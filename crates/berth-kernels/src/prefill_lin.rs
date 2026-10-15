//! EXL3 linears for `rows = bsz * seq_len > 1`.
//!
//! [`exl3_linear`] follows `LinearEXL3.forward` (`modules/quant/exl3.py`).
//! Rows at or under 144 go through `BC_LinearEXL3::run` → `exl3_gemm`
//! (`libtorch/linear.cpp`). That tries the plain int8 sq GEMV at `m = 2`
//! (`EXL3_INT8_GEMV=2`) and otherwise the cooperative `exl3_gemm_kernel`
//! for K = 4, mul1, shapes 1–6. The autotune key is the same value
//! `exl3_gemm` hashes (`max(rows, 2)`, K, N, bits, dtype, device, cc, SMs,
//! codebook), so a serve-cache hit launches the cached shape.
//!
//! Rows above 144 take `reconstruct_hgemm`. The reconstructed fp16 weight is
//! written into `w_scratch` and dropped by the caller: the fork allocates
//! that buffer inside the call and does not keep it. Fused reconstruct
//! (both Hadamards folded into the weight, no standalone `had_r_128`) runs
//! when K and N are multiples of 128, `EXL3_NO_FUSED_RECONSTRUCT` is unset
//! or `"0"`, and rows are at least 1024. The GEMM is `hgemm_recon`:
//! `f16acc::gemm_kernel<out, 128, false>` when the fork's coverage and
//! worthwhile checks pass and `EXL3_HGEMM_F16ACC=auto` enables it, otherwise
//! `cublasGemmEx` on torch's `libcublas.so.13` (see `csrc/prefill_lin/hgemm.cu`).
//!
//! [`exl3_mgemm`] is the sliced multi-linear (`ext.exl3_mgemm`). It does not
//! reconstruct. The model itself only calls it for `rows <= 32`; the kernel
//! accepts any row count, which is what the parity test launches.
//!
//! Pointers are device addresses. `stream` is a `cudaStream_t` (`null` is
//! the default stream). Bias is not applied. 27B body linears have none.

use std::ffi::{c_void, CStr};

use crate::ops;
use crate::{Codebook, Error, OutDtype, Result};

mod ffi {
    use std::ffi::{c_char, c_int, c_void};

    unsafe extern "C" {
        pub fn berth_prefill_exl3_linear(
            a: *const c_void,
            trellis: *const c_void,
            c: *mut c_void,
            suh: *const c_void,
            a_had: *mut c_void,
            svh: *const c_void,
            w_scratch: *mut c_void,
            size_m: c_int,
            size_k: c_int,
            size_n: c_int,
            k_bits: c_int,
            codebook: c_int,
            c_fp32: c_int,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_prefill_hgemm(
            a: *const c_void,
            b: *const c_void,
            c: *mut c_void,
            size_m: c_int,
            size_k: c_int,
            size_n: c_int,
            ldc: c_int,
            c_fp32: c_int,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_prefill_last_error() -> *const c_char;
        pub fn berth_prefill_last_launch() -> *const c_char;
    }
}

/// One prefill EXL3 linear: `C = A @ trellis`, rows > 1.
///
/// Shapes, device memory, row-major:
///
/// * `a`: fp16 activation, `size_m * size_k`.
/// * `trellis`: int16, `(size_k / 16, size_n / 16, 16 * k_bits)`.
/// * `c`: `size_m * size_n` elements, dtype [`OutDtype`].
/// * `suh`, `svh`: fp16 Hadamard signs, lengths `size_k` and `size_n`.
/// * `a_had`: fp16 scratch, `size_m * size_k`. The int8 and cooperative
///   kernels write the transformed activation here. Rows above 144 and below
///   1024 use it as the Hadamard image of `a`. The fused path (rows ≥ 1024)
///   does not read it; it still has to be non-null.
/// * `w_scratch`: fp16 weight scratch, `size_k * min(size_n, 32768)`
///   elements, required when `size_m > 144` and ignored otherwise (null).
///   Every call reconstructs into it. Nothing is cached.
///
/// `k_bits` is 4 and [`Codebook::Mul1`] for the 27B body. `size_m` must be
/// greater than 1. A GEMM cache miss times candidates on `stream` and appends
/// the winner to the coop autotune file.
///
/// # Errors
///
/// Null pointers, a non-positive size, `size_m <= 1`, or a K / codebook this
/// shim does not link return [`Error::Launch`] before a kernel launch. QTIP
/// (`m <= 8` after int8 declines) is refused rather than replaced with a
/// different kernel. A CUDA failure is the same error.
///
/// # Safety
///
/// Every non-null pointer except `stream` must address device memory large
/// enough for the shape above, and must stay alive until `stream` completes.
/// `k_bits`, [`Codebook`], and [`OutDtype`] must describe that trellis.
#[allow(clippy::too_many_arguments)]
pub unsafe fn exl3_linear(
    a: *const c_void,
    trellis: *const c_void,
    c: *mut c_void,
    suh: *const c_void,
    a_had: *mut c_void,
    svh: *const c_void,
    w_scratch: *mut c_void,
    size_m: i32,
    size_k: i32,
    size_n: i32,
    k_bits: i32,
    codebook: Codebook,
    out: OutDtype,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe {
        ffi::berth_prefill_exl3_linear(
            a,
            trellis,
            c,
            suh,
            a_had,
            svh,
            w_scratch,
            size_m,
            size_k,
            size_n,
            k_bits,
            codebook as i32,
            out.as_c(),
            stream,
        )
    };
    if rc == 0 {
        return Ok(());
    }
    let message = last_error();
    let message = if message.is_empty() {
        format!("prefill exl3 linear launch failed ({rc})")
    } else {
        message
    };
    Err(Error::Launch(message))
}

/// Plain fp16 GEMM, `C[m, n] = A[m, k] @ B[k, n]`, as the fork's `ext.hgemm`
/// runs it: its own `f16acc::gemm_kernel` when that covers the shape,
/// otherwise `cublasGemmEx` (torch's libcublas) with the fork's arguments.
///
/// * `a`: fp16 `[size_m, size_k]`, row-major.
/// * `b`: fp16 `[size_k, size_n]`, row-major (the `x @ W` layout).
/// * `c`: `[size_m, ldc]` of [`OutDtype`]; columns `0..size_n` are written.
///
/// GDN prefill uses this for the b / a projections.
///
/// # Errors
///
/// Null pointers or a bad shape return [`Error::Launch`] before any launch.
/// A CUDA or cuBLAS failure is the same error.
///
/// # Safety
///
/// The pointers must address device memory of the sizes above and stay alive
/// until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn hgemm(
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    size_m: i32,
    size_k: i32,
    size_n: i32,
    ldc: i32,
    out: OutDtype,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe {
        ffi::berth_prefill_hgemm(a, b, c, size_m, size_k, size_n, ldc, out.as_c(), stream)
    };
    if rc == 0 {
        return Ok(());
    }
    let message = last_error();
    let message = if message.is_empty() {
        format!("prefill hgemm launch failed ({rc})")
    } else {
        message
    };
    Err(Error::Launch(message))
}

/// Kernel the last successful [`exl3_linear`] on this thread launched.
pub fn last_launch() -> String {
    c_string(unsafe { ffi::berth_prefill_last_launch() })
}

/// Message from the last failed [`exl3_linear`] on this thread.
pub fn last_error() -> String {
    c_string(unsafe { ffi::berth_prefill_last_error() })
}

fn c_string(ptr: *const std::ffi::c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

/// Sliced `exl3_mgemm` at any row count, including prefill rows past 32.
///
/// Same launch as [`ops::exl3_mgemm`]. The 27B bundles are GDN `in_proj_qkv`
/// with `in_proj_z` (fp32, slice 2048) and attention `q_proj` (interleaved
/// gate) with `k_proj` and `v_proj` (fp16, slice 1024). A cache miss appends
/// to the same coop autotune file. [`ops::last_launch`] names the kernel.
///
/// # Safety
///
/// The pointer rules are [`ops::exl3_mgemm`]'s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn exl3_mgemm(
    a: *const c_void,
    b_list: *const c_void,
    c: *mut c_void,
    suh_list: *const c_void,
    a_had: *mut c_void,
    svh_list: *const c_void,
    size_m: i32,
    size_k: i32,
    size_n: i32,
    k_bits: i32,
    codebook: Codebook,
    out: OutDtype,
    bszm_in: i32,
    bszm_out: i32,
    size_n_list: *const i32,
    c_ptrs: *const c_void,
    n_stride_list: *const i32,
    had_src_list: *const i32,
    num_had_src: i32,
    stream: *mut c_void,
) -> Result<i32> {
    unsafe {
        ops::exl3_mgemm(
            a,
            b_list,
            c,
            suh_list,
            a_had,
            svh_list,
            size_m,
            size_k,
            size_n,
            k_bits,
            codebook,
            out,
            bszm_in,
            bszm_out,
            size_n_list,
            c_ptrs,
            n_stride_list,
            had_src_list,
            num_had_src,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;

    #[test]
    fn symbols_link() {
        let linear = ffi::berth_prefill_exl3_linear as *const ();
        let err = ffi::berth_prefill_last_error as *const ();
        let launch = ffi::berth_prefill_last_launch as *const ();
        assert!(!linear.is_null());
        assert!(!err.is_null());
        assert!(!launch.is_null());
    }

    #[test]
    fn null_and_m1_do_not_launch() {
        let rc = unsafe {
            exl3_linear(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null_mut(),
                2,
                128,
                256,
                4,
                Codebook::Mul1,
                OutDtype::F16,
                std::ptr::null_mut(),
            )
        };
        let err = rc.expect_err("null pointers");
        assert!(err.to_string().contains("null"), "{err}");
        assert!(last_launch().is_empty());

        let p = std::ptr::dangling::<c_void>();
        let rc = unsafe {
            exl3_linear(
                p,
                p,
                p as *mut c_void,
                p,
                p as *mut c_void,
                p,
                std::ptr::null_mut(),
                1,
                128,
                256,
                4,
                Codebook::Mul1,
                OutDtype::F16,
                std::ptr::null_mut(),
            )
        };
        let err = rc.expect_err("m = 1");
        assert!(err.to_string().contains("size_m"), "{err}");
    }
}

//! EXL3 linear launches on raw device pointers.
//!
//! The static library is compiled from the exllamav3 fork at `BERTH_EXL3_SRC`
//! (default `/home/sam/qwen38-exl3-dflash2/exllamav3`). `build.rs` requires
//! `git rev-parse HEAD:exllamav3/exllamav3_ext` to be
//! `de35ffbe336bdc6d617e49e208cd7c730a789787`, the tree the 27B service loads.
//! Sources are included from that tree; they are not vendored here.
//!
//! nvcc is `/usr/local/cuda-13.3/bin/nvcc` (`BERTH_NVCC` overrides it), with the
//! fork's flags: `-O3 --use_fast_math -lineinfo -arch=sm_86` and
//! `-Xcudafe --diag_suppress=177,20012`. The link is the CUDA 13.3 shared
//! runtime, `libcudart.so.13`, plus `libstdc++`. Each translation unit keeps
//! its own fatbin; there is no relocatable device link.
//!
//! Linked instances, the m = 1 decode cases in `testdata/exl3_ref/`:
//!
//! * `K = 4`, mul1, `EXL3_INT8_GEMV=2`, `EXL3_INT8_GEMV_MAX_K=5`:
//!   `exl3_gemv_int8_sq_kernel<4,1,0,0>` (fp16 out) and `<4,1,1,0>` (fp32 out).
//! * `K = 6`, mul1, fp16 out: cooperative `exl3_gemm_kernel<6,0,2,…>` chosen by
//!   the fork's coop autotune. For `lm_head` (`m = 1`, `K = 5120`, `N = 248320`,
//!   device 0, 82 SMs, Ampere) the on-disk cache selects shape 4
//!   (block 256, 82 SMs, concurrency 1).
//!
//! Other shapes fail closed: residual int8 (`EXL3_INT8_GEMV=1`), the M = 2 sq
//! instance, cooperative int8, QTIP `exl3_gemv`, and GEMM other than K = 6
//! mul1 fp16 are not linked. `last_launch` names the kernel that ran.
//!
//! This crate does not use `berth-cuda` types. Pointers are device addresses
//! and `stream` is a `cudaStream_t` (`null` is the default stream). Bias is
//! not applied; the 27B language-model linears have none, and the reference
//! outputs match the kernel output.

mod error;
mod ffi;

pub use error::{Error, Result};

use std::ffi::{c_void, CStr};

/// Codebook selector. Matches the fork's `cb` argument.
///
/// A checkpoint stores this as a marker tensor, not as a weight the kernel
/// reads. `<prefix>.mul1` present means [`Codebook::Mul1`]. `<prefix>.mcg`
/// present means [`Codebook::Mcg`]. Neither marker is [`Codebook::Default`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Codebook {
    /// Original 3-instruction codebook. No `mcg` or `mul1` tensor.
    Default = 0,
    /// `<prefix>.mcg` is present.
    Mcg = 1,
    /// `<prefix>.mul1` is present. The 27B linears and `lm_head` use this.
    Mul1 = 2,
}

/// Device dtype of the output `C`.
///
/// Not stored in the checkpoint. On the 27B, gate, up, and `lm_head` are
/// [`OutDtype::F16`]. Down and `o_proj` / `out_proj` are [`OutDtype::F32`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutDtype {
    /// `C` is fp16. Gate, up, and `lm_head`.
    F16,
    /// `C` is fp32. Down and `o_proj` / `out_proj`.
    F32,
}

impl OutDtype {
    fn as_c(self) -> i32 {
        match self {
            Self::F16 => 0,
            Self::F32 => 1,
        }
    }
}

/// Launch one EXL3 linear: `C = A @ trellis`, with the Hadamard scales.
///
/// Shapes, all device memory, row-major where they are matrices:
///
/// * `a`: fp16 activation, `size_m * size_k` elements. `size_k` is
///   `x.shape[-1]` (in-features). `size_m` is the product of the leading
///   dimensions. Decode passes `size_m = 1`.
/// * `trellis`: `<prefix>.trellis`, int16, shape
///   `(size_k / 16, size_n / 16, 16 * k_bits)`. `k_bits` is
///   `trellis.shape[-1] / 16`. `size_n` is out-features (`y.shape[-1]`).
/// * `c`: output, `size_m * size_n` elements, dtype [`OutDtype`].
/// * `suh`: `<prefix>.suh`, fp16, length `size_k`. Input Hadamard signs.
/// * `a_had`: fp16 scratch, `size_m * size_k` elements, same layout as `a`.
///   The kernel writes the transformed activation here. The Python bsz-1
///   cache sometimes allocates this buffer in the output dtype; the kernel
///   still writes fp16, so the buffer must be at least `size_m * size_k`
///   halves.
/// * `svh`: `<prefix>.svh`, fp16, length `size_n`. Output Hadamard signs.
///
/// `k_bits` is 4 for the 27B body and 6 for `lm_head`. `size_k` must be a
/// multiple of 16 and `size_n` a multiple of 128. The int8 sq path also
/// requires `size_k % 128 == 0` and `size_n % 256 == 0`.
///
/// `stream` is a `cudaStream_t` (`*mut c_void`). Null is the default stream.
/// The final kernel is asynchronous. A GEMM cache miss times candidates on
/// this stream before that launch and synchronizes those measurements; it
/// also appends the winner to the coop autotune cache (`EXLLAMAV3_TUNE_CACHE`,
/// or `~/.cache/exllamav3/autotune/coop_autotune_v1.bin`). A hit launches once
/// and does not write the cache.
///
/// Locks for GEMM and the 16 MiB int8 workspace are allocated inside the shim,
/// once per device, and never grown.
///
/// # Errors
///
/// Host checks (null pointers, sizes, `k_bits` outside `1..=8`, an unknown
/// codebook) return [`Error::Launch`] before any CUDA call. A shape this
/// shim has no kernel for, or a CUDA failure, is the same error. The message
/// is also available from [`last_error`] until the next launch on this thread.
///
/// # Safety
///
/// Every pointer except a null `stream` must be a device allocation large
/// enough for the shape above, and must stay alive until `stream` has
/// completed the launch. The shim does not check the bytes behind the
/// pointers. `k_bits`, the sizes, and [`Codebook`] must describe that
/// trellis. Concurrent launches on the same device that both miss the GEMM
/// cache share the autotune tables; the fork does not lock that path either.
#[allow(clippy::too_many_arguments)]
pub unsafe fn exl3_linear(
    a: *const c_void,
    trellis: *const c_void,
    c: *mut c_void,
    suh: *const c_void,
    a_had: *mut c_void,
    svh: *const c_void,
    size_m: i32,
    size_k: i32,
    size_n: i32,
    k_bits: i32,
    codebook: Codebook,
    out: OutDtype,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe {
        ffi::berth_exl3_linear(
            a,
            trellis,
            c,
            suh,
            a_had,
            svh,
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
        format!("exl3 linear launch failed ({rc})")
    } else {
        message
    };
    Err(Error::Launch(message))
}

/// Kernel the last successful [`exl3_linear`] on this thread launched.
///
/// Empty after a failed launch, and before any launch. The string is a copy;
/// the next launch on this thread replaces the shim's buffer.
pub fn last_launch() -> String {
    c_string(unsafe { ffi::berth_exl3_last_launch() })
}

/// Message from the last failed [`exl3_linear`] on this thread.
///
/// Empty after a successful launch. Copied, same as [`last_launch`].
pub fn last_error() -> String {
    c_string(unsafe { ffi::berth_exl3_last_error() })
}

fn c_string(ptr: *const std::ffi::c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;
    use std::process::Command;

    #[test]
    fn symbols_link() {
        let linear = ffi::berth_exl3_linear as *const ();
        let err = ffi::berth_exl3_last_error as *const ();
        let launch = ffi::berth_exl3_last_launch as *const ();
        let tune = ffi::berth_exl3_gemm_tune_key as *const ();
        let salt = ffi::berth_exl3_salt_hash as *const ();
        assert!(!linear.is_null());
        assert!(!err.is_null());
        assert!(!launch.is_null());
        assert!(!tune.is_null());
        assert!(!salt.is_null());
    }

    #[test]
    fn lm_head_tune_key_matches_cache() {
        // device 0, CC_AMPERE = 2, 82 SMs. m = 1 is keyed as max(m, 2).
        // Salted key is the disk record (shape 4, block 256, 82 SMs, concurrency 1).
        let key = unsafe {
            ffi::berth_exl3_gemm_tune_key(1, 5120, 248320, 6, 0, 0, 2, 82, Codebook::Mul1 as i32)
        };
        assert_eq!(key, 0x51f4_0c1b_d7ca_e3a7);
        let salted = unsafe { ffi::berth_exl3_salt_hash(key) };
        assert_eq!(salted, 0x0c94_364f_adc0_cdf9);
    }

    #[test]
    fn null_and_bad_sizes_do_not_call_cuda() {
        let rc = unsafe {
            exl3_linear(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null(),
                1,
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

        // Non-null and never dereferenced: the size check returns first.
        let p = std::ptr::dangling::<c_void>();
        let rc = unsafe {
            exl3_linear(
                p,
                p,
                p as *mut c_void,
                p,
                p as *mut c_void,
                p,
                1,
                15,
                256,
                4,
                Codebook::Mul1,
                OutDtype::F32,
                std::ptr::null_mut(),
            )
        };
        let err = rc.expect_err("size_k");
        assert!(err.to_string().contains("multiple of 16"), "{err}");

        let rc = unsafe {
            exl3_linear(
                p,
                p,
                p as *mut c_void,
                p,
                p as *mut c_void,
                p,
                1,
                16,
                100,
                4,
                Codebook::Mul1,
                OutDtype::F16,
                std::ptr::null_mut(),
            )
        };
        let err = rc.expect_err("size_n");
        assert!(err.to_string().contains("multiple of 128"), "{err}");

        let rc = unsafe {
            exl3_linear(
                p,
                p,
                p as *mut c_void,
                p,
                p as *mut c_void,
                p,
                0,
                16,
                128,
                4,
                Codebook::Mul1,
                OutDtype::F16,
                std::ptr::null_mut(),
            )
        };
        let err = rc.expect_err("size_m");
        assert!(err.to_string().contains("positive"), "{err}");
    }

    fn serve_blocks_gpu() -> bool {
        let output = match Command::new("ss").args(["-ltn"]).output() {
            Ok(output) => output,
            Err(_) => return true,
        };
        if !output.status.success() {
            return true;
        }
        let text = String::from_utf8_lossy(&output.stdout);
        text.contains(":8889 ")
    }

    #[test]
    fn gpu_smoke_when_enabled() {
        if std::env::var("BERTH_GPU_TESTS").ok().as_deref() != Some("1") {
            return;
        }
        if serve_blocks_gpu() {
            eprintln!("berth-kernels: skipping GPU smoke; port 8889 is in use or ss failed");
            return;
        }

        let rc = unsafe { ffi::berth_exl3_smoke_int8(0) };
        assert_eq!(rc, 0, "{}", last_error());
        let launch = last_launch();
        assert!(
            launch.contains("exl3_gemv_int8_sq_kernel<4,1,0,0>"),
            "{launch}"
        );
        eprintln!("int8 fp16: {launch}");

        let rc = unsafe { ffi::berth_exl3_smoke_int8(1) };
        assert_eq!(rc, 0, "{}", last_error());
        let launch = last_launch();
        assert!(
            launch.contains("exl3_gemv_int8_sq_kernel<4,1,1,0>"),
            "{launch}"
        );
        eprintln!("int8 fp32: {launch}");

        let rc = unsafe { ffi::berth_exl3_smoke_gemm() };
        assert_eq!(rc, 0, "{}", last_error());
        let launch = last_launch();
        assert!(launch.contains("exl3_gemm_kernel<6,0,2,"), "{launch}");
        assert!(launch.contains("smem=92160"), "{launch}");
        eprintln!("gemm: {launch}");
    }
}

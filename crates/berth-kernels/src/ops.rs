//! Decode-path ops other than the single EXL3 linear: fused multi-linear,
//! RMSNorm, SiLU glue, the fp32 residual add, and the token embedding.
//!
//! Pointers are device addresses. `stream` is a `cudaStream_t` (`null` is the
//! default stream). Host checks return [`Error::Launch`] before any CUDA call.
//! [`last_launch`] and [`last_error`] are the ops launches on this thread,
//! separate from the single-linear shim.

use std::ffi::{c_void, CStr};

use crate::{Codebook, Error, OutDtype, Result};

mod ffi {
    use std::ffi::{c_char, c_int, c_void};

    unsafe extern "C" {
        pub fn berth_ops_mgemm(
            a: *const c_void,
            b_list: *const c_void,
            c: *mut c_void,
            suh_list: *const c_void,
            a_had: *mut c_void,
            svh_list: *const c_void,
            size_m: c_int,
            size_k: c_int,
            size_n: c_int,
            k_bits: c_int,
            codebook: c_int,
            c_fp32: c_int,
            bszm_in: c_int,
            bszm_out: c_int,
            size_n_list: *const c_int,
            c_ptrs: *const c_void,
            n_stride_list: *const c_int,
            had_src_list: *const c_int,
            num_had_src: c_int,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_ops_mgemm_tune_key(
            size_m: c_int,
            size_k: c_int,
            size_n: c_int,
            k_bits: c_int,
            c_fp32: c_int,
            device: c_int,
            cc: c_int,
            sms: c_int,
            cb: c_int,
            bszm_in: c_int,
            bszm_out: c_int,
            sliced: c_int,
        ) -> u64;

        pub fn berth_ops_rms_norm(
            x: *const c_void,
            weight: *const c_void,
            y: *mut c_void,
            rows: c_int,
            dim: c_int,
            epsilon: f32,
            constant_bias: f32,
            constant_scale: f32,
            weight_bf16: c_int,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_ops_rms_norm_res_in(
            x: *const c_void,
            weight: *const c_void,
            y: *mut c_void,
            residual: *mut c_void,
            rows: c_int,
            dim: c_int,
            epsilon: f32,
            constant_bias: f32,
            constant_scale: f32,
            weight_bf16: c_int,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_ops_silu_mul(
            gate: *const c_void,
            up: *const c_void,
            out: *mut c_void,
            numel: i64,
            act_limit: f32,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_ops_add_f32(
            a: *const c_void,
            b: *const c_void,
            c: *mut c_void,
            n: c_int,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_ops_embed_tokens(
            weight: *const c_void,
            ids: *const c_int,
            out: *mut c_void,
            hidden: c_int,
            n: c_int,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_ops_rms_norm_f16(
            x: *const c_void,
            weight: *const c_void,
            y: *mut c_void,
            rows: c_int,
            dim: c_int,
            epsilon: f32,
            constant_bias: f32,
            constant_scale: f32,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_ops_embed_tokens_f16(
            weight: *const c_void,
            ids: *const c_int,
            out: *mut c_void,
            hidden: c_int,
            n: c_int,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_ops_memcpy_d2d(
            dst: *mut c_void,
            src: *const c_void,
            bytes: u64,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_ops_cat_f16_last(
            embedding: *const c_void,
            hidden: *const c_void,
            out: *mut c_void,
            rows: c_int,
            hidden_size: c_int,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_ops_last_error() -> *const c_char;
        pub fn berth_ops_last_launch() -> *const c_char;
    }
}

/// Dtype of an RMSNorm weight row.
///
/// `model.language_model.layers.*.input_layernorm.weight`,
/// `model.language_model.layers.*.post_attention_layernorm.weight`, and
/// `model.language_model.norm.weight` are BF16 on the 27B. The fork loader
/// keeps BF16 (`allow_bf16=True`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormWeight {
    /// fp16 weight. Not what the 27B layernorms store.
    F16,
    /// bf16 weight. The 27B layernorms and the final norm.
    Bf16,
}

impl NormWeight {
    fn as_c(self) -> i32 {
        match self {
            Self::F16 => 0,
            Self::Bf16 => 1,
        }
    }
}

fn c_string(ptr: *const std::ffi::c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn error_message(rc: i32) -> String {
    let message = last_error();
    if message.is_empty() {
        format!("ops launch failed ({rc})")
    } else {
        message
    }
}

/// Kernel the last successful ops launch on this thread ran.
///
/// Empty after a failed launch, and before any launch. The next ops launch on
/// this thread replaces it. Single-linear launches do not.
pub fn last_launch() -> String {
    c_string(unsafe { ffi::berth_ops_last_launch() })
}

/// Message from the last failed ops launch on this thread.
///
/// Empty after a successful launch.
pub fn last_error() -> String {
    c_string(unsafe { ffi::berth_ops_last_error() })
}

/// Unsalted coop-autotune key for [`exl3_mgemm`].
///
/// This is `mgemm_autotune_hash` from `quant/exl3_gemm.cu`. `size_m` goes
/// through `gemm_autotune_m_key` directly, so `m = 1` stays 1. The regular
/// GEMM key uses `max(m, 2)` and does not mix `bszm`. When `sliced` is set
/// the key is xored with `0x9e3779b97f4a7c15`, which is what `exl3_mgemm_gr`
/// does once `had_src_list` is present. No CUDA call. The autotuner salts
/// this value (version 4) before the disk lookup.
#[allow(clippy::too_many_arguments)]
pub fn mgemm_tune_key(
    size_m: i32,
    size_k: i32,
    size_n: i32,
    k_bits: i32,
    out: OutDtype,
    device: i32,
    cc: i32,
    sms: i32,
    codebook: Codebook,
    bszm_in: i32,
    bszm_out: i32,
    sliced: bool,
) -> u64 {
    unsafe {
        ffi::berth_ops_mgemm_tune_key(
            size_m,
            size_k,
            size_n,
            k_bits,
            out.as_c(),
            device,
            cc,
            sms,
            codebook as i32,
            bszm_in,
            bszm_out,
            i32::from(sliced),
        )
    }
}

/// One fused EXL3 multi-linear: `exl3_mgemm_gr` / `ext.exl3_mgemm`.
///
/// Sliced mode is `SlicedMultiLinear` (`modules/multilinear.py`). The 27B
/// decode uses it twice per token, both `m = 1`, `K = 4`, mul1:
///
/// * GDN layer 0, `BC_GatedDeltaNetSplit` `project_qkvz_sliced`. Sources are
///   `model.language_model.layers.0.linear_attn.in_proj_qkv` (`N = 10240`)
///   and `in_proj_z` (`N = 6144`). Slice width is their gcd, 2048 (8 slices).
///   Outputs are fp32. The measured kernel is
///   `exl3_mgemm_kernel<4,1,2,16,16,512,4,3>`.
/// * Full attention layer 3, `project_qkv_sliced` with `interleaved_gate`.
///   Sources are `model.language_model.layers.3.self_attn.q_proj`
///   (`N = 12288`, q and g interleaved), `k_proj` (`N = 1024`), and `v_proj`
///   (`N = 1024`). Slice width 1024 (14 slices). Outputs are fp16, before
///   `deinterleave_qg`. The measured kernel is
///   `exl3_mgemm_kernel<4,0,2,16,16,512,4,3>`.
///
/// Shapes, device memory:
///
/// * `a`: fp16 activation, `(bszm_in, size_m, size_k)`. Decode passes
///   `bszm_in = 1`, `size_m = 1`, `size_k = 5120`.
/// * `b_list`: `uint64` device pointers to int16 trellis bases, one per slice
///   in sliced mode. Slice `n0` of a source is
///   `trellis + (n0 / 16) * 16 * K * 2` bytes (`<prefix>.trellis`).
/// * `c`: carrier, `(bszm_out, size_m, size_n)` in the output dtype. In sliced
///   mode the kernel writes `c_ptrs` and uses `c` for the dtype and for
///   `size_n` (the slice width, not the full `N`).
/// * `suh_list`: `uint64` pointers to fp16 `<prefix>.suh`. One entry per
///   source in sliced mode, not per slice. Length `size_k`.
/// * `a_had`: fp16 scratch. Sliced mode needs `num_had_src * size_m * size_k`
///   elements (one slab per source). Uniform mode needs `max(bszm_in, bszm_out)`
///   slabs.
/// * `svh_list`: `uint64` pointers to fp16 `<prefix>.svh`, offset by
///   `n0 * 2` bytes for a slice.
/// * `size_n_list`, `n_stride_list`, `had_src_list`: int32 device arrays, one
///   entry per slice. Width, full source `N` (row stride), and source index.
///   All four list pointers are null in uniform mode. Sliced mode requires
///   all four, `num_had_src >= 1`, and `bszm_in = 1`. `bszm_out` is the slice
///   count (`c_ptrs` length), which is what `exl3_mgemm_gr` stores from
///   `c_ptrs.size(0)` before hashing.
/// * `c_ptrs`: `uint64` device pointers, output of slice `n0` at
///   `out.data_ptr() + n0 * element_size`.
///
/// `k_bits` is 4 and [`Codebook::Mul1`] for these linears (`<prefix>.mul1`
/// present, no `.mcg`, no bias). Other `K` / codebook values fail closed.
/// `size_k` must be a multiple of 16 and `size_n` (the slice width) a multiple
/// of 128.
///
/// Indices, weights, and expert-range filtering are not parameters. They stay
/// null, `min_index = max_index = -1`, `num_tokens = 1`, `force_shape_idx = -1`.
/// The fused qkv bundles do not use them.
///
/// A cache hit launches once and does not write the coop autotune file. A miss
/// times candidates on `stream` and appends the winner (`EXLLAMAV3_TUNE_CACHE`,
/// or `~/.cache/exllamav3/autotune/coop_autotune_v1.bin`). Returns the shape
/// tag. Locks are the same per-device buffer as the single-linear GEMM.
///
/// # Errors
///
/// Null pointers, bad sizes, a partial slice-list, or a `K` / codebook with
/// no linked kernel return [`Error::Launch`] before CUDA. A CUDA failure is
/// the same error.
///
/// # Safety
///
/// Every non-null pointer except `stream` must address device memory large
/// enough for the shape above, and must stay alive until `stream` completes.
/// `k_bits`, [`Codebook`], and [`OutDtype`] must describe those trellises.
/// Concurrent launches that both miss the cache share the autotune tables.
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
    let rc = unsafe {
        ffi::berth_ops_mgemm(
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
            codebook as i32,
            out.as_c(),
            bszm_in,
            bszm_out,
            size_n_list,
            c_ptrs,
            n_stride_list,
            had_src_list,
            num_had_src,
            stream,
        )
    };
    if rc > 0 {
        return Ok(rc);
    }
    Err(Error::Launch(error_message(rc)))
}

/// RMSNorm without a fused residual: `ext.rms_norm`, `rms_norm_kernel<0, ...>`.
///
/// `RMSNorm.forward` with no `residual_in`. On the 27B that is:
///
/// * `model.language_model.layers.{i}.input_layernorm`, the attention input
///   norm. `TransformerBlock` calls it with `out_dtype = torch.half`. Input
///   is the fp32 residual stream. 64 launches per token.
/// * `model.language_model.norm`, the final norm, constructed with
///   `out_dtype = torch.half`. One launch per token, on the fp32 residual
///   after the last layer.
///
/// Together those are the 65 `rms_norm_kernel<0, ...>` launches. Weight is
/// BF16, `constant_bias = 1`, `constant_scale = 1`, `rms_norm_eps = 1e-6`,
/// `groups = 1`, `span_heads = false`. Output is fp16. `rows` is the product
/// of the leading dimensions; decode passes `rows = 1` and `dim = 5120`.
///
/// `y[row] = rms(x[row]) * (weight + constant_bias) * constant_scale`, with
/// `rms` the reciprocal square root of the mean square plus `epsilon`.
///
/// # Errors
///
/// A null pointer, non-positive `rows` or `dim`, or `dim` not a multiple of 4
/// returns [`Error::Launch`] before CUDA.
///
/// # Safety
///
/// `x` is fp32, `weight` is the dtype in `weight` with `dim` elements, and
/// `y` is fp16, each `rows` rows. They must stay alive until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn rms_norm(
    x: *const c_void,
    weight: *const c_void,
    y: *mut c_void,
    rows: i32,
    dim: i32,
    epsilon: f32,
    constant_bias: f32,
    constant_scale: f32,
    weight_dtype: NormWeight,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe {
        ffi::berth_ops_rms_norm(
            x,
            weight,
            y,
            rows,
            dim,
            epsilon,
            constant_bias,
            constant_scale,
            weight_dtype.as_c(),
            stream,
        )
    };
    if rc == 0 {
        return Ok(());
    }
    Err(Error::Launch(error_message(rc)))
}

/// Fused pre-norm residual: `ext.rms_norm_res_in`, `rms_norm_kernel<2, ...>`.
///
/// `post_attention_layernorm` (`model.language_model.layers.{i}.post_attention_layernorm`).
/// `TransformerBlock` defers the attention residual when `can_fuse_residual`
/// accepts the fp32 stream and the fp32 attention output, then calls
/// `mlp_norm.forward(y_resid, out_dtype=torch.half, residual_in=x)`. That is
/// 64 launches per token.
///
/// `residual += x` in place, in the residual dtype (fp32, so the sum is exact),
/// then `y = rms(residual) * (weight + constant_bias) * constant_scale` in
/// fp16. Same epsilon, bias, and scale as [`rms_norm`]. `x` is the attention
/// output, not the norm weight.
///
/// # Errors
///
/// Same as [`rms_norm`], plus a null `residual`.
///
/// # Safety
///
/// `x` and `residual` are fp32 of `rows * dim` elements. `residual` is
/// overwritten. `weight` and `y` match [`rms_norm`]. All four stay alive
/// until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn rms_norm_res_in(
    x: *const c_void,
    weight: *const c_void,
    y: *mut c_void,
    residual: *mut c_void,
    rows: i32,
    dim: i32,
    epsilon: f32,
    constant_bias: f32,
    constant_scale: f32,
    weight_dtype: NormWeight,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe {
        ffi::berth_ops_rms_norm_res_in(
            x,
            weight,
            y,
            residual,
            rows,
            dim,
            epsilon,
            constant_bias,
            constant_scale,
            weight_dtype.as_c(),
            stream,
        )
    };
    if rc == 0 {
        return Ok(());
    }
    Err(Error::Launch(error_message(rc)))
}

/// `silu(gate) * up` in fp16: `ext.silu_mul`, `act_mul_kernel_h<0>`.
///
/// `GatedMLP` (`modules/mlp.py`, `libtorch/mlp.cpp` `silu_mul_gr`) with
/// `interm_dtype = torch.half` and `act_limit = 0`. Gate and up are separate
/// int8 GEMVs on the 27B; this launch is only the product. `numel` is
/// `intermediate_size` (17408) per token. `out` may alias `gate` or `up`; the
/// graphed bsz-1 path uses a separate buffer.
///
/// `act_limit = 0` skips the clamp. A non-zero limit clamps the half2 product
/// the way `act_mul_kernel_h` does.
///
/// # Errors
///
/// A null pointer, or `numel` that is not positive and even, returns
/// [`Error::Launch`] before CUDA.
///
/// # Safety
///
/// `gate`, `up`, and `out` are fp16 device pointers of `numel` elements and
/// stay alive until `stream` completes.
pub unsafe fn silu_mul(
    gate: *const c_void,
    up: *const c_void,
    out: *mut c_void,
    numel: i64,
    act_limit: f32,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe { ffi::berth_ops_silu_mul(gate, up, out, numel, act_limit, stream) };
    if rc == 0 {
        return Ok(());
    }
    Err(Error::Launch(error_message(rc)))
}

/// `c = a + b` in fp32, bit-identical to `torch.add`.
///
/// The decode residual after the MLP is not a fork kernel. With no MLP
/// post-norm and no hyperconnection, `TransformerBlock.forward` does
/// `x += y` on the fp32 stream (`CUDAFunctor_add<float>`, 64 times per
/// token). The attention residual is [`rms_norm_res_in`], not this add.
/// There is no `exllamav3_ext` binding; the reference is `torch.add`.
///
/// `n` is `hidden_size` (5120) for one token. `c` may alias `a`.
///
/// # Errors
///
/// A null pointer or `n < 1` returns [`Error::Launch`] before CUDA.
///
/// # Safety
///
/// `a`, `b`, and `c` are fp32 device pointers of `n` elements and stay alive
/// until `stream` completes.
pub unsafe fn add_f32(
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    n: i32,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe { ffi::berth_ops_add_f32(a, b, c, n, stream) };
    if rc == 0 {
        return Ok(());
    }
    Err(Error::Launch(error_message(rc)))
}

/// Gather embedding rows and widen BF16 to fp32.
///
/// `model.language_model.embed_tokens.weight` is BF16 `[248320, 5120]`.
/// `modules/embedding.py` loads it with `allow_bf16=True` into `nn.Embedding`.
/// `qwen3_5.Embedding` leaves `out_dtype` at `torch.float`, so `forward` is
/// the gather followed by `to2(..., torch.float)`. `normalize` is false and
/// `multiplier` is 1. Indexed embeddings are not on the text path. There is
/// no extension binding; the reference is `F.embedding(ids, weight).float()`.
///
/// `__bfloat162float` pads the mantissa, which is the same bits as that cast.
/// `ids` are `i32` token indices into `weight`'s row axis and must already
/// be in range. The kernel does not read a vocab size.
///
/// # Errors
///
/// A null pointer, or non-positive `hidden` or `n`, returns [`Error::Launch`]
/// before CUDA.
///
/// # Safety
///
/// `weight` is BF16 row-major with stride `hidden` and covers every id.
/// `ids` has `n` `i32` elements. `out` is fp32 `[n, hidden]`. All three stay
/// alive until `stream` completes.
pub unsafe fn embed_tokens(
    weight: *const c_void,
    ids: *const i32,
    out: *mut c_void,
    hidden: i32,
    n: i32,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe { ffi::berth_ops_embed_tokens(weight, ids, out, hidden, n, stream) };
    if rc == 0 {
        return Ok(());
    }
    Err(Error::Launch(error_message(rc)))
}

/// RMSNorm, fp16 in and fp16 out, bf16 weight, no residual.
///
/// Fork: `rms_norm_kernel<0, half, half, bfloat16, half>` (`RES_NONE`), the
/// arm `exllamav3_ext/norm.cu` takes when `x` and `y` are fp16 and the weight
/// is bf16. MTP `pre_fc_norm_hidden` and `pre_fc_norm_embedding` are
/// constructed with `out_dtype=torch.half` and `constant_bias=1`
/// (`modules/arch_specific/qwen3_5_mtp.py`). The activation they see is
/// already fp16 (the target post-norm export, and the embedding cast).
/// `w_groups` is 1. This does not change [`rms_norm`], which stays float in.
///
/// # Errors
///
/// A null pointer, non-positive `rows` or `dim`, or `dim` not a multiple of
/// 4 returns [`Error::Launch`] before CUDA.
///
/// # Safety
///
/// `x` and `y` are fp16, `rows` by `dim`. `weight` is bf16, one row of
/// `dim`. They stay alive until `stream` completes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn rms_norm_f16(
    x: *const c_void,
    weight: *const c_void,
    y: *mut c_void,
    rows: i32,
    dim: i32,
    epsilon: f32,
    constant_bias: f32,
    constant_scale: f32,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe {
        ffi::berth_ops_rms_norm_f16(
            x,
            weight,
            y,
            rows,
            dim,
            epsilon,
            constant_bias,
            constant_scale,
            stream,
        )
    };
    if rc == 0 {
        return Ok(());
    }
    Err(Error::Launch(error_message(rc)))
}

/// Gather embedding rows and round BF16 to fp16.
///
/// MTP calls the target embedding with `out_dtype=torch.half`
/// (`modules/arch_specific/qwen3_5_mtp.py`). `nn.Embedding` returns the bf16
/// row, then `to2` (`util/tensor.py`) is `.to(torch.half)`. Widening bf16 to
/// fp32 is exact, and fp32 to fp16 is round-to-nearest-even, so
/// `__float2half(__bfloat162float(v))` equals that cast. Decode keeps
/// [`embed_tokens`] (fp32). `ids` must already be in range.
///
/// # Errors
///
/// A null pointer, or non-positive `hidden` or `n`, returns [`Error::Launch`]
/// before CUDA.
///
/// # Safety
///
/// `weight` is BF16 row-major with stride `hidden` and covers every id.
/// `ids` has `n` `i32` elements. `out` is fp16 `[n, hidden]`. All three stay
/// alive until `stream` completes.
pub unsafe fn embed_tokens_f16(
    weight: *const c_void,
    ids: *const i32,
    out: *mut c_void,
    hidden: i32,
    n: i32,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe { ffi::berth_ops_embed_tokens_f16(weight, ids, out, hidden, n, stream) };
    if rc == 0 {
        return Ok(());
    }
    Err(Error::Launch(error_message(rc)))
}

/// `cudaMemcpyAsync` of `bytes` from `src` to `dst`, device to device.
///
/// Ordered on `stream`. The regions must not overlap. Used to build the MTP
/// shifted hidden (`cat(carry, export[:-1])`) without a host round trip.
///
/// # Errors
///
/// A null pointer or `bytes == 0` returns [`Error::Launch`] before CUDA.
///
/// # Safety
///
/// `src` and `dst` are device allocations of at least `bytes` and do not
/// overlap. They stay alive until `stream` completes.
pub unsafe fn memcpy_d2d(
    dst: *mut c_void,
    src: *const c_void,
    bytes: u64,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe { ffi::berth_ops_memcpy_d2d(dst, src, bytes, stream) };
    if rc == 0 {
        return Ok(());
    }
    Err(Error::Launch(error_message(rc)))
}

/// `out = cat((embedding, hidden), dim=-1)` in fp16.
///
/// `embedding` and `hidden` are `[rows, hidden_size]`. `out` is
/// `[rows, 2 * hidden_size]`, embedding in the first half. This is
/// `torch.cat((x, y), dim=-1)` in `Qwen3_5MTPInputLayer.forward`.
///
/// # Errors
///
/// A null pointer, or non-positive `rows` or `hidden_size`, returns
/// [`Error::Launch`] before CUDA.
///
/// # Safety
///
/// The three buffers are fp16, do not overlap, and stay alive until
/// `stream` completes.
pub unsafe fn cat_f16_last(
    embedding: *const c_void,
    hidden: *const c_void,
    out: *mut c_void,
    rows: i32,
    hidden_size: i32,
    stream: *mut c_void,
) -> Result<()> {
    let rc =
        unsafe { ffi::berth_ops_cat_f16_last(embedding, hidden, out, rows, hidden_size, stream) };
    if rc == 0 {
        return Ok(());
    }
    Err(Error::Launch(error_message(rc)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mgemm_tune_keys_match_the_27b_cache_entries() {
        // device 0, CC_AMPERE = 2, 82 SMs. m = 1 stays 1. Salted forms of
        // these keys are shape-4 records in coop_autotune_v1.bin
        // (block 256, 10 SMs, concurrency 8).
        let gdn = mgemm_tune_key(
            1,
            5120,
            2048,
            4,
            OutDtype::F32,
            0,
            2,
            82,
            Codebook::Mul1,
            1,
            8,
            true,
        );
        assert_eq!(gdn, 0xd229_7a8a_33a3_3f5b);
        let attn = mgemm_tune_key(
            1,
            5120,
            1024,
            4,
            OutDtype::F16,
            0,
            2,
            82,
            Codebook::Mul1,
            1,
            14,
            true,
        );
        assert_eq!(attn, 0x9c25_a075_2178_fb4a);
    }

    #[test]
    fn rejected_launches_do_not_call_cuda() {
        let err = unsafe {
            exl3_mgemm(
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null_mut(),
                std::ptr::null(),
                1,
                5120,
                2048,
                4,
                Codebook::Mul1,
                OutDtype::F32,
                1,
                8,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
            )
        }
        .expect_err("null");
        assert!(err.to_string().contains("null"), "{err}");
        assert!(last_launch().is_empty());

        let p = std::ptr::dangling::<c_void>();
        let err = unsafe {
            exl3_mgemm(
                p,
                p,
                p as *mut c_void,
                p,
                p as *mut c_void,
                p,
                1,
                5120,
                2048,
                6,
                Codebook::Mul1,
                OutDtype::F32,
                1,
                8,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
            )
        }
        .expect_err("k");
        assert!(err.to_string().contains("no linked kernel"), "{err}");

        let err = unsafe {
            exl3_mgemm(
                p,
                p,
                p as *mut c_void,
                p,
                p as *mut c_void,
                p,
                1,
                15,
                2048,
                4,
                Codebook::Mul1,
                OutDtype::F16,
                1,
                1,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
            )
        }
        .expect_err("size_k");
        assert!(err.to_string().contains("multiple of 16"), "{err}");

        let err = unsafe {
            rms_norm(
                std::ptr::null(),
                p,
                p as *mut c_void,
                1,
                5120,
                1e-6,
                1.0,
                1.0,
                NormWeight::Bf16,
                std::ptr::null_mut(),
            )
        }
        .expect_err("norm null");
        assert!(err.to_string().contains("null"), "{err}");

        let err = unsafe {
            rms_norm(
                p,
                p,
                p as *mut c_void,
                1,
                6,
                1e-6,
                1.0,
                1.0,
                NormWeight::Bf16,
                std::ptr::null_mut(),
            )
        }
        .expect_err("dim");
        assert!(err.to_string().contains("multiple of 4"), "{err}");

        let err = unsafe { silu_mul(p, p, p as *mut c_void, 3, 0.0, std::ptr::null_mut()) }
            .expect_err("silu");
        assert!(err.to_string().contains("even"), "{err}");

        let err =
            unsafe { add_f32(p, p, p as *mut c_void, 0, std::ptr::null_mut()) }.expect_err("add");
        assert!(err.to_string().contains("positive"), "{err}");

        let err = unsafe {
            embed_tokens(
                p,
                std::ptr::null(),
                p as *mut c_void,
                5120,
                1,
                std::ptr::null_mut(),
            )
        }
        .expect_err("embed");
        assert!(err.to_string().contains("null"), "{err}");
    }
}

//! Full-attention prefill for one chunk. bsz 1, cq3, `EXL3_QC_STAGING=1`.
//!
//! The fork path this follows, for `q_len > 16` so the eager prefill runs
//! rather than the q_len-1 decode cubins:
//!
//! * `deinterleave_qg` (`activation.cu` `deinterleave_qg_kernel`, called from
//!   `modules/attn.py` `project_qkv` when the interleaved gate is on).
//! * `rope` with fused bf16 q/k norms (`rope.cu` `rope_kernel<2,1>`).
//!   `prepare_flash_attn` sets `positions = cache_seqlens` when a block table
//!   is present (`modules/attn.py`).
//! * `quant_cache_paged` (`cache/quant.py` `update_kv_direct`, before attention,
//!   `modules/attention_fn/dispatch.py`).
//! * `paged_attn_triton_prefill` (`modules/attention_fn/triton_paged.py`).
//!   `q_len >= 256` dequantizes the referenced pages into an fp16 scratch
//!   (`dequant_cache_paged_window`, `q_cache.cu`) and runs the fp16 kernel.
//!   Shorter chunks dequantize inside `_paged_attn_prefill_kernel`.
//! * `mul_sigmoid_` on the attention output (`modules/attn.py` `decode_flash_attn`).
//!
//! Q, K, and V projections are not in this module. The caller passes the
//! projection outputs: interleaved `qg`, and `k` / `v` before RoPE.
//!
//! The cubins are the specializations captured by
//! `scripts/build_triton_cubins.py`. Triton specializes every integer runtime
//! argument of `_paged_attn_prefill_kernel`: value 1 becomes a constexpr,
//! a multiple of 16 is a distinct cubin (`tt.divisibility = 16`), and any
//! other value is a plain i32. [`prefill_cubin_key`] is that rule. The
//! captured product is the qc path and the fp16 staging path over the q_len
//! and page-count classes this call can take (`kv_append_len` equals
//! `q_len`, both windows are -1). `num_splits` is 1 below the 8192-token
//! bound and 2..=8 above it, so the split class is always `other`.
//! `num_splits > 1` also launches `_paged_attn_prefill_combine_kernel`
//! ([`combine_cubin_key`]) before the output gate. A shape whose class was
//! not captured is an error.

use std::ffi::{c_void, CStr};
use std::path::Path;

use berth_cuda::{Context, Cubin, Dim3, Function, Stream};

use crate::attn::{
    deinterleave_qg, mul_sigmoid, quant_cache_paged, rope, QuantCacheLaunch, RopeLaunch,
    TRITON_CUBIN_BUILD,
};
use crate::{Error, Result};

/// Query heads on the 27B.
pub const Q_HEADS: i32 = 24;
/// Key/value heads on the 27B.
pub const KV_HEADS: i32 = 4;
/// Head dimension on the 27B.
pub const HEAD_DIM: i32 = 256;
/// `PAGE_SIZE` in the fork's `constants.py`.
pub const PAGE_SIZE: i32 = 256;
/// Prefill `BLOCK_M` for `head_dim` 256 (`triton_paged.py` tile table).
pub const BLOCK_M: i32 = 64;
/// `EXL3_QC_PF_TWO_PASS_MIN_Q`. At or above this, prefill stages through fp16.
pub const TWO_PASS_MIN_Q: i32 = 256;
/// `n_kv_heads * head_dim`.
pub const TOKEN_DIM: i32 = KV_HEADS * HEAD_DIM;
/// RoPE rotary span. `partial_rotary_factor` 0.25 of `head_dim`.
pub const ROTARY_DIM: i32 = 64;
/// RMSNorm epsilon on the 27B.
pub const NORM_EPS: f32 = 1e-6;
/// `RMSNorm.constant_bias` on the q/k norms.
pub const NORM_BIAS: f32 = 1.0;

/// Ampere's default dynamic shared-memory cap. Above this, load opts in.
const DEFAULT_DYNAMIC_SMEM: u32 = 48 * 1024;

/// Integer runtime arguments of `_paged_attn_prefill_kernel`, definition order.
const SPECIALIZED_INTS: &[&str] = &[
    "num_splits",
    "q_len",
    "kv_append_len",
    "num_pages_per_seq",
    "WINDOW_LEFT",
    "WINDOW_RIGHT",
];

const PREFILL_PARAMS: &[&str] = &[
    "q",
    "k_cache",
    "v_cache",
    "block_table",
    "cache_seqlens",
    "out",
    "partial_o",
    "partial_ml",
    "k_scales",
    "v_scales",
    "h32",
    "k_new",
    "v_new",
    "sinks",
    "num_splits",
    "q_len",
    "kv_append_len",
    "num_pages_per_seq",
    "WINDOW_LEFT",
    "WINDOW_RIGHT",
    "global_scratch",
    "profile_scratch",
];

/// Runtime arguments of `_paged_attn_prefill_combine_kernel`, definition order.
/// Triton appends the two scratch pointers. `num_splits` and `q_len` specialize
/// the same way as the prefill kernel; the other integers are constexprs.
const COMBINE_PARAMS: &[&str] = &[
    "partial_o",
    "partial_ml",
    "out",
    "h32",
    "sinks",
    "num_splits",
    "q_len",
    "global_scratch",
    "profile_scratch",
];

const COMBINE_INTS: &[&str] = &["num_splits", "q_len"];

/// fp32 partial buffer cap in `paged_attn_triton_prefill` (`128 * 1024 * 1024`).
pub const PARTIAL_CAP_BYTES: usize = 128 * 1024 * 1024;

/// Triton's specialization class for one integer runtime argument.
///
/// `eq1` is a constexpr (the parameter is omitted). `div16` keeps the i32 and
/// sets `tt.divisibility = 16`. `other` is a plain i32. The `%` is Python's:
/// a negative multiple of 16 is `div16`, and `-1` is `other`.
#[must_use]
pub fn int_class(value: i32) -> &'static str {
    if value == 1 {
        "eq1"
    } else if value.rem_euclid(16) == 0 {
        "div16"
    } else {
        "other"
    }
}

/// Manifest key of the cubin this call would JIT.
///
/// `staging` is the fp16 path (`q_len >= 256`). The six integers are
/// `num_splits`, `q_len`, `kv_append_len`, `num_pages_per_seq`,
/// `WINDOW_LEFT`, and `WINDOW_RIGHT`, in that order, each reduced with
/// [`int_class`].
#[must_use]
pub fn prefill_cubin_key(
    staging: bool,
    num_splits: i32,
    q_len: i32,
    kv_append_len: i32,
    num_pages_per_seq: i32,
    window_left: i32,
    window_right: i32,
) -> String {
    let path = if staging { "fp16" } else { "qc" };
    format!(
        "_paged_attn_prefill_kernel_{path}_ns-{}_q-{}_kv-{}_p-{}_wl-{}_wr-{}",
        int_class(num_splits),
        int_class(q_len),
        int_class(kv_append_len),
        int_class(num_pages_per_seq),
        int_class(window_left),
        int_class(window_right),
    )
}

/// Manifest key of the combine cubin this split call would JIT.
///
/// `staging` is the fp16 path (`QCV == 0`). The qc path has `QCV == 3`.
/// The two integers are `num_splits` and `q_len`. On the split path
/// `num_splits` is 2..=8, so its class is `other`.
#[must_use]
pub fn combine_cubin_key(staging: bool, num_splits: i32, q_len: i32) -> String {
    let path = if staging { "fp16" } else { "qc" };
    format!(
        "_paged_attn_prefill_combine_kernel_{path}_ns-{}_q-{}",
        int_class(num_splits),
        int_class(q_len),
    )
}

/// Bytes of the fp32 partial buffers for one split launch, `(partial_o, partial_ml)`.
///
/// Shapes from `paged_attn_triton_prefill`: `programs * num_splits * BLOCK_M * HD_PAD`
/// and `programs * num_splits * BLOCK_M * 2`, with `programs = cdiv(q_len, BLOCK_M) * n_q_heads`
/// at batch 1. `partial_o` above [`PARTIAL_CAP_BYTES`] is an error; the split
/// formula already caps the count so a real launch stays inside it.
///
/// # Errors
///
/// `q_len` or `num_splits` is not positive, or `partial_o` would exceed the cap.
pub fn partial_bytes(q_len: i32, num_splits: i32) -> Result<(usize, usize)> {
    if q_len < 1 || num_splits < 1 {
        return Err(Error::Launch(
            "partial_bytes: q_len and num_splits must be positive".into(),
        ));
    }
    let q_blocks = ceil_div_i64(i64::from(q_len), i64::from(BLOCK_M))?;
    let programs = q_blocks * i64::from(Q_HEADS);
    let splits = i64::from(num_splits);
    let tile = i64::from(BLOCK_M);
    let o_elems = programs
        .checked_mul(splits)
        .and_then(|n| n.checked_mul(tile))
        .and_then(|n| n.checked_mul(i64::from(HEAD_DIM)))
        .ok_or_else(|| Error::Launch("partial_o element count overflow".into()))?;
    let ml_elems = programs
        .checked_mul(splits)
        .and_then(|n| n.checked_mul(tile))
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| Error::Launch("partial_ml element count overflow".into()))?;
    let o_bytes = usize::try_from(o_elems)
        .ok()
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| Error::Launch("partial_o byte count overflow".into()))?;
    let ml_bytes = usize::try_from(ml_elems)
        .ok()
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| Error::Launch("partial_ml byte count overflow".into()))?;
    if o_bytes > PARTIAL_CAP_BYTES {
        return Err(Error::Launch(format!(
            "partial_o is {o_bytes} bytes, above the {PARTIAL_CAP_BYTES}-byte cap"
        )));
    }
    Ok((o_bytes, ml_bytes))
}

/// True when `paged_attn_triton_prefill` dequantizes into an fp16 scratch.
#[must_use]
pub fn needs_staging(q_len: i32) -> bool {
    q_len >= TWO_PASS_MIN_Q
}

/// Scratch pages allocated by the staging path.
///
/// `max(1, 1 << (bsz * bt_width - 1).bit_length())` in `triton_paged.py`.
/// The kernel's block table is still `bt_width` wide; the extra pages are
/// the allocator's power-of-two rounding and are not indexed.
///
/// # Errors
///
/// `bsz` or `bt_width` is not positive, or the page count does not fit in `i32`.
pub fn staging_pages(bsz: i32, bt_width: i32) -> Result<i32> {
    if bsz < 1 || bt_width < 1 {
        return Err(Error::Launch(
            "staging_pages: bsz and bt_width must be positive".into(),
        ));
    }
    let bsz_u = u32::try_from(bsz).map_err(|_| Error::Launch("staging_pages: bsz".into()))?;
    let width_u =
        u32::try_from(bt_width).map_err(|_| Error::Launch("staging_pages: bt_width".into()))?;
    let n = bsz_u
        .checked_mul(width_u)
        .ok_or_else(|| Error::Launch("staging_pages: product overflow".into()))?;
    if n <= 1 {
        return Ok(1);
    }
    let shift = (n - 1).ilog2() + 1;
    let pages = 1u32
        .checked_shl(shift)
        .ok_or_else(|| Error::Launch("staging_pages: shift overflow".into()))?;
    i32::try_from(pages).map_err(|_| Error::Launch("staging_pages does not fit in i32".into()))
}

/// fp16 elements in one staging scratch (`k` or `v`).
///
/// Shape `(staging_pages, 256, n_kv_heads, head_dim)`.
///
/// # Errors
///
/// [`staging_pages`] fails, or the product does not fit in `usize`.
pub fn staging_scratch_elems(bsz: i32, bt_width: i32) -> Result<usize> {
    let pages = i64::from(staging_pages(bsz, bt_width)?);
    let n = pages * i64::from(PAGE_SIZE) * i64::from(KV_HEADS) * i64::from(HEAD_DIM);
    usize::try_from(n).map_err(|_| Error::Launch("staging scratch does not fit in usize".into()))
}

/// Inputs to the split-count formula in `paged_attn_triton_prefill`.
#[derive(Clone, Copy, Debug)]
pub struct SplitShape {
    /// Chunk length.
    pub q_len: i32,
    /// Block-table width (`num_pages_per_seq`).
    pub bt_width: i32,
    /// Page length. 256.
    pub page_size: i32,
    /// Rows appended before attention. Equal to `q_len` on this path.
    pub kv_append_len: i32,
    /// `BLOCK_M`.
    pub block_m: i32,
    /// Query heads.
    pub n_q_heads: i32,
    /// Batch. 1 for this module's launch.
    pub bsz: i32,
    /// Padded head dimension. 256.
    pub hd_pad: i32,
    /// Device SM count.
    pub sm_count: i32,
}

/// `num_splits` from `paged_attn_triton_prefill`.
///
/// 1 unless `bt_width * page_size + kv_append_len >= 8192`. Above that, the
/// candidate in `(1, 2, 3, 4, 5, 6, 8)` with the smallest
/// `ceil(programs * cand / sms) / cand + 0.03 * (cand - 1)` wins, then the
/// 128 MiB partial-buffer cap applies. Ties keep the earlier candidate.
///
/// # Errors
///
/// An argument is out of range, or the result does not fit in `i32`.
pub fn prefill_num_splits(shape: &SplitShape) -> Result<i32> {
    let s = shape;
    if s.q_len < 1
        || s.bt_width < 1
        || s.page_size < 1
        || s.kv_append_len < 0
        || s.block_m < 1
        || s.n_q_heads < 1
        || s.bsz < 1
        || s.hd_pad < 1
        || s.sm_count < 1
    {
        return Err(Error::Launch(
            "prefill_num_splits: arguments out of range".into(),
        ));
    }
    let bound = i64::from(s.bt_width) * i64::from(s.page_size) + i64::from(s.kv_append_len);
    let q_blocks = ceil_div_i64(i64::from(s.q_len), i64::from(s.block_m))?;
    let programs = q_blocks * i64::from(s.bsz) * i64::from(s.n_q_heads);
    let mut num_splits = 1i64;
    if bound >= 8192 && programs > 0 {
        let sms = i64::from(s.sm_count);
        let mut best: Option<(f64, i64)> = None;
        for cand in [1i64, 2, 3, 4, 5, 6, 8] {
            let waves = (programs * cand + sms - 1) / sms;
            let cost = (waves as f64) / (cand as f64) + 0.03 * ((cand - 1) as f64);
            let replace = match best {
                None => true,
                Some((prev, _)) => cost < prev,
            };
            if replace {
                best = Some((cost, cand));
            }
        }
        num_splits = best
            .ok_or_else(|| Error::Launch("prefill_num_splits: no candidate".into()))?
            .1;
        let max_partial_bytes = 128i64 * 1024 * 1024;
        let denom = programs * i64::from(s.block_m) * i64::from(s.hd_pad) * 4;
        if denom <= 0 {
            return Err(Error::Launch(
                "prefill_num_splits: partial buffer size overflow".into(),
            ));
        }
        let max_splits = 1i64.max(max_partial_bytes / denom);
        num_splits = num_splits.min(max_splits);
    }
    i32::try_from(num_splits)
        .map_err(|_| Error::Launch("prefill_num_splits does not fit in i32".into()))
}

/// True when the wrapper would set `WIDE_INDEX` (`triton_paged.py`).
#[must_use]
pub fn wide_index(shape: &SplitShape, num_splits: i32) -> bool {
    let q_blocks = if shape.block_m < 1 {
        0
    } else {
        (i64::from(shape.q_len) + i64::from(shape.block_m) - 1) / i64::from(shape.block_m)
    };
    let q_elems = i64::from(shape.bsz)
        * i64::from(shape.q_len)
        * i64::from(shape.n_q_heads)
        * i64::from(HEAD_DIM);
    let partial = q_blocks
        * i64::from(shape.bsz)
        * i64::from(shape.n_q_heads)
        * i64::from(num_splits)
        * i64::from(shape.block_m)
        * i64::from(shape.hd_pad);
    q_elems.max(partial) >= (1i64 << 31)
}

/// Device pointers for [`PagedPrefill::prefill_chunk`].
///
/// Every address is a device pointer ([`berth_cuda::DevicePtr::as_u64`]).
/// Null is 0. Layouts are contiguous, bsz 1, fp16 unless noted.
///
/// * `qg`: `(q_len, n_q_heads * 2 * head_dim)`, interleaved `[q | g]` per head.
/// * `k`, `v`: `(q_len, n_kv_heads, head_dim)`, before RoPE. RoPE writes `k`
///   in place. `v` is not rotated.
/// * `q`, `g`: deinterleave outputs. `q` is `(q_len, n_q_heads, head_dim)`.
///   `g` is `(q_len, n_q_heads * head_dim)`. RoPE writes `q` in place.
/// * `o`: attention output, same shape as `q`. The gate is applied in place.
/// * `k_cache`, `v_cache`: cq3 pages, int32, `(num_pages, 256, 96)`.
/// * `k_scales`, `v_scales`: fp16, `(num_pages, 256, 32)`.
/// * `block_table`: int32, `(1, bt_width)`, the real page map.
/// * `cache_seqlens`: int32, length 1. The value is the position of the first
///   new token. RoPE reads it as `positions`. It must equal the host
///   `past_len` passed to the launch.
/// * `inv_freq`: fp32, length 32.
/// * `q_norm`, `k_norm`: bf16, length `head_dim`.
/// * `h32`: fp16, `(32, 32)`, Sylvester Hadamard over `sqrt(32)`.
/// * `scratch_k`, `scratch_v`: fp16 staging buffers,
///   [`staging_scratch_elems`] elements each. Required when `q_len >= 256`.
/// * `identity_table`: int32 `[0, 1, …, bt_width)`. Required when staging.
///   The attention kernel's table width stays `bt_width`, not the rounded
///   scratch page count.
/// * `global_scratch`, `profile_scratch`: Triton scratch, in bytes, sized by
///   [`PagedPrefill::scratch_bytes`]. Pass 0 when that size is 0.
/// * `partial_o`, `partial_ml`: fp32 split buffers, [`partial_bytes`] each.
///   Required when [`prefill_num_splits`] is greater than 1. Pass 0 when the
///   launch is unsplit; the kernel then receives `q` and does not read them.
#[derive(Clone, Copy, Debug)]
pub struct PrefillChunk {
    /// Interleaved query and gate, fp16.
    pub qg: u64,
    /// Key, fp16, before RoPE.
    pub k: u64,
    /// Value, fp16.
    pub v: u64,
    /// Deinterleaved query, fp16. RoPE output and attention query.
    pub q: u64,
    /// Deinterleaved gate, fp16.
    pub g: u64,
    /// Attention output, fp16. Gate writes this in place.
    pub o: u64,
    /// Packed K pages, int32.
    pub k_cache: u64,
    /// Packed V pages, int32.
    pub v_cache: u64,
    /// K scales, fp16.
    pub k_scales: u64,
    /// V scales, fp16.
    pub v_scales: u64,
    /// Real page map, int32.
    pub block_table: u64,
    /// Past length, int32, length 1. Also the RoPE `positions` tensor.
    pub cache_seqlens: u64,
    /// Inverse frequencies, fp32, length 32.
    pub inv_freq: u64,
    /// bf16 query RMSNorm weight, length 256.
    pub q_norm: u64,
    /// bf16 key RMSNorm weight, length 256.
    pub k_norm: u64,
    /// 32×32 Hadamard, fp16.
    pub h32: u64,
    /// fp16 K scratch for staging. 0 when `q_len < 256`.
    pub scratch_k: u64,
    /// fp16 V scratch for staging. 0 when `q_len < 256`.
    pub scratch_v: u64,
    /// Identity block table, int32. 0 when `q_len < 256`.
    pub identity_table: u64,
    /// Triton global scratch. 0 when the cubin asks for none.
    pub global_scratch: u64,
    /// Triton profile scratch. 0 when the cubin asks for none.
    pub profile_scratch: u64,
    /// fp32 partial outputs. 0 when `num_splits` is 1.
    pub partial_o: u64,
    /// fp32 partial max/sum. 0 when `num_splits` is 1.
    pub partial_ml: u64,
}

/// Arguments for [`dequant_cache_paged_window`].
///
/// Matches `dequant_cache_paged_window` (`q_cache.cu`). `bonus_len` is the
/// pre-appended chunk. `compact_out` is fixed at 1 and `sliding_window` at
/// -1 inside the launcher. `k_bits` and `v_bits` must be 3.
pub struct DequantWindow {
    /// Packed K pages, int32.
    pub k_in: *const c_void,
    /// K scales, fp16.
    pub k_scales: *const c_void,
    /// Compact fp16 K scratch.
    pub k_out: *mut c_void,
    /// Packed V pages, int32.
    pub v_in: *const c_void,
    /// V scales, fp16.
    pub v_scales: *const c_void,
    /// Compact fp16 V scratch.
    pub v_out: *mut c_void,
    /// Past length, int32, length `bsz`.
    pub cache_seqlens: *const c_void,
    /// Page map, int32, `(bsz, pages_per_seq)`.
    pub block_table: *const c_void,
    /// Batch.
    pub bsz: i32,
    /// Block-table width.
    pub pages_per_seq: i32,
    /// `n_kv_heads * head_dim`.
    pub token_dim: i32,
    /// K bitrate. Must be 3.
    pub k_bits: i32,
    /// V bitrate. Must be 3.
    pub v_bits: i32,
    /// Page length. Must be 256.
    pub page_size: i32,
    /// Rows past `cache_seqlens` to include.
    pub bonus_len: i32,
    /// Compander coefficient. 0 on the 27B.
    pub compand_a: f32,
}

/// Dequantize the block table's window into a compact fp16 scratch.
///
/// # Errors
///
/// A null pointer, a page size other than 256, a bitrate other than 3, or a
/// CUDA failure.
///
/// # Safety
///
/// Every pointer must be a device allocation of the size implied by
/// [`DequantWindow`] and must stay allocated until `stream` finishes.
/// `k_out` must hold at least `bsz * pages_per_seq` pages.
pub unsafe fn dequant_cache_paged_window(args: &DequantWindow, stream: *mut c_void) -> Result<()> {
    let rc = unsafe {
        ffi::berth_attn_prefill_dequant_window(
            args.k_in,
            args.k_scales,
            args.k_out,
            args.v_in,
            args.v_scales,
            args.v_out,
            args.cache_seqlens,
            args.block_table,
            args.bsz,
            args.pages_per_seq,
            args.token_dim,
            args.k_bits,
            args.v_bits,
            args.page_size,
            args.bonus_len,
            args.compand_a,
            stream,
        )
    };
    fail(rc, "dequant_cache_paged_window")
}

/// Every prefill cubin the build script captured, keyed by
/// [`prefill_cubin_key`], plus the combine cubins keyed by
/// [`combine_cubin_key`].
pub struct PagedPrefill {
    images: Vec<PrefillImage>,
    combine: Vec<CombineImage>,
    sm_count: i32,
}

struct PrefillImage {
    /// [`prefill_cubin_key`] for the class this cubin was captured with.
    key: String,
    /// Fp16 staging path (`QCK == 0`). The qc path dequantizes in the kernel.
    staging: bool,
    function: Function,
    name: String,
    shared: u32,
    num_warps: u32,
    params: Vec<Param>,
    global_scratch: u32,
    profile_scratch: u32,
    /// `num_splits` when Triton specialized it away. `None` when it is a launch argument.
    baked_splits: Option<i32>,
    /// `num_pages_per_seq` when Triton specialized it away.
    baked_pages: Option<i32>,
}

struct CombineImage {
    /// [`combine_cubin_key`] for the class this cubin was captured with.
    key: String,
    /// Fp16 staging path (`QCV == 0`). The qc path has `QCV == 3`.
    staging: bool,
    function: Function,
    name: String,
    shared: u32,
    num_warps: u32,
    params: Vec<Param>,
    global_scratch: u32,
    profile_scratch: u32,
    /// `num_splits` when Triton specialized it away.
    baked_splits: Option<i32>,
    /// `q_len` when Triton specialized it away.
    baked_q_len: Option<i32>,
}

struct Param {
    name: String,
    kind: ParamKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ParamKind {
    Ptr,
    I32,
}

impl PagedPrefill {
    /// Load every prefill class from `dir` (`manifest.json` plus one cubin each).
    ///
    /// # Errors
    ///
    /// A missing manifest or cubin, a class key that does not match the
    /// recorded specialization, a constexpr that is not the 27B tile, a
    /// parameter list that is not the prefill launch order, or a driver error.
    /// The missing-file error includes [`TRITON_CUBIN_BUILD`].
    pub fn load(ctx: &Context, dir: &Path) -> Result<Self> {
        let manifest_path = dir.join("manifest.json");
        let text = std::fs::read_to_string(&manifest_path).map_err(|_| {
            Error::Launch(format!(
                "prefill attention cubins are missing at {}. Run: {TRITON_CUBIN_BUILD}",
                manifest_path.display()
            ))
        })?;
        let manifest = parse_json(&text).map_err(|err| {
            Error::Launch(format!(
                "prefill attention manifest {}: {err}",
                manifest_path.display()
            ))
        })?;
        let kernels = manifest
            .get("kernels")
            .ok_or_else(|| Error::Launch("manifest has no kernels object".into()))?;
        let classes = manifest
            .get("prefill")
            .and_then(|prefill| prefill.get("classes"))
            .and_then(Json::as_array)
            .ok_or_else(|| Error::Launch("manifest prefill.classes is missing".into()))?;
        if classes.is_empty() {
            return Err(Error::Launch("manifest prefill.classes is empty".into()));
        }
        let mut images = Vec::with_capacity(classes.len());
        for item in classes {
            let key = item.as_str().ok_or_else(|| {
                Error::Launch("manifest prefill.classes entry is not a string".into())
            })?;
            images.push(load_image(ctx, dir, kernels, key)?);
        }
        let qc = images.iter().any(|image| !image.staging);
        let staged = images.iter().any(|image| image.staging);
        if !qc || !staged {
            return Err(Error::Launch(
                "prefill manifest is missing the qc path or the fp16 staging path".into(),
            ));
        }
        let combine_classes = manifest
            .get("prefill")
            .and_then(|prefill| prefill.get("combine_classes"))
            .and_then(Json::as_array)
            .ok_or_else(|| Error::Launch("manifest prefill.combine_classes is missing".into()))?;
        if combine_classes.is_empty() {
            return Err(Error::Launch(
                "manifest prefill.combine_classes is empty".into(),
            ));
        }
        let mut combine = Vec::with_capacity(combine_classes.len());
        for item in combine_classes {
            let key = item.as_str().ok_or_else(|| {
                Error::Launch("manifest prefill.combine_classes entry is not a string".into())
            })?;
            combine.push(load_combine(ctx, dir, kernels, key)?);
        }
        let qc_combine = combine.iter().any(|image| !image.staging);
        let staged_combine = combine.iter().any(|image| image.staging);
        if !qc_combine || !staged_combine {
            return Err(Error::Launch(
                "prefill manifest is missing a combine cubin for the qc path or the fp16 path"
                    .into(),
            ));
        }
        let sms = ctx.multiprocessor_count().map_err(cuda_err)?;
        let sm_count = i32::try_from(sms)
            .map_err(|_| Error::Launch("multiprocessor count does not fit in i32".into()))?;
        if sm_count < 1 {
            return Err(Error::Launch("multiprocessor count is 0".into()));
        }
        Ok(Self {
            images,
            combine,
            sm_count,
        })
    }

    /// `num_splits` for this device, from [`prefill_num_splits`].
    ///
    /// # Errors
    ///
    /// [`prefill_num_splits`] fails.
    pub fn num_splits(&self, q_len: i32, bt_width: i32) -> Result<i32> {
        prefill_num_splits(&SplitShape {
            q_len,
            bt_width,
            page_size: PAGE_SIZE,
            kv_append_len: q_len,
            block_m: BLOCK_M,
            n_q_heads: Q_HEADS,
            bsz: 1,
            hd_pad: HEAD_DIM,
            sm_count: self.sm_count,
        })
    }

    /// Triton scratch bytes `(global, profile)` for this `q_len`.
    ///
    /// The larger of every prefill cubin and every combine cubin on that path
    /// (qc when `q_len < 256`, fp16 staging otherwise). Both are 0 for the
    /// captured 27B cubins. A nonzero value is the buffer the launch reads
    /// from [`PrefillChunk::global_scratch`] and
    /// [`PrefillChunk::profile_scratch`].
    #[must_use]
    pub fn scratch_bytes(&self, q_len: i32) -> (u32, u32) {
        let staging = needs_staging(q_len);
        let mut global = 0u32;
        let mut profile = 0u32;
        for image in &self.images {
            if image.staging != staging {
                continue;
            }
            global = global.max(image.global_scratch);
            profile = profile.max(image.profile_scratch);
        }
        for image in &self.combine {
            if image.staging != staging {
                continue;
            }
            global = global.max(image.global_scratch);
            profile = profile.max(image.profile_scratch);
        }
        (global, profile)
    }

    /// Run one prefill chunk and write the gated output and the cq3 cache.
    ///
    /// `past_len` is the host copy of `cache_seqlens[0]`. `past_len + q_len`
    /// must fit in `bt_width` pages. bsz is 1.
    ///
    /// # Errors
    ///
    /// The chunk does not fit, no cubin was captured for this argument class,
    /// the cubin would need `WIDE_INDEX`, a required pointer is null, or a
    /// launch fails. The missing-class error names the key from
    /// [`prefill_cubin_key`].
    ///
    /// # Safety
    ///
    /// Every non-zero pointer in `args` must address device memory of the size
    /// described on [`PrefillChunk`] and must stay allocated until `stream`
    /// finishes. `cache_seqlens[0]` must equal `past_len`. The cache pages
    /// outside this chunk are left as they were. Staging scratch may be
    /// overwritten.
    pub unsafe fn prefill_chunk(
        &self,
        stream: &Stream,
        args: &PrefillChunk,
        q_len: i32,
        bt_width: i32,
        past_len: i32,
    ) -> Result<()> {
        if q_len < 1 || bt_width < 1 || past_len < 0 {
            return Err(Error::Launch(
                "prefill: q_len and bt_width must be positive and past_len non-negative".into(),
            ));
        }
        let end = i64::from(past_len) + i64::from(q_len);
        let cap = i64::from(bt_width) * i64::from(PAGE_SIZE);
        if end > cap {
            return Err(Error::Launch(format!(
                "prefill: past_len {past_len} + q_len {q_len} does not fit in {bt_width} pages"
            )));
        }
        require_ptr(args.qg, "qg")?;
        require_ptr(args.k, "k")?;
        require_ptr(args.v, "v")?;
        require_ptr(args.q, "q")?;
        require_ptr(args.g, "g")?;
        require_ptr(args.o, "o")?;
        require_ptr(args.k_cache, "k_cache")?;
        require_ptr(args.v_cache, "v_cache")?;
        require_ptr(args.k_scales, "k_scales")?;
        require_ptr(args.v_scales, "v_scales")?;
        require_ptr(args.block_table, "block_table")?;
        require_ptr(args.cache_seqlens, "cache_seqlens")?;
        require_ptr(args.inv_freq, "inv_freq")?;
        require_ptr(args.q_norm, "q_norm")?;
        require_ptr(args.k_norm, "k_norm")?;
        require_ptr(args.h32, "h32")?;

        let shape = SplitShape {
            q_len,
            bt_width,
            page_size: PAGE_SIZE,
            kv_append_len: q_len,
            block_m: BLOCK_M,
            n_q_heads: Q_HEADS,
            bsz: 1,
            hd_pad: HEAD_DIM,
            sm_count: self.sm_count,
        };
        let splits = prefill_num_splits(&shape)?;
        if wide_index(&shape, splits) {
            return Err(Error::Launch(
                "prefill cubin is the WIDE_INDEX=false specialization".into(),
            ));
        }
        // kv_append_len is q_len: k is None and the new rows are already in
        // the cache. Windows stay -1 (full attention, HAS_WINDOW_* false).
        let image = self.select(q_len, q_len, bt_width, splits, -1, -1)?;
        if let Some(pages) = image.baked_pages {
            if bt_width != pages {
                return Err(Error::Launch(format!(
                    "prefill cubin {} baked num_pages_per_seq={pages}; this block table is {bt_width} wide",
                    image.key
                )));
            }
        }
        if let Some(n) = image.baked_splits {
            if splits != n {
                return Err(Error::Launch(format!(
                    "prefill cubin {} baked num_splits={n}; this shape wants {splits} splits",
                    image.key
                )));
            }
        }

        let raw = stream.handle().as_ptr();
        let q_numel = i64::from(q_len) * i64::from(Q_HEADS) * i64::from(HEAD_DIM);
        unsafe {
            deinterleave_qg(
                args.qg as *const c_void,
                args.q as *mut c_void,
                args.g as *mut c_void,
                HEAD_DIM,
                q_numel,
                raw,
            )?;
            rope(&rope_launch(args, q_len), raw)?;
            quant_cache_paged(&quant_launch(args, q_len, bt_width), raw)?;
        }

        let staging = needs_staging(q_len);
        let (k_cache, v_cache, table, scales_k, scales_v, h32) = if staging {
            require_ptr(args.scratch_k, "scratch_k")?;
            require_ptr(args.scratch_v, "scratch_v")?;
            require_ptr(args.identity_table, "identity_table")?;
            unsafe {
                dequant_cache_paged_window(&dequant_launch(args, q_len, bt_width), raw)?;
            }
            (
                args.scratch_k,
                args.scratch_v,
                args.identity_table,
                args.q,
                args.q,
                args.q,
            )
        } else {
            (
                args.k_cache,
                args.v_cache,
                args.block_table,
                args.k_scales,
                args.v_scales,
                args.h32,
            )
        };

        // Unsplit passes `q` for both partials. The cubin was captured that
        // way (`IS_SPLIT` false) and does not load them. A split stores fp32
        // partials and the combine kernel reads them before the gate.
        let (partial_o, partial_ml) = if splits > 1 {
            require_ptr(args.partial_o, "partial_o")?;
            require_ptr(args.partial_ml, "partial_ml")?;
            (args.partial_o, args.partial_ml)
        } else {
            (args.q, args.q)
        };
        let mut runtime = PrefillRuntime {
            q: args.q,
            k_cache,
            v_cache,
            block_table: table,
            cache_seqlens: args.cache_seqlens,
            out: args.o,
            partial_o,
            partial_ml,
            k_scales: scales_k,
            v_scales: scales_v,
            h32,
            k_new: args.q,
            v_new: args.q,
            sinks: args.q,
            num_splits: splits,
            q_len,
            kv_append_len: q_len,
            num_pages_per_seq: bt_width,
            window_left: -1,
            window_right: -1,
            global_scratch: 0,
            profile_scratch: 0,
        };
        runtime.global_scratch =
            scratch_ptr(image.global_scratch, args.global_scratch, "global_scratch")?;
        runtime.profile_scratch = scratch_ptr(
            image.profile_scratch,
            args.profile_scratch,
            "profile_scratch",
        )?;
        let q_blocks = ceil_div_i64(i64::from(q_len), i64::from(BLOCK_M))?;
        let grid_x = u32_fit(q_blocks, "q_blocks")?;
        let grid_y = u32_fit(i64::from(Q_HEADS), "grid_y")?;
        let grid_z = u32_fit(i64::from(splits), "num_splits")?;
        launch_image(image, stream, &runtime, Dim3::new(grid_x, grid_y, grid_z))?;
        if splits > 1 {
            let combine = self.select_combine(staging, splits, q_len)?;
            if let Some(n) = combine.baked_splits {
                if splits != n {
                    return Err(Error::Launch(format!(
                        "prefill combine cubin {} baked num_splits={n}; this shape wants {splits}",
                        combine.key
                    )));
                }
            }
            if let Some(n) = combine.baked_q_len {
                if q_len != n {
                    return Err(Error::Launch(format!(
                        "prefill combine cubin {} baked q_len={n}; this chunk is {q_len}",
                        combine.key
                    )));
                }
            }
            runtime.global_scratch = scratch_ptr(
                combine.global_scratch,
                args.global_scratch,
                "global_scratch",
            )?;
            runtime.profile_scratch = scratch_ptr(
                combine.profile_scratch,
                args.profile_scratch,
                "profile_scratch",
            )?;
            launch_listed(
                &combine.name,
                &combine.function,
                combine.num_warps,
                combine.shared,
                &combine.params,
                stream,
                &runtime,
                Dim3::new(grid_x, grid_y, 1),
            )?;
        }
        unsafe { mul_sigmoid(args.o as *mut c_void, args.g as *const c_void, q_numel, raw)? };
        Ok(())
    }

    fn select(
        &self,
        q_len: i32,
        kv_append_len: i32,
        bt_width: i32,
        splits: i32,
        window_left: i32,
        window_right: i32,
    ) -> Result<&PrefillImage> {
        let key = prefill_cubin_key(
            needs_staging(q_len),
            splits,
            q_len,
            kv_append_len,
            bt_width,
            window_left,
            window_right,
        );
        self.images
            .iter()
            .find(|image| image.key == key)
            .ok_or_else(|| {
                Error::Launch(format!(
                    "prefill cubin {key} is missing \
                     (q_len={q_len}, kv_append_len={kv_append_len}, \
                     num_pages_per_seq={bt_width}, num_splits={splits}, \
                     WINDOW_LEFT={window_left}, WINDOW_RIGHT={window_right}). \
                     Triton class is eq1 when the value is 1, div16 when it is \
                     divisible by 16, and other otherwise. Run: {TRITON_CUBIN_BUILD}"
                ))
            })
    }

    fn select_combine(&self, staging: bool, splits: i32, q_len: i32) -> Result<&CombineImage> {
        let key = combine_cubin_key(staging, splits, q_len);
        self.combine
            .iter()
            .find(|image| image.key == key)
            .ok_or_else(|| {
                Error::Launch(format!(
                    "prefill combine cubin {key} is missing \
                     (q_len={q_len}, num_splits={splits}). \
                     Run: {TRITON_CUBIN_BUILD}"
                ))
            })
    }
}

fn rope_launch(args: &PrefillChunk, q_len: i32) -> RopeLaunch {
    RopeLaunch {
        q: args.q as *const c_void,
        out_q: args.q as *mut c_void,
        k: args.k as *const c_void,
        out_k: args.k as *mut c_void,
        inv_freq: args.inv_freq as *const c_void,
        bsz: 1,
        seq_len: q_len,
        num_heads_q: Q_HEADS,
        num_heads_k: KV_HEADS,
        head_dim: HEAD_DIM,
        q_head_stride: HEAD_DIM,
        k_head_stride: HEAD_DIM,
        partial_head_dim: ROTARY_DIM,
        position: 0,
        positions: args.cache_seqlens as *const c_void,
        position_ids: std::ptr::null(),
        rope_mode: 2,
        attn_factor: 1.0,
        q_norm: args.q_norm as *const c_void,
        k_norm: args.k_norm as *const c_void,
        norm_eps: NORM_EPS,
        norm_constant_bias: NORM_BIAS,
        inv_freq_table: false,
        inv_freq_stride: 0,
        llama_4_scaling_beta: 0.0,
        llama_4_scaling_original: 0,
        position_ids_stride: 1,
        rotate_dims: 1,
        rotate_offset: 0,
    }
}

fn quant_launch(args: &PrefillChunk, q_len: i32, bt_width: i32) -> QuantCacheLaunch {
    QuantCacheLaunch {
        k_in: args.k as *const c_void,
        k_out: args.k_cache as *mut c_void,
        k_scales: args.k_scales as *mut c_void,
        v_in: args.v as *const c_void,
        v_out: args.v_cache as *mut c_void,
        v_scales: args.v_scales as *mut c_void,
        cache_seqlens: args.cache_seqlens as *const c_void,
        block_table: args.block_table as *const c_void,
        bsz: 1,
        seq_len: q_len,
        blocks_per_seq: bt_width,
        token_dim: TOKEN_DIM,
        k_bits: 3,
        v_bits: 3,
        page_size: PAGE_SIZE,
        compand_a: 0.0,
        in_contiguous: true,
    }
}

fn dequant_launch(args: &PrefillChunk, q_len: i32, bt_width: i32) -> DequantWindow {
    DequantWindow {
        k_in: args.k_cache as *const c_void,
        k_scales: args.k_scales as *const c_void,
        k_out: args.scratch_k as *mut c_void,
        v_in: args.v_cache as *const c_void,
        v_scales: args.v_scales as *const c_void,
        v_out: args.scratch_v as *mut c_void,
        cache_seqlens: args.cache_seqlens as *const c_void,
        block_table: args.block_table as *const c_void,
        bsz: 1,
        pages_per_seq: bt_width,
        token_dim: TOKEN_DIM,
        k_bits: 3,
        v_bits: 3,
        page_size: PAGE_SIZE,
        bonus_len: q_len,
        compand_a: 0.0,
    }
}

fn require_ptr(ptr: u64, what: &str) -> Result<()> {
    if ptr == 0 {
        Err(Error::Launch(format!("prefill: {what} is null")))
    } else {
        Ok(())
    }
}

fn scratch_ptr(bytes: u32, ptr: u64, what: &str) -> Result<u64> {
    if bytes == 0 {
        Ok(0)
    } else if ptr == 0 {
        Err(Error::Launch(format!(
            "prefill: {what} is null and the cubin asks for {bytes} bytes"
        )))
    } else {
        Ok(ptr)
    }
}

struct PrefillRuntime {
    q: u64,
    k_cache: u64,
    v_cache: u64,
    block_table: u64,
    cache_seqlens: u64,
    out: u64,
    partial_o: u64,
    partial_ml: u64,
    k_scales: u64,
    v_scales: u64,
    h32: u64,
    k_new: u64,
    v_new: u64,
    sinks: u64,
    num_splits: i32,
    q_len: i32,
    kv_append_len: i32,
    num_pages_per_seq: i32,
    window_left: i32,
    window_right: i32,
    global_scratch: u64,
    profile_scratch: u64,
}

fn launch_image(
    image: &PrefillImage,
    stream: &Stream,
    runtime: &PrefillRuntime,
    grid: Dim3,
) -> Result<()> {
    launch_listed(
        &image.name,
        &image.function,
        image.num_warps,
        image.shared,
        &image.params,
        stream,
        runtime,
        grid,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_listed(
    name: &str,
    function: &Function,
    num_warps: u32,
    shared: u32,
    params: &[Param],
    stream: &Stream,
    runtime: &PrefillRuntime,
    grid: Dim3,
) -> Result<()> {
    let mut slots = Vec::with_capacity(params.len());
    for param in params {
        slots.push(slot(param, runtime)?);
    }
    let mut argv: Vec<*mut c_void> = slots
        .iter_mut()
        .map(|item| match item.kind {
            ParamKind::Ptr => &mut item.ptr as *mut u64 as *mut c_void,
            ParamKind::I32 => &mut item.scalar as *mut i32 as *mut c_void,
        })
        .collect();
    let block = Dim3::new(32 * num_warps, 1, 1);
    unsafe { function.launch(stream, grid, block, shared, &mut argv) }
        .map_err(|err| Error::Launch(format!("{name}: {err}")))
}

struct Slot {
    ptr: u64,
    scalar: i32,
    kind: ParamKind,
}

fn slot(param: &Param, runtime: &PrefillRuntime) -> Result<Slot> {
    let ptr_of = |value: u64| Slot {
        ptr: value,
        scalar: 0,
        kind: ParamKind::Ptr,
    };
    let i32_of = |value: i32| Slot {
        ptr: 0,
        scalar: value,
        kind: ParamKind::I32,
    };
    let item = match param.name.as_str() {
        "global_scratch" => ptr_of(runtime.global_scratch),
        "profile_scratch" => ptr_of(runtime.profile_scratch),
        "q" => ptr_of(runtime.q),
        "k_cache" => ptr_of(runtime.k_cache),
        "v_cache" => ptr_of(runtime.v_cache),
        "block_table" => ptr_of(runtime.block_table),
        "cache_seqlens" => ptr_of(runtime.cache_seqlens),
        "out" => ptr_of(runtime.out),
        "partial_o" => ptr_of(runtime.partial_o),
        "partial_ml" => ptr_of(runtime.partial_ml),
        "k_scales" => ptr_of(runtime.k_scales),
        "v_scales" => ptr_of(runtime.v_scales),
        "h32" => ptr_of(runtime.h32),
        "k_new" => ptr_of(runtime.k_new),
        "v_new" => ptr_of(runtime.v_new),
        "sinks" => ptr_of(runtime.sinks),
        "num_splits" => i32_of(runtime.num_splits),
        "q_len" => i32_of(runtime.q_len),
        "kv_append_len" => i32_of(runtime.kv_append_len),
        "num_pages_per_seq" => i32_of(runtime.num_pages_per_seq),
        "WINDOW_LEFT" => i32_of(runtime.window_left),
        "WINDOW_RIGHT" => i32_of(runtime.window_right),
        other => {
            return Err(Error::Launch(format!(
                "manifest parameter {other} is not a prefill argument"
            )))
        }
    };
    if item.kind != param.kind {
        return Err(Error::Launch(format!(
            "manifest parameter {} has the wrong type",
            param.name
        )));
    }
    Ok(item)
}

fn load_image(ctx: &Context, dir: &Path, kernels: &Json, key: &str) -> Result<PrefillImage> {
    let record = kernels
        .get(key)
        .ok_or_else(|| Error::Launch(format!("manifest has no {key}")))?;
    let file = record
        .get("cubin")
        .and_then(Json::as_str)
        .ok_or_else(|| Error::Launch(format!("{key}: cubin filename missing")))?;
    let symbol = record
        .get("name")
        .and_then(Json::as_str)
        .ok_or_else(|| Error::Launch(format!("{key}: symbol name missing")))?;
    let shared = record
        .get("shared")
        .and_then(Json::as_u32)
        .ok_or_else(|| Error::Launch(format!("{key}: shared missing")))?;
    let num_warps = record
        .get("num_warps")
        .and_then(Json::as_u32)
        .ok_or_else(|| Error::Launch(format!("{key}: num_warps missing")))?;
    let num_stages = record
        .get("num_stages")
        .and_then(Json::as_i64)
        .ok_or_else(|| Error::Launch(format!("{key}: num_stages missing")))?;
    if num_warps != 8 || num_stages != 2 {
        return Err(Error::Launch(format!(
            "{key}: expected num_warps 8 and num_stages 2, got {num_warps} and {num_stages}"
        )));
    }
    let global_scratch = scratch_size(record, key, "global_scratch_size")?;
    let profile_scratch = scratch_size(record, key, "profile_scratch_size")?;
    let consts = record
        .get("constexprs")
        .ok_or_else(|| Error::Launch(format!("{key}: constexprs missing")))?;
    let is_split = consts
        .get("IS_SPLIT")
        .and_then(Json::as_bool)
        .ok_or_else(|| Error::Launch(format!("{key}: constexpr IS_SPLIT missing")))?;
    let qck = need_i32(consts, "QCK")?;
    let (block_n, cache_ty, staging) = match qck {
        3 => (64, "*i32", false),
        0 => (32, "*fp16", true),
        other => {
            return Err(Error::Launch(format!(
                "{key}: constexpr QCK is {other}, want 3 (qc) or 0 (fp16 staging)"
            )))
        }
    };
    check_constexprs(consts, key, qck, block_n, is_split)?;
    let spec = specialization(record, key)?;
    let built = key_from_classes(staging, &spec);
    if built != key {
        return Err(Error::Launch(format!(
            "{key}: specialization does not match the class key ({built})"
        )));
    }
    let (params, baked_splits, baked_pages) = param_list(record, key, cache_ty, consts)?;
    check_spec_against_params(&spec, &params, consts, key)?;
    let partial_ty = if is_split { "*fp32" } else { "*fp16" };
    check_named_type(record, key, "partial_o", partial_ty)?;
    check_named_type(record, key, "partial_ml", partial_ty)?;
    // The wrapper sets IS_SPLIT from `num_splits > 1` and bakes 1 as a constexpr.
    if is_split {
        if baked_splits.is_some() {
            return Err(Error::Launch(format!(
                "{key}: IS_SPLIT is true but num_splits was specialized to 1"
            )));
        }
    } else if baked_splits != Some(1) {
        return Err(Error::Launch(format!(
            "{key}: IS_SPLIT is false but num_splits is not the constexpr 1"
        )));
    }
    let path = dir.join(file);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) if !bytes.is_empty() => bytes,
        _ => {
            return Err(Error::Launch(format!(
                "prefill cubin is missing at {}. Run: {TRITON_CUBIN_BUILD}",
                path.display()
            )))
        }
    };
    let cubin = Cubin::load_bytes(ctx, &bytes).map_err(cuda_err)?;
    let function = cubin.function(symbol).map_err(cuda_err)?;
    if shared > DEFAULT_DYNAMIC_SMEM {
        function
            .set_max_dynamic_shared_bytes(shared)
            .map_err(cuda_err)?;
    }
    Ok(PrefillImage {
        key: key.to_string(),
        staging,
        function,
        name: symbol.to_string(),
        shared,
        num_warps,
        params,
        global_scratch,
        profile_scratch,
        baked_splits,
        baked_pages,
    })
}

fn load_combine(ctx: &Context, dir: &Path, kernels: &Json, key: &str) -> Result<CombineImage> {
    let record = kernels
        .get(key)
        .ok_or_else(|| Error::Launch(format!("manifest has no {key}")))?;
    let file = record
        .get("cubin")
        .and_then(Json::as_str)
        .ok_or_else(|| Error::Launch(format!("{key}: cubin filename missing")))?;
    let symbol = record
        .get("name")
        .and_then(Json::as_str)
        .ok_or_else(|| Error::Launch(format!("{key}: symbol name missing")))?;
    let shared = record
        .get("shared")
        .and_then(Json::as_u32)
        .ok_or_else(|| Error::Launch(format!("{key}: shared missing")))?;
    let num_warps = record
        .get("num_warps")
        .and_then(Json::as_u32)
        .ok_or_else(|| Error::Launch(format!("{key}: num_warps missing")))?;
    let num_stages = record
        .get("num_stages")
        .and_then(Json::as_i64)
        .ok_or_else(|| Error::Launch(format!("{key}: num_stages missing")))?;
    if num_warps != 8 || num_stages != 1 {
        return Err(Error::Launch(format!(
            "{key}: expected num_warps 8 and num_stages 1, got {num_warps} and {num_stages}"
        )));
    }
    let global_scratch = scratch_size(record, key, "global_scratch_size")?;
    let profile_scratch = scratch_size(record, key, "profile_scratch_size")?;
    let consts = record
        .get("constexprs")
        .ok_or_else(|| Error::Launch(format!("{key}: constexprs missing")))?;
    let qcv = need_i32(consts, "QCV")?;
    let staging = match qcv {
        0 => true,
        3 => false,
        other => {
            return Err(Error::Launch(format!(
                "{key}: constexpr QCV is {other}, want 3 (qc) or 0 (fp16 staging)"
            )))
        }
    };
    check_combine_constexprs(consts, key, qcv)?;
    let spec = combine_specialization(record, key)?;
    let built = combine_key_from_classes(staging, &spec);
    if built != key {
        return Err(Error::Launch(format!(
            "{key}: specialization does not match the class key ({built})"
        )));
    }
    if spec
        .iter()
        .any(|(name, kind)| *name == "num_splits" && kind == "eq1")
    {
        return Err(Error::Launch(format!(
            "{key}: combine cubin specialized num_splits to 1; the wrapper only launches it when num_splits > 1"
        )));
    }
    let (params, baked_splits, baked_q_len) = combine_param_list(record, key, consts)?;
    check_spec_against_params(&spec, &params, consts, key)?;
    check_named_type(record, key, "partial_o", "*fp32")?;
    check_named_type(record, key, "partial_ml", "*fp32")?;
    for name in ["out", "h32", "sinks"] {
        check_named_type(record, key, name, "*fp16")?;
    }
    let path = dir.join(file);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) if !bytes.is_empty() => bytes,
        _ => {
            return Err(Error::Launch(format!(
                "prefill combine cubin is missing at {}. Run: {TRITON_CUBIN_BUILD}",
                path.display()
            )))
        }
    };
    let cubin = Cubin::load_bytes(ctx, &bytes).map_err(cuda_err)?;
    let function = cubin.function(symbol).map_err(cuda_err)?;
    if shared > DEFAULT_DYNAMIC_SMEM {
        function
            .set_max_dynamic_shared_bytes(shared)
            .map_err(cuda_err)?;
    }
    Ok(CombineImage {
        key: key.to_string(),
        staging,
        function,
        name: symbol.to_string(),
        shared,
        num_warps,
        params,
        global_scratch,
        profile_scratch,
        baked_splits,
        baked_q_len,
    })
}

fn key_from_classes(staging: bool, classes: &[(&'static str, String)]) -> String {
    let path = if staging { "fp16" } else { "qc" };
    let mut out = format!("_paged_attn_prefill_kernel_{path}");
    for (tag, name) in [
        ("ns", "num_splits"),
        ("q", "q_len"),
        ("kv", "kv_append_len"),
        ("p", "num_pages_per_seq"),
        ("wl", "WINDOW_LEFT"),
        ("wr", "WINDOW_RIGHT"),
    ] {
        let kind = classes
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, kind)| kind.as_str())
            .unwrap_or("missing");
        out.push('_');
        out.push_str(tag);
        out.push('-');
        out.push_str(kind);
    }
    out
}

fn combine_key_from_classes(staging: bool, classes: &[(&'static str, String)]) -> String {
    let path = if staging { "fp16" } else { "qc" };
    let kind = |name: &str| {
        classes
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, kind)| kind.as_str())
            .unwrap_or("missing")
    };
    format!(
        "_paged_attn_prefill_combine_kernel_{path}_ns-{}_q-{}",
        kind("num_splits"),
        kind("q_len"),
    )
}

fn combine_specialization(record: &Json, key: &str) -> Result<Vec<(&'static str, String)>> {
    let obj = record
        .get("specialization")
        .ok_or_else(|| Error::Launch(format!("{key}: specialization missing")))?;
    let mut out = Vec::with_capacity(COMBINE_INTS.len());
    for name in COMBINE_INTS {
        let kind = obj
            .get(name)
            .and_then(Json::as_str)
            .ok_or_else(|| Error::Launch(format!("{key}: specialization {name} missing")))?;
        if !matches!(kind, "eq1" | "div16" | "other") {
            return Err(Error::Launch(format!(
                "{key}: specialization {name} is {kind}, want eq1, div16, or other"
            )));
        }
        out.push((*name, kind.to_string()));
    }
    Ok(out)
}

fn combine_param_list(
    record: &Json,
    key: &str,
    consts: &Json,
) -> Result<(Vec<Param>, Option<i32>, Option<i32>)> {
    let list = record
        .get("parameters")
        .and_then(Json::as_array)
        .ok_or_else(|| Error::Launch(format!("{key}: parameters missing")))?;
    let mut index = 0usize;
    let mut baked_splits = None;
    let mut baked_q_len = None;
    let mut out = Vec::with_capacity(list.len());
    for expected in COMBINE_PARAMS {
        let matches_next = list
            .get(index)
            .and_then(|item| item.get("name"))
            .and_then(Json::as_str)
            == Some(*expected);
        if !matches_next {
            if !COMBINE_INTS.contains(expected) {
                return Err(Error::Launch(format!(
                    "{key}: manifest is missing parameter {expected}"
                )));
            }
            let v = need_i32(consts, expected)?;
            if v != 1 {
                return Err(Error::Launch(format!(
                    "{key}: baked {expected} is {v}; Triton only specializes the value 1 to a constexpr"
                )));
            }
            match *expected {
                "num_splits" => baked_splits = Some(v),
                "q_len" => baked_q_len = Some(v),
                _ => {}
            }
            continue;
        }
        let item = &list[index];
        let ty = item
            .get("type")
            .and_then(Json::as_str)
            .ok_or_else(|| Error::Launch(format!("{key}: parameter {expected} has no type")))?;
        let kind = if ty.starts_with('*') {
            ParamKind::Ptr
        } else if ty == "i32" {
            ParamKind::I32
        } else {
            return Err(Error::Launch(format!(
                "{key}: parameter {expected} has unsupported type {ty}"
            )));
        };
        out.push(Param {
            name: (*expected).to_string(),
            kind,
        });
        index += 1;
    }
    if index != list.len() {
        return Err(Error::Launch(format!(
            "{key}: manifest has {} parameters, the combine signature uses {index}",
            list.len()
        )));
    }
    Ok((out, baked_splits, baked_q_len))
}

fn check_combine_constexprs(consts: &Json, key: &str, qcv: i32) -> Result<()> {
    let want_i32 = [
        ("QCV", qcv),
        ("n_q_heads", Q_HEADS),
        ("head_dim", HEAD_DIM),
        ("HD_PAD", HEAD_DIM),
        ("BLOCK_M", BLOCK_M),
    ];
    for (name, want) in want_i32 {
        let got = need_i32(consts, name)?;
        if got != want {
            return Err(Error::Launch(format!(
                "{key}: constexpr {name} is {got}, want {want}"
            )));
        }
    }
    for name in ["HAS_SINKS", "WIDE_INDEX"] {
        let got = consts
            .get(name)
            .and_then(Json::as_bool)
            .ok_or_else(|| Error::Launch(format!("{key}: constexpr {name} missing")))?;
        if got {
            return Err(Error::Launch(format!(
                "{key}: constexpr {name} is true, want false"
            )));
        }
    }
    Ok(())
}

fn specialization(record: &Json, key: &str) -> Result<Vec<(&'static str, String)>> {
    let obj = record
        .get("specialization")
        .ok_or_else(|| Error::Launch(format!("{key}: specialization missing")))?;
    let mut out = Vec::with_capacity(SPECIALIZED_INTS.len());
    for name in SPECIALIZED_INTS {
        let kind = obj
            .get(name)
            .and_then(Json::as_str)
            .ok_or_else(|| Error::Launch(format!("{key}: specialization {name} missing")))?;
        if !matches!(kind, "eq1" | "div16" | "other") {
            return Err(Error::Launch(format!(
                "{key}: specialization {name} is {kind}, want eq1, div16, or other"
            )));
        }
        out.push((*name, kind.to_string()));
    }
    Ok(out)
}

fn check_spec_against_params(
    spec: &[(&'static str, String)],
    params: &[Param],
    consts: &Json,
    key: &str,
) -> Result<()> {
    for (name, kind) in spec {
        let present = params.iter().any(|param| param.name == *name);
        let baked = consts.get(name).and_then(Json::as_i64);
        match kind.as_str() {
            "eq1" => {
                if present || baked != Some(1) {
                    return Err(Error::Launch(format!(
                        "{key}: {name} is eq1 but present={present} baked={baked:?}"
                    )));
                }
            }
            "div16" | "other" => {
                if !present || baked.is_some() {
                    return Err(Error::Launch(format!(
                        "{key}: {name} is {kind} but present={present} baked={baked:?}"
                    )));
                }
            }
            other => {
                return Err(Error::Launch(format!(
                    "{key}: specialization {name} is {other}"
                )))
            }
        }
    }
    Ok(())
}

fn scratch_size(record: &Json, key: &str, field: &str) -> Result<u32> {
    let size = record
        .get(field)
        .and_then(Json::as_i64)
        .ok_or_else(|| Error::Launch(format!("{key}: {field} missing")))?;
    if size < 0 {
        return Err(Error::Launch(format!("{key}: {field} is {size}")));
    }
    u32::try_from(size).map_err(|_| Error::Launch(format!("{key}: {field} does not fit in u32")))
}

fn check_named_type(record: &Json, key: &str, name: &str, want: &str) -> Result<()> {
    let list = record
        .get("parameters")
        .and_then(Json::as_array)
        .ok_or_else(|| Error::Launch(format!("{key}: parameters missing")))?;
    let got = list
        .iter()
        .find(|item| item.get("name").and_then(Json::as_str) == Some(name))
        .and_then(|item| item.get("type"))
        .and_then(Json::as_str)
        .ok_or_else(|| Error::Launch(format!("{key}: parameter {name} missing")))?;
    if got != want {
        return Err(Error::Launch(format!(
            "{key}: parameter {name} type {got} != {want}"
        )));
    }
    Ok(())
}

fn check_constexprs(
    consts: &Json,
    key: &str,
    qck: i32,
    block_n: i32,
    is_split: bool,
) -> Result<()> {
    let want_i32 = [
        ("NEW_KV", 0),
        ("QCK", qck),
        ("QCV", qck),
        ("n_q_heads", Q_HEADS),
        ("n_kv_heads", KV_HEADS),
        ("page_size", PAGE_SIZE),
        ("head_dim", HEAD_DIM),
        ("HD_PAD", HEAD_DIM),
        ("BLOCK_M", BLOCK_M),
        ("BLOCK_N", block_n),
    ];
    for (name, want) in want_i32 {
        let got = need_i32(consts, name)?;
        if got != want {
            return Err(Error::Launch(format!(
                "{key}: constexpr {name} is {got}, want {want}"
            )));
        }
    }
    let want_bool = [
        ("IS_SPLIT", is_split),
        ("CAUSAL", true),
        ("HAS_WINDOW_LEFT", false),
        ("HAS_WINDOW_RIGHT", false),
        ("HAS_SINKS", false),
        ("WIDE_INDEX", false),
    ];
    for (name, want) in want_bool {
        let got = consts
            .get(name)
            .and_then(Json::as_bool)
            .ok_or_else(|| Error::Launch(format!("{key}: constexpr {name} missing")))?;
        if got != want {
            return Err(Error::Launch(format!(
                "{key}: constexpr {name} is {got}, want {want}"
            )));
        }
    }
    let scale = need_f64(consts, "scale")?;
    if scale != 0.0625 {
        return Err(Error::Launch(format!(
            "{key}: constexpr scale is {scale}, want 0.0625"
        )));
    }
    let softcap = need_f64(consts, "SOFTCAP")?;
    if softcap != 0.0 {
        return Err(Error::Launch(format!(
            "{key}: constexpr SOFTCAP is {softcap}, want 0"
        )));
    }
    Ok(())
}

fn param_list(
    record: &Json,
    key: &str,
    cache_ty: &str,
    consts: &Json,
) -> Result<(Vec<Param>, Option<i32>, Option<i32>)> {
    // Triton drops an integer argument from the launch signature when its value
    // is 1 and bakes that 1 as a constexpr. div16 and other both stay i32
    // parameters; the cubin file, not the parameter list, tells them apart.
    // A pointer or scratch argument is never specialized away.
    let list = record
        .get("parameters")
        .and_then(Json::as_array)
        .ok_or_else(|| Error::Launch(format!("{key}: parameters missing")))?;
    let mut index = 0usize;
    let mut baked_splits = None;
    let mut baked_pages = None;
    let mut out = Vec::with_capacity(list.len());
    for expected in PREFILL_PARAMS {
        let matches_next = list
            .get(index)
            .and_then(|item| item.get("name"))
            .and_then(Json::as_str)
            == Some(*expected);
        if !matches_next {
            if !SPECIALIZED_INTS.contains(expected) {
                return Err(Error::Launch(format!(
                    "{key}: manifest is missing parameter {expected}"
                )));
            }
            let v = need_i32(consts, expected)?;
            if v != 1 {
                return Err(Error::Launch(format!(
                    "{key}: baked {expected} is {v}; Triton only specializes the value 1 to a constexpr"
                )));
            }
            match *expected {
                "num_splits" => baked_splits = Some(v),
                "num_pages_per_seq" => baked_pages = Some(v),
                _ => {}
            }
            continue;
        }
        let item = &list[index];
        let ty = item
            .get("type")
            .and_then(Json::as_str)
            .ok_or_else(|| Error::Launch(format!("{key}: parameter {expected} has no type")))?;
        if matches!(*expected, "k_cache" | "v_cache") && ty != cache_ty {
            return Err(Error::Launch(format!(
                "{key}: {expected} type {ty} != {cache_ty}"
            )));
        }
        let kind = if ty.starts_with('*') {
            ParamKind::Ptr
        } else if ty == "i32" {
            ParamKind::I32
        } else {
            return Err(Error::Launch(format!(
                "{key}: parameter {expected} has unsupported type {ty}"
            )));
        };
        out.push(Param {
            name: (*expected).to_string(),
            kind,
        });
        index += 1;
    }
    if index != list.len() {
        return Err(Error::Launch(format!(
            "{key}: manifest has {} parameters, the prefill signature uses {index}",
            list.len()
        )));
    }
    Ok((out, baked_splits, baked_pages))
}

fn need_i32(obj: &Json, key: &str) -> Result<i32> {
    let v = obj
        .get(key)
        .and_then(Json::as_i64)
        .ok_or_else(|| Error::Launch(format!("constexpr {key} missing")))?;
    i32::try_from(v).map_err(|_| Error::Launch(format!("constexpr {key} does not fit in i32")))
}

fn need_f64(obj: &Json, key: &str) -> Result<f64> {
    obj.get(key)
        .and_then(Json::as_f64)
        .ok_or_else(|| Error::Launch(format!("constexpr {key} missing")))
}

fn ceil_div_i64(n: i64, d: i64) -> Result<i64> {
    if d < 1 || n < 0 {
        return Err(Error::Launch("ceil_div: bad arguments".into()));
    }
    Ok((n + d - 1) / d)
}

fn u32_fit(v: i64, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::Launch(format!("{what} does not fit in u32")))
}

fn fail(rc: i32, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let message = last_error_c();
    let message = if message.is_empty() {
        format!("{what} launch failed ({rc})")
    } else {
        message
    };
    Err(Error::Launch(message))
}

fn last_error_c() -> String {
    let ptr = unsafe { ffi::berth_attn_last_error() };
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn cuda_err(err: berth_cuda::Error) -> Error {
    Error::Launch(err.to_string())
}

mod ffi {
    use std::ffi::{c_char, c_void};

    #[allow(clippy::too_many_arguments)]
    unsafe extern "C" {
        pub fn berth_attn_last_error() -> *const c_char;

        pub fn berth_attn_prefill_dequant_window(
            k_in: *const c_void,
            k_scales: *const c_void,
            k_out: *mut c_void,
            v_in: *const c_void,
            v_scales: *const c_void,
            v_out: *mut c_void,
            cache_seqlens: *const c_void,
            block_table: *const c_void,
            bsz: i32,
            pages_per_seq: i32,
            token_dim: i32,
            k_bits: i32,
            v_bits: i32,
            page_size: i32,
            bonus_len: i32,
            compand_a: f32,
            stream: *mut c_void,
        ) -> i32;
    }
}

// Manifest JSON. `json.dumps` output: objects, arrays, strings, numbers, bools, null.

struct Json {
    kind: JsonKind,
}

enum JsonKind {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match &self.kind {
            JsonKind::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match &self.kind {
            JsonKind::String(s) => Some(s),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[Json]> {
        match &self.kind {
            JsonKind::Array(items) => Some(items),
            _ => None,
        }
    }

    fn as_bool(&self) -> Option<bool> {
        match &self.kind {
            JsonKind::Bool(v) => Some(*v),
            _ => None,
        }
    }

    fn as_i64(&self) -> Option<i64> {
        match &self.kind {
            JsonKind::Number(v)
                if v.fract() == 0.0 && *v >= i64::MIN as f64 && *v <= i64::MAX as f64 =>
            {
                Some(*v as i64)
            }
            _ => None,
        }
    }

    fn as_u32(&self) -> Option<u32> {
        self.as_i64().and_then(|v| u32::try_from(v).ok())
    }

    fn as_f64(&self) -> Option<f64> {
        match &self.kind {
            JsonKind::Number(v) => Some(*v),
            _ => None,
        }
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

fn parse_json(text: &str) -> std::result::Result<Json, String> {
    let mut p = Parser {
        s: text.as_bytes(),
        i: 0,
    };
    let value = p.value()?;
    p.skip();
    if p.i != p.s.len() {
        return Err("trailing data".into());
    }
    Ok(value)
}

impl<'a> Parser<'a> {
    fn skip(&mut self) {
        while let Some(b) = self.s.get(self.i) {
            if b.is_ascii_whitespace() {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    fn value(&mut self) -> std::result::Result<Json, String> {
        self.skip();
        let b = self.peek().ok_or_else(|| "unexpected end".to_string())?;
        match b {
            b'n' => self.lit(b"null", JsonKind::Null),
            b't' => self.lit(b"true", JsonKind::Bool(true)),
            b'f' => self.lit(b"false", JsonKind::Bool(false)),
            b'"' => Ok(Json {
                kind: JsonKind::String(self.string()?),
            }),
            b'[' => self.array(),
            b'{' => self.object(),
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(format!("unexpected byte {b} at {}", self.i)),
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn lit(&mut self, word: &[u8], kind: JsonKind) -> std::result::Result<Json, String> {
        if self.s[self.i..].starts_with(word) {
            self.i += word.len();
            Ok(Json { kind })
        } else {
            Err(format!("expected {}", String::from_utf8_lossy(word)))
        }
    }

    fn string(&mut self) -> std::result::Result<String, String> {
        self.i += 1;
        let mut out = String::new();
        while let Some(b) = self.peek() {
            self.i += 1;
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    let e = self.peek().ok_or_else(|| "truncated escape".to_string())?;
                    self.i += 1;
                    match e {
                        b'"' | b'\\' | b'/' => out.push(e as char),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            if self.i + 4 > self.s.len() {
                                return Err("truncated unicode escape".into());
                            }
                            let hex = std::str::from_utf8(&self.s[self.i..self.i + 4])
                                .map_err(|_| "unicode escape is not utf-8".to_string())?;
                            let code = u32::from_str_radix(hex, 16)
                                .map_err(|_| format!("bad unicode escape {hex}"))?;
                            out.push(
                                char::from_u32(code).ok_or_else(|| "bad code point".to_string())?,
                            );
                            self.i += 4;
                        }
                        _ => return Err(format!("bad escape {e}")),
                    }
                }
                _ => out.push(b as char),
            }
        }
        Err("unterminated string".into())
    }

    fn array(&mut self) -> std::result::Result<Json, String> {
        self.i += 1;
        let mut items = Vec::new();
        self.skip();
        if self.peek() == Some(b']') {
            self.i += 1;
            return Ok(Json {
                kind: JsonKind::Array(items),
            });
        }
        loop {
            items.push(self.value()?);
            self.skip();
            match self.peek() {
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json {
                        kind: JsonKind::Array(items),
                    });
                }
                Some(b',') => self.i += 1,
                _ => return Err("expected ',' or ']'".into()),
            }
        }
    }

    fn object(&mut self) -> std::result::Result<Json, String> {
        self.i += 1;
        let mut pairs = Vec::new();
        self.skip();
        if self.peek() == Some(b'}') {
            self.i += 1;
            return Ok(Json {
                kind: JsonKind::Object(pairs),
            });
        }
        loop {
            self.skip();
            if self.peek() != Some(b'"') {
                return Err("expected object key".into());
            }
            let key = self.string()?;
            self.skip();
            if self.peek() != Some(b':') {
                return Err("expected ':'".into());
            }
            self.i += 1;
            pairs.push((key, self.value()?));
            self.skip();
            match self.peek() {
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json {
                        kind: JsonKind::Object(pairs),
                    });
                }
                Some(b',') => self.i += 1,
                _ => return Err("expected ',' or '}'".into()),
            }
        }
    }

    fn number(&mut self) -> std::result::Result<Json, String> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.i += 1;
        }
        if self.peek() == Some(b'.') {
            self.i += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.i += 1;
            }
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).map_err(|_| "number".to_string())?;
        let v: f64 = text.parse().map_err(|_| format!("bad number {text}"))?;
        Ok(Json {
            kind: JsonKind::Number(v),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape_27b(q_len: i32, bt_width: i32, sm_count: i32) -> SplitShape {
        SplitShape {
            q_len,
            bt_width,
            page_size: PAGE_SIZE,
            kv_append_len: q_len,
            block_m: BLOCK_M,
            n_q_heads: Q_HEADS,
            bsz: 1,
            hd_pad: HEAD_DIM,
            sm_count,
        }
    }

    #[test]
    fn integer_class_matches_triton() {
        assert_eq!(int_class(1), "eq1");
        assert_eq!(int_class(0), "div16");
        assert_eq!(int_class(16), "div16");
        assert_eq!(int_class(-16), "div16");
        assert_eq!(int_class(256), "div16");
        assert_eq!(int_class(1280), "div16");
        assert_eq!(int_class(240), "div16");
        assert_eq!(int_class(-1), "other");
        assert_eq!(int_class(6), "other");
        assert_eq!(int_class(47), "other");
        assert_eq!(int_class(219), "other");
        assert_eq!(int_class(300), "other");
        assert_eq!(int_class(5), "other");
        // The three layer-test launches, plus a 16-page table.
        assert_eq!(
            prefill_cubin_key(false, 1, 99, 99, 1, -1, -1),
            "_paged_attn_prefill_kernel_qc_ns-eq1_q-other_kv-other_p-eq1_wl-other_wr-other"
        );
        assert_eq!(
            prefill_cubin_key(false, 1, 219, 219, 6, -1, -1),
            "_paged_attn_prefill_kernel_qc_ns-eq1_q-other_kv-other_p-other_wl-other_wr-other"
        );
        assert_eq!(
            prefill_cubin_key(true, 1, 1280, 1280, 5, -1, -1),
            "_paged_attn_prefill_kernel_fp16_ns-eq1_q-div16_kv-div16_p-other_wl-other_wr-other"
        );
        assert_eq!(
            prefill_cubin_key(false, 1, 47, 47, 16, -1, -1),
            "_paged_attn_prefill_kernel_qc_ns-eq1_q-other_kv-other_p-div16_wl-other_wr-other"
        );
        assert_eq!(
            prefill_cubin_key(true, 1, 256, 256, 1, -1, -1),
            "_paged_attn_prefill_kernel_fp16_ns-eq1_q-div16_kv-div16_p-eq1_wl-other_wr-other"
        );
    }

    #[test]
    fn staging_page_round_up_matches_the_fork() {
        assert_eq!(staging_pages(1, 1).unwrap(), 1);
        assert_eq!(staging_pages(1, 2).unwrap(), 2);
        assert_eq!(staging_pages(1, 6).unwrap(), 8);
        assert_eq!(staging_pages(1, 8).unwrap(), 8);
        assert!(staging_pages(0, 1).is_err());
    }

    #[test]
    fn tight_tables_stay_unsplit_and_a_wide_table_does_not() {
        for (q_len, width) in [(47, 1), (300, 2), (1499, 6), (300, 8)] {
            let n = prefill_num_splits(&shape_27b(q_len, width, 82)).unwrap();
            assert_eq!(n, 1, "q_len {q_len} width {width}");
            assert!(!wide_index(&shape_27b(q_len, width, 82), n));
        }
        // 32 pages is the profiled 8192-token table. q_len 47 wants 3 splits.
        let wide = prefill_num_splits(&shape_27b(47, 32, 82)).unwrap();
        assert_eq!(wide, 3);
    }

    #[test]
    fn long_chunks_split_and_stay_inside_int32_indexes() {
        // 3090, 82 SMs. A 2048-row chunk is capped at 2 splits by the 128 MiB
        // partial buffer. The cost model wanted 5. Eight splits is the largest
        // candidate and still does not set WIDE_INDEX.
        let wide_shape = shape_27b(2048, 33, 82);
        assert_eq!(prefill_num_splits(&wide_shape).unwrap(), 2);
        assert_eq!(prefill_num_splits(&shape_27b(2048, 32, 82)).unwrap(), 2);
        assert_eq!(prefill_num_splits(&shape_27b(2048, 24, 82)).unwrap(), 2);
        assert_eq!(prefill_num_splits(&shape_27b(207, 33, 82)).unwrap(), 5);
        assert_eq!(prefill_num_splits(&shape_27b(47, 32, 82)).unwrap(), 3);
        assert_eq!(prefill_num_splits(&shape_27b(240, 33, 82)).unwrap(), 5);
        assert_eq!(prefill_num_splits(&shape_27b(240, 32, 82)).unwrap(), 5);
        assert_eq!(prefill_num_splits(&shape_27b(253, 32, 82)).unwrap(), 5);
        assert_eq!(prefill_num_splits(&shape_27b(1792, 32, 82)).unwrap(), 3);
        // 31 pages plus a short tail is one under the threshold.
        assert_eq!(prefill_num_splits(&shape_27b(255, 31, 82)).unwrap(), 1);
        assert!(!wide_index(&wide_shape, 2));
        assert!(!wide_index(&wide_shape, 8));
        let (o_bytes, ml_bytes) = partial_bytes(2048, 2).unwrap();
        assert!(o_bytes <= PARTIAL_CAP_BYTES);
        assert!(ml_bytes < o_bytes);
        assert!(partial_bytes(2048, 8).is_err());
        assert_eq!(
            combine_cubin_key(true, 2, 2048),
            "_paged_attn_prefill_combine_kernel_fp16_ns-other_q-div16"
        );
        assert_eq!(
            combine_cubin_key(false, 5, 207),
            "_paged_attn_prefill_combine_kernel_qc_ns-other_q-other"
        );
        assert_eq!(
            prefill_cubin_key(true, 2, 2048, 2048, 33, -1, -1),
            "_paged_attn_prefill_kernel_fp16_ns-other_q-div16_kv-div16_p-other_wl-other_wr-other"
        );
        assert_eq!(
            prefill_cubin_key(false, 5, 240, 240, 32, -1, -1),
            "_paged_attn_prefill_kernel_qc_ns-other_q-div16_kv-div16_p-div16_wl-other_wr-other"
        );
    }

    #[test]
    fn dequant_rejects_a_bad_page_size_before_launch() {
        let p = std::ptr::dangling();
        let rc = unsafe {
            dequant_cache_paged_window(
                &DequantWindow {
                    k_in: p,
                    k_scales: p,
                    k_out: p as *mut c_void,
                    v_in: p,
                    v_scales: p,
                    v_out: p as *mut c_void,
                    cache_seqlens: p,
                    block_table: p,
                    bsz: 1,
                    pages_per_seq: 1,
                    token_dim: TOKEN_DIM,
                    k_bits: 3,
                    v_bits: 3,
                    page_size: 128,
                    bonus_len: 1,
                    compand_a: 0.0,
                },
                std::ptr::null_mut(),
            )
        };
        let err = rc.expect_err("page size");
        assert!(err.to_string().contains("256"), "{err}");
    }

    #[test]
    fn scale_json_number_round_trips() {
        let v = parse_json("{\"scale\": 0.0625, \"SOFTCAP\": 0.0}").unwrap();
        assert_eq!(v.get("scale").and_then(Json::as_f64), Some(0.0625));
        assert_eq!(v.get("SOFTCAP").and_then(Json::as_f64), Some(0.0));
    }

    fn params(names: &[&str]) -> String {
        names
            .iter()
            .map(|name| {
                let ty = if name.starts_with("WINDOW")
                    || *name == "q_len"
                    || *name == "kv_append_len"
                    || *name == "num_splits"
                    || *name == "num_pages_per_seq"
                {
                    "i32"
                } else if *name == "k_cache" || *name == "v_cache" {
                    "*i32"
                } else {
                    "*fp16"
                };
                format!(r#"{{"name":"{name}","type":"{ty}"}}"#)
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    #[test]
    fn one_valued_integers_drop_out_of_the_launch_signature() {
        let online_names = [
            "q",
            "k_cache",
            "v_cache",
            "block_table",
            "cache_seqlens",
            "out",
            "partial_o",
            "partial_ml",
            "k_scales",
            "v_scales",
            "h32",
            "k_new",
            "v_new",
            "sinks",
            "q_len",
            "kv_append_len",
            "WINDOW_LEFT",
            "WINDOW_RIGHT",
            "global_scratch",
            "profile_scratch",
        ];
        let online = parse_json(&format!(
            r#"{{"parameters":[{}],"constexprs":{{"num_splits":1,"num_pages_per_seq":1}}}}"#,
            params(&online_names)
        ))
        .unwrap();
        let consts = online.get("constexprs").unwrap();
        let (got, splits, pages) = param_list(&online, "online", "*i32", consts).unwrap();
        assert_eq!(got.len(), online_names.len());
        assert!(got
            .iter()
            .all(|p| p.name != "num_splits" && p.name != "num_pages_per_seq"));
        assert_eq!(splits, Some(1));
        assert_eq!(pages, Some(1));

        let mut staged_names = online_names.to_vec();
        staged_names.insert(16, "num_pages_per_seq");
        // k/v of the staged cubin are fp16. The type check below only fires for *i32.
        let staged_body = params(&staged_names).replace(r#""type":"*i32""#, r#""type":"*fp16""#);
        let staged = parse_json(&format!(
            r#"{{"parameters":[{staged_body}],"constexprs":{{"num_splits":1}}}}"#
        ))
        .unwrap();
        let (got, splits, pages) = param_list(
            &staged,
            "staged",
            "*fp16",
            staged.get("constexprs").unwrap(),
        )
        .unwrap();
        assert!(got.iter().any(|p| p.name == "num_pages_per_seq"));
        assert_eq!(splits, Some(1));
        assert_eq!(pages, None);

        let missing_q = parse_json(
            r#"{"parameters":[{"name":"q","type":"*fp16"}],"constexprs":{"num_splits":1}}"#,
        )
        .unwrap();
        let err = match param_list(
            &missing_q,
            "online",
            "*i32",
            missing_q.get("constexprs").unwrap(),
        ) {
            Ok(_) => panic!("q_len stays runtime"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("k_cache"), "{err}");
    }
}

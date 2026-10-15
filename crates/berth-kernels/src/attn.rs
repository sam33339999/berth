//! Full-attention decode for the 27B, one token at a time.
//!
//! Per layer, after the fused QKV/QG projection (not in this module) and
//! before `o_proj`, `BC_Attention::run_gr` in
//! `exllamav3_ext/libtorch/attention.cpp` does:
//!
//! 1. [`deinterleave_qg`] — `deinterleave_qg_kernel` (`activation.cu`)
//! 2. [`rope`] — `rope_kernel<2,1>` (`rope.cu`), NEOX, bf16 q/k norm fused
//! 3. [`quant_cache_paged`] — `quant_cache_paged_kernel<3,3>` (`cache/q_cache.cu`)
//! 4. [`PagedDecode::launch`] — Triton `_paged_attn_decode_split_kernel` then
//!    `_paged_attn_decode_combine_kernel`
//! 5. [`mul_sigmoid`] — `mul_sigmoid_kernel_h` (`activation.cu`)
//!
//! Gate mode is the interleaved swish gate (`gate_mode == 3`). There is no
//! v-norm and no QSA. The 16 full-attention layers are every fourth layer.
//! Checkpoint tensors this path reads, besides the cache it writes:
//!
//! * `model.language_model.layers.{idx}.self_attn.q_norm.weight`, bf16,
//!   length `head_dim` (256). `constant_bias` is 1.0.
//! * `model.language_model.layers.{idx}.self_attn.k_norm.weight`, same.
//! * `rms_norm_eps` is `1e-6` (`text_config` in `config.json`).
//!
//! RoPE is NEOX (`RopeStyle` 2). `rope_theta` is `1e7`,
//! `partial_rotary_factor` is 0.25, so the rotary dim is 64 and `inv_freq`
//! has length 32. `attn_factor` is 1, `rotate_dims` is 1, and the Llama-4
//! scale is off. Text decode passes a scalar position (the cache length
//! before the new token) and null `positions` / `position_ids`.
//!
//! Pointers are device addresses. `stream` on the CUDA launches is a
//! `cudaStream_t` (`null` is the default stream). The Triton pair is loaded
//! with [`berth_cuda::Cubin`] and launched with [`berth_cuda::Function`].
//! Above Ampere's 48 KiB default, the function opts in with
//! [`berth_cuda::Function::set_max_dynamic_shared_bytes`], which is what
//! `TritonKernel` does in `triton_kernel.cpp`.

use std::ffi::{c_void, CStr};
use std::path::Path;

use berth_cuda::{Context, Cubin, Dim3, Function, Stream};

use crate::{Error, Result};

const Q_HEADS: i32 = 24;
const KV_HEADS: i32 = 4;
const HEAD_DIM: i32 = 256;
const PAGE_SIZE: i32 = 256;
const Q_LEN: i32 = 1;
/// Ampere's default dynamic shared-memory cap. Above this, the function opts
/// in with `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)`, matching
/// `TritonKernel` (`triton_kernel.cpp`).
const DEFAULT_DYNAMIC_SMEM: u32 = 48 * 1024;

/// Command that writes `target/triton-cubins/` when the cubins are missing.
pub const TRITON_CUBIN_BUILD: &str = "\
PYTHONPATH=~/qwen38-exl3-dflash2/exllamav3 \
TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \
~/qwen38-exl3-dflash2/env/bin/python scripts/build_triton_cubins.py";

mod ffi {
    use std::ffi::{c_char, c_void};

    #[allow(clippy::too_many_arguments)]
    unsafe extern "C" {
        pub fn berth_attn_last_error() -> *const c_char;

        pub fn berth_attn_deinterleave_qg(
            qg: *const c_void,
            q: *mut c_void,
            g: *mut c_void,
            head_dim: i32,
            q_numel: i64,
            stream: *mut c_void,
        ) -> i32;

        pub fn berth_attn_mul_sigmoid(
            x: *mut c_void,
            y: *const c_void,
            numel: i64,
            stream: *mut c_void,
        ) -> i32;

        pub fn berth_attn_rope(
            q: *const c_void,
            out_q: *mut c_void,
            k: *const c_void,
            out_k: *mut c_void,
            inv_freq: *const c_void,
            bsz: i32,
            seq_len: i32,
            num_heads_q: i32,
            num_heads_k: i32,
            head_dim: i32,
            q_head_stride: i32,
            k_head_stride: i32,
            partial_head_dim: i32,
            position: u32,
            positions: *const c_void,
            position_ids: *const c_void,
            rope_mode: i32,
            attn_factor: f32,
            q_norm: *const c_void,
            k_norm: *const c_void,
            norm_eps: f32,
            norm_constant_bias: f32,
            inv_freq_table: i32,
            inv_freq_stride: i32,
            llama_4_scaling_beta: f32,
            llama_4_scaling_original: i32,
            position_ids_stride: i32,
            rotate_dims: i32,
            rotate_offset: i32,
            stream: *mut c_void,
        ) -> i32;

        pub fn berth_attn_quant_cache_paged(
            k_in: *const c_void,
            k_out: *mut c_void,
            k_scales: *mut c_void,
            v_in: *const c_void,
            v_out: *mut c_void,
            v_scales: *mut c_void,
            cache_seqlens: *const c_void,
            block_table: *const c_void,
            bsz: i32,
            seq_len: i32,
            blocks_per_seq: i32,
            token_dim: i32,
            k_bits: i32,
            v_bits: i32,
            page_size: i32,
            compand_a: f32,
            in_contiguous: i32,
            stream: *mut c_void,
        ) -> i32;
    }
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

/// Split interleaved Q and gate.
///
/// `deinterleave_qg_gr` (`activation.cu`). Gate mode 3 stores, per head,
/// `[q: head_dim | g: head_dim]` inside `qg`. The kernel writes contiguous
/// `q` and `g`.
///
/// Shapes, fp16, contiguous, device memory. `rows` is `bsz * q_len` (1 at
/// decode):
///
/// * `qg`: `(rows, n_q_heads * 2 * head_dim)`. For the 27B, `(1, 12288)`.
/// * `q`, `g`: `(rows, n_q_heads * head_dim)`. For the 27B, `(1, 6144)`.
///
/// `head_dim` is 256 and must be a multiple of 8. `q_numel` is the number of
/// fp16 elements in `q` (and in `g`), not in `qg`. The launch is
/// `ceil(q_numel / 8 / 256)` blocks of 256 threads. `uint4` moves 8 halves
/// per thread.
///
/// # Errors
///
/// A null pointer, a `head_dim` that is not a positive multiple of 8, or a
/// `q_numel` that is not a positive multiple of both 8 and `head_dim`.
///
/// # Safety
///
/// `qg`, `q`, and `g` must be device allocations of those sizes and must stay
/// allocated until `stream` finishes the launch. The three regions must not
/// overlap.
pub unsafe fn deinterleave_qg(
    qg: *const c_void,
    q: *mut c_void,
    g: *mut c_void,
    head_dim: i32,
    q_numel: i64,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe { ffi::berth_attn_deinterleave_qg(qg, q, g, head_dim, q_numel, stream) };
    fail(rc, "deinterleave_qg")
}

/// In-place output gate: `x *= sigmoid(y)`.
///
/// `mul_sigmoid__gr` (`activation.cu`), `mul_sigmoid_kernel_h`. Interleaved
/// and full gates use this. The headwise gate uses the broadcast variant,
/// which this decode path does not.
///
/// `x` and `y` are fp16, contiguous, the same number of elements. At decode
/// that is `(1, n_q_heads * head_dim)` = `(1, 6144)`, the attention output
/// and the deinterleaved gate. `numel` must be even. The kernel walks
/// `half2` pairs, `ceil(numel / 512)` blocks of 256 threads. `x` is updated
/// in place. `y` is not written.
///
/// # Errors
///
/// A null pointer, or a `numel` that is not positive and even.
///
/// # Safety
///
/// Both pointers must address device memory of `numel` fp16 elements and must
/// stay allocated until `stream` finishes. `x` and `y` may not overlap.
pub unsafe fn mul_sigmoid(
    x: *mut c_void,
    y: *const c_void,
    numel: i64,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe { ffi::berth_attn_mul_sigmoid(x, y, numel, stream) };
    fail(rc, "mul_sigmoid")
}

/// Arguments for [`rope`]. Matches `rope_gr` (`rope.cu`).
///
/// Only `rope_mode == 2` (NEOX) is launched, and the norm instance is the
/// bf16 one (`rope_kernel<2,1>`). Passing a norm pointer means that buffer
/// is bf16. Both norms are set, or both are null. Both `k` and `out_k` are
/// set, or both are null.
///
/// `q` is `(bsz, seq_len, num_heads_q, head_dim)` fp16. `k` is
/// `(bsz, seq_len, num_heads_k, head_dim)` fp16. Decode uses `bsz = 1`,
/// `seq_len = 1`, 24 query heads, 4 key heads, `head_dim = 256`. `out_q` may
/// alias `q` and `out_k` may alias `k`; `run_gr` does both.
///
/// `q_head_stride` and `k_head_stride` are `stride(2)` in **half elements**,
/// not bytes. A contiguous head is `head_dim` (256). The innermost dim is dense.
///
/// `inv_freq` is fp32. A 1-D table has length `partial_head_dim / 2` (32 for
/// the 27B) and `inv_freq_table` is false. `partial_head_dim` is
/// `inv_freq.shape[-1] * 2` (64). `position` is the position id of the first
/// token in the tile. `positions` (u32, length `bsz`) and `position_ids`
/// (i32, `(bsz, seq_len)` or with a trailing `rotate_dims`) are optional and
/// mutually exclusive; null selects the scalar `position`.
///
/// `q_norm` and `k_norm` are the checkpoint weights above, bf16, length
/// `head_dim`. The kernel adds `norm_constant_bias` (1.0) to each weight and
/// uses `norm_eps` (`1e-6`). `attn_factor` is 1. `rotate_dims` is 1 and
/// `rotate_offset` is 0. `llama_4_scaling_beta` is 0 and
/// `llama_4_scaling_original` is unused in that case.
pub struct RopeLaunch {
    /// Query, fp16.
    pub q: *const c_void,
    /// Query output, fp16. May equal `q`.
    pub out_q: *mut c_void,
    /// Key, fp16. Null when `num_heads_k` is 0.
    pub k: *const c_void,
    /// Key output, fp16. May equal `k`. Null when `k` is null.
    pub out_k: *mut c_void,
    /// Inverse frequencies, fp32.
    pub inv_freq: *const c_void,
    /// Batch.
    pub bsz: i32,
    /// Tokens in this tile.
    pub seq_len: i32,
    /// Query heads. 24 on the 27B.
    pub num_heads_q: i32,
    /// Key heads. 4 on the 27B. 0 omits `k`.
    pub num_heads_k: i32,
    /// Head dimension. 256 on the 27B.
    pub head_dim: i32,
    /// `q.stride(2)` in half elements.
    pub q_head_stride: i32,
    /// `k.stride(2)` in half elements. Ignored when `k` is null.
    pub k_head_stride: i32,
    /// Rotary span. `inv_freq` length times 2. 64 on the 27B.
    pub partial_head_dim: i32,
    /// Position id of token 0 in the tile.
    pub position: u32,
    /// Optional per-batch base position, u32, length `bsz`.
    pub positions: *const c_void,
    /// Optional per-token position ids, i32.
    pub position_ids: *const c_void,
    /// `ROPESTYLE_NEOX` is 2. Anything else is rejected.
    pub rope_mode: i32,
    /// Scale on sin and cos. 1.0 on the 27B.
    pub attn_factor: f32,
    /// bf16 RMSNorm weight, length `head_dim`, or null.
    pub q_norm: *const c_void,
    /// bf16 RMSNorm weight, length `head_dim`, or null.
    pub k_norm: *const c_void,
    /// RMSNorm epsilon. `1e-6` on the 27B.
    pub norm_eps: f32,
    /// Added to the norm weight. 1.0 on the 27B (`RMSNorm.constant_bias`).
    pub norm_constant_bias: f32,
    /// True when `inv_freq` is a position table rather than one frequency vector.
    pub inv_freq_table: bool,
    /// `inv_freq.shape[-1] * inv_freq.shape[-2]` when `inv_freq_table` is set.
    pub inv_freq_stride: i32,
    /// Llama-4 query scale. 0 on the 27B.
    pub llama_4_scaling_beta: f32,
    /// Llama-4 original context. Unused when the beta is 0.
    pub llama_4_scaling_original: i32,
    /// Trailing size of `position_ids`. 1 unless that tensor is 3-D.
    pub position_ids_stride: i32,
    /// How many rotary spans fit in the head. 1 on the 27B. At most 4.
    pub rotate_dims: i32,
    /// First rotated element. 0 on the 27B.
    pub rotate_offset: i32,
}

/// Fused RMSNorm and NEOX RoPE. See [`RopeLaunch`].
///
/// The grid matches `rope_gr`: `warps = ceil((head_dim/2) / 32)`, block
/// `(warps * 32, parallel_heads)`, grid `(seq_len, bsz, head_groups)`.
/// `parallel_heads` is `min(1024 / threads_x, n_q + n_kv)`. `head_groups`
/// covers every head, and drops to 1 when `seq_len * bsz >= 32`. The 27B
/// decode tile is grid `(1, 1, 4)` and block `(128, 8)`.
///
/// # Errors
///
/// Host checks (null `q` / `out_q` / `inv_freq`, rope mode other than NEOX,
/// a one-sided norm or key pointer, a rotate range the kernel cannot hold)
/// return [`Error::Launch`] before any launch. A CUDA failure is the same
/// error.
///
/// # Safety
///
/// Every non-null pointer must be a device allocation of the size described
/// on [`RopeLaunch`] and must stay allocated until `stream` finishes. Strides
/// must match the allocation. Norm pointers, when set, must be bf16.
pub unsafe fn rope(args: &RopeLaunch, stream: *mut c_void) -> Result<()> {
    let rc = unsafe {
        ffi::berth_attn_rope(
            args.q,
            args.out_q,
            args.k,
            args.out_k,
            args.inv_freq,
            args.bsz,
            args.seq_len,
            args.num_heads_q,
            args.num_heads_k,
            args.head_dim,
            args.q_head_stride,
            args.k_head_stride,
            args.partial_head_dim,
            args.position,
            args.positions,
            args.position_ids,
            args.rope_mode,
            args.attn_factor,
            args.q_norm,
            args.k_norm,
            args.norm_eps,
            args.norm_constant_bias,
            i32::from(args.inv_freq_table),
            args.inv_freq_stride,
            args.llama_4_scaling_beta,
            args.llama_4_scaling_original,
            args.position_ids_stride,
            args.rotate_dims,
            args.rotate_offset,
            stream,
        )
    };
    fail(rc, "rope")
}

/// Arguments for [`quant_cache_paged`]. Matches `quant_cache_paged_gr`.
///
/// # Paged cq layout
///
/// `CacheLayer_quant` (`cache/quant.py`) allocates, for `token_dim = n_kv_heads * head_dim`
/// (1024 on the 27B) and bitrate 3:
///
/// * `k_out` / `v_out`: int32, shape
///   `(num_pages, page_size, token_dim / 32 * bits)` =
///   `(num_pages, 256, 96)`. These are `qk` / `qv`.
/// * `k_scales` / `v_scales`: fp16, shape
///   `(num_pages, page_size, token_dim / 32)` =
///   `(num_pages, 256, 32)`. These are `sk` / `sv`.
///
/// The kernel addresses both as flat buffers. Group `g` of physical token
/// `token_pos` starts at element `token_pos * groups_per_token + g`, with
/// `groups_per_token = token_dim / 32` (32). Each group is 32 fp16 values,
/// Hadamard-rotated by H32, scaled by its absmax (one fp16), and packed
/// little-endian into `bits` uint32 words. `compand_a` is 0 on the 27B, so
/// the quantizer is the linear midpoint grid, not the cubic compander.
///
/// `token_pos = block_table[batch * blocks_per_seq + page] * page_size + (seq_index % page_size)`,
/// where `page = seq_index / page_size` and
/// `seq_index = cache_seqlens[batch] + append_pos`. `cache_seqlens` is the
/// length **before** this append, int32, one per batch. `block_table` is
/// int32, shape `(bsz, blocks_per_seq)`, row-major. Page ids are non-negative.
///
/// `k_in` and `v_in` are fp16. With `in_contiguous` (what `run_gr` passes),
/// row `(batch * seq_len + append_pos)` is token `append_pos` of that batch,
/// laid out as `n_kv_heads * head_dim` contiguous halves. That is `k4` / `v4`:
/// `(bsz, seq_len, n_kv_heads, head_dim)`. `seq_len` is `q_len` (1 at decode).
/// A false flag indexes the input at the paged `token_pos` instead, which is
/// the non-direct cache update.
///
/// `page_size` must be 256. `k_bits` and `v_bits` must be 3. The header also
/// builds the other bitrates; this launcher does not call them.
pub struct QuantCacheLaunch {
    /// New K, fp16.
    pub k_in: *const c_void,
    /// K pages, int32. See the layout above.
    pub k_out: *mut c_void,
    /// K scales, fp16.
    pub k_scales: *mut c_void,
    /// New V, fp16.
    pub v_in: *const c_void,
    /// V pages, int32.
    pub v_out: *mut c_void,
    /// V scales, fp16.
    pub v_scales: *mut c_void,
    /// Length before the append, int32, length `bsz`.
    pub cache_seqlens: *const c_void,
    /// Page map, int32, `(bsz, blocks_per_seq)`.
    pub block_table: *const c_void,
    /// Batch.
    pub bsz: i32,
    /// Tokens appended per batch. `q_len` in `run_gr`.
    pub seq_len: i32,
    /// `block_table` width.
    pub blocks_per_seq: i32,
    /// `n_kv_heads * head_dim`. 1024 on the 27B.
    pub token_dim: i32,
    /// K bitrate. Must be 3.
    pub k_bits: i32,
    /// V bitrate. Must be 3.
    pub v_bits: i32,
    /// Page length. Must be 256.
    pub page_size: i32,
    /// Cubic compander coefficient. 0 on the 27B.
    pub compand_a: f32,
    /// True: input rows are contiguous `(batch, append_pos)`, not paged positions.
    pub in_contiguous: bool,
}

/// Quantize one appended K/V tile into the paged cq3 cache. See [`QuantCacheLaunch`].
///
/// The grid matches `quant_cache_paged_gr`. For the 27B token dim, that is
/// grid `(1, seq_len, bsz)` and 256 threads.
///
/// # Errors
///
/// A null pointer, a page size other than 256, a bitrate other than 3, or a
/// `token_dim` that is not a positive multiple of 32.
///
/// # Safety
///
/// Every pointer must be a device allocation of the size implied by
/// [`QuantCacheLaunch`] and must stay allocated until `stream` finishes.
/// `cache_seqlens[batch] + seq_len` must land inside the pages named by
/// `block_table`. Output slots the kernel does not touch are left as they were.
pub unsafe fn quant_cache_paged(args: &QuantCacheLaunch, stream: *mut c_void) -> Result<()> {
    let rc = unsafe {
        ffi::berth_attn_quant_cache_paged(
            args.k_in,
            args.k_out,
            args.k_scales,
            args.v_in,
            args.v_out,
            args.v_scales,
            args.cache_seqlens,
            args.block_table,
            args.bsz,
            args.seq_len,
            args.blocks_per_seq,
            args.token_dim,
            args.k_bits,
            args.v_bits,
            args.page_size,
            args.compand_a,
            i32::from(args.in_contiguous),
            stream,
        )
    };
    fail(rc, "quant_cache_paged")
}

/// Live split count and split length.
///
/// `split_config` in `libtorch/attention.cpp`:
///
/// * `bound = bt_width * page_size + q_len`
/// * `num_splits = max(1, min(splits_cap, ceil(bound / (4 * block_n))))`
/// * `split_len = ceil(ceil(bound / num_splits) / block_n) * block_n`
///
/// `bt_width` is `block_table.size(1)`. `block_n` is the cubin's `BLOCK_N`
/// (32 for the 27B). `splits_cap` is not a constexpr; on this 3090 it is 41.
/// An empty page table (`bt_width = 1`, `q_len = 1`) is `(3, 96)`. A table
/// wide enough for ~300 tokens (`bt_width = 2`) is `(5, 128)`.
///
/// # Errors
///
/// Any argument that is not positive, or a result that does not fit in `i32`.
pub fn split_config(
    bt_width: i32,
    page_size: i32,
    q_len: i32,
    block_n: i32,
    splits_cap: i32,
) -> Result<(i32, i32)> {
    if bt_width < 1 || page_size < 1 || q_len < 1 || block_n < 1 || splits_cap < 1 {
        return Err(Error::Launch(
            "split_config: bt_width, page_size, q_len, block_n, and splits_cap must be positive"
                .into(),
        ));
    }
    let bound = i64::from(bt_width) * i64::from(page_size) + i64::from(q_len);
    let denom = 4 * i64::from(block_n);
    let ceil_splits = (bound + denom - 1) / denom;
    let num_splits = 1.max(ceil_splits.min(i64::from(splits_cap)));
    let per = (bound + num_splits - 1) / num_splits;
    let bn = i64::from(block_n);
    let split_len = ((per + bn - 1) / bn) * bn;
    let num_splits = i32_fit(num_splits, "num_splits")?;
    let split_len = i32_fit(split_len, "split_len")?;
    Ok((num_splits, split_len))
}

/// `bc_attn._configure`: `max(1, min(2 * sm_count / programs, 128))`.
///
/// 4 programs and 82 SMs (this 3090) give 41.
///
/// # Errors
///
/// `programs` or `sm_count` is not positive.
pub fn splits_cap(programs: i32, sm_count: i32) -> Result<i32> {
    if programs < 1 || sm_count < 1 {
        return Err(Error::Launch(
            "splits_cap: programs and sm_count must be positive".into(),
        ));
    }
    let target = 2 * i64::from(sm_count);
    let cap = 1.max((target / i64::from(programs)).min(128));
    i32_fit(cap, "splits_cap")
}

fn i32_fit(v: i64, what: &str) -> Result<i32> {
    i32::try_from(v).map_err(|_| Error::Launch(format!("{what} does not fit in i32")))
}

fn u32_fit(v: i64, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::Launch(format!("{what} does not fit in u32")))
}

/// Device pointers for one [`PagedDecode::launch`].
///
/// All of them are device addresses ([`berth_cuda::DevicePtr::as_u64`]).
/// Layouts match `BC_Attention::run_gr`'s regime-0 launch
/// (`libtorch/attention.cpp`).
///
/// * `q`: fp16, `(bsz, q_len, n_q_heads, head_dim)`, after RoPE. `q_len` is 1.
/// * `k_cache`, `v_cache`: cq3 pages. See [`QuantCacheLaunch`].
/// * `k_scales`, `v_scales`: fp16 scales beside those pages.
/// * `block_table`: int32, `(bsz, bt_width)`.
/// * `cache_seqlens`: int32, length `bsz`, length **before** the append.
///   The split kernel adds the constexpr `kv_append_len` (`q_len`) itself.
///   The cache write for this token must already have happened.
/// * `out`: fp16, same shape as `q`. The combine kernel writes it. The gate
///   is a separate [`mul_sigmoid`] on this buffer.
/// * `partial_o`: fp32, `programs * splits_cap * BLOCK_ROWS * HD_PAD` elements.
///   `programs = bsz * n_kv_heads * cdiv(n_q_heads / n_kv_heads, BLOCK_H)`.
///   For bsz 1 that is `4 * 41 * 16 * 256`.
/// * `partial_ml`: fp32, `programs * splits_cap * BLOCK_ROWS * 2` elements
///   (online-softmax `m` and `l` per row per live split).
/// * `h32`: fp16, `(32, 32)`, the Sylvester Hadamard divided by `sqrt(32)`.
///   `_get_h32` in `modules/attention_fn/triton_paged.py`. Q and the output
///   are rotated by it because the cache bitrate is nonzero.
/// * `sinks`: dead. The cubin is `HAS_SINKS = false`, and `run_gr` passes
///   `q`'s pointer. Any device pointer is enough; it is not loaded.
pub struct PagedDecodeArgs {
    /// Query, fp16.
    pub q: u64,
    /// Packed K pages, int32.
    pub k_cache: u64,
    /// Packed V pages, int32.
    pub v_cache: u64,
    /// Page map, int32.
    pub block_table: u64,
    /// Length before the append, int32.
    pub cache_seqlens: u64,
    /// Attention output, fp16, before the gate.
    pub out: u64,
    /// Split accumulators, fp32.
    pub partial_o: u64,
    /// Split softmax stats, fp32.
    pub partial_ml: u64,
    /// K group scales, fp16.
    pub k_scales: u64,
    /// V group scales, fp16.
    pub v_scales: u64,
    /// 32×32 Hadamard, fp16.
    pub h32: u64,
    /// Unused sink pointer. `run_gr` passes `q`.
    pub sinks: u64,
}

struct KernelImage {
    function: Function,
    name: String,
    shared: u32,
    num_warps: u32,
    params: Vec<Param>,
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

struct Tile {
    n_q_heads: i32,
    n_kv_heads: i32,
    q_len: i32,
    page_size: i32,
    head_dim: i32,
    block_h: i32,
    block_rows: i32,
    block_n: i32,
    hd_pad: i32,
}

/// The two decode cubins, split then combine.
///
/// Built by `scripts/build_triton_cubins.py` into `target/triton-cubins/`
/// (`manifest.json` plus the two cubin files). The manifest's `parameters`
/// list is the launch order, including Triton's trailing `global_scratch`
/// and `profile_scratch` pointers. Both scratch sizes are 0; those pointers
/// are null.
///
/// `run_gr` launches split at grid `(programs, splits_cap, 1)` and combine
/// at `(programs, 1, 1)`. The block is `(32 * num_warps, 1, 1)` with
/// `num_warps = 4`, so 128 threads. `FINAL` is false on this cubin, so
/// combine always runs, including when the live split count is 1. Splits
/// at or above the live count exit without storing.
///
/// Both kernels are launched with [`berth_cuda::Function::launch`]. When
/// the cubin asks for more than 48 KiB of dynamic shared memory, load calls
/// [`berth_cuda::Function::set_max_dynamic_shared_bytes`] first, which is
/// what `TritonKernel` does. The 27B decode cubins are under that cap.
pub struct PagedDecode {
    split: KernelImage,
    combine: KernelImage,
    tile: Tile,
}

impl PagedDecode {
    /// Load `manifest.json` and both cubins from `dir`.
    ///
    /// # Errors
    ///
    /// A missing manifest or cubin, a parameter list that is not the
    /// `run_gr` order, a constexpr that is not the 27B decode tile, or a
    /// driver error while loading the image. The missing-file error includes
    /// [`TRITON_CUBIN_BUILD`].
    pub fn load(ctx: &Context, dir: &Path) -> Result<Self> {
        let manifest_path = dir.join("manifest.json");
        let text = std::fs::read_to_string(&manifest_path).map_err(|_| {
            Error::Launch(format!(
                "paged attention cubins are missing at {}. Run: {TRITON_CUBIN_BUILD}",
                manifest_path.display()
            ))
        })?;
        let manifest = parse_json(&text).map_err(|err| {
            Error::Launch(format!(
                "paged attention manifest {}: {err}",
                manifest_path.display()
            ))
        })?;
        let kernels = manifest
            .get("kernels")
            .ok_or_else(|| Error::Launch("manifest has no kernels object".into()))?;
        let split = load_kernel(
            ctx,
            dir,
            kernels,
            "_paged_attn_decode_split_kernel",
            &[
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
                "split_len",
                "num_pages_per_seq",
                "num_splits",
                "sinks",
                "global_scratch",
                "profile_scratch",
            ],
        )?;
        let combine = load_kernel(
            ctx,
            dir,
            kernels,
            "_paged_attn_decode_combine_kernel",
            &[
                "partial_o",
                "partial_ml",
                "out",
                "h32",
                "num_splits",
                "sinks",
                "global_scratch",
                "profile_scratch",
            ],
        )?;
        let tile = tile_from(split_record(kernels)?)?;
        check_combine_tile(kernels, &tile)?;
        if split.num_warps != combine.num_warps {
            return Err(Error::Launch(format!(
                "split num_warps {} != combine num_warps {}",
                split.num_warps, combine.num_warps
            )));
        }
        Ok(Self {
            split,
            combine,
            tile,
        })
    }

    /// fp32 elements of `partial_o` for this `bsz` and `splits_cap`.
    ///
    /// # Errors
    ///
    /// `bsz` or `splits_cap` is not positive, or the product overflows.
    pub fn partial_o_elems(&self, bsz: i32, splits_cap: i32) -> Result<usize> {
        let programs = self.programs(bsz)?;
        let n = i64::from(programs)
            * i64::from(splits_cap)
            * i64::from(self.tile.block_rows)
            * i64::from(self.tile.hd_pad);
        usize_fit(n, "partial_o")
    }

    /// fp32 elements of `partial_ml` for this `bsz` and `splits_cap`.
    ///
    /// # Errors
    ///
    /// `bsz` or `splits_cap` is not positive, or the product overflows.
    pub fn partial_ml_elems(&self, bsz: i32, splits_cap: i32) -> Result<usize> {
        let programs = self.programs(bsz)?;
        let n = i64::from(programs) * i64::from(splits_cap) * i64::from(self.tile.block_rows) * 2;
        usize_fit(n, "partial_ml")
    }

    /// `bsz * n_kv_heads * cdiv(group, BLOCK_H)`.
    ///
    /// # Errors
    ///
    /// `bsz` is not positive.
    pub fn programs(&self, bsz: i32) -> Result<i32> {
        if bsz < 1 {
            return Err(Error::Launch("programs: bsz must be positive".into()));
        }
        let group = self.tile.n_q_heads / self.tile.n_kv_heads;
        let h_blocks = ceil_div_i32(group, self.tile.block_h)?;
        i32_fit(
            i64::from(bsz) * i64::from(self.tile.n_kv_heads) * i64::from(h_blocks),
            "programs",
        )
    }

    /// Split, then combine, for one decode step.
    ///
    /// `bt_width` is the block-table width (`num_pages_per_seq`).
    /// `splits_cap` is the grid Y of the split kernel. On this 3090 with
    /// bsz 1 that is 41 ([`splits_cap`]). The live `num_splits` and
    /// `split_len` are computed with [`split_config`] from `bt_width`, the
    /// cubin's page size, `q_len` (1), `BLOCK_N`, and `splits_cap`.
    ///
    /// # Errors
    ///
    /// A non-positive `bt_width` or `splits_cap`, or a launch rejected by
    /// the driver.
    ///
    /// # Safety
    ///
    /// Every address in `args` must be a device allocation of the size
    /// described on [`PagedDecodeArgs`], live until `stream` finishes both
    /// kernels. `partial_o` and `partial_ml` must hold at least
    /// [`Self::partial_o_elems`] and [`Self::partial_ml_elems`] elements.
    /// The cache pages must already contain this step's new K and V.
    pub unsafe fn launch(
        &self,
        stream: &Stream,
        args: &PagedDecodeArgs,
        bsz: i32,
        bt_width: i32,
        splits_cap: i32,
    ) -> Result<()> {
        if bt_width < 1 || splits_cap < 1 {
            return Err(Error::Launch(
                "paged decode: bt_width and splits_cap must be positive".into(),
            ));
        }
        let programs = self.programs(bsz)?;
        let (num_splits, split_len) = split_config(
            bt_width,
            self.tile.page_size,
            self.tile.q_len,
            self.tile.block_n,
            splits_cap,
        )?;
        let programs_u = u32_fit(i64::from(programs), "programs")?;
        let cap_u = u32_fit(i64::from(splits_cap), "splits_cap")?;
        let runtime = Runtime {
            q: args.q,
            k_cache: args.k_cache,
            v_cache: args.v_cache,
            block_table: args.block_table,
            cache_seqlens: args.cache_seqlens,
            out: args.out,
            partial_o: args.partial_o,
            partial_ml: args.partial_ml,
            k_scales: args.k_scales,
            v_scales: args.v_scales,
            h32: args.h32,
            split_len,
            num_pages_per_seq: bt_width,
            num_splits,
            sinks: args.sinks,
        };
        launch_image(
            &self.split,
            stream,
            &runtime,
            Dim3::new(programs_u, cap_u, 1),
        )?;
        launch_image(&self.combine, stream, &runtime, Dim3::new(programs_u, 1, 1))?;
        Ok(())
    }
}

struct Runtime {
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
    split_len: i32,
    num_pages_per_seq: i32,
    num_splits: i32,
    sinks: u64,
}

fn ceil_div_i32(n: i32, d: i32) -> Result<i32> {
    if d < 1 || n < 0 {
        return Err(Error::Launch("ceil_div: bad arguments".into()));
    }
    i32_fit((i64::from(n) + i64::from(d) - 1) / i64::from(d), "ceil_div")
}

fn usize_fit(v: i64, what: &str) -> Result<usize> {
    if v < 0 {
        return Err(Error::Launch(format!("{what} is negative")));
    }
    usize::try_from(v).map_err(|_| Error::Launch(format!("{what} does not fit in usize")))
}

fn launch_image(image: &KernelImage, stream: &Stream, runtime: &Runtime, grid: Dim3) -> Result<()> {
    let mut slots = Vec::with_capacity(image.params.len());
    for param in &image.params {
        slots.push(slot(param, runtime)?);
    }
    let mut params: Vec<*mut c_void> = slots
        .iter_mut()
        .map(|slot| match slot.kind {
            ParamKind::Ptr => &mut slot.ptr as *mut u64 as *mut c_void,
            ParamKind::I32 => &mut slot.scalar as *mut i32 as *mut c_void,
        })
        .collect();
    let block = Dim3::new(32 * image.num_warps, 1, 1);
    unsafe {
        image
            .function
            .launch(stream, grid, block, image.shared, &mut params)
    }
    .map_err(|err| Error::Launch(format!("{}: {err}", image.name)))
}

struct Slot {
    ptr: u64,
    scalar: i32,
    kind: ParamKind,
}

fn slot(param: &Param, runtime: &Runtime) -> Result<Slot> {
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
    let slot = match param.name.as_str() {
        "global_scratch" | "profile_scratch" => ptr_of(0),
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
        "sinks" => ptr_of(runtime.sinks),
        "split_len" => i32_of(runtime.split_len),
        "num_pages_per_seq" => i32_of(runtime.num_pages_per_seq),
        "num_splits" => i32_of(runtime.num_splits),
        other => {
            return Err(Error::Launch(format!(
                "manifest parameter {other} is not a run_gr argument"
            )))
        }
    };
    if slot.kind != param.kind {
        return Err(Error::Launch(format!(
            "manifest parameter {} has the wrong type",
            param.name
        )));
    }
    Ok(slot)
}

fn split_record(kernels: &Json) -> Result<&Json> {
    kernels
        .get("_paged_attn_decode_split_kernel")
        .ok_or_else(|| Error::Launch("manifest has no split kernel".into()))
}

fn load_kernel(
    ctx: &Context,
    dir: &Path,
    kernels: &Json,
    key: &str,
    expect: &[&str],
) -> Result<KernelImage> {
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
    if num_warps == 0 || num_warps > 32 {
        return Err(Error::Launch(format!("{key}: num_warps {num_warps}")));
    }
    for scratch in ["global_scratch_size", "profile_scratch_size"] {
        if let Some(size) = record.get(scratch).and_then(Json::as_i64) {
            if size != 0 {
                return Err(Error::Launch(format!(
                    "{key}: {scratch} is {size}; TritonKernel passes a null pointer"
                )));
            }
        }
    }
    let params = param_list(record, expect)?;
    let path = dir.join(file);
    let bytes = read_cubin(&path)?;
    let cubin = Cubin::load_bytes(ctx, &bytes).map_err(cuda_err)?;
    let function = cubin.function(symbol).map_err(cuda_err)?;
    if shared > DEFAULT_DYNAMIC_SMEM {
        function
            .set_max_dynamic_shared_bytes(shared)
            .map_err(cuda_err)?;
    }
    Ok(KernelImage {
        function,
        name: symbol.to_string(),
        shared,
        num_warps,
        params,
    })
}

fn read_cubin(path: &Path) -> Result<Vec<u8>> {
    match std::fs::read(path) {
        Ok(bytes) if !bytes.is_empty() => Ok(bytes),
        _ => Err(Error::Launch(format!(
            "paged attention cubin is missing at {}. Run: {TRITON_CUBIN_BUILD}",
            path.display()
        ))),
    }
}

fn param_list(record: &Json, expect: &[&str]) -> Result<Vec<Param>> {
    let list = record
        .get("parameters")
        .and_then(Json::as_array)
        .ok_or_else(|| Error::Launch("manifest parameters missing".into()))?;
    if list.len() != expect.len() {
        return Err(Error::Launch(format!(
            "manifest has {} parameters, run_gr passes {}",
            list.len(),
            expect.len()
        )));
    }
    let mut out = Vec::with_capacity(list.len());
    for (item, expected) in list.iter().zip(expect.iter()) {
        let name = item
            .get("name")
            .and_then(Json::as_str)
            .ok_or_else(|| Error::Launch("parameter name missing".into()))?;
        if name != *expected {
            return Err(Error::Launch(format!(
                "manifest parameter {name} != run_gr argument {expected}"
            )));
        }
        let ty = item
            .get("type")
            .and_then(Json::as_str)
            .ok_or_else(|| Error::Launch(format!("parameter {name} has no type")))?;
        let kind = if ty.starts_with('*') {
            ParamKind::Ptr
        } else if ty == "i32" {
            ParamKind::I32
        } else {
            return Err(Error::Launch(format!(
                "parameter {name} has unsupported type {ty}"
            )));
        };
        out.push(Param {
            name: name.to_string(),
            kind,
        });
    }
    Ok(out)
}

fn tile_from(record: &Json) -> Result<Tile> {
    let c = record
        .get("constexprs")
        .ok_or_else(|| Error::Launch("split constexprs missing".into()))?;
    let tile = Tile {
        n_q_heads: need_i32(c, "n_q_heads")?,
        n_kv_heads: need_i32(c, "n_kv_heads")?,
        q_len: need_i32(c, "q_len")?,
        page_size: need_i32(c, "page_size")?,
        head_dim: need_i32(c, "head_dim")?,
        block_h: need_i32(c, "BLOCK_H")?,
        block_rows: need_i32(c, "BLOCK_ROWS")?,
        block_n: need_i32(c, "BLOCK_N")?,
        hd_pad: need_i32(c, "HD_PAD")?,
    };
    let qck = need_i32(c, "QCK")?;
    let qcv = need_i32(c, "QCV")?;
    let kv_append = need_i32(c, "kv_append_len")?;
    let block_m = need_i32(c, "BLOCK_M")?;
    let final_flag = c
        .get("FINAL")
        .and_then(Json::as_bool)
        .ok_or_else(|| Error::Launch("FINAL constexpr missing".into()))?;
    let sinks = c
        .get("HAS_SINKS")
        .and_then(Json::as_bool)
        .ok_or_else(|| Error::Launch("HAS_SINKS constexpr missing".into()))?;
    if tile.n_q_heads != Q_HEADS
        || tile.n_kv_heads != KV_HEADS
        || tile.head_dim != HEAD_DIM
        || tile.page_size != PAGE_SIZE
        || tile.q_len != Q_LEN
        || qck != 3
        || qcv != 3
        || kv_append != Q_LEN
        || block_m != 1
        || tile.block_h != 16
        || tile.block_rows != 16
        || tile.block_n != 32
        || tile.hd_pad != 256
        || final_flag
        || sinks
    {
        return Err(Error::Launch(
            "split cubin constexprs are not the 27B q_len-1 cq3 decode tile".into(),
        ));
    }
    Ok(tile)
}

fn check_combine_tile(kernels: &Json, tile: &Tile) -> Result<()> {
    let record = kernels
        .get("_paged_attn_decode_combine_kernel")
        .ok_or_else(|| Error::Launch("manifest has no combine kernel".into()))?;
    let c = record
        .get("constexprs")
        .ok_or_else(|| Error::Launch("combine constexprs missing".into()))?;
    let same = need_i32(c, "n_q_heads")? == tile.n_q_heads
        && need_i32(c, "n_kv_heads")? == tile.n_kv_heads
        && need_i32(c, "head_dim")? == tile.head_dim
        && need_i32(c, "q_len")? == tile.q_len
        && need_i32(c, "HD_PAD")? == tile.hd_pad
        && need_i32(c, "BLOCK_H")? == tile.block_h
        && need_i32(c, "BLOCK_ROWS")? == tile.block_rows
        && need_i32(c, "BLOCK_M")? == 1
        && need_i32(c, "QCV")? == 3
        && c.get("HAS_SINKS").and_then(Json::as_bool) == Some(false);
    if !same {
        return Err(Error::Launch(
            "combine cubin constexprs do not match the split cubin".into(),
        ));
    }
    Ok(())
}

fn need_i32(obj: &Json, key: &str) -> Result<i32> {
    let v = obj
        .get(key)
        .and_then(Json::as_i64)
        .ok_or_else(|| Error::Launch(format!("constexpr {key} missing")))?;
    i32_fit(v, key)
}

// --- manifest JSON. The file is `json.dumps` output: objects, arrays,
// strings, numbers, bools, and null. No other crate parses JSON here.

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

    #[test]
    fn split_config_matches_attention_cpp() {
        assert_eq!(split_config(1, 256, 1, 32, 41).unwrap(), (3, 96));
        assert_eq!(split_config(2, 256, 1, 32, 41).unwrap(), (5, 128));
        assert_eq!(splits_cap(4, 82).unwrap(), 41);
    }

    #[test]
    fn split_config_rejects_non_positive() {
        assert!(split_config(0, 256, 1, 32, 41).is_err());
        assert!(splits_cap(0, 82).is_err());
    }

    #[test]
    fn manifest_parser_keeps_parameter_order() {
        let text = r#"{
            "kernels": {
                "_paged_attn_decode_split_kernel": {
                    "shared": 45056,
                    "parameters": [
                        {"name": "q", "type": "*fp16"},
                        {"name": "split_len", "type": "i32"}
                    ],
                    "constexprs": {"FINAL": false, "q_len": 1}
                }
            }
        }"#;
        let v = parse_json(text).unwrap();
        let params = v
            .get("kernels")
            .unwrap()
            .get("_paged_attn_decode_split_kernel")
            .unwrap()
            .get("parameters")
            .unwrap()
            .as_array()
            .unwrap();
        assert_eq!(params[0].get("name").unwrap().as_str(), Some("q"));
        assert_eq!(params[1].get("type").unwrap().as_str(), Some("i32"));
        assert_eq!(
            v.get("kernels")
                .unwrap()
                .get("_paged_attn_decode_split_kernel")
                .unwrap()
                .get("shared")
                .unwrap()
                .as_u32(),
            Some(45056)
        );
        assert_eq!(
            v.get("kernels")
                .unwrap()
                .get("_paged_attn_decode_split_kernel")
                .unwrap()
                .get("constexprs")
                .unwrap()
                .get("FINAL")
                .unwrap()
                .as_bool(),
            Some(false)
        );
    }

    #[test]
    fn on_disk_manifest_is_the_27b_decode_tile() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/triton-cubins/manifest.json");
        if !path.is_file() {
            eprintln!(
                "skipping manifest tile check: {} is missing",
                path.display()
            );
            return;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let manifest = parse_json(&text).expect("manifest.json");
        let kernels = manifest.get("kernels").expect("kernels");
        let split = split_record(kernels).unwrap();
        let tile = tile_from(split).unwrap();
        check_combine_tile(kernels, &tile).unwrap();
        param_list(
            split,
            &[
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
                "split_len",
                "num_pages_per_seq",
                "num_splits",
                "sinks",
                "global_scratch",
                "profile_scratch",
            ],
        )
        .unwrap();
        let combine = kernels.get("_paged_attn_decode_combine_kernel").unwrap();
        param_list(
            combine,
            &[
                "partial_o",
                "partial_ml",
                "out",
                "h32",
                "num_splits",
                "sinks",
                "global_scratch",
                "profile_scratch",
            ],
        )
        .unwrap();
        assert_eq!(split.get("num_warps").and_then(Json::as_u32), Some(4));
        assert_eq!(combine.get("num_warps").and_then(Json::as_u32), Some(4));
        let shared = split.get("shared").and_then(Json::as_u32).unwrap();
        assert!(shared > 0, "split shared {shared}");
    }

    #[test]
    fn host_checks_do_not_launch() {
        let bad = unsafe {
            mul_sigmoid(
                std::ptr::dangling_mut(),
                std::ptr::dangling(),
                1,
                std::ptr::null_mut(),
            )
        };
        assert!(bad.is_err());
        let bad = unsafe {
            deinterleave_qg(
                std::ptr::dangling(),
                std::ptr::dangling_mut(),
                std::ptr::dangling_mut(),
                7,
                8,
                std::ptr::null_mut(),
            )
        };
        assert!(bad.is_err());
    }
}

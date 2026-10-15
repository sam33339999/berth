//! Prompt prefill for one sequence.
//!
//! The fork's `Job.prefill` (`generator/job.py`) forwards `ids[:-1]` only.
//! The last prompt token is a decode step ([`Model::decode_step`]), which
//! also produces the logits. One call of `Job.prefill` runs one chunk.
//! `Generator.iterate` calls it again until `kv_position == len(ids) - 1`.
//!
//! Chunking (`job.py` `prefill`, `constants.py` `PAGE_SIZE`)
//! ----------------------------------------------------------
//! `max_chunk_size` is the serve default 2048 (`generator.py`). There is no
//! prefix-cache hit and no multimodal span on this path. From `kv_position`:
//!
//! * `end = align_down(pos + max_chunk_size, 256)`, then `min(end, len-1)`
//! * recurrent models (`qwen3_5.py` sets `recurrent_states`) cut the chunk
//!   at `last_page_b = (len-1) // 256 * 256` when `pos < last_page_b <= end`
//!
//! [`chunk_bounds`] is that rule. 1500 tokens is `[0, 1280)` then
//! `[1280, 1499)`. 100 tokens is `[0, 99)`.
//!
//! One chunk (`model_ls.py` `prefill_ls`, `transformer.py` `forward`)
//! --------------------------------------------------------------------
//! Batch 1. The residual is fp32 `[1, n, 5120]`, `n = end - start`. Embed
//! the `n` ids (`Embedding` `out_dtype` is fp32). Then each
//! `TransformerBlock`, absolute positions `start .. end`:
//!
//! ```text
//! h  = rms_norm(x)            input_layernorm, fp32 -> fp16, rows flattened
//! y  = mixer(h)               Gdn or Attention, fp16 -> fp32
//! h2 = rms_norm_res_in(x, y)  post_attention_layernorm; x += y
//! m  = mlp(h2)                gate, up, silu_mul, down; fp16 -> fp32
//! x  = x + m
//! ```
//!
//! `cache_seqlens` is `start`. Rope positions are `start .. end-1`. The GDN
//! recurrent slot is the same slot across chunks. Attention writes cq3 pages
//! at `cache_seqlens`.
//!
//! The last KV module is layer 63 (full attention). `prefill_ls` sets
//! `params["prefill"]` only for that module and then breaks. The block still
//! runs the mixer, so the cache is updated, then returns the incoming
//! residual and skips the MLP (`transformer.py`). Its digest equals layer
//! 62. Final norm and `lm_head` do not run. The following decode step does.
//!
//! `n > 8` on every chunk this rule emits for a prompt of at least a few
//! pages, so the MLP is not `BC_GatedMLP::run_bszN` (`mlp.py` `MAX_BSZN`).
//! `n > 16` skips `BC_GatedDeltaNetSplit` and `BC_Attention` (`MAX_QLEN`).
//! Those graphs stay on the decode step.
//!
//! Kernel APIs still to land
//! -------------------------
//! Tasks LIN, GDNP, and ATTNP own the implementations. Names below are the
//! fork ops. Template arguments are the 27B prefill profile (K = 4 except
//! `lm_head` K = 6, which is the decode step, not a prefill chunk).
//!
//! Already in this crate, shared with decode: `rms_norm` /
//! `rms_norm_kernel`, `rms_norm_res_in`, `silu_mul` / `act_mul_kernel_h<0>`,
//! the fp32 residual add, the embedding gather, and the last-token
//! `lm_head` `exl3_gemm` K = 6. Prefill does not call them through a new API.
//!
//! LIN (`modules/quant/exl3.py`). `rows = numel / last_dim`
//! (`AUTO_RECONSTRUCT_THRESHOLD` is 144, `MAX_RECONSTRUCT_SLICE_N` is 32768).
//!
//! * `rows <= 144` and `rows > 1`: `exl3_gemm`. The 100-token chunk (n = 99)
//!   takes this, not a reconstruct. Profile:
//!   `exl3_gemm_kernel<4, (bool)0, 2, 64, 32, 128, 4, 3>` and the fp32-out
//!   sibling `(bool)1` in that second flag. The int8 gemv
//!   `exl3_gemv_int8_sq_kernel` is the m = 1 decode path (`EXL3_INT8_GEMV=2`,
//!   `EXL3_INT8_GEMV_MAX_K=5`), not these chunks.
//! * `rows > 144`: `reconstruct_hgemm`. Fused when both features are
//!   divisible by 128 and `rows >= 1024`: `reconstruct_had_slice`
//!   (`reconstruct_had_kernel<4, 2>`) then `hgemm_recon`. No standalone Hadamard.
//!   Otherwise `had_r_128` (`had_hf_r_128_kernel`, then `had_ff_r_128_kernel`
//!   on the output), `reconstruct` / `reconstruct_slice`
//!   (`reconstruct_kernel<4, 2>`), `hgemm_recon`, `had_r_128`. The 1280-row
//!   chunk is fused. The 219-row chunk is not. `EXL3_HGEMM_F16ACC=auto`
//!   selects `f16acc::gemm_kernel<(bool)0, 128, (bool)0>` and
//!   `f16acc::gemm_kernel<(bool)1, 128, (bool)0>` (fp32 out). 27B out-features
//!   used here are below 32768, so the non-slice entry is the one taken; the
//!   slice entry is the same branch when `out_features` is larger.
//! * `m <= 32` bundles, instead of the separate linears: attention
//!   `exl3_mgemm` through `multi_qg` (`attn.py` `project_qkv`); MLP
//!   `exl3_mgemm` through `multi_gu` when `8 < m <= 32`; GDN `multi_qkvz`
//!   when it is set and `m <= 32` (`gated_delta_net.py`). The profiled
//!   chunks are all `m > 32`, so they miss these and use separate projections.
//!   A short tail chunk would not. Profiled `exl3_mgemm_kernel<4, ...>`
//!   launches are the decode step.
//!
//! GDNP (`gated_delta_net.py` torch path, `gated_delta_net_fn/`, `vendor/fla/`).
//! Qwen3.5 is split projections (`qkv_proj`, `z_proj`, `b_proj`, `a_proj`),
//! not fused `qkvz`. Those four linears are LIN.
//!
//! * `gated_delta_net_fused_op_2` (`gated_delta_net_fused_op_2_kernel`): beta
//!   bf16 and g fp32 from `b` and `a`. (`gated_delta_net_fused_op` is the
//!   fused-qkvz architecture. `gated_delta_net_fused_op_3` and
//!   `gdn_ba_gemv_kernel` are the decode BC path.)
//! * `causal_conv1d_update`. `seqlen > 32` (`MAX_CUDA_SEQLEN`) skips
//!   `cuda_causal_conv1d_update` / `conv1d_update_kernel` (that kernel is the
//!   decode step). Triton slotted: `seqlen <= 256` launches
//!   `_causal_conv1d_update_slotted_kernel` only; `seqlen > 256` launches
//!   `_causal_conv1d_update_slotted_output_kernel` and
//!   `_causal_conv1d_update_slotted_state_kernel`. The 1280 chunk splits.
//!   The 219 and 99 chunks do not.
//! * Delta rule (`gated_delta_rule.py`): `seqlen >= num_v_heads` (48) and
//!   history off calls `chunk_gated_delta_rule` with
//!   `use_qk_l2norm_in_kernel` and `chunk_size` 64. Kernels, in order:
//!   `l2norm_fwd_kernel`, `chunk_local_cumsum_scalar_kernel` (scale
//!   `RCP_LN2` = 1.4426950216), `chunk_gated_delta_rule_fwd_kkt_solve_kernel`,
//!   `recompute_w_u_fwd_kernel`, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`,
//!   `chunk_fwd_kernel_o`. The final state is copied into
//!   `recurrent_state[slot, 0]`. `seqlen < 48` would take
//!   `cuda_recurrent_gated_delta_rule` /
//!   `cuda_recurrent_gated_delta_rule_kernel_128` instead. The profiled
//!   chunks are all `>= 48`; the decode step is the recurrent kernel.
//! * `gated_rms_norm` (`gated_rms_norm_kernel`) with the `z` gate, then
//!   `o_proj` (LIN).
//!
//! ATTNP (`attn.py` `decode_flash_attn`, `attention_fn/dispatch.py`,
//! `triton_paged.py`). Eager, because `seqlen > 16`.
//!
//! * `m > 32`: separate `q_proj` / `k_proj` / `v_proj` (LIN). Interleaved
//!   gate, `head_dim % 8 == 0`, fp16: `deinterleave_qg`
//!   (`deinterleave_qg_kernel`). `m <= 32` uses LIN's `multi_qg` `exl3_mgemm`
//!   instead.
//! * q/k RMSNorm is fused into `rope` (`rope_kernel`), via `RoPE.apply` with
//!   the norm tensors. Not a standalone norm launch.
//! * `EXL3_QC_STAGING=1`: `CacheLayer_quant.update_kv_direct` ->
//!   `quant_cache_paged` (`quant_cache_paged_kernel<3, 3>`), then
//!   `fn_triton_paged_attn_prefill_qc` (`q_len > 16`) ->
//!   `paged_attn_triton_prefill` -> `_paged_attn_prefill_kernel`. New K/V are
//!   already in the cache (`pre_appended_len = q_len`). When `q_len >= 256`
//!   (`EXL3_QC_PF_TWO_PASS_MIN_Q`) the same function first calls
//!   `dequant_cache_paged_window` (`dequant_cache_paged_kernel<3, 3>`) into
//!   an fp16 scratch and runs the fp16 prefill kernel over that. The 1280
//!   chunk does. The 219 and 99 chunks read the packed cache inside
//!   `_paged_attn_prefill_kernel` and do not dequant. `_paged_attn_prefill_combine_kernel`
//!   runs only when `num_splits > 1` (`bound_kv >= 8192`). `bound_kv` uses
//!   the job's allocated block-table width ([`prefill_block_pages`]), which
//!   is constant across chunks. The 100- and 1500-token prompts stay under
//!   the threshold. 8190 and 8400 do not, so those chunks combine.
//! * `mul_sigmoid_` (`mul_sigmoid_kernel_h`) for the interleaved gate, then
//!   `o_proj` (LIN).
//! * Decode-only, not a prefill chunk: `_paged_attn_decode_split_kernel`,
//!   `_paged_attn_decode_combine_kernel`.
//!
//! [`Model::prefill`] runs this loop. [`Model::prefill_chunk`] is one chunk
//! and [`Model::prefill_prompt`] runs `ids[:-1]`. Digest hex is not computed
//! in the library (`blake2` is a test dependency). The reference test hashes
//! residuals from [`Model::prefill_with`], and the cache and GDN state from
//! [`Model::attn_cache`] and [`Model::gdn_state`].

use crate::model::Model;
use crate::{Error, Gpu, Result};

/// `constants.py` `PAGE_SIZE`. Chunk ends and the KV page share it.
pub const PAGE_SIZE: u32 = 256;

/// `Generator` default `max_chunk_size`. The serve does not override it.
pub const MAX_CHUNK_SIZE: u32 = 2048;

/// One fp32 residual digest, fork shape `[1, n, hidden]`.
///
/// A leading 1 is the batch. The bytes match `[n, hidden]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockDigest {
    pub layer: u32,
    /// `"float32"`.
    pub dtype: String,
    pub shape: Vec<usize>,
    /// blake2b-256 hex of the little-endian residual.
    pub digest: String,
}

/// Digests of one prefill chunk, layers in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkDigest {
    /// Absolute start, included.
    pub start: u32,
    /// Absolute end, excluded. The fork's `prefill_end`.
    pub end: u32,
    pub blocks: Vec<BlockDigest>,
}

/// blake2b-256 of one cache or state tensor, little-endian.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDigest {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub digest: String,
}

/// Used rows of one attention layer after `ids[:-1]` have been written.
///
/// `qk` and `qv` are int32, `sk` and `sv` are fp16. The bytes are logical
/// block-table order: page `p` is physical `block_table[p]`, and only
/// `min(PAGE_SIZE, tokens_written - p * PAGE_SIZE)` rows. `tokens_written`
/// is `prompt_len - 1`. The tail of a partial page is not included.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttnCacheDigest {
    pub layer: u32,
    pub qk: TensorDigest,
    pub qv: TensorDigest,
    pub sk: TensorDigest,
    pub sv: TensorDigest,
}

/// One GDN layer's live slot after prefill.
///
/// `conv` is bf16 (the raw 16-bit pattern). `recurrent` is fp32.
/// `max_batch_size` 1 and `max_history` 0, so the slot is the allocation:
/// conv `[fdim, conv_kernel]`, recurrent `[1, num_v_heads, k_dim, v_dim]`
/// at history index 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GdnStateDigest {
    pub layer: u32,
    pub conv: TensorDigest,
    pub recurrent: TensorDigest,
}

/// Little-endian bytes of one tensor, for a caller to hash.
///
/// [`Model::attn_cache`] and [`Model::gdn_state`] return these. The digest
/// hex stays in the test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostTensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// Used rows of one attention layer, in block-table order.
///
/// `qk` and `qv` are int32. `sk` and `sv` are fp16. The tail of a partial
/// page is not included.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttnCacheImage {
    pub layer: u32,
    pub qk: HostTensor,
    pub qv: HostTensor,
    pub sk: HostTensor,
    pub sv: HostTensor,
}

/// One GDN layer's live slot.
///
/// `conv` is bf16. `recurrent` is fp32. `max_batch` 1 and `max_history` 0,
/// so the allocation is the slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GdnStateImage {
    pub layer: u32,
    pub conv: HostTensor,
    pub recurrent: HostTensor,
}

/// Chunk ranges from [`Model::prefill_prompt`].
///
/// `blocks`, `attn`, and `gdn` are empty. The forward has run; the bytes are
/// read through [`Model::prefill_with`], [`Model::attn_cache`], and
/// [`Model::gdn_state`]. Logits are not here. They come from
/// [`Model::decode_step`] on the last prompt token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefillOutput {
    pub chunks: Vec<ChunkDigest>,
    pub attn: Vec<AttnCacheDigest>,
    pub gdn: Vec<GdnStateDigest>,
}

/// Pages in the block table prefill sees for one job.
///
/// The fork allocates them in `Sequence.prepare` before the first chunk:
/// `max_rq_tokens` defaults to `max_new_tokens + 1` when there is no draft
/// (`Job.prepare_for_queue`), and `total_pages` is
/// `ceil((len(sequence_ids) + max_rq_tokens) / PAGE_SIZE)`. Every chunk of
/// that job is launched with this width. It is not the 16-padded decode
/// width from `iterate_gen`.
#[must_use]
pub fn prefill_block_pages(prompt_len: u32, max_new_tokens: u32) -> u32 {
    let max_rq = u64::from(max_new_tokens) + 1;
    let total = u64::from(prompt_len) + max_rq;
    let pages = total.div_ceil(u64::from(PAGE_SIZE));
    u32::try_from(pages).unwrap_or(u32::MAX).max(1)
}

/// Chunk ranges `[start, end)` covering `ids[:-1]`.
///
/// `prompt_len` is `ids.len()`, including the last token that prefill does
/// not run. Empty when `prompt_len < 2` (a one-token prompt is already
/// `is_prefill_done`). See the module comment for the fork's cuts.
pub fn chunk_bounds(prompt_len: u32) -> Vec<(u32, u32)> {
    let max_chunk = u64::from(MAX_CHUNK_SIZE);
    let page = u64::from(PAGE_SIZE);
    let mut out = Vec::new();
    if prompt_len < 2 {
        return out;
    }
    let limit = u64::from(prompt_len) - 1;
    let mut pos = 0u64;
    while pos < limit {
        let mut end = (pos + max_chunk) / page * page;
        if end > limit {
            end = limit;
        }
        let last_page_b = limit / page * page;
        if pos < last_page_b && last_page_b <= end {
            end = last_page_b;
        }
        // Serve's max_chunk is a multiple of the page, so `end` stays above
        // `pos`. Stop rather than spin if a caller ever changes that.
        if end <= pos {
            break;
        }
        out.push((pos as u32, end as u32));
        pos = end;
    }
    out
}

impl Model {
    /// Prefill one chunk of absolute positions `[start, start + ids.len())`.
    ///
    /// `ids` is that chunk, not the whole prompt. The block loop and the
    /// per-layer ops are the module comment. The residual stream, the
    /// attention pages, and the GDN slot are updated in place. Layer 63
    /// returns the residual it was given.
    ///
    pub fn prefill_chunk(&mut self, gpu: &Gpu, ids: &[u32], start: u32) -> Result<()> {
        self.prefill_chunk_at(gpu, ids, start)
    }

    /// Prefill `ids[:-1]` in [`chunk_bounds`] order.
    ///
    /// Does not decode the last id and does not write logits. The caller
    /// runs [`Model::decode_step`] on `ids[ids.len() - 1]` at position
    /// `ids.len() - 1` when it wants the first sampled token. A prompt shorter
    /// than two tokens does not launch anything.
    ///
    /// The returned chunk ranges are the fork's cuts. Block, cache, and state
    /// digests are not filled here.
    pub fn prefill_prompt(&mut self, gpu: &Gpu, ids: &[u32]) -> Result<PrefillOutput> {
        let n = u32::try_from(ids.len())
            .map_err(|_| Error::Input("prompt is longer than u32::MAX".into()))?;
        let bounds = chunk_bounds(n);
        if ids.len() >= 2 {
            self.prefill(gpu, &ids[..ids.len() - 1])?;
        }
        Ok(PrefillOutput {
            chunks: bounds
                .into_iter()
                .map(|(start, end)| ChunkDigest {
                    start,
                    end,
                    blocks: Vec::new(),
                })
                .collect(),
            attn: Vec::new(),
            gdn: Vec::new(),
        })
    }
}

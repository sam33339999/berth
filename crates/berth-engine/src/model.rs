//! The whole text model for one decode step: embedding, 64 blocks, final
//! norm, lm_head.
//!
//! Owner: task MODEL. Uses [`crate::gdn::GdnLayer`], [`crate::attn::AttnLayer`],
//! and [`crate::mlp::MlpLayer`] through their `load` / `forward`. Between
//! requests, [`Model::reset_sequence`] also calls each mixer's state reset.
//!
//! One block follows [`crate`]'s decode sketch, which is
//! `TransformerBlock.forward` (`modules/transformer.py`) at batch 1:
//!
//! ```text
//! h  = rms_norm(x)                 input_layernorm, fp32 x -> fp16 h
//! y  = mixer(h)                    GdnLayer or AttnLayer, fp16 -> fp32
//! h2 = rms_norm_res_in(y, x)       post_attention_layernorm; x += y in place
//! m  = mlp(h2)                     MlpLayer, fp16 -> fp32
//! x  = x + m                       add_f32
//! ```
//!
//! The kernel's `x` argument to `rms_norm_res_in` is the mixer output. The
//! residual stream is the buffer that is updated in place.
//!
//! # Embedding residency
//!
//! `model.language_model.embed_tokens.weight` is BF16 `[248320, 5120]`,
//! 2,542,796,800 bytes (about 2.54 GB). [`Model::load`] keeps that whole
//! table on the device. Decode reads one row per token
//! ([`berth_kernels::ops::embed_tokens`]); the row never crosses PCIe.
//! The measured decode weights without the table are about 13.2 GB
//! (`docs/kernels.md`), so the table fits in the 24 GB card beside them
//! (`docs/memory.md` keeps this stack in VRAM). The fork's module sets
//! `prefer_cpu` and `model_ls.py` loads it on the host; that choice is for
//! the service, which also keeps a draft model on the same card.

use std::path::{Path, PathBuf};

use berth_cuda::{ByteBuf, PinnedBuf};
use berth_exl3::{Checkpoint, DType};
use berth_kernels::attn::PagedDecode;
use berth_kernels::ops::{self, NormWeight};
use berth_kernels::OutDtype;
use berth_plan::{Ffn, LayerKind, Mixer, ModelPlan, ModelShape, ARCH_QWEN3_5};

use crate::attn::AttnLayer;
use crate::device::{self, ptr, Exl3Linear};
use crate::gdn::GdnLayer;
use crate::mlp::MlpLayer;
use crate::{Error, Gpu, Result};

/// Tensor prefix of every Qwen3.5 text module.
///
/// `Qwen3_5VLModel`, `Qwen3_5Model`, `Qwen3_5VLMoeModel`, and `Qwen3_5MoeModel`
/// all pass `key_prefix = "model.language_model"` (`architecture/qwen3_5.py`).
/// `lm_head` is the exception: its key is `"lm_head"`.
const TEXT_PREFIX: &str = "model.language_model";

/// Full-attention KV page, in tokens. `AttnLayer` sizes its cache in these
/// pages (`berth_kernels::attn`, page size 256).
const ATTN_PAGE_TOKENS: u32 = 256;

/// `constant_bias` passed to every Qwen3.5 text RMSNorm.
///
/// `Qwen3_5BaseModel` sets `constant_bias = 1.0` on `input_layernorm`,
/// `post_attention_layernorm`, and the final `norm`
/// (`architecture/qwen3_5.py`). The kernel adds it to the weight, so the
/// scale is `1 + w`.
const QWEN35_NORM_BIAS: f32 = 1.0;

/// `constant_scale` for those norms. `RMSNorm.__init__` defaults it to 1
/// (`modules/rmsnorm.py`); the architecture does not override it.
const QWEN35_NORM_SCALE: f32 = 1.0;

/// Epsilon, bias, and scale for one text RMSNorm.
///
/// `epsilon` is `text_config.rms_norm_eps` (`1e-6` on the 27B). Bias and
/// scale are the Qwen3.5 constructor values above, not checkpoint tensors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextNorm {
    /// Added inside the reciprocal square root, before the multiply.
    pub epsilon: f32,
    /// Added to the bf16 weight. Qwen3.5 uses 1.
    pub constant_bias: f32,
    /// Multiplies the normalized row. Qwen3.5 uses 1.
    pub constant_scale: f32,
}

impl TextNorm {
    /// Qwen3.5 bias and scale, with `epsilon` from the checkpoint.
    pub fn qwen3_5(epsilon: f32) -> Self {
        Self {
            epsilon,
            constant_bias: QWEN35_NORM_BIAS,
            constant_scale: QWEN35_NORM_SCALE,
        }
    }
}

// Attention carries the paged cache and the one-slot sequence staging
// buffers, so it is wider than a GDN layer. Both arms stay unboxed: the
// slot is moved only at load, and decode matches it by reference.
#[allow(clippy::large_enum_variant)]
enum MixerSlot {
    Gdn(GdnLayer),
    FullAttn(AttnLayer),
}

struct Block {
    input_norm: ByteBuf,
    post_norm: ByteBuf,
    mixer: MixerSlot,
    mlp: MlpLayer,
}

/// The loaded 27B text stack and its per-sequence state.
pub struct Model {
    hidden: i32,
    vocab: i32,
    max_tokens: u32,
    norm: TextNorm,
    /// Full bf16 embedding table. See the module comment.
    embed: ByteBuf,
    blocks: Vec<Block>,
    final_norm: ByteBuf,
    lm_head: Exl3Linear,
    /// Triton split/combine pair, one copy for every attention layer.
    paged: PagedDecode,
    /// Token id and the shared position word, staged before each step.
    staged: StagedInputs,
    /// fp32 `[hidden]` residual stream.
    residual: ByteBuf,
    /// fp16 `[hidden]`. Mixer input, then the final norm output.
    norm_out: ByteBuf,
    /// fp32 `[hidden]` mixer output.
    mixer_out: ByteBuf,
    /// fp16 `[hidden]` MLP input (post-attention layernorm).
    mlp_in: ByteBuf,
    /// fp32 `[hidden]` MLP output.
    mlp_out: ByteBuf,
    /// fp16 `[vocab]` logits from the last successful [`Self::decode_step`].
    logits: ByteBuf,
    /// fp32 residual after each block, copied out on the decode stream.
    ///
    /// `device.rs` has no device-to-device copy. `rms_norm_res_in` updates
    /// the residual in place, so the next block would overwrite a device
    /// snapshot that aliased it. Each block instead queues a device-to-host
    /// copy into pinned memory on the same stream, before the next kernel
    /// writes the residual. The copy is asynchronous. [`Self::synchronize`]
    /// (or any sync of `gpu.stream`) finishes it before [`Self::block_residual`]
    /// is read.
    block_residuals: Vec<PinnedBuf<u8>>,
    /// Set when a `decode_step` has queued its snapshot copies.
    ///
    /// The bytes are not ready to read until the stream is synchronized.
    captured: bool,
}

/// How many decode steps may be queued before [`Model::synchronize`].
///
/// Each step takes its own pinned slot so the host can write the next token
/// while an earlier async copy is still in flight. 128 covers the 64-step
/// bench with the warm-up step included, without a host sync inside
/// `decode_step`.
const STAGED_STEPS: usize = 128;

/// Device words the kernels read, plus the pinned slots that fill them.
///
/// `token` is one i32, the embedding id. `seq` is one i32 that every
/// attention layer uses as `cache_seqlens` and as the rope `positions`
/// pointer. Qwen3.5 text decode has no MRoPE offset, so the position id
/// and the cache length before the token are the same integer. The fork
/// assigns `positions = cache_seqlens` in that case (`modules/attn.py`).
struct StagedInputs {
    token: ByteBuf,
    seq: ByteBuf,
    token_host: Vec<PinnedBuf<u8>>,
    seq_host: Vec<PinnedBuf<u8>>,
    used: usize,
}

impl StagedInputs {
    fn new(gpu: &Gpu) -> Result<StagedInputs> {
        let mut token_host = Vec::with_capacity(STAGED_STEPS);
        let mut seq_host = Vec::with_capacity(STAGED_STEPS);
        for _ in 0..STAGED_STEPS {
            token_host.push(PinnedBuf::zeroed(&gpu.ctx, 4)?);
            seq_host.push(PinnedBuf::zeroed(&gpu.ctx, 4)?);
        }
        Ok(Self {
            token: gpu.zeroed(4)?,
            seq: gpu.zeroed(4)?,
            token_host,
            seq_host,
            used: 0,
        })
    }

    fn enqueue(&mut self, gpu: &Gpu, token: [u8; 4], seq: [u8; 4]) -> Result<()> {
        if self.used >= STAGED_STEPS {
            return Err(Error::Input(format!(
                "staged {STAGED_STEPS} decode steps without Model::synchronize"
            )));
        }
        let index = self.used;
        self.token_host[index]
            .as_mut_slice()
            .copy_from_slice(&token);
        self.seq_host[index].as_mut_slice().copy_from_slice(&seq);
        let StagedInputs {
            token: token_dev,
            seq: seq_dev,
            token_host,
            seq_host,
            used,
        } = self;
        // SAFETY: slot `index` is not written again until `reset`, and
        // `reset` runs only after `gpu.stream` has been synchronized. Both
        // device words outlive the copies. Copies already queued for earlier
        // slots use other pages. This copy is ordered after the previous
        // step's kernels on `gpu.stream`, so those kernels still read the
        // previous word.
        unsafe {
            token_dev.copy_from_pinned_async(&gpu.stream, &token_host[index])?;
            seq_dev.copy_from_pinned_async(&gpu.stream, &seq_host[index])?;
        }
        *used += 1;
        Ok(())
    }

    fn reset(&mut self) {
        self.used = 0;
    }
}

impl Model {
    /// Load every decode-path weight `plan` names onto the device, with
    /// caches sized for `max_tokens`.
    ///
    /// `max_tokens` is the attention cache length and must be a positive
    /// multiple of [`ATTN_PAGE_TOKENS`]. `plan` must be the dense Qwen3.5
    /// stack (`ARCH_QWEN3_5`): GDN or full attention, dense MLP, no PLE.
    ///
    /// The Triton cubins come from `target/triton-cubins` in this workspace;
    /// [`Self::load_with_cubins`] takes another directory.
    pub fn load(gpu: &Gpu, ckpt: &Checkpoint, plan: &ModelPlan, max_tokens: u32) -> Result<Self> {
        Self::load_with_cubins(gpu, ckpt, plan, max_tokens, &default_cubin_dir())
    }

    /// [`Self::load`], with the Triton attention cubins read from `cubin_dir`.
    pub fn load_with_cubins(
        gpu: &Gpu,
        ckpt: &Checkpoint,
        plan: &ModelPlan,
        max_tokens: u32,
        cubin_dir: &Path,
    ) -> Result<Self> {
        let shape = validate_plan(plan, max_tokens)?;
        let hidden = i32_width(plan.hidden_size, "hidden_size")?;
        let vocab = i32_width(plan.vocab_size, "vocab_size")?;
        checked_hidden(hidden)?;
        let norm = TextNorm::qwen3_5(read_rms_eps(ckpt)?);
        // Cubins are a few megabytes. Load them before the embedding table
        // so a missing `target/triton-cubins/` fails without that upload.
        let paged = PagedDecode::load(&gpu.ctx, cubin_dir)?;
        let embed = load_embedding(gpu, ckpt, plan.vocab_size, plan.hidden_size)?;
        let mut blocks = Vec::with_capacity(plan.layers.len());
        for layer in &plan.layers {
            let input_norm = load_norm(
                gpu,
                ckpt,
                &format!(
                    "{TEXT_PREFIX}.layers.{}.input_layernorm.weight",
                    layer.index
                ),
                plan.hidden_size,
            )?;
            let mixer = match layer.mixer {
                Mixer::Gdn => MixerSlot::Gdn(GdnLayer::load(gpu, ckpt, &shape, layer.index)?),
                Mixer::FullAttn => MixerSlot::FullAttn(AttnLayer::load(
                    gpu,
                    ckpt,
                    &shape,
                    layer.index,
                    max_tokens,
                )?),
                Mixer::Qsa => {
                    return Err(Error::Weights(format!(
                        "layer {}: QSA is not a 27B mixer",
                        layer.index
                    )))
                }
            };
            let post_norm = load_norm(
                gpu,
                ckpt,
                &format!(
                    "{TEXT_PREFIX}.layers.{}.post_attention_layernorm.weight",
                    layer.index
                ),
                plan.hidden_size,
            )?;
            let mlp = match layer.ffn {
                Ffn::DenseMlp => MlpLayer::load(gpu, ckpt, layer.index)?,
                Ffn::Moe { .. } => {
                    return Err(Error::Weights(format!(
                        "layer {}: MoE MLP is not the 27B decode path",
                        layer.index
                    )))
                }
            };
            blocks.push(Block {
                input_norm,
                post_norm,
                mixer,
                mlp,
            });
        }
        let final_norm = load_norm(
            gpu,
            ckpt,
            &format!("{TEXT_PREFIX}.norm.weight"),
            plan.hidden_size,
        )?;
        let lm_head = Exl3Linear::load(gpu, ckpt, "lm_head")?;
        if lm_head.size_k != hidden || lm_head.size_n != vocab {
            return Err(Error::Weights(format!(
                "lm_head is {}x{}, plan is {hidden}x{vocab}",
                lm_head.size_k, lm_head.size_n
            )));
        }
        let hidden_f32 = row_bytes(hidden, 4)?;
        let hidden_f16 = row_bytes(hidden, 2)?;
        let mut block_residuals = Vec::with_capacity(blocks.len());
        for _ in 0..blocks.len() {
            block_residuals.push(PinnedBuf::zeroed(&gpu.ctx, hidden_f32)?);
        }
        let staged = StagedInputs::new(gpu)?;
        // One device word for every attention layer. The address is the
        // allocation's, so moving `staged` later does not change it.
        let seq_addr = staged.seq.ptr().as_u64();
        for block in &mut blocks {
            if let MixerSlot::FullAttn(layer) = &mut block.mixer {
                layer.bind_sequence(seq_addr, seq_addr);
            }
        }
        Ok(Self {
            hidden,
            vocab,
            max_tokens,
            norm,
            embed,
            blocks,
            final_norm,
            lm_head,
            paged,
            staged,
            residual: gpu.zeroed(hidden_f32)?,
            norm_out: gpu.zeroed(hidden_f16)?,
            mixer_out: gpu.zeroed(hidden_f32)?,
            mlp_in: gpu.zeroed(hidden_f16)?,
            mlp_out: gpu.zeroed(hidden_f32)?,
            logits: gpu.zeroed(row_bytes(vocab, 2)?)?,
            block_residuals,
            captured: false,
        })
    }

    /// Run `token` at `position` through every block. Returns the fp16
    /// `[vocab]` logits buffer, valid until the next call.
    ///
    /// `position` is 0-based. The attention cache already holds `position`
    /// tokens and must have room for this one (`position < max_tokens`).
    ///
    /// The token id, rope positions, and `cache_seqlens` are written first,
    /// by async host-to-device copies from pinned slots on `gpu.stream`.
    /// The kernels that follow only read those device words. This function
    /// does not allocate, does not synchronize the host, and does not make
    /// a synchronous copy.
    ///
    /// # Warm-up
    ///
    /// The first step after process start can still synchronize, inside
    /// berth-kernels rather than here. Sliced `exl3_mgemm` (GDN qkv+z and
    /// attention qkv) and the cooperative `lm_head` GEMM autotune on a
    /// memory-cache miss (`csrc/autotune.cu`, `tune` calls
    /// `cudaStreamSynchronize` / `cudaEventSynchronize`). A hit in
    /// `g_launch_cache` launches and returns. The int8 GEMV workspace and
    /// the GEMM lock buffer are `cudaMalloc`'d on first use (`int8_sq.cu`
    /// `workspace`, `linear.cu` `berth_device_locks`), with a synchronous
    /// `cudaMemset`. One full step touches every decode key, including a
    /// disk hit that only fills the memory cache. Call [`Self::decode_step`]
    /// once, then [`Self::synchronize`], before treating a later step as
    /// free of host syncs and device allocations. Do not warm up on a cache
    /// you still need empty: the warm-up token is written into the K/V pages
    /// and the GDN state.
    ///
    /// # What still varies as a kernel argument
    ///
    /// With the block table fixed at load (`max_tokens / 256` pages) none of
    /// these change from token to token. The fork patches them on every
    /// replay anyway, because a growing table changes the width without
    /// recapturing. The next round writes those graph nodes:
    ///
    /// * Triton split, computed in `PagedDecode::launch` from `split_config`
    ///   of the `num_pages` and `splits_cap` this step passes through
    ///   [`AttnLayer`]: `split_len` (argument 11), `num_pages_per_seq`
    ///   (argument 12), `num_splits` (argument 13). Fork:
    ///   `BC_Attention::run_gr` records `GP_attn_split_len`,
    ///   `GP_attn_num_pages`, and `GP_attn_num_splits`
    ///   (`libtorch/attention.cpp`).
    /// * Triton combine: `num_splits` (argument 4). Fork: `GP_attn_num_splits`
    ///   on the combine kernel.
    /// * `quant_cache_paged`: `blocks_per_seq` (argument 8), set in
    ///   `AttnLayer::quant_launch` to `num_pages`. Fork:
    ///   `GP_qcache_blocks_per_seq` (`cache/q_cache.cu`).
    /// * Rope's scalar `position` (argument 13) is constant 0. The live
    ///   position is `positions[batch]` (`rope.cu` `get_pos`). The fork
    ///   still records `GP_rope_position` (`rope.cu`) for the mode where
    ///   the positions pointer is null.
    ///
    /// `cache_seqlens` and rope `positions` are device pointers, not
    /// scalars. The fork records the pointers (`GP_qcache_seqlens`,
    /// `GP_attn_seqlens`, `GP_rope_positions`) because the tensor storage
    /// can change between replays. Here the allocation lives as long as
    /// the model; only the int32 in it changes, via the copy below.
    ///
    /// [`Self::block_residual`], [`Self::final_norm`], and the logits are
    /// readable after [`Self::synchronize`]. [`Self::block_residual`] is the
    /// fp32 stream at the end of each layer.
    pub fn decode_step(&mut self, gpu: &Gpu, token: u32, position: u32) -> Result<&ByteBuf> {
        self.captured = false;
        self.check_step(token, position)?;
        self.stage_inputs(gpu, token, position)?;
        self.run_kernels(gpu)?;
        self.captured = true;
        Ok(&self.logits)
    }

    /// Wait until every copy and kernel queued by [`Self::decode_step`] has
    /// finished, then reuse the pinned staging slots.
    ///
    /// `gpu.stream` must be the stream the steps were launched on. After
    /// this returns, [`Self::block_residual`] and [`Self::final_norm`] are
    /// safe to read. `decode_step` does not call this.
    pub fn synchronize(&mut self, gpu: &Gpu) -> Result<()> {
        gpu.stream.synchronize()?;
        self.staged.reset();
        Ok(())
    }

    /// Drop per-sequence state so the next token is position 0.
    ///
    /// A fresh fork job with no prefix hit does the same thing and then
    /// prefills from the start (`generator/job.py` allocates pages, then
    /// `Cache.get_new_state` with `clear=True` when no GDN checkpoint was
    /// stashed). Prefix reuse is out of scope here.
    ///
    /// * Every GDN layer's conv and recurrent state is zeroed in its existing
    ///   device allocation (`GDNLayerState.clear`).
    /// * Every attention layer's cache length goes back to 0. The fork's
    ///   `Sequence` starts at `kv_position = 0` and `add_ref_unique` does
    ///   not scrub K/V bytes; the next write overwrites a position before
    ///   attention reads it. The live length is the shared `cache_seqlens`
    ///   word this model stages, plus each layer's private word for the
    ///   unbound path.
    ///
    /// Finishes any queued kernels first. After this returns, [`Self::logits`]
    /// is not readable until the next [`Self::decode_step`].
    pub fn reset_sequence(&mut self, gpu: &Gpu) -> Result<()> {
        self.synchronize(gpu)?;
        self.captured = false;
        for block in &mut self.blocks {
            match &mut block.mixer {
                MixerSlot::Gdn(layer) => layer.reset_state(gpu)?,
                MixerSlot::FullAttn(layer) => layer.reset_cache(gpu)?,
            }
        }
        // Attention is bound to this word (`bind_sequence`). Zeroing the
        // private layer words does not clear it.
        self.staged.seq.copy_from_host(&gpu.stream, &[0u8; 4])?;
        Ok(())
    }

    /// Number of decoder layers loaded.
    pub fn num_layers(&self) -> usize {
        self.blocks.len()
    }

    /// fp16 `[vocab]` logits of the last [`Self::decode_step`].
    ///
    /// The device buffer. Synchronize `gpu.stream` before copying it out.
    pub fn logits(&self) -> Result<&ByteBuf> {
        if !self.captured {
            return Err(Error::Input("decode_step has not completed".to_string()));
        }
        Ok(&self.logits)
    }

    /// Mutable fp16 `[vocab]` logits of the last [`Self::decode_step`].
    ///
    /// The sampler penalizes this row in place. Synchronize `gpu.stream`
    /// before the sampler reads it. The next [`Self::decode_step`] overwrites it.
    pub fn logits_mut(&mut self) -> Result<&mut ByteBuf> {
        if !self.captured {
            return Err(Error::Input("decode_step has not completed".to_string()));
        }
        Ok(&mut self.logits)
    }

    /// fp16 `[hidden]` final norm of the last [`Self::decode_step`].
    ///
    /// This is the buffer `lm_head` reads. Synchronize `gpu.stream` before
    /// copying it out.
    pub fn final_norm(&self) -> Result<&ByteBuf> {
        if !self.captured {
            return Err(Error::Input("decode_step has not completed".to_string()));
        }
        Ok(&self.norm_out)
    }

    /// fp32 `[hidden]` residual after layer `index` of the last
    /// [`Self::decode_step`].
    ///
    /// The slice is pinned host memory. It stays valid until the next
    /// `decode_step` or until `self` is dropped. Synchronize `gpu.stream`
    /// (see [`Self::synchronize`]) before reading it: the device-to-host
    /// copy is asynchronous.
    pub fn block_residual(&self, index: u32) -> Result<&[u8]> {
        if !self.captured {
            return Err(Error::Input("decode_step has not completed".to_string()));
        }
        let slot = self.block_residuals.get(index as usize).ok_or_else(|| {
            Error::Input(format!(
                "layer {index} is past {} layers",
                self.block_residuals.len()
            ))
        })?;
        Ok(slot.as_slice())
    }

    /// Gather one token from a bf16 table into fp32 `[hidden]`.
    ///
    /// `weight` is row-major BF16 with stride `hidden`. It must cover row
    /// `token` (a prefix of the table is enough). `out` is fp32 `[hidden]`.
    /// This is `F.embedding(ids, weight).float()`: `qwen3_5.Embedding` leaves
    /// `out_dtype` at `torch.float`, `normalize` false, and `multiplier` 1
    /// (`modules/embedding.py`).
    pub fn embed_token(
        gpu: &Gpu,
        weight: &ByteBuf,
        token: u32,
        hidden: i32,
        out: &mut ByteBuf,
    ) -> Result<()> {
        let hidden = checked_hidden(hidden)?;
        let id = i32::try_from(token)
            .map_err(|_| Error::Input(format!("token {token} does not fit in i32")))?;
        let need = row_span(token, hidden, 2)?;
        expect_bytes(weight, need, "embedding weight")?;
        expect_bytes(out, row_bytes(hidden, 4)?, "embedding output")?;
        let ids = berth_cuda::DeviceBuffer::from_host(&gpu.stream, &id.to_le_bytes())?;
        embed_ids(gpu, weight, &ids, hidden, 1, out)
    }

    /// `input_layernorm` and the final norm. fp32 `[hidden]` to fp16 `[hidden]`.
    ///
    /// `TransformerBlock` calls the input norm with `out_dtype = torch.half`.
    /// The final norm is constructed with `out_dtype = torch.half`
    /// (`architecture/qwen3_5.py`). Weight is one bf16 row.
    pub fn norm_fp32_to_fp16(
        gpu: &Gpu,
        x: &ByteBuf,
        weight: &ByteBuf,
        y: &mut ByteBuf,
        hidden: i32,
        norm: TextNorm,
    ) -> Result<()> {
        let hidden = checked_hidden(hidden)?;
        expect_row(x, hidden, 4, "norm input")?;
        expect_row(weight, hidden, 2, "norm weight")?;
        expect_row(y, hidden, 2, "norm output")?;
        // SAFETY: `x` is fp32, `weight` is bf16, and `y` is fp16, one row of
        // `hidden` each. They stay allocated until `gpu.stream` finishes.
        // Callers synchronize before reading `y`.
        unsafe {
            ops::rms_norm(
                ptr(x),
                ptr(weight),
                ptr(y),
                1,
                hidden,
                norm.epsilon,
                norm.constant_bias,
                norm.constant_scale,
                NormWeight::Bf16,
                gpu.raw_stream(),
            )?;
        }
        Ok(())
    }

    /// `post_attention_layernorm`: `residual += mixer`, then fp16 norm of the sum.
    ///
    /// `mixer` is the fp32 mixer output (the kernel's `x`). `residual` is the
    /// fp32 stream and is updated in place. `mlp_in` is the fp16 norm output.
    /// `TransformerBlock` takes this path when `can_fuse_residual` accepts
    /// the fp32 stream and the fp32 mixer output (`modules/transformer.py`).
    pub fn norm_residual(
        gpu: &Gpu,
        mixer: &ByteBuf,
        weight: &ByteBuf,
        mlp_in: &mut ByteBuf,
        residual: &mut ByteBuf,
        hidden: i32,
        norm: TextNorm,
    ) -> Result<()> {
        let hidden = checked_hidden(hidden)?;
        expect_row(mixer, hidden, 4, "mixer output")?;
        expect_row(weight, hidden, 2, "post-attention norm weight")?;
        expect_row(mlp_in, hidden, 2, "mlp input")?;
        expect_row(residual, hidden, 4, "residual")?;
        // SAFETY: `mixer` and `residual` are fp32, `weight` is bf16, and
        // `mlp_in` is fp16, one row of `hidden` each. `residual` is
        // overwritten with `residual + mixer`. All four stay allocated
        // until `gpu.stream` finishes.
        unsafe {
            ops::rms_norm_res_in(
                ptr(mixer),
                ptr(weight),
                ptr(mlp_in),
                ptr(residual),
                1,
                hidden,
                norm.epsilon,
                norm.constant_bias,
                norm.constant_scale,
                NormWeight::Bf16,
                gpu.raw_stream(),
            )?;
        }
        Ok(())
    }

    /// `x += m` in fp32. `mlp` is the MLP output. `residual` is both an
    /// addend and the destination, which is the block's `x += y`
    /// (`modules/transformer.py`, no MLP post-norm).
    pub fn add_residual_in_place(
        gpu: &Gpu,
        residual: &mut ByteBuf,
        mlp: &ByteBuf,
        hidden: i32,
    ) -> Result<()> {
        let hidden = checked_hidden(hidden)?;
        expect_row(residual, hidden, 4, "residual")?;
        expect_row(mlp, hidden, 4, "mlp output")?;
        let stream = gpu.raw_stream();
        let a = ptr(residual);
        let b = ptr(mlp);
        // SAFETY: both buffers are fp32 of `hidden` elements. The kernel
        // allows the destination to alias the first addend. They stay
        // allocated until `gpu.stream` finishes.
        unsafe {
            ops::add_f32(a, b, a, hidden, stream)?;
        }
        Ok(())
    }

    /// `lm_head` for one token. `x` is fp16 `[hidden]`; `logits` is fp16 `[vocab]`.
    ///
    /// The module is constructed without `out_dtype`, so `Linear.forward`
    /// uses `torch.half` (`modules/linear.py`).
    pub fn project_lm_head(
        gpu: &Gpu,
        lm_head: &mut Exl3Linear,
        x: &ByteBuf,
        logits: &mut ByteBuf,
    ) -> Result<()> {
        lm_head.forward(gpu, x, logits, OutDtype::F16)
    }

    fn check_step(&self, token: u32, position: u32) -> Result<()> {
        let vocab = u32::try_from(self.vocab).expect("vocab is a positive i32");
        if token >= vocab {
            return Err(Error::Input(format!(
                "token {token} is past the vocabulary ({vocab})"
            )));
        }
        if position >= self.max_tokens {
            return Err(Error::Input(format!(
                "position {position} is past the cache ({})",
                self.max_tokens
            )));
        }
        Ok(())
    }

    fn stage_inputs(&mut self, gpu: &Gpu, token: u32, position: u32) -> Result<()> {
        let id = i32::try_from(token).expect("token was checked against vocab");
        let seq = i32::try_from(position)
            .map_err(|_| Error::Input(format!("position {position} does not fit in i32")))?;
        // Same four bytes: rope reads them as uint32 (`positions[batch]`),
        // the cache kernels read them as int32 (`cache_seqlens[batch]`).
        self.staged
            .enqueue(gpu, id.to_le_bytes(), seq.to_le_bytes())
    }

    /// Kernel launches only. Capture this, not [`Self::stage_inputs`].
    ///
    /// The async copies of the token id and the position word are already
    /// queued on `gpu.stream`. No allocation, no host copy, no sync.
    /// The per-block residual snapshots are async device-to-host copies
    /// into pinned buffers allocated at load; they are how
    /// [`Self::block_residual`] observes the stream.
    fn run_kernels(&mut self, gpu: &Gpu) -> Result<()> {
        self.embed_step(gpu)?;
        let layers = self.blocks.len();
        for index in 0..layers {
            self.forward_block(gpu, index)?;
        }
        self.project(gpu)
    }

    fn embed_step(&mut self, gpu: &Gpu) -> Result<()> {
        let hidden = self.hidden;
        let Model {
            embed,
            staged,
            residual,
            ..
        } = self;
        embed_ids(gpu, embed, &staged.token, hidden, 1, residual)
    }

    fn forward_block(&mut self, gpu: &Gpu, index: usize) -> Result<()> {
        let hidden = self.hidden;
        let norm = self.norm;
        let Model {
            blocks,
            residual,
            norm_out,
            mixer_out,
            mlp_in,
            mlp_out,
            paged,
            block_residuals,
            ..
        } = self;

        Self::norm_fp32_to_fp16(
            gpu,
            residual,
            &blocks[index].input_norm,
            norm_out,
            hidden,
            norm,
        )?;
        match &mut blocks[index].mixer {
            MixerSlot::Gdn(layer) => layer.forward(gpu, norm_out, mixer_out)?,
            MixerSlot::FullAttn(layer) => layer.forward_bound(gpu, paged, norm_out, mixer_out)?,
        }
        Self::norm_residual(
            gpu,
            mixer_out,
            &blocks[index].post_norm,
            mlp_in,
            residual,
            hidden,
            norm,
        )?;
        blocks[index].mlp.forward(gpu, mlp_in, mlp_out)?;
        Self::add_residual_in_place(gpu, residual, mlp_out, hidden)?;
        // SAFETY: the copy is queued on `gpu.stream` before any later kernel
        // on that stream writes `residual`. The caller synchronizes that
        // stream before `block_residual` reads the pinned pages, and the
        // pages live on `self` until then.
        unsafe {
            residual.copy_to_pinned_async(&gpu.stream, &mut block_residuals[index])?;
        }
        Ok(())
    }

    fn project(&mut self, gpu: &Gpu) -> Result<()> {
        let hidden = self.hidden;
        let norm = self.norm;
        let Model {
            residual,
            final_norm,
            norm_out,
            lm_head,
            logits,
            ..
        } = self;
        Self::norm_fp32_to_fp16(gpu, residual, final_norm, norm_out, hidden, norm)?;
        Self::project_lm_head(gpu, lm_head, norm_out, logits)
    }
}

fn embed_ids(
    gpu: &Gpu,
    weight: &ByteBuf,
    ids: &ByteBuf,
    hidden: i32,
    n: i32,
    out: &mut ByteBuf,
) -> Result<()> {
    let hidden = checked_hidden(hidden)?;
    if n < 1 {
        return Err(Error::Input("embedding batch must be positive".into()));
    }
    let n_usize = usize::try_from(n).expect("n is positive and fits i32");
    expect_bytes(ids, n_usize * 4, "token ids")?;
    expect_bytes(out, row_bytes(hidden, 4)? * n_usize, "embedding output")?;
    // SAFETY: `weight` is bf16 with stride `hidden` and covers every id in
    // `ids` (checked by `embed_token` for one id, and by `load` for the full
    // vocabulary). `ids` holds `n` i32 values. `out` is fp32 `[n, hidden]`.
    // All three stay allocated until `gpu.stream` finishes.
    unsafe {
        ops::embed_tokens(
            ptr(weight),
            ptr(ids).cast::<i32>(),
            ptr(out),
            hidden,
            n,
            gpu.raw_stream(),
        )?;
    }
    Ok(())
}

fn validate_plan(plan: &ModelPlan, max_tokens: u32) -> Result<ModelShape> {
    if plan.arch_string != ARCH_QWEN3_5 {
        return Err(Error::Weights(format!(
            "decode load is the dense Qwen3.5 stack, got {}",
            plan.arch_string
        )));
    }
    if plan.layers.is_empty() {
        return Err(Error::Weights("plan has no layers".into()));
    }
    if max_tokens == 0 || !max_tokens.is_multiple_of(ATTN_PAGE_TOKENS) {
        return Err(Error::Input(format!(
            "max_tokens {max_tokens} must be a positive multiple of {ATTN_PAGE_TOKENS}"
        )));
    }
    if plan.embedding.hidden_size != plan.hidden_size
        || plan.final_norm.hidden_size != plan.hidden_size
        || plan.lm_head.hidden_size != plan.hidden_size
        || plan.embedding.vocab_size != plan.vocab_size
        || plan.lm_head.vocab_size != plan.vocab_size
    {
        return Err(Error::Weights(
            "plan embedding, final norm, and lm_head widths disagree".into(),
        ));
    }
    if plan.hidden_size == 0 || !plan.hidden_size.is_multiple_of(4) {
        return Err(Error::Weights(format!(
            "hidden_size {} must be a positive multiple of 4",
            plan.hidden_size
        )));
    }
    let mut layer_kinds = Vec::with_capacity(plan.layers.len());
    for (i, layer) in plan.layers.iter().enumerate() {
        if layer.index != u32::try_from(i).expect("layer count fits u32") {
            return Err(Error::Weights(format!(
                "layer {i} is numbered {}",
                layer.index
            )));
        }
        if layer.extra.is_some() {
            return Err(Error::Weights(format!(
                "layer {i}: PLE is not on the 27B decode path"
            )));
        }
        if !matches!(layer.ffn, Ffn::DenseMlp) {
            return Err(Error::Weights(format!(
                "layer {i}: MoE MLP is not the 27B decode path"
            )));
        }
        layer_kinds.push(match layer.mixer {
            Mixer::Gdn => LayerKind::LinearAttention,
            Mixer::FullAttn => LayerKind::FullAttention,
            Mixer::Qsa => return Err(Error::Weights(format!("layer {i}: QSA is not a 27B mixer"))),
        });
    }
    let num_layers = u32::try_from(plan.layers.len()).expect("layer count fits u32");
    Ok(ModelShape {
        arch_string: plan.arch_string.clone(),
        num_layers,
        layer_kinds,
        hidden_size: plan.hidden_size,
        head_dim: plan.head_dim,
        num_q_heads: plan.num_q_heads,
        num_kv_heads: plan.num_kv_heads,
        gdn: plan.gdn,
        moe: None,
        vocab_size: plan.vocab_size,
    })
}

fn default_cubin_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/triton-cubins")
}

/// `text_config.rms_norm_eps`, the value `Qwen3_5VLBaseConfig` reads.
///
/// `ModelConfig` does not keep this field. The fork reads it from
/// `config.json` (`read_cfg(float, "text_config->rms_norm_eps")`). On the
/// 27B that number is `1e-6`.
fn read_rms_eps(ckpt: &Checkpoint) -> Result<f32> {
    let path = ckpt.directory.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|source| Error::Read {
        name: path.display().to_string(),
        source,
    })?;
    parse_rms_eps(&text)
}

fn parse_rms_eps(text: &str) -> Result<f32> {
    let key = "\"rms_norm_eps\"";
    let mut values = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find(key) {
        let after = rest[at + key.len()..].trim_start();
        let Some(after) = after.strip_prefix(':') else {
            return Err(Error::Weights("rms_norm_eps is not followed by ':'".into()));
        };
        let after = after.trim_start();
        let end = after
            .find(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-')))
            .unwrap_or(after.len());
        if end == 0 {
            return Err(Error::Weights("rms_norm_eps has no number".into()));
        }
        let number = &after[..end];
        let value: f32 = number
            .parse()
            .map_err(|_| Error::Weights(format!("rms_norm_eps `{number}` is not a float")))?;
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::Weights(format!(
                "rms_norm_eps `{number}` is not a positive finite float"
            )));
        }
        values.push(value);
        rest = &after[end..];
    }
    match values.as_slice() {
        [] => Err(Error::Weights("config.json has no rms_norm_eps".into())),
        [one] => Ok(*one),
        many => {
            let first = many[0].to_bits();
            if many.iter().any(|value| value.to_bits() != first) {
                Err(Error::Weights(
                    "config.json rms_norm_eps values disagree".into(),
                ))
            } else {
                Ok(many[0])
            }
        }
    }
}

fn load_embedding(gpu: &Gpu, ckpt: &Checkpoint, vocab: u32, hidden: u32) -> Result<ByteBuf> {
    let name = format!("{TEXT_PREFIX}.embed_tokens.weight");
    let info = tensor(ckpt, &name)?;
    if info.dtype != DType::Bf16 {
        return Err(Error::Weights(format!(
            "{name}: dtype is {}, expected BF16",
            info.dtype
        )));
    }
    if info.shape != [u64::from(vocab), u64::from(hidden)] {
        return Err(Error::Weights(format!(
            "{name}: shape {:?}, expected [{vocab}, {hidden}]",
            info.shape
        )));
    }
    device::upload_tensor(gpu, ckpt, &name)
}

fn load_norm(gpu: &Gpu, ckpt: &Checkpoint, name: &str, hidden: u32) -> Result<ByteBuf> {
    let info = tensor(ckpt, name)?;
    if info.dtype != DType::Bf16 {
        return Err(Error::Weights(format!(
            "{name}: dtype is {}, expected BF16",
            info.dtype
        )));
    }
    if info.shape != [u64::from(hidden)] {
        return Err(Error::Weights(format!(
            "{name}: shape {:?}, expected [{hidden}]",
            info.shape
        )));
    }
    device::upload_tensor(gpu, ckpt, name)
}

fn tensor<'a>(ckpt: &'a Checkpoint, name: &str) -> Result<&'a berth_exl3::TensorInfo> {
    ckpt.tensors
        .get(name)
        .ok_or_else(|| Error::Weights(format!("tensor {name} is missing")))
}

fn i32_width(value: u32, what: &str) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::Weights(format!("{what} {value} does not fit i32")))
}

fn checked_hidden(hidden: i32) -> Result<i32> {
    if hidden <= 0 || hidden % 4 != 0 {
        return Err(Error::Input(format!(
            "hidden {hidden} must be a positive multiple of 4"
        )));
    }
    Ok(hidden)
}

fn row_bytes(hidden: i32, elem: usize) -> Result<usize> {
    let width = usize::try_from(hidden).expect("hidden was checked positive");
    width
        .checked_mul(elem)
        .ok_or_else(|| Error::Input("row byte count overflows".into()))
}

fn row_span(token: u32, hidden: i32, elem: usize) -> Result<usize> {
    let rows = usize::try_from(token)
        .expect("token fits usize")
        .checked_add(1)
        .ok_or_else(|| Error::Input("token index overflows".into()))?;
    row_bytes(hidden, elem)?
        .checked_mul(rows)
        .ok_or_else(|| Error::Input("embedding span overflows".into()))
}

fn expect_row(buf: &ByteBuf, hidden: i32, elem: usize, what: &str) -> Result<()> {
    expect_bytes(buf, row_bytes(hidden, elem)?, what)
}

fn expect_bytes(buf: &ByteBuf, bytes: usize, what: &str) -> Result<()> {
    if buf.num_bytes() < bytes {
        return Err(Error::Input(format!(
            "{what} has {} bytes, needs {bytes}",
            buf.num_bytes()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_27b_rms_norm_eps() {
        // Bits recorded in testdata/kernels/ops/meta.json as epsilon_f32_bits.
        let eps = parse_rms_eps(r#"{ "text_config": { "rms_norm_eps": 1e-06 } }"#).unwrap();
        assert_eq!(eps.to_bits(), 897_988_541);
        assert_eq!(eps.to_bits(), 1e-6_f32.to_bits());
        let norm = TextNorm::qwen3_5(eps);
        assert_eq!(norm.constant_bias, 1.0);
        assert_eq!(norm.constant_scale, 1.0);
    }

    #[test]
    fn rejects_a_missing_or_inconsistent_eps() {
        let missing = parse_rms_eps("{}").unwrap_err();
        assert!(missing.to_string().contains("no rms_norm_eps"), "{missing}");
        let disagree =
            parse_rms_eps(r#"{"rms_norm_eps": 1e-06, "text_config": {"rms_norm_eps": 1e-05}}"#)
                .unwrap_err();
        assert!(disagree.to_string().contains("disagree"), "{disagree}");
    }

    #[test]
    fn identical_eps_entries_collapse() {
        let eps =
            parse_rms_eps(r#"{"rms_norm_eps": 1e-06, "text_config": {"rms_norm_eps": 1e-06}}"#)
                .unwrap();
        assert_eq!(eps.to_bits(), 1e-6_f32.to_bits());
    }
}

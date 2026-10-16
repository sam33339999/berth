//! MTP head: one input projection and one transformer block.
//!
//! Fork: `architecture/qwen3_5_mtp.py` and
//! `modules/arch_specific/qwen3_5_mtp.py`. The head borrows the target
//! embedding and `lm_head` from [`crate::model::Model`]. It does not store
//! a model.
//!
//! # Positions
//!
//! The target's full-attention layers are `bind_sequence`'d to the model's
//! shared length word. This head's attention layer is not.
//! [`MtpHead::draft_step`] calls [`AttnLayer::forward`], which writes
//! `position` into the layer's private `seqlens` and `positions` words.
//! That position is the draft index (`kv_position + idx` in the fork), not
//! the target's cache length. [`MtpHead::prefill_chunk`] calls
//! [`AttnLayer::prefill`], which stages the chunk start and `start + rows`
//! into those same private words. The kernels read `seqlens_addr()`, and
//! that address prefers a bound word, so binding this layer would make the
//! draft write at the target length. Both entry points error if the layer
//! is bound.

use std::ffi::c_void;
use std::path::Path;

use berth_cuda::{ByteBuf, DeviceBuffer};
use berth_exl3::{Bitrate, Checkpoint};
use berth_kernels::attn::PagedDecode;
use berth_kernels::{ops, sample, OutDtype};
use berth_plan::ModelShape;

use crate::attn::{AttnLayer, AttnPrefillScratch};
use crate::device::{find_linear, ptr, Exl3Linear, RowScratch};
use crate::mlp::MlpLayer;
use crate::model::{self, Model, TextNorm};
use crate::prefill::{self, MAX_CHUNK_SIZE};
use crate::{Error, Gpu, Result};

/// `mtp.fc`. 2H → H, fp32 out, K = 4 (`qwen3_5_mtp.py`).
const MTP_FC: &str = "mtp.fc";
/// RMSNorm on the target's post-norm export. fp16 in, fp16 out.
const MTP_PRE_FC_HIDDEN: &str = "mtp.pre_fc_norm_hidden.weight";
/// RMSNorm on the fp16 embedding gather.
const MTP_PRE_FC_EMBED: &str = "mtp.pre_fc_norm_embedding.weight";
/// Block `input_layernorm`.
const MTP_INPUT_NORM: &str = "mtp.layers.0.input_layernorm.weight";
/// Block `post_attention_layernorm`. Draft only; prefill stops before it.
const MTP_POST_NORM: &str = "mtp.layers.0.post_attention_layernorm.weight";
/// `mtp.norm`. fp32 in, fp16 out, same bias as the target final norm.
const MTP_NORM: &str = "mtp.norm.weight";
/// `mtp.layers.0.self_attn`.
const MTP_ATTN: &str = "mtp.layers.0.self_attn";
/// `mtp.layers.0.mlp`. Dense `GatedMLP` (`use_moe` is false on the 27B).
const MTP_MLP: &str = "mtp.layers.0.mlp";

/// `sample::fused` partials: 128 blocks times 3 fp32s (`FUSED_MAX_BLOCKS`).
const FUSED_WORKSPACE_BYTES: usize = 128 * 3 * 4;

/// Chunks of this many rows or fewer need the q_len attention path (P3).
const REFRESH_MAX_ROWS: u32 = 16;

/// Where one row of `cat(carry, export[:-1])` is copied from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftSource {
    /// Previous chunk's last export row. Zeros on the first chunk.
    Carry,
    /// `export[index]` of this chunk.
    Export(u32),
}

/// Source of shifted-hidden row `index`.
///
/// Row 0 is the carry. Row `i > 0` is `export[i - 1]`. The last export row
/// is not in the shifted buffer; it is [`next_carry_row`].
#[must_use]
pub fn shifted_row(index: u32) -> ShiftSource {
    match index.checked_sub(1) {
        None => ShiftSource::Carry,
        Some(prev) => ShiftSource::Export(prev),
    }
}

/// Export row kept as the next chunk's carry. `None` when `rows` is 0.
#[must_use]
pub fn next_carry_row(rows: u32) -> Option<u32> {
    rows.checked_sub(1)
}

/// One MTP layer: input projection, attention cache, and MLP.
///
/// The attention cache is a second cq3 allocation of `max_tokens` pages, the
/// fork's draft cache. Page count and page indices match a target layer
/// loaded with the same `max_tokens`. Decode width is set per step with
/// [`AttnLayer::set_decode_pages`] and is not the target's width.
pub struct MtpHead {
    hidden: i32,
    vocab: i32,
    max_tokens: u32,
    max_rows: u32,
    norm: TextNorm,
    pre_fc_norm_hidden: ByteBuf,
    pre_fc_norm_embedding: ByteBuf,
    input_norm: ByteBuf,
    post_norm: ByteBuf,
    final_norm: ByteBuf,
    fc: Exl3Linear,
    attn: AttnLayer,
    mlp: MlpLayer,
    /// Decode split/combine for [`MtpHead::draft_step`]. Not the target's.
    paged: PagedDecode,
    attn_scratch: AttnPrefillScratch,
    /// Shared by `mtp.fc` and the attention linears on a prefill chunk.
    ///
    /// `max_k` covers `mtp.fc` (2H) and `o_proj` (query heads × head dim).
    /// `max_n` covers the gated `q_proj`. MLP down is K = 17408 on the 27B.
    /// Prefill stops before the MLP, and draft is one row
    /// ([`MlpLayer::forward`]), so that K is not in this scratch.
    row_scratch: RowScratch,
    /// fp16 `[max_rows, H]`. `pre_fc_norm_hidden` output.
    y: ByteBuf,
    /// fp16 `[max_rows, H]`. Embedding gather, before its norm.
    embedded: ByteBuf,
    /// fp16 `[max_rows, H]`. `pre_fc_norm_embedding` output.
    x_norm: ByteBuf,
    /// fp16 `[max_rows, 2H]`. Embedding in the first half, hidden norm in
    /// the second. `torch.cat((x, y), dim=-1)`.
    cat: ByteBuf,
    /// fp32 `[max_rows, H]`. `mtp.fc` output, then the block residual.
    residual: ByteBuf,
    /// fp16 `[max_rows, H]`. `input_layernorm` output.
    block_norm: ByteBuf,
    /// fp32 `[max_rows, H]`. Attention output.
    mixer_out: ByteBuf,
    /// fp16 `[H]`. Draft `post_attention_layernorm` output.
    mlp_in: ByteBuf,
    /// fp32 `[H]`. Draft MLP output.
    mlp_out: ByteBuf,
    /// fp16 `[H]`. `mtp.norm` of the last [`MtpHead::draft_step`].
    norm_out: ByteBuf,
    /// fp16 `[vocab]`. Target `lm_head` of that norm row.
    logits: ByteBuf,
    /// One `uint64` argmax index.
    token_out: DeviceBuffer<u64>,
    /// 1536-byte workspace for [`sample::fused`].
    sample_ws: ByteBuf,
    /// `max_rows` i32 token ids. A call uses the prefix.
    ids: ByteBuf,
}

impl MtpHead {
    /// Load `mtp.*` for one draft layer and allocate its cache and scratch.
    ///
    /// `max_tokens` is a positive multiple of the 256-token page, the same
    /// rule as [`AttnLayer::load`]. `cubin_dir` is `target/triton-cubins`.
    /// Rope, the q/k norms, and the interleaved gate come from `shape`,
    /// which is the target config (`qwen3_5_mtp.py` builds the attention
    /// with `Qwen3_5Config`).
    ///
    /// # Errors
    ///
    /// A missing tensor, a bitrate other than K = 4, a bias, a width that
    /// is not 2H → H for `mtp.fc` or the config intermediate for the MLP,
    /// or a `max_tokens` the attention cache rejects.
    pub fn load(
        gpu: &Gpu,
        ckpt: &Checkpoint,
        shape: &ModelShape,
        max_tokens: u32,
        cubin_dir: &Path,
    ) -> Result<Self> {
        let hidden = checked_hidden(shape)?;
        let vocab = checked_vocab(shape)?;
        if max_tokens == 0 || !max_tokens.is_multiple_of(prefill::PAGE_SIZE) {
            return Err(Error::Input(format!(
                "max_tokens {max_tokens} is not a positive multiple of {}",
                prefill::PAGE_SIZE
            )));
        }
        let intermediate = ckpt
            .config
            .intermediate_size
            .ok_or_else(|| Error::Weights("config has no intermediate_size".to_string()))?;
        require_k4(ckpt, MTP_FC)?;
        require_k4(ckpt, &format!("{MTP_ATTN}.q_proj"))?;
        let norm = TextNorm::qwen3_5(model::read_rms_eps(ckpt)?);
        let weights = load_weights(gpu, ckpt, shape, hidden, max_tokens, intermediate)?;
        let (max_k, max_n) = scratch_limits(shape, hidden)?;
        let scratch = alloc_scratch(
            gpu, shape, hidden, vocab, max_k, max_n, max_tokens, cubin_dir,
        )?;
        Ok(Self {
            hidden,
            vocab,
            max_tokens,
            max_rows: MAX_CHUNK_SIZE,
            norm,
            pre_fc_norm_hidden: weights.pre_fc_norm_hidden,
            pre_fc_norm_embedding: weights.pre_fc_norm_embedding,
            input_norm: weights.input_norm,
            post_norm: weights.post_norm,
            final_norm: weights.final_norm,
            fc: weights.fc,
            attn: weights.attn,
            mlp: weights.mlp,
            paged: scratch.paged,
            attn_scratch: scratch.attn_scratch,
            row_scratch: scratch.row_scratch,
            y: scratch.y,
            embedded: scratch.embedded,
            x_norm: scratch.x_norm,
            cat: scratch.cat,
            residual: scratch.residual,
            block_norm: scratch.block_norm,
            mixer_out: scratch.mixer_out,
            mlp_in: scratch.mlp_in,
            mlp_out: scratch.mlp_out,
            norm_out: scratch.norm_out,
            logits: scratch.logits,
            token_out: scratch.token_out,
            sample_ws: scratch.sample_ws,
            ids: scratch.ids,
        })
    }

    /// Input layer plus attention for one prefill chunk.
    ///
    /// Writes the MTP cache at `[start, start + rows)`. Stops before the
    /// residual add, the MLP, and `mtp.norm`
    /// (`modules/transformer.py` returns the incoming residual when
    /// `params["prefill"]` is set; the cache write is inside attention).
    /// `hidden` is fp16 `[rows, H]`, the shifted target export.
    /// `block_pages` is this job's block-table width.
    ///
    /// `rows > 16` uses [`AttnLayer::prefill`]. `rows <= 16` is the refresh
    /// path and returns before any launch.
    ///
    /// # Errors
    ///
    /// The layer is bound, the chunk is empty, `rows <= 16`, a token is
    /// past the vocabulary, or the chunk does not fit the cache.
    pub fn prefill_chunk(
        &mut self,
        gpu: &Gpu,
        model: &Model,
        ids: &[u32],
        hidden: &ByteBuf,
        start: u32,
        block_pages: u32,
    ) -> Result<()> {
        self.check_unbound()?;
        let rows = chunk_rows(ids)?;
        if rows <= REFRESH_MAX_ROWS {
            return Err(Error::Input(
                "mtp refresh of <= 16 rows needs the q_len path (P3)".into(),
            ));
        }
        self.check_model(model)?;
        self.check_tokens(ids)?;
        self.check_span(hidden, start, rows)?;
        self.attn.set_prefill_pages(block_pages)?;
        self.stage_ids(gpu, ids)?;
        self.project_input(gpu, model.embedding(), hidden, rows)?;
        let width = self.hidden;
        let norm = self.norm;
        Model::norm_rows(
            gpu,
            &self.residual,
            &self.input_norm,
            &mut self.block_norm,
            width,
            rows,
            norm,
        )?;
        let MtpHead {
            attn,
            attn_scratch,
            row_scratch,
            block_norm,
            mixer_out,
            ..
        } = self;
        attn.prefill(
            gpu,
            row_scratch,
            attn_scratch,
            block_norm,
            mixer_out,
            start,
            rows,
        )?;
        // The pinned seqlens and positions slots are single-buffered.
        gpu.stream.synchronize()?;
        Ok(())
    }

    /// One draft token. Returns the argmax id.
    ///
    /// Full one-row block: input layer, attention at `position` on this
    /// head's cache, residual, MLP, residual, `mtp.norm`. Then the target
    /// `lm_head` on that fp16 row and a first-max argmax.
    /// `decode_width_pages` is the block-table width from `docs/mtp.md`
    /// section 3; this stores it with [`AttnLayer::set_decode_pages`] and
    /// does not change the target's width. [`MtpHead::draft_norm`] is the
    /// `mtp.norm` row, the next step's hidden.
    ///
    /// # Errors
    ///
    /// The layer is bound, `id` is past the vocabulary, `position` is past
    /// the cache, or `decode_width_pages` does not fit this cache.
    pub fn draft_step(
        &mut self,
        gpu: &Gpu,
        model: &mut Model,
        id: u32,
        hidden: &ByteBuf,
        position: u32,
        decode_width_pages: u32,
    ) -> Result<u32> {
        self.check_unbound()?;
        self.check_model(model)?;
        self.check_tokens(&[id])?;
        self.check_span(hidden, position, 1)?;
        self.attn.set_decode_pages(decode_width_pages)?;
        self.stage_ids(gpu, &[id])?;
        self.project_input(gpu, model.embedding(), hidden, 1)?;
        self.draft_block(gpu, position)?;
        self.project_and_argmax(gpu, model)
    }

    /// fp16 `[H]` `mtp.norm` output of the last [`Self::draft_step`].
    ///
    /// The next draft step reads this row as its target hidden. The buffer
    /// is ready when `draft_step` returns: the argmax copy synchronizes
    /// `gpu.stream`.
    pub fn draft_norm(&self) -> &ByteBuf {
        &self.norm_out
    }

    fn check_unbound(&self) -> Result<()> {
        if self.attn.sequence_is_bound() {
            return Err(Error::Input(
                "mtp attention is bound to the target sequence word".into(),
            ));
        }
        Ok(())
    }

    fn check_model(&self, model: &Model) -> Result<()> {
        if model.hidden_size() != self.hidden || model.vocab_size() != self.vocab {
            return Err(Error::Input(format!(
                "mtp head is {}x{}, model is {}x{}",
                self.hidden,
                self.vocab,
                model.hidden_size(),
                model.vocab_size()
            )));
        }
        Ok(())
    }

    fn check_tokens(&self, ids: &[u32]) -> Result<()> {
        let vocab = u32::try_from(self.vocab).expect("vocab is a positive i32");
        for &token in ids {
            if token >= vocab {
                return Err(Error::Input(format!(
                    "token {token} is past the vocabulary ({vocab})"
                )));
            }
        }
        Ok(())
    }

    fn check_span(&self, hidden: &ByteBuf, start: u32, rows: u32) -> Result<()> {
        if rows > self.max_rows {
            return Err(Error::Input(format!(
                "mtp chunk of {rows} rows exceeds the {}-row scratch",
                self.max_rows
            )));
        }
        let end = start
            .checked_add(rows)
            .ok_or_else(|| Error::Input("mtp chunk end overflows".into()))?;
        if end > self.max_tokens {
            return Err(Error::Input(format!(
                "mtp chunk [{start}, {end}) does not fit the {}-token cache",
                self.max_tokens
            )));
        }
        let need = matrix_bytes(rows, self.hidden, 2)?;
        if hidden.num_bytes() < need {
            return Err(Error::Input(format!(
                "mtp hidden has {} bytes, needs {need}",
                hidden.num_bytes()
            )));
        }
        Ok(())
    }

    fn stage_ids(&mut self, gpu: &Gpu, ids: &[u32]) -> Result<()> {
        let bytes = self.ids.num_bytes();
        if ids.len().saturating_mul(4) > bytes {
            return Err(Error::Input(format!(
                "mtp chunk of {} tokens exceeds the id buffer",
                ids.len()
            )));
        }
        let mut host = vec![0u8; bytes];
        for (index, token) in ids.iter().enumerate() {
            let id = i32::try_from(*token)
                .map_err(|_| Error::Input(format!("token {token} does not fit in i32")))?;
            host[index * 4..index * 4 + 4].copy_from_slice(&id.to_le_bytes());
        }
        self.ids.copy_from_host(&gpu.stream, &host)?;
        Ok(())
    }

    /// `Qwen3_5MTPInputLayer.forward` for `rows` rows. fp32 `[rows, H]` lands
    /// in `residual`.
    fn project_input(
        &mut self,
        gpu: &Gpu,
        embed: &ByteBuf,
        hidden_in: &ByteBuf,
        rows: u32,
    ) -> Result<()> {
        let rows_i =
            i32::try_from(rows).map_err(|_| Error::Input("mtp rows do not fit in i32".into()))?;
        let width = self.hidden;
        let norm = self.norm;
        expect_embed(embed, self.vocab, width)?;
        let stream = gpu.raw_stream();
        let MtpHead {
            pre_fc_norm_hidden,
            pre_fc_norm_embedding,
            y,
            embedded,
            x_norm,
            cat,
            fc,
            residual,
            ids,
            row_scratch,
            ..
        } = self;
        // SAFETY: `hidden_in` is fp16 `[rows, H]`, the norm weights are one
        // bf16 row, `embed` is the bf16 table, and `ids` holds `rows` i32s
        // already checked against the vocabulary. `y`, `embedded`, `x_norm`,
        // and `cat` are this head's buffers and do not alias. They stay
        // allocated until `gpu.stream` finishes.
        unsafe {
            ops::rms_norm_f16(
                ptr(hidden_in),
                ptr(pre_fc_norm_hidden),
                ptr(y),
                rows_i,
                width,
                norm.epsilon,
                norm.constant_bias,
                norm.constant_scale,
                stream,
            )?;
            ops::embed_tokens_f16(
                ptr(embed),
                ptr(ids).cast::<i32>(),
                ptr(embedded),
                width,
                rows_i,
                stream,
            )?;
            ops::rms_norm_f16(
                ptr(embedded),
                ptr(pre_fc_norm_embedding),
                ptr(x_norm),
                rows_i,
                width,
                norm.epsilon,
                norm.constant_bias,
                norm.constant_scale,
                stream,
            )?;
            // Embedding first, then the hidden norm. `torch.cat((x, y), -1)`.
            ops::cat_f16_last(ptr(x_norm), ptr(y), ptr(cat), rows_i, width, stream)?;
        }
        if rows == 1 {
            fc.forward(gpu, cat, residual, OutDtype::F32)?;
        } else {
            fc.forward_rows(gpu, row_scratch, cat, residual, rows, OutDtype::F32)?;
        }
        Ok(())
    }

    /// One-row block after the input layer. Leaves `mtp.norm` in `norm_out`.
    fn draft_block(&mut self, gpu: &Gpu, position: u32) -> Result<()> {
        let width = self.hidden;
        let norm = self.norm;
        Model::norm_rows(
            gpu,
            &self.residual,
            &self.input_norm,
            &mut self.block_norm,
            width,
            1,
            norm,
        )?;
        let MtpHead {
            attn,
            paged,
            block_norm,
            mixer_out,
            ..
        } = &mut *self;
        attn.forward(gpu, paged, block_norm, mixer_out, position)?;
        Model::norm_residual(
            gpu,
            &self.mixer_out,
            &self.post_norm,
            &mut self.mlp_in,
            &mut self.residual,
            width,
            norm,
        )?;
        let MtpHead {
            mlp,
            mlp_in,
            mlp_out,
            ..
        } = &mut *self;
        mlp.forward(gpu, mlp_in, mlp_out)?;
        Model::add_residual_in_place(gpu, &mut self.residual, &self.mlp_out, width)?;
        Model::norm_rows(
            gpu,
            &self.residual,
            &self.final_norm,
            &mut self.norm_out,
            width,
            1,
            norm,
        )
    }

    /// Target `lm_head` on `norm_out`, then first-max argmax of the fp16 logits.
    ///
    /// Fork: `torch.argmax` (`qwen3_5_mtp.py` `sample_from_state`). Ties keep
    /// the lowest index. [`sample::fused`] mode 0 is the same rule
    /// (`fs_amax_first` in `generator/sampling_fused.cu`): fp16 widens to
    /// fp32 exactly, so equal fp16 values stay tied.
    fn project_and_argmax(&mut self, gpu: &Gpu, model: &mut Model) -> Result<u32> {
        let vocab = self.vocab;
        let norm_out = &self.norm_out;
        let logits = &mut self.logits;
        let head = model.lm_head_mut();
        if head.size_n != vocab {
            return Err(Error::Input(format!(
                "lm_head N is {}, MTP argmax width is {vocab}",
                head.size_n
            )));
        }
        Model::project_lm_head(gpu, head, norm_out, logits)?;
        let stream = gpu.raw_stream();
        let logits_ptr = ptr(&self.logits);
        let out_ptr = self.token_out.ptr().as_u64() as *mut c_void;
        let workspace = ptr(&self.sample_ws);
        // SAFETY: `logits` is fp16 `[vocab]`, `token_out` is one uint64, and
        // `sample_ws` is 1536 bytes from a CUDA allocation. All three stay
        // allocated until `gpu.stream` finishes.
        unsafe {
            sample::fused(
                logits_ptr, out_ptr, workspace, vocab, vocab, true, 1.0, 0, 0, stream,
            )?;
        }
        let mut word = [0u64; 1];
        self.token_out.copy_to_host(&gpu.stream, &mut word)?;
        u32::try_from(word[0])
            .map_err(|_| Error::Input(format!("argmax {} does not fit in u32", word[0])))
    }
}

struct MtpWeights {
    pre_fc_norm_hidden: ByteBuf,
    pre_fc_norm_embedding: ByteBuf,
    input_norm: ByteBuf,
    post_norm: ByteBuf,
    final_norm: ByteBuf,
    fc: Exl3Linear,
    attn: AttnLayer,
    mlp: MlpLayer,
}

fn load_weights(
    gpu: &Gpu,
    ckpt: &Checkpoint,
    shape: &ModelShape,
    hidden: i32,
    max_tokens: u32,
    intermediate: u32,
) -> Result<MtpWeights> {
    let hidden_u = u32::try_from(hidden).expect("hidden is positive");
    let pre_fc_norm_hidden = model::load_norm(gpu, ckpt, MTP_PRE_FC_HIDDEN, hidden_u)?;
    let pre_fc_norm_embedding = model::load_norm(gpu, ckpt, MTP_PRE_FC_EMBED, hidden_u)?;
    let input_norm = model::load_norm(gpu, ckpt, MTP_INPUT_NORM, hidden_u)?;
    let post_norm = model::load_norm(gpu, ckpt, MTP_POST_NORM, hidden_u)?;
    let final_norm = model::load_norm(gpu, ckpt, MTP_NORM, hidden_u)?;
    let fc = Exl3Linear::load(gpu, ckpt, MTP_FC)?;
    let expect_k = hidden
        .checked_mul(2)
        .ok_or_else(|| Error::Weights("mtp.fc K overflows".into()))?;
    if fc.size_k != expect_k || fc.size_n != hidden {
        return Err(Error::Weights(format!(
            "{MTP_FC} is {}x{}, expected {expect_k}x{hidden}",
            fc.size_k, fc.size_n
        )));
    }
    let attn = AttnLayer::load_prefixed(gpu, ckpt, shape, MTP_ATTN, max_tokens)?;
    let mlp = MlpLayer::load_prefixed(gpu, ckpt, MTP_MLP)?;
    let got = u32::try_from(mlp.intermediate()).unwrap_or(u32::MAX);
    if mlp.hidden_size() != hidden || got != intermediate {
        return Err(Error::Weights(format!(
            "{MTP_MLP} is {}x{got}, expected {hidden}x{intermediate}",
            mlp.hidden_size()
        )));
    }
    Ok(MtpWeights {
        pre_fc_norm_hidden,
        pre_fc_norm_embedding,
        input_norm,
        post_norm,
        final_norm,
        fc,
        attn,
        mlp,
    })
}

struct MtpScratch {
    paged: PagedDecode,
    attn_scratch: AttnPrefillScratch,
    row_scratch: RowScratch,
    y: ByteBuf,
    embedded: ByteBuf,
    x_norm: ByteBuf,
    cat: ByteBuf,
    residual: ByteBuf,
    block_norm: ByteBuf,
    mixer_out: ByteBuf,
    mlp_in: ByteBuf,
    mlp_out: ByteBuf,
    norm_out: ByteBuf,
    logits: ByteBuf,
    token_out: DeviceBuffer<u64>,
    sample_ws: ByteBuf,
    ids: ByteBuf,
}

#[allow(clippy::too_many_arguments)]
fn alloc_scratch(
    gpu: &Gpu,
    shape: &ModelShape,
    hidden: i32,
    vocab: i32,
    max_k: u32,
    max_n: u32,
    max_tokens: u32,
    cubin_dir: &Path,
) -> Result<MtpScratch> {
    let paged = PagedDecode::load(&gpu.ctx, cubin_dir)?;
    let attn_scratch = AttnPrefillScratch::new(gpu, shape, MAX_CHUNK_SIZE, max_tokens, cubin_dir)?;
    let row_scratch = RowScratch::new(gpu, MAX_CHUNK_SIZE, max_k, max_n)?;
    let rows = MAX_CHUNK_SIZE;
    let f16 = matrix_bytes(rows, hidden, 2)?;
    let f32 = matrix_bytes(rows, hidden, 4)?;
    let cat = matrix_bytes(
        rows,
        hidden
            .checked_mul(2)
            .ok_or_else(|| Error::Input("mtp concat width overflows".into()))?,
        2,
    )?;
    let one_f16 = matrix_bytes(1, hidden, 2)?;
    let one_f32 = matrix_bytes(1, hidden, 4)?;
    let vocab_f16 = matrix_bytes(1, vocab, 2)?;
    let id_bytes = matrix_bytes(rows, 1, 4)?;
    Ok(MtpScratch {
        paged,
        attn_scratch,
        row_scratch,
        y: gpu.zeroed(f16)?,
        embedded: gpu.zeroed(f16)?,
        x_norm: gpu.zeroed(f16)?,
        cat: gpu.zeroed(cat)?,
        residual: gpu.zeroed(f32)?,
        block_norm: gpu.zeroed(f16)?,
        mixer_out: gpu.zeroed(f32)?,
        mlp_in: gpu.zeroed(one_f16)?,
        mlp_out: gpu.zeroed(one_f32)?,
        norm_out: gpu.zeroed(one_f16)?,
        logits: gpu.zeroed(vocab_f16)?,
        token_out: DeviceBuffer::zeroed(&gpu.stream, 1)?,
        sample_ws: gpu.zeroed(FUSED_WORKSPACE_BYTES)?,
        ids: gpu.zeroed(id_bytes)?,
    })
}

fn scratch_limits(shape: &ModelShape, hidden: i32) -> Result<(u32, u32)> {
    let q = shape
        .num_q_heads
        .checked_mul(shape.head_dim)
        .ok_or_else(|| Error::Weights("query width overflows".into()))?;
    let q_gate = q
        .checked_mul(2)
        .ok_or_else(|| Error::Weights("gated query width overflows".into()))?;
    let hidden_u = u32::try_from(hidden).expect("hidden is positive");
    let doubled = hidden_u
        .checked_mul(2)
        .ok_or_else(|| Error::Weights("mtp.fc K overflows".into()))?;
    // fc K is 2H. o_proj K is the ungated query width. q_proj N is the gated
    // width. See the `row_scratch` field comment for the MLP.
    Ok((doubled.max(q), hidden_u.max(q_gate)))
}

fn require_k4(ckpt: &Checkpoint, prefix: &str) -> Result<()> {
    let linear = find_linear(ckpt, prefix)?;
    if linear.bias.is_some() {
        return Err(Error::Weights(format!(
            "{prefix}: bias is present and is not applied"
        )));
    }
    match linear.k {
        Bitrate::Int(4) => {}
        other => {
            return Err(Error::Weights(format!(
                "{prefix}: rate {other} is not K = 4"
            )))
        }
    }
    if !matches!(linear.codebook, berth_exl3::Codebook::Mul1) {
        return Err(Error::Weights(format!("{prefix}: codebook is not mul1")));
    }
    Ok(())
}

fn checked_hidden(shape: &ModelShape) -> Result<i32> {
    let hidden = i32::try_from(shape.hidden_size)
        .map_err(|_| Error::Weights("hidden_size does not fit in i32".into()))?;
    if hidden <= 0 || hidden % 4 != 0 {
        return Err(Error::Weights(format!(
            "hidden {hidden} must be a positive multiple of 4"
        )));
    }
    Ok(hidden)
}

fn checked_vocab(shape: &ModelShape) -> Result<i32> {
    let vocab = i32::try_from(shape.vocab_size)
        .map_err(|_| Error::Weights("vocab_size does not fit in i32".into()))?;
    if vocab < 1 {
        return Err(Error::Weights("vocab_size is 0".into()));
    }
    Ok(vocab)
}

fn expect_embed(embed: &ByteBuf, vocab: i32, hidden: i32) -> Result<()> {
    let need = matrix_bytes(u32::try_from(vocab).unwrap_or(u32::MAX), hidden, 2)?;
    if embed.num_bytes() < need {
        return Err(Error::Input(format!(
            "embedding table has {} bytes, needs {need}",
            embed.num_bytes()
        )));
    }
    Ok(())
}

fn matrix_bytes(rows: u32, cols: i32, elem: usize) -> Result<usize> {
    if cols < 1 {
        return Err(Error::Input(format!("mtp width {cols} must be positive")));
    }
    let rows = usize::try_from(rows).unwrap_or(usize::MAX);
    let cols = usize::try_from(cols).expect("cols is positive");
    rows.checked_mul(cols)
        .and_then(|n| n.checked_mul(elem))
        .ok_or_else(|| Error::Input("mtp buffer size overflows".into()))
}

fn chunk_rows(ids: &[u32]) -> Result<u32> {
    if ids.is_empty() {
        return Err(Error::Input("mtp prefill chunk is empty".into()));
    }
    u32::try_from(ids.len()).map_err(|_| Error::Input("mtp chunk is longer than u32::MAX".into()))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::Path;

    use berth_exl3::{Bitrate, Checkpoint, DType};

    use super::*;

    struct Spec {
        name: String,
        dtype: DType,
        shape: Vec<u64>,
    }

    fn push_norm(out: &mut Vec<Spec>, name: &str, dim: u64) {
        out.push(Spec {
            name: name.to_string(),
            dtype: DType::Bf16,
            shape: vec![dim],
        });
    }

    fn push_linear(out: &mut Vec<Spec>, prefix: &str, k: u64, n: u64) {
        assert!(k.is_multiple_of(16) && n.is_multiple_of(16), "{prefix}");
        out.push(Spec {
            name: format!("{prefix}.trellis"),
            dtype: DType::I16,
            shape: vec![k / 16, n / 16, 64],
        });
        out.push(Spec {
            name: format!("{prefix}.suh"),
            dtype: DType::F16,
            shape: vec![k],
        });
        out.push(Spec {
            name: format!("{prefix}.svh"),
            dtype: DType::F16,
            shape: vec![n],
        });
        out.push(Spec {
            name: format!("{prefix}.mul1"),
            dtype: DType::I32,
            shape: vec![],
        });
    }

    /// Every `mtp.*` tensor [`MtpHead::load`] reads on the 27B.
    fn mtp_27b_spec() -> Vec<Spec> {
        let hidden = 5120u64;
        let intermediate = 17408u64;
        let q = 24u64 * 256;
        let kv = 4u64 * 256;
        let mut out = Vec::new();
        push_norm(&mut out, MTP_PRE_FC_HIDDEN, hidden);
        push_norm(&mut out, MTP_PRE_FC_EMBED, hidden);
        push_norm(&mut out, MTP_INPUT_NORM, hidden);
        push_norm(&mut out, MTP_POST_NORM, hidden);
        push_norm(&mut out, &format!("{MTP_ATTN}.q_norm.weight"), 256);
        push_norm(&mut out, &format!("{MTP_ATTN}.k_norm.weight"), 256);
        push_norm(&mut out, MTP_NORM, hidden);
        push_linear(&mut out, MTP_FC, hidden * 2, hidden);
        push_linear(&mut out, &format!("{MTP_ATTN}.q_proj"), hidden, q * 2);
        push_linear(&mut out, &format!("{MTP_ATTN}.k_proj"), hidden, kv);
        push_linear(&mut out, &format!("{MTP_ATTN}.v_proj"), hidden, kv);
        push_linear(&mut out, &format!("{MTP_ATTN}.o_proj"), q, hidden);
        push_linear(
            &mut out,
            &format!("{MTP_MLP}.gate_proj"),
            hidden,
            intermediate,
        );
        push_linear(
            &mut out,
            &format!("{MTP_MLP}.up_proj"),
            hidden,
            intermediate,
        );
        push_linear(
            &mut out,
            &format!("{MTP_MLP}.down_proj"),
            intermediate,
            hidden,
        );
        out
    }

    fn linear_prefixes() -> Vec<String> {
        let mut out = vec![MTP_FC.to_string()];
        for part in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            out.push(format!("{MTP_ATTN}.{part}"));
        }
        for part in ["gate_proj", "up_proj", "down_proj"] {
            out.push(format!("{MTP_MLP}.{part}"));
        }
        out
    }

    #[test]
    fn mtp_tensor_list_has_39_names() {
        let spec = mtp_27b_spec();
        assert_eq!(spec.len(), 39);
        let names: BTreeSet<&str> = spec.iter().map(|item| item.name.as_str()).collect();
        assert_eq!(names.len(), 39);
        assert!(names.iter().all(|name| name.starts_with("mtp.")));
        assert!(names.contains(MTP_PRE_FC_HIDDEN));
        assert!(names.contains(MTP_NORM));
        assert!(names.contains("mtp.fc.trellis"));
        assert!(names.contains("mtp.layers.0.self_attn.q_norm.weight"));
        assert!(names.contains("mtp.layers.0.mlp.down_proj.trellis"));
    }

    #[test]
    fn mtp_27b_tensors_match_the_checkpoint() {
        let dir = std::env::var("BERTH_27B_DIR")
            .unwrap_or_else(|_| "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3".to_string());
        if !Path::new(&dir).is_dir() {
            eprintln!("skipping mtp 27b tensors: directory missing: {dir}");
            return;
        }
        let ckpt = Checkpoint::open(&dir).expect("27b checkpoint headers");
        let spec = mtp_27b_spec();
        let expected: BTreeSet<String> = spec.iter().map(|item| item.name.clone()).collect();
        let mut found = BTreeSet::new();
        for (name, info) in ckpt.tensors.iter() {
            if !name.starts_with("mtp.") {
                continue;
            }
            found.insert(name.to_string());
            let item = spec
                .iter()
                .find(|item| item.name == name)
                .unwrap_or_else(|| panic!("{name} is not a tensor MtpHead::load reads"));
            assert_eq!(info.dtype, item.dtype, "{name}");
            assert_eq!(info.shape, item.shape, "{name}");
        }
        assert_eq!(found, expected);
        for prefix in linear_prefixes() {
            let linear = ckpt
                .linears()
                .iter()
                .find(|linear| linear.prefix == prefix)
                .unwrap_or_else(|| panic!("{prefix} is not an EXL3 linear"));
            assert_eq!(linear.k, Bitrate::Int(4), "{prefix}");
            assert!(linear.bias.is_none(), "{prefix}");
        }
    }

    #[test]
    fn shifted_hidden_rows() {
        assert_eq!(shifted_row(0), ShiftSource::Carry);
        assert_eq!(shifted_row(1), ShiftSource::Export(0));
        assert_eq!(shifted_row(2), ShiftSource::Export(1));
        assert_eq!(shifted_row(3), ShiftSource::Export(2));
        assert_eq!(next_carry_row(4), Some(3));
        // One-row chunk: the only shifted row is the old carry, and the
        // next carry is export row 0.
        assert_eq!(shifted_row(0), ShiftSource::Carry);
        assert_eq!(next_carry_row(1), Some(0));
        assert_eq!(next_carry_row(0), None);
    }

    #[test]
    fn chunk_bounds_ignore_draft_width() {
        // `N` does not change the cuts. 100 tokens is one chunk of the
        // prefix; 1500 is the page cut at 1280, then the tail.
        assert_eq!(prefill::chunk_bounds(100), vec![(0, 99)]);
        assert_eq!(prefill::chunk_bounds(1500), vec![(0, 1280), (1280, 1499)]);
    }

    #[test]
    fn prefill_pages_8190_grows_by_draft_tokens() {
        assert_eq!(prefill::prefill_block_pages(8190, 1), 32);
        assert_eq!(prefill::prefill_block_pages_mtp(8190, 1, 2), 33);
    }
}

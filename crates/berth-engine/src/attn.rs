//! Full attention mixer, one decode token (16 of the 27B's 64 layers).
//!
//! Owner: task ATTN. Reproduces `BC_Attention::run` → `run_gr` for bsz 1,
//! one token, cq3 K and V, interleaved swish gate (`gate_mode == 3`):
//!
//! 1. sliced `exl3_mgemm` of `q_proj`, `k_proj`, `v_proj` (fp16 out)
//! 2. `deinterleave_qg`
//! 3. fused q/k RMSNorm + NEOX RoPE, in place
//! 4. `quant_cache_paged` into this layer's own pages
//! 5. Triton split/combine ([`PagedDecode`])
//! 6. `mul_sigmoid` gate
//! 7. `o_proj` via [`Exl3Linear`], fp32 out
//!
//! `Exl3Linear` keeps its trellis pointers private, so the q/k/v bundle is
//! loaded here. `o_proj` goes through `Exl3Linear`. Rope theta, the partial
//! rotary factor, and `rms_norm_eps` are not on [`berth_plan::ModelShape`];
//! they are read from `config.json` with the fork's key order.

use std::path::Path;

use berth_cuda::{ByteBuf, DeviceBuffer, PinnedBuf};
use berth_exl3::{Bitrate, Checkpoint, DType, HadamardScale};
use berth_kernels::attn::{
    deinterleave_qg, mul_sigmoid, quant_cache_paged, rope, splits_cap, PagedDecode,
    PagedDecodeArgs, QuantCacheLaunch, RopeLaunch,
};
use berth_kernels::attn_prefill::{
    needs_staging, staging_pages, PagedPrefill, PrefillChunk, NORM_EPS, ROTARY_DIM, TWO_PASS_MIN_Q,
};
use berth_kernels::ops::exl3_mgemm;
use berth_kernels::{Codebook, OutDtype};
use berth_plan::{LayerKind, ModelShape};

use crate::device::{find_linear, ptr, read_tensor, upload_tensor};
use crate::{Error, Exl3Linear, Gpu, Result};

/// Cubin tile. `PagedDecode::load` rejects anything else.
const Q_HEADS: i32 = 24;
const KV_HEADS: i32 = 4;
const HEAD_DIM: i32 = 256;
const PAGE: u32 = 256;
const CACHE_BITS: u32 = 3;
const BLOCK_H: i32 = 16;
const BLOCK_ROWS: i32 = 16;
const HD_PAD: i32 = 256;
const ROPE_NEOX: i32 = 2;
/// Qwen3.5 passes this when `text_config` has no `rope_theta`.
const DEFAULT_ROPE_THETA: f64 = 10_000_000.0;
/// `project_qkv` uses the sliced bundle only while `bsz * q_len <= 32`.
const SLICED_MAX_ROWS: u32 = 32;

const _: () = {
    assert!(Q_HEADS == berth_kernels::attn_prefill::Q_HEADS);
    assert!(KV_HEADS == berth_kernels::attn_prefill::KV_HEADS);
    assert!(HEAD_DIM == berth_kernels::attn_prefill::HEAD_DIM);
    assert!(PAGE as i32 == berth_kernels::attn_prefill::PAGE_SIZE);
};

/// One attention layer: its weights, its own paged cq3 K/V cache for a single
/// sequence of up to `max_tokens`, and scratch.
pub struct AttnLayer {
    hidden: i32,
    head_dim: i32,
    q_heads: i32,
    kv_heads: i32,
    /// fp16 elements in `q` (and in the gate, and in the attention output).
    q_numel: i64,
    /// `num_kv_heads * head_dim`.
    token_dim: i32,
    width: i32,
    slices: i32,
    size_k: i32,
    k_bits: i32,
    codebook: Codebook,
    num_pages: i32,
    /// Block-table width the decode attention is launched with. The fork
    /// sizes it per request, not per cache: `ceil(max_seq_len / 256)` pages
    /// rounded up to a multiple of 16 (`generator.py` `iterate_gen`), and the
    /// split count follows it. [`Self::set_decode_width`] sets it; it starts
    /// as the whole cache.
    decode_width: i32,
    max_tokens: u32,
    programs: i32,
    splits_cap: i32,
    partial_head_dim: i32,
    norm_eps: f32,
    /// q, k, v. The device slice lists alias the weight buffers.
    qkv: [Proj; 3],
    b_list: ByteBuf,
    suh_list: ByteBuf,
    svh_list: ByteBuf,
    c_ptrs: ByteBuf,
    size_n_list: ByteBuf,
    n_stride_list: ByteBuf,
    had_src_list: ByteBuf,
    a_had: ByteBuf,
    carrier: ByteBuf,
    q: ByteBuf,
    g: ByteBuf,
    attn_out: ByteBuf,
    q_norm: ByteBuf,
    k_norm: ByteBuf,
    inv_freq: ByteBuf,
    h32: ByteBuf,
    k_pages: ByteBuf,
    v_pages: ByteBuf,
    k_scales: ByteBuf,
    v_scales: ByteBuf,
    block_table: ByteBuf,
    /// int32 length before this token, for the standalone [`Self::forward`].
    ///
    /// [`Self::bind_sequence`] points the kernels at the model's shared word
    /// instead. The fork reads one `cache_seqlens` tensor from every layer
    /// (`BC_Attention::run_gr`).
    seqlens: ByteBuf,
    /// u32 rope base for the standalone path. Same integer as `seqlens` on
    /// this decode (no MRoPE offset). Ignored when `bound_positions` is set.
    positions: ByteBuf,
    /// Pinned source for [`Self::forward`]'s async copy into `seqlens`.
    ///
    /// One slot. The caller synchronizes `gpu.stream` before the next
    /// `forward`, which is what the layer test's download does. A second
    /// `forward` without that wait races this page.
    seq_host: PinnedBuf<u8>,
    /// Pinned source for the matching copy into `positions`.
    pos_host: PinnedBuf<u8>,
    /// Device address of the shared int32 `cache_seqlens`, when the model owns it.
    bound_seqlens: Option<u64>,
    /// Device address of the shared u32 rope `positions`.
    bound_positions: Option<u64>,
    partial_o: ByteBuf,
    partial_ml: ByteBuf,
    o_proj: Exl3Linear,
    /// q/k/v again, so `rows > 32` can call [`Exl3Linear::forward_rows`].
    ///
    /// [`Exl3Linear`] keeps its trellis private, and the sliced bundle above
    /// has to stay for decode. These are the fork's separate linears
    /// (`project_qkv` when `bsz * q_len > 32`). The `rows <= 32` path does
    /// not read them.
    q_linear: Exl3Linear,
    k_linear: Exl3Linear,
    v_linear: Exl3Linear,
}

/// One EXL3 source in the sliced bundle.
///
/// `trellis`, `suh`, and `svh` are not named again after load: the slice
/// lists hold their device addresses, and dropping the buffers would free
/// that memory out from under the lists.
#[allow(dead_code)]
struct Proj {
    trellis: ByteBuf,
    suh: ByteBuf,
    svh: ByteBuf,
    out: ByteBuf,
}

struct Source {
    n: i32,
    k_bits: i32,
    codebook: Codebook,
    proj: Proj,
}

struct SliceLists {
    b: Vec<u64>,
    suh: Vec<u64>,
    svh: Vec<u64>,
    c: Vec<u64>,
    size_n: Vec<i32>,
    stride: Vec<i32>,
    had: Vec<i32>,
}

#[derive(Debug)]
struct RopeConsts {
    /// The parsed theta. Unit tests check it; `inv_freq` is what the kernel reads.
    #[allow(dead_code)]
    theta: f32,
    partial_head_dim: i32,
    norm_eps: f32,
    inv_freq: Vec<f32>,
}

impl AttnLayer {
    /// Load layer `layer`'s `self_attn.*` weights and allocate an empty cache
    /// for `max_tokens` (a multiple of the 256-token page).
    pub fn load(
        gpu: &Gpu,
        ckpt: &Checkpoint,
        shape: &ModelShape,
        layer: u32,
        max_tokens: u32,
    ) -> Result<Self> {
        if max_tokens == 0 || !max_tokens.is_multiple_of(PAGE) {
            return Err(Error::Input(format!(
                "max_tokens {max_tokens} is not a positive multiple of {PAGE}"
            )));
        }
        let kind = shape.layer_kinds.get(layer as usize).ok_or_else(|| {
            Error::Input(format!(
                "layer {layer} is out of range ({})",
                shape.num_layers
            ))
        })?;
        if *kind != LayerKind::FullAttention {
            return Err(Error::Input(format!(
                "layer {layer} is {kind}, not full attention"
            )));
        }
        if shape.num_q_heads != u32::try_from(Q_HEADS).unwrap_or(0)
            || shape.num_kv_heads != u32::try_from(KV_HEADS).unwrap_or(0)
            || shape.head_dim != u32::try_from(HEAD_DIM).unwrap_or(0)
        {
            return Err(Error::Weights(
                "paged decode cubin is the 27B tile (24 query heads, 4 kv heads, head dim 256)"
                    .into(),
            ));
        }
        let hidden = i32::try_from(shape.hidden_size)
            .map_err(|_| Error::Weights("hidden_size does not fit in i32".into()))?;
        if hidden < 1 {
            return Err(Error::Weights("hidden_size is 0".into()));
        }

        let rope = read_rope(
            &read_config(&ckpt.directory)?,
            u32::try_from(HEAD_DIM).unwrap_or(0),
        )?;
        if rope.partial_head_dim > HEAD_DIM {
            return Err(Error::Weights(format!(
                "partial rotary dim {} is wider than head dim {HEAD_DIM}",
                rope.partial_head_dim
            )));
        }

        let base = format!("model.language_model.layers.{layer}.self_attn");
        let q_features = Q_HEADS * HEAD_DIM;
        let q = load_source(gpu, ckpt, &format!("{base}.q_proj"), q_features * 2, hidden)?;
        let k = load_source(
            gpu,
            ckpt,
            &format!("{base}.k_proj"),
            KV_HEADS * HEAD_DIM,
            hidden,
        )?;
        let v = load_source(
            gpu,
            ckpt,
            &format!("{base}.v_proj"),
            KV_HEADS * HEAD_DIM,
            hidden,
        )?;
        if k.k_bits != q.k_bits
            || v.k_bits != q.k_bits
            || k.codebook != q.codebook
            || v.codebook != q.codebook
        {
            return Err(Error::Weights(format!(
                "{base}: q/k/v bitrate or codebook disagree"
            )));
        }
        let width = gcd(gcd(q.n, k.n), v.n);
        if width < 256 || width % 128 != 0 {
            return Err(Error::Weights(format!(
                "{base}: slice width {width} is not a multiple of 128 of at least 256"
            )));
        }
        let lists = slice_lists([&q, &k, &v], width, q.k_bits);
        let slices = i32::try_from(lists.had.len())
            .map_err(|_| Error::Weights(format!("{base}: too many slices")))?;
        if slices < 1 {
            return Err(Error::Weights(format!("{base}: sliced bundle is empty")));
        }

        let q_norm = load_norm(gpu, ckpt, &format!("{base}.q_norm.weight"), HEAD_DIM)?;
        let k_norm = load_norm(gpu, ckpt, &format!("{base}.k_norm.weight"), HEAD_DIM)?;
        let o_prefix = format!("{base}.o_proj");
        let o_linear = find_linear(ckpt, &o_prefix)?;
        if o_linear.bias.is_some() {
            return Err(Error::Weights(format!("{o_prefix}: bias is not applied")));
        }
        match o_linear.k {
            Bitrate::Int(4) => {}
            other => {
                return Err(Error::Weights(format!(
                    "{o_prefix}: rate {other:?} is not K = 4"
                )))
            }
        }
        if !matches!(o_linear.codebook, berth_exl3::Codebook::Mul1) {
            return Err(Error::Weights(format!("{o_prefix}: codebook is not mul1")));
        }
        let o_proj = Exl3Linear::load(gpu, ckpt, &o_prefix)?;
        if o_proj.size_k != q_features || o_proj.size_n != hidden {
            return Err(Error::Weights(format!(
                "{o_prefix}: {}x{}, expected {q_features}x{hidden}; padded hidden is not handled",
                o_proj.size_k, o_proj.size_n
            )));
        }
        let q_linear =
            load_rows_linear(gpu, ckpt, &format!("{base}.q_proj"), hidden, q_features * 2)?;
        let kv_n = KV_HEADS * HEAD_DIM;
        let k_linear = load_rows_linear(gpu, ckpt, &format!("{base}.k_proj"), hidden, kv_n)?;
        let v_linear = load_rows_linear(gpu, ckpt, &format!("{base}.v_proj"), hidden, kv_n)?;

        let num_pages_u = max_tokens / PAGE;
        let num_pages = i32::try_from(num_pages_u)
            .map_err(|_| Error::Input("page count does not fit in i32".into()))?;
        let token_dim = KV_HEADS * HEAD_DIM;
        if token_dim % 32 != 0 {
            return Err(Error::Weights("kv width is not a multiple of 32".into()));
        }
        let group = Q_HEADS / KV_HEADS;
        let programs = KV_HEADS * ((group + BLOCK_H - 1) / BLOCK_H);
        let cap = splits_cap(programs, gpu.sm_count)?;
        let q_bytes = usize::try_from(q_features).unwrap_or(0) * 2;
        let partial_o = partial_elems(programs, cap, HD_PAD)?;
        let partial_ml = partial_elems(programs, cap, 2)?;
        let had_bytes = 3usize
            .checked_mul(usize::try_from(hidden).unwrap_or(0))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| Error::Input("qkv scratch overflows".into()))?;
        let carrier_bytes = usize::try_from(slices)
            .ok()
            .and_then(|n| n.checked_mul(usize::try_from(width).unwrap_or(0)))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| Error::Input("qkv carrier overflows".into()))?;

        // Room for the widest padded decode width; columns past the cache
        // are 0 and never read (the kernels stop at cache_seqlens).
        let table_len = (num_pages + 15) / 16 * 16;
        let table: Vec<i32> = (0..table_len)
            .map(|p| if p < num_pages { p } else { 0 })
            .collect();
        let qkv = [q.proj, k.proj, v.proj];
        Ok(Self {
            hidden,
            head_dim: HEAD_DIM,
            q_heads: Q_HEADS,
            kv_heads: KV_HEADS,
            q_numel: i64::from(q_features),
            token_dim,
            width,
            slices,
            size_k: hidden,
            k_bits: q.k_bits,
            codebook: q.codebook,
            num_pages,
            decode_width: num_pages,
            max_tokens,
            programs,
            splits_cap: cap,
            partial_head_dim: rope.partial_head_dim,
            norm_eps: rope.norm_eps,
            qkv,
            b_list: upload_u64(gpu, &lists.b)?,
            suh_list: upload_u64(gpu, &lists.suh)?,
            svh_list: upload_u64(gpu, &lists.svh)?,
            c_ptrs: upload_u64(gpu, &lists.c)?,
            size_n_list: upload_i32(gpu, &lists.size_n)?,
            n_stride_list: upload_i32(gpu, &lists.stride)?,
            had_src_list: upload_i32(gpu, &lists.had)?,
            a_had: gpu.zeroed(had_bytes)?,
            carrier: gpu.zeroed(carrier_bytes)?,
            q: gpu.zeroed(q_bytes)?,
            g: gpu.zeroed(q_bytes)?,
            attn_out: gpu.zeroed(q_bytes)?,
            q_norm,
            k_norm,
            inv_freq: upload_f32(gpu, &rope.inv_freq)?,
            h32: upload_bytes(gpu, &h32_bytes())?,
            k_pages: gpu.zeroed(cache_bytes(num_pages_u, token_dim)?)?,
            v_pages: gpu.zeroed(cache_bytes(num_pages_u, token_dim)?)?,
            k_scales: gpu.zeroed(scale_bytes(num_pages_u, token_dim)?)?,
            v_scales: gpu.zeroed(scale_bytes(num_pages_u, token_dim)?)?,
            block_table: upload_i32(gpu, &table)?,
            seqlens: gpu.zeroed(4)?,
            positions: gpu.zeroed(4)?,
            seq_host: PinnedBuf::zeroed(&gpu.ctx, 4)?,
            pos_host: PinnedBuf::zeroed(&gpu.ctx, 4)?,
            bound_seqlens: None,
            bound_positions: None,
            partial_o: gpu.zeroed(
                partial_o
                    .checked_mul(4)
                    .ok_or_else(|| Error::Input("partial_o scratch overflows".into()))?,
            )?,
            partial_ml: gpu.zeroed(
                partial_ml
                    .checked_mul(4)
                    .ok_or_else(|| Error::Input("partial_ml scratch overflows".into()))?,
            )?,
            o_proj,
            q_linear,
            k_linear,
            v_linear,
        })
    }

    /// Point rope and the cache kernels at device words the model stages.
    ///
    /// `seqlens` is one int32 (length before this token). `positions` is one
    /// u32, the rope base. On this decode they are the same address: the
    /// value is the position, and `rope.cu`'s `get_pos` reads
    /// `positions[batch]` as `uint32_t` while `quant_cache_paged` reads the
    /// same bits as the int32 length. The fork aliases them the same way
    /// when there is no MRoPE offset (`modules/attn.py`, `positions =
    /// cache_seqlens`).
    pub(crate) fn bind_sequence(&mut self, seqlens: u64, positions: u64) {
        self.bound_seqlens = Some(seqlens);
        self.bound_positions = Some(positions);
    }

    /// Set this layer's cache length to 0.
    ///
    /// A fresh job starts at `kv_position = 0` (`generator/pagetable.py`,
    /// `Sequence.__init__` and `CachePage.add_ref_unique`). Those paths do
    /// not zero K/V pages. The next token overwrites position 0 before
    /// attention reads it, and the kernel stops at `cache_seqlens`.
    ///
    /// Writes the private `seqlens` and rope `positions` words. When
    /// [`Self::bind_sequence`] is in effect the kernels read the shared word
    /// instead; [`crate::model::Model::reset_sequence`] clears that too.
    /// The page allocations are unchanged.
    /// Set the decode block-table width for a request whose prompt plus
    /// `max_new_tokens` is `max_seq_len` tokens, the way the fork's
    /// `iterate_gen` sizes it.
    pub fn set_decode_width(&mut self, max_seq_len: u32) {
        let pages = max_seq_len.div_ceil(PAGE).max(1);
        let padded = pages.div_ceil(16) * 16;
        let limit = u32::try_from(self.num_pages).unwrap_or(0).div_ceil(16) * 16;
        self.decode_width = i32::try_from(padded.min(limit)).unwrap_or(self.num_pages);
    }

    pub fn reset_cache(&mut self, gpu: &Gpu) -> Result<()> {
        let zero = [0u8; 4];
        self.seqlens.copy_from_host(&gpu.stream, &zero)?;
        self.positions.copy_from_host(&gpu.stream, &zero)?;
        Ok(())
    }

    /// `output = self_attn(input)` for the token at `position` (0-based; the
    /// cache holds `position` earlier tokens), writing this token's K/V.
    ///
    /// `input` is fp16 `[hidden]`; `output` is fp32 `[hidden]`. `kernels` is the
    /// Triton split/combine pair, loaded once and shared by every layer.
    ///
    /// Stages `position` into this layer's own device words with an async
    /// copy from pinned memory, then launches. Synchronize `gpu.stream`
    /// before calling this again: the pinned slot is single-buffered.
    /// [`Self::forward_bound`] is the model path and does not copy.
    pub fn forward(
        &mut self,
        gpu: &Gpu,
        kernels: &PagedDecode,
        input: &ByteBuf,
        output: &mut ByteBuf,
        position: u32,
    ) -> Result<()> {
        if self.bound_seqlens.is_some() {
            return Err(Error::Input(
                "attention reads the shared sequence buffers; do not pass a scalar position".into(),
            ));
        }
        self.check_forward(input, output, position)?;
        self.stage_local(gpu, position)?;
        self.launch(gpu, kernels, input, output)
    }

    /// Attention for a token whose `cache_seqlens` and rope `positions` are
    /// already in the device words from [`Self::bind_sequence`].
    ///
    /// No copy and no host sync. The model writes those words before the step.
    pub(crate) fn forward_bound(
        &mut self,
        gpu: &Gpu,
        kernels: &PagedDecode,
        input: &ByteBuf,
        output: &mut ByteBuf,
    ) -> Result<()> {
        if self.bound_seqlens.is_none() || self.bound_positions.is_none() {
            return Err(Error::Input(
                "attention sequence buffers are not bound".into(),
            ));
        }
        self.check_buffers(input, output)?;
        self.launch(gpu, kernels, input, output)
    }

    fn launch(
        &mut self,
        gpu: &Gpu,
        kernels: &PagedDecode,
        input: &ByteBuf,
        output: &mut ByteBuf,
    ) -> Result<()> {
        // SAFETY: buffers are device allocations owned by `self` or the caller,
        // sized for this tile, and live until `gpu.stream` finishes. qg, q, and
        // g do not overlap. Rope writes q and k in place. The cache write is
        // ordered before the paged read on this stream. The gate buffers do
        // not overlap. `o_proj` reads `attn_out` only after these launches.
        // `cache_seqlens` and rope `positions` already hold this token.
        unsafe { self.run_attention(gpu, kernels, input)? };
        self.o_proj
            .forward(gpu, &self.attn_out, output, OutDtype::F32)?;
        Ok(())
    }

    fn check_forward(&self, input: &ByteBuf, output: &ByteBuf, position: u32) -> Result<()> {
        if position >= self.max_tokens {
            return Err(Error::Input(format!(
                "position {position} is past the {}-token cache",
                self.max_tokens
            )));
        }
        if i32::try_from(position).is_err() {
            return Err(Error::Input(format!(
                "position {position} does not fit in i32"
            )));
        }
        self.check_buffers(input, output)
    }

    fn check_buffers(&self, input: &ByteBuf, output: &ByteBuf) -> Result<()> {
        let hidden = usize::try_from(self.hidden).unwrap_or(0);
        if input.num_bytes() < hidden * 2 {
            return Err(Error::Input(
                "attention input is shorter than hidden fp16".into(),
            ));
        }
        if output.num_bytes() < hidden * 4 {
            return Err(Error::Input(
                "attention output is shorter than hidden fp32".into(),
            ));
        }
        Ok(())
    }

    /// Async H2D of `position` into this layer's seqlens and rope positions.
    ///
    /// The two words are the same four bytes. `get_pos` (`rope.cu`) adds
    /// `positions[batch]` when the pointer is set. `cache_seqlens` is that
    /// length as int32, which matches for every position this cache can hold.
    fn stage_local(&mut self, gpu: &Gpu, position: u32) -> Result<()> {
        let seq = i32::try_from(position).expect("check_forward rejected an oversized position");
        let bytes = seq.to_le_bytes();
        self.seq_host.as_mut_slice().copy_from_slice(&bytes);
        self.pos_host.as_mut_slice().copy_from_slice(&bytes);
        let AttnLayer {
            seqlens,
            positions,
            seq_host,
            pos_host,
            ..
        } = self;
        // SAFETY: `seq_host` and `pos_host` stay allocated on `self`. The
        // caller synchronizes `gpu.stream` before the next `stage_local`,
        // so this copy is finished before those pages are written again.
        // The device words outlive the copy.
        unsafe {
            seqlens.copy_from_pinned_async(&gpu.stream, seq_host)?;
            positions.copy_from_pinned_async(&gpu.stream, pos_host)?;
        }
        Ok(())
    }

    unsafe fn run_attention(
        &self,
        gpu: &Gpu,
        kernels: &PagedDecode,
        input: &ByteBuf,
    ) -> Result<()> {
        let programs = kernels.programs(1)?;
        if programs != self.programs {
            return Err(Error::Input(format!(
                "paged decode programs {programs} != the {} this layer allocated",
                self.programs
            )));
        }
        let cap = splits_cap(programs, gpu.sm_count)?;
        if cap != self.splits_cap {
            return Err(Error::Input(format!(
                "splits_cap {cap} != the {} this layer allocated",
                self.splits_cap
            )));
        }
        let o_elems = kernels.partial_o_elems(1, cap)?;
        let ml_elems = kernels.partial_ml_elems(1, cap)?;
        if self.partial_o.num_bytes() != o_elems * 4 || self.partial_ml.num_bytes() != ml_elems * 4
        {
            return Err(Error::Input(
                "paged decode scratch does not match this cubin's tile".into(),
            ));
        }

        let stream = gpu.raw_stream();
        let qg = ptr(&self.qkv[0].out);
        let q = ptr(&self.q);
        let g = ptr(&self.g);
        let o = ptr(&self.attn_out);
        unsafe {
            exl3_mgemm(
                ptr(input).cast_const(),
                ptr(&self.b_list).cast_const(),
                ptr(&self.carrier),
                ptr(&self.suh_list).cast_const(),
                ptr(&self.a_had),
                ptr(&self.svh_list).cast_const(),
                1,
                self.size_k,
                self.width,
                self.k_bits,
                self.codebook,
                OutDtype::F16,
                1,
                self.slices,
                ptr(&self.size_n_list) as *const i32,
                ptr(&self.c_ptrs).cast_const(),
                ptr(&self.n_stride_list) as *const i32,
                ptr(&self.had_src_list) as *const i32,
                3,
                stream,
            )?;
            deinterleave_qg(qg.cast_const(), q, g, self.head_dim, self.q_numel, stream)?;
            rope(&self.rope_launch(), stream)?;
            quant_cache_paged(&self.quant_launch(), stream)?;
            kernels.launch(
                &gpu.stream,
                &self.paged_args(),
                1,
                self.decode_width,
                self.splits_cap,
            )?;
            mul_sigmoid(o, g.cast_const(), self.q_numel, stream)?;
        }
        Ok(())
    }

    fn rope_launch(&self) -> RopeLaunch {
        let q = ptr(&self.q);
        let k = ptr(&self.qkv[1].out);
        // Scalar `position` stays 0. `get_pos` (`rope.cu`) uses
        // `positions[batch]` whenever that pointer is set and ignores the
        // scalar. The fork's graph path passes the staged tensor
        // (`GP_rope_positions`, arg 14 in `rope.cu`) and still records the
        // scalar (`GP_rope_position`, arg 13) for the null-pointer mode.
        // Keeping the scalar at 0 means a captured launch does not depend on it.
        RopeLaunch {
            q: q.cast_const(),
            out_q: q,
            k: k.cast_const(),
            out_k: k,
            inv_freq: ptr(&self.inv_freq).cast_const(),
            bsz: 1,
            seq_len: 1,
            num_heads_q: self.q_heads,
            num_heads_k: self.kv_heads,
            head_dim: self.head_dim,
            q_head_stride: self.head_dim,
            k_head_stride: self.head_dim,
            partial_head_dim: self.partial_head_dim,
            position: 0,
            positions: self.positions_ptr(),
            position_ids: std::ptr::null(),
            rope_mode: ROPE_NEOX,
            attn_factor: 1.0,
            q_norm: ptr(&self.q_norm).cast_const(),
            k_norm: ptr(&self.k_norm).cast_const(),
            norm_eps: self.norm_eps,
            norm_constant_bias: 1.0,
            inv_freq_table: false,
            inv_freq_stride: 0,
            llama_4_scaling_beta: 0.0,
            llama_4_scaling_original: 0,
            position_ids_stride: 1,
            rotate_dims: 1,
            rotate_offset: 0,
        }
    }

    fn quant_launch(&self) -> QuantCacheLaunch {
        QuantCacheLaunch {
            k_in: ptr(&self.qkv[1].out).cast_const(),
            k_out: ptr(&self.k_pages),
            k_scales: ptr(&self.k_scales),
            v_in: ptr(&self.qkv[2].out).cast_const(),
            v_out: ptr(&self.v_pages),
            v_scales: ptr(&self.v_scales),
            cache_seqlens: self.seqlens_ptr(),
            block_table: ptr(&self.block_table).cast_const(),
            bsz: 1,
            seq_len: 1,
            blocks_per_seq: self.num_pages,
            token_dim: self.token_dim,
            k_bits: i32::try_from(CACHE_BITS).unwrap_or(3),
            v_bits: i32::try_from(CACHE_BITS).unwrap_or(3),
            page_size: i32::try_from(PAGE).unwrap_or(256),
            compand_a: 0.0,
            in_contiguous: true,
        }
    }

    fn paged_args(&self) -> PagedDecodeArgs {
        let q = self.q.ptr().as_u64();
        PagedDecodeArgs {
            q,
            k_cache: self.k_pages.ptr().as_u64(),
            v_cache: self.v_pages.ptr().as_u64(),
            block_table: self.block_table.ptr().as_u64(),
            cache_seqlens: self.seqlens_addr(),
            out: self.attn_out.ptr().as_u64(),
            partial_o: self.partial_o.ptr().as_u64(),
            partial_ml: self.partial_ml.ptr().as_u64(),
            k_scales: self.k_scales.ptr().as_u64(),
            v_scales: self.v_scales.ptr().as_u64(),
            h32: self.h32.ptr().as_u64(),
            sinks: q,
        }
    }

    fn seqlens_addr(&self) -> u64 {
        self.bound_seqlens.unwrap_or(self.seqlens.ptr().as_u64())
    }

    fn seqlens_ptr(&self) -> *const std::ffi::c_void {
        self.seqlens_addr() as *const std::ffi::c_void
    }

    fn positions_ptr(&self) -> *const std::ffi::c_void {
        self.bound_positions
            .unwrap_or(self.positions.ptr().as_u64()) as *const std::ffi::c_void
    }
}

fn load_source(
    gpu: &Gpu,
    ckpt: &Checkpoint,
    prefix: &str,
    expect_n: i32,
    expect_k: i32,
) -> Result<Source> {
    let linear = find_linear(ckpt, prefix)?;
    if linear.bias.is_some() {
        return Err(Error::Weights(format!("{prefix}: bias is not applied")));
    }
    let k_bits = match linear.k {
        Bitrate::Int(4) => 4,
        other => {
            return Err(Error::Weights(format!(
                "{prefix}: rate {other:?}, the sliced kernel is K = 4"
            )))
        }
    };
    let codebook = match linear.codebook {
        berth_exl3::Codebook::Mul1 => Codebook::Mul1,
        berth_exl3::Codebook::Mcg => {
            return Err(Error::Weights(format!(
                "{prefix}: codebook mcg is not mul1"
            )))
        }
        berth_exl3::Codebook::Default => {
            return Err(Error::Weights(format!("{prefix}: codebook is not mul1")))
        }
    };
    let shape = &linear.trellis.shape;
    if linear.trellis.dtype != DType::I16 || shape.len() != 3 {
        return Err(Error::Weights(format!(
            "{prefix}: trellis is not rank-3 int16"
        )));
    }
    let n = shape_dim(shape[1], 16, prefix, "N")?;
    let k_in = shape_dim(shape[0], 16, prefix, "K")?;
    if shape[2] != u64::from(k_bits as u32) * 16 {
        return Err(Error::Weights(format!(
            "{prefix}: trellis tail {} is not 16 * K",
            shape[2]
        )));
    }
    if k_in != expect_k {
        return Err(Error::Weights(format!(
            "{prefix}: in_features {k_in} != hidden {expect_k}; padded hidden is not handled"
        )));
    }
    if n != expect_n {
        return Err(Error::Weights(format!(
            "{prefix}: N is {n}, expected {expect_n}"
        )));
    }
    let (suh, svh) = match (&linear.input_scale, &linear.output_scale) {
        (HadamardScale::Unpacked(suh), HadamardScale::Unpacked(svh)) => (suh, svh),
        _ => {
            return Err(Error::Weights(format!(
                "{prefix}: packed su / sv signs are not handled yet"
            )))
        }
    };
    if suh.dtype != DType::F16 || suh.length != u64::from(k_in as u32) * 2 {
        return Err(Error::Weights(format!("{prefix}: suh is not K fp16")));
    }
    if svh.dtype != DType::F16 || svh.length != u64::from(n as u32) * 2 {
        return Err(Error::Weights(format!("{prefix}: svh is not N fp16")));
    }
    Ok(Source {
        n,
        k_bits,
        codebook,
        proj: Proj {
            trellis: upload_bytes(gpu, &read_tensor(&linear.trellis)?)?,
            suh: upload_bytes(gpu, &read_tensor(suh)?)?,
            svh: upload_bytes(gpu, &read_tensor(svh)?)?,
            out: gpu.zeroed(usize::try_from(n).unwrap_or(0) * 2)?,
        },
    })
}

/// The same linear as [`load_source`], through [`Exl3Linear`] so prefill can
/// call [`Exl3Linear::forward_rows`]. `load_source` already rejected a bias
/// and a non-mul1 K = 4 trellis.
fn load_rows_linear(
    gpu: &Gpu,
    ckpt: &Checkpoint,
    prefix: &str,
    expect_k: i32,
    expect_n: i32,
) -> Result<Exl3Linear> {
    let linear = Exl3Linear::load(gpu, ckpt, prefix)?;
    if linear.size_k != expect_k || linear.size_n != expect_n {
        return Err(Error::Weights(format!(
            "{prefix}: {}x{}, expected {expect_k}x{expect_n}",
            linear.size_k, linear.size_n
        )));
    }
    Ok(linear)
}

fn shape_dim(groups: u64, scale: u64, prefix: &str, what: &str) -> Result<i32> {
    let n = groups
        .checked_mul(scale)
        .ok_or_else(|| Error::Weights(format!("{prefix}: {what} overflows")))?;
    i32::try_from(n).map_err(|_| Error::Weights(format!("{prefix}: {what} does not fit in i32")))
}

fn load_norm(gpu: &Gpu, ckpt: &Checkpoint, name: &str, head_dim: i32) -> Result<ByteBuf> {
    let info = ckpt
        .tensors
        .get(name)
        .ok_or_else(|| Error::Weights(format!("tensor {name} is missing")))?;
    if info.dtype != DType::Bf16 {
        return Err(Error::Weights(format!(
            "{name}: dtype {}, expected BF16",
            info.dtype
        )));
    }
    let numel = info
        .shape
        .iter()
        .try_fold(1u64, |acc, dim| acc.checked_mul(*dim));
    let expect = u64::try_from(head_dim).unwrap_or(0);
    if numel != Some(expect) || info.length != expect * 2 {
        return Err(Error::Weights(format!(
            "{name}: expected {head_dim} bf16 values"
        )));
    }
    upload_tensor(gpu, ckpt, name)
}

fn slice_lists(sources: [&Source; 3], width: i32, k_bits: i32) -> SliceLists {
    let mut lists = SliceLists {
        b: Vec::new(),
        suh: Vec::new(),
        svh: Vec::new(),
        c: Vec::new(),
        size_n: Vec::new(),
        stride: Vec::new(),
        had: Vec::new(),
    };
    for (index, source) in sources.iter().enumerate() {
        lists.suh.push(source.proj.suh.ptr().as_u64());
        let mut n0 = 0;
        while n0 < source.n {
            // `SlicedMultiLinear`: trellis + (n0 // 16) * 16 * K * sizeof(int16),
            // svh and the output at column n0.
            let trellis_off = i64::from(n0 / 16) * 16 * i64::from(k_bits) * 2;
            lists
                .b
                .push(source.proj.trellis.ptr().as_u64() + u64::try_from(trellis_off).unwrap_or(0));
            let column = u64::try_from(n0).unwrap_or(0) * 2;
            lists.svh.push(source.proj.svh.ptr().as_u64() + column);
            lists.c.push(source.proj.out.ptr().as_u64() + column);
            lists.size_n.push(width);
            lists.stride.push(source.n);
            lists.had.push(i32::try_from(index).unwrap_or(0));
            n0 += width;
        }
    }
    lists
}

fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a.abs()
}

fn cache_bytes(num_pages: u32, token_dim: i32) -> Result<usize> {
    // int32 words: (token_dim / 32) * bits per token, page of 256.
    let groups = u64::from(u32::try_from(token_dim).unwrap_or(0) / 32);
    let bytes = u64::from(num_pages) * u64::from(PAGE) * groups * u64::from(CACHE_BITS) * 4;
    usize::try_from(bytes).map_err(|_| Error::Input("k/v cache does not fit in memory".into()))
}

fn scale_bytes(num_pages: u32, token_dim: i32) -> Result<usize> {
    let groups = u64::from(u32::try_from(token_dim).unwrap_or(0) / 32);
    let bytes = u64::from(num_pages) * u64::from(PAGE) * groups * 2;
    usize::try_from(bytes).map_err(|_| Error::Input("k/v scales do not fit in memory".into()))
}

fn partial_elems(programs: i32, cap: i32, tail: i32) -> Result<usize> {
    let n = i64::from(programs) * i64::from(cap) * i64::from(BLOCK_ROWS) * i64::from(tail);
    usize::try_from(n).map_err(|_| Error::Input("attention scratch overflows".into()))
}

fn upload_bytes(gpu: &Gpu, bytes: &[u8]) -> Result<ByteBuf> {
    Ok(DeviceBuffer::from_host(&gpu.stream, bytes)?)
}

fn upload_u64(gpu: &Gpu, values: &[u64]) -> Result<ByteBuf> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    upload_bytes(gpu, &bytes)
}

fn upload_i32(gpu: &Gpu, values: &[i32]) -> Result<ByteBuf> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    upload_bytes(gpu, &bytes)
}

fn upload_f32(gpu: &Gpu, values: &[f32]) -> Result<ByteBuf> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    upload_bytes(gpu, &bytes)
}

fn read_config(dir: &Path) -> Result<Vec<u8>> {
    let path = dir.join("config.json");
    let meta = std::fs::metadata(&path).map_err(|source| Error::Read {
        name: path.display().to_string(),
        source,
    })?;
    if meta.len() > 32 * 1024 * 1024 {
        return Err(Error::Weights(format!(
            "{} is larger than 32 MiB",
            path.display()
        )));
    }
    std::fs::read(&path).map_err(|source| Error::Read {
        name: path.display().to_string(),
        source,
    })
}

/// `RoPE._rope_params_default`, with the one CUDA-vs-libm `powf` ulp folded in.
///
/// `rotary_dim` is the fork's `dim` (`rotary_dim`, or
/// `int(head_dim * partial_rotary_factor)`). Frequencies are
/// `arange(0, dim, 2) / dim`.
fn inv_freq(theta: f32, rotary_dim: i32) -> Vec<f32> {
    let dim = rotary_dim as f32;
    (0..rotary_dim)
        .step_by(2)
        .map(|i| {
            let exp = i as f32 / dim;
            1.0 / cuda_powf(theta, exp)
        })
        .collect()
}

/// Device `powf` for the 27B table.
///
/// Host `f32::powf` matches CUDA `powf` on every entry of
/// `testdata/kernels/attn/inv_freq.bin` except index 28, exponent `56/64`.
/// libm returns pow bits `0x49a2c88b`; the device returns `0x49a2c88c`.
/// The reciprocal then differs by one ulp (`0x35494c57` vs `0x35494c56`).
fn cuda_powf(base: f32, exp: f32) -> f32 {
    let mut pow = base.powf(exp);
    if exp.to_bits() == 0x3f60_0000
        && base.to_bits() == 10_000_000f32.to_bits()
        && pow.to_bits() == 0x49a2_c88b
    {
        pow = f32::from_bits(0x49a2_c88c);
    }
    pow
}

/// `_get_h32`: Sylvester Hadamard of order 32, divided by `sqrt(32)`, fp16.
fn h32_bytes() -> [u8; 2048] {
    let mut n = 1usize;
    let mut h = vec![1.0f32];
    while n < 32 {
        let n2 = n * 2;
        let mut next = vec![0.0f32; n2 * n2];
        for row in 0..n {
            for col in 0..n {
                let value = h[row * n + col];
                next[row * n2 + col] = value;
                next[row * n2 + col + n] = value;
                next[(row + n) * n2 + col] = value;
                next[(row + n) * n2 + col + n] = -value;
            }
        }
        h = next;
        n = n2;
    }
    let scale = 32f32.sqrt();
    let mut out = [0u8; 2048];
    for (index, value) in h.iter().enumerate() {
        out[index * 2..index * 2 + 2].copy_from_slice(&f32_to_f16_rne(value / scale).to_le_bytes());
    }
    out
}

/// Round-to-nearest-even f32 → fp16 bits.
fn f32_to_f16_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let abs = bits & 0x7fff_ffff;
    if abs >= 0x7f80_0000 {
        return if abs == 0x7f80_0000 {
            sign | 0x7c00
        } else {
            sign | 0x7e00
        };
    }
    if abs >= 0x4780_0000 {
        return sign | 0x7c00;
    }
    let exp = abs >> 23;
    let mantissa = abs & 0x007f_ffff;
    if abs < 0x3880_0000 {
        if exp == 0 {
            return sign;
        }
        let shift = 126 - exp;
        let significand = mantissa | 0x0080_0000;
        let half = 1u32 << (shift - 1);
        let truncated = significand >> shift;
        let rem = significand & ((1u32 << shift) - 1);
        let round_up = rem > half || (rem == half && (truncated & 1) == 1);
        let rounded = truncated + u32::from(round_up);
        if rounded > 0x03ff {
            return sign | 0x0400;
        }
        return sign | rounded as u16;
    }
    let mut half_exp = exp - 112;
    let mut half_mant = mantissa >> 13;
    let rem = mantissa & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (half_mant & 1) == 1) {
        half_mant += 1;
        if half_mant == 0x400 {
            half_mant = 0;
            half_exp += 1;
        }
    }
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((half_exp as u16) << 10) | (half_mant as u16)
}

/// Fork rope lookup inside `text_config`.
///
/// Theta: `rope_theta`, then `rope_parameters.rope_theta`, else `1e7`
/// (Qwen3.5's default). Partial factor: `partial_rotary_factor`, then the
/// same key under `rope_parameters`, else `1`. `rope_scaling` wins over
/// `rope_parameters` for the type. `default` and `mrope` share
/// `_rope_params_default`. `rms_norm_eps` is required.
fn read_rope(bytes: &[u8], head_dim: u32) -> Result<RopeConsts> {
    let root = parse_json(bytes).map_err(|err| Error::Weights(format!("config.json: {err}")))?;
    let text = match dig(&root, &["text_config"])? {
        Some(obj @ Json::Object(_)) => obj,
        Some(_) => return Err(Error::Weights("text_config is not an object".into())),
        None => return Err(Error::Weights("config.json has no text_config".into())),
    };
    let norm_eps = finite_f32(
        lookup_number(text, &[&["rms_norm_eps"]], false, None, "rms_norm_eps")?,
        "rms_norm_eps",
    )?;
    if norm_eps <= 0.0 {
        return Err(Error::Weights(format!(
            "rms_norm_eps {norm_eps} is not positive"
        )));
    }
    let theta = finite_f32(
        lookup_number(
            text,
            &[&["rope_theta"], &["rope_parameters", "rope_theta"]],
            true,
            Some(DEFAULT_ROPE_THETA),
            "rope_theta",
        )?,
        "rope_theta",
    )?;
    if theta <= 0.0 {
        return Err(Error::Weights(format!(
            "rope_theta {theta} is not positive"
        )));
    }
    let partial = lookup_number(
        text,
        &[
            &["partial_rotary_factor"],
            &["rope_parameters", "partial_rotary_factor"],
        ],
        true,
        Some(1.0),
        "partial_rotary_factor",
    )?;
    let explicit = lookup_int(text, &["rotary_dim"])?;
    let rotary = rotary_dim(head_dim, partial, explicit)?;
    let scaling = lookup_object(text, &[&["rope_scaling"], &["rope_parameters"]])?;
    let rope_type = match scaling {
        None => None,
        Some(obj) => match obj_get(obj, "rope_type").or_else(|| obj_get(obj, "type")) {
            None | Some(Json::Null) => None,
            Some(Json::String(kind)) => Some(kind.as_str()),
            Some(_) => return Err(Error::Weights("rope_type is not a string".into())),
        },
    };
    match rope_type {
        None | Some("default") | Some("mrope") => {}
        Some(other) => {
            return Err(Error::Weights(format!(
                "rope_type {other} is not the default inv_freq"
            )))
        }
    }
    let table = inv_freq(theta, rotary);
    if table.is_empty() {
        return Err(Error::Weights("inv_freq is empty".into()));
    }
    let partial_head_dim = i32::try_from(table.len() * 2)
        .map_err(|_| Error::Weights("partial rotary dim does not fit in i32".into()))?;
    if partial_head_dim > i32::try_from(head_dim).unwrap_or(i32::MAX) {
        return Err(Error::Weights(format!(
            "partial rotary dim {partial_head_dim} is wider than head dim {head_dim}"
        )));
    }
    Ok(RopeConsts {
        theta,
        partial_head_dim,
        norm_eps,
        inv_freq: table,
    })
}

fn rotary_dim(head_dim: u32, partial: f64, explicit: Option<i32>) -> Result<i32> {
    if let Some(dim) = explicit.filter(|dim| *dim > 0) {
        return Ok(dim);
    }
    if !partial.is_finite() || partial <= 0.0 {
        return Err(Error::Weights(format!(
            "partial_rotary_factor {partial} is not positive"
        )));
    }
    let dim = (f64::from(head_dim) * partial).trunc();
    if !(2.0..f64::from(i32::MAX)).contains(&dim) {
        return Err(Error::Weights(format!("rotary dim {dim} is out of range")));
    }
    Ok(dim as i32)
}

fn finite_f32(value: f64, what: &str) -> Result<f32> {
    let narrow = value as f32;
    if !narrow.is_finite() {
        return Err(Error::Weights(format!(
            "{what} {value} is not a finite f32"
        )));
    }
    Ok(narrow)
}

fn lookup_number(
    obj: &Json,
    paths: &[&[&str]],
    skip_wrong_type: bool,
    default: Option<f64>,
    what: &str,
) -> Result<f64> {
    for path in paths {
        match dig(obj, path)? {
            Some(Json::Number(n)) => return Ok(*n),
            None | Some(Json::Null) => {}
            Some(_) if skip_wrong_type => {}
            Some(_) => return Err(Error::Weights(format!("{what} has the wrong json type"))),
        }
    }
    default.ok_or_else(|| Error::Weights(format!("config.json is missing {what}")))
}

fn lookup_int(obj: &Json, path: &[&str]) -> Result<Option<i32>> {
    match dig(obj, path)? {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Number(n)) => {
            if n.fract() != 0.0 || *n < 0.0 || *n > f64::from(i32::MAX) {
                return Err(Error::Weights(format!(
                    "{} {n} is not a non-negative i32",
                    path.join(".")
                )));
            }
            Ok(Some(*n as i32))
        }
        Some(_) => Err(Error::Weights(format!(
            "{} has the wrong json type",
            path.join(".")
        ))),
    }
}

fn lookup_object<'a>(obj: &'a Json, paths: &[&[&str]]) -> Result<Option<&'a Json>> {
    for path in paths {
        match dig(obj, path)? {
            Some(value @ Json::Object(_)) => return Ok(Some(value)),
            None | Some(Json::Null) => {}
            Some(_) => {
                return Err(Error::Weights(format!(
                    "{} is not an object",
                    path.join(".")
                )))
            }
        }
    }
    Ok(None)
}

fn dig<'a>(value: &'a Json, path: &[&str]) -> Result<Option<&'a Json>> {
    let mut cur = value;
    for (depth, key) in path.iter().enumerate() {
        let Some(next) = obj_get(cur, key) else {
            return Ok(None);
        };
        if depth + 1 == path.len() {
            return Ok(Some(next));
        }
        if !matches!(next, Json::Object(_)) {
            return Err(Error::Weights(format!(
                "config.json {} is not an object",
                path[..=depth].join(".")
            )));
        }
        cur = next;
    }
    Ok(None)
}

fn obj_get<'a>(value: &'a Json, key: &str) -> Option<&'a Json> {
    match value {
        Json::Object(pairs) => pairs
            .iter()
            .rev()
            .find(|(name, _)| name == key)
            .map(|(_, v)| v),
        _ => None,
    }
}

/// Enough of JSON to walk `config.json`. Bools and arrays are stored so the
/// walk can pass them; the rope lookup does not read those values.
#[allow(dead_code)]
enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

fn parse_json(bytes: &[u8]) -> std::result::Result<Json, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "config.json is not utf-8".to_string())?;
    let mut i = 0;
    skip_ws(text, &mut i);
    let value = parse_value(text, &mut i, 0)?;
    skip_ws(text, &mut i);
    if i != text.len() {
        return Err(format!("trailing input at byte {i}"));
    }
    Ok(value)
}

fn parse_value(text: &str, i: &mut usize, depth: usize) -> std::result::Result<Json, String> {
    if depth > 64 {
        return Err("config.json is nested too deeply".into());
    }
    skip_ws(text, i);
    let b = *text
        .as_bytes()
        .get(*i)
        .ok_or("unexpected end of config.json")?;
    match b {
        b'{' => parse_object(text, i, depth),
        b'[' => parse_array(text, i, depth),
        b'"' => Ok(Json::String(parse_string(text, i)?)),
        b't' | b'f' | b'n' => parse_literal(text, i),
        b'-' | b'0'..=b'9' => Ok(Json::Number(parse_number(text, i)?)),
        _ => Err(format!("unexpected byte {b:#x} at {i}")),
    }
}

fn parse_object(text: &str, i: &mut usize, depth: usize) -> std::result::Result<Json, String> {
    *i += 1;
    let mut pairs = Vec::new();
    loop {
        skip_ws(text, i);
        if text.as_bytes().get(*i).copied() == Some(b'}') {
            if !pairs.is_empty() {
                return Err(format!("trailing comma at {i}"));
            }
            *i += 1;
            return Ok(Json::Object(pairs));
        }
        if text.as_bytes().get(*i).copied() != Some(b'"') {
            return Err(format!("expected a string key at {i}"));
        }
        let key = parse_string(text, i)?;
        skip_ws(text, i);
        expect(text, i, b':')?;
        pairs.push((key, parse_value(text, i, depth + 1)?));
        skip_ws(text, i);
        match text.as_bytes().get(*i).copied() {
            Some(b',') => *i += 1,
            Some(b'}') => {
                *i += 1;
                return Ok(Json::Object(pairs));
            }
            _ => return Err(format!("expected ',' or '}}' at {i}")),
        }
    }
}

fn parse_array(text: &str, i: &mut usize, depth: usize) -> std::result::Result<Json, String> {
    *i += 1;
    let mut items = Vec::new();
    loop {
        skip_ws(text, i);
        if text.as_bytes().get(*i).copied() == Some(b']') {
            if !items.is_empty() {
                return Err(format!("trailing comma at {i}"));
            }
            *i += 1;
            return Ok(Json::Array(items));
        }
        items.push(parse_value(text, i, depth + 1)?);
        skip_ws(text, i);
        match text.as_bytes().get(*i).copied() {
            Some(b',') => *i += 1,
            Some(b']') => {
                *i += 1;
                return Ok(Json::Array(items));
            }
            _ => return Err(format!("expected ',' or ']' at {i}")),
        }
    }
}

fn parse_string(text: &str, i: &mut usize) -> std::result::Result<String, String> {
    if text.as_bytes().get(*i).copied() != Some(b'"') {
        return Err(format!("expected a string at {i}"));
    }
    *i += 1;
    let mut out = String::new();
    loop {
        let rest = text.get(*i..).ok_or("unterminated string")?;
        let ch = rest.chars().next().ok_or("unterminated string")?;
        match ch {
            '"' => {
                *i += 1;
                return Ok(out);
            }
            '\\' => {
                *i += 1;
                let esc = text
                    .get(*i..)
                    .and_then(|s| s.chars().next())
                    .ok_or("bad escape")?;
                *i += esc.len_utf8();
                match esc {
                    '"' | '\\' | '/' => out.push(esc),
                    'b' => out.push('\u{0008}'),
                    'f' => out.push('\u{000c}'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'u' => push_unicode(&mut out, text, i)?,
                    _ => return Err(format!("bad escape \\{esc}")),
                }
            }
            c if c.is_control() => return Err("unescaped control character in string".into()),
            c => {
                out.push(c);
                *i += c.len_utf8();
            }
        }
    }
}

fn push_unicode(out: &mut String, text: &str, i: &mut usize) -> std::result::Result<(), String> {
    let unit = hex4(text, i)?;
    if (0xD800..=0xDBFF).contains(&unit) {
        let rest = text.get(*i..).unwrap_or("");
        if !rest.starts_with("\\u") {
            return Err("lone surrogate".into());
        }
        *i += 2;
        let low = hex4(text, i)?;
        if !(0xDC00..=0xDFFF).contains(&low) {
            return Err("bad surrogate pair".into());
        }
        let merged = 0x10000 + (((unit - 0xD800) << 10) | (low - 0xDC00));
        out.push(char::from_u32(merged).ok_or("bad unicode scalar")?);
        return Ok(());
    }
    if (0xDC00..=0xDFFF).contains(&unit) {
        return Err("lone surrogate".into());
    }
    out.push(char::from_u32(unit).ok_or("bad unicode scalar")?);
    Ok(())
}

fn hex4(text: &str, i: &mut usize) -> std::result::Result<u32, String> {
    let end = i.checked_add(4).ok_or("bad unicode escape")?;
    let digits = text.get(*i..end).ok_or("bad unicode escape")?;
    let unit =
        u32::from_str_radix(digits, 16).map_err(|_| format!("bad unicode escape {digits}"))?;
    *i = end;
    Ok(unit)
}

fn parse_literal(text: &str, i: &mut usize) -> std::result::Result<Json, String> {
    for (lit, value) in [
        ("true", Json::Bool(true)),
        ("false", Json::Bool(false)),
        ("null", Json::Null),
    ] {
        if text[*i..].starts_with(lit) {
            let after = text.as_bytes().get(*i + lit.len()).copied();
            if after.is_some_and(|c| c.is_ascii_alphanumeric() || c == b'_') {
                break;
            }
            *i += lit.len();
            return Ok(value);
        }
    }
    Err(format!("invalid json at {i}"))
}

fn parse_number(text: &str, i: &mut usize) -> std::result::Result<f64, String> {
    let start = *i;
    let bytes = text.as_bytes();
    if bytes.get(*i) == Some(&b'-') {
        *i += 1;
    }
    if bytes.get(*i) == Some(&b'0') {
        *i += 1;
    } else if bytes.get(*i).is_some_and(|c| c.is_ascii_digit()) {
        while bytes.get(*i).is_some_and(|c| c.is_ascii_digit()) {
            *i += 1;
        }
    } else {
        return Err(format!("expected a number at {start}"));
    }
    if bytes.get(*i) == Some(&b'.') {
        *i += 1;
        let frac = *i;
        while bytes.get(*i).is_some_and(|c| c.is_ascii_digit()) {
            *i += 1;
        }
        if *i == frac {
            return Err(format!("expected digits after decimal at {frac}"));
        }
    }
    if bytes.get(*i).is_some_and(|c| *c == b'e' || *c == b'E') {
        *i += 1;
        if bytes.get(*i).is_some_and(|c| *c == b'+' || *c == b'-') {
            *i += 1;
        }
        let exp = *i;
        while bytes.get(*i).is_some_and(|c| c.is_ascii_digit()) {
            *i += 1;
        }
        if *i == exp {
            return Err(format!("expected an exponent at {exp}"));
        }
    }
    text[start..*i]
        .parse::<f64>()
        .map_err(|_| format!("bad number {}", &text[start..*i]))
}

fn expect(text: &str, i: &mut usize, byte: u8) -> std::result::Result<(), String> {
    if text.as_bytes().get(*i).copied() != Some(byte) {
        return Err(format!("expected {} at {i}", byte as char));
    }
    *i += 1;
    Ok(())
}

fn skip_ws(text: &str, i: &mut usize) {
    let bytes = text.as_bytes();
    while matches!(bytes.get(*i), Some(b' ' | b'\n' | b'\r' | b'\t')) {
        *i += 1;
    }
}

/// Scratch for [`AttnLayer::prefill`], allocated once for the largest chunk
/// and shared by every attention layer, plus the Triton prefill kernels.
///
/// Holds the q/k/v/gate/output rows, the sliced-mgemm Hadamard image (up to
/// 32 rows), the cq3 staging buffers, and an identity block table. It does
/// not hold a second copy of any layer's weights.
///
/// Owner: prefill task ATTN.
pub struct AttnPrefillScratch {
    prefill: PagedPrefill,
    max_rows: u32,
    max_pages: i32,
    hidden: i32,
    width: i32,
    slices: i32,
    /// Rows the sliced mgemm Hadamard image can hold (`min(max_rows, 32)`).
    mgemm_rows: u32,
    /// Interleaved q and gate, fp16 `[max_rows, 2 * q_heads * head_dim]`.
    qg: ByteBuf,
    /// Key and value before RoPE, fp16 `[max_rows, kv_heads * head_dim]`.
    k: ByteBuf,
    v: ByteBuf,
    /// Deinterleaved query, gate, and gated attention output.
    q: ByteBuf,
    g: ByteBuf,
    o: ByteBuf,
    /// fp16 Hadamard image for the sliced bundle: 3 slabs of `mgemm_rows`.
    mgemm_had: ByteBuf,
    /// Per-slice output pointers into `qg`, `k`, and `v`. Stable for the
    /// life of those buffers.
    c_ptrs: ByteBuf,
    /// fp16 staging for chunks of `TWO_PASS_MIN_Q` rows or more, `stage_pages`
    /// pages each. Allocated on first use and grown by
    /// [`Self::reserve_staging`], never at load: a cache-sized pair is over
    /// 1 GiB at 270336 tokens. The fork allocates this per call from the block
    /// table's span (`triton_paged.py`, `pages_alloc`).
    scratch_k: Option<ByteBuf>,
    scratch_v: Option<ByteBuf>,
    stage_pages: i32,
    /// int32 `[0, 1, …, max_pages)`. The launch reads the first `bt_width`.
    identity: ByteBuf,
    global_scratch: ByteBuf,
    profile_scratch: ByteBuf,
}

/// One prefill chunk that fits this layer and `scratch`.
struct PrefillSpan {
    rows: i32,
    start: i32,
    end: i32,
    pages: i32,
}

impl AttnPrefillScratch {
    /// Room for chunks of up to `max_rows` tokens over a `max_tokens` cache.
    /// `cubin_dir` is the Triton cubin directory (`target/triton-cubins`).
    pub fn new(
        gpu: &Gpu,
        shape: &ModelShape,
        max_rows: u32,
        max_tokens: u32,
        cubin_dir: &std::path::Path,
    ) -> Result<Self> {
        if shape.num_q_heads != u32::try_from(Q_HEADS).unwrap_or(0)
            || shape.num_kv_heads != u32::try_from(KV_HEADS).unwrap_or(0)
            || shape.head_dim != u32::try_from(HEAD_DIM).unwrap_or(0)
        {
            return Err(Error::Weights(
                "paged prefill cubin is the 27B tile (24 query heads, 4 kv heads, head dim 256)"
                    .into(),
            ));
        }
        if max_rows < 2 {
            return Err(Error::Input(format!(
                "attention prefill max_rows {max_rows} is below 2"
            )));
        }
        if max_tokens == 0 || !max_tokens.is_multiple_of(PAGE) {
            return Err(Error::Input(format!(
                "max_tokens {max_tokens} is not a positive multiple of {PAGE}"
            )));
        }
        let max_pages = i32::try_from(max_tokens / PAGE)
            .map_err(|_| Error::Input("page count does not fit in i32".into()))?;
        let hidden = i32::try_from(shape.hidden_size)
            .map_err(|_| Error::Weights("hidden_size does not fit in i32".into()))?;
        if hidden < 1 {
            return Err(Error::Weights("hidden_size is 0".into()));
        }

        let prefill = PagedPrefill::load(&gpu.ctx, cubin_dir)?;
        let (width, slices, q_n, kv_n) = qkv_layout()?;
        let mgemm_rows = max_rows.min(SLICED_MAX_ROWS);
        let q_cols = Q_HEADS * HEAD_DIM;
        let qg = gpu.zeroed(fp16_bytes(max_rows, q_n)?)?;
        let k = gpu.zeroed(fp16_bytes(max_rows, kv_n)?)?;
        let v = gpu.zeroed(fp16_bytes(max_rows, kv_n)?)?;
        let ptrs = slice_out_ptrs(
            qg.ptr().as_u64(),
            k.ptr().as_u64(),
            v.ptr().as_u64(),
            q_n,
            kv_n,
            width,
        )?;
        if i32::try_from(ptrs.len()).ok() != Some(slices) {
            return Err(Error::Input(
                "prefill slice pointer count does not match the bundle".into(),
            ));
        }
        let (g_short, p_short) = prefill.scratch_bytes(1);
        let (g_wide, p_wide) = prefill.scratch_bytes(TWO_PASS_MIN_Q);
        let identity: Vec<i32> = (0..max_pages).collect();
        Ok(Self {
            prefill,
            max_rows,
            max_pages,
            hidden,
            width,
            slices,
            mgemm_rows,
            qg,
            k,
            v,
            q: gpu.zeroed(fp16_bytes(max_rows, q_cols)?)?,
            g: gpu.zeroed(fp16_bytes(max_rows, q_cols)?)?,
            o: gpu.zeroed(fp16_bytes(max_rows, q_cols)?)?,
            mgemm_had: gpu.zeroed(mgemm_had_bytes(mgemm_rows, hidden)?)?,
            c_ptrs: upload_u64(gpu, &ptrs)?,
            scratch_k: None,
            scratch_v: None,
            stage_pages: 0,
            identity: upload_i32(gpu, &identity)?,
            global_scratch: gpu.zeroed(u32_len(g_short.max(g_wide))?)?,
            profile_scratch: gpu.zeroed(u32_len(p_short.max(p_wide))?)?,
        })
    }

    /// Make the staging pair hold at least `pages` pages.
    ///
    /// The kernels touch `pages` pages (the launch's `bt_width`). Growth
    /// rounds up to a power of two, as the fork's `pages_alloc` does, so a
    /// growing context reallocates a handful of times; the cap is the cache's
    /// page count, since a 1056-page cache would otherwise round to 2048. The
    /// old pair is freed first (its drop waits for the context), so the peak
    /// is the new size. If the allocation fails, the pair is empty and the
    /// next call tries again at its own size.
    fn reserve_staging(&mut self, gpu: &Gpu, pages: i32) -> Result<()> {
        if pages <= self.stage_pages {
            return Ok(());
        }
        let want = staging_pages(1, pages)?.min(self.max_pages).max(pages);
        self.scratch_k = None;
        self.scratch_v = None;
        self.stage_pages = 0;
        let bytes = stage_bytes(want)?;
        let k = gpu
            .zeroed(bytes)
            .map_err(|err| staging_oom(err, want, bytes))?;
        let v = gpu
            .zeroed(bytes)
            .map_err(|err| staging_oom(err, want, bytes))?;
        self.scratch_k = Some(k);
        self.scratch_v = Some(v);
        self.stage_pages = want;
        Ok(())
    }
}

/// Bytes of one fp16 staging buffer of `pages` pages.
fn stage_bytes(pages: i32) -> Result<usize> {
    let per_page = usize::try_from(PAGE * KV_HEADS as u32 * HEAD_DIM as u32 * 2).unwrap_or(0);
    usize::try_from(pages)
        .ok()
        .and_then(|p| p.checked_mul(per_page))
        .ok_or_else(|| Error::Input("attention staging scratch overflows".into()))
}

fn staging_oom(err: Error, pages: i32, bytes: usize) -> Error {
    Error::Input(format!(
        "attention prefill staging: {pages} pages ({} MiB each for k and v) did not fit: {err}",
        bytes >> 20
    ))
}

impl AttnLayer {
    /// `output = self_attn(input)` for `rows > 1` tokens at positions
    /// `start .. start + rows`; the cache holds `start` earlier tokens and
    /// gets these. The fork's prefill path (`decode_flash_attn` for
    /// `q_len > 1`).
    ///
    /// `input` is fp16 `[rows, hidden]`; `output` is fp32 `[rows, hidden]`.
    ///
    /// `rows <= 32` is one sliced `exl3_mgemm` (`project_qkv_sliced`). Larger
    /// chunks are separate `q_proj`, `k_proj`, and `v_proj` linears. Either
    /// way [`PagedPrefill::prefill_chunk`] deinterleaves, applies fused q/k
    /// norm and RoPE at `start .. start + rows`, writes cq3 at
    /// `cache_seqlens = start`, runs the Triton prefill (fp16 staging when
    /// `rows >= 256`), and applies the output gate. The cubin is the Triton
    /// specialization class of this call: `q_len` and `kv_append_len` are
    /// `rows`, `num_pages_per_seq` is the live page count, `num_splits` comes
    /// from the fork's formula, and both windows are -1.
    /// [`PagedPrefill::prefill_chunk`] selects that cubin and errors if the
    /// class was not captured. `o_proj` is
    /// [`Exl3Linear::forward_rows`] with fp32 out.
    ///
    /// The private cache length becomes `start + rows` after the launch.
    /// A [`Self::bind_sequence`] word is left unchanged: every attention
    /// layer in the chunk still reads `start` from it. Synchronize
    /// `gpu.stream` before the next `prefill` or [`Self::forward`] on this
    /// layer. The pinned slots are single-buffered, same as `forward`.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill(
        &mut self,
        gpu: &Gpu,
        rows_scratch: &mut crate::device::RowScratch,
        scratch: &mut AttnPrefillScratch,
        input: &ByteBuf,
        output: &mut ByteBuf,
        start: u32,
        rows: u32,
    ) -> Result<()> {
        let span = self.prefill_span(scratch, input, output, start, rows)?;
        if needs_staging(span.rows) {
            scratch.reserve_staging(gpu, span.pages)?;
        }
        // Both host values are written before either copy. `seq_host` is
        // `start` and `pos_host` is `start + rows`. The first copy publishes
        // `start` for the kernels; the second publishes the new length after
        // them. Do not write either page again until `gpu.stream` completes.
        self.seq_host
            .as_mut_slice()
            .copy_from_slice(&span.start.to_le_bytes());
        self.pos_host
            .as_mut_slice()
            .copy_from_slice(&span.end.to_le_bytes());
        self.copy_length_from(gpu, true)?;
        if rows <= SLICED_MAX_ROWS {
            self.project_qkv_sliced(gpu, scratch, input, span.rows)?;
        } else {
            self.q_linear.forward_rows(
                gpu,
                rows_scratch,
                input,
                &mut scratch.qg,
                rows,
                OutDtype::F16,
            )?;
            self.k_linear.forward_rows(
                gpu,
                rows_scratch,
                input,
                &mut scratch.k,
                rows,
                OutDtype::F16,
            )?;
            self.v_linear.forward_rows(
                gpu,
                rows_scratch,
                input,
                &mut scratch.v,
                rows,
                OutDtype::F16,
            )?;
        }
        let args = self.prefill_args(scratch, span.rows);
        // SAFETY: qg, k, v, q, g, and o are scratch rows of at least `span.rows`
        // and do not alias each other. Cache pages, scales, and the identity
        // table cover `span.pages`. `cache_seqlens` is this layer's private
        // word, just staged to `start`, unless [`Self::bind_sequence`] points
        // it at the model's word (the model set that to `start`). Norms,
        // inv_freq, and h32 belong to the layer. Every buffer outlives
        // `gpu.stream`. The cq3 write is ordered before the attention read.
        unsafe {
            scratch
                .prefill
                .prefill_chunk(&gpu.stream, &args, span.rows, span.pages, span.start)?;
        }
        self.o_proj
            .forward_rows(gpu, rows_scratch, &scratch.o, output, rows, OutDtype::F32)?;
        self.copy_length_from(gpu, false)?;
        Ok(())
    }

    /// Async copy of `seq_host` (`start`) or `pos_host` (`start + rows`) into
    /// the private length words. Never the [`Self::bind_sequence`] word.
    fn copy_length_from(&mut self, gpu: &Gpu, start_value: bool) -> Result<()> {
        let AttnLayer {
            seqlens,
            positions,
            seq_host,
            pos_host,
            ..
        } = self;
        let host = if start_value { &*seq_host } else { &*pos_host };
        // SAFETY: `host` stays allocated on `self` and is not written again
        // until the caller synchronizes `gpu.stream`. The device words outlive
        // the copy. Both words read the same pinned page.
        unsafe {
            seqlens.copy_from_pinned_async(&gpu.stream, host)?;
            positions.copy_from_pinned_async(&gpu.stream, host)?;
        }
        Ok(())
    }

    fn prefill_span(
        &self,
        scratch: &AttnPrefillScratch,
        input: &ByteBuf,
        output: &ByteBuf,
        start: u32,
        rows: u32,
    ) -> Result<PrefillSpan> {
        if rows < 2 {
            return Err(Error::Input("attention prefill needs rows > 1".into()));
        }
        if rows > scratch.max_rows {
            return Err(Error::Input(format!(
                "prefill rows {rows} exceeds the scratch's {}",
                scratch.max_rows
            )));
        }
        if scratch.hidden != self.hidden
            || scratch.width != self.width
            || scratch.slices != self.slices
        {
            return Err(Error::Input(format!(
                "prefill scratch is {}x{} ({} slices); this layer is {}x{} ({} slices)",
                scratch.hidden, scratch.width, scratch.slices, self.hidden, self.width, self.slices
            )));
        }
        if self.partial_head_dim != ROTARY_DIM || self.norm_eps.to_bits() != NORM_EPS.to_bits() {
            return Err(Error::Input(format!(
                "prefill rope cubin is rotary dim {ROTARY_DIM} and eps {NORM_EPS}; this layer is {} / {}",
                self.partial_head_dim, self.norm_eps
            )));
        }
        let end = start
            .checked_add(rows)
            .ok_or_else(|| Error::Input("start + rows overflows".into()))?;
        if end > self.max_tokens {
            return Err(Error::Input(format!(
                "prefill end {end} is past the {}-token cache",
                self.max_tokens
            )));
        }
        let start_i = i32::try_from(start)
            .map_err(|_| Error::Input(format!("prefill start {start} does not fit in i32")))?;
        let end_i = i32::try_from(end)
            .map_err(|_| Error::Input(format!("prefill end {end} does not fit in i32")))?;
        let rows_i = i32::try_from(rows)
            .map_err(|_| Error::Input(format!("prefill rows {rows} does not fit in i32")))?;
        let pages = i32::try_from(end.div_ceil(PAGE))
            .map_err(|_| Error::Input("page count does not fit in i32".into()))?;
        if pages > self.num_pages || pages > scratch.max_pages {
            return Err(Error::Input(format!(
                "prefill needs {pages} pages; layer has {}, scratch has {}",
                self.num_pages, scratch.max_pages
            )));
        }
        let hidden = usize::try_from(self.hidden).unwrap_or(0);
        let rows_u = usize::try_from(rows).unwrap_or(0);
        let in_bytes = rows_u
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| Error::Input("prefill input size overflows".into()))?;
        let out_bytes = rows_u
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error::Input("prefill output size overflows".into()))?;
        need_bytes(input, in_bytes, "attention input")?;
        need_bytes(output, out_bytes, "attention output")?;
        let q_cols = Q_HEADS * HEAD_DIM;
        let kv_cols = KV_HEADS * HEAD_DIM;
        need_bytes(&scratch.qg, fp16_bytes(rows, q_cols * 2)?, "qg")?;
        need_bytes(&scratch.k, fp16_bytes(rows, kv_cols)?, "k")?;
        need_bytes(&scratch.v, fp16_bytes(rows, kv_cols)?, "v")?;
        need_bytes(&scratch.q, fp16_bytes(rows, q_cols)?, "q")?;
        need_bytes(&scratch.g, fp16_bytes(rows, q_cols)?, "g")?;
        need_bytes(&scratch.o, fp16_bytes(rows, q_cols)?, "o")?;
        if rows <= SLICED_MAX_ROWS {
            if rows > scratch.mgemm_rows {
                return Err(Error::Input(format!(
                    "sliced prefill rows {rows} exceeds the mgemm scratch's {}",
                    scratch.mgemm_rows
                )));
            }
            need_bytes(
                &scratch.mgemm_had,
                mgemm_had_bytes(rows, self.hidden)?,
                "mgemm hadamard",
            )?;
            let ptr_bytes = usize::try_from(self.slices)
                .unwrap_or(0)
                .checked_mul(8)
                .ok_or_else(|| Error::Input("mgemm c_ptrs overflow".into()))?;
            need_bytes(&scratch.c_ptrs, ptr_bytes, "mgemm c_ptrs")?;
        }
        if needs_staging(rows_i) {
            let table_bytes = usize::try_from(pages)
                .unwrap_or(0)
                .checked_mul(4)
                .ok_or_else(|| Error::Input("identity table overflows".into()))?;
            need_bytes(&scratch.identity, table_bytes, "identity block table")?;
        }
        let (global, profile) = scratch.prefill.scratch_bytes(rows_i);
        need_bytes(
            &scratch.global_scratch,
            u32_len(global)?,
            "triton global scratch",
        )?;
        need_bytes(
            &scratch.profile_scratch,
            u32_len(profile)?,
            "triton profile scratch",
        )?;
        Ok(PrefillSpan {
            rows: rows_i,
            start: start_i,
            end: end_i,
            pages,
        })
    }

    /// Interleaved q/g plus k and v, one sliced mgemm. `prefill_chunk`
    /// deinterleaves; doing it here would run that kernel twice.
    fn project_qkv_sliced(
        &self,
        gpu: &Gpu,
        scratch: &AttnPrefillScratch,
        input: &ByteBuf,
        rows: i32,
    ) -> Result<()> {
        // SAFETY: the slice lists alias this layer's q/k/v trellis. `c_ptrs`
        // address `scratch`'s qg/k/v at each slice column, with the source
        // width as the row stride. `mgemm_had` holds 3 * `rows` * hidden fp16
        // (the kernel strides by `rows`, not by the allocation). The carrier
        // is not written when `c_ptrs` is set; it carries the fp16 dtype and
        // the slice width. All of them outlive `gpu.stream`.
        let _shape = unsafe {
            berth_kernels::prefill_lin::exl3_mgemm(
                ptr(input).cast_const(),
                ptr(&self.b_list).cast_const(),
                ptr(&self.carrier),
                ptr(&self.suh_list).cast_const(),
                ptr(&scratch.mgemm_had),
                ptr(&self.svh_list).cast_const(),
                rows,
                self.size_k,
                self.width,
                self.k_bits,
                self.codebook,
                OutDtype::F16,
                1,
                self.slices,
                ptr(&self.size_n_list) as *const i32,
                ptr(&scratch.c_ptrs).cast_const(),
                ptr(&self.n_stride_list) as *const i32,
                ptr(&self.had_src_list) as *const i32,
                3,
                gpu.raw_stream(),
            )?
        };
        Ok(())
    }

    fn prefill_args(&self, scratch: &AttnPrefillScratch, rows: i32) -> PrefillChunk {
        let staging = needs_staging(rows);
        PrefillChunk {
            qg: scratch.qg.ptr().as_u64(),
            k: scratch.k.ptr().as_u64(),
            v: scratch.v.ptr().as_u64(),
            q: scratch.q.ptr().as_u64(),
            g: scratch.g.ptr().as_u64(),
            o: scratch.o.ptr().as_u64(),
            k_cache: self.k_pages.ptr().as_u64(),
            v_cache: self.v_pages.ptr().as_u64(),
            k_scales: self.k_scales.ptr().as_u64(),
            v_scales: self.v_scales.ptr().as_u64(),
            block_table: self.block_table.ptr().as_u64(),
            cache_seqlens: self.seqlens_addr(),
            inv_freq: self.inv_freq.ptr().as_u64(),
            q_norm: self.q_norm.ptr().as_u64(),
            k_norm: self.k_norm.ptr().as_u64(),
            h32: self.h32.ptr().as_u64(),
            scratch_k: match &scratch.scratch_k {
                Some(buf) if staging => buf.ptr().as_u64(),
                _ => 0,
            },
            scratch_v: match &scratch.scratch_v {
                Some(buf) if staging => buf.ptr().as_u64(),
                _ => 0,
            },
            identity_table: if staging {
                scratch.identity.ptr().as_u64()
            } else {
                0
            },
            global_scratch: scratch.global_scratch.ptr().as_u64(),
            profile_scratch: scratch.profile_scratch.ptr().as_u64(),
        }
    }
}

fn qkv_layout() -> Result<(i32, i32, i32, i32)> {
    let q_n = Q_HEADS * HEAD_DIM * 2;
    let kv_n = KV_HEADS * HEAD_DIM;
    let width = gcd(q_n, kv_n);
    if width < 256 || width % 128 != 0 || q_n % width != 0 || kv_n % width != 0 {
        return Err(Error::Weights(format!(
            "prefill slice width {width} does not tile q/k/v"
        )));
    }
    let slices = (q_n / width) + (kv_n / width) * 2;
    Ok((width, slices, q_n, kv_n))
}

fn slice_out_ptrs(qg: u64, k: u64, v: u64, q_n: i32, kv_n: i32, width: i32) -> Result<Vec<u64>> {
    let mut ptrs = Vec::new();
    push_slice_ptrs(&mut ptrs, qg, q_n, width)?;
    push_slice_ptrs(&mut ptrs, k, kv_n, width)?;
    push_slice_ptrs(&mut ptrs, v, kv_n, width)?;
    Ok(ptrs)
}

fn push_slice_ptrs(ptrs: &mut Vec<u64>, base: u64, n: i32, width: i32) -> Result<()> {
    if width < 1 {
        return Err(Error::Weights("prefill slice width is 0".into()));
    }
    let mut n0 = 0i32;
    while n0 < n {
        ptrs.push(base + u64::from(u32::try_from(n0).unwrap_or(0)) * 2);
        n0 = n0
            .checked_add(width)
            .ok_or_else(|| Error::Weights("prefill slice column overflows".into()))?;
    }
    Ok(())
}

fn fp16_bytes(rows: u32, cols: i32) -> Result<usize> {
    let cols = u64::try_from(cols).map_err(|_| Error::Input("negative prefill width".into()))?;
    let bytes = u64::from(rows)
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| Error::Input("attention prefill buffer overflows".into()))?;
    usize::try_from(bytes).map_err(|_| Error::Input("attention prefill buffer overflows".into()))
}

fn mgemm_had_bytes(rows: u32, hidden: i32) -> Result<usize> {
    fp16_bytes(rows, hidden)?
        .checked_mul(3)
        .ok_or_else(|| Error::Input("qkv hadamard scratch overflows".into()))
}

fn u32_len(bytes: u32) -> Result<usize> {
    usize::try_from(bytes).map_err(|_| Error::Input("scratch size does not fit usize".into()))
}

fn need_bytes(buf: &ByteBuf, bytes: usize, what: &str) -> Result<()> {
    if buf.num_bytes() < bytes {
        return Err(Error::Input(format!(
            "{what} has {} bytes, needs {bytes}",
            buf.num_bytes()
        )));
    }
    Ok(())
}

/// Host copy of this layer's cq3 pages and block table.
///
/// `qk` / `qv` are int32 code pages, `sk` / `sv` are fp16 scales, both in
/// physical page order. `q_cols` is the int32 width of one token
/// (`token_dim / 32 * bits`). `s_cols` is the fp16 scale width
/// (`token_dim / 32`). The model gathers used rows in block-table order.
pub(crate) struct CachePages {
    pub qk: Vec<u8>,
    pub qv: Vec<u8>,
    pub sk: Vec<u8>,
    pub sv: Vec<u8>,
    pub block_table: Vec<i32>,
    pub q_cols: usize,
    pub s_cols: usize,
    pub page_tokens: usize,
}

impl AttnLayer {
    /// Copy the cache pages and the block table. The allocations stay on device.
    ///
    /// This is how the prefill reference reads used rows. Decode does not call it.
    pub(crate) fn copy_cache(&self, gpu: &Gpu) -> Result<CachePages> {
        let groups = usize::try_from(self.token_dim).unwrap_or(0) / 32;
        Ok(CachePages {
            qk: download_bytes(gpu, &self.k_pages)?,
            qv: download_bytes(gpu, &self.v_pages)?,
            sk: download_bytes(gpu, &self.k_scales)?,
            sv: download_bytes(gpu, &self.v_scales)?,
            block_table: download_i32(gpu, &self.block_table)?,
            q_cols: groups * usize::try_from(CACHE_BITS).unwrap_or(0),
            s_cols: groups,
            page_tokens: usize::try_from(PAGE).unwrap_or(0),
        })
    }
}

fn download_bytes(gpu: &Gpu, buf: &ByteBuf) -> Result<Vec<u8>> {
    let mut out = vec![0u8; buf.num_bytes()];
    buf.copy_to_host(&gpu.stream, &mut out)?;
    Ok(out)
}

fn download_i32(gpu: &Gpu, buf: &ByteBuf) -> Result<Vec<i32>> {
    let bytes = download_bytes(gpu, buf)?;
    if !bytes.len().is_multiple_of(4) {
        return Err(Error::Input(
            "block table byte length is not a multiple of 4".into(),
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|word| i32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bin(name: &str) -> Vec<u8> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../testdata/kernels/attn")
            .join(name);
        std::fs::read(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()))
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn inv_freq_matches_the_cuda_table() {
        let got = inv_freq(10_000_000.0, 64);
        assert_eq!(f32_bytes(&got), bin("inv_freq.bin"));
        assert_eq!(got[28].to_bits(), 0x3549_4c56);
    }

    #[test]
    fn h32_matches_the_sylvester_table() {
        assert_eq!(h32_bytes().as_slice(), bin("h32.bin"));
    }

    #[test]
    fn f16_known_values() {
        assert_eq!(f32_to_f16_rne(0.0), 0);
        assert_eq!(f32_to_f16_rne(-0.0), 0x8000);
        assert_eq!(f32_to_f16_rne(1.0), 0x3c00);
        assert_eq!(f32_to_f16_rne(-2.0), 0xc000);
        assert_eq!(f32_to_f16_rne(2f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_f16_rne(2f32.powi(-25)), 0);
        assert_eq!(f32_to_f16_rne(2f32.powi(-14)), 0x0400);
        assert_eq!(f32_to_f16_rne(1.0 / 32f32.sqrt()), 0x31a8);
        assert_eq!(f32_to_f16_rne(-1.0 / 32f32.sqrt()), 0xb1a8);
    }

    #[test]
    fn rope_settings_follow_the_fork_lookup() {
        let json = r#"{"text_config":{"partial_rotary_factor":0.25,"rms_norm_eps":1e-06,"rope_parameters":{"rope_theta":10000000,"rope_type":"default","partial_rotary_factor":0.5}}}"#;
        let rope = read_rope(json.as_bytes(), 256).unwrap();
        assert_eq!(rope.theta.to_bits(), 10_000_000f32.to_bits());
        assert_eq!(rope.partial_head_dim, 64);
        assert_eq!(rope.norm_eps.to_bits(), 1e-6f32.to_bits());
        assert_eq!(f32_bytes(&rope.inv_freq), bin("inv_freq.bin"));
    }

    #[test]
    fn non_default_rope_is_rejected() {
        let json = r#"{"text_config":{"rms_norm_eps":1e-6,"rope_parameters":{"rope_type":"yarn","rope_theta":10000}}}"#;
        let err = read_rope(json.as_bytes(), 256).unwrap_err();
        assert!(err.to_string().contains("yarn"), "{err}");
    }

    #[test]
    fn the_27b_config_matches_the_inv_freq_bin() {
        let dir = std::env::var("BERTH_27B_DIR")
            .unwrap_or_else(|_| "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3".to_string());
        let path = Path::new(&dir).join("config.json");
        if !path.is_file() {
            eprintln!("skipping 27b config: {}", path.display());
            return;
        }
        let bytes = std::fs::read(&path).unwrap();
        let rope = read_rope(&bytes, 256).unwrap();
        assert_eq!(rope.theta.to_bits(), 10_000_000f32.to_bits());
        assert_eq!(rope.partial_head_dim, 64);
        assert_eq!(rope.norm_eps.to_bits(), 1e-6f32.to_bits());
        assert_eq!(f32_bytes(&rope.inv_freq), bin("inv_freq.bin"));
    }
}

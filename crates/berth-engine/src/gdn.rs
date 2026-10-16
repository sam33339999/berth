//! Gated DeltaNet mixer (48 of the 27B's 64 layers).
//!
//! Decode ([`GdnLayer::forward`]) reproduces `BC_GatedDeltaNetSplit::run_bszN`
//! for one token, history off: the sliced qkv+z `exl3_mgemm`, then the five
//! GDN kernels, then `out_proj`.
//!
//! Prefill ([`GdnLayer::prefill`]) is the torch path. `seq_len > 1` misses the
//! BC graph (`gated_delta_net.py` 1034–1048) even when `seq_len <= 16`.

use std::ffi::c_void;
use std::path::PathBuf;

use berth_cuda::{ByteBuf, DeviceBuffer, DeviceCopy};
use berth_exl3::{Bitrate, Checkpoint, DType, HadamardScale, LinearExl3};
use berth_kernels::{Codebook, OutDtype};
use berth_plan::{LayerKind, ModelShape};

use crate::device::{self, ptr};
use crate::{Error, Exl3Linear, Gpu, Result};

/// `GatedDeltaNet.beta_scale` default. Qwen3.5 does not override it
/// (`gated_delta_net.py`).
const BETA_SCALE: f32 = 1.0;

/// `project_qkvz_sliced` runs only when `bsz * seqlen <= 32`
/// (`gated_delta_net.py`). Longer chunks use the two linears.
const SLICED_QKVZ_MAX: i32 = 32;

/// `AUTO_RECONSTRUCT_THRESHOLD`. Rows above this reconstruct the fp16 weight.
const RECONSTRUCT_ABOVE: i32 = 144;

/// Pinned fla cubins reject a longer chunk (`gdn_prefill`).
const PREFILL_MAX_ROWS: u32 = 4096;

/// 27B `text_config.rms_norm_eps`. The fork passes `config.rms_norm_eps` into
/// `GatedRMSNorm`. [`berth_exl3::ModelConfig`] does not store the field, so
/// the constant lives here. The checkpoint's value is `1e-06`.
const RMS_NORM_EPS: f32 = 1e-6;

/// One GDN layer: its weights, its conv and recurrent state, and scratch.
pub struct GdnLayer {
    dims: Dims,
    qkvz: Qkvz,
    /// fp16 `[2 * num_v_heads, hidden]`. Rows `0..H` are `in_proj_b`, the rest
    /// `in_proj_a` (`gated_delta_net.py` `cat([b, a], dim=-1).T`). Decode only.
    ba_weight: ByteBuf,
    /// fp16 `[hidden, num_v_heads]`. Checkpoint `in_proj_b.weight` is
    /// `[num_v_heads, hidden]`; `load_fp16` transposes it (`transposed_load`,
    /// default true) into the `x @ W` layout `ext.hgemm` multiplies.
    b_hgemm: ByteBuf,
    /// fp16 `[hidden, num_v_heads]`, the same transpose of `in_proj_a.weight`.
    a_hgemm: ByteBuf,
    /// fp16 `[2 * num_v_heads]`, or absent. The 27B checkpoint has neither bias.
    ba_bias: Option<ByteBuf>,
    /// bf16 `[num_v_heads]`, `linear_attn.dt_bias`.
    dt_bias: ByteBuf,
    /// bf16 `[num_v_heads]`, `linear_attn.A_log`.
    a_log: ByteBuf,
    /// bf16 `[fdim, conv_k]`. The checkpoint stores `[fdim, 1, conv_k]`; a
    /// squeeze of that axis does not move bytes.
    conv_weight: ByteBuf,
    /// bf16 `[fdim]`, or absent.
    conv_bias: Option<ByteBuf>,
    /// bf16 `[v_head_dim]`, shared by every V head (`w_groups = 1`).
    norm_weight: ByteBuf,
    out_proj: Exl3Linear,
    /// bf16 `[1, fdim, conv_k + max_history]`. `GDNLayerState.conv_state`
    /// (`gated_delta_net.py`). Load uses `max_history = 0`, so this is
    /// `[1, fdim, conv_k]` and the kernel's `state_size` is `conv_k`.
    conv_state: ByteBuf,
    /// f32 `[1, max_history + 1, num_v_heads, k_head_dim, v_head_dim]`.
    /// Load uses `max_history = 0`, so the leading history axis is 1 and
    /// `history_stride` is 1. K-major inside a head. History-off launches
    /// read and write index 0 only; the pointer stays the base of this buffer.
    recurrent_state: ByteBuf,
    /// Extra conv taps and recurrent history slots past the live state.
    /// Zero at load. [`Self::set_max_history`] reallocates both buffers.
    max_history: u32,
    /// History index written by the last [`Self::forward_history`].
    ///
    /// `seqlen - 1` after a history launch (`advance_recurrent_states`).
    /// Zero after [`Self::rewind`], [`Self::reset_state`], and a successful
    /// [`Self::set_max_history`]. Decode and prefill do not change it.
    last_history: u32,
    scratch: Scratch,
}

/// Widths for one token. Every product is checked into `i32` at load.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Dims {
    hidden: i32,
    /// `2 * num_k_heads * k_head_dim + num_v_heads * v_head_dim`.
    fdim: i32,
    /// `num_v_heads * v_head_dim`. `in_proj_z` and `out_proj` K.
    v_dim: i32,
    num_k_heads: i32,
    num_v_heads: i32,
    k_head_dim: i32,
    v_head_dim: i32,
    conv_k: i32,
    /// `2 * num_v_heads`.
    ba_n: i32,
}

/// One EXL3 source of the qkv+z bundle. The device buffers stay alive for the
/// slice pointer tables, which alias them.
struct Projection {
    n: i32,
    k_bits: i32,
    trellis: ByteBuf,
    suh: ByteBuf,
    svh: ByteBuf,
}

/// Sliced qkv+z bundle. Pointer tables are device arrays built once in `load`,
/// the same tables `SlicedMultiLinear` and `set_qkvz_bundle` keep.
struct Qkvz {
    qkv: Projection,
    /// `in_proj_z` weights. The slice tables alias this allocation. Prefill
    /// rows above 32 also launch this trellis on its own.
    z: Projection,
    b_list: DeviceBuffer<u64>,
    suh_list: DeviceBuffer<u64>,
    svh_list: DeviceBuffer<u64>,
    c_ptrs: DeviceBuffer<u64>,
    size_n_list: DeviceBuffer<i32>,
    n_stride_list: DeviceBuffer<i32>,
    had_src_list: DeviceBuffer<i32>,
    width: i32,
    slices: i32,
}

/// Per-token scratch. Shapes match `_bc_configure_slot` for bsz = seqlen = 1.
struct Scratch {
    /// fp32 `[fdim]`. `in_proj_qkv` output, `[B, S, F]`.
    qkv: ByteBuf,
    /// fp32 `[v_dim]`. `in_proj_z` output, also the RMSNorm gate `[Nv, Hv]`.
    z: ByteBuf,
    /// fp32 `[ba_n]`.
    ba: ByteBuf,
    /// bf16 `[num_v_heads]`.
    beta: ByteBuf,
    /// fp32 `[num_v_heads]`.
    g: ByteBuf,
    /// bf16 `[fdim]`. Fused-op output `[B, F, S]` and the conv input.
    mixed_qkv: ByteBuf,
    /// bf16 `[fdim]`. Conv output `[B, S, F]`, the recurrent input.
    conv_out: ByteBuf,
    /// bf16 `[v_dim]`. `[B, S, Nv, Hv]`.
    core_attn_out: ByteBuf,
    /// fp16 `[v_dim]`. Gated RMSNorm output, `out_proj` input.
    core_attn_out_f: ByteBuf,
    /// fp16 `[2, hidden]`. One Hadamard slab per qkv/z source.
    a_had: ByteBuf,
    /// fp32 `[slices, width]`. Carrier; the kernel writes `c_ptrs`.
    carrier: ByteBuf,
}

impl GdnLayer {
    /// Load layer `layer`'s `linear_attn.*` weights and allocate zeroed state.
    pub fn load(gpu: &Gpu, ckpt: &Checkpoint, shape: &ModelShape, layer: u32) -> Result<Self> {
        let dims = dims_for(shape, layer)?;
        let base = format!("model.language_model.layers.{layer}.linear_attn");
        let qkv = load_projection(
            gpu,
            ckpt,
            &format!("{base}.in_proj_qkv"),
            dims.hidden,
            dims.fdim,
        )?;
        let z = load_projection(
            gpu,
            ckpt,
            &format!("{base}.in_proj_z"),
            dims.hidden,
            dims.v_dim,
        )?;
        if qkv.k_bits != z.k_bits {
            return Err(Error::Weights(format!(
                "{base}: in_proj_qkv K is {}, in_proj_z K is {}",
                qkv.k_bits, z.k_bits
            )));
        }
        let (width, slices) = slice_plan(qkv.n, z.n)?;
        let scratch = alloc_scratch(gpu, &dims, slices, width)?;
        let qkvz = pack_qkvz(gpu, qkv, z, &scratch.qkv, &scratch.z, width, slices)?;

        let ba_weight = cat_ba(gpu, ckpt, &base, dims.num_v_heads, dims.hidden)?;
        let b_hgemm = hgemm_weight(
            gpu,
            ckpt,
            &format!("{base}.in_proj_b.weight"),
            dims.num_v_heads,
            dims.hidden,
        )?;
        let a_hgemm = hgemm_weight(
            gpu,
            ckpt,
            &format!("{base}.in_proj_a.weight"),
            dims.num_v_heads,
            dims.hidden,
        )?;
        let ba_bias = cat_ba_bias(gpu, ckpt, &base, dims.num_v_heads)?;
        let dt_bias = upload_exact(
            gpu,
            ckpt,
            &format!("{base}.dt_bias"),
            DType::Bf16,
            &[u64_dim(dims.num_v_heads)],
        )?;
        let a_log = upload_exact(
            gpu,
            ckpt,
            &format!("{base}.A_log"),
            DType::Bf16,
            &[u64_dim(dims.num_v_heads)],
        )?;
        let conv_weight = upload_exact(
            gpu,
            ckpt,
            &format!("{base}.conv1d.weight"),
            DType::Bf16,
            &[u64_dim(dims.fdim), 1, u64_dim(dims.conv_k)],
        )?;
        let conv_bias = optional_conv_bias(gpu, ckpt, &base, dims.fdim)?;
        let norm_weight = upload_exact(
            gpu,
            ckpt,
            &format!("{base}.norm.weight"),
            DType::Bf16,
            &[u64_dim(dims.v_head_dim)],
        )?;

        let out_prefix = format!("{base}.out_proj");
        let out_linear = device::find_linear(ckpt, &out_prefix)?;
        if out_linear.bias.is_some() {
            return Err(Error::Weights(format!(
                "{out_prefix}: bias is present; this path does not add it"
            )));
        }
        let out_proj = Exl3Linear::load(gpu, ckpt, &out_prefix)?;
        if out_proj.size_k != dims.v_dim || out_proj.size_n != dims.hidden {
            return Err(Error::Weights(format!(
                "{out_prefix}: expected K {} N {}, got K {} N {}",
                dims.v_dim, dims.hidden, out_proj.size_k, out_proj.size_n
            )));
        }

        let conv_elems = mul_i32(dims.fdim, dims.conv_k, "conv state")?;
        let recurrent_elems = mul_i32(
            mul_i32(dims.num_v_heads, dims.k_head_dim, "recurrent state")?,
            dims.v_head_dim,
            "recurrent state",
        )?;
        Ok(Self {
            dims,
            qkvz,
            ba_weight,
            b_hgemm,
            a_hgemm,
            ba_bias,
            dt_bias,
            a_log,
            conv_weight,
            conv_bias,
            norm_weight,
            out_proj,
            conv_state: gpu.zeroed(nbytes(conv_elems, 2)?)?,
            recurrent_state: gpu.zeroed(nbytes(recurrent_elems, 4)?)?,
            max_history: 0,
            last_history: 0,
            scratch,
        })
    }

    /// `output = linear_attn(input)` for one token, advancing the state.
    ///
    /// `input` is fp16 `[hidden]` (the input_layernorm output); `output` is
    /// fp32 `[hidden]`.
    pub fn forward(&mut self, gpu: &Gpu, input: &ByteBuf, output: &mut ByteBuf) -> Result<()> {
        let hidden = self.dims.hidden as usize;
        if input.num_bytes() < hidden * 2 || output.num_bytes() < hidden * 4 {
            return Err(Error::Input(format!(
                "gdn: expected fp16 [{hidden}] in and fp32 [{hidden}] out"
            )));
        }
        // Same order as `BC_GatedDeltaNetSplit::run_bszN_gr` (gated_delta_net.cpp:234-323)
        // for the non-KDA sliced bundle, bsz = seqlen = 1, history off.
        self.launch(input, gpu.raw_stream())?;
        self.out_proj
            .forward(gpu, &self.scratch.core_attn_out_f, output, OutDtype::F32)
    }

    /// Kernels up to the gated norm. `out_proj` follows in [`Self::forward`].
    ///
    /// The conv and recurrent kernels write `conv_state` and `recurrent_state`
    /// through these pointers. That is the only mutation; the Rust buffers
    /// stay the allocations from `load`.
    fn launch(&self, input: &ByteBuf, stream: *mut c_void) -> Result<()> {
        let d = &self.dims;
        let s = &self.scratch;
        let q = &self.qkvz;
        // `max_history == 0` is `(1, conv_k)`, the literals this launch used
        // before the field existed.
        let (history_stride, state_size) = gdn_state_launch(d.conv_k, self.max_history)?;
        // SAFETY: every pointer is a device allocation owned by this layer or
        // by the caller, sized for the 27B decode shapes checked at load, and
        // live until `stream` finishes. `input` is fp16 `[hidden]`. Optional
        // biases are null when the checkpoint has none. Slots are null, so
        // slot 0 is batch row 0. Launches are ordered on `stream`.
        unsafe {
            let _shape = berth_kernels::ops::exl3_mgemm(
                ptr(input),
                raw(&q.b_list),
                ptr(&s.carrier),
                raw(&q.suh_list),
                ptr(&s.a_had),
                raw(&q.svh_list),
                1,
                d.hidden,
                q.width,
                q.qkv.k_bits,
                Codebook::Mul1,
                OutDtype::F32,
                1,
                q.slices,
                q.size_n_list.ptr().as_u64() as *const i32,
                raw(&q.c_ptrs),
                q.n_stride_list.ptr().as_u64() as *const i32,
                q.had_src_list.ptr().as_u64() as *const i32,
                2,
                stream,
            )?;
            berth_kernels::gdn::gdn_ba_gemv(
                ptr(input),
                ptr(&self.ba_weight),
                self.ba_bias
                    .as_ref()
                    .map(ptr)
                    .unwrap_or(std::ptr::null_mut()),
                ptr(&s.ba),
                1,
                d.hidden,
                d.ba_n,
                stream,
            )?;
            berth_kernels::gdn::gated_delta_net_fused_op_3(
                ptr(&s.qkv),
                ptr(&s.ba),
                ptr(&self.dt_bias),
                ptr(&self.a_log),
                ptr(&s.mixed_qkv),
                ptr(&s.beta),
                ptr(&s.g),
                1,
                1,
                d.fdim,
                d.num_v_heads,
                BETA_SCALE,
                stream,
            )?;
            berth_kernels::gdn::conv1d_update(
                ptr(&s.mixed_qkv),
                ptr(&self.conv_state),
                std::ptr::null(),
                ptr(&self.conv_weight),
                self.conv_bias
                    .as_ref()
                    .map(ptr)
                    .unwrap_or(std::ptr::null_mut()),
                ptr(&s.conv_out),
                1,
                d.fdim,
                1,
                state_size,
                d.conv_k,
                stream,
            )?;
            berth_kernels::gdn::recurrent_gated_delta_rule(
                ptr(&s.conv_out),
                ptr(&s.g),
                ptr(&s.beta),
                ptr(&self.recurrent_state),
                ptr(&s.core_attn_out),
                std::ptr::null(),
                1,
                1,
                d.num_k_heads,
                d.num_v_heads,
                d.k_head_dim,
                d.v_head_dim,
                history_stride,
                stream,
            )?;
            berth_kernels::gdn::gated_rms_norm(
                ptr(&s.core_attn_out),
                ptr(&self.norm_weight),
                ptr(&s.core_attn_out_f),
                ptr(&s.z),
                RMS_NORM_EPS,
                0.0,
                d.num_v_heads,
                d.v_head_dim,
                1,
                0,
                0,
                stream,
            )?;
        }
        Ok(())
    }

    /// Zero conv and recurrent state in the current allocations.
    ///
    /// `GDNLayerState.clear` (`gated_delta_net.py`) does `zero_()` on the
    /// slot. `Cache.get_new_state` asks for that when a job has no stashed
    /// checkpoint. Nothing is reallocated. `berth-cuda` has no memset on an
    /// existing buffer, so the zeros are copied into the same device memory,
    /// including the history tail after [`Self::set_max_history`]. The caller
    /// synchronizes `gpu.stream` first if a forward is still queued.
    pub fn reset_state(&mut self, gpu: &Gpu) -> Result<()> {
        // Clear the cursor first. A failed zero then still refuses a rewind
        // that would read a half-cleared history slot.
        self.last_history = 0;
        zero_in_place(gpu, &mut self.conv_state)?;
        zero_in_place(gpu, &mut self.recurrent_state)?;
        Ok(())
    }

    /// Reallocate conv and recurrent state for `n` history tokens, and zero them.
    ///
    /// Shapes are `GDNLayerState` (`gated_delta_net.py`): conv bf16
    /// `[1, fdim, conv_k + n]`, recurrent fp32
    /// `[1, n + 1, num_v_heads, k_head_dim, v_head_dim]`. `n == 0` restores
    /// the load-time sizes. The old allocation is dropped; synchronize
    /// `gpu.stream` first if a forward is still queued. A `state_size` past
    /// 19 is stored here and rejected when prefill launches.
    pub fn set_max_history(&mut self, gpu: &Gpu, n: u32) -> Result<()> {
        let (slots, state_size) = gdn_state_launch(self.dims.conv_k, n)?;
        let conv_elems = mul_i32(self.dims.fdim, state_size, "conv state")?;
        let per_slot = mul_i32(
            mul_i32(
                self.dims.num_v_heads,
                self.dims.k_head_dim,
                "recurrent state",
            )?,
            self.dims.v_head_dim,
            "recurrent state",
        )?;
        let recurrent_elems = mul_i32(slots, per_slot, "recurrent state")?;
        self.conv_state = gpu.zeroed(nbytes(conv_elems, 2)?)?;
        self.recurrent_state = gpu.zeroed(nbytes(recurrent_elems, 4)?)?;
        self.max_history = n;
        self.last_history = 0;
        Ok(())
    }
}

/// `(history_stride, state_size)` for a GDN launch.
///
/// `history_stride = max_history + 1` and `state_size = conv_k + max_history`,
/// the axes of `GDNLayerState`. `max_history == 0` returns `(1, conv_k)`,
/// which is what decode and prefill passed before this field existed (`conv_k`
/// is 4 on the 27B).
pub fn gdn_state_launch(conv_k: i32, max_history: u32) -> Result<(i32, i32)> {
    if conv_k < 1 {
        return Err(Error::Input(format!(
            "gdn conv kernel {conv_k} must be positive"
        )));
    }
    let stride_u = max_history
        .checked_add(1)
        .ok_or_else(|| Error::Input("gdn max_history + 1 overflows".into()))?;
    let history_stride = i32::try_from(stride_u)
        .map_err(|_| Error::Input("gdn history_stride does not fit in i32".into()))?;
    let extra = i32::try_from(max_history)
        .map_err(|_| Error::Input("gdn max_history does not fit in i32".into()))?;
    let state_size = conv_k
        .checked_add(extra)
        .ok_or_else(|| Error::Input("gdn state_size overflows".into()))?;
    Ok((history_stride, state_size))
}

fn zero_in_place(gpu: &Gpu, buf: &mut ByteBuf) -> Result<()> {
    let zeros = vec![0u8; buf.num_bytes()];
    buf.copy_from_host(&gpu.stream, &zeros)?;
    Ok(())
}

fn dims_for(shape: &ModelShape, layer: u32) -> Result<Dims> {
    let Some(kind) = shape.layer_kinds.get(layer as usize) else {
        return Err(Error::Weights(format!(
            "layer {layer} is past {} layers",
            shape.layer_kinds.len()
        )));
    };
    if *kind != LayerKind::LinearAttention {
        return Err(Error::Weights(format!(
            "layer {layer} is {kind}, not linear_attention"
        )));
    }
    dims_from_shape(shape)
}

fn dims_from_shape(shape: &ModelShape) -> Result<Dims> {
    let g = &shape.gdn;
    let hidden = as_i32(shape.hidden_size, "hidden_size")?;
    let num_k_heads = as_i32(g.num_k_heads, "num_k_heads")?;
    let num_v_heads = as_i32(g.num_v_heads, "num_v_heads")?;
    let k_head_dim = as_i32(g.key_head_dim, "key_head_dim")?;
    let v_head_dim = as_i32(g.value_head_dim, "value_head_dim")?;
    let conv_k = as_i32(g.conv_kernel_dim, "conv_kernel_dim")?;
    let qk = mul_i32(
        mul_i32(2, num_k_heads, "qkv width")?,
        k_head_dim,
        "qkv width",
    )?;
    let v_dim = mul_i32(num_v_heads, v_head_dim, "z width")?;
    let fdim = qk
        .checked_add(v_dim)
        .ok_or_else(|| Error::Weights("qkv width overflows".into()))?;
    let ba_n = mul_i32(2, num_v_heads, "b/a width")?;
    Ok(Dims {
        hidden,
        fdim,
        v_dim,
        num_k_heads,
        num_v_heads,
        k_head_dim,
        v_head_dim,
        conv_k,
        ba_n,
    })
}

fn as_i32(value: u32, what: &str) -> Result<i32> {
    if value == 0 {
        return Err(Error::Weights(format!("{what} is 0")));
    }
    i32::try_from(value).map_err(|_| Error::Weights(format!("{what} does not fit in i32")))
}

fn mul_i32(a: i32, b: i32, what: &str) -> Result<i32> {
    a.checked_mul(b)
        .ok_or_else(|| Error::Weights(format!("{what} overflows")))
}

fn nbytes(elems: i32, elem_size: usize) -> Result<usize> {
    let elems = usize::try_from(elems).map_err(|_| Error::Weights("negative size".into()))?;
    elems
        .checked_mul(elem_size)
        .ok_or_else(|| Error::Weights("allocation overflows".into()))
}

fn u64_dim(value: i32) -> u64 {
    u64::try_from(value).expect("dimension was checked positive")
}

fn load_projection(
    gpu: &Gpu,
    ckpt: &Checkpoint,
    prefix: &str,
    size_k: i32,
    size_n: i32,
) -> Result<Projection> {
    let linear = device::find_linear(ckpt, prefix)?;
    if linear.bias.is_some() {
        return Err(Error::Weights(format!(
            "{prefix}: sliced qkv+z refuses a bias"
        )));
    }
    if linear.codebook != berth_exl3::Codebook::Mul1 {
        return Err(Error::Weights(format!(
            "{prefix}: qkv/z bundle requires the mul1 codebook"
        )));
    }
    let Bitrate::Int(k_bits) = linear.k else {
        return Err(Error::Weights(format!("{prefix}: half-integer rate")));
    };
    if k_bits != 4 {
        return Err(Error::Weights(format!(
            "{prefix}: mgemm is linked for K = 4, got {k_bits}"
        )));
    }
    expect_trellis(linear, prefix, size_k, size_n, i32::from(k_bits))?;
    let (suh, svh) = match (&linear.input_scale, &linear.output_scale) {
        (HadamardScale::Unpacked(suh), HadamardScale::Unpacked(svh)) => (suh, svh),
        _ => {
            return Err(Error::Weights(format!(
                "{prefix}: packed su / sv signs are not handled"
            )))
        }
    };
    expect_vector(suh, DType::F16, size_k)?;
    expect_vector(svh, DType::F16, size_n)?;
    Ok(Projection {
        n: size_n,
        k_bits: i32::from(k_bits),
        trellis: upload_info(gpu, &linear.trellis)?,
        suh: upload_info(gpu, suh)?,
        svh: upload_info(gpu, svh)?,
    })
}

fn expect_trellis(
    linear: &LinearExl3,
    prefix: &str,
    size_k: i32,
    size_n: i32,
    k_bits: i32,
) -> Result<()> {
    let shape = &linear.trellis.shape;
    if shape.len() != 3 {
        return Err(Error::Weights(format!(
            "{prefix}.trellis: expected rank 3, got {}",
            shape.len()
        )));
    }
    let got_k = shape[0].checked_mul(16);
    let got_n = shape[1].checked_mul(16);
    let words = i32::try_from(shape[2]).ok();
    let matches = got_k == Some(u64_dim(size_k))
        && got_n == Some(u64_dim(size_n))
        && words == Some(k_bits * 16);
    if !matches {
        return Err(Error::Weights(format!(
            "{prefix}.trellis: expected [{}, {}, {}], got {shape:?}",
            u64_dim(size_k) / 16,
            u64_dim(size_n) / 16,
            k_bits * 16
        )));
    }
    Ok(())
}

fn expect_vector(info: &berth_exl3::TensorInfo, dtype: DType, n: i32) -> Result<()> {
    if info.dtype != dtype || info.shape != vec![u64_dim(n)] {
        return Err(Error::Weights(format!(
            "{}: expected {dtype} [{}], got {} {:?}",
            info.name,
            u64_dim(n),
            info.dtype,
            info.shape
        )));
    }
    Ok(())
}

fn slice_plan(n_qkv: i32, n_z: i32) -> Result<(i32, i32)> {
    let width = gcd(n_qkv, n_z);
    // `SlicedMultiLinear`: the gcd must be a whole number of 128-column blocks
    // and at least 256, or the fork refuses the bundle.
    if width < 256 || width % 128 != 0 {
        return Err(Error::Weights(format!(
            "qkv+z slice width {width} is not a usable mgemm width"
        )));
    }
    let slices = n_qkv / width + n_z / width;
    Ok((width, slices))
}

fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a
}

fn pack_qkvz(
    gpu: &Gpu,
    qkv: Projection,
    z: Projection,
    qkv_out: &ByteBuf,
    z_out: &ByteBuf,
    width: i32,
    slices: i32,
) -> Result<Qkvz> {
    let mut b_host = Vec::with_capacity(slices as usize);
    let mut svh_host = Vec::with_capacity(slices as usize);
    let mut suh_host = Vec::with_capacity(2);
    let mut c_host = Vec::with_capacity(slices as usize);
    let mut size_n = Vec::with_capacity(slices as usize);
    let mut strides = Vec::with_capacity(slices as usize);
    let mut had_src = Vec::with_capacity(slices as usize);
    for (index, (proj, out)) in [&qkv, &z].into_iter().zip([qkv_out, z_out]).enumerate() {
        suh_host.push(proj.suh.ptr().as_u64());
        let mut n0 = 0;
        while n0 < proj.n {
            // `SlicedMultiLinear`: trellis + (n0 // 16) * 16 * K * sizeof(int16),
            // svh + n0 * sizeof(fp16), output + n0 * sizeof(fp32).
            let trellis_off = (n0 / 16) * 16 * proj.k_bits * 2;
            b_host.push(proj.trellis.ptr().as_u64() + u64::try_from(trellis_off).unwrap());
            svh_host.push(proj.svh.ptr().as_u64() + u64::try_from(n0).unwrap() * 2);
            c_host.push(out.ptr().as_u64() + u64::try_from(n0).unwrap() * 4);
            size_n.push(width);
            strides.push(proj.n);
            had_src.push(i32::try_from(index).unwrap());
            n0 += width;
        }
    }
    if had_src.len() != slices as usize || suh_host.len() != 2 {
        return Err(Error::Weights(format!(
            "qkv+z slice count {} does not match the plan {slices}",
            had_src.len()
        )));
    }
    Ok(Qkvz {
        qkv,
        z,
        b_list: DeviceBuffer::from_host(&gpu.stream, &b_host)?,
        suh_list: DeviceBuffer::from_host(&gpu.stream, &suh_host)?,
        svh_list: DeviceBuffer::from_host(&gpu.stream, &svh_host)?,
        c_ptrs: DeviceBuffer::from_host(&gpu.stream, &c_host)?,
        size_n_list: DeviceBuffer::from_host(&gpu.stream, &size_n)?,
        n_stride_list: DeviceBuffer::from_host(&gpu.stream, &strides)?,
        had_src_list: DeviceBuffer::from_host(&gpu.stream, &had_src)?,
        width,
        slices,
    })
}

fn alloc_scratch(gpu: &Gpu, dims: &Dims, slices: i32, width: i32) -> Result<Scratch> {
    let had = mul_i32(2, dims.hidden, "a_had")?;
    let carrier = mul_i32(slices, width, "mgemm carrier")?;
    Ok(Scratch {
        qkv: gpu.zeroed(nbytes(dims.fdim, 4)?)?,
        z: gpu.zeroed(nbytes(dims.v_dim, 4)?)?,
        ba: gpu.zeroed(nbytes(dims.ba_n, 4)?)?,
        beta: gpu.zeroed(nbytes(dims.num_v_heads, 2)?)?,
        g: gpu.zeroed(nbytes(dims.num_v_heads, 4)?)?,
        mixed_qkv: gpu.zeroed(nbytes(dims.fdim, 2)?)?,
        conv_out: gpu.zeroed(nbytes(dims.fdim, 2)?)?,
        core_attn_out: gpu.zeroed(nbytes(dims.v_dim, 2)?)?,
        core_attn_out_f: gpu.zeroed(nbytes(dims.v_dim, 2)?)?,
        a_had: gpu.zeroed(nbytes(had, 2)?)?,
        carrier: gpu.zeroed(nbytes(carrier, 4)?)?,
    })
}

fn cat_ba(gpu: &Gpu, ckpt: &Checkpoint, base: &str, num_v: i32, hidden: i32) -> Result<ByteBuf> {
    let shape = [u64_dim(num_v), u64_dim(hidden)];
    let b = read_exact(
        ckpt,
        &format!("{base}.in_proj_b.weight"),
        DType::F16,
        &shape,
    )?;
    let a = read_exact(
        ckpt,
        &format!("{base}.in_proj_a.weight"),
        DType::F16,
        &shape,
    )?;
    let mut rows = Vec::with_capacity(b.len() + a.len());
    rows.extend_from_slice(&b);
    rows.extend_from_slice(&a);
    Ok(DeviceBuffer::from_host(&gpu.stream, &rows)?)
}

/// Checkpoint `[num_v, hidden]` transposed to the runtime `[hidden, num_v]`
/// `LinearFP16` weight (`linear.py` `load_fp16`, `transpose = transposed_load`).
fn hgemm_weight(gpu: &Gpu, ckpt: &Checkpoint, name: &str, n: i32, k: i32) -> Result<ByteBuf> {
    let bytes = read_exact(ckpt, name, DType::F16, &[u64_dim(n), u64_dim(k)])?;
    let transposed = transpose_fp16(&bytes, n as usize, k as usize)?;
    Ok(DeviceBuffer::from_host(&gpu.stream, &transposed)?)
}

/// `[n, k]` fp16 to `[k, n]` fp16. Bits are copied; nothing is converted.
fn transpose_fp16(bytes: &[u8], n: usize, k: usize) -> Result<Vec<u8>> {
    if n == 0 || k == 0 {
        return Err(Error::Weights("b/a weight has a zero dimension".into()));
    }
    let need = n
        .checked_mul(k)
        .and_then(|elems| elems.checked_mul(2))
        .ok_or_else(|| Error::Weights("b/a weight size overflows".into()))?;
    if bytes.len() != need {
        return Err(Error::Weights(format!(
            "b/a weight has {} bytes, expected {need}",
            bytes.len()
        )));
    }
    let mut out = vec![0u8; need];
    for (row, src_row) in bytes.chunks_exact(k * 2).enumerate() {
        for (col, elem) in src_row.chunks_exact(2).enumerate() {
            let dst = (col * n + row) * 2;
            out[dst..dst + 2].copy_from_slice(elem);
        }
    }
    Ok(out)
}

fn cat_ba_bias(gpu: &Gpu, ckpt: &Checkpoint, base: &str, num_v: i32) -> Result<Option<ByteBuf>> {
    let b_name = format!("{base}.in_proj_b.bias");
    let a_name = format!("{base}.in_proj_a.bias");
    let b = ckpt.tensors.get(&b_name);
    let a = ckpt.tensors.get(&a_name);
    if b.is_none() && a.is_none() {
        return Ok(None);
    }
    let mut row = Vec::new();
    row.extend_from_slice(&bias_or_zero(b, num_v, &b_name)?);
    row.extend_from_slice(&bias_or_zero(a, num_v, &a_name)?);
    Ok(Some(DeviceBuffer::from_host(&gpu.stream, &row)?))
}

fn bias_or_zero(info: Option<&berth_exl3::TensorInfo>, n: i32, name: &str) -> Result<Vec<u8>> {
    let Some(info) = info else {
        return Ok(vec![0u8; nbytes(n, 2)?]);
    };
    expect_vector(info, DType::F16, n).map_err(|_| {
        Error::Weights(format!(
            "{name}: expected F16 [{n}], got {} {:?}",
            info.dtype, info.shape
        ))
    })?;
    device::read_tensor(info)
}

fn optional_conv_bias(
    gpu: &Gpu,
    ckpt: &Checkpoint,
    base: &str,
    fdim: i32,
) -> Result<Option<ByteBuf>> {
    let name = format!("{base}.conv1d.bias");
    if ckpt.tensors.get(&name).is_none() {
        return Ok(None);
    }
    Ok(Some(upload_exact(
        gpu,
        ckpt,
        &name,
        DType::Bf16,
        &[u64_dim(fdim)],
    )?))
}

fn upload_exact(
    gpu: &Gpu,
    ckpt: &Checkpoint,
    name: &str,
    dtype: DType,
    shape: &[u64],
) -> Result<ByteBuf> {
    let bytes = read_exact(ckpt, name, dtype, shape)?;
    Ok(DeviceBuffer::from_host(&gpu.stream, &bytes)?)
}

fn read_exact(ckpt: &Checkpoint, name: &str, dtype: DType, shape: &[u64]) -> Result<Vec<u8>> {
    let info = ckpt
        .tensors
        .get(name)
        .ok_or_else(|| Error::Weights(format!("tensor {name} is missing")))?;
    if info.dtype != dtype || info.shape.as_slice() != shape {
        return Err(Error::Weights(format!(
            "{name}: expected {dtype} {shape:?}, got {} {:?}",
            info.dtype, info.shape
        )));
    }
    device::read_tensor(info)
}

fn upload_info(gpu: &Gpu, info: &berth_exl3::TensorInfo) -> Result<ByteBuf> {
    Ok(DeviceBuffer::from_host(
        &gpu.stream,
        &device::read_tensor(info)?,
    )?)
}

fn raw<T: DeviceCopy>(buf: &DeviceBuffer<T>) -> *mut c_void {
    buf.ptr().as_u64() as *mut c_void
}

/// Scratch for [`GdnLayer::prefill`], allocated once by the model for the
/// largest chunk and shared by all 48 GDN layers.
///
/// The qkv and z trellises stay on the layer. This holds the chunk buffers
/// those linears write, the `gdn_prefill` workspace, and the Hadamard /
/// reconstruct scratch the separate linears need. `out_proj` uses the
/// caller's [`crate::device::RowScratch`] instead of a second copy.
pub struct GdnPrefillScratch {
    dims: Dims,
    max_rows: u32,
    width: i32,
    slices: i32,
    /// fp32 `[max_rows, fdim]`.
    qkv: ByteBuf,
    /// fp32 `[max_rows, v_dim]`.
    z: ByteBuf,
    /// fp32 `[max_rows, num_v_heads]`, `b_proj` output.
    b: ByteBuf,
    /// fp32 `[max_rows, num_v_heads]`, `a_proj` output.
    a: ByteBuf,
    /// fp16 `[max_rows, v_dim]`. Gated RMSNorm output, `out_proj` input.
    core: ByteBuf,
    /// `gdn_prefill::scratch_bytes(max_rows)`.
    workspace: ByteBuf,
    /// fp16 `[max_rows, hidden]` for the separate `exl3_linear` calls.
    a_had: ByteBuf,
    /// fp16 reconstruct buffer, `hidden * fdim` (qkv is wider than z).
    weight: ByteBuf,
    /// fp16 `[2, mgemm_rows, hidden]` for `project_qkvz_sliced`.
    mgemm_had: ByteBuf,
    /// fp32 `[slices, mgemm_rows, width]`. Sliced mode writes `c_ptrs`.
    carrier: ByteBuf,
    /// Device pointers, `out + n0 * 4`, into `qkv` then `z`.
    c_ptrs: DeviceBuffer<u64>,
    kernels: berth_kernels::gdn_prefill::GdnPrefill,
}

impl GdnPrefillScratch {
    /// Room for chunks of up to `max_rows` tokens.
    ///
    /// The cubins are the 27B path (16/48 heads, dim 128, conv 4, `F = 10240`).
    /// `max_rows` is `2..=4096`. The model passes 2048.
    pub fn new(gpu: &Gpu, shape: &ModelShape, max_rows: u32) -> Result<Self> {
        let dims = dims_from_shape(shape)?;
        expect_kernel_shape(&dims)?;
        let rows_i = i32::try_from(max_rows)
            .map_err(|_| Error::Input("gdn prefill max_rows does not fit in i32".into()))?;
        if !(2..=PREFILL_MAX_ROWS).contains(&max_rows) {
            return Err(Error::Input(format!(
                "gdn prefill max_rows {max_rows} is outside 2..={PREFILL_MAX_ROWS}"
            )));
        }
        let kernels = berth_kernels::gdn_prefill::GdnPrefill::load(&gpu.ctx, &fla_cubin_dir())?;
        let (width, slices) = slice_plan(dims.fdim, dims.v_dim)?;
        let rows = max_rows as usize;
        let hidden = dims.hidden as usize;
        let fdim = dims.fdim as usize;
        let v_dim = dims.v_dim as usize;
        let heads = dims.num_v_heads as usize;
        let mgemm_rows = rows.min(SLICED_QKVZ_MAX as usize);
        let qkv = dev_bytes(gpu, mul_usize(rows, fdim, "qkv")?, 4, "qkv")?;
        let z = dev_bytes(gpu, mul_usize(rows, v_dim, "z")?, 4, "z")?;
        let workspace_bytes = berth_kernels::gdn_prefill::scratch_bytes(rows_i)?;
        let workspace = gpu.zeroed(workspace_bytes)?;
        if !workspace.ptr().as_u64().is_multiple_of(256) {
            return Err(Error::Input(
                "gdn prefill workspace is not 256-byte aligned".into(),
            ));
        }
        let c_ptrs = qkvz_c_ptrs(gpu, &qkv, &z, dims.fdim, dims.v_dim, width, slices)?;
        let carrier_elems = mul_usize(slices as usize, mgemm_rows, "mgemm carrier")?;
        let carrier_elems = mul_usize(carrier_elems, width as usize, "mgemm carrier")?;
        let mgemm_had_elems = mul_usize(2, mgemm_rows, "mgemm a_had")?;
        let mgemm_had_elems = mul_usize(mgemm_had_elems, hidden, "mgemm a_had")?;
        Ok(Self {
            dims,
            max_rows,
            width,
            slices,
            qkv,
            z,
            b: dev_bytes(gpu, mul_usize(rows, heads, "b")?, 4, "b")?,
            a: dev_bytes(gpu, mul_usize(rows, heads, "a")?, 4, "a")?,
            core: dev_bytes(gpu, mul_usize(rows, v_dim, "core")?, 2, "core")?,
            workspace,
            a_had: dev_bytes(gpu, mul_usize(rows, hidden, "a_had")?, 2, "a_had")?,
            weight: dev_bytes(
                gpu,
                mul_usize(hidden, fdim, "reconstruct weight")?,
                2,
                "reconstruct weight",
            )?,
            mgemm_had: dev_bytes(gpu, mgemm_had_elems, 2, "mgemm a_had")?,
            carrier: dev_bytes(gpu, carrier_elems, 4, "mgemm carrier")?,
            c_ptrs,
            kernels,
        })
    }
}

fn fla_cubin_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/triton-cubins/fla")
}

fn expect_kernel_shape(dims: &Dims) -> Result<()> {
    let ok = dims.hidden == 5120
        && dims.num_k_heads == 16
        && dims.num_v_heads == 48
        && dims.k_head_dim == 128
        && dims.v_head_dim == 128
        && dims.conv_k == 4
        && dims.fdim == 10240
        && dims.v_dim == 6144;
    if ok {
        return Ok(());
    }
    Err(Error::Input(format!(
        "gdn prefill cubins are the 27B shape (hidden 5120, 16/48 heads of 128, conv 4, F 10240); \
         got hidden {} heads {}/{} dims {}/{} conv {} F {} V {}",
        dims.hidden,
        dims.num_k_heads,
        dims.num_v_heads,
        dims.k_head_dim,
        dims.v_head_dim,
        dims.conv_k,
        dims.fdim,
        dims.v_dim
    )))
}

fn dev_bytes(gpu: &Gpu, elems: usize, elem: usize, what: &str) -> Result<ByteBuf> {
    let bytes = elems
        .checked_mul(elem)
        .ok_or_else(|| Error::Input(format!("gdn prefill {what} overflows")))?;
    gpu.zeroed(bytes)
}

fn mul_usize(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| Error::Input(format!("gdn prefill {what} overflows")))
}

/// `SlicedMultiLinear` output bases: `out.data_ptr() + n0 * sizeof(fp32)`.
fn qkvz_c_ptrs(
    gpu: &Gpu,
    qkv: &ByteBuf,
    z: &ByteBuf,
    fdim: i32,
    v_dim: i32,
    width: i32,
    slices: i32,
) -> Result<DeviceBuffer<u64>> {
    let mut host = Vec::with_capacity(slices as usize);
    for (out, n) in [(qkv, fdim), (z, v_dim)] {
        let mut n0 = 0;
        while n0 < n {
            host.push(out.ptr().as_u64() + u64::try_from(n0).unwrap() * 4);
            n0 += width;
        }
    }
    if host.len() != slices as usize {
        return Err(Error::Input(format!(
            "gdn prefill built {} c_ptrs, expected {slices}",
            host.len()
        )));
    }
    Ok(DeviceBuffer::from_host(&gpu.stream, &host)?)
}

impl GdnLayer {
    /// `output = linear_attn(input)` for a chunk of `rows > 1` tokens, the
    /// fork's prefill path. Continues from this layer's conv and recurrent
    /// state and leaves the state after the chunk, as the fork does.
    ///
    /// `input` is fp16 `[rows, hidden]`; `output` is fp32 `[rows, hidden]`.
    ///
    /// Fork order (`gated_delta_net.py` 1096–1154): qkv and z (1097–1101),
    /// `b_proj` and `a_proj` (1103–1104), `gated_delta_net_fused_op_2` plus
    /// the conv, delta rule, and gated norm (`gdn_prefill`), then `o_proj`
    /// (1154).
    pub fn prefill(
        &mut self,
        gpu: &Gpu,
        rows_scratch: &mut crate::device::RowScratch,
        scratch: &mut GdnPrefillScratch,
        input: &ByteBuf,
        output: &mut ByteBuf,
        rows: u32,
    ) -> Result<()> {
        let rows_i = i32::try_from(rows)
            .map_err(|_| Error::Input("gdn prefill rows do not fit in i32".into()))?;
        if rows < 2 || rows > scratch.max_rows {
            return Err(Error::Input(format!(
                "gdn prefill rows {rows} is outside 2..={}",
                scratch.max_rows
            )));
        }
        if self.dims != scratch.dims
            || self.qkvz.width != scratch.width
            || self.qkvz.slices != scratch.slices
        {
            return Err(Error::Input(
                "gdn prefill scratch was built for a different shape".into(),
            ));
        }
        if self.ba_bias.is_some() {
            return Err(Error::Input(
                "gdn prefill: in_proj_b/a bias is present; LinearFP16 would add it".into(),
            ));
        }
        if self.conv_bias.is_some() {
            return Err(Error::Input(
                "gdn prefill: conv1d bias is present; the triton conv has no bias".into(),
            ));
        }
        let hidden = self.dims.hidden as usize;
        let n = rows as usize;
        if input.num_bytes() < n * hidden * 2 || output.num_bytes() < n * hidden * 4 {
            return Err(Error::Input(format!(
                "gdn prefill: expected fp16 [{rows}, {hidden}] in and fp32 [{rows}, {hidden}] out"
            )));
        }
        self.project_qkvz(gpu, scratch, input, rows_i)?;
        self.project_ba(gpu, scratch, input, rows_i)?;
        self.run_gdn_prefill(gpu, scratch, rows_i)?;
        self.out_proj.forward_rows(
            gpu,
            rows_scratch,
            &scratch.core,
            output,
            rows,
            OutDtype::F32,
        )
    }

    /// `gated_delta_net.py` 1097–1101. `rows <= 32` is one sliced mgemm.
    /// Longer chunks are `qkv_proj.forward` then `z_proj.forward`, both fp32.
    fn project_qkvz(
        &self,
        gpu: &Gpu,
        scratch: &GdnPrefillScratch,
        input: &ByteBuf,
        rows: i32,
    ) -> Result<()> {
        let d = &self.dims;
        let q = &self.qkvz;
        let stream = gpu.raw_stream();
        if rows <= SLICED_QKVZ_MAX {
            // SAFETY: slice tables alias this layer's trellises. `c_ptrs`
            // address `scratch.qkv` / `scratch.z`, which hold `max_rows`
            // rows. `mgemm_had` has two slabs of at least `rows`. The
            // carrier is `(slices, mgemm_rows, width)` with `mgemm_rows >= rows`.
            unsafe {
                berth_kernels::prefill_lin::exl3_mgemm(
                    ptr(input),
                    raw(&q.b_list),
                    ptr(&scratch.carrier),
                    raw(&q.suh_list),
                    ptr(&scratch.mgemm_had),
                    raw(&q.svh_list),
                    rows,
                    d.hidden,
                    q.width,
                    q.qkv.k_bits,
                    Codebook::Mul1,
                    OutDtype::F32,
                    1,
                    q.slices,
                    q.size_n_list.ptr().as_u64() as *const i32,
                    raw(&scratch.c_ptrs),
                    q.n_stride_list.ptr().as_u64() as *const i32,
                    q.had_src_list.ptr().as_u64() as *const i32,
                    2,
                    stream,
                )?;
            }
            return Ok(());
        }
        let weight = if rows > RECONSTRUCT_ABOVE {
            ptr(&scratch.weight)
        } else {
            std::ptr::null_mut()
        };
        launch_exl3(
            input,
            &q.qkv,
            &scratch.qkv,
            &scratch.a_had,
            weight,
            rows,
            d.hidden,
            d.fdim,
            stream,
        )?;
        launch_exl3(
            input,
            &q.z,
            &scratch.z,
            &scratch.a_had,
            weight,
            rows,
            d.hidden,
            d.v_dim,
            stream,
        )
    }

    /// `b_proj.forward` then `a_proj.forward` (`gated_delta_net.py` 1103–1104).
    ///
    /// Both are `LinearFP16` with `out_dtype` float (`gated_delta_net.py`
    /// 480–483). `y` is fp32 and `x` is fp16, so `fp16.py` 93–96 calls
    /// `ext.hgemm`, not `matmul`. Two `N = 48` GEMMs, not one `N = 96`.
    fn project_ba(
        &self,
        gpu: &Gpu,
        scratch: &GdnPrefillScratch,
        input: &ByteBuf,
        rows: i32,
    ) -> Result<()> {
        let stream = gpu.raw_stream();
        hgemm_ba(
            input,
            &self.b_hgemm,
            &scratch.b,
            rows,
            self.dims.hidden,
            self.dims.num_v_heads,
            stream,
            "b",
        )?;
        hgemm_ba(
            input,
            &self.a_hgemm,
            &scratch.a,
            rows,
            self.dims.hidden,
            self.dims.num_v_heads,
            stream,
            "a",
        )?;
        Ok(())
    }

    /// Fused op 2, conv, delta rule, gated norm. Writes this layer's state.
    fn run_gdn_prefill(&self, gpu: &Gpu, scratch: &GdnPrefillScratch, rows: i32) -> Result<()> {
        let (history_stride, state_size) = gdn_state_launch(self.dims.conv_k, self.max_history)?;
        let args = berth_kernels::gdn_prefill::GdnPrefillArgs {
            qkv: scratch.qkv.ptr().as_u64(),
            z: scratch.z.ptr().as_u64(),
            b: scratch.b.ptr().as_u64(),
            a: scratch.a.ptr().as_u64(),
            dt_bias: self.dt_bias.ptr().as_u64(),
            a_log: self.a_log.ptr().as_u64(),
            conv_weight: self.conv_weight.ptr().as_u64(),
            norm_weight: self.norm_weight.ptr().as_u64(),
            conv_state: self.conv_state.ptr().as_u64(),
            recurrent_state: self.recurrent_state.ptr().as_u64(),
            out: scratch.core.ptr().as_u64(),
            workspace: scratch.workspace.ptr().as_u64(),
            workspace_bytes: scratch.workspace.num_bytes(),
            seqlen: rows,
            slot: 0,
            num_slots: 1,
            history_stride,
            beta_scale: BETA_SCALE,
        };
        // FLA (`gated_delta_rule.py`) reads `recurrent_state[slot, 0]` and
        // copies the new state back into that view. `GdnPrefill::launch_chunk`
        // adds `slot * history_stride * state_bytes` before the pointer. Slot
        // 0 keeps offset 0 for any stride, so the live state stays the first
        // `[num_v_heads, k_head_dim, v_head_dim]` block. The recurrent kernel
        // (`seqlen < 48`) indexes history 0 the same way and uses
        // `history_stride` only as the slot stride. History-off launches do
        // not read the tail. `max_history == 0` passes stride 1 and
        // `state_size == conv_k` (4 on the 27B), and that case calls `run`,
        // which is the state_size-4 cubin. A wider conv uses
        // `run_with_state_size`.
        //
        // SAFETY: every address is a device allocation sized for this chunk
        // (the scratch holds `max_rows >= rows`; state is this layer's buffer,
        // wide enough for `state_size` and `history_stride`). `conv_state`
        // and `recurrent_state` are read and written. `gpu.stream` is the
        // stream `GdnPrefill::load` was given a context for.
        if state_size == 4 {
            unsafe { scratch.kernels.run(&gpu.stream, &args)? };
        } else {
            unsafe {
                scratch
                    .kernels
                    .run_with_state_size(&gpu.stream, &args, state_size)?
            };
        }
        Ok(())
    }
}

/// One separate prefill linear, fp32 out. `weight` is null at or under 144 rows.
#[allow(clippy::too_many_arguments)]
fn launch_exl3(
    a: &ByteBuf,
    proj: &Projection,
    c: &ByteBuf,
    a_had: &ByteBuf,
    weight: *mut c_void,
    rows: i32,
    size_k: i32,
    size_n: i32,
    stream: *mut c_void,
) -> Result<()> {
    // SAFETY: trellis, suh, and svh are the checkpoint uploads. `a_had` is
    // `max_rows * hidden` fp16. `weight`, when non-null, is `hidden * fdim`
    // fp16, which covers both projections (`size_n` is `fdim` or `v_dim`).
    // `c` is fp32 `[rows, size_n]`. The caller's `rows` is at most `max_rows`.
    unsafe {
        berth_kernels::prefill_lin::exl3_linear(
            ptr(a),
            ptr(&proj.trellis),
            ptr(c),
            ptr(&proj.suh),
            ptr(a_had),
            ptr(&proj.svh),
            weight,
            rows,
            size_k,
            size_n,
            proj.k_bits,
            Codebook::Mul1,
            OutDtype::F32,
            stream,
        )?;
    }
    Ok(())
}

/// One `N = 48` fp32 hgemm. A non-zero return means f16acc ran; `LinearFP16`
/// never takes that path, so the result would not match the fork.
#[allow(clippy::too_many_arguments)]
fn hgemm_ba(
    a: &ByteBuf,
    weight: &ByteBuf,
    c: &ByteBuf,
    rows: i32,
    k: i32,
    n: i32,
    stream: *mut c_void,
    which: &str,
) -> Result<()> {
    // SAFETY: `a` is fp16 `[rows, k]`, `weight` is fp16 `[k, n]`, `c` is
    // fp32 `[rows, n]`. All three stay alive until `stream` finishes.
    // `berth_kernels::prefill_lin::hgemm` is the fork's `ext.hgemm` choice.
    // For `N = 48` the f16acc tile does not divide `N`, so it runs
    // `cublasGemmEx` with `hgemm_gemmex_impl`'s arguments
    // (`exllamav3_ext/hgemm.cu` 65-75), which is what `LinearFP16` uses.
    unsafe {
        berth_kernels::prefill_lin::hgemm(
            ptr(a),
            ptr(weight),
            ptr(c),
            rows,
            k,
            n,
            n,
            OutDtype::F32,
            stream,
        )?;
    }
    let launched = berth_kernels::prefill_lin::last_launch();
    if launched != "cublasGemmEx" {
        return Err(Error::Kernel(berth_kernels::Error::Launch(format!(
            "gdn prefill: in_proj_{which} ran {launched}; LinearFP16 always uses cublasGemmEx"
        ))));
    }
    Ok(())
}

/// Host copy of the one live GDN slot (`max_batch` 1).
///
/// `max_history` is 0 unless [`GdnLayer::set_max_history`] widened the buffers.
pub(crate) struct GdnStateBytes {
    pub conv: Vec<u8>,
    pub recurrent: Vec<u8>,
    pub fdim: u32,
    pub conv_k: u32,
    pub num_v_heads: u32,
    pub k_head_dim: u32,
    pub v_head_dim: u32,
    pub max_history: u32,
}

impl GdnLayer {
    /// Copy conv and recurrent state. Both buffers are that one slot, including
    /// a history tail when [`Self::set_max_history`] widened them.
    ///
    /// Conv is bf16 `[fdim, conv_k + max_history]`. Recurrent is fp32
    /// `[max_history + 1, num_v_heads, k_head_dim, v_head_dim]`. With
    /// `max_history == 0` those are `[fdim, conv_k]` and
    /// `[1, num_v_heads, k_head_dim, v_head_dim]`. Decode does not call this.
    pub(crate) fn copy_state(&self, gpu: &Gpu) -> Result<GdnStateBytes> {
        let mut conv = vec![0u8; self.conv_state.num_bytes()];
        self.conv_state.copy_to_host(&gpu.stream, &mut conv)?;
        let mut recurrent = vec![0u8; self.recurrent_state.num_bytes()];
        self.recurrent_state
            .copy_to_host(&gpu.stream, &mut recurrent)?;
        Ok(GdnStateBytes {
            conv,
            recurrent,
            fdim: u32::try_from(self.dims.fdim).unwrap_or(0),
            conv_k: u32::try_from(self.dims.conv_k).unwrap_or(0),
            num_v_heads: u32::try_from(self.dims.num_v_heads).unwrap_or(0),
            k_head_dim: u32::try_from(self.dims.k_head_dim).unwrap_or(0),
            v_head_dim: u32::try_from(self.dims.v_head_dim).unwrap_or(0),
            max_history: self.max_history,
        })
    }
}

/// n-row history scratch for [`GdnLayer::forward_history`].
///
/// One scratch is shared by every GDN layer. `mtp_steps == 0` never
/// allocates it. Buffers match `BC_GatedDeltaNetSplit::run_bszN` with
/// `history = true` (`gated_delta_net.cpp` `run_bszN_gr`).
pub struct GdnHistoryScratch {
    dims: Dims,
    max_rows: u32,
    width: i32,
    slices: i32,
    /// fp32 `[max_rows, fdim]`.
    qkv: ByteBuf,
    /// fp32 `[max_rows, v_dim]`. `in_proj_z`, the gated-norm gate.
    z: ByteBuf,
    /// fp32 `[max_rows, ba_n]`.
    ba: ByteBuf,
    /// bf16 `[max_rows, num_v_heads]`.
    beta: ByteBuf,
    /// fp32 `[max_rows, num_v_heads]`. Recurrent decay, not the z gate.
    g: ByteBuf,
    /// bf16 `[1, fdim, max_rows]`. Packed with the launched `seqlen`.
    mixed_qkv: ByteBuf,
    /// bf16 `[1, max_rows, fdim]`.
    conv_out: ByteBuf,
    /// bf16 `[max_rows, v_dim]`.
    core_attn: ByteBuf,
    /// fp16 `[max_rows, v_dim]`. Gated RMSNorm output, `o_proj` input.
    core: ByteBuf,
    /// fp16 `[2, max_rows, hidden]`. Packed with stride `rows`.
    a_had: ByteBuf,
    /// fp32 `[slices, max_rows, width]`.
    carrier: ByteBuf,
    /// `out + n0 * 4` into `qkv` then `z`.
    c_ptrs: DeviceBuffer<u64>,
}

impl GdnLayer {
    /// Allocate [`GdnHistoryScratch`] for `1..=max_rows` tokens. `max_rows` is 1..=16.
    ///
    /// # Errors
    ///
    /// `max_rows` is outside 1..=16.
    pub fn history_scratch(&self, gpu: &Gpu, max_rows: u32) -> Result<GdnHistoryScratch> {
        if !(1..=16).contains(&max_rows) {
            return Err(Error::Input(format!(
                "gdn history max_rows {max_rows} is outside 1..=16"
            )));
        }
        let d = &self.dims;
        let rows = max_rows as usize;
        let qkv = dev_bytes(
            gpu,
            mul_usize(rows, d.fdim as usize, "history qkv")?,
            4,
            "history qkv",
        )?;
        let z = dev_bytes(
            gpu,
            mul_usize(rows, d.v_dim as usize, "history z")?,
            4,
            "history z",
        )?;
        let c_ptrs = qkvz_c_ptrs(
            gpu,
            &qkv,
            &z,
            d.fdim,
            d.v_dim,
            self.qkvz.width,
            self.qkvz.slices,
        )?;
        let heads = d.num_v_heads as usize;
        let hidden = d.hidden as usize;
        let width = self.qkvz.width as usize;
        let slices = self.qkvz.slices as usize;
        let had = mul_usize(
            mul_usize(2, rows, "history a_had")?,
            hidden,
            "history a_had",
        )?;
        let carrier = mul_usize(
            mul_usize(slices, rows, "history carrier")?,
            width,
            "history carrier",
        )?;
        Ok(GdnHistoryScratch {
            dims: *d,
            max_rows,
            width: self.qkvz.width,
            slices: self.qkvz.slices,
            qkv,
            z,
            ba: dev_bytes(
                gpu,
                mul_usize(rows, d.ba_n as usize, "history ba")?,
                4,
                "history ba",
            )?,
            beta: dev_bytes(
                gpu,
                mul_usize(rows, heads, "history beta")?,
                2,
                "history beta",
            )?,
            g: dev_bytes(gpu, mul_usize(rows, heads, "history g")?, 4, "history g")?,
            mixed_qkv: dev_bytes(
                gpu,
                mul_usize(d.fdim as usize, rows, "history mixed_qkv")?,
                2,
                "history mixed_qkv",
            )?,
            conv_out: dev_bytes(
                gpu,
                mul_usize(rows, d.fdim as usize, "history conv_out")?,
                2,
                "history conv_out",
            )?,
            core_attn: dev_bytes(
                gpu,
                mul_usize(rows, d.v_dim as usize, "history core")?,
                2,
                "history core",
            )?,
            core: dev_bytes(
                gpu,
                mul_usize(rows, d.v_dim as usize, "history core f16")?,
                2,
                "history core f16",
            )?,
            a_had: dev_bytes(gpu, had, 2, "history a_had")?,
            carrier: dev_bytes(gpu, carrier, 4, "history carrier")?,
            c_ptrs,
        })
    }

    /// `output = linear_attn(input)` for `rows` tokens with history on.
    ///
    /// Seven launches from `docs/mtp.md` section 4: qkvz sliced mgemm, ba
    /// GEMV, `fused_op_3`, `conv1d_update_history`, `recurrent_gated_delta_rule_history`,
    /// gated RMSNorm, `o_proj`. `history_stride = max_history + 1` and
    /// `state_size = conv_k + max_history`. After the conv and recurrent
    /// kernels return, `last_history` becomes `rows - 1` even if the norm or
    /// `o_proj` then fails.
    ///
    /// # Errors
    ///
    /// `rows` is outside 1..=16, the scratch shape does not match, or
    /// `history_stride < rows`.
    pub fn forward_history(
        &mut self,
        gpu: &Gpu,
        rows_scratch: &mut crate::device::RowScratch,
        scratch: &GdnHistoryScratch,
        input: &ByteBuf,
        output: &mut ByteBuf,
        rows: u32,
    ) -> Result<()> {
        let rows_i = self.check_history(scratch, input, output, rows)?;
        self.project_history(gpu, scratch, input, rows_i)?;
        self.state_history(gpu, scratch, rows_i)?;
        self.out_history(gpu, rows_scratch, scratch, output, rows)
    }

    /// Conv and recurrent rewind for `num_tokens`, then `last_history = 0`.
    ///
    /// One layer. [`rewind_jobs`] plus the batched launch is how the model
    /// rewinds every GDN layer in two kernels. `num_tokens == 0` skips the
    /// state job and still copies conv when `last_history > 0`.
    ///
    /// # Errors
    ///
    /// `num_tokens > last_history`. The cursor is left unchanged.
    pub fn rewind(&mut self, gpu: &Gpu, num_tokens: u32) -> Result<()> {
        let (conv, state) = self.rewind_jobs(num_tokens)?;
        let conv_jobs: Vec<_> = conv.into_iter().collect();
        let state_jobs: Vec<_> = state.into_iter().collect();
        // SAFETY: addresses come from this layer's state buffers and stay
        // allocated until `gpu.stream` finishes. An empty job list does not
        // call CUDA.
        unsafe {
            berth_kernels::gdn::batched_conv_rewind(&conv_jobs, gpu.raw_stream())?;
            berth_kernels::gdn::batched_state_rewind(&state_jobs, gpu.raw_stream())?;
        }
        self.last_history = 0;
        Ok(())
    }

    /// Addresses for one layer, matching `GDNState.rewind`.
    ///
    /// `None` conv when `last_history == 0`. `None` state when `num_tokens == 0`.
    /// Does not clear the cursor and does not launch.
    ///
    /// # Errors
    ///
    /// `num_tokens > last_history`, or the conv window would start before the buffer.
    pub fn rewind_jobs(
        &self,
        num_tokens: u32,
    ) -> Result<(
        Option<berth_kernels::gdn::ConvRewindJob>,
        Option<berth_kernels::gdn::StateRewindJob>,
    )> {
        let (history_stride, state_size) = gdn_state_launch(self.dims.conv_k, self.max_history)?;
        let last = i32::try_from(self.last_history)
            .map_err(|_| Error::Input("gdn last_history does not fit in i32".into()))?;
        let n = i32::try_from(num_tokens)
            .map_err(|_| Error::Input("gdn rewind num_tokens does not fit in i32".into()))?;
        let fdim = i64::from(self.dims.fdim);
        let state_i = i64::from(state_size);
        let conv = berth_kernels::gdn::rewind_conv_addrs(
            self.conv_state.ptr().as_u64(),
            0,
            fdim * state_i,
            state_i,
            1,
            self.dims.fdim,
            state_size,
            self.dims.conv_k,
            2,
            last,
            n,
        )?;
        let per_slot = i64::from(self.dims.num_v_heads)
            * i64::from(self.dims.k_head_dim)
            * i64::from(self.dims.v_head_dim);
        let stride0 = i64::from(history_stride) * per_slot;
        let state = berth_kernels::gdn::rewind_state_addrs(
            self.recurrent_state.ptr().as_u64(),
            0,
            stride0,
            per_slot,
            4,
            last,
            n,
        )?;
        Ok((conv, state))
    }

    /// `last_history = 0` after a batched rewind has been queued.
    pub(crate) fn clear_history_cursor(&mut self) {
        self.last_history = 0;
    }

    /// `history_stride` covers `rows`. Called on every layer before any launch.
    pub(crate) fn check_history_rows(&self, rows: u32) -> Result<()> {
        if !(1..=16).contains(&rows) {
            return Err(Error::Input(format!(
                "gdn history rows {rows} is outside 1..=16"
            )));
        }
        let (stride, _) = gdn_state_launch(self.dims.conv_k, self.max_history)?;
        let rows_i = i32::try_from(rows).expect("rows is at most 16");
        if stride < rows_i {
            return Err(Error::Input(format!(
                "gdn history_stride {stride} is shorter than {rows} rows (max_history is {})",
                self.max_history
            )));
        }
        Ok(())
    }

    fn check_history(
        &self,
        scratch: &GdnHistoryScratch,
        input: &ByteBuf,
        output: &ByteBuf,
        rows: u32,
    ) -> Result<i32> {
        self.check_history_rows(rows)?;
        if rows > scratch.max_rows || self.dims != scratch.dims {
            return Err(Error::Input(
                "gdn history scratch was built for a different shape".into(),
            ));
        }
        if self.qkvz.width != scratch.width || self.qkvz.slices != scratch.slices {
            return Err(Error::Input(
                "gdn history scratch slice plan does not match this layer".into(),
            ));
        }
        let hidden = self.dims.hidden as usize;
        let rows_u = rows as usize;
        let in_bytes = rows_u
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| Error::Input("gdn history input size overflows".into()))?;
        let out_bytes = rows_u
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error::Input("gdn history output size overflows".into()))?;
        if input.num_bytes() < in_bytes || output.num_bytes() < out_bytes {
            return Err(Error::Input(
                "gdn history input or output is shorter than the row span".into(),
            ));
        }
        Ok(i32::try_from(rows).expect("rows is at most 16"))
    }

    fn project_history(
        &self,
        gpu: &Gpu,
        scratch: &GdnHistoryScratch,
        input: &ByteBuf,
        rows: i32,
    ) -> Result<()> {
        let d = &self.dims;
        let q = &self.qkvz;
        let stream = gpu.raw_stream();
        // SAFETY: slice lists alias this layer's qkv+z trellis. `c_ptrs`
        // address the scratch's fp32 qkv and z. Buffers stay allocated until
        // `stream` finishes. Same codebook as the one-token launch.
        unsafe {
            let _shape = berth_kernels::ops::exl3_mgemm(
                ptr(input),
                raw(&q.b_list),
                ptr(&scratch.carrier),
                raw(&q.suh_list),
                ptr(&scratch.a_had),
                raw(&q.svh_list),
                rows,
                d.hidden,
                q.width,
                q.qkv.k_bits,
                Codebook::Mul1,
                OutDtype::F32,
                1,
                q.slices,
                q.size_n_list.ptr().as_u64() as *const i32,
                raw(&scratch.c_ptrs),
                q.n_stride_list.ptr().as_u64() as *const i32,
                q.had_src_list.ptr().as_u64() as *const i32,
                2,
                stream,
            )?;
            berth_kernels::gdn::gdn_ba_gemv(
                ptr(input),
                ptr(&self.ba_weight),
                self.ba_bias
                    .as_ref()
                    .map(ptr)
                    .unwrap_or(std::ptr::null_mut()),
                ptr(&scratch.ba),
                rows,
                d.hidden,
                d.ba_n,
                stream,
            )?;
            berth_kernels::gdn::gated_delta_net_fused_op_3(
                ptr(&scratch.qkv),
                ptr(&scratch.ba),
                ptr(&self.dt_bias),
                ptr(&self.a_log),
                ptr(&scratch.mixed_qkv),
                ptr(&scratch.beta),
                ptr(&scratch.g),
                1,
                rows,
                d.fdim,
                d.num_v_heads,
                BETA_SCALE,
                stream,
            )?;
        }
        Ok(())
    }

    fn state_history(&mut self, gpu: &Gpu, scratch: &GdnHistoryScratch, rows: i32) -> Result<()> {
        let d = &self.dims;
        let (history_stride, state_size) = gdn_state_launch(d.conv_k, self.max_history)?;
        let stream = gpu.raw_stream();
        // SAFETY: conv and recurrent state belong to this layer. Scratch
        // rows are the `rows` prefix. History writes indexes `0..rows`.
        unsafe {
            berth_kernels::gdn::conv1d_update_history(
                ptr(&scratch.mixed_qkv),
                ptr(&self.conv_state),
                std::ptr::null(),
                ptr(&self.conv_weight),
                self.conv_bias
                    .as_ref()
                    .map(ptr)
                    .unwrap_or(std::ptr::null_mut()),
                ptr(&scratch.conv_out),
                1,
                d.fdim,
                rows,
                state_size,
                d.conv_k,
                true,
                stream,
            )?;
            berth_kernels::gdn::recurrent_gated_delta_rule_history(
                ptr(&scratch.conv_out),
                ptr(&scratch.g),
                ptr(&scratch.beta),
                ptr(&self.recurrent_state),
                ptr(&scratch.core_attn),
                std::ptr::null(),
                1,
                rows,
                d.num_k_heads,
                d.num_v_heads,
                d.k_head_dim,
                d.v_head_dim,
                history_stride,
                true,
                stream,
            )?;
        }
        self.last_history = u32::try_from(rows - 1).expect("rows is positive");
        Ok(())
    }

    fn out_history(
        &mut self,
        gpu: &Gpu,
        rows_scratch: &mut crate::device::RowScratch,
        scratch: &GdnHistoryScratch,
        output: &mut ByteBuf,
        rows: u32,
    ) -> Result<()> {
        let d = &self.dims;
        let rows_i = i32::try_from(rows).expect("rows is at most 16");
        let norm_rows = rows_i
            .checked_mul(d.num_v_heads)
            .ok_or_else(|| Error::Input("gdn history norm rows overflow".into()))?;
        // SAFETY: `core_attn` is bf16 `[rows, Nv, Hv]`, `z` is fp32 of the
        // same element count, and `core` is fp16. The kernel views them as
        // `rows * num_v_heads` by `v_head_dim`.
        unsafe {
            berth_kernels::gdn::gated_rms_norm(
                ptr(&scratch.core_attn),
                ptr(&self.norm_weight),
                ptr(&scratch.core),
                ptr(&scratch.z),
                RMS_NORM_EPS,
                0.0,
                norm_rows,
                d.v_head_dim,
                1,
                0,
                0,
                gpu.raw_stream(),
            )?;
        }
        if rows == 1 {
            self.out_proj
                .forward(gpu, &scratch.core, output, OutDtype::F32)
        } else {
            self.out_proj.forward_rows(
                gpu,
                rows_scratch,
                &scratch.core,
                output,
                rows,
                OutDtype::F32,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn transposes_ba_weight_to_hgemm_layout() {
        // [2, 3] fp16 rows (1, 2, 3) and (4, 5, 6) become [3, 2] columns.
        let src = [1u16, 2, 3, 4, 5, 6];
        let bytes: Vec<u8> = src.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out = super::transpose_fp16(&bytes, 2, 3).unwrap();
        let got: Vec<u16> = out
            .chunks_exact(2)
            .map(|elem| u16::from_le_bytes([elem[0], elem[1]]))
            .collect();
        assert_eq!(got, vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn gdn_state_launch_matches_max_history() {
        // conv_k 4 is the 27B kernel. max_history 0 is today's launch.
        assert_eq!(super::gdn_state_launch(4, 0).unwrap(), (1, 4));
        assert_eq!(super::gdn_state_launch(4, 2).unwrap(), (3, 6));
        assert_eq!(super::gdn_state_launch(4, 4).unwrap(), (5, 8));
    }
}

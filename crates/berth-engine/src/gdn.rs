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
    /// bf16 `[1, fdim, conv_k]`. `GDNLayerState.conv_state` with one slot and
    /// `max_history = 0`, which is the reference cache. History off, so the
    /// kernel uses this whole prefix (`state_size = conv_k`).
    conv_state: ByteBuf,
    /// f32 `[1, 1, num_v_heads, k_head_dim, v_head_dim]`. History index 0 only,
    /// so `history_stride = 1`. K-major inside a head.
    recurrent_state: ByteBuf,
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
                d.conv_k,
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
                1,
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

    /// Zero conv and recurrent state in the allocations from [`Self::load`].
    ///
    /// `GDNLayerState.clear` (`gated_delta_net.py`) does `zero_()` on the
    /// slot. `Cache.get_new_state` asks for that when a job has no stashed
    /// checkpoint. Nothing is reallocated. `berth-cuda` has no memset on an
    /// existing buffer, so the zeros are copied into the same device memory.
    /// The caller synchronizes `gpu.stream` first if a forward is still queued.
    pub fn reset_state(&mut self, gpu: &Gpu) -> Result<()> {
        zero_in_place(gpu, &mut self.conv_state)?;
        zero_in_place(gpu, &mut self.recurrent_state)?;
        Ok(())
    }
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
            history_stride: 1,
            beta_scale: BETA_SCALE,
        };
        // SAFETY: every address is a device allocation sized for this chunk
        // (the scratch holds `max_rows >= rows`; state is the one-slot layer
        // buffer). `conv_state` and `recurrent_state` are read and written.
        // `gpu.stream` is the stream `GdnPrefill::load` was given a context for.
        unsafe { scratch.kernels.run(&gpu.stream, &args)? };
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

/// Host copy of the one live GDN slot (`max_batch` 1, `max_history` 0).
pub(crate) struct GdnStateBytes {
    pub conv: Vec<u8>,
    pub recurrent: Vec<u8>,
    pub fdim: u32,
    pub conv_k: u32,
    pub num_v_heads: u32,
    pub k_head_dim: u32,
    pub v_head_dim: u32,
}

impl GdnLayer {
    /// Copy conv and recurrent state. Both buffers are that one slot.
    ///
    /// Conv is bf16 `[fdim, conv_k]`. Recurrent is fp32
    /// `[num_v_heads, k_head_dim, v_head_dim]` (the reference shape has a
    /// leading 1, which adds no elements). Decode does not call this.
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
        })
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
}

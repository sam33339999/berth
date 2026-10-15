//! Gated DeltaNet mixer, one decode token (48 of the 27B's 64 layers).
//!
//! Owner: task GDN. Reproduces `BC_GatedDeltaNetSplit::run_bszN` for bsz 1,
//! history off: the sliced qkv+z `exl3_mgemm`, then the five GDN kernels, then
//! `out_proj`.

use std::ffi::c_void;

use berth_cuda::{ByteBuf, DeviceBuffer, DeviceCopy};
use berth_exl3::{Bitrate, Checkpoint, DType, HadamardScale, LinearExl3};
use berth_kernels::{Codebook, OutDtype};
use berth_plan::{LayerKind, ModelShape};

use crate::device::{self, ptr};
use crate::{Error, Exl3Linear, Gpu, Result};

/// `GatedDeltaNet.beta_scale` default. Qwen3.5 does not override it
/// (`gated_delta_net.py`).
const BETA_SCALE: f32 = 1.0;

/// 27B `text_config.rms_norm_eps`. The fork passes `config.rms_norm_eps` into
/// `GatedRMSNorm`. [`berth_exl3::ModelConfig`] does not store the field, so
/// the constant lives here. The checkpoint's value is `1e-06`.
const RMS_NORM_EPS: f32 = 1e-6;

/// One GDN layer: its weights, its conv and recurrent state, and scratch.
pub struct GdnLayer {
    dims: Dims,
    qkvz: Qkvz,
    /// fp16 `[2 * num_v_heads, hidden]`. Rows `0..H` are `in_proj_b`, the rest
    /// `in_proj_a` (`gated_delta_net.py` `cat([b, a], dim=-1).T`).
    ba_weight: ByteBuf,
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
    /// `in_proj_z` weights. The slice tables alias this allocation, so it has
    /// to stay alive even though launch reads the tables, not the field.
    #[allow(dead_code)]
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

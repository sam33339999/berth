//! Gated DeltaNet prefill for one chunk, batch 1, history off, state saved.
//!
//! This is the non-BC branch of `GatedDeltaNet.forward`
//! (`modules/gated_delta_net.py` 1096–1151) between the projections and
//! `out_proj`. `seq_len > 1` misses the BC graph (`gated_delta_net.py`
//! 1034–1048). The fork then does:
//!
//! 1. `ext.gated_delta_net_fused_op_2` (`gdn.cu` 277–336, kernel 237–270)
//!    on `b` and `a`, then `qkv.transpose(1, 2).to(bfloat16).contiguous()`
//!    (`gated_delta_net.py` 1106, 1111–1117). `z` stays fp32; it is the
//!    RMSNorm gate, not a conv input.
//! 2. `causal_conv1d_update` (`gated_delta_net_fn/conv1d.py` 364–419).
//!    `seq_len > 32` (`MAX_CUDA_SEQLEN`) takes the Triton slotted kernels.
//!    `seq_len <= 256` is one combined kernel; longer sequences launch the
//!    output kernel and then the state kernel (`conv1d.py` 280–337).
//!    `transpose_output` is true, `history` is false, and there is no conv
//!    bias (the host passes the weight pointer as the dummy bias).
//! 3. `gated_delta_rule_fn` (`gated_delta_net_fn/gated_delta_rule.py`).
//!    `seq_len >= num_v_heads` (48) and history off calls
//!    `vendor/fla.chunk_gated_delta_rule` (`__init__.py` 28–67, call site
//!    `gated_delta_rule.py` 165–193): l2norm, `chunk_local_cumsum`, fused
//!    kkt+solve, `recompute_w_u`, `chunk_gated_delta_rule_fwd_h`,
//!    `chunk_fwd_o`. Shorter sequences call
//!    `ext.cuda_recurrent_gated_delta_rule` (`gated_delta_rule.py` 195–222),
//!    which this module launches through [`crate::gdn::recurrent_gated_delta_rule`].
//! 4. `GatedRMSNorm.forward` (`gated_rmsnorm.py` 104–121) via
//!    `ext.gated_rms_norm` (`norm.cu` 608–706), launched through
//!    [`crate::gdn::gated_rms_norm`].
//!
//! # `b` and `a` are not computed here
//!
//! For many rows the fork's `b_proj` and `a_proj` are `LinearFP16` with
//! `out_dtype` float (`gated_delta_net.py` 475–481). `LinearFP16.forward`
//! (`modules/quant/fp16.py` 69–96) sees `y` dtype float and `x` dtype half,
//! so it calls `ext.hgemm`, not `torch.matmul`. `hgemm_gemmex_impl`
//! (`exllamav3_ext/hgemm.cu` 19–77) is row-major `a @ b -> c`:
//!
//! * `cublasGemmEx`, `CUBLAS_OP_N`, `CUBLAS_OP_N`
//! * `size_n = 48` (`b.size(-1)`), `size_m = seqlen` (`a.numel() / size_k`),
//!   `size_k = 5120` (`a.size(-1)`)
//! * pointer order in the call is the weight (`b_ptr`) then `x` (`a_ptr`)
//! * leading dimensions `ldb = size_n = 48`, `lda = size_k = 5120`,
//!   `ldc = c.stride(-2) = 48` for a contiguous `y`
//! * A and B `CUDA_R_16F`, C `CUDA_R_32F`, compute `CUBLAS_COMPUTE_32F`,
//!   algo `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, alpha 1, beta 0
//!
//! Runtime weight is fp16 `[5120, 48]` (the checkpoint stores the transpose;
//! `LinearFP16.get_tensors` writes `weight.T`). `x` is the layer input, fp16
//! `[seqlen, 5120]`, not the qkv projection. `y` is fp32 `[seqlen, 48]`.
//! The 100- and 1500-token profiles show `cutlass_80_tensorop_s16816gemm_f16_64x64_32x6_nn_align8`
//! and a float `splitKreduce` for this GEMM. Task LIN owns it. This module
//! takes the two fp32 results as `b` and `a`.
//!
//! # Triton cubins
//!
//! `scripts/build_triton_fla_cubins.py` launches the fork functions at the
//! 27B shapes, reads the autotune `best_config` Triton actually selected
//! (on-disk cache under `~/.triton/cache`, device sm_86), and saves that
//! cubin. Conv `seq_len` is a runtime integer. Triton still specializes
//! divisibility by 16, and it folds the value 1 into a constexpr, so the
//! combined `BLOCK_S` 1 cubin has no runtime `seq_len`. Each combined
//! `BLOCK_S` and the long output/state pair are stored for both
//! divisibility classes when the images differ. Load refuses a cubin whose
//! constexprs are not this 27B path. f32 literals (`1e-6`, `RCP_LN2`,
//! `128 ** -0.5`) are the bits the script recorded, not a recomputation.
//!
//! # State
//!
//! Conv state is bf16 `[num_slots, 10240, 4]`: `state_size` is 4 because
//! the profiled cache has `max_history == 0` (`conv1d.py`, `state_size =
//! conv_state.shape[-1]`, `BLOCK_STATE = next_pow2(4)`). Recurrent state is
//! fp32 `[num_slots, history_stride, 48, 128, 128]`. History off reads and
//! writes history index 0 only. `history_stride` is `recurrent_state.size(1)`
//! (`gdn.cu` 933; `max_history + 1` in the real cache). The kernel multiplies
//! the slot by that stride (`gdn.cu` 691–699) and writes index 0 in place.
//! The chunk path has no slots argument: the initial state is the pointer at
//! `slot * history_stride * 48 * 128 * 128` fp32 elements, and the separate
//! final-state buffer is copied back there (`gated_delta_rule.py` 180–190,
//! `state.copy_(new_state)`). The recurrent kernel applies the same
//! indexing itself (`gdn.cu` slot math). A second prefill chunk passes a
//! nonzero initial state in those buffers.
//!
//! Shapes are the 27B text config: 16 K heads, 48 V heads, head dim 128,
//! conv kernel 4, `F = 10240`. `seq_len` above 4096 is rejected because the
//! pinned l2norm cubin was compiled with `NB = 1` (`cdiv(T, 2048 * 32)` for
//! the flattened `q`/`k` length `seq_len * 16`).

use std::ffi::{c_char, c_int, c_void, CStr};
use std::path::Path;

use berth_cuda::{Context, Cubin, Dim3, Function, Stream};

use crate::gdn::{gated_rms_norm, recurrent_gated_delta_rule};
use crate::{Error, Result};

/// Command that writes `target/triton-cubins/fla/` when the cubins are missing.
pub const TRITON_FLA_BUILD: &str = "\
PYTHONPATH=~/qwen38-exl3-dflash2/exllamav3 \
TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \
~/qwen38-exl3-dflash2/env/bin/python scripts/build_triton_fla_cubins.py";

const NUM_K_HEADS: i32 = 16;
const NUM_V_HEADS: i32 = 48;
const HEAD_DIM: i32 = 128;
const QKV_DIM: i32 = 2 * NUM_K_HEADS * HEAD_DIM + NUM_V_HEADS * HEAD_DIM;
const K_DIM: i32 = NUM_K_HEADS * HEAD_DIM;
const V_DIM: i32 = NUM_V_HEADS * HEAD_DIM;
const CHUNK: i32 = 64;
const CONV_STATE: i32 = 4;
/// Flattened q/k length `seq_len * 16` must stay in the `NB = 1` bucket.
const MAX_SEQLEN: i32 = 4096;
const ALIGN: usize = 256;
const DEFAULT_DYNAMIC_SMEM: u32 = 48 * 1024;
const STATE_BYTES: u64 = (NUM_V_HEADS as u64) * (HEAD_DIM as u64) * (HEAD_DIM as u64) * 4;

/// Device pointers for one prefill chunk. Every address is a device pointer.
///
/// * `qkv` — f32 `[1, seqlen, 10240]`, the qkv projection output.
/// * `z` — f32 `[1, seqlen, 48, 128]`, the z projection, RMSNorm gate.
/// * `b`, `a` — f32 `[1, seqlen, 48]`, the `hgemm` results documented above.
/// * `dt_bias`, `a_log` — bf16 `[48]`, checkpoint `linear_attn.dt_bias` and
///   `linear_attn.A_log` (layer 0 is bf16 `[48]`).
/// * `conv_weight` — bf16 `[10240, 4]`, `conv1d.weight` squeezed on dim 1.
/// * `norm_weight` — bf16 `[128]`, `linear_attn.norm.weight`.
/// * `conv_state` — bf16 `[num_slots, 10240, 4]`, read and written.
/// * `recurrent_state` — f32 `[num_slots, history_stride, 48, 128, 128]`,
///   read and written at history index 0.
/// * `out` — fp16 `[1, seqlen, 6144]`, the gated-norm output (`[1, seqlen, 48, 128]`).
/// * `workspace` — `scratch_bytes(seqlen)` bytes, 256-byte aligned.
pub struct GdnPrefillArgs {
    /// f32 `[1, seqlen, 10240]`.
    pub qkv: u64,
    /// f32 `[1, seqlen, 48, 128]`.
    pub z: u64,
    /// f32 `[1, seqlen, 48]`.
    pub b: u64,
    /// f32 `[1, seqlen, 48]`.
    pub a: u64,
    /// bf16 `[48]`.
    pub dt_bias: u64,
    /// bf16 `[48]`.
    pub a_log: u64,
    /// bf16 `[10240, 4]`.
    pub conv_weight: u64,
    /// bf16 `[128]`.
    pub norm_weight: u64,
    /// bf16 `[num_slots, 10240, 4]`.
    pub conv_state: u64,
    /// f32 `[num_slots, history_stride, 48, 128, 128]`.
    pub recurrent_state: u64,
    /// fp16 `[1, seqlen, 6144]`.
    pub out: u64,
    /// Scratch base. 256-byte aligned.
    pub workspace: u64,
    /// Bytes available at `workspace`.
    pub workspace_bytes: usize,
    /// Chunk length. `1..=4096`.
    pub seqlen: i32,
    /// Recurrent and conv slot. `0 <= slot < num_slots`.
    pub slot: i32,
    /// Leading dimension of both state tensors.
    pub num_slots: i32,
    /// `recurrent_state.size(1)`. At least 1. Index 0 is the live state.
    pub history_stride: i32,
    /// `GatedDeltaNet.beta_scale`. The 27B default is 1.
    pub beta_scale: f32,
}

/// Workspace bytes for `seqlen`, including the chunk-path buffers.
///
/// The same size is used below and above 48 tokens. The recurrent path
/// leaves the fla buffers unused.
///
/// # Errors
///
/// `seqlen` is outside `1..=4096`.
pub fn scratch_bytes(seqlen: i32) -> Result<usize> {
    Ok(layout(seqlen)?.end)
}

/// Loaded 27B prefill cubins.
pub struct GdnPrefill {
    l2norm: Image,
    cumsum: Image,
    kkt: Image,
    recompute: Image,
    chunk_h: Image,
    chunk_o: Image,
    conv: Vec<(String, Image)>,
    l2_bt: i32,
    h_bv: i32,
    o_bv: i32,
    l2norm_eps: f32,
    rms_eps: f32,
    rcp_ln2: f32,
    k_scale: f32,
}

struct Image {
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
    F32,
}

struct Region {
    off: usize,
    bytes: usize,
}

struct Layout {
    slot: Region,
    mixed: Region,
    beta: Region,
    g: Region,
    conv_out: Region,
    q: Region,
    k: Region,
    v: Region,
    qn: Region,
    kn: Region,
    rstd: Region,
    g_cum: Region,
    a_mat: Region,
    w: Region,
    u: Region,
    v_new: Region,
    core: Region,
    h: Region,
    ht: Region,
    end: usize,
}

impl GdnPrefill {
    /// Load `manifest.json` and the cubins from `dir`.
    ///
    /// # Errors
    ///
    /// A missing file, a constexpr that is not the 27B prefill path, a
    /// nonzero Triton scratch size, or a driver error while loading.
    pub fn load(ctx: &Context, dir: &Path) -> Result<Self> {
        let manifest_path = dir.join("manifest.json");
        let text = std::fs::read_to_string(&manifest_path).map_err(|_| {
            Error::Launch(format!(
                "gdn prefill cubins are missing at {}. Run: {TRITON_FLA_BUILD}",
                manifest_path.display()
            ))
        })?;
        let manifest = parse_json(&text).map_err(|err| {
            Error::Launch(format!(
                "gdn prefill manifest {}: {err}",
                manifest_path.display()
            ))
        })?;
        let arch = manifest
            .get("arch")
            .and_then(Json::as_str)
            .ok_or_else(|| Error::Launch("manifest arch missing".into()))?;
        if arch != "sm_86" {
            return Err(Error::Launch(format!("manifest arch {arch}, want sm_86")));
        }
        let bits = manifest
            .get("f32_bits")
            .ok_or_else(|| Error::Launch("manifest f32_bits missing".into()))?;
        let l2norm_eps = f32_bits(bits, "l2norm_eps")?;
        let rms_eps = f32_bits(bits, "rms_eps")?;
        let rcp_ln2 = f32_bits(bits, "rcp_ln2")?;
        let k_scale = f32_bits(bits, "k_scale")?;
        let kernels = manifest
            .get("kernels")
            .ok_or_else(|| Error::Launch("manifest has no kernels object".into()))?;

        let l2norm = load_image(ctx, dir, kernels, "l2norm_fwd_kernel", "l2norm")?;
        check_l2norm(&l2norm)?;
        let l2_bt = const_i32(kernels, "l2norm_fwd_kernel", "BT")?;
        if !(8..=128).contains(&l2_bt) || !(l2_bt as u32).is_power_of_two() {
            return Err(Error::Launch(format!("l2norm BT {l2_bt}")));
        }

        let cumsum = load_image(
            ctx,
            dir,
            kernels,
            "chunk_local_cumsum_scalar_kernel",
            "cumsum",
        )?;
        check_cumsum(&cumsum)?;

        let kkt = load_image(
            ctx,
            dir,
            kernels,
            "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
            "kkt",
        )?;
        check_kkt(&kkt)?;

        let recompute = load_image(ctx, dir, kernels, "recompute_w_u_fwd_kernel", "recompute")?;
        check_recompute(&recompute)?;

        let chunk_h = load_image(
            ctx,
            dir,
            kernels,
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
            "chunk_h",
        )?;
        check_chunk_h(&chunk_h)?;
        let h_bv = const_i32(
            kernels,
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
            "BV",
        )?;
        if h_bv < 1 || HEAD_DIM % h_bv != 0 {
            return Err(Error::Launch(format!("chunk_h BV {h_bv}")));
        }

        let chunk_o = load_image(ctx, dir, kernels, "chunk_fwd_kernel_o", "chunk_o")?;
        check_chunk_o(&chunk_o)?;
        let o_bv = const_i32(kernels, "chunk_fwd_kernel_o", "BV")?;
        if o_bv < 1 || HEAD_DIM % o_bv != 0 {
            return Err(Error::Launch(format!("chunk_o BV {o_bv}")));
        }

        let mut conv = Vec::new();
        for block_s in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
            let key = format!("conv_combined_bs{block_s}");
            conv.push((
                key.clone(),
                load_conv(ctx, dir, kernels, &key, "conv_combined", block_s)?,
            ));
            if block_s >= 16 {
                let div = format!("{key}_div16");
                conv.push((
                    div.clone(),
                    load_conv(ctx, dir, kernels, &div, "conv_combined", block_s)?,
                ));
            }
        }
        for (key, grid) in [
            ("conv_output", "conv_output"),
            ("conv_output_div16", "conv_output"),
            ("conv_state", "conv_state"),
            ("conv_state_div16", "conv_state"),
        ] {
            let block_s = if grid == "conv_state" { 0 } else { 256 };
            conv.push((
                key.to_string(),
                load_conv(ctx, dir, kernels, key, grid, block_s)?,
            ));
        }

        Ok(Self {
            l2norm,
            cumsum,
            kkt,
            recompute,
            chunk_h,
            chunk_o,
            conv,
            l2_bt,
            h_bv,
            o_bv,
            l2norm_eps,
            rms_eps,
            rcp_ln2,
            k_scale,
        })
    }

    /// Run one prefill chunk on `stream`.
    ///
    /// `seq_len < 48` uses the CUDA recurrent kernel. `seq_len >= 48` uses
    /// the fla chunk cubins. Both write the final conv taps and the
    /// recurrent state at history index 0.
    ///
    /// # Errors
    ///
    /// A null pointer, a shape outside the 27B path, a short or unaligned
    /// workspace, or a launch rejected by the driver.
    ///
    /// # Safety
    ///
    /// Every address in `args` must be a device allocation of at least the
    /// size described on [`GdnPrefillArgs`], live until `stream` finishes.
    /// `conv_state` and `recurrent_state` are read and written. `stream`
    /// must belong to the context `load` used.
    pub unsafe fn run(&self, stream: &Stream, args: &GdnPrefillArgs) -> Result<()> {
        let seqlen = args.seqlen;
        let lay = layout(seqlen)?;
        if args.workspace == 0 || !args.workspace.is_multiple_of(ALIGN as u64) {
            return Err(Error::Launch(
                "gdn prefill: workspace is null or not 256-byte aligned".into(),
            ));
        }
        if args.workspace_bytes < lay.end {
            return Err(Error::Launch(format!(
                "gdn prefill: workspace has {} bytes, need {}",
                args.workspace_bytes, lay.end
            )));
        }
        if args.num_slots < 1 || args.slot < 0 || args.slot >= args.num_slots {
            return Err(Error::Launch(
                "gdn prefill: slot must satisfy 0 <= slot < num_slots".into(),
            ));
        }
        if args.history_stride < 1 {
            return Err(Error::Launch(
                "gdn prefill: history_stride must be positive".into(),
            ));
        }
        let qkv = need(args.qkv, "qkv")?;
        let z = need(args.z, "z")?;
        let b = need(args.b, "b")?;
        let a = need(args.a, "a")?;
        let dt_bias = need(args.dt_bias, "dt_bias")?;
        let a_log = need(args.a_log, "a_log")?;
        let conv_weight = need(args.conv_weight, "conv_weight")?;
        let norm_weight = need(args.norm_weight, "norm_weight")?;
        let conv_state = need(args.conv_state, "conv_state")?;
        let recurrent_state = need(args.recurrent_state, "recurrent_state")?;
        let out = need(args.out, "out")?;

        let base = args.workspace;
        let slot_ptr = addr(base, lay.slot.off)?;
        let mixed = addr(base, lay.mixed.off)?;
        let beta = addr(base, lay.beta.off)?;
        let g = addr(base, lay.g.off)?;
        let conv_out = addr(base, lay.conv_out.off)?;
        let core = addr(base, lay.core.off)?;
        let st = stream.handle().as_ptr();

        check_c(
            unsafe {
                ffi::berth_gdn_prefill_prepare(
                    qkv as *const c_void,
                    mixed as *mut c_void,
                    b as *const c_void,
                    a as *const c_void,
                    dt_bias as *const c_void,
                    a_log as *const c_void,
                    beta as *mut c_void,
                    g as *mut c_void,
                    slot_ptr as *mut c_void,
                    args.slot,
                    1,
                    seqlen,
                    QKV_DIM,
                    NUM_V_HEADS,
                    args.beta_scale,
                    st,
                )
            },
            "gdn_prefill_prepare",
        )?;

        self.launch_conv(
            stream,
            seqlen,
            mixed,
            conv_state,
            slot_ptr,
            conv_weight,
            conv_out,
        )?;

        if seqlen < NUM_V_HEADS {
            unsafe {
                recurrent_gated_delta_rule(
                    conv_out as *const c_void,
                    g as *const c_void,
                    beta as *const c_void,
                    recurrent_state as *mut c_void,
                    core as *mut c_void,
                    slot_ptr as *const c_void,
                    1,
                    seqlen,
                    NUM_K_HEADS,
                    NUM_V_HEADS,
                    HEAD_DIM,
                    HEAD_DIM,
                    args.history_stride,
                    st,
                )
            }?;
        } else {
            self.launch_chunk(
                stream,
                seqlen,
                &lay,
                base,
                conv_out,
                beta,
                g,
                core,
                recurrent_state,
                args.slot,
                args.history_stride,
            )?;
        }

        let rows = i32_fit(i64::from(seqlen) * i64::from(NUM_V_HEADS), "rms rows")?;
        unsafe {
            gated_rms_norm(
                core as *const c_void,
                norm_weight as *const c_void,
                out as *mut c_void,
                z as *const c_void,
                self.rms_eps,
                0.0,
                rows,
                HEAD_DIM,
                1,
                0,
                0,
                st,
            )
        }
    }
}

impl GdnPrefill {
    #[allow(clippy::too_many_arguments)]
    fn launch_conv(
        &self,
        stream: &Stream,
        seqlen: i32,
        mixed: u64,
        conv_state: u64,
        slots: u64,
        weight: u64,
        out: u64,
    ) -> Result<()> {
        let div16 = seqlen % 16 == 0;
        if seqlen <= 256 {
            let block_s = next_pow2(seqlen)?;
            let key = conv_combined_key(block_s, div16);
            let image = self.conv_image(&key)?;
            let grid = Dim3::new(1, cdiv_u32(QKV_DIM, 32)?, 1);
            launch_named(
                image,
                stream,
                grid,
                &[
                    ("x", Scalar::Ptr(mixed)),
                    ("conv_state", Scalar::Ptr(conv_state)),
                    ("slots", Scalar::Ptr(slots)),
                    ("weight", Scalar::Ptr(weight)),
                    ("bias", Scalar::Ptr(weight)),
                    ("out", Scalar::Ptr(out)),
                    ("seq_len", Scalar::I32(seqlen)),
                ],
            )
        } else {
            let output_key = if div16 {
                "conv_output_div16"
            } else {
                "conv_output"
            };
            let state_key = if div16 {
                "conv_state_div16"
            } else {
                "conv_state"
            };
            let output = self.conv_image(output_key)?;
            let state = self.conv_image(state_key)?;
            let y = cdiv_u32(QKV_DIM, 32)?;
            let z = cdiv_u32(seqlen, 256)?;
            launch_named(
                output,
                stream,
                Dim3::new(1, y, z),
                &[
                    ("x", Scalar::Ptr(mixed)),
                    ("conv_state", Scalar::Ptr(conv_state)),
                    ("slots", Scalar::Ptr(slots)),
                    ("weight", Scalar::Ptr(weight)),
                    ("bias", Scalar::Ptr(weight)),
                    ("out", Scalar::Ptr(out)),
                    ("seq_len", Scalar::I32(seqlen)),
                ],
            )?;
            launch_named(
                state,
                stream,
                Dim3::new(1, y, 1),
                &[
                    ("x", Scalar::Ptr(mixed)),
                    ("conv_state", Scalar::Ptr(conv_state)),
                    ("slots", Scalar::Ptr(slots)),
                    ("seq_len", Scalar::I32(seqlen)),
                ],
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_chunk(
        &self,
        stream: &Stream,
        seqlen: i32,
        lay: &Layout,
        base: u64,
        conv_out: u64,
        beta: u64,
        g: u64,
        core: u64,
        recurrent_state: u64,
        slot: i32,
        history_stride: i32,
    ) -> Result<()> {
        let q = addr(base, lay.q.off)?;
        let k = addr(base, lay.k.off)?;
        let v = addr(base, lay.v.off)?;
        let qn = addr(base, lay.qn.off)?;
        let kn = addr(base, lay.kn.off)?;
        let rstd = addr(base, lay.rstd.off)?;
        let g_cum = addr(base, lay.g_cum.off)?;
        let a_mat = addr(base, lay.a_mat.off)?;
        let w = addr(base, lay.w.off)?;
        let u = addr(base, lay.u.off)?;
        let v_new = addr(base, lay.v_new.off)?;
        let h = addr(base, lay.h.off)?;
        let ht = addr(base, lay.ht.off)?;
        let st = stream.handle().as_ptr();

        check_c(
            unsafe {
                ffi::berth_gdn_prefill_pack(
                    conv_out as *const c_void,
                    q as *mut c_void,
                    k as *mut c_void,
                    v as *mut c_void,
                    1,
                    seqlen,
                    QKV_DIM,
                    K_DIM,
                    V_DIM,
                    st,
                )
            },
            "gdn_prefill_pack",
        )?;
        check_c(
            unsafe { ffi::berth_gdn_prefill_zero(a_mat as *mut c_void, lay.a_mat.bytes, st) },
            "gdn_prefill_zero A",
        )?;
        check_c(
            unsafe { ffi::berth_gdn_prefill_zero(ht as *mut c_void, lay.ht.bytes, st) },
            "gdn_prefill_zero ht",
        )?;

        let t_flat = i32_fit(i64::from(seqlen) * i64::from(NUM_K_HEADS), "l2norm T")?;
        let l2_grid = Dim3::new(cdiv_u32(t_flat, self.l2_bt)?, 1, 1);
        for (x, y) in [(q, qn), (k, kn)] {
            launch_named(
                &self.l2norm,
                stream,
                l2_grid,
                &[
                    ("x", Scalar::Ptr(x)),
                    ("y", Scalar::Ptr(y)),
                    ("rstd", Scalar::Ptr(rstd)),
                    ("eps", Scalar::F32(self.l2norm_eps)),
                    ("T", Scalar::I32(t_flat)),
                ],
            )?;
        }

        let nt = cdiv_u32(seqlen, CHUNK)?;
        launch_named(
            &self.cumsum,
            stream,
            Dim3::new(nt, cdiv_u32(NUM_V_HEADS, 1)?, 1),
            &[
                ("s", Scalar::Ptr(g)),
                ("o", Scalar::Ptr(g_cum)),
                ("scale", Scalar::F32(self.rcp_ln2)),
                ("T", Scalar::I32(seqlen)),
            ],
        )?;
        let hv_grid = Dim3::new(nt, u32_fit(i64::from(NUM_V_HEADS), "HV")?, 1);
        launch_named(
            &self.kkt,
            stream,
            hv_grid,
            &[
                ("k", Scalar::Ptr(kn)),
                ("g", Scalar::Ptr(g_cum)),
                ("beta", Scalar::Ptr(beta)),
                ("A", Scalar::Ptr(a_mat)),
                ("T", Scalar::I32(seqlen)),
            ],
        )?;
        launch_named(
            &self.recompute,
            stream,
            hv_grid,
            &[
                ("k", Scalar::Ptr(kn)),
                ("v", Scalar::Ptr(v)),
                ("beta", Scalar::Ptr(beta)),
                ("w", Scalar::Ptr(w)),
                ("u", Scalar::Ptr(u)),
                ("A", Scalar::Ptr(a_mat)),
                ("g", Scalar::Ptr(g_cum)),
                ("T", Scalar::I32(seqlen)),
            ],
        )?;

        let h0_off = state_byte_offset(slot, history_stride)?;
        let h0 = recurrent_state
            .checked_add(h0_off)
            .ok_or_else(|| Error::Launch("gdn prefill: recurrent state pointer overflow".into()))?;
        launch_named(
            &self.chunk_h,
            stream,
            Dim3::new(
                cdiv_u32(HEAD_DIM, self.h_bv)?,
                u32_fit(i64::from(NUM_V_HEADS), "HV")?,
                1,
            ),
            &[
                ("k", Scalar::Ptr(kn)),
                ("v", Scalar::Ptr(u)),
                ("w", Scalar::Ptr(w)),
                ("v_new", Scalar::Ptr(v_new)),
                ("g", Scalar::Ptr(g_cum)),
                ("h", Scalar::Ptr(h)),
                ("h0", Scalar::Ptr(h0)),
                ("ht", Scalar::Ptr(ht)),
                ("T", Scalar::I32(seqlen)),
            ],
        )?;
        launch_named(
            &self.chunk_o,
            stream,
            Dim3::new(
                cdiv_u32(HEAD_DIM, self.o_bv)?,
                nt,
                u32_fit(i64::from(NUM_V_HEADS), "HV")?,
            ),
            &[
                ("q", Scalar::Ptr(qn)),
                ("k", Scalar::Ptr(kn)),
                ("v", Scalar::Ptr(v_new)),
                ("h", Scalar::Ptr(h)),
                ("g", Scalar::Ptr(g_cum)),
                ("o", Scalar::Ptr(core)),
                ("scale", Scalar::F32(self.k_scale)),
                ("T", Scalar::I32(seqlen)),
            ],
        )?;
        check_c(
            unsafe {
                ffi::berth_gdn_prefill_copy(
                    h0 as *mut c_void,
                    ht as *const c_void,
                    lay.ht.bytes,
                    st,
                )
            },
            "gdn_prefill_copy state",
        )
    }

    fn conv_image(&self, key: &str) -> Result<&Image> {
        self.conv
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, image)| image)
            .ok_or_else(|| Error::Launch(format!("gdn prefill: no cubin {key}")))
    }
}

fn conv_combined_key(block_s: i32, div16: bool) -> String {
    let mut key = format!("conv_combined_bs{block_s}");
    if div16 {
        key.push_str("_div16");
    }
    key
}

fn next_pow2(n: i32) -> Result<i32> {
    if n < 1 {
        return Err(Error::Launch("next_pow2: n must be positive".into()));
    }
    let mut p = 1i32;
    while p < n {
        p = p
            .checked_mul(2)
            .ok_or_else(|| Error::Launch("next_pow2 overflow".into()))?;
    }
    Ok(p)
}

fn state_byte_offset(slot: i32, history_stride: i32) -> Result<u64> {
    let slot = u64::try_from(slot).map_err(|_| Error::Launch("slot".into()))?;
    let hs = u64::try_from(history_stride).map_err(|_| Error::Launch("history_stride".into()))?;
    slot.checked_mul(hs)
        .and_then(|n| n.checked_mul(STATE_BYTES))
        .ok_or_else(|| Error::Launch("recurrent state offset overflows".into()))
}

fn layout(seqlen: i32) -> Result<Layout> {
    if !(1..=MAX_SEQLEN).contains(&seqlen) {
        return Err(Error::Launch(format!(
            "gdn prefill: seqlen {seqlen} is outside 1..={MAX_SEQLEN}"
        )));
    }
    let s = seqlen as usize;
    let nt = s.div_ceil(CHUNK as usize);
    let mut cursor = 0usize;
    let slot = take(&mut cursor, 4)?;
    let mixed = take(&mut cursor, bytes2(s, QKV_DIM as usize)?)?;
    let beta = take(&mut cursor, bytes2(s, NUM_V_HEADS as usize)?)?;
    let g = take(&mut cursor, bytes4(s, NUM_V_HEADS as usize)?)?;
    let conv_out = take(&mut cursor, bytes2(s, QKV_DIM as usize)?)?;
    let q = take(&mut cursor, bytes2(s, K_DIM as usize)?)?;
    let k = take(&mut cursor, bytes2(s, K_DIM as usize)?)?;
    let v = take(&mut cursor, bytes2(s, V_DIM as usize)?)?;
    let qn = take(&mut cursor, bytes2(s, K_DIM as usize)?)?;
    let kn = take(&mut cursor, bytes2(s, K_DIM as usize)?)?;
    let rstd = take(&mut cursor, bytes4(s, NUM_K_HEADS as usize)?)?;
    let g_cum = take(&mut cursor, bytes4(s, NUM_V_HEADS as usize)?)?;
    let a_mat = take(
        &mut cursor,
        bytes2(s, (NUM_V_HEADS as usize) * (CHUNK as usize))?,
    )?;
    let w = take(&mut cursor, bytes2(s, V_DIM as usize)?)?;
    let u = take(&mut cursor, bytes2(s, V_DIM as usize)?)?;
    let v_new = take(&mut cursor, bytes2(s, V_DIM as usize)?)?;
    let core = take(&mut cursor, bytes2(s, V_DIM as usize)?)?;
    let h = take(
        &mut cursor,
        bytes2(
            nt,
            (NUM_V_HEADS as usize) * (HEAD_DIM as usize) * (HEAD_DIM as usize),
        )?,
    )?;
    let ht = take(
        &mut cursor,
        (NUM_V_HEADS as usize)
            .checked_mul(HEAD_DIM as usize)
            .and_then(|n| n.checked_mul(HEAD_DIM as usize))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error::Launch("ht size overflows".into()))?,
    )?;
    Ok(Layout {
        slot,
        mixed,
        beta,
        g,
        conv_out,
        q,
        k,
        v,
        qn,
        kn,
        rstd,
        g_cum,
        a_mat,
        w,
        u,
        v_new,
        core,
        h,
        ht,
        end: cursor,
    })
}

fn bytes2(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| Error::Launch("bf16 buffer size overflows".into()))
}

fn bytes4(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| Error::Launch("f32 buffer size overflows".into()))
}

fn take(cursor: &mut usize, bytes: usize) -> Result<Region> {
    let off = align_up(*cursor)?;
    let end = off
        .checked_add(bytes)
        .ok_or_else(|| Error::Launch("workspace offset overflows".into()))?;
    *cursor = end;
    Ok(Region { off, bytes })
}

fn align_up(n: usize) -> Result<usize> {
    n.checked_add(ALIGN - 1)
        .map(|v| v & !(ALIGN - 1))
        .ok_or_else(|| Error::Launch("workspace alignment overflows".into()))
}

fn addr(base: u64, off: usize) -> Result<u64> {
    let off = u64::try_from(off).map_err(|_| Error::Launch("offset does not fit".into()))?;
    base.checked_add(off)
        .ok_or_else(|| Error::Launch("workspace pointer overflows".into()))
}

fn need(ptr: u64, name: &str) -> Result<u64> {
    if ptr == 0 {
        Err(Error::Launch(format!("gdn prefill: {name} is null")))
    } else {
        Ok(ptr)
    }
}

enum Scalar {
    Ptr(u64),
    I32(i32),
    F32(f32),
}

struct Slot {
    ptr: u64,
    bits: u32,
    kind: ParamKind,
}

fn launch_named(image: &Image, stream: &Stream, grid: Dim3, vals: &[(&str, Scalar)]) -> Result<()> {
    let mut slots = Vec::with_capacity(image.params.len());
    for param in &image.params {
        slots.push(fill_slot(param, vals)?);
    }
    let mut params: Vec<*mut c_void> = slots
        .iter_mut()
        .map(|slot| match slot.kind {
            ParamKind::Ptr => &mut slot.ptr as *mut u64 as *mut c_void,
            ParamKind::I32 | ParamKind::F32 => &mut slot.bits as *mut u32 as *mut c_void,
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

fn fill_slot(param: &Param, vals: &[(&str, Scalar)]) -> Result<Slot> {
    if param.name == "global_scratch" || param.name == "profile_scratch" {
        if param.kind != ParamKind::Ptr {
            return Err(Error::Launch(format!(
                "{} scratch parameter is not a pointer",
                param.name
            )));
        }
        return Ok(Slot {
            ptr: 0,
            bits: 0,
            kind: ParamKind::Ptr,
        });
    }
    let value = vals
        .iter()
        .find(|(name, _)| *name == param.name)
        .map(|(_, value)| value)
        .ok_or_else(|| Error::Launch(format!("manifest parameter {} has no value", param.name)))?;
    let slot = match value {
        Scalar::Ptr(ptr) => Slot {
            ptr: *ptr,
            bits: 0,
            kind: ParamKind::Ptr,
        },
        Scalar::I32(v) => Slot {
            ptr: 0,
            bits: *v as u32,
            kind: ParamKind::I32,
        },
        Scalar::F32(v) => Slot {
            ptr: 0,
            bits: v.to_bits(),
            kind: ParamKind::F32,
        },
    };
    if slot.kind != param.kind {
        return Err(Error::Launch(format!(
            "manifest parameter {} has the wrong type",
            param.name
        )));
    }
    Ok(slot)
}

fn cdiv_u32(n: i32, d: i32) -> Result<u32> {
    if n < 1 || d < 1 {
        return Err(Error::Launch("cdiv: arguments must be positive".into()));
    }
    u32_fit((i64::from(n) + i64::from(d) - 1) / i64::from(d), "grid")
}

fn i32_fit(v: i64, what: &str) -> Result<i32> {
    i32::try_from(v).map_err(|_| Error::Launch(format!("{what} does not fit in i32")))
}

fn u32_fit(v: i64, what: &str) -> Result<u32> {
    u32::try_from(v).map_err(|_| Error::Launch(format!("{what} does not fit in u32")))
}

fn check_c(rc: c_int, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    Err(Error::Launch(format!("{what}: {}", last_error())))
}

fn last_error() -> String {
    let ptr = unsafe { ffi::berth_gdn_prefill_last_error() };
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

mod ffi {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    unsafe extern "C" {
        pub fn berth_gdn_prefill_prepare(
            qkv: *const c_void,
            mixed_qkv: *mut c_void,
            b: *const c_void,
            a: *const c_void,
            dt_bias: *const c_void,
            a_log: *const c_void,
            beta: *mut c_void,
            g: *mut c_void,
            slot_dev: *mut c_void,
            slot_host: c_int,
            batch: c_int,
            seqlen: c_int,
            features: c_int,
            heads: c_int,
            beta_scale: f32,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_gdn_prefill_pack(
            conv: *const c_void,
            q: *mut c_void,
            k: *mut c_void,
            v: *mut c_void,
            batch: c_int,
            seqlen: c_int,
            features: c_int,
            k_dim: c_int,
            v_dim: c_int,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_gdn_prefill_zero(ptr: *mut c_void, bytes: usize, stream: *mut c_void)
            -> c_int;

        pub fn berth_gdn_prefill_copy(
            dst: *mut c_void,
            src: *const c_void,
            bytes: usize,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_gdn_prefill_last_error() -> *const c_char;
    }
}

fn f32_bits(obj: &Json, key: &str) -> Result<f32> {
    let v = obj
        .get(key)
        .and_then(Json::as_i64)
        .ok_or_else(|| Error::Launch(format!("f32_bits.{key} missing")))?;
    let bits =
        u32::try_from(v).map_err(|_| Error::Launch(format!("f32_bits.{key} is not a u32")))?;
    Ok(f32::from_bits(bits))
}

fn const_i32(kernels: &Json, key: &str, field: &str) -> Result<i32> {
    let c = kernels
        .get(key)
        .and_then(|k| k.get("constexprs"))
        .ok_or_else(|| Error::Launch(format!("{key} constexprs missing")))?;
    need_i32(c, field)
}

fn load_image(ctx: &Context, dir: &Path, kernels: &Json, key: &str, grid: &str) -> Result<Image> {
    let record = kernels
        .get(key)
        .ok_or_else(|| Error::Launch(format!("manifest has no {key}")))?;
    let got = record
        .get("grid")
        .and_then(Json::as_str)
        .ok_or_else(|| Error::Launch(format!("{key}: grid missing")))?;
    if got != grid {
        return Err(Error::Launch(format!("{key}: grid {got}, want {grid}")));
    }
    load_record(ctx, dir, key, record)
}

fn load_conv(
    ctx: &Context,
    dir: &Path,
    kernels: &Json,
    key: &str,
    grid: &str,
    block_s: i32,
) -> Result<Image> {
    let image = load_image(ctx, dir, kernels, key, grid)?;
    if image.num_warps != 4 {
        return Err(Error::Launch(format!(
            "{key}: num_warps {}, the fork passes 4",
            image.num_warps
        )));
    }
    let record = kernels.get(key).ok_or_else(|| Error::Launch(key.into()))?;
    let c = record
        .get("constexprs")
        .ok_or_else(|| Error::Launch(format!("{key} constexprs missing")))?;
    require_i32(c, "dim", QKV_DIM)?;
    require_i32(c, "state_size", CONV_STATE)?;
    require_i32(c, "conv_kernel_size", CONV_STATE)?;
    require_i32(c, "BLOCK_D", 32)?;
    // Triton folds an integer argument whose value is 1 into a constexpr.
    // That is only the combined kernel for a 1-token chunk. Every longer
    // cubin keeps `seq_len` as a runtime i32, which is what the launch passes.
    let baked_seq = match c.get("seq_len") {
        Some(value) if !matches!(value.kind, JsonKind::Null) => Some(need_i32(c, "seq_len")?),
        _ => None,
    };
    if let Some(seq) = baked_seq {
        if grid != "conv_combined" || block_s != 1 || seq != 1 {
            return Err(Error::Launch(format!(
                "{key}: seq_len constexpr {seq} is only valid for the combined BLOCK_S=1 cubin"
            )));
        }
    }
    let seq_name: &[&str] = if baked_seq.is_some() {
        &[]
    } else {
        &["seq_len"]
    };
    if grid == "conv_state" {
        // The state kernel has `history` and no bias/output constexprs.
        require_bool(c, "history", false)?;
        require_i32(c, "BLOCK_STATE", CONV_STATE)?;
        allow_only(
            c,
            &[
                "dim",
                "state_size",
                "conv_kernel_size",
                "history",
                "seq_len",
                "BLOCK_D",
                "BLOCK_STATE",
            ],
        )?;
        require_names(
            &image,
            &concat_names(
                &["x", "conv_state", "slots"],
                seq_name,
                &["global_scratch", "profile_scratch"],
            ),
            &[],
        )?;
    } else {
        // Output has no `history`. Combined has `history` and `BLOCK_STATE`.
        // `has_bias` is false; the fork still passes the weight pointer as the
        // dummy bias, and Triton may keep that pointer in the signature.
        require_bool(c, "has_bias", false)?;
        require_bool(c, "transpose_output", true)?;
        require_i32(c, "BLOCK_S", block_s)?;
        require_i32(c, "BLOCK_K", CONV_STATE)?;
        if grid == "conv_combined" {
            require_bool(c, "history", false)?;
            require_i32(c, "BLOCK_STATE", CONV_STATE)?;
        }
        allow_only(
            c,
            &[
                "dim",
                "state_size",
                "conv_kernel_size",
                "history",
                "has_bias",
                "transpose_output",
                "seq_len",
                "BLOCK_D",
                "BLOCK_S",
                "BLOCK_K",
                "BLOCK_STATE",
            ],
        )?;
        require_names(
            &image,
            &concat_names(
                &["x", "conv_state", "slots", "weight", "out"],
                seq_name,
                &["global_scratch", "profile_scratch"],
            ),
            &["bias"],
        )?;
    }
    Ok(image)
}

fn concat_names<'a>(head: &[&'a str], mid: &[&'a str], tail: &[&'a str]) -> Vec<&'a str> {
    let mut names = Vec::with_capacity(head.len() + mid.len() + tail.len());
    names.extend_from_slice(head);
    names.extend_from_slice(mid);
    names.extend_from_slice(tail);
    names
}

fn check_l2norm(image: &Image) -> Result<()> {
    require_names(
        image,
        &[
            "x",
            "y",
            "rstd",
            "eps",
            "T",
            "global_scratch",
            "profile_scratch",
        ],
        &[],
    )
}

fn check_cumsum(image: &Image) -> Result<()> {
    require_names(
        image,
        &["s", "o", "scale", "T", "global_scratch", "profile_scratch"],
        &[],
    )
}

fn check_kkt(image: &Image) -> Result<()> {
    require_names(
        image,
        &[
            "k",
            "g",
            "beta",
            "A",
            "T",
            "global_scratch",
            "profile_scratch",
        ],
        &[],
    )
}

fn check_recompute(image: &Image) -> Result<()> {
    require_names(
        image,
        &[
            "k",
            "v",
            "beta",
            "w",
            "u",
            "A",
            "g",
            "T",
            "global_scratch",
            "profile_scratch",
        ],
        &[],
    )
}

fn check_chunk_h(image: &Image) -> Result<()> {
    require_names(
        image,
        &[
            "k",
            "v",
            "w",
            "v_new",
            "g",
            "h",
            "h0",
            "ht",
            "T",
            "global_scratch",
            "profile_scratch",
        ],
        &[],
    )
}

fn check_chunk_o(image: &Image) -> Result<()> {
    require_names(
        image,
        &[
            "q",
            "k",
            "v",
            "h",
            "g",
            "o",
            "scale",
            "T",
            "global_scratch",
            "profile_scratch",
        ],
        &[],
    )
}

fn require_names(image: &Image, required: &[&str], optional: &[&str]) -> Result<()> {
    for name in required {
        if !image.params.iter().any(|p| p.name == *name) {
            return Err(Error::Launch(format!(
                "{}: manifest is missing parameter {name}",
                image.name
            )));
        }
    }
    for param in &image.params {
        if !required.contains(&param.name.as_str()) && !optional.contains(&param.name.as_str()) {
            return Err(Error::Launch(format!(
                "{}: unexpected parameter {}",
                image.name, param.name
            )));
        }
    }
    Ok(())
}

fn load_record(ctx: &Context, dir: &Path, key: &str, record: &Json) -> Result<Image> {
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
    let ctas = record.get("num_ctas").and_then(Json::as_i64).unwrap_or(1);
    if ctas != 1 {
        return Err(Error::Launch(format!("{key}: num_ctas {ctas}")));
    }
    for scratch in ["global_scratch_size", "profile_scratch_size"] {
        if let Some(size) = record.get(scratch).and_then(Json::as_i64) {
            if size != 0 {
                return Err(Error::Launch(format!(
                    "{key}: {scratch} is {size}; the launcher passes a null pointer"
                )));
            }
        }
    }
    let params = param_list(key, record)?;
    let path = dir.join(file);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) if !bytes.is_empty() => bytes,
        _ => {
            return Err(Error::Launch(format!(
                "gdn prefill cubin is missing at {}. Run: {TRITON_FLA_BUILD}",
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
    if key.starts_with("l2norm") || key.starts_with("chunk_") || key.starts_with("recompute") {
        check_fla_constexprs(key, record)?;
    }
    Ok(Image {
        function,
        name: symbol.to_string(),
        shared,
        num_warps,
        params,
    })
}

fn check_fla_constexprs(key: &str, record: &Json) -> Result<()> {
    let c = record
        .get("constexprs")
        .ok_or_else(|| Error::Launch(format!("{key} constexprs missing")))?;
    if key.starts_with("l2norm") {
        require_i32(c, "D", HEAD_DIM)?;
        require_i32(c, "BD", HEAD_DIM)?;
        require_i32(c, "NB", 1)?;
        allow_only(c, &["D", "BD", "NB", "BT"])?;
    } else if key.contains("cumsum") {
        require_i32(c, "B", 1)?;
        require_i32(c, "H", NUM_V_HEADS)?;
        require_i32(c, "BT", CHUNK)?;
        require_bool(c, "REVERSE", false)?;
        require_bool(c, "HAS_SCALE", true)?;
        require_bool(c, "IS_VARLEN", false)?;
        require_bool(c, "HEAD_FIRST", false)?;
        allow_only(
            c,
            &[
                "B",
                "H",
                "BT",
                "REVERSE",
                "HAS_SCALE",
                "IS_VARLEN",
                "HEAD_FIRST",
            ],
        )?;
    } else if key.contains("kkt") {
        require_i32(c, "H", NUM_K_HEADS)?;
        require_i32(c, "HV", NUM_V_HEADS)?;
        require_i32(c, "K", HEAD_DIM)?;
        require_i32(c, "BT", CHUNK)?;
        require_i32(c, "BC", 16)?;
        require_bool(c, "USE_G", true)?;
        require_bool(c, "IS_VARLEN", false)?;
        allow_only(c, &["H", "HV", "K", "BT", "BC", "BK", "USE_G", "IS_VARLEN"])?;
    } else if key.starts_with("recompute") {
        require_i32(c, "H", NUM_K_HEADS)?;
        require_i32(c, "HV", NUM_V_HEADS)?;
        require_i32(c, "K", HEAD_DIM)?;
        require_i32(c, "V", HEAD_DIM)?;
        require_i32(c, "BT", CHUNK)?;
        require_i32(c, "BK", 64)?;
        require_i32(c, "BV", 64)?;
        require_bool(c, "USE_G", true)?;
        require_bool(c, "IS_VARLEN", false)?;
        allow_only(
            c,
            &["H", "HV", "K", "V", "BT", "BK", "BV", "USE_G", "IS_VARLEN"],
        )?;
    } else if key.contains("kernel_h") {
        require_i32(c, "H", NUM_K_HEADS)?;
        require_i32(c, "HV", NUM_V_HEADS)?;
        require_i32(c, "K", HEAD_DIM)?;
        require_i32(c, "V", HEAD_DIM)?;
        require_i32(c, "BT", CHUNK)?;
        require_bool(c, "USE_G", true)?;
        require_bool(c, "USE_GK", false)?;
        require_bool(c, "USE_INITIAL_STATE", true)?;
        require_bool(c, "STORE_FINAL_STATE", true)?;
        require_bool(c, "SAVE_NEW_VALUE", true)?;
        require_bool(c, "STATE_V_FIRST", false)?;
        require_bool(c, "IS_VARLEN", false)?;
        allow_only(
            c,
            &[
                "H",
                "HV",
                "K",
                "V",
                "BT",
                "BV",
                "USE_G",
                "USE_GK",
                "USE_INITIAL_STATE",
                "STORE_FINAL_STATE",
                "SAVE_NEW_VALUE",
                "STATE_V_FIRST",
                "IS_VARLEN",
            ],
        )?;
    } else if key.contains("kernel_o") {
        require_i32(c, "H", NUM_K_HEADS)?;
        require_i32(c, "HV", NUM_V_HEADS)?;
        require_i32(c, "K", HEAD_DIM)?;
        require_i32(c, "V", HEAD_DIM)?;
        require_i32(c, "BT", CHUNK)?;
        require_bool(c, "USE_G", true)?;
        require_bool(c, "USE_G_GAMMA", false)?;
        require_bool(c, "STATE_V_FIRST", false)?;
        require_bool(c, "IS_VARLEN", false)?;
        allow_only(
            c,
            &[
                "H",
                "HV",
                "K",
                "V",
                "BT",
                "BK",
                "BV",
                "USE_G",
                "USE_G_GAMMA",
                "STATE_V_FIRST",
                "IS_VARLEN",
            ],
        )?;
    }
    Ok(())
}

fn allow_only(obj: &Json, allowed: &[&str]) -> Result<()> {
    let JsonKind::Object(pairs) = &obj.kind else {
        return Err(Error::Launch("constexprs are not an object".into()));
    };
    for (key, value) in pairs {
        if matches!(value.kind, JsonKind::Null) {
            continue;
        }
        if !allowed.contains(&key.as_str()) {
            return Err(Error::Launch(format!(
                "unexpected constexpr {key}; the 27B path does not set it"
            )));
        }
    }
    Ok(())
}

fn require_i32(obj: &Json, key: &str, want: i32) -> Result<()> {
    let got = need_i32(obj, key)?;
    if got != want {
        return Err(Error::Launch(format!(
            "constexpr {key} is {got}, want {want}"
        )));
    }
    Ok(())
}

fn require_bool(obj: &Json, key: &str, want: bool) -> Result<()> {
    let got = obj
        .get(key)
        .and_then(Json::as_bool)
        .ok_or_else(|| Error::Launch(format!("constexpr {key} missing")))?;
    if got != want {
        return Err(Error::Launch(format!(
            "constexpr {key} is {got}, want {want}"
        )));
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

fn param_list(key: &str, record: &Json) -> Result<Vec<Param>> {
    let list = record
        .get("parameters")
        .and_then(Json::as_array)
        .ok_or_else(|| Error::Launch(format!("{key}: parameters missing")))?;
    let mut out = Vec::with_capacity(list.len());
    for item in list {
        let name = item
            .get("name")
            .and_then(Json::as_str)
            .ok_or_else(|| Error::Launch(format!("{key}: parameter name missing")))?;
        let ty = item
            .get("type")
            .and_then(Json::as_str)
            .ok_or_else(|| Error::Launch(format!("{key}: parameter {name} has no type")))?;
        let kind = if ty.starts_with('*') {
            ParamKind::Ptr
        } else if ty == "i32" || ty == "u32" {
            ParamKind::I32
        } else if ty == "fp32" || ty == "f32" {
            ParamKind::F32
        } else {
            return Err(Error::Launch(format!(
                "{key}: parameter {name} has unsupported type {ty}"
            )));
        };
        out.push(Param {
            name: name.to_string(),
            kind,
        });
    }
    Ok(out)
}

fn cuda_err(err: berth_cuda::Error) -> Error {
    Error::Launch(err.to_string())
}

// Manifest JSON. `json.dumps` output: objects, arrays, strings, ints, bools, null.

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
    fn scratch_rejects_out_of_range() {
        assert!(scratch_bytes(0).is_err());
        assert!(scratch_bytes(-1).is_err());
        assert!(scratch_bytes(4097).is_err());
        assert!(scratch_bytes(1).is_ok());
        assert!(scratch_bytes(4096).is_ok());
    }

    #[test]
    fn scratch_grows_and_regions_do_not_overlap() {
        let small = scratch_bytes(47).unwrap();
        let mid = scratch_bytes(300).unwrap();
        let large = scratch_bytes(1499).unwrap();
        assert!(small < mid && mid < large);
        assert!(large > 150_000_000 && large < 400_000_000);
        let lay = layout(1499).unwrap();
        let regions = [
            lay.slot,
            lay.mixed,
            lay.beta,
            lay.g,
            lay.conv_out,
            lay.q,
            lay.k,
            lay.v,
            lay.qn,
            lay.kn,
            lay.rstd,
            lay.g_cum,
            lay.a_mat,
            lay.w,
            lay.u,
            lay.v_new,
            lay.core,
            lay.h,
            lay.ht,
        ];
        for region in &regions {
            assert!(region.off.is_multiple_of(ALIGN));
            assert!(region.bytes > 0);
            assert!(region.off + region.bytes <= lay.end);
        }
        let mut spans: Vec<(usize, usize)> =
            regions.iter().map(|r| (r.off, r.off + r.bytes)).collect();
        spans.sort_unstable();
        for pair in spans.windows(2) {
            assert!(pair[0].1 <= pair[1].0);
        }
    }
}

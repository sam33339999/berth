//! Gated DeltaNet decode launches on raw device pointers.
//!
//! One token, one batch. These are the five kernels `docs/kernels.md` records
//! for each of the 48 `linear_attention` layers, in the order
//! `BC_GatedDeltaNetSplit::run_bszN` runs them after the fused qkv/z projection
//! (that projection is `exl3_mgemm`, not this module) and before `out_proj`
//! (`exl3_linear`):
//!
//! 1. [`gdn_ba_gemv`] — `gdn_ba_gemv_kernel`
//! 2. [`gated_delta_net_fused_op_3`] — `gated_delta_net_fused_op_3_kernel<bf16>`
//! 3. [`conv1d_update`] — `conv1d_update_kernel<1,0>`
//! 4. [`recurrent_gated_delta_rule`] — `cuda_recurrent_gated_delta_rule_kernel_128<0,4,0>`
//! 5. [`gated_rms_norm`] — `gated_rms_norm_kernel<32,half,bf16,float>`
//!
//! The device code is the fork's (`exllamav3_ext/gdn.cu`, `norm.cu`). Those
//! functions are not in a torch-free header, so `csrc/gdn/gdn_launch.cu` carries
//! a verbatim copy of the cited line ranges and launches them the way
//! `gdn_ba_gemv_gr`, `gated_delta_net_fused_op_3_gr`,
//! `cuda_causal_conv1d_update_gr`, `cuda_recurrent_gated_delta_rule_gr`, and
//! `gated_rms_norm_gr` do. Dynamic shared memory is 0 for all five. A shape
//! that would select a different template instance is rejected before any CUDA
//! call.
//!
//! 27B shapes (`text_config`): 16 K heads, 48 V heads, head dim 128, conv
//! kernel 4, hidden 5120. `F = 2 * 16 * 128 + 48 * 128 = 10240`. `rms_norm_eps`
//! is `1e-6`. The norm gate is silu (`output_gate_type` `"swish"`). Checkpoint
//! prefix `model.language_model.layers.{i}.linear_attn`.
//!
//! Pointers are device addresses. `stream` is a `cudaStream_t` (`null` is the
//! default stream). [`last_launch`] names the kernel the last successful launch
//! on this thread queued.

use std::ffi::{c_char, c_int, c_void, CStr};

use crate::{Error, Result};

/// One conv-state rewind, matching `gdn.cuh` `ConvRewindJob`.
///
/// `src` and `dst` are device addresses. `stride` is the per-channel element
/// stride (`conv_state.stride(1)`), not a byte stride. The kernel copies
/// `cdim` bf16 values, `stride` elements apart, from `src` onto `dst`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvRewindJob {
    pub src: u64,
    pub dst: u64,
    pub dim: i32,
    pub cdim: i32,
    pub stride: i32,
}

/// One recurrent-state rewind, matching `gdn.cuh` `StateRewindJob`.
///
/// Flat fp32 copy of `num_elements` contiguous elements. The fork requires
/// `num_elements` to be a multiple of 4 (the kernel moves `float4`s).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateRewindJob {
    pub src: u64,
    pub dst: u64,
    pub num_elements: i64,
}

unsafe extern "C" {
    fn berth_gdn_ba_gemv(
        x: *const c_void,
        w_t: *const c_void,
        bias: *const c_void,
        y: *mut c_void,
        rows: c_int,
        k: c_int,
        n: c_int,
        stream: *mut c_void,
    ) -> c_int;

    fn berth_gdn_conv1d_update(
        x: *const c_void,
        conv_state: *mut c_void,
        slots: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        out: *mut c_void,
        bsz: c_int,
        dim: c_int,
        seqlen: c_int,
        state_size: c_int,
        k: c_int,
        stream: *mut c_void,
    ) -> c_int;

    fn berth_gdn_fused_op_3(
        qkv: *const c_void,
        ba: *const c_void,
        dt_bias: *const c_void,
        a_log: *const c_void,
        mixed_qkv: *mut c_void,
        beta: *mut c_void,
        g: *mut c_void,
        b: c_int,
        s: c_int,
        f: c_int,
        h: c_int,
        beta_scale: f32,
        stream: *mut c_void,
    ) -> c_int;

    fn berth_gdn_recurrent(
        mixed_qkv: *const c_void,
        g: *const c_void,
        beta: *const c_void,
        recurrent_state: *mut c_void,
        core_attn_out: *mut c_void,
        slots: *const c_void,
        bsz: c_int,
        seqlen: c_int,
        num_k_heads: c_int,
        num_v_heads: c_int,
        k_head_dim: c_int,
        v_head_dim: c_int,
        history_stride: c_int,
        stream: *mut c_void,
    ) -> c_int;

    fn berth_gdn_gated_rms_norm(
        x: *const c_void,
        w: *const c_void,
        y: *mut c_void,
        g: *const c_void,
        epsilon: f32,
        constant_bias: f32,
        rows: c_int,
        dim: c_int,
        w_groups: c_int,
        gate_first: c_int,
        gate_act: c_int,
        stream: *mut c_void,
    ) -> c_int;

    fn berth_gdn_conv1d_update_history(
        x: *const c_void,
        conv_state: *mut c_void,
        slots: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        out: *mut c_void,
        bsz: c_int,
        dim: c_int,
        seqlen: c_int,
        state_size: c_int,
        k: c_int,
        history: c_int,
        stream: *mut c_void,
    ) -> c_int;

    fn berth_gdn_recurrent_history(
        mixed_qkv: *const c_void,
        g: *const c_void,
        beta: *const c_void,
        recurrent_state: *mut c_void,
        core_attn_out: *mut c_void,
        slots: *const c_void,
        bsz: c_int,
        seqlen: c_int,
        num_k_heads: c_int,
        num_v_heads: c_int,
        k_head_dim: c_int,
        v_head_dim: c_int,
        history_stride: c_int,
        history: c_int,
        stream: *mut c_void,
    ) -> c_int;

    fn berth_gdn_batched_conv_rewind(
        jobs: *const ConvRewindJob,
        n: c_int,
        stream: *mut c_void,
    ) -> c_int;

    fn berth_gdn_batched_state_rewind(
        jobs: *const StateRewindJob,
        n: c_int,
        stream: *mut c_void,
    ) -> c_int;

    fn berth_gdn_rewind_last_error() -> *const c_char;
    fn berth_gdn_rewind_last_launch() -> *const c_char;

    fn berth_gdn_last_error() -> *const c_char;
    fn berth_gdn_last_launch() -> *const c_char;
}

fn check(rc: c_int, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let message = last_error();
    let message = if message.is_empty() {
        format!("{what} launch failed ({rc})")
    } else {
        message
    };
    Err(Error::Launch(message))
}

/// fp16 GEMV `y = x @ w_t.T`, accumulated and stored as fp32.
///
/// Fork: `gdn_ba_gemv` / `gdn_ba_gemv_gr` (`gdn.cu`). One warp per output
/// feature, eight warps per block. Not templated.
///
/// * `x` — fp16 `[rows, k]`, row-major. On the 27B decode `rows` is 1 and `k`
///   is `hidden_size` (5120): the layer input, not a checkpoint tensor.
/// * `w_t` — fp16 `[n, k]`. `n` is `2 * num_v_heads` (96). Rows `0..num_v_heads`
///   are b and rows `num_v_heads..n` are a. Checkpoint tensors
///   `linear_attn.in_proj_b.weight` and `linear_attn.in_proj_a.weight` are fp16
///   `[48, 5120]` (out, in). The runtime `LinearFP16` weight is the transpose
///   (`x @ W`); the module then does `cat([b, a], dim=-1).T`, which is this
///   `[96, 5120]` matrix.
/// * `bias` — fp16 `[n]`, or null. The 27B checkpoint has no `in_proj_a.bias`
///   and no `in_proj_b.bias`.
/// * `y` — fp32 `[rows, n]`. Column `0..H` is b and `H..2H` is a, which is the
///   `ba` argument of [`gated_delta_net_fused_op_3`].
///
/// `k` must be even (the kernel loads `half2`). For the 27B shape the launch is
/// `gdn_ba_gemv_kernel grid=12x1x1 block=256x1x1 smem=0`.
///
/// # Errors
///
/// A null `x`, `w_t`, or `y`, a non-positive size, or an odd `k` is
/// [`Error::Launch`] and does not call CUDA. A CUDA failure is the same error.
///
/// # Safety
///
/// Every non-null pointer must be a device allocation of at least the bytes
/// implied by the shape, aligned for `half2` loads, and live until `stream`
/// finishes the launch. `bias` and `stream` may be null. The shim does not
/// read the bytes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gdn_ba_gemv(
    x: *const c_void,
    w_t: *const c_void,
    bias: *const c_void,
    y: *mut c_void,
    rows: i32,
    k: i32,
    n: i32,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe { berth_gdn_ba_gemv(x, w_t, bias, y, rows, k, n, stream) };
    check(rc, "gdn_ba_gemv")
}

/// Causal conv1d update, sigmoid activation, no history writeback.
///
/// Fork: `cuda_causal_conv1d_update` / `cuda_causal_conv1d_update_gr` with
/// `activation=true` and `history=false`, which is `conv1d_update_kernel<1,0>`
/// (`gdn.cu`). The sigmoid is `_sigmoid_fast_exp`, not silu. The other
/// `(activation, history)` instances are not linked here.
/// [`conv1d_update_history`] also links `<1,1>`.
///
/// * `x` — bf16 `[bsz, dim, seqlen]`. This is `mixed_qkv` from
///   [`gated_delta_net_fused_op_3`]: `[B, F, S]` with `F = 10240` and `S = 1`.
/// * `conv_state` — bf16 `[num_slots, dim, state_size]`, updated in place.
///   The cache tensor is `GDNLayerState.conv_state`,
///   `[max_batch, fdim_qkv, conv_kernel_size + max_history]`. With history off,
///   the kernel reads and writes only the first `k` taps of each channel
///   (`state_d[0..k]`); `state_size` is the stride and must be `>= k`. The
///   live window is that prefix. The 27B checkpoint weight is
///   `linear_attn.conv1d.weight`, bf16 `[10240, 1, 4]`, squeezed to
///   `[10240, 4]`. There is no `conv1d.bias`.
/// * `slots` — int32 `[bsz]`, or null. Null means slot `b` for batch row `b`.
/// * `weight` — bf16 `[dim, k]`, the squeezed conv weight.
/// * `bias` — bf16 `[dim]`, or null.
/// * `out` — bf16 `[bsz, seqlen, dim]`. This is the recurrent kernel's
///   `mixed_qkv` layout: per token, Q heads, then K heads, then V heads.
///
/// `k` is the conv kernel size, 1..=16. The 27B value is 4. For `dim = 10240`
/// and `bsz = 1` the launch is
/// `conv1d_update_kernel<1,0> grid=40x1x1 block=256x1x1 smem=0`.
///
/// # Errors
///
/// A null required pointer, a non-positive `bsz` / `dim` / `seqlen`, `k`
/// outside 1..=16, or `state_size < k` is [`Error::Launch`] and does not call
/// CUDA.
///
/// # Safety
///
/// Non-null pointers must be device allocations of at least the stated shapes
/// and must live until `stream` finishes. `conv_state` is read and written.
/// `slots`, `bias`, and `stream` may be null.
#[allow(clippy::too_many_arguments)]
pub unsafe fn conv1d_update(
    x: *const c_void,
    conv_state: *mut c_void,
    slots: *const c_void,
    weight: *const c_void,
    bias: *const c_void,
    out: *mut c_void,
    bsz: i32,
    dim: i32,
    seqlen: i32,
    state_size: i32,
    k: i32,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe {
        berth_gdn_conv1d_update(
            x, conv_state, slots, weight, bias, out, bsz, dim, seqlen, state_size, k, stream,
        )
    };
    check(rc, "conv1d_update")
}

/// Transpose/cast qkv to bf16 and compute beta and g from the packed b/a GEMV.
///
/// Fork: `gated_delta_net_fused_op_3_gr` (`gdn.cu`). `a_log` is bf16, so this
/// is `gated_delta_net_fused_op_3_kernel<bf16>`. The fp32 `a_log` instance is
/// not linked. There is no Python binding for this op; the module's eager path
/// is a transpose plus `gated_delta_net_fused_op_2`, which applies the same
/// per-element math.
///
/// * `qkv` — f32 `[B, S, F]`, the `in_proj_qkv` output (EXL3, not a checkpoint
///   tensor). `F = 10240` on the 27B.
/// * `ba` — f32 `[B, S, 2H]`, the [`gdn_ba_gemv`] output. `H` is `num_v_heads`
///   (48). The first `H` columns are b, the second `H` are a.
/// * `dt_bias` — bf16 `[H]`, checkpoint `linear_attn.dt_bias`.
/// * `a_log` — bf16 `[H]`, checkpoint `linear_attn.A_log`.
/// * `mixed_qkv` — bf16 `[B, F, S]`, `trunc_bf16` (`__float2bfloat16_rn`) of
///   `qkv[b, s, f]`. This is the conv kernel's `x`.
/// * `beta` — bf16 `[B, S, H]`, `sigmoid(b) * beta_scale` rounded to bf16
///   nearest-even.
/// * `g` — f32 `[B, S, H]`, `-softplus(a + dt_bias) * exp(a_log)` with the
///   fork's device `softplus` (threshold 20, `__expf` / `log1pf`).
///
/// `beta_scale` is the module's `beta_scale` (1.0 unless overridden). For
/// `B = S = 1`, `F = 10240`, `H = 48` the launch is
/// `gated_delta_net_fused_op_3_kernel<bf16> grid=41x1x1 block=256x1x1 smem=0`.
///
/// # Errors
///
/// A null pointer, a non-positive size, or a `B * S * (F + H)` that does not
/// fit in `int32` (the kernel's index type) is [`Error::Launch`] and does not
/// call CUDA.
///
/// # Safety
///
/// Non-null pointers must be device allocations of at least the stated shapes
/// and must live until `stream` finishes. `stream` may be null.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gated_delta_net_fused_op_3(
    qkv: *const c_void,
    ba: *const c_void,
    dt_bias: *const c_void,
    a_log: *const c_void,
    mixed_qkv: *mut c_void,
    beta: *mut c_void,
    g: *mut c_void,
    b: i32,
    s: i32,
    f: i32,
    h: i32,
    beta_scale: f32,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe {
        berth_gdn_fused_op_3(
            qkv, ba, dt_bias, a_log, mixed_qkv, beta, g, b, s, f, h, beta_scale, stream,
        )
    };
    check(rc, "gated_delta_net_fused_op_3")
}

/// Gated delta rule, in place on the current recurrent state.
///
/// Fork: `cuda_recurrent_gated_delta_rule` / `cuda_recurrent_gated_delta_rule_gr`
/// with `history=false`, head dim 128, and `v_split = 4`. That is
/// `cuda_recurrent_gated_delta_rule_kernel_128<0,4,0>` (`gdn.cu`):
/// `save_history=false`, `V_SPLIT=4`, `CHANNELWISE=false`. KDA (per-channel
/// decay) is not linked. The history-saving instance is
/// [`recurrent_gated_delta_rule_history`].
///
/// * `mixed_qkv` — bf16 `[bsz, seqlen, F]`, the conv output. Per token the
///   layout is Q (`num_k_heads * k_head_dim`), then K (same), then V
///   (`num_v_heads * v_head_dim`).
/// * `g` — f32 `[bsz, seqlen, num_v_heads]`, from [`gated_delta_net_fused_op_3`].
/// * `beta` — bf16 `[bsz, seqlen, num_v_heads]`, from the same kernel.
/// * `recurrent_state` — f32
///   `[num_slots, history_stride, num_v_heads, k_head_dim, v_head_dim]`,
///   updated in place. The cache tensor is `GDNLayerState.recurrent_state`.
///   With history off the kernel reads and writes history index 0 only.
///   Within a head the layout is k-major: element `(k, v)` is at
///   `k * v_head_dim + v`. `history_stride` is `recurrent_state.size(1)`,
///   which is `max_history + 1` in the real cache (at least 5 while drafting).
///   Slot 0's live state is the base of the buffer, so a buffer that stores
///   only the current state passes `history_stride = 1`. A non-zero slot
///   address depends on `history_stride`.
/// * `core_attn_out` — bf16 `[bsz, seqlen, num_v_heads, v_head_dim]`. Values
///   are `__float2bfloat16_rz` of `v_out * scale`, with
///   `scale = 1/sqrt(k_head_dim)` computed in the launcher (host `sqrtf`).
///   Q and K are L2-normalized inside the kernel with epsilon `1e-6`, which
///   is not the RMSNorm epsilon.
/// * `slots` — int32 `[bsz]`, or null (slot = batch index).
///
/// The kernel's trailing `D` pointer is nullptr. GDN has no Mamba2 skip.
/// `bsz` must be 1, both head dims must be 128, and `num_v_heads` must be a
/// positive multiple of `num_k_heads` and at most 64. Those are the conditions
/// under which the fork picks `<0,4,0>`. The 27B launch is
/// `cuda_recurrent_gated_delta_rule_kernel_128<0,4,0> grid=1x48x4 block=128x4x1 smem=0`.
///
/// # Errors
///
/// A null required pointer or a shape outside that instance is
/// [`Error::Launch`] and does not call CUDA.
///
/// # Safety
///
/// Non-null pointers must be device allocations of at least the stated shapes
/// and must live until `stream` finishes. `recurrent_state` is read and
/// written. `slots` and `stream` may be null.
#[allow(clippy::too_many_arguments)]
pub unsafe fn recurrent_gated_delta_rule(
    mixed_qkv: *const c_void,
    g: *const c_void,
    beta: *const c_void,
    recurrent_state: *mut c_void,
    core_attn_out: *mut c_void,
    slots: *const c_void,
    bsz: i32,
    seqlen: i32,
    num_k_heads: i32,
    num_v_heads: i32,
    k_head_dim: i32,
    v_head_dim: i32,
    history_stride: i32,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe {
        berth_gdn_recurrent(
            mixed_qkv,
            g,
            beta,
            recurrent_state,
            core_attn_out,
            slots,
            bsz,
            seqlen,
            num_k_heads,
            num_v_heads,
            k_head_dim,
            v_head_dim,
            history_stride,
            stream,
        )
    };
    check(rc, "recurrent_gated_delta_rule")
}

/// Gated RMSNorm: `y = rms(x) * w * silu(g)` for the 27B norm.
///
/// Fork: `gated_rms_norm` / `gated_rms_norm_gr` (`norm.cu`). `x` is bf16, `w`
/// is bf16, `y` is fp16, `g` is f32, and `dim <= 256`, so the instance is
/// `gated_rms_norm_kernel<32,half,bf16,float>`. The 1024-thread instance used
/// for `dim > 256` is not linked. Call the extension, not `forward_torch`:
/// the torch fallback is not bit-exact.
///
/// * `x` — bf16 `[rows, dim]`, row-major. This is `core_attn_out` viewed as
///   `rows = B * S * num_v_heads` and `dim = v_head_dim` (48 by 128 on the
///   27B decode). The bytes match a contiguous `[B, S, Nv, Hv]` tensor.
/// * `w` — bf16 `[w_groups, dim]`, row-major. Checkpoint
///   `linear_attn.norm.weight` is bf16 `[128]`. `w_groups` is 1, so one weight
///   vector is shared by every V head: the kernel picks row `row % w_groups`.
/// * `y` — fp16 `[rows, dim]`. This is the `core_attn_out_f` slot. `out_proj`
///   consumes it and is not this module.
/// * `g` — f32 `[rows, dim]`, the `in_proj_z` output (EXL3, not a checkpoint
///   tensor), same element count as `x`. Not the recurrent `g`.
/// * `epsilon` — `rms_norm_eps` from the model config (`1e-6` on the 27B).
/// * `constant_bias` — added to the weight. The 27B norm passes 0.
/// * `gate_first` — 0 or 1. 0 applies the gate after the norm (GDN). 1 applies
///   it before (Mamba2). The 27B passes 0.
/// * `gate_act` — 0 is silu, 1 is sigmoid. The 27B passes 0 (`"swish"`).
///
/// For 48 rows the launch is
/// `gated_rms_norm_kernel<32,half,bf16,float> grid=48x1x1 block=32x1x1 smem=0`.
///
/// # Errors
///
/// A null pointer, non-positive `rows` or `w_groups`, `dim` outside the
/// positive multiples of 4 up through 256, or a flag other than 0 or 1 is
/// [`Error::Launch`] and does not call CUDA.
///
/// # Safety
///
/// Non-null pointers must be device allocations of at least the stated shapes,
/// aligned for the kernel's 8-byte and 16-byte vector loads, and live until
/// `stream` finishes. `stream` may be null.
#[allow(clippy::too_many_arguments)]
pub unsafe fn gated_rms_norm(
    x: *const c_void,
    w: *const c_void,
    y: *mut c_void,
    g: *const c_void,
    epsilon: f32,
    constant_bias: f32,
    rows: i32,
    dim: i32,
    w_groups: i32,
    gate_first: i32,
    gate_act: i32,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe {
        berth_gdn_gated_rms_norm(
            x,
            w,
            y,
            g,
            epsilon,
            constant_bias,
            rows,
            dim,
            w_groups,
            gate_first,
            gate_act,
            stream,
        )
    };
    check(rc, "gated_rms_norm")
}

/// Kernel the last successful GDN launch on this thread queued.
///
/// Empty after a failed launch, and before any launch. The string is a copy;
/// the next launch on this thread replaces the shim's buffer. A 27B decode
/// step produces, in order:
///
/// * `gdn_ba_gemv_kernel grid=12x1x1 block=256x1x1 smem=0`
/// * `gated_delta_net_fused_op_3_kernel<bf16> grid=41x1x1 block=256x1x1 smem=0`
/// * `conv1d_update_kernel<1,0> grid=40x1x1 block=256x1x1 smem=0`
/// * `cuda_recurrent_gated_delta_rule_kernel_128<0,4,0> grid=1x48x4 block=128x4x1 smem=0`
/// * `gated_rms_norm_kernel<32,half,bf16,float> grid=48x1x1 block=32x1x1 smem=0`
pub fn last_launch() -> String {
    c_string(unsafe { berth_gdn_last_launch() })
}

/// Message from the last failed GDN launch on this thread.
///
/// Empty after a successful launch. Copied, same as [`last_launch`].
/// Rewind failures are reported by the rewind wrappers themselves; they do
/// not write this string.
pub fn last_error() -> String {
    c_string(unsafe { berth_gdn_last_error() })
}

/// Causal conv1d update with an explicit history flag.
///
/// Same tensors and grid as [`conv1d_update`]. `history` selects
/// `conv1d_update_kernel<1,1>` (the fork's `cuda_causal_conv1d_update` with
/// `activation=true` and `history=true`). `history == false` launches the
/// already-linked `<1,0>` instance. `state_size` is `conv_state.size(2)`.
///
/// # Errors
///
/// The same shape checks as [`conv1d_update`]. A CUDA failure is
/// [`Error::Launch`].
///
/// # Safety
///
/// Same as [`conv1d_update`]. With history on, the kernel writes the tail of
/// each channel, up to `state_size` elements.
#[allow(clippy::too_many_arguments)]
pub unsafe fn conv1d_update_history(
    x: *const c_void,
    conv_state: *mut c_void,
    slots: *const c_void,
    weight: *const c_void,
    bias: *const c_void,
    out: *mut c_void,
    bsz: i32,
    dim: i32,
    seqlen: i32,
    state_size: i32,
    k: i32,
    history: bool,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe {
        berth_gdn_conv1d_update_history(
            x,
            conv_state,
            slots,
            weight,
            bias,
            out,
            bsz,
            dim,
            seqlen,
            state_size,
            k,
            c_int::from(history),
            stream,
        )
    };
    check(rc, "conv1d_update_history")
}

/// Gated delta rule with an explicit history flag.
///
/// Same tensors as [`recurrent_gated_delta_rule`]. `history` selects
/// `cuda_recurrent_gated_delta_rule_kernel_128<1,4,0>`. `v_split` is the
/// fork's value: 4 when `bsz == 1`, both head dims are 128, and
/// `num_v_heads <= 64`. Any other shape is rejected; `<*,1>` is not linked.
/// When `history` is set, `history_stride` must be at least `seqlen`
/// (`recurrent_state.size(1)`).
///
/// # Errors
///
/// A null required pointer, a shape that is not `kernel_128` with
/// `v_split == 4`, or `history_stride < seqlen` while history is on, is
/// [`Error::Launch`] and does not call CUDA.
///
/// # Safety
///
/// Same as [`recurrent_gated_delta_rule`]. With history on, indexes
/// `0..seqlen` of the history dimension are read and written.
#[allow(clippy::too_many_arguments)]
pub unsafe fn recurrent_gated_delta_rule_history(
    mixed_qkv: *const c_void,
    g: *const c_void,
    beta: *const c_void,
    recurrent_state: *mut c_void,
    core_attn_out: *mut c_void,
    slots: *const c_void,
    bsz: i32,
    seqlen: i32,
    num_k_heads: i32,
    num_v_heads: i32,
    k_head_dim: i32,
    v_head_dim: i32,
    history_stride: i32,
    history: bool,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe {
        berth_gdn_recurrent_history(
            mixed_qkv,
            g,
            beta,
            recurrent_state,
            core_attn_out,
            slots,
            bsz,
            seqlen,
            num_k_heads,
            num_v_heads,
            k_head_dim,
            v_head_dim,
            history_stride,
            c_int::from(history),
            stream,
        )
    };
    check(rc, "recurrent_gated_delta_rule_history")
}

/// Launch `batched_conv_rewind_kernel` for `jobs`.
///
/// An empty slice returns without calling CUDA. Jobs are packed into chunks
/// of 64. The grid is `(ceil(max_dim / 256), n_chunk)` and the block is 256,
/// matching `batched_conv_rewind` in `gdn.cu`. `cdim` must be in `0..=16`.
///
/// # Errors
///
/// `cdim` outside `0..=16`, a negative `dim`, or a CUDA failure is
/// [`Error::Launch`]. The rewind error channel is separate from [`last_error`].
///
/// # Safety
///
/// Each job's `src` and `dst` must be device addresses of a bf16 conv state.
/// The kernel reads `cdim` elements into registers before writing them back,
/// so an overlapping window is safe. `stream` may be null.
pub unsafe fn batched_conv_rewind(jobs: &[ConvRewindJob], stream: *mut c_void) -> Result<()> {
    let n = i32::try_from(jobs.len())
        .map_err(|_| Error::Launch("batched_conv_rewind: too many jobs".into()))?;
    let ptr = if n == 0 {
        std::ptr::null()
    } else {
        jobs.as_ptr()
    };
    let rc = unsafe { berth_gdn_batched_conv_rewind(ptr, n, stream) };
    check_rewind(rc, "batched_conv_rewind")
}

/// Launch `batched_state_rewind_kernel` for `jobs`.
///
/// An empty slice returns without calling CUDA. The grid is
/// `(ceil((max_elements / 4) / 256), n_chunk)` and the block is 256.
/// `num_elements` must be a non-negative multiple of 4.
///
/// # Errors
///
/// A `num_elements` that is not a multiple of 4, a count that does not fit
/// in `int` after dividing by 4, or a CUDA failure is [`Error::Launch`].
///
/// # Safety
///
/// Each job's `src` and `dst` must be device addresses of `num_elements`
/// fp32 values. The fork's sources do not overlap the destination.
pub unsafe fn batched_state_rewind(jobs: &[StateRewindJob], stream: *mut c_void) -> Result<()> {
    let n = i32::try_from(jobs.len())
        .map_err(|_| Error::Launch("batched_state_rewind: too many jobs".into()))?;
    let ptr = if n == 0 {
        std::ptr::null()
    } else {
        jobs.as_ptr()
    };
    let rc = unsafe { berth_gdn_batched_state_rewind(ptr, n, stream) };
    check_rewind(rc, "batched_state_rewind")
}

/// Name of the last successful rewind launch on this thread.
///
/// Empty after a failed rewind and before any rewind. Separate from
/// [`last_launch`].
pub fn rewind_last_launch() -> String {
    c_string(unsafe { berth_gdn_rewind_last_launch() })
}

fn check_rewind(rc: c_int, what: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let message = c_string(unsafe { berth_gdn_rewind_last_error() });
    let message = if message.is_empty() {
        format!("{what} launch failed ({rc})")
    } else {
        message
    };
    Err(Error::Launch(message))
}

fn rewind_counts(last_history: i32, num_tokens: i32) -> Result<()> {
    if last_history < 0 || num_tokens < 0 {
        return Err(Error::Launch(
            "rewind: last_history and num_tokens must be non-negative".into(),
        ));
    }
    if num_tokens > last_history {
        return Err(Error::Launch(
            "rewind: num_tokens must be <= last_history".into(),
        ));
    }
    Ok(())
}

fn rewind_slot_base(base: u64, slot: i64, stride0: i64, elem_size: i64) -> Result<u64> {
    if slot < 0 || stride0 < 0 || elem_size <= 0 {
        return Err(Error::Launch(
            "rewind: slot, stride0, and element size must be non-negative".into(),
        ));
    }
    let delta = i128::from(slot)
        .checked_mul(i128::from(stride0))
        .and_then(|n| n.checked_mul(i128::from(elem_size)))
        .ok_or_else(|| Error::Launch("rewind: slot offset overflows".into()))?;
    add_ptr(base, delta)
}

fn add_ptr(base: u64, delta: i128) -> Result<u64> {
    let addr = i128::from(base)
        .checked_add(delta)
        .ok_or_else(|| Error::Launch("rewind: pointer overflows".into()))?;
    u64::try_from(addr).map_err(|_| Error::Launch("rewind: pointer overflows".into()))
}

/// Conv rewind addresses, matching `GDNLayerState.rewind_conv_job`.
///
/// `None` when `last_history == 0`. Otherwise `p = state_size - num_tokens`
/// and the job copies elements `[p - conv_k, p)` onto `[0, conv_k)`.
///
/// * `base` — `conv_state.data_ptr()`.
/// * `slot` — which slot. The slot base is `base + slot * stride0 * elem_size`.
/// * `stride0`, `stride1`, `stride2` — element strides of `[slots, dim, state]`.
/// * `shape1` — `conv_state.shape[1]` (channels). Stored as `dim`.
/// * `state_size` — `conv_state.shape[-1]`.
/// * `conv_k` — `conv_kernel_size`. Stored as `cdim`.
/// * `elem_size` — `element_size()` in bytes (2 for bf16).
/// * `stride` on the job is `stride1` in elements, which must fit in `i32`.
///
/// `num_tokens > last_history` is an error, same as `GDNState.rewind`.
///
/// # Errors
///
/// A negative count, `num_tokens > last_history`, a source window that
/// starts before the buffer, or an offset that overflows.
#[allow(clippy::too_many_arguments)]
pub fn rewind_conv_addrs(
    base: u64,
    slot: i64,
    stride0: i64,
    stride1: i64,
    stride2: i64,
    shape1: i32,
    state_size: i32,
    conv_k: i32,
    elem_size: i64,
    last_history: i32,
    num_tokens: i32,
) -> Result<Option<ConvRewindJob>> {
    rewind_counts(last_history, num_tokens)?;
    if last_history == 0 {
        return Ok(None);
    }
    if shape1 < 0 || state_size < 1 || conv_k < 1 || stride1 < 0 || stride2 < 0 {
        return Err(Error::Launch(
            "rewind: conv shape and strides must be non-negative".into(),
        ));
    }
    let stride = i32::try_from(stride1)
        .map_err(|_| Error::Launch("rewind: conv stride1 does not fit in i32".into()))?;
    let p = state_size
        .checked_sub(num_tokens)
        .ok_or_else(|| Error::Launch("rewind: conv p underflows".into()))?;
    let elem_off = i64::from(p)
        .checked_sub(i64::from(conv_k))
        .ok_or_else(|| Error::Launch("rewind: conv source starts before the buffer".into()))?;
    if elem_off < 0 {
        return Err(Error::Launch(
            "rewind: conv source starts before the buffer".into(),
        ));
    }
    let slot_base = rewind_slot_base(base, slot, stride0, elem_size)?;
    let byte_off = i128::from(elem_off)
        .checked_mul(i128::from(stride2))
        .and_then(|n| n.checked_mul(i128::from(elem_size)))
        .ok_or_else(|| Error::Launch("rewind: conv source offset overflows".into()))?;
    let src = add_ptr(slot_base, byte_off)?;
    Ok(Some(ConvRewindJob {
        src,
        dst: slot_base,
        dim: shape1,
        cdim: conv_k,
        stride,
    }))
}

/// Recurrent rewind addresses, matching `GDNLayerState.rewind_state_job`.
///
/// `None` when `num_tokens == 0`. Otherwise the source history index is
/// `last_history + 1 - num_tokens`, and `num_elements` is `stride1`
/// (one history slice, not the product of the head dims).
///
/// * `base` — `recurrent_state.data_ptr()`.
/// * `stride0`, `stride1` — element strides. `stride1` is the history stride
///   in elements (`48 * 128 * 128` when the tail is contiguous).
/// * `elem_size` — 4 for the fp32 recurrent state.
///
/// # Errors
///
/// A negative count, `num_tokens > last_history`, or an offset that overflows.
pub fn rewind_state_addrs(
    base: u64,
    slot: i64,
    stride0: i64,
    stride1: i64,
    elem_size: i64,
    last_history: i32,
    num_tokens: i32,
) -> Result<Option<StateRewindJob>> {
    rewind_counts(last_history, num_tokens)?;
    if num_tokens == 0 {
        return Ok(None);
    }
    if stride1 < 0 {
        return Err(Error::Launch(
            "rewind: recurrent stride1 must be non-negative".into(),
        ));
    }
    let index = i64::from(last_history) + 1 - i64::from(num_tokens);
    let slot_base = rewind_slot_base(base, slot, stride0, elem_size)?;
    let byte_off = i128::from(index)
        .checked_mul(i128::from(stride1))
        .and_then(|n| n.checked_mul(i128::from(elem_size)))
        .ok_or_else(|| Error::Launch("rewind: state source offset overflows".into()))?;
    let src = add_ptr(slot_base, byte_off)?;
    Ok(Some(StateRewindJob {
        src,
        dst: slot_base,
        num_elements: stride1,
    }))
}

fn c_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake() -> *const c_void {
        0x1000 as *const c_void
    }

    fn fake_mut() -> *mut c_void {
        0x1000 as *mut c_void
    }

    #[test]
    fn symbols_link() {
        let ba = berth_gdn_ba_gemv as *const ();
        let conv = berth_gdn_conv1d_update as *const ();
        let fused = berth_gdn_fused_op_3 as *const ();
        let recurrent = berth_gdn_recurrent as *const ();
        let rms = berth_gdn_gated_rms_norm as *const ();
        let err = berth_gdn_last_error as *const ();
        let launch = berth_gdn_last_launch as *const ();
        let conv_h = berth_gdn_conv1d_update_history as *const ();
        let recurrent_h = berth_gdn_recurrent_history as *const ();
        let conv_rw = berth_gdn_batched_conv_rewind as *const ();
        let state_rw = berth_gdn_batched_state_rewind as *const ();
        let rewind_launch = berth_gdn_rewind_last_launch as *const ();
        assert!(!ba.is_null());
        assert!(!conv.is_null());
        assert!(!fused.is_null());
        assert!(!recurrent.is_null());
        assert!(!rms.is_null());
        assert!(!err.is_null());
        assert!(!launch.is_null());
        assert!(!conv_h.is_null());
        assert!(!recurrent_h.is_null());
        assert!(!conv_rw.is_null());
        assert!(!state_rw.is_null());
        assert!(!rewind_launch.is_null());
    }

    #[test]
    fn ba_gemv_rejects_bad_shapes_without_cuda() {
        let err = unsafe {
            gdn_ba_gemv(
                std::ptr::null(),
                fake(),
                std::ptr::null(),
                fake_mut(),
                1,
                5120,
                96,
                std::ptr::null_mut(),
            )
        }
        .expect_err("null x");
        assert!(err.to_string().contains("x, w_t, and y"), "{err}");
        assert!(last_launch().is_empty(), "{}", last_launch());

        let err = unsafe {
            gdn_ba_gemv(
                fake(),
                fake(),
                std::ptr::null(),
                fake_mut(),
                1,
                3,
                96,
                std::ptr::null_mut(),
            )
        }
        .expect_err("odd k");
        assert!(err.to_string().contains("even"), "{err}");

        let err = unsafe {
            gdn_ba_gemv(
                fake(),
                fake(),
                std::ptr::null(),
                fake_mut(),
                0,
                2,
                96,
                std::ptr::null_mut(),
            )
        }
        .expect_err("rows");
        assert!(err.to_string().contains("positive"), "{err}");
    }

    #[test]
    fn conv_rejects_kernel_sizes_outside_the_linked_instance() {
        let err = unsafe {
            conv1d_update(
                fake(),
                fake_mut(),
                std::ptr::null(),
                fake(),
                std::ptr::null(),
                fake_mut(),
                1,
                10240,
                1,
                4,
                17,
                std::ptr::null_mut(),
            )
        }
        .expect_err("K");
        assert!(err.to_string().contains("1..=16"), "{err}");

        let err = unsafe {
            conv1d_update(
                fake(),
                fake_mut(),
                std::ptr::null(),
                fake(),
                std::ptr::null(),
                fake_mut(),
                1,
                10240,
                1,
                3,
                4,
                std::ptr::null_mut(),
            )
        }
        .expect_err("state");
        assert!(err.to_string().contains("state_size"), "{err}");
        assert!(last_launch().is_empty(), "{}", last_launch());
    }

    #[test]
    fn fused_op_3_rejects_an_index_that_does_not_fit() {
        let err = unsafe {
            gated_delta_net_fused_op_3(
                fake(),
                fake(),
                fake(),
                fake(),
                fake_mut(),
                fake_mut(),
                fake_mut(),
                46341,
                46341,
                2,
                1,
                1.0,
                std::ptr::null_mut(),
            )
        }
        .expect_err("overflow");
        assert!(err.to_string().contains("int32"), "{err}");
        assert!(last_launch().is_empty(), "{}", last_launch());
    }

    #[test]
    fn recurrent_rejects_shapes_that_are_not_kernel_128_vsplit4() {
        let err = unsafe {
            recurrent_gated_delta_rule(
                fake(),
                fake(),
                fake(),
                fake_mut(),
                fake_mut(),
                std::ptr::null(),
                2,
                1,
                16,
                48,
                128,
                128,
                1,
                std::ptr::null_mut(),
            )
        }
        .expect_err("bsz");
        assert!(err.to_string().contains("bsz must be 1"), "{err}");

        let err = unsafe {
            recurrent_gated_delta_rule(
                fake(),
                fake(),
                fake(),
                fake_mut(),
                fake_mut(),
                std::ptr::null(),
                1,
                1,
                16,
                48,
                64,
                128,
                1,
                std::ptr::null_mut(),
            )
        }
        .expect_err("head dim");
        assert!(err.to_string().contains("128"), "{err}");

        let err = unsafe {
            recurrent_gated_delta_rule(
                fake(),
                fake(),
                fake(),
                fake_mut(),
                fake_mut(),
                std::ptr::null(),
                1,
                1,
                1,
                65,
                128,
                128,
                1,
                std::ptr::null_mut(),
            )
        }
        .expect_err("v heads");
        assert!(err.to_string().contains("at most 64"), "{err}");
        assert!(last_launch().is_empty(), "{}", last_launch());
    }

    #[test]
    fn gated_rms_rejects_the_1024_thread_instance() {
        let err = unsafe {
            gated_rms_norm(
                fake(),
                fake(),
                fake_mut(),
                fake(),
                1e-6,
                0.0,
                48,
                260,
                1,
                0,
                0,
                std::ptr::null_mut(),
            )
        }
        .expect_err("dim");
        assert!(err.to_string().contains("at most 256"), "{err}");

        let err = unsafe {
            gated_rms_norm(
                fake(),
                fake(),
                fake_mut(),
                fake(),
                1e-6,
                0.0,
                48,
                2,
                1,
                0,
                2,
                std::ptr::null_mut(),
            )
        }
        .expect_err("dim not multiple of 4");
        assert!(err.to_string().contains("multiple of 4"), "{err}");

        let err = unsafe {
            gated_rms_norm(
                fake(),
                fake(),
                fake_mut(),
                fake(),
                1e-6,
                0.0,
                48,
                128,
                1,
                0,
                2,
                std::ptr::null_mut(),
            )
        }
        .expect_err("gate_act");
        assert!(err.to_string().contains("silu"), "{err}");
        assert!(last_launch().is_empty(), "{}", last_launch());
    }

    #[test]
    fn rewind_job_layout() {
        assert_eq!(std::mem::size_of::<ConvRewindJob>(), 32);
        assert_eq!(std::mem::size_of::<StateRewindJob>(), 24);
        assert_eq!(std::mem::offset_of!(ConvRewindJob, src), 0);
        assert_eq!(std::mem::offset_of!(ConvRewindJob, dst), 8);
        assert_eq!(std::mem::offset_of!(ConvRewindJob, dim), 16);
        assert_eq!(std::mem::offset_of!(ConvRewindJob, cdim), 20);
        assert_eq!(std::mem::offset_of!(ConvRewindJob, stride), 24);
        assert_eq!(std::mem::offset_of!(StateRewindJob, num_elements), 16);
    }

    #[test]
    fn rewind_addrs_match_section6() {
        // docs/mtp.md section 6. K = 4, conv_k = 4, state_size = 8, n = 5,
        // last_history = n - 1 = 4. Contiguous bf16 conv [1, 10240, 8].
        let base = 0x1_0000u64;
        let es = 2i64;
        let dim = 10240i32;
        let state_size = 8i32;
        let conv_k = 4i32;
        let stride2 = 1i64;
        let stride1 = i64::from(state_size);
        let stride0 = i64::from(dim) * stride1;

        // Reject at i = 0: num_rejected = 4. Source element offset is 0,
        // so the window copied back is [0, 4).
        let conv = rewind_conv_addrs(
            base, 0, stride0, stride1, stride2, dim, state_size, conv_k, es, 4, 4,
        )
        .unwrap()
        .unwrap();
        assert_eq!(conv.src, base);
        assert_eq!(conv.dst, base);
        assert_eq!(conv.dim, dim);
        assert_eq!(conv.cdim, conv_k);
        assert_eq!(conv.stride, 8);

        // Contiguous fp32 recurrent [1, 5, 48, 128, 128]. Source index is 1.
        let heads = 48i64 * 128 * 128;
        let history = 5i64;
        let state = rewind_state_addrs(base, 0, history * heads, heads, 4, 4, 4)
            .unwrap()
            .unwrap();
        assert_eq!(state.src, base + (heads as u64) * 4);
        assert_eq!(state.dst, base);
        assert_eq!(state.num_elements, heads);

        // Full accept, rewind(0): no state job. Conv copies elements [4, 8).
        assert!(rewind_state_addrs(base, 0, history * heads, heads, 4, 4, 0)
            .unwrap()
            .is_none());
        let accept = rewind_conv_addrs(
            base, 0, stride0, stride1, stride2, dim, state_size, conv_k, es, 4, 0,
        )
        .unwrap()
        .unwrap();
        assert_eq!(accept.src, base + 8);
        assert_eq!(accept.dst, base);
        assert_eq!(accept.cdim, conv_k);
        assert_eq!(accept.stride, 8);

        // last_history == 0 skips the conv job. num_tokens == 0 skips state.
        assert!(rewind_conv_addrs(
            base, 0, stride0, stride1, stride2, dim, state_size, conv_k, es, 0, 0,
        )
        .unwrap()
        .is_none());
        assert!(rewind_state_addrs(base, 0, history * heads, heads, 4, 4, 0)
            .unwrap()
            .is_none());

        // Slot 1 shifts by stride0 * element size.
        let slotted = rewind_conv_addrs(
            base, 1, stride0, stride1, stride2, dim, state_size, conv_k, es, 4, 4,
        )
        .unwrap()
        .unwrap();
        let slot_bytes = (stride0 * es) as u64;
        assert_eq!(slotted.dst, base + slot_bytes);
        assert_eq!(slotted.src, slotted.dst);

        assert!(rewind_conv_addrs(
            base, 0, stride0, stride1, stride2, dim, state_size, conv_k, es, 4, 5,
        )
        .is_err());
        assert!(rewind_state_addrs(base, 0, history * heads, heads, 4, 3, 4).is_err());
    }

    #[test]
    fn rewind_and_history_reject_before_cuda() {
        let bad = ConvRewindJob {
            src: 0x1000,
            dst: 0x2000,
            dim: 10240,
            cdim: 17,
            stride: 8,
        };
        let err = unsafe { batched_conv_rewind(&[bad], std::ptr::null_mut()) }.unwrap_err();
        assert!(err.to_string().contains("CONV1D_MAX_K"), "{err}");

        assert!(unsafe { batched_conv_rewind(&[], std::ptr::null_mut()) }.is_ok());
        assert!(unsafe { batched_state_rewind(&[], std::ptr::null_mut()) }.is_ok());

        let bad_state = StateRewindJob {
            src: 0x1000,
            dst: 0x2000,
            num_elements: 2,
        };
        let err = unsafe { batched_state_rewind(&[bad_state], std::ptr::null_mut()) }.unwrap_err();
        assert!(err.to_string().contains("multiple of 4"), "{err}");

        let err = unsafe {
            conv1d_update_history(
                fake(),
                fake_mut(),
                std::ptr::null(),
                fake(),
                std::ptr::null(),
                fake_mut(),
                1,
                10240,
                5,
                8,
                17,
                true,
                std::ptr::null_mut(),
            )
        }
        .unwrap_err();
        assert!(err.to_string().contains("1..=16"), "{err}");

        let err = unsafe {
            recurrent_gated_delta_rule_history(
                fake(),
                fake(),
                fake(),
                fake_mut(),
                fake_mut(),
                std::ptr::null(),
                1,
                5,
                16,
                48,
                128,
                128,
                4,
                true,
                std::ptr::null_mut(),
            )
        }
        .unwrap_err();
        assert!(err.to_string().contains("history_stride"), "{err}");

        let err = unsafe {
            recurrent_gated_delta_rule_history(
                fake(),
                fake(),
                fake(),
                fake_mut(),
                fake_mut(),
                std::ptr::null(),
                2,
                1,
                16,
                48,
                128,
                128,
                1,
                true,
                std::ptr::null_mut(),
            )
        }
        .unwrap_err();
        assert!(err.to_string().contains("v_split"), "{err}");
        assert!(last_launch().is_empty(), "{}", last_launch());
        assert!(rewind_last_launch().is_empty(), "{}", rewind_last_launch());
    }
}

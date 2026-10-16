//! Gated MLP, one decode token (all 64 layers).
//!
//! Owner: task MLP. Reproduces `BC_GatedMLP::run_bszN` for bsz 1.
//!
//! Gate and up are two int8 GEMVs with fp16 output. `InferParams::use_mgemm`
//! is false for these linears (K = 4, N = 17408, `EXL3_INT8_GEMV=2`), which
//! is the measured pair of fp16 GEMVs per layer. `silu_mul` then runs with
//! the fork's `act_limit` of 0, and down is an int8 GEMV with fp32 output.
//!
//! Prefill ([`MlpLayer::prefill`]) is the same product for `rows > 1`, through
//! [`Exl3Linear::forward_rows`]. Gate and up stay separate fp16 linears.

use berth_cuda::ByteBuf;
use berth_exl3::Checkpoint;
use berth_kernels::OutDtype;

use crate::device::{find_linear, ptr};
use crate::{Error, Exl3Linear, Gpu, Result};

/// `GatedMLP` default. Qwen3.5's dense MLP does not pass `act_limit`
/// (`architecture/qwen3_5.py`), and `0` skips the clamp in `act_mul_kernel_h`.
const ACT_LIMIT: f32 = 0.0;

/// One layer's `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`, and scratch.
pub struct MlpLayer {
    gate: Exl3Linear,
    up: Exl3Linear,
    down: Exl3Linear,
    /// fp16 `[intermediate]`. Gate GEMV output.
    gate_out: ByteBuf,
    /// fp16 `[intermediate]`. Up GEMV output.
    up_out: ByteBuf,
    /// fp16 `[intermediate]`. `silu(gate) * up`, the fork's separate `a` buffer.
    activated: ByteBuf,
    /// `gate_proj` out-features. 17408 on the 27B.
    intermediate: i64,
    /// `gate_proj` in-features, and the down projection's out-features.
    hidden: i32,
}

impl MlpLayer {
    /// `gate_proj` in-features.
    pub(crate) fn hidden_size(&self) -> i32 {
        self.hidden
    }

    /// `gate_proj` out-features.
    pub(crate) fn intermediate(&self) -> i64 {
        self.intermediate
    }
}

impl MlpLayer {
    /// Load layer `layer`'s MLP weights.
    pub fn load(gpu: &Gpu, ckpt: &Checkpoint, layer: u32) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        Self::load_prefixed(gpu, ckpt, &prefix)
    }

    /// Load `{prefix}.gate_proj`, `.up_proj`, and `.down_proj`.
    ///
    /// [`Self::load`] passes `model.language_model.layers.{layer}.mlp`. The
    /// MTP head passes `mtp.layers.0.mlp`.
    pub fn load_prefixed(gpu: &Gpu, ckpt: &Checkpoint, prefix: &str) -> Result<Self> {
        let gate_prefix = format!("{prefix}.gate_proj");
        let up_prefix = format!("{prefix}.up_proj");
        let down_prefix = format!("{prefix}.down_proj");
        // `BC_GatedMLP::run_bszN_gr` adds a bias when the linear has one. The
        // 27B language-model linears do not, and `Exl3Linear` does not add one.
        for part in [&gate_prefix, &up_prefix, &down_prefix] {
            if find_linear(ckpt, part)?.bias.is_some() {
                return Err(Error::Weights(format!(
                    "{part}: bias is present and is not applied"
                )));
            }
        }

        let gate = Exl3Linear::load(gpu, ckpt, &gate_prefix)?;
        let up = Exl3Linear::load(gpu, ckpt, &up_prefix)?;
        let down = Exl3Linear::load(gpu, ckpt, &down_prefix)?;
        if gate.size_k != up.size_k || gate.size_n != up.size_n {
            return Err(Error::Weights(format!(
                "{prefix}: gate is {}x{}, up is {}x{}",
                gate.size_k, gate.size_n, up.size_k, up.size_n
            )));
        }
        if down.size_k != gate.size_n || down.size_n != gate.size_k {
            return Err(Error::Weights(format!(
                "{prefix}: down is {}x{}, gate is {}x{}",
                down.size_k, down.size_n, gate.size_k, gate.size_n
            )));
        }
        let hidden = gate.size_k;
        let intermediate = gate.size_n;
        // `silu_mul` requires a positive even count (`act_mul_kernel_h` steps by half2).
        if intermediate < 2 || intermediate % 2 != 0 {
            return Err(Error::Weights(format!(
                "{prefix}: intermediate {intermediate} must be positive and even"
            )));
        }
        let bytes = intermediate as usize * 2;
        Ok(Self {
            gate,
            up,
            down,
            gate_out: gpu.zeroed(bytes)?,
            up_out: gpu.zeroed(bytes)?,
            activated: gpu.zeroed(bytes)?,
            intermediate: i64::from(intermediate),
            hidden,
        })
    }

    /// `output = down(silu(gate(input)) * up(input))`.
    ///
    /// `input` is fp16 `[hidden]` (post_attention_layernorm output); `output`
    /// is fp32 `[hidden]`.
    pub fn forward(&mut self, gpu: &Gpu, input: &ByteBuf, output: &mut ByteBuf) -> Result<()> {
        if input.num_bytes() < self.hidden as usize * 2
            || output.num_bytes() < self.hidden as usize * 4
        {
            return Err(Error::Input(format!(
                "{}: buffer too small",
                self.gate.prefix
            )));
        }

        // Unfused `run_bszN_gr`: gate GEMV, up GEMV, then `silu_mul_gr`, then down.
        self.gate
            .forward(gpu, input, &mut self.gate_out, OutDtype::F16)?;
        self.up
            .forward(gpu, input, &mut self.up_out, OutDtype::F16)?;
        // SAFETY: the three buffers are device fp16 of `intermediate` elements,
        // allocated in `load`, and stay alive for this stream. `activated` does
        // not alias the GEMV outputs. Later work on `gpu`'s stream is ordered
        // after this launch.
        unsafe {
            berth_kernels::ops::silu_mul(
                ptr(&self.gate_out),
                ptr(&self.up_out),
                ptr(&self.activated),
                self.intermediate,
                ACT_LIMIT,
                gpu.raw_stream(),
            )?;
        }
        self.down
            .forward(gpu, &self.activated, output, OutDtype::F32)?;
        Ok(())
    }
}

/// Scratch for [`MlpLayer::prefill`], allocated once by the model for the
/// largest chunk and shared by all 64 MLPs.
///
/// Owner: prefill task MLP.
///
/// Gate and up are not fused. `GatedMLP.forward` (`modules/mlp.py`) calls
/// `exl3_mgemm` through `multi_gu` only when that bundle was built and
/// `bsz * q_len <= 32`. Above 32 rows the two linears always run separately.
/// On the 27B `InferParams::use_mgemm` is false for K = 4 and N = 17408
/// (`EXL3_INT8_GEMV=2`, K threshold 6, N threshold 8192), so `multi_gu` is
/// never built and every chunk above `MAX_BSZN` takes the unfused pair.
/// Both projections write fp16 (`interm_dtype`).
pub struct MlpPrefillScratch {
    /// fp16 `[max_rows, intermediate]`. Gate projection.
    gate: ByteBuf,
    /// fp16 `[max_rows, intermediate]`. Up projection. `silu_mul` then writes
    /// over it: `interm_dtype` is half, so the fork sets `a = u`.
    up: ByteBuf,
    /// `gate_proj` in-features. Every layer that shares this scratch matches it.
    hidden: u32,
    /// `gate_proj` out-features. Even, so `rows * intermediate` stays even.
    intermediate: u32,
    max_rows: u32,
}

impl MlpPrefillScratch {
    /// Room for chunks of up to `max_rows` tokens.
    ///
    /// `hidden` and `intermediate` are the 27B widths (5120 and 17408). The
    /// model passes `max_rows` 2048.
    pub fn new(gpu: &Gpu, hidden: u32, intermediate: u32, max_rows: u32) -> Result<Self> {
        if hidden == 0 || max_rows == 0 {
            return Err(Error::Input(
                "MLP prefill scratch needs a positive hidden size and max_rows".into(),
            ));
        }
        // `act_mul_kernel_h` steps by half2. An even intermediate keeps
        // `rows * intermediate` even for every chunk. Zero is a multiple of
        // two, so the lower bound stays.
        if intermediate < 2 || !intermediate.is_multiple_of(2) {
            return Err(Error::Input(format!(
                "MLP prefill scratch intermediate {intermediate} must be positive and even"
            )));
        }
        let bytes = matrix_bytes(max_rows, intermediate, 2)?;
        Ok(Self {
            gate: gpu.zeroed(bytes)?,
            up: gpu.zeroed(bytes)?,
            hidden,
            intermediate,
            max_rows,
        })
    }
}

impl MlpLayer {
    /// `output = down(silu(gate(input)) * up(input))` for `rows > 1` rows,
    /// the fork's prefill path (`GatedMLP.forward` past `MAX_BSZN`).
    ///
    /// Gate and up are separate fp16 linears. Down is fp32. `silu_mul` covers
    /// `rows * intermediate` elements and writes over the up projection.
    ///
    /// `input` is fp16 `[rows, hidden]`; `output` is fp32 `[rows, hidden]`.
    /// Rows at or under `MAX_BSZN` (8) are `BC_GatedMLP::run_bszN` in the
    /// fork, not this path.
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn prefill(
        &mut self,
        gpu: &Gpu,
        rows_scratch: &mut crate::device::RowScratch,
        scratch: &mut MlpPrefillScratch,
        input: &ByteBuf,
        output: &mut ByteBuf,
        rows: u32,
    ) -> Result<()> {
        if rows < 2 {
            return Err(Error::Input(format!(
                "{}: prefill needs rows > 1",
                self.gate.prefix
            )));
        }
        if rows > scratch.max_rows {
            return Err(Error::Input(format!(
                "{}: {rows} rows exceeds the MLP prefill scratch ({})",
                self.gate.prefix, scratch.max_rows
            )));
        }
        if i64::from(self.hidden) != i64::from(scratch.hidden)
            || self.intermediate != i64::from(scratch.intermediate)
        {
            return Err(Error::Input(format!(
                "{}: layer is {}x{}, scratch is {}x{}",
                self.gate.prefix,
                self.hidden,
                self.intermediate,
                scratch.hidden,
                scratch.intermediate
            )));
        }
        let in_bytes = matrix_bytes(rows, scratch.hidden, 2)?;
        let out_bytes = matrix_bytes(rows, scratch.hidden, 4)?;
        if input.num_bytes() < in_bytes || output.num_bytes() < out_bytes {
            return Err(Error::Input(format!(
                "{}: buffer too small",
                self.gate.prefix
            )));
        }

        // Unfused branch of `GatedMLP.forward`: `multi_gu is None or rows > 32`.
        // `out_dtype` on gate and up is `interm_dtype` (fp16).
        self.gate.forward_rows(
            gpu,
            rows_scratch,
            input,
            &mut scratch.gate,
            rows,
            OutDtype::F16,
        )?;
        self.up.forward_rows(
            gpu,
            rows_scratch,
            input,
            &mut scratch.up,
            rows,
            OutDtype::F16,
        )?;

        // `a = u` when `interm_dtype` is half, then `silu_mul(g, u, a, act_limit)`.
        let numel = i64::from(rows)
            .checked_mul(self.intermediate)
            .ok_or_else(|| Error::Input("MLP prefill activation count overflows".into()))?;
        let gate_ptr = ptr(&scratch.gate);
        let up_ptr = ptr(&scratch.up);
        // SAFETY: both buffers are device fp16 of `max_rows * intermediate`
        // elements and `numel` is only the `rows` prefix. `up_ptr` is the
        // fork's aliased destination; each thread reads its half2 before
        // writing it. Later work on this stream is ordered after the launch.
        unsafe {
            berth_kernels::ops::silu_mul(
                gate_ptr,
                up_ptr,
                up_ptr,
                numel,
                ACT_LIMIT,
                gpu.raw_stream(),
            )?;
        }

        // Down's `out_dtype` is fp32 (`architecture/qwen3_5.py`).
        self.down
            .forward_rows(gpu, rows_scratch, &scratch.up, output, rows, OutDtype::F32)?;
        Ok(())
    }
}

fn matrix_bytes(rows: u32, cols: u32, elem: usize) -> Result<usize> {
    (rows as usize)
        .checked_mul(cols as usize)
        .and_then(|n| n.checked_mul(elem))
        .ok_or_else(|| Error::Input("MLP prefill byte count overflows".into()))
}

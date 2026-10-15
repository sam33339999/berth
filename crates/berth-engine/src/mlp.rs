//! Gated MLP, one decode token (all 64 layers).
//!
//! Owner: task MLP. Reproduces `BC_GatedMLP::run_bszN` for bsz 1.
//!
//! Gate and up are two int8 GEMVs with fp16 output. `InferParams::use_mgemm`
//! is false for these linears (K = 4, N = 17408, `EXL3_INT8_GEMV=2`), which
//! is the measured pair of fp16 GEMVs per layer. `silu_mul` then runs with
//! the fork's `act_limit` of 0, and down is an int8 GEMV with fp32 output.

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
    /// Load layer `layer`'s MLP weights.
    pub fn load(gpu: &Gpu, ckpt: &Checkpoint, layer: u32) -> Result<Self> {
        let gate_prefix = format!("model.language_model.layers.{layer}.mlp.gate_proj");
        let up_prefix = format!("model.language_model.layers.{layer}.mlp.up_proj");
        let down_prefix = format!("model.language_model.layers.{layer}.mlp.down_proj");
        // `BC_GatedMLP::run_bszN_gr` adds a bias when the linear has one. The
        // 27B language-model linears do not, and `Exl3Linear` does not add one.
        for prefix in [&gate_prefix, &up_prefix, &down_prefix] {
            if find_linear(ckpt, prefix)?.bias.is_some() {
                return Err(Error::Weights(format!(
                    "{prefix}: bias is present and is not applied"
                )));
            }
        }

        let gate = Exl3Linear::load(gpu, ckpt, &gate_prefix)?;
        let up = Exl3Linear::load(gpu, ckpt, &up_prefix)?;
        let down = Exl3Linear::load(gpu, ckpt, &down_prefix)?;
        if gate.size_k != up.size_k || gate.size_n != up.size_n {
            return Err(Error::Weights(format!(
                "layer {layer}: gate is {}x{}, up is {}x{}",
                gate.size_k, gate.size_n, up.size_k, up.size_n
            )));
        }
        if down.size_k != gate.size_n || down.size_n != gate.size_k {
            return Err(Error::Weights(format!(
                "layer {layer}: down is {}x{}, gate is {}x{}",
                down.size_k, down.size_n, gate.size_k, gate.size_n
            )));
        }
        let hidden = gate.size_k;
        let intermediate = gate.size_n;
        // `silu_mul` requires a positive even count (`act_mul_kernel_h` steps by half2).
        if intermediate < 2 || intermediate % 2 != 0 {
            return Err(Error::Weights(format!(
                "layer {layer}: intermediate {intermediate} must be positive and even"
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

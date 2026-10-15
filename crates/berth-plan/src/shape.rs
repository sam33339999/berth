//! Checkpoint shape that [`crate::build_plan`] consumes.
//!
//! `berth-exl3` is not a dependency of this crate yet. The reader maps
//! `config.json` onto [`ModelShape`] and the safetensors byte sizes onto
//! [`crate::TensorBytes`].

use std::fmt;

/// `text_config.layer_types` entry for one decoder layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated DeltaNet. Checkpoint string `linear_attention`.
    LinearAttention,
    /// Full attention. Checkpoint string `full_attention`.
    ///
    /// Flash-Next still uses this string for its QSA layers. Mapping those
    /// to [`crate::Mixer::Qsa`] is stage 4.
    FullAttention,
}

impl LayerKind {
    /// Parse one `layer_types` string.
    pub fn from_config(name: &str) -> Result<Self, ShapeError> {
        match name {
            "linear_attention" => Ok(Self::LinearAttention),
            "full_attention" => Ok(Self::FullAttention),
            other => Err(ShapeError::UnknownLayerKind(other.to_string())),
        }
    }

    /// The `layer_types` string this kind was parsed from.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LinearAttention => "linear_attention",
            Self::FullAttention => "full_attention",
        }
    }
}

impl fmt::Display for LayerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Gated DeltaNet width. One shape for every linear-attention layer.
///
/// Config keys: `linear_num_key_heads`, `linear_num_value_heads`,
/// `linear_key_head_dim`, `linear_value_head_dim`, `linear_conv_kernel_dim`.
/// These models do not vary the width per layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnDims {
    pub num_k_heads: u32,
    pub num_v_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub conv_kernel_dim: u32,
}

/// Routed experts plus the shared expert.
///
/// Config keys: `num_experts`, `num_experts_per_tok`, `moe_intermediate_size`,
/// `shared_expert_intermediate_size`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeDims {
    pub num_experts: u32,
    pub experts_per_tok: u32,
    pub moe_intermediate: u32,
    pub shared_intermediate: u32,
}

/// Stage 1. Qwen3.8-27B, dense hybrid.
pub const ARCH_QWEN3_5: &str = "Qwen3_5ForConditionalGeneration";

/// Stage 2. Qwen3.6-35B-A3B, MoE plus a shared expert.
pub const ARCH_QWEN3_5_MOE: &str = "Qwen3_5MoeForConditionalGeneration";

/// Stage 3. MiniCPM-V 4.7. [`crate::build_plan`] returns [`crate::PlanError::NotYet`].
pub const ARCH_MINICPM_V4_7: &str = "MiniCPMV4_7ForConditionalGeneration";

/// Stage 4. Qwen3.8-Flash-Next. [`crate::build_plan`] returns [`crate::PlanError::NotYet`].
pub const ARCH_QWEN4_EXP: &str = "Qwen4ExpForConditionalGeneration";

/// Text-stack shape for one `arch_string`.
///
/// Vision and MTP are not fields here. The vision tower is prefill, and MTP
/// is a separate draft (`docs/runtime.md`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelShape {
    /// exllamav3 `arch_string`, for example [`ARCH_QWEN3_5`].
    pub arch_string: String,
    pub num_layers: u32,
    /// Length must equal [`Self::num_layers`]. Resolved `layer_types`, including
    /// the fallback `(index + 1) % full_attention_interval == 0`.
    pub layer_kinds: Vec<LayerKind>,
    pub hidden_size: u32,
    pub head_dim: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub gdn: GdnDims,
    /// Set for MoE archs. Dense archs reject `Some`.
    pub moe: Option<MoeDims>,
    pub vocab_size: u32,
}

/// A shape [`crate::build_plan`] will not turn into a plan.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ShapeError {
    #[error("num_layers is 0")]
    NoLayers,
    #[error("layer_kinds has {got} entries but num_layers is {num_layers}")]
    LayerCount { got: usize, num_layers: u32 },
    #[error("unknown layer kind `{0}`")]
    UnknownLayerKind(String),
    #[error("{arch} is dense and does not take MoE dims")]
    UnexpectedMoe { arch: String },
    #[error("{arch} requires MoE dims")]
    MissingMoe { arch: String },
    #[error("{field} must be non-zero")]
    Zero { field: &'static str },
    #[error("experts_per_tok ({experts_per_tok}) exceeds num_experts ({num_experts})")]
    ExpertsPerTok {
        experts_per_tok: u32,
        num_experts: u32,
    },
}

/// PLE n-gram injection in front of this layer.
///
/// Flash-Next only. `docs/runtime.md` puts it in front of layer 2; the
/// checkpoint's `ple_layer_ids` (1-based) names the layer. Stage 4 sets this.
/// Stages 1 and 2 leave [`crate::LayerPlan::extra`] empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PleInject;

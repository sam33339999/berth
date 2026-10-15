//! `arch_string` → [`ModelPlan`].

use crate::shape::{
    GdnDims, LayerKind, ModelShape, MoeDims, PleInject, ShapeError, ARCH_MINICPM_V4_7,
    ARCH_QWEN3_5, ARCH_QWEN3_5_MOE, ARCH_QWEN4_EXP,
};

/// Per-layer mixer. The three arms are the schedule sketch in `docs/runtime.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mixer {
    /// Gated DeltaNet (`linear_attn`). Width is [`ModelPlan::gdn`].
    Gdn,
    /// Full attention (`self_attn`). Width is the head fields on [`ModelPlan`].
    FullAttn,
    /// Sparse full attention on Qwen3.8-Flash-Next.
    ///
    /// Those layers are still `full_attention` in `layer_types`. Stage 4 maps
    /// them here. [`build_plan`] does not produce this arm.
    Qsa,
}

/// Router in front of the routed experts (`mlp.gate`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Router {
    pub num_experts: u32,
    pub experts_per_tok: u32,
}

/// Shared expert (`mlp.shared_expert`), one per MoE layer, always resident.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedExpert {
    pub intermediate_size: u32,
}

/// Routed expert pool (`mlp.experts`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RoutedExperts {
    pub num_experts: u32,
    pub experts_per_tok: u32,
    pub intermediate_size: u32,
}

/// Per-layer feed-forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ffn {
    /// Dense gated MLP (`mlp.gate_proj` / `up_proj` / `down_proj`).
    ///
    /// Intermediate width is not on [`ModelShape`]. The checkpoint reader
    /// still owns it; placement only needs the module kind.
    DenseMlp,
    /// Routed MoE plus one shared expert.
    Moe {
        router: Router,
        shared: SharedExpert,
        experts: RoutedExperts,
    },
}

/// One decoder layer in the schedule sketch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerPlan {
    pub index: u32,
    pub mixer: Mixer,
    pub ffn: Ffn,
    /// [`Some`](Option::Some) only for a Flash-Next PLE site. Empty in stages 1 and 2.
    pub extra: Option<PleInject>,
}

/// Token embedding (`model.language_model.embed_tokens`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Embedding {
    pub vocab_size: u32,
    pub hidden_size: u32,
}

/// Final RMSNorm (`model.language_model.norm`).
///
/// Qwen3.5 and Qwen3.5-MoE both have one. Flash-Next folds the stream stack
/// instead; that arch is [`PlanError::NotYet`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinalNorm {
    pub hidden_size: u32,
}

/// Output projection (`lm_head`), separate from the embedding.
///
/// Tied embeddings are not a plan flag. A tied checkpoint still lists both
/// modules; the reader passes a single tensor to placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LmHead {
    pub hidden_size: u32,
    pub vocab_size: u32,
}

/// Decode schedule for one text stack, plus the global modules around it.
///
/// The vision tower is prefill and is not a module here. MTP is a separate draft.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelPlan {
    pub arch_string: String,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub head_dim: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub gdn: GdnDims,
    pub embedding: Embedding,
    pub layers: Vec<LayerPlan>,
    pub final_norm: FinalNorm,
    pub lm_head: LmHead,
}

/// [`build_plan`] refused a shape.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PlanError {
    /// Stage 3 or 4. The string is the `arch_string`.
    #[error("architecture `{0}` is a later stage and is not planned yet")]
    NotYet(String),
    /// Not one of the four arch strings in `docs/runtime.md`.
    #[error("unknown architecture `{0}`")]
    Unknown(String),
    /// The arch is known, but the shape contradicts it.
    #[error(transparent)]
    Shape(#[from] ShapeError),
}

/// Build the stage-1 or stage-2 schedule.
///
/// `MiniCPMV4_7ForConditionalGeneration` and `Qwen4ExpForConditionalGeneration`
/// return [`PlanError::NotYet`]. Any other string returns [`PlanError::Unknown`],
/// including the text-only `Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM` names.
/// The stage table lists the `ForConditionalGeneration` strings only.
pub fn build_plan(shape: &ModelShape) -> Result<ModelPlan, PlanError> {
    match shape.arch_string.as_str() {
        ARCH_QWEN3_5 => build_dense(shape),
        ARCH_QWEN3_5_MOE => build_moe(shape),
        ARCH_MINICPM_V4_7 | ARCH_QWEN4_EXP => Err(PlanError::NotYet(shape.arch_string.clone())),
        _ => Err(PlanError::Unknown(shape.arch_string.clone())),
    }
}

fn build_dense(shape: &ModelShape) -> Result<ModelPlan, PlanError> {
    validate_common(shape)?;
    if shape.moe.is_some() {
        return Err(ShapeError::UnexpectedMoe {
            arch: shape.arch_string.clone(),
        }
        .into());
    }
    assemble(shape, Ffn::DenseMlp)
}

fn build_moe(shape: &ModelShape) -> Result<ModelPlan, PlanError> {
    validate_common(shape)?;
    let Some(dims) = shape.moe else {
        return Err(ShapeError::MissingMoe {
            arch: shape.arch_string.clone(),
        }
        .into());
    };
    validate_moe(dims)?;
    assemble(shape, moe_ffn(dims))
}

fn assemble(shape: &ModelShape, ffn: Ffn) -> Result<ModelPlan, PlanError> {
    let mut layers = Vec::with_capacity(shape.layer_kinds.len());
    for (i, kind) in shape.layer_kinds.iter().enumerate() {
        // `i < num_layers <= u32::MAX`, so the cast preserves the index.
        let index = i as u32;
        let mixer = match kind {
            LayerKind::LinearAttention => Mixer::Gdn,
            LayerKind::FullAttention => Mixer::FullAttn,
        };
        layers.push(LayerPlan {
            index,
            mixer,
            ffn,
            extra: None,
        });
    }
    Ok(ModelPlan {
        arch_string: shape.arch_string.clone(),
        hidden_size: shape.hidden_size,
        vocab_size: shape.vocab_size,
        head_dim: shape.head_dim,
        num_q_heads: shape.num_q_heads,
        num_kv_heads: shape.num_kv_heads,
        gdn: shape.gdn,
        embedding: Embedding {
            vocab_size: shape.vocab_size,
            hidden_size: shape.hidden_size,
        },
        layers,
        final_norm: FinalNorm {
            hidden_size: shape.hidden_size,
        },
        lm_head: LmHead {
            hidden_size: shape.hidden_size,
            vocab_size: shape.vocab_size,
        },
    })
}

fn moe_ffn(dims: MoeDims) -> Ffn {
    Ffn::Moe {
        router: Router {
            num_experts: dims.num_experts,
            experts_per_tok: dims.experts_per_tok,
        },
        shared: SharedExpert {
            intermediate_size: dims.shared_intermediate,
        },
        experts: RoutedExperts {
            num_experts: dims.num_experts,
            experts_per_tok: dims.experts_per_tok,
            intermediate_size: dims.moe_intermediate,
        },
    }
}

fn validate_common(shape: &ModelShape) -> Result<(), ShapeError> {
    if shape.layer_kinds.len() != shape.num_layers as usize {
        return Err(ShapeError::LayerCount {
            got: shape.layer_kinds.len(),
            num_layers: shape.num_layers,
        });
    }
    if shape.num_layers == 0 {
        return Err(ShapeError::NoLayers);
    }
    nonzero(shape.hidden_size, "hidden_size")?;
    nonzero(shape.head_dim, "head_dim")?;
    nonzero(shape.num_q_heads, "num_q_heads")?;
    nonzero(shape.num_kv_heads, "num_kv_heads")?;
    nonzero(shape.vocab_size, "vocab_size")?;
    if shape.layer_kinds.contains(&LayerKind::LinearAttention) {
        let gdn = &shape.gdn;
        nonzero(gdn.num_k_heads, "gdn.num_k_heads")?;
        nonzero(gdn.num_v_heads, "gdn.num_v_heads")?;
        nonzero(gdn.key_head_dim, "gdn.key_head_dim")?;
        nonzero(gdn.value_head_dim, "gdn.value_head_dim")?;
        nonzero(gdn.conv_kernel_dim, "gdn.conv_kernel_dim")?;
    }
    Ok(())
}

fn validate_moe(dims: MoeDims) -> Result<(), ShapeError> {
    nonzero(dims.num_experts, "moe.num_experts")?;
    nonzero(dims.experts_per_tok, "moe.experts_per_tok")?;
    nonzero(dims.moe_intermediate, "moe.moe_intermediate")?;
    nonzero(dims.shared_intermediate, "moe.shared_intermediate")?;
    if dims.experts_per_tok > dims.num_experts {
        return Err(ShapeError::ExpertsPerTok {
            experts_per_tok: dims.experts_per_tok,
            num_experts: dims.num_experts,
        });
    }
    Ok(())
}

fn nonzero(value: u32, field: &'static str) -> Result<(), ShapeError> {
    if value == 0 {
        Err(ShapeError::Zero { field })
    } else {
        Ok(())
    }
}

//! [`berth_exl3::ModelConfig`] → [`crate::ModelShape`].

use berth_exl3::{LayerType, ModelConfig};

use crate::{GdnDims, LayerKind, ModelShape, MoeDims};

/// [`model_shape`] could not copy a checkpoint config.
///
/// Zero widths, a layer-count mismatch, and an `arch_string` that disagrees
/// with the MoE option are [`crate::PlanError`] from [`crate::build_plan`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ConfigError {
    /// `layer_types[index]` is not `linear_attention` or `full_attention`.
    #[error("layer {index}: unknown layer type `{name}`")]
    UnknownLayerType { index: usize, name: String },

    /// Some of the four MoE fields are set and the rest are absent.
    ///
    /// `missing` names the absent keys in field order: `num_experts`,
    /// `num_experts_per_tok`, `moe_intermediate_size`,
    /// `shared_expert_intermediate_size`.
    #[error("incomplete MoE config, missing {}", .missing.join(", "))]
    IncompleteMoe { missing: Vec<&'static str> },
}

/// Copy `config` onto a [`ModelShape`].
///
/// Widths and `layer_types` are copied through. The arch string is copied
/// with them; [`crate::build_plan`] accepts or rejects the MoE option against
/// that string. MoE dims are [`Some`] when `num_experts`,
/// `num_experts_per_tok`, `moe_intermediate_size`, and
/// `shared_expert_intermediate_size` are all present, and [`None`] when all
/// four are absent.
pub fn model_shape(config: &ModelConfig) -> Result<ModelShape, ConfigError> {
    Ok(ModelShape {
        arch_string: config.arch_string.clone(),
        num_layers: config.num_hidden_layers,
        layer_kinds: layer_kinds(&config.layer_types)?,
        hidden_size: config.hidden_size,
        head_dim: config.head_dim,
        num_q_heads: config.num_attention_heads,
        num_kv_heads: config.num_key_value_heads,
        gdn: GdnDims {
            num_k_heads: config.linear_num_key_heads,
            num_v_heads: config.linear_num_value_heads,
            key_head_dim: config.linear_key_head_dim,
            value_head_dim: config.linear_value_head_dim,
            conv_kernel_dim: config.linear_conv_kernel_dim,
        },
        moe: moe_dims(config)?,
        vocab_size: config.vocab_size,
    })
}

fn layer_kinds(types: &[LayerType]) -> Result<Vec<LayerKind>, ConfigError> {
    types
        .iter()
        .enumerate()
        .map(|(index, layer)| match layer {
            LayerType::LinearAttention => Ok(LayerKind::LinearAttention),
            LayerType::FullAttention => Ok(LayerKind::FullAttention),
            LayerType::Other(name) => Err(ConfigError::UnknownLayerType {
                index,
                name: name.clone(),
            }),
        })
        .collect()
}

fn moe_dims(config: &ModelConfig) -> Result<Option<MoeDims>, ConfigError> {
    let num_experts = config.num_experts;
    let experts_per_tok = config.num_experts_per_tok;
    let moe_intermediate = config.moe_intermediate_size;
    let shared_intermediate = config.shared_expert_intermediate_size;
    match (
        num_experts,
        experts_per_tok,
        moe_intermediate,
        shared_intermediate,
    ) {
        (None, None, None, None) => Ok(None),
        (
            Some(num_experts),
            Some(experts_per_tok),
            Some(moe_intermediate),
            Some(shared_intermediate),
        ) => Ok(Some(MoeDims {
            num_experts,
            experts_per_tok,
            moe_intermediate,
            shared_intermediate,
        })),
        _ => {
            let mut missing = Vec::new();
            if num_experts.is_none() {
                missing.push("num_experts");
            }
            if experts_per_tok.is_none() {
                missing.push("num_experts_per_tok");
            }
            if moe_intermediate.is_none() {
                missing.push("moe_intermediate_size");
            }
            if shared_intermediate.is_none() {
                missing.push("shared_expert_intermediate_size");
            }
            Err(ConfigError::IncompleteMoe { missing })
        }
    }
}

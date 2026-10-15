//! `ModelConfig` → `ModelShape`.

use std::path::Path;

use berth_exl3::{Checkpoint, LayerType, ModelConfig};
use berth_plan::from_exl3::{model_shape, ConfigError};
use berth_plan::{
    build_plan, GdnDims, LayerKind, Mixer, ModelShape, MoeDims, ARCH_QWEN3_5, ARCH_QWEN3_5_MOE,
};

struct Sample {
    arch: &'static str,
    num_layers: u32,
    layer_types: &'static str,
    hidden_size: u32,
    head_dim: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    vocab_size: u32,
    gdn: GdnDims,
    /// Extra `text_config` keys, without a leading comma. Empty when dense.
    moe_fields: &'static str,
}

fn dense_sample() -> Sample {
    Sample {
        arch: ARCH_QWEN3_5,
        num_layers: 2,
        layer_types: r#"["full_attention", "linear_attention"]"#,
        hidden_size: 96,
        head_dim: 40,
        num_q_heads: 6,
        num_kv_heads: 3,
        vocab_size: 12_345,
        gdn: GdnDims {
            num_k_heads: 7,
            num_v_heads: 11,
            key_head_dim: 13,
            value_head_dim: 17,
            conv_kernel_dim: 5,
        },
        moe_fields: "",
    }
}

fn parse(sample: &Sample) -> ModelConfig {
    let moe = if sample.moe_fields.is_empty() {
        String::new()
    } else {
        format!(",\n    {}", sample.moe_fields)
    };
    let json = format!(
        r#"{{
  "architectures": [{arch:?}],
  "text_config": {{
    "num_hidden_layers": {num_layers},
    "hidden_size": {hidden_size},
    "head_dim": {head_dim},
    "num_attention_heads": {num_q_heads},
    "num_key_value_heads": {num_kv_heads},
    "vocab_size": {vocab_size},
    "tie_word_embeddings": false,
    "layer_types": {layer_types},
    "linear_conv_kernel_dim": {conv},
    "linear_num_key_heads": {k_heads},
    "linear_num_value_heads": {v_heads},
    "linear_key_head_dim": {key_dim},
    "linear_value_head_dim": {value_dim},
    "attn_output_gate": true{moe}
  }},
  "quantization_config": {{
    "quant_method": "exl3",
    "bits": 4.0,
    "head_bits": 6
  }}
}}"#,
        arch = sample.arch,
        num_layers = sample.num_layers,
        hidden_size = sample.hidden_size,
        head_dim = sample.head_dim,
        num_q_heads = sample.num_q_heads,
        num_kv_heads = sample.num_kv_heads,
        vocab_size = sample.vocab_size,
        layer_types = sample.layer_types,
        conv = sample.gdn.conv_kernel_dim,
        k_heads = sample.gdn.num_k_heads,
        v_heads = sample.gdn.num_v_heads,
        key_dim = sample.gdn.key_head_dim,
        value_dim = sample.gdn.value_head_dim,
        moe = moe,
    );
    ModelConfig::from_slice(json.as_bytes()).unwrap_or_else(|err| panic!("{err}\n{json}"))
}

#[test]
fn dense_config_copies_widths_and_layer_order() {
    let sample = dense_sample();
    let shape = model_shape(&parse(&sample)).unwrap();
    assert_eq!(
        shape,
        ModelShape {
            arch_string: ARCH_QWEN3_5.to_string(),
            num_layers: 2,
            layer_kinds: vec![LayerKind::FullAttention, LayerKind::LinearAttention],
            hidden_size: 96,
            head_dim: 40,
            num_q_heads: 6,
            num_kv_heads: 3,
            gdn: sample.gdn,
            moe: None,
            vocab_size: 12_345,
        }
    );
    let plan = build_plan(&shape).unwrap();
    assert_eq!(
        plan.layers
            .iter()
            .map(|layer| layer.mixer)
            .collect::<Vec<_>>(),
        vec![Mixer::FullAttn, Mixer::Gdn]
    );
}

#[test]
fn moe_config_copies_all_four_expert_fields() {
    let sample = Sample {
        arch: ARCH_QWEN3_5_MOE,
        num_layers: 2,
        layer_types: r#"["linear_attention", "full_attention"]"#,
        hidden_size: 2048,
        head_dim: 256,
        num_q_heads: 16,
        num_kv_heads: 2,
        vocab_size: 248_144,
        gdn: GdnDims {
            num_k_heads: 16,
            num_v_heads: 32,
            key_head_dim: 128,
            value_head_dim: 64,
            conv_kernel_dim: 4,
        },
        moe_fields: r#""num_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 512, "shared_expert_intermediate_size": 768"#,
    };
    let shape = model_shape(&parse(&sample)).unwrap();
    assert_eq!(
        shape,
        ModelShape {
            arch_string: ARCH_QWEN3_5_MOE.to_string(),
            num_layers: 2,
            layer_kinds: vec![LayerKind::LinearAttention, LayerKind::FullAttention],
            hidden_size: 2048,
            head_dim: 256,
            num_q_heads: 16,
            num_kv_heads: 2,
            gdn: sample.gdn,
            moe: Some(MoeDims {
                num_experts: 256,
                experts_per_tok: 8,
                moe_intermediate: 512,
                shared_intermediate: 768,
            }),
            vocab_size: 248_144,
        }
    );
    let plan = build_plan(&shape).unwrap();
    assert_eq!(plan.layers.len(), 2);
    assert_eq!(plan.layers[0].mixer, Mixer::Gdn);
    assert_eq!(plan.layers[1].mixer, Mixer::FullAttn);
}

#[test]
fn unknown_layer_type_names_the_index_and_string() {
    let mut sample = dense_sample();
    sample.layer_types = r#"["linear_attention", "qsa"]"#;
    let err = model_shape(&parse(&sample)).unwrap_err();
    assert_eq!(
        err,
        ConfigError::UnknownLayerType {
            index: 1,
            name: "qsa".to_string(),
        }
    );
    assert_eq!(err.to_string(), "layer 1: unknown layer type `qsa`");
}

#[test]
fn partial_moe_names_the_missing_fields() {
    let mut sample = dense_sample();
    sample.moe_fields = r#""num_experts": 256, "moe_intermediate_size": 512"#;
    let err = model_shape(&parse(&sample)).unwrap_err();
    assert_eq!(
        err,
        ConfigError::IncompleteMoe {
            missing: vec!["num_experts_per_tok", "shared_expert_intermediate_size"],
        }
    );
    assert_eq!(
        err.to_string(),
        "incomplete MoE config, missing num_experts_per_tok, shared_expert_intermediate_size"
    );

    sample.moe_fields = r#""shared_expert_intermediate_size": 32"#;
    let err = model_shape(&parse(&sample)).unwrap_err();
    assert_eq!(
        err,
        ConfigError::IncompleteMoe {
            missing: vec![
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
            ],
        }
    );
    assert_eq!(
        err.to_string(),
        "incomplete MoE config, missing num_experts, num_experts_per_tok, moe_intermediate_size"
    );
}

#[test]
fn moe_option_follows_the_four_fields() {
    let mut with_experts = dense_sample();
    with_experts.moe_fields = r#""num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 8, "shared_expert_intermediate_size": 16"#;
    let shape = model_shape(&parse(&with_experts)).unwrap();
    assert_eq!(shape.arch_string, ARCH_QWEN3_5);
    assert_eq!(
        shape.moe,
        Some(MoeDims {
            num_experts: 4,
            experts_per_tok: 2,
            moe_intermediate: 8,
            shared_intermediate: 16,
        })
    );

    let mut bare = dense_sample();
    bare.arch = ARCH_QWEN3_5_MOE;
    let shape = model_shape(&parse(&bare)).unwrap();
    assert_eq!(shape.arch_string, ARCH_QWEN3_5_MOE);
    assert_eq!(shape.moe, None);
}

fn model_dir() -> String {
    std::env::var("BERTH_27B_DIR")
        .unwrap_or_else(|_| "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3".to_string())
}

#[test]
fn qwen38_27b_shape_builds_a_plan() {
    let dir = model_dir();
    if !Path::new(&dir).is_dir() {
        eprintln!("skipping qwen38_27b: directory missing: {dir}");
        return;
    }

    let ckpt = Checkpoint::open(&dir).unwrap();
    let shape = model_shape(&ckpt.config).unwrap();
    let plan = build_plan(&shape).unwrap();

    assert_eq!(shape.arch_string, ARCH_QWEN3_5);
    assert_eq!(shape.num_layers, 64);
    assert_eq!(shape.layer_kinds.len(), 64);
    assert_eq!(shape.hidden_size, 5120);
    assert_eq!(shape.head_dim, 256);
    assert_eq!(shape.num_q_heads, 24);
    assert_eq!(shape.num_kv_heads, 4);
    assert_eq!(
        shape.gdn,
        GdnDims {
            num_k_heads: 16,
            num_v_heads: 48,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_dim: 4,
        }
    );
    assert_eq!(shape.vocab_size, 248_320);
    assert_eq!(shape.moe, None);

    assert_eq!(plan.layers.len(), 64);
    let mut gdn = 0;
    let mut full = 0;
    for (index, layer_type) in ckpt.config.layer_types.iter().enumerate() {
        let (kind, mixer) = match layer_type {
            LayerType::LinearAttention => {
                gdn += 1;
                (LayerKind::LinearAttention, Mixer::Gdn)
            }
            LayerType::FullAttention => {
                full += 1;
                (LayerKind::FullAttention, Mixer::FullAttn)
            }
            LayerType::Other(name) => panic!("layer {index}: {name}"),
        };
        assert_eq!(shape.layer_kinds[index], kind, "layer {index}");
        assert_eq!(plan.layers[index].mixer, mixer, "layer {index}");
        assert_eq!(plan.layers[index].index, index as u32);
    }
    assert_eq!(gdn, 48);
    assert_eq!(full, 16);
}

//! Read the real Qwen3.8-27B EXL3 checkpoint. Headers only.

use std::path::Path;

use berth_exl3::{Bitrate, Checkpoint, Codebook, DType, HadamardScale, LayerType};

fn model_dir() -> String {
    std::env::var("BERTH_27B_DIR")
        .unwrap_or_else(|_| "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3".to_string())
}

#[test]
fn qwen38_27b_checkpoint() {
    let dir = model_dir();
    if !Path::new(&dir).is_dir() {
        eprintln!("skipping qwen38_27b: directory missing: {dir}");
        return;
    }

    let ckpt = Checkpoint::open(&dir).unwrap();
    let cfg = &ckpt.config;
    assert_eq!(cfg.arch_string, "Qwen3_5ForConditionalGeneration");
    assert_eq!(cfg.num_hidden_layers, 64);
    assert_eq!(cfg.layer_types.len(), 64);
    let full = cfg
        .layer_types
        .iter()
        .filter(|kind| matches!(kind, LayerType::FullAttention))
        .count();
    let gdn = cfg
        .layer_types
        .iter()
        .filter(|kind| matches!(kind, LayerType::LinearAttention))
        .count();
    assert_eq!(full, 16);
    assert_eq!(gdn, 48);
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.head_dim, 256);
    // Not the exllamav3 default of 32, so this was read from text_config.
    assert_eq!(cfg.linear_num_value_heads, 48);
    assert!(cfg.attn_output_gate);
    assert_eq!(cfg.intermediate_size, Some(17408));
    assert_eq!(cfg.quantization.quant_method, "exl3");
    assert_eq!(cfg.quantization.bits, 4.0);
    assert_eq!(cfg.quantization.head_bits, 6);
    assert_eq!(cfg.quantization.codebook.as_deref(), Some("mul1"));

    assert_eq!(ckpt.tensors.len(), 3080);
    assert_eq!(ckpt.linears().len(), 573);
    assert_eq!(ckpt.other_tensors().len(), 624);

    let mut layer_linears = 0usize;
    for linear in ckpt.linears() {
        assert_eq!(linear.codebook, Codebook::Mul1, "{}", linear.prefix);
        if linear.prefix.starts_with("model.language_model.layers.") {
            assert_eq!(linear.k, Bitrate::Int(4), "{}", linear.prefix);
            layer_linears += 1;
        }
    }
    assert_eq!(layer_linears, 400);

    let head = ckpt
        .linears()
        .iter()
        .find(|linear| linear.prefix == "lm_head")
        .expect("lm_head");
    assert_eq!(head.k, Bitrate::Int(6));
    assert_eq!(head.trellis.dtype, DType::I16);
    assert_eq!(head.trellis.shape, vec![320, 15520, 96]);
    assert_eq!(head.trellis.offset, 6_301_900_764);
    assert_eq!(head.trellis.length, 953_548_800);
    assert!(head
        .trellis
        .shard
        .ends_with("model-00002-of-00002.safetensors"));

    let q = ckpt
        .linears()
        .iter()
        .find(|linear| linear.prefix == "model.language_model.layers.3.self_attn.q_proj")
        .expect("layer 3 q_proj");
    assert_eq!(q.k, Bitrate::Int(4));
    assert!(q.bias.is_none());
    assert_eq!(q.trellis.shape, vec![320, 768, 64]);
    assert_eq!(q.trellis.offset, 3_123_577_554);
    assert_eq!(q.trellis.length, 31_457_280);
    assert!(q
        .trellis
        .shard
        .ends_with("model-00001-of-00002.safetensors"));
    match &q.input_scale {
        HadamardScale::Unpacked(info) => {
            assert_eq!(info.dtype, DType::F16);
            assert_eq!(info.shape, vec![5120]);
            assert_eq!(info.offset, 3_123_542_734);
            assert_eq!(info.length, 10_240);
        }
        HadamardScale::Packed(_) => panic!("q_proj stores suh, not packed su"),
    }

    let embed = ckpt
        .tensors
        .get("model.language_model.embed_tokens.weight")
        .expect("embed");
    assert_eq!(embed.dtype, DType::Bf16);
    assert_eq!(embed.shape, vec![248_320, 5120]);
    assert_eq!(embed.offset, 125_762);
    assert_eq!(embed.length, 2_542_796_800);

    let others: Vec<&str> = ckpt
        .other_tensors()
        .iter()
        .map(|info| info.name.as_str())
        .collect();
    for name in [
        "model.language_model.embed_tokens.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.dt_bias",
        "model.language_model.layers.0.linear_attn.conv1d.weight",
    ] {
        assert!(others.contains(&name), "missing other tensor {name}");
    }
    assert!(others.iter().all(|name| !name.ends_with(".trellis")));
    assert!(others.iter().all(|name| !name.ends_with(".suh")));
}

use berth_plan::{
    build_plan, is_routed_expert, place, Berth, Ffn, GdnDims, LayerKind, MachineBudget, Mixer,
    ModelShape, MoeDims, PlaceError, PlanError, ShapeError, TensorBytes, ARCH_MINICPM_V4_7,
    ARCH_QWEN3_5, ARCH_QWEN3_5_MOE, ARCH_QWEN4_EXP,
};

/// Qwen3.5 `full_attention_interval`: layers 3, 7, 11, … are full attention.
fn every_fourth_full(n: u32) -> Vec<LayerKind> {
    (0..n)
        .map(|index| {
            if (index + 1) % 4 == 0 {
                LayerKind::FullAttention
            } else {
                LayerKind::LinearAttention
            }
        })
        .collect()
}

fn gdn_27b() -> GdnDims {
    // qwen38-27b-exl3 `text_config`. `num_v_heads` is 48; that is not the
    // count of GDN layers (also 48 on this model).
    GdnDims {
        num_k_heads: 16,
        num_v_heads: 48,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel_dim: 4,
    }
}

fn shape_27b() -> ModelShape {
    ModelShape {
        arch_string: ARCH_QWEN3_5.to_string(),
        num_layers: 64,
        layer_kinds: every_fourth_full(64),
        hidden_size: 5120,
        head_dim: 256,
        num_q_heads: 24,
        num_kv_heads: 4,
        gdn: gdn_27b(),
        moe: None,
        vocab_size: 248_320,
    }
}

fn shape_35b() -> ModelShape {
    // Head counts other than KV come from the shape the caller supplies.
    // `docs/models.md` fixes hidden 2048, head dim 256, 2 KV heads, 256
    // experts, top-8, intermediate 512, and one shared expert. GDN width is
    // the Qwen3.5 loader default; this checkpoint is not on disk here.
    ModelShape {
        arch_string: ARCH_QWEN3_5_MOE.to_string(),
        num_layers: 40,
        layer_kinds: every_fourth_full(40),
        hidden_size: 2048,
        head_dim: 256,
        num_q_heads: 16,
        num_kv_heads: 2,
        gdn: GdnDims {
            num_k_heads: 16,
            num_v_heads: 32,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_dim: 4,
        },
        moe: Some(MoeDims {
            num_experts: 256,
            experts_per_tok: 8,
            moe_intermediate: 512,
            shared_intermediate: 512,
        }),
        vocab_size: 248_320,
    }
}

fn count_mixers(plan: &berth_plan::ModelPlan) -> (usize, usize, usize) {
    let mut gdn = 0;
    let mut full = 0;
    let mut qsa = 0;
    for layer in &plan.layers {
        match layer.mixer {
            Mixer::Gdn => gdn += 1,
            Mixer::FullAttn => full += 1,
            Mixer::Qsa => qsa += 1,
        }
    }
    (gdn, full, qsa)
}

fn tensor(name: &str, bytes: u64) -> (String, u64) {
    (name.to_string(), bytes)
}

/// `docs/models.md` table: 16.1 + 1.2 + 0.25 = 17.55 GB, quoted as about 17.6.
fn bytes_35b() -> TensorBytes {
    TensorBytes::from(vec![
        tensor("model.language_model.embed_tokens.weight", 400_000_000),
        tensor(
            "model.language_model.layers.0.mlp.shared_expert.down_proj.trellis",
            500_000_000,
        ),
        tensor(
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            10_000,
        ),
        tensor("model.language_model.layers.0.mlp.gate.weight", 49_990_000),
        tensor("model.language_model.norm.weight", 10_000),
        tensor("lm_head.weight", 249_990_000),
        tensor("model.visual.patch_embed.proj.weight", 250_000_000),
        tensor(
            "model.language_model.layers.0.mlp.experts.gate_up_proj.trellis",
            10_000_000_000,
        ),
        tensor(
            "model.language_model.layers.3.mlp.experts.down_proj.trellis",
            6_100_000_000,
        ),
    ])
}

fn assert_interval(plan: &berth_plan::ModelPlan) {
    for layer in &plan.layers {
        let expect = if (layer.index + 1) % 4 == 0 {
            Mixer::FullAttn
        } else {
            Mixer::Gdn
        };
        assert_eq!(layer.mixer, expect, "layer {}", layer.index);
        assert!(layer.extra.is_none());
    }
}

#[test]
fn qwen3_5_27b_plan_fits_in_vram() {
    let plan = build_plan(&shape_27b()).unwrap();
    assert_eq!(plan.arch_string, ARCH_QWEN3_5);
    assert_eq!(plan.layers.len(), 64);
    assert_eq!(count_mixers(&plan), (48, 16, 0));
    assert_interval(&plan);
    assert!(plan.layers.iter().all(|layer| layer.ffn == Ffn::DenseMlp));
    assert_eq!(plan.gdn, gdn_27b());
    assert_eq!(plan.hidden_size, 5120);
    assert_eq!(plan.head_dim, 256);
    assert_eq!(plan.num_q_heads, 24);
    assert_eq!(plan.num_kv_heads, 4);
    assert_eq!(plan.vocab_size, 248_320);
    assert_eq!(plan.embedding.vocab_size, 248_320);
    assert_eq!(plan.embedding.hidden_size, 5120);
    assert_eq!(plan.final_norm.hidden_size, 5120);
    assert_eq!(plan.lm_head.hidden_size, 5120);
    assert_eq!(plan.lm_head.vocab_size, 248_320);

    let weights = TensorBytes::from(vec![
        tensor("model.language_model.embed_tokens.weight", 2_000_000_000),
        tensor(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.trellis",
            4_000_000_000,
        ),
        tensor(
            "model.language_model.layers.3.self_attn.q_proj.trellis",
            1_000_000_000,
        ),
        tensor(
            "model.language_model.layers.0.mlp.down_proj.trellis",
            3_000_000_000,
        ),
        tensor("model.language_model.norm.weight", 20_000),
        tensor("lm_head.weight", 3_199_980_000),
    ]);
    let total: u64 = weights.tensors.iter().map(|(_, bytes)| bytes).sum();
    assert_eq!(total, 13_200_000_000);

    let placed = place(&plan, &weights, &MachineBudget::this_machine(4_600_000_000)).unwrap();
    assert!(placed
        .tensors
        .iter()
        .all(|tensor| tensor.berth == Berth::Vram));
    assert_eq!(placed.resident.vram, total);
    assert_eq!(placed.resident.pinned_ram, 0);
    assert_eq!(placed.resident.ssd, 0);
    assert_eq!(placed.bytes_moved.vram, total);
    assert_eq!(placed.bytes_moved.pinned_ram, 0);
    assert_eq!(placed.bytes_moved.ssd, 0);
    assert_eq!(
        placed
            .tensors
            .iter()
            .map(|tensor| tensor.name.as_str())
            .collect::<Vec<_>>(),
        weights
            .tensors
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>()
    );
}

#[test]
fn qwen3_5_moe_35b_fits_in_vram() {
    let plan = build_plan(&shape_35b()).unwrap();
    assert_eq!(plan.layers.len(), 40);
    assert_eq!(count_mixers(&plan), (30, 10, 0));
    assert_interval(&plan);
    assert_eq!(plan.num_kv_heads, 2);
    assert_eq!(plan.head_dim, 256);
    assert_eq!(plan.hidden_size, 2048);
    for layer in &plan.layers {
        match layer.ffn {
            Ffn::Moe {
                router,
                shared,
                experts,
            } => {
                assert_eq!(router.num_experts, 256);
                assert_eq!(router.experts_per_tok, 8);
                assert_eq!(shared.intermediate_size, 512);
                assert_eq!(experts.num_experts, 256);
                assert_eq!(experts.experts_per_tok, 8);
                assert_eq!(experts.intermediate_size, 512);
            }
            Ffn::DenseMlp => panic!("layer {} is dense", layer.index),
        }
    }

    let weights = bytes_35b();
    let total: u64 = weights.tensors.iter().map(|(_, bytes)| bytes).sum();
    assert_eq!(total, 17_550_000_000);

    let placed = place(&plan, &weights, &MachineBudget::this_machine(2_500_000_000)).unwrap();
    assert!(placed
        .tensors
        .iter()
        .all(|tensor| tensor.berth == Berth::Vram));
    assert_eq!(placed.resident.vram, total);
    assert_eq!(placed.resident.pinned_ram, 0);
    // Non-expert names in this list sum to 1.45 GB: the 1.2 GB language
    // stack plus the 0.25 GB vision tower. Listed vision counts as backbone.
    // Expert traffic is 8/256 of the 16.1 GB expert tensors.
    let expert_slice = 16_100_000_000u64 * 8 / 256;
    assert_eq!(expert_slice, 503_125_000);
    assert_eq!(placed.bytes_moved.vram, 1_450_000_000 + expert_slice);
    assert_eq!(placed.bytes_moved.pinned_ram, 0);
    assert_eq!(placed.bytes_moved.ssd, 0);
}

#[test]
fn qwen3_5_moe_spills_experts_to_pinned_ram() {
    let plan = build_plan(&shape_35b()).unwrap();
    let weights = bytes_35b();
    let mut budget = MachineBudget::this_machine(2_500_000_000);
    budget.vram_bytes = 8_000_000_000;

    let placed = place(&plan, &weights, &budget).unwrap();
    for tensor in &placed.tensors {
        let expect = if is_routed_expert(&tensor.name) {
            Berth::PinnedRam
        } else {
            Berth::Vram
        };
        assert_eq!(tensor.berth, expect, "{}", tensor.name);
    }
    assert_eq!(placed.resident.vram, 1_450_000_000);
    assert_eq!(placed.resident.pinned_ram, 16_100_000_000);
    assert_eq!(placed.resident.ssd, 0);
    assert_eq!(placed.bytes_moved.vram, 1_450_000_000);
    assert_eq!(placed.bytes_moved.pinned_ram, 503_125_000);
    assert_eq!(placed.bytes_moved.ssd, 0);

    let shared = placed
        .tensors
        .iter()
        .find(|tensor| tensor.name.contains("shared_expert.down_proj"))
        .unwrap();
    assert_eq!(shared.berth, Berth::Vram);
    let gate = placed
        .tensors
        .iter()
        .find(|tensor| tensor.name.ends_with("mlp.gate.weight"))
        .unwrap();
    assert_eq!(gate.berth, Berth::Vram);
}

#[test]
fn later_archs_are_not_yet_and_unknown_archs_error() {
    for arch in [ARCH_MINICPM_V4_7, ARCH_QWEN4_EXP] {
        let mut shape = shape_35b();
        shape.arch_string = arch.to_string();
        assert_eq!(
            build_plan(&shape).unwrap_err(),
            PlanError::NotYet(arch.to_string())
        );
    }
    for arch in [
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
        "LlamaForCausalLM",
        "",
    ] {
        let mut shape = shape_27b();
        shape.arch_string = arch.to_string();
        assert_eq!(
            build_plan(&shape).unwrap_err(),
            PlanError::Unknown(arch.to_string())
        );
    }
}

#[test]
fn bad_shapes_are_shape_errors() {
    let mut dense = shape_27b();
    dense.layer_kinds.pop();
    assert!(matches!(
        build_plan(&dense).unwrap_err(),
        PlanError::Shape(ShapeError::LayerCount {
            got: 63,
            num_layers: 64
        })
    ));

    let mut with_moe = shape_27b();
    with_moe.moe = shape_35b().moe;
    assert!(matches!(
        build_plan(&with_moe).unwrap_err(),
        PlanError::Shape(ShapeError::UnexpectedMoe { .. })
    ));

    let mut missing = shape_35b();
    missing.moe = None;
    assert!(matches!(
        build_plan(&missing).unwrap_err(),
        PlanError::Shape(ShapeError::MissingMoe { .. })
    ));

    let mut zero = shape_27b();
    zero.hidden_size = 0;
    assert!(matches!(
        build_plan(&zero).unwrap_err(),
        PlanError::Shape(ShapeError::Zero {
            field: "hidden_size"
        })
    ));

    let mut too_many = shape_35b();
    too_many.moe.as_mut().unwrap().experts_per_tok = 300;
    assert!(matches!(
        build_plan(&too_many).unwrap_err(),
        PlanError::Shape(ShapeError::ExpertsPerTok {
            experts_per_tok: 300,
            num_experts: 256
        })
    ));

    assert!(matches!(
        LayerKind::from_config("sliding_attention").unwrap_err(),
        ShapeError::UnknownLayerKind(name) if name == "sliding_attention"
    ));
    assert_eq!(
        LayerKind::from_config("linear_attention").unwrap(),
        LayerKind::LinearAttention
    );
}

#[test]
fn dense_overflow_does_not_spill() {
    let plan = build_plan(&shape_27b()).unwrap();
    let weights = TensorBytes::from(vec![tensor("lm_head.weight", 13_200_000_000)]);
    let mut budget = MachineBudget::this_machine(4_600_000_000);
    budget.vram_bytes = 16 * 1024 * 1024 * 1024;
    let err = place(&plan, &weights, &budget).unwrap_err();
    assert!(matches!(err, PlaceError::DenseExceedsVram { .. }), "{err}");
}

#[test]
fn moe_overflow_errors_when_backbone_or_pinned_ram_cannot_hold() {
    let plan = build_plan(&shape_35b()).unwrap();
    let weights = bytes_35b();

    let err = place(
        &plan,
        &weights,
        &MachineBudget::this_machine(24 * 1024 * 1024 * 1024 + 1),
    )
    .unwrap_err();
    assert!(
        matches!(err, PlaceError::ReservationExceedsVram { .. }),
        "{err}"
    );

    let mut tight = MachineBudget::this_machine(2_500_000_000);
    tight.vram_bytes = 3_000_000_000;
    let err = place(&plan, &weights, &tight).unwrap_err();
    assert!(
        matches!(err, PlaceError::BackboneExceedsVram { .. }),
        "{err}"
    );

    let mut pinned = MachineBudget::this_machine(2_500_000_000);
    pinned.vram_bytes = 8_000_000_000;
    pinned.pinned_ram_bytes = 1_000_000_000;
    let err = place(&plan, &weights, &pinned).unwrap_err();
    assert!(
        matches!(err, PlaceError::ExpertsExceedPinnedRam { .. }),
        "{err}"
    );
}

#[test]
fn routed_expert_name_rule() {
    assert!(is_routed_expert(
        "model.language_model.layers.0.mlp.experts.gate_up_proj.trellis"
    ));
    assert!(is_routed_expert(
        "model.language_model.layers.0.mlp.experts.12.down_proj.trellis"
    ));
    assert!(!is_routed_expert(
        "model.language_model.layers.0.mlp.shared_expert.down_proj.trellis"
    ));
    assert!(!is_routed_expert(
        "model.language_model.layers.0.mlp.shared_expert_gate.weight"
    ));
    assert!(!is_routed_expert(
        "model.language_model.layers.0.mlp.gate.weight"
    ));
    assert!(!is_routed_expert(
        "model.language_model.layers.0.linear_attn.out_proj.trellis"
    ));
    assert!(!is_routed_expert("lm_head.weight"));
}

#[test]
fn this_machine_is_24_gib_vram_and_18_gib_pinned() {
    let budget = MachineBudget::this_machine(4);
    assert_eq!(budget.vram_bytes, 24 << 30);
    assert_eq!(budget.pinned_ram_bytes, 18 << 30);
    assert_eq!(budget.vram_reserved_bytes, 4);
    assert_eq!(MachineBudget::default().vram_bytes, budget.vram_bytes);
    assert_eq!(
        MachineBudget::default().pinned_ram_bytes,
        budget.pinned_ram_bytes
    );
    assert_eq!(MachineBudget::default().vram_reserved_bytes, 0);
}

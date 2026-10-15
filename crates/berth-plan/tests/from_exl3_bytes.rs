//! `Checkpoint` → decode-path `TensorBytes`.

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use berth_exl3::Checkpoint;
use berth_plan::from_exl3::{tensor_bytes, MTP_PREFIX, VISION_TOWER_PREFIX};
use berth_plan::{
    build_plan, is_routed_expert, place, Berth, GdnDims, LayerKind, MachineBudget, ModelShape,
    ARCH_QWEN3_5,
};

/// Prefixes cited from the 27B header dump. Literals, not the library
/// constants, so a drifted constant fails this file.
const VISION: &str = "model.visual.";
const MTP: &str = "mtp.";

const LINEAR_PARTS: &[&str] = &["trellis", "suh", "svh", "su", "sv", "bias", "mcg", "mul1"];

fn model_dir() -> String {
    std::env::var("BERTH_27B_DIR")
        .unwrap_or_else(|_| "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3".to_string())
}

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

fn shape_27b() -> ModelShape {
    // Hand-written. `crates/berth-plan/tests/plan.rs` uses the same shape.
    // `num_v_heads` is 48; that is not the count of GDN layers.
    ModelShape {
        arch_string: ARCH_QWEN3_5.to_string(),
        num_layers: 64,
        layer_kinds: every_fourth_full(64),
        hidden_size: 5120,
        head_dim: 256,
        num_q_heads: 24,
        num_kv_heads: 4,
        gdn: GdnDims {
            num_k_heads: 16,
            num_v_heads: 48,
            key_head_dim: 128,
            value_head_dim: 128,
            conv_kernel_dim: 4,
        },
        moe: None,
        vocab_size: 248_320,
    }
}

fn off_decode_path(name: &str) -> bool {
    name.starts_with(VISION) || name.starts_with(MTP)
}

#[test]
fn qwen38_27b_decode_bytes_fit_in_vram() {
    let dir = model_dir();
    if !Path::new(&dir).is_dir() {
        eprintln!("skipping qwen38_27b decode bytes: directory missing: {dir}");
        return;
    }

    assert_eq!(VISION_TOWER_PREFIX, VISION);
    assert_eq!(MTP_PREFIX, MTP);

    let ckpt = Checkpoint::open(&dir).unwrap();
    assert!(
        ckpt.tensors
            .iter()
            .any(|(name, _)| name.starts_with(VISION)),
        "27B header has no vision tensors under {VISION}"
    );
    assert!(
        ckpt.tensors.iter().any(|(name, _)| name.starts_with(MTP)),
        "27B header has no MTP tensors under {MTP}"
    );

    let bytes = tensor_bytes(&ckpt);
    assert!(bytes.tensors.windows(2).all(|pair| pair[0].0 < pair[1].0));
    for (name, _) in &bytes.tensors {
        assert!(!off_decode_path(name), "{name}");
    }

    let all: u64 = ckpt.tensors.iter().map(|(_, info)| info.length).sum();
    let vision: u64 = ckpt
        .tensors
        .iter()
        .filter(|(name, _)| name.starts_with(VISION))
        .map(|(_, info)| info.length)
        .sum();
    let mtp: u64 = ckpt
        .tensors
        .iter()
        .filter(|(name, _)| name.starts_with(MTP))
        .map(|(_, info)| info.length)
        .sum();
    let excluded = vision + mtp;
    let listed: u64 = bytes.tensors.iter().map(|(_, len)| len).sum();
    eprintln!("decode bytes: {listed}");
    eprintln!("vision/MTP bytes excluded: {excluded} (vision {vision}, mtp {mtp})");
    assert_eq!(
        listed + excluded,
        all,
        "listed {listed} + excluded {excluded} != index {all}"
    );

    let decode_linears = ckpt
        .linears()
        .iter()
        .filter(|linear| !off_decode_path(&linear.prefix))
        .count();
    let decode_others = ckpt
        .other_tensors()
        .iter()
        .filter(|info| !off_decode_path(&info.name))
        .count();
    assert_eq!(bytes.tensors.len(), decode_linears + decode_others);

    let lm_head = bytes
        .tensors
        .iter()
        .find(|(name, _)| name == "lm_head")
        .expect("lm_head");
    assert_eq!(lm_head.1, owned_len(&ckpt, "lm_head"));
    let embed = ckpt
        .tensors
        .get("model.language_model.embed_tokens.weight")
        .expect("embed");
    let embed_len = bytes
        .tensors
        .iter()
        .find(|(name, _)| name == "model.language_model.embed_tokens.weight")
        .map(|(_, len)| *len);
    assert_eq!(embed_len, Some(embed.length));
    assert!(bytes
        .tensors
        .iter()
        .any(|(name, _)| name == "model.language_model.layers.0.linear_attn.A_log"));

    let plan = build_plan(&shape_27b()).unwrap();
    let placed = place(&plan, &bytes, &MachineBudget::this_machine(0)).unwrap();
    assert!(placed
        .tensors
        .iter()
        .all(|tensor| tensor.berth == Berth::Vram));
    assert_eq!(placed.resident.vram, listed);
    assert_eq!(placed.resident.pinned_ram, 0);
    assert_eq!(placed.resident.ssd, 0);
    assert_eq!(placed.bytes_moved.vram, listed);
    assert_eq!(placed.bytes_moved.pinned_ram, 0);
    assert_eq!(placed.bytes_moved.ssd, 0);
}

fn owned_len(ckpt: &Checkpoint, prefix: &str) -> u64 {
    LINEAR_PARTS
        .iter()
        .filter_map(|suffix| {
            ckpt.tensors
                .get(&format!("{prefix}.{suffix}"))
                .map(|info| info.length)
        })
        .sum()
}

const CONFIG: &str = r#"{
  "architectures": ["TestForCausalLM"],
  "text_config": {
    "num_hidden_layers": 2,
    "hidden_size": 32,
    "intermediate_size": 64,
    "head_dim": 16,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "vocab_size": 128,
    "tie_word_embeddings": false,
    "layer_types": ["linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 4,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "attn_output_gate": false
  },
  "quantization_config": {
    "quant_method": "exl3",
    "bits": 4,
    "head_bits": 6,
    "codebook": "mul1"
  }
}"#;

struct Tmp(PathBuf);

impl Tmp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("berth-plan-bytes-{}-{n}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct OwnedTensor {
    name: String,
    dtype: &'static str,
    shape: Vec<u64>,
    data: Vec<u8>,
}

fn tensor(name: &str, dtype: &'static str, shape: &[u64]) -> OwnedTensor {
    let elem: usize = match dtype {
        "I16" | "F16" | "BF16" => 2,
        "I32" | "F32" => 4,
        other => panic!("test dtype {other}"),
    };
    let numel = shape.iter().product::<u64>() as usize;
    OwnedTensor {
        name: name.to_string(),
        dtype,
        shape: shape.to_vec(),
        data: vec![0; numel * elem],
    }
}

fn push_part(
    out: &mut Vec<OwnedTensor>,
    prefix: &str,
    suffix: &str,
    dtype: &'static str,
    shape: &[u64],
) {
    out.push(tensor(&format!("{prefix}.{suffix}"), dtype, shape));
}

fn write_safetensors(path: &Path, tensors: &[OwnedTensor]) {
    let mut data = Vec::new();
    let mut entries = Vec::new();
    for tensor in tensors {
        let start = data.len();
        data.extend_from_slice(&tensor.data);
        let end = data.len();
        let shape = tensor
            .shape
            .iter()
            .map(|dim| dim.to_string())
            .collect::<Vec<_>>()
            .join(",");
        entries.push(format!(
            r#""{}":{{"dtype":"{}","shape":[{}],"data_offsets":[{start},{end}]}}"#,
            tensor.name, tensor.dtype, shape
        ));
    }
    let header = format!("{{{}}}", entries.join(","));
    let mut file = File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(&data).unwrap();
}

fn part_len(tensors: &[OwnedTensor], prefix: &str) -> u64 {
    tensors
        .iter()
        .filter(|tensor| {
            LINEAR_PARTS
                .iter()
                .any(|suffix| tensor.name == format!("{prefix}.{suffix}"))
        })
        .map(|tensor| tensor.data.len() as u64)
        .sum()
}

#[test]
fn sums_owned_parts_keeps_expert_names_and_drops_vision_mtp() {
    // Indexed expert stores both `su` and `suh`. `HadamardScale` keeps `suh`;
    // `su` is still this linear's tensor. Shared expert is packed `su`/`sv`
    // plus bias and `mcg`. `down_proj` is the default codebook (no marker).
    let expert = "model.language_model.layers.0.mlp.experts.3.gate_proj";
    let fused = "model.language_model.layers.1.mlp.experts.down_proj";
    let shared = "model.language_model.layers.0.mlp.shared_expert.down_proj";
    let dense = "model.language_model.layers.0.mlp.down_proj";
    let vision = "model.visual.blocks.0.attn.q_proj";
    let mtp_fc = "mtp.fc";

    let mut tensors = Vec::new();
    push_part(&mut tensors, expert, "trellis", "I16", &[1, 1, 64]);
    push_part(&mut tensors, expert, "suh", "F16", &[4]);
    push_part(&mut tensors, expert, "svh", "F16", &[4]);
    push_part(&mut tensors, expert, "su", "I16", &[1]);
    push_part(&mut tensors, expert, "mul1", "I32", &[]);
    push_part(&mut tensors, fused, "trellis", "I16", &[1, 1, 16]);
    push_part(&mut tensors, fused, "suh", "F16", &[2]);
    push_part(&mut tensors, fused, "svh", "F16", &[2]);
    push_part(&mut tensors, fused, "mul1", "I32", &[]);
    push_part(&mut tensors, shared, "trellis", "I16", &[1, 1, 32]);
    push_part(&mut tensors, shared, "su", "I16", &[1]);
    push_part(&mut tensors, shared, "sv", "I16", &[1]);
    push_part(&mut tensors, shared, "bias", "F16", &[2]);
    push_part(&mut tensors, shared, "mcg", "I32", &[]);
    push_part(&mut tensors, dense, "trellis", "I16", &[1, 1, 16]);
    push_part(&mut tensors, dense, "suh", "F16", &[2]);
    push_part(&mut tensors, dense, "svh", "F16", &[2]);
    push_part(&mut tensors, vision, "trellis", "I16", &[1, 1, 16]);
    push_part(&mut tensors, vision, "suh", "F16", &[2]);
    push_part(&mut tensors, vision, "svh", "F16", &[2]);
    push_part(&mut tensors, vision, "bias", "F16", &[2]);
    push_part(&mut tensors, vision, "mul1", "I32", &[]);
    push_part(&mut tensors, mtp_fc, "trellis", "I16", &[1, 1, 16]);
    push_part(&mut tensors, mtp_fc, "suh", "F16", &[2]);
    push_part(&mut tensors, mtp_fc, "svh", "F16", &[2]);
    push_part(&mut tensors, mtp_fc, "mul1", "I32", &[]);
    let others = [
        tensor("model.language_model.embed_tokens.weight", "BF16", &[2, 2]),
        tensor("model.language_model.norm.weight", "BF16", &[4]),
        tensor(
            "model.language_model.layers.0.input_layernorm.weight",
            "BF16",
            &[4],
        ),
        tensor(
            "model.language_model.layers.0.linear_attn.A_log",
            "F32",
            &[2],
        ),
        tensor(
            "model.language_model.layers.0.linear_attn.dt_bias",
            "F32",
            &[2],
        ),
        tensor(
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            "BF16",
            &[2, 1, 4],
        ),
        tensor(
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            "F32",
            &[4],
        ),
        tensor("model.visual.pos_embed.weight", "BF16", &[4]),
        tensor("mtp.norm.weight", "BF16", &[4]),
    ];
    tensors.extend(others);

    let tmp = Tmp::new();
    fs::write(tmp.0.join("config.json"), CONFIG).unwrap();
    write_safetensors(&tmp.0.join("model.safetensors"), &tensors);
    let ckpt = Checkpoint::open(&tmp.0).unwrap();
    let bytes = tensor_bytes(&ckpt);

    assert!(bytes.tensors.windows(2).all(|pair| pair[0].0 < pair[1].0));
    for (name, _) in &bytes.tensors {
        assert!(!off_decode_path(name), "{name}");
    }

    let entry = |name: &str| {
        bytes
            .tensors
            .iter()
            .find(|(got, _)| got == name)
            .unwrap_or_else(|| panic!("missing {name}"))
            .1
    };
    assert_eq!(entry(expert), part_len(&tensors, expert));
    assert_eq!(entry(fused), part_len(&tensors, fused));
    assert_eq!(entry(shared), part_len(&tensors, shared));
    assert_eq!(entry(dense), part_len(&tensors, dense));
    assert!(is_routed_expert(expert));
    assert!(is_routed_expert(fused));
    assert!(!is_routed_expert(shared));
    assert!(!is_routed_expert(
        "model.language_model.layers.0.mlp.shared_expert_gate.weight"
    ));

    // Both scales on the indexed expert: dropping packed `su` (2 bytes) or
    // the `mul1` marker (4 bytes) would miss `part_len`.
    assert!(part_len(&tensors, expert) > 128 + 8 + 8);

    for other in [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.layers.0.linear_attn.dt_bias",
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        "model.language_model.layers.0.mlp.shared_expert_gate.weight",
    ] {
        let len = tensors
            .iter()
            .find(|tensor| tensor.name == other)
            .unwrap()
            .data
            .len() as u64;
        assert_eq!(entry(other), len, "{other}");
    }

    let all: u64 = tensors.iter().map(|tensor| tensor.data.len() as u64).sum();
    let excluded: u64 = tensors
        .iter()
        .filter(|tensor| off_decode_path(&tensor.name))
        .map(|tensor| tensor.data.len() as u64)
        .sum();
    let listed: u64 = bytes.tensors.iter().map(|(_, len)| len).sum();
    assert!(excluded > 0);
    assert_eq!(listed + excluded, all);
}

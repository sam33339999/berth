use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use berth_exl3::{
    Bitrate, Checkpoint, Codebook, DType, Error, HadamardScale, HalfBitrate, LayerType,
    FP16_NEG_ONE, FP16_ONE,
};

const CONFIG: &str = r#"{
  "architectures": ["TestForCausalLM"],
  "ignored_top": true,
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
    "attn_output_gate": false,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 8,
    "shared_expert_intermediate_size": 16
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
        let path = std::env::temp_dir().join(format!("berth-exl3-{}-{n}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Clone)]
struct OwnedTensor {
    name: String,
    dtype: &'static str,
    shape: Vec<u64>,
    data: Vec<u8>,
}

fn zeros(n: usize) -> Vec<u8> {
    vec![0; n]
}

fn words(values: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn numel(shape: &[u64]) -> usize {
    shape.iter().product::<u64>() as usize
}

fn write_config(dir: &Path, text: &str) {
    fs::write(dir.join("config.json"), text).unwrap();
}

fn write_tensors(path: &Path, tensors: &[OwnedTensor]) -> u64 {
    write_tensors_padded(path, tensors, 0)
}

fn write_tensors_padded(path: &Path, tensors: &[OwnedTensor], pad: usize) -> u64 {
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
    let mut header = format!("{{{}}}", entries.join(","));
    header.push_str(&" ".repeat(pad));
    let mut file = File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(&data).unwrap();
    header.len() as u64
}

fn header_len(path: &Path) -> u64 {
    let mut file = File::open(path).unwrap();
    let mut buf = [0u8; 8];
    file.read_exact(&mut buf).unwrap();
    u64::from_le_bytes(buf)
}

fn fp16(n: usize) -> OwnedTensor {
    OwnedTensor {
        name: String::new(),
        dtype: "F16",
        shape: vec![n as u64],
        data: zeros(n * 2),
    }
}

fn marker(dtype_name: &'static str) -> OwnedTensor {
    OwnedTensor {
        name: String::new(),
        dtype: dtype_name,
        shape: Vec::new(),
        data: vec![1, 0, 0, 0],
    }
}

/// One EXL3 linear plus the caller fills names.
fn linear_parts(
    prefix: &str,
    last: u64,
    codebook: Option<&str>,
    packed: bool,
    bias: bool,
    su_word: u16,
    sv_word: u16,
) -> Vec<OwnedTensor> {
    let mut parts = vec![OwnedTensor {
        name: format!("{prefix}.trellis"),
        dtype: "I16",
        shape: vec![1, 1, last],
        data: zeros((last as usize) * 2),
    }];
    if packed {
        parts.push(OwnedTensor {
            name: format!("{prefix}.su"),
            dtype: "I16",
            shape: vec![1],
            data: words(&[su_word]),
        });
        parts.push(OwnedTensor {
            name: format!("{prefix}.sv"),
            dtype: "I16",
            shape: vec![1],
            data: words(&[sv_word]),
        });
    } else {
        let mut suh = fp16(4);
        suh.name = format!("{prefix}.suh");
        let mut svh = fp16(4);
        svh.name = format!("{prefix}.svh");
        parts.push(suh);
        parts.push(svh);
    }
    if let Some(kind) = codebook {
        let mut mark = marker("I32");
        mark.name = format!("{prefix}.{kind}");
        parts.push(mark);
    }
    if bias {
        let mut bias_tensor = fp16(2);
        bias_tensor.name = format!("{prefix}.bias");
        parts.push(bias_tensor);
    }
    parts
}

fn others() -> Vec<OwnedTensor> {
    vec![
        OwnedTensor {
            name: "model.norm.weight".into(),
            dtype: "F16",
            shape: vec![4],
            data: zeros(8),
        },
        OwnedTensor {
            name: "model.norm.bias".into(),
            dtype: "F16",
            shape: vec![4],
            data: zeros(8),
        },
        OwnedTensor {
            name: "model.layers.0.linear_attn.A_log".into(),
            dtype: "F32",
            shape: vec![2],
            data: zeros(8),
        },
        OwnedTensor {
            name: "model.layers.0.linear_attn.dt_bias".into(),
            dtype: "F32",
            shape: vec![2],
            data: zeros(8),
        },
        OwnedTensor {
            name: "model.layers.0.linear_attn.conv1d.weight".into(),
            dtype: "BF16",
            shape: vec![2, 1, 4],
            data: zeros(numel(&[2, 1, 4]) * 2),
        },
        OwnedTensor {
            name: "model.embed.weight".into(),
            dtype: "BF16",
            shape: vec![2, 2],
            data: zeros(8),
        },
    ]
}

fn demo_tensors() -> Vec<OwnedTensor> {
    let mut tensors = Vec::new();
    tensors.extend(linear_parts(
        "block.0.down",
        64,
        Some("mul1"),
        false,
        false,
        0,
        0,
    ));
    tensors.extend(linear_parts(
        "block.1.q",
        24,
        Some("mul1"),
        true,
        true,
        0x0005,
        0x8000,
    ));
    tensors.extend(linear_parts(
        "block.0.k",
        64,
        Some("mcg"),
        false,
        false,
        0,
        0,
    ));
    tensors.extend(linear_parts("block.0.v", 64, None, false, false, 0, 0));
    tensors.extend(linear_parts(
        "wide.up",
        128,
        Some("mul1"),
        false,
        false,
        0,
        0,
    ));
    tensors.extend(linear_parts(
        "narrow.q",
        16,
        Some("mul1"),
        false,
        false,
        0,
        0,
    ));
    tensors.extend(linear_parts("mid.o", 40, Some("mul1"), true, false, 0, 0));
    tensors.extend(linear_parts(
        "mid.down",
        56,
        Some("mul1"),
        true,
        false,
        0,
        0,
    ));
    let mut both = linear_parts("both.proj", 64, Some("mul1"), false, false, 0, 0);
    both.push(OwnedTensor {
        name: "both.proj.su".into(),
        dtype: "I16",
        shape: vec![1],
        data: words(&[0xffff]),
    });
    tensors.extend(both);
    tensors.extend(others());
    tensors
}

#[test]
fn synthetic_checkpoint_groups_linears_and_unpacks_signs() {
    let tmp = Tmp::new();
    write_config(tmp.path(), CONFIG);
    let tensors = demo_tensors();
    let shard = tmp.path().join("model.safetensors");
    let header = write_tensors(&shard, &tensors);

    let ckpt = Checkpoint::open(tmp.path()).unwrap();
    assert_eq!(ckpt.config.arch_string, "TestForCausalLM");
    assert_eq!(ckpt.config.hidden_size, 32);
    assert_eq!(ckpt.config.intermediate_size, Some(64));
    assert_eq!(ckpt.config.num_experts, Some(4));
    assert_eq!(ckpt.config.shared_expert_intermediate_size, Some(16));
    assert!(!ckpt.config.attn_output_gate);
    let full = ckpt
        .config
        .layer_types
        .iter()
        .filter(|kind| matches!(kind, LayerType::FullAttention))
        .count();
    assert_eq!(full, 1);
    assert_eq!(ckpt.config.quantization.bits, 4.0);

    assert_eq!(ckpt.linears().len(), 9);
    let by = |name: &str| {
        ckpt.linears()
            .iter()
            .find(|linear| linear.prefix == name)
            .unwrap_or_else(|| panic!("missing {name}"))
    };
    assert_eq!(by("block.0.down").k, Bitrate::Int(4));
    assert_eq!(by("block.0.down").codebook, Codebook::Mul1);
    assert!(by("block.0.down").bias.is_none());
    assert!(!by("block.0.down").input_scale.is_packed());

    let half = by("block.1.q");
    assert_eq!(half.k, Bitrate::Half(HalfBitrate::OnePointFive));
    assert_eq!(half.k.as_f64(), 1.5);
    assert!(half.input_scale.is_packed());
    assert!(half.bias.is_some());
    assert_eq!(half.bias.as_ref().unwrap().dtype, DType::F16);
    assert_eq!(
        half.unpack_input_signs().unwrap(),
        vec![
            FP16_NEG_ONE,
            FP16_ONE,
            FP16_NEG_ONE,
            FP16_ONE,
            FP16_ONE,
            FP16_ONE,
            FP16_ONE,
            FP16_ONE,
            FP16_ONE,
            FP16_ONE,
            FP16_ONE,
            FP16_ONE,
            FP16_ONE,
            FP16_ONE,
            FP16_ONE,
            FP16_ONE,
        ]
    );
    let output = half.unpack_output_signs().unwrap();
    assert_eq!(output[15], FP16_NEG_ONE);
    assert!(output[..15].iter().all(|&bit| bit == FP16_ONE));

    assert_eq!(by("block.0.k").codebook, Codebook::Mcg);
    assert_eq!(by("block.0.v").codebook, Codebook::Default);
    assert_eq!(by("wide.up").k, Bitrate::Int(8));
    assert_eq!(by("narrow.q").k, Bitrate::Int(1));
    assert_eq!(by("mid.o").k, Bitrate::Half(HalfBitrate::TwoPointFive));
    assert_eq!(by("mid.down").k, Bitrate::Half(HalfBitrate::ThreePointFive));

    let both = by("both.proj");
    assert!(matches!(both.input_scale, HadamardScale::Unpacked(_)));
    let err = both.unpack_input_signs().unwrap_err();
    assert!(err.to_string().contains("suh"), "{err}");
    assert!(ckpt.tensors.contains("both.proj.su"));

    let other_names: Vec<&str> = ckpt
        .other_tensors()
        .iter()
        .map(|info| info.name.as_str())
        .collect();
    for name in [
        "model.norm.weight",
        "model.norm.bias",
        "model.layers.0.linear_attn.A_log",
        "model.layers.0.linear_attn.dt_bias",
        "model.layers.0.linear_attn.conv1d.weight",
        "model.embed.weight",
    ] {
        assert!(other_names.contains(&name), "{name}");
    }
    assert!(other_names.iter().all(|name| !name.ends_with(".trellis")));
    assert!(other_names.iter().all(|name| !name.ends_with(".su")));
    assert!(other_names.iter().all(|name| !name.ends_with(".suh")));

    let trellis = ckpt.tensors.get("block.0.down.trellis").unwrap();
    assert_eq!(trellis.offset, 8 + header);
    assert_eq!(trellis.length, 128);
    assert_eq!(trellis.dtype, DType::I16);
    assert_eq!(trellis.shape, vec![1, 1, 64]);
    let suh = ckpt.tensors.get("block.0.down.suh").unwrap();
    assert_eq!(suh.offset, trellis.offset + trellis.length);
    assert_eq!(header_len(&shard), header);
}

#[test]
fn padded_header_still_locates_tensors() {
    let tmp = Tmp::new();
    write_config(tmp.path(), CONFIG);
    let tensors = linear_parts("only.proj", 16, Some("mul1"), false, false, 0, 0);
    let shard = tmp.path().join("model.safetensors");
    let header = write_tensors_padded(&shard, &tensors, 7);
    let ckpt = Checkpoint::open(tmp.path()).unwrap();
    let trellis = &ckpt.linears()[0].trellis;
    assert_eq!(trellis.offset, 8 + header);
    assert!(header > 7, "header should include the padding");
}

#[test]
fn sharded_index_checks_total_size() {
    let tmp = Tmp::new();
    write_config(tmp.path(), CONFIG);
    let shard_a = linear_parts("a.proj", 16, Some("mul1"), false, false, 0, 0);
    let mut shard_b = linear_parts("b.proj", 32, Some("mcg"), false, false, 0, 0);
    shard_b.extend(others());
    write_tensors(
        &tmp.path().join("model-00001-of-00002.safetensors"),
        &shard_a,
    );
    write_tensors(
        &tmp.path().join("model-00002-of-00002.safetensors"),
        &shard_b,
    );
    let bytes = shard_a
        .iter()
        .chain(&shard_b)
        .map(|tensor| tensor.data.len() as u64)
        .sum::<u64>();
    let index = format!(
        r#"{{
        "metadata": {{"total_size": {bytes}}},
        "weight_map": {{
            "a.proj.trellis": "model-00001-of-00002.safetensors",
            "a.proj.suh": "model-00001-of-00002.safetensors",
            "a.proj.svh": "model-00001-of-00002.safetensors",
            "a.proj.mul1": "model-00001-of-00002.safetensors",
            "b.proj.trellis": "model-00002-of-00002.safetensors",
            "b.proj.suh": "model-00002-of-00002.safetensors",
            "b.proj.svh": "model-00002-of-00002.safetensors",
            "b.proj.mcg": "model-00002-of-00002.safetensors",
            "model.norm.weight": "model-00002-of-00002.safetensors",
            "model.norm.bias": "model-00002-of-00002.safetensors",
            "model.layers.0.linear_attn.A_log": "model-00002-of-00002.safetensors",
            "model.layers.0.linear_attn.dt_bias": "model-00002-of-00002.safetensors",
            "model.layers.0.linear_attn.conv1d.weight": "model-00002-of-00002.safetensors",
            "model.embed.weight": "model-00002-of-00002.safetensors"
        }}
    }}"#
    );
    fs::write(tmp.path().join("model.safetensors.index.json"), index).unwrap();
    // A decoy unsharded file must be ignored when the index is present.
    fs::write(tmp.path().join("model.safetensors"), b"not a checkpoint").unwrap();

    let ckpt = Checkpoint::open(tmp.path()).unwrap();
    assert_eq!(ckpt.linears().len(), 2);
    assert_eq!(ckpt.linears()[0].prefix, "a.proj");
    assert_eq!(ckpt.linears()[1].codebook, Codebook::Mcg);
    assert!(ckpt.linears()[0]
        .trellis
        .shard
        .ends_with("model-00001-of-00002.safetensors"));
}

#[test]
fn rejects_bad_linears_and_headers() {
    assert_open_err(
        &[linear_parts("h.proj", 24, Some("mcg"), false, false, 0, 0)],
        "mul1",
    );
    assert_open_err(
        &[{
            let mut parts = linear_parts("h.proj", 64, Some("mul1"), false, false, 0, 0);
            parts.push({
                let mut mcg = marker("I32");
                mcg.name = "h.proj.mcg".into();
                mcg
            });
            parts
        }],
        "both mcg and mul1",
    );
    assert_open_err(
        &[linear_parts(
            "h.proj",
            144,
            Some("mul1"),
            false,
            false,
            0,
            0,
        )],
        "1..=8",
    );
    assert_open_err(
        &[linear_parts("h.proj", 72, Some("mul1"), false, false, 0, 0)],
        "4.5",
    );
    assert_open_err(
        &[linear_parts("h.proj", 20, Some("mul1"), false, false, 0, 0)],
        "16*K",
    );
    assert_open_err(
        &[linear_parts("h.proj", 64, Some("mul1"), false, false, 0, 0)
            .into_iter()
            .filter(|tensor| !tensor.name.ends_with(".suh") && !tensor.name.ends_with(".svh"))
            .collect::<Vec<_>>()],
        "suh",
    );

    let tmp = Tmp::new();
    write_config(tmp.path(), CONFIG);
    let mut bad = linear_parts("h.proj", 16, Some("mul1"), false, false, 0, 0);
    bad[0].shape = vec![16];
    bad[0].data = zeros(32);
    write_tensors(&tmp.path().join("model.safetensors"), &bad);
    let err = Checkpoint::open(tmp.path()).unwrap_err();
    assert!(err.to_string().contains("rank 3"), "{err}");
}

#[test]
fn rejects_a_mismatched_span_and_a_short_file() {
    let tmp = Tmp::new();
    write_config(tmp.path(), CONFIG);
    let header = r#"{"h.proj.trellis":{"dtype":"I16","shape":[1],"data_offsets":[0,4]}}"#;
    write_raw(&tmp.path().join("model.safetensors"), header, &[0, 0, 0, 0]);
    let err = Checkpoint::open(tmp.path()).unwrap_err();
    assert!(err.to_string().contains("h.proj.trellis"), "{err}");

    let tmp = Tmp::new();
    write_config(tmp.path(), CONFIG);
    let header = r#"{"h.proj.trellis":{"dtype":"I16","shape":[1],"data_offsets":[0,2]}}"#;
    write_raw(&tmp.path().join("model.safetensors"), header, &[]);
    let err = Checkpoint::open(tmp.path()).unwrap_err();
    assert!(err.to_string().contains("h.proj.trellis"), "{err}");
}

#[test]
fn rejects_a_bad_index() {
    let tmp = Tmp::new();
    write_config(tmp.path(), CONFIG);
    let parts = linear_parts("a.proj", 16, Some("mul1"), false, false, 0, 0);
    write_tensors(&tmp.path().join("model-00001-of-00002.safetensors"), &parts);
    fs::write(
        tmp.path().join("model.safetensors.index.json"),
        r#"{"metadata":{"total_size":1},"weight_map":{"a.proj.trellis":"model-00001-of-00002.safetensors","a.proj.suh":"model-00001-of-00002.safetensors","a.proj.svh":"model-00001-of-00002.safetensors","a.proj.mul1":"model-00001-of-00002.safetensors"}}"#,
    )
    .unwrap();
    let err = Checkpoint::open(tmp.path()).unwrap_err();
    assert!(err.to_string().contains("total_size"), "{err}");

    fs::write(
        tmp.path().join("model.safetensors.index.json"),
        r#"{"weight_map":{"missing.trellis":"model-00001-of-00002.safetensors"}}"#,
    )
    .unwrap();
    let err = Checkpoint::open(tmp.path()).unwrap_err();
    // Header keys are visited in name order, so the first tensor the index
    // does not list is `a.proj.mul1`, before the missing `missing.trellis`.
    assert!(
        err.to_string().contains("a.proj.mul1") || err.to_string().contains("missing.trellis"),
        "{err}"
    );

    fs::write(
        tmp.path().join("model.safetensors.index.json"),
        r#"{"weight_map":{"a.proj.trellis":"../outside.safetensors"}}"#,
    )
    .unwrap();
    let err = Checkpoint::open(tmp.path()).unwrap_err();
    assert!(err.to_string().contains("weight_map"), "{err}");
}

#[test]
fn missing_checkpoint_names_the_reason() {
    let tmp = Tmp::new();
    write_config(tmp.path(), CONFIG);
    let err = Checkpoint::open(tmp.path()).unwrap_err();
    assert!(err.to_string().contains("model.safetensors"), "{err}");

    let err = Checkpoint::open(tmp.path().join("nope")).unwrap_err();
    assert!(matches!(err, Error::BadField { .. }), "{err}");
}

fn assert_open_err(groups: &[Vec<OwnedTensor>], needle: &str) {
    let tmp = Tmp::new();
    write_config(tmp.path(), CONFIG);
    let tensors: Vec<OwnedTensor> = groups.iter().flatten().cloned().collect();
    write_tensors(&tmp.path().join("model.safetensors"), &tensors);
    let err = Checkpoint::open(tmp.path()).unwrap_err();
    assert!(err.to_string().contains(needle), "{err}");
}

fn write_raw(path: &Path, header: &str, data: &[u8]) {
    let mut file = File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(data).unwrap();
}

//! Block-0 prefill glue, without running the mixer or the MLP.
//!
//! GPU-gated. From the fork's `block_in` and `mixer` tensors, the
//! post-attention norm produces `mlp_in` and the updated residual, then the
//! reference MLP output is added. Block 1's input is not stored, so the sum
//! is compared with block 0's residual digest in `testdata/prefill_ref`.
//! Prompts 100 and 1500, chunk 0. Skips when the layer dump is missing.
//!
//! ```text
//! BERTH_GPU_TESTS=1 cargo test -p berth-engine --test prefill_block -- --nocapture
//! ```

mod common;

use std::path::{Path, PathBuf};

use berth_engine::model::{Model, TextNorm};
use berth_engine::Gpu;
use berth_exl3::Checkpoint;
use blake2::digest::consts::U32;
use blake2::digest::Digest;
use blake2::Blake2b;

/// `scripts/dump_27b_prefill_layers.py` writes here. Not in git.
const LAYERS: &str = "/home/sam/berth/target/prefill_layers";

#[test]
fn block0_glue_matches_the_fork() {
    let dir = match common::gpu_ready() {
        Ok(dir) => dir,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    if !Path::new(LAYERS).is_dir() {
        eprintln!("skipping: {LAYERS} is missing");
        return;
    }
    let _slot = match common::full_model_slot() {
        Ok(slot) => slot,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    let ckpt = Checkpoint::open(&dir).unwrap();
    let eps = rms_eps(&dir);
    let norm = TextNorm::qwen3_5(eps);
    let gpu = Gpu::new().unwrap();
    let weight = berth_engine::upload_tensor(
        &gpu,
        &ckpt,
        "model.language_model.layers.0.post_attention_layernorm.weight",
    )
    .unwrap();

    let meta = ref_meta();
    let hidden = meta["hidden"].as_u64().unwrap() as i32;
    assert_eq!(weight.num_bytes(), usize::try_from(hidden).unwrap() * 2);

    for (prompt, tag) in [("100", "100_c0_b00"), ("1500", "1500_c0_b00")] {
        let block_in = layer_file(&format!("{tag}_block_in.bin"));
        let mixer = layer_file(&format!("{tag}_mixer.bin"));
        let mlp_in_ref = layer_file(&format!("{tag}_mlp_in.bin"));
        let mlp = layer_file(&format!("{tag}_mlp.bin"));
        let row = usize::try_from(hidden).unwrap();
        assert_eq!(block_in.len() % (row * 4), 0, "{tag} block_in");
        let rows = u32::try_from(block_in.len() / (row * 4)).unwrap();
        assert_eq!(mixer.len(), block_in.len(), "{tag} mixer");
        assert_eq!(mlp.len(), block_in.len(), "{tag} mlp");
        assert_eq!(mlp_in_ref.len(), block_in.len() / 2, "{tag} mlp_in");
        let chunk = &meta["prompts"][prompt]["chunks"][0];
        assert_eq!(chunk["start"].as_u64().unwrap(), 0, "{prompt}");
        assert_eq!(chunk["end"].as_u64().unwrap(), u64::from(rows), "{prompt}");
        let want_digest = chunk["blocks"][0]["digest"].as_str().unwrap();
        assert_eq!(chunk["blocks"][0]["layer"].as_u64().unwrap(), 0);

        let mut residual = common::upload(&gpu, &block_in);
        let mixer_dev = common::upload(&gpu, &mixer);
        let mlp_dev = common::upload(&gpu, &mlp);
        let mut mlp_in = gpu.zeroed(mlp_in_ref.len()).unwrap();
        Model::norm_residual_rows(
            &gpu,
            &mixer_dev,
            &weight,
            &mut mlp_in,
            &mut residual,
            hidden,
            rows,
            norm,
        )
        .unwrap();
        let got_mlp_in = common::download(&gpu, &mlp_in);
        require_equal(&format!("{tag} mlp_in"), &got_mlp_in, &mlp_in_ref, 2);

        Model::add_residual_rows(&gpu, &mut residual, &mlp_dev, hidden, rows).unwrap();
        let got = common::download(&gpu, &residual);
        let digest = blake2b_256(&got);
        if !digest.eq_ignore_ascii_case(want_digest) {
            eprintln!(
                "first difference: {prompt} chunk 0 block 0 residual {digest} != {want_digest}"
            );
            panic!("{prompt} block 0 residual digest mismatch");
        }
        eprintln!("{prompt} chunk 0 block 0 residual digest matches");
    }
}

fn layer_file(name: &str) -> Vec<u8> {
    let path = Path::new(LAYERS).join(name);
    std::fs::read(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()))
}

fn ref_meta() -> serde_json::Value {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/prefill_ref/meta.json");
    let bytes = std::fs::read(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    serde_json::from_slice(&bytes).unwrap()
}

fn rms_eps(dir: &str) -> f32 {
    let path = Path::new(dir).join("config.json");
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    let eps = value["text_config"]["rms_norm_eps"]
        .as_f64()
        .unwrap_or_else(|| panic!("text_config.rms_norm_eps is missing"));
    let eps = eps as f32;
    assert_eq!(eps.to_bits(), 1e-6f32.to_bits(), "rms_norm_eps {eps}");
    eps
}

fn require_equal(label: &str, got: &[u8], want: &[u8], elem: usize) {
    assert_eq!(got.len(), want.len(), "{label}: byte length");
    let mut first = None;
    let mut diff = 0usize;
    for (index, (a, b)) in got.chunks(elem).zip(want.chunks(elem)).enumerate() {
        if a != b {
            diff += 1;
            if first.is_none() {
                first = Some(index);
            }
        }
    }
    if diff != 0 {
        let index = first.unwrap_or(0);
        eprintln!("first difference: {label} element {index} ({diff} differing)");
        panic!("{label}: {diff} differing, first at {index}");
    }
    eprintln!("{label}: 0 differing");
}

fn blake2b_256(bytes: &[u8]) -> String {
    let mut hasher = Blake2b::<U32>::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0xf)]));
    }
    out
}

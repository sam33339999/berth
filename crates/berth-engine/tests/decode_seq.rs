//! Many eager decode steps, GPU-gated.
//!
//! Teacher-forces `testdata/decode_seq/meta.json` when that directory is
//! present: `input_tokens[i]` at position `i`, then blake2b-256 of each
//! block residual, the final norm, and the logits. Prints the first
//! `(step, name)` whose digest differs.
//!
//! When the directory is missing (task SEQ has not landed), runs a sanity
//! check instead: 300 steps finish, every step's logits are finite, and
//! step 0 matches `testdata/decode_ref` bit for bit.
//!
//! ```text
//! BERTH_GPU_TESTS=1 cargo test -p berth-engine --test decode_seq -- --nocapture
//! ```

mod common;

use std::path::PathBuf;

use berth_engine::model::Model;
use berth_engine::Gpu;
use berth_exl3::Checkpoint;
use berth_plan::{build_plan, from_exl3::model_shape};
use blake2::digest::consts::U32;
use blake2::digest::Digest;
use blake2::Blake2b;

/// Same cache geometry as `testdata/decode_ref` (16 pages). `split_config`
/// keys off that width, so a longer table would not match step 0.
const MAX_TOKENS: u32 = 4096;
const SANITY_STEPS: u32 = 300;

#[test]
fn decode_sequence() {
    let dir = match common::gpu_ready() {
        Ok(dir) => dir,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    let _slot = match common::full_model_slot() {
        Ok(slot) => slot,
        Err(why) => return eprintln!("skipping: {why}"),
    };

    let ckpt = Checkpoint::open(&dir).unwrap();
    let shape = model_shape(&ckpt.config).unwrap();
    let plan = build_plan(&shape).unwrap();
    let gpu = Gpu::new().unwrap();
    let mut model = Model::load(&gpu, &ckpt, &plan, MAX_TOKENS).unwrap();

    let seq_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/decode_seq");
    let meta_path = seq_dir.join("meta.json");
    if !meta_path.is_file() {
        eprintln!(
            "skipping digest compare: testdata/decode_seq is missing (it arrives with task SEQ)"
        );
        sanity(&gpu, &mut model);
        return;
    }
    compare_digests(&gpu, &mut model, &meta_path);
}

fn sanity(gpu: &Gpu, model: &mut Model) {
    let token0 = common::reference_token();
    for step in 0..SANITY_STEPS {
        let token = if step == 0 { token0 } else { (step % 1024) + 1 };
        model.decode_step(gpu, token, step).unwrap();
        model.synchronize(gpu).unwrap();
        let logits = common::download(gpu, model.logits().unwrap());
        assert!(
            fp16_finite(&logits),
            "step {step}: logits have a non-finite value"
        );
        if step == 0 {
            assert_step0(gpu, model, &logits);
        }
        if step % 50 == 0 {
            eprintln!("sanity step {step} ok");
        }
    }
    eprintln!(
        "sanity: {SANITY_STEPS} steps complete, logits finite, step 0 matches testdata/decode_ref"
    );
}

fn assert_step0(gpu: &Gpu, model: &Model, logits: &[u8]) {
    let layers = model.num_layers();
    for index in 0..layers {
        let name = format!("block_{index:02}");
        let got = model.block_residual(u32::try_from(index).unwrap()).unwrap();
        let want = common::reference(&name);
        let diff = common::differing(got, &want, 4);
        let n = want.len() / 4;
        eprintln!("{name}: {diff} of {n} elements differ");
        assert_eq!(diff, 0, "first differing block: {name}");
    }
    // `synchronize` already finished the stream. `download` syncs again.
    let got_norm = common::download(gpu, model.final_norm().unwrap());
    let want_norm = common::reference("final_norm");
    let diff = common::differing(&got_norm, &want_norm, 2);
    eprintln!(
        "final_norm: {diff} of {} elements differ",
        want_norm.len() / 2
    );
    assert_eq!(diff, 0, "final_norm");

    let want_logits = common::reference("logits");
    let diff = common::differing(logits, &want_logits, 2);
    eprintln!(
        "logits: {diff} of {} elements differ",
        want_logits.len() / 2
    );
    assert_eq!(diff, 0, "logits");
}

fn compare_digests(gpu: &Gpu, model: &mut Model, meta_path: &std::path::Path) {
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(meta_path).unwrap())
        .unwrap_or_else(|err| panic!("{}: {err}", meta_path.display()));
    let tokens = meta["input_tokens"]
        .as_array()
        .unwrap_or_else(|| panic!("input_tokens is missing"))
        .iter()
        .map(|value| {
            value
                .as_u64()
                .unwrap_or_else(|| panic!("input token {value} is not an integer"))
                as u32
        })
        .collect::<Vec<_>>();
    let digests = meta["digests"]
        .as_array()
        .unwrap_or_else(|| panic!("digests is missing"));
    assert_eq!(
        tokens.len(),
        digests.len(),
        "input_tokens and digests have different lengths"
    );
    assert!(!tokens.is_empty(), "input_tokens is empty");
    assert!(
        tokens.len() <= usize::try_from(MAX_TOKENS).unwrap(),
        "sequence of {} tokens does not fit the {}-token cache",
        tokens.len(),
        MAX_TOKENS
    );
    // `digests[step]` is one hex digest per entry of `meta["names"]`, in that order.
    let order: Vec<String> = meta["names"]
        .as_array()
        .unwrap_or_else(|| panic!("names is missing"))
        .iter()
        .map(|name| name.as_str().unwrap().to_string())
        .collect();
    let names = digest_names();
    assert_eq!(
        names, order,
        "meta.json names are not the order this test hashes"
    );
    for (step, token) in tokens.iter().copied().enumerate() {
        let position = u32::try_from(step).unwrap();
        model.decode_step(gpu, token, position).unwrap();
        model.synchronize(gpu).unwrap();
        let row = &digests[step];
        for name in &names {
            let got = blake2b_256(&tensor_bytes(gpu, model, name));
            let column = order.iter().position(|n| n == name).unwrap();
            let want = row[column].as_str().unwrap_or_else(|| {
                panic!("step {step} {name}: digest missing");
            });
            if !hex_eq(&got, want) {
                eprintln!("first difference: step {step} {name}");
                eprintln!("  got  {got}");
                eprintln!("  want {want}");
                panic!("first difference: ({step}, {name})");
            }
        }
        if step % 50 == 0 {
            eprintln!("digest step {step} ok");
        }
    }
    eprintln!("decode_seq: {} steps, 0 differing", tokens.len());
}

fn digest_names() -> Vec<String> {
    let mut names = Vec::with_capacity(66);
    for index in 0..64 {
        names.push(format!("block_{index:02}"));
    }
    names.push("final_norm".to_string());
    names.push("logits".to_string());
    names
}

fn tensor_bytes(gpu: &Gpu, model: &Model, name: &str) -> Vec<u8> {
    if name == "logits" {
        return common::download(gpu, model.logits().unwrap());
    }
    if name == "final_norm" {
        return common::download(gpu, model.final_norm().unwrap());
    }
    let index: u32 = name
        .strip_prefix("block_")
        .unwrap_or_else(|| panic!("unknown tensor {name}"))
        .parse()
        .unwrap_or_else(|_| panic!("unknown tensor {name}"));
    model.block_residual(index).unwrap().to_vec()
}

fn blake2b_256(bytes: &[u8]) -> String {
    let mut hasher = Blake2b::<U32>::new();
    hasher.update(bytes);
    hex_encode(hasher.finalize().as_slice())
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0xf)]));
    }
    out
}

fn hex_eq(got: &str, want: &str) -> bool {
    got.eq_ignore_ascii_case(want.trim())
}

fn fp16_finite(bytes: &[u8]) -> bool {
    bytes.chunks(2).all(|chunk| {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        bits & 0x7c00 != 0x7c00
    })
}

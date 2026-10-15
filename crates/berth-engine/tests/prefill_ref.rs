//! Prefill reference against `testdata/prefill_ref/`.
//!
//! `chunk_bounds` is checked on the CPU against the fork's recorded chunks.
//! The GPU test loads the 27B and calls `Model::prefill_prompt`. Until the
//! kernels land, that returns `prefill not implemented` and the test passes.
//! A second GPU test runs the 100-token prompt through `decode_step` and
//! prints the max abs logit difference from the fork's prefill logits.
//!
//! ```text
//! BERTH_GPU_TESTS=1 cargo test -p berth-engine --test prefill_ref -- --nocapture
//! ```

mod common;

use std::path::PathBuf;
use std::sync::Mutex;

use berth_engine::model::Model;
use berth_engine::prefill::{self, AttnCacheDigest, GdnStateDigest, PrefillOutput, TensorDigest};
use berth_engine::Gpu;
use berth_exl3::Checkpoint;
use berth_plan::{build_plan, from_exl3::model_shape};

/// One full-model load at a time inside this binary. `flock` covers other
/// processes; it does not cover two tests of this process.
static GPU_TESTS: Mutex<()> = Mutex::new(());

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/prefill_ref")
}

fn meta() -> serde_json::Value {
    let path = ref_dir().join("meta.json");
    let bytes = std::fs::read(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|err| panic!("{}: {err}", path.display()))
}

fn prompt<'a>(meta: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    meta["prompts"]
        .get(name)
        .unwrap_or_else(|| panic!("prompts.{name} is missing"))
}

fn token_ids(prompt: &serde_json::Value) -> Vec<u32> {
    prompt["token_ids"]
        .as_array()
        .unwrap_or_else(|| panic!("token_ids is missing"))
        .iter()
        .map(|value| value.as_u64().unwrap_or_else(|| panic!("token id {value}")) as u32)
        .collect()
}

fn cache_tokens(meta: &serde_json::Value) -> u32 {
    let n = meta["cache_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("cache_tokens")) as u32;
    assert!(n.is_multiple_of(256) && n > 0, "cache_tokens {n}");
    n
}

fn recorded_chunks(prompt: &serde_json::Value) -> Vec<(u32, u32)> {
    prompt["chunks"]
        .as_array()
        .unwrap_or_else(|| panic!("chunks is missing"))
        .iter()
        .map(|chunk| {
            let start = chunk["start"].as_u64().unwrap() as u32;
            let end = chunk["end"].as_u64().unwrap() as u32;
            (start, end)
        })
        .collect()
}

#[test]
fn chunk_bounds_match_the_fork() {
    assert!(prefill::chunk_bounds(0).is_empty());
    assert!(prefill::chunk_bounds(1).is_empty());
    assert_eq!(prefill::chunk_bounds(100), vec![(0, 99)]);
    assert_eq!(prefill::chunk_bounds(1500), vec![(0, 1280), (1280, 1499)]);
    assert_eq!(prefill::chunk_bounds(257), vec![(0, 256)]);
    assert_eq!(prefill::chunk_bounds(258), vec![(0, 256), (256, 257)]);
    assert_eq!(prefill::chunk_bounds(2049), vec![(0, 2048)]);
    assert_eq!(prefill::chunk_bounds(2050), vec![(0, 2048), (2048, 2049)]);

    let meta = meta();
    assert_eq!(
        meta["page_size"].as_u64(),
        Some(u64::from(prefill::PAGE_SIZE))
    );
    assert_eq!(
        meta["max_chunk_size"].as_u64(),
        Some(u64::from(prefill::MAX_CHUNK_SIZE))
    );
    let short = token_ids(prompt(&meta, "100"));
    let long = token_ids(prompt(&meta, "1500"));
    assert_eq!(short.len(), 100);
    assert_eq!(long.len(), 1500);
    assert_eq!(&long[..short.len()], short.as_slice());
    for name in ["100", "1500"] {
        let prompt = prompt(&meta, name);
        let ids = token_ids(prompt);
        let got = prefill::chunk_bounds(u32::try_from(ids.len()).unwrap());
        let want = recorded_chunks(prompt);
        assert_eq!(got, want, "{name}: chunk bounds");
        // Layer 63 returns the residual layer 62 produced.
        for chunk in prompt["chunks"].as_array().unwrap() {
            let blocks = chunk["blocks"].as_array().unwrap();
            assert_eq!(blocks.len(), 64, "{name}");
            assert_eq!(
                blocks[62]["digest"], blocks[63]["digest"],
                "{name}: early return"
            );
            assert!(
                blocks[63]["early_return"].as_bool().unwrap(),
                "{name}: layer 63 early_return"
            );
            assert_eq!(blocks[0]["dtype"], "float32");
        }
    }

    assert_eq!(fp16_to_f32(0x3c00), 1.0);
    assert_eq!(fp16_to_f32(0x4000), 2.0);
    assert_eq!(fp16_to_f32(0xbc00), -1.0);
    assert_eq!(fp16_to_f32(0x0000), 0.0);
    assert_eq!(fp16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
}

#[test]
fn prefill_prompt_matches_fork_or_skips() {
    let _guard = GPU_TESTS.lock().unwrap_or_else(|err| err.into_inner());
    let dir = match common::gpu_ready() {
        Ok(dir) => dir,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    let meta = meta();
    let max_tokens = cache_tokens(&meta);
    let _slot = match common::full_model_slot() {
        Ok(slot) => slot,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    let ckpt = Checkpoint::open(&dir).unwrap();
    let shape = model_shape(&ckpt.config).unwrap();
    let plan = build_plan(&shape).unwrap();
    let gpu = Gpu::new().unwrap();
    let mut model = Model::load(&gpu, &ckpt, &plan, max_tokens).unwrap();

    for name in ["100", "1500"] {
        let prompt = prompt(&meta, name);
        let ids = token_ids(prompt);
        match model.prefill_prompt(&gpu, &ids) {
            Err(err) if err.to_string().contains("prefill not implemented") => {
                eprintln!("prefill not implemented");
                return;
            }
            Ok(output) => compare_prefill(name, prompt, &output),
            Err(err) => panic!("{name}: {err}"),
        }
        model.reset_sequence(&gpu).unwrap();
    }
}

#[test]
fn decode_step_logits_versus_fork_prefill() {
    let _guard = GPU_TESTS.lock().unwrap_or_else(|err| err.into_inner());
    let dir = match common::gpu_ready() {
        Ok(dir) => dir,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    let meta = meta();
    let max_tokens = cache_tokens(&meta);
    let prompt = prompt(&meta, "100");
    let ids = token_ids(prompt);
    assert!(
        ids.len() <= usize::try_from(max_tokens).unwrap(),
        "100-token prompt does not fit the cache"
    );
    let want = std::fs::read(ref_dir().join(prompt["logits"]["file"].as_str().unwrap()))
        .unwrap_or_else(|err| panic!("logits_100: {err}"));
    let fork_argmax = prompt["logits"]["argmax"].as_u64().unwrap() as usize;

    let _slot = match common::full_model_slot() {
        Ok(slot) => slot,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    let ckpt = Checkpoint::open(&dir).unwrap();
    let shape = model_shape(&ckpt.config).unwrap();
    let plan = build_plan(&shape).unwrap();
    let gpu = Gpu::new().unwrap();
    let mut model = Model::load(&gpu, &ckpt, &plan, max_tokens).unwrap();

    // 100 steps fit in the 128-deep staging buffer, so one sync at the end
    // is enough. The logits are the forward of the last prompt token.
    for (position, &token) in ids.iter().enumerate() {
        let position = u32::try_from(position).unwrap();
        model.decode_step(&gpu, token, position).unwrap();
    }
    model.synchronize(&gpu).unwrap();
    let got = common::download(&gpu, model.logits().unwrap());
    assert!(
        fp16_finite(&got),
        "decode_step logits have a non-finite value"
    );
    let (max_abs, got_argmax) = max_abs_fp16(&got, &want);
    eprintln!("max abs logit difference {max_abs}");
    eprintln!("decode_step argmax {got_argmax}");
    eprintln!("fork prefill argmax {fork_argmax}");
}

fn compare_prefill(name: &str, prompt: &serde_json::Value, output: &PrefillOutput) {
    let want_chunks = prompt["chunks"].as_array().unwrap();
    assert_eq!(
        output.chunks.len(),
        want_chunks.len(),
        "{name}: chunk count"
    );
    for (got, want) in output.chunks.iter().zip(want_chunks) {
        assert_eq!(
            u64::from(got.start),
            want["start"].as_u64().unwrap(),
            "{name}"
        );
        assert_eq!(u64::from(got.end), want["end"].as_u64().unwrap(), "{name}");
        let want_blocks = want["blocks"].as_array().unwrap();
        assert_eq!(got.blocks.len(), want_blocks.len(), "{name}");
        for (block, want_block) in got.blocks.iter().zip(want_blocks) {
            let layer = want_block["layer"].as_u64().unwrap();
            assert_eq!(u64::from(block.layer), layer, "{name}");
            assert!(
                block.dtype.contains("float32"),
                "{name} layer {layer} dtype"
            );
            assert_eq!(
                numel(&block.shape),
                numel_json(&want_block["shape"]),
                "{name}"
            );
            let want_digest = want_block["digest"].as_str().unwrap();
            assert!(
                hex_eq(&block.digest, want_digest),
                "{name} layer {layer}: {} != {want_digest}",
                block.digest
            );
        }
    }
    compare_tensors(
        name,
        "attn",
        &output.attn,
        prompt["after_prefill"]["attn"].as_array().unwrap(),
    );
    compare_gdn(
        name,
        &output.gdn,
        prompt["after_prefill"]["gdn"].as_array().unwrap(),
    );
}

fn compare_tensors(name: &str, kind: &str, got: &[AttnCacheDigest], want: &[serde_json::Value]) {
    assert_eq!(got.len(), want.len(), "{name}: {kind} layers");
    for (layer, want_layer) in got.iter().zip(want) {
        assert_eq!(
            u64::from(layer.layer),
            want_layer["layer"].as_u64().unwrap()
        );
        for (which, tensor) in [
            ("qk", &layer.qk),
            ("qv", &layer.qv),
            ("sk", &layer.sk),
            ("sv", &layer.sv),
        ] {
            expect_tensor(name, layer.layer, which, tensor, &want_layer[which]);
        }
    }
}

fn compare_gdn(name: &str, got: &[GdnStateDigest], want: &[serde_json::Value]) {
    assert_eq!(got.len(), want.len(), "{name}: gdn layers");
    for (layer, want_layer) in got.iter().zip(want) {
        assert_eq!(
            u64::from(layer.layer),
            want_layer["layer"].as_u64().unwrap()
        );
        expect_tensor(name, layer.layer, "conv", &layer.conv, &want_layer["conv"]);
        expect_tensor(
            name,
            layer.layer,
            "recurrent",
            &layer.recurrent,
            &want_layer["recurrent"],
        );
    }
}

fn expect_tensor(
    name: &str,
    layer: u32,
    which: &str,
    got: &TensorDigest,
    want: &serde_json::Value,
) {
    let want_dtype = want["dtype"].as_str().unwrap();
    assert!(
        got.dtype.contains(want_dtype),
        "{name} layer {layer} {which}: {} does not contain {want_dtype}",
        got.dtype
    );
    assert_eq!(
        numel(&got.shape),
        numel_json(&want["shape"]),
        "{name} layer {layer} {which}"
    );
    let want_digest = want["digest"].as_str().unwrap();
    assert!(
        hex_eq(&got.digest, want_digest),
        "{name} layer {layer} {which}: {} != {want_digest}",
        got.digest
    );
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn numel_json(shape: &serde_json::Value) -> usize {
    shape
        .as_array()
        .unwrap()
        .iter()
        .map(|dim| usize::try_from(dim.as_u64().unwrap()).unwrap())
        .product()
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

/// IEEE fp16 to f32. Subnormals included. No `half` crate in the workspace.
fn fp16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = (bits >> 10) & 0x1f;
    let frac = u32::from(bits & 0x03ff);
    let bits32 = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mag = (frac as f32) * (-14f32).exp2() / 1024.0;
            return if sign == 0 { mag } else { -mag };
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        let exp32 = u32::from(exp) + (127 - 15);
        sign | (exp32 << 23) | (frac << 13)
    };
    f32::from_bits(bits32)
}

fn max_abs_fp16(got: &[u8], want: &[u8]) -> (f32, usize) {
    assert_eq!(got.len(), want.len(), "logit byte length");
    assert_eq!(got.len() % 2, 0, "fp16 byte length");
    let mut max_abs = 0f32;
    let mut best = f32::NEG_INFINITY;
    let mut argmax = 0usize;
    for (index, (g, w)) in got.chunks(2).zip(want.chunks(2)).enumerate() {
        let gv = fp16_to_f32(u16::from_le_bytes([g[0], g[1]]));
        let wv = fp16_to_f32(u16::from_le_bytes([w[0], w[1]]));
        let diff = (gv - wv).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        if gv > best {
            best = gv;
            argmax = index;
        }
    }
    (max_abs, argmax)
}

//! Twenty-step sampler parity with the fork's `CustomSampler`.
//!
//! `crates/berth-kernels/tests/sample_ref.py` writes
//! `testdata/kernels/sample/meta.json`. Logit rows and the prompt are
//! regenerated from the LCG documented there. Each case must emit the same
//! 20 token ids and the same post-penalty logit bytes:
//!
//! * greedy: argmax, penalties off, post bytes are the masked fp16 row
//! * serve defaults at temperature 0.7: DRY only, post bytes are fp32
//! * rep 1.1, presence 0.2, frequency 0.2, DRY on: fp32 after all three
//!
//! GPU only, `BERTH_GPU_TESTS=1`, and skipped when port 8889 is listening.
//! This does not load the 27B.

use std::path::{Path, PathBuf};
use std::process::Command;

use berth_cuda::ByteBuf;
use berth_engine::{Gpu, Sampler, SamplerConfig};
use blake2::digest::consts::U32;
use blake2::digest::Digest;
use blake2::Blake2b;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/kernels/sample")
}

fn load_meta() -> serde_json::Value {
    let path = data_dir().join("meta.json");
    let bytes = std::fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    serde_json::from_slice(&bytes).unwrap()
}

fn serve_listening() -> bool {
    let output = match Command::new("ss").args(["-ltn"]).output() {
        Ok(output) => output,
        Err(_) => return true,
    };
    if !output.status.success() {
        return true;
    }
    String::from_utf8_lossy(&output.stdout).contains(":8889 ")
}

fn lcg_next(x: u32) -> u32 {
    x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

fn lcg_f16(seed: u32, n: usize) -> Vec<u8> {
    let mut x = seed;
    let mut out = vec![0u8; n * 2];
    for i in 0..n {
        x = lcg_next(x);
        let mant = (x >> 8) & 0x3ff;
        let exp = 14 + ((x >> 20) & 3);
        let sign = x >> 31;
        let bits = (sign << 15) | (exp << 10) | mant;
        out[i * 2..i * 2 + 2].copy_from_slice(&(bits as u16).to_le_bytes());
    }
    out
}

fn lcg_ids(seed: u32, n: usize, modulo: u32, repeat_tail: usize) -> Vec<u32> {
    let mut x = seed;
    let mut ids = Vec::with_capacity(n);
    for _ in 0..n {
        x = lcg_next(x);
        ids.push(x % modulo);
    }
    if repeat_tail > 0 && n >= 2 * repeat_tail {
        let start = n - repeat_tail;
        let pattern_at = n - 2 * repeat_tail;
        for i in 0..repeat_tail {
            ids[start + i] = ids[pattern_at + i];
        }
    }
    ids
}

fn history_bytes(ids: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ids.len() * 4);
    for id in ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out
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

fn u32_at(value: &serde_json::Value, key: &str) -> u32 {
    value[key]
        .as_u64()
        .unwrap_or_else(|| panic!("missing {key}")) as u32
}

fn f32_bits(value: &serde_json::Value, key: &str) -> f32 {
    f32::from_bits(u32_at(value, key))
}

fn read_breakers(name: &str) -> Vec<u32> {
    let bytes = std::fs::read(data_dir().join(name)).unwrap();
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn download(gpu: &Gpu, buf: &ByteBuf) -> Vec<u8> {
    let mut out = vec![0u8; buf.num_bytes()];
    buf.copy_to_host(&gpu.stream, &mut out).unwrap();
    out
}

fn config_from(spec: &serde_json::Value) -> SamplerConfig {
    SamplerConfig {
        temperature: f32_bits(spec, "temperature_bits"),
        rep_penalty: f32_bits(spec, "rep_penalty_bits"),
        presence_penalty: f32_bits(spec, "presence_penalty_bits"),
        frequency_penalty: f32_bits(spec, "frequency_penalty_bits"),
        penalty_range: u32_at(spec, "penalty_range"),
        dry_multiplier: f32_bits(spec, "dry_multiplier_bits"),
        dry_base: f32_bits(spec, "dry_base_bits"),
        dry_allowed_length: u32_at(spec, "dry_allowed_length"),
        dry_range: u32_at(spec, "dry_range"),
        seed: spec["seed"].as_u64().unwrap(),
    }
}

#[test]
fn twenty_steps_match_the_fork() {
    if std::env::var("BERTH_GPU_TESTS").ok().as_deref() != Some("1") {
        eprintln!("skipping sampler: set BERTH_GPU_TESTS=1");
        return;
    }
    if serve_listening() {
        eprintln!("skipping sampler: port 8889 is in use or ss failed");
        return;
    }
    let meta = load_meta();
    let vocab = meta["vocab"].as_u64().unwrap() as u32;
    let sampler_meta = &meta["sampler"];
    let limit = u32_at(sampler_meta, "prompt_mod");
    assert_eq!(limit, meta["logit_limit"].as_u64().unwrap() as u32);
    let prompt = lcg_ids(
        u32_at(sampler_meta, "prompt_seed"),
        u32_at(sampler_meta, "prompt_len") as usize,
        limit,
        u32_at(sampler_meta, "repeat_tail") as usize,
    );
    assert_eq!(
        blake2b_256(&history_bytes(&prompt)),
        sampler_meta["prompt_blake2b256"].as_str().unwrap(),
        "prompt LCG drifted"
    );
    let steps = u32_at(sampler_meta, "steps") as usize;
    let seed0 = u32_at(sampler_meta, "logits_seed0");
    let rows: Vec<Vec<u8>> = (0..steps)
        .map(|i| lcg_f16(seed0.wrapping_add(i as u32), vocab as usize))
        .collect();
    let hashes = sampler_meta["input_blake2b256"].as_array().unwrap();
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(
            blake2b_256(row),
            hashes[i].as_str().unwrap(),
            "logit row {i} drifted"
        );
    }
    let breakers = read_breakers(sampler_meta["breakers_file"].as_str().unwrap());
    assert_eq!(
        breakers.len(),
        sampler_meta["breaker_count"].as_u64().unwrap() as usize
    );

    let gpu = Gpu::new().unwrap();
    let mut sampler = Sampler::new(&gpu, vocab, 4096).unwrap();
    sampler.set_logit_limit(limit).unwrap();
    sampler.set_sequence_breakers(&gpu, &breakers).unwrap();

    for name in ["greedy", "defaults", "penalties"] {
        let spec = &sampler_meta["cases"][name];
        sampler.reset(&gpu, &config_from(spec), &prompt).unwrap();
        let expect_tokens = spec["tokens"].as_array().unwrap();
        let expect_posts = spec["post_blake2b256"].as_array().unwrap();
        let post_dtype = spec["post_dtype"].as_str().unwrap();
        let mut got_tokens = Vec::with_capacity(steps);
        for step in 0..steps {
            let mut logits = ByteBuf::from_host(&gpu.stream, &rows[step]).unwrap();
            let token = sampler.sample(&gpu, &mut logits).unwrap();
            got_tokens.push(token);
            let expect = expect_tokens[step].as_u64().unwrap() as u32;
            assert_eq!(
                token, expect,
                "{name} step {step}: token {token} != {expect}"
            );
            let post = match sampler.penalized_logits() {
                Some(buf) => {
                    assert_eq!(post_dtype, "f32", "{name}");
                    download(&gpu, buf)
                }
                None => {
                    assert_eq!(post_dtype, "f16", "{name}");
                    download(&gpu, &logits)
                }
            };
            let hash = blake2b_256(&post);
            let expect_hash = expect_posts[step].as_str().unwrap();
            assert_eq!(
                hash, expect_hash,
                "{name} step {step}: post-penalty logits differ"
            );
        }
        assert_eq!(got_tokens.len(), expect_tokens.len(), "{name}");
        eprintln!("{name}: {got_tokens:?}");
    }
}

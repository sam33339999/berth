//! Non-MTP generate against `testdata/generate_ref`. GPU-gated like `generate`.
//!
//! One load, `mtp_steps` 0, cache length from the meta (12288). Six cases:
//! prompts 4090 and 100, samplers `greedy`, `sample`, and `greedy_dry`.
//! Each step compares the fp16 logit digest, then the token. A case stops
//! at its first mismatch. The test then reports every case, so a block-table
//! miss does not hide a DRY-breaker miss. `BERTH_GENERATE_REF_CASE` is
//! `<prompt>_<sampler>` and keeps one case.
//!
//! ```text
//! BERTH_GPU_TESTS=1 cargo test --release -p berth-engine --test generate_ref -- --nocapture --test-threads=1
//! ```

mod common;

use std::path::{Path, PathBuf};

use berth_engine::{Engine, Finish, Flow, GenerateRequest, SamplerConfig};
use blake2::digest::consts::U32;
use blake2::digest::Digest;
use blake2::Blake2b;
use serde_json::Value;

#[test]
fn generate_ref_matches_fork() {
    let dir = match common::gpu_ready() {
        Ok(dir) => dir,
        Err(why) => {
            eprintln!("skipping generate_ref: {why}");
            return;
        }
    };
    let meta = load_meta();
    check_sampler_bits(&meta);
    let only = env_case();
    let cases = meta["cases"]
        .as_array()
        .unwrap_or_else(|| panic!("cases is not an array"));
    let selected: Vec<&Value> = cases
        .iter()
        .filter(|case| {
            let label = case_label(case);
            only.as_ref().is_none_or(|want| want == &label)
        })
        .collect();
    if selected.is_empty() {
        panic!(
            "BERTH_GENERATE_REF_CASE={} matched no case",
            only.unwrap_or_default()
        );
    }

    let _slot = common::full_model_slot().unwrap_or_else(|why| panic!("generate_ref: {why}"));
    let cubin = cubin_dir();
    if !cubin.is_dir() {
        panic!("cubin dir {} is missing", cubin.display());
    }
    let cache_tokens = u32_of(&meta["cache_tokens"], "cache_tokens");
    eprintln!("generate_ref loading cache_tokens={cache_tokens} mtp_steps=0");
    let mut engine = Engine::load(Path::new(&dir), &cubin, cache_tokens, 0)
        .unwrap_or_else(|err| panic!("Engine::load: {err}"));
    if engine.vocab_size() != u32_of(&meta["vocab"], "vocab") {
        panic!(
            "vocab {} != meta {}",
            engine.vocab_size(),
            u32_of(&meta["vocab"], "vocab")
        );
    }

    let mut failures = Vec::new();
    for case in selected {
        let label = case_label(case);
        eprintln!("generate_ref case={label}");
        if let Some(failure) = run_case(&mut engine, &meta, case, &label) {
            eprintln!("generate_ref {failure}");
            failures.push(failure);
        } else {
            eprintln!("generate_ref case={label} ok");
        }
    }
    if !failures.is_empty() {
        panic!(
            "generate_ref {} case(s) mismatched:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }
}

/// `None` when the case matches. Otherwise the first mismatch, naming the
/// case, the step, and the field.
fn run_case(engine: &mut Engine, meta: &Value, case: &Value, label: &str) -> Option<String> {
    let prompt_name = str_of(&case["prompt"], "prompt");
    let sampler_name = str_of(&case["sampler"], "sampler");
    let ids = prompt_ids(meta, prompt_name);
    let prompt_len = u32_of(&case["prompt_len"], "prompt_len");
    if ids.len() != usize::try_from(prompt_len).unwrap_or(usize::MAX) {
        return Some(format!(
            "case={label} step=- field=prompt_len: meta {prompt_len} but prompts.{prompt_name} has {} ids",
            ids.len()
        ));
    }
    let steps = case["steps"]
        .as_array()
        .unwrap_or_else(|| panic!("case={label} steps is not an array"));
    if steps.len() != 32 {
        return Some(format!(
            "case={label} step=- field=steps: meta has {} steps, expected 32",
            steps.len()
        ));
    }
    let bin_name = str_of(&case["logits_bin"], "logits_bin");
    let bin = std::fs::read(ref_dir().join(bin_name))
        .unwrap_or_else(|err| panic!("case={label} {bin_name}: {err}"));
    let step0 = str_of(&steps[0]["logits_blake2b"], "step0 logits");
    if blake2b_256(&bin) != step0 {
        return Some(format!(
            "case={label} step=0 field=logits_bin: file {bin_name} digest {} != meta {step0}",
            blake2b_256(&bin)
        ));
    }

    let max_new = u32_of(&meta["max_new_tokens"], "max_new_tokens");
    let request = GenerateRequest {
        prompt: ids,
        max_new_tokens: max_new,
        stop_tokens: Vec::new(),
        sampler: sampler_config(sampler_name),
    };
    let mut tokens = Vec::new();
    let mut logits = Vec::new();
    let generated = engine.generate_observed(
        &request,
        &mut |token| {
            tokens.push(token);
            Flow::Continue
        },
        &mut |row| logits.push(row.to_vec()),
    );
    let (finish, usage) = match generated {
        Ok(pair) => pair,
        Err(err) => {
            return Some(format!("case={label} step=- field=error: {err}"));
        }
    };

    for (index, step) in steps.iter().enumerate() {
        let pages = u32_of(&step["block_pages"], "block_pages");
        let want_logits = str_of(&step["logits_blake2b"], "logits_blake2b");
        let Some(row) = logits.get(index) else {
            return Some(format!(
                "case={label} step={index} field=logits: missing (run produced {} rows, finish={finish:?}, fork block_pages={pages})",
                logits.len()
            ));
        };
        let got_logits = blake2b_256(row);
        if got_logits != want_logits {
            return Some(format!(
                "case={label} step={index} field=logits: got {got_logits} expect {want_logits} (fork block_pages={pages})"
            ));
        }
        if index == 0 && row.as_slice() != bin.as_slice() {
            return Some(format!(
                "case={label} step=0 field=logits_bin: digest matched meta but bytes differ from {bin_name}"
            ));
        }
        let Some(&token) = tokens.get(index) else {
            return Some(format!(
                "case={label} step={index} field=token: missing (run produced {} tokens, finish={finish:?})",
                tokens.len()
            ));
        };
        let want_token = u32_of(&step["token"], "token");
        if token != want_token {
            return Some(format!(
                "case={label} step={index} field=token: got {token} expect {want_token} (fork block_pages={pages})"
            ));
        }
    }
    if logits.len() != steps.len() {
        return Some(format!(
            "case={label} step={} field=logits: extra rows {} (meta {})",
            steps.len(),
            logits.len(),
            steps.len()
        ));
    }
    if tokens.len() != steps.len() {
        return Some(format!(
            "case={label} step={} field=token: extra tokens {} (meta {})",
            steps.len(),
            tokens.len(),
            steps.len()
        ));
    }
    if finish != Finish::Length {
        return Some(format!(
            "case={label} step=- field=finish: got {finish:?} expect Length"
        ));
    }
    if usage.prompt_tokens != prompt_len || usage.completion_tokens != max_new {
        return Some(format!(
            "case={label} step=- field=usage: got prompt {} completion {} expect {prompt_len} {max_new}",
            usage.prompt_tokens, usage.completion_tokens
        ));
    }
    None
}

fn sampler_config(name: &str) -> SamplerConfig {
    match name {
        "greedy" => SamplerConfig::greedy(),
        "sample" => SamplerConfig::serve_defaults(0.7, 42),
        "greedy_dry" => SamplerConfig::serve_defaults(0.0, 0),
        other => panic!("sampler {other} is not greedy, sample, or greedy_dry"),
    }
}

/// The live sampler bits the dump recorded. A mismatch here is the test
/// setup, so it fails before a generate.
fn check_sampler_bits(meta: &Value) {
    let greedy = &meta["samplers"]["greedy"];
    let greedy_fused = &greedy["collapsed_steps"][0];
    same_hex(
        "greedy inv_temp",
        "3f800000",
        str_of(&greedy_fused["inv_temp_f32_bits"], "greedy inv_temp"),
    );
    let greedy_cfg = SamplerConfig::greedy();
    assert_eq!(greedy_cfg.temperature.to_bits(), 0, "greedy temperature");
    assert_eq!(
        greedy_cfg.dry_multiplier.to_bits(),
        0,
        "greedy dry_multiplier"
    );
    assert_eq!(greedy_cfg.seed, 0, "greedy seed");
    assert_eq!(u32_of(&greedy["seed"], "greedy seed"), 0);
    assert_eq!(
        u32_of(&greedy_fused["mode"], "greedy mode"),
        0,
        "greedy fused mode"
    );

    check_dry_stack("sample", &meta["samplers"]["sample"], 0.7, 42, 1);
    check_dry_stack("greedy_dry", &meta["samplers"]["greedy_dry"], 0.0, 0, 0);
}

fn check_dry_stack(name: &str, entry: &Value, temperature: f32, seed: u64, fused_mode: u64) {
    let cfg = if temperature == 0.0 {
        SamplerConfig::serve_defaults(0.0, seed)
    } else {
        SamplerConfig::serve_defaults(temperature, seed)
    };
    assert_eq!(
        cfg.temperature.to_bits(),
        temperature.to_bits(),
        "{name} temperature"
    );
    assert_eq!(cfg.seed, seed, "{name} seed");
    assert_eq!(u32_of(&entry["seed"], "seed"), u32_try(seed, "seed"));
    assert_eq!(cfg.rep_penalty.to_bits(), 1.0f32.to_bits(), "{name} rep");
    assert_eq!(cfg.presence_penalty.to_bits(), 0, "{name} presence");
    assert_eq!(cfg.frequency_penalty.to_bits(), 0, "{name} frequency");
    assert_eq!(
        cfg.penalty_range,
        u32_of(&entry["penalty_range"], "penalty_range"),
        "{name} penalty_range"
    );
    let steps = entry["collapsed_steps"]
        .as_array()
        .unwrap_or_else(|| panic!("{name} collapsed_steps is not an array"));
    let dry = steps
        .iter()
        .find(|step| step["type"] == "SS_DRY")
        .unwrap_or_else(|| panic!("{name} stack has no SS_DRY"));
    let fused = steps
        .iter()
        .find(|step| step["type"] == "SS_Fused")
        .unwrap_or_else(|| panic!("{name} stack has no SS_Fused"));
    same_hex(
        &format!("{name} dry_multiplier"),
        str_of(&dry["dry_multiplier_f32_bits"], "dry_multiplier"),
        &hex_f32(cfg.dry_multiplier),
    );
    same_hex(
        &format!("{name} dry_base"),
        str_of(&dry["dry_base_f32_bits"], "dry_base"),
        &hex_f32(cfg.dry_base),
    );
    assert_eq!(
        u32_of(&dry["dry_allowed_length"], "dry_allowed_length"),
        cfg.dry_allowed_length,
        "{name} dry_allowed_length"
    );
    assert_eq!(
        u32_of(&dry["dry_range"], "dry_range"),
        cfg.dry_range,
        "{name} dry_range"
    );
    assert_eq!(
        u32_of(&fused["mode"], "fused mode"),
        u32_try(fused_mode, "fused mode"),
        "{name} fused mode"
    );
    if fused_mode == 1 {
        let inv = (1.0f64 / f64::from(temperature)) as f32;
        same_hex(
            &format!("{name} inv_temp"),
            str_of(&fused["inv_temp_f32_bits"], "inv_temp"),
            &hex_f32(inv),
        );
    }
}

fn case_label(case: &Value) -> String {
    format!(
        "{}_{}",
        str_of(&case["prompt"], "prompt"),
        str_of(&case["sampler"], "sampler")
    )
}

fn env_case() -> Option<String> {
    let raw = std::env::var("BERTH_GENERATE_REF_CASE").ok()?;
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let (prompt, sampler) = raw
        .rsplit_once('_')
        .unwrap_or_else(|| panic!("BERTH_GENERATE_REF_CASE must be <prompt>_<sampler>, got {raw}"));
    if prompt != "4090" && prompt != "100" {
        panic!("BERTH_GENERATE_REF_CASE prompt must be 4090 or 100, got {raw}");
    }
    if sampler != "greedy" && sampler != "sample" && sampler != "greedy_dry" {
        panic!("BERTH_GENERATE_REF_CASE sampler must be greedy, sample, or greedy_dry, got {raw}");
    }
    Some(format!("{prompt}_{sampler}"))
}

fn prompt_ids(meta: &Value, name: &str) -> Vec<u32> {
    meta["prompts"][name]
        .as_array()
        .unwrap_or_else(|| panic!("prompts.{name} is missing"))
        .iter()
        .map(|value| u32_of(value, "prompt id"))
        .collect()
}

fn load_meta() -> Value {
    let path = ref_dir().join("meta.json");
    let bytes = std::fs::read(&path).unwrap_or_else(|_| {
        panic!(
            "{} 不在。先跑 `scripts/dump_27b_generate_ref.py`",
            path.display()
        );
    });
    let meta: Value =
        serde_json::from_slice(&bytes).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    if meta["schema_version"] != 1 {
        panic!("{} schema_version is not 1", path.display());
    }
    if meta["num_draft_tokens"] != 0 {
        panic!("{} num_draft_tokens is not 0", path.display());
    }
    if meta["hash"] != "blake2b-256" {
        panic!("{} hash is not blake2b-256", path.display());
    }
    meta
}

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/generate_ref")
}

fn cubin_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/triton-cubins")
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

fn hex_f32(value: f32) -> String {
    format!("{:08x}", value.to_bits())
}

fn same_hex(what: &str, expected: &str, actual: &str) {
    assert_eq!(expected, actual, "{what} f32 bits");
}

fn u32_of(value: &Value, what: &str) -> u32 {
    value
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or_else(|| panic!("{what} is not a u32"))
}

fn u32_try(value: u64, what: &str) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| panic!("{what} does not fit in u32"))
}

fn str_of<'a>(value: &'a Value, what: &str) -> &'a str {
    value
        .as_str()
        .unwrap_or_else(|| panic!("{what} is not a string"))
}

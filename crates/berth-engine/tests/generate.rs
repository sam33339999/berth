//! Generation loop on the full 27B. GPU-gated like `decode_seq`.
//!
//! `reset_repeats_teacher_forced_logits` drives [`Model`] directly.
//!
//! `generate_stops_for_length_cancel_and_greedy` checks, on one load:
//! greedy `argmax[0..299]`, a second request with the same tokens, cancel
//! after 5, and `max_new_tokens`.
//!
//! ```text
//! BERTH_GPU_TESTS=1 cargo test -p berth-engine --test generate -- --nocapture
//! ```

mod common;

use std::path::{Path, PathBuf};

use berth_engine::model::Model;
use berth_engine::{Engine, Finish, Flow, GenerateRequest, Gpu, SamplerConfig, Usage};
use berth_exl3::Checkpoint;
use berth_plan::build_plan;
use berth_plan::from_exl3::model_shape;
use blake2::digest::consts::U32;
use blake2::digest::Digest;
use blake2::Blake2b;

/// Same cache geometry as `testdata/decode_seq` and `testdata/decode_ref`.
/// `split_config` keys off the page-table width, so a different `max_tokens`
/// would not match the argmax chain.
const MAX_TOKENS: u32 = 4096;
/// `decode_seq` prompt token, also `decode_ref`'s token.
const PROMPT: u32 = 32;
const GREEDY_NEW: usize = 299;
const RESET_STEPS: usize = 20;

fn cubin_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/triton-cubins")
}

fn decode_seq_meta_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/decode_seq/meta.json")
}

/// Needs Sampler (task SAMPLE). One load covers length, cancel, the greedy
/// chain, and a second request that must repeat it.
#[test]
fn generate_stops_for_length_cancel_and_greedy() {
    let dir = match common::gpu_ready() {
        Ok(dir) => dir,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    let _slot = match common::full_model_slot() {
        Ok(slot) => slot,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    let vocab = vocab_of(&dir);

    eprintln!("loading engine");
    let mut engine = Engine::load(Path::new(&dir), &cubin_dir(), MAX_TOKENS, 0).unwrap();
    assert_eq!(engine.vocab_size(), vocab);
    assert_eq!(engine.max_tokens(), MAX_TOKENS);
    assert_eq!(engine.default_stop_tokens(), &[248_044, 248_046]);

    let argmax = argmax_from_meta();
    // The chain's token 258 is eos. Stops stay empty so all 299 argmax
    // tokens are sampled; `Finish::Stop` is covered by the id list above
    // and by the library tests of the parser.
    let open = request(u32::try_from(GREEDY_NEW).unwrap(), &[]);

    let (length_tokens, length_finish, length_usage) = collect(&mut engine, &request(4, &[]), None);
    assert_eq!(length_finish, Finish::Length, "max_new_tokens");
    assert_eq!(
        length_usage,
        Usage {
            prompt_tokens: 1,
            completion_tokens: 4
        }
    );
    assert_eq!(length_tokens, argmax[..4]);

    let (cancel_tokens, cancel_finish, cancel_usage) = collect(&mut engine, &open, Some(5));
    assert_eq!(cancel_finish, Finish::Cancelled, "cancel after 5");
    assert_eq!(cancel_usage.completion_tokens, 5);
    assert_eq!(cancel_tokens.len(), 5);
    assert_eq!(cancel_tokens, argmax[..5]);

    eprintln!("greedy chain, {GREEDY_NEW} tokens");
    let (first, first_finish, first_usage) = collect(&mut engine, &open, None);
    assert_eq!(first_finish, Finish::Length);
    assert_eq!(
        first_usage,
        Usage {
            prompt_tokens: 1,
            completion_tokens: u32::try_from(GREEDY_NEW).unwrap(),
        }
    );
    assert_eq!(first, argmax[..GREEDY_NEW], "greedy argmax[0..299]");

    eprintln!("second request");
    let (second, second_finish, second_usage) = collect(&mut engine, &open, None);
    assert_eq!(second_finish, Finish::Length);
    assert_eq!(second_usage, first_usage);
    assert_eq!(second, first, "back-to-back requests");
}

/// Two runs of the same 20 teacher-forced steps, with [`Model::reset_sequence`]
/// before each. Logits digests must match, and step 0 must match
/// `testdata/decode_ref` (token 32, cache 4096). Does not need Sampler.
#[test]
fn reset_repeats_teacher_forced_logits() {
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
    eprintln!("loading model for the reset check");
    let mut model = Model::load(&gpu, &ckpt, &plan, MAX_TOKENS).unwrap();

    let tokens = teacher_tokens();
    assert_eq!(tokens[0], common::reference_token());
    let want = meta_logits_digests(tokens.len());

    model.reset_sequence(&gpu).unwrap();
    let first = logits_digests(&gpu, &mut model, &tokens);
    if let Some(want) = &want {
        assert_eq!(first, *want, "first run vs decode_seq logits digests");
    }
    model.reset_sequence(&gpu).unwrap();
    let second = logits_digests(&gpu, &mut model, &tokens);
    assert_eq!(
        second, first,
        "reset did not restore the teacher-forced logits"
    );
    eprintln!("reset: {RESET_STEPS} steps match across two runs");
}

fn request(max_new_tokens: u32, stop_tokens: &[u32]) -> GenerateRequest {
    GenerateRequest {
        prompt: vec![PROMPT],
        max_new_tokens,
        stop_tokens: stop_tokens.to_vec(),
        sampler: SamplerConfig::greedy(),
    }
}

fn collect(
    engine: &mut Engine,
    request: &GenerateRequest,
    cancel_after: Option<usize>,
) -> (Vec<u32>, Finish, Usage) {
    let mut tokens = Vec::new();
    let (finish, usage) = engine
        .generate(request, &mut |token| {
            tokens.push(token);
            if cancel_after == Some(tokens.len()) {
                Flow::Cancel
            } else {
                Flow::Continue
            }
        })
        .unwrap();
    (tokens, finish, usage)
}

fn vocab_of(dir: &str) -> u32 {
    let ckpt = Checkpoint::open(dir).unwrap();
    let shape = model_shape(&ckpt.config).unwrap();
    build_plan(&shape).unwrap().vocab_size
}

fn teacher_tokens() -> Vec<u32> {
    let path = decode_seq_meta_path();
    if !path.is_file() {
        eprintln!("decode_seq meta is missing; teacher-forcing a synthetic prompt");
        return (0..RESET_STEPS)
            .map(|i| {
                if i == 0 {
                    PROMPT
                } else {
                    u32::try_from(i).unwrap()
                }
            })
            .collect();
    }
    let meta = read_meta(&path);
    let tokens = u32_list(&meta["input_tokens"], "input_tokens");
    assert!(
        tokens.len() >= RESET_STEPS,
        "input_tokens has {} entries",
        tokens.len()
    );
    tokens[..RESET_STEPS].to_vec()
}

fn argmax_from_meta() -> Vec<u32> {
    let path = decode_seq_meta_path();
    assert!(path.is_file(), "missing {}", path.display());
    let meta = read_meta(&path);
    let argmax = u32_list(&meta["argmax"], "argmax");
    assert!(
        argmax.len() >= GREEDY_NEW,
        "argmax has {} entries",
        argmax.len()
    );
    argmax
}

fn meta_logits_digests(steps: usize) -> Option<Vec<String>> {
    let path = decode_seq_meta_path();
    if !path.is_file() {
        eprintln!("skipping digest compare: {}", path.display());
        return None;
    }
    let meta = read_meta(&path);
    let names = meta["names"].as_array().expect("names");
    let column = names
        .iter()
        .position(|name| name.as_str() == Some("logits"))
        .expect("logits column");
    let rows = meta["digests"].as_array().expect("digests");
    assert!(rows.len() >= steps, "digests has {} rows", rows.len());
    Some(
        rows[..steps]
            .iter()
            .map(|row| {
                row.get(column)
                    .and_then(|value| value.as_str())
                    .expect("logits digest")
                    .to_string()
            })
            .collect(),
    )
}

fn read_meta(path: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(path).unwrap())
        .unwrap_or_else(|err| panic!("{}: {err}", path.display()))
}

fn u32_list(value: &serde_json::Value, what: &str) -> Vec<u32> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("{what} is missing"))
        .iter()
        .map(|item| {
            let number = item
                .as_u64()
                .unwrap_or_else(|| panic!("{what} entry {item} is not an integer"));
            u32::try_from(number).unwrap_or_else(|_| panic!("{what} entry {number} exceeds u32"))
        })
        .collect()
}

fn logits_digests(gpu: &Gpu, model: &mut Model, tokens: &[u32]) -> Vec<String> {
    let mut digests = Vec::with_capacity(tokens.len());
    for (step, token) in tokens.iter().copied().enumerate() {
        let position = u32::try_from(step).unwrap();
        model.decode_step(gpu, token, position).unwrap();
        model.synchronize(gpu).unwrap();
        let logits = common::download(gpu, model.logits().unwrap());
        if step == 0 {
            let want = common::reference("logits");
            let diff = common::differing(&logits, &want, 2);
            assert_eq!(diff, 0, "step 0 logits vs decode_ref");
        }
        digests.push(blake2b_256(&logits));
    }
    digests
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

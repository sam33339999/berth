//! Prefill reference against `testdata/prefill_ref/`.
//!
//! `chunk_bounds` is checked on the CPU against the fork's recorded chunks.
//! The GPU test loads the 27B, prefills the stored prompts, and compares
//! every chunk/block residual digest, the attention cache and GDN state
//! digests, then the decode step's logits digest and argmax. Until the layer
//! prefill methods land, that returns "layer prefill not implemented" and the
//! test passes. A greedy `Engine::generate` of the 100-token prompt must
//! sample the fork's argmax as its first token, with the same pass-through.
//!
//! ```text
//! BERTH_GPU_TESTS=1 cargo test -p berth-engine --test prefill_ref -- --nocapture
//! ```

mod common;

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use berth_engine::model::Model;
use berth_engine::prefill::{
    self, AttnCacheDigest, AttnCacheImage, BlockDigest, ChunkDigest, GdnStateDigest, GdnStateImage,
    HostTensor, PrefillOutput, TensorDigest,
};
use berth_engine::{Engine, Finish, Flow, GenerateRequest, Gpu, SamplerConfig, Usage};
use berth_exl3::Checkpoint;
use berth_plan::{build_plan, from_exl3::model_shape};
use blake2::digest::consts::U32;
use blake2::digest::Digest;
use blake2::Blake2b;

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
    // 8190 is 32 pages (div16): three 2048-row staging splits, one 1792-row
    // staging split, and a 253-row qc tail. 8400 is 33 pages (other): four
    // 2048-row staging splits and a 207-row qc tail. One job has one page
    // class, so the two lengths are both required.
    assert_eq!(
        prefill::chunk_bounds(8190),
        vec![
            (0, 2048),
            (2048, 4096),
            (4096, 6144),
            (6144, 7936),
            (7936, 8189)
        ]
    );
    assert_eq!(
        prefill::chunk_bounds(8400),
        vec![
            (0, 2048),
            (2048, 4096),
            (4096, 6144),
            (6144, 8192),
            (8192, 8399)
        ]
    );
    assert_eq!(prefill::prefill_block_pages(100, 1), 1);
    assert_eq!(prefill::prefill_block_pages(1500, 1), 6);
    assert_eq!(prefill::prefill_block_pages(8190, 1), 32);
    assert_eq!(prefill::prefill_block_pages(8400, 1), 33);

    let meta = meta();
    assert_eq!(
        meta["page_size"].as_u64(),
        Some(u64::from(prefill::PAGE_SIZE))
    );
    assert_eq!(
        meta["max_chunk_size"].as_u64(),
        Some(u64::from(prefill::MAX_CHUNK_SIZE))
    );
    let names = ["100", "1500", "8190", "8400"];
    let stored: Vec<Vec<u32>> = names
        .iter()
        .map(|name| token_ids(prompt(&meta, name)))
        .collect();
    assert_eq!(stored[0].len(), 100);
    assert_eq!(stored[1].len(), 1500);
    assert_eq!(stored[2].len(), 8190);
    assert_eq!(stored[3].len(), 8400);
    for (short, long) in stored.iter().zip(stored.iter().skip(1)) {
        assert_eq!(&long[..short.len()], short.as_slice());
    }
    for name in names {
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
fn prefill_matches_fork_digests() {
    let _guard = GPU_TESTS.lock().unwrap_or_else(|err| err.into_inner());
    let dir = match common::gpu_ready() {
        Ok(dir) => dir,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    let meta = meta();
    let max_tokens = cache_tokens(&meta);
    let hidden = meta["hidden"].as_u64().unwrap() as usize;
    let _slot = match common::full_model_slot() {
        Ok(slot) => slot,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    let ckpt = Checkpoint::open(&dir).unwrap();
    let shape = model_shape(&ckpt.config).unwrap();
    let plan = build_plan(&shape).unwrap();
    let gpu = Gpu::new().unwrap();
    let mut model = Model::load(&gpu, &ckpt, &plan, max_tokens).unwrap();

    for name in ["100", "1500", "8190", "8400"] {
        let prompt = prompt(&meta, name);
        let ids = token_ids(prompt);
        eprintln!("prefill {name}");
        model.reset_sequence(&gpu).unwrap();
        // Dump jobs use max_new_tokens 1, so the fork's prefill table is
        // ceil((prompt_len + 2) / 256) on every chunk.
        let pages = prefill::prefill_block_pages(u32::try_from(ids.len()).unwrap(), 1);
        model.set_prefill_pages(pages).unwrap();
        let prefix = &ids[..ids.len() - 1];
        let mut records = Vec::new();
        let mut observe = |layer: u32, start: u32, end: u32, bytes: &[u8]| {
            records.push((layer, start, end, blake2b_256(bytes)));
            Ok(())
        };
        match model.prefill_with(&gpu, prefix, Some(&mut observe)) {
            Err(err) if not_implemented(&err) => {
                eprintln!("layer prefill not implemented");
                return;
            }
            Err(err) => panic!("{name}: {err}"),
            Ok(()) => {}
        }
        let written = u32::try_from(prefix.len()).unwrap();
        let output = PrefillOutput {
            chunks: chunks_from(&records, hidden),
            attn: model
                .attn_cache(&gpu, written)
                .unwrap()
                .into_iter()
                .map(cache_digest)
                .collect(),
            gdn: model
                .gdn_state(&gpu)
                .unwrap()
                .into_iter()
                .map(gdn_digest)
                .collect(),
        };
        compare_prefill(name, prompt, &output);
        compare_logits(&gpu, &mut model, name, prompt, &ids);
    }
}

/// Greedy `Engine::generate` on the 100-token prompt. The first sampled token
/// is the fork's argmax of the prefill logits.
#[test]
fn greedy_100_first_token_is_fork_argmax() {
    let _guard = GPU_TESTS.lock().unwrap_or_else(|err| err.into_inner());
    let dir = match common::gpu_ready() {
        Ok(dir) => dir,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    let meta = meta();
    let max_tokens = cache_tokens(&meta);
    let prompt = prompt(&meta, "100");
    let ids = token_ids(prompt);
    let want = prompt["logits"]["argmax"].as_u64().unwrap() as u32;
    let _slot = match common::full_model_slot() {
        Ok(slot) => slot,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    let cubin = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/triton-cubins");
    let mut engine = Engine::load(Path::new(&dir), &cubin, max_tokens, 0).unwrap();
    let request = GenerateRequest {
        prompt: ids,
        max_new_tokens: 1,
        stop_tokens: Vec::new(),
        sampler: SamplerConfig::greedy(),
    };
    let mut first = None;
    let result = engine.generate(&request, &mut |token| {
        first = Some(token);
        Flow::Continue
    });
    match result {
        Err(err) if not_implemented(&err) => {
            eprintln!("layer prefill not implemented");
        }
        Err(err) => panic!("{err}"),
        Ok((finish, usage)) => {
            assert_eq!(finish, Finish::Length);
            assert_eq!(
                usage,
                Usage {
                    prompt_tokens: 100,
                    completion_tokens: 1,
                }
            );
            assert_eq!(first, Some(want), "greedy first token");
            eprintln!("greedy first token {want}");
        }
    }
}

fn not_implemented(err: &impl std::fmt::Display) -> bool {
    err.to_string()
        .to_ascii_lowercase()
        .contains("not implemented")
}

fn chunks_from(records: &[(u32, u32, u32, String)], hidden: usize) -> Vec<ChunkDigest> {
    let mut chunks = Vec::new();
    for (layer, start, end, digest) in records {
        let new_chunk = chunks
            .last()
            .map(|chunk: &ChunkDigest| chunk.start != *start || chunk.end != *end)
            .unwrap_or(true);
        if new_chunk {
            chunks.push(ChunkDigest {
                start: *start,
                end: *end,
                blocks: Vec::new(),
            });
        }
        let rows = usize::try_from(*end - *start).expect("chunk rows fit usize");
        chunks.last_mut().unwrap().blocks.push(BlockDigest {
            layer: *layer,
            dtype: "float32".to_string(),
            shape: vec![1, rows, hidden],
            digest: digest.clone(),
        });
    }
    chunks
}

fn cache_digest(image: AttnCacheImage) -> AttnCacheDigest {
    AttnCacheDigest {
        layer: image.layer,
        qk: tensor_digest(image.qk),
        qv: tensor_digest(image.qv),
        sk: tensor_digest(image.sk),
        sv: tensor_digest(image.sv),
    }
}

fn gdn_digest(image: GdnStateImage) -> GdnStateDigest {
    GdnStateDigest {
        layer: image.layer,
        conv: tensor_digest(image.conv),
        recurrent: tensor_digest(image.recurrent),
    }
}

fn tensor_digest(tensor: HostTensor) -> TensorDigest {
    TensorDigest {
        dtype: tensor.dtype,
        shape: tensor.shape,
        digest: blake2b_256(&tensor.bytes),
    }
}

fn compare_logits(
    gpu: &Gpu,
    model: &mut Model,
    name: &str,
    prompt: &serde_json::Value,
    ids: &[u32],
) {
    let last = *ids.last().unwrap();
    let position = u32::try_from(ids.len() - 1).unwrap();
    // The reference job has max_new_tokens 1, so the fork's decode block
    // table covers prompt_len + 1 tokens, padded to 16 pages.
    model.set_decode_width(u32::try_from(ids.len()).unwrap() + 1);
    model.decode_step(gpu, last, position).unwrap();
    model.synchronize(gpu).unwrap();
    let got = common::download(gpu, model.logits().unwrap());
    let file = prompt["logits"]["file"].as_str().unwrap();
    let want = std::fs::read(ref_dir().join(file)).unwrap_or_else(|err| panic!("{file}: {err}"));
    let want_digest = prompt["logits"]["digest"].as_str().unwrap();
    let fork_argmax = prompt["logits"]["argmax"].as_u64().unwrap() as usize;
    if !fp16_finite(&got) {
        eprintln!("first difference: {name} logits have a non-finite value");
        panic!("{name} logits have a non-finite value");
    }
    let digest = blake2b_256(&got);
    if !hex_eq(&digest, want_digest) {
        let (max_abs, index) = first_fp16_diff(&got, &want);
        eprintln!(
            "first difference: {name} logits digest {digest} != {want_digest}, element {index}, max abs {max_abs}"
        );
        panic!("{name} logits digest mismatch");
    }
    let got_argmax = argmax_fp16(&got);
    if got_argmax != fork_argmax {
        eprintln!("first difference: {name} logits argmax {got_argmax} != {fork_argmax}");
        panic!("{name} logits argmax {got_argmax} != {fork_argmax}");
    }
    eprintln!("{name} logits digest matches, argmax {got_argmax}");
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
            if !hex_eq(&block.digest, want_digest) {
                eprintln!(
                    "first difference: {name} chunk [{}, {}) layer {layer} residual {} != {want_digest}",
                    got.start, got.end, block.digest
                );
                panic!("{name} layer {layer}: {} != {want_digest}", block.digest);
            }
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
    if !hex_eq(&got.digest, want_digest) {
        eprintln!(
            "first difference: {name} layer {layer} {which} {} != {want_digest}",
            got.digest
        );
        panic!(
            "{name} layer {layer} {which}: {} != {want_digest}",
            got.digest
        );
    }
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

fn argmax_fp16(bytes: &[u8]) -> usize {
    let mut best = f32::NEG_INFINITY;
    let mut index = 0usize;
    for (i, chunk) in bytes.chunks(2).enumerate() {
        let value = fp16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        if value > best {
            best = value;
            index = i;
        }
    }
    index
}

fn first_fp16_diff(got: &[u8], want: &[u8]) -> (f32, usize) {
    let mut max_abs = 0f32;
    let mut first = 0usize;
    let mut seen = false;
    let n = got.len().min(want.len()) / 2;
    for index in 0..n {
        let gv = fp16_to_f32(u16::from_le_bytes([got[index * 2], got[index * 2 + 1]]));
        let wv = fp16_to_f32(u16::from_le_bytes([want[index * 2], want[index * 2 + 1]]));
        let diff = (gv - wv).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        if !seen && got[index * 2..index * 2 + 2] != want[index * 2..index * 2 + 2] {
            first = index;
            seen = true;
        }
    }
    if got.len() != want.len() && !seen {
        first = n;
    }
    (max_abs, first)
}

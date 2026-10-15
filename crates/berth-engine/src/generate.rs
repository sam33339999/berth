//! One request, end to end: feed the prompt, then sample until a stop.
//!
//! A prompt longer than one token is the fork's prefill: [`Model::prefill`]
//! on every id except the last (`generator/job.py` `Job.prefill` forwards
//! `ids[:-1]`), then [`Model::decode_step`] of that last id. A one-token
//! prompt is only the decode step.
//!
//! Prompt tokens occupy positions `0..n-1` and are not sampled. The logits
//! of the last prompt token are the first sample. Each later sample is fed
//! back at the next position. Generation stops on a stop token, on
//! `max_new_tokens`, on [`Flow::Cancel`], or when the next token would not
//! fit in the cache. The stop token is reported.

use std::path::Path;

use berth_exl3::Checkpoint;
use berth_plan::build_plan;
use berth_plan::from_exl3::model_shape;

use crate::model::Model;
use crate::sampler::{Sampler, SamplerConfig};
use crate::{Error, Gpu, Result};

/// What the caller asks for.
#[derive(Clone, Debug)]
pub struct GenerateRequest {
    /// Prompt token ids, already templated and tokenized.
    pub prompt: Vec<u32>,
    pub max_new_tokens: u32,
    /// Generation ends after any of these is sampled (it is still reported).
    pub stop_tokens: Vec<u32>,
    pub sampler: SamplerConfig,
}

/// Returned by the per-token callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    Continue,
    /// Stop now, e.g. the client closed the connection.
    Cancel,
}

/// Why generation ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// A stop token was sampled.
    Stop,
    /// `max_new_tokens` reached.
    Length,
    /// The callback returned [`Flow::Cancel`].
    Cancelled,
    /// The context (`max_tokens` of the cache) is full.
    ContextFull,
}

/// Counts for the response's `usage`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

/// The loaded model, its sampler, and the device. One request at a time.
///
/// `Send`, so the server can move it onto its own thread. Fields stay private
/// and are ordinary owned buffers; nothing here is `unsafe impl Send`.
pub struct Engine {
    gpu: Gpu,
    model: Model,
    sampler: Sampler,
    vocab: u32,
    max_tokens: u32,
    /// Stop ids the serve would put on a job. See [`Self::default_stop_tokens`].
    stop_tokens: Vec<u32>,
}

/// `Engine` moves to the server thread. A non-`Send` field fails this build
/// instead of being papered over with `unsafe impl Send`.
const _: fn(&Engine) = assert_engine_send;

fn assert_engine_send<T: Send>(_: &T) {}

impl Engine {
    /// Open the checkpoint at `model_dir` and load it with a `max_tokens` cache.
    /// `cubin_dir` holds the Triton attention cubins (`target/triton-cubins`).
    pub fn load(model_dir: &Path, cubin_dir: &Path, max_tokens: u32) -> Result<Self> {
        let stop_tokens = read_default_stop_tokens(model_dir)?;
        let ckpt = Checkpoint::open(model_dir)?;
        let shape = model_shape(&ckpt.config).map_err(|err| Error::Weights(err.to_string()))?;
        let plan = build_plan(&shape).map_err(|err| Error::Weights(err.to_string()))?;
        let gpu = Gpu::new()?;
        let model = Model::load_with_cubins(&gpu, &ckpt, &plan, max_tokens, cubin_dir)?;
        let sampler = Sampler::new(&gpu, plan.vocab_size, max_tokens)?;
        Ok(Self {
            gpu,
            model,
            sampler,
            vocab: plan.vocab_size,
            max_tokens,
            stop_tokens,
        })
    }

    /// Vocabulary size of the loaded model.
    pub fn vocab_size(&self) -> u32 {
        self.vocab
    }

    /// Cache length in tokens; prompt plus output must fit.
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    /// Stop token ids, in the serve's order.
    ///
    /// `serve_openai.py` sets `stop_ids` from `model.config.eos_token_id_list`
    /// after `Tokenizer` has extended it, then appends `tokenizer.eos_token_id`
    /// when that id is missing. The list is `config.json`'s `eos_token_id`
    /// (top level, else `text_config`), plus the `tokenizer_config.json`
    /// `eos_token` piece when its id is non-zero and new, plus every
    /// `generation_config.json` `eos_token_id`, plus the tokenizer's own eos
    /// id (`tokenizer.py` `get_default_token_id`, default 2 only when the
    /// config and the tokenizer config both lack one).
    pub fn default_stop_tokens(&self) -> &[u32] {
        &self.stop_tokens
    }

    /// Run one request. Every sampled token, including a final stop token,
    /// goes to `on_token` as soon as it is sampled.
    ///
    /// Per-sequence state is cleared first: attention cache length, GDN conv
    /// and recurrent state, and [`Sampler::reset`] with this prompt. A cancel
    /// reported for a token wins over that token also being a stop token.
    /// A stop token wins over `max_new_tokens` when the same token would trip
    /// both, matching `Job.receive_token` (stop check, then the length check).
    pub fn generate(
        &mut self,
        request: &GenerateRequest,
        on_token: &mut dyn FnMut(u32) -> Flow,
    ) -> Result<(Finish, Usage)> {
        let prompt_len = self.check_request(request)?;
        self.model.reset_sequence(&self.gpu)?;
        // The fork sizes the decode block table from the job's whole length
        // (`Job.get_max_seq_len`: prompt plus max_new_tokens).
        self.model
            .set_decode_width(prompt_len.saturating_add(request.max_new_tokens));
        // Prefill sees the job's allocated pages, not the pages written so
        // far and not the 16-padded decode width. `max_rq_tokens` is
        // `max_new_tokens + 1` when the generator has no draft.
        let prefill_pages = crate::prefill::prefill_block_pages(prompt_len, request.max_new_tokens);
        self.model.set_prefill_pages(prefill_pages)?;
        self.sampler
            .reset(&self.gpu, &request.sampler, &request.prompt)?;

        let max_tokens = self.max_tokens;
        let Engine {
            gpu,
            model,
            sampler,
            ..
        } = self;
        feed_prompt(model, gpu, &request.prompt)?;

        let mut completion = 0u32;
        loop {
            let token = sampler.sample(gpu, model.logits_mut()?)?;
            completion += 1;
            let usage = Usage {
                prompt_tokens: prompt_len,
                completion_tokens: completion,
            };
            if on_token(token) == Flow::Cancel {
                return Ok((Finish::Cancelled, usage));
            }
            if request.stop_tokens.contains(&token) {
                return Ok((Finish::Stop, usage));
            }
            if completion >= request.max_new_tokens {
                return Ok((Finish::Length, usage));
            }
            // The token just sampled is decoded at this position to produce
            // the next logits. `prompt_len + completion - 1` is that index:
            // the first sample comes from the last prompt position and is
            // fed back at `prompt_len`.
            let Some(next_pos) = prompt_len.checked_add(completion - 1) else {
                return Ok((Finish::ContextFull, usage));
            };
            if next_pos >= max_tokens {
                return Ok((Finish::ContextFull, usage));
            }
            model.decode_step(gpu, token, next_pos)?;
            model.synchronize(gpu)?;
        }
    }

    fn check_request(&self, request: &GenerateRequest) -> Result<u32> {
        if request.prompt.is_empty() {
            return Err(Error::Input("prompt is empty".into()));
        }
        if request.max_new_tokens == 0 {
            return Err(Error::Input("max_new_tokens must be at least 1".into()));
        }
        let prompt_len = u32::try_from(request.prompt.len())
            .map_err(|_| Error::Input("prompt is longer than u32::MAX".into()))?;
        if prompt_len > self.max_tokens {
            return Err(Error::Input(format!(
                "prompt of {prompt_len} tokens does not fit the {}-token cache",
                self.max_tokens
            )));
        }
        for &token in &request.prompt {
            if token >= self.vocab {
                return Err(Error::Input(format!(
                    "token {token} is past the vocabulary ({})",
                    self.vocab
                )));
            }
        }
        Ok(prompt_len)
    }
}

/// Prompt tokens at positions `0..prompt.len()`.
///
/// A longer prompt is `Job.prefill` (`generator/job.py`): [`Model::prefill`]
/// on `prompt[..len - 1]`, then one [`Model::decode_step`] of the last id at
/// position `len - 1`. A one-token prompt is only that decode step. The step
/// is synchronized so the logits are finished before sampling.
fn feed_prompt(model: &mut Model, gpu: &Gpu, prompt: &[u32]) -> Result<()> {
    if prompt.len() > 1 {
        model.prefill(gpu, &prompt[..prompt.len() - 1])?;
    }
    let last = *prompt.last().expect("prompt is non-empty");
    let position = u32::try_from(prompt.len() - 1).expect("prompt length fits u32");
    model.decode_step(gpu, last, position)?;
    model.synchronize(gpu)?;
    Ok(())
}

/// Stop ids in the order `serve_openai.py` builds `self.stop_ids`.
///
/// Fork:
/// * `model/config.py` reads `eos_token_id`, else `text_config->eos_token_id`,
///   and stores a list with `None` entries removed. A list's first element,
///   before that filter, is `eos_token_id`.
/// * `tokenizer/tokenizer.py` resolves `tokenizer_config.json` `eos_token`
///   through `token_to_id`. A non-zero id missing from the list is appended
///   (`if e:`). `generation_config.json` `eos_token_id` is appended the same
///   way. The tokenizer's own `eos_token_id` stays at the config value when
///   the config had one; otherwise it is the resolved piece, or `2` when the
///   tokenizer config has no `eos_token` key.
/// * `serve_openai.py` copies `eos_token_id_list` and appends
///   `tokenizer.eos_token_id` when it is not `None` and not already listed.
fn read_default_stop_tokens(model_dir: &Path) -> Result<Vec<u32>> {
    let config = read_required_json(&model_dir.join("config.json"))?;
    let (mut list, config_first) = config_eos(&config)?;
    let tokenizer = read_optional_json(&model_dir.join("tokenizer_config.json"))?;
    let pieces = piece_map(tokenizer.as_ref())?;
    let piece_id = match tokenizer_eos_piece(tokenizer.as_ref()) {
        Some(piece) => lookup_piece(model_dir, piece, &pieces)?,
        None => None,
    };
    // `if e:` in `tokenizer.py`. Id 0 is a real token but it is skipped here.
    if let Some(id) = piece_id {
        if id != 0 {
            push_new(&mut list, id);
        }
    }
    if let Some(generation) = read_optional_json(&model_dir.join("generation_config.json"))? {
        append_generation_eos(&mut list, &generation)?;
    }
    let serve_eos = match config_first {
        Some(id) => Some(id),
        None if tokenizer_eos_key_absent(tokenizer.as_ref()) => Some(2),
        None => piece_id,
    };
    if let Some(id) = serve_eos {
        push_new(&mut list, id);
    }
    Ok(list)
}

fn push_new(list: &mut Vec<u32>, id: u32) {
    if !list.contains(&id) {
        list.push(id);
    }
}

fn config_eos(root: &serde_json::Value) -> Result<(Vec<u32>, Option<u32>)> {
    let Some(value) = first_present(root, &["eos_token_id", "text_config->eos_token_id"]) else {
        return Ok((Vec::new(), None));
    };
    match value {
        serde_json::Value::Array(items) if items.is_empty() => Ok((Vec::new(), None)),
        serde_json::Value::Array(items) => {
            let first = if items[0].is_null() {
                None
            } else {
                Some(json_u32(&items[0], "eos_token_id")?)
            };
            let mut list = Vec::new();
            for (index, item) in items.iter().enumerate() {
                if item.is_null() {
                    continue;
                }
                list.push(json_u32(item, &format!("eos_token_id[{index}]"))?);
            }
            Ok((list, first))
        }
        other => {
            let id = json_u32(other, "eos_token_id")?;
            Ok((vec![id], Some(id)))
        }
    }
}

fn append_generation_eos(list: &mut Vec<u32>, root: &serde_json::Value) -> Result<()> {
    let Some(value) = root.get("eos_token_id") else {
        return Ok(());
    };
    // `if ee:` is false for null, for 0, and for an empty list.
    if value.is_null() {
        return Ok(());
    }
    match value {
        serde_json::Value::Array(items) if items.is_empty() => Ok(()),
        serde_json::Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                if item.is_null() {
                    continue;
                }
                let id = json_u32(
                    item,
                    &format!("generation_config.json eos_token_id[{index}]"),
                )?;
                push_new(list, id);
            }
            Ok(())
        }
        serde_json::Value::Number(number)
            if number.as_u64() == Some(0) || number.as_i64() == Some(0) =>
        {
            Ok(())
        }
        other => {
            push_new(
                list,
                json_u32(other, "generation_config.json eos_token_id")?,
            );
            Ok(())
        }
    }
}

fn tokenizer_eos_key_absent(config: Option<&serde_json::Value>) -> bool {
    match config {
        None => true,
        Some(value) => value.get("eos_token").is_none(),
    }
}

fn tokenizer_eos_piece(config: Option<&serde_json::Value>) -> Option<&str> {
    let value = config?.get("eos_token")?;
    match value {
        serde_json::Value::String(text) => Some(text.as_str()),
        serde_json::Value::Object(map) => map.get("content").and_then(serde_json::Value::as_str),
        _ => None,
    }
}

fn piece_map(config: Option<&serde_json::Value>) -> Result<std::collections::HashMap<String, u32>> {
    let mut map = std::collections::HashMap::new();
    let Some(config) = config else {
        return Ok(map);
    };
    let Some(decoder) = config.get("added_tokens_decoder") else {
        return Ok(map);
    };
    if decoder.is_null() {
        return Ok(map);
    }
    let obj = decoder.as_object().ok_or_else(|| {
        Error::Weights("tokenizer_config.json added_tokens_decoder is not an object".into())
    })?;
    for (key, spec) in obj {
        let id = key.parse::<u32>().map_err(|_| {
            Error::Weights(format!(
                "added_tokens_decoder key `{key}` is not a u32 token id"
            ))
        })?;
        let content = spec
            .get("content")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                Error::Weights(format!("added_tokens_decoder[{key}] has no string content"))
            })?;
        match map.entry(content.to_string()) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(id);
            }
            std::collections::hash_map::Entry::Occupied(slot) if *slot.get() == id => {}
            std::collections::hash_map::Entry::Occupied(slot) => {
                return Err(Error::Weights(format!(
                    "added token `{content}` is id {} and {id}",
                    slot.get()
                )));
            }
        }
    }
    Ok(map)
}

/// `Tokenizer.token_to_id` for a special piece.
///
/// `added_tokens_decoder` in `tokenizer_config.json` is the table the 27B
/// uses (`<|im_end|>` is 248046). A piece that is only in `tokenizer.json`'s
/// `added_tokens` array is resolved from that file, which is what the
/// HuggingFace tokenizer reads.
fn lookup_piece(
    model_dir: &Path,
    piece: &str,
    known: &std::collections::HashMap<String, u32>,
) -> Result<Option<u32>> {
    if let Some(id) = known.get(piece) {
        return Ok(Some(*id));
    }
    let Some(value) = read_optional_json(&model_dir.join("tokenizer.json"))? else {
        return Ok(None);
    };
    let Some(added) = value
        .get("added_tokens")
        .and_then(serde_json::Value::as_array)
    else {
        return Ok(None);
    };
    for item in added {
        let Some(content) = item.get("content").and_then(serde_json::Value::as_str) else {
            continue;
        };
        if content != piece {
            continue;
        }
        let id = item.get("id").ok_or_else(|| {
            Error::Weights(format!("tokenizer.json added token `{piece}` has no id"))
        })?;
        return Ok(Some(json_u32(id, "tokenizer.json added token id")?));
    }
    Ok(None)
}

fn first_present<'a>(root: &'a serde_json::Value, keys: &[&str]) -> Option<&'a serde_json::Value> {
    keys.iter().find_map(|key| dig(root, key))
}

fn dig<'a>(root: &'a serde_json::Value, key: &str) -> Option<&'a serde_json::Value> {
    let mut cur = root;
    for part in key.split("->") {
        match cur.get(part) {
            Some(serde_json::Value::Null) | None => return None,
            Some(next) => cur = next,
        }
    }
    Some(cur)
}

fn json_u32(value: &serde_json::Value, what: &str) -> Result<u32> {
    if let Some(number) = value.as_u64() {
        return u32::try_from(number)
            .map_err(|_| Error::Weights(format!("{what} {number} does not fit in u32")));
    }
    if let Some(number) = value.as_i64() {
        return u32::try_from(number)
            .map_err(|_| Error::Weights(format!("{what} {number} is not a token id")));
    }
    if let Some(number) = value.as_f64() {
        let max = f64::from(u32::MAX);
        // `1.0` is a JSON float. An integral value in range is still a token id.
        if number.is_finite() && number.fract() == 0.0 && (0.0..=max).contains(&number) {
            let text = format!("{number:.0}");
            return text
                .parse::<u32>()
                .map_err(|_| Error::Weights(format!("{what} {number} does not fit in u32")));
        }
    }
    Err(Error::Weights(format!("{what} is not an integer token id")))
}

fn read_required_json(path: &Path) -> Result<serde_json::Value> {
    match read_optional_json(path)? {
        Some(value) => Ok(value),
        None => Err(Error::Read {
            name: path.display().to_string(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"),
        }),
    }
}

fn read_optional_json(path: &Path) -> Result<Option<serde_json::Value>> {
    match std::fs::read(path) {
        Ok(bytes) => {
            let value: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|err| Error::Weights(format!("{}: {err}", path.display())))?;
            if !value.is_object() {
                return Err(Error::Weights(format!(
                    "{}: expected a JSON object",
                    path.display()
                )));
            }
            Ok(Some(value))
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(Error::Read {
            name: path.display().to_string(),
            source,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicU64, Ordering};

    static FIXTURES: AtomicU64 = AtomicU64::new(0);

    struct Tmp {
        path: std::path::PathBuf,
    }

    impl Tmp {
        fn write(files: &[(&str, &str)]) -> Self {
            let n = FIXTURES.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("berth-stop-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&path).unwrap();
            for (name, body) in files {
                std::fs::write(path.join(name), body).unwrap();
            }
            Self { path }
        }
    }

    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn stops(dir: &Path) -> Vec<u32> {
        read_default_stop_tokens(dir).unwrap()
    }

    #[test]
    fn serve_order_is_config_then_tokenizer_piece_then_generation() {
        let dir = Tmp::write(&[
            ("config.json", r#"{"text_config": {"eos_token_id": 10}}"#),
            (
                "tokenizer_config.json",
                r#"{"eos_token": "<|im_end|>", "added_tokens_decoder": {"20": {"content": "<|im_end|>"}}}"#,
            ),
            ("generation_config.json", r#"{"eos_token_id": [30, 10]}"#),
        ]);
        assert_eq!(stops(&dir.path), vec![10, 20, 30]);
    }

    #[test]
    fn top_level_eos_wins_over_text_config() {
        let dir = Tmp::write(&[(
            "config.json",
            r#"{"eos_token_id": 1, "text_config": {"eos_token_id": 2}}"#,
        )]);
        assert_eq!(stops(&dir.path), vec![1]);
    }

    #[test]
    fn missing_eos_falls_back_to_tokenizer_default_2() {
        let dir = Tmp::write(&[("config.json", "{}")]);
        assert_eq!(stops(&dir.path), vec![2]);
    }

    #[test]
    fn zero_piece_id_is_not_appended_by_the_tokenizer_update() {
        // `if e:` skips 0. The config already supplied 10, so the serve's
        // own eos id is 10 and 0 never enters the list.
        let dir = Tmp::write(&[
            ("config.json", r#"{"eos_token_id": 10}"#),
            (
                "tokenizer_config.json",
                r#"{"eos_token": "<eos>", "added_tokens_decoder": {"0": {"content": "<eos>"}}}"#,
            ),
        ]);
        assert_eq!(stops(&dir.path), vec![10]);
    }

    #[test]
    fn zero_piece_id_is_the_serve_eos_when_config_has_none() {
        let dir = Tmp::write(&[
            ("config.json", "{}"),
            (
                "tokenizer_config.json",
                r#"{"eos_token": "<eos>", "added_tokens_decoder": {"0": {"content": "<eos>"}}}"#,
            ),
        ]);
        assert_eq!(stops(&dir.path), vec![0]);
    }

    #[test]
    fn eos_token_object_uses_its_content_field() {
        let dir = Tmp::write(&[
            ("config.json", r#"{"eos_token_id": 4}"#),
            (
                "tokenizer_config.json",
                r#"{"eos_token": {"content": "</s>"}, "added_tokens_decoder": {"8": {"content": "</s>"}}}"#,
            ),
        ]);
        assert_eq!(stops(&dir.path), vec![4, 8]);
    }

    #[test]
    fn generation_config_integer_is_appended() {
        let dir = Tmp::write(&[
            ("config.json", r#"{"eos_token_id": 4}"#),
            ("generation_config.json", r#"{"eos_token_id": 9}"#),
        ]);
        assert_eq!(stops(&dir.path), vec![4, 9]);
    }

    #[test]
    fn generation_config_float_integer_is_appended() {
        let dir = Tmp::write(&[
            ("config.json", r#"{"eos_token_id": 4}"#),
            ("generation_config.json", r#"{"eos_token_id": 9.0}"#),
        ]);
        assert_eq!(stops(&dir.path), vec![4, 9]);
    }

    #[test]
    fn qwen38_27b_stop_tokens_match_the_serve() {
        let dir = std::env::var_os("BERTH_27B_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from("/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3")
            });
        if !dir.join("generation_config.json").is_file() {
            eprintln!("skipping: {} has no generation_config.json", dir.display());
            return;
        }
        assert_eq!(stops(&dir), vec![248_044, 248_046]);
    }
}

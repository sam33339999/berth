//! Tokenizer, chat template, and end-of-sequence ids for one model directory.
//!
//! Rendering follows `serve_openai.py` `render_chat` → `Tokenizer.hf_chat_template`
//! → transformers `apply_chat_template` (`tokenization_utils_base.py`).

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use minijinja::syntax::SyntaxConfig;
use minijinja::value::Serde;
use minijinja::{AutoEscape, Environment, Error as JinjaError, ErrorKind, Value as JinjaValue};
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

use crate::{Error, Result};

pub struct Assets {
    tokenizer: Tokenizer,
    env: Environment<'static>,
    special: Vec<(String, String)>,
    eos_ids: Vec<u32>,
}

impl Assets {
    pub fn render_chat(
        &self,
        messages: &[Value],
        tools: Option<&Value>,
        template_kwargs: &Map<String, Value>,
    ) -> Result<String> {
        let normalized = normalize_messages(messages);
        let mut ctx = Map::new();
        for (key, value) in &self.special {
            ctx.insert(key.clone(), Value::String(value.clone()));
        }
        for (key, value) in template_kwargs {
            ctx.insert(key.clone(), value.clone());
        }
        ctx.insert("messages".into(), Value::Array(normalized));
        ctx.insert("add_generation_prompt".into(), Value::Bool(true));
        ctx.insert("documents".into(), Value::Null);
        ctx.insert("tools".into(), tools.cloned().unwrap_or(Value::Null));
        let value = JinjaValue::from(Serde(Value::Object(ctx)));
        let template = self
            .env
            .get_template("chat")
            .map_err(|err| Error::new(format!("chat template: {err}")))?;
        template
            .render(value)
            .map_err(|err| Error::new(format!("chat template: {err}")))
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|err| Error::new(format!("tokenize: {err}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|err| Error::new(format!("detokenize: {err}")))
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    pub fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

pub fn load_assets(model_dir: &Path) -> Result<Assets> {
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|err| Error::new(format!("tokenizer.json: {err}")))?;
    let template = std::fs::read_to_string(model_dir.join("chat_template.jinja"))
        .map_err(|err| Error::new(format!("chat_template.jinja: {err}")))?;
    let config: Value = read_json(&model_dir.join("config.json"))?;
    let tokenizer_config: Value = read_json(&model_dir.join("tokenizer_config.json"))?;
    let eos_ids = eos_ids(&config, &tokenizer_config, &tokenizer);
    let special = special_tokens(&tokenizer_config);
    let env = chat_environment(template)?;
    Ok(Assets {
        tokenizer,
        env,
        special,
        eos_ids,
    })
}

fn read_json(path: &Path) -> Result<Value> {
    let file = File::open(path).map_err(|err| Error::new(format!("{}: {err}", path.display())))?;
    serde_json::from_reader(BufReader::new(file))
        .map_err(|err| Error::new(format!("{}: {err}", path.display())))
}

/// `exllamav3` `Config.eos_token_id_list`, plus the tokenizer eos if it is absent.
/// `serve_openai.py` `Server.__init__` stores that list as `stop_ids`.
fn eos_ids(config: &Value, tokenizer_config: &Value, tokenizer: &Tokenizer) -> Vec<u32> {
    let mut ids = Vec::new();
    let eos = config
        .get("eos_token_id")
        .filter(|value| !value.is_null())
        .or_else(|| config.pointer("/text_config/eos_token_id"));
    match eos {
        Some(Value::Number(number)) => {
            if let Some(id) = number_u32(number) {
                ids.push(id);
            }
        }
        Some(Value::Array(values)) => {
            for value in values {
                if let Some(id) = value.as_number().and_then(number_u32) {
                    ids.push(id);
                }
            }
        }
        _ => {}
    }
    if let Some(token) = tokenizer_config.get("eos_token").and_then(Value::as_str) {
        if let Some(id) = tokenizer.token_to_id(token) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
    }
    ids
}

fn number_u32(number: &serde_json::Number) -> Option<u32> {
    if let Some(value) = number.as_u64() {
        return u32::try_from(value).ok();
    }
    let value = number.as_f64()?;
    if value.is_finite() && value >= 0.0 && value <= u32::MAX as f64 && value.fract() == 0.0 {
        Some(value as u32)
    } else {
        None
    }
}

/// Transformers 5 `special_tokens_map`: named tokens that are set, then
/// `extra_special_tokens` promoted by `_set_model_specific_special_tokens`.
fn special_tokens(config: &Value) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for key in [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
    ] {
        if let Some(token) = config.get(key).and_then(Value::as_str) {
            out.push((key.to_string(), token.to_string()));
        }
    }
    if let Some(extra) = config
        .get("extra_special_tokens")
        .and_then(Value::as_object)
    {
        for (key, value) in extra {
            if let Some(token) = value.as_str() {
                out.push((key.clone(), token.to_string()));
            }
        }
    }
    out
}

fn chat_environment(template: String) -> Result<Environment<'static>> {
    let mut env = Environment::new();
    env.set_syntax(
        SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .expect("default delimiters"),
    );
    env.set_auto_escape_callback(|_| AutoEscape::None);
    env.add_function("raise_exception", raise_exception);
    env.add_filter("tojson", tojson);
    env.set_unknown_method_callback(unknown_method);
    env.add_template_owned("chat", template)
        .map_err(|err| Error::new(format!("chat template: {err}")))?;
    Ok(env)
}

fn raise_exception(message: String) -> std::result::Result<JinjaValue, JinjaError> {
    Err(JinjaError::new(ErrorKind::InvalidOperation, message))
}

/// Hugging Face `tojson`: `json.dumps(..., ensure_ascii=False)` with the default
/// separators `(', ', ': ')`. Jinja's built-in filter HTML-escapes.
fn tojson(value: JinjaValue) -> std::result::Result<String, JinjaError> {
    let json = serde_json::to_value(&value)
        .map_err(|err| JinjaError::new(ErrorKind::InvalidOperation, format!("tojson: {err}")))?;
    Ok(py_dumps(&json))
}

fn py_dumps(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(true) => "true".to_string(),
        Value::Bool(false) => "false".to_string(),
        Value::Number(number) => number.to_string(),
        Value::String(text) => serde_json::to_string(text).unwrap_or_else(|_| "\"\"".to_string()),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(py_dumps).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(key, value)| {
                    let key = serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string());
                    format!("{key}: {}", py_dumps(value))
                })
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

fn unknown_method(
    _state: &mut minijinja::State<'_, '_>,
    value: &JinjaValue,
    method: &str,
    args: &[JinjaValue],
) -> std::result::Result<JinjaValue, JinjaError> {
    let Some(text) = value.as_str() else {
        return Err(JinjaError::new(
            ErrorKind::InvalidOperation,
            format!("{method}() on a non-string"),
        ));
    };
    let argument = args.first().and_then(JinjaValue::as_str).unwrap_or("");
    match method {
        "startswith" => Ok(JinjaValue::from(text.starts_with(argument))),
        "endswith" => Ok(JinjaValue::from(text.ends_with(argument))),
        _ => Err(JinjaError::new(ErrorKind::UnknownMethod, method.to_owned())),
    }
}

/// `serve_openai.py` `normalize_messages`: tool arguments must be a mapping
/// because the chat template walks them with `|items`.
fn normalize_messages(messages: &[Value]) -> Vec<Value> {
    messages.iter().map(normalize_message).collect()
}

fn normalize_message(message: &Value) -> Value {
    let Some(object) = message.as_object() else {
        return message.clone();
    };
    let mut object = object.clone();
    if let Some(calls) = object
        .get("tool_calls")
        .cloned()
        .and_then(|value| match value {
            Value::Array(calls) => Some(calls),
            _ => None,
        })
    {
        let fixed = calls.iter().map(normalize_call).collect();
        object.insert("tool_calls".into(), Value::Array(fixed));
    }
    Value::Object(object)
}

fn normalize_call(call: &Value) -> Value {
    let Some(object) = call.as_object() else {
        return call.clone();
    };
    let mut object = object.clone();
    if let Some(function) = object.get("function").and_then(Value::as_object) {
        let mut function = function.clone();
        if let Some(arguments) = function.get("arguments") {
            let mapped = arguments_mapping(arguments);
            function.insert("arguments".into(), mapped);
        }
        object.insert("function".into(), Value::Object(function));
    } else if let Some(arguments) = object.get("arguments").cloned() {
        object.insert("arguments".into(), arguments_mapping(&arguments));
    }
    Value::Object(object)
}

fn arguments_mapping(arguments: &Value) -> Value {
    match arguments {
        Value::Null => Value::Object(Map::new()),
        Value::String(text) if text.is_empty() => Value::Object(Map::new()),
        Value::Object(_) => arguments.clone(),
        Value::String(text) => match serde_json::from_str::<Value>(text) {
            Ok(Value::Object(map)) => Value::Object(map),
            Ok(other) => {
                let mut map = Map::new();
                map.insert("value".into(), other);
                Value::Object(map)
            }
            Err(_) => {
                let mut map = Map::new();
                map.insert("arguments".into(), Value::String(text.clone()));
                Value::Object(map)
            }
        },
        other => {
            let mut map = Map::new();
            map.insert("value".into(), other.clone());
            Value::Object(map)
        }
    }
}

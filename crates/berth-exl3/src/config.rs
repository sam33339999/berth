use serde_json::{Map, Value};

use crate::{Error, Result};

/// One entry of `layer_types`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayerType {
    /// Gated delta net (`"linear_attention"`).
    LinearAttention,
    /// Standard attention (`"full_attention"`).
    FullAttention,
    /// A string this reader does not special-case.
    Other(String),
}

/// `quantization_config.calibration`, when the checkpoint recorded one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Calibration {
    /// Calibration rows.
    pub rows: u32,
    /// Calibration columns.
    pub cols: u32,
}

/// `quantization_config` from `config.json`.
///
/// Unknown keys are ignored. `codebook` is the conversion-wide setting; each
/// linear's codebook is detected from its own `mcg` / `mul1` tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizationConfig {
    /// Quantization method. EXL3 checkpoints use `"exl3"`.
    pub quant_method: String,
    /// Target average bits per weight. May be fractional.
    pub bits: f64,
    /// Bits used for `lm_head`.
    pub head_bits: u32,
    /// Conversion codebook name (`"mul1"`, `"mcg"`, `"3inst"`), when written.
    pub codebook: Option<String>,
    /// exllamav3 version string, when written.
    pub version: Option<String>,
    /// Vision-tower bitrate, when written.
    pub vision_bits: Option<u32>,
    /// MTP bitrate, when written.
    pub mtp_bits: Option<u32>,
    /// Output-scale policy (`"always"`, `"never"`, `"auto"`), when written.
    pub out_scales: Option<String>,
    /// Calibration shape, when written.
    pub calibration: Option<Calibration>,
}

/// Text stack and quantization settings from `config.json`.
///
/// Text fields are read from `text_config` when that object contains the key,
/// otherwise from the top level. `architectures` and `quantization_config` are
/// always top-level. Unknown keys are ignored.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    /// `architectures[0]`.
    pub arch_string: String,
    /// Number of transformer layers.
    pub num_hidden_layers: u32,
    /// Residual width.
    pub hidden_size: u32,
    /// Dense MLP intermediate width. Absent on MoE configs.
    pub intermediate_size: Option<u32>,
    /// Full-attention head dimension.
    pub head_dim: u32,
    /// Query heads.
    pub num_attention_heads: u32,
    /// Key / value heads.
    pub num_key_value_heads: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Whether `lm_head` is tied to the token embedding.
    pub tie_word_embeddings: bool,
    /// Per-layer mixer. Length matches `num_hidden_layers`.
    pub layer_types: Vec<LayerType>,
    /// GDN convolution kernel width.
    pub linear_conv_kernel_dim: u32,
    /// GDN key heads.
    pub linear_num_key_heads: u32,
    /// GDN value heads.
    pub linear_num_value_heads: u32,
    /// GDN key head dimension.
    pub linear_key_head_dim: u32,
    /// GDN value head dimension.
    pub linear_value_head_dim: u32,
    /// Whether full attention multiplies by an output gate.
    pub attn_output_gate: bool,
    /// Routed experts, when the config is MoE.
    pub num_experts: Option<u32>,
    /// Experts selected per token, when the config is MoE.
    pub num_experts_per_tok: Option<u32>,
    /// Routed-expert intermediate width, when present.
    pub moe_intermediate_size: Option<u32>,
    /// Shared-expert intermediate width, when present.
    pub shared_expert_intermediate_size: Option<u32>,
    /// EXL3 quantization block.
    pub quantization: QuantizationConfig,
}

impl ModelConfig {
    /// Parse `config.json` bytes.
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        let value =
            serde_json::from_slice(bytes).map_err(|source| Error::parse("config.json", source))?;
        Self::from_value(&value)
    }

    /// Parse an already-decoded `config.json` value.
    pub fn from_value(value: &Value) -> Result<Self> {
        let root = value
            .as_object()
            .ok_or_else(|| Error::bad("config.json", "expected a JSON object"))?;
        let text = match root.get("text_config") {
            None => None,
            Some(Value::Object(map)) => Some(map),
            Some(other) => {
                return Err(Error::bad(
                    "text_config",
                    format!("expected an object, got {}", describe(other)),
                ));
            }
        };

        let arch_string = arch_string(root)?;
        let num_hidden_layers = require_u32(root, text, "num_hidden_layers")?;
        let layer_types = parse_layer_types(require(root, text, "layer_types")?)?;
        if layer_types.len() != num_hidden_layers as usize {
            return Err(Error::bad(
                "layer_types",
                format!(
                    "length {} does not match num_hidden_layers {num_hidden_layers}",
                    layer_types.len()
                ),
            ));
        }

        let quant_value = root
            .get("quantization_config")
            .ok_or(Error::MissingField("quantization_config"))?;

        Ok(Self {
            arch_string,
            num_hidden_layers,
            hidden_size: require_u32(root, text, "hidden_size")?,
            intermediate_size: optional_u32(root, text, "intermediate_size")?,
            head_dim: require_u32(root, text, "head_dim")?,
            num_attention_heads: require_u32(root, text, "num_attention_heads")?,
            num_key_value_heads: require_u32(root, text, "num_key_value_heads")?,
            vocab_size: require_u32(root, text, "vocab_size")?,
            tie_word_embeddings: require_bool(root, text, "tie_word_embeddings")?,
            layer_types,
            linear_conv_kernel_dim: require_u32(root, text, "linear_conv_kernel_dim")?,
            linear_num_key_heads: require_u32(root, text, "linear_num_key_heads")?,
            linear_num_value_heads: require_u32(root, text, "linear_num_value_heads")?,
            linear_key_head_dim: require_u32(root, text, "linear_key_head_dim")?,
            linear_value_head_dim: require_u32(root, text, "linear_value_head_dim")?,
            attn_output_gate: require_bool(root, text, "attn_output_gate")?,
            num_experts: optional_u32(root, text, "num_experts")?,
            num_experts_per_tok: optional_u32(root, text, "num_experts_per_tok")?,
            moe_intermediate_size: optional_u32(root, text, "moe_intermediate_size")?,
            shared_expert_intermediate_size: optional_u32(
                root,
                text,
                "shared_expert_intermediate_size",
            )?,
            quantization: parse_quant(quant_value)?,
        })
    }
}

fn arch_string(root: &Map<String, Value>) -> Result<String> {
    let value = root
        .get("architectures")
        .ok_or(Error::MissingField("architectures"))?;
    let list = value.as_array().ok_or_else(|| {
        Error::bad(
            "architectures",
            format!("expected an array, got {}", describe(value)),
        )
    })?;
    let first = list
        .first()
        .ok_or_else(|| Error::bad("architectures", "expected a non-empty array"))?;
    let name = first.as_str().ok_or_else(|| {
        Error::bad(
            "architectures",
            format!("expected a string, got {}", describe(first)),
        )
    })?;
    if name.is_empty() {
        return Err(Error::bad("architectures", "architecture string is empty"));
    }
    Ok(name.to_string())
}

fn parse_layer_types(value: &Value) -> Result<Vec<LayerType>> {
    let list = value.as_array().ok_or_else(|| {
        Error::bad(
            "layer_types",
            format!("expected an array, got {}", describe(value)),
        )
    })?;
    list.iter()
        .map(|item| {
            let name = item.as_str().ok_or_else(|| {
                Error::bad(
                    "layer_types",
                    format!("expected strings, got {}", describe(item)),
                )
            })?;
            Ok(match name {
                "linear_attention" => LayerType::LinearAttention,
                "full_attention" => LayerType::FullAttention,
                other => LayerType::Other(other.to_string()),
            })
        })
        .collect()
}

fn parse_quant(value: &Value) -> Result<QuantizationConfig> {
    let obj = value.as_object().ok_or_else(|| {
        Error::bad(
            "quantization_config",
            format!("expected an object, got {}", describe(value)),
        )
    })?;
    let calibration = match obj.get("calibration") {
        None => None,
        Some(raw) => Some(parse_calibration(raw)?),
    };
    Ok(QuantizationConfig {
        quant_method: require_string(obj, "quant_method")?,
        bits: require_f64(obj, "bits")?,
        head_bits: require_u32_in(obj, "head_bits")?,
        codebook: optional_string(obj, "codebook")?,
        version: optional_string(obj, "version")?,
        vision_bits: optional_u32_in(obj, "vision_bits")?,
        mtp_bits: optional_u32_in(obj, "mtp_bits")?,
        out_scales: optional_string(obj, "out_scales")?,
        calibration,
    })
}

fn parse_calibration(value: &Value) -> Result<Calibration> {
    let obj = value.as_object().ok_or_else(|| {
        Error::bad(
            "calibration",
            format!("expected an object, got {}", describe(value)),
        )
    })?;
    Ok(Calibration {
        rows: require_u32_at(obj, "rows", "calibration.rows")?,
        cols: require_u32_at(obj, "cols", "calibration.cols")?,
    })
}

fn lookup<'a>(
    root: &'a Map<String, Value>,
    text: Option<&'a Map<String, Value>>,
    key: &'static str,
) -> Option<&'a Value> {
    if let Some(text) = text {
        if let Some(value) = text.get(key) {
            return Some(value);
        }
    }
    root.get(key)
}

fn require<'a>(
    root: &'a Map<String, Value>,
    text: Option<&'a Map<String, Value>>,
    key: &'static str,
) -> Result<&'a Value> {
    lookup(root, text, key).ok_or(Error::MissingField(key))
}

fn require_u32(
    root: &Map<String, Value>,
    text: Option<&Map<String, Value>>,
    key: &'static str,
) -> Result<u32> {
    expect_u32(require(root, text, key)?, key)
}

fn optional_u32(
    root: &Map<String, Value>,
    text: Option<&Map<String, Value>>,
    key: &'static str,
) -> Result<Option<u32>> {
    match lookup(root, text, key) {
        None => Ok(None),
        Some(value) => Ok(Some(expect_u32(value, key)?)),
    }
}

fn require_bool(
    root: &Map<String, Value>,
    text: Option<&Map<String, Value>>,
    key: &'static str,
) -> Result<bool> {
    let value = require(root, text, key)?;
    value
        .as_bool()
        .ok_or_else(|| Error::bad(key, format!("expected a bool, got {}", describe(value))))
}

fn require_u32_in(obj: &Map<String, Value>, key: &'static str) -> Result<u32> {
    require_u32_at(obj, key, key)
}

fn require_u32_at(obj: &Map<String, Value>, key: &str, field: &'static str) -> Result<u32> {
    let value = obj.get(key).ok_or(Error::MissingField(field))?;
    expect_u32(value, field)
}

fn optional_u32_in(obj: &Map<String, Value>, key: &'static str) -> Result<Option<u32>> {
    match obj.get(key) {
        None => Ok(None),
        Some(value) => Ok(Some(expect_u32(value, key)?)),
    }
}

fn require_string(obj: &Map<String, Value>, key: &'static str) -> Result<String> {
    let value = obj.get(key).ok_or(Error::MissingField(key))?;
    expect_string(value, key)
}

fn optional_string(obj: &Map<String, Value>, key: &'static str) -> Result<Option<String>> {
    match obj.get(key) {
        None => Ok(None),
        Some(value) => Ok(Some(expect_string(value, key)?)),
    }
}

fn require_f64(obj: &Map<String, Value>, key: &'static str) -> Result<f64> {
    let value = obj.get(key).ok_or(Error::MissingField(key))?;
    let number = value
        .as_f64()
        .ok_or_else(|| Error::bad(key, format!("expected a number, got {}", describe(value))))?;
    if !number.is_finite() || number < 0.0 {
        return Err(Error::bad(
            key,
            format!("expected a non-negative finite number, got {number}"),
        ));
    }
    Ok(number)
}

fn expect_string(value: &Value, field: &'static str) -> Result<String> {
    value
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| Error::bad(field, format!("expected a string, got {}", describe(value))))
}

/// Integers, or floats that are exactly an integer (`6.0`). Config writers emit both.
fn expect_u32(value: &Value, field: &'static str) -> Result<u32> {
    if let Some(number) = value.as_u64() {
        return u32::try_from(number)
            .map_err(|_| Error::bad(field, format!("integer {number} does not fit in u32")));
    }
    if let Some(number) = value.as_f64() {
        if number.is_finite()
            && number >= 0.0
            && number.fract() == 0.0
            && number <= f64::from(u32::MAX)
        {
            return Ok(number as u32);
        }
    }
    Err(Error::bad(
        field,
        format!("expected a non-negative integer, got {}", describe(value)),
    ))
}

fn describe(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(flag) => flag.to_string(),
        Value::Number(number) => number.to_string(),
        Value::String(text) if text.len() < 64 => format!("string {text:?}"),
        Value::String(_) => "a long string".to_string(),
        Value::Array(_) => "an array".to_string(),
        Value::Object(_) => "an object".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample() -> Value {
        json!({
            "architectures": ["Qwen3_5ForConditionalGeneration", "Ignored"],
            "unused_top": 1,
            "tie_word_embeddings": true,
            "text_config": {
                "num_hidden_layers": 2,
                "hidden_size": 32,
                "head_dim": 16,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 128,
                "tie_word_embeddings": false,
                "layer_types": ["linear_attention", "full_attention"],
                "linear_conv_kernel_dim": 4,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "attn_output_gate": true,
                "future_field": "ignored"
            },
            "quantization_config": {
                "quant_method": "exl3",
                "version": "1.5.0",
                "bits": 4.0,
                "head_bits": 6,
                "codebook": "mul1",
                "vision_bits": 6,
                "mtp_bits": 4,
                "out_scales": "always",
                "calibration": {"rows": 250, "cols": 2048},
                "original_quantization_config": {"quant_method": "other"}
            }
        })
    }

    #[test]
    fn reads_text_config_ahead_of_the_top_level() {
        let cfg = ModelConfig::from_value(&sample()).unwrap();
        assert_eq!(cfg.arch_string, "Qwen3_5ForConditionalGeneration");
        assert_eq!(cfg.num_hidden_layers, 2);
        assert_eq!(cfg.hidden_size, 32);
        assert_eq!(cfg.head_dim, 16);
        assert!(!cfg.tie_word_embeddings);
        assert_eq!(cfg.intermediate_size, None);
        assert_eq!(cfg.num_experts, None);
        assert_eq!(
            cfg.layer_types,
            vec![LayerType::LinearAttention, LayerType::FullAttention]
        );
        assert_eq!(cfg.linear_num_value_heads, 48);
        assert!(cfg.attn_output_gate);
        assert_eq!(cfg.quantization.quant_method, "exl3");
        assert_eq!(cfg.quantization.bits, 4.0);
        assert_eq!(cfg.quantization.head_bits, 6);
        assert_eq!(cfg.quantization.codebook.as_deref(), Some("mul1"));
        assert_eq!(cfg.quantization.vision_bits, Some(6));
        assert_eq!(
            cfg.quantization.calibration,
            Some(Calibration {
                rows: 250,
                cols: 2048
            })
        );
    }

    #[test]
    fn falls_back_to_the_top_level_per_field() {
        let mut value = sample();
        value["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("tie_word_embeddings");
        let cfg = ModelConfig::from_value(&value).unwrap();
        assert!(cfg.tie_word_embeddings);

        let mut bare = sample();
        let text = bare["text_config"].take();
        let root = bare.as_object_mut().unwrap();
        for (key, item) in text.as_object().unwrap() {
            root.insert(key.clone(), item.clone());
        }
        root.remove("text_config");
        let cfg = ModelConfig::from_value(&bare).unwrap();
        assert_eq!(cfg.hidden_size, 32);
        assert!(!cfg.tie_word_embeddings);
    }

    #[test]
    fn missing_required_field_names_the_field() {
        for key in [
            "head_dim",
            "attn_output_gate",
            "layer_types",
            "linear_conv_kernel_dim",
        ] {
            let mut value = sample();
            value["text_config"].as_object_mut().unwrap().remove(key);
            let err = ModelConfig::from_value(&value).unwrap_err();
            assert!(
                matches!(err, Error::MissingField(missing) if missing == key),
                "{key}: {err}"
            );
        }
        let mut value = sample();
        value.as_object_mut().unwrap().remove("quantization_config");
        let err = ModelConfig::from_value(&value).unwrap_err();
        assert!(err.to_string().contains("quantization_config"), "{err}");
    }

    #[test]
    fn optional_moe_and_other_layer_type() {
        let mut value = sample();
        let text = value["text_config"].as_object_mut().unwrap();
        text.insert("intermediate_size".into(), json!(17408));
        text.insert("num_experts".into(), json!(256));
        text.insert("num_experts_per_tok".into(), json!(8));
        text.insert("moe_intermediate_size".into(), json!(512));
        text.insert("shared_expert_intermediate_size".into(), json!(512));
        text["layer_types"] = json!(["linear_attention", "qsa"]);
        let cfg = ModelConfig::from_value(&value).unwrap();
        assert_eq!(cfg.intermediate_size, Some(17408));
        assert_eq!(cfg.num_experts, Some(256));
        assert_eq!(cfg.num_experts_per_tok, Some(8));
        assert_eq!(cfg.moe_intermediate_size, Some(512));
        assert_eq!(cfg.shared_expert_intermediate_size, Some(512));
        assert_eq!(cfg.layer_types[1], LayerType::Other("qsa".to_string()));
    }

    #[test]
    fn layer_types_must_match_the_layer_count() {
        let mut value = sample();
        value["text_config"]["layer_types"] = json!(["full_attention"]);
        let err = ModelConfig::from_value(&value).unwrap_err();
        assert!(err.to_string().contains("layer_types"), "{err}");
    }

    #[test]
    fn empty_architectures_names_the_field() {
        let mut value = sample();
        value["architectures"] = json!([]);
        let err = ModelConfig::from_value(&value).unwrap_err();
        assert!(err.to_string().contains("architectures"), "{err}");
    }

    #[test]
    fn integer_valued_float_counts_as_an_integer() {
        let mut value = sample();
        value["quantization_config"]["head_bits"] = json!(6.0);
        let cfg = ModelConfig::from_value(&value).unwrap();
        assert_eq!(cfg.quantization.head_bits, 6);
    }
}

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde_json::Value;

use crate::linear::group_linears;
use crate::safetensors::{read_shard, TensorIndex};
use crate::{Error, LinearExl3, ModelConfig, Result, TensorInfo};

/// `config.json` larger than this is rejected.
const MAX_CONFIG_BYTES: u64 = 32 * 1024 * 1024;

/// An EXL3 checkpoint: config, every tensor's location, and the grouped linears.
///
/// Opening a checkpoint reads `config.json`, `model.safetensors.index.json` when
/// the model is sharded, and the JSON header of each shard. Tensor bodies stay
/// on disk. `ngram_embedding.safetensors` and any other file that is not a model
/// shard is left alone.
#[derive(Clone, Debug)]
pub struct Checkpoint {
    /// Directory that was opened.
    pub directory: PathBuf,
    /// Parsed `config.json`.
    pub config: ModelConfig,
    /// Every tensor in the model shards.
    pub tensors: TensorIndex,
    linears: Vec<LinearExl3>,
    other: Vec<TensorInfo>,
}

impl Checkpoint {
    /// Open a checkpoint directory.
    ///
    /// A sharded model is the shards named by `model.safetensors.index.json`.
    /// An unsharded model is `model.safetensors`.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        if !dir.is_dir() {
            return Err(Error::bad(
                dir.display().to_string(),
                "not a checkpoint directory",
            ));
        }
        let config = read_config(&dir.join("config.json"))?;
        let (tensors, total_size) = read_tensors(dir)?;
        if let Some(expected) = total_size {
            let sum = tensors.iter().try_fold(0u64, |acc, (_, info)| {
                acc.checked_add(info.length).ok_or_else(|| {
                    Error::bad("metadata.total_size", "tensor byte sum overflows u64")
                })
            })?;
            if sum != expected {
                return Err(Error::bad(
                    "metadata.total_size",
                    format!("index says {expected}, tensor bytes sum to {sum}"),
                ));
            }
        }
        let (linears, other) = group_linears(&tensors)?;
        Ok(Self {
            directory: dir.to_path_buf(),
            config,
            tensors,
            linears,
            other,
        })
    }

    /// EXL3 linears, sorted by prefix.
    pub fn linears(&self) -> &[LinearExl3] {
        &self.linears
    }

    /// Tensors that are not part of an EXL3 linear: norms, embeddings, `conv1d`,
    /// `A_log`, `dt_bias`, and unquantized weights. Sorted by name.
    pub fn other_tensors(&self) -> &[TensorInfo] {
        &self.other
    }
}

fn read_config(path: &Path) -> Result<ModelConfig> {
    let len = fs::metadata(path)
        .map_err(|source| Error::read(path, source))?
        .len();
    if len > MAX_CONFIG_BYTES {
        return Err(Error::bad("config.json", format!("config is {len} bytes")));
    }
    let bytes = fs::read(path).map_err(|source| Error::read(path, source))?;
    let value: Value =
        serde_json::from_slice(&bytes).map_err(|source| Error::parse(path, source))?;
    ModelConfig::from_value(&value)
}

fn read_tensors(dir: &Path) -> Result<(TensorIndex, Option<u64>)> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.is_file() {
        let (weight_map, total_size) = read_weight_map(&index_path)?;
        return Ok((read_sharded(dir, &weight_map)?, total_size));
    }
    let single = dir.join("model.safetensors");
    if single.is_file() {
        let mut index = TensorIndex::default();
        for info in read_shard(&single)? {
            index.insert(info)?;
        }
        return Ok((index, None));
    }
    Err(Error::bad(
        "model.safetensors",
        "no model.safetensors.index.json and no model.safetensors",
    ))
}

fn read_weight_map(path: &Path) -> Result<(BTreeMap<String, String>, Option<u64>)> {
    let bytes = fs::read(path).map_err(|source| Error::read(path, source))?;
    let value: Value =
        serde_json::from_slice(&bytes).map_err(|source| Error::parse(path, source))?;
    let root = value
        .as_object()
        .ok_or_else(|| Error::bad("model.safetensors.index.json", "expected a JSON object"))?;
    let map_value = root
        .get("weight_map")
        .ok_or(Error::MissingField("weight_map"))?;
    let map = map_value
        .as_object()
        .ok_or_else(|| Error::bad("weight_map", "expected an object"))?;
    let mut weight_map = BTreeMap::new();
    for (name, shard) in map {
        let shard = shard
            .as_str()
            .ok_or_else(|| Error::bad(format!("weight_map.{name}"), "expected a shard filename"))?;
        if shard.is_empty() {
            return Err(Error::bad(format!("weight_map.{name}"), "empty shard path"));
        }
        weight_map.insert(name.clone(), shard.to_string());
    }
    let total_size = match root.get("metadata") {
        None => None,
        Some(Value::Object(meta)) => match meta.get("total_size") {
            None => None,
            Some(value) => Some(
                value
                    .as_u64()
                    .ok_or_else(|| Error::bad("metadata.total_size", "expected an integer"))?,
            ),
        },
        Some(_) => {
            return Err(Error::bad("metadata", "expected an object"));
        }
    };
    Ok((weight_map, total_size))
}

fn read_sharded(dir: &Path, weight_map: &BTreeMap<String, String>) -> Result<TensorIndex> {
    let mut by_shard: BTreeMap<PathBuf, BTreeSet<String>> = BTreeMap::new();
    for (name, shard) in weight_map {
        let path = shard_path(dir, shard)?;
        by_shard.entry(path).or_default().insert(name.clone());
    }
    let mut index = TensorIndex::default();
    for (path, expected) in &by_shard {
        let found = read_shard(path)?;
        let mut saw = HashSet::new();
        for info in found {
            if !expected.contains(&info.name) {
                return Err(Error::bad(
                    &info.name,
                    format!(
                        "present in {} but not in model.safetensors.index.json",
                        path.display()
                    ),
                ));
            }
            if !saw.insert(info.name.clone()) {
                return Err(Error::bad(&info.name, "duplicated inside one shard header"));
            }
            index.insert(info)?;
        }
        for name in expected {
            if !saw.contains(name) {
                return Err(Error::bad(
                    name,
                    format!(
                        "weight_map lists it in {} but the shard header does not",
                        path.display()
                    ),
                ));
            }
        }
    }
    Ok(index)
}

fn shard_path(dir: &Path, rel: &str) -> Result<PathBuf> {
    let rel_path = Path::new(rel);
    let escapes = rel_path.is_absolute()
        || rel_path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::Prefix(_)));
    if escapes {
        return Err(Error::bad(
            "weight_map",
            format!("shard `{rel}` is not a path inside the checkpoint directory"),
        ));
    }
    Ok(dir.join(rel_path))
}

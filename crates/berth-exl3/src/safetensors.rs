use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::dtype::parse_dtype;
use crate::{DType, Error, Result};

/// A header larger than this is rejected so a corrupt length cannot allocate.
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;

/// One tensor, located but not loaded.
///
/// `offset` is the absolute byte offset in `shard` (the 8-byte length, the
/// JSON header, then the safetensors `data_offsets` start). `length` is
/// `data_offsets[1] - data_offsets[0]` and equals `dtype.size * numel`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    /// Tensor name, as stored in the header.
    pub name: String,
    /// Shard file that holds the bytes.
    pub shard: PathBuf,
    /// Element type.
    pub dtype: DType,
    /// Shape. An empty shape is a scalar, whose element count is 1.
    pub shape: Vec<u64>,
    /// Absolute offset of the first byte in `shard`.
    pub offset: u64,
    /// Byte length of the tensor body.
    pub length: u64,
}

/// Every tensor in the checkpoint, keyed by name. Iteration is sorted by name.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TensorIndex {
    by_name: BTreeMap<String, TensorInfo>,
}

impl TensorIndex {
    /// Look up one tensor by its full name.
    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.by_name.get(name)
    }

    /// Whether `name` is in the index.
    pub fn contains(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    /// Whether the checkpoint recorded no tensors.
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    /// Names and records, sorted by name.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TensorInfo)> {
        self.by_name
            .iter()
            .map(|(name, info)| (name.as_str(), info))
    }

    pub(crate) fn insert(&mut self, info: TensorInfo) -> Result<()> {
        let name = info.name.clone();
        if self.by_name.insert(name.clone(), info).is_some() {
            return Err(Error::bad(name, "tensor name appears more than once"));
        }
        Ok(())
    }
}

/// Read one shard's header. Tensor bodies are not read.
pub(crate) fn read_shard(path: &Path) -> Result<Vec<TensorInfo>> {
    let mut file = File::open(path).map_err(|source| Error::read(path, source))?;
    let file_len = file
        .metadata()
        .map_err(|source| Error::read(path, source))?
        .len();
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|source| Error::read(path, source))?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len > MAX_HEADER_BYTES || header_len > file_len.saturating_sub(8) {
        return Err(Error::bad(
            path.display().to_string(),
            format!(
                "safetensors header length {header_len} is not usable in a {file_len}-byte file"
            ),
        ));
    }
    let header_usize = usize::try_from(header_len).map_err(|_| {
        Error::bad(
            path.display().to_string(),
            format!("safetensors header length {header_len} does not fit usize"),
        )
    })?;
    let mut header = vec![0u8; header_usize];
    file.read_exact(&mut header)
        .map_err(|source| Error::read(path, source))?;
    let value: Value =
        serde_json::from_slice(&header).map_err(|source| Error::parse(path, source))?;
    let obj = value.as_object().ok_or_else(|| {
        Error::bad(
            path.display().to_string(),
            "safetensors header is not a JSON object",
        )
    })?;
    let data_start = 8u64.checked_add(header_len).ok_or_else(|| {
        Error::bad(
            path.display().to_string(),
            "safetensors header length overflows u64",
        )
    })?;

    let mut tensors = Vec::new();
    for (name, entry) in obj {
        if name == "__metadata__" {
            continue;
        }
        tensors.push(parse_tensor(path, name, entry, data_start, file_len)?);
    }
    Ok(tensors)
}

fn parse_tensor(
    path: &Path,
    name: &str,
    entry: &Value,
    data_start: u64,
    file_len: u64,
) -> Result<TensorInfo> {
    let obj = entry.as_object().ok_or_else(|| {
        Error::bad(
            name,
            format!("expected a tensor object, got {}", json_kind(entry)),
        )
    })?;
    let dtype_value = obj
        .get("dtype")
        .ok_or_else(|| Error::bad(format!("{name}.dtype"), "missing dtype"))?;
    let dtype_name = dtype_value
        .as_str()
        .ok_or_else(|| Error::bad(format!("{name}.dtype"), "expected a string"))?;
    let dtype = parse_dtype(name, dtype_name)?;
    let shape_value = obj
        .get("shape")
        .ok_or_else(|| Error::bad(format!("{name}.shape"), "missing shape"))?;
    let shape = expect_shape(name, shape_value)?;
    let offsets = obj
        .get("data_offsets")
        .ok_or_else(|| Error::bad(format!("{name}.data_offsets"), "missing data_offsets"))?;
    let (start, end) = expect_offsets(name, offsets)?;
    let span = end - start;
    let numel = numel(name, &shape)?;
    let nbytes = numel
        .checked_mul(dtype.size())
        .ok_or_else(|| Error::bad(name, "byte length overflows u64"))?;
    if span != nbytes {
        return Err(Error::bad(
            name,
            format!("data_offsets span {span} does not match {dtype} shape byte length {nbytes}"),
        ));
    }
    let offset = data_start
        .checked_add(start)
        .ok_or_else(|| Error::bad(name, "byte offset overflows u64"))?;
    let end_abs = offset
        .checked_add(span)
        .ok_or_else(|| Error::bad(name, "tensor end overflows u64"))?;
    if end_abs > file_len {
        return Err(Error::bad(
            name,
            format!("tensor extends to byte {end_abs} but the shard is {file_len} bytes"),
        ));
    }
    Ok(TensorInfo {
        name: name.to_string(),
        shard: path.to_path_buf(),
        dtype,
        shape,
        offset,
        length: span,
    })
}

fn numel(name: &str, shape: &[u64]) -> Result<u64> {
    let mut count = 1u64;
    for &dim in shape {
        count = count
            .checked_mul(dim)
            .ok_or_else(|| Error::bad(name, "shape product overflows u64"))?;
    }
    Ok(count)
}

fn expect_shape(name: &str, value: &Value) -> Result<Vec<u64>> {
    let list = value.as_array().ok_or_else(|| {
        Error::bad(
            format!("{name}.shape"),
            format!("expected an array, got {}", json_kind(value)),
        )
    })?;
    list.iter()
        .map(|dim| expect_u64(&format!("{name}.shape"), dim))
        .collect()
}

fn expect_offsets(name: &str, value: &Value) -> Result<(u64, u64)> {
    let field = format!("{name}.data_offsets");
    let list = value.as_array().ok_or_else(|| {
        Error::bad(
            field.clone(),
            format!("expected an array, got {}", json_kind(value)),
        )
    })?;
    if list.len() != 2 {
        return Err(Error::bad(
            field,
            format!("expected 2 offsets, got {}", list.len()),
        ));
    }
    let start = expect_u64(&format!("{name}.data_offsets"), &list[0])?;
    let end = expect_u64(&format!("{name}.data_offsets"), &list[1])?;
    if end < start {
        return Err(Error::bad(
            format!("{name}.data_offsets"),
            format!("end {end} is before start {start}"),
        ));
    }
    Ok((start, end))
}

fn expect_u64(field: &str, value: &Value) -> Result<u64> {
    value.as_u64().ok_or_else(|| {
        Error::bad(
            field,
            format!("expected an integer, got {}", json_kind(value)),
        )
    })
}

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a bool",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

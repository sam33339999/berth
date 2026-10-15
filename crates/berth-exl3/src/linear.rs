use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use crate::signs::unpack_signs;
use crate::{DType, Error, Result, TensorIndex, TensorInfo};

/// Packed sign reads refuse anything larger than this. A real `su` / `sv` is
/// one bit per channel; this stops a mis-labeled trellis from being loaded.
const MAX_PACKED_SIGN_BYTES: u64 = 64 * 1024 * 1024;

const LINEAR_PARTS: &[&str] = &["trellis", "suh", "svh", "su", "sv", "bias", "mcg", "mul1"];

/// Which codebook a linear was quantized with.
///
/// Detected from the marker tensors, the same way exllamav3 does: `mul1` wins
/// over nothing, `mcg` likewise, and neither marker is the original
/// 3-instruction codebook. Both markers at once is an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codebook {
    /// `mcg` tensor is present.
    Mcg,
    /// `mul1` tensor is present.
    Mul1,
    /// Neither marker. The original 3-instruction codebook.
    Default,
}

/// Half-integer EXL3 rates. Only these three exist, and only with [`Codebook::Mul1`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HalfBitrate {
    /// 1.5 bits per weight.
    OnePointFive,
    /// 2.5 bits per weight.
    TwoPointFive,
    /// 3.5 bits per weight.
    ThreePointFive,
}

/// Bits per weight, read from `trellis.shape[-1]`.
///
/// An integer rate stores `16 * K` uint16s in the last dimension. A half-integer
/// rate stores eight extra words, so `K = last / 16` is `n + 0.5`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Bitrate {
    /// Integer rate, `1..=8`.
    Int(u8),
    /// Half-integer rate. See [`HalfBitrate`].
    Half(HalfBitrate),
}

impl Bitrate {
    /// The rate as a number. Integer rates are exact; half rates are `*.5`.
    pub fn as_f64(self) -> f64 {
        match self {
            Self::Int(k) => f64::from(k),
            Self::Half(HalfBitrate::OnePointFive) => 1.5,
            Self::Half(HalfBitrate::TwoPointFive) => 2.5,
            Self::Half(HalfBitrate::ThreePointFive) => 3.5,
        }
    }
}

impl fmt::Display for Bitrate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(k) => write!(f, "{k}"),
            Self::Half(half) => write!(f, "{half}"),
        }
    }
}

impl HalfBitrate {
    fn as_str(self) -> &'static str {
        match self {
            Self::OnePointFive => "1.5",
            Self::TwoPointFive => "2.5",
            Self::ThreePointFive => "3.5",
        }
    }
}

impl fmt::Display for HalfBitrate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Input-side (`su` / `suh`) or output-side (`sv` / `svh`) Hadamard signs.
///
/// When a checkpoint has both the packed and the unpacked tensor, the unpacked
/// fp16 vector is the one recorded here. That matches exllamav3, which uses
/// `suh` whenever it is present. The packed tensor stays in the tensor index
/// and is not listed as a non-linear tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HadamardScale {
    /// fp16 vector (`suh` or `svh`).
    Unpacked(TensorInfo),
    /// Packed int16 signs (`su` or `sv`).
    Packed(TensorInfo),
}

impl HadamardScale {
    /// The tensor this scale is stored in.
    pub fn tensor(&self) -> &TensorInfo {
        match self {
            Self::Unpacked(info) | Self::Packed(info) => info,
        }
    }

    /// Whether the checkpoint stored packed signs rather than fp16.
    pub fn is_packed(&self) -> bool {
        matches!(self, Self::Packed(_))
    }
}

/// One EXL3 linear, grouped by the tensor prefix.
///
/// `model.language_model.layers.3.self_attn.q_proj` is the prefix of
/// `...q_proj.trellis`, `...q_proj.suh`, and the rest of that linear.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearExl3 {
    /// Tensor prefix, without `.trellis`.
    pub prefix: String,
    /// int16 trellis, rank 3. The last dimension encodes `k`.
    pub trellis: TensorInfo,
    /// Input-side Hadamard signs.
    pub input_scale: HadamardScale,
    /// Output-side Hadamard signs.
    pub output_scale: HadamardScale,
    /// Optional bias. `F16`, `BF16`, and `F32` are accepted and the stored dtype
    /// is kept. Absent on language-model linears in the Qwen3.8-27B checkpoint.
    pub bias: Option<TensorInfo>,
    /// Codebook marker for this linear.
    pub codebook: Codebook,
    /// Bitrate of `trellis`.
    pub k: Bitrate,
}

impl LinearExl3 {
    /// Unpack packed `su` into fp16 ±1 bit patterns.
    ///
    /// Errors when this linear stores `suh` instead. Only the packed tensor's
    /// own byte range is read.
    pub fn unpack_input_signs(&self) -> Result<Vec<u16>> {
        unpack_scale(&self.input_scale, "su", "suh")
    }

    /// Unpack packed `sv` into fp16 ±1 bit patterns.
    ///
    /// Errors when this linear stores `svh` instead.
    pub fn unpack_output_signs(&self) -> Result<Vec<u16>> {
        unpack_scale(&self.output_scale, "sv", "svh")
    }
}

pub(crate) fn group_linears(index: &TensorIndex) -> Result<(Vec<LinearExl3>, Vec<TensorInfo>)> {
    let mut consumed = std::collections::HashSet::<String>::new();
    let mut linears = Vec::new();
    for (name, _) in index.iter() {
        let Some(prefix) = name.strip_suffix(".trellis") else {
            continue;
        };
        if prefix.is_empty() {
            return Err(Error::bad(
                name,
                "trellis tensor has an empty linear prefix",
            ));
        }
        linears.push(build_linear(index, prefix)?);
        for suffix in LINEAR_PARTS {
            let sibling = format!("{prefix}.{suffix}");
            if index.contains(&sibling) {
                consumed.insert(sibling);
            }
        }
    }
    linears.sort_by(|a, b| a.prefix.cmp(&b.prefix));
    let other = index
        .iter()
        .filter(|(name, _)| !consumed.contains(*name))
        .map(|(_, info)| info.clone())
        .collect();
    Ok((linears, other))
}

fn build_linear(index: &TensorIndex, prefix: &str) -> Result<LinearExl3> {
    let trellis = require_part(index, prefix, "trellis")?;
    if trellis.dtype != DType::I16 {
        return Err(Error::bad(
            &trellis.name,
            format!("expected I16, got {}", trellis.dtype),
        ));
    }
    if trellis.shape.len() != 3 {
        return Err(Error::bad(
            &trellis.name,
            format!("expected rank 3, got rank {}", trellis.shape.len()),
        ));
    }

    let mcg = index.get(&format!("{prefix}.mcg"));
    let mul1 = index.get(&format!("{prefix}.mul1"));
    let codebook = match (mcg, mul1) {
        (Some(_), Some(_)) => {
            return Err(Error::bad(
                format!("{prefix}.mcg"),
                "both mcg and mul1 are present",
            ));
        }
        (Some(_), None) => Codebook::Mcg,
        (None, Some(_)) => Codebook::Mul1,
        (None, None) => Codebook::Default,
    };
    let k = bitrate(&trellis.name, trellis.shape[2], codebook)?;

    let input_scale = scale(index, prefix, "suh", "su")?;
    let output_scale = scale(index, prefix, "svh", "sv")?;
    let bias = match index.get(&format!("{prefix}.bias")) {
        None => None,
        Some(info) => {
            if !matches!(info.dtype, DType::F16 | DType::Bf16 | DType::F32) {
                return Err(Error::bad(
                    &info.name,
                    format!("expected a float bias, got {}", info.dtype),
                ));
            }
            Some(info.clone())
        }
    };

    Ok(LinearExl3 {
        prefix: prefix.to_string(),
        trellis: trellis.clone(),
        input_scale,
        output_scale,
        bias,
        codebook,
        k,
    })
}

fn scale(
    index: &TensorIndex,
    prefix: &str,
    unpacked_suffix: &str,
    packed_suffix: &str,
) -> Result<HadamardScale> {
    let unpacked = index.get(&format!("{prefix}.{unpacked_suffix}"));
    let packed = index.get(&format!("{prefix}.{packed_suffix}"));
    if let Some(info) = unpacked {
        if info.dtype != DType::F16 {
            return Err(Error::bad(
                &info.name,
                format!("expected F16, got {}", info.dtype),
            ));
        }
    }
    if let Some(info) = packed {
        if info.dtype != DType::I16 {
            return Err(Error::bad(
                &info.name,
                format!("expected I16, got {}", info.dtype),
            ));
        }
    }
    match (unpacked, packed) {
        (Some(info), _) => Ok(HadamardScale::Unpacked(info.clone())),
        (None, Some(info)) => Ok(HadamardScale::Packed(info.clone())),
        (None, None) => Err(Error::bad(
            format!("{prefix}.{unpacked_suffix}"),
            format!("missing {unpacked_suffix} or {packed_suffix}"),
        )),
    }
}

fn require_part<'a>(index: &'a TensorIndex, prefix: &str, suffix: &str) -> Result<&'a TensorInfo> {
    let name = format!("{prefix}.{suffix}");
    index
        .get(&name)
        .ok_or_else(|| Error::bad(name, format!("missing {suffix}")))
}

/// Last dimension is `16 * K` words, or `16 * floor(K) + 8` for a half rate.
fn bitrate(tensor: &str, last: u64, codebook: Codebook) -> Result<Bitrate> {
    if last.is_multiple_of(16) {
        let k = last / 16;
        if (1..=8).contains(&k) {
            return Ok(Bitrate::Int(k as u8));
        }
        return Err(Error::bad(
            tensor,
            format!("integer bitrate {k} is outside 1..=8 (last dim {last})"),
        ));
    }
    if last % 16 == 8 {
        let floor = last / 16;
        let half = match floor {
            1 => Some(HalfBitrate::OnePointFive),
            2 => Some(HalfBitrate::TwoPointFive),
            3 => Some(HalfBitrate::ThreePointFive),
            _ => None,
        };
        let Some(half) = half else {
            return Err(Error::bad(
                tensor,
                format!(
                    "half-integer bitrate {floor}.5 is not one of 1.5, 2.5, 3.5 (last dim {last})"
                ),
            ));
        };
        if codebook != Codebook::Mul1 {
            return Err(Error::bad(
                tensor,
                format!("half-integer bitrate {half} requires mul1 (last dim {last})"),
            ));
        }
        return Ok(Bitrate::Half(half));
    }
    Err(Error::bad(
        tensor,
        format!("last dim {last} is not 16*K or 16*K+8"),
    ))
}

fn unpack_scale(scale: &HadamardScale, packed: &str, unpacked: &str) -> Result<Vec<u16>> {
    match scale {
        HadamardScale::Packed(info) => read_packed_signs(info),
        HadamardScale::Unpacked(info) => Err(Error::bad(
            &info.name,
            format!("scale is unpacked `{unpacked}` fp16, not packed `{packed}`"),
        )),
    }
}

fn read_packed_signs(info: &TensorInfo) -> Result<Vec<u16>> {
    if info.dtype != DType::I16 {
        return Err(Error::bad(
            &info.name,
            format!("expected I16, got {}", info.dtype),
        ));
    }
    if info.length > MAX_PACKED_SIGN_BYTES {
        return Err(Error::bad(
            &info.name,
            format!(
                "refusing to read {} bytes; packed sign reads are capped at {MAX_PACKED_SIGN_BYTES}",
                info.length
            ),
        ));
    }
    let len = usize::try_from(info.length)
        .map_err(|_| Error::bad(&info.name, "tensor length does not fit usize"))?;
    let mut file = File::open(&info.shard).map_err(|source| Error::read(&info.shard, source))?;
    file.seek(SeekFrom::Start(info.offset))
        .map_err(|source| Error::read(&info.shard, source))?;
    let mut bytes = vec![0u8; len];
    file.read_exact(&mut bytes)
        .map_err(|source| Error::read(&info.shard, source))?;
    unpack_signs(&bytes)
}

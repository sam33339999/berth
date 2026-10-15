use std::fmt;

use crate::{Error, Result};

/// Safetensors dtype. Element size is in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    U8,
    I8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F16,
    Bf16,
    F32,
    F64,
    F8E4M3,
    F8E5M2,
    F8E8M0,
}

impl DType {
    /// Parse a safetensors dtype string such as `"I16"` or `"BF16"`.
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "BOOL" => Self::Bool,
            "U8" => Self::U8,
            "I8" => Self::I8,
            "I16" => Self::I16,
            "U16" => Self::U16,
            "I32" => Self::I32,
            "U32" => Self::U32,
            "I64" => Self::I64,
            "U64" => Self::U64,
            "F16" => Self::F16,
            "BF16" => Self::Bf16,
            "F32" => Self::F32,
            "F64" => Self::F64,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E5M2" => Self::F8E5M2,
            "F8_E8M0" => Self::F8E8M0,
            _ => return None,
        })
    }

    /// The safetensors spelling of this dtype.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bool => "BOOL",
            Self::U8 => "U8",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::U16 => "U16",
            Self::I32 => "I32",
            Self::U32 => "U32",
            Self::I64 => "I64",
            Self::U64 => "U64",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::F8E4M3 => "F8_E4M3",
            Self::F8E5M2 => "F8_E5M2",
            Self::F8E8M0 => "F8_E8M0",
        }
    }

    /// Bytes per element. A scalar (rank 0) still occupies one element.
    pub fn size(self) -> u64 {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 | Self::F8E8M0 => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

pub(crate) fn parse_dtype(tensor: &str, name: &str) -> Result<DType> {
    DType::parse(name).ok_or_else(|| {
        Error::bad(
            format!("{tensor}.dtype"),
            format!("unknown safetensors dtype `{name}`"),
        )
    })
}

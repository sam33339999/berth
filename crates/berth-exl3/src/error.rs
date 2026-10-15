use std::path::PathBuf;

/// Failure while reading an EXL3 checkpoint.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A required JSON field is absent from both `text_config` and the top level.
    #[error("missing field `{0}`")]
    MissingField(&'static str),

    /// A field is present but has the wrong type, or a tensor fails a format check.
    /// `field` is the config key or the tensor name.
    #[error("field `{field}`: {detail}")]
    BadField { field: String, detail: String },

    /// Filesystem error, with the path that was opened.
    #[error("{path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// `config.json`, the shard index, or a safetensors header is not JSON.
    #[error("{path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

impl Error {
    pub(crate) fn bad(field: impl AsRef<str>, detail: impl Into<String>) -> Self {
        Self::BadField {
            field: field.as_ref().to_string(),
            detail: detail.into(),
        }
    }

    pub(crate) fn read(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Read {
            path: path.into(),
            source,
        }
    }

    pub(crate) fn parse(path: impl Into<PathBuf>, source: serde_json::Error) -> Self {
        Self::Parse {
            path: path.into(),
            source,
        }
    }
}

/// Result of an EXL3 read.
pub type Result<T> = std::result::Result<T, Error>;

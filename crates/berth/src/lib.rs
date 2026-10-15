//! OpenAI-compatible HTTP serving over a [`Backend`].
//!
//! The binary loads [`berth_engine::Engine`]. Tests pass a scripted backend and
//! never call the engine. Specs: `docs/cache.md`, `docs/repetition.md`.

#![forbid(unsafe_code)]

mod assets;
mod backend;
mod serve;
mod think;
mod turn;

pub use assets::{load_assets, Assets};
pub use backend::Backend;
pub use berth_engine::{Finish, Flow, GenerateRequest, SamplerConfig, Usage};
pub use serve::{serve, ServeConfig, ServeHandle};

/// Default `--model` directory (`scripts/serve-host.sh` `TARGET`).
pub const DEFAULT_MODEL_DIR: &str = "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3";

/// Default bind address (`scripts/serve-host.sh` `HOST`).
pub const DEFAULT_HOST: &str = "0.0.0.0";

/// Default bind port. Tests bind an ephemeral port instead.
pub const DEFAULT_PORT: u16 = 8889;

/// Default `--cache-tokens`: KV cache length (`CACHE_TOKENS` in `serve-host.sh`).
pub const DEFAULT_CACHE_TOKENS: u32 = 270336;

/// Default `--max-model-len` (`CTX` in `serve-host.sh`).
pub const DEFAULT_MAX_MODEL_LEN: u32 = 262144;

/// Default `--max-concurrent`. 0 admits every request; the GPU still runs one job.
pub const DEFAULT_MAX_CONCURRENT: u32 = 0;

/// Default `--cubins` directory.
pub const DEFAULT_CUBINS: &str = "target/triton-cubins";

/// `serve_openai.py` `--model-name`.
pub const DEFAULT_MODEL_NAME: &str = "qwen38-27b-exl3-dflash2";

/// Chat `max_tokens` when the client omits it (`MAX_TOKENS`).
pub const DEFAULT_CHAT_MAX_TOKENS: u32 = 32768;

/// Completion `max_tokens` when the client omits it (`serve_openai.py`).
pub const DEFAULT_COMPLETION_MAX_TOKENS: u32 = 256;

/// `--dry-multiplier` (`DRY_MULTIPLIER`). Same as [`SamplerConfig::serve_defaults`].
pub const DEFAULT_DRY_MULTIPLIER: f32 = 0.2;
/// `--dry-base` (`DRY_BASE`).
pub const DEFAULT_DRY_BASE: f32 = 1.75;
/// `--dry-allowed-length` (`DRY_ALLOWED_LENGTH`).
pub const DEFAULT_DRY_ALLOWED_LENGTH: u32 = 6;
/// `--dry-range` (`DRY_RANGE`). 0 means the whole context.
pub const DEFAULT_DRY_RANGE: u32 = 4096;
/// `--rep-penalty` (`REP_PENALTY`). 1.0 is off.
pub const DEFAULT_REP_PENALTY: f32 = 1.0;
/// `--freq-penalty` (`FREQ_PENALTY`). 0 is off.
pub const DEFAULT_FREQ_PENALTY: f32 = 0.0;
/// `--freq-range` (`FREQ_RANGE`). Window for rep, frequency, and presence.
pub const DEFAULT_FREQ_RANGE: u32 = 512;
/// `--presence-penalty` (`PRESENCE_PENALTY`). 0 is off.
pub const DEFAULT_PRESENCE_PENALTY: f32 = 0.0;

/// Request body cap (`MAX_BODY`).
pub const MAX_BODY: usize = 32 * 1024 * 1024;

/// Failure while loading model files or accepting a request before generation.
#[derive(Debug)]
pub struct Error {
    message: String,
}

impl Error {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

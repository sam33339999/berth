//! EXL3 checkpoint reader.
//!
//! Reads `config.json` and the `.safetensors` shards into a tensor index and
//! one [`LinearExl3`] per quantized linear: trellis, Hadamard scales (`suh` /
//! `svh`, or packed `su` / `sv`), optional bias, codebook (`mcg`, `mul1`, or
//! the default 3-instruction codebook), and bitrate `K`.
//!
//! Tensor bodies stay on disk. The only bytes read past a safetensors header
//! are packed `su` / `sv` sign words, and only when
//! [`LinearExl3::unpack_input_signs`] or [`LinearExl3::unpack_output_signs`]
//! is called. [`unpack_signs`] does that expansion on bytes the caller already
//! holds.
//!
//! Format: `docs/exl3.md`.

mod checkpoint;
mod config;
mod dtype;
mod error;
mod linear;
mod safetensors;
mod signs;

pub use checkpoint::Checkpoint;
pub use config::{Calibration, LayerType, ModelConfig, QuantizationConfig};
pub use dtype::DType;
pub use error::{Error, Result};
pub use linear::{Bitrate, Codebook, HadamardScale, HalfBitrate, LinearExl3};
pub use safetensors::{TensorIndex, TensorInfo};
pub use signs::{unpack_signs, FP16_NEG_ONE, FP16_ONE};

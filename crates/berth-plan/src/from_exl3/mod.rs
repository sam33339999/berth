//! Glue from `berth-exl3` to this crate's inputs.
//!
//! [`model_shape`] maps `config.json` onto [`crate::ModelShape`].
//! [`tensor_bytes`] sums checkpoint tensors into [`crate::TensorBytes`].

mod bytes;
mod shape;

pub use bytes::*;
pub use shape::*;

//! EXL3 checkpoint reader.
//!
//! Reads `config.json` and the `.safetensors` shards into per-Linear descriptors:
//! `trellis`, `suh` / `svh` (or packed `su` / `sv`), bias, codebook (mcg or mul1),
//! and the bitrate `K = trellis.shape[-1] / 16`. Pure Rust; no GPU.
//! Format: `docs/exl3.md`.

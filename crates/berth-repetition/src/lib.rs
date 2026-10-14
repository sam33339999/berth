//! Generation repetition check.
//!
//! Per-channel buffers of accepted tokens (`thinking`, `response`), the tail and
//! exact-repeat checks, trigger labels, and the 256-byte / 64-trigger caps.
//! Pure Rust; no GPU.
//! Spec: `docs/repetition.md`.

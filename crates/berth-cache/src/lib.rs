//! Paged prefix table.
//!
//! Full-page blake2b-128 chain hashes, reference counts, tree eviction by root
//! `access_serial`, and the keys for GDN checkpoints and rewind. Works on page
//! indices only and never touches GPU memory. Pure Rust; no GPU.
//! Spec: `docs/cache.md`.

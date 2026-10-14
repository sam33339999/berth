//! Layer plan and placement.
//!
//! Builds a `LayerPlan` per `arch_string` (mixer `Gdn | FullAttn | Qsa`, ffn
//! `DenseMlp | Moe`, extra `PleInject`) and decides where each tensor sits:
//! VRAM, pinned RAM or SSD. Pure Rust; no GPU.
//! Specs: `docs/runtime.md` (Schedule), `docs/memory.md`.

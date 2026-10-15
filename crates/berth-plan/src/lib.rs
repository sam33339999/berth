//! Layer plan and placement.
//!
//! Builds a [`ModelPlan`] per `arch_string` (mixer [`Mixer::Gdn`] |
//! [`Mixer::FullAttn`] | [`Mixer::Qsa`], ffn [`Ffn::DenseMlp`] | [`Ffn::Moe`],
//! extra [`PleInject`]) and decides where each tensor sits: VRAM, pinned RAM,
//! or SSD. Pure Rust; no GPU.
//!
//! Specs: `docs/runtime.md` (Schedule, 順序), `docs/memory.md`, `docs/models.md`.
//!
//! # Later stages
//!
//! * [`Mixer::Qsa`] and [`PleInject`] are in the schedule sketch. Only
//!   Flash-Next uses them, and that arch returns [`PlanError::NotYet`].
//! * [`Berth::Ssd`] is the Flash-Next n-gram table and cold-expert berth.
//!   [`place`] never assigns it.
//! * `docs/memory.md` keeps a hot-expert cache in VRAM after a spill. The
//!   cache size is not decided, so a spill moves every routed expert to
//!   [`Berth::PinnedRam`].
//! * Dense MLP intermediate width is not on [`ModelShape`]. [`Ffn::DenseMlp`]
//!   records the module kind; the checkpoint reader owns the width.

mod place;
mod plan;
mod shape;

pub use place::{
    is_routed_expert, place, Berth, BerthBytes, MachineBudget, PlaceError, PlacedTensor, Placement,
    TensorBytes, THIS_MACHINE_PINNED_RAM_BYTES, THIS_MACHINE_VRAM_BYTES,
};
pub use plan::{
    build_plan, Embedding, Ffn, FinalNorm, LayerPlan, LmHead, Mixer, ModelPlan, PlanError,
    RoutedExperts, Router, SharedExpert,
};
pub use shape::{
    GdnDims, LayerKind, ModelShape, MoeDims, PleInject, ShapeError, ARCH_MINICPM_V4_7,
    ARCH_QWEN3_5, ARCH_QWEN3_5_MOE, ARCH_QWEN4_EXP,
};

//! Forward pass and decode loop.
//!
//! Runs a `LayerPlan`, captures CUDA graphs, samples on the GPU, and hands each
//! accepted token to the repetition check. Draft accept / rewind lands here later.
//! Tests that touch the GPU run only with `BERTH_GPU_TESTS=1`.
//! Spec: `docs/runtime.md`.

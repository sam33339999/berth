//! CUDA driver layer.
//!
//! Device and pinned memory, streams, events, graphs whose kernel parameters can
//! be patched after capture (replacing exllamav3 `graph.cu`), cubin loading
//! (replacing `TritonKernel`), and per-device context (replacing `DevCtx`).
//! Tests that touch the GPU run only with `BERTH_GPU_TESTS=1`.
//! See `docs/kernels.md`.

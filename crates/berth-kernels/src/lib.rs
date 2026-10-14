//! exllamav3 kernel bindings.
//!
//! Raw-pointer shims over the kernels in the exllamav3 fork the live service
//! loads (`BERTH_EXL3_SRC`, commit `355c6ee`), compiled by nvcc, plus the decode
//! attention cubins built ahead of time by `scripts/build_triton_cubins.py`.
//! Tests that touch the GPU run only with `BERTH_GPU_TESTS=1`.
//! Kernel list: `docs/kernels.md`.

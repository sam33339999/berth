//! CUDA driver layer.
//!
//! Device and pinned memory, streams, events, graphs whose kernel parameters can
//! be patched after capture (replacing exllamav3 `graph.cu`), cubin loading
//! (replacing `TritonKernel`), and the primary context on device 0 (SM count
//! and compute capability from exllamav3 `DevCtx`).
//!
//! The GEMM lock buffer `DevCtx` also owns is not allocated here. Callers that
//! need it allocate their own [`DeviceBuffer`].
//!
//! Tests that touch the GPU run only with `BERTH_GPU_TESTS=1`.
//! See `docs/kernels.md`.

mod buffer;
mod context;
mod copy;
mod error;
mod event;
mod graph;
mod module;
mod pinned;
mod ptr;
mod raw;
mod stream;

pub use buffer::{ByteBuf, DeviceBuffer};
pub use context::{ComputeCapability, Context};
pub use copy::DeviceCopy;
pub use error::{Error, Result};
pub use event::Event;
pub use graph::{
    Capture, Graph, ParamSlot, ParamUpdate, MAX_PARAM_BYTES, MAX_PARAM_INDEX, PARAM_END,
};
pub use module::{Cubin, Dim3, Function};
pub use pinned::PinnedBuf;
pub use ptr::DevicePtr;
pub use stream::{Stream, StreamHandle};

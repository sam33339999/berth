//! Non-blocking streams, and the handle `berth-kernels` launches on.

use std::ffi::c_void;
use std::sync::Arc;

use cuda_core::CudaStream;

use crate::context::Context;
use crate::error::{Error, Result};
use crate::event::Event;
use crate::raw::{bits_to_stream, stream_bits};

/// Opaque driver stream handle.
///
/// The integer is the `CUstream` pointer. A driver `CUstream` and a runtime
/// `cudaStream_t` are the same handle: the runtime header typedefs both to
/// `struct CUstream_st *`. `berth-kernels` declares its nvcc launches as
/// `extern "C"` functions that take `cudaStream_t`; pass [`Self::as_ptr`] as
/// that argument.
///
/// Zero is the legacy default stream. Streams created by [`crate::Context::stream`]
/// are non-blocking and non-null.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StreamHandle(u64);

impl StreamHandle {
    /// Wraps a raw handle. Zero is the default stream.
    #[inline]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The handle as an integer.
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Pointer to pass as `cudaStream_t` / `CUstream`.
    #[inline]
    pub fn as_ptr(self) -> *mut c_void {
        bits_to_stream(self.0).cast()
    }

    /// `true` for the legacy default stream.
    #[inline]
    pub const fn is_default(self) -> bool {
        self.0 == 0
    }
}

/// A non-blocking stream on a [`Context`].
///
/// Cloning shares the same driver stream. The last clone destroys it. The
/// driver finishes work still queued on the stream before the handle goes away.
#[derive(Clone, Debug)]
pub struct Stream {
    pub(crate) inner: Arc<CudaStream>,
    pub(crate) ctx: Context,
}

impl Stream {
    /// Context this stream was created on.
    #[inline]
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// Raw handle for FFI. See [`StreamHandle`].
    #[inline]
    pub fn handle(&self) -> StreamHandle {
        StreamHandle(stream_bits(self.inner.cu_stream()))
    }

    /// Waits until work queued on this stream has finished.
    pub fn synchronize(&self) -> Result<()> {
        self.inner.synchronize().map_err(Error::driver)
    }

    /// Makes later work on this stream wait until `event` has completed.
    ///
    /// This is `cuStreamWaitEvent`. The event must belong to the same context.
    pub fn wait_event(&self, event: &Event) -> Result<()> {
        self.same_context(event.context())?;
        self.inner.wait(event.cuda()).map_err(Error::driver)
    }

    pub(crate) fn cu(&self) -> cuda_core::sys::CUstream {
        self.inner.cu_stream()
    }

    pub(crate) fn same_context(&self, ctx: &Context) -> Result<()> {
        if self.ctx.same(ctx) {
            Ok(())
        } else {
            Err(Error::ContextMismatch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_handle_is_a_null_cuda_stream() {
        let handle = StreamHandle::from_raw(0);
        assert!(handle.is_default());
        assert!(handle.as_ptr().is_null());
        assert_eq!(handle.as_u64(), 0);
    }

    #[test]
    fn handle_round_trip_keeps_the_pointer_bits() {
        let handle = StreamHandle::from_raw(0x1000);
        assert!(!handle.is_default());
        assert_eq!(handle.as_ptr() as usize, 0x1000);
        assert_eq!(handle.as_u64(), 0x1000);
    }
}

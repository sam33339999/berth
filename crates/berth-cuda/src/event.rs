//! Timing-enabled events.

use cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT;
use cuda_core::CudaEvent;

use crate::context::Context;
use crate::error::{Error, Result};
use crate::stream::Stream;

/// A CUDA event with timing enabled.
///
/// Record it on a stream, then either wait on it from another stream or read
/// [`Self::elapsed_ms`] between two recorded events.
pub struct Event {
    inner: CudaEvent,
    ctx: Context,
}

impl Event {
    pub(crate) fn new(ctx: &Context) -> Result<Self> {
        let inner = ctx
            .inner
            .new_event(Some(CUevent_flags_enum_CU_EVENT_DEFAULT))
            .map_err(Error::driver)?;
        Ok(Self {
            inner,
            ctx: ctx.clone(),
        })
    }

    /// Context this event was created on.
    #[inline]
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// Records the event after the work currently queued on `stream`.
    pub fn record(&self, stream: &Stream) -> Result<()> {
        stream.same_context(&self.ctx)?;
        self.inner.record(&stream.inner).map_err(Error::driver)
    }

    /// Waits until the recorded work has finished.
    pub fn synchronize(&self) -> Result<()> {
        self.inner.synchronize().map_err(Error::driver)
    }

    /// Milliseconds of device time from `self` to `end`.
    ///
    /// Both events are synchronized before the query, so the host waits for
    /// `end`. `self` must have been recorded at an earlier point in the same
    /// context than `end`.
    pub fn elapsed_ms(&self, end: &Event) -> Result<f32> {
        if !self.ctx.same(&end.ctx) {
            return Err(Error::ContextMismatch);
        }
        self.inner.elapsed_ms(&end.inner).map_err(Error::driver)
    }

    pub(crate) fn cuda(&self) -> &CudaEvent {
        &self.inner
    }
}

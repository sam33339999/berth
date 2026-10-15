//! Primary context on device 0.

use std::sync::Arc;

use cuda_core::CudaContext;

use crate::error::{Error, Result};
use crate::event::Event;
use crate::stream::Stream;

/// Compute capability `(major, minor)`, for example `(8, 6)` on an RTX 3090.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComputeCapability {
    /// Major version. Ampere is 8.
    pub major: i32,
    /// Minor version. `sm_86` is minor 6.
    pub minor: i32,
}

impl ComputeCapability {
    /// `major * 10 + minor`, so 8.6 is 86.
    ///
    /// Minor versions above 9 would collide with the next major. Shipping
    /// devices use a single minor digit.
    #[inline]
    pub const fn sm_version(self) -> i32 {
        self.major * 10 + self.minor
    }
}

impl std::fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// Retained primary context for device 0.
///
/// Cloning shares the same context. The last clone releases it.
#[derive(Clone, Debug)]
pub struct Context {
    pub(crate) inner: Arc<CudaContext>,
}

impl Context {
    /// Retains the primary context of device 0 and makes it current.
    pub fn new() -> Result<Self> {
        let inner = CudaContext::new(0).map_err(Error::driver)?;
        Ok(Self { inner })
    }

    /// Device ordinal. [`Self::new`] opens device 0.
    #[inline]
    pub fn ordinal(&self) -> usize {
        self.inner.ordinal()
    }

    /// Streaming multiprocessor count
    /// (`CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`).
    pub fn multiprocessor_count(&self) -> Result<u32> {
        self.inner.multiprocessor_count().map_err(Error::driver)
    }

    /// Device compute capability.
    pub fn compute_capability(&self) -> Result<ComputeCapability> {
        let (major, minor) = self.inner.compute_capability().map_err(Error::driver)?;
        Ok(ComputeCapability { major, minor })
    }

    /// A non-blocking stream on this context.
    pub fn stream(&self) -> Result<Stream> {
        let inner = self.inner.new_stream().map_err(Error::driver)?;
        Ok(Stream {
            inner,
            ctx: self.clone(),
        })
    }

    /// A timing-enabled event on this context.
    ///
    /// Timing is on (`CU_EVENT_DEFAULT`) so [`Event::elapsed_ms`] can read it.
    pub fn event(&self) -> Result<Event> {
        Event::new(self)
    }

    /// Waits until every stream on this context has finished.
    pub fn synchronize(&self) -> Result<()> {
        self.inner.synchronize().map_err(Error::driver)
    }

    pub(crate) fn bind(&self) -> Result<()> {
        self.inner.bind_to_thread().map_err(Error::driver)
    }

    pub(crate) fn same(&self, other: &Self) -> bool {
        self.inner.cu_ctx() == other.inner.cu_ctx()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sm_version_packs_major_and_minor() {
        let cc = ComputeCapability { major: 8, minor: 6 };
        assert_eq!(cc.sm_version(), 86);
        assert_eq!(cc.to_string(), "8.6");
    }
}

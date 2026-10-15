//! Device allocations.

use std::marker::PhantomData;

use cuda_core::simt::memory;

use crate::context::Context;
use crate::copy::DeviceCopy;
use crate::error::{Error, Result};
use crate::pinned::PinnedBuf;
use crate::ptr::DevicePtr;
use crate::raw::{bits_to_device_ptr, device_ptr_bits};
use crate::stream::Stream;

/// Device memory of `T` elements.
///
/// `T: DeviceCopy` so a byte copy is a valid value on both sides. [`ByteBuf`]
/// is the `u8` form.
///
/// Drop waits for this context, then frees the allocation. Recycling a buffer
/// is cheaper than dropping it on the decode path.
pub struct DeviceBuffer<T: DeviceCopy> {
    ptr: u64,
    len: usize,
    bytes: usize,
    ctx: Context,
    _marker: PhantomData<T>,
}

/// Device memory viewed as bytes.
pub type ByteBuf = DeviceBuffer<u8>;

impl<T: DeviceCopy> DeviceBuffer<T> {
    /// Allocates `len` elements. The contents are unspecified until something
    /// writes them.
    pub fn alloc(stream: &Stream, len: usize) -> Result<Self> {
        let bytes = byte_len::<T>(len)?;
        if bytes == 0 {
            return Ok(Self::empty(stream.context().clone(), len));
        }
        stream.context().bind()?;
        let ptr = unsafe { memory::malloc_sync(bytes) }.map_err(Error::driver)?;
        Ok(Self {
            ptr: device_ptr_bits(ptr),
            len,
            bytes,
            ctx: stream.context().clone(),
            _marker: PhantomData,
        })
    }

    /// Allocates `len` elements, fills them with zero bytes, and waits until
    /// that memset has finished on `stream`.
    pub fn zeroed(stream: &Stream, len: usize) -> Result<Self> {
        let buf = Self::alloc(stream, len)?;
        if buf.bytes == 0 {
            return Ok(buf);
        }
        buf.ctx.bind()?;
        unsafe { memory::memset_d8_async(bits_to_device_ptr(buf.ptr), 0, buf.bytes, stream.cu()) }
            .map_err(Error::driver)?;
        stream.synchronize()?;
        Ok(buf)
    }

    /// Allocates a buffer and copies `data` into it, waiting until the copy
    /// has finished on `stream`.
    pub fn from_host(stream: &Stream, data: &[T]) -> Result<Self> {
        let mut buf = Self::alloc(stream, data.len())?;
        buf.copy_from_host(stream, data)?;
        Ok(buf)
    }

    /// Element count.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` when [`Self::len`] is zero.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Size in bytes.
    #[inline]
    pub fn num_bytes(&self) -> usize {
        self.bytes
    }

    /// Device address of the first element. Null when the buffer is empty
    /// or `T` is zero-sized.
    #[inline]
    pub fn ptr(&self) -> DevicePtr {
        DevicePtr::from_u64(self.ptr)
    }

    /// Copies `src` into this buffer and waits until the copy has finished.
    pub fn copy_from_host(&mut self, stream: &Stream, src: &[T]) -> Result<()> {
        // SAFETY: `stream.synchronize` below finishes the copy before this
        // function returns, so the driver is done reading `src`.
        unsafe { self.copy_from_host_async(stream, src)? };
        if self.bytes == 0 {
            Ok(())
        } else {
            stream.synchronize()
        }
    }

    /// Enqueues a host-to-device copy on `stream` and returns before it finishes.
    ///
    /// Pageable host memory is staged by the driver. Overlap with other GPU
    /// work needs a [`PinnedBuf`].
    ///
    /// # Safety
    ///
    /// `src` must stay allocated and unmodified until the copy has finished
    /// on `stream` (`stream.synchronize()`, or an event recorded after the
    /// copy).
    pub unsafe fn copy_from_host_async(&mut self, stream: &Stream, src: &[T]) -> Result<()> {
        self.same_len(stream, src.len())?;
        if self.bytes == 0 {
            return Ok(());
        }
        self.ctx.bind()?;
        unsafe {
            memory::memcpy_htod_async(
                bits_to_device_ptr(self.ptr),
                src.as_ptr(),
                self.bytes,
                stream.cu(),
            )
        }
        .map_err(Error::driver)
    }

    /// Copies this buffer into `dst` and waits until the copy has finished.
    pub fn copy_to_host(&self, stream: &Stream, dst: &mut [T]) -> Result<()> {
        // SAFETY: synchronizing `stream` before returning means the driver is
        // done writing `dst`.
        unsafe { self.copy_to_host_async(stream, dst)? };
        if self.bytes == 0 {
            Ok(())
        } else {
            stream.synchronize()
        }
    }

    /// Enqueues a device-to-host copy on `stream` and returns before it finishes.
    ///
    /// # Safety
    ///
    /// `dst` must stay allocated and must not be read until the copy has
    /// finished on `stream`.
    pub unsafe fn copy_to_host_async(&self, stream: &Stream, dst: &mut [T]) -> Result<()> {
        self.same_len(stream, dst.len())?;
        if self.bytes == 0 {
            return Ok(());
        }
        self.ctx.bind()?;
        unsafe {
            memory::memcpy_dtoh_async(
                dst.as_mut_ptr(),
                bits_to_device_ptr(self.ptr),
                self.bytes,
                stream.cu(),
            )
        }
        .map_err(Error::driver)
    }

    /// Copies `src` into this buffer and waits until the copy has finished.
    pub fn copy_from_pinned(&mut self, stream: &Stream, src: &PinnedBuf<T>) -> Result<()> {
        // SAFETY: the stream is synchronized before this returns, so `src`
        // is not freed or mutated while the driver reads it.
        unsafe { self.copy_from_pinned_async(stream, src)? };
        if self.bytes == 0 {
            Ok(())
        } else {
            stream.synchronize()
        }
    }

    /// Enqueues a pinned-host to device copy on `stream`.
    ///
    /// # Safety
    ///
    /// `src` must not be dropped or mutated until the copy has finished on
    /// `stream`. Dropping it earlier frees the pages while the copy reads them.
    pub unsafe fn copy_from_pinned_async(
        &mut self,
        stream: &Stream,
        src: &PinnedBuf<T>,
    ) -> Result<()> {
        self.same_len(stream, src.len())?;
        stream.same_context(src.context())?;
        if self.bytes == 0 {
            return Ok(());
        }
        self.ctx.bind()?;
        unsafe {
            memory::memcpy_htod_async(
                bits_to_device_ptr(self.ptr),
                src.as_ptr(),
                self.bytes,
                stream.cu(),
            )
        }
        .map_err(Error::driver)
    }

    /// Copies this buffer into `dst` and waits until the copy has finished.
    pub fn copy_to_pinned(&self, stream: &Stream, dst: &mut PinnedBuf<T>) -> Result<()> {
        // SAFETY: the stream is synchronized before this returns.
        unsafe { self.copy_to_pinned_async(stream, dst)? };
        if self.bytes == 0 {
            Ok(())
        } else {
            stream.synchronize()
        }
    }

    /// Enqueues a device to pinned-host copy on `stream`.
    ///
    /// # Safety
    ///
    /// `dst` must not be dropped or read until the copy has finished on
    /// `stream`.
    pub unsafe fn copy_to_pinned_async(
        &self,
        stream: &Stream,
        dst: &mut PinnedBuf<T>,
    ) -> Result<()> {
        self.same_len(stream, dst.len())?;
        stream.same_context(dst.context())?;
        if self.bytes == 0 {
            return Ok(());
        }
        self.ctx.bind()?;
        unsafe {
            memory::memcpy_dtoh_async(
                dst.as_mut_ptr(),
                bits_to_device_ptr(self.ptr),
                self.bytes,
                stream.cu(),
            )
        }
        .map_err(Error::driver)
    }

    fn same_len(&self, stream: &Stream, host: usize) -> Result<()> {
        stream.same_context(&self.ctx)?;
        if host != self.len {
            Err(Error::Length {
                host,
                device: self.len,
            })
        } else {
            Ok(())
        }
    }

    fn empty(ctx: Context, len: usize) -> Self {
        Self {
            ptr: 0,
            len,
            bytes: 0,
            ctx,
            _marker: PhantomData,
        }
    }
}

impl<T: DeviceCopy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.bytes == 0 {
            return;
        }
        // `cuMemFree` on a pointer that still has queued work is a
        // use-after-free. Wait for the context first.
        let _ = self.ctx.synchronize();
        let _ = self.ctx.bind();
        let _ = unsafe { memory::free_sync(bits_to_device_ptr(self.ptr)) };
    }
}

pub(crate) fn byte_len<T>(len: usize) -> Result<usize> {
    len.checked_mul(std::mem::size_of::<T>())
        .ok_or(Error::TooLarge {
            len,
            elem: std::mem::size_of::<T>(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_len_rejects_overflow() {
        let err = byte_len::<u64>(usize::MAX).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("overflows"), "{text}");
    }

    #[test]
    fn byte_len_of_empty_is_zero() {
        assert_eq!(byte_len::<u32>(0).unwrap(), 0);
        assert_eq!(byte_len::<u8>(4).unwrap(), 4);
    }
}

//! Page-locked host memory.

use std::marker::PhantomData;
use std::ptr::NonNull;

use cuda_core::simt::memory;

use crate::buffer::byte_len;
use crate::context::Context;
use crate::copy::DeviceCopy;
use crate::error::{Error, Result};

/// Page-locked host memory (`cuMemAllocHost`).
///
/// Copies that should overlap other GPU work go through this, not through a
/// pageable slice. Drop waits for the context, then unlocks the pages.
pub struct PinnedBuf<T: DeviceCopy> {
    ptr: NonNull<T>,
    len: usize,
    bytes: usize,
    ctx: Context,
    _marker: PhantomData<T>,
}

// SAFETY: the allocation is host memory owned by this value. `T: DeviceCopy`
// is `Send + Sync`, and the pointer is not aliased once the buffer is.
unsafe impl<T: DeviceCopy> Send for PinnedBuf<T> {}
// SAFETY: shared access is a shared slice of `T: Sync`.
unsafe impl<T: DeviceCopy> Sync for PinnedBuf<T> {}

impl<T: DeviceCopy> PinnedBuf<T> {
    /// Allocates `len` zeroed elements.
    pub fn zeroed(ctx: &Context, len: usize) -> Result<Self> {
        let buf = Self::alloc(ctx, len)?;
        if buf.bytes != 0 {
            unsafe {
                std::ptr::write_bytes(buf.ptr.as_ptr().cast::<u8>(), 0, buf.bytes);
            }
        }
        Ok(buf)
    }

    /// Allocates a buffer and copies `data` into it.
    pub fn from_slice(ctx: &Context, data: &[T]) -> Result<Self> {
        let buf = Self::alloc(ctx, data.len())?;
        if !data.is_empty() && buf.bytes != 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), buf.ptr.as_ptr(), data.len());
            }
        }
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

    /// Context the pages were locked on.
    #[inline]
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// Host pointer. Dangling when the buffer occupies no bytes.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Mutable host pointer. Dangling when the buffer occupies no bytes.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// The locked pages as a slice.
    ///
    /// Reading this while a device-to-host copy into the buffer is still
    /// queued is a data race with the driver. The async copy methods say
    /// when that window ends.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// The locked pages as a mutable slice.
    ///
    /// Mutating this while a host-to-device copy from the buffer is still
    /// queued races the driver.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    fn alloc(ctx: &Context, len: usize) -> Result<Self> {
        let bytes = byte_len::<T>(len)?;
        if bytes == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len,
                bytes: 0,
                ctx: ctx.clone(),
                _marker: PhantomData,
            });
        }
        ctx.bind()?;
        let raw = unsafe { memory::malloc_host(bytes) }.map_err(Error::driver)?;
        let ptr = NonNull::new(raw.cast::<T>()).ok_or_else(|| Error::Driver {
            code: 1,
            detail: "cuMemAllocHost returned a null pointer".to_string(),
        })?;
        Ok(Self {
            ptr,
            len,
            bytes,
            ctx: ctx.clone(),
            _marker: PhantomData,
        })
    }
}

impl<T: DeviceCopy> Drop for PinnedBuf<T> {
    fn drop(&mut self) {
        if self.bytes == 0 {
            return;
        }
        let _ = self.ctx.synchronize();
        let _ = self.ctx.bind();
        let _ = unsafe { memory::free_host(self.ptr.as_ptr().cast()) };
    }
}

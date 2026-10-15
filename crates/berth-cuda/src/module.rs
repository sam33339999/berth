//! Cubin loading and `cuLaunchKernel`.
//!
//! This replaces exllamav3 `TritonKernel`: load an image from a path or from
//! bytes, look up a function by name, and launch it with a raw parameter list.

use std::ffi::c_void;
use std::path::Path;
use std::sync::Arc;

use cuda_core::simt::launch_kernel_on_stream;
use cuda_core::{CudaFunction, CudaModule};

use crate::context::Context;
use crate::error::{Error, Result};
use crate::stream::Stream;

/// Grid or block dimensions passed to `cuLaunchKernel`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dim3 {
    /// X dimension.
    pub x: u32,
    /// Y dimension.
    pub y: u32,
    /// Z dimension.
    pub z: u32,
}

impl Dim3 {
    /// A launch dimension of `(x, y, z)`.
    #[inline]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// A one-dimensional extent `(n, 1, 1)`.
    #[inline]
    pub const fn one(n: u32) -> Self {
        Self { x: n, y: 1, z: 1 }
    }
}

/// A cubin or fatbin loaded into a [`Context`].
///
/// Cloning shares the module. The last clone unloads it. [`Function`] handles
/// keep the module loaded while they are alive.
#[derive(Clone, Debug)]
pub struct Cubin {
    inner: Arc<CudaModule>,
    ctx: Context,
}

impl Cubin {
    /// Loads a cubin or fatbin from `path` (`cuModuleLoad`).
    ///
    /// The path must be valid UTF-8 and must not contain an interior `NUL`.
    pub fn load_file(ctx: &Context, path: &Path) -> Result<Self> {
        let path = utf8_path(path)?;
        let inner = ctx
            .inner
            .load_module_from_file(path)
            .map_err(Error::driver)?;
        Ok(Self {
            inner,
            ctx: ctx.clone(),
        })
    }

    /// Loads a cubin, fatbin, or PTX image from memory (`cuModuleLoadData`).
    pub fn load_bytes(ctx: &Context, image: &[u8]) -> Result<Self> {
        require_image(image)?;
        let inner = ctx
            .inner
            .load_module_from_image(image)
            .map_err(Error::driver)?;
        Ok(Self {
            inner,
            ctx: ctx.clone(),
        })
    }

    /// Context the module was loaded into.
    #[inline]
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// Looks up `name` (`cuModuleGetFunction`).
    ///
    /// `name` must not contain an interior `NUL`.
    pub fn function(&self, name: &str) -> Result<Function> {
        c_name(name)?;
        let inner = self.inner.load_function(name).map_err(Error::driver)?;
        Ok(Function {
            inner,
            ctx: self.ctx.clone(),
        })
    }
}

/// A kernel entry point in a loaded [`Cubin`].
///
/// Cloning is cheap. The module stays loaded while any clone is alive.
#[derive(Clone, Debug)]
pub struct Function {
    inner: CudaFunction,
    ctx: Context,
}

impl Function {
    /// Context the owning module was loaded into.
    #[inline]
    pub fn context(&self) -> &Context {
        &self.ctx
    }

    /// Launches the kernel on `stream` (`cuLaunchKernel`).
    ///
    /// `params` is the `kernelParams` array: one pointer per argument, each
    /// pointing at the argument's host bytes. A device pointer argument is the
    /// address of a `u64` holding the device address ([`crate::DevicePtr::as_u64`]),
    /// not the device address cast to a pointer. A scalar is the address of
    /// that scalar. The driver copies the bytes during this call, so the
    /// pointed-to values only need to live until `launch` returns.
    ///
    /// # Safety
    ///
    /// - `params` must match the kernel's signature: count, size, and alignment.
    /// - Grid and block dimensions must be legal for this device and kernel.
    /// - Device pointers inside the arguments must stay allocated until the
    ///   launch finishes on `stream`.
    pub unsafe fn launch(
        &self,
        stream: &Stream,
        grid: Dim3,
        block: Dim3,
        shared_mem_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<()> {
        stream.same_context(&self.ctx)?;
        unsafe {
            launch_kernel_on_stream(
                &self.inner,
                (grid.x, grid.y, grid.z),
                (block.x, block.y, block.z),
                shared_mem_bytes,
                &stream.inner,
                params,
            )
        }
        .map_err(Error::driver)
    }

    pub(crate) fn raw_function(&self) -> cuda_core::sys::CUfunction {
        // SAFETY: `CudaFunction` holds the module. The handle is valid while
        // `self` is alive, and the caller does not unload it.
        unsafe { self.inner.cu_function() }
    }
}

pub(crate) fn c_name(name: &str) -> Result<()> {
    if name.as_bytes().contains(&0) {
        Err(Error::InteriorNul)
    } else {
        Ok(())
    }
}

fn utf8_path(path: &Path) -> Result<&str> {
    let text = path
        .to_str()
        .ok_or_else(|| Error::PathNotUtf8(path.display().to_string()))?;
    c_name(text)?;
    Ok(text)
}

fn require_image(image: &[u8]) -> Result<()> {
    if image.is_empty() {
        Err(Error::EmptyImage)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_is_a_linear_extent() {
        assert_eq!(Dim3::one(4), Dim3::new(4, 1, 1));
    }

    #[test]
    fn interior_nul_in_a_name_is_rejected() {
        let err = c_name("fill\0i32").unwrap_err();
        assert!(matches!(err, Error::InteriorNul));
    }

    #[test]
    fn empty_image_is_rejected() {
        let err = require_image(&[]).unwrap_err();
        assert!(matches!(err, Error::EmptyImage));
        assert!(require_image(&[0x7f]).is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_path_is_rejected_before_the_driver() {
        use std::os::unix::ffi::OsStrExt;

        let path = Path::new(std::ffi::OsStr::from_bytes(b"cubin/\xff.bin"));
        let err = utf8_path(path).unwrap_err();
        assert!(matches!(err, Error::PathNotUtf8(_)));
    }
}

//! Conversions between driver handles and the integers this crate exposes.
//!
//! The integers stay on our side of the API. Callers never see `CUdeviceptr`
//! or `CUstream`.

use cuda_core::sys::{CUdeviceptr, CUstream};

#[inline]
pub(crate) fn device_ptr_bits(ptr: CUdeviceptr) -> u64 {
    ptr
}

#[inline]
pub(crate) fn bits_to_device_ptr(bits: u64) -> CUdeviceptr {
    bits
}

#[inline]
pub(crate) fn stream_bits(stream: CUstream) -> u64 {
    stream as usize as u64
}

#[inline]
pub(crate) fn bits_to_stream(bits: u64) -> CUstream {
    bits as usize as CUstream
}

/// `CUdevice` is `int` in the driver headers. Bindgen may emit that as
/// `c_int` already; the cast stays so a distinct enum type still converts.
#[inline]
pub(crate) fn device_as_c_int(device: cuda_core::sys::CUdevice) -> std::ffi::c_int {
    #[allow(clippy::unnecessary_cast)]
    let device = device as std::ffi::c_int;
    device
}

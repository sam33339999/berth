//! Device addresses passed to kernels.

use crate::copy::DeviceCopy;

/// Address of device memory, as the `u64` the driver calls `CUdeviceptr`.
///
/// Kernel launches take a pointer to this value (the `cuLaunchKernel` rule:
/// each entry of the parameter array points at the argument). [`Self::to_ne_bytes`]
/// is the same bits, for [`crate::Graph::set_params`].
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DevicePtr(u64);

impl DevicePtr {
    /// The null device address.
    pub const NULL: Self = Self(0);

    /// Wraps a raw device address.
    ///
    /// This does not check that `raw` is mapped. Passing it to a kernel is
    /// only valid when it came from [`crate::DeviceBuffer::ptr`] or is null.
    #[inline]
    pub const fn from_u64(raw: u64) -> Self {
        Self(raw)
    }

    /// The address as an integer.
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Native-endian bytes of the address, suitable as an 8-byte kernel argument.
    #[inline]
    pub const fn to_ne_bytes(self) -> [u8; 8] {
        self.0.to_ne_bytes()
    }

    /// `true` when the address is zero.
    #[inline]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }
}

impl From<DevicePtr> for u64 {
    fn from(ptr: DevicePtr) -> Self {
        ptr.0
    }
}

// SAFETY: `DevicePtr` is a transparent `u64`. Every bit pattern is a valid
// address value, and the all-zero pattern is the null device pointer.
unsafe impl DeviceCopy for DevicePtr {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_pointer_is_zero() {
        assert!(DevicePtr::NULL.is_null());
        assert_eq!(DevicePtr::NULL.as_u64(), 0);
        assert_eq!(u64::from(DevicePtr::NULL), 0);
    }

    #[test]
    fn round_trip_preserves_bits() {
        let ptr = DevicePtr::from_u64(0x0000_7f00_1234_0000);
        assert_eq!(ptr.as_u64(), 0x0000_7f00_1234_0000);
        assert_eq!(ptr.to_ne_bytes(), 0x0000_7f00_1234_0000u64.to_ne_bytes());
        assert!(!ptr.is_null());
    }
}

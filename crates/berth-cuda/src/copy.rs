//! Values that may cross the host/device boundary as raw bytes.

/// Plain data that a memcpy may move between host memory and device memory.
///
/// # Safety
///
/// Implement this only when every bit pattern of `size_of::<Self>()` bytes is
/// a valid value of `Self`, including the all-zero pattern. The type must not
/// own heap memory, and its drop glue must be a no-op. `bool`, `char`, and
/// `NonZero*` fail the bit-pattern rule.
pub unsafe trait DeviceCopy: Copy + Send + Sync + 'static {}

macro_rules! impl_device_copy {
    ($($ty:ty),+ $(,)?) => {
        $(
            unsafe impl DeviceCopy for $ty {}
        )+
    };
}

impl_device_copy!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f32, f64,);

unsafe impl<T: DeviceCopy, const N: usize> DeviceCopy for [T; N] {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DevicePtr;

    fn assert_device_copy<T: DeviceCopy>() {}

    #[test]
    fn numeric_arrays_and_device_ptr_are_device_copy() {
        assert_device_copy::<u8>();
        assert_device_copy::<i32>();
        assert_device_copy::<f32>();
        assert_device_copy::<u64>();
        assert_device_copy::<DevicePtr>();
        assert_device_copy::<[u16; 4]>();
    }
}

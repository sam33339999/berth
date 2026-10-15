//! C ABI of the nvcc static library. Pointers are device addresses.
//! `stream` is a `cudaStream_t` (`null` is the default stream).

use std::ffi::{c_char, c_int, c_void};

unsafe extern "C" {
    pub fn berth_exl3_linear(
        a: *const c_void,
        trellis: *const c_void,
        c: *mut c_void,
        suh: *const c_void,
        a_had: *mut c_void,
        svh: *const c_void,
        size_m: c_int,
        size_k: c_int,
        size_n: c_int,
        k_bits: c_int,
        codebook: c_int,
        out_fp32: c_int,
        stream: *mut c_void,
    ) -> c_int;

    pub fn berth_exl3_last_error() -> *const c_char;
    pub fn berth_exl3_last_launch() -> *const c_char;
}

#[cfg(test)]
unsafe extern "C" {
    pub fn berth_exl3_smoke_int8(out_fp32: c_int) -> c_int;
    pub fn berth_exl3_smoke_gemm() -> c_int;

    /// Unsalted coop-autotune key, including `max(size_m, 2)`. No CUDA call.
    pub fn berth_exl3_gemm_tune_key(
        size_m: c_int,
        size_k: c_int,
        size_n: c_int,
        k_bits: c_int,
        out_fp32: c_int,
        device: c_int,
        cc: c_int,
        sms: c_int,
        codebook: c_int,
    ) -> u64;

    /// `coop_autotune.cu` version salt (version 4). No CUDA call.
    pub fn berth_exl3_salt_hash(hash: u64) -> u64;
}

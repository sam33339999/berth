//! GPU sampling kernels: repetition penalties, DRY, Gumbel noise, and the
//! fused argmax the serve's `CustomSampler` collapses onto.
//!
//! Device code comes from the exllamav3 fork at build time (`FORK_EXTRACTS`
//! in `build.rs`). Pointers are device addresses. `stream` is a
//! `cudaStream_t` (`null` is the default stream). Host checks return
//! [`Error::Launch`] before any CUDA call. [`last_launch`] and [`last_error`]
//! are this thread's sampler launches, separate from the ops and the
//! single-linear shim.
//!
//! Batch size is 1. Logit rows are contiguous. Penalty outputs are fp32.
//! `past_ids` is `int64`, length `past_len`, and must be non-null even when
//! `past_len` is 0.

use std::ffi::{c_void, CStr};

use crate::{Error, Result};

mod ffi {
    use std::ffi::{c_char, c_int, c_uint, c_void};

    unsafe extern "C" {
        pub fn berth_sample_apply_rep_pens(
            in_logits: *const c_void,
            out_logits: *mut c_void,
            past_ids: *const c_void,
            past_len: c_int,
            vocab: c_int,
            input_fp16: c_int,
            rep_p: f32,
            sustain_range: c_int,
            decay_range: c_int,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_sample_apply_pres_freq_pens(
            in_logits: *const c_void,
            out_logits: *mut c_void,
            past_ids: *const c_void,
            past_len: c_int,
            vocab: c_int,
            input_fp16: c_int,
            pres_p: f32,
            freq_p: f32,
            sustain_range: c_int,
            decay_range: c_int,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_sample_dry_penalty(
            in_logits: *const c_void,
            out_logits: *mut c_void,
            past_ids: *const c_void,
            breakers: *const c_void,
            scratch: *mut c_void,
            past_len: c_int,
            vocab: c_int,
            input_fp16: c_int,
            multiplier: f32,
            base: f32,
            allowed_length: c_int,
            range: c_int,
            max_exponent: c_int,
            match_cap: c_int,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_sample_gumbel_f16(
            in_logits: *const c_void,
            out_logits: *mut c_void,
            size: c_int,
            random: c_uint,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_sample_gumbel_f32(
            in_logits: *const c_void,
            out_logits: *mut c_void,
            size: c_int,
            random: c_uint,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_sample_fused(
            logits: *const c_void,
            out_token: *mut c_void,
            workspace: *mut c_void,
            dim: c_int,
            size: c_int,
            input_fp16: c_int,
            inv_temp: f32,
            random: c_uint,
            mode: c_int,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_sample_mask_tail_f16(
            logits: *mut c_void,
            limit: c_int,
            vocab: c_int,
            stream: *mut c_void,
        ) -> c_int;

        pub fn berth_sample_last_error() -> *const c_char;
        pub fn berth_sample_last_launch() -> *const c_char;
    }
}

fn c_string(ptr: *const std::ffi::c_char) -> String {
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn error_message(rc: i32) -> String {
    let message = last_error();
    if message.is_empty() {
        format!("sample launch failed ({rc})")
    } else {
        message
    }
}

fn check(rc: i32) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(Error::Launch(error_message(rc)))
    }
}

fn flag(input_fp16: bool) -> i32 {
    i32::from(input_fp16)
}

/// Kernel the last successful sampler launch on this thread ran.
///
/// Empty after a failed launch, and before any launch.
pub fn last_launch() -> String {
    c_string(unsafe { ffi::berth_sample_last_launch() })
}

/// Message from the last failed sampler launch on this thread.
///
/// Empty after a successful launch.
pub fn last_error() -> String {
    c_string(unsafe { ffi::berth_sample_last_error() })
}

/// `apply_rep_pens`. `out_logits` is fp32 and may alias `in_logits` when the
/// input is already fp32. `input_fp16` selects the `<true>` instance.
///
/// # Safety
///
/// Every pointer except a null `stream` must address device memory that stays
/// live until `stream` finishes the launch. `in_logits` holds `vocab`
/// elements of fp16 or fp32. `out_logits` holds `vocab` fp32s. `past_ids`
/// holds `past_len` int64s.
#[allow(clippy::too_many_arguments)]
pub unsafe fn apply_rep_pens(
    in_logits: *const c_void,
    out_logits: *mut c_void,
    past_ids: *const c_void,
    past_len: i32,
    vocab: i32,
    input_fp16: bool,
    rep_p: f32,
    sustain_range: i32,
    decay_range: i32,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe {
        ffi::berth_sample_apply_rep_pens(
            in_logits,
            out_logits,
            past_ids,
            past_len,
            vocab,
            flag(input_fp16),
            rep_p,
            sustain_range,
            decay_range,
            stream,
        )
    };
    check(rc)
}

/// `apply_pres_freq_pens`. Same buffer rules as [`apply_rep_pens`].
///
/// # Safety
///
/// Same as [`apply_rep_pens`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn apply_pres_freq_pens(
    in_logits: *const c_void,
    out_logits: *mut c_void,
    past_ids: *const c_void,
    past_len: i32,
    vocab: i32,
    input_fp16: bool,
    pres_p: f32,
    freq_p: f32,
    sustain_range: i32,
    decay_range: i32,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe {
        ffi::berth_sample_apply_pres_freq_pens(
            in_logits,
            out_logits,
            past_ids,
            past_len,
            vocab,
            flag(input_fp16),
            pres_p,
            freq_p,
            sustain_range,
            decay_range,
            stream,
        )
    };
    check(rc)
}

/// `dry_penalty`. `scratch` is `(vocab + 1)` int32s; the launcher fills it
/// with `-1`. `breakers` is a `vocab`-long bool mask, or null for no breakers.
///
/// # Safety
///
/// Same lifetime rule as [`apply_rep_pens`]. `scratch` must hold `vocab + 1`
/// int32s. A non-null `breakers` must hold `vocab` bytes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn dry_penalty(
    in_logits: *const c_void,
    out_logits: *mut c_void,
    past_ids: *const c_void,
    breakers: *const c_void,
    scratch: *mut c_void,
    past_len: i32,
    vocab: i32,
    input_fp16: bool,
    multiplier: f32,
    base: f32,
    allowed_length: i32,
    range: i32,
    max_exponent: i32,
    match_cap: i32,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe {
        ffi::berth_sample_dry_penalty(
            in_logits,
            out_logits,
            past_ids,
            breakers,
            scratch,
            past_len,
            vocab,
            flag(input_fp16),
            multiplier,
            base,
            allowed_length,
            range,
            max_exponent,
            match_cap,
            stream,
        )
    };
    check(rc)
}

/// `gumbel_noise_f16`. `out_logits` is fp16 and must not be odd-sized at an
/// unaligned address; the kernel reads `half2` pairs. `random` is the Philox
/// key (`randint` from the job).
///
/// # Safety
///
/// `in_logits` and `out_logits` hold `size` fp16 elements and stay live until
/// `stream` finishes. They may alias.
pub unsafe fn gumbel_f16(
    in_logits: *const c_void,
    out_logits: *mut c_void,
    size: i32,
    random: u32,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe { ffi::berth_sample_gumbel_f16(in_logits, out_logits, size, random, stream) };
    check(rc)
}

/// `gumbel_noise_f32`. `out_logits` may alias `in_logits`.
///
/// # Safety
///
/// Both buffers hold `size` fp32 elements and stay live until `stream` finishes.
pub unsafe fn gumbel_f32(
    in_logits: *const c_void,
    out_logits: *mut c_void,
    size: i32,
    random: u32,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe { ffi::berth_sample_gumbel_f32(in_logits, out_logits, size, random, stream) };
    check(rc)
}

/// Fused argmax. `mode` 0 is greedy. `mode` 1 is `argmax(logit * inv_temp + gumbel)`.
///
/// `out_token` receives one `uint64` index. `workspace` holds at least
/// `128 * 3` fp32s. `size` is the scan bound and must be in `1..=dim`.
///
/// # Safety
///
/// `logits` holds `dim` elements of fp16 or fp32. `out_token` holds one
/// `uint64`. `workspace` holds at least 1536 bytes and is 8-byte aligned
/// (a CUDA allocation is). All stay live until `stream` finishes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused(
    logits: *const c_void,
    out_token: *mut c_void,
    workspace: *mut c_void,
    dim: i32,
    size: i32,
    input_fp16: bool,
    inv_temp: f32,
    random: u32,
    mode: i32,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe {
        ffi::berth_sample_fused(
            logits,
            out_token,
            workspace,
            dim,
            size,
            flag(input_fp16),
            inv_temp,
            random,
            mode,
            stream,
        )
    };
    check(rc)
}

/// Write fp16 `-inf` (`0xFC00`) into `logits[limit..vocab]`.
///
/// `limit >= vocab` does not launch. This is the tail fill
/// `CustomSampler.forward` does when `actual_vocab_size` is below the logit
/// width.
///
/// # Safety
///
/// `logits` holds `vocab` fp16 elements and stays live until `stream` finishes.
pub unsafe fn mask_tail_f16(
    logits: *mut c_void,
    limit: i32,
    vocab: i32,
    stream: *mut c_void,
) -> Result<()> {
    let rc = unsafe { ffi::berth_sample_mask_tail_f16(logits, limit, vocab, stream) };
    check(rc)
}

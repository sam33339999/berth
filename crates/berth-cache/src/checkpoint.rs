//! Where a GDN checkpoint is allowed to sit.
//!
//! Intervals match the live service: 2048 tokens while generating and near the
//! end of the prompt (`default_recurrent_checkpoint_interval` for Qwen3.8),
//! 32768 during earlier prefill (`recurrent_checkpoint_interval_pp`), and the
//! prompt's last full page boundary (job passes `PAGE_SIZE` as the interval).
//! Every interval is a multiple of [`PAGE_SIZE`](crate::PAGE_SIZE). Positions
//! are measured from the end of the cached prefix so a restored checkpoint
//! lines up with the KV pages.

use crate::{Error, PAGE_SIZE};

/// Generate / near-tail interval. Qwen3.8 `default_recurrent_checkpoint_interval`.
pub const DEFAULT_RECURRENT_CHECKPOINT_INTERVAL: usize = 2048;

/// Prefill interval, before it is ceiled to the chunk size.
pub const DEFAULT_RECURRENT_CHECKPOINT_INTERVAL_PP: usize = 32768;

/// Interval used for the prompt's last full page (`maybe_stash_recurrent(..., PAGE_SIZE)`).
pub const LAST_PAGE_CHECKPOINT_INTERVAL: usize = PAGE_SIZE;

/// Checkpoint spacing for one model.
///
/// `interval_pp` is `ceil_span(requested_pp, max_chunk_size)`, matching
/// `Generator.__init__`. The 4 GiB recurrent-cache byte cap is not applied
/// here: this crate stores hashes, not checkpoint tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointPolicy {
    /// Spacing near generation and in the tail of the prompt.
    pub interval: usize,
    /// Spacing during earlier prefill.
    pub interval_pp: usize,
    /// Prefill chunk size. The tail switch is two chunks before the prompt end.
    pub max_chunk_size: usize,
}

impl CheckpointPolicy {
    /// Build a policy. All three values must be positive multiples of [`PAGE_SIZE`](crate::PAGE_SIZE).
    pub fn new(interval: usize, interval_pp: usize, max_chunk_size: usize) -> Result<Self, Error> {
        page_multiple("recurrent_checkpoint_interval", interval)?;
        page_multiple("recurrent_checkpoint_interval_pp", interval_pp)?;
        page_multiple("max_chunk_size", max_chunk_size)?;
        Ok(Self {
            interval,
            interval_pp: ceil_span(interval_pp, max_chunk_size),
            max_chunk_size,
        })
    }

    /// Qwen3.8 defaults: 2048 / 32768, with the prefill interval ceiled to `max_chunk_size`.
    ///
    /// The serve process does not override these. Its default chunk size is
    /// 2048, which already divides 32768, so the prefill interval stays 32768.
    pub fn qwen3_8(max_chunk_size: usize) -> Result<Self, Error> {
        Self::new(
            DEFAULT_RECURRENT_CHECKPOINT_INTERVAL,
            DEFAULT_RECURRENT_CHECKPOINT_INTERVAL_PP,
            max_chunk_size,
        )
    }
}

fn page_multiple(what: &'static str, value: usize) -> Result<(), Error> {
    // Zero is a multiple of PAGE_SIZE, and it is still not a usable interval.
    if value == 0 || !value.is_multiple_of(PAGE_SIZE) {
        Err(Error::NotPageMultiple { what, value })
    } else {
        Ok(())
    }
}

/// Smallest multiple of `b` that is >= `a`. `b` is non-zero.
fn ceil_span(a: usize, b: usize) -> usize {
    a.div_ceil(b) * b
}

/// Python `Job.is_checkpoint_boundary`.
///
/// `prompt_len` is `len(sequence_ids) - 1`. `seq_pos` is `kv_position`.
/// `override_interval` is the explicit interval (`PAGE_SIZE` for the last
/// full page). Otherwise a position within two chunks of the prompt end uses
/// [`CheckpointPolicy::interval`]; an earlier position uses
/// [`CheckpointPolicy::interval_pp`]. The offset is
/// `seq_pos - cached_pages * PAGE_SIZE`, and Python's `%` (floor modulo,
/// including a negative offset) is `rem_euclid`.
pub fn is_checkpoint_boundary(
    seq_pos: usize,
    prompt_len: usize,
    cached_pages: usize,
    policy: &CheckpointPolicy,
    override_interval: Option<usize>,
) -> bool {
    if let Some(interval) = override_interval {
        // `is_multiple_of(0)` is false, matching a rejected override.
        return seq_pos.is_multiple_of(interval);
    }
    let tail = prompt_len as i128 - (policy.max_chunk_size as i128) * 2;
    let interval = if seq_pos as i128 >= tail {
        policy.interval
    } else {
        policy.interval_pp
    };
    let offset = seq_pos as i128 - (cached_pages as i128) * (PAGE_SIZE as i128);
    offset.rem_euclid(interval as i128) == 0
}

/// Whether `maybe_stash_recurrent` would store a checkpoint.
///
/// Position 0 is never stashed. A repeat of `last_checkpoint_pos` is not
/// stashed again. The position must also be page-aligned; Python asserts that
/// once the boundary test has passed.
pub fn should_stash_checkpoint(
    seq_pos: usize,
    prompt_len: usize,
    cached_pages: usize,
    policy: &CheckpointPolicy,
    override_interval: Option<usize>,
    last_checkpoint_pos: Option<usize>,
) -> bool {
    // Position 0 is page-aligned and still must not be stashed.
    if seq_pos == 0 || !seq_pos.is_multiple_of(PAGE_SIZE) {
        return false;
    }
    if last_checkpoint_pos == Some(seq_pos) {
        return false;
    }
    is_checkpoint_boundary(seq_pos, prompt_len, cached_pages, policy, override_interval)
}

/// Token position of the prompt's last full page boundary.
///
/// `seq_len` is `len(sequence_ids)`. The prompt body is `seq_len - 1` tokens
/// (`seqlen // PAGE_SIZE * PAGE_SIZE` in `Job.prefill`).
pub fn last_full_page_boundary(seq_len: usize) -> usize {
    let seqlen = seq_len.saturating_sub(1);
    seqlen / PAGE_SIZE * PAGE_SIZE
}

/// Cut a prefill chunk so the last full page can be checkpointed on its own.
///
/// Returns the boundary when `prefill_start < boundary <= prefill_end`.
/// The caller then stashes with [`LAST_PAGE_CHECKPOINT_INTERVAL`].
pub fn prefill_cut_for_last_page(
    prefill_start: usize,
    prefill_end: usize,
    seq_len: usize,
) -> Option<usize> {
    let boundary = last_full_page_boundary(seq_len);
    if prefill_start < boundary && boundary <= prefill_end {
        Some(boundary)
    } else {
        None
    }
}

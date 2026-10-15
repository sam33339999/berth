//! Tail-check seam.
//!
//! The integer `t` in `tail_repetition:<t>@<channel>` is not defined by the spec yet.
//! Smaller `t` is denser, and one generation may fire several `t`s, but the comparison
//! formula is undecided. This crate does not guess that `t` is a period.

use crate::{Channel, TokenId};

/// Decides which tail-repetition integers fire for one channel window.
///
/// The engine calls this only when the request enabled `x-berth-tail-check`,
/// once per accepted token on that channel.
///
/// `window` is that channel's most recent accepted tokens, already cut to the
/// request lookback (`min(buffered, lookback)`). It is never tokens from the
/// other channel, and it never contains a rejected draft token.
///
/// The returned integers are written as `tail_repetition:<t>@<channel>` in the
/// order given. What `t` measures is the implementor's choice; this crate does
/// not rank, filter, or reinterpret it. Duplicate values in one return are
/// recorded once.
pub trait TailDetector {
    fn detect(&mut self, channel: Channel, window: &[TokenId]) -> Vec<u32>;
}

impl<D: TailDetector + ?Sized> TailDetector for &mut D {
    fn detect(&mut self, channel: Channel, window: &[TokenId]) -> Vec<u32> {
        (**self).detect(channel, window)
    }
}

impl<D: TailDetector + ?Sized> TailDetector for Box<D> {
    fn detect(&mut self, channel: Channel, window: &[TokenId]) -> Vec<u32> {
        (**self).detect(channel, window)
    }
}

/// Detector that reports nothing. Used when the caller has no tail formula yet,
/// or when the tail check is off and a value is still required.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoTail;

impl TailDetector for NoTail {
    fn detect(&mut self, _channel: Channel, _window: &[TokenId]) -> Vec<u32> {
        Vec::new()
    }
}

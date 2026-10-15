//! Generation repetition check.
//!
//! After the sampler accepts a token, the engine hands it to [`RepetitionCheck`]
//! with the channel it was written to. Rejected draft tokens are not stored.
//! Each channel keeps its own buffer of at most [`MAX_WINDOW`] accepted tokens.
//! A run that starts in `thinking` does not continue into `response`. Models
//! with no thinking channel pass every accepted token as [`Channel::Response`].
//!
//! Two checks, each off unless its header is present:
//!
//! - `x-berth-tail-check` — lookback window, then [`TailDetector`]. The integer
//!   `t` in `tail_repetition:<t>@<channel>` has no formula in v1. This crate
//!   does not treat `t` as a period. See [`TailDetector`].
//! - `x-berth-exact-repeat-check` — a block of at least L tokens copied
//!   back-to-back. One different token between copies does not count.
//!   Low-diversity / `low_logprob` is not implemented and is never emitted.
//!
//! Labels accumulate for the request: insertion order, duplicates skipped,
//! 256 bytes and 64 labels max. A new label is the signal to send the
//! mid-stream event again, carrying the whole set. An empty set produces no
//! event and no completion field.
//!
//! Spec: `docs/repetition.md`.
//!
//! # Exact-repeat label
//!
//! On each accepted token the check looks only at a suffix of that channel.
//! Among blocks with `len >= L` and `count >= 2` that tile that suffix, it
//! records the smallest `len`, and for that `len` the largest `count`.
//! A longer multiple of the same run is not also recorded at that step.
//! The next copy, when it completes, is a new label: `exact_repetition:64x2@thinking`
//! then `exact_repetition:64x3@thinking`. Both stay in the set.
//!
//! Detection is one Z-array over the reversed window: **O(W) time per accepted
//! token**, O(W) scratch, W <= 4096. The bound does not grow with the generation.
//!
//! # Malformed headers
//!
//! After ASCII trim, a value is well-formed only when it is a non-empty run of
//! ASCII digits (no sign, no fraction). Anything else turns that check off.
//! A well-formed tail length outside 512..=4096 enables the tail check at 4096.
//! A well-formed exact minimum of `0`, or one that does not fit in `u32`, turns
//! exact-repeat off. There is no clamp range for the exact minimum.
//! The first occurrence of each header wins. Names are case-insensitive.
//! [`PROPOSED_TAIL_LOOKBACK`] and [`PROPOSED_EXACT_MIN_TOKENS`] are what a client
//! sends; a missing header does not apply them.

#![forbid(unsafe_code)]

mod channel;
mod event;
mod exact;
mod header;
mod tail;
mod trigger;

pub use channel::Channel;
pub use event::{
    format_completion_field, format_mid_stream_event, REPETITION_CHECK_EVENT,
    REPETITION_CHECK_FIELD,
};
pub use header::{
    parse_exact_min_tokens, parse_tail_lookback, CheckConfig, EXACT_REPEAT_CHECK_HEADER,
    MAX_WINDOW, PROPOSED_EXACT_MIN_TOKENS, PROPOSED_TAIL_LOOKBACK, TAIL_CHECK_HEADER,
    TAIL_LOOKBACK_MAX, TAIL_LOOKBACK_MIN,
};
pub use tail::{NoTail, TailDetector};
pub use trigger::{TriggerSet, MAX_LABEL_BYTES, MAX_TRIGGERS};

use std::collections::VecDeque;

/// Sampler token id. Compared as an integer; never decoded to text.
pub type TokenId = u32;

/// One position from the sampler, in the order the draft was scored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampled {
    /// Kept. Enters the channel buffer and is checked immediately, so a later
    /// accepted token in the same draft can already see it.
    Accepted(TokenId),
    /// Thrown away. The id is ignored. It does not enter the buffer and does
    /// not sit between the accepted tokens on either side.
    Rejected(TokenId),
}

/// What the engine should do with the stream after one [`RepetitionCheck::observe`].
///
/// `new_trigger` is the edge: send [`Self::mid_stream_event`] only when it is set.
/// The event text, when present, carries the full accumulated set, not only the
/// label just added. No event is produced when the set is empty.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "new_trigger means the engine should send mid_stream_event"]
pub struct Outcome {
    /// At least one label was stored on this call.
    pub new_trigger: bool,
    /// Mid-stream SSE text to send when [`Self::new_trigger`] is set.
    pub mid_stream_event: Option<String>,
}

impl Outcome {
    fn silent() -> Self {
        Self {
            new_trigger: false,
            mid_stream_event: None,
        }
    }
}

/// Per-request repetition check.
///
/// `D` is only called when the tail check is enabled. [`NoTail`] is the
/// stand-in when the caller has no formula.
pub struct RepetitionCheck<D: TailDetector> {
    config: CheckConfig,
    detector: D,
    thinking: VecDeque<TokenId>,
    response: VecDeque<TokenId>,
    triggers: TriggerSet,
    snapshot: Vec<TokenId>,
    reversed: Vec<TokenId>,
    z: Vec<usize>,
}

impl<D: TailDetector + Default> Default for RepetitionCheck<D> {
    fn default() -> Self {
        Self::new(CheckConfig::disabled(), D::default())
    }
}

impl<D: TailDetector> RepetitionCheck<D> {
    pub fn new(config: CheckConfig, detector: D) -> Self {
        Self {
            config,
            detector,
            thinking: VecDeque::with_capacity(MAX_WINDOW),
            response: VecDeque::with_capacity(MAX_WINDOW),
            triggers: TriggerSet::new(),
            snapshot: Vec::with_capacity(MAX_WINDOW),
            reversed: Vec::with_capacity(MAX_WINDOW),
            z: Vec::with_capacity(MAX_WINDOW),
        }
    }

    pub fn config(&self) -> CheckConfig {
        self.config
    }

    /// Labels recorded so far, in insertion order.
    pub fn triggers(&self) -> &[String] {
        self.triggers.as_slice()
    }

    /// Accepted tokens currently retained for `channel` (at most [`MAX_WINDOW`]).
    ///
    /// Rejected tokens are not counted. When both checks are off, nothing is retained.
    pub fn accepted_len(&self, channel: Channel) -> usize {
        self.buffer(channel).len()
    }

    /// The tail detector, so a test or the engine can read what it recorded.
    pub fn detector_mut(&mut self) -> &mut D {
        &mut self.detector
    }

    /// Mid-stream SSE event for the accumulated set, or `None` when nothing has fired.
    ///
    /// This does not remember whether the event was already sent. The engine sends
    /// it when [`Outcome::new_trigger`] is set, and may call this again to re-read
    /// the same bytes.
    pub fn mid_stream_event(&self) -> Option<String> {
        format_mid_stream_event(self.triggers())
    }

    /// `repetition_check` field for `response.completed` / `response.incomplete`.
    ///
    /// `None` when nothing has fired. The field is omitted in that case, not sent empty.
    pub fn completion_field(&self) -> Option<String> {
        format_completion_field(self.triggers())
    }

    /// Push one sampled token and run the enabled checks.
    ///
    /// Accepted tokens append to that channel's ring (the oldest falls off past
    /// [`MAX_WINDOW`]) and are checked before this call returns. Exact-repeat, when
    /// enabled, is O(W) in the retained window: copy the window, build one Z-array
    /// on its reverse, scan for the smallest qualifying period. Tail detection, when
    /// enabled, is one [`TailDetector::detect`] call on the lookback suffix
    /// (`min(retained, lookback)` tokens of this channel only) and adds no further
    /// asymptotic term here.
    ///
    /// On a token where both checks fire, the exact-repeat label is considered
    /// first, then each distinct `t` from the detector in return order.
    /// A label that loses to the 256-byte cap or the 64-label cap does not count
    /// as new, so it does not by itself cause another mid-stream event.
    pub fn observe(&mut self, channel: Channel, sampled: Sampled) -> Outcome {
        let token = match sampled {
            Sampled::Accepted(token) => token,
            Sampled::Rejected(_) => return Outcome::silent(),
        };
        if !self.config.tail_enabled() && !self.config.exact_enabled() {
            return Outcome::silent();
        }
        self.push(channel, token);
        self.load_snapshot(channel);

        let mut batch = Vec::new();
        if let Some(min_len) = self.config.exact_min_tokens() {
            if let Some((len, count)) =
                exact::best_run(&self.snapshot, min_len, &mut self.reversed, &mut self.z)
            {
                batch.push(format!("exact_repetition:{len}x{count}@{channel}"));
            }
        }
        if let Some(lookback) = self.config.tail_lookback() {
            let start = self.snapshot.len().saturating_sub(lookback as usize);
            let found = self.detector.detect(channel, &self.snapshot[start..]);
            let mut seen = Vec::new();
            for t in found {
                if seen.contains(&t) {
                    continue;
                }
                seen.push(t);
                batch.push(format!("tail_repetition:{t}@{channel}"));
            }
        }

        let mut added = false;
        for label in &batch {
            if self.triggers.insert(label) {
                added = true;
            }
        }
        if !added {
            return Outcome::silent();
        }
        let mid_stream_event = format_mid_stream_event(self.triggers.as_slice());
        debug_assert!(mid_stream_event.is_some());
        Outcome {
            new_trigger: true,
            mid_stream_event,
        }
    }

    fn push(&mut self, channel: Channel, token: TokenId) {
        let buf = self.buffer_mut(channel);
        if buf.len() == MAX_WINDOW {
            let _ = buf.pop_front();
        }
        buf.push_back(token);
    }

    fn load_snapshot(&mut self, channel: Channel) {
        match channel {
            Channel::Thinking => copy_window(&self.thinking, &mut self.snapshot),
            Channel::Response => copy_window(&self.response, &mut self.snapshot),
        }
    }

    fn buffer(&self, channel: Channel) -> &VecDeque<TokenId> {
        match channel {
            Channel::Thinking => &self.thinking,
            Channel::Response => &self.response,
        }
    }

    fn buffer_mut(&mut self, channel: Channel) -> &mut VecDeque<TokenId> {
        match channel {
            Channel::Thinking => &mut self.thinking,
            Channel::Response => &mut self.response,
        }
    }
}

fn copy_window(src: &VecDeque<TokenId>, dst: &mut Vec<TokenId>) {
    dst.clear();
    dst.extend(src.iter().copied());
}

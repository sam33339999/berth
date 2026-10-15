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
//! 256 bytes and 64 labels max. One exact-repeat label is kept per
//! `(length, channel)`. A larger count rewrites that label in place and does
//! not take a new slot. A smaller count does not. A new label, or a rewritten
//! exact count, is the signal to send the mid-stream event again, carrying
//! the whole set. An empty set produces no event and no completion field.
//!
//! Spec: `docs/repetition.md`.
//!
//! # Exact-repeat label
//!
//! On each accepted token the check looks only at a suffix of that channel.
//! Among blocks with `len >= L` and `count >= 2` that tile that suffix, it
//! records the smallest `len`, and for that `len` the largest `count`.
//! A longer multiple of the same run is not also recorded at that step.
//! The same `(len, channel)` keeps a single label. The next copy rewrites it:
//! `exact_repetition:64x2@thinking` becomes `exact_repetition:64x3@thinking`.
//! The same length on the other channel is a different label.
//!
//! The retained window is at most 4096 tokens. A run that continues past that
//! still raises the count. The new copy is checked against the window, and
//! the copies already counted stay in the label.
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
/// label just added or rewritten. No event is produced when the set is empty.
#[derive(Clone, Debug, PartialEq, Eq)]
#[must_use = "new_trigger means the engine should send mid_stream_event"]
pub struct Outcome {
    /// At least one label was stored, or an exact-repeat count was raised.
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
    /// Accepted tokens ever pushed on each channel, including ones that have
    /// fallen out of the window. Exact-run anchors are in this count.
    thinking_seen: u64,
    response_seen: u64,
    /// One anchor per exact `(channel, len)` whose label was stored.
    exact_runs: Vec<ExactRun>,
    triggers: TriggerSet,
    snapshot: Vec<TokenId>,
    reversed: Vec<TokenId>,
    z: Vec<usize>,
}

/// High-water mark for one exact `(channel, len)`, and the accepted-token
/// count where that mark was written.
struct ExactRun {
    channel: Channel,
    len: u32,
    count: u32,
    at: u64,
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
            thinking_seen: 0,
            response_seen: 0,
            exact_runs: Vec::new(),
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
    /// as new. Rewriting an exact-repeat count does, including when 64 labels
    /// are already stored: the rewrite does not add a slot.
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

        let mut changed = false;
        if let Some(min_len) = self.config.exact_min_tokens() {
            if let Some((len, visible)) =
                exact::best_run(&self.snapshot, min_len, &mut self.reversed, &mut self.z)
            {
                let count = self.running_exact_count(channel, len, visible);
                if record_exact(&mut self.triggers, len, count, channel) {
                    // Anchor the next copy to this token. A count that does not
                    // raise the stored one leaves the anchor alone, so a shorter
                    // later run cannot walk the old count up one block at a time.
                    self.remember_exact(channel, len, count);
                    changed = true;
                }
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
                if self
                    .triggers
                    .insert(&format!("tail_repetition:{t}@{channel}"))
                {
                    changed = true;
                }
            }
        }

        if !changed {
            return Outcome::silent();
        }
        let mid_stream_event = format_mid_stream_event(self.triggers.as_slice());
        debug_assert!(mid_stream_event.is_some());
        Outcome {
            new_trigger: true,
            mid_stream_event,
        }
    }

    /// Count to record for this `(len, channel)`.
    ///
    /// [`exact::best_run`] only sees the retained window, so once a run has
    /// filled it, each further copy still reports that same `visible` count.
    /// If this copy ends exactly one block after the count already stored, the
    /// run is still going and the stored count goes up by one.
    fn running_exact_count(&self, channel: Channel, len: u32, visible: u32) -> u32 {
        let seen = self.seen(channel);
        let step = u64::from(len);
        for run in &self.exact_runs {
            if run.channel == channel && run.len == len && seen.saturating_sub(run.at) == step {
                return visible.max(run.count.saturating_add(1));
            }
        }
        visible
    }

    fn remember_exact(&mut self, channel: Channel, len: u32, count: u32) {
        let at = self.seen(channel);
        for run in &mut self.exact_runs {
            if run.channel == channel && run.len == len {
                run.count = count;
                run.at = at;
                return;
            }
        }
        self.exact_runs.push(ExactRun {
            channel,
            len,
            count,
            at,
        });
    }

    fn seen(&self, channel: Channel) -> u64 {
        match channel {
            Channel::Thinking => self.thinking_seen,
            Channel::Response => self.response_seen,
        }
    }

    fn push(&mut self, channel: Channel, token: TokenId) {
        match channel {
            Channel::Thinking => self.thinking_seen += 1,
            Channel::Response => self.response_seen += 1,
        }
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

/// One exact label per `(len, channel)`. A larger `count` rewrites that label
/// in place. A smaller or equal `count` leaves the set unchanged. The first
/// label for the pair is an ordinary insert, so the byte cap and the 64-label
/// cap still apply to it.
fn record_exact(triggers: &mut TriggerSet, len: u32, count: u32, channel: Channel) -> bool {
    if let Some((index, stored)) = find_exact(triggers.as_slice(), len, channel) {
        if count <= stored {
            return false;
        }
        return triggers.replace_at(index, &format_exact(len, count, channel));
    }
    triggers.insert(&format_exact(len, count, channel))
}

fn format_exact(len: u32, count: u32, channel: Channel) -> String {
    format!("exact_repetition:{len}x{count}@{channel}")
}

fn find_exact(labels: &[String], len: u32, channel: Channel) -> Option<(usize, u32)> {
    for (index, label) in labels.iter().enumerate() {
        if let Some((parsed_len, count, parsed_channel)) = parse_exact_label(label) {
            if parsed_len == len && parsed_channel == channel {
                return Some((index, count));
            }
        }
    }
    None
}

fn parse_exact_label(label: &str) -> Option<(u32, u32, Channel)> {
    let rest = label.strip_prefix("exact_repetition:")?;
    let (len_text, rest) = rest.split_once('x')?;
    let (count_text, channel_text) = rest.split_once('@')?;
    if !is_ascii_digits(len_text) || !is_ascii_digits(count_text) {
        return None;
    }
    let len = len_text.parse().ok()?;
    let count = count_text.parse().ok()?;
    let channel = match channel_text {
        "thinking" => Channel::Thinking,
        "response" => Channel::Response,
        _ => return None,
    };
    Some((len, count, channel))
}

fn is_ascii_digits(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit())
}

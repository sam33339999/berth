//! One generation: incremental detokenize, stop strings, think split, repetition.

use std::time::Duration;

use berth_repetition::{Channel, CheckConfig, NoTail, Outcome, RepetitionCheck, Sampled};
use tokenizers::Tokenizer;

use crate::think::{ThinkPart, ThinkStream};
use crate::Flow;

/// Something the HTTP layer writes to the client, in order.
#[derive(Debug)]
pub enum Outbound {
    Reasoning(String),
    Content(String),
    Text(String),
    /// `format_mid_stream_event` text. The writer adds the SSE frame's blank line.
    Repetition(String),
}

pub struct Turn {
    dec: Incremental,
    stops: StopHold,
    think: Option<ThinkStream>,
    rep: RepetitionCheck<NoTail>,
    stop_ids: Vec<u32>,
    raw: String,
    pieces: Vec<Outbound>,
    stop_string: bool,
}

impl Turn {
    pub fn new(
        chat: bool,
        in_think: bool,
        skip_special_tokens: bool,
        stop_ids: &[u32],
        stop_strings: &[String],
        repetition: CheckConfig,
    ) -> Self {
        Self {
            dec: Incremental::new(skip_special_tokens),
            stops: StopHold::new(stop_strings),
            think: chat.then(|| ThinkStream::new(in_think)),
            rep: RepetitionCheck::new(repetition, NoTail),
            stop_ids: stop_ids.to_vec(),
            raw: String::new(),
            pieces: Vec::new(),
            stop_string: false,
        }
    }

    pub fn on_token(&mut self, tokenizer: &Tokenizer, id: u32) -> Flow {
        let channel = self.channel();
        let outcome = self.rep.observe(channel, Sampled::Accepted(id));
        self.note_repetition(outcome);
        if self.stop_ids.contains(&id) {
            // The engine reports the stop token and then ends with Finish::Stop.
            // Keep it out of the text, the way skip_special_tokens drops eos.
            return Flow::Continue;
        }
        let text = self.dec.push(tokenizer, id);
        let (emit, matched) = self.stops.push(&text);
        self.append_visible(&emit);
        if matched {
            self.stop_string = true;
            return Flow::Cancel;
        }
        Flow::Continue
    }

    pub fn finish(&mut self, tokenizer: &Tokenizer) -> Vec<Outbound> {
        let tail = self.dec.finish(tokenizer);
        let (emit, matched) = self.stops.push(&tail);
        if matched {
            self.stop_string = true;
            self.append_visible(&emit);
        } else {
            self.append_visible(&emit);
            let rest = self.stops.finish();
            self.append_visible(&rest);
        }
        if let Some(think) = self.think.as_mut() {
            for part in think.finish() {
                self.push_think(part);
            }
        }
        self.drain_pieces()
    }

    pub fn drain_pieces(&mut self) -> Vec<Outbound> {
        std::mem::take(&mut self.pieces)
    }

    pub fn stop_string(&self) -> bool {
        self.stop_string
    }

    pub fn raw(&self) -> &str {
        &self.raw
    }

    pub fn triggers(&self) -> &[String] {
        self.rep.triggers()
    }

    fn channel(&self) -> Channel {
        match self.think.as_ref() {
            Some(think) if think.in_think() => Channel::Thinking,
            _ => Channel::Response,
        }
    }

    fn note_repetition(&mut self, outcome: Outcome) {
        if outcome.new_trigger {
            if let Some(event) = outcome.mid_stream_event {
                self.pieces.push(Outbound::Repetition(event));
            }
        }
    }

    fn append_visible(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.raw.push_str(text);
        if let Some(think) = self.think.as_mut() {
            let parts = think.feed(text);
            for part in parts {
                self.push_think(part);
            }
        } else {
            self.pieces.push(Outbound::Text(text.to_string()));
        }
    }

    fn push_think(&mut self, part: ThinkPart) {
        match part {
            ThinkPart::Reasoning(text) if !text.is_empty() => {
                self.pieces.push(Outbound::Reasoning(text));
            }
            ThinkPart::Content(text) if !text.is_empty() => {
                self.pieces.push(Outbound::Content(text));
            }
            _ => {}
        }
    }
}

/// Byte-level BPE can split one codepoint across tokens. A decode that ends in
/// U+FFFD is an incomplete sequence (`tokenizers` 0.22, `errors: replace`).
/// Hold those tokens and emit them with the token that completes the character,
/// so a stream chunk never contains a torn codepoint.
struct Incremental {
    window: Vec<u32>,
    skip_special_tokens: bool,
}

impl Incremental {
    fn new(skip_special_tokens: bool) -> Self {
        Self {
            window: Vec::new(),
            skip_special_tokens,
        }
    }

    fn push(&mut self, tokenizer: &Tokenizer, id: u32) -> String {
        self.window.push(id);
        match tokenizer.decode(&self.window, self.skip_special_tokens) {
            Ok(text) if !text.ends_with('\u{FFFD}') => {
                self.window.clear();
                text
            }
            _ => String::new(),
        }
    }

    fn finish(&mut self, tokenizer: &Tokenizer) -> String {
        if self.window.is_empty() {
            return String::new();
        }
        let text = tokenizer
            .decode(&self.window, self.skip_special_tokens)
            .unwrap_or_default();
        self.window.clear();
        text
    }
}

/// Hold back a proper prefix of a stop string so it is not streamed and then
/// retracted. A full match drops the stop and the rest of that piece.
struct StopHold {
    stops: Vec<String>,
    pending: String,
}

impl StopHold {
    fn new(stops: &[String]) -> Self {
        Self {
            stops: stops
                .iter()
                .filter(|stop| !stop.is_empty())
                .cloned()
                .collect(),
            pending: String::new(),
        }
    }

    fn push(&mut self, text: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (text.to_string(), false);
        }
        if text.is_empty() && self.pending.is_empty() {
            return (String::new(), false);
        }
        self.pending.push_str(text);
        if let Some(index) = self.find_stop() {
            let emit = self.pending[..index].to_string();
            self.pending.clear();
            return (emit, true);
        }
        let hold = hold_len(&self.pending, &self.stops);
        let cut = self.pending.len() - hold;
        let emit = self.pending[..cut].to_string();
        self.pending.drain(..cut);
        (emit, false)
    }

    fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }

    fn find_stop(&self) -> Option<usize> {
        self.stops
            .iter()
            .filter_map(|stop| self.pending.find(stop.as_str()))
            .min()
    }
}

fn hold_len(pending: &str, stops: &[String]) -> usize {
    let mut hold = 0;
    for (index, _) in pending.char_indices() {
        let suffix = &pending[index..];
        for stop in stops {
            if suffix.len() < stop.len() && stop.starts_with(suffix) && suffix.len() > hold {
                hold = suffix.len();
            }
        }
    }
    hold
}

/// Poll a capacity-1 channel without sleeping inside the async runtime.
/// A full channel waits briefly so a disconnect flag is noticed on the next token.
pub fn send_outbound(
    tx: &tokio::sync::mpsc::Sender<Outbound>,
    cancel: &std::sync::atomic::AtomicBool,
    mut piece: Outbound,
) -> Flow {
    use std::sync::atomic::Ordering;
    loop {
        if cancel.load(Ordering::SeqCst) {
            return Flow::Cancel;
        }
        match tx.try_send(piece) {
            Ok(()) => return Flow::Continue,
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => return Flow::Cancel,
            Err(tokio::sync::mpsc::error::TrySendError::Full(returned)) => {
                piece = returned;
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }
}

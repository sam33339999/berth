//! Wire forms from `docs/repetition.md`.
//!
//! The mid-stream `data` line is compact JSON, matching the spec's SSE example.
//! The completion-field snippet uses one space after `:` and `,`, matching the
//! spec's `response.completed` / `response.incomplete` example. Both are pinned
//! byte-for-byte by tests. Neither is emitted for an empty trigger list.

/// SSE `event:` name, also the `type` field of the data payload.
pub const REPETITION_CHECK_EVENT: &str = "response.repetition_check";

/// Object key on `response.completed` and `response.incomplete`.
pub const REPETITION_CHECK_FIELD: &str = "repetition_check";

/// `event:` and `data:` lines for the mid-stream check.
///
/// `None` when `triggers` is empty: the engine must not send an empty event.
/// The string is the two spec lines, each ending in `\n`. It does not add a
/// further blank line; a writer that concatenates SSE frames adds that separator.
pub fn format_mid_stream_event<S: AsRef<str>>(triggers: &[S]) -> Option<String> {
    if triggers.is_empty() {
        return None;
    }
    let mut data = String::from("{\"type\":\"");
    data.push_str(REPETITION_CHECK_EVENT);
    data.push_str("\",\"");
    data.push_str(REPETITION_CHECK_FIELD);
    data.push_str("\":{\"triggers\":[");
    for (index, trigger) in triggers.iter().enumerate() {
        if index > 0 {
            data.push(',');
        }
        push_json_string(&mut data, trigger.as_ref());
    }
    data.push_str("]}}");
    Some(format!("event: {REPETITION_CHECK_EVENT}\ndata: {data}\n"))
}

/// `"repetition_check": {"triggers": [...]}` for a terminal response object.
///
/// `None` when `triggers` is empty: the field is omitted, not sent as an empty list.
/// Whitespace matches the spec example (space after `:` and after `,`).
pub fn format_completion_field<S: AsRef<str>>(triggers: &[S]) -> Option<String> {
    if triggers.is_empty() {
        return None;
    }
    let mut out = String::from("\"");
    out.push_str(REPETITION_CHECK_FIELD);
    out.push_str("\": {\"triggers\": [");
    for (index, trigger) in triggers.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        push_json_string(&mut out, trigger.as_ref());
    }
    out.push_str("]}");
    Some(out)
}

fn push_json_string(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => push_unicode_escape(out, c),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn push_unicode_escape(out: &mut String, ch: char) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let value = ch as u32;
    out.push_str("\\u");
    out.push(char::from(HEX[((value >> 12) & 0xf) as usize]));
    out.push(char::from(HEX[((value >> 8) & 0xf) as usize]));
    out.push(char::from(HEX[((value >> 4) & 0xf) as usize]));
    out.push(char::from(HEX[(value & 0xf) as usize]));
}

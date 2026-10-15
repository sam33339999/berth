//! Request headers that turn the two checks on.
//!
//! Missing header means that check does not run. The proposed client defaults
//! ([`PROPOSED_TAIL_LOOKBACK`], [`PROPOSED_EXACT_MIN_TOKENS`]) are not filled in
//! for an absent header.

/// Tail-check header. The value is a decimal lookback length, in tokens.
pub const TAIL_CHECK_HEADER: &str = "x-berth-tail-check";

/// Exact-repeat header. The value is the minimum block length, in tokens.
pub const EXACT_REPEAT_CHECK_HEADER: &str = "x-berth-exact-repeat-check";

/// Lookback a client sends when it wants the tail check at the proposed default.
/// Absence of the header does not apply this value.
pub const PROPOSED_TAIL_LOOKBACK: u32 = 1024;

/// Minimum block length a client sends when it wants exact-repeat at the proposed default.
/// Absence of the header does not apply this value.
pub const PROPOSED_EXACT_MIN_TOKENS: u32 = 64;

/// Smallest tail lookback kept as sent.
pub const TAIL_LOOKBACK_MIN: u32 = 512;

/// Lookback used when a well-formed tail value falls outside [`TAIL_LOOKBACK_MIN`]..=[`TAIL_LOOKBACK_MAX`].
/// Also the per-channel buffer cap ([`crate::MAX_WINDOW`]).
pub const TAIL_LOOKBACK_MAX: u32 = 4096;

/// Accepted tokens retained per channel. Exact-repeat reads only this buffer.
/// A count already recorded can still rise after older tokens fall off: the new
/// copy is checked against the buffer, and the copies already counted stay in
/// the total.
pub const MAX_WINDOW: usize = TAIL_LOOKBACK_MAX as usize;

const _: () = assert!(MAX_WINDOW == 4096);

/// Parsed switches for one request.
///
/// Both checks start off. A missing header stays off. A well-formed tail value
/// outside 512..=4096 is enabled at 4096. A malformed value stays off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckConfig {
    tail_lookback: Option<u32>,
    exact_min_tokens: Option<u32>,
}

impl Default for CheckConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

impl CheckConfig {
    /// Both checks off.
    pub fn disabled() -> Self {
        Self {
            tail_lookback: None,
            exact_min_tokens: None,
        }
    }

    /// Build a config from already-split values.
    ///
    /// `tail_lookback`: `None` disables the tail check. `Some` enables it.
    /// Values outside 512..=4096 become 4096, same as the header.
    /// `exact_min_tokens`: `None` or `Some(0)` disables exact-repeat.
    /// Any other `Some` is kept. A minimum above [`MAX_WINDOW`] / 2 never matches:
    /// two copies have to fit in the retained window.
    pub fn new(tail_lookback: Option<u32>, exact_min_tokens: Option<u32>) -> Self {
        Self {
            tail_lookback: tail_lookback.map(clamp_tail_lookback),
            exact_min_tokens: exact_min_tokens.filter(|len| *len >= 1),
        }
    }

    /// First occurrence of each check header wins. Names are compared case-insensitively.
    /// Unrelated headers are ignored.
    pub fn parse_headers<'a>(headers: impl IntoIterator<Item = (&'a str, &'a str)>) -> Self {
        let mut tail = None;
        let mut exact = None;
        let mut saw_tail = false;
        let mut saw_exact = false;
        for (name, value) in headers {
            if !saw_tail && name.eq_ignore_ascii_case(TAIL_CHECK_HEADER) {
                saw_tail = true;
                tail = parse_tail_lookback(value);
            } else if !saw_exact && name.eq_ignore_ascii_case(EXACT_REPEAT_CHECK_HEADER) {
                saw_exact = true;
                exact = parse_exact_min_tokens(value);
            }
        }
        // `parse_*` already normalized the values. `new` clamps again; that is idempotent.
        Self::new(tail, exact)
    }

    /// Tail lookback in tokens, or `None` when the tail check is off.
    pub fn tail_lookback(self) -> Option<u32> {
        self.tail_lookback
    }

    /// Exact-repeat minimum block length, or `None` when that check is off.
    pub fn exact_min_tokens(self) -> Option<u32> {
        self.exact_min_tokens
    }

    pub fn tail_enabled(self) -> bool {
        self.tail_lookback.is_some()
    }

    pub fn exact_enabled(self) -> bool {
        self.exact_min_tokens.is_some()
    }
}

/// Parse one `x-berth-tail-check` value.
///
/// Grammar, after trimming ASCII whitespace: a non-empty run of ASCII digits, no sign.
/// That is the well-formed decimal token count.
///
/// - Value in 512..=4096: used as sent.
/// - Any other well-formed integer, including `0`, values above `u32::MAX`, and digit
///   strings that overflow `u128`: the check is on, with lookback 4096.
/// - Anything else (empty, sign, fraction, trailing junk): the check is off.
///
/// This does not apply [`PROPOSED_TAIL_LOOKBACK`]. The caller passes `None` / omits the
/// header for off.
pub fn parse_tail_lookback(value: &str) -> Option<u32> {
    match parse_decimal(value) {
        Decimal::Malformed => None,
        Decimal::TooBig => Some(TAIL_LOOKBACK_MAX),
        Decimal::Value(n) => Some(match u32::try_from(n) {
            Ok(n) if (TAIL_LOOKBACK_MIN..=TAIL_LOOKBACK_MAX).contains(&n) => n,
            _ => TAIL_LOOKBACK_MAX,
        }),
    }
}

/// Parse one `x-berth-exact-repeat-check` value.
///
/// Same digit grammar as [`parse_tail_lookback`]. The spec gives no clamp range for this
/// header, so a well-formed integer in `1..=u32::MAX` is kept as the minimum block length.
/// `0` is not a block length and is treated as malformed (check off). A digit string that
/// does not fit in `u32` is malformed (check off): there is no "use 4096" rule here.
pub fn parse_exact_min_tokens(value: &str) -> Option<u32> {
    match parse_decimal(value) {
        Decimal::Value(n) => u32::try_from(n).ok().filter(|len| *len >= 1),
        Decimal::Malformed | Decimal::TooBig => None,
    }
}

fn clamp_tail_lookback(value: u32) -> u32 {
    if (TAIL_LOOKBACK_MIN..=TAIL_LOOKBACK_MAX).contains(&value) {
        value
    } else {
        TAIL_LOOKBACK_MAX
    }
}

enum Decimal {
    /// Not a signless ASCII integer.
    Malformed,
    /// Digit string whose value fits in `u128`.
    Value(u128),
    /// Digit string that overflows `u128`. Still a well-formed integer for the tail clamp.
    TooBig,
}

fn parse_decimal(raw: &str) -> Decimal {
    let text = raw.trim_ascii();
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return Decimal::Malformed;
    }
    let mut value: u128 = 0;
    for byte in text.bytes() {
        let digit = u128::from(byte - b'0');
        if value > (u128::MAX - digit) / 10 {
            return Decimal::TooBig;
        }
        value = value * 10 + digit;
    }
    Decimal::Value(value)
}

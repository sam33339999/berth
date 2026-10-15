//! Generation channel. The caller passes this; it is not inferred from text.

use std::fmt;

/// Where an accepted token was written.
///
/// Models with no thinking channel put every accepted token in [`Channel::Response`].
/// A repetition never crosses from one channel to the other: each channel has its own buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Channel {
    /// Tokens the model writes for itself.
    Thinking,
    /// Tokens the caller shows to the user.
    Response,
}

impl Channel {
    /// Wire name used in trigger labels: `thinking` or `response`.
    pub fn as_str(self) -> &'static str {
        match self {
            Channel::Thinking => "thinking",
            Channel::Response => "response",
        }
    }
}

impl fmt::Display for Channel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

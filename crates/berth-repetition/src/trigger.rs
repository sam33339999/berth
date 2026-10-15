//! Per-request trigger labels.
//!
//! Insertion-ordered and deduplicated. A label longer than 256 bytes is dropped.
//! The 65th distinct label is dropped. Dropped labels are not truncated and do
//! not count as a new trigger.

/// Maximum stored label length, in bytes (not chars).
pub const MAX_LABEL_BYTES: usize = 256;

/// Maximum labels retained for one request.
pub const MAX_TRIGGERS: usize = 64;

/// Accumulated trigger labels for one request.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TriggerSet {
    labels: Vec<String>,
}

impl TriggerSet {
    pub fn new() -> Self {
        Self { labels: Vec::new() }
    }

    /// Insert `label` if it is new, at most [`MAX_LABEL_BYTES`] bytes, and the
    /// set is under [`MAX_TRIGGERS`].
    ///
    /// Returns whether this call stored a new label. A duplicate, an oversized
    /// label, or a label past the cap returns `false` and leaves the set unchanged.
    /// Oversized labels do not consume a slot.
    pub fn insert(&mut self, label: &str) -> bool {
        if self.labels.len() >= MAX_TRIGGERS {
            return false;
        }
        if label.len() > MAX_LABEL_BYTES {
            return false;
        }
        if self.labels.iter().any(|stored| stored == label) {
            return false;
        }
        self.labels.push(label.to_string());
        true
    }

    pub fn as_slice(&self) -> &[String] {
        &self.labels
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }
}

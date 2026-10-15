//! Per-request trigger labels.
//!
//! Insertion-ordered and deduplicated. A label longer than 256 bytes is dropped.
//! The 65th distinct label is dropped. Dropped labels are not truncated and do
//! not count as a new trigger. Replacing the text of a label already in the
//! list does not take a slot, and it still runs when the list is full.

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

    /// Overwrite the label at `index`.
    ///
    /// The list length does not change, so this still works at [`MAX_TRIGGERS`].
    /// Returns whether the stored string changed. An out-of-range index, a
    /// label longer than [`MAX_LABEL_BYTES`], the same string, or a string
    /// already stored at another index leaves the set unchanged.
    pub(crate) fn replace_at(&mut self, index: usize, label: &str) -> bool {
        if label.len() > MAX_LABEL_BYTES || index >= self.labels.len() {
            return false;
        }
        if self.labels.iter().any(|stored| stored == label) {
            return false;
        }
        self.labels[index] = label.to_string();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replace_at_rewrites_a_full_list_in_place() {
        let mut set = TriggerSet::new();
        for index in 0..MAX_TRIGGERS {
            assert!(set.insert(&format!("label-{index}")));
        }
        assert_eq!(set.len(), MAX_TRIGGERS);

        assert!(set.replace_at(0, "label-0b"));
        assert_eq!(set.len(), MAX_TRIGGERS);
        assert_eq!(set.as_slice()[0], "label-0b");
        assert_eq!(set.as_slice()[1], "label-1");
        assert_eq!(
            set.as_slice()[MAX_TRIGGERS - 1],
            format!("label-{}", MAX_TRIGGERS - 1)
        );

        assert!(!set.replace_at(0, "label-1"), "duplicate");
        assert_eq!(set.as_slice()[0], "label-0b");
        assert!(!set.replace_at(0, "label-0b"), "same string");

        let oversized = "z".repeat(MAX_LABEL_BYTES + 1);
        assert!(!set.replace_at(1, &oversized));
        assert_eq!(set.as_slice()[1], "label-1");
        assert!(!set.insert("one-past-cap"));
        assert_eq!(set.len(), MAX_TRIGGERS);
    }
}

use std::fmt;

use crate::PAGE_SIZE;

/// Failure from page-table bookkeeping or from a checkpoint interval that is
/// not a multiple of [`PAGE_SIZE`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// [`crate::PageTable::new`] was asked for zero pages.
    EmptyPool,
    /// Page count does not fit in a `u32` index.
    PoolTooLarge {
        /// Requested page count.
        pages: usize,
    },
    /// Every page is still referenced, so eviction cannot free one.
    OutOfPages,
    /// An interval or chunk size is zero or not a multiple of [`PAGE_SIZE`].
    NotPageMultiple {
        /// Which value failed the check.
        what: &'static str,
        /// The rejected value.
        value: usize,
    },
    /// [`crate::hash_page`] was given a slice that is not one full page.
    PartialPage {
        /// Number of token ids passed in.
        len: usize,
    },
    /// Page index or allocation ordinal is outside its collection.
    BadIndex {
        /// The rejected index.
        index: usize,
    },
    /// A reference count would go below zero.
    RefUnderflow {
        /// Pool index of the page.
        page: u32,
    },
    /// The page's reference count is not the one this operation requires.
    BadRefCount {
        /// Pool index of the page.
        page: u32,
        /// Count observed at the call.
        ref_count: u32,
    },
    /// Two live pages would share one hash. Treated as a bookkeeping failure,
    /// not as a token mismatch: lookups do not compare token ids.
    HashCollision,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::EmptyPool => write!(f, "page table needs at least one page"),
            Error::PoolTooLarge { pages } => {
                write!(f, "page count {pages} does not fit in a u32 index")
            }
            Error::OutOfPages => write!(f, "no unreferenced page left to allocate"),
            Error::NotPageMultiple { what, value } => write!(
                f,
                "{what} ({value}) must be a positive multiple of {PAGE_SIZE}"
            ),
            Error::PartialPage { len } => {
                write!(f, "page hash expects {PAGE_SIZE} token ids, got {len}")
            }
            Error::BadIndex { index } => write!(f, "page index {index} is out of range"),
            Error::RefUnderflow { page } => {
                write!(f, "reference count underflow on page {page}")
            }
            Error::BadRefCount { page, ref_count } => {
                write!(f, "page {page} has reference count {ref_count}")
            }
            Error::HashCollision => write!(f, "two live pages resolved to the same hash"),
        }
    }
}

impl std::error::Error for Error {}

//! Paged prefix table.
//!
//! Full-page blake2b-128 chain hashes, reference counts, tree eviction by root
//! `access_serial`, and the keys for GDN checkpoints and rewind. Works on page
//! indices only and never touches GPU memory. Pure Rust; no GPU.
//! Spec: `docs/cache.md`.
//!
//! There is no salt and no per-caller key. The isolation unit is the process:
//! two requests with the same token prefix share one chain of pages.
//!
//! Physical defrag and the CPU page tier are not implemented. Defrag only
//! renumbers pages while nothing is referenced; it does not change hash
//! identity, and this table's indices are that identity. The live service
//! runs with `cpu_cache_size = 0`, so an evicted page is gone.

mod checkpoint;
mod error;
mod hash;
mod table;

pub use checkpoint::{
    is_checkpoint_boundary, last_full_page_boundary, prefill_cut_for_last_page,
    should_stash_checkpoint, CheckpointPolicy, DEFAULT_RECURRENT_CHECKPOINT_INTERVAL,
    DEFAULT_RECURRENT_CHECKPOINT_INTERVAL_PP, LAST_PAGE_CHECKPOINT_INTERVAL,
};
pub use error::Error;
pub use hash::{
    fresh_page_count, hash_page, hash_sequence, hashed_page_count, is_content_hash, PageHash,
    HASH_LEN, PAGE_SIZE,
};
pub use table::{Allocation, Metrics, PageInfo, PageTable, DEFAULT_NUM_PAGES};

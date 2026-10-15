//! Chained blake2b-128 over full pages of token ids.
//!
//! Byte layout matches `exllamav3` `_tensor_blake2b_checksum`:
//! `hashlib.blake2b(digest_size=16)`, then the previous digest if there is
//! one, then this page's tokens. Token ids are little-endian `i64`, which is
//! `torch.long` / `numpy().tobytes()` on the little-endian host the service
//! runs on. The first page passes no previous hash at all. Sixteen zero bytes
//! are a different input and produce a different digest.

use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;

use crate::Error;

/// Tokens in one cache page. `exllamav3.constants.PAGE_SIZE`.
pub const PAGE_SIZE: usize = 256;

/// Digest length. Blake2b-128.
pub const HASH_LEN: usize = 16;

/// Blake2b-128 digest of one full page, chained to the previous page.
pub type PageHash = [u8; HASH_LEN];

/// Number of full pages `Sequence.prepare` hashes: `(len - 1) / PAGE_SIZE`.
///
/// The final token is outside the hashed prefix because prefill stops before
/// it (`len(sequence_ids) - 1`). A buffer whose length is an exact multiple of
/// [`PAGE_SIZE`] therefore hashes one page fewer than `len / PAGE_SIZE`. An
/// empty buffer hashes nothing. Python's floor division of `-1` for a zero
/// length is not used.
pub fn hashed_page_count(seq_len: usize) -> usize {
    seq_len.saturating_sub(1) / PAGE_SIZE
}

/// Pages a job must allocate beyond the hashed prefix.
///
/// Mirrors `Sequence.prepare`: the partial tail of the prompt is not hashed,
/// so it counts as a fresh page, as do pages reserved for `max_new_tokens`.
/// `has_prefix_token` adds the one extra token Python adds to `max_len`.
pub fn fresh_page_count(seq_len: usize, max_new_tokens: usize, has_prefix_token: bool) -> usize {
    let mut max_len = seq_len.saturating_add(max_new_tokens);
    if has_prefix_token {
        max_len = max_len.saturating_add(1);
    }
    let total_pages = max_len.div_ceil(PAGE_SIZE);
    total_pages.saturating_sub(hashed_page_count(seq_len))
}

/// True when `hash` is a content digest rather than a placeholder counter.
///
/// Placeholder hashes are a big-endian counter in 16 bytes, so their top
/// eight bytes stay zero. A blake2b-128 digest matches that with vanishing
/// probability. Lookups still key only on the hash.
pub fn is_content_hash(hash: &PageHash) -> bool {
    hash[..8] != [0u8; 8]
}

/// Blake2b-128 of one full page.
///
/// `prev` is the previous page's digest. `None` means the first page: those
/// 16 bytes are not fed to the hasher. `token_ids` must be exactly
/// [`PAGE_SIZE`] long.
pub fn hash_page(prev: Option<&PageHash>, token_ids: &[i64]) -> Result<PageHash, Error> {
    if token_ids.len() != PAGE_SIZE {
        return Err(Error::PartialPage {
            len: token_ids.len(),
        });
    }
    let mut hasher = Blake2bVar::new(HASH_LEN).expect("blake2b accepts a 16-byte digest");
    if let Some(prev) = prev {
        hasher.update(prev);
    }
    let mut bytes = [0u8; PAGE_SIZE * 8];
    for (i, token) in token_ids.iter().enumerate() {
        bytes[i * 8..i * 8 + 8].copy_from_slice(&token.to_le_bytes());
    }
    hasher.update(&bytes);
    let mut out = [0u8; HASH_LEN];
    hasher
        .finalize_variable(&mut out)
        .expect("blake2b output length is 16");
    Ok(out)
}

/// Hash the `(len - 1) / PAGE_SIZE` full pages of a sequence, each chained to
/// the previous digest. The partial tail is not hashed.
pub fn hash_sequence(token_ids: &[i64]) -> Result<Vec<PageHash>, Error> {
    let pages = hashed_page_count(token_ids.len());
    let mut out = Vec::with_capacity(pages);
    let mut prev: Option<PageHash> = None;
    for i in 0..pages {
        let start = i * PAGE_SIZE;
        let hash = hash_page(prev.as_ref(), &token_ids[start..start + PAGE_SIZE])?;
        out.push(hash);
        prev = Some(hash);
    }
    Ok(out)
}

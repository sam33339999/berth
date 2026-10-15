//! Behaviour tests for the paged prefix table. Pure CPU.
//!
//! The blake2b vector is the output of `tests/page_hash_vector.py`
//! (`python3 -I`). Page 0 is tokens `-1, 0, 1, ..., 254`. Page 1 is
//! `1000, 1001, ..., 1255`, chained onto page 0.

use berth_cache::{
    fresh_page_count, hash_page, hash_sequence, hashed_page_count, is_checkpoint_boundary,
    last_full_page_boundary, prefill_cut_for_last_page, should_stash_checkpoint, CheckpointPolicy,
    Error, PageTable, DEFAULT_NUM_PAGES, LAST_PAGE_CHECKPOINT_INTERVAL, PAGE_SIZE,
};

fn seq(pages: usize, seed: i64) -> Vec<i64> {
    let n = pages * PAGE_SIZE + 1;
    (0..n as i64)
        .map(|i| seed.wrapping_mul(1_000_003).wrapping_add(i))
        .collect()
}

fn page0() -> Vec<i64> {
    (0..PAGE_SIZE as i64).map(|i| i - 1).collect()
}

fn page1() -> Vec<i64> {
    (0..PAGE_SIZE as i64).map(|i| 1000 + i).collect()
}

#[test]
fn hash_matches_python_vector() {
    // tests/page_hash_vector.py
    let first = hash_page(None, &page0()).unwrap();
    let second = hash_page(Some(&first), &page1()).unwrap();
    let with_zero_prev = hash_page(Some(&[0u8; 16]), &page0()).unwrap();
    assert_eq!(hex(&first), "644aa0ad86e7b59e814598bf64626e55");
    assert_eq!(hex(&second), "f13d91d071f64e13cd35e82fccadc633");
    assert_eq!(hex(&with_zero_prev), "8c585dff802c85d8e1ad4a00b3fc47d7");
    assert_ne!(first, with_zero_prev);

    let mut both = page0();
    both.extend(page1());
    both.push(7);
    assert_eq!(hash_sequence(&both).unwrap(), vec![first, second]);
}

#[test]
fn partial_last_page_is_not_hashed() {
    assert_eq!(hashed_page_count(0), 0);
    assert_eq!(hashed_page_count(PAGE_SIZE), 0);
    assert_eq!(hashed_page_count(PAGE_SIZE + 1), 1);
    assert_eq!(hashed_page_count(PAGE_SIZE * 2), 1);
    assert_eq!(hashed_page_count(PAGE_SIZE * 2 + 1), 2);
    assert!(matches!(
        hash_page(None, &[1, 2, 3]).unwrap_err(),
        Error::PartialPage { len: 3 }
    ));

    let a = vec![4i64; PAGE_SIZE * 2];
    let mut b = a.clone();
    b[PAGE_SIZE] = 99;
    assert_eq!(hash_sequence(&a).unwrap(), hash_sequence(&b).unwrap());
    b[0] = 5;
    assert_ne!(hash_sequence(&a).unwrap(), hash_sequence(&b).unwrap());

    // Prompt tail that is not a full hashed page still needs a fresh slot.
    assert_eq!(fresh_page_count(PAGE_SIZE + 1, 0, false), 1);
    assert_eq!(fresh_page_count(PAGE_SIZE + 1, 1, false), 1);
    assert_eq!(fresh_page_count(PAGE_SIZE + 1, PAGE_SIZE, false), 2);
    assert_eq!(fresh_page_count(PAGE_SIZE + 1, 0, true), 1);
    assert_eq!(DEFAULT_NUM_PAGES * PAGE_SIZE, 270_336);
}

#[test]
fn prefix_hit_miss_and_partial_share() {
    let mut table = PageTable::new(8).unwrap();
    let mut full = seq(2, 1);
    let mut same_tail = full.clone();
    let last = same_tail.len() - 1;
    same_tail[last] = same_tail[last].wrapping_add(1);
    // Same hashed pages: the only difference is the excluded final token.
    assert_eq!(
        hash_sequence(&full).unwrap(),
        hash_sequence(&same_tail).unwrap()
    );

    let mut first = table.allocate(&full, 0, false).unwrap();
    assert_eq!(first.cached_pages, 0);
    assert_eq!(first.kv_only_pages, 0);
    assert_eq!(table.metrics().evictions, 2);
    assert_eq!(table.metrics().evictions_live, 0);
    let pages = first.page_indices.clone();
    table.seal_prefix(&mut first).unwrap();
    table.deallocate(first).unwrap();

    let again = table.allocate(&same_tail, 0, false).unwrap();
    assert_eq!(again.cached_pages, 2);
    assert_eq!(again.kv_position, 2 * PAGE_SIZE);
    assert_eq!(again.page_indices, pages);
    assert_eq!(table.metrics().evictions, 2);
    table.deallocate(again).unwrap();

    full[PAGE_SIZE] ^= 1;
    let diverged = table.allocate(&full, 0, false).unwrap();
    assert_eq!(diverged.cached_pages, 1);
    assert_eq!(diverged.page_indices[0], pages[0]);
    assert_ne!(diverged.page_indices[1], pages[1]);
    table.deallocate(diverged).unwrap();

    let other = table.allocate(&seq(1, 9), 0, false).unwrap();
    assert_eq!(other.cached_pages, 0);
    table.deallocate(other).unwrap();
}

#[test]
fn ref_counts_survive_one_deallocate() {
    let mut table = PageTable::new(1).unwrap();
    let tokens = seq(1, 3);
    let mut first = table.allocate(&tokens, 0, false).unwrap();
    table.seal_prefix(&mut first).unwrap();
    let page = first.page_indices[0];
    let hash = table.page(page).unwrap().hash;
    let second = table.allocate(&tokens, 0, false).unwrap();
    assert_eq!(second.cached_pages, 1);
    assert_eq!(second.page_indices, vec![page]);
    assert_eq!(table.page(page).unwrap().ref_count, 2);

    table.deallocate(first).unwrap();
    assert_eq!(table.page(page).unwrap().ref_count, 1);
    assert_eq!(table.page(page).unwrap().hash, hash);
    let err = table.allocate(&seq(1, 4), 0, false).unwrap_err();
    assert!(matches!(err, Error::OutOfPages));

    table.deallocate(second).unwrap();
    assert_eq!(table.page(page).unwrap().ref_count, 0);
    assert_eq!(table.page(page).unwrap().hash, hash);
    let third = table.allocate(&tokens, 0, false).unwrap();
    assert_eq!(third.cached_pages, 1);
    assert_eq!(third.page_indices, vec![page]);
}

#[test]
fn unused_unique_page_reverts() {
    let mut table = PageTable::new(1).unwrap();
    let tokens = seq(1, 4);
    let mut held = table.allocate(&tokens, 0, false).unwrap();
    table.seal_prefix(&mut held).unwrap();
    let hash = table.page(held.page_indices[0]).unwrap().hash;
    table.put_checkpoint(hash);
    table.deallocate(held).unwrap();

    let fresh = table.allocate(&[], 1, false).unwrap();
    assert_eq!(fresh.cached_pages, 0);
    assert_eq!(table.metrics().stashes_stranded, 1);
    assert_eq!(table.metrics().evictions_live, 1);
    table.deallocate(fresh).unwrap();

    assert!(table.has_checkpoint(&hash));
    assert_eq!(table.prune_stranded(), 0);
    let restored = table.allocate(&tokens, 0, false).unwrap();
    assert_eq!(restored.cached_pages, 1);
    assert_eq!(table.page(restored.page_indices[0]).unwrap().hash, hash);
}

#[test]
fn published_generation_page_is_a_later_hit() {
    let mut table = PageTable::new(4).unwrap();
    let mut prefix = vec![3i64; PAGE_SIZE + 1];
    let mut alloc = table.allocate(&prefix, 1, false).unwrap();
    table.seal_prefix(&mut alloc).unwrap();
    let generated: Vec<i64> = (0..PAGE_SIZE as i64).map(|i| 50 + i).collect();
    table.publish_page(&mut alloc, 1, &generated).unwrap();
    table.deallocate(alloc).unwrap();

    prefix.truncate(PAGE_SIZE);
    prefix.extend_from_slice(&generated);
    prefix.push(7);
    assert_eq!(hashed_page_count(prefix.len()), 2);
    let hit = table.allocate(&prefix, 0, false).unwrap();
    assert_eq!(hit.cached_pages, 2);
}

#[test]
fn eviction_free_then_oldest_tree_leaf_first() {
    let mut table = PageTable::new(6).unwrap();
    let mut chain = table.allocate(&seq(3, 1), 0, false).unwrap();
    let root = chain.page_indices[0];
    let mid = chain.page_indices[1];
    let leaf = chain.page_indices[2];
    table.seal_prefix(&mut chain).unwrap();
    table.deallocate(chain).unwrap();

    let mut other = table.allocate(&seq(2, 2), 0, false).unwrap();
    let other_pages = other.page_indices.clone();
    table.seal_prefix(&mut other).unwrap();
    table.deallocate(other).unwrap();

    let used: Vec<u32> = [root, mid, leaf]
        .into_iter()
        .chain(other_pages.iter().copied())
        .collect();
    let mut empty: Vec<u32> = (0..6u32).filter(|i| !used.contains(i)).collect();
    empty.sort_by_key(|&i| table.page(i).unwrap().access_serial);
    assert_eq!(empty.len(), 1);

    let miss = table.allocate(&seq(3, 3), 0, false).unwrap();
    assert_eq!(miss.cached_pages, 0);
    assert_eq!(
        miss.page_indices,
        vec![empty[0], leaf, mid],
        "empty page, then the older tree's leaf, then its parent"
    );
    assert_eq!(table.page(root).unwrap().ref_count, 0);
    assert_eq!(table.page(other_pages[0]).unwrap().ref_count, 0);
    assert_eq!(table.page(other_pages[1]).unwrap().ref_count, 0);
}

#[test]
fn live_prefix_extension_is_pruned_before_its_root() {
    let mut table = PageTable::new(3).unwrap();
    let mut chain = table.allocate(&seq(2, 1), 0, false).unwrap();
    let root = chain.page_indices[0];
    let leaf = chain.page_indices[1];
    table.seal_prefix(&mut chain).unwrap();
    table.deallocate(chain).unwrap();

    let mut shared = seq(2, 1);
    shared[PAGE_SIZE] ^= 1;
    let branch = table.allocate(&shared, 0, false).unwrap();
    assert_eq!(branch.cached_pages, 1);
    assert_eq!(branch.page_indices[0], root);
    assert_ne!(branch.page_indices[1], leaf);

    let taken = table.allocate(&seq(1, 8), 0, false).unwrap();
    assert_eq!(taken.page_indices, vec![leaf]);
    assert_eq!(table.page(root).unwrap().ref_count, 1);
}

#[test]
fn hit_refreshes_root_serial_and_protects_that_tree() {
    let mut table = PageTable::new(8).unwrap();
    let mut chain = table.allocate(&seq(3, 1), 0, false).unwrap();
    let root = chain.page_indices[0];
    table.seal_prefix(&mut chain).unwrap();
    table.deallocate(chain).unwrap();
    let before = table.page(root).unwrap().access_serial;

    let mut other = table.allocate(&seq(1, 2), 0, false).unwrap();
    let other_page = other.page_indices[0];
    table.seal_prefix(&mut other).unwrap();
    table.deallocate(other).unwrap();
    let other_serial = table.page(other_page).unwrap().access_serial;
    assert!(other_serial > before);

    let hit = table.allocate(&seq(3, 1), 0, false).unwrap();
    assert_eq!(hit.cached_pages, 3);
    assert_eq!(hit.page_indices[0], root);
    let after = table.page(root).unwrap().access_serial;
    assert!(after > before);
    assert!(after > other_serial);
    table.deallocate(hit).unwrap();

    // Four empty pages, then the older root, which is now `other`.
    let miss = table.allocate(&seq(5, 5), 0, false).unwrap();
    assert_eq!(miss.page_indices[4], other_page);
    assert!(!miss.page_indices.contains(&root));
}

#[test]
fn recurrent_gate_truncates_to_last_checkpoint() {
    let mut table = PageTable::new(6).unwrap();
    let tokens = seq(3, 6);
    let mut alloc = table.allocate(&tokens, 0, false).unwrap();
    assert_eq!(alloc.prefix_hashes.len(), 3);
    table.seal_prefix(&mut alloc).unwrap();
    let hashes = alloc.prefix_hashes.clone();
    table.deallocate(alloc).unwrap();

    // KV is present, but no checkpoint: the whole run is excluded.
    let none = table.allocate(&tokens, 0, true).unwrap();
    assert_eq!(none.cached_pages, 0);
    assert_eq!(none.kv_only_pages, 3);
    table.deallocate(none).unwrap();

    // Page 1 has no checkpoint. The hit still runs through page 2, the last
    // page whose hash does.
    table.put_checkpoint(hashes[0]);
    table.put_checkpoint(hashes[2]);
    let through_last = table.allocate(&tokens, 0, true).unwrap();
    assert_eq!(through_last.cached_pages, 3);
    assert_eq!(through_last.kv_only_pages, 0);
    table.deallocate(through_last).unwrap();

    let plain = table.allocate(&tokens, 0, false).unwrap();
    assert_eq!(plain.cached_pages, 3);
    assert_eq!(plain.kv_only_pages, 0);
    table.deallocate(plain).unwrap();

    // A checkpoint on a page that was never sealed does not extend the hit.
    let raw = table.allocate(&seq(1, 11), 0, false).unwrap();
    let raw_hash = raw.prefix_hashes[0];
    table.deallocate(raw).unwrap();
    table.put_checkpoint(raw_hash);
    let gated = table.allocate(&seq(1, 11), 0, true).unwrap();
    assert_eq!(gated.cached_pages, 0);
}

#[test]
fn recurrent_gate_stops_at_middle_checkpoint() {
    let mut table = PageTable::new(4).unwrap();
    let tokens = seq(3, 6);
    let mut alloc = table.allocate(&tokens, 0, false).unwrap();
    table.seal_prefix(&mut alloc).unwrap();
    let mid_hash = alloc.prefix_hashes[1];
    table.deallocate(alloc).unwrap();

    table.put_checkpoint(mid_hash);
    let before = table.metrics().alloc_kv_only_pages;
    let gated = table.allocate(&tokens, 0, true).unwrap();
    assert_eq!(gated.cached_pages, 2);
    assert_eq!(gated.kv_only_pages, 1);
    assert_eq!(gated.kv_position, 2 * PAGE_SIZE);
    assert_eq!(table.metrics().alloc_kv_only_pages, before + 1);
}

#[test]
fn stranded_checkpoint_is_counted_then_pruned() {
    let mut table = PageTable::new(2).unwrap();
    let mut first = table.allocate(&seq(1, 1), 0, false).unwrap();
    table.seal_prefix(&mut first).unwrap();
    let hash_a = first.prefix_hashes[0];
    table.put_checkpoint(hash_a);
    table.deallocate(first).unwrap();

    let mut second = table.allocate(&seq(1, 2), 0, false).unwrap();
    table.seal_prefix(&mut second).unwrap();
    let hash_b = second.prefix_hashes[0];
    table.put_checkpoint(hash_b);
    table.deallocate(second).unwrap();
    assert_eq!(table.metrics().stashes_stranded, 0);

    let third = table.allocate(&seq(1, 3), 0, false).unwrap();
    assert_eq!(third.cached_pages, 0);
    assert_eq!(table.metrics().stashes_stranded, 1);
    assert!(table.has_checkpoint(&hash_a));
    assert_eq!(table.prune_stranded(), 1);
    assert!(!table.has_checkpoint(&hash_a));
    assert!(table.has_checkpoint(&hash_b));
    assert_eq!(table.prune_stranded(), 0);
    assert_eq!(table.metrics().pruned_stranded, 1);
}

#[test]
fn checkpoint_intervals_match_job_rules() {
    let policy = CheckpointPolicy::qwen3_8(2048).unwrap();
    assert_eq!(policy.interval, 2048);
    assert_eq!(policy.interval_pp, 32_768);

    let wide = CheckpointPolicy::new(2048, 32_768, 3072).unwrap();
    assert_eq!(wide.interval_pp, 33_792);

    assert!(matches!(
        CheckpointPolicy::qwen3_8(1000).unwrap_err(),
        Error::NotPageMultiple { .. }
    ));
    assert!(CheckpointPolicy::new(2000, 32_768, 2048).is_err());
    assert!(PageTable::new(0).is_err());

    let prompt = 100_000usize;
    assert!(is_checkpoint_boundary(32_768, prompt, 0, &policy, None));
    assert!(!is_checkpoint_boundary(2048, prompt, 0, &policy, None));
    assert!(is_checkpoint_boundary(96_256, prompt, 0, &policy, None));
    assert!(!is_checkpoint_boundary(94_208, prompt, 0, &policy, None));
    assert!(is_checkpoint_boundary(
        1024 + 32_768,
        prompt,
        4,
        &policy,
        None
    ));
    assert!(!is_checkpoint_boundary(
        1024 + 2048,
        prompt,
        4,
        &policy,
        None
    ));

    assert!(is_checkpoint_boundary(
        256,
        257,
        0,
        &policy,
        Some(LAST_PAGE_CHECKPOINT_INTERVAL)
    ));
    assert!(!is_checkpoint_boundary(
        100,
        257,
        0,
        &policy,
        Some(LAST_PAGE_CHECKPOINT_INTERVAL)
    ));
    assert!(!should_stash_checkpoint(
        0,
        prompt,
        0,
        &policy,
        Some(LAST_PAGE_CHECKPOINT_INTERVAL),
        None
    ));
    assert!(should_stash_checkpoint(
        256,
        257,
        0,
        &policy,
        Some(LAST_PAGE_CHECKPOINT_INTERVAL),
        None
    ));
    assert!(!should_stash_checkpoint(
        256,
        257,
        0,
        &policy,
        Some(LAST_PAGE_CHECKPOINT_INTERVAL),
        Some(256)
    ));

    assert_eq!(last_full_page_boundary(257), 256);
    assert_eq!(last_full_page_boundary(256), 0);
    assert_eq!(last_full_page_boundary(513), 512);
    assert_eq!(prefill_cut_for_last_page(0, 2048, 513), Some(512));
    assert_eq!(prefill_cut_for_last_page(0, 512, 513), Some(512));
    assert_eq!(prefill_cut_for_last_page(512, 2048, 513), None);
    assert_eq!(prefill_cut_for_last_page(0, 256, 513), None);
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0xf) as usize] as char);
    }
    out
}

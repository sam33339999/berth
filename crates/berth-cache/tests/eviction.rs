//! Parity with exllamav3 `PageTable.build_eviction_order`.
//!
//! Expected orders are the stdout of `python3 -I tests/eviction_order.py`.
//! That script is a stdlib transcription of `generator/pagetable.py` lines
//! 387-496. The states below are the ones the script models: empty pages,
//! a chain whose parent hash was overwritten, two intact trees, a branch,
//! referenced pages, and a protected prefix. A full miss consumes the raw
//! order. The protected case also checks the allocation trace printed as
//! `protected_broken_consumed`, which walks that order the way
//! `allocate_pages` / `next_evictable` does.
//!
//! If a Rust order disagrees, keep the Python numbers and ignore the test.

use berth_cache::{hash_sequence, Allocation, PageTable, PAGE_SIZE};

fn seq(pages: usize, seed: i64) -> Vec<i64> {
    let n = pages * PAGE_SIZE + 1;
    (0..n as i64)
        .map(|i| seed.wrapping_mul(1_000_003).wrapping_add(i))
        .collect()
}

fn replacement_page() -> Vec<i64> {
    vec![7; PAGE_SIZE]
}

fn diverge_second_page(seed: i64) -> Vec<i64> {
    let mut tokens = seq(2, seed);
    tokens[PAGE_SIZE] ^= 1;
    tokens
}

fn seal_dealloc(table: &mut PageTable, tokens: &[i64]) -> Vec<u32> {
    let mut alloc = table.allocate(tokens, 0, false).unwrap();
    let indices = alloc.page_indices.clone();
    table.seal_prefix(&mut alloc).unwrap();
    table.deallocate(alloc).unwrap();
    indices
}

/// Every unreferenced page, in the order a complete miss takes them.
fn take_all(table: &mut PageTable, seed: i64) -> Vec<u32> {
    let n = table.unreferenced_pages();
    let alloc = table.allocate(&seq(n, seed), 0, false).unwrap();
    assert_eq!(
        alloc.cached_pages, 0,
        "probe hashed a page that was already live"
    );
    let order = alloc.page_indices.clone();
    table.deallocate(alloc).unwrap();
    order
}

fn assert_prev_missing(table: &PageTable, index: u32) {
    let prev = table
        .page(index)
        .unwrap()
        .prev_hash
        .expect("broken page still records a parent hash");
    let live = (0..table.num_pages() as u32).any(|i| table.page(i).unwrap().hash == prev);
    assert!(!live, "page {index} parent hash is still live");
}

fn assert_prev_is(table: &PageTable, child: u32, parent: u32) {
    let prev = table
        .page(child)
        .unwrap()
        .prev_hash
        .expect("child has no parent hash");
    assert_eq!(prev, table.page(parent).unwrap().hash);
}

/// Chain of three, then a chain of two, then overwrite the first root's hash.
///
/// `publish_page` on a one-page hit has no following slot, so it does not
/// retarget the old child. That child's `prev_hash` still names the digest
/// `update_hash` removed.
fn broken_beside_intact(pool: usize) -> PageTable {
    let mut table = PageTable::new(pool).unwrap();
    assert_eq!(seal_dealloc(&mut table, &seq(3, 1)), vec![0, 1, 2]);
    assert_eq!(seal_dealloc(&mut table, &seq(2, 2)), vec![3, 4]);

    let mut root = table.allocate(&seq(1, 1), 0, false).unwrap();
    assert_eq!(root.page_indices, vec![0]);
    assert_eq!(root.cached_pages, 1);
    table
        .publish_page(&mut root, 0, &replacement_page())
        .unwrap();
    table.deallocate(root).unwrap();

    let original = hash_sequence(&seq(3, 1)).unwrap();
    assert_ne!(table.page(0).unwrap().hash, original[0]);
    assert_eq!(table.page(1).unwrap().hash, original[1]);
    assert_eq!(table.page(2).unwrap().hash, original[2]);
    let other = hash_sequence(&seq(2, 2)).unwrap();
    assert_eq!(table.page(3).unwrap().hash, other[0]);
    assert_eq!(table.page(4).unwrap().hash, other[1]);

    assert_prev_missing(&table, 1);
    assert_prev_is(&table, 2, 1);
    assert_prev_is(&table, 4, 3);
    assert!(table.page(0).unwrap().prev_hash.is_none());
    assert!(table.page(3).unwrap().prev_hash.is_none());
    for index in [0, 1, 2, 3, 4] {
        assert_eq!(
            table.page(index).unwrap().kv_position,
            PAGE_SIZE as u32,
            "page {index} is not a full page"
        );
        assert_eq!(table.page(index).unwrap().ref_count, 0);
    }
    // The overwritten root was touched last, so its serial is newer than the
    // intact tree's root. The broken leaf is newer than its own orphan root;
    // leaf-to-root still emits that leaf first.
    assert!(table.page(3).unwrap().access_serial < table.page(0).unwrap().access_serial);
    assert!(table.page(1).unwrap().access_serial < table.page(2).unwrap().access_serial);
    table
}

fn branch(hold: bool) -> (PageTable, Option<Allocation>) {
    let mut table = PageTable::new(5).unwrap();
    assert_eq!(seal_dealloc(&mut table, &seq(2, 1)), vec![0, 1]);
    let mut branched = table.allocate(&diverge_second_page(1), 0, false).unwrap();
    assert_eq!(branched.page_indices, vec![0, 2]);
    assert_eq!(branched.cached_pages, 1);
    table.seal_prefix(&mut branched).unwrap();

    assert_prev_is(&table, 1, 0);
    assert_prev_is(&table, 2, 0);
    assert!(table.page(0).unwrap().prev_hash.is_none());
    assert!(table.page(1).unwrap().access_serial < table.page(2).unwrap().access_serial);
    assert_eq!(table.page(3).unwrap().kv_position, 0);
    assert_eq!(table.page(4).unwrap().kv_position, 0);
    assert_eq!(table.page(1).unwrap().kv_position, PAGE_SIZE as u32);
    assert_eq!(table.page(2).unwrap().kv_position, PAGE_SIZE as u32);

    if hold {
        assert_eq!(table.page(0).unwrap().ref_count, 1);
        assert_eq!(table.page(2).unwrap().ref_count, 1);
        assert_eq!(table.page(1).unwrap().ref_count, 0);
        (table, Some(branched))
    } else {
        table.deallocate(branched).unwrap();
        (table, None)
    }
}

#[test]
fn empty_pages_then_broken_chain_then_older_intact_tree() {
    // python3 -I tests/eviction_order.py
    // empty_broken_two_trees 5 2 1 4 3 0
    let mut table = broken_beside_intact(6);
    assert_eq!(table.page(5).unwrap().kv_position, 0);
    assert!(table.page(5).unwrap().prev_hash.is_none());
    assert_eq!(take_all(&mut table, 99), vec![5, 2, 1, 4, 3, 0]);
}

#[test]
fn branch_evicts_older_leaf_before_newer_leaf_before_root() {
    // python3 -I tests/eviction_order.py
    // branch_two_children 3 4 1 2 0
    let (mut table, held) = branch(false);
    assert!(held.is_none());
    assert_eq!(take_all(&mut table, 99), vec![3, 4, 1, 2, 0]);
}

#[test]
fn referenced_pages_are_not_evicted_and_block_the_root() {
    // python3 -I tests/eviction_order.py
    // referenced_blocks_root 3 4 1
    let (mut table, held) = branch(true);
    let held = held.expect("branch stays referenced");
    let order = take_all(&mut table, 99);
    assert_eq!(order, vec![3, 4, 1]);
    assert!(!order.contains(&held.page_indices[0]));
    assert!(!order.contains(&held.page_indices[1]));
    table.deallocate(held).unwrap();
}

#[test]
fn protected_prefix_does_not_cannibalize_broken_tail() {
    // python3 -I tests/eviction_order.py
    // protected_setup_unprotected 2 1 4 3 0
    // protected_broken_raw       4 3 0 1 2
    // protected_broken_consumed  4 1 2 3 0
    //
    // The raw protected order is not what `allocate` returns. One leading
    // miss, hits on the surviving tail, then two fresh pages walk it the way
    // `next_evictable` skips pages referenced after the order was built.
    let mut unprotected = broken_beside_intact(5);
    assert_eq!(take_all(&mut unprotected, 99), vec![2, 1, 4, 3, 0]);

    let mut table = broken_beside_intact(5);
    let alloc = table.allocate(&seq(3, 1), 2, false).unwrap();
    assert_eq!(alloc.page_indices, vec![4, 1, 2, 3, 0]);
    let original = hash_sequence(&seq(3, 1)).unwrap();
    assert_eq!(table.page(1).unwrap().hash, original[1]);
    assert_eq!(table.page(2).unwrap().hash, original[2]);
    assert_eq!(table.page(1).unwrap().kv_position, PAGE_SIZE as u32);
    assert_eq!(table.page(2).unwrap().kv_position, PAGE_SIZE as u32);
    table.deallocate(alloc).unwrap();
}

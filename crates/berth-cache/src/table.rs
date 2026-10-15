//! Fixed pool of cache pages, keyed by chained content hashes.
//!
//! Eviction policy is `"tree"`: empty pages (`kv_position == 0`), then broken
//! chains, then whole trees ordered by the root's `access_serial`, oldest
//! first. Inside a tree, pages go leaf to root, oldest leaf first. A prefix
//! hit `add_ref`s every matched page, and the root is the first of those, so
//! the hit refreshes the root serial.
//!
//! When `recurrent` is set, the reported hit stops at the last page of the
//! leading KV run whose hash has a checkpoint. Earlier pages in that run do
//! not each need a checkpoint. Trailing KV-only pages stay allocated but are
//! not part of the hit; the caller recomputes them. `cache.md` describes the
//! same cut. Pages past the last checkpoint are what Python counts as
//! `alloc_kv_only_pages`.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use crate::{
    fresh_page_count, hash_page, hash_sequence, is_content_hash, Error, PageHash, HASH_LEN,
    PAGE_SIZE,
};

/// `CACHE_TOKENS / PAGE_SIZE` for the live service (`270336 / 256`).
pub const DEFAULT_NUM_PAGES: usize = 1056;

/// One claim on the pool. Drop it with [`PageTable::deallocate`].
///
/// Not `Clone`: deallocating the same claim twice would drop two references.
#[derive(Debug)]
pub struct Allocation {
    /// Pool indices, hashed prefix pages first, then fresh pages.
    pub page_indices: Vec<u32>,
    /// Content hashes of the `(len - 1) / PAGE_SIZE` prefix pages, in order.
    pub prefix_hashes: Vec<PageHash>,
    /// Leading pages whose KV can be skipped. Already cut by recurrent gating.
    pub cached_pages: usize,
    /// KV pages past the last usable checkpoint. Zero when `recurrent` is false.
    pub kv_only_pages: usize,
    /// `cached_pages * PAGE_SIZE`. Resume prefill here.
    pub kv_position: usize,
}

/// Counters from `PageTable.metrics`. `pruned_stranded` is the page table's
/// own tally of [`PageTable::prune_stranded`]; Python keeps that number on
/// the recurrent cache as `stash_pruned`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Metrics {
    /// Unreferenced pages repurposed.
    pub evictions: u64,
    /// Of those, complete content-hashed pages.
    pub evictions_live: u64,
    /// Of the live evictions, pages whose hash still had a checkpoint.
    /// An event counter: a later revert does not decrement it.
    pub stashes_stranded: u64,
    /// Pages claimed by [`PageTable::allocate`], including fresh pages.
    pub alloc_pages: u64,
    /// Of those, reused as a resumable prefix (after recurrent gating).
    pub alloc_cached_pages: u64,
    /// Cached KV pages dropped from the hit because no checkpoint covers them.
    pub alloc_kv_only_pages: u64,
    /// Checkpoints dropped by [`PageTable::prune_stranded`].
    pub pruned_stranded: u64,
}

/// Snapshot of one pool slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageInfo {
    /// Pool index.
    pub index: u32,
    /// Current hash. Placeholder counter, or a content digest once the page is full.
    pub hash: PageHash,
    /// Previous page's hash. `None` is a root.
    pub prev_hash: Option<PageHash>,
    /// Jobs currently holding the page.
    pub ref_count: u32,
    /// Last time the page was claimed. Trees are ordered by the root's value.
    pub access_serial: u64,
    /// Tokens of KV valid on this page. [`PAGE_SIZE`] means a full cached page.
    pub kv_position: u32,
    /// [`crate::is_content_hash`] of `hash`.
    pub content_hash: bool,
}

#[derive(Debug, Clone)]
struct Backup {
    hash: PageHash,
    prev_hash: Option<PageHash>,
    access_serial: u64,
    kv_position: u32,
}

#[derive(Debug)]
struct Page {
    index: u32,
    hash: PageHash,
    prev_hash: Option<PageHash>,
    ref_count: u32,
    access_serial: u64,
    kv_position: u32,
    /// Set by `add_ref_unique`. `deallocate` restores `backup` while this is set,
    /// so an unused generation page gives the displaced entry back.
    can_revert: bool,
    backup: Option<Backup>,
}

/// Fixed page pool. Not internally synchronized.
#[derive(Debug)]
pub struct PageTable {
    pages: Vec<Page>,
    referenced: HashMap<PageHash, u32>,
    unreferenced: HashMap<PageHash, u32>,
    access_serial: u64,
    /// Big-endian counter used as a placeholder hash. Per table, not
    /// process-global: content hashes do not depend on it.
    placeholder_counter: u64,
    metrics: Metrics,
    checkpoints: HashSet<PageHash>,
}

impl PageTable {
    /// Empty pool of `num_pages` slots. Each starts unreferenced, with
    /// `kv_position == 0` and `access_serial == index`.
    pub fn new(num_pages: usize) -> Result<Self, Error> {
        if num_pages == 0 {
            return Err(Error::EmptyPool);
        }
        if num_pages > u32::MAX as usize {
            return Err(Error::PoolTooLarge { pages: num_pages });
        }
        let mut table = Self {
            pages: Vec::with_capacity(num_pages),
            referenced: HashMap::new(),
            unreferenced: HashMap::with_capacity(num_pages),
            access_serial: num_pages as u64,
            placeholder_counter: 0,
            metrics: Metrics::default(),
            checkpoints: HashSet::new(),
        };
        for idx in 0..num_pages {
            let hash = table.alloc_placeholder()?;
            table.unreferenced.insert(hash, idx as u32);
            table.pages.push(Page {
                index: idx as u32,
                hash,
                prev_hash: None,
                ref_count: 0,
                access_serial: idx as u64,
                kv_position: 0,
                can_revert: false,
                backup: None,
            });
        }
        Ok(table)
    }

    /// Number of slots in the pool.
    pub fn num_pages(&self) -> usize {
        self.pages.len()
    }

    /// Eviction and hit counters.
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    /// One slot. Errors if `index` is outside the pool.
    pub fn page(&self, index: u32) -> Result<PageInfo, Error> {
        let page = self.pages.get(index as usize).ok_or(Error::BadIndex {
            index: index as usize,
        })?;
        Ok(PageInfo {
            index: page.index,
            hash: page.hash,
            prev_hash: page.prev_hash,
            ref_count: page.ref_count,
            access_serial: page.access_serial,
            kv_position: page.kv_position,
            content_hash: is_content_hash(&page.hash),
        })
    }

    /// Slots with `ref_count == 0`.
    pub fn unreferenced_pages(&self) -> usize {
        self.unreferenced.len()
    }

    /// Record that `hash` has a GDN checkpoint.
    ///
    /// The checkpoint bytes are not stored. Presence is what prefix gating
    /// and [`Self::prune_stranded`] consult. The 4 GiB LRU cap needs those
    /// byte sizes, so it is not applied here.
    pub fn put_checkpoint(&mut self, hash: PageHash) {
        self.checkpoints.insert(hash);
    }

    /// Whether [`Self::put_checkpoint`] recorded `hash` and it has not been pruned.
    pub fn has_checkpoint(&self, hash: &PageHash) -> bool {
        self.checkpoints.contains(hash)
    }

    /// How many checkpoint hashes are stored.
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// Drop checkpoints whose page chain is not fully live.
    ///
    /// A hash is resumable when every page from it back to a root is present
    /// with `kv_position == PAGE_SIZE`. Returns how many hashes were removed.
    /// Does not change [`Metrics::stashes_stranded`], which counts overwrite
    /// events rather than how many checkpoints are stranded right now.
    pub fn prune_stranded(&mut self) -> usize {
        let stranded: Vec<PageHash> = self
            .checkpoints
            .iter()
            .copied()
            .filter(|hash| !self.is_resumable(hash))
            .collect();
        let n = stranded.len();
        for hash in stranded {
            self.checkpoints.remove(&hash);
        }
        self.metrics.pruned_stranded += n as u64;
        n
    }

    /// Allocate pages for `token_ids`.
    ///
    /// Only `(len - 1) / PAGE_SIZE` pages are hashed. `new_unique_pages` is
    /// the fresh tail plus generation headroom (`fresh_page_count`). A hash
    /// that is already in the pool is reused and its reference count goes up.
    /// A miss takes the next page from the tree eviction order and keeps the
    /// wanted hash with `kv_position == 0` until [`Self::seal_prefix`] or
    /// [`Self::publish_page`].
    ///
    /// Hashes this call is about to look up are protected: a miss will not
    /// cannibalize a later hit of the same sequence unless nothing else is
    /// left. `recurrent` cuts the hit as described on the module.
    pub fn allocate(
        &mut self,
        token_ids: &[i64],
        new_unique_pages: usize,
        recurrent: bool,
    ) -> Result<Allocation, Error> {
        let prefix_hashes = hash_sequence(token_ids)?;
        let protected: HashSet<PageHash> = prefix_hashes.iter().copied().collect();
        let mut allocated: Vec<u32> = Vec::with_capacity(prefix_hashes.len() + new_unique_pages);
        let mut order: Option<VecDeque<u32>> = None;

        for hash in &prefix_hashes {
            self.access_serial = self.access_serial.saturating_add(1);
            let serial = self.access_serial;
            if let Some(idx) = self.referenced.get(hash).copied() {
                self.add_ref(idx as usize, serial)?;
                allocated.push(idx);
            } else if let Some(idx) = self.unreferenced.get(hash).copied() {
                self.add_ref(idx as usize, serial)?;
                allocated.push(idx);
            } else {
                let idx = self.next_evictable(&mut order, &protected)?;
                self.note_evict(idx);
                self.add_ref_clear(idx, serial, *hash)?;
                allocated.push(idx as u32);
            }
        }

        for _ in 0..new_unique_pages {
            self.access_serial = self.access_serial.saturating_add(1);
            let serial = self.access_serial;
            let idx = self.next_evictable(&mut order, &protected)?;
            self.note_evict(idx);
            self.add_ref_unique(idx, serial)?;
            let prev_hash = allocated.last().and_then(|&prev_idx| {
                let prev = &self.pages[prev_idx as usize];
                (prev.kv_position == PAGE_SIZE as u32).then_some(prev.hash)
            });
            if let Some(prev_hash) = prev_hash {
                self.pages[idx].prev_hash = Some(prev_hash);
            }
            allocated.push(idx as u32);
        }

        let mut cached_pages = 0usize;
        for &idx in &allocated {
            if self.pages[idx as usize].kv_position == PAGE_SIZE as u32 {
                cached_pages += 1;
            } else {
                break;
            }
        }

        let mut kv_only_pages = 0usize;
        if recurrent {
            // Highest checkpoint strictly inside the leading KV run. Pages
            // before it are part of the hit even with no stash of their own.
            let mut max_recur = 0usize;
            for (pi, hash) in prefix_hashes.iter().enumerate() {
                if pi < cached_pages && self.checkpoints.contains(hash) {
                    max_recur = pi + 1;
                }
            }
            kv_only_pages = cached_pages - max_recur;
            self.metrics.alloc_kv_only_pages += kv_only_pages as u64;
            cached_pages = max_recur;
        }

        self.metrics.alloc_pages += allocated.len() as u64;
        self.metrics.alloc_cached_pages += cached_pages as u64;
        self.debug_check();

        Ok(Allocation {
            page_indices: allocated,
            prefix_hashes,
            cached_pages,
            kv_only_pages,
            kv_position: cached_pages * PAGE_SIZE,
        })
    }

    /// [`Self::allocate`] with `new_unique_pages` from [`fresh_page_count`].
    pub fn allocate_job(
        &mut self,
        token_ids: &[i64],
        max_new_tokens: usize,
        has_prefix_token: bool,
        recurrent: bool,
    ) -> Result<Allocation, Error> {
        let fresh = fresh_page_count(token_ids.len(), max_new_tokens, has_prefix_token);
        self.allocate(token_ids, fresh, recurrent)
    }

    /// Mark the hashed prefix as full KV and link `prev_hash`.
    ///
    /// Call after prefill has written those pages. A hit is already full;
    /// this still confirms the chain. The first fresh page, if any, is linked
    /// to the last prefix page and loses `can_revert`, matching prefill when
    /// it finishes the last hashed page and steps one page past it.
    pub fn seal_prefix(&mut self, alloc: &mut Allocation) -> Result<(), Error> {
        let n = alloc.prefix_hashes.len();
        for i in 0..n {
            let hash = alloc.prefix_hashes[i];
            let prev = if i == 0 {
                None
            } else {
                Some(alloc.prefix_hashes[i - 1])
            };
            self.install_full_page(alloc, i, hash, prev)?;
        }
        self.debug_check();
        Ok(())
    }

    /// Publish one allocated page once its 256 tokens are known.
    ///
    /// This is the page-boundary path in `Job` sampling. The previous page's
    /// current hash is the chain parent (`None` for ordinal 0). If another
    /// referenced page already has the digest, this claim adopts that page
    /// and the displaced slot is released (an unused unique page reverts).
    /// If an unreferenced page has the digest, that page is cleared first.
    pub fn publish_page(
        &mut self,
        alloc: &mut Allocation,
        ordinal: usize,
        token_ids: &[i64],
    ) -> Result<(), Error> {
        if ordinal >= alloc.page_indices.len() {
            return Err(Error::BadIndex { index: ordinal });
        }
        let prev = if ordinal == 0 {
            None
        } else {
            let prev_index = alloc.page_indices[ordinal - 1] as usize;
            let prev_page = self
                .pages
                .get(prev_index)
                .ok_or(Error::BadIndex { index: prev_index })?;
            Some(prev_page.hash)
        };
        let new_hash = hash_page(prev.as_ref(), token_ids)?;
        self.install_full_page(alloc, ordinal, new_hash, prev)?;
        if ordinal < alloc.prefix_hashes.len() {
            alloc.prefix_hashes[ordinal] = new_hash;
        }
        self.debug_check();
        Ok(())
    }

    /// Drop one reference per claimed page.
    ///
    /// At zero the page stays in the pool under its hash, so a later job can
    /// hit it. A fresh page that was never published reverts to the identity
    /// it displaced.
    pub fn deallocate(&mut self, alloc: Allocation) -> Result<(), Error> {
        for idx in alloc.page_indices {
            self.sub_ref(idx as usize)?;
        }
        self.debug_check();
        Ok(())
    }

    fn install_full_page(
        &mut self,
        alloc: &mut Allocation,
        ordinal: usize,
        new_hash: PageHash,
        prev: Option<PageHash>,
    ) -> Result<(), Error> {
        if ordinal >= alloc.page_indices.len() {
            return Err(Error::BadIndex { index: ordinal });
        }
        let mut index = alloc.page_indices[ordinal] as usize;
        if index >= self.pages.len() {
            return Err(Error::BadIndex { index });
        }
        if self.pages[index].ref_count == 0 {
            return Err(Error::BadRefCount {
                page: index as u32,
                ref_count: 0,
            });
        }

        if let Some(existing) = self.referenced.get(&new_hash).copied() {
            let existing = existing as usize;
            if existing != index {
                let serial = self.pages[index].access_serial;
                self.sub_ref(index)?;
                self.add_ref(existing, serial)?;
                alloc.page_indices[ordinal] = existing as u32;
                index = existing;
            }
            let page = &mut self.pages[index];
            page.kv_position = PAGE_SIZE as u32;
            page.prev_hash = prev;
            page.can_revert = false;
            page.backup = None;
        } else {
            if let Some(other) = self.unreferenced.get(&new_hash).copied() {
                self.clear_page(other as usize)?;
            }
            self.pages[index].kv_position = PAGE_SIZE as u32;
            self.pages[index].prev_hash = prev;
            self.update_hash(index, new_hash)?;
        }

        if ordinal + 1 < alloc.page_indices.len() {
            let next = alloc.page_indices[ordinal + 1] as usize;
            if next < self.pages.len() {
                let page = &mut self.pages[next];
                page.prev_hash = Some(new_hash);
                page.can_revert = false;
            }
        }
        Ok(())
    }

    fn is_resumable(&self, hash: &PageHash) -> bool {
        let mut hash = *hash;
        for _ in 0..=self.pages.len() {
            let Some(idx) = self
                .referenced
                .get(&hash)
                .or_else(|| self.unreferenced.get(&hash))
                .copied()
            else {
                return false;
            };
            let page = &self.pages[idx as usize];
            if page.kv_position != PAGE_SIZE as u32 {
                return false;
            }
            match page.prev_hash {
                None => return true,
                Some(prev) => hash = prev,
            }
        }
        false
    }

    fn next_evictable(
        &mut self,
        order: &mut Option<VecDeque<u32>>,
        protected: &HashSet<PageHash>,
    ) -> Result<usize, Error> {
        if order.is_none() {
            *order = Some(self.build_eviction_order(protected));
        }
        let queue = order.as_mut().expect("eviction order was just built");
        loop {
            let Some(idx) = queue.pop_front() else {
                return Err(Error::OutOfPages);
            };
            // A hash hit later in this allocate may already hold the page.
            if self.pages[idx as usize].ref_count == 0 {
                return Ok(idx as usize);
            }
        }
    }

    fn note_evict(&mut self, index: usize) {
        self.metrics.evictions += 1;
        let page = &self.pages[index];
        if page.kv_position == PAGE_SIZE as u32 && is_content_hash(&page.hash) {
            self.metrics.evictions_live += 1;
            // No CPU tier. The checkpoint stays until `prune_stranded`; the
            // counter only records that the anchor was overwritten.
            if self.checkpoints.contains(&page.hash) {
                self.metrics.stashes_stranded += 1;
            }
        }
    }

    /// Order unreferenced pages: empty, then broken chains, then rooted trees.
    ///
    /// Referenced pages stay in the tree as non-evictable nodes so an
    /// unreferenced extension of a live prefix is still pruned from the leaf.
    /// A referenced or protected page blocks its ancestors. Anything the walk
    /// does not emit is appended at the end, protected hashes last.
    fn build_eviction_order(&self, protected: &HashSet<PageHash>) -> VecDeque<u32> {
        let mut order = Vec::new();
        let mut emitted = HashSet::new();

        let mut empty: Vec<u32> = self
            .pages
            .iter()
            .filter(|page| {
                page.ref_count == 0 && page.kv_position == 0 && !protected.contains(&page.hash)
            })
            .map(|page| page.index)
            .collect();
        empty.sort_by_key(|&index| self.pages[index as usize].access_serial);
        for index in empty {
            order.push(index);
            emitted.insert(index);
        }

        let mut by_hash: HashMap<PageHash, u32> = HashMap::with_capacity(self.pages.len());
        for page in &self.pages {
            by_hash.insert(page.hash, page.index);
        }

        let mut children = vec![Vec::new(); self.pages.len()];
        let mut parent_of = vec![None; self.pages.len()];
        let mut roots = Vec::new();
        let mut orphan_roots = Vec::new();
        for page in &self.pages {
            match page.prev_hash {
                None => roots.push(page.index),
                Some(prev) => match by_hash.get(&prev).copied() {
                    Some(parent) if parent != page.index => {
                        children[parent as usize].push(page.index);
                        parent_of[page.index as usize] = Some(parent);
                    }
                    _ => orphan_roots.push(page.index),
                },
            }
        }
        roots.sort_by_key(|&index| self.pages[index as usize].access_serial);
        orphan_roots.sort_by_key(|&index| self.pages[index as usize].access_serial);

        for root in orphan_roots {
            prune_tree(
                root,
                &children,
                &parent_of,
                &self.pages,
                protected,
                &mut order,
                &mut emitted,
            );
        }
        for root in roots {
            prune_tree(
                root,
                &children,
                &parent_of,
                &self.pages,
                protected,
                &mut order,
                &mut emitted,
            );
        }

        let mut leftovers: Vec<u32> = self
            .pages
            .iter()
            .filter(|page| page.ref_count == 0 && !emitted.contains(&page.index))
            .map(|page| page.index)
            .collect();
        leftovers.sort_by_key(|&index| {
            let page = &self.pages[index as usize];
            (protected.contains(&page.hash), page.access_serial)
        });
        order.extend(leftovers);
        VecDeque::from(order)
    }

    fn add_ref(&mut self, index: usize, serial: u64) -> Result<(), Error> {
        if index >= self.pages.len() {
            return Err(Error::BadIndex { index });
        }
        let hash = self.pages[index].hash;
        if self.pages[index].ref_count == 0 {
            if self.referenced.contains_key(&hash) {
                return Err(Error::HashCollision);
            }
            self.unreferenced.remove(&hash);
            self.referenced.insert(hash, index as u32);
        }
        let page = &mut self.pages[index];
        page.ref_count = page.ref_count.saturating_add(1);
        // Prefix walks claim page 0, the root, before the rest of the chain.
        page.access_serial = page.access_serial.max(serial);
        page.can_revert = false;
        page.backup = None;
        Ok(())
    }

    fn add_ref_clear(
        &mut self,
        index: usize,
        serial: u64,
        new_hash: PageHash,
    ) -> Result<(), Error> {
        if index >= self.pages.len() {
            return Err(Error::BadIndex { index });
        }
        if self.pages[index].ref_count != 0 {
            return Err(Error::BadRefCount {
                page: index as u32,
                ref_count: self.pages[index].ref_count,
            });
        }
        let old = self.pages[index].hash;
        if self.referenced.contains_key(&new_hash)
            || (new_hash != old && self.unreferenced.contains_key(&new_hash))
        {
            return Err(Error::HashCollision);
        }
        self.unreferenced.remove(&old);
        let page = &mut self.pages[index];
        page.hash = new_hash;
        page.prev_hash = None;
        page.kv_position = 0;
        page.ref_count = 1;
        page.access_serial = serial;
        page.can_revert = false;
        page.backup = None;
        self.referenced.insert(new_hash, index as u32);
        Ok(())
    }

    fn add_ref_unique(&mut self, index: usize, serial: u64) -> Result<(), Error> {
        if index >= self.pages.len() {
            return Err(Error::BadIndex { index });
        }
        if self.pages[index].ref_count != 0 {
            return Err(Error::BadRefCount {
                page: index as u32,
                ref_count: self.pages[index].ref_count,
            });
        }
        let backup = Backup {
            hash: self.pages[index].hash,
            prev_hash: self.pages[index].prev_hash,
            access_serial: self.pages[index].access_serial,
            kv_position: self.pages[index].kv_position,
        };
        self.unreferenced.remove(&backup.hash);
        let new_hash = match self.alloc_placeholder() {
            Ok(hash) => hash,
            Err(err) => {
                self.unreferenced.insert(backup.hash, index as u32);
                return Err(err);
            }
        };
        let page = &mut self.pages[index];
        page.backup = Some(backup);
        page.hash = new_hash;
        page.prev_hash = None;
        page.kv_position = 0;
        page.ref_count = 1;
        page.access_serial = serial;
        page.can_revert = true;
        self.referenced.insert(new_hash, index as u32);
        Ok(())
    }

    fn sub_ref(&mut self, index: usize) -> Result<(), Error> {
        if index >= self.pages.len() {
            return Err(Error::BadIndex { index });
        }
        if self.pages[index].ref_count == 0 {
            return Err(Error::RefUnderflow { page: index as u32 });
        }
        self.pages[index].ref_count -= 1;
        if self.pages[index].ref_count > 0 {
            return Ok(());
        }
        let current = self.pages[index].hash;
        self.referenced.remove(&current);
        if self.pages[index].can_revert {
            if let Some(backup) = self.pages[index].backup.take() {
                let page = &mut self.pages[index];
                page.hash = backup.hash;
                page.prev_hash = backup.prev_hash;
                page.access_serial = backup.access_serial;
                page.kv_position = backup.kv_position;
                page.can_revert = false;
            }
        }
        let restored = self.pages[index].hash;
        if self.referenced.contains_key(&restored) || self.unreferenced.contains_key(&restored) {
            let replacement = self.alloc_placeholder()?;
            let page = &mut self.pages[index];
            page.hash = replacement;
            page.prev_hash = None;
        }
        let final_hash = self.pages[index].hash;
        self.unreferenced.insert(final_hash, index as u32);
        Ok(())
    }

    fn clear_page(&mut self, index: usize) -> Result<(), Error> {
        if index >= self.pages.len() {
            return Err(Error::BadIndex { index });
        }
        if self.pages[index].ref_count != 0 {
            return Err(Error::BadRefCount {
                page: index as u32,
                ref_count: self.pages[index].ref_count,
            });
        }
        let old = self.pages[index].hash;
        self.unreferenced.remove(&old);
        let new_hash = match self.alloc_placeholder() {
            Ok(hash) => hash,
            Err(err) => {
                self.unreferenced.insert(old, index as u32);
                return Err(err);
            }
        };
        let page = &mut self.pages[index];
        page.hash = new_hash;
        page.prev_hash = None;
        page.kv_position = 0;
        page.can_revert = false;
        page.backup = None;
        self.unreferenced.insert(new_hash, index as u32);
        Ok(())
    }

    fn update_hash(&mut self, index: usize, new_hash: PageHash) -> Result<(), Error> {
        if self.pages[index].ref_count == 0 {
            return Err(Error::BadRefCount {
                page: index as u32,
                ref_count: 0,
            });
        }
        if self.pages[index].kv_position != PAGE_SIZE as u32 {
            return Err(Error::PartialPage {
                len: self.pages[index].kv_position as usize,
            });
        }
        let old = self.pages[index].hash;
        if old == new_hash {
            self.pages[index].can_revert = false;
            return Ok(());
        }
        if self.referenced.contains_key(&new_hash) || self.unreferenced.contains_key(&new_hash) {
            return Err(Error::HashCollision);
        }
        self.referenced.remove(&old);
        self.pages[index].hash = new_hash;
        self.pages[index].can_revert = false;
        self.referenced.insert(new_hash, index as u32);
        Ok(())
    }

    fn alloc_placeholder(&mut self) -> Result<PageHash, Error> {
        for _ in 0..=self.pages.len().saturating_add(1) {
            self.placeholder_counter = self.placeholder_counter.saturating_add(1);
            if self.placeholder_counter == 0 {
                return Err(Error::HashCollision);
            }
            let hash = placeholder_hash(self.placeholder_counter);
            if !self.referenced.contains_key(&hash) && !self.unreferenced.contains_key(&hash) {
                return Ok(hash);
            }
        }
        Err(Error::HashCollision)
    }

    fn debug_check(&self) {
        debug_assert_eq!(
            self.referenced.len() + self.unreferenced.len(),
            self.pages.len(),
            "referenced and unreferenced maps must partition the pool"
        );
        for (hash, &idx) in &self.referenced {
            let page = &self.pages[idx as usize];
            debug_assert_eq!(page.hash, *hash);
            debug_assert!(page.ref_count > 0);
            debug_assert!(!self.unreferenced.contains_key(hash));
        }
        for (hash, &idx) in &self.unreferenced {
            let page = &self.pages[idx as usize];
            debug_assert_eq!(page.hash, *hash);
            debug_assert_eq!(page.ref_count, 0);
        }
    }
}

fn placeholder_hash(counter: u64) -> PageHash {
    let mut hash = [0u8; HASH_LEN];
    hash[8..].copy_from_slice(&counter.to_be_bytes());
    hash
}

fn prune_tree(
    root: u32,
    children: &[Vec<u32>],
    parent_of: &[Option<u32>],
    pages: &[Page],
    protected: &HashSet<PageHash>,
    order: &mut Vec<u32>,
    emitted: &mut HashSet<u32>,
) {
    let mut remaining: HashMap<u32, i32> = HashMap::new();
    let mut heap: BinaryHeap<(Reverse<u64>, Reverse<u32>)> = BinaryHeap::new();
    let mut stack = vec![root];
    let mut seen = HashSet::new();
    while let Some(node) = stack.pop() {
        if !seen.insert(node) {
            continue;
        }
        let kids = &children[node as usize];
        remaining.insert(node, kids.len() as i32);
        if kids.is_empty() {
            heap.push((Reverse(pages[node as usize].access_serial), Reverse(node)));
        }
        for &child in kids.iter().rev() {
            stack.push(child);
        }
    }

    while let Some((_, Reverse(node))) = heap.pop() {
        if !emitted.contains(&node) {
            let page = &pages[node as usize];
            // A page still in use blocks its ancestors: do not free the parent.
            if page.ref_count > 0 || protected.contains(&page.hash) {
                continue;
            }
            order.push(node);
            emitted.insert(node);
        }
        if let Some(parent) = parent_of[node as usize] {
            if let Some(left) = remaining.get_mut(&parent) {
                *left -= 1;
                if *left == 0 {
                    heap.push((
                        Reverse(pages[parent as usize].access_serial),
                        Reverse(parent),
                    ));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq(pages: usize, seed: i64) -> Vec<i64> {
        let n = pages * PAGE_SIZE + 1;
        (0..n as i64)
            .map(|i| seed.wrapping_mul(1_000_003).wrapping_add(i))
            .collect()
    }

    #[test]
    fn broken_chain_before_intact_tree_leaf_first() {
        let mut table = PageTable::new(4).unwrap();
        let mut chain = table.allocate(&seq(3, 1), 0, false).unwrap();
        let root = chain.page_indices[0];
        let mid = chain.page_indices[1];
        let leaf = chain.page_indices[2];
        table.seal_prefix(&mut chain).unwrap();
        table.deallocate(chain).unwrap();

        let mut other = table.allocate(&seq(1, 2), 0, false).unwrap();
        let other_page = other.page_indices[0];
        table.seal_prefix(&mut other).unwrap();
        table.deallocate(other).unwrap();

        // Parent of `mid` no longer resolves. `leaf` still names `mid`.
        table.pages[mid as usize].prev_hash = Some([0xAB; HASH_LEN]);

        let first = table.allocate(&seq(1, 3), 0, false).unwrap();
        assert_eq!(first.page_indices, vec![leaf]);
        let second = table.allocate(&seq(1, 4), 0, false).unwrap();
        assert_eq!(second.page_indices, vec![mid]);
        assert_eq!(table.page(root).unwrap().ref_count, 0);
        assert_eq!(table.page(other_page).unwrap().ref_count, 0);
    }
}

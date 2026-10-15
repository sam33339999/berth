#!/usr/bin/env python3
"""Eviction-order vectors for berth-cache.

Stdlib only. Run with `python3 -I`. Do not import torch or exllamav3.

Transcription of `PageTable.build_eviction_order` from
exllamav3 `generator/pagetable.py` lines 387-496. `CachePage` keeps only the
fields that function reads. `id(page)` is the original object identity; nothing
here replaces it. The `"lru"` branch is kept so the tree path stays in the same
place, and every scenario below uses `"tree"`.

Consumption of that order follows `allocate_pages`' `next_evictable`
(pagetable.py:553-564) and the prefix / unique loops (566-615): a hash hit
does not pop, a miss pops the next page that is still unreferenced.

Each scenario is the page state produced by the public `PageTable` calls in
`tests/eviction.rs` (initial `access_serial == index`, table counter starts at
`num_pages` and is incremented before it is written onto a claimed page; a hit
keeps `max(old, counter)`). Hashes are stand-ins, not blake2b digests. The
script prints one line per order, indices separated by spaces.
"""

from __future__ import annotations

import heapq
from collections import deque

# exllamav3.constants.PAGE_SIZE. The tree walk only tests kv_position == 0.
PAGE_SIZE = 256


class CachePage:
    """Fields read by `build_eviction_order`. See pagetable.py:45-77."""

    def __init__(
        self,
        page_index: int,
        phash: bytes,
        prev_hash: bytes | None,
        ref_count: int,
        access_serial: int,
        kv_position: int,
    ) -> None:
        self.page_index = page_index
        self.phash = phash
        self.prev_hash = prev_hash
        self.ref_count = ref_count
        self.access_serial = access_serial
        self.kv_position = kv_position
        # Filled in by build_eviction_order. pagetable.py:76, 439.
        self.children: list[CachePage] = []


class PageTable:
    """Just enough of pagetable.py:300-496 to order unreferenced pages."""

    def __init__(self) -> None:
        self.all_pages: list[CachePage] = []
        self.unreferenced_pages: dict[bytes, CachePage] = {}
        self.eviction_policy = "tree"

    def build_eviction_order(self, protected_hashes: set | None = None) -> deque:
        # pagetable.py:387-496
        if protected_hashes is None:
            protected_hashes = set()

        if self.eviction_policy == "lru":
            order = [p for p in self.unreferenced_pages.values() if p.phash not in protected_hashes]
            order.sort(key = lambda p: p.access_serial)
            order += sorted(
                (p for p in self.unreferenced_pages.values() if p.phash in protected_hashes),
                key = lambda p: p.access_serial
            )
            return deque(order)

        order = []
        emitted = set()

        # Class 1: empty pages. pagetable.py:425-433
        empty = [
            p for p in self.unreferenced_pages.values()
            if p.kv_position == 0 and p.phash not in protected_hashes
        ]
        empty.sort(key = lambda p: p.access_serial)
        for p in empty:
            order.append(p)
            emitted.add(id(p))

        # Link pages into trees. Referenced pages participate as (non-evictable) internal nodes so that the
        # unreferenced extensions of a live prefix are still pruned tail-first.
        # pagetable.py:435-457
        page_index = {}
        for p in self.all_pages:
            p.children = []
            page_index[p.phash] = p

        roots = []
        orphan_roots = []
        parent_of = {}
        for p in self.all_pages:
            if p.prev_hash is None:
                roots.append(p)
                continue
            parent = page_index.get(p.prev_hash)
            if parent is None or parent is p:
                orphan_roots.append(p)
            else:
                parent.children.append(p)
                parent_of[id(p)] = parent

        roots.sort(key = lambda p: p.access_serial)
        orphan_roots.sort(key = lambda p: p.access_serial)

        def prune(root):
            # pagetable.py:459-483
            # Emit the tree's evictable pages in the order given by repeatedly removing the oldest current leaf.
            # A page that cannot be evicted (referenced or protected) blocks its ancestors, since evicting an
            # ancestor would orphan a chain that is still in use.
            remaining = {}
            heap = []
            stack = [root]
            while stack:
                p = stack.pop()
                remaining[id(p)] = len(p.children)
                if not p.children:
                    heapq.heappush(heap, (p.access_serial, p.page_index, p))
                stack.extend(p.children)
            while heap:
                _, _, p = heapq.heappop(heap)
                if id(p) not in emitted:
                    if p.ref_count > 0 or p.phash in protected_hashes:
                        continue
                    order.append(p)
                    emitted.add(id(p))
                parent = parent_of.get(id(p))
                if parent is not None:
                    remaining[id(parent)] -= 1
                    if remaining[id(parent)] == 0:
                        heapq.heappush(heap, (parent.access_serial, parent.page_index, parent))

        # Classes 2 and 3. pagetable.py:485-489
        for root in orphan_roots:
            prune(root)
        for root in roots:
            prune(root)

        # Last resort. pagetable.py:491-494
        leftovers = [p for p in self.unreferenced_pages.values() if id(p) not in emitted]
        leftovers.sort(key = lambda p: (p.phash in protected_hashes, p.access_serial))
        order += leftovers

        return deque(order)


def consume(order: deque, steps: list[tuple[str, int | None]]) -> list[int]:
    """Prefix-loop consumption. pagetable.py:553-590, unique pages at 602-615.

    `("evict", None)` is one miss (`next_evictable`). `("hit", index)` is a
    hash hit: the page becomes referenced and the queue is not popped. A later
    miss skips pages claimed after the order was built.
    """
    queue: deque[int] = deque(page.page_index for page in order)
    referenced: set[int] = set()
    taken: list[int] = []
    for kind, index in steps:
        if kind == "hit":
            if index is None:
                raise SystemExit("hit without a page index")
            referenced.add(index)
            taken.append(index)
            continue
        if kind != "evict":
            raise SystemExit(kind)
        while queue and queue[0] in referenced:
            queue.popleft()
        if not queue:
            raise SystemExit("out of pages")
        chosen = queue.popleft()
        referenced.add(chosen)
        taken.append(chosen)
    return taken


def make_table(rows: list[tuple[int, bytes, bytes | None, int, int, int]]) -> PageTable:
    """One row is index, hash, prev hash, ref_count, access_serial, kv_position.

    `ref_count == 0` pages go in `unreferenced_pages`, matching the partition
    `sub_ref` / `add_ref` keep. Dict insertion follows index order, which is
    the stable-sort tie break `all_pages` order gives the original.
    """
    table = PageTable()
    for index, phash, prev_hash, ref_count, access_serial, kv_position in rows:
        if index != len(table.all_pages):
            raise SystemExit(f"page index {index} is not the next slot")
        page = CachePage(index, phash, prev_hash, ref_count, access_serial, kv_position)
        table.all_pages.append(page)
        if ref_count == 0:
            table.unreferenced_pages[phash] = page
    return table


def check_order(name: str, table: PageTable, order: deque) -> list[int]:
    indices = [page.page_index for page in order]
    evictable = sorted(page.page_index for page in table.all_pages if page.ref_count == 0)
    if sorted(indices) != evictable:
        raise SystemExit(f"{name}: {indices} is not the unreferenced set {evictable}")
    return indices


def line(name: str, indices: list[int]) -> None:
    print(name, *indices)


def empty_broken_two_trees() -> None:
    """Pool of 6. Chain A on pages 0-2, chain B on pages 3-4, page 5 never used.

    `publish_page` then rewrites page 0's hash, so page 1's prev no longer
    resolves. Page 2 still names page 1: one broken chain, leaf then orphan
    root. Chain B's root serial (10) is older than page 0's refreshed serial
    (12), so the intact tree is pages 4 then 3, and page 0 is last.
    """
    table = make_table([
        (0, b"H0x", None, 0, 12, PAGE_SIZE),
        (1, b"H1", b"dead-H0", 0, 8, PAGE_SIZE),
        (2, b"H2", b"H1", 0, 9, PAGE_SIZE),
        (3, b"B0", None, 0, 10, PAGE_SIZE),
        (4, b"B1", b"B0", 0, 11, PAGE_SIZE),
        (5, b"E5", None, 0, 5, 0),
    ])
    line("empty_broken_two_trees", check_order("empty_broken_two_trees", table, table.build_eviction_order()))


def branch_two_children() -> None:
    """Pool of 5. Page 0 has two full children; pages 3 and 4 are still empty.

    Leaf serials are 7 then 9, so the older leaf goes before the newer one,
    and the root (serial 8) waits until both leaves are gone.
    """
    table = make_table([
        (0, b"H0", None, 0, 8, PAGE_SIZE),
        (1, b"H1", b"H0", 0, 7, PAGE_SIZE),
        (2, b"H1b", b"H0", 0, 9, PAGE_SIZE),
        (3, b"E3", None, 0, 3, 0),
        (4, b"E4", None, 0, 4, 0),
    ])
    line("branch_two_children", check_order("branch_two_children", table, table.build_eviction_order()))


def referenced_blocks_root() -> None:
    """Same branch, but the root and the newer child are still referenced.

    The unreferenced child is the only page the tree emits. The referenced
    child blocks the root, and neither referenced page is in the order.
    """
    table = make_table([
        (0, b"H0", None, 1, 8, PAGE_SIZE),
        (1, b"H1", b"H0", 0, 7, PAGE_SIZE),
        (2, b"H1b", b"H0", 1, 9, PAGE_SIZE),
        (3, b"E3", None, 0, 3, 0),
        (4, b"E4", None, 0, 4, 0),
    ])
    line("referenced_blocks_root", check_order("referenced_blocks_root", table, table.build_eviction_order()))


def protected_broken_tail() -> None:
    """Pool of 5, no empty page. Same overwrite as `empty_broken_two_trees`.

    Resuming the original chain protects H0 (already gone), H1, and H2.
    The broken tail is deferred to the leftovers. The intact tree is eaten
    first: leaf 4, root 3, then the rewritten root 0. Protected leftovers
    follow, older serial first (pagetable.py:492-494), which is the orphan
    root and then its leaf — not leaf-to-root.

    `protected_broken_consumed` is that raw order walked by one leading miss,
    hits on pages 1 and 2, then two more misses (`new_unique_pages == 2`).
    """
    table = make_table([
        (0, b"H0x", None, 0, 11, PAGE_SIZE),
        (1, b"H1", b"dead-H0", 0, 7, PAGE_SIZE),
        (2, b"H2", b"H1", 0, 8, PAGE_SIZE),
        (3, b"B0", None, 0, 9, PAGE_SIZE),
        (4, b"B1", b"B0", 0, 10, PAGE_SIZE),
    ])
    protected = {b"H0", b"H1", b"H2"}
    raw = table.build_eviction_order(protected)
    line("protected_broken_raw", check_order("protected_broken_raw", table, raw))
    line(
        "protected_broken_consumed",
        consume(raw, [("evict", None), ("hit", 1), ("hit", 2), ("evict", None), ("evict", None)]),
    )
    bare = table.build_eviction_order()
    line("protected_setup_unprotected", check_order("protected_setup_unprotected", table, bare))


def main() -> None:
    empty_broken_two_trees()
    branch_two_children()
    referenced_blocks_root()
    protected_broken_tail()


if __name__ == "__main__":
    main()

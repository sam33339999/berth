#!/usr/bin/env python3
"""Blake2b-128 page-hash vector for berth-cache.

Stdlib only. Run with `python3 -I`. Do not import torch or exllamav3.

Matches exllamav3 `generator/pagetable.py` `_tensor_blake2b_checksum`:
hashlib.blake2b(digest_size=16), previous digest omitted entirely for the
first page (not 16 zero bytes), then this page's 256 token ids as
little-endian int64. That is torch.long / numpy().tobytes() on the
little-endian host the service runs on.

Prints three hex digests:
  page 0 with no previous hash
  page 1 chained onto page 0
  page 0 again but with 16 zero bytes as a fake previous hash
"""

import hashlib

PAGE_SIZE = 256


def page_hash(prev: bytes | None, token_ids: list[int]) -> bytes:
    if len(token_ids) != PAGE_SIZE:
        raise SystemExit(f"page must be {PAGE_SIZE} tokens, got {len(token_ids)}")
    hasher = hashlib.blake2b(digest_size=16)
    if prev is not None:
        hasher.update(prev)
    for token in token_ids:
        hasher.update(int(token).to_bytes(8, "little", signed=True))
    return hasher.digest()


def main() -> None:
    page0 = [i - 1 for i in range(PAGE_SIZE)]  # -1, 0, 1, ..., 254
    page1 = [1000 + i for i in range(PAGE_SIZE)]
    first = page_hash(None, page0)
    second = page_hash(first, page1)
    zero_prev = page_hash(bytes(16), page0)
    print(first.hex())
    print(second.hex())
    print(zero_prev.hex())


if __name__ == "__main__":
    main()

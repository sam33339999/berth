"""Print safetensors tensor headers from an EXL3 checkpoint.

Uses only the Python standard library. Reads the 8-byte header length and the
JSON header of each shard. Does not read tensor bodies, and does not import
torch, safetensors, or exllamav3.

Usage:
    python3 -I dump_headers.py [checkpoint_dir]

Each line is ``name<TAB>dtype<TAB>shape``.
"""

import json
import os
import struct
import sys


def header(path):
    with open(path, "rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise SystemExit(f"{path}: file is shorter than a safetensors header")
        (nbytes,) = struct.unpack("<Q", raw_len)
        payload = handle.read(nbytes)
        if len(payload) != nbytes:
            raise SystemExit(f"{path}: header length {nbytes} runs past end of file")
    return json.loads(payload)


def shards(directory):
    index_path = os.path.join(directory, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        names = sorted(set(weight_map.values()))
        return [os.path.join(directory, name) for name in names]
    single = os.path.join(directory, "model.safetensors")
    if os.path.isfile(single):
        return [single]
    raise SystemExit(f"{directory}: no safetensors index and no model.safetensors")


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "BERTH_27B_DIR", "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3"
    )
    rows = []
    for path in shards(directory):
        for name, meta in header(path).items():
            if name == "__metadata__":
                continue
            shape = "[" + ",".join(str(dim) for dim in meta["shape"]) + "]"
            rows.append((name, meta["dtype"], shape))
    for name, dtype, shape in sorted(rows):
        print(f"{name}\t{dtype}\t{shape}")


if __name__ == "__main__":
    main()

"""Fork reference for prefill EXL3 linears.

The Rust test writes a manifest and the fp16 inputs. This process runs
`LinearEXL3.forward` or `ext.exl3_mgemm` on the 27B weights and writes raw
outputs next to the inputs. Env matches the serve: plain int8, auto f16acc.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("EXL3_HGEMM_F16ACC", "auto")
os.environ.setdefault("EXL3_INT8_GEMV", "2")
os.environ.setdefault("EXL3_INT8_GEMV_MAX_K", "5")
os.environ.setdefault("EXL3_QC_STAGING", "1")

import torch
from safetensors import safe_open

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3 import LinearEXL3


def load_index(model: Path) -> dict[str, str]:
    index = json.loads((model / "model.safetensors.index.json").read_text())
    return index["weight_map"]


def load_tensor(model: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    shard = model / weight_map[name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def read_input(path: Path, rows: int, k_in: int) -> torch.Tensor:
    raw = bytearray(path.read_bytes())
    expect = rows * k_in * 2
    if len(raw) != expect:
        raise SystemExit(f"{path}: {len(raw)} bytes, expected {expect}")
    return torch.frombuffer(raw, dtype=torch.float16).clone().view(rows, k_in)


def write_raw(path: Path, tensor: torch.Tensor) -> None:
    cpu = tensor.detach().contiguous().cpu()
    path.write_bytes(cpu.numpy().tobytes())


def run_linear(model: Path, weight_map: dict[str, str], case: dict, out_dir: Path) -> None:
    prefix = case["prefix"]
    rows = case["rows"]
    trellis = load_tensor(model, weight_map, f"{prefix}.trellis").cuda()
    suh = load_tensor(model, weight_map, f"{prefix}.suh").cuda()
    svh = load_tensor(model, weight_map, f"{prefix}.svh").cuda()
    k_in = trellis.shape[0] * 16
    n = trellis.shape[1] * 16
    out_dtype = torch.float32 if case["out"] == "f32" else torch.float16
    linear = LinearEXL3(
        None,
        k_in,
        n,
        suh=suh,
        svh=svh,
        trellis=trellis,
        mul1=torch.empty(0, dtype=torch.int16, device="cuda"),
        out_dtype=out_dtype,
    )
    x = read_input(out_dir / case["input"], rows, k_in).cuda()
    y = linear.forward(x, {})
    torch.cuda.synchronize()
    write_raw(out_dir / f"{case['id']}.out", y)
    del y, x, linear, trellis, suh, svh
    torch.cuda.empty_cache()


def run_mgemm(model: Path, weight_map: dict[str, str], case: dict, out_dir: Path) -> None:
    rows = case["rows"]
    out_dtype = torch.float32 if case["out"] == "f32" else torch.float16
    parts = []
    for prefix in case["prefixes"]:
        trellis = load_tensor(model, weight_map, f"{prefix}.trellis").cuda()
        suh = load_tensor(model, weight_map, f"{prefix}.suh").cuda()
        svh = load_tensor(model, weight_map, f"{prefix}.svh").cuda()
        parts.append((trellis, suh, svh))
    k_bits = parts[0][0].shape[2] // 16
    k_in = parts[0][0].shape[0] * 16
    widths = [trellis.shape[1] * 16 for trellis, _, _ in parts]
    for trellis, _, _ in parts:
        if trellis.shape[0] * 16 != k_in or trellis.shape[2] // 16 != k_bits:
            raise SystemExit(f"{case['id']}: source shapes disagree")
    width = widths[0]
    for item in widths[1:]:
        width = math.gcd(width, item)
    if width % 128 != 0 or width < 256:
        raise SystemExit(f"{case['id']}: slice width {width}")

    trellis_ptrs = []
    svh_ptrs = []
    targets = []
    offsets = []
    strides = []
    srcs = []
    for index, (trellis, _, svh) in enumerate(parts):
        n = widths[index]
        for n0 in range(0, n, width):
            trellis_ptrs.append(trellis.data_ptr() + (n0 // 16) * 16 * k_bits * trellis.element_size())
            svh_ptrs.append(svh.data_ptr() + n0 * svh.element_size())
            targets.append(index)
            offsets.append(n0)
            strides.append(n)
            srcs.append(index)
    num_slices = len(targets)
    num_src = len(parts)

    x = read_input(out_dir / case["input"], rows, k_in).cuda().view(1, rows, k_in)
    xh = torch.empty((num_src, rows, k_in), dtype=torch.float16, device="cuda")
    outputs = [torch.empty((rows, n), dtype=out_dtype, device="cuda") for n in widths]
    c_host = []
    for target, n0 in zip(targets, offsets):
        out = outputs[target]
        c_host.append(out.data_ptr() + n0 * out.element_size())
    carrier = torch.empty((num_slices, rows, width), dtype=out_dtype, device="cuda")
    ext.exl3_mgemm(
        x,
        torch.tensor(trellis_ptrs, dtype=torch.int64, device="cuda"),
        carrier,
        torch.tensor([suh.data_ptr() for _, suh, _ in parts], dtype=torch.int64, device="cuda"),
        xh,
        torch.tensor(svh_ptrs, dtype=torch.int64, device="cuda"),
        None,
        None,
        k_bits,
        -1,
        False,
        True,
        -1,
        -1,
        0,
        1,
        torch.full((num_slices,), width, dtype=torch.int32, device="cuda"),
        torch.tensor(c_host, dtype=torch.int64, device="cuda"),
        torch.tensor(strides, dtype=torch.int32, device="cuda"),
        torch.tensor(srcs, dtype=torch.int32, device="cuda"),
        num_src,
    )
    torch.cuda.synchronize()
    for index, out in enumerate(outputs):
        write_raw(out_dir / f"{case['id']}.{index}.out", out)
    del outputs, x, xh, carrier, parts
    torch.cuda.empty_cache()


def main() -> None:
    manifest_path = Path(sys.argv[1])
    manifest = json.loads(manifest_path.read_text())
    model = Path(manifest["model"])
    out_dir = Path(manifest["out_dir"])
    weight_map = load_index(model)
    print(f"f16acc_status {ext.hgemm_f16acc_status(0)}", flush=True)
    for case in manifest["cases"]:
        print(f"ref {case['id']}", flush=True)
        if case["kind"] == "linear":
            run_linear(model, weight_map, case, out_dir)
        elif case["kind"] == "mgemm":
            run_mgemm(model, weight_map, case, out_dir)
        else:
            raise SystemExit(f"unknown kind {case['kind']}")
    print("ref done", flush=True)


if __name__ == "__main__":
    main()

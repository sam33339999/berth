#!/usr/bin/env python3
"""Reference outputs of single EXL3 linears from the fork. Needs the GPU.

Do not run while the :8889 serve is up: this loads a second 27B.

For each linear in CASES, builds a seeded fp16 input of shape [rows, in] and
runs exllamav3's `Linear.forward` on it. Writes, per case, little-endian raw
`<name>.x.bin` (fp16) and `<name>.y.bin` (the output dtype), plus one
`meta.json` with shapes, dtypes, the seed, the env flags, and the fork's
commit and `exllamav3_ext` tree hash. `berth-kernels` compares against these.

Run through the serve's environment (`scripts/serve-host.sh` exports):

    python scripts/dump_exl3_reference.py [OUT_DIR]
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from exllamav3 import Config, Model

TARGET = os.environ.get(
    "BERTH_27B_DIR", "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3"
)
FORK = Path(os.environ.get("BERTH_EXL3_SRC", "/home/sam/qwen38-exl3-dflash2/exllamav3"))
SEED = 1234

# (file name, module key, rows). Kernel paths are from the nsys decode pass
# in docs/kernels.md: K=4 at m=1 is int8 GEMV, lm_head K=6 is the cooperative GEMM.
CASES = [
    ("l0_down_m1", "model.language_model.layers.0.mlp.down_proj", 1),
    ("l0_gate_m1", "model.language_model.layers.0.mlp.gate_proj", 1),
    ("l3_o_m1", "model.language_model.layers.3.self_attn.o_proj", 1),
    ("lm_head_m1", "lm_head", 1),
]
ENV_FLAGS = ["EXL3_INT8_GEMV", "EXL3_INT8_GEMV_MAX_K", "EXL3_QC_STAGING", "EXL3_HGEMM_F16ACC"]


def git(*args):
    return subprocess.check_output(["git", "-C", str(FORK), *args], text=True).strip()


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "testdata/exl3_ref")
    out.mkdir(parents=True, exist_ok=True)

    model = Model.from_config(Config.from_directory(TARGET))
    model.load(progressbar=False)

    meta = {
        "target": TARGET,
        "fork_commit": git("rev-parse", "HEAD"),
        "ext_tree": git("rev-parse", "HEAD:exllamav3/exllamav3_ext"),
        "torch": torch.__version__,
        "seed": SEED,
        "env": {k: os.environ.get(k) for k in ENV_FLAGS},
        "cases": [],
    }
    for name, key, rows in CASES:
        linear = model.find_module(key)
        gen = torch.Generator(device="cuda").manual_seed(SEED)
        x = torch.randn((1, rows, linear.in_features), generator=gen, device="cuda",
                        dtype=torch.float32).half()
        with torch.inference_mode():
            y = linear.forward(x, {})
        torch.cuda.synchronize()
        x.cpu().contiguous().numpy().tofile(out / f"{name}.x.bin")
        y.cpu().contiguous().numpy().tofile(out / f"{name}.y.bin")
        meta["cases"].append({
            "name": name, "key": key,
            "x_shape": list(x.shape), "x_dtype": str(x.dtype).removeprefix("torch."),
            "y_shape": list(y.shape), "y_dtype": str(y.dtype).removeprefix("torch."),
            "y_abs_max": float(y.abs().max()), "y_finite": bool(torch.isfinite(y).all()),
        })
        print(name, key, tuple(x.shape), "->", tuple(y.shape), y.dtype, file=sys.stderr)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()

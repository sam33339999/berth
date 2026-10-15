#!/usr/bin/env python3
"""Bit-exact references for the 27B full-attention prefill core.

Fork path, bsz 1, cq3, EXL3_QC_STAGING=1:

  ext.deinterleave_qg
  ext.rope (positions = cache_seqlens, fused bf16 q/k norm)
  ext.quant_cache_paged (in_contiguous)
  paged_attn_triton_prefill  (staging when q_len >= 256, else online dequant)
  ext.mul_sigmoid_

Cases come from testdata/kernels/attn_prefill/meta.json: chunks of 47, 300,
and 1499 tokens at position 0, then a 300-token chunk at position 1499 over
the cache the 1499-token chunk wrote. q_norm and k_norm are the layer-3
checkpoint weights. Other tensors are seeded fp16.

Large outputs go to /tmp/berth-attn-prefill (or the directory on argv). They
are not written under testdata. Exits before touching the GPU if :8889 is
listening.
"""

import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

SEED = 20261015
N_Q = 24
N_KV = 4
HD = 256
PAGE = 256
ROTARY = 64
THETA = 10_000_000.0
EPS = 1e-6
BIAS = 1.0
GROUPS = N_KV * HD // 32
K_WORDS = GROUPS * 3

ROOT = Path(__file__).resolve().parents[3]
FORK = Path(os.environ.get("BERTH_EXL3_SRC", "/home/sam/qwen38-exl3-dflash2/exllamav3"))
MODEL = Path(os.environ.get(
    "BERTH_27B_DIR",
    os.path.expanduser("~/qwen38-exl3-dflash2/models/qwen38-27b-exl3"),
))
SPEC = ROOT / "testdata" / "kernels" / "attn_prefill" / "meta.json"
CUBIN_DIR = ROOT / "target" / "triton-cubins"
BUILD_SCRIPT = ROOT / "scripts" / "build_triton_cubins.py"
Q_NORM = "model.language_model.layers.3.self_attn.q_norm.weight"
K_NORM = "model.language_model.layers.3.self_attn.k_norm.weight"


def port_busy() -> bool:
    text = subprocess.check_output(["ss", "-ltn"], text=True)
    return ":8889 " in text


def reexec_with_runtime_env() -> None:
    if os.environ.get("BERTH_LD_READY") == "1":
        return
    nvidia = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import pathlib, torch\n"
            "base = pathlib.Path(torch.__file__).resolve().parent.parent / 'nvidia'\n"
            "libs = sorted(p for p in base.glob('*/lib') if p.is_dir())\n"
            "print(':'.join(str(p) for p in libs))",
        ],
        text=True,
    ).strip()
    if not nvidia:
        raise SystemExit("torch nvidia lib directories were not found")
    env = os.environ.copy()
    prev = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = nvidia + ((":" + prev) if prev else "")
    env["TRITON_LIBCUDA_PATH"] = env.get("TRITON_LIBCUDA_PATH", "/usr/lib/x86_64-linux-gnu")
    env["EXL3_QC_STAGING"] = "1"
    env["PYTHONPATH"] = env.get("PYTHONPATH", str(FORK))
    env["BERTH_LD_READY"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def save(out: Path, name: str, tensor) -> str:
    import ctypes

    host = tensor.detach().contiguous().cpu()
    raw = ctypes.string_at(host.data_ptr(), host.numel() * host.element_size())
    path = out / name
    path.write_bytes(raw)
    if not any(b != 0 for b in raw[: min(len(raw), 64)]) and raw == bytes(len(raw)):
        raise SystemExit(f"{name} is all zeros")
    return name


def load_ext():
    so = next(FORK.glob("exllamav3_ext*.so"))
    spec = importlib.util.spec_from_file_location("exllamav3_ext", so)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exllamav3_ext"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_triton_paged():
    """Import triton_paged.py without running exllamav3/__init__.py."""
    pkg = FORK / "exllamav3"

    def stub(name: str, path: Path):
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]
        mod.__package__ = name
        sys.modules[name] = mod
        return mod

    stub("exllamav3", pkg)
    stub("exllamav3.modules", pkg / "modules")
    stub("exllamav3.modules.attention_fn", pkg / "modules" / "attention_fn")

    def load(qual: str, path: Path):
        spec = importlib.util.spec_from_file_location(qual, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[qual] = mod
        spec.loader.exec_module(mod)
        return mod

    load(
        "exllamav3.modules.attention_fn.common",
        pkg / "modules" / "attention_fn" / "common.py",
    )
    return load(
        "exllamav3.modules.attention_fn.triton_paged",
        pkg / "modules" / "attention_fn" / "triton_paged.py",
    )


def load_norms(torch):
    from safetensors import safe_open

    index = json.loads((MODEL / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    found = {}
    for key in (Q_NORM, K_NORM):
        shard = MODEL / weight_map[key]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            found[key] = handle.get_tensor(key).contiguous()
    q_norm = found[Q_NORM]
    k_norm = found[K_NORM]
    if q_norm.dtype != torch.bfloat16 or tuple(q_norm.shape) != (HD,):
        raise SystemExit(f"q_norm is {tuple(q_norm.shape)} {q_norm.dtype}")
    if k_norm.dtype != torch.bfloat16 or tuple(k_norm.shape) != (HD,):
        raise SystemExit(f"k_norm is {tuple(k_norm.shape)} {k_norm.dtype}")
    return q_norm.cuda(), k_norm.cuda()


def randn(torch, gen, shape):
    return torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32).to(torch.float16)


def empty_pages(torch, n):
    k = torch.zeros(n, PAGE, K_WORDS, dtype=torch.int32, device="cuda")
    v = torch.zeros(n, PAGE, K_WORDS, dtype=torch.int32, device="cuda")
    sk = torch.zeros(n, PAGE, GROUPS, dtype=torch.float16, device="cuda")
    sv = torch.zeros(n, PAGE, GROUPS, dtype=torch.float16, device="cuda")
    return k, v, sk, sv


def jit_map(fn):
    found = {}
    for _dev, entry in fn.device_caches.items():
        for key, kern in entry[0].items():
            found[key] = kern
    return found


def expect_new_cubin(fn, before: set, filename: str):
    after = jit_map(fn)
    new = [key for key in after if key not in before]
    if len(new) != 1:
        raise SystemExit(f"{filename}: expected 1 new prefill cubin, got {len(new)}")
    blob = after[new[0]].asm["cubin"]
    if not isinstance(blob, (bytes, bytearray)):
        blob = bytes(blob)
    disk = (CUBIN_DIR / filename).read_bytes()
    if bytes(blob) != disk:
        raise SystemExit(f"{filename}: JIT cubin bytes differ from {CUBIN_DIR / filename}")
    return set(after)


def run_chunk(torch, ext, paged, out, name, q_len, past_len, pages, bundle, gen, cache, save_in):
    q_norm, k_norm, inv_freq = bundle
    qg = randn(torch, gen, (1, q_len, N_Q * 2 * HD))
    k = randn(torch, gen, (1, q_len, N_KV, HD))
    v = randn(torch, gen, (1, q_len, N_KV, HD))
    save(out, f"{name}_qg.bin", qg)
    save(out, f"{name}_k.bin", k)
    save(out, f"{name}_v.bin", v)
    if cache is None:
        qk, qv, sk, sv = empty_pages(torch, pages)
    else:
        qk, qv, sk, sv = cache
    if save_in:
        save(out, f"{name}_k_pages_in.bin", qk)
        save(out, f"{name}_v_pages_in.bin", qv)
        save(out, f"{name}_k_scales_in.bin", sk)
        save(out, f"{name}_v_scales_in.bin", sv)
    q = torch.empty(1, q_len, N_Q, HD, dtype=torch.float16, device="cuda")
    g = torch.empty(1, q_len, N_Q * HD, dtype=torch.float16, device="cuda")
    ext.deinterleave_qg(qg, q, g, HD)
    seqlens = torch.tensor([past_len], dtype=torch.int32, device="cuda")
    table = torch.arange(pages, dtype=torch.int32, device="cuda").view(1, pages)
    ext.rope(
        q, q, k, k, inv_freq,
        0, seqlens, None,
        2, 1.0, q_norm, k_norm, EPS, BIAS,
        0.0, 0, 1, 0,
    )
    ext.quant_cache_paged(k, qk, sk, v, qv, sv, seqlens, table, PAGE, q_len, 0.0, True)
    attn = paged.paged_attn_triton_prefill(
        q, None, None, qk, qv, table, seqlens,
        causal=True,
        softmax_scale=HD ** -0.5,
        window_size=-1,
        softcap=0.0,
        sinks=None,
        qc=(sk, sv, 3, 3),
        pre_appended_len=q_len,
        n_kv_heads_override=N_KV,
        max_kv_len=None,
    )
    gated = attn.view(1, q_len, N_Q * HD)
    ext.mul_sigmoid_(gated, g)
    torch.cuda.synchronize()
    save(out, f"{name}_o.bin", gated)
    save(out, f"{name}_k_pages.bin", qk)
    save(out, f"{name}_v_pages.bin", qv)
    save(out, f"{name}_k_scales.bin", sk)
    save(out, f"{name}_v_scales.bin", sv)
    return qk, qv, sk, sv


def main() -> None:
    if len(sys.argv) > 2:
        raise SystemExit("usage: attn_prefill_ref.py [OUT_DIR]")
    if port_busy():
        raise SystemExit("port 8889 is listening; not touching the GPU")
    reexec_with_runtime_env()
    if port_busy():
        raise SystemExit("port 8889 is listening; not touching the GPU")

    import torch

    spec = json.loads(SPEC.read_text())
    if spec["seed"] != SEED:
        raise SystemExit(f"meta seed {spec['seed']} != {SEED}")
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/berth-attn-prefill")
    out.mkdir(parents=True, exist_ok=True)

    ext = load_ext()
    paged = load_triton_paged()
    if paged._qc_staging != 1 or paged._qc_prefill_two_pass_min_q != 256:
        raise SystemExit(
            f"staging {paged._qc_staging} threshold {paged._qc_prefill_two_pass_min_q}"
        )

    q_norm, k_norm = load_norms(torch)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(SEED)
    idx = torch.arange(0, ROTARY, 2, device="cuda", dtype=torch.float32)
    inv_freq = 1.0 / (THETA ** (idx / ROTARY))
    h32 = paged._get_h32(torch.device("cuda"))
    save(out, "h32.bin", h32)
    save(out, "inv_freq.bin", inv_freq)
    save(out, "q_norm.bin", q_norm)
    save(out, "k_norm.bin", k_norm)
    bundle = (q_norm, k_norm, inv_freq)

    fn = paged._paged_attn_prefill_kernel
    seen = set(jit_map(fn))
    produced = {}
    for case in spec["cases"]:
        name = case["name"]
        q_len = int(case["q_len"])
        past = int(case["past_len"])
        pages = int(case["pages"])
        cache = None
        if case.get("cache_in"):
            if name != "q300_at_1499":
                raise SystemExit(f"{name}: cache_in is only the follow-up chunk")
            src = produced["q1499"]
            cache = empty_pages(torch, pages)
            n = src[0].shape[0]
            for dst, src_t in zip(cache, src):
                dst[:n].copy_(src_t)
        before = set(jit_map(fn))
        produced[name] = run_chunk(
            torch, ext, paged, out, name, q_len, past, pages, bundle, gen, cache, bool(case.get("cache_in"))
        )
        if name == "q47":
            seen = expect_new_cubin(fn, before, "_paged_attn_prefill_kernel_qc.cubin")
        elif name == "q300":
            seen = expect_new_cubin(fn, before, "_paged_attn_prefill_kernel_fp16.cubin")
        else:
            new = [key for key in jit_map(fn) if key not in seen]
            if new:
                raise SystemExit(f"{name} compiled an unexpected prefill cubin")

    manifest = {
        "seed": SEED,
        "cases": spec["cases"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    used = torch.cuda.memory_allocated()
    print(f"wrote {out} gpu_allocated {used}")


if __name__ == "__main__":
    main()

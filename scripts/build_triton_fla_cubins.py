#!/usr/bin/env python3
"""Prebuild the Qwen3.8-27B Gated DeltaNet prefill Triton cubins.

The non-BC prefill path (`modules/gated_delta_net.py` forward, seq_len > 1,
history off) launches the slotted causal conv1d kernels in
`modules/gated_delta_net_fn/conv1d.py` and, when seq_len >= num_v_heads (48),
the vendored fla kernels behind `vendor/fla.chunk_gated_delta_rule`.

fla autotunes. This script launches the real fork functions at the 27B shapes
so Triton's on-disk autotune cache (the configs the serve picked) selects
`best_config`, then saves that CompiledKernel's cubin. None pointer arguments
are already dropped, and pointer/integer divisibility attributes match the
launch. A later cache rebench cannot change what berth loads.

Conv constexprs are not autotuned: num_warps=4, dim=10240, K=4, state_size=4
(max_history 0), history off, no bias, transpose_output. Combined BLOCK_S is
next_pow2(seq_len) for seq_len <= 256. seq_len is a runtime integer that
Triton specializes on divisibility by 16, so each BLOCK_S is built twice when
both classes exist and the cubins differ. `build_wide_conv_cubins`, called
after that manifest is written, captures the same conv specializations for
state_size 5..19 (BLOCK_STATE = next_pow2(state_size), history off). Those
files are named with `_ss{state_size}`. state_size 4 and the fla kernels
are not rewritten.

Does not load the model. If anything is listening on :8889 the script exits
before touching the GPU.

    PYTHONPATH=~/qwen38-exl3-dflash2/exllamav3 \\
    TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \\
    ~/qwen38-exl3-dflash2/env/bin/python scripts/build_triton_fla_cubins.py [OUT_DIR]

`OUT_DIR` defaults to `target/triton-cubins/fla`.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# 27B text_config / GatedDeltaNet construction (architecture/qwen3_5.py).
NUM_K_HEADS = 16
NUM_V_HEADS = 48
HEAD_DIM = 128
QKV_DIM = 2 * NUM_K_HEADS * HEAD_DIM + NUM_V_HEADS * HEAD_DIM  # 10240
K_DIM = NUM_K_HEADS * HEAD_DIM  # 2048
V_DIM = NUM_V_HEADS * HEAD_DIM  # 6144
CONV_K = 4
# Cache max_history 0 (profile script and Cache default): state length is K.
STATE_SIZE = 4
CHUNK = 64

TRAILING_SCRATCH = [
    ("global_scratch", "*i8"),
    ("profile_scratch", "*i8"),
]

FLA_KERNELS = [
    "l2norm_fwd_kernel",
    "chunk_local_cumsum_scalar_kernel",
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
    "recompute_w_u_fwd_kernel",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_fwd_kernel_o",
]


def port_8889_listening() -> bool:
    out = subprocess.check_output(["ss", "-ltn"], text=True)
    return ":8889 " in out


def ptx_param_kinds(ptx: str, entry: str) -> list[str]:
    match = re.search(
        rf"\.visible\s+\.entry\s+{re.escape(entry)}\s*\((.*?)\)",
        ptx,
        re.S,
    )
    if match is None:
        raise SystemExit(f"PTX has no .visible .entry {entry}")
    kinds = []
    for line in match.group(1).splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith(".param"):
            continue
        if ".u64" in line and ".ptr" in line:
            kinds.append("ptr")
        elif ".f32" in line:
            kinds.append("f32")
        elif re.search(r"\.u32\b", line):
            kinds.append("i32")
        elif ".u64" in line:
            kinds.append("u64")
        else:
            raise SystemExit(f"{entry}: unhandled PTX param {line}")
    return kinds


def kind_of(ty: str) -> str:
    if ty.startswith("*"):
        return "ptr"
    if ty in ("i32", "u32"):
        return "i32"
    if ty in ("fp32", "f32"):
        return "f32"
    raise SystemExit(f"no PTX kind for Triton type {ty}")


def py_value(value):
    """JSON value for a Triton constexpr. None is a dropped pointer (JSON null)."""
    if hasattr(value, "value") and type(value).__name__ == "constexpr":
        value = value.value
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        return value
    raise SystemExit(f"cannot serialize constexpr {value!r} ({type(value)})")


def unwrap(kernel):
    """Heuristics -> Autotuner -> JITFunction. Return (autotuner or None, jit)."""
    fn = kernel
    auto = None
    for _ in range(6):
        if hasattr(fn, "best_config") and hasattr(fn, "configs") and hasattr(fn, "cache"):
            auto = fn
        if hasattr(fn, "device_caches"):
            return auto, fn
        if not hasattr(fn, "fn"):
            break
        fn = fn.fn
    raise SystemExit(f"could not unwrap {kernel}")


def device_cache(jit):
    import triton
    device = triton.runtime.driver.active.get_current_device()
    cache, _key_cache, _target, _backend, _binder = jit.device_caches[device]
    return cache


def f32_bits(value: float) -> int:
    import torch
    bits = int(torch.tensor(value, dtype=torch.float32).view(torch.int32).item())
    return bits & 0xFFFFFFFF


def record_from_compiled(compiled, grid: str, autotune: dict | None) -> dict:
    meta = compiled.metadata
    if meta.target.arch != 86:
        raise SystemExit(f"{meta.name} targeted sm_{meta.target.arch}, want sm_86")
    if getattr(meta, "num_ctas", 1) != 1:
        raise SystemExit(f"{meta.name} num_ctas {meta.num_ctas} != 1")
    ptx = compiled.asm["ptx"]
    if not isinstance(ptx, str):
        ptx = ptx.decode()
    names = list(compiled.src.fn.arg_names)
    signature = compiled.src.signature
    if list(signature.keys()) != names:
        raise SystemExit(f"{meta.name} signature order {list(signature)} != args {names}")
    params = []
    for name in names:
        ty = signature[name]
        if ty == "constexpr":
            continue
        if not isinstance(ty, str):
            raise SystemExit(f"{meta.name} param {name} type {ty!r}")
        params.append({"name": name, "type": ty})
    kinds = ptx_param_kinds(ptx, meta.name)
    expect = [kind_of(p["type"]) for p in params] + ["ptr", "ptr"]
    if kinds != expect:
        raise SystemExit(
            f"{meta.name} PTX params {kinds} != runtime types plus two scratch pointers {expect}"
        )
    params = params + [{"name": n, "type": t} for n, t in TRAILING_SCRATCH]
    consts = {}
    for key, val in compiled.src.constants.items():
        idx = key[0] if isinstance(key, tuple) else key
        consts[names[idx]] = py_value(val)
    cubin = compiled.asm["cubin"]
    if not isinstance(cubin, (bytes, bytearray)) or len(cubin) == 0:
        raise SystemExit(f"{meta.name} cubin missing")
    return {
        "name": meta.name,
        "shared": int(meta.shared),
        "num_warps": int(meta.num_warps),
        "num_stages": int(meta.num_stages),
        "num_ctas": int(meta.num_ctas),
        "global_scratch_size": int(meta.global_scratch_size),
        "global_scratch_align": int(getattr(meta, "global_scratch_align", 1)),
        "profile_scratch_size": int(meta.profile_scratch_size),
        "profile_scratch_align": int(getattr(meta, "profile_scratch_align", 1)),
        "grid": grid,
        "parameters": params,
        "constexprs": consts,
        "autotune": autotune,
        "cubin_bytes": bytes(cubin),
        "sha256": hashlib.sha256(cubin).hexdigest(),
    }


def autotune_of(kernel) -> dict:
    auto, _jit = unwrap(kernel)
    if auto is None or auto.best_config is None:
        raise SystemExit(f"{kernel} has no best_config after launch")
    cfg = auto.best_config
    return {
        "num_warps": int(cfg.num_warps),
        "num_stages": int(cfg.num_stages),
        "num_ctas": int(cfg.num_ctas),
        "kwargs": {k: py_value(v) for k, v in cfg.kwargs.items()},
    }


def harvest_new(kernel, before: set):
    _auto, jit = unwrap(kernel)
    cache = device_cache(jit)
    new = [cache[k] for k in cache.keys() if k not in before]
    return new


def select_compiled(new, autotune: dict | None, name: str):
    if not new:
        raise SystemExit(f"{name}: launch compiled nothing")
    if autotune is None:
        if len(new) != 1:
            raise SystemExit(f"{name}: expected 1 cubin, got {len(new)}")
        return new[0]
    matched = []
    for compiled in new:
        meta = compiled.metadata
        if meta.name != name:
            continue
        if int(meta.num_warps) != autotune["num_warps"]:
            continue
        if int(meta.num_stages) != autotune["num_stages"]:
            continue
        names = list(compiled.src.fn.arg_names)
        consts = {}
        for key, val in compiled.src.constants.items():
            idx = key[0] if isinstance(key, tuple) else key
            consts[names[idx]] = py_value(val)
        if all(consts.get(k) == v for k, v in autotune["kwargs"].items()):
            matched.append(compiled)
    if len(matched) != 1:
        raise SystemExit(
            f"{name}: {len(matched)} cubins match best_config {autotune} "
            f"out of {len(new)} new compiles"
        )
    return matched[0]


def cache_key_snapshot(kernels) -> dict:
    out = {}
    for kernel in kernels:
        _auto, jit = unwrap(kernel)
        out[id(jit)] = set(device_cache(jit).keys())
    return out


def combined_cases():
    """(BLOCK_S, seq_len, div16) for every combined-kernel specialization."""
    cases = []
    for block_s in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        lo = 1 if block_s == 1 else block_s // 2 + 1
        unaligned = None
        aligned = None
        for seq in range(lo, block_s + 1):
            if seq % 16 == 0:
                aligned = seq
            elif unaligned is None:
                unaligned = seq
        if unaligned is not None:
            cases.append((block_s, unaligned, False))
        if aligned is not None:
            cases.append((block_s, aligned, True))
    return cases


def launch_conv(conv_mod, seq: int):
    import torch
    x = torch.empty(1, QKV_DIM, seq, dtype=torch.bfloat16, device="cuda")
    state = torch.empty(1, QKV_DIM, STATE_SIZE, dtype=torch.bfloat16, device="cuda")
    weight = torch.empty(QKV_DIM, CONV_K, dtype=torch.bfloat16, device="cuda")
    slots = torch.zeros(1, dtype=torch.int32, device="cuda")
    conv_mod.causal_conv1d_update_slotted_triton(
        x, state, slots, weight, None, transpose_output=True, history=False,
    )
    torch.cuda.synchronize()


def main() -> None:
    if len(sys.argv) > 2:
        raise SystemExit("usage: build_triton_fla_cubins.py [OUT_DIR]")
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "target/triton-cubins/fla")

    if port_8889_listening():
        raise SystemExit("port 8889 is listening; not touching the GPU")

    import torch
    import triton

    free, _total = torch.cuda.mem_get_info()
    if free < 1 << 30:
        raise SystemExit(f"GPU has {free} bytes free; refusing (need 1 GiB headroom)")

    from exllamav3.vendor.fla import chunk_gated_delta_rule
    from exllamav3.vendor.fla.l2norm import l2norm_fwd_kernel
    from exllamav3.vendor.fla.cumsum import chunk_local_cumsum_scalar_kernel
    from exllamav3.vendor.fla.gdn_chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
    from exllamav3.vendor.fla.gdn_wy_fast import recompute_w_u_fwd_kernel
    from exllamav3.vendor.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    from exllamav3.vendor.fla.chunk_o import chunk_fwd_kernel_o
    from exllamav3.modules.gated_delta_net_fn import conv1d as conv_mod

    fla = {
        "l2norm_fwd_kernel": l2norm_fwd_kernel,
        "chunk_local_cumsum_scalar_kernel": chunk_local_cumsum_scalar_kernel,
        "chunk_gated_delta_rule_fwd_kkt_solve_kernel": chunk_gated_delta_rule_fwd_kkt_solve_kernel,
        "recompute_w_u_fwd_kernel": recompute_w_u_fwd_kernel,
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
        "chunk_fwd_kernel_o": chunk_fwd_kernel_o,
    }
    grids = {
        "l2norm_fwd_kernel": "l2norm",
        "chunk_local_cumsum_scalar_kernel": "cumsum",
        "chunk_gated_delta_rule_fwd_kkt_solve_kernel": "kkt",
        "recompute_w_u_fwd_kernel": "recompute",
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": "chunk_h",
        "chunk_fwd_kernel_o": "chunk_o",
    }

    # RCP_LN2 and 128**-0.5 are Python f64 literals narrowed to f32 at the
    # kernel boundary. Record those bits; Rust must not recompute them.
    from exllamav3.vendor.fla import RCP_LN2
    eps = 1e-6
    k_scale = HEAD_DIM ** -0.5
    bits = {
        "l2norm_eps": f32_bits(eps),
        "rms_eps": f32_bits(eps),
        "rcp_ln2": f32_bits(RCP_LN2),
        "k_scale": f32_bits(k_scale),
    }
    print(
        f"f32 bits eps={bits['l2norm_eps']:08x} rcp_ln2={bits['rcp_ln2']:08x} "
        f"k_scale={bits['k_scale']:08x}",
        flush=True,
    )

    cache_root = Path(os.path.expanduser("~/.triton/cache"))
    before_mtime = 0.0
    if cache_root.is_dir():
        for path in cache_root.glob("*/*.autotune.json"):
            before_mtime = max(before_mtime, path.stat().st_mtime)

    before = {}
    for name, kernel in fla.items():
        _auto, jit = unwrap(kernel)
        before[name] = set(device_cache(jit).keys())

    print("launch chunk_gated_delta_rule T=64", flush=True)
    q = torch.randn(1, 64, NUM_K_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, 64, NUM_K_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 64, NUM_V_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(1, 64, NUM_V_HEADS, dtype=torch.float32, device="cuda")
    beta = torch.rand(1, 64, NUM_V_HEADS, dtype=torch.bfloat16, device="cuda")
    state = torch.randn(1, NUM_V_HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float32, device="cuda")
    chunk_gated_delta_rule(
        q, k, v, g, beta,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()

    # T is do_not_specialize. A second length must not compile another cubin.
    print("launch chunk_gated_delta_rule T=96 (no new cubin expected)", flush=True)
    keys_after_first = cache_key_snapshot(fla.values())
    q2 = torch.randn(1, 96, NUM_K_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    k2 = torch.randn(1, 96, NUM_K_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    v2 = torch.randn(1, 96, NUM_V_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    g2 = torch.randn(1, 96, NUM_V_HEADS, dtype=torch.float32, device="cuda")
    beta2 = torch.rand(1, 96, NUM_V_HEADS, dtype=torch.bfloat16, device="cuda")
    chunk_gated_delta_rule(
        q2, k2, v2, g2, beta2,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    keys_after_second = cache_key_snapshot(fla.values())
    if keys_after_second != keys_after_first:
        raise SystemExit("fla recompiled for T=96; T is specialized and one cubin is not enough")

    rebenched = False
    if cache_root.is_dir():
        for path in cache_root.glob("*/*.autotune.json"):
            if path.stat().st_mtime > before_mtime + 1e-6:
                rebenched = True
                print(f"autotune cache written: {path}", flush=True)

    records = {}
    for name in FLA_KERNELS:
        kernel = fla[name]
        tuned = autotune_of(kernel)
        compiled = select_compiled(harvest_new(kernel, before[name]), tuned, name)
        rec = record_from_compiled(compiled, grids[name], tuned)
        if rec["num_warps"] != tuned["num_warps"] or rec["num_stages"] != tuned["num_stages"]:
            raise SystemExit(f"{name} metadata does not match best_config {tuned}")
        records[name] = rec
        print(
            f"{name} warps={rec['num_warps']} stages={rec['num_stages']} "
            f"shared={rec['shared']} scratch={rec['global_scratch_size']}/"
            f"{rec['profile_scratch_size']} params={[p['name'] for p in rec['parameters']]} "
            f"constexprs={rec['constexprs']}",
            flush=True,
        )

    conv_kernels = {
        "combined": conv_mod._causal_conv1d_update_slotted_kernel,
        "output": conv_mod._causal_conv1d_update_slotted_output_kernel,
        "state": conv_mod._causal_conv1d_update_slotted_state_kernel,
    }
    # One launch can compile up to three kernels. Snapshot all three caches.
    print("launch conv specializations", flush=True)
    conv_records = {}

    def take_new(kernel, before_keys: set, label: str):
        _auto, jit = unwrap(kernel)
        new = [device_cache(jit)[k] for k in device_cache(jit).keys() if k not in before_keys]
        if len(new) > 1:
            raise SystemExit(f"{label}: {len(new)} new cubins")
        return new

    for block_s, seq, div16 in combined_cases():
        kernel = conv_kernels["combined"]
        _auto, jit = unwrap(kernel)
        before_keys = set(device_cache(jit).keys())
        launch_conv(conv_mod, seq)
        new = take_new(kernel, before_keys, f"combined BLOCK_S={block_s} seq={seq}")
        if len(new) == 1:
            rec = record_from_compiled(new[0], "conv_combined", None)
        else:
            # Same specialization as a cubin already built for this BLOCK_S.
            prev = f"conv_combined_bs{block_s}"
            if prev not in conv_records:
                raise SystemExit(f"combined BLOCK_S={block_s} seq={seq}: no cubin")
            rec = conv_records[prev]
        key = f"conv_combined_bs{block_s}" + ("_div16" if div16 else "")
        conv_records[key] = rec
        print(f"{key} seq={seq} sha={rec['sha256'][:12]} warps={rec['num_warps']}", flush=True)

    for seq, div16 in ((300, False), (512, True)):
        before_conv = {}
        for kind in ("output", "state"):
            _auto, jit = unwrap(conv_kernels[kind])
            before_conv[kind] = set(device_cache(jit).keys())
        launch_conv(conv_mod, seq)
        for kind, grid in (("output", "conv_output"), ("state", "conv_state")):
            new = take_new(conv_kernels[kind], before_conv[kind], f"{kind} seq={seq}")
            base = f"conv_{kind}"
            if len(new) == 1:
                rec = record_from_compiled(new[0], grid, None)
            elif base in conv_records:
                rec = conv_records[base]
            else:
                raise SystemExit(f"{kind} seq={seq}: no cubin")
            key = base + ("_div16" if div16 else "")
            conv_records[key] = rec
            print(f"{key} seq={seq} sha={rec['sha256'][:12]}", flush=True)

    # If the div16 cubin matches the other class, keep one file and point both
    # manifest keys at it. Rust still looks up the key for the live seq_len.
    out.mkdir(parents=True, exist_ok=True)
    manifest_kernels = {}

    def emit(key: str, rec: dict, filename: str) -> None:
        path = out / filename
        path.write_bytes(rec["cubin_bytes"])
        manifest_kernels[key] = {
            "cubin": filename,
            "sha256": rec["sha256"],
            "name": rec["name"],
            "shared": rec["shared"],
            "num_warps": rec["num_warps"],
            "num_stages": rec["num_stages"],
            "num_ctas": rec["num_ctas"],
            "global_scratch_size": rec["global_scratch_size"],
            "global_scratch_align": rec["global_scratch_align"],
            "profile_scratch_size": rec["profile_scratch_size"],
            "profile_scratch_align": rec["profile_scratch_align"],
            "grid": rec["grid"],
            "parameters": rec["parameters"],
            "constexprs": rec["constexprs"],
            "autotune": rec["autotune"],
        }

    for name, rec in records.items():
        emit(name, rec, f"{name}.cubin")

    written_hash = {}
    for key, rec in conv_records.items():
        filename = f"{key}.cubin"
        # Share the file when an earlier key has the same image.
        for prev, prev_hash in written_hash.items():
            if prev_hash == rec["sha256"] and prev.split("_div16")[0] == key.split("_div16")[0]:
                filename = manifest_kernels[prev]["cubin"]
                break
        if filename == f"{key}.cubin":
            (out / filename).write_bytes(rec["cubin_bytes"])
        written_hash[key] = rec["sha256"]
        manifest_kernels[key] = {
            "cubin": filename,
            "sha256": rec["sha256"],
            "name": rec["name"],
            "shared": rec["shared"],
            "num_warps": rec["num_warps"],
            "num_stages": rec["num_stages"],
            "num_ctas": rec["num_ctas"],
            "global_scratch_size": rec["global_scratch_size"],
            "global_scratch_align": rec["global_scratch_align"],
            "profile_scratch_size": rec["profile_scratch_size"],
            "profile_scratch_align": rec["profile_scratch_align"],
            "grid": rec["grid"],
            "parameters": rec["parameters"],
            "constexprs": rec["constexprs"],
            "autotune": rec["autotune"],
        }

    manifest = {
        "arch": "sm_86",
        "triton": triton.__version__,
        "autotune_rebenched": rebenched,
        "f32_bits": bits,
        "shapes": {
            "num_k_heads": NUM_K_HEADS,
            "num_v_heads": NUM_V_HEADS,
            "head_dim": HEAD_DIM,
            "qkv_dim": QKV_DIM,
            "k_dim": K_DIM,
            "v_dim": V_DIM,
            "conv_k": CONV_K,
            "state_size": STATE_SIZE,
            "chunk": CHUNK,
        },
        "hgemm": {
            "owner": "LIN",
            "torch_op": "ext.hgemm",
            "source": "modules/quant/fp16.py LinearFP16.forward, exllamav3_ext/hgemm.cu hgemm_gemmex_impl",
            "note": (
                "b_proj and a_proj are LinearFP16 with out_dtype float. x is fp16, "
                "so forward calls ext.hgemm rather than torch.matmul. "
                "cublasGemmEx CUBLAS_OP_N, CUBLAS_OP_N, size_n=48, size_m=seqlen, "
                "size_k=5120, A and B CUDA_R_16F, C CUDA_R_32F, CUBLAS_COMPUTE_32F, "
                "CUBLAS_GEMM_DEFAULT_TENSOR_OP, alpha 1, beta 0. Runtime weight is "
                "[5120, 48] fp16 (checkpoint stores the transpose). y is [seqlen, 48] "
                "fp32. This script does not build that GEMM."
            ),
        },
        "kernels": manifest_kernels,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out} ({len(manifest_kernels)} kernels, rebenched={rebenched})", flush=True)
    wide_notes = build_wide_conv_cubins(conv_mod, out)
    for note in wide_notes:
        print(note, flush=True)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _conv_share_stem(key: str) -> str:
    """Identity of a conv cubin ignoring the divisibility-16 suffix."""
    return key.replace("_div16", "")


def launch_conv_state(conv_mod, seq: int, state_size: int):
    import torch
    x = torch.empty(1, QKV_DIM, seq, dtype=torch.bfloat16, device="cuda")
    state = torch.empty(1, QKV_DIM, state_size, dtype=torch.bfloat16, device="cuda")
    weight = torch.empty(QKV_DIM, CONV_K, dtype=torch.bfloat16, device="cuda")
    slots = torch.zeros(1, dtype=torch.int32, device="cuda")
    conv_mod.causal_conv1d_update_slotted_triton(
        x, state, slots, weight, None, transpose_output=True, history=False,
    )
    torch.cuda.synchronize()


def _check_wide_conv(rec: dict, state_size: int, kind: str) -> None:
    consts = rec["constexprs"]
    block_state = _next_pow2(state_size)
    if consts.get("state_size") != state_size:
        raise SystemExit(f"{kind} state_size {consts.get('state_size')} != {state_size}")
    if consts.get("conv_kernel_size") != CONV_K or consts.get("dim") != QKV_DIM:
        raise SystemExit(f"{kind} ss{state_size} dim/K constexprs {consts}")
    if kind == "output":
        if "BLOCK_STATE" in consts:
            raise SystemExit(f"output kernel ss{state_size} has BLOCK_STATE")
    else:
        if consts.get("BLOCK_STATE") != block_state:
            raise SystemExit(
                f"{kind} ss{state_size} BLOCK_STATE {consts.get('BLOCK_STATE')} != {block_state}"
            )
        if consts.get("history") is not False:
            raise SystemExit(f"{kind} ss{state_size} history {consts.get('history')}")
    if kind != "state":
        if consts.get("BLOCK_K") != CONV_K:
            raise SystemExit(f"{kind} ss{state_size} BLOCK_K {consts.get('BLOCK_K')}")
        if consts.get("has_bias") is not False or consts.get("transpose_output") is not True:
            raise SystemExit(f"{kind} ss{state_size} bias/transpose {consts}")


def _wide_entry(rec: dict, filename: str) -> dict:
    return {
        "cubin": filename,
        "sha256": rec["sha256"],
        "name": rec["name"],
        "shared": rec["shared"],
        "num_warps": rec["num_warps"],
        "num_stages": rec["num_stages"],
        "num_ctas": rec["num_ctas"],
        "global_scratch_size": rec["global_scratch_size"],
        "global_scratch_align": rec["global_scratch_align"],
        "profile_scratch_size": rec["profile_scratch_size"],
        "profile_scratch_align": rec["profile_scratch_align"],
        "grid": rec["grid"],
        "parameters": rec["parameters"],
        "constexprs": rec["constexprs"],
        "autotune": rec["autotune"],
    }


def build_wide_conv_cubins(conv_mod, out: Path) -> list[str]:
    """Capture conv cubins for state_size 5..19, history off.

    Reloads the manifest main() just wrote and adds `_ss` keys. Does not
    rewrite state_size 4 cubins or the fla kernels. Identical images are
    shared only within one state_size, and only across the `_div16` pair.
    """
    import torch

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    kernels = manifest["kernels"]
    if manifest.get("shapes", {}).get("state_size") != STATE_SIZE:
        raise SystemExit("refusing to change the state_size 4 manifest field")

    conv_kernels = {
        "combined": conv_mod._causal_conv1d_update_slotted_kernel,
        "output": conv_mod._causal_conv1d_update_slotted_output_kernel,
        "state": conv_mod._causal_conv1d_update_slotted_state_kernel,
    }
    notes = []

    def take_new(kernel, before_keys: set, label: str):
        _auto, jit = unwrap(kernel)
        new = [device_cache(jit)[k] for k in device_cache(jit).keys() if k not in before_keys]
        if len(new) > 1:
            raise SystemExit(f"{label}: {len(new)} new cubins")
        return new

    def store(key: str, rec: dict, written_hash: dict) -> None:
        if "_ss" not in key:
            raise SystemExit(f"refusing wide conv key without _ss: {key}")
        filename = f"{key}.cubin"
        stem = _conv_share_stem(key)
        for prev, prev_hash in written_hash.items():
            if prev_hash == rec["sha256"] and _conv_share_stem(prev) == stem:
                filename = kernels[prev]["cubin"]
                break
        if "_ss" not in filename:
            raise SystemExit(f"{key}: refusing to write {filename}")
        path = out / filename
        data = rec["cubin_bytes"]
        if path.is_file() and path.read_bytes() != data:
            raise SystemExit(
                f"{key} cubin bytes differ from the existing {path} "
                f"({path.stat().st_size} on disk, {len(data)} newly compiled). "
                "Refusing to overwrite."
            )
        if filename == f"{key}.cubin":
            path.write_bytes(data)
        written_hash[key] = rec["sha256"]
        kernels[key] = _wide_entry(rec, filename)
        notes.append(f"{key} sha={rec['sha256'][:12]} file={filename}")

    for state_size in range(5, 20):
        conv_records = {}
        written_hash = {}
        for block_s, seq, div16 in combined_cases():
            kernel = conv_kernels["combined"]
            _auto, jit = unwrap(kernel)
            before_keys = set(device_cache(jit).keys())
            launch_conv_state(conv_mod, seq, state_size)
            label = f"combined BLOCK_S={block_s} seq={seq} ss={state_size}"
            new = take_new(kernel, before_keys, label)
            if len(new) == 1:
                rec = record_from_compiled(new[0], "conv_combined", None)
                _check_wide_conv(rec, state_size, "combined")
            else:
                prev = f"conv_combined_bs{block_s}_ss{state_size}"
                if prev not in conv_records:
                    raise SystemExit(f"{label}: no cubin")
                rec = conv_records[prev]
            key = f"conv_combined_bs{block_s}" + ("_div16" if div16 else "") + f"_ss{state_size}"
            conv_records[key] = rec
        for seq, div16 in ((300, False), (512, True)):
            before_conv = {}
            for kind in ("output", "state"):
                _auto, jit = unwrap(conv_kernels[kind])
                before_conv[kind] = set(device_cache(jit).keys())
            launch_conv_state(conv_mod, seq, state_size)
            for kind, grid in (("output", "conv_output"), ("state", "conv_state")):
                label = f"{kind} seq={seq} ss={state_size}"
                new = take_new(conv_kernels[kind], before_conv[kind], label)
                base = f"conv_{kind}_ss{state_size}"
                if len(new) == 1:
                    rec = record_from_compiled(new[0], grid, None)
                    _check_wide_conv(rec, state_size, kind)
                elif base in conv_records:
                    rec = conv_records[base]
                else:
                    raise SystemExit(f"{label}: no cubin")
                key = f"conv_{kind}" + ("_div16" if div16 else "") + f"_ss{state_size}"
                conv_records[key] = rec
        for key, rec in conv_records.items():
            store(key, rec, written_hash)
        torch.cuda.empty_cache()
        used = torch.cuda.memory_allocated()
        if used > 1 << 30:
            raise SystemExit(f"GPU allocation {used} bytes exceeds 1 GiB after state_size {state_size}")

    manifest["conv_state_sizes"] = list(range(5, 20))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return notes


if __name__ == "__main__":
    main()

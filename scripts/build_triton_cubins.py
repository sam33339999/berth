#!/usr/bin/env python3
"""Prebuild the Qwen3.8-27B decode and prefill attention cubins.

Compiles `_paged_attn_decode_split_kernel` and `_paged_attn_decode_combine_kernel`
with the same `triton.compile` arguments `BCAttn._configure` uses
(`exllamav3/modules/attention_fn/bc_attn.py`): signature, constexprs, num_warps,
num_stages. The slot is autoregressive decode, q_len 1, causal, cq3 K and V.
`splits_cap` is not a constexpr.

Also JIT-captures `_paged_attn_prefill_kernel` by calling
`paged_attn_triton_prefill` (`triton_paged.py`) with `EXL3_QC_STAGING=1`,
once for every specialization class the 27B full-attention prefill can
launch while it stays unsplit. Triton specializes each integer runtime
argument of that kernel: value 1 becomes a constexpr (the parameter drops
out of the launch), a value divisible by 16 is a separate cubin with
`tt.divisibility = 16`, and any other value is a plain i32. The specialized
integers are `num_splits`, `q_len`, `kv_append_len`, `num_pages_per_seq`,
`WINDOW_LEFT`, and `WINDOW_RIGHT`. On this path `kv_append_len` equals
`q_len`, both windows are -1, and `num_splits` is 1, so the captured product
is the qc path and the fp16 staging path, each over the q_len classes and
page-count classes those calls can actually take. Cubins are named by that
class and the class key is recorded in manifest.json. Decode cubin bytes
are checked against the files already in the output directory and are not
replaced when they differ.

Does not load the model. `triton.compile` needs a CUDA device; if anything is
listening on :8889 the script exits before touching the GPU.

    PYTHONPATH=~/qwen38-exl3-dflash2/exllamav3 \\
    TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \\
    ~/qwen38-exl3-dflash2/env/bin/python scripts/build_triton_cubins.py [OUT_DIR]

`OUT_DIR` defaults to `target/triton-cubins/`. Checkpoint directory is
`BERTH_27B_DIR` (default `~/qwen38-exl3-dflash2/models/qwen38-27b-exl3`).
Fork is `BERTH_EXL3_SRC` (default `~/qwen38-exl3-dflash2/exllamav3`).
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

# Autoregressive decode. DFlash2's verify window is a different q_len and a
# different cubin; this script only builds the q_len 1 pair.
Q_LEN = 1
# Serve default, profiled in docs/kernels.md. Not in config.json.
K_BITS = 3
V_BITS = 3
# Qwen3_5 Attention() is constructed without these (architecture/qwen3_5.py),
# so Attn.__init__ defaults apply: sliding_window -1, logit_softcapping 0,
# sinks None. Interleaved gate does not enter these kernels.
CAUSAL = True
WINDOW_SIZE = -1
SOFTCAP = 0.0
HAS_SINKS = False

# What BC_Attention::run_gr pushes, in order, before TritonKernel appends the
# two scratch pointers. libtorch/attention.cpp, the regime == 0 launches.
SPLIT_CPP_ARGS = [
    "q", "k_cache", "v_cache", "block_table", "cache_seqlens", "out",
    "partial_o", "partial_ml", "k_scales", "v_scales", "h32",
    "split_len", "num_pages_per_seq", "num_splits", "sinks",
]
COMBINE_CPP_ARGS = [
    "partial_o", "partial_ml", "out", "h32", "num_splits", "sinks",
]
# triton_kernel.cpp: launch() always pushes these after the caller's args.
# Names match the comments there and Triton's own launcher (driver.c).
TRAILING_SCRATCH = [
    ("global_scratch", "*i8"),
    ("profile_scratch", "*i8"),
]

SPLIT_NAME = "_paged_attn_decode_split_kernel"
COMBINE_NAME = "_paged_attn_decode_combine_kernel"


def port_8889_listening() -> bool:
    # Same probe the workspace rule uses. The trailing space avoids :88890.
    out = subprocess.check_output(["ss", "-ltn"], text=True)
    return ":8889 " in out


def git_ext_tree(fork: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(fork), "rev-parse", "HEAD:exllamav3/exllamav3_ext"],
        text=True,
    ).strip()


def read_page_size(fork: Path) -> int:
    text = (fork / "exllamav3" / "constants.py").read_text()
    match = re.search(r"^PAGE_SIZE\s*=\s*(\d+)\s*$", text, re.M)
    if match is None:
        raise SystemExit(f"PAGE_SIZE missing in {fork}/exllamav3/constants.py")
    return int(match.group(1))


def load_triton_paged(fork: Path):
    """Import triton_paged.py without running exllamav3/__init__.py.

    The package init pulls the model stack and the C++ extension. The kernel
    module only needs torch and triton.
    """
    pkg = fork / "exllamav3"

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


def read_shape(model_dir: Path, page_size: int) -> dict:
    cfg = json.loads((model_dir / "config.json").read_text())
    arch = cfg.get("architectures", [])
    if arch != ["Qwen3_5ForConditionalGeneration"]:
        raise SystemExit(f"expected Qwen3_5ForConditionalGeneration, got {arch}")
    text = cfg["text_config"]
    n_q = int(text["num_attention_heads"])
    n_kv = int(text["num_key_value_heads"])
    head_dim = int(text["head_dim"])
    if n_q % n_kv != 0:
        raise SystemExit(f"n_q_heads {n_q} is not a multiple of n_kv_heads {n_kv}")
    if head_dim > 512 or head_dim % 32 != 0:
        raise SystemExit(f"head_dim {head_dim} is outside the cq kernel limits")
    # Same expression Attn.__init__ uses when sm_scale is omitted.
    sm_scale = head_dim ** -0.5
    return {
        "n_q_heads": n_q,
        "n_kv_heads": n_kv,
        "head_dim": head_dim,
        "page_size": page_size,
        "q_len": Q_LEN,
        "k_bits": K_BITS,
        "v_bits": V_BITS,
        "sm_scale": float(sm_scale),
    }


def tile(shape: dict, triton) -> dict:
    """BCAttn._configure's tile math for this q_len. bsz is not a constexpr."""
    hd_pad = triton.next_power_of_2(shape["head_dim"])
    group = shape["n_q_heads"] // shape["n_kv_heads"]
    block_n = max(16, 8192 // hd_pad)
    block_m = triton.next_power_of_2(shape["q_len"])
    block_h = max(16 // block_m, 1)
    block_rows = block_m * block_h
    h_blocks = triton.cdiv(group, block_h)
    return {
        "hd_pad": hd_pad,
        "block_n": block_n,
        "block_m": block_m,
        "block_h": block_h,
        "block_rows": block_rows,
        "h_blocks": h_blocks,
        "group": group,
    }


def compile_kernel(fn, signature: dict, constexprs: dict, num_warps: int, num_stages: int):
    """Same steps as bc_attn._compile_kernel, without ext.TritonKernel.

    The `:16` suffix is how that function forwards the JIT's divisibility
    specialization. These two signatures do not use it; the loop stays so a
    later signature that does is compiled the same way.
    """
    import triton
    from triton.compiler import ASTSource

    attrs = {}
    sig = {}
    for name, ty in signature.items():
        if isinstance(ty, str) and ty.endswith(":16"):
            sig[name] = ty[:-3]
            attrs[(fn.arg_names.index(name),)] = [["tt.divisibility", 16]]
        else:
            sig[name] = ty
    import torch
    with torch.cuda.device(torch.device("cuda", 0)):
        src = ASTSource(fn=fn, signature=sig, constexprs=constexprs, attrs=attrs)
        return triton.compile(src, options={"num_warps": num_warps, "num_stages": num_stages})


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
        elif re.search(r"\.u32\b", line):
            kinds.append("i32")
        elif ".u64" in line:
            kinds.append("u64")
        else:
            kinds.append(line)
    return kinds


def kind_of(ty: str) -> str:
    if ty.startswith("*"):
        return "ptr"
    if ty == "i32":
        return "i32"
    raise SystemExit(f"no PTX kind for Triton type {ty}")


def runtime_params(fn, signature: dict) -> list[dict]:
    missing = [n for n in fn.arg_names if n not in signature]
    extra = [n for n in signature if n not in fn.arg_names]
    if missing or extra:
        raise SystemExit(f"{fn.__name__} signature drifted: missing {missing}, extra {extra}")
    params = []
    for name in fn.arg_names:
        ty = signature[name]
        if ty == "constexpr":
            continue
        params.append({"name": name, "type": ty})
    return params


def check_against_cpp(fn_name: str, params: list[dict], cpp_args: list[str]) -> None:
    got = [p["name"] for p in params]
    if got != cpp_args:
        raise SystemExit(
            f"{fn_name} runtime params {got} != attention.cpp arg order {cpp_args}"
        )


def check_against_ptx(entry: str, params: list[dict], ptx: str) -> list[dict]:
    """Return the launch list TritonKernel passes, including the two scratch pointers.

    Triton's cubin has those two pointer params even when both scratch sizes are 0.
    A newer Triton that appends something else will not match triton_kernel.cpp,
    which always pushes exactly two nulls. Fail instead of writing a list the
    C++ launcher does not pass.
    """
    kinds = ptx_param_kinds(ptx, entry)
    expect = [kind_of(p["type"]) for p in params] + ["ptr", "ptr"]
    if kinds != expect:
        raise SystemExit(
            f"{entry} PTX params {kinds} != runtime types plus two scratch pointers {expect}"
        )
    return params + [{"name": n, "type": t} for n, t in TRAILING_SCRATCH]


def programs_for(shape: dict, geom: dict, bsz: int = 1) -> int:
    return bsz * shape["n_kv_heads"] * geom["h_blocks"]


def splits_cap_for(programs: int, sm_count: int) -> int:
    # bc_attn._configure: max(1, min(2 * sm_count // programs, 128))
    return max(1, min((2 * sm_count) // programs, 128))


def grid_note_split(shape: dict, geom: dict, num_warps: int, sm_count: int) -> str:
    programs = programs_for(shape, geom)
    cap = splits_cap_for(programs, sm_count)
    return (
        "programs = bsz * n_kv_heads * cdiv(n_q_heads / n_kv_heads, BLOCK_H). "
        "splits_cap = max(1, min(2 * sm_count / programs, 128)); not a constexpr. "
        "BC_Attention::run_gr launches k_split at grid (programs, splits_cap, 1) "
        "so the captured grid stays put; splits at or above the live count exit. "
        "TritonKernel::launch uses block (32 * num_warps, 1, 1). "
        "Live num_splits and split_len come from attention.cpp split_config: "
        "bound = bt_width * page_size + q_len, "
        "num_splits = max(1, min(splits_cap, ceil(bound / (4 * BLOCK_N)))), "
        "split_len = ceil(ceil(bound / num_splits) / BLOCK_N) * BLOCK_N. "
        f"bsz 1 on this device (sm_count {sm_count}): programs {programs}, "
        f"splits_cap {cap}, grid ({programs}, {cap}, 1), "
        f"block ({32 * num_warps}, 1, 1)."
    )


def grid_note_combine(shape: dict, geom: dict, num_warps: int, sm_count: int) -> str:
    programs = programs_for(shape, geom)
    return (
        "Same programs as the split kernel. "
        "BC_Attention::run_gr launches k_combine at grid (programs, 1, 1). "
        "FINAL is false on this cubin, so the split kernel always writes partials "
        "and the combine kernel always runs, including when the live split count is 1. "
        "TritonKernel::launch uses block (32 * num_warps, 1, 1). "
        f"bsz 1 on this device (sm_count {sm_count}): programs {programs}, "
        f"grid ({programs}, 1, 1), block ({32 * num_warps}, 1, 1)."
    )


def jsonable_constexprs(consts: dict) -> dict:
    out = {}
    for key, value in consts.items():
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            out[key] = value
        elif isinstance(value, float):
            out[key] = value
        else:
            raise SystemExit(f"constexpr {key} has unsupported type {type(value)}")
    return out


def main() -> None:
    if len(sys.argv) > 2:
        raise SystemExit("usage: build_triton_cubins.py [OUT_DIR]")
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "target/triton-cubins")

    if port_8889_listening():
        raise SystemExit(
            "port 8889 is listening; not touching the GPU. "
            "Stop the serve before building cubins."
        )
    if os.environ.get("BERTH_CUBIN_ENV") != "1":
        reexec_with_runtime_env()

    fork = Path(os.environ.get(
        "BERTH_EXL3_SRC", os.path.expanduser("~/qwen38-exl3-dflash2/exllamav3")
    )).resolve()
    model_dir = Path(os.environ.get(
        "BERTH_27B_DIR",
        os.path.expanduser("~/qwen38-exl3-dflash2/models/qwen38-27b-exl3"),
    )).resolve()
    if not (fork / "exllamav3" / "modules" / "attention_fn" / "triton_paged.py").is_file():
        raise SystemExit(f"fork is missing triton_paged.py: {fork}")
    if not (model_dir / "config.json").is_file():
        raise SystemExit(f"checkpoint config.json missing: {model_dir}")

    ext_tree = git_ext_tree(fork)
    page_size = read_page_size(fork)
    shape = read_shape(model_dir, page_size)

    # Snapshot cache paths first, so a cubin this process writes does not count
    # as a pre-existing bc_attn compile.
    fork_marker = str(fork)
    preexisting = {}
    cache_root = Path(os.path.expanduser("~/.triton/cache"))
    if cache_root.is_dir():
        for name in (SPLIT_NAME, COMBINE_NAME):
            preexisting[name] = [
                p for p in cache_root.glob(f"*/{name}.cubin") if p.is_file()
            ]

    import torch
    import triton
    import triton.knobs as knobs

    free, _total = torch.cuda.mem_get_info()
    if free < 1 << 30:
        raise SystemExit(f"GPU has {free} bytes free; refusing to compile (need 1 GiB headroom)")

    paged = load_triton_paged(fork)
    window_left, window_right = paged._normalize_window(WINDOW_SIZE)
    geom = tile(shape, triton)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count

    cache_t = "*i32"  # k_bits/v_bits > 0
    split_sig = {
        "q": "*fp16", "k_cache": cache_t, "v_cache": cache_t,
        "block_table": "*i32", "cache_seqlens": "*i32", "out": "*fp16",
        "partial_o": "*fp32", "partial_ml": "*fp32",
        "k_scales": "*fp16", "v_scales": "*fp16", "h32": "*fp16",
        "split_len": "i32", "num_pages_per_seq": "i32", "num_splits": "i32",
        "sinks": "*fp32",
    } | {n: "constexpr" for n in (
        "QCK", "QCV", "q_len", "kv_append_len", "n_q_heads", "n_kv_heads",
        "page_size", "head_dim", "HD_PAD", "scale", "CAUSAL", "WINDOW_LEFT", "WINDOW_RIGHT",
        "SOFTCAP", "FINAL", "HAS_SINKS", "BLOCK_M", "BLOCK_H", "BLOCK_ROWS", "BLOCK_N")}
    split_consts = dict(
        QCK=shape["k_bits"], QCV=shape["v_bits"],
        q_len=shape["q_len"], kv_append_len=shape["q_len"],
        n_q_heads=shape["n_q_heads"], n_kv_heads=shape["n_kv_heads"],
        page_size=shape["page_size"], head_dim=shape["head_dim"], HD_PAD=geom["hd_pad"],
        scale=float(shape["sm_scale"]),
        CAUSAL=bool(CAUSAL), WINDOW_LEFT=window_left, WINDOW_RIGHT=window_right,
        SOFTCAP=float(SOFTCAP or 0.0), FINAL=False, HAS_SINKS=False,
        BLOCK_M=geom["block_m"], BLOCK_H=geom["block_h"], BLOCK_ROWS=geom["block_rows"],
        BLOCK_N=geom["block_n"],
    )
    combine_sig = {
        "partial_o": "*fp32", "partial_ml": "*fp32", "out": "*fp16", "h32": "*fp16",
        "num_splits": "i32", "sinks": "*fp32",
    } | {n: "constexpr" for n in (
        "QCV", "HAS_SINKS", "q_len", "n_q_heads", "n_kv_heads", "head_dim", "HD_PAD",
        "BLOCK_M", "BLOCK_H", "BLOCK_ROWS")}
    combine_consts = dict(
        QCV=shape["v_bits"], HAS_SINKS=HAS_SINKS, q_len=shape["q_len"],
        n_q_heads=shape["n_q_heads"], n_kv_heads=shape["n_kv_heads"],
        head_dim=shape["head_dim"], HD_PAD=geom["hd_pad"],
        BLOCK_M=geom["block_m"], BLOCK_H=geom["block_h"], BLOCK_ROWS=geom["block_rows"],
    )

    events = []

    def listener(*, src, metadata, metadata_group, times, cache_hit):
        events.append({
            "name": metadata["name"],
            "cache_hit": cache_hit,
            "shared": metadata["shared"],
            "num_warps": metadata["num_warps"],
        })

    knobs.compilation.listener = listener

    specs = [
        (paged._paged_attn_decode_split_kernel, split_sig, split_consts, 4, 2, SPLIT_CPP_ARGS, "split"),
        (paged._paged_attn_decode_combine_kernel, combine_sig, combine_consts, 4, 1, COMBINE_CPP_ARGS, "combine"),
    ]
    compiled = []
    for fn, sig, consts, num_warps, num_stages, cpp_args, kind in specs:
        ck = compile_kernel(fn, sig, consts, num_warps, num_stages)
        if ck.metadata.target.arch != 86:
            raise SystemExit(f"{fn.__name__} targeted sm_{ck.metadata.target.arch}, want sm_86")
        if ck.metadata.global_scratch_size or ck.metadata.profile_scratch_size:
            raise SystemExit(
                f"{fn.__name__} scratch sizes "
                f"{ck.metadata.global_scratch_size}/{ck.metadata.profile_scratch_size}; "
                "TritonKernel passes null and requires both to be 0"
            )
        if ck.metadata.num_warps != num_warps:
            raise SystemExit(
                f"{fn.__name__} metadata num_warps {ck.metadata.num_warps} != requested {num_warps}"
            )
        params = runtime_params(fn, sig)
        check_against_cpp(fn.__name__, params, cpp_args)
        ptx = ck.asm["ptx"]
        if not isinstance(ptx, str):
            ptx = ptx.decode()
        launch_params = check_against_ptx(ck.metadata.name, params, ptx)
        cubin = ck.asm["cubin"]
        if not isinstance(cubin, (bytes, bytearray)) or len(cubin) == 0:
            raise SystemExit(f"{fn.__name__} cubin is empty")
        note = grid_note_split(shape, geom, num_warps, sm_count) if kind == "split" \
            else grid_note_combine(shape, geom, num_warps, sm_count)
        compiled.append({
            "fn": fn.__name__,
            "cubin_name": f"{fn.__name__}.cubin",
            "cubin": bytes(cubin),
            "record": {
                "cubin": f"{fn.__name__}.cubin",
                "name": ck.metadata.name,
                "num_warps": ck.metadata.num_warps,
                "num_stages": num_stages,
                "shared": ck.metadata.shared,
                "global_scratch_size": ck.metadata.global_scratch_size,
                "profile_scratch_size": ck.metadata.profile_scratch_size,
                "parameters": launch_params,
                "constexprs": jsonable_constexprs(consts),
                "grid": note,
                "triton_version": triton.__version__,
                "ext_tree": ext_tree,
            },
        })

    used = torch.cuda.memory_allocated()
    if used > 1 << 30:
        raise SystemExit(f"GPU allocation {used} bytes exceeds 1 GiB")

    out.mkdir(parents=True, exist_ok=True)
    kernels = {}
    for item in compiled:
        dest = out / item["cubin_name"]
        if dest.is_file() and dest.read_bytes() != item["cubin"]:
            raise SystemExit(
                f"{item['fn']} cubin bytes differ from the existing {dest} "
                f"({dest.stat().st_size} on disk, {len(item['cubin'])} newly compiled). "
                "Refusing to overwrite decode cubins."
            )
        dest.write_bytes(item["cubin"])
        kernels[item["fn"]] = item["record"]
        prior = []
        for path in preexisting.get(item["fn"], []):
            src = path.with_suffix(".source")
            if not src.is_file():
                continue
            if fork_marker not in src.read_text(errors="replace")[:4000]:
                continue
            if path.read_bytes() == item["cubin"]:
                meta = json.loads(path.with_suffix(".json").read_text())
                prior.append({
                    "path": str(path),
                    "name": meta.get("name"),
                    "shared": meta.get("shared"),
                    "num_warps": meta.get("num_warps"),
                })
        item["prior"] = prior

    manifest = {
        "triton_version": triton.__version__,
        "ext_tree": ext_tree,
        "fork": str(fork),
        "model_dir": str(model_dir),
        "arch": 86,
        "shape": {
            "n_q_heads": shape["n_q_heads"],
            "n_kv_heads": shape["n_kv_heads"],
            "head_dim": shape["head_dim"],
            "page_size": shape["page_size"],
            "q_len": shape["q_len"],
            "k_bits": shape["k_bits"],
            "v_bits": shape["v_bits"],
            "causal": CAUSAL,
            "sm_scale": shape["sm_scale"],
        },
        "kernels": kernels,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if len(events) != len(compiled):
        raise SystemExit(f"compiler listener saw {len(events)} events for {len(compiled)} kernels")

    # Prefill JIT runs after the decode listener check so those two events
    # stay the decode compiles. The listener is cleared first.
    knobs.compilation.listener = None
    prefill_notes = build_prefill_cubins(paged, shape, out, fork_marker, ext_tree, triton.__version__)

    print(f"wrote {out}")
    for item, ev in zip(compiled, events):
        rec = item["record"]
        print(
            f"{item['fn']}: symbol {rec['name']} warps {rec['num_warps']} "
            f"stages {rec['num_stages']} shared {rec['shared']} "
            f"params {len(rec['parameters'])} bytes {len(item['cubin'])} "
            f"cache_hit {ev['cache_hit']}"
        )
        if item["prior"]:
            for hit in item["prior"]:
                print(
                    f"  cache bytes match {hit['path']} "
                    f"name {hit['name']} shared {hit['shared']} warps {hit['num_warps']}"
                )
        else:
            print("  no pre-existing fork cache cubin with the same bytes")
        print(f"  cross-check: parameter list matches attention.cpp and PTX ({len(rec['parameters'])} launch slots)")
    for note in prefill_notes:
        print(note)


# Runtime pointer args of _paged_attn_prefill_kernel, in definition order.
# Triton appends the two scratch pointers at launch. The integer runtime args
# follow these and are listed in SPECIALIZED_INTS; a class of eq1 drops that
# integer out of this list.
_PREFILL_PREFIX = [
    "q", "k_cache", "v_cache", "block_table", "cache_seqlens", "out",
    "partial_o", "partial_ml", "k_scales", "v_scales", "h32",
    "k_new", "v_new", "sinks",
]

# Integer runtime arguments, definition order. native_specialize_impl (Triton
# 3.7.1, specialize=True, align=True, which is the default on this kernel):
#   value == 1            -> constexpr 1, argument omitted
#   value % 16 == 0       -> i32 with tt.divisibility 16  (Python modulo)
#   otherwise             -> plain i32
# Confirmed against native_specialize_impl for 0, 1, -1, -16, 6, 16, 256.
SPECIALIZED_INTS = [
    "num_splits",
    "q_len",
    "kv_append_len",
    "num_pages_per_seq",
    "WINDOW_LEFT",
    "WINDOW_RIGHT",
]

# One representative call per reachable unsplit class.
# qc is q_len < 256; staging is q_len >= 256. kv_append_len is q_len and both
# windows are -1, so those classes ride along. num_splits stays 1 because
# pages * 256 + q_len < 8192 for every row. q_len == 1 is the decode kernel
# (prefill starts at q_len > 16). A staging chunk cannot have a 1-page table
# unless q_len == 256, which is the div16 class, so (staging, q other, pages
# eq1) is not a call this path can make. Split counts above 1 are a different
# constexpr (IS_SPLIT) plus the combine kernel; they are not captured here.
PREFILL_CAPTURES = [
    # qc, q_len other
    (47, 1),
    (47, 6),
    (47, 16),
    # qc, q_len div16 (240 % 16 == 0 and 240 < 256)
    (240, 1),
    (240, 6),
    (240, 16),
    # staging, q_len other
    (300, 2),
    (300, 16),
    # staging, q_len div16. 256 is the only staging length that fits in 1 page.
    (256, 1),
    (1280, 6),
    (1280, 16),
]


def reexec_with_runtime_env() -> None:
    """Set the serve's loader path and EXL3_QC_STAGING, then replace this process.

    LD_LIBRARY_PATH is read by the dynamic linker when the extension .so is
    loaded. The prefill staging path imports that .so. The child runs this
    script once; BERTH_CUBIN_ENV stops a second exec.
    """
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
    env["BERTH_CUBIN_ENV"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


def _unwrap_constexpr(value):
    if type(value).__name__ == "constexpr" and hasattr(value, "value"):
        return value.value
    return value


def _type_name(ty) -> str:
    text = ty if isinstance(ty, str) else str(ty)
    text = text.strip()
    aliases = {
        "*fp16": "*fp16",
        "*f16": "*fp16",
        "*i32": "*i32",
        "*i64": "*i64",
        "i32": "i32",
        "i64": "i64",
        "*fp32": "*fp32",
        "*f32": "*fp32",
        "*i8": "*i8",
        "*u8": "*i8",
    }
    if text not in aliases:
        raise SystemExit(f"unrecognized Triton type {text!r}")
    return aliases[text]


def _constexprs_from(ck, fn) -> dict:
    out = {}
    for key, value in ck.src.constants.items():
        if isinstance(key, tuple):
            idx = key[0]
        elif isinstance(key, int):
            idx = key
        else:
            raise SystemExit(f"constexpr key {key!r} is not an argument index")
        name = fn.arg_names[idx]
        out[name] = _unwrap_constexpr(value)
    return out


def _runtime_from(ck, fn, expected: list[str]) -> list[dict]:
    sig = ck.src.signature
    params = []
    for name in fn.arg_names:
        if name not in sig:
            continue
        ty = sig[name]
        text = ty if isinstance(ty, str) else str(ty)
        if text == "constexpr" or "constexpr" in text:
            continue
        params.append({"name": name, "type": _type_name(text)})
    got = [p["name"] for p in params]
    if got != expected:
        raise SystemExit(f"prefill runtime params {got} != {expected}")
    return params


def _expect_const(consts: dict, key: str, want) -> None:
    got = consts.get(key)
    if isinstance(want, float):
        if got is None or abs(float(got) - want) > 1e-12:
            raise SystemExit(f"constexpr {key}={got!r} want {want!r}; all={consts}")
        return
    if got != want:
        raise SystemExit(f"constexpr {key}={got!r} want {want!r}; all={consts}")


def int_class(value: int) -> str:
    """Triton's class for one specialized integer. See SPECIALIZED_INTS."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"specialization value {value!r} is not an int")
    if value == 1:
        return "eq1"
    if value % 16 == 0:
        return "div16"
    return "other"


def prefill_class_key(path: str, classes: dict) -> str:
    """Manifest key. `path` is `qc` or `fp16`. Classes are eq1, div16, or other."""
    short = [
        ("ns", "num_splits"),
        ("q", "q_len"),
        ("kv", "kv_append_len"),
        ("p", "num_pages_per_seq"),
        ("wl", "WINDOW_LEFT"),
        ("wr", "WINDOW_RIGHT"),
    ]
    parts = [f"_paged_attn_prefill_kernel_{path}"]
    for tag, name in short:
        kind = classes[name]
        if kind not in ("eq1", "div16", "other"):
            raise SystemExit(f"{name} class {kind!r} is not eq1, div16, or other")
        parts.append(f"{tag}-{kind}")
    return "_".join(parts)


def _classes_for(q_len: int, pages: int) -> tuple[str, dict]:
    """Class of one capture. kv_append_len is q_len; windows are -1; splits are 1."""
    if pages < 1 or q_len < 1:
        raise SystemExit(f"bad capture q_len {q_len} pages {pages}")
    if q_len > pages * 256:
        raise SystemExit(f"q_len {q_len} does not fit in {pages} pages")
    bound = pages * 256 + q_len
    if bound >= 8192:
        raise SystemExit(f"q_len {q_len} pages {pages} wants splits (bound {bound})")
    path = "fp16" if q_len >= 256 else "qc"
    classes = {
        "num_splits": int_class(1),
        "q_len": int_class(q_len),
        "kv_append_len": int_class(q_len),
        "num_pages_per_seq": int_class(pages),
        "WINDOW_LEFT": int_class(-1),
        "WINDOW_RIGHT": int_class(-1),
    }
    return path, classes


def _div16_indexes(attrs) -> set[int]:
    found = set()
    for key, val in attrs.items():
        if isinstance(key, tuple) and len(key) == 1 and isinstance(key[0], int):
            idx = key[0]
        elif isinstance(key, int):
            idx = key
        else:
            raise SystemExit(f"unexpected Triton attr key {key!r}")
        div16 = False
        if isinstance(val, (list, tuple)):
            for item in val:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) >= 2
                    and item[0] == "tt.divisibility"
                    and int(item[1]) == 16
                ):
                    div16 = True
        elif val not in ([], ()):
            raise SystemExit(f"unexpected Triton attr {key!r}={val!r}")
        if div16:
            found.add(idx)
        elif val not in ([], ()):
            raise SystemExit(f"unexpected Triton attr {key!r}={val!r}")
    return found


def _jit_keys(fn) -> dict:
    found = {}
    for _dev, entry in fn.device_caches.items():
        for key, kern in entry[0].items():
            found[key] = kern
    return found


def _capture_prefill(paged, shape: dict, q_len: int, pages: int):
    """One real paged_attn_triton_prefill call. Returns the new CompiledKernel."""
    import torch

    fn = paged._paged_attn_prefill_kernel
    before = set(_jit_keys(fn))
    device = torch.device("cuda", 0)
    n_q = shape["n_q_heads"]
    n_kv = shape["n_kv_heads"]
    hd = shape["head_dim"]
    page = shape["page_size"]
    groups = n_kv * hd // 32
    words = groups * shape["k_bits"]
    q = torch.empty((1, q_len, n_q, hd), dtype=torch.float16, device=device)
    k_cache = torch.zeros((pages, page, words), dtype=torch.int32, device=device)
    v_cache = torch.zeros_like(k_cache)
    k_scales = torch.zeros((pages, page, groups), dtype=torch.float16, device=device)
    v_scales = torch.zeros_like(k_scales)
    block_table = torch.arange(pages, dtype=torch.int32, device=device).view(1, pages)
    cache_seqlens = torch.zeros((1,), dtype=torch.int32, device=device)
    paged.paged_attn_triton_prefill(
        q, None, None, k_cache, v_cache, block_table, cache_seqlens,
        causal=True,
        softmax_scale=shape["sm_scale"],
        window_size=-1,
        softcap=0.0,
        sinks=None,
        qc=(k_scales, v_scales, shape["k_bits"], shape["v_bits"]),
        pre_appended_len=q_len,
        n_kv_heads_override=n_kv,
        max_kv_len=None,
    )
    torch.cuda.synchronize()
    after = _jit_keys(fn)
    new = [key for key in after if key not in before]
    if len(new) != 1:
        raise SystemExit(
            f"q_len {q_len} pages {pages}: expected 1 new prefill cubin, got {len(new)}"
        )
    ck = after[new[0]]
    del q, k_cache, v_cache, k_scales, v_scales, block_table, cache_seqlens
    return ck


def _cache_matches(cubin: bytes, fork_marker: str) -> list[dict]:
    cache_root = Path(os.path.expanduser("~/.triton/cache"))
    hits = []
    if not cache_root.is_dir():
        return hits
    name = "_paged_attn_prefill_kernel"
    for path in cache_root.glob(f"*/{name}.cubin"):
        if not path.is_file() or path.read_bytes() != cubin:
            continue
        src = path.with_suffix(".source")
        text = src.read_text(errors="replace")[:8000] if src.is_file() else ""
        if fork_marker not in text and "triton_paged.py" not in text:
            continue
        meta = json.loads(path.with_suffix(".json").read_text())
        hits.append({
            "path": str(path),
            "name": meta.get("name"),
            "shared": meta.get("shared"),
            "num_warps": meta.get("num_warps"),
            "num_stages": meta.get("num_stages"),
        })
    return hits


def _sig_text(ck, name: str):
    if name not in ck.src.signature:
        return None
    ty = ck.src.signature[name]
    return ty if isinstance(ty, str) else str(ty)


def _is_runtime_sig(ck, name: str) -> bool:
    """True when the argument is still passed at launch.

    A value specialized to constexpr stays in `src.signature` with type
    `constexpr`. That is not a runtime parameter.
    """
    text = _sig_text(ck, name)
    if text is None:
        return False
    return "constexpr" not in text


def _record_prefill(ck, fn, consts_expect: dict, classes: dict, key: str, q_len: int, pages: int, ext_tree: str, triton_version: str):
    consts = _constexprs_from(ck, fn)
    for name, want in consts_expect.items():
        _expect_const(consts, name, want)
    div16 = _div16_indexes(ck.src.attrs)
    for name in SPECIALIZED_INTS:
        kind = classes[name]
        idx = fn.arg_names.index(name)
        runtime = _is_runtime_sig(ck, name)
        if kind == "eq1":
            _expect_const(consts, name, 1)
            if runtime:
                raise SystemExit(
                    f"{key}: {name} is eq1 but still a runtime parameter ({_sig_text(ck, name)!r})"
                )
            if idx in div16:
                raise SystemExit(f"{key}: {name} is eq1 but has a divisibility attr")
        else:
            if name in consts:
                raise SystemExit(f"{key}: {name} class {kind} was baked as {consts[name]!r}")
            if not runtime:
                raise SystemExit(f"{key}: {name} class {kind} is missing from the runtime signature")
            got_ty = _sig_text(ck, name)
            if _type_name(got_ty) != "i32":
                raise SystemExit(f"{key}: {name} type {got_ty!r} != i32")
            got = idx in div16
            if kind == "div16" and not got:
                raise SystemExit(f"{key}: {name} is div16 but Triton did not set tt.divisibility 16")
            if kind == "other" and got:
                raise SystemExit(f"{key}: {name} is other but Triton set tt.divisibility 16")
    # Pointers from torch.empty / arange are 16-byte aligned, so every pointer
    # argument of the captured cubin is the div16 pointer class. CUDA device
    # allocs used at launch are aligned the same way. An unaligned pointer
    # would be a different cubin; refuse to save one.
    for name in _PREFILL_PREFIX:
        idx = fn.arg_names.index(name)
        if idx not in div16:
            raise SystemExit(
                f"{key}: pointer {name} is not tt.divisibility 16 "
                f"(attrs={ck.src.attrs!r})"
            )
    if ck.metadata.target.arch != 86:
        raise SystemExit(f"{key} targeted sm_{ck.metadata.target.arch}, want sm_86")
    if ck.metadata.num_warps != 8:
        raise SystemExit(f"{key} num_warps {ck.metadata.num_warps} != 8")
    stages = getattr(ck.metadata, "num_stages", None)
    if stages != 2:
        raise SystemExit(f"{key} num_stages {stages} != 2")
    expected_runtime = _PREFILL_PREFIX + [name for name in SPECIALIZED_INTS if classes[name] != "eq1"]
    params = _runtime_from(ck, fn, expected_runtime)
    ptx = ck.asm["ptx"]
    if not isinstance(ptx, str):
        ptx = ptx.decode()
    launch_params = check_against_ptx(ck.metadata.name, params, ptx)
    cubin = ck.asm["cubin"]
    if not isinstance(cubin, (bytes, bytearray)) or len(cubin) == 0:
        raise SystemExit(f"{key} cubin is empty")
    # Pointer dtypes the tests rely on. Partials are fp16 because num_splits is 1
    # and the wrapper passes `q` for every unused tensor.
    cache_ty = "*i32" if consts_expect["QCK"] else "*fp16"
    by_name = {p["name"]: p["type"] for p in params}
    for name in ("k_cache", "v_cache"):
        if by_name[name] != cache_ty:
            raise SystemExit(f"{key} {name} type {by_name[name]} != {cache_ty}")
    for name in ("partial_o", "partial_ml", "k_new", "v_new", "sinks", "h32", "k_scales", "v_scales", "q", "out"):
        if by_name[name] != "*fp16":
            raise SystemExit(f"{key} {name} type {by_name[name]} != *fp16")
    path = "qc" if consts_expect["QCK"] else "fp16"
    if prefill_class_key(path, classes) != key:
        raise SystemExit(f"class key drifted: {prefill_class_key(path, classes)} != {key}")
    desc = ", ".join(f"{name}={classes[name]}" for name in SPECIALIZED_INTS)
    return bytes(cubin), {
        "name": ck.metadata.name,
        "num_warps": ck.metadata.num_warps,
        "num_stages": stages,
        "shared": ck.metadata.shared,
        "global_scratch_size": ck.metadata.global_scratch_size,
        "profile_scratch_size": ck.metadata.profile_scratch_size,
        "parameters": launch_params,
        "constexprs": jsonable_constexprs(consts),
        "specialization": dict(classes),
        "path": path,
        "grid": (
            "grid (cdiv(q_len, BLOCK_M), bsz * n_q_heads, num_splits), "
            "block (32 * num_warps, 1, 1). "
            f"Specialization {key}: {desc}. "
            "eq1 is a constexpr and the argument is omitted from the launch. "
            "div16 is the same i32 parameter with tt.divisibility 16. "
            "other is a plain i32. "
            "num_splits follows paged_attn_triton_prefill and is 1 "
            "(IS_SPLIT false) for this cubin."
        ),
        "triton_version": triton_version,
        "ext_tree": ext_tree,
        "capture_q_len": q_len,
        "capture_pages": pages,
    }


def build_prefill_cubins(paged, shape: dict, out: Path, fork_marker: str, ext_tree: str, triton_version: str) -> list[str]:
    """JIT every unsplit 27B prefill class and append it to manifest.json.

    Decode kernel records already in the manifest are left as they were.
    """
    import torch

    if paged._qc_staging != 1:
        raise SystemExit(f"EXL3_QC_STAGING imported as {paged._qc_staging}, want 1")
    if paged._qc_prefill_two_pass_min_q != 256:
        raise SystemExit(
            f"two-pass threshold is {paged._qc_prefill_two_pass_min_q}, want 256"
        )
    common = dict(
        IS_SPLIT=False,
        NEW_KV=0,
        n_q_heads=shape["n_q_heads"],
        n_kv_heads=shape["n_kv_heads"],
        page_size=shape["page_size"],
        head_dim=shape["head_dim"],
        HD_PAD=256,
        scale=0.0625,
        CAUSAL=True,
        HAS_WINDOW_LEFT=False,
        HAS_WINDOW_RIGHT=False,
        SOFTCAP=0.0,
        HAS_SINKS=False,
        WIDE_INDEX=False,
        BLOCK_M=64,
    )
    fn = paged._paged_attn_prefill_kernel
    notes = []
    written = []
    seen = {}
    for q_len, pages in PREFILL_CAPTURES:
        path, classes = _classes_for(q_len, pages)
        key = prefill_class_key(path, classes)
        if key in seen:
            raise SystemExit(f"{key} captured twice ({seen[key]} and q_len {q_len} pages {pages})")
        seen[key] = (q_len, pages)
        if path == "qc":
            expect = {**common, "QCK": 3, "QCV": 3, "BLOCK_N": 64}
        else:
            expect = {**common, "QCK": 0, "QCV": 0, "BLOCK_N": 32}
        ck = _capture_prefill(paged, shape, q_len, pages)
        cubin, record = _record_prefill(
            ck, fn, expect, classes, key, q_len, pages, ext_tree, triton_version
        )
        filename = f"{key}.cubin"
        record["cubin"] = filename
        hits = _cache_matches(cubin, fork_marker)
        if not hits:
            raise SystemExit(
                f"{key} cubin has no matching ~/.triton/cache entry whose source "
                "names triton_paged.py"
            )
        (out / filename).write_bytes(cubin)
        written.append((key, record))
        hit = hits[0]
        notes.append(
            f"{key}: q_len {q_len} pages {pages} symbol {record['name']} "
            f"warps {record['num_warps']} stages {record['num_stages']} "
            f"shared {record['shared']} "
            f"scratch {record['global_scratch_size']}/{record['profile_scratch_size']} "
            f"bytes {len(cubin)} cache {hit['path']}"
        )
        del ck
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    used = torch.cuda.memory_allocated()
    if used > 1 << 30:
        raise SystemExit(f"GPU allocation {used} bytes exceeds 1 GiB after prefill capture")

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    kernels = manifest["kernels"]
    # Drop prefill records from a previous capture so a renamed class cannot
    # linger beside the new keys. Decode records are the ones main() just wrote.
    stale_keys = [name for name in list(kernels) if name.startswith("_paged_attn_prefill_kernel")]
    for name in stale_keys:
        del kernels[name]
    for key, record in written:
        kernels[key] = record
    for stale in out.glob("_paged_attn_prefill_kernel*.cubin"):
        if stale.name not in {record["cubin"] for _key, record in written}:
            stale.unlink()
    manifest["prefill"] = {
        "staging": 1,
        "two_pass_min_q": 256,
        "specialized_ints": list(SPECIALIZED_INTS),
        "int_class": "eq1 if value == 1 else div16 if value % 16 == 0 else other",
        "classes": [key for key, _record in written],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return notes


if __name__ == "__main__":
    main()

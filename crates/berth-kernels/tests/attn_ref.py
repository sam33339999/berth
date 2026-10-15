#!/usr/bin/env python3
"""Bit-exact references for the 27B full-attention decode kernels.

Writes little-endian raw tensors and meta.json under testdata/kernels/attn/.
Fork ops: ext.deinterleave_qg, ext.rope, ext.quant_cache_paged, ext.mul_sigmoid_.

The paged decode pair has no Python launch binding. ext.TritonKernel is
constructible, and TritonKernel::launch is not on the pybind. This script
compiles _paged_attn_decode_split_kernel and _paged_attn_decode_combine_kernel
with the same triton.compile arguments as scripts/build_triton_cubins.py
(FINAL false, cq3, q_len 1), checks the cubin bytes against
target/triton-cubins/, and launches that image the way TritonKernel::launch
does: cuLaunchKernel, grid (programs, splits_cap, 1) then (programs, 1, 1),
block 128, trailing null scratch pointers. It also launches the Triton
CompiledKernel and requires the two outputs to match.

Does not load the 27B. Exits before touching the GPU if :8889 is listening.
"""

import ctypes
import importlib.util
import json
import subprocess
import sys
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
GROUPS = N_KV * HD // 32  # 32
K_WORDS = GROUPS * 3  # cq3: 96 int32s per token

ROOT = Path(__file__).resolve().parents[3]
FORK = Path("/home/sam/qwen38-exl3-dflash2/exllamav3")
OUT = ROOT / "testdata" / "kernels" / "attn"
CUBIN_DIR = ROOT / "target" / "triton-cubins"
BUILD_SCRIPT = ROOT / "scripts" / "build_triton_cubins.py"


def port_busy() -> bool:
    text = subprocess.check_output(["ss", "-ltn"], text=True)
    return ":8889 " in text


def save(name: str, tensor) -> str:
    host = tensor.detach().contiguous().cpu()
    raw = ctypes.string_at(host.data_ptr(), host.numel() * host.element_size())
    path = OUT / name
    path.write_bytes(raw)
    return name


def load_ext():
    so = next(FORK.glob("exllamav3_ext*.so"))
    spec = importlib.util.spec_from_file_location("exllamav3_ext", so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_builder():
    spec = importlib.util.spec_from_file_location("build_triton_cubins", BUILD_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def split_config(bt_width: int, page_size: int, q_len: int, block_n: int, splits_cap: int):
    bound = bt_width * page_size + q_len
    denom = 4 * block_n
    num_splits = max(1, min(splits_cap, (bound + denom - 1) // denom))
    per = (bound + num_splits - 1) // num_splits
    split_len = ((per + block_n - 1) // block_n) * block_n
    return num_splits, split_len


class Driver:
    """cuLaunchKernel with TritonKernel's argument layout."""

    def __init__(self):
        lib = ctypes.CDLL("libcuda.so.1")
        self.lib = lib
        lib.cuInit.argtypes = [ctypes.c_uint]
        lib.cuInit.restype = ctypes.c_int
        rc = lib.cuInit(0)
        if rc != 0:
            raise SystemExit(f"cuInit returned {rc}")
        # Torch's runtime context is not current for a fresh driver call.
        # Retain the primary context the runtime already created and make it current.
        lib.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        lib.cuDeviceGet.restype = ctypes.c_int
        lib.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        lib.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
        lib.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        lib.cuCtxSetCurrent.restype = ctypes.c_int
        dev = ctypes.c_int()
        rc = lib.cuDeviceGet(ctypes.byref(dev), 0)
        if rc != 0:
            raise SystemExit(f"cuDeviceGet returned {rc}")
        ctx = ctypes.c_void_p()
        rc = lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
        if rc != 0:
            raise SystemExit(f"cuDevicePrimaryCtxRetain returned {rc}")
        rc = lib.cuCtxSetCurrent(ctx)
        if rc != 0:
            raise SystemExit(f"cuCtxSetCurrent returned {rc}")
        self.ctx = ctx
        lib.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        lib.cuModuleLoadData.restype = ctypes.c_int
        lib.cuModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        lib.cuModuleGetFunction.restype = ctypes.c_int
        lib.cuFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.cuFuncSetAttribute.restype = ctypes.c_int
        lib.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        lib.cuLaunchKernel.restype = ctypes.c_int
        self.modules = []

    def load(self, image: bytes, name: str, shared: int):
        buf = ctypes.create_string_buffer(image)
        mod = ctypes.c_void_p()
        rc = self.lib.cuModuleLoadData(ctypes.byref(mod), buf)
        if rc != 0:
            raise SystemExit(f"cuModuleLoadData({name}) returned {rc}")
        fn = ctypes.c_void_p()
        rc = self.lib.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode())
        if rc != 0:
            raise SystemExit(f"cuModuleGetFunction({name}) returned {rc}")
        # 8 is CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES.
        if shared > 48 * 1024:
            rc = self.lib.cuFuncSetAttribute(fn, 8, shared)
            if rc != 0:
                raise SystemExit(f"cuFuncSetAttribute({name}) returned {rc}")
        self.modules.append((mod, buf))
        return fn

    def launch(self, fn, grid, shared, stream, values):
        packed = (ctypes.c_uint64 * len(values))(*[v & 0xFFFFFFFFFFFFFFFF for v in values])
        params = (ctypes.c_void_p * len(values))()
        base = ctypes.addressof(packed)
        for i in range(len(values)):
            params[i] = base + i * ctypes.sizeof(ctypes.c_uint64)
        rc = self.lib.cuLaunchKernel(
            fn,
            grid[0],
            grid[1],
            grid[2],
            128,
            1,
            1,
            shared,
            ctypes.c_void_p(stream),
            params,
            None,
        )
        if rc != 0:
            raise SystemExit(f"cuLaunchKernel returned {rc}")


def compile_pair(bt, paged, triton):
    shape = {
        "n_q_heads": N_Q,
        "n_kv_heads": N_KV,
        "head_dim": HD,
        "page_size": PAGE,
        "q_len": bt.Q_LEN,
        "k_bits": bt.K_BITS,
        "v_bits": bt.V_BITS,
        "sm_scale": float(HD ** -0.5),
    }
    window_left, window_right = paged._normalize_window(bt.WINDOW_SIZE)
    geom = bt.tile(shape, triton)
    cache_t = "*i32"
    split_sig = {
        "q": "*fp16",
        "k_cache": cache_t,
        "v_cache": cache_t,
        "block_table": "*i32",
        "cache_seqlens": "*i32",
        "out": "*fp16",
        "partial_o": "*fp32",
        "partial_ml": "*fp32",
        "k_scales": "*fp16",
        "v_scales": "*fp16",
        "h32": "*fp16",
        "split_len": "i32",
        "num_pages_per_seq": "i32",
        "num_splits": "i32",
        "sinks": "*fp32",
    } | {
        n: "constexpr"
        for n in (
            "QCK",
            "QCV",
            "q_len",
            "kv_append_len",
            "n_q_heads",
            "n_kv_heads",
            "page_size",
            "head_dim",
            "HD_PAD",
            "scale",
            "CAUSAL",
            "WINDOW_LEFT",
            "WINDOW_RIGHT",
            "SOFTCAP",
            "FINAL",
            "HAS_SINKS",
            "BLOCK_M",
            "BLOCK_H",
            "BLOCK_ROWS",
            "BLOCK_N",
        )
    }
    split_consts = dict(
        QCK=shape["k_bits"],
        QCV=shape["v_bits"],
        q_len=shape["q_len"],
        kv_append_len=shape["q_len"],
        n_q_heads=shape["n_q_heads"],
        n_kv_heads=shape["n_kv_heads"],
        page_size=shape["page_size"],
        head_dim=shape["head_dim"],
        HD_PAD=geom["hd_pad"],
        scale=float(shape["sm_scale"]),
        CAUSAL=bool(bt.CAUSAL),
        WINDOW_LEFT=window_left,
        WINDOW_RIGHT=window_right,
        SOFTCAP=float(bt.SOFTCAP or 0.0),
        FINAL=False,
        HAS_SINKS=False,
        BLOCK_M=geom["block_m"],
        BLOCK_H=geom["block_h"],
        BLOCK_ROWS=geom["block_rows"],
        BLOCK_N=geom["block_n"],
    )
    combine_sig = {
        "partial_o": "*fp32",
        "partial_ml": "*fp32",
        "out": "*fp16",
        "h32": "*fp16",
        "num_splits": "i32",
        "sinks": "*fp32",
    } | {
        n: "constexpr"
        for n in (
            "QCV",
            "HAS_SINKS",
            "q_len",
            "n_q_heads",
            "n_kv_heads",
            "head_dim",
            "HD_PAD",
            "BLOCK_M",
            "BLOCK_H",
            "BLOCK_ROWS",
        )
    }
    combine_consts = dict(
        QCV=shape["v_bits"],
        HAS_SINKS=bt.HAS_SINKS,
        q_len=shape["q_len"],
        n_q_heads=shape["n_q_heads"],
        n_kv_heads=shape["n_kv_heads"],
        head_dim=shape["head_dim"],
        HD_PAD=geom["hd_pad"],
        BLOCK_M=geom["block_m"],
        BLOCK_H=geom["block_h"],
        BLOCK_ROWS=geom["block_rows"],
    )
    split_ck = bt.compile_kernel(paged._paged_attn_decode_split_kernel, split_sig, split_consts, 4, 2)
    combine_ck = bt.compile_kernel(
        paged._paged_attn_decode_combine_kernel, combine_sig, combine_consts, 4, 1
    )
    return split_ck, combine_ck, geom


def cubin_bytes(ck) -> bytes:
    raw = ck.asm["cubin"]
    if isinstance(raw, str):
        raw = raw.encode()
    return bytes(raw)


def require_same_cubin(ck, filename: str) -> bytes:
    disk = (CUBIN_DIR / filename).read_bytes()
    got = cubin_bytes(ck)
    if got != disk:
        raise SystemExit(
            f"{filename}: triton.compile cubin ({len(got)} bytes) != "
            f"target/triton-cubins ({len(disk)} bytes)"
        )
    return disk


def randn(gen, shape, dtype):
    import torch

    return torch.randn(*shape, generator=gen, device="cuda", dtype=torch.float32).to(dtype)


def pages(torch, n):
    k = torch.zeros(n, PAGE, K_WORDS, dtype=torch.int32, device="cuda")
    v = torch.zeros(n, PAGE, K_WORDS, dtype=torch.int32, device="cuda")
    ks = torch.zeros(n, PAGE, GROUPS, dtype=torch.float16, device="cuda")
    vs = torch.zeros(n, PAGE, GROUPS, dtype=torch.float16, device="cuda")
    return k, v, ks, vs


def main():
    if port_busy():
        raise SystemExit(":8889 is listening; not touching the GPU")
    if not (CUBIN_DIR / "manifest.json").is_file():
        raise SystemExit(f"missing {CUBIN_DIR / 'manifest.json'}; run scripts/build_triton_cubins.py")

    import torch
    import triton

    if torch.cuda.memory_allocated() > 1 << 30:
        raise SystemExit("GPU allocation already exceeds 1 GiB")

    bt = load_builder()
    paged = bt.load_triton_paged(FORK)
    ext = load_ext()
    if hasattr(ext, "TritonKernel") and hasattr(ext.TritonKernel, "launch"):
        raise SystemExit("TritonKernel.launch is bound; this script's note is stale")

    split_ck, combine_ck, geom = compile_pair(bt, paged, triton)
    split_image = require_same_cubin(split_ck, "_paged_attn_decode_split_kernel.cubin")
    combine_image = require_same_cubin(combine_ck, "_paged_attn_decode_combine_kernel.cubin")
    manifest = json.loads((CUBIN_DIR / "manifest.json").read_text())
    split_meta = manifest["kernels"]["_paged_attn_decode_split_kernel"]
    combine_meta = manifest["kernels"]["_paged_attn_decode_combine_kernel"]
    split_shared = int(split_meta["shared"])
    combine_shared = int(combine_meta["shared"])
    if split_ck.metadata.shared != split_shared or combine_ck.metadata.shared != combine_shared:
        raise SystemExit(
            f"shared mismatch compile {split_ck.metadata.shared}/{combine_ck.metadata.shared} "
            f"manifest {split_shared}/{combine_shared}"
        )

    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    programs = 1 * N_KV * triton.cdiv(N_Q // N_KV, geom["block_h"])
    splits_cap = max(1, min(2 * sm_count // programs, 128))
    if (sm_count, programs, splits_cap) != (82, 4, 41):
        raise SystemExit(f"expected sm 82, programs 4, splits_cap 41; got {sm_count}, {programs}, {splits_cap}")
    block_n = geom["block_n"]
    if block_n != 32 or geom["block_rows"] != 16 or geom["hd_pad"] != 256:
        raise SystemExit(f"unexpected tile {geom}")

    drv = Driver()
    split_fn = drv.load(split_image, split_meta["name"], split_shared)
    combine_fn = drv.load(combine_image, combine_meta["name"], combine_shared)
    stream = torch.cuda.current_stream().cuda_stream

    def attention(q, k_cache, v_cache, k_scales, v_scales, block_table, cache_seqlens, bt_width):
        num_splits, split_len = split_config(bt_width, PAGE, 1, block_n, splits_cap)
        out = torch.zeros_like(q)
        partial_o = torch.zeros(programs * splits_cap * 16 * 256, dtype=torch.float32, device="cuda")
        partial_ml = torch.zeros(programs * splits_cap * 16 * 2, dtype=torch.float32, device="cuda")
        sinks = q
        values = [
            q.data_ptr(),
            k_cache.data_ptr(),
            v_cache.data_ptr(),
            block_table.data_ptr(),
            cache_seqlens.data_ptr(),
            out.data_ptr(),
            partial_o.data_ptr(),
            partial_ml.data_ptr(),
            k_scales.data_ptr(),
            v_scales.data_ptr(),
            h32.data_ptr(),
            split_len,
            bt_width,
            num_splits,
            sinks.data_ptr(),
            0,
            0,
        ]
        drv.launch(split_fn, (programs, splits_cap, 1), split_shared, stream, values)
        drv.launch(
            combine_fn,
            (programs, 1, 1),
            combine_shared,
            stream,
            [
                partial_o.data_ptr(),
                partial_ml.data_ptr(),
                out.data_ptr(),
                h32.data_ptr(),
                num_splits,
                sinks.data_ptr(),
                0,
                0,
            ],
        )
        torch.cuda.synchronize()

        out_triton = torch.zeros_like(q)
        po_t = torch.zeros_like(partial_o)
        ml_t = torch.zeros_like(partial_ml)
        split_ck[(programs, splits_cap, 1)](
            q,
            k_cache,
            v_cache,
            block_table,
            cache_seqlens,
            out_triton,
            po_t,
            ml_t,
            k_scales,
            v_scales,
            h32,
            split_len,
            bt_width,
            num_splits,
            sinks,
        )
        combine_ck[(programs, 1, 1)](po_t, ml_t, out_triton, h32, num_splits, sinks)
        torch.cuda.synchronize()
        if not torch.equal(out, out_triton):
            differ = (out.view(torch.int16) != out_triton.view(torch.int16)).sum().item()
            raise SystemExit(f"driver launch and Triton CompiledKernel differ in {differ} fp16 elements")
        return out, num_splits, split_len

    gen = torch.Generator(device="cuda")
    gen.manual_seed(SEED)
    torch.manual_seed(SEED)

    h32 = paged._get_h32(torch.device("cuda"))
    idx = torch.arange(0, ROTARY, 2, device="cuda", dtype=torch.float32)
    inv_freq = 1.0 / (THETA ** (idx / ROTARY))
    q_norm = randn(gen, (HD,), torch.bfloat16)
    k_norm = randn(gen, (HD,), torch.bfloat16)

    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        "h32": save("h32.bin", h32),
        "inv_freq": save("inv_freq.bin", inv_freq),
        "q_norm": save("q_norm.bin", q_norm),
        "k_norm": save("k_norm.bin", k_norm),
    }

    qg = randn(gen, (1, N_Q * 2 * HD), torch.float16)
    q = torch.empty(1, N_Q * HD, dtype=torch.float16, device="cuda")
    g = torch.empty_like(q)
    ext.deinterleave_qg(qg, q, g, HD)
    deinterleave = {
        "qg": save("deinterleave_qg.bin", qg),
        "q": save("deinterleave_q.bin", q),
        "g": save("deinterleave_g.bin", g),
        "head_dim": HD,
        "q_numel": N_Q * HD,
    }

    q4 = randn(gen, (1, 1, N_Q, HD), torch.float16)
    k4 = randn(gen, (1, 1, N_KV, HD), torch.float16)
    q_in = q4.clone()
    k_in = k4.clone()
    ext.rope(
        q4,
        q4,
        k4,
        k4,
        inv_freq,
        300,
        None,
        None,
        2,
        1.0,
        q_norm,
        k_norm,
        EPS,
        BIAS,
        0.0,
        0,
        1,
        0,
    )
    rope = {
        "q_in": save("rope_q_in.bin", q_in),
        "k_in": save("rope_k_in.bin", k_in),
        "q_out": save("rope_q_out.bin", q4),
        "k_out": save("rope_k_out.bin", k4),
        "position": 300,
    }

    x = randn(gen, (1, N_Q * HD), torch.float16)
    y = randn(gen, (1, N_Q * HD), torch.float16)
    x_in = x.clone()
    ext.mul_sigmoid_(x, y)
    mul = {
        "x_in": save("mul_x_in.bin", x_in),
        "y": save("mul_y.bin", y),
        "x_out": save("mul_x_out.bin", x),
        "numel": N_Q * HD,
    }

    seq = 300
    k_tok = randn(gen, (1, seq, N_KV, HD), torch.float16)
    v_tok = randn(gen, (1, seq, N_KV, HD), torch.float16)
    seqlens0 = torch.zeros(1, dtype=torch.int32, device="cuda")
    table2 = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")
    qk, qv, sk, sv = pages(torch, 2)
    ext.quant_cache_paged(k_tok, qk, sk, v_tok, qv, sv, seqlens0, table2, PAGE, seq, 0.0, True)
    quant = {
        "k_in": save("quant_k_in.bin", k_tok),
        "v_in": save("quant_v_in.bin", v_tok),
        "k_pages": save("quant_k_pages.bin", qk),
        "v_pages": save("quant_v_pages.bin", qv),
        "k_scales": save("quant_k_scales.bin", sk),
        "v_scales": save("quant_v_scales.bin", sv),
        "seq_len": seq,
        "blocks_per_seq": 2,
        "block_table": [0, 1],
        "cache_seqlens": [0],
        "num_pages": 2,
    }

    def core(tag, position, seqlens_val, table_host, num_pages, cache):
        qg_c = randn(gen, (1, N_Q * 2 * HD), torch.float16)
        k_c = randn(gen, (1, 1, N_KV, HD), torch.float16)
        v_c = randn(gen, (1, 1, N_KV, HD), torch.float16)
        # Save the pre-chain inputs. RoPE updates k in place; v is not roped.
        saved_in = {
            "qg": save(f"{tag}_qg.bin", qg_c),
            "k": save(f"{tag}_k.bin", k_c),
            "v": save(f"{tag}_v.bin", v_c),
        }
        q_c = torch.empty(1, N_Q * HD, dtype=torch.float16, device="cuda")
        g_c = torch.empty_like(q_c)
        ext.deinterleave_qg(qg_c, q_c, g_c, HD)
        q_view = q_c.view(1, 1, N_Q, HD)
        ext.rope(
            q_view,
            q_view,
            k_c,
            k_c,
            inv_freq,
            position,
            None,
            None,
            2,
            1.0,
            q_norm,
            k_norm,
            EPS,
            BIAS,
            0.0,
            0,
            1,
            0,
        )
        seqlens = torch.tensor([seqlens_val], dtype=torch.int32, device="cuda")
        table = torch.tensor([table_host], dtype=torch.int32, device="cuda")
        bt_width = len(table_host)
        ext.quant_cache_paged(
            k_c, cache[0], cache[2], v_c, cache[1], cache[3], seqlens, table, PAGE, 1, 0.0, True
        )
        o, num_splits, split_len = attention(
            q_view, cache[0], cache[1], cache[2], cache[3], table, seqlens, bt_width
        )
        o = o.view(1, N_Q * HD)
        ext.mul_sigmoid_(o, g_c)
        torch.cuda.synchronize()
        return {
            "qg": saved_in["qg"],
            "k": saved_in["k"],
            "v": saved_in["v"],
            "o": save(f"{tag}_o.bin", o),
            "k_pages": save(f"{tag}_k_pages.bin", cache[0]),
            "v_pages": save(f"{tag}_v_pages.bin", cache[1]),
            "k_scales": save(f"{tag}_k_scales.bin", cache[2]),
            "v_scales": save(f"{tag}_v_scales.bin", cache[3]),
            "position": position,
            "cache_seqlens": [seqlens_val],
            "block_table": table_host,
            "bt_width": bt_width,
            "num_pages": num_pages,
            "num_splits": num_splits,
            "split_len": split_len,
            "k_pages_in": quant["k_pages"] if tag == "core_cached" else None,
            "v_pages_in": quant["v_pages"] if tag == "core_cached" else None,
            "k_scales_in": quant["k_scales"] if tag == "core_cached" else None,
            "v_scales_in": quant["v_scales"] if tag == "core_cached" else None,
        }

    empty_cache = pages(torch, 1)
    core_empty = core("core_empty", 0, 0, [0], 1, empty_cache)
    cached = (qk.clone(), qv.clone(), sk.clone(), sv.clone())
    core_cached = core("core_cached", 300, 300, [0, 1], 2, cached)

    meta = {
        "seed": SEED,
        "sm_count": sm_count,
        "programs": programs,
        "splits_cap": splits_cap,
        "n_q_heads": N_Q,
        "n_kv_heads": N_KV,
        "head_dim": HD,
        "page_size": PAGE,
        "rotary_dim": ROTARY,
        "rope_theta": THETA,
        "norm_eps": EPS,
        "norm_constant_bias": BIAS,
        "attn_factor": 1.0,
        "compand_a": 0.0,
        "token_dim": N_KV * HD,
        "shared": files,
        "deinterleave": deinterleave,
        "rope": rope,
        "mul_sigmoid": mul,
        "quant": quant,
        "core_empty": core_empty,
        "core_cached": core_cached,
        "paged_attn": (
            "triton.compile of _paged_attn_decode_split_kernel and "
            "_paged_attn_decode_combine_kernel with the build_triton_cubins.py "
            "constexprs (FINAL false). Cubin bytes match target/triton-cubins/. "
            "ext.TritonKernel is constructible; TritonKernel.launch is not a Python "
            "binding. The saved outputs are cuLaunchKernel of that image with "
            "TritonKernel's argument layout, checked equal to CompiledKernel."
        ),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    used = torch.cuda.max_memory_allocated()
    print(f"wrote {OUT} ({total} bytes), peak GPU allocated {used} bytes")
    if total >= 3 * 1024 * 1024:
        raise SystemExit(f"testdata {total} bytes exceeds 3 MB")
    if used >= 1 << 30:
        raise SystemExit(f"GPU allocation {used} bytes exceeds 1 GiB")


if __name__ == "__main__":
    main()

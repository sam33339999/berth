#!/usr/bin/env python3
"""Fork reference for the decode ops shim. Needs the GPU. Does not load a model.

Writes little-endian raw bins and meta.json under testdata/kernels/ops. Trellis,
suh, svh, norm weights, and the embedding table stay in the checkpoint; this
script only stores activations and outputs.

mgemm is ext.exl3_mgemm in sliced mode (SlicedMultiLinear packing), on the real
layer-0 GDN qkv+z projections and the layer-3 attention q/k/v projections.
Norms are ext.rms_norm and ext.rms_norm_res_in. silu is ext.silu_mul.
The residual add and the embedding have no exllamav3_ext binding: the
references are torch.add and F.embedding(...).float().

Aborts before any mgemm launch unless the coop autotune cache already holds
the measured shape-4 records, and restores the cache file if a launch appends.

    EXL3_INT8_GEMV=2 EXL3_INT8_GEMV_MAX_K=5 EXL3_QC_STAGING=1 \\
    EXL3_HGEMM_F16ACC=auto \\
    PYTHONPATH=/home/sam/qwen38-exl3-dflash2/exllamav3 \\
    TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \\
    python crates/berth-kernels/tests/ops_ref.py [OUT_DIR]
"""

import ctypes
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

TARGET = Path(os.environ.get(
    "BERTH_27B_DIR", "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3"
))
SEED = 0x27B0515
MASK = (1 << 64) - 1
FNV_OFF = 1469598103934665603
FNV_PRIME = 1099511628211
SLICED_XOR = 0x9E3779B97F4A7C15

GDN_KEY = 0xD2297A8A33A33F5B
ATTN_KEY = 0x9C25A0752178FB4A
GDN_SALTED = 0xBFBA97D5BE64AE6D
ATTN_SALTED = 0xCCEAF507E093058A
CACHE_HIT = (4, 256, 10, 8)

DTYPES = {
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
}


def serve_listening():
    try:
        out = subprocess.run(["ss", "-ltn"], capture_output=True, text=True, check=False)
    except OSError:
        return True
    if out.returncode != 0:
        return True
    return ":8889 " in out.stdout


def tune_cache_path():
    override = os.environ.get("EXLLAMAV3_TUNE_CACHE")
    if override:
        path = Path(override)
        if path.is_dir():
            return path / "coop_autotune_v1.bin"
        return path
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "exllamav3" / "autotune" / "coop_autotune_v1.bin"


def roundup_pow2(x):
    if x == 0:
        return 1
    x = (x - 1) & MASK
    x |= x >> 1
    x |= x >> 2
    x |= x >> 4
    x |= x >> 8
    x |= x >> 16
    x |= x >> 32
    return (x + 1) & MASK


def m_key(size_m):
    if size_m <= 16:
        return min(roundup_pow2(size_m), 16)
    if size_m <= 31:
        return 32
    if size_m <= 63:
        return 64
    if size_m <= 127:
        return 128
    return 512


def mix(h, v):
    h = (h ^ (v & MASK)) & MASK
    return (h * FNV_PRIME) & MASK


def mgemm_key(size_m, size_k, size_n, k_bits, c_fp32, device, cc, sms, cb,
              bszm_in, bszm_out, sliced):
    h = FNV_OFF
    for v in (
        m_key(size_m), size_k, size_n, k_bits, 1 if c_fp32 else 0,
        device, cc, sms, cb, min(bszm_in, 24), min(bszm_out, 24),
    ):
        h = mix(h, v)
    if sliced:
        h = (h ^ SLICED_XOR) & MASK
    return h


def salt(h):
    return ((h ^ 4) * FNV_PRIME) & MASK


def cache_records(data):
    if len(data) < 16 or data[:8] != b"EX3ATUNE":
        raise SystemExit("autotune cache: missing EX3ATUNE header")
    fmt = int.from_bytes(data[8:12], "little")
    rec_size = int.from_bytes(data[12:16], "little")
    if fmt != 1 or rec_size < 32:
        raise SystemExit(f"autotune cache: format {fmt} record {rec_size}")
    found = {}
    off = 16
    while off + rec_size <= len(data):
        chunk = data[off:off + rec_size]
        if chunk[:8] == b"EX3ATUNE":
            off += 16
            continue
        h = int.from_bytes(chunk[0:8], "little")
        tag = int.from_bytes(chunk[8:12], "little", signed=True)
        block = int.from_bytes(chunk[12:16], "little", signed=True)
        sms = int.from_bytes(chunk[16:20], "little", signed=True)
        conc = int.from_bytes(chunk[20:24], "little", signed=True)
        found[h] = (tag, block, sms, conc)
        off += rec_size
    return found


def require_cache_hit(records, salted, label):
    got = records.get(salted)
    if got != CACHE_HIT:
        raise SystemExit(
            f"{label}: cache record {got} is not {CACHE_HIT}; refusing to launch mgemm"
        )


class Shard:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            self.header_len = n
            self.header = json.loads(f.read(n))

    def tensor(self, name, device):
        info = self.header[name]
        start, end = info["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(8 + self.header_len + start)
            data = f.read(end - start)
        dtype = DTYPES[info["dtype"]]
        shape = tuple(info["shape"])
        cpu = torch.empty(shape, dtype=dtype)
        if len(data) != cpu.numel() * cpu.element_size():
            raise SystemExit(f"{name}: {len(data)} bytes, tensor is {cpu.numel() * cpu.element_size()}")
        ctypes.memmove(cpu.data_ptr(), data, len(data))
        return cpu.to(device)


class Checkpoint:
    def __init__(self, directory):
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self.directory = directory
        self.shards = {}

    def has(self, name):
        return name in self.weight_map

    def tensor(self, name, device="cuda"):
        shard_name = self.weight_map[name]
        shard = self.shards.get(shard_name)
        if shard is None:
            shard = Shard(self.directory / shard_name)
            self.shards[shard_name] = shard
        return shard.tensor(name, device)

    def row(self, name, row_index, device="cuda"):
        shard_name = self.weight_map[name]
        shard = self.shards.get(shard_name)
        if shard is None:
            shard = Shard(self.directory / shard_name)
            self.shards[shard_name] = shard
        info = shard.header[name]
        shape = info["shape"]
        if len(shape) != 2:
            raise SystemExit(f"{name}: expected rank 2, got {shape}")
        start, _end = info["data_offsets"]
        dtype = DTYPES[info["dtype"]]
        width = shape[1]
        elem = torch.empty((), dtype=dtype).element_size()
        row_bytes = width * elem
        with open(shard.path, "rb") as f:
            f.seek(8 + shard.header_len + start + row_index * row_bytes)
            data = f.read(row_bytes)
        cpu = torch.empty((width,), dtype=dtype)
        ctypes.memmove(cpu.data_ptr(), data, row_bytes)
        return cpu.to(device)


def write_tensor(path, tensor):
    t = tensor.detach().contiguous().cpu()
    n = t.numel() * t.element_size()
    buf = bytearray(n)
    ctypes.memmove((ctypes.c_char * n).from_buffer(buf), t.data_ptr(), n)
    path.write_bytes(buf)


def ptrs_to_cuda(values):
    raw = bytearray()
    for value in values:
        raw += (int(value) & MASK).to_bytes(8, "little")
    cpu = torch.empty(len(values), dtype=torch.int64)
    ctypes.memmove(cpu.data_ptr(), bytes(raw), len(raw))
    return cpu.cuda()


def load_linear(ckpt, prefix):
    for suffix in ("su", "sv", "mcg", "bias"):
        if ckpt.has(prefix + "." + suffix):
            raise SystemExit(f"{prefix} has .{suffix}; the 27B fused linears do not")
    if not ckpt.has(prefix + ".mul1"):
        raise SystemExit(f"{prefix} has no .mul1")
    trellis = ckpt.tensor(prefix + ".trellis")
    suh = ckpt.tensor(prefix + ".suh")
    svh = ckpt.tensor(prefix + ".svh")
    if trellis.dtype != torch.int16 or trellis.dim() != 3:
        raise SystemExit(f"{prefix}.trellis {trellis.dtype} {tuple(trellis.shape)}")
    k_bits = trellis.shape[-1] // 16
    n = trellis.shape[1] * 16
    k_in = trellis.shape[0] * 16
    if suh.dtype != torch.float16 or suh.numel() != k_in:
        raise SystemExit(f"{prefix}.suh {suh.dtype} {tuple(suh.shape)}")
    if svh.dtype != torch.float16 or svh.numel() != n:
        raise SystemExit(f"{prefix}.svh {svh.dtype} {tuple(svh.shape)}")
    if not trellis.is_contiguous() or not suh.is_contiguous() or not svh.is_contiguous():
        raise SystemExit(f"{prefix} scales are not contiguous")
    return {
        "prefix": prefix,
        "trellis": trellis,
        "suh": suh,
        "svh": svh,
        "n": int(n),
        "k": int(k_bits),
        "k_in": int(k_in),
    }


def run_sliced(ext, sources, x, out_dtype):
    width = math.gcd(*(src["n"] for src in sources))
    if width % 128 != 0 or width < 256:
        raise SystemExit(f"slice width {width}")
    k_bits = sources[0]["k"]
    k_in = sources[0]["k_in"]
    if any(src["k"] != k_bits or src["k_in"] != k_in for src in sources):
        raise SystemExit("sources disagree on K or in-features")
    if x.shape != (1, 1, k_in) or x.dtype != torch.float16:
        raise SystemExit(f"activation {tuple(x.shape)} {x.dtype}")

    trellis_ptrs, svh_ptrs, strides, srcs, c_host = [], [], [], [], []
    outputs = []
    for i, src in enumerate(sources):
        out = torch.empty((1, src["n"]), dtype=out_dtype, device="cuda")
        outputs.append(out)
        for n0 in range(0, src["n"], width):
            trellis_ptrs.append(
                src["trellis"].data_ptr() + (n0 // 16) * 16 * k_bits * src["trellis"].element_size()
            )
            svh_ptrs.append(src["svh"].data_ptr() + n0 * src["svh"].element_size())
            strides.append(src["n"])
            srcs.append(i)
            c_host.append(out.data_ptr() + n0 * out.element_size())

    num_slices = len(srcs)
    num_src = len(sources)
    carrier = torch.empty((num_slices, 1, width), dtype=out_dtype, device="cuda")
    a_had = torch.empty((num_src, 1, k_in), dtype=torch.float16, device="cuda")
    tag = ext.exl3_mgemm(
        x,
        ptrs_to_cuda(trellis_ptrs),
        carrier,
        ptrs_to_cuda([src["suh"].data_ptr() for src in sources]),
        a_had,
        ptrs_to_cuda(svh_ptrs),
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
        ptrs_to_cuda(c_host),
        torch.tensor(strides, dtype=torch.int32, device="cuda"),
        torch.tensor(srcs, dtype=torch.int32, device="cuda"),
        num_src,
    )
    torch.cuda.synchronize()
    if tag != 4:
        raise SystemExit(f"exl3_mgemm returned shape {tag}, expected 4")
    return outputs, width, num_slices


def f32_bits(value):
    t = torch.tensor([value], dtype=torch.float64).float()
    return int(t.view(torch.int32).item()) & 0xFFFFFFFF, float(t.item())


def main():
    if serve_listening():
        raise SystemExit("port 8889 is listening; refusing to use the GPU")

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "testdata/kernels/ops")
    out.mkdir(parents=True, exist_ok=True)

    gdn_key = mgemm_key(1, 5120, 2048, 4, True, 0, 2, 82, 2, 1, 8, True)
    attn_key = mgemm_key(1, 5120, 1024, 4, False, 0, 2, 82, 2, 1, 14, True)
    if gdn_key != GDN_KEY or attn_key != ATTN_KEY:
        raise SystemExit(f"tune key mismatch gdn={gdn_key:#x} attn={attn_key:#x}")
    if salt(gdn_key) != GDN_SALTED or salt(attn_key) != ATTN_SALTED:
        raise SystemExit("salted tune key mismatch")

    cache_path = tune_cache_path()
    cache_before = cache_path.read_bytes()
    records = cache_records(cache_before)
    require_cache_hit(records, GDN_SALTED, "gdn")
    require_cache_hit(records, ATTN_SALTED, "attn")

    config = json.loads((TARGET / "config.json").read_text())
    text = config["text_config"]
    hidden = int(text["hidden_size"])
    intermediate = int(text["intermediate_size"])
    vocab = int(text["vocab_size"])
    if (hidden, intermediate, vocab) != (5120, 17408, 248320):
        raise SystemExit(f"unexpected 27B config {(hidden, intermediate, vocab)}")
    eps_bits, eps = f32_bits(float(text["rms_norm_eps"]))
    bias = 1.0
    scale = 1.0

    from exllamav3.ext import exllamav3_ext as ext

    torch.cuda.set_device(0)
    if ext.g_get_num_sms(0) != 82 or ext.g_get_cc(0) != 2:
        raise SystemExit(
            f"device sms={ext.g_get_num_sms(0)} cc={ext.g_get_cc(0)}; refusing to retune"
        )

    ckpt = Checkpoint(TARGET)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(SEED)

    def randn(shape, dtype):
        values = torch.randn(shape, generator=gen, device="cuda", dtype=torch.float32)
        return values if dtype == torch.float32 else values.to(dtype)

    x = randn((1, 1, hidden), torch.float16)
    write_tensor(out / "mgemm.x.bin", x)

    gdn_names = [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
    ]
    gdn = [load_linear(ckpt, name) for name in gdn_names]
    if [src["n"] for src in gdn] != [10240, 6144]:
        raise SystemExit(f"gdn N {[src['n'] for src in gdn]}")
    gdn_out, gdn_width, gdn_slices = run_sliced(ext, gdn, x, torch.float32)
    if (gdn_width, gdn_slices) != (2048, 8):
        raise SystemExit(f"gdn slices {gdn_width} x {gdn_slices}")
    gdn_files = ["gdn_l0.qkv.bin", "gdn_l0.z.bin"]
    for path, tensor in zip(gdn_files, gdn_out):
        write_tensor(out / path, tensor)
    del gdn, gdn_out
    torch.cuda.empty_cache()

    attn_names = [
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.k_proj",
        "model.language_model.layers.3.self_attn.v_proj",
    ]
    attn = [load_linear(ckpt, name) for name in attn_names]
    if [src["n"] for src in attn] != [12288, 1024, 1024]:
        raise SystemExit(f"attn N {[src['n'] for src in attn]}")
    attn_out, attn_width, attn_slices = run_sliced(ext, attn, x, torch.float16)
    if (attn_width, attn_slices) != (1024, 14):
        raise SystemExit(f"attn slices {attn_width} x {attn_slices}")
    attn_files = ["attn_l3.q.bin", "attn_l3.k.bin", "attn_l3.v.bin"]
    for path, tensor in zip(attn_files, attn_out):
        write_tensor(out / path, tensor)
    del attn, attn_out
    torch.cuda.empty_cache()

    cache_after = cache_path.read_bytes()
    if cache_after != cache_before:
        cache_path.write_bytes(cache_before)
        raise SystemExit("mgemm appended to the autotune cache; restored the original bytes")

    def rms(weight_name, x_in, y_name, kind, residual=None):
        weight = ckpt.tensor(weight_name)
        if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (hidden,):
            raise SystemExit(f"{weight_name} {weight.dtype} {tuple(weight.shape)}")
        y = torch.empty_like(x_in, dtype=torch.float16)
        if kind == "none":
            ext.rms_norm(x_in, weight, y, eps, bias, scale, False, False, 1)
            torch.cuda.synchronize()
            write_tensor(out / y_name, y)
            return
        r = residual
        ext.rms_norm_res_in(x_in, weight, y, r, eps, bias, scale)
        torch.cuda.synchronize()
        write_tensor(out / y_name, y)

    norm_x = randn((1, hidden), torch.float32)
    write_tensor(out / "norm.x.bin", norm_x)
    rms("model.language_model.layers.0.input_layernorm.weight", norm_x, "norm_input.y.bin", "none")
    rms("model.language_model.norm.weight", norm_x, "norm_final.y.bin", "none")

    res_x = randn((1, hidden), torch.float32)
    res_r = randn((1, hidden), torch.float32)
    write_tensor(out / "norm_res.x.bin", res_x)
    write_tensor(out / "norm_res.r.bin", res_r)
    res_live = res_r.clone()
    rms(
        "model.language_model.layers.0.post_attention_layernorm.weight",
        res_x,
        "norm_res.y.bin",
        "res_in",
        res_live,
    )
    write_tensor(out / "norm_res.r_out.bin", res_live)

    gate = randn((intermediate,), torch.float16)
    up = randn((intermediate,), torch.float16)
    activated = torch.empty_like(gate)
    ext.silu_mul(gate, up, activated, 0.0)
    torch.cuda.synchronize()
    write_tensor(out / "silu.gate.bin", gate)
    write_tensor(out / "silu.up.bin", up)
    write_tensor(out / "silu.out.bin", activated)

    a = randn((hidden,), torch.float32)
    b = randn((hidden,), torch.float32)
    # torch.add is the reference: the decode residual is ATen, not an ext op.
    c = torch.add(a, b)
    write_tensor(out / "add.a.bin", a)
    write_tensor(out / "add.b.bin", b)
    write_tensor(out / "add.c.bin", c)

    token_ids = [0, 1, 42, 1024, 248056, 248319]
    weight_name = "model.language_model.embed_tokens.weight"
    rows = torch.stack([ckpt.row(weight_name, token) for token in token_ids])
    if rows.dtype != torch.bfloat16 or tuple(rows.shape) != (len(token_ids), hidden):
        raise SystemExit(f"embed rows {rows.dtype} {tuple(rows.shape)}")
    local = torch.tensor([4, 0, 5, 2, 1, 3], dtype=torch.int32)
    # F.embedding + .float() is modules/embedding.py with out_dtype torch.float.
    embedded = F.embedding(local.to(dtype=torch.long, device="cuda"), rows).float()
    torch.cuda.synchronize()
    write_tensor(out / "embed.ids.bin", local)
    write_tensor(out / "embed.out.bin", embedded)

    meta = {
        "hidden": hidden,
        "intermediate": intermediate,
        "vocab": vocab,
        "epsilon_f32_bits": eps_bits,
        "constant_bias": bias,
        "constant_scale": scale,
        "seed": SEED,
        "x_file": "mgemm.x.bin",
        "mgemm": [
            {
                "name": "gdn_l0",
                "out_dtype": "float32",
                "c_fp32": 1,
                "width": gdn_width,
                "slices": gdn_slices,
                "sources": 2,
                "k_bits": 4,
                "prefixes": gdn_names,
                "out_files": gdn_files,
                "out_elems": [10240, 6144],
                "kernel": "exl3_mgemm_kernel<4,1,2,16,16,512,4,3> block=256 grid=10x1x8 smem=92160",
                "reference": "ext.exl3_mgemm, SlicedMultiLinear of in_proj_qkv and in_proj_z",
            },
            {
                "name": "attn_l3",
                "out_dtype": "float16",
                "c_fp32": 0,
                "width": attn_width,
                "slices": attn_slices,
                "sources": 3,
                "k_bits": 4,
                "prefixes": attn_names,
                "out_files": attn_files,
                "out_elems": [12288, 1024, 1024],
                "kernel": "exl3_mgemm_kernel<4,0,2,16,16,512,4,3> block=256 grid=10x1x8 smem=92160",
                "reference": "ext.exl3_mgemm, SlicedMultiLinear of q_proj, k_proj, v_proj, before deinterleave_qg",
            },
        ],
        "norms": [
            {
                "name": "input_layernorm",
                "kind": "none",
                "weight": "model.language_model.layers.0.input_layernorm.weight",
                "x": "norm.x.bin",
                "y": "norm_input.y.bin",
                "kernel": "rms_norm_kernel<0,float,half,bfloat16,float> block=1024 grid=1",
                "reference": "ext.rms_norm",
            },
            {
                "name": "final",
                "kind": "none",
                "weight": "model.language_model.norm.weight",
                "x": "norm.x.bin",
                "y": "norm_final.y.bin",
                "kernel": "rms_norm_kernel<0,float,half,bfloat16,float> block=1024 grid=1",
                "reference": "ext.rms_norm",
            },
            {
                "name": "post_attention",
                "kind": "res_in",
                "weight": "model.language_model.layers.0.post_attention_layernorm.weight",
                "x": "norm_res.x.bin",
                "residual": "norm_res.r.bin",
                "y": "norm_res.y.bin",
                "residual_out": "norm_res.r_out.bin",
                "kernel": "rms_norm_kernel<2,float,half,bfloat16,float> block=1024 grid=1",
                "reference": "ext.rms_norm_res_in",
            },
        ],
        "silu": {
            "numel": intermediate,
            "act_limit": 0.0,
            "gate": "silu.gate.bin",
            "up": "silu.up.bin",
            "out": "silu.out.bin",
            "kernel": "act_mul_kernel_h<0> block=256 grid=34",
            "reference": "ext.silu_mul",
        },
        "add": {
            "n": hidden,
            "a": "add.a.bin",
            "b": "add.b.bin",
            "c": "add.c.bin",
            "kernel": "add_f32 block=256 grid=20",
            "reference": "torch.add; no exllamav3_ext binding. TransformerBlock post-MLP x += y",
        },
        "embed": {
            "weight": weight_name,
            "hidden": hidden,
            "token_ids": token_ids,
            "local_ids": "embed.ids.bin",
            "out": "embed.out.bin",
            "kernel": "embed_bf16_f32 block=256 grid=6",
            "reference": "F.embedding(local_ids, gathered_rows).float(); no exllamav3_ext binding",
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    if total >= 3 * 1024 * 1024:
        raise SystemExit(f"testdata is {total} bytes")
    print(f"wrote {out} ({total} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()

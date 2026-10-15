#!/usr/bin/env python3
"""Fork reference for Gated DeltaNet prefill.

Runs the non-BC path in `modules/gated_delta_net.py` (fused_op_2, the torch
transpose, slotted causal conv1d, gated_delta_rule_fn, gated_rms_norm) on the
seeded inputs described by testdata/kernels/gdn_prefill/meta.json. Layer-0
A_log, dt_bias, conv1d.weight, and norm.weight come from the 27B checkpoint.
fla autotune configs are replaced with the single config in the cubin manifest
before the first launch, so this process executes the cubins berth loads.

Writes, under --out:

* weights/{a_log,dt_bias,conv_weight,norm_weight}.bin
* case_XX/{out.bin,conv_state.bin,recurrent.bin} and 32-byte prefixes of the
  inputs taken before the kernels run (qkv, z, b, a, conv state, recurrent state)
"""

import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

NUM_K_HEADS = 16
NUM_V_HEADS = 48
HEAD_DIM = 128
QKV_DIM = 2 * NUM_K_HEADS * HEAD_DIM + NUM_V_HEADS * HEAD_DIM
K_DIM = NUM_K_HEADS * HEAD_DIM
V_DIM = NUM_V_HEADS * HEAD_DIM
CONV_K = 4


def port_busy() -> bool:
    out = subprocess.check_output(["ss", "-ltn"], text=True)
    return ":8889 " in out


def lcg_bits(seed: int, n: int) -> list[int]:
    x = seed & 0xFFFFFFFF
    out = []
    for _ in range(n):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        bits = (x & 0x007FFFFF) | 0x3F000000
        if x & 0x100:
            bits |= 0x80000000
        out.append(bits)
    return out


def lcg_f32(seed: int, n: int):
    import torch
    bits = lcg_bits(seed, n)
    raw = struct.pack("<" + "I" * n, *bits)
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).clone()


def lcg_bf16_trunc(seed: int, n: int):
    import torch
    bits = lcg_bits(seed, n)
    raw = struct.pack("<" + "H" * n, *[(b >> 16) & 0xFFFF for b in bits])
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone()


def unwrap(kernel):
    """Heuristics -> Autotuner -> JITFunction. Return (autotuner or None, jit).

    `best_config` is assigned on the first run, so a pin before any launch
    recognizes the autotuner by `configs`, `cache`, and `keys`.
    """
    fn = kernel
    auto = None
    for _ in range(6):
        if hasattr(fn, "configs") and hasattr(fn, "cache") and hasattr(fn, "keys"):
            auto = fn
        if hasattr(fn, "device_caches"):
            return auto, fn
        if not hasattr(fn, "fn"):
            break
        fn = fn.fn
    raise SystemExit(f"could not unwrap {kernel}")


def pin(manifest: dict) -> None:
    import triton
    from exllamav3.vendor.fla.l2norm import l2norm_fwd_kernel
    from exllamav3.vendor.fla.cumsum import chunk_local_cumsum_scalar_kernel
    from exllamav3.vendor.fla.gdn_chunk_fwd import chunk_gated_delta_rule_fwd_kkt_solve_kernel
    from exllamav3.vendor.fla.gdn_wy_fast import recompute_w_u_fwd_kernel
    from exllamav3.vendor.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    from exllamav3.vendor.fla.chunk_o import chunk_fwd_kernel_o

    kernels = {
        "l2norm_fwd_kernel": l2norm_fwd_kernel,
        "chunk_local_cumsum_scalar_kernel": chunk_local_cumsum_scalar_kernel,
        "chunk_gated_delta_rule_fwd_kkt_solve_kernel": chunk_gated_delta_rule_fwd_kkt_solve_kernel,
        "recompute_w_u_fwd_kernel": recompute_w_u_fwd_kernel,
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
        "chunk_fwd_kernel_o": chunk_fwd_kernel_o,
    }
    for name, kernel in kernels.items():
        spec = manifest["kernels"][name]["autotune"]
        if not spec:
            raise SystemExit(f"{name} has no autotune pin in the manifest")
        auto, _jit = unwrap(kernel)
        if auto is None:
            raise SystemExit(f"{name} is not an autotuned kernel")
        # len(configs) == 1 skips the disk autotune cache (autotuner.py run).
        auto.configs = [
            triton.Config(
                spec["kwargs"],
                num_warps=spec["num_warps"],
                num_stages=spec["num_stages"],
                num_ctas=spec["num_ctas"],
            )
        ]
        auto.cache.clear()
        print(
            f"pin {name} warps={spec['num_warps']} stages={spec['num_stages']} kwargs={spec['kwargs']}",
            flush=True,
        )


def load_weights():
    import torch
    from safetensors import safe_open

    root = Path(os.environ.get(
        "BERTH_27B_DIR",
        "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3",
    ))
    shard = root / "model-00001-of-00002.safetensors"
    names = {
        "a_log": "model.language_model.layers.0.linear_attn.A_log",
        "dt_bias": "model.language_model.layers.0.linear_attn.dt_bias",
        "conv": "model.language_model.layers.0.linear_attn.conv1d.weight",
        "norm": "model.language_model.layers.0.linear_attn.norm.weight",
    }
    with safe_open(shard, framework="pt", device="cpu") as handle:
        tensors = {key: handle.get_tensor(name).contiguous() for key, name in names.items()}
    a_log = tensors["a_log"].view(-1).contiguous()
    dt_bias = tensors["dt_bias"].view(-1).contiguous()
    conv = tensors["conv"].squeeze(1).contiguous()
    norm = tensors["norm"].view(-1).contiguous()
    for name, tensor, shape in (
        ("a_log", a_log, (NUM_V_HEADS,)),
        ("dt_bias", dt_bias, (NUM_V_HEADS,)),
        ("conv", conv, (QKV_DIM, CONV_K)),
        ("norm", norm, (HEAD_DIM,)),
    ):
        if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
            raise SystemExit(f"{name} is {tuple(tensor.shape)} {tensor.dtype}, want {shape} bf16")
    return a_log, dt_bias, conv, norm


def write_bytes(path: Path, tensor) -> None:
    import torch
    tensor = tensor.detach().contiguous().cpu()
    path.write_bytes(tensor.view(torch.uint8).numpy().tobytes())


def byte_prefix(tensor, n: int) -> bytes:
    """First `n` bytes of `tensor`, before any kernel reads or writes it."""
    import torch
    raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    if int(raw.numel()) < n:
        raise SystemExit(f"prefix of {n} bytes exceeds a {int(raw.numel())}-byte tensor")
    return bytes(raw[:n].cpu().numpy())


def run_case(case, seeds, prefix_n, beta_scale, weights, out_dir: Path) -> None:
    import torch
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.gated_delta_net_fn.conv1d import causal_conv1d_update
    from exllamav3.modules.gated_delta_net_fn.gated_delta_rule import gated_delta_rule_fn

    seqlen = int(case["seqlen"])
    zero = bool(case["zero_state"])
    slot = int(case["slot"])
    num_slots = int(case["num_slots"])
    history_stride = int(case["history_stride"])
    a_log, dt_bias, conv_w, norm_w = weights

    qkv = lcg_f32(seeds["qkv"], seqlen * QKV_DIM).view(1, seqlen, QKV_DIM).cuda()
    z = lcg_f32(seeds["z"], seqlen * V_DIM).view(1, seqlen, NUM_V_HEADS, HEAD_DIM).cuda()
    b = lcg_f32(seeds["b"], seqlen * NUM_V_HEADS).view(1, seqlen, NUM_V_HEADS).cuda()
    a = lcg_f32(seeds["a"], seqlen * NUM_V_HEADS).view(1, seqlen, NUM_V_HEADS).cuda()
    conv_n = num_slots * QKV_DIM * CONV_K
    rec_n = num_slots * history_stride * NUM_V_HEADS * HEAD_DIM * HEAD_DIM
    if zero:
        conv_state = torch.zeros(num_slots, QKV_DIM, CONV_K, dtype=torch.bfloat16, device="cuda")
        recurrent = torch.zeros(
            num_slots, history_stride, NUM_V_HEADS, HEAD_DIM, HEAD_DIM,
            dtype=torch.float32, device="cuda",
        )
    else:
        conv_state = lcg_bf16_trunc(seeds["conv"], conv_n).view(num_slots, QKV_DIM, CONV_K).cuda()
        recurrent = lcg_f32(seeds["rec"], rec_n).view(
            num_slots, history_stride, NUM_V_HEADS, HEAD_DIM, HEAD_DIM,
        ).cuda()

    prefix = int(prefix_n)
    prefixes = {
        "qkv_prefix.bin": byte_prefix(qkv, prefix),
        "z_prefix.bin": byte_prefix(z, prefix),
        "b_prefix.bin": byte_prefix(b, prefix),
        "a_prefix.bin": byte_prefix(a, prefix),
        "conv_prefix.bin": byte_prefix(conv_state, prefix),
        "rec_prefix.bin": byte_prefix(recurrent, prefix),
    }

    # gated_delta_net.py 1106 and 1111-1117.
    mixed = qkv.transpose(1, 2).to(torch.bfloat16).contiguous()
    beta = torch.empty(1, seqlen, NUM_V_HEADS, dtype=torch.bfloat16, device="cuda")
    g = torch.empty(1, seqlen, NUM_V_HEADS, dtype=torch.float32, device="cuda")
    ext.gated_delta_net_fused_op_2(
        b, a, dt_bias.cuda(), a_log.cuda(), beta, g, float(beta_scale),
    )

    slots = torch.tensor([slot], dtype=torch.int32, device="cuda")
    slots_cpu = torch.tensor([slot], dtype=torch.int32)
    # conv1d.py 364-419. seq_len > 32 takes the Triton kernels.
    mixed = causal_conv1d_update(
        mixed_qkv=mixed,
        conv_state=conv_state,
        recurrent_slots=slots,
        conv1d_weight=conv_w.cuda(),
        conv1d_bias=None,
        history=False,
        params={},
    )
    # gated_delta_rule.py. The chunk path reads the CPU slot tensor from params.
    core = gated_delta_rule_fn(
        mixed_qkv=mixed,
        beta=beta,
        g=g,
        recurrent_state=recurrent,
        recurrent_slots=slots,
        history=False,
        save_state=True,
        num_k_heads=NUM_K_HEADS,
        num_v_heads=NUM_V_HEADS,
        k_dim=K_DIM,
        v_dim=V_DIM,
        k_head_dim=HEAD_DIM,
        v_head_dim=HEAD_DIM,
        params={"recurrent_slots": slots_cpu},
        channelwise_g=False,
    )
    y = torch.empty_like(core, dtype=torch.float16)
    # gated_rmsnorm.py 104-121 / norm.cu gated_rms_norm. silu, bias 0, groups 1.
    ext.gated_rms_norm(core, norm_w.cuda(), y, z, 1e-6, 0.0, 1, False, 0)
    torch.cuda.synchronize()

    out_dir.mkdir(parents=True, exist_ok=True)
    write_bytes(out_dir / "out.bin", y.view(torch.float16))
    write_bytes(out_dir / "conv_state.bin", conv_state)
    write_bytes(out_dir / "recurrent.bin", recurrent)
    for name, raw in prefixes.items():
        (out_dir / name).write_bytes(raw)
    print(
        f"case seq={seqlen} zero={zero} slot={slot} slots={num_slots} hs={history_stride} "
        f"out={y.numel() * 2} conv={conv_state.numel() * 2} rec={recurrent.numel() * 4}",
        flush=True,
    )
    del qkv, z, b, a, mixed, beta, g, core, y, conv_state, recurrent
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--meta", required=True)
    args = parser.parse_args()
    if port_busy():
        raise SystemExit("port 8889 is listening; not touching the GPU")

    import torch
    free, _total = torch.cuda.mem_get_info()
    # The 1499-token chunk keeps fla intermediates under 3 GiB. Leave 2 GiB free
    # so this process cannot start on a GPU that is already busy.
    if free < 2 << 30:
        raise SystemExit(f"GPU has {free} bytes free; refusing")

    meta = json.loads(Path(args.meta).read_text())
    manifest = json.loads(Path(args.manifest).read_text())
    pin(manifest)
    weights = load_weights()
    out = Path(args.out)
    weight_dir = out / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in zip(("a_log", "dt_bias", "conv_weight", "norm_weight"), weights):
        write_bytes(weight_dir / f"{name}.bin", tensor)

    for index, case in enumerate(meta["cases"]):
        run_case(case, meta["seeds"], meta["prefix_bytes"], meta["beta_scale"], weights, out / f"case_{index:02d}")


if __name__ == "__main__":
    main()

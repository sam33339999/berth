#!/usr/bin/env python3
"""Bit-exact references for the 27B Gated DeltaNet decode kernels.

Writes little-endian raw tensors and meta.json under testdata/kernels/gdn/.
Run with the serve environment (torch's nvidia libs before the system CUDA):

    PYTHON=/home/sam/qwen38-exl3-dflash2/env/bin/python
    export PYTHONPATH=/home/sam/qwen38-exl3-dflash2/exllamav3
    export TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu
    export EXL3_INT8_GEMV=2 EXL3_INT8_GEMV_MAX_K=5 EXL3_QC_STAGING=1
    export EXL3_HGEMM_F16ACC=auto
    # LD_LIBRARY_PATH: torch's bundled nvidia/*/lib first, as serve-host.sh does.

Does not load the 27B. Exits before importing torch if :8889 is listening.

gated_delta_net_fused_op_3 has no Python binding (bindings.cpp registers
fused_op, fused_op_2, and not fused_op_3). mixed_qkv is
qkv.transpose(1, 2).contiguous().to(bfloat16), which is the rn cast the kernel
calls trunc_bf16. beta and g come from ext.gated_delta_net_fused_op_2 on
b = ba[..., :H] and a = ba[..., H:]. That kernel uses the same __expf, softplus,
and __float2bfloat16_rn as fused_op_3, one element at a time. The other four
ops are the extension entry points the decode path calls.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / "testdata" / "kernels" / "gdn"

HIDDEN = 5120
NV = 48
NK = 16
HD = 128
F = 2 * NK * HD + NV * HD  # 10240
BA_N = 2 * NV  # 96
CONV_K = 4
TORCH_SEED = 0x6D6E3147
LCG_SEED = 0x6D6E3147
BETA_SCALE = 1.0
EPS = 1e-6

LAUNCHES = {
    "ba": "gdn_ba_gemv_kernel grid=12x1x1 block=256x1x1 smem=0",
    "fused": "gated_delta_net_fused_op_3_kernel<bf16> grid=41x1x1 block=256x1x1 smem=0",
    "conv": "conv1d_update_kernel<1,0> grid=40x1x1 block=256x1x1 smem=0",
    "recurrent": "cuda_recurrent_gated_delta_rule_kernel_128<0,4,0> grid=1x48x4 block=128x4x1 smem=0",
    "rms": "gated_rms_norm_kernel<32,half,bf16,float> grid=48x1x1 block=32x1x1 smem=0",
}


def port_blocks_gpu() -> bool:
    try:
        out = subprocess.run(["ss", "-ltn"], capture_output=True, text=True, check=False)
    except OSError:
        return True
    if out.returncode != 0:
        return True
    return ":8889 " in out.stdout


def lcg_f32_bytes(seed: int, n: int) -> bytes:
    """Same generator the Rust test uses for the recurrent state.

    x = seed; repeat n times:
        x = x * 1664525 + 1013904223 (mod 2^32)
        bits = (x & 0x007fffff) | 0x3f000000
        if x & 0x100: bits |= 0x80000000
        store bits as little-endian u32
    Values land in [0.5, 1) with the sign bit taken from LCG bit 8.
    """
    x = seed & 0xFFFFFFFF
    out = bytearray()
    for _ in range(n):
        x = (x * 1664525 + 1013904223) & 0xFFFFFFFF
        bits = (x & 0x007FFFFF) | 0x3F000000
        if x & 0x100:
            bits |= 0x80000000
        out += bits.to_bytes(4, "little")
    return bytes(out)


def main() -> None:
    if sys.byteorder != "little":
        raise SystemExit("this script writes little-endian raw tensors")
    if port_blocks_gpu():
        raise SystemExit("port 8889 is listening or ss failed; not touching the GPU")

    import torch
    from exllamav3.ext import exllamav3_ext as ext

    torch.manual_seed(0)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(TORCH_SEED)

    def randn(shape, scale, dtype):
        t = torch.randn(shape, generator=gen, dtype=torch.float32) * scale
        if dtype is torch.bfloat16:
            return t.to(torch.bfloat16)
        if dtype is torch.float16:
            return t.to(torch.float16)
        return t

    def raw_bytes(t: "torch.Tensor") -> bytes:
        t = t.detach().contiguous().cpu()
        if t.dtype is torch.bfloat16:
            t = t.view(torch.int16)
        elif t.dtype is torch.float16:
            t = t.view(torch.int16)
        elif t.dtype is torch.float32:
            t = t.view(torch.int32)
        elif t.dtype is torch.int32:
            pass
        else:
            raise SystemExit(f"unsupported dtype {t.dtype}")
        return t.numpy().tobytes()

    files: dict[str, int] = {}

    def save(name: str, t) -> None:
        data = raw_bytes(t)
        (OUT / name).write_bytes(data)
        files[name] = len(data)

    def finite(name: str, t) -> None:
        if not torch.isfinite(t.detach().float()).all():
            raise SystemExit(f"{name} has non-finite values")

    OUT.mkdir(parents=True, exist_ok=True)

    x = randn((1, HIDDEN), 0.5, torch.float16).cuda()
    w_ba = randn((BA_N, HIDDEN), 0.02, torch.float16).cuda()
    bias = randn((BA_N,), 0.1, torch.float16).cuda()
    y = torch.empty(1, BA_N, dtype=torch.float32, device="cuda")
    y_bias = torch.empty_like(y)
    ext.gdn_ba_gemv(x, w_ba, None, y)
    ext.gdn_ba_gemv(x, w_ba, bias, y_bias)
    finite("y_ba", y)
    finite("y_ba_bias", y_bias)

    qkv = (torch.randn(1, 1, F, generator=gen, dtype=torch.float32) * 0.5).cuda()
    dt_bias = randn((NV,), 0.25, torch.bfloat16).cuda()
    a_log = randn((NV,), 0.25, torch.bfloat16).cuda()
    # Closest fork path to gated_delta_net_fused_op_3. See the module docstring.
    mixed = qkv.transpose(1, 2).contiguous().to(torch.bfloat16)
    ba = y.view(1, 1, BA_N)
    b = ba[..., :NV].contiguous()
    a = ba[..., NV:].contiguous()
    beta = torch.empty(1, 1, NV, dtype=torch.bfloat16, device="cuda")
    g = torch.empty(1, 1, NV, dtype=torch.float32, device="cuda")
    ext.gated_delta_net_fused_op_2(b, a, dt_bias, a_log, beta, g, BETA_SCALE)
    finite("mixed_qkv", mixed)
    finite("beta", beta)
    finite("g", g)

    w_conv = randn((F, CONV_K), 0.05, torch.bfloat16).cuda()
    conv_out_zero = torch.empty(1, 1, F, dtype=torch.bfloat16, device="cuda")
    state_zero = torch.zeros(1, F, CONV_K, dtype=torch.bfloat16, device="cuda")
    ext.cuda_causal_conv1d_update(
        mixed, state_zero, None, w_conv, None, conv_out_zero, True, False
    )
    finite("conv_out_zero", conv_out_zero)
    finite("conv_state_zero_out", state_zero)
    if not state_zero.float().abs().sum() > 0:
        raise SystemExit("zero-state conv writeback stayed zero")

    conv_state_nz = randn((1, F, CONV_K), 0.25, torch.bfloat16).cuda()
    conv_state_nz_init = conv_state_nz.clone()
    conv_out_nz = torch.empty_like(conv_out_zero)
    slots = torch.tensor([0], dtype=torch.int32, device="cuda")
    ext.cuda_causal_conv1d_update(
        mixed, conv_state_nz, slots, w_conv, None, conv_out_nz, True, False
    )
    finite("conv_out_nz", conv_out_nz)
    if torch.equal(conv_out_zero, conv_out_nz):
        raise SystemExit("conv outputs did not depend on the initial state")
    if torch.equal(conv_state_nz_init, conv_state_nz):
        raise SystemExit("nonzero conv state was not updated")

    def recurrent(conv_out, state, slot):
        core = torch.empty(1, 1, NV, HD, dtype=torch.bfloat16, device="cuda")
        ext.cuda_recurrent_gated_delta_rule(
            conv_out, g, beta, state, core, NK, NV, HD, HD, slot, False
        )
        finite("core_attn_out", core)
        finite("recurrent_state", state)
        return core

    state_rec_zero = torch.zeros(1, 1, NV, HD, HD, dtype=torch.float32, device="cuda")
    core_zero = recurrent(conv_out_zero, state_rec_zero, None)
    if not state_rec_zero.float().abs().sum() > 0:
        raise SystemExit("zero recurrent state produced a zero update")

    n_state = NV * HD * HD
    lcg_raw = lcg_f32_bytes(LCG_SEED, n_state)
    lcg_cpu = torch.frombuffer(bytearray(lcg_raw), dtype=torch.float32).clone().view(1, 1, NV, HD, HD)
    state_rec_nz = lcg_cpu.cuda()
    core_nz = recurrent(conv_out_nz, state_rec_nz, slots)
    if torch.equal(core_zero, core_nz):
        raise SystemExit("core_attn_out did not depend on the recurrent state")

    w_norm = (1.0 + randn((HD,), 0.1, torch.float32)).to(torch.bfloat16).cuda()
    z = (torch.randn(1, 1, NV, HD, generator=gen, dtype=torch.float32) * 0.5).cuda()
    y_rms_zero = torch.empty(1, 1, NV, HD, dtype=torch.float16, device="cuda")
    y_rms_nz = torch.empty_like(y_rms_zero)
    ext.gated_rms_norm(core_zero, w_norm, y_rms_zero, z, EPS, 0.0, 1, False, 0)
    ext.gated_rms_norm(core_nz, w_norm, y_rms_nz, z, EPS, 0.0, 1, False, 0)
    finite("y_rms_zero", y_rms_zero)
    finite("y_rms_nz", y_rms_nz)

    save("w_ba.bin", w_ba)
    save("x_ba.bin", x)
    save("bias_ba.bin", bias)
    save("y_ba.bin", y)
    save("y_ba_bias.bin", y_bias)
    save("qkv.bin", qkv)
    save("dt_bias.bin", dt_bias)
    save("a_log.bin", a_log)
    save("mixed_qkv.bin", mixed)
    save("beta.bin", beta)
    save("g.bin", g)
    save("w_conv.bin", w_conv)
    save("conv_state_nz.bin", conv_state_nz_init)
    save("conv_out_zero.bin", conv_out_zero)
    save("conv_state_zero_out.bin", state_zero)
    save("conv_out_nz.bin", conv_out_nz)
    save("conv_state_nz_out.bin", conv_state_nz)
    save("core_zero.bin", core_zero)
    save("core_nz.bin", core_nz)
    save("w_norm.bin", w_norm)
    save("z.bin", z)
    save("y_rms_zero.bin", y_rms_zero)
    save("y_rms_nz.bin", y_rms_nz)
    save("slots_nz.bin", slots)

    zero_hash = hashlib.sha256(raw_bytes(state_rec_zero)).hexdigest()
    nz_hash = hashlib.sha256(raw_bytes(state_rec_nz)).hexdigest()
    lcg_hash = hashlib.sha256(lcg_raw).hexdigest()
    total = sum(files.values())
    if total >= 3_000_000:
        raise SystemExit(f"testdata is {total} bytes, budget is 3000000")

    meta = {
        "shapes": {
            "hidden": HIDDEN,
            "num_k_heads": NK,
            "num_v_heads": NV,
            "head_dim": HD,
            "f": F,
            "ba_n": BA_N,
            "conv_k": CONV_K,
            "beta_scale": BETA_SCALE,
            "epsilon": EPS,
            "constant_bias": 0.0,
            "w_groups": 1,
            "gate_first": 0,
            "gate_act": 0,
            "history_stride": 1,
            "bsz": 1,
            "seqlen": 1,
        },
        "launches": LAUNCHES,
        "torch_seed": TORCH_SEED,
        "lcg": {
            "seed": LCG_SEED,
            "count": n_state,
            "prefix_hex": lcg_raw[:32].hex(),
            "sha256": lcg_hash,
            "algorithm": (
                "x = seed; x = x*1664525+1013904223 mod 2^32; "
                "bits = (x & 0x007fffff) | 0x3f000000; "
                "if x & 0x100: bits |= 0x80000000; store le u32"
            ),
        },
        "zero": {
            "slots": None,
            "conv_state": "zeros",
            "recurrent_state": "zeros",
            "recurrent_sha256": zero_hash,
        },
        "nonzero": {
            "slots": [0],
            "conv_state": "conv_state_nz.bin",
            "recurrent_state": "lcg",
            "recurrent_sha256": nz_hash,
        },
        "bytes": files,
        "total_bytes": total,
        "fused_op_3": (
            "No Python binding. mixed_qkv is qkv.transpose(1, 2).contiguous().to(bfloat16) "
            "(rn cast). beta and g are ext.gated_delta_net_fused_op_2 on b=ba[...,:H], "
            "a=ba[...,H:], same __expf / softplus / __float2bfloat16_rn as fused_op_3."
        ),
        "fork_ops": {
            "y_ba": "ext.gdn_ba_gemv",
            "y_ba_bias": "ext.gdn_ba_gemv",
            "mixed_qkv": "torch transpose + bf16 cast (fused_op_3 substitute)",
            "beta": "ext.gated_delta_net_fused_op_2",
            "g": "ext.gated_delta_net_fused_op_2",
            "conv": "ext.cuda_causal_conv1d_update(..., activation=True, history=False)",
            "recurrent": "ext.cuda_recurrent_gated_delta_rule(..., history=False)",
            "rms": "ext.gated_rms_norm",
        },
        "checkpoint": {
            "w_ba": "cat of linear_attn.in_proj_b.weight and in_proj_a.weight as [96, 5120] fp16, b then a. No bias in the checkpoint; bias_ba.bin is an extra case.",
            "dt_bias": "linear_attn.dt_bias bf16 [48]",
            "a_log": "linear_attn.A_log bf16 [48]",
            "w_conv": "linear_attn.conv1d.weight bf16 [10240, 1, 4] squeezed to [10240, 4]. No conv1d.bias.",
            "w_norm": "linear_attn.norm.weight bf16 [128], w_groups=1",
            "conv_state": "bf16 [num_slots, 10240, state_size]. Cache is [max_batch, F, conv_k + max_history]. history=false uses the first conv_k taps.",
            "recurrent_state": "f32 [num_slots, history_stride, 48, 128, 128], k-major. history=false uses history index 0. Not stored; input is the LCG, updated state is sha256.",
        },
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {OUT} ({total} bytes, budget 3000000)")
    print(f"recurrent zero sha256 {zero_hash}")
    print(f"recurrent nonzero sha256 {nz_hash}")


if __name__ == "__main__":
    main()

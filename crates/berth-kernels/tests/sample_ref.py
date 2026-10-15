#!/usr/bin/env python3
"""Fork reference for the sampler kernels and the serve's CustomSampler.

Writes testdata/kernels/sample/meta.json and breakers.u32.bin. Logit rows are
not stored: both sides regenerate them from the LCG described in meta.json.
Post-penalty rows and kernel outputs are blake2b-256 digests. Fused outputs
are token ids.

The sampler cases are the serve's stack, not a hand-rolled chain:

  CustomSampler([SS_RepP, SS_PresFreqP, SS_DRY, SS_Argmax | (SS_Temperature, SS_Sample)])

Identity steps drop out and the tail collapses into SS_Fused. EXL3_FUSED_SAMPLER
must stay at its default (fused on). The job draws randint(0, 2**32-1) before
forward; this script does the same with random.Random(seed).

Aborts when port 8889 is listening. Does not load the 27B weights.
"""

import hashlib
import json
import os
import random
import socket
import struct
import sys

import numpy as np
import torch

FORK = "/home/sam/qwen38-exl3-dflash2/exllamav3"
MODEL = "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3"
sys.path.insert(0, FORK)

from exllamav3 import Config, Tokenizer  # noqa: E402
from exllamav3.ext import exllamav3_ext as ext  # noqa: E402
from exllamav3.generator.sampler.custom import (  # noqa: E402
    SS_Argmax,
    SS_DRY,
    SS_Fused,
    SS_PresFreqP,
    SS_RepP,
    SS_Sample,
    SS_Temperature,
    CustomSampler,
    _DRY_MATCH_CAP,
    _dry_max_exponent,
    dry_default_sequence_breaker_tokens,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "testdata", "kernels", "sample"))


def port_listening(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex(("127.0.0.1", port)) == 0
    except OSError:
        return True


def blake(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def f32_bits(value) -> int:
    """Bits of the f32 the fork's pybind cast would pass to the kernel."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", int(bits) & 0xFFFFFFFF))[0]


def lcg_next(x: int) -> int:
    return (x * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF


def lcg_row(seed: int, n: int, dtype: str) -> bytes:
    """One logit row. fp16 and fp32 packings are documented in meta.json."""
    x = seed & 0xFFFFFFFF
    if dtype == "f16":
        out = bytearray(n * 2)
        for i in range(n):
            x = lcg_next(x)
            mant = (x >> 8) & 0x3FF
            exp = 14 + ((x >> 20) & 3)
            sign = x >> 31
            bits = (sign << 15) | (exp << 10) | mant
            struct.pack_into("<H", out, i * 2, bits)
        return bytes(out)
    if dtype == "f32":
        out = bytearray(n * 4)
        for i in range(n):
            x = lcg_next(x)
            bits = (x & 0x007FFFFF) | 0x3F800000
            if x & 1:
                bits |= 0x80000000
            struct.pack_into("<I", out, i * 4, bits)
        return bytes(out)
    raise ValueError(dtype)


def lcg_ids(seed: int, n: int, mod: int, repeat_tail: int) -> list:
    x = seed & 0xFFFFFFFF
    ids = []
    for _ in range(n):
        x = lcg_next(x)
        ids.append(x % mod)
    if repeat_tail and n >= 2 * repeat_tail:
        pattern = ids[n - 2 * repeat_tail : n - repeat_tail]
        ids[n - repeat_tail :] = pattern
    return ids


def history_bytes(ids: list) -> bytes:
    return b"".join(struct.pack("<I", int(t)) for t in ids)


def apply_patch(row: bytes, patch, dtype: str) -> bytes:
    if not patch:
        return row
    buf = bytearray(row)
    index = int(patch["index"])
    bits = int(patch["f16_bits"]) & 0xFFFF
    if dtype != "f16":
        raise ValueError("patch is fp16")
    struct.pack_into("<H", buf, index * 2, bits)
    return bytes(buf)


def cuda_f(row: bytes, dtype: str, vocab: int):
    np_dtype = np.float16 if dtype == "f16" else np.float32
    host = np.frombuffer(row, dtype=np_dtype).copy()
    return torch.tensor(host, dtype=torch.float16 if dtype == "f16" else torch.float32, device="cuda").view(1, vocab)


def cuda_ids(ids: list):
    return torch.tensor([ids], dtype=torch.long, device="cuda")


def tensor_bytes(t: torch.Tensor) -> bytes:
    flat = t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return flat


def run_kernel_case(case, vocab):
    dtype = case["dtype"]
    row = apply_patch(lcg_row(case["seed"], vocab, dtype), case.get("patch"), dtype)
    case["input_blake2b256"] = blake(row)
    ids = []
    if case["op"] in ("rep", "pres", "dry"):
        ids = lcg_ids(case["history_seed"], case["history_len"], case["history_mod"], case["repeat_tail"])
        case["history_blake2b256"] = blake(history_bytes(ids))
        if case["op"] == "dry" and case["breakers"] == "from_history":
            # A breaker that sits inside the repeated tail, so the scan stops early.
            case["breakers"] = [int(ids[-3])]
    logits = cuda_f(row, dtype, vocab)
    past = cuda_ids(ids) if ids else None
    out_dtype = torch.float32
    if case["op"].startswith("gumbel") and dtype == "f16":
        out_dtype = torch.float16
    if case["inplace"]:
        out = logits
    elif case["op"] == "fused":
        out = torch.empty((1, 1), dtype=torch.long, device="cuda")
    else:
        elem = torch.float16 if out_dtype == torch.float16 else torch.float32
        out = torch.empty((1, vocab), dtype=elem, device="cuda")

    p = case["params"]
    if case["op"] == "rep":
        ext.apply_rep_pens(logits, out, past, bits_to_f32(p["rep_p_bits"]), int(p["sustain"]), int(p["decay"]))
    elif case["op"] == "pres":
        ext.apply_pres_freq_pens(
            logits, out, past,
            bits_to_f32(p["pres_p_bits"]), bits_to_f32(p["freq_p_bits"]),
            int(p["sustain"]), int(p["decay"]),
        )
    elif case["op"] == "dry":
        breakers = None
        if case["breakers"]:
            mask = torch.zeros((vocab,), dtype=torch.bool, device="cuda")
            keep = [t for t in case["breakers"] if 0 <= int(t) < vocab]
            if keep:
                mask[torch.tensor(keep, dtype=torch.long, device="cuda")] = True
            breakers = mask
        scratch = torch.full((vocab + 1,), -1, dtype=torch.int32, device="cuda")
        ext.dry_penalty(
            logits, out, past, breakers, scratch[:vocab], scratch[vocab:],
            bits_to_f32(p["multiplier_bits"]), bits_to_f32(p["base_bits"]),
            int(p["allowed_length"]), int(p["range"]), int(p["max_exponent"]), int(p["match_cap"]),
        )
    elif case["op"] == "gumbel_f16":
        ext.gumbel_noise_f16(logits, out, int(p["random"]) & 0xFFFFFFFF)
    elif case["op"] == "gumbel_f32":
        ext.gumbel_noise_f32(logits, out, int(p["random"]) & 0xFFFFFFFF)
    elif case["op"] == "fused":
        workspace = torch.empty((ext.FUSED_SAMPLER_MAX_BLOCKS * 3,), dtype=torch.float32, device="cuda")
        # Argument order is the C++ host function: minp_log sits between inv_temp and random.
        ext.fused_sampler(
            logits, None, None, out, workspace, int(p["size"]),
            bits_to_f32(p["inv_temp_bits"]), 0.0,
            int(p["random"]) & 0xFFFFFFFF, int(p["mode"]),
            0, 0, 1.0, 1.0, None,
        )
    else:
        raise ValueError(case["op"])
    torch.cuda.synchronize()
    if case["op"] == "fused":
        case["token"] = int(out.item())
    else:
        case["output_blake2b256"] = blake(tensor_bytes(out))
    return case


def serve_sampler(temperature, rep, pres, freq, freq_range, dry_mult, dry_base, dry_allowed, dry_range):
    steps = [
        SS_RepP(rep, freq_range, 0),
        SS_PresFreqP(pres, freq, freq_range, 0),
        SS_DRY(dry_mult, dry_base, dry_allowed, dry_range),
    ]
    if not temperature:
        steps.append(SS_Argmax())
    else:
        steps.append(SS_Temperature(temperature))
        steps.append(SS_Sample())
    sampler = CustomSampler(steps)
    described = []
    for step in sampler.steps:
        name = type(step).__name__
        if isinstance(step, SS_Fused):
            described.append(f"SS_Fused(mode={step.mode},inv_temp_bits={f32_bits(step.inv_temp)})")
        else:
            described.append(name)
    return sampler, described


def run_sampler_case(spec, rows, prompt, tokenizer, vocab):
    temperature = spec["temperature"]
    sampler, stack = serve_sampler(
        temperature,
        spec["rep_penalty"],
        spec["presence_penalty"],
        spec["frequency_penalty"],
        spec["penalty_range"],
        spec["dry_multiplier"],
        spec["dry_base"],
        spec["dry_allowed_length"],
        spec["dry_range"],
    )
    assert stack == spec["expect_stack"], (stack, spec["expect_stack"])
    rng = random.Random(spec["seed"])
    past_list = list(prompt)
    tokens = []
    posts = []
    post_dtype = None
    for row in rows:
        logits = cuda_f(row, "f16", vocab)
        rand_u32 = rng.randint(0, (1 << 32) - 1)
        past = cuda_ids(past_list)
        state = sampler.forward(
            logits, past, rand_u32, tokenizer, return_state=True,
        )
        token = int(state.sample.item())
        tokens.append(token)
        if state.logits is not None:
            posts.append(blake(tensor_bytes(state.logits)))
            post_dtype = "f32"
        else:
            posts.append(blake(tensor_bytes(logits)))
            post_dtype = "f16"
        past_list.append(token)
    spec["stack"] = stack
    spec["tokens"] = tokens
    spec["post_blake2b256"] = posts
    spec["post_dtype"] = post_dtype
    if spec["post_dtype"] == "f32":
        # The penalty path upcasts; the tail -inf is inside that fp32 row.
        assert state.logits.dtype == torch.float32
        assert state.logits.numel() == vocab
    return spec


def main():
    if port_listening(8889):
        raise SystemExit("port 8889 is listening; refusing to use the GPU")
    if os.environ.get("EXL3_FUSED_SAMPLER") == "0":
        raise SystemExit("EXL3_FUSED_SAMPLER=0 would leave the unfused path; unset it")

    os.makedirs(OUT, exist_ok=True)
    cfg = Config.from_directory(MODEL)
    tokenizer = Tokenizer.from_config(cfg)
    vocab = int(cfg.vocab_size)
    limit = int(tokenizer.actual_vocab_size)
    if vocab != 248320 or limit != 248077:
        raise SystemExit(f"unexpected vocab {vocab} limit {limit}")

    breaker_ids = sorted(int(t) for t in dry_default_sequence_breaker_tokens(tokenizer))
    breaker_path = os.path.join(OUT, "breakers.u32.bin")
    with open(breaker_path, "wb") as fh:
        fh.write(b"".join(struct.pack("<I", t) for t in breaker_ids))

    inv_07 = f32_bits(1.0 / 0.7)
    assert inv_07 == f32_bits(1.0 / float(np.float32(0.7)))
    assert _dry_max_exponent(1.75) == 158
    assert _dry_max_exponent(2.0) == 128
    assert _dry_max_exponent(1.0) == 0
    assert _dry_max_exponent(1.000001) == 0

    cases = [
        dict(name="rep_f16", op="rep", seed=11, history_seed=12, history_len=600, history_mod=vocab,
             repeat_tail=12, dtype="f16", inplace=False, breakers=[], patch=None,
             params=dict(rep_p_bits=f32_bits(1.1), sustain=512, decay=0),
             launch_contains=["apply_rep_pens_kernel<1>"]),
        dict(name="rep_f32", op="rep", seed=13, history_seed=14, history_len=40, history_mod=vocab,
             repeat_tail=0, dtype="f32", inplace=True, breakers=[], patch=None,
             params=dict(rep_p_bits=f32_bits(1.25), sustain=8, decay=16),
             launch_contains=["apply_rep_pens_kernel<0>"]),
        dict(name="pres_f16", op="pres", seed=21, history_seed=22, history_len=600, history_mod=vocab,
             repeat_tail=12, dtype="f16", inplace=False, breakers=[], patch=None,
             params=dict(pres_p_bits=f32_bits(0.2), freq_p_bits=f32_bits(0.2), sustain=512, decay=0),
             launch_contains=["apply_pres_freq_pens_kernel<1>"]),
        dict(name="pres_f32", op="pres", seed=23, history_seed=24, history_len=30, history_mod=vocab,
             repeat_tail=0, dtype="f32", inplace=True, breakers=[], patch=None,
             params=dict(pres_p_bits=f32_bits(0.2), freq_p_bits=f32_bits(0.5), sustain=4, decay=8),
             launch_contains=["apply_pres_freq_pens_kernel<0>"]),
        dict(name="dry_f16", op="dry", seed=31, history_seed=32, history_len=200, history_mod=vocab,
             repeat_tail=12, dtype="f16", inplace=False, breakers=[], patch=None,
             params=dict(multiplier_bits=f32_bits(0.2), base_bits=f32_bits(1.75),
                         allowed_length=6, range=4096, max_exponent=_dry_max_exponent(1.75),
                         match_cap=_DRY_MATCH_CAP),
             launch_contains=["dry_penalty_kernel<1>"]),
        dict(name="dry_f32", op="dry", seed=33, history_seed=34, history_len=80, history_mod=vocab,
             repeat_tail=12, dtype="f32", inplace=True, breakers="from_history", patch=None,
             params=dict(multiplier_bits=f32_bits(0.3), base_bits=f32_bits(2.0),
                         allowed_length=2, range=64, max_exponent=_dry_max_exponent(2.0),
                         match_cap=_DRY_MATCH_CAP),
             launch_contains=["dry_penalty_kernel<0>"]),
        dict(name="gumbel_f32", op="gumbel_f32", seed=41, dtype="f32", inplace=True,
             breakers=[], patch=None,
             params=dict(random=0xC0FFEE),
             launch_contains=["gumbel_noise_kernel_f32"]),
        dict(name="gumbel_f16", op="gumbel_f16", seed=42, dtype="f16", inplace=False,
             breakers=[], patch=None,
             params=dict(random=0xC0FFEE),
             launch_contains=["gumbel_noise_kernel_f16"]),
        dict(name="fused_greedy_f16", op="fused", seed=51, dtype="f16", inplace=False,
             breakers=[], patch=None,
             params=dict(mode=0, size=vocab, random=0x11111111, inv_temp_bits=f32_bits(1.0)),
             launch_contains=["fs_partial_argmax_kernel<half,0>", "fs_finalize_kernel"]),
        dict(name="fused_greedy_f32", op="fused", seed=52, dtype="f32", inplace=False,
             breakers=[], patch=None,
             params=dict(mode=0, size=vocab, random=0x22222222, inv_temp_bits=f32_bits(1.0)),
             launch_contains=["fs_partial_argmax_kernel<float,0>", "fs_finalize_kernel"]),
        dict(name="fused_sample_f16", op="fused", seed=53, dtype="f16", inplace=False,
             breakers=[], patch=None,
             params=dict(mode=1, size=vocab, random=0x33333333, inv_temp_bits=inv_07),
             launch_contains=["fs_partial_argmax_kernel<half,1>", "fs_finalize_kernel"]),
        dict(name="fused_sample_f32", op="fused", seed=54, dtype="f32", inplace=False,
             breakers=[], patch=None,
             params=dict(mode=1, size=vocab, random=0x44444444, inv_temp_bits=inv_07),
             launch_contains=["fs_partial_argmax_kernel<float,1>", "fs_finalize_kernel"]),
        dict(name="fused_sample_f16_bounded", op="fused", seed=55, dtype="f16", inplace=False,
             breakers=[], patch={"index": vocab - 1, "f16_bits": 0x7BFF},
             params=dict(mode=1, size=limit, random=0x55555555, inv_temp_bits=inv_07),
             launch_contains=["fs_partial_argmax_kernel<half,1>", "fs_finalize_kernel"]),
    ]
    for case in cases:
        run_kernel_case(case, vocab)
        if case["name"] == "fused_sample_f16_bounded":
            assert case["token"] != vocab - 1, case["token"]

    prompt_seed = 0xC0FFEE
    prompt_len = 600
    steps = 20
    logits_seed0 = 0x5A170001
    prompt = lcg_ids(prompt_seed, prompt_len, limit, 12)
    rows = [lcg_row((logits_seed0 + i) & 0xFFFFFFFF, vocab, "f16") for i in range(steps)]
    input_hashes = [blake(row) for row in rows]

    greedy_stack = ["SS_Fused(mode=0,inv_temp_bits=%d)" % f32_bits(1.0)]
    sample_stack_dry = [
        "SS_DRY",
        "SS_Fused(mode=1,inv_temp_bits=%d)" % inv_07,
    ]
    penalty_stack = [
        "SS_RepP",
        "SS_PresFreqP",
        "SS_DRY",
        "SS_Fused(mode=1,inv_temp_bits=%d)" % inv_07,
    ]
    sampler_cases = {
        "greedy": dict(
            temperature=0.0, seed=1,
            rep_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0, penalty_range=512,
            dry_multiplier=0.0, dry_base=1.75, dry_allowed_length=6, dry_range=4096,
            expect_stack=greedy_stack,
        ),
        "defaults": dict(
            temperature=0.7, seed=42,
            rep_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0, penalty_range=512,
            dry_multiplier=0.2, dry_base=1.75, dry_allowed_length=6, dry_range=4096,
            expect_stack=sample_stack_dry,
        ),
        "penalties": dict(
            temperature=0.7, seed=99,
            rep_penalty=1.1, presence_penalty=0.2, frequency_penalty=0.2, penalty_range=512,
            dry_multiplier=0.2, dry_base=1.75, dry_allowed_length=6, dry_range=4096,
            expect_stack=penalty_stack,
        ),
    }
    for name, spec in sampler_cases.items():
        run_sampler_case(spec, rows, prompt, tokenizer, vocab)
        assert spec["post_dtype"] == ("f16" if name == "greedy" else "f32")
        assert len(spec["tokens"]) == steps

    # Float parameters the Rust sampler should load with f32::from_bits.
    for spec in sampler_cases.values():
        for key in ("temperature", "rep_penalty", "presence_penalty", "frequency_penalty",
                    "dry_multiplier", "dry_base"):
            spec[key + "_bits"] = f32_bits(spec[key])

    meta = {
        "vocab": vocab,
        "logit_limit": limit,
        "lcg": "x = (x * 1664525 + 1013904223) & 0xffffffff, starting from seed; emit after each step",
        "fp16": "mant=(x>>8)&0x3ff, exp=14+((x>>20)&3), sign=x>>31, bits=(sign<<15)|(exp<<10)|mant, little-endian",
        "fp32": "bits=(x & 0x007fffff) | 0x3f800000; if x & 1 { bits |= 0x80000000 }; little-endian",
        "history": "ids[i] = next() % history_mod; if repeat_tail L and n >= 2L: ids[n-L:] = ids[n-2L:n-L]; hash is little-endian u32 bytes",
        "kernel_cases": cases,
        "sampler": {
            "prompt_seed": prompt_seed,
            "prompt_len": prompt_len,
            "prompt_mod": limit,
            "repeat_tail": 12,
            "prompt_blake2b256": blake(history_bytes(prompt)),
            "steps": steps,
            "logits_seed0": logits_seed0,
            "input_blake2b256": input_hashes,
            "breakers_file": "breakers.u32.bin",
            "breaker_count": len(breaker_ids),
            "cases": sampler_cases,
        },
    }
    meta_path = os.path.join(OUT, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    total = os.path.getsize(meta_path) + os.path.getsize(breaker_path)
    if total > 3 * 1024 * 1024:
        raise SystemExit(f"testdata is {total} bytes, over 3 MiB")
    print(f"wrote {meta_path} ({total} bytes, {len(breaker_ids)} breakers)")
    for case in cases:
        extra = case.get("token", case.get("output_blake2b256", "")[:16])
        print(f"  kernel {case['name']}: {extra}")
    for name, spec in sampler_cases.items():
        print(f"  sampler {name}: {spec['tokens'][:4]}... dtype {spec['post_dtype']} stack {spec['stack']}")


if __name__ == "__main__":
    main()

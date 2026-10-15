#!/usr/bin/env python3
"""Profile Qwen3.8-27B autoregressive decode under nsys. Needs the GPU.

Do not run while the :8889 serve is up: this loads a second 27B.

Loads the target only (no DFlash2 draft), with the serve's cache layer (cq3)
and sampler steps, so every decode step is one plain forward. The first job
warms graphs and autotune. The second job is profiled from its first decoded
token for DECODE_STEPS steps; prefill stays outside the capture range.

Run through the serve's environment (`scripts/serve-host.sh` exports), e.g.:

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \\
        --cuda-graph-trace=node --trace=cuda,nvtx -o decode \\
        python scripts/profile_27b_decode.py
"""

import os
import sys

import torch

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler.custom import (
    CustomSampler, SS_DRY, SS_PresFreqP, SS_RepP, SS_Sample, SS_Temperature,
)

TARGET = os.environ.get(
    "BERTH_27B_DIR", "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3"
)
DECODE_STEPS = int(os.environ.get("DECODE_STEPS", "32"))
CACHE_TOKENS = 8192
PROMPT = (
    "Explain, step by step, how a paged KV cache with content-hashed pages "
    "lets several requests share a common prompt prefix, and what has to "
    "happen when a page that another request still references is evicted. "
) * 4


def sampler():
    # serve_openai.py `_sampler` with the serve-host.sh defaults, minus the
    # serve's own think-exit step.
    return CustomSampler([
        SS_RepP(1.0, 512, 0),
        SS_PresFreqP(0.0, 0.0, 512, 0),
        SS_DRY(0.2, 1.75, 6, 4096),
        SS_Temperature(0.6),
        SS_Sample(),
    ])


def run(gen, ids, max_new, profile):
    gen.enqueue(Job(input_ids=ids, max_new_tokens=max_new, sampler=sampler(), seed=1))
    decoded = 0
    started = False
    while gen.num_remaining_jobs():
        if profile and decoded >= 1 and not started:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            started = True
        for r in gen.iterate():
            if r.get("stage") == "streaming" and r.get("token_ids") is not None:
                decoded += r["token_ids"].shape[-1]
    torch.cuda.synchronize()
    if started:
        torch.cuda.cudart().cudaProfilerStop()
    return decoded


def main():
    cfg = Config.from_directory(TARGET)
    model = Model.from_config(cfg)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS, layer_type=CacheLayer_quant,
                  k_bits=3, v_bits=3, max_batch_size=1)
    model.load(progressbar=False)
    tokenizer = Tokenizer.from_config(cfg)
    gen = Generator(model, cache, tokenizer)

    ids = tokenizer.encode(PROMPT)
    print(f"prompt tokens {ids.shape[-1]}", file=sys.stderr)
    run(gen, ids, 64, profile=False)
    n = run(gen, tokenizer.encode("Second prompt. " + PROMPT), DECODE_STEPS + 1, profile=True)
    print(f"decoded {n} tokens; profiled {DECODE_STEPS} steps after the first", file=sys.stderr)


if __name__ == "__main__":
    main()

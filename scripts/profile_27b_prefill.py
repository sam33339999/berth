#!/usr/bin/env python3
"""Profile Qwen3.8-27B prompt prefill under nsys. Needs the GPU.

Do not run while the :8889 serve is up: this loads a second 27B.

Same setup as profile_27b_decode.py (target only, cq3, the serve's env).
A warm-up job prefills a prompt of PREFILL_TOKENS tokens; a second job with
a different prompt of the same length is profiled from just before its
prefill to just after the first sampled token.

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \\
        --cuda-graph-trace=node --trace=cuda,nvtx -o prefill \\
        python scripts/profile_27b_prefill.py
"""

import os
import sys

import torch

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler.custom import CustomSampler, SS_Argmax

TARGET = os.environ.get(
    "BERTH_27B_DIR", "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3"
)
PREFILL_TOKENS = int(os.environ.get("PREFILL_TOKENS", "1500"))
CACHE_TOKENS = 8192


def prompt_ids(tokenizer, seed_word, n):
    text = (f"{seed_word} " + "The quick brown fox jumps over the lazy dog. ") * (n // 8 + 8)
    ids = tokenizer.encode(text)
    return ids[:, :n]


def run(gen, ids, profile):
    gen.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=CustomSampler([SS_Argmax()])))
    if profile:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
    while gen.num_remaining_jobs():
        gen.iterate()
    torch.cuda.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStop()


def main():
    cfg = Config.from_directory(TARGET)
    model = Model.from_config(cfg)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS, layer_type=CacheLayer_quant,
                  k_bits=3, v_bits=3, max_batch_size=1)
    model.load(progressbar=False)
    tokenizer = Tokenizer.from_config(cfg)
    gen = Generator(model, cache, tokenizer)
    run(gen, prompt_ids(tokenizer, "Warm", PREFILL_TOKENS), profile=False)
    ids = prompt_ids(tokenizer, "Second", PREFILL_TOKENS)
    print(f"profiled prompt tokens {ids.shape[-1]}", file=sys.stderr)
    run(gen, ids, profile=True)


if __name__ == "__main__":
    main()

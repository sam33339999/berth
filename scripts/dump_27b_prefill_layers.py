#!/usr/bin/env python3
"""Full tensors from inside blocks 0 and 3 during the fork's prefill. Needs the GPU.

Do not run while the :8889 serve is up: this loads the 27B.

Same model, cache and prompts as `dump_27b_prefill_reference.py` (100- and
1500-token prompts; the fork prefills ids[:-1] in chunks, [0, 99) and
[0, 1280) + [1280, 1499)). For every prefill chunk, blocks 0 (GDN) and 3
(full attention) record, as the fork holds them:

    block_in   the residual entering the block (fp32 [1, n, 5120])
    mixer_in   input_layernorm output (fp16)
    mixer      linear_attn / self_attn output (fp32)
    mlp_in     post_attention_layernorm output (fp16)
    mlp        mlp output (fp32)

Files are large (a 1280-row fp32 tensor is 26 MB), so they go to
`target/prefill_layers/` (or the first argument), not into the repo:
`{prompt}_c{chunk}_b{block:02d}_{name}.bin`, raw little-endian, plus
`meta.json` with each file's dtype, shape and chunk bounds. Run it twice and
compare the digests in meta.json to check determinism.
"""

import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dump_27b_prefill_reference as ref  # noqa: E402

from exllamav3 import Cache, Config, Generator, Model, Tokenizer  # noqa: E402
from exllamav3 import Job  # noqa: E402
from exllamav3.cache import CacheLayer_quant  # noqa: E402
from exllamav3.modules.transformer import TransformerBlock  # noqa: E402

BLOCKS = (0, 3)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ref.ROOT / "target" / "prefill_layers"
    out.mkdir(parents=True, exist_ok=True)
    ref.wait_for_gpu()
    cfg = Config.from_directory(ref.TARGET)
    model = Model.from_config(cfg)
    cache = Cache(model, max_num_tokens=ref.CACHE_TOKENS, layer_type=CacheLayer_quant,
                  k_bits=3, v_bits=3, max_batch_size=1)
    model.load(progressbar=False)
    tokenizer = Tokenizer.from_config(cfg)
    gen = Generator(model, cache, tokenizer)

    state = {"prompt": None, "chunk": -1, "in_prefill": False}
    files = []

    def save(tag, block, tensor):
        if not state["in_prefill"]:
            return
        dtype, shape, raw = ref.le_bytes(tensor)
        name = f"{state['prompt']}_c{state['chunk']}_b{block:02d}_{tag}.bin"
        (out / name).write_bytes(raw)
        files.append({
            "file": name, "prompt": state["prompt"], "chunk": state["chunk"],
            "block": block, "name": tag, "dtype": dtype,
            "shape": shape, "blake2b": hashlib.blake2b(raw, digest_size=32).hexdigest(),
        })

    orig_prefill = model.prefill

    def wrapped_prefill(input_ids, params=None):
        state["chunk"] += 1
        state["in_prefill"] = True
        try:
            return orig_prefill(input_ids, params)
        finally:
            state["in_prefill"] = False

    model.prefill = wrapped_prefill

    blocks = [m for m in model.modules if isinstance(m, TransformerBlock)]
    for i in BLOCKS:
        block = blocks[i]
        orig_block = block.forward

        def block_fwd(x, params, out_dtype=None, _orig=orig_block, _i=i):
            save("block_in", _i, x)
            return _orig(x, params, out_dtype)

        block.forward = block_fwd
        for mod, tag in ((block.attn_norm, "mixer_in"), (block.attn, "mixer"),
                         (block.mlp_norm, "mlp_in"), (block.mlp, "mlp")):
            orig = mod.forward

            def sub_fwd(*args, _orig=orig, _tag=tag, _i=i, **kwargs):
                y = _orig(*args, **kwargs)
                save(_tag, _i, y[0] if isinstance(y, tuple) else y)
                return y

            mod.forward = sub_fwd

    ids = ref.prompt_ids(tokenizer)
    for length in (100, 1500):
        state["prompt"], state["chunk"] = length, -1
        ref.reset_generator(gen)
        ref.zero_cache(cache)
        gen.enqueue(Job(input_ids=ids[length].clone(), max_new_tokens=1,
                        sampler=ref.ArgmaxSampler()))
        while gen.num_remaining_jobs():
            gen.iterate()
        torch.cuda.synchronize()

    (out / "meta.json").write_text(json.dumps({
        "source": "scripts/dump_27b_prefill_layers.py",
        "blocks": list(BLOCKS),
        "token_ids": {str(n): [int(v) for v in ids[n][0].tolist()] for n in (100, 1500)},
        "files": files,
    }, indent=1) + "\n")
    print(f"wrote {len(files)} tensors to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prefill reference for the 27B target. Needs the GPU.

Do not run while the :8889 serve is up: this loads the 27B (~16 GB). The
script waits while ``nvidia-smi`` reports 4000 MiB or more, up to 20 minutes,
then refuses.

Same job the serve runs for one prompt, target only (no draft)
--------------------------------------------------------------
``CacheLayer_quant`` cq3, ``max_batch_size`` 1, ``max_num_tokens`` 12288.
The cache has to hold the longest job: ``ceil((8400 + 2) / 256) = 33`` pages.
``Generator`` keeps its default ``max_chunk_size`` of 2048. Serve env flags
are set before import. ``max_new_tokens`` is 1, so ``max_rq_tokens`` is 2
and every chunk of a job sees the same block-table width
``ceil((prompt_len + 2) / 256)``. One prompt therefore has one page class.
8190 and 8400 are both required: 32 pages is ``div16``, 33 is ``other``.

``Job.prefill`` (``generator/job.py``) forwards ``ids[:-1]`` only. The last
prompt token is a decode ``model.forward``. Chunk end is aligned down to
``PAGE_SIZE`` (256), then cut at the last page boundary when the model has a
recurrent cache (``qwen3_5.py`` sets ``recurrent_states``). Lengths:

- 1500 tokens, 6 pages, no split: ``[0, 1280)`` then ``[1280, 1499)``,
  then decode position 1499
- 100 tokens, 1 page, no split: ``[0, 99)``, then decode position 99
- 8190 tokens, 32 pages (``div16``). Staging chunks ``[0, 2048)``,
  ``[2048, 4096)``, ``[4096, 6144)`` are q_len 2048, ``num_splits`` 2
  (case b: staging split, pages ``div16``). ``[6144, 7936)`` is q_len 1792,
  ``num_splits`` 3, same cubin class. The tail ``[7936, 8189)`` is qc
  q_len 253, ``num_splits`` 5 (case c on a ``div16`` table). Decode
  position 8189.
- 8400 tokens, 33 pages (``other``). Staging chunks ``[0, 2048)``,
  ``[2048, 4096)``, ``[4096, 6144)``, ``[6144, 8192)`` are q_len 2048,
  ``num_splits`` 2 (case a: staging split, pages ``other``). The tail
  ``[8192, 8399)`` is qc q_len 207, ``num_splits`` 5 (case c on an
  ``other`` table). Decode position 8399.

``prefill_ls`` stops at the last KV module (layer 63, full attention) and sets
``params["prefill"]``. ``TransformerBlock.forward`` still runs that mixer, so
the cache updates, then returns the incoming residual unchanged and skips the
MLP. Final norm and ``lm_head`` do not run during prefill. Logits come from
the decode of the last prompt token.

What is saved
-------------
Under ``testdata/prefill_ref/`` (kept under 4 MB):

- ``meta.json``: token ids, the chunk boundaries the fork actually ran, per
  block blake2b-256 of the fp32 residual ``[1, n, 5120]``, per attention layer
  the digest of used cache rows (logical block-table order; qk/qv int32,
  sk/sv fp16), per GDN layer the slot's conv (bf16) and recurrent (fp32)
  state, and the logits digest plus argmax.
- ``logits_<len>.bin`` for 100, 1500, 8190, and 8400: raw little-endian
  fp16 vocab row from that decode, cloned before the sampler.

Two measurement jobs per length must match. Two earlier jobs of the same ids
are discarded: the first decode is the eager BC warmup, the second captures
the ``(bsz=1, q_len=1)`` graph. Measurements are replays. Between jobs the
page table is reset and the KV / GDN bytes are zeroed, so the prefix cache
cannot skip the second prefill and a dirty page tail cannot change attention.

Run with the serve's environment (``scripts/serve-host.sh`` exports)::

    PYTHONPATH=/home/sam/qwen38-exl3-dflash2/exllamav3 \\
    TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \\
    LD_LIBRARY_PATH=<torch nvidia libs>:$LD_LIBRARY_PATH \\
    EXL3_INT8_GEMV=2 EXL3_INT8_GEMV_MAX_K=5 EXL3_QC_STAGING=1 \\
    EXL3_HGEMM_F16ACC=auto \\
    /home/sam/qwen38-exl3-dflash2/env/bin/python \\
        scripts/dump_27b_prefill_reference.py [OUT_DIR]
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Read by the extension when the first GEMM is built. Set the serve defaults
# before import so a bare invocation still matches serve-host.sh.
os.environ.setdefault("EXL3_INT8_GEMV", "2")
os.environ.setdefault("EXL3_INT8_GEMV_MAX_K", "5")
os.environ.setdefault("EXL3_QC_STAGING", "1")
os.environ.setdefault("EXL3_HGEMM_F16ACC", "auto")

import torch

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler.presets import ArgmaxSampler
from exllamav3.modules.transformer import TransformerBlock

ROOT = Path(__file__).resolve().parent.parent
TARGET = os.environ.get(
    "BERTH_27B_DIR", "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3"
)
SENTENCE = "The quick brown fox jumps over the lazy dog. "
# 1500 then 100 first: 1500's decode capture is what both short lengths
# replay. 8190 (32 pages, div16) and 8400 (33 pages, other) follow.
MEASURE_ORDER = (1500, 100, 8190, 8400)
LENGTHS = MEASURE_ORDER
PAGE_SIZE = 256
MAX_CHUNK = 2048
# 33 pages for the 8400 job, plus the 48-wide padded decode table.
CACHE_TOKENS = 12288
HIDDEN = 5120
VOCAB = 248320
LAYERS = 64
LAST_LAYER = LAYERS - 1
MEM_LIMIT_MIB = 4000
MEM_WAIT_S = 60
MEM_WAIT_BUDGET_S = 20 * 60
# Eager decode, then the capture launch, then the two replays we keep.
WARMUPS = 2
MEASURES = 2
ENV_FLAGS = (
    "EXL3_INT8_GEMV",
    "EXL3_INT8_GEMV_MAX_K",
    "EXL3_QC_STAGING",
    "EXL3_HGEMM_F16ACC",
)


def serve_listening():
    # Same probe the task requires: `ss -ltn | grep -q ':8889 '`.
    return subprocess.run(
        "ss -ltn | grep -q ':8889 '",
        shell=True,
    ).returncode == 0


def gpu_mem_mib():
    """Max ``memory.used`` across GPUs, in MiB. nvidia-smi prints ``N MiB``."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
        text=True,
    )
    vals = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        vals.append(int(line.split()[0]))
    if not vals:
        raise RuntimeError("nvidia-smi returned no memory.used rows")
    return max(vals)


def wait_for_gpu():
    """Port 8889 aborts immediately. A busy GPU is waited out, up to 20 min."""
    if serve_listening():
        raise SystemExit("port 8889 is listening; refusing to use the GPU")
    deadline = time.monotonic() + MEM_WAIT_BUDGET_S
    while True:
        if serve_listening():
            raise SystemExit("port 8889 is listening; refusing to use the GPU")
        used = gpu_mem_mib()
        if used < MEM_LIMIT_MIB:
            print(f"GPU memory {used} MiB", file=sys.stderr, flush=True)
            return used
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"GPU memory stayed at {used} MiB (>= {MEM_LIMIT_MIB}) for "
                f"{MEM_WAIT_BUDGET_S // 60} min; refusing to load the 27B"
            )
        print(
            f"GPU memory {used} MiB, waiting {MEM_WAIT_S}s",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(MEM_WAIT_S)


def blake2b_256(raw):
    return hashlib.blake2b(raw, digest_size=32).hexdigest()


def dtype_name(dt):
    if dt == torch.float32:
        return "float32"
    if dt == torch.float16:
        return "float16"
    if dt == torch.bfloat16:
        return "bfloat16"
    if dt == torch.int32:
        return "int32"
    raise RuntimeError(f"unexpected dtype {dt}")


def le_bytes(tensor):
    """Little-endian payload. bf16 and fp16 are their raw 16-bit patterns."""
    if sys.byteorder != "little":
        raise RuntimeError("digest bytes are defined as little-endian")
    t = tensor.detach().to("cpu").contiguous()
    name = dtype_name(t.dtype)
    if t.dtype == torch.bfloat16 or t.dtype == torch.float16:
        t = t.view(torch.int16)
    raw = t.numpy().tobytes()
    return name, list(t.shape), raw


def chunk_bounds(prompt_len, max_chunk=MAX_CHUNK, page=PAGE_SIZE):
    """``Job.prefill`` chunk ends for a cold recurrent job (``job.py``).

    Prefill covers ``[0, prompt_len)`` of the prompt excluding the last id.
    No prefix-cache skip, no multimodal cut. ``end`` is aligned down to
    ``page`` after adding ``max_chunk``, then clipped to the prefill limit,
    then cut at the last page boundary when that boundary sits inside the
    chunk.
    """
    if prompt_len < 2:
        return []
    limit = prompt_len - 1
    pos = 0
    out = []
    while pos < limit:
        end = ((pos + max_chunk) // page) * page
        end = min(end, limit)
        last_page_b = (limit // page) * page
        if pos < last_page_b <= end:
            end = last_page_b
        if end <= pos:
            raise RuntimeError(
                f"chunk did not advance at {pos} (limit {limit}, max_chunk {max_chunk})"
            )
        out.append((pos, end))
        pos = end
    return out


def prompt_ids(tokenizer):
    # Start from the same 400 repeats the 100/1500 dump used, then extend.
    # The token prefix stays the repeated sentence, so the short prompts are
    # the same ids as before.
    repeats = 400
    text = SENTENCE * repeats
    ids = tokenizer.encode(text)
    while ids.shape[-1] < max(LENGTHS):
        repeats *= 2
        text = SENTENCE * repeats
        ids = tokenizer.encode(text)
    full = ids[:, : max(LENGTHS)].contiguous()
    return {n: full[:, :n].contiguous() for n in LENGTHS}


class Capture:
    """Hooks around one job. Residuals are hashed on the CPU and dropped."""

    def __init__(self, model):
        self.model = model
        self.prompt_len = None
        self.in_prefill = False
        self.chunks = []
        self.current = None
        self.after = None
        self.logits = None
        self.forwards = 0
        self._install()

    def _install(self):
        model = self.model
        cap = self
        orig_prefill = model.prefill
        orig_forward = model.forward

        def wrapped_prefill(input_ids, params=None):
            if params is None:
                raise RuntimeError("prefill was called without params")
            start = int(params["cache_seqlens"].detach().to("cpu").view(-1)[0].item())
            n = int(input_ids.shape[-1])
            if input_ids.shape[0] != 1:
                raise RuntimeError(f"prefill batch {tuple(input_ids.shape)} is not 1")
            cap.current = {"start": start, "end": start + n, "n": n, "blocks": []}
            cap.in_prefill = True
            try:
                y = orig_prefill(input_ids, params)
                if y is not None:
                    raise RuntimeError(f"prefill_ls returned {type(y).__name__}, expected None")
                blocks = cap.current["blocks"]
                if [b["layer"] for b in blocks] != list(range(LAYERS)):
                    raise RuntimeError(
                        f"chunk [{start}, {start + n}) recorded layers "
                        f"{[b['layer'] for b in blocks]}"
                    )
                if blocks[LAST_LAYER]["digest"] != blocks[LAST_LAYER - 1]["digest"]:
                    raise RuntimeError(
                        "layer 63 residual does not equal layer 62 "
                        "(last KV module must return its input)"
                    )
                blocks[LAST_LAYER]["early_return"] = True
                cap.chunks.append(cap.current)
                print(
                    f"  prefill [{start}, {start + n}) layers {len(blocks)}",
                    file=sys.stderr,
                    flush=True,
                )
                if start + n == cap.prompt_len - 1:
                    cap.after = snapshot_state(model, params, cap.prompt_len - 1)
                elif start + n > cap.prompt_len - 1:
                    raise RuntimeError(
                        f"prefill chunk [{start}, {start + n}) passed the prompt tail "
                        f"{cap.prompt_len - 1}"
                    )
            finally:
                cap.in_prefill = False
                cap.current = None
            return y

        def wrapped_forward(input_ids, params=None):
            y = orig_forward(input_ids, params)
            cap.forwards += 1
            if not isinstance(y, torch.Tensor):
                raise RuntimeError(f"model.forward returned {type(y).__name__}")
            name, shape, raw = le_bytes(y)
            if name != "float16" or shape[-1] != VOCAB or y.numel() != VOCAB:
                raise RuntimeError(f"logits are {name} {shape}, expected fp16 [*, {VOCAB}]")
            cap.logits = {
                "dtype": name,
                "shape": shape,
                "digest": blake2b_256(raw),
                "argmax": int(y.detach().view(-1).argmax().item()),
                "bytes": raw,
            }
            return y

        model.prefill = wrapped_prefill
        model.forward = wrapped_forward

        blocks = [m for m in model.modules if isinstance(m, TransformerBlock)]
        if [b.layer_idx for b in blocks] != list(range(LAYERS)):
            raise RuntimeError(f"transformer blocks are {[b.layer_idx for b in blocks]}")
        for block in blocks:
            orig = block.forward

            def wrapped_block(x, params, out_dtype=None, _orig=orig, _layer=block.layer_idx):
                y = _orig(x, params, out_dtype=out_dtype)
                if cap.in_prefill:
                    name, shape, raw = le_bytes(y)
                    n = cap.current["n"]
                    if name != "float32" or shape != [1, n, HIDDEN]:
                        raise RuntimeError(
                            f"layer {_layer} residual is {name} {shape}, "
                            f"expected float32 [1, {n}, {HIDDEN}]"
                        )
                    cap.current["blocks"].append({
                        "layer": _layer,
                        "dtype": name,
                        "shape": shape,
                        "digest": blake2b_256(raw),
                        "early_return": False,
                    })
                return y

            block.forward = wrapped_block

    def begin_job(self, prompt_len):
        self.prompt_len = prompt_len
        self.chunks = []
        self.current = None
        self.after = None
        self.logits = None
        self.forwards = 0

    def record(self, sampled):
        if self.after is None:
            raise RuntimeError("prefill finished without a snapshot of the last chunk")
        if self.forwards != 1 or self.logits is None:
            raise RuntimeError(
                f"expected one decode forward, saw {self.forwards} "
                f"(logits captured: {self.logits is not None})"
            )
        if len(sampled) != 1:
            raise RuntimeError(f"expected one sampled token, got {sampled}")
        if sampled[0] != self.logits["argmax"]:
            raise RuntimeError(
                f"sampler emitted {sampled[0]}, logits argmax is {self.logits['argmax']}"
            )
        return {
            "chunks": self.chunks,
            "after_prefill": self.after,
            "logits": {
                "dtype": self.logits["dtype"],
                "shape": self.logits["shape"],
                "digest": self.logits["digest"],
                "argmax": self.logits["argmax"],
            },
            "logits_bytes": self.logits["bytes"],
            "sampled_token": sampled[0],
        }


def snapshot_state(model, params, tokens_written):
    """Cache and GDN state after the last prefill chunk, before the decode."""
    torch.cuda.synchronize()
    cache = params["cache"]
    bt = params["block_table"].detach().to("cpu").contiguous()
    if bt.dim() != 2 or bt.shape[0] != 1:
        raise RuntimeError(f"block_table shape {tuple(bt.shape)}")
    pages = (tokens_written + PAGE_SIZE - 1) // PAGE_SIZE
    if bt.shape[1] < pages:
        raise RuntimeError(f"block_table width {bt.shape[1]} < {pages} used pages")
    table = [int(bt[0, p].item()) for p in range(bt.shape[1])]
    rs = params.get("recurrent_states")
    if not rs or len(rs) != 1:
        raise RuntimeError("prefill params have no single recurrent state")
    slot = int(rs[0].slot)

    attn = []
    for key in sorted(cache.layers.keys()):
        layer_idx, instance = key
        layer = cache.layers[key]
        entry = {"layer": int(layer_idx), "instance": int(instance)}
        for name, tensor in (("qk", layer.qk), ("qv", layer.qv), ("sk", layer.sk), ("sv", layer.sv)):
            if tensor.shape[0] <= max(table[:pages]):
                raise RuntimeError(f"layer {layer_idx} {name} has no page {max(table[:pages])}")
            used = gather_used(tensor, table, tokens_written)
            dtype, shape, raw = le_bytes(used)
            expect = "float16" if name in ("sk", "sv") else "int32"
            if dtype != expect:
                raise RuntimeError(f"layer {layer_idx} {name} is {dtype}, expected {expect}")
            entry[name] = {"dtype": dtype, "shape": shape, "digest": blake2b_256(raw)}
        attn.append(entry)
    if len(attn) != 16:
        raise RuntimeError(f"expected 16 attention caches, found {len(attn)}")

    gdn = []
    for key in sorted(cache.recurrent_layers.keys()):
        layer_idx, instance = key
        conv, recurrent = cache.recurrent_layers[key].get_state_tensors()
        if slot >= conv.shape[0] or slot >= recurrent.shape[0]:
            raise RuntimeError(f"GDN slot {slot} is outside layer {layer_idx}")
        conv_dtype, conv_shape, conv_raw = le_bytes(conv[slot])
        rec_dtype, rec_shape, rec_raw = le_bytes(recurrent[slot])
        if conv_dtype != "bfloat16" or rec_dtype != "float32":
            raise RuntimeError(
                f"layer {layer_idx} state is conv {conv_dtype} recurrent {rec_dtype}"
            )
        gdn.append({
            "layer": int(layer_idx),
            "instance": int(instance),
            "slot": slot,
            "conv": {"dtype": conv_dtype, "shape": conv_shape, "digest": blake2b_256(conv_raw)},
            "recurrent": {
                "dtype": rec_dtype,
                "shape": rec_shape,
                "digest": blake2b_256(rec_raw),
            },
        })
    if len(gdn) != 48:
        raise RuntimeError(f"expected 48 GDN states, found {len(gdn)}")

    return {
        "tokens_written": tokens_written,
        "block_table": table,
        "attn": attn,
        "gdn": gdn,
    }


def gather_used(tensor, block_table, tokens_written):
    parts = []
    for p in range((tokens_written + PAGE_SIZE - 1) // PAGE_SIZE):
        phys = block_table[p]
        used = min(PAGE_SIZE, tokens_written - p * PAGE_SIZE)
        parts.append(tensor[phys, :used])
    return torch.cat(parts, dim=0).contiguous()


@torch.inference_mode()
def zero_cache(cache):
    # Cache tensors are allocated under inference mode. zero_ outside it is rejected.
    for layer in cache.layers.values():
        layer.qk.zero_()
        layer.qv.zero_()
        layer.sk.zero_()
        layer.sv.zero_()
    for layer in cache.recurrent_layers.values():
        layer.conv_state.zero_()
        layer.recurrent_state.zero_()
    torch.cuda.synchronize()


def reset_generator(gen):
    """Drop prefix-cache hits. ``reset_page_table`` does not scrub KV bytes."""
    if gen.num_remaining_jobs():
        raise RuntimeError("resetting the page table while a job is still queued")
    gen.pagetable.reset_page_table()
    gen.recurrent_cache.clear()
    gen.recurrent_cache.current_size = 0
    zero_cache(gen.cache)


def sampled_tokens(results):
    found = []
    for result in results:
        src = result if "token_ids" in result else (result.get("held") or {})
        if "token_ids" not in src:
            continue
        ids = src["token_ids"].detach().to("cpu").view(-1)
        found.extend(int(v) for v in ids.tolist())
    return found


def run_job(gen, cap, ids):
    cap.begin_job(int(ids.shape[-1]))
    gen.enqueue(Job(
        input_ids=ids.clone(),
        max_new_tokens=1,
        sampler=ArgmaxSampler(),
    ))
    sampled = []
    while gen.num_remaining_jobs():
        sampled.extend(sampled_tokens(gen.iterate()))
    torch.cuda.synchronize()
    return cap.record(sampled)


def public_record(ids, record):
    logits = dict(record["logits"])
    return {
        "token_ids": [int(v) for v in ids[0].tolist()],
        "chunks": record["chunks"],
        "after_prefill": record["after_prefill"],
        "logits": logits,
        "sampled_token": record["sampled_token"],
    }


def first_mismatch(a, b, path=""):
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return f"{path}: keys {sorted(a.keys() ^ b.keys())}"
        for key in a:
            found = first_mismatch(a[key], b[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            found = first_mismatch(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    if a != b:
        return f"{path}: {a!r} vs {b!r}"
    return None


def measure(gen, cap, ids, length):
    expect = chunk_bounds(length)
    saved = None
    saved_bytes = None
    warmup_pub = None
    warmup_bytes = None
    for index in range(WARMUPS + MEASURES):
        kind = "warmup" if index < WARMUPS else "measure"
        ordinal = index if index < WARMUPS else index - WARMUPS
        print(f"{length}: {kind} {ordinal}", file=sys.stderr, flush=True)
        record = run_job(gen, cap, ids)
        got = [(c["start"], c["end"]) for c in record["chunks"]]
        if got != expect:
            raise RuntimeError(f"{length}: chunks {got} != {expect}")
        pub = public_record(ids, record)
        if index < WARMUPS:
            warmup_pub = pub
            warmup_bytes = record["logits_bytes"]
        elif saved is None:
            saved = pub
            saved_bytes = record["logits_bytes"]
            mismatch = first_mismatch(warmup_pub, pub)
            if mismatch or warmup_bytes != saved_bytes:
                where = mismatch or "logits bytes"
                print(f"{length}: last warmup differs from replay at {where}", file=sys.stderr, flush=True)
            else:
                print(f"{length}: last warmup matches the replay", file=sys.stderr, flush=True)
        else:
            mismatch = first_mismatch(saved, pub)
            if mismatch or saved_bytes != record["logits_bytes"]:
                raise RuntimeError(f"{length}: measurement runs differ at {mismatch}")
        # The next job must not reuse this prompt's pages or GDN slot contents.
        reset_generator(gen)
    # Compare the last warmup's public record only when we still have it.
    # `saved` is the first measurement. Re-running is unnecessary; the two
    # measurements already matched.
    print(
        f"{length}: argmax {saved['logits']['argmax']} "
        f"chunks {[(c['start'], c['end']) for c in saved['chunks']]}",
        file=sys.stderr,
        flush=True,
    )
    return saved, saved_bytes


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "testdata" / "prefill_ref"
    wait_for_gpu()
    cfg = Config.from_directory(TARGET)
    model = Model.from_config(cfg)
    cache = Cache(
        model,
        max_num_tokens=CACHE_TOKENS,
        layer_type=CacheLayer_quant,
        k_bits=3,
        v_bits=3,
        max_batch_size=1,
    )
    model.load(progressbar=False)
    tokenizer = Tokenizer.from_config(cfg)
    gen = Generator(model, cache, tokenizer)
    if gen.max_chunk_size != MAX_CHUNK:
        raise RuntimeError(f"Generator max_chunk_size is {gen.max_chunk_size}, expected {MAX_CHUNK}")
    if gen.recurrent_cache is None:
        raise RuntimeError("target has no recurrent cache; chunk cuts would differ")
    cap = Capture(model)
    ids = prompt_ids(tokenizer)
    prefix_order = (100, 1500, 8190, 8400)
    for shorter, longer in zip(prefix_order, prefix_order[1:]):
        if ids[shorter][0].tolist() != ids[longer][0, :shorter].tolist():
            raise RuntimeError(f"{shorter}-token prompt is not a prefix of the {longer}-token prompt")

    prompts = {}
    blobs = {}
    for length in MEASURE_ORDER:
        record, raw = measure(gen, cap, ids[length], length)
        record["logits"]["file"] = f"logits_{length}.bin"
        prompts[str(length)] = record
        blobs[length] = raw

    meta = {
        "cache_tokens": CACHE_TOKENS,
        "serve_default_cache_tokens": 270336,
        "page_size": PAGE_SIZE,
        "max_chunk_size": MAX_CHUNK,
        "hidden": HIDDEN,
        "vocab": VOCAB,
        "k_bits": 3,
        "v_bits": 3,
        "max_batch_size": 1,
        "max_history": 0,
        "layers": LAYERS,
        "env": {name: os.environ.get(name) for name in ENV_FLAGS},
        "checkpoint": TARGET,
        "prompt_sentence": SENTENCE,
        "notes": {
            "prefill_range": (
                "The fork prefills ids[:-1], chunked by max_chunk_size and aligned "
                "down to PAGE_SIZE, then cuts a recurrent model at the last page "
                "boundary (generator/job.py). The last prompt token is a decode forward."
            ),
            "last_block": (
                "Layer 63 is the last KV module. prefill_ls sets params['prefill'] only "
                "there (model/model_ls.py). TransformerBlock.forward runs the mixer, "
                "then returns the incoming residual and skips the MLP "
                "(modules/transformer.py). That digest equals layer 62. Final norm and "
                "lm_head do not run during prefill."
            ),
            "cache_width": (
                "cache_tokens is 12288 so the 8400-token job's 33 pages fit, and "
                "so does a decode table padded out to 48. The serve default is "
                "270336. Prefill num_splits follows the job allocation "
                "ceil((prompt_len + max_new_tokens + 1) / 256), which is constant "
                "for every chunk. 8190 is 32 pages (div16) and 8400 is 33 (other); "
                "one prompt cannot hit both page classes. Decode num_splits follows "
                "the current sequence length padded up to a multiple of 16."
            ),
            "residual_digest": (
                "blake2b-256 of the little-endian fp32 bytes of the block return, "
                "shape [1, n, 5120], for that prefill chunk."
            ),
            "attn_digest": (
                "blake2b-256 of used rows in logical block-table order. Page p uses "
                "physical index block_table[p], rows min(256, tokens_written - 256*p). "
                "tokens_written is prompt_len - 1. qk and qv are int32, sk and sv fp16. "
                "The tail of a partial page is not included."
            ),
            "gdn_digest": (
                "blake2b-256 of the job slot. conv_state[slot] is bf16 (raw 16-bit "
                "pattern); recurrent_state[slot] is fp32. max_batch_size is 1 and "
                "max_history is 0, so the slot is the live state."
            ),
            "logits": (
                "fp16 row returned by model.forward on the last prompt token, cloned "
                "before the sampler. The .bin file is that row, little-endian."
            ),
        },
        "prompts": prompts,
    }

    out.mkdir(parents=True, exist_ok=True)
    for length, raw in blobs.items():
        (out / f"logits_{length}.bin").write_bytes(raw)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    if total >= 4 * 1024 * 1024:
        raise RuntimeError(f"{out} is {total} bytes, over the 4 MB cap")
    print(f"wrote {out} ({total} bytes)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

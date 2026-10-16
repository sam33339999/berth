#!/usr/bin/env python3
"""Non-MTP generate reference for the 27B target. Needs the GPU.

Do not run while the :8889 serve is up: this loads the 27B (~16 GB). The
script waits while ``nvidia-smi`` reports 4000 MiB or more, up to 20 minutes,
then refuses. It refuses immediately if port 8889 is listening. It does not
import torch until that check has passed. Do not edit the fork.

Fork target only, no draft
--------------------------
``Cache(model, max_num_tokens=12288, layer_type=CacheLayer_quant, k_bits=3,
v_bits=3, max_batch_size=1)`` then ``model.load()``, then
``Generator(model, cache, tokenizer)``. ``num_draft_tokens`` stays 0
(``generator.py`` 171). The cache is built before ``load()``, which
``Generator.iterate`` requires.

Prompts are prefixes of one tokenization of the sentence in
``dump_27b_prefill_reference.py`` (the same ids as that script's
``prompt_ids`` / the MTP dump). Lengths:

- 4090. The last prompt token is decoded at position 4089
  (``len(sequence_ids) == 4090``, 16 pages). Token 7 of the completion is
  decoded at position 4096 (``len == 4097``, 17 pages padded to 32). The
  block table therefore crosses the 16-page boundary during the 32 new
  tokens.
- 100. ``len`` stays at 132, one page padded to 16 the whole way. This
  length does not move the block table. It is here so DRY's default
  sequence breakers can change the sampled token.

Samplers, each a fresh object per job. ``max_new_tokens`` is 32,
``min_new_tokens`` is 32, ``stop_conditions=()``, ``token_healing=False``.

- ``greedy``: ``ArgmaxSampler``, ``Job(seed=0)``. Penalties off. This is
  ``SamplerConfig::greedy()``, not ``serve_defaults(0)`` (that stack still
  applies DRY 0.2).
- ``sample``: temperature 0.7, ``Job(seed=42)``, built the way
  ``scripts/serve_openai.py`` ``_sampler`` builds a request.
  ``SS_RepP(1.0, 512, 0)``, ``SS_PresFreqP(0.0, 0.0, 512, 0)``,
  ``SS_DRY(0.2, 1.75, 6, 4096)`` with no breaker set and no tokenizer
  (``default_breakers`` stays true; ``receive_logits`` passes the generator
  tokenizer and ``custom.py`` 1008–1010 resolves the default mask), then
  ``SS_Temperature(0.7)`` and ``SS_Sample()``.
- ``greedy_dry``: the serve's ``temperature = 0`` path. The same penalty
  steps, then ``SS_Argmax()`` instead of temperature plus sample.
  ``Job(seed=0)``, matching ``serve.rs`` ``base_sampler``. DRY stays on.
  Formal greedy still penalizes.

``EXL3_FUSED_SAMPLER`` is left at its default (do not set it). ``alt()``
drops the repetition and presence/frequency steps. The tail collapses to
``SS_Fused``.

What one step records
---------------------
Each decode ``model.forward`` (the last prompt token, then each accepted
token) contributes one fp16 ``[vocab]`` logit row, cloned after ``forward``
returns and before the sampler reads the original tensor. The clone is not
inside the captured graph. The token is the id appended to ``sequence_ids``.
Both are blake2b-256 of the little-endian fp16 payload. The first row of
each case is also written raw, with no header.

``block_pages`` is ``params["block_table"].shape[-1]``, the padded width
``iterate_gen`` built from ``get_max_seq_len()``. The script requires it to
equal ``ceil((position + 1) / 256)`` pages rounded up to a multiple of 16,
where ``position`` is ``cache_seqlens`` (the token about to enter the cache).

Warmup
------
Each new CUDA-graph slot does eager, then capture, then replay. For each
``(prompt, sampler)`` the script discards two identical full 32-token jobs,
then records the third. Those two jobs cover every prefill chunk length and
both decode widths (16 and 32 pages) that the 4090 case hits. Hooks are not
installed on the discarded jobs. Between jobs the page table is reset and
KV / GDN bytes are zeroed, and the recurrent checkpoint cache is cleared, so
the prefix cache cannot skip the next prefill.

Output
------
``testdata/generate_ref/meta.json`` and one
``logits_<prompt>_<sampler>_step0.bin`` per case. Regeneration is described
in ``testdata/generate_ref/README.md``.
"""

import json
import os
import struct
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

ROOT = Path(__file__).resolve().parent.parent
TARGET = os.environ.get(
    "BERTH_27B_DIR", "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3"
)
FORK = Path(os.environ.get("BERTH_EXL3_SRC", "/home/sam/qwen38-exl3-dflash2/exllamav3"))
CACHE_TOKENS = 12288
PAGE_SIZE = 256
HIDDEN = 5120
VOCAB = 248320
MAX_NEW = 32
WARMUPS = 2
MEM_LIMIT_MIB = 4000
MEM_WAIT_S = 60
MEM_WAIT_BUDGET_S = 20 * 60
PROMPT_ORDER = (4090, 100)
SAMPLER_ORDER = ("greedy", "sample", "greedy_dry")
PENALTY_RANGE = 512
SAMPLE_TEMPERATURE = 0.7
SAMPLE_SEED = 42
GREEDY_SEED = 0
INV_TEMP_07_F32 = "3fb6db6e"
ENV_FLAGS = (
    "EXL3_INT8_GEMV",
    "EXL3_INT8_GEMV_MAX_K",
    "EXL3_QC_STAGING",
    "EXL3_HGEMM_F16ACC",
)


def git(*args):
    return subprocess.check_output(["git", "-C", str(FORK), *args], text=True).strip()


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


def f32_bits(value):
    """Lowercase hex of the f32 bit pattern, matching a Rust ``to_bits()``."""
    return f"{struct.unpack('<I', struct.pack('<f', float(value)))[0]:08x}"


def padded_pages(seq_len):
    """``iterate_gen`` block-table width for a sequence of ``seq_len`` tokens."""
    pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
    if pages < 1:
        pages = 1
    return (pages + 15) // 16 * 16


def constructed_penalty_steps(tail):
    return [
        {
            "type": "SS_RepP",
            "rep_p": 1.0,
            "sustain_range": PENALTY_RANGE,
            "decay_range": 0,
            "dropped": True,
        },
        {
            "type": "SS_PresFreqP",
            "pres_p": 0.0,
            "freq_p": 0.0,
            "sustain_range": PENALTY_RANGE,
            "decay_range": 0,
            "dropped": True,
        },
        {
            "type": "SS_DRY",
            "dry_multiplier": 0.2,
            "dry_base": 1.75,
            "dry_allowed_length": 6,
            "dry_range": 4096,
            "sequence_breakers": None,
            "tokenizer": None,
            "dropped": False,
        },
        *tail,
    ]


def make_sampler(kind):
    from exllamav3.generator.sampler.custom import (
        CustomSampler,
        SS_Argmax,
        SS_DRY,
        SS_Fused,
        SS_PresFreqP,
        SS_RepP,
        SS_Sample,
        SS_Temperature,
    )
    from exllamav3.generator.sampler.presets import ArgmaxSampler

    if kind == "greedy":
        sampler = ArgmaxSampler()
        seed = GREEDY_SEED
    elif kind == "sample":
        sampler = CustomSampler([
            SS_RepP(1.0, PENALTY_RANGE, 0),
            SS_PresFreqP(0.0, 0.0, PENALTY_RANGE, 0),
            SS_DRY(0.2, 1.75, 6, 4096),
            SS_Temperature(SAMPLE_TEMPERATURE),
            SS_Sample(),
        ])
        seed = SAMPLE_SEED
    elif kind == "greedy_dry":
        # serve_openai._sampler when temperature is 0: the penalty stack, then
        # SS_Argmax. No breaker argument, no tokenizer argument.
        sampler = CustomSampler([
            SS_RepP(1.0, PENALTY_RANGE, 0),
            SS_PresFreqP(0.0, 0.0, PENALTY_RANGE, 0),
            SS_DRY(0.2, 1.75, 6, 4096),
            SS_Argmax(),
        ])
        seed = GREEDY_SEED
    else:
        raise RuntimeError(f"unknown sampler {kind}")
    if kind != "greedy":
        dry = next(step for step in sampler.steps if isinstance(step, SS_DRY))
        if not dry.default_breakers:
            raise RuntimeError(f"{kind} SS_DRY did not keep default_breakers")
    collapsed = []
    for step in sampler.steps:
        entry = {"type": type(step).__name__}
        if isinstance(step, SS_Fused):
            entry["mode"] = int(step.mode)
            entry["inv_temp_f32_bits"] = f32_bits(step.inv_temp)
        if isinstance(step, SS_DRY):
            entry["dry_multiplier"] = step.dry_multiplier
            entry["dry_base"] = step.dry_base
            entry["dry_allowed_length"] = step.dry_allowed_length
            entry["dry_range"] = step.dry_range
            entry["dry_max_exponent"] = step.max_exponent
            entry["dry_multiplier_f32_bits"] = f32_bits(step.dry_multiplier)
            entry["dry_base_f32_bits"] = f32_bits(step.dry_base)
        collapsed.append(entry)
    if kind == "greedy":
        expect = [{
            "type": "SS_Fused",
            "mode": SS_Fused.MODE_GREEDY,
            "inv_temp_f32_bits": f32_bits(1.0),
        }]
        if collapsed != expect:
            raise RuntimeError(
                f"greedy stack collapsed to {collapsed}. "
                "Leave EXL3_FUSED_SAMPLER at its default."
            )
    elif kind == "sample":
        if (
            len(collapsed) != 2
            or collapsed[0]["type"] != "SS_DRY"
            or collapsed[1]["type"] != "SS_Fused"
            or collapsed[1]["mode"] != SS_Fused.MODE_SAMPLE
            or collapsed[1]["inv_temp_f32_bits"] != INV_TEMP_07_F32
        ):
            raise RuntimeError(
                f"sample stack collapsed to {collapsed}, expected DRY plus "
                f"SS_Fused mode {SS_Fused.MODE_SAMPLE} inv_temp {INV_TEMP_07_F32}. "
                "Leave EXL3_FUSED_SAMPLER at its default."
            )
    else:
        if (
            len(collapsed) != 2
            or collapsed[0]["type"] != "SS_DRY"
            or collapsed[1]["type"] != "SS_Fused"
            or collapsed[1]["mode"] != SS_Fused.MODE_GREEDY
            or collapsed[1]["inv_temp_f32_bits"] != f32_bits(1.0)
        ):
            raise RuntimeError(
                f"greedy_dry stack collapsed to {collapsed}, expected DRY plus "
                f"SS_Fused mode {SS_Fused.MODE_GREEDY}. "
                "Leave EXL3_FUSED_SAMPLER at its default."
            )
    return sampler, seed, collapsed


class LogitHook:
    """Clone each decode ``model.forward`` result after it returns.

    The clone is a later kernel on the same stream. It is not part of a
    graph captured inside ``forward``, and the tensor the sampler reads is
    the original return. Installed only on the recorded job.
    """

    def __init__(self):
        self.saved = None
        self.pending = []

    def install(self, model):
        if self.saved is not None:
            raise RuntimeError("logit hook is already installed")
        orig = model.forward
        hook = self

        def wrapped(input_ids, params=None, *args, **kwargs):
            y = orig(input_ids, params, *args, **kwargs)
            if y is None:
                raise RuntimeError("model.forward returned None")
            if tuple(input_ids.shape) != (1, 1):
                raise RuntimeError(
                    f"decode forward ids {tuple(input_ids.shape)}, expected (1, 1)"
                )
            block_table = None if params is None else params.get("block_table")
            if block_table is None:
                raise RuntimeError("decode forward has no block_table")
            seqlens = None if params is None else params.get("cache_seqlens")
            if seqlens is None:
                raise RuntimeError("decode forward has no cache_seqlens")
            flat = seqlens.detach().reshape(-1)
            if flat.numel() != 1:
                raise RuntimeError(f"cache_seqlens has {flat.numel()} entries")
            hook.pending.append((
                y.detach().clone(),
                int(block_table.shape[-1]),
                int(flat[0].item()),
            ))
            return y

        self.pending = []
        model.forward = wrapped
        self.saved = (model, orig)

    def uninstall(self):
        if self.saved is None:
            return
        model, orig = self.saved
        model.forward = orig
        self.saved = None

    def take(self):
        import torch

        torch.cuda.synchronize()
        out = []
        for tensor, width, position in self.pending:
            cpu = tensor.detach().to("cpu").contiguous()
            del tensor
            out.append((cpu, width, position))
        self.pending = []
        return out


def run_job(gen, ids, sampler, seed, hook):
    from exllamav3 import Job

    job = Job(
        input_ids=ids.clone(),
        max_new_tokens=MAX_NEW,
        min_new_tokens=MAX_NEW,
        sampler=sampler,
        seed=seed,
        stop_conditions=(),
        token_healing=False,
    )
    if hook is not None:
        hook.install(gen.model)
    eos_reason = None
    try:
        gen.enqueue(job)
        while gen.num_remaining_jobs():
            for result in gen.iterate():
                if result.get("eos"):
                    eos_reason = result.get("eos_reason")
    finally:
        if hook is not None:
            hook.uninstall()
    return job, eos_reason


def blake_of(raw):
    import dump_27b_prefill_reference as ref

    return ref.blake2b_256(raw)


def logit_payload(tensor):
    import dump_27b_prefill_reference as ref

    name, shape, raw = ref.le_bytes(tensor)
    if name != "float16" or shape != [VOCAB]:
        raise RuntimeError(f"logits are {name} {shape}, expected float16 [{VOCAB}]")
    if len(raw) != VOCAB * 2:
        raise RuntimeError(f"logits payload is {len(raw)} bytes, expected {VOCAB * 2}")
    return raw


def sequence_ids(job):
    if len(job.sequences) != 1:
        raise RuntimeError(f"job has {len(job.sequences)} sequences")
    seq = job.sequences[0].sequence_ids.torch().detach().to("cpu").view(-1)
    return [int(v) for v in seq.tolist()]


def check_recorded(label, prompt_ids, rows, output_tokens, eos_reason, max_rq):
    prompt_len = len(prompt_ids)
    if eos_reason != "max_new_tokens":
        raise RuntimeError(f"{label} eos_reason {eos_reason}")
    if len(output_tokens) != MAX_NEW:
        raise RuntimeError(f"{label} emitted {len(output_tokens)} tokens, expected {MAX_NEW}")
    if len(rows) != MAX_NEW:
        raise RuntimeError(f"{label} captured {len(rows)} logit rows, expected {MAX_NEW}")
    if max_rq != MAX_NEW + 1:
        raise RuntimeError(
            f"{label} max_rq_tokens {max_rq}, expected {MAX_NEW + 1} with no draft"
        )
    steps = []
    for index, ((tensor, width, position), token) in enumerate(zip(rows, output_tokens)):
        if position != prompt_len - 1 + index:
            raise RuntimeError(
                f"{label} step {index} cache position {position}, "
                f"expected {prompt_len - 1 + index}"
            )
        seq_len = position + 1
        expect_width = padded_pages(seq_len)
        if width != expect_width:
            raise RuntimeError(
                f"{label} step {index} block_pages {width}, "
                f"expected {expect_width} for sequence length {seq_len}"
            )
        if tensor.shape != (1, 1, VOCAB):
            raise RuntimeError(f"{label} step {index} logits shape {tuple(tensor.shape)}")
        raw = logit_payload(tensor.reshape(-1))
        steps.append({
            "token": token,
            "position": position,
            "block_pages": width,
            "logits_blake2b": blake_of(raw),
            "raw": raw,
        })
    # The crossing this reference exists to catch: 4090's early steps stay on
    # 16 pages and a later step pads to 32. 100 never leaves 16.
    widths = [step["block_pages"] for step in steps]
    if prompt_len == 4090:
        if widths[0] != 16 or 32 not in widths:
            raise RuntimeError(f"{label} widths {widths} did not cross 16 -> 32")
    elif prompt_len == 100:
        if any(width != 16 for width in widths):
            raise RuntimeError(f"{label} widths {widths}, expected 16 throughout")
    return steps


def sampler_meta_entry(kind, seed, collapsed, breakers):
    entry = {
        "seed": seed,
        "collapsed_steps": collapsed,
    }
    if kind == "greedy":
        entry["temperature"] = 0.0
        entry["dry_multiplier"] = 0.0
        entry["constructed_steps"] = [{"type": "SS_Argmax", "dropped": False}]
        return entry
    entry["temperature"] = 0.0 if kind == "greedy_dry" else SAMPLE_TEMPERATURE
    entry["rep_penalty"] = 1.0
    entry["presence_penalty"] = 0.0
    entry["frequency_penalty"] = 0.0
    entry["penalty_range"] = PENALTY_RANGE
    entry["dry_multiplier"] = 0.2
    entry["dry_base"] = 1.75
    entry["dry_allowed_length"] = 6
    entry["dry_range"] = 4096
    entry["dry_breaker_token_ids"] = breakers
    if kind == "sample":
        tail = [
            {
                "type": "SS_Temperature",
                "temperature": SAMPLE_TEMPERATURE,
                "dropped": False,
                "collapsed_into_fused": True,
            },
            {"type": "SS_Sample", "dropped": False, "collapsed_into_fused": True},
        ]
    else:
        tail = [{"type": "SS_Argmax", "dropped": False, "collapsed_into_fused": True}]
    entry["constructed_steps"] = constructed_penalty_steps(tail)
    return entry


def main():
    if sys.byteorder != "little":
        raise SystemExit("digest bytes are defined as little-endian")
    if not FORK.is_dir():
        raise SystemExit(f"fork not found: {FORK}")
    ext_tree = git("rev-parse", "HEAD:exllamav3/exllamav3_ext")
    if not ext_tree.startswith("de35ffb"):
        raise SystemExit(f"ext tree {ext_tree} is not de35ffb")
    dirty = git("status", "--porcelain")
    if dirty:
        raise SystemExit(f"fork worktree is dirty; refusing to dump a reference:\n{dirty}")
    if os.environ.get("EXL3_BC_ATTN") == "0":
        raise SystemExit(
            "EXL3_BC_ATTN=0 turns graph capture off. Leave it unset so this dump "
            "matches the serve (BC stays on)."
        )
    if os.environ.get("EXL3_FUSED_SAMPLER") == "0":
        raise SystemExit(
            "EXL3_FUSED_SAMPLER=0 disables the fused tail. Leave it unset."
        )

    # Port check before torch. dump_27b_prefill_reference imports torch.
    wait_for_gpu()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dump_27b_prefill_reference as ref
    import torch
    from exllamav3 import Cache, Config, Generator, Model, Tokenizer
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.generator.sampler.custom import dry_default_sequence_breaker_tokens

    out = ROOT / "testdata" / "generate_ref"
    out.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        cfg = Config.from_directory(TARGET)
        if int(cfg.hidden_size) != HIDDEN or int(cfg.vocab_size) != VOCAB:
            raise SystemExit(
                f"config hidden {cfg.hidden_size} vocab {cfg.vocab_size}, "
                f"expected {HIDDEN} {VOCAB}"
            )
        model = Model.from_config(cfg)
        cache = Cache(
            model,
            max_num_tokens=CACHE_TOKENS,
            layer_type=CacheLayer_quant,
            k_bits=3,
            v_bits=3,
            max_batch_size=1,
        )
        print("loading target (no draft)", file=sys.stderr, flush=True)
        model.load(progressbar=False)
        if not cache.initialized:
            raise SystemExit("load() did not mark the cache initialized")
        tokenizer = Tokenizer.from_config(cfg)
        encoded = ref.prompt_ids(tokenizer)
        full = encoded[max(ref.LENGTHS)]
        if int(full.shape[-1]) < max(PROMPT_ORDER):
            raise SystemExit(
                f"prefill prompt is {int(full.shape[-1])} tokens, "
                f"shorter than {max(PROMPT_ORDER)}"
            )
        prompts = {}
        for length in PROMPT_ORDER:
            piece = full[:, :length].contiguous()
            if length in encoded and not torch.equal(piece, encoded[length]):
                raise SystemExit(f"prompt {length} is not prompt_ids({length})")
            if length == 100 and not torch.equal(piece, encoded[100]):
                raise SystemExit("prompt 100 is not the prefill prompt_ids prefix")
            prompts[length] = piece
        if not torch.equal(prompts[100], prompts[4090][:, :100]):
            raise SystemExit("prompt 100 is not a prefix of prompt 4090")
        breakers = sorted(int(v) for v in dry_default_sequence_breaker_tokens(tokenizer))
        if not breakers:
            raise SystemExit("default DRY breaker set is empty")

        gen = Generator(model, cache, tokenizer)
        if gen.num_draft_tokens != 0:
            raise SystemExit(f"generator drafts {gen.num_draft_tokens}, expected 0")
        if gen.max_chunk_size != ref.MAX_CHUNK:
            raise SystemExit(
                f"max_chunk_size {gen.max_chunk_size}, expected {ref.MAX_CHUNK}"
            )
        if gen.recurrent_cache is None:
            raise SystemExit("target has no recurrent cache; chunk cuts would differ")

        cases = []
        bins = {}
        sampler_meta = {}
        hook = LogitHook()
        for length in PROMPT_ORDER:
            ids = prompts[length]
            prompt_len = int(ids.shape[-1])
            prompt_list = [int(v) for v in ids[0].tolist()]
            for kind in SAMPLER_ORDER:
                label = f"{length}_{kind}"
                for pass_i in range(WARMUPS + 1):
                    record = pass_i == WARMUPS
                    sampler, seed, collapsed = make_sampler(kind)
                    if kind not in sampler_meta:
                        sampler_meta[kind] = sampler_meta_entry(
                            kind, seed, collapsed, breakers
                        )
                    elif sampler_meta[kind]["collapsed_steps"] != collapsed:
                        raise SystemExit(f"{kind} stack changed: {collapsed}")
                    elif sampler_meta[kind]["seed"] != seed:
                        raise SystemExit(f"{kind} seed changed")
                    print(
                        f"prompt={length} sampler={kind} "
                        f"{'record' if record else f'warmup {pass_i + 1}'}",
                        file=sys.stderr,
                        flush=True,
                    )
                    ref.reset_generator(gen)
                    job, eos_reason = run_job(
                        gen, ids, sampler, seed, hook if record else None
                    )
                    if not record:
                        if hook.pending:
                            raise SystemExit("warmup captured logits; the hook leaked")
                        continue
                    rows = hook.take()
                    got_ids = sequence_ids(job)
                    if got_ids[:prompt_len] != prompt_list:
                        raise SystemExit(f"{label} prompt ids changed during the job")
                    output_tokens = got_ids[prompt_len:]
                    steps = check_recorded(
                        label,
                        prompt_list,
                        rows,
                        output_tokens,
                        eos_reason,
                        job.max_rq_tokens,
                    )
                    filename = f"logits_{length}_{kind}_step0.bin"
                    bins[filename] = steps[0]["raw"]
                    if blake_of(steps[0]["raw"]) != steps[0]["logits_blake2b"]:
                        raise SystemExit(f"{label} step0 digest does not match its payload")
                    cases.append({
                        "prompt": str(length),
                        "prompt_len": prompt_len,
                        "sampler": kind,
                        "seed": seed,
                        "eos_reason": eos_reason,
                        "max_rq_tokens": job.max_rq_tokens,
                        "logits_bin": filename,
                        "steps": [
                            {
                                "token": step["token"],
                                "position": step["position"],
                                "block_pages": step["block_pages"],
                                "logits_blake2b": step["logits_blake2b"],
                            }
                            for step in steps
                        ],
                    })
                    print(
                        f"{label}: {len(steps)} tokens, "
                        f"widths {sorted(set(step['block_pages'] for step in steps))}",
                        file=sys.stderr,
                        flush=True,
                    )

        expected = len(PROMPT_ORDER) * len(SAMPLER_ORDER)
        if len(cases) != expected:
            raise SystemExit(f"recorded {len(cases)} cases, expected {expected}")
        total_bin = sum(len(raw) for raw in bins.values())
        meta = {
            "schema_version": 1,
            "hash": "blake2b-256",
            "byte_order": "little",
            "source": "scripts/dump_27b_generate_ref.py",
            "model_dir": TARGET,
            "ext_tree": ext_tree,
            "fork_clean": True,
            "env": {name: os.environ.get(name) for name in ENV_FLAGS},
            "num_draft_tokens": 0,
            "cache_tokens": CACHE_TOKENS,
            "page_size": PAGE_SIZE,
            "max_new_tokens": MAX_NEW,
            "min_new_tokens": MAX_NEW,
            "hidden": HIDDEN,
            "vocab": VOCAB,
            "k_bits": 3,
            "v_bits": 3,
            "max_batch_size": 1,
            "warmup_jobs": WARMUPS,
            "prompt_sentence": ref.SENTENCE,
            "prompt_lengths": list(PROMPT_ORDER),
            "prompts": {
                str(length): [int(v) for v in prompts[length][0].tolist()]
                for length in PROMPT_ORDER
            },
            "samplers": sampler_meta,
            "cases": cases,
            "logit_bins": {
                "dtype": "float16",
                "shape": [VOCAB],
                "bytes": (
                    "raw little-endian fp16, no header. step 0 only, the decode "
                    "of the last prompt token, cloned before the sampler"
                ),
                "files": [{"file": name, "bytes": len(raw)} for name, raw in bins.items()],
                "total_bytes": total_bin,
            },
        }
        for name, raw in bins.items():
            (out / name).write_bytes(raw)
        tmp = out / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1) + "\n")
        tmp.replace(out / "meta.json")
        keep = set(bins) | {"meta.json", "README.md"}
        for stale in out.iterdir():
            if stale.name not in keep:
                stale.unlink()
        print(
            f"wrote {len(cases)} cases and {len(bins)} step0 rows "
            f"({total_bin} bytes) to {out / 'meta.json'}",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()

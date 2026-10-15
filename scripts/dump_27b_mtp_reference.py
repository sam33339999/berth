#!/usr/bin/env python3
"""MTP reference for the 27B target plus the fork MTP head. Needs the GPU.

Do not run while the :8889 serve is up: this loads the 27B twice over
(target and the MTP component, about 16 GB plus the draft). Do not run it
at the same time as a Rust 27B load. The script waits while ``nvidia-smi``
reports 4000 MiB or more, up to 20 minutes, then refuses. It refuses
immediately if port 8889 is listening. It does not import torch until that
check has passed.

Parity objects (``mtp-host/common.md``). One process is one ``N``
(``--num-draft``, an integer from 2 to 15). ``N`` is ``num_draft_tokens``
and the target cache's ``max_history``. The weights are loaded once, for
that ``N`` only. A second ``N`` is another process: ``BC_GatedDeltaNetSplit``
bakes conv width and history stride into the ``(bsz, seqlen, history)``
graph (``gated_delta_net.cpp`` ``run_bszN``), and replaying that graph
against a different geometry stays eager. Verify ``q_len`` is ``N + 1``,
which for this range is 3..16, the ``BC_Attention`` / GDN ``MAX_QLEN``.

- target ``Cache(model, max_num_tokens=12288, layer_type=CacheLayer_quant,
  k_bits=3, v_bits=3, max_history=N, max_batch_size=1)``
- ``Model.from_config(cfg, component="mtp")``
- draft ``Cache(draft_model, max_num_tokens=12288, layer_type=CacheLayer_quant,
  k_bits=3, v_bits=3, max_batch_size=1)`` — ``max_history`` stays 0. The MTP
  head has no recurrent layers (``generator.py`` asserts that).
- ``Generator(model, cache, tokenizer, draft_model=..., draft_cache=...,
  num_draft_tokens=N)``. ``dynamic_draft_tokens`` stays False.

``Generator.iterate`` asserts both caches were constructed before the
matching ``load()`` (``generator.py``). This script builds the target cache,
calls ``model.load()``, builds the draft cache, then calls ``draft.load()``.
12288 is a multiple of 256 and holds an 8190-token prompt, 32 new tokens,
and the draft window.

The reference set is two invocations, ``--num-draft 4`` and
``--num-draft 2``. Prompts are the single token ``"A"`` (id 32; the fork
does not prefill it), then 100, 1500 and 8190. 100, 1500 and 8190 are
prefixes of one tokenization of the sentence in
``dump_27b_prefill_reference.py`` (``prompt_ids``). Samplers, each a fresh
object per job:

- ``greedy``: ``ArgmaxSampler``, ``Job(seed=0)``. Penalties off. This is
  ``SamplerConfig::greedy()``, not ``serve_defaults(0)`` (that stack still
  applies DRY 0.2).
- ``sample``: temperature 0.7, ``Job(seed=42)``, built the way
  ``scripts/serve_openai.py`` ``_sampler`` builds a request. ``SS_RepP(1.0,
  512, 0)``, ``SS_PresFreqP(0.0, 0.0, 512, 0)``, ``SS_DRY(0.2, 1.75, 6,
  4096)`` with no tokenizer bound (default breakers resolve at sample time),
  then ``SS_Temperature(0.7)`` and ``SS_Sample()``. ``freq_range`` 512 is
  the serve ``--freq-range`` default. ``alt()`` drops the repetition and
  presence/frequency steps because they are identities. DRY stays.
  ``EXL3_FUSED_SAMPLER`` is left at its default ``"1"`` (do not set it), so
  the tail collapses to ``SS_Fused(MODE_SAMPLE, 0.7)``.

``max_new_tokens`` is 32 and ``min_new_tokens`` is 32, with
``stop_conditions=()`` and ``token_healing=False``. Nothing stops the job
early, so the 8190 prompt reaches the recurrent checkpoint at sequence
position 8192. ``eos_reason`` is ``max_new_tokens`` and the stream is 32
tokens. 100 and 1500 stay below the next multiple of 2048. 8190 + 32 crosses
8192 = 4 × 2048 on purpose.

How the seed reaches ``randint``
---------------------------------
``Job.__init__`` (``generator/job.py``) sets ``self.rng = random.Random(seed)``
when ``seed`` is not None. Every ``receive_logits``, including greedy, calls
``sampler.forward(logits, current_device_ids, self.rng.randint(0, 2**32 - 1),
tokenizer, logit_mask)``. ``CustomSampler.forward`` stores that integer as
``SamplingState.rand_u32``. ``SS_Sample.reqs_torch_seed`` is false, so this
path does not call ``torch.manual_seed`` or ``random.seed``. The integer is
the argument of ``ext.fused_sampler`` / the gumbel kernels. One draw per
position ``receive_logits`` actually reaches. ``SS_Fused`` mode 0 (greedy)
still receives the integer and does not use it to choose the token. Mode 1
uses it as the gumbel stream. Seeding a small Python ``int`` is the
``init_by_array`` path ``berth`` ``PyRandom::seed_u64`` reproduces. Kernel
tests that pass an explicit ``rand_u32`` are a different entry point; this
dump goes through ``Job`` so the draws match the generator.

``SS_Fused.inv_temp`` is the Python float ``1.0 / temperature``. The
extension narrows it to f32. For 0.7 that is ``0x3fb6db6e``, the same bits as
Rust ``(1.0f64 / f64::from(0.7f32)) as f32``. The script records the live
object's bits and refuses a sample stack whose bits differ.

Warmup
------
Each new CUDA-graph slot does eager, then capture, then replay
(``dump_27b_decode_sequence.py``). This process has one cache geometry, so
every slot is captured against this ``N``. The slots are target ``q_len`` 1
with ``recurrent_history`` false (the ``"A"`` bootstrap: a one-token prompt
leaves ``mtp_last_hidden`` unset and ``iterate_draftmodel_mtp_gen`` returns
None), target ``q_len`` ``N + 1`` with history true (verify), and the draft
refresh lengths ``accepted_length - 1``. History on and history off are
different GDN slots. ``BC_Attention`` is one object per cache layer
(``id(layer)`` in ``attn.py``); this process allocates that layer once.
Block-table width is patched on replay (``attention.cpp``), not recaptured.
``BC_GatedMLP`` is keyed by token count and does not read the cache.

For each ``(prompt, sampler)`` the script discards two identical full
32-token jobs, then records the third. Those two jobs are the eager run and
the capture for every slot that case hits, including ``q_len`` 1 and
``q_len`` ``N + 1``. A one-token warmup would never enter the later refresh
lengths, so those slots would capture on the recorded job. Hooks are not
installed on the two discarded jobs: the capture must record only the
fork's kernels. Clones and digests run after the wrapped forward returns
and are not inside the capture. Between jobs the page table is reset and
the target KV, the draft KV and every GDN state are zeroed, and the
recurrent checkpoint cache is cleared, so the prefix cache cannot skip the
second prefill. The script refuses to start if ``EXL3_BC_ATTN`` is ``0``.
BC stays on. berth does not capture graphs; these replays are the reference
(``docs/mtp.md`` 未確認 2).

What one ``iterate()`` is
-------------------------
``Job.prefill`` runs at most one chunk of ``ids[:-1]``. With MTP that chunk
is ``model.forward`` with ``last_tokens_only`` (not ``model.prefill``), so
layer 63 runs its MLP and the final norm exports every row. The same chunk
is then ``draft_model.prefill`` with the shifted post-norm states. The last
chunk of the prompt finishes prefill inside that same ``iterate()``, which
then drafts and verifies. 100 is one iterate that does both. 1500 and 8190
have prefill-only iterates before that. ``"A"`` has no prefill; its first
iterate is the 1-token forward.

Page and row addressing
-----------------------
``logical = pos // 256``, ``row = pos % 256``. The physical page is
``seq.allocated_pages[logical].page_index``, equal to
``seq.block_index_tensor[0, logical]`` while the sequence is live. The
verify block table is padded up to a multiple of 16 with zeros; it is not
used here. ``reset_page_table`` rebuilds pages with fresh hashes and the
tree eviction order, so a physical index is not ``logical``. Both are
recorded.

``Job.deallocate_pages`` on the finishing iterate drops the page list and
returns the GDN slot, and does not wipe KV or GDN bytes. Physical indices
are snapshotted during the forward, while ``allocated_pages`` still exists,
and hashed after ``iterate()`` returns (rewind and refresh have already
run). ``free()`` does not zero the state. Defrag cannot move those pages:
``reset_page_table`` puts ``access_serial`` back under the defrag threshold
before every job.

A target digest is the whole physical page, all 256 rows, one blake2b per
tensor. ``qk`` / ``qv`` are the raw int32 packed words
(``token_dim // 32 * bits`` per row, ``torch.int``). ``sk`` / ``sv`` are the
fp16 scales (``token_dim // 32`` per row). For the 27B attention,
``token_dim = 4 * 256 = 1024``, so the packed key/value width is 96 and the
scale width is 32. Rejected rows stay in the page: ``reject_remainder``
rewinds ``page.kv_position`` and the GDN state and does not erase cq3.

A draft digest is the concatenation, in increasing position order, of only
the rows that write touched. Rows in other positions on the same page are
not included. Prompt MTP prefill rows and refresh rows are separate
objects. They do not overlap: prefill writes ``[chunk_start, chunk_end)``
and that end is ``prompt_len - 1``, while the first refresh starts at
``prompt_len``. The speculative ``draft_model.forward`` rows (the anchor at
position ``P`` and the draft tail refresh does not overwrite) are not
hashed.

Checkpoint
----------
``is_checkpoint_boundary`` with no override uses
``(seq_pos - cached_pages * 256) % interval``. ``cached_pages`` stays 0
because the page table was reset. During generation ``seq_pos`` equals
``len(sequence_ids) - 1``, which is past ``prompt_len - 2 * max_chunk``, so
``interval`` is 2048. ``maybe_stash_recurrent`` calls the same method; those
hits are ``where: "stash"``. The acceptance loop calls it directly; those
hits are ``where: "accept"``. A hit on the last logit of a window
(``i == q_len - 1``) is not an accept stop, because the loop only checks
while a draft remains. After prefill, ``kv_position`` is 8189. The third
new token moves it to 8192. For ``N = 4`` that index can fall inside the
window (``i < q_len - 1``), so the accept loop records it and the next
iterate's ``maybe_stash_recurrent`` records it again (the accept path does
not set ``last_recurrent_checkpoint_pos``). For ``N = 2`` the same position
can be the last logit, which the accept loop skips; the next iterate then
records the stash. The script only requires that 8192 appears, as either
kind. Prefill of 8190 also stashes 4096 and 6144 (those positions use the
2048 interval; 2048 itself uses ``interval_pp`` 32768 and does not hit).
The script requires 8192, 4096 and 6144 on every 8190 case, and requires
no multiple of 2048 on the other prompts.

``meta_N<N>.json`` schema (schema_version 1)
---------------------------------------------
UTF-8 JSON, ``indent`` 1. Hex strings are lowercase blake2b-256 of the
little-endian contiguous payload (C order, no header, no stride padding).
fp16 and bf16 are the raw 16-bit patterns (viewed as int16 before
``tobytes``), not a conversion to f32. fp32 and int32 are little-endian.
The machine must be little-endian. No tensor bodies in this file.

Top level:

- ``schema_version`` int 1, ``hash`` ``"blake2b-256"``, ``byte_order``
  ``"little"``, ``source`` this script.
- ``model_dir``, ``ext_tree`` (``git rev-parse HEAD:exllamav3/exllamav3_ext``;
  must start with ``de35ffb``), ``fork_clean`` true.
- ``env``: the four ``EXL3_*`` strings. ``EXL3_FUSED_SAMPLER`` is not set.
- ``num_draft_tokens`` int, the ``--num-draft`` of this process. Every
  case in the file uses this ``N``.
- ``cache_tokens`` 12288, ``page_size`` 256, ``max_new_tokens`` 32,
  ``min_new_tokens`` 32, ``hidden`` 5120, ``vocab`` 248320, ``k_bits`` 3,
  ``v_bits`` 3, ``max_batch_size`` 1, ``warmup_jobs`` 2.
- ``checkpoint_interval`` 2048, ``checkpoint_interval_pp`` 32768.
- ``prompt_sentence``, ``prompt_a`` ``"A"``.
- ``prompts``: ``{"A": [32], "100": [...], "1500": [...], "8190": [...]}``.
  100 and 1500 are prefixes of 8190. Ints.
- ``layers``: ``attn``, ``gdn``, ``draft_attn``. Each is a list of
  ``[layer_idx, instance]`` sorted by that pair. Target attention is 16
  entries, GDN is 48, draft attention is 1. Later per-layer arrays follow
  this order and do not repeat the index.
- ``layouts``: dtype and shape of every digested tensor. ``by_num_draft_tokens``
  has exactly one key, the decimal string of this ``N``. Shapes that depend
  on ``N`` live there (``conv`` ``[10240, 4+N]`` bf16, ``recurrent``
  ``[N+1, 48, 128, 128]`` fp32). cq3 page shapes do not depend on ``N``.
- ``samplers.greedy`` and ``samplers.sample``: ``seed``, the steps passed
  to ``CustomSampler`` (identity steps marked ``dropped``; temperature and
  multinomial marked ``collapsed_into_fused`` because ``alt()`` folds them
  into ``SS_Fused``), and ``collapsed_steps`` actually executed (``type``,
  and for ``SS_Fused`` ``mode`` plus ``inv_temp_f32_bits``). Sample also
  has the f32-rounded DRY fields, ``dry_*_f32_bits``, and
  ``dry_breaker_token_ids`` (sorted), the set
  ``dry_default_sequence_breaker_tokens`` resolves for this tokenizer.
  ``collapsed_steps`` is what runs.
- ``cases``: 8 objects, this ``N`` only, prompts ``A`` / ``100`` / ``1500``
  / ``8190``, samplers ``greedy`` then ``sample``.
- ``logit_bins``: the 8 filenames and the byte contract below. The same
  bytes are written under ``testdata/mtp_ref`` and under ``target/mtp-ref``.
- ``omissions``: speculative draft-forward rows are not hashed; 8190
  ``block_in`` is not comparable to the old 32-page prefill dump.

Each case:

- ``prompt`` string, ``prompt_len`` int, ``num_draft_tokens`` int,
  ``sampler`` ``"greedy"`` or ``"sample"``, ``seed`` int.
- ``pages_allocated`` int, ``page_class`` ``"div16"`` if that count is a
  multiple of 16, otherwise ``"other"``. This is the job's page count
  (``max_rq_tokens = max_new_tokens + 1 + N``), not the number of pages one
  round touched. Page count follows ``ceil((prompt_len + max_rq_tokens) / 256)``
  with ``max_rq_tokens = 33 + N``. 100 is 1 page at ``N`` 2 and 4. 1500 is
  6 pages at ``N = 2`` and 7 at ``N = 4``; both are ``other`` and below the
  attention split, so the 100/1500 residuals stay in the same Triton class
  as ``testdata/prefill_ref``. 8190 is 33 pages (``other``) for ``N`` 2..15.
  The old prefill dump used ``max_new_tokens = 1`` and got 32 pages
  (``div16``). Do not compare 8190 ``block_in`` to that dump. The file
  records the live ``pages_allocated`` for this ``N``.
- ``max_rq_tokens`` int. ``compare_block_in_to_prefill_ref`` bool, true only
  for 100 and 1500.
- ``rounds``: one object per ``iterate()``, in order.
- ``output_tokens`` int list, length 32. ``eos_reason`` string.
- ``logits_bin`` filename. ``checkpoint_hits``: list of
  ``{"iterate": int, "seq_pos": int, "where": "accept"|"stash",
  "override": null or int}``.

Each round:

- ``i`` iterate index from 0.
- ``prefill`` null, or ``{"start", "end", "block_in", "export"}``.
  ``block_in`` is 64 hex strings, layers 0..63, each the fp32 residual
  entering ``TransformerBlock.forward``, shape ``[1, end-start, 5120]``.
  ``block_in[k+1]`` is the tensor ``testdata/prefill_ref`` hashes as the
  block-output residual of layer ``k`` (that dump hashes the block return;
  this one hashes the next block's input). Layer 63 of the old dump
  early-returns, so its digest equals this ``block_in[63]``. ``block_in[0]``
  is the embedding residual and is not in the old dump. ``export`` is one
  hex of the fp16 post-norm rows, shape ``[1, end-start, 5120]``
  (``params["export_states"][-1]``, and that list has length 1).
- ``verify`` null, or ``{"index", "kv_pos", "q_len", "recurrent_history",
  "draft_ids", "logit_rows", "export"}``. ``index`` counts verify forwards
  only, from 0. ``kv_pos`` is ``cache_seqlens`` at the forward (the write
  starts here; ``seq.kv_position`` is still this value, while
  ``page.kv_position`` has already been advanced by ``q_len``).
  ``draft_ids`` is null or ``N`` ints. Null only for the ``"A"`` bootstrap
  (``q_len`` 1, ``recurrent_history`` false). Otherwise ``q_len == N + 1``
  and ``recurrent_history`` is true. ``logit_rows`` is one hex per row,
  each fp16 ``[1, 1, 248320]``, row-major. ``export`` is fp16
  ``[1, q_len, 5120]``.
- ``emitted`` int list. The ids appended to ``sequence_ids`` during this
  iterate. The cursor starts at ``prompt_len``. Empty when ``verify`` is
  null. Otherwise its length is ``1 + accepted_delta`` (the anchor sample
  plus each accepted draft; a mismatched draft is not appended, the
  target's sample at that index is). This is not the streamed
  ``token_ids``. ``receive_sample`` can return ``emit`` without
  ``emit_held`` while a token piece ends in an incomplete UTF-8 character,
  so those ids show up in a later result. The ``max_new_tokens`` path
  passes ``emit_held=True`` (``job.py`` ``emit``), and with
  ``stop_conditions=()`` the stop-string matcher is not installed
  (``_strings_to_utf32`` of an empty tuple returns no offsets). At eos the
  concatenation of the streamed ids equals ``output_tokens``. The script
  checks that.
- ``accepted_delta``, ``rejected_delta``, ``accepted_total``,
  ``rejected_total``. Totals are the job's cumulative counters
  (``job.py``). Deltas are this iterate. A prefill-only round has deltas 0.
  When ``draft_ids`` is not null, ``accepted_delta + rejected_delta == N``.
- ``gdn``: list parallel to ``layers.gdn``. Each entry is ``{"conv",
  "recurrent"}``, hex of slot 0, the entire tensor including every history
  index. Present on prefill-only rounds too. Hashed after ``iterate()``.
- ``target_pages``: the logical pages this iterate wrote (prefill span,
  verify span, or both), sorted by ``logical``. Each is ``{"logical",
  "physical", "qk", "qv", "sk", "sv"}``. The four arrays are parallel to
  ``layers.attn``. Each hex is the whole physical page (256 rows), hashed
  after rewind, so a rejected tail is still in the page.
- ``draft_prefill`` and ``draft_refresh``: null, or ``{"start", "length",
  "qk", "qv", "sk", "sv"}``. The four arrays are parallel to
  ``layers.draft_attn``. Each hex is the concatenation of ``length`` rows
  starting at ``start``, shape ``[length, packed]``, not the rest of the
  page. ``draft_prefill.length`` equals the target prefill span. Refresh
  exists only when ``accepted_delta > 0``; then ``start == kv_pos + 1`` and
  ``length == accepted_delta``.

``logits_<prompt>_N<N>_<sampler>_round0.bin`` is the raw little-endian fp16
of verify ``index`` 0, row 0, shape ``[248320]``, no header. Those bytes are
the same bytes as ``logit_rows[0]`` (the digest is of shape ``[1, 1, 248320]``,
which is the same payload). One row is 496640 bytes. Eight cases are
3973120 bytes, about 3.79 MiB. A full verify window for every case of one
``N`` is ``8 * (N + 1)`` rows (about 19.9 MiB at ``N = 4``), over the 10 MB
cap, so only row 0 of the first verify is stored. The other rows are
digests in ``logit_rows``. ``round0`` is the first verify forward, not
iterate 0: 1500 and 8190 prefill before they verify.

The script writes ``testdata/mtp_ref/meta_N<N>.json`` and the eight bins
only after every case has succeeded. The meta file is written last, via
``meta_N<N>.json.tmp`` then replace. A crash leaves no partial meta. It
does not delete ``README.md``. The same bin bytes are also written to
``target/mtp-ref/logits_<prompt>_N<N>_<sampler>_round0.bin``. Every file
under ``target/mtp-ref`` has this ``N`` in the name, so a second process
does not overwrite the first. ``[META_DIR]`` overrides the directory of the
meta file and of the testdata copy of the bins. The ``target/mtp-ref``
copy stays under the repo ``target/``.

Run once per ``N``. The reference set is 4 and 2. Use the serve's
environment (``scripts/serve-host.sh`` exports). ``<torch nvidia libs>`` is
the ``nvidia/*/lib`` set that script computes from this interpreter's
torch::

    PYTHONPATH=/home/sam/qwen38-exl3-dflash2/exllamav3 \\
    TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \\
    LD_LIBRARY_PATH=<torch nvidia libs>:$LD_LIBRARY_PATH \\
    EXL3_INT8_GEMV=2 EXL3_INT8_GEMV_MAX_K=5 EXL3_QC_STAGING=1 \\
    EXL3_HGEMM_F16ACC=auto \\
    /home/sam/qwen38-exl3-dflash2/env/bin/python \\
        scripts/dump_27b_mtp_reference.py --num-draft 4 [META_DIR]

    /home/sam/qwen38-exl3-dflash2/env/bin/python \\
        scripts/dump_27b_mtp_reference.py --num-draft 2 [META_DIR]

``META_DIR`` defaults to ``testdata/mtp_ref``. ``--num-draft`` is required.
The GPU must be free.
"""

import argparse
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
LAYERS = 64
MAX_NEW = 32
WARMUPS = 2
MEM_LIMIT_MIB = 4000
MEM_WAIT_S = 60
MEM_WAIT_BUDGET_S = 20 * 60
BIN_BUDGET = 10 * 1024 * 1024
PROMPT_ORDER = ("A", "100", "1500", "8190")
SAMPLER_ORDER = ("greedy", "sample")
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
UNSET = object()


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


def digest(tensor, expect_dtype, expect_shape):
    import dump_27b_prefill_reference as ref

    name, shape, raw = ref.le_bytes(tensor)
    if name != expect_dtype or shape != list(expect_shape):
        raise RuntimeError(
            f"digest expected {expect_dtype} {list(expect_shape)}, got {name} {shape}"
        )
    return ref.blake2b_256(raw), raw


def digest_hex(tensor, expect_dtype, expect_shape):
    hexdigest, _raw = digest(tensor, expect_dtype, expect_shape)
    return hexdigest


def seqlens0(params):
    tensor = params.get("cache_seqlens")
    if tensor is None:
        raise RuntimeError("forward params have no cache_seqlens")
    flat = tensor.detach().to("cpu").reshape(-1)
    if flat.numel() != 1:
        raise RuntimeError(f"cache_seqlens has {flat.numel()} entries")
    return int(flat[0].item())


def layer_pairs(keys):
    pairs = []
    for key in keys:
        if not isinstance(key, tuple) or len(key) != 2:
            raise RuntimeError(f"cache layer key {key!r} is not (layer, instance)")
        pairs.append([int(key[0]), int(key[1])])
    if pairs != sorted(pairs):
        raise RuntimeError(f"layer keys are not sorted: {pairs}")
    return pairs


def note_span(seq, start, length, phys_of):
    """Record physical pages touched by positions ``[start, start+length)``.

    ``logical = pos // 256`` and ``row = pos % 256``. The physical page is
    ``allocated_pages[logical].page_index``. That list is the sequence's own
    map; the verify block table is wider and zero-padded, so it is not read
    here. The same logical page must keep the same physical index for the
    whole iterate.
    """
    if length <= 0:
        raise RuntimeError(f"empty cache span at {start}")
    if start < 0:
        raise RuntimeError(f"negative cache position {start}")
    last = start + length - 1
    first_page = start // PAGE_SIZE
    last_page = last // PAGE_SIZE
    for logical in range(first_page, last_page + 1):
        if seq.allocated_pages is None or logical >= len(seq.allocated_pages):
            raise RuntimeError(
                f"logical page {logical} is outside the {len(seq.allocated_pages or [])} "
                f"allocated pages (span [{start}, {start + length}))"
            )
        phys = int(seq.allocated_pages[logical].page_index)
        table = seq.block_index_tensor
        if table is None or table.dim() != 2 or int(table.shape[1]) <= logical:
            raise RuntimeError(
                f"block_index_tensor does not cover logical page {logical}"
            )
        if int(table[0, logical].item()) != phys:
            raise RuntimeError(
                f"logical page {logical}: allocated_pages says {phys}, "
                f"block_index_tensor says {int(table[0, logical].item())}"
            )
        prev = phys_of.get(logical)
        if prev is not None and prev != phys:
            raise RuntimeError(
                f"logical page {logical} changed physical index {prev} -> {phys}"
            )
        phys_of[logical] = phys


def rows_digest(tensor, start, length, phys_of, expect_dtype, packed):
    """Concatenate the written rows in position order and hash them.

    Position ``p`` is ``tensor[phys_of[p // 256], p % 256]``. The other rows
    of those pages are not part of the payload. Shape is ``[length, packed]``.
    """
    import torch

    parts = []
    for pos in range(start, start + length):
        logical = pos // PAGE_SIZE
        row = pos % PAGE_SIZE
        if logical not in phys_of:
            raise RuntimeError(f"position {pos} has no snapshotted physical page")
        parts.append(tensor[phys_of[logical], row])
    stacked = torch.stack(parts, dim=0).contiguous()
    return digest_hex(stacked, expect_dtype, [length, packed])


def page_digest(tensor, phys, expect_dtype, row_shape):
    page = tensor[phys]
    return digest_hex(page, expect_dtype, [PAGE_SIZE, *row_shape])


def reset_pair(gen, target_cache, draft_cache):
    import dump_27b_prefill_reference as ref

    if gen.num_remaining_jobs():
        raise RuntimeError("resetting the page table while a job is still queued")
    gen.pagetable.reset_page_table()
    gen.recurrent_cache.clear()
    gen.recurrent_cache.current_size = 0
    ref.zero_cache(target_cache)
    ref.zero_cache(draft_cache)


def active_job(gen):
    if len(gen.active_jobs) != 1:
        raise RuntimeError(f"expected one active job, found {len(gen.active_jobs)}")
    job = gen.active_jobs[0]
    if len(job.sequences) != 1:
        raise RuntimeError(f"expected one sequence, found {len(job.sequences)}")
    state = job.recurrent_state
    if state is None or int(state.slot) != 0:
        raise RuntimeError(
            f"GDN slot is {None if state is None else state.slot}, expected 0"
        )
    return job, job.sequences[0]


def emitted_ids(results):
    found = []
    eos = None
    for result in results:
        if result.get("eos"):
            if eos is not None:
                raise RuntimeError(f"two eos results in one iterate ({eos}, {result.get('eos_reason')})")
            eos = result.get("eos_reason")
        src = result if "token_ids" in result else (result.get("held") or {})
        if "token_ids" not in src:
            continue
        ids = src["token_ids"].detach().to("cpu").reshape(-1)
        found.extend(int(v) for v in ids.tolist())
    return found, eos


def job_from(results, gen):
    for result in reversed(results):
        job = result.get("job")
        if job is not None:
            return job
    if len(gen.active_jobs) == 1:
        return gen.active_jobs[0]
    raise RuntimeError("iterate returned no job")


class Recorder:
    """Per-job digests. Hooks append; ``finish_iterate`` hashes after rewind."""

    def __init__(self):
        self.rounds = []
        self.output_tokens = []
        self.checkpoint_hits = []
        self.eos_reason = None
        self.row0 = None
        self.pages_allocated = None
        self.max_rq_tokens = None
        self.verify_seen = False
        self.verify_index = 0
        self.prev_accepted = 0
        self.prev_rejected = 0
        self.num_draft = None
        self.in_forward = False
        self.forward_last_only = False
        self.in_stash = 0
        self.expect_rows = None
        self.block_in = []
        self.iterate_i = -1
        self.draft_ids = UNSET
        self.prefill = None
        self.verify = None
        self.draft_prefill = None
        self.draft_refresh = None
        self.phys_of = {}
        self.seq_cursor = None
        self.stream_tokens = []

    def begin_job(self, num_draft, prompt_len):
        self.__init__()
        self.num_draft = num_draft
        self.seq_cursor = int(prompt_len)

    def begin_iterate(self):
        self.iterate_i += 1
        self.draft_ids = UNSET
        self.prefill = None
        self.verify = None
        self.draft_prefill = None
        self.draft_refresh = None
        self.phys_of = {}
        self.block_in = []
        self.in_forward = False
        self.forward_last_only = False
        self.expect_rows = None

    def on_draft_ids(self, ids):
        if self.draft_ids is not UNSET:
            raise RuntimeError("iterate_draftmodel_mtp_gen ran twice in one iterate")
        if ids is None:
            self.draft_ids = None
            return
        flat = [int(v) for v in ids.detach().to("cpu").reshape(-1).tolist()]
        if len(flat) != self.num_draft:
            raise RuntimeError(f"draft returned {len(flat)} ids, N is {self.num_draft}")
        self.draft_ids = flat

    def on_draft_prefill(self, gen, input_ids, params):
        start = seqlens0(params)
        length = int(input_ids.shape[-1])
        if input_ids.shape[0] != 1:
            raise RuntimeError(f"draft prefill batch {tuple(input_ids.shape)}")
        _job, seq = active_job(gen)
        note_span(seq, start, length, self.phys_of)
        self.pages_allocated = self._remember_pages(seq)
        span = {"start": start, "length": length}
        # Prompt prefill runs before the first verify of this job. Refresh runs
        # after a verify forward in the same or a later iterate. ``verify_seen``
        # flips only at the end of that forward, so both can land in one iterate.
        if self.verify_seen:
            if self.draft_refresh is not None:
                raise RuntimeError("two draft refreshes in one iterate")
            self.draft_refresh = span
        else:
            if self.draft_prefill is not None:
                raise RuntimeError("two MTP prompt prefills in one iterate")
            self.draft_prefill = span

    def on_block_in(self, layer, x):
        if not self.in_forward or not self.forward_last_only:
            return
        if layer != len(self.block_in):
            raise RuntimeError(
                f"block_in layer {layer} arrived, expected {len(self.block_in)}"
            )
        self.block_in.append(
            digest_hex(x, "float32", [1, self.expect_rows, HIDDEN])
        )

    def on_checkpoint(self, job, hit, override):
        if not hit:
            return
        seq_pos = int(job.sequences[0].kv_position)
        where = "stash" if self.in_stash else "accept"
        self.checkpoint_hits.append({
            "iterate": self.iterate_i,
            "seq_pos": seq_pos,
            "where": where,
            "override": None if override is None else int(override),
        })

    def on_target_forward(self, gen, input_ids, params, y):
        import torch

        last_only = bool(params.get("last_tokens_only"))
        if input_ids.shape[0] != 1:
            raise RuntimeError(f"target forward batch {tuple(input_ids.shape)}")
        rows = int(input_ids.shape[-1])
        start = seqlens0(params)
        _job, seq = active_job(gen)
        note_span(seq, start, rows, self.phys_of)
        self.pages_allocated = self._remember_pages(seq)
        exports = params.get("export_states")
        if not isinstance(exports, list) or len(exports) != 1:
            raise RuntimeError(
                f"export_states length is {None if exports is None else len(exports)}, expected 1"
            )
        export_hex = digest_hex(exports[-1], "float16", [1, rows, HIDDEN])
        if last_only:
            if self.prefill is not None:
                raise RuntimeError("two MTP prefills in one iterate")
            if len(self.block_in) != LAYERS:
                raise RuntimeError(
                    f"MTP prefill recorded {len(self.block_in)} block_in residuals, expected {LAYERS}"
                )
            self.prefill = {
                "start": start,
                "end": start + rows,
                "block_in": self.block_in,
                "export": export_hex,
            }
            self.block_in = []
            return
        if not isinstance(y, torch.Tensor):
            raise RuntimeError(f"model.forward returned {type(y).__name__}")
        if y.dtype != torch.float16 or list(y.shape) != [1, rows, VOCAB]:
            raise RuntimeError(
                f"verify logits are {y.dtype} {tuple(y.shape)}, expected fp16 [1, {rows}, {VOCAB}]"
            )
        logit_rows = []
        for i in range(rows):
            hexdigest, raw = digest(y[:, i:i + 1, :], "float16", [1, 1, VOCAB])
            logit_rows.append(hexdigest)
            if self.verify_index == 0 and i == 0:
                if len(raw) != VOCAB * 2:
                    raise RuntimeError(f"logit row 0 is {len(raw)} bytes, expected {VOCAB * 2}")
                self.row0 = raw
        history = bool(params.get("recurrent_history"))
        self.verify = {
            "index": self.verify_index,
            "kv_pos": start,
            "q_len": rows,
            "recurrent_history": history,
            "draft_ids": None if self.draft_ids is None else list(self.draft_ids),
            "logit_rows": logit_rows,
            "export": export_hex,
        }
        self.verify_index += 1
        self.verify_seen = True

    def _remember_pages(self, seq):
        count = len(seq.allocated_pages)
        if self.pages_allocated is None:
            self.pages_allocated = count
        elif self.pages_allocated != count:
            raise RuntimeError(
                f"allocated pages changed {self.pages_allocated} -> {count}"
            )
        return self.pages_allocated

    def finish_iterate(self, results, gen, target_cache, draft_cache, attn_keys, gdn_keys, draft_keys):
        import torch

        if self.draft_ids is UNSET:
            raise RuntimeError("iterate_draftmodel_mtp_gen was not called")
        torch.cuda.synchronize()
        job = job_from(results, gen)
        if self.max_rq_tokens is None:
            self.max_rq_tokens = int(job.max_rq_tokens)
        accepted = int(job.accepted_draft_tokens)
        rejected = int(job.rejected_draft_tokens)
        accepted_delta = accepted - self.prev_accepted
        rejected_delta = rejected - self.prev_rejected
        if accepted_delta < 0 or rejected_delta < 0:
            raise RuntimeError(
                f"draft counters went backwards accepted {self.prev_accepted}->{accepted} "
                f"rejected {self.prev_rejected}->{rejected}"
            )
        self.prev_accepted = accepted
        self.prev_rejected = rejected
        # Streamed token_ids can lag sequence_ids while a piece ends in an
        # incomplete UTF-8 character (emit without emit_held). The ids this
        # iterate actually appended are the growth of sequence_ids.
        stream, eos = emitted_ids(results)
        self.stream_tokens.extend(stream)
        seq_ids = job.sequences[0].sequence_ids.torch().detach().to("cpu").reshape(-1)
        all_ids = [int(v) for v in seq_ids.tolist()]
        if self.seq_cursor is None or len(all_ids) < self.seq_cursor:
            raise RuntimeError(
                f"sequence shrank to {len(all_ids)} from cursor {self.seq_cursor}"
            )
        emitted = all_ids[self.seq_cursor:]
        self.seq_cursor = len(all_ids)
        if eos is not None:
            if self.eos_reason is not None:
                raise RuntimeError(f"eos twice: {self.eos_reason} then {eos}")
            self.eos_reason = eos
        self.output_tokens.extend(emitted)
        if eos is not None and self.stream_tokens != self.output_tokens:
            raise RuntimeError(
                f"streamed token_ids ({len(self.stream_tokens)}) diverged from "
                f"sequence_ids ({len(self.output_tokens)}) at eos"
            )

        verify = self.verify
        if verify is None:
            if self.draft_ids is not None or emitted or accepted_delta or rejected_delta:
                raise RuntimeError(
                    f"prefill-only iterate {self.iterate_i} drafted or sampled: "
                    f"draft_ids={self.draft_ids} emitted={emitted} "
                    f"acc+{accepted_delta} rej+{rejected_delta}"
                )
            if self.draft_refresh is not None:
                raise RuntimeError("refresh without a verify forward")
        else:
            draft_ids = verify["draft_ids"]
            if draft_ids is None:
                if verify["q_len"] != 1 or verify["recurrent_history"]:
                    raise RuntimeError(
                        f"bootstrap verify is q_len {verify['q_len']} "
                        f"history {verify['recurrent_history']}"
                    )
                if accepted_delta or rejected_delta:
                    raise RuntimeError("q_len 1 verify changed draft counters")
            else:
                if (
                    len(draft_ids) != self.num_draft
                    or verify["q_len"] != self.num_draft + 1
                    or not verify["recurrent_history"]
                ):
                    raise RuntimeError(
                        f"verify q_len {verify['q_len']} history {verify['recurrent_history']} "
                        f"drafts {len(draft_ids)} for N={self.num_draft}"
                    )
                if accepted_delta + rejected_delta != self.num_draft:
                    raise RuntimeError(
                        f"accepted {accepted_delta} + rejected {rejected_delta} != N {self.num_draft}"
                    )
            if len(emitted) != 1 + accepted_delta:
                raise RuntimeError(
                    f"emitted {len(emitted)} tokens, expected {1 + accepted_delta} "
                    f"(anchor plus accepted drafts)"
                )
            refresh = self.draft_refresh
            if accepted_delta > 0:
                if refresh is None:
                    raise RuntimeError("accepted drafts but draft cache was not refreshed")
                if refresh["start"] != verify["kv_pos"] + 1 or refresh["length"] != accepted_delta:
                    raise RuntimeError(
                        f"refresh [{refresh['start']}, +{refresh['length']}) != "
                        f"kv {verify['kv_pos']}+1 length {accepted_delta}"
                    )
            elif refresh is not None:
                raise RuntimeError("refresh ran with no accepted draft")

        prefill = self.prefill
        prompt_draft = self.draft_prefill
        if prefill is None:
            if prompt_draft is not None:
                raise RuntimeError("draft prompt prefill without a target prefill")
        else:
            span = prefill["end"] - prefill["start"]
            if prompt_draft is None or prompt_draft["start"] != prefill["start"] or prompt_draft["length"] != span:
                raise RuntimeError(
                    f"draft prefill {prompt_draft} does not match target prefill "
                    f"[{prefill['start']}, {prefill['end']})"
                )

        gdn = []
        for key in gdn_keys:
            conv, recurrent = target_cache.get_all_recurrent_layers()[key].get_state_tensors()
            gdn.append({
                "conv": digest_hex(
                    conv[0], "bfloat16", [int(conv.shape[1]), int(conv.shape[2])]
                ),
                "recurrent": digest_hex(
                    recurrent[0], "float32", [int(s) for s in recurrent.shape[1:]]
                ),
            })
        target_pages = self._hash_target_pages(target_cache, attn_keys)
        if prompt_draft is not None:
            self._hash_draft_rows(draft_cache, draft_keys, prompt_draft)
        if self.draft_refresh is not None:
            self._hash_draft_rows(draft_cache, draft_keys, self.draft_refresh)

        self.rounds.append({
            "i": self.iterate_i,
            "prefill": prefill,
            "verify": verify,
            "emitted": emitted,
            "accepted_delta": accepted_delta,
            "rejected_delta": rejected_delta,
            "accepted_total": accepted,
            "rejected_total": rejected,
            "gdn": gdn,
            "target_pages": target_pages,
            "draft_prefill": prompt_draft,
            "draft_refresh": self.draft_refresh,
        })

    def _hash_target_pages(self, cache, attn_keys):
        pages = []
        for logical in sorted(self.phys_of):
            phys = self.phys_of[logical]
            entry = {"logical": logical, "physical": phys, "qk": [], "qv": [], "sk": [], "sv": []}
            for key in attn_keys:
                layer = cache.layers[key]
                packed_k = int(layer.qshape_k[-1])
                packed_v = int(layer.qshape_v[-1])
                packed_s = int(layer.qshape_s[-1])
                entry["qk"].append(page_digest(layer.qk, phys, "int32", [packed_k]))
                entry["qv"].append(page_digest(layer.qv, phys, "int32", [packed_v]))
                entry["sk"].append(page_digest(layer.sk, phys, "float16", [packed_s]))
                entry["sv"].append(page_digest(layer.sv, phys, "float16", [packed_s]))
            pages.append(entry)
        return pages

    def _hash_draft_rows(self, cache, draft_keys, span):
        span["qk"] = []
        span["qv"] = []
        span["sk"] = []
        span["sv"] = []
        for key in draft_keys:
            layer = cache.layers[key]
            packed_k = int(layer.qshape_k[-1])
            packed_v = int(layer.qshape_v[-1])
            packed_s = int(layer.qshape_s[-1])
            span["qk"].append(rows_digest(
                layer.qk, span["start"], span["length"], self.phys_of, "int32", packed_k
            ))
            span["qv"].append(rows_digest(
                layer.qv, span["start"], span["length"], self.phys_of, "int32", packed_v
            ))
            span["sk"].append(rows_digest(
                layer.sk, span["start"], span["length"], self.phys_of, "float16", packed_s
            ))
            span["sv"].append(rows_digest(
                layer.sv, span["start"], span["length"], self.phys_of, "float16", packed_s
            ))


class Hooks:
    """Instance and class wrappers. Installed only around the recorded job."""

    def __init__(self, model, draft, blocks):
        self.model = model
        self.draft = draft
        self.blocks = blocks
        self.saved = []
        self.rec = None
        self.gen = None

    def bind(self, obj, name, fn):
        self.saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, fn)

    def install(self, rec, gen):
        from exllamav3 import Job

        self.rec = rec
        self.gen = gen
        try:
            self._install(Job)
        except Exception:
            self.uninstall()
            raise

    def _install(self, job_cls):
        rec_box = self
        orig_draft = self.gen.iterate_draftmodel_mtp_gen
        orig_forward = self.model.forward
        orig_prefill = self.draft.prefill
        orig_boundary = job_cls.is_checkpoint_boundary
        orig_stash = job_cls.maybe_stash_recurrent

        def draft_ids(results):
            ids = orig_draft(results)
            rec_box.rec.on_draft_ids(ids)
            return ids

        def target_forward(input_ids, params=None):
            if params is None:
                raise RuntimeError("target forward was called without params")
            rec = rec_box.rec
            rec.in_forward = True
            rec.forward_last_only = bool(params.get("last_tokens_only"))
            rec.expect_rows = int(input_ids.shape[-1])
            rec.block_in = []
            try:
                y = orig_forward(input_ids, params)
            finally:
                rec.in_forward = False
            rec.on_target_forward(rec_box.gen, input_ids, params, y)
            return y

        def draft_prefill(input_ids, params=None):
            if params is None:
                raise RuntimeError("draft prefill was called without params")
            rec_box.rec.on_draft_prefill(rec_box.gen, input_ids, params)
            return orig_prefill(input_ids, params)

        def boundary(self, override_interval=None):
            hit = orig_boundary(self, override_interval)
            rec_box.rec.on_checkpoint(self, hit, override_interval)
            return hit

        def stash(self, cache, interval=None):
            rec_box.rec.in_stash += 1
            try:
                return orig_stash(self, cache, interval)
            finally:
                rec_box.rec.in_stash -= 1

        self.bind(self.gen, "iterate_draftmodel_mtp_gen", draft_ids)
        self.bind(self.model, "forward", target_forward)
        self.bind(self.draft, "prefill", draft_prefill)
        self.bind(job_cls, "is_checkpoint_boundary", boundary)
        self.bind(job_cls, "maybe_stash_recurrent", stash)
        for block in self.blocks:
            orig_block = block.forward

            def block_fwd(x, params, out_dtype=None, _orig=orig_block, _layer=block.layer_idx):
                rec_box.rec.on_block_in(_layer, x)
                return _orig(x, params, out_dtype)

            self.bind(block, "forward", block_fwd)

    def uninstall(self):
        while self.saved:
            obj, name, orig = self.saved.pop()
            setattr(obj, name, orig)
        self.rec = None
        self.gen = None


def constructed_sample_steps():
    return [
        {"type": "SS_RepP", "rep_p": 1.0, "sustain_range": PENALTY_RANGE, "decay_range": 0, "dropped": True},
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
            "dropped": False,
        },
        {
            "type": "SS_Temperature",
            "temperature": SAMPLE_TEMPERATURE,
            "dropped": False,
            "collapsed_into_fused": True,
        },
        {"type": "SS_Sample", "dropped": False, "collapsed_into_fused": True},
    ]


def make_sampler(kind):
    from exllamav3.generator.sampler.custom import (
        CustomSampler,
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
        # Same class serve_openai._sampler uses. ArgmaxSampler() takes no
        # step list; calling it with one raises TypeError.
        sampler = CustomSampler([
            SS_RepP(1.0, PENALTY_RANGE, 0),
            SS_PresFreqP(0.0, 0.0, PENALTY_RANGE, 0),
            SS_DRY(0.2, 1.75, 6, 4096),
            SS_Temperature(SAMPLE_TEMPERATURE),
            SS_Sample(),
        ])
        seed = SAMPLE_SEED
    else:
        raise RuntimeError(f"unknown sampler {kind}")
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
        if collapsed != [{"type": "SS_Fused", "mode": SS_Fused.MODE_GREEDY, "inv_temp_f32_bits": f32_bits(1.0)}]:
            raise RuntimeError(
                f"greedy stack collapsed to {collapsed}. Leave EXL3_FUSED_SAMPLER at its default."
            )
    else:
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
    return sampler, seed, collapsed


def cq3_layout(cache, name):
    if not cache.layers:
        raise RuntimeError(f"{name} cache has no attention layers")
    layer = next(iter(cache.layers.values()))
    token_dim = int(layer.token_dim)
    return {
        "k_bits": int(layer.k_bits),
        "v_bits": int(layer.v_bits),
        "token_dim": token_dim,
        "page_rows": PAGE_SIZE,
        "qk": {"dtype": "int32", "shape": [PAGE_SIZE, token_dim // 32 * int(layer.k_bits)]},
        "qv": {"dtype": "int32", "shape": [PAGE_SIZE, token_dim // 32 * int(layer.v_bits)]},
        "sk": {"dtype": "float16", "shape": [PAGE_SIZE, token_dim // 32]},
        "sv": {"dtype": "float16", "shape": [PAGE_SIZE, token_dim // 32]},
        "bytes": (
            "one physical page, all 256 rows, C order. qk/qv are the packed int32 "
            "words (token_dim/32*bits per row). sk/sv are the fp16 scales "
            "(token_dim/32 per row). Four separate digests, not concatenated."
        ),
    }


def gdn_layout(cache, num_draft):
    if not cache.recurrent_layers:
        raise RuntimeError("target cache has no GDN layers")
    layer = next(iter(cache.recurrent_layers.values()))
    conv, recurrent = layer.get_state_tensors()
    conv_shape = [int(conv.shape[1]), int(conv.shape[2])]
    recurrent_shape = [int(s) for s in recurrent.shape[1:]]
    if conv_shape != [10240, 4 + num_draft] or recurrent_shape != [num_draft + 1, 48, 128, 128]:
        raise RuntimeError(
            f"N={num_draft} GDN slot shapes conv {conv_shape} recurrent {recurrent_shape}"
        )
    if str(conv.dtype) != "torch.bfloat16" or str(recurrent.dtype) != "torch.float32":
        raise RuntimeError(
            f"N={num_draft} GDN dtypes conv {conv.dtype} recurrent {recurrent.dtype}"
        )
    return {
        "conv": {
            "dtype": "bfloat16",
            "shape": conv_shape,
            "bytes": "slot 0, the whole tensor, including the history tail past conv_k",
        },
        "recurrent": {
            "dtype": "float32",
            "shape": recurrent_shape,
            "bytes": "slot 0, every history index, shape [max_history+1, 48, 128, 128]",
        },
    }


def run_job(gen, hooks, rec, ids, sampler, seed, record):
    from exllamav3 import Job

    if record:
        rec.begin_job(gen.num_draft_tokens, int(ids.shape[-1]))
        hooks.install(rec, gen)
    try:
        gen.enqueue(Job(
            input_ids=ids.clone(),
            max_new_tokens=MAX_NEW,
            min_new_tokens=MAX_NEW,
            sampler=sampler,
            seed=seed,
            stop_conditions=(),
            token_healing=False,
        ))
        while gen.num_remaining_jobs():
            if record:
                rec.begin_iterate()
            results = gen.iterate()
            if record:
                rec.finish_iterate(
                    results,
                    gen,
                    gen.cache,
                    gen.draft_cache,
                    sorted(gen.cache.layers),
                    sorted(gen.cache.get_all_recurrent_layers()),
                    sorted(gen.draft_cache.layers),
                )
    finally:
        if record:
            hooks.uninstall()


def check_case(rec, label, prompt_len, num_draft, chunk_bounds):
    if rec.eos_reason != "max_new_tokens":
        raise RuntimeError(f"{label} N={num_draft} eos_reason {rec.eos_reason}")
    if len(rec.output_tokens) != MAX_NEW:
        raise RuntimeError(
            f"{label} N={num_draft} emitted {len(rec.output_tokens)} tokens, expected {MAX_NEW}"
        )
    if rec.row0 is None:
        raise RuntimeError(f"{label} N={num_draft} recorded no verify row 0")
    if rec.max_rq_tokens != MAX_NEW + 1 + num_draft:
        raise RuntimeError(
            f"{label} N={num_draft} max_rq_tokens {rec.max_rq_tokens}, "
            f"expected {MAX_NEW + 1 + num_draft}"
        )
    got = [(r["prefill"]["start"], r["prefill"]["end"]) for r in rec.rounds if r["prefill"]]
    if label == "A":
        if got:
            raise RuntimeError(f'"A" prefilled {got}')
        if rec.rounds[0]["verify"] is None or rec.rounds[0]["verify"]["index"] != 0:
            raise RuntimeError('"A" first iterate is not verify 0')
    elif got != chunk_bounds(prompt_len):
        raise RuntimeError(f"{label} prefill spans {got}, chunk_bounds {chunk_bounds(prompt_len)}")
    positions = [hit["seq_pos"] for hit in rec.checkpoint_hits]
    if label == "8190":
        for pos in (4096, 6144, 8192):
            if pos not in positions:
                raise RuntimeError(f"8190 N={num_draft} never recorded checkpoint {pos}: {rec.checkpoint_hits}")
        for pos in (4096, 6144):
            if not any(h["seq_pos"] == pos and h["where"] == "stash" for h in rec.checkpoint_hits):
                raise RuntimeError(f"8190 N={num_draft} checkpoint {pos} was not a prefill stash")
    else:
        bad = [p for p in positions if p % 2048 == 0]
        if bad:
            raise RuntimeError(f"{label} crossed a 2048 boundary at {bad}")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Dump one MTP num_draft_tokens reference.")
    parser.add_argument(
        "--num-draft",
        type=int,
        required=True,
        dest="num_draft",
        help="num_draft_tokens for this process, an integer from 2 to 15",
    )
    parser.add_argument(
        "meta_dir",
        nargs="?",
        default=None,
        help="directory for meta_N<N>.json and the testdata copy of the round-0 bins",
    )
    args = parser.parse_args(argv)
    if not 2 <= args.num_draft <= 15:
        parser.error("--num-draft must be an integer from 2 to 15")
    return args


def main():
    args = parse_args(sys.argv[1:])
    num_draft = args.num_draft
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
    meta_dir = Path(args.meta_dir) if args.meta_dir else ROOT / "testdata" / "mtp_ref"
    bulk_dir = ROOT / "target" / "mtp-ref"
    if CACHE_TOKENS % PAGE_SIZE != 0:
        raise SystemExit(f"CACHE_TOKENS {CACHE_TOKENS} is not a multiple of {PAGE_SIZE}")
    # The extension reads this when it is imported. Popping it afterwards does
    # not turn capture back on, and it would hide a caller override.
    if os.environ.get("EXL3_BC_ATTN") == "0":
        raise SystemExit(
            "EXL3_BC_ATTN=0 turns graph capture off. Leave it unset so this dump "
            "matches the serve (BC stays on)."
        )

    # Port check before torch. dump_27b_prefill_reference imports torch.
    wait_for_gpu()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dump_27b_prefill_reference as ref
    import torch
    from exllamav3 import Cache, Config, Generator, Model, Tokenizer
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.generator.sampler.custom import dry_default_sequence_breaker_tokens
    from exllamav3.modules.transformer import TransformerBlock

    if ref.CACHE_TOKENS != CACHE_TOKENS:
        raise SystemExit(
            f"CACHE_TOKENS {CACHE_TOKENS} != dump_27b_prefill_reference.CACHE_TOKENS {ref.CACHE_TOKENS}"
        )

    with torch.inference_mode():
        cfg = Config.from_directory(TARGET)
        if int(cfg.hidden_size) != HIDDEN or int(cfg.vocab_size) != VOCAB:
            raise SystemExit(
                f"config hidden {cfg.hidden_size} vocab {cfg.vocab_size}, expected {HIDDEN} {VOCAB}"
            )
        model = Model.from_config(cfg)
        blocks = [m for m in model.modules if isinstance(m, TransformerBlock)]
        if [b.layer_idx for b in blocks] != list(range(LAYERS)):
            raise SystemExit(f"target blocks are {[b.layer_idx for b in blocks]}")
        cases = []
        bins = {}
        sampler_meta = None

        # One N, one cache. Construct both caches before the matching load().
        target_cache = Cache(
            model,
            max_num_tokens=CACHE_TOKENS,
            layer_type=CacheLayer_quant,
            k_bits=3,
            v_bits=3,
            max_history=num_draft,
            max_batch_size=1,
        )
        model.load(progressbar=False)
        draft = Model.from_config(cfg, component="mtp")
        draft_cache = Cache(
            draft,
            max_num_tokens=CACHE_TOKENS,
            layer_type=CacheLayer_quant,
            k_bits=3,
            v_bits=3,
            max_batch_size=1,
        )
        draft.load(progressbar=False)
        if not target_cache.initialized or not draft_cache.initialized:
            raise SystemExit("load() did not mark the caches initialized")
        tokenizer = Tokenizer.from_config(cfg)
        encoded = ref.prompt_ids(tokenizer)
        full = encoded[8190]
        prompt_tensors = {"8190": full}
        for length in (100, 1500):
            piece = encoded[length]
            if not torch.equal(piece, full[:, :length]):
                raise SystemExit(f"prompt {length} is not a prefix of 8190")
            prompt_tensors[str(length)] = piece
        token_a = tokenizer.encode("A")
        if tuple(token_a.shape) != (1, 1) or int(token_a[0, 0].item()) != 32:
            raise SystemExit(f'"A" encoded as shape {tuple(token_a.shape)} value {token_a.tolist()}')
        prompt_tensors["A"] = token_a
        hooks = Hooks(model, draft, blocks)
        breakers = sorted(int(v) for v in dry_default_sequence_breaker_tokens(tokenizer))

        if int(target_cache.max_history) != num_draft or int(draft_cache.max_history) != 0:
            raise SystemExit(
                f"history target {target_cache.max_history} draft {draft_cache.max_history} "
                f"for N={num_draft}"
            )
        attn_keys = sorted(target_cache.layers)
        gdn_keys = sorted(target_cache.get_all_recurrent_layers())
        draft_keys = sorted(draft_cache.layers)
        if len(attn_keys) != 16 or len(gdn_keys) != 48 or len(draft_keys) != 1:
            raise SystemExit(
                f"layers attn {len(attn_keys)} gdn {len(gdn_keys)} draft {len(draft_keys)}"
            )
        layer_lists = {
            "attn": layer_pairs(attn_keys),
            "gdn": layer_pairs(gdn_keys),
            "draft_attn": layer_pairs(draft_keys),
        }
        layouts_n = {str(num_draft): gdn_layout(target_cache, num_draft)}

        gen = Generator(
            model,
            target_cache,
            tokenizer,
            draft_model=draft,
            draft_cache=draft_cache,
            num_draft_tokens=num_draft,
        )
        if gen.num_draft_tokens != num_draft or gen.dynamic_draft:
            raise SystemExit(
                f"generator drafts {gen.num_draft_tokens} dynamic {gen.dynamic_draft}"
            )
        interval = int(gen.recurrent_checkpoint_interval)
        interval_pp = int(gen.recurrent_checkpoint_interval_pp)
        if interval != 2048 or interval_pp != 32768:
            raise SystemExit(
                f"checkpoint interval {interval} / pp {interval_pp}, expected 2048 / 32768"
            )
        target_layout = cq3_layout(target_cache, "target")
        draft_layout = cq3_layout(draft_cache, "draft")
        if target_layout["qk"]["shape"] != [PAGE_SIZE, 96] or target_layout["sk"]["shape"] != [PAGE_SIZE, 32]:
            raise SystemExit(f"target cq3 layout {target_layout}")
        if draft_layout["qk"] != target_layout["qk"] or draft_layout["sk"] != target_layout["sk"]:
            raise SystemExit(f"draft cq3 layout {draft_layout} != target {target_layout}")

        rec = Recorder()
        for label in PROMPT_ORDER:
            ids = prompt_tensors[label]
            prompt_len = int(ids.shape[-1])
            for kind in SAMPLER_ORDER:
                if sampler_meta is None:
                    sampler_meta = {}
                collapsed_seen = sampler_meta.get(kind, {}).get("collapsed_steps")
                for pass_i in range(WARMUPS + 1):
                    record = pass_i == WARMUPS
                    sampler, seed, collapsed = make_sampler(kind)
                    if collapsed_seen is None:
                        sampler_meta[kind] = {
                            "seed": seed,
                            "collapsed_steps": collapsed,
                        }
                        if kind == "sample":
                            sampler_meta[kind]["temperature"] = SAMPLE_TEMPERATURE
                            sampler_meta[kind]["constructed_steps"] = constructed_sample_steps()
                            sampler_meta[kind]["dry_breaker_token_ids"] = breakers
                        else:
                            sampler_meta[kind]["constructed_steps"] = [
                                {"type": "SS_Argmax", "dropped": False}
                            ]
                        collapsed_seen = collapsed
                    elif collapsed != collapsed_seen:
                        raise SystemExit(f"{kind} stack changed: {collapsed}")
                    print(
                        f"N={num_draft} prompt={label} sampler={kind} "
                        f"{'record' if record else f'warmup {pass_i + 1}'}",
                        file=sys.stderr,
                        flush=True,
                    )
                    reset_pair(gen, target_cache, draft_cache)
                    run_job(gen, hooks, rec, ids, sampler, seed, record)
                check_case(rec, label, prompt_len, num_draft, ref.chunk_bounds)
                filename = f"logits_{label}_N{num_draft}_{kind}_round0.bin"
                bins[filename] = rec.row0
                page_class = "div16" if rec.pages_allocated % 16 == 0 else "other"
                cases.append({
                    "prompt": label,
                    "prompt_len": prompt_len,
                    "num_draft_tokens": num_draft,
                    "sampler": kind,
                    "seed": seed,
                    "pages_allocated": rec.pages_allocated,
                    "page_class": page_class,
                    "max_rq_tokens": rec.max_rq_tokens,
                    "compare_block_in_to_prefill_ref": label in ("100", "1500"),
                    "rounds": rec.rounds,
                    "output_tokens": rec.output_tokens,
                    "eos_reason": rec.eos_reason,
                    "logits_bin": filename,
                    "checkpoint_hits": rec.checkpoint_hits,
                })

        total_bin = sum(len(raw) for raw in bins.values())
        if total_bin > BIN_BUDGET:
            raise SystemExit(f"logit bins are {total_bin} bytes, over the {BIN_BUDGET} budget")
        expected_cases = len(PROMPT_ORDER) * len(SAMPLER_ORDER)
        if len(cases) != expected_cases:
            raise SystemExit(f"recorded {len(cases)} cases, expected {expected_cases}")

        prompts = {
            label: [int(v) for v in prompt_tensors[label][0].tolist()]
            for label in PROMPT_ORDER
        }
        meta = {
            "schema_version": 1,
            "hash": "blake2b-256",
            "byte_order": "little",
            "source": "scripts/dump_27b_mtp_reference.py",
            "model_dir": TARGET,
            "ext_tree": ext_tree,
            "fork_clean": True,
            "env": {name: os.environ.get(name) for name in ENV_FLAGS},
            "num_draft_tokens": num_draft,
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
            "checkpoint_interval": interval,
            "checkpoint_interval_pp": interval_pp,
            "prompt_sentence": ref.SENTENCE,
            "prompt_a": "A",
            "prompts": prompts,
            "layers": layer_lists,
            "layouts": {
                "logit_row": {
                    "dtype": "float16",
                    "shape": [1, 1, VOCAB],
                    "bytes": "one verify row, raw fp16, same payload as the round0 bin",
                },
                "export": {
                    "dtype": "float16",
                    "shape": [1, "rows", HIDDEN],
                    "rows": "prefill chunk length, or verify q_len",
                    "bytes": "params['export_states'][-1] after the target forward; the list has length 1",
                },
                "block_in": {
                    "dtype": "float32",
                    "shape": [1, "rows", HIDDEN],
                    "rows": "prefill chunk length",
                    "bytes": "input x of TransformerBlock.forward, layers 0..63 in order",
                },
                "cq3_page": target_layout,
                "cq3_rows": {
                    "dtype_qk": "int32",
                    "dtype_qv": "int32",
                    "dtype_sk": "float16",
                    "dtype_sv": "float16",
                    "shape": ["length", "packed"],
                    "packed_qk": target_layout["qk"]["shape"][-1],
                    "packed_qv": target_layout["qv"]["shape"][-1],
                    "packed_s": target_layout["sk"]["shape"][-1],
                    "bytes": (
                        "draft only. concatenation of the written rows in position order. "
                        "position p is page phys_of[p//256], row p%256. other rows on the page "
                        "are not included. speculative draft_model.forward rows are not included."
                    ),
                },
                "by_num_draft_tokens": layouts_n,
            },
            "samplers": sampler_meta,
            "cases": cases,
            "logit_bins": {
                "dtype": "float16",
                "shape": [VOCAB],
                "bytes": "raw little-endian fp16, no header. verify index 0, row 0 only",
                "files": [
                    {
                        "file": name,
                        "bytes": len(raw),
                    }
                    for name, raw in bins.items()
                ],
                "total_bytes": total_bin,
            },
            "omissions": [
                "speculative draft_model.forward rows (the anchor at position P and the draft tail refresh does not overwrite) are not hashed",
                "8190 block_in is not comparable to testdata/prefill_ref (33 pages, class other, versus the old 32-page div16 dump)",
                "100 and 1500 block_in[k+1] corresponds to prefill_ref's block-output digest of layer k; this script does not compare them",
            ],
        }
        meta_name = f"meta_N{num_draft}.json"
        meta_dir.mkdir(parents=True, exist_ok=True)
        bulk_dir.mkdir(parents=True, exist_ok=True)
        for name, raw in bins.items():
            (meta_dir / name).write_bytes(raw)
            (bulk_dir / name).write_bytes(raw)
        tmp = meta_dir / f"{meta_name}.tmp"
        tmp.write_text(json.dumps(meta, indent=1) + "\n")
        tmp.replace(meta_dir / meta_name)
        print(
            f"wrote {len(cases)} cases and {len(bins)} logit rows ({total_bin} bytes) "
            f"to {meta_dir / meta_name} and {bulk_dir}",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()

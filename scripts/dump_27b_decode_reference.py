#!/usr/bin/env python3
"""One-token decode reference for the 27B target. Needs the GPU.

Do not run while the :8889 serve is up: this loads the 27B.

Where the pure decode happens
-----------------------------
exllamav3 does not prefill the last prompt token.

- ``Sequence.__init__`` (``generator/pagetable.py``) sets ``kv_position = 0``.
- ``Job.prefill`` (``generator/job.py``) sets
  ``prefill_end = min(chunk end, len(sequence_ids) - 1)`` and, when
  ``prefill_end <= kv_position``, ``continue``s without calling
  ``model.prefill``. A 1-token prompt has length 1, so ``prefill_end`` is 0
  and the prefill forward is skipped. The KV pages stay empty and the job's
  recurrent slot stays the zeroed slot from ``Cache.get_new_state``
  (``GDNState`` with ``clear=True``).
- ``Job.is_prefill_done`` is ``kv_position == len(sequence_ids) - 1``, which
  is already true (0 == 0). ``Generator.iterate`` (``generator/generator.py``)
  still calls ``job.prefill`` (a no-op here) and then ``iterate_gen``, which
  copies ``cache_seqlens`` from ``kv_position`` and calls ``model.forward``
  on that single token. That forward is a decode at position 0.

``Model.forward`` -> ``forward_ls`` (``model/model_ls.py``) walks
``fwd_modules``, which for this checkpoint is ``model.modules``: embedding,
64 ``TransformerBlock``s, the final RMSNorm, and ``lm_head``. Inside
``TransformerBlock.forward`` (``modules/transformer.py``) each block runs
``attn_norm``, the mixer, ``mlp_norm`` (fused residual add), and the MLP.
The block returns only the residual. The mixer output, the MLP output, and
the normed inputs are the values those submodule forwards return; this
script wraps those forwards and clones after they return.

Code path
---------
The serve's decode does not run the Python eager bodies. It enters:

- GDN: ``BC_GatedDeltaNetSplit::run_bszN`` (``libtorch/gated_delta_net.cpp``)
- full attention: ``BC_Attention::run`` (``libtorch/attention.cpp``)
- MLP: ``BC_GatedMLP::run_bszN`` (``libtorch/mlp.cpp``)

Each of those, per (batch, q_len) slot: the first call runs the ``*_gr``
kernels eagerly and arms recording; the second call captures a CUDA graph
and then launches it; every later call only replays the graph. Norms,
residual adds, the embedding lookup, and ``lm_head`` stay direct launches
between those graphs (one token is 64 layers x 2 graphs, not one graph).

This script therefore discards two identical one-token jobs (eager warmup,
then capture) and saves the third, which is the graph replay the serve
uses after warmup. A fourth replay checks that the saved step repeats.
The first job's tensors are compared to the saved step and the result is
recorded; they are not what is written.

Hooks
-----
exllamav3 ``Module`` is not ``torch.nn.Module``, so this wraps ``.forward``
and calls the original. The CUDA graph is captured inside the C++ ``run_*``
during that call. The wrapper clones the returned tensor only after
``run_*`` has returned, does not replace it, and is not installed on the
capture job. Wrapping does not flip ``EXL3_BC_ATTN`` or otherwise select
the Python eager fallback. The script counts BC entries and refuses to
write if a layer took that fallback.

Run with the serve's environment (``scripts/serve-host.sh`` exports)::

    PYTHONPATH=/home/sam/qwen38-exl3-dflash2/exllamav3 \\
    TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \\
    LD_LIBRARY_PATH=<torch nvidia libs>:$LD_LIBRARY_PATH \\
    EXL3_INT8_GEMV=2 EXL3_INT8_GEMV_MAX_K=5 EXL3_QC_STAGING=1 \\
    EXL3_HGEMM_F16ACC=auto \\
    /home/sam/qwen38-exl3-dflash2/env/bin/python \\
        scripts/dump_27b_decode_reference.py [OUT_DIR]
"""

import json
import os
import subprocess
import sys
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
from exllamav3.modules.attention_fn.bc_attn import MAX_BSZ, MAX_QLEN
from exllamav3.modules.mlp import MAX_BSZN

TARGET = os.environ.get(
    "BERTH_27B_DIR", "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3"
)
FORK = Path(os.environ.get("BERTH_EXL3_SRC", "/home/sam/qwen38-exl3-dflash2/exllamav3"))
CACHE_TOKENS = 4096
PROMPT_TEXT = "A"
PROBE_LAYERS = (0, 3)
ENV_FLAGS = [
    "EXL3_INT8_GEMV",
    "EXL3_INT8_GEMV_MAX_K",
    "EXL3_QC_STAGING",
    "EXL3_HGEMM_F16ACC",
]


def git(*args):
    return subprocess.check_output(["git", "-C", str(FORK), *args], text=True).strip()


def serve_listening():
    # Same probe the task requires: `ss -ltn | grep -q ':8889 '`.
    return subprocess.run(
        "ss -ltn | grep -q ':8889 '",
        shell=True,
    ).returncode == 0


def dtype_name(t):
    return str(t.dtype).removeprefix("torch.")


def raw_numpy(t):
    """Little-endian byte view. bfloat16 has no numpy dtype; alias it as int16."""
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy()
    return t.numpy()


def raw_bytes(t):
    return raw_numpy(t).tobytes()


def differing_elements(a, b):
    """Count elements whose raw little-endian bytes differ. None if the tensors do not match."""
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        return None
    ba = raw_bytes(a)
    bb = raw_bytes(b)
    if len(ba) != len(bb):
        return None
    if ba == bb:
        return 0
    es = a.element_size()
    return sum(ba[i:i + es] != bb[i:i + es] for i in range(0, len(ba), es))


def zero_kv(cache):
    # Cache tensors are created under Generator/model inference mode. Inplace
    # updates on them have to stay inside that mode.
    with torch.inference_mode():
        for layer in cache.layers.values():
            for name in ("qk", "qv", "sk", "sv"):
                t = getattr(layer, name, None)
                if t is not None:
                    t.zero_()


def state_nonzero(cache, slot):
    total = None
    for layer in cache.get_all_recurrent_layers().values():
        conv, rec = layer.get_state_tensors()
        n = torch.count_nonzero(conv[slot]) + torch.count_nonzero(rec[slot])
        total = n if total is None else total + n
    if total is None:
        return 0
    return int(total.item())


def kv_nonzero(cache):
    total = None
    for layer in cache.layers.values():
        for name in ("qk", "qv", "sk", "sv"):
            t = getattr(layer, name, None)
            if t is None:
                continue
            n = torch.count_nonzero(t)
            total = n if total is None else total + n
    if total is None:
        return 0
    return int(total.item())


class ActivationCapture:
    """Clone module outputs after the original forward returns.

    The clone is a later kernel on the same stream as the module that just
    ran, including a BC graph launch queued inside that forward. It is not
    part of the captured graph, and the tensor passed downstream is the
    original return value.
    """

    def __init__(self):
        self.saved = []
        self.records = []

    def install(self, specs):
        if self.saved:
            raise RuntimeError("activation capture is already installed")
        self.records = []
        for mod, name, key in specs:
            orig = mod.forward

            def wrapped(x, params=None, *args, _orig=orig, _name=name, _key=key, **kwargs):
                y = _orig(x, params, *args, **kwargs)
                self.records.append((_name, _key, y.detach().clone()))
                return y

            mod.forward = wrapped
            self.saved.append((mod, orig))

    def uninstall(self):
        for mod, orig in self.saved:
            mod.forward = orig
        self.saved = []

    def to_cpu(self):
        torch.cuda.synchronize()
        out = []
        for name, key, t in self.records:
            out.append((name, key, t.detach().cpu()))
            del t
        self.records = []
        return out


def gdn_uses_bc(mod, x, params):
    bsz, seqlen, _ = x.shape
    rsg = params.get("recurrent_states") if params else None
    return bool(
        mod.bc_split
        and mod.bc is not None
        and rsg
        and 1 <= bsz <= MAX_BSZ
        and 1 <= seqlen <= MAX_QLEN
    )


def mlp_uses_bc(mod, x):
    bsz, q_len, _ = x.shape
    return bool(mod.bc is not None and mod.num_slices > 0 and bsz * q_len <= MAX_BSZN)


def install_bc_counters(blocks, counts, entered):
    """Count entries into the BC calls. The C++ object then picks eager, capture, or replay."""
    for block in blocks:
        layer = block.layer_idx
        attn = block.attn
        if type(attn).__name__ == "GatedDeltaNet":
            orig = attn.forward

            def wrapped(x, params=None, *args, _orig=orig, _mod=attn, _layer=layer, **kwargs):
                used = gdn_uses_bc(_mod, x, params)
                y = _orig(x, params, *args, **kwargs)
                counts["gdn"][_layer] += 1
                entered["gdn"][_layer] = used
                return y

            attn.forward = wrapped
        elif type(attn).__name__ == "Attention":
            orig = attn.bc_attn_step

            def wrapped(*args, _orig=orig, _layer=layer, **kwargs):
                y = _orig(*args, **kwargs)
                counts["attn"][_layer] += 1
                entered["attn"][_layer] = y is not None
                return y

            attn.bc_attn_step = wrapped
        else:
            raise RuntimeError(f"layer {layer}: unexpected mixer {type(attn).__name__}")

        mlp = block.mlp
        orig_mlp = mlp.forward

        def wrapped_mlp(x, params=None, *args, _orig=orig_mlp, _mod=mlp, _layer=layer, **kwargs):
            used = mlp_uses_bc(_mod, x)
            y = _orig(x, params, *args, **kwargs)
            counts["mlp"][_layer] += 1
            entered["mlp"][_layer] = used
            return y

        mlp.forward = wrapped_mlp


def capture_specs(modules):
    specs = []
    for mod in modules:
        cls = type(mod).__name__
        if cls == "Embedding":
            specs.append((mod, "embedding", mod.key))
        elif cls == "TransformerBlock":
            i = mod.layer_idx
            if i in PROBE_LAYERS:
                specs.append((mod.attn_norm, f"block_{i:02d}_mixer_in", mod.attn_norm.key))
                specs.append((mod.attn, f"block_{i:02d}_mixer", mod.attn.key))
                specs.append((mod.mlp_norm, f"block_{i:02d}_mlp_in", mod.mlp_norm.key))
                specs.append((mod.mlp, f"block_{i:02d}_mlp", mod.mlp.key))
            specs.append((mod, f"block_{i:02d}", mod.key))
        elif cls == "RMSNorm":
            specs.append((mod, "final_norm", mod.key))
        elif cls == "Linear" and mod.key == "lm_head":
            specs.append((mod, "logits", mod.key))
        else:
            raise RuntimeError(f"unexpected module {cls} key={mod.key}")
    return specs


def expected_names(n_blocks):
    names = ["embedding"]
    for i in range(n_blocks):
        if i in PROBE_LAYERS:
            names += [
                f"block_{i:02d}_mixer_in",
                f"block_{i:02d}_mixer",
                f"block_{i:02d}_mlp_in",
                f"block_{i:02d}_mlp",
            ]
        names.append(f"block_{i:02d}")
    names += ["final_norm", "logits"]
    return names


def assert_modules(model):
    mods = list(model.modules)
    if len(mods) < 4:
        raise RuntimeError(f"expected embedding, blocks, norm, lm_head; got {len(mods)} modules")
    names = [type(m).__name__ for m in mods]
    if names[0] != "Embedding" or names[-1] != "Linear" or names[-2] != "RMSNorm":
        raise RuntimeError(f"module list starts/ends unexpectedly: {names[:2]} ... {names[-2:]}")
    blocks = mods[1:-2]
    if len(blocks) != 64 or any(type(b).__name__ != "TransformerBlock" for b in blocks):
        raise RuntimeError(f"expected 64 TransformerBlocks, got {[type(b).__name__ for b in blocks]}")
    if [b.layer_idx for b in blocks] != list(range(64)):
        raise RuntimeError("block layer_idx is not 0..63 in order")
    if mods[-1].key != "lm_head":
        raise RuntimeError(f"last module is {mods[-1].key}, not lm_head")
    walked = [m for m, _inst, _idx in model.fwd_modules]
    if walked != mods:
        raise RuntimeError("forward_ls fwd_modules is not model.modules")
    if type(blocks[0].attn).__name__ != "GatedDeltaNet":
        raise RuntimeError(f"block 0 mixer is {type(blocks[0].attn).__name__}, expected GatedDeltaNet")
    if type(blocks[3].attn).__name__ != "Attention":
        raise RuntimeError(f"block 3 mixer is {type(blocks[3].attn).__name__}, expected Attention")
    for i in PROBE_LAYERS:
        if blocks[i].attn_norm is None or blocks[i].mlp_norm is None or blocks[i].mlp is None:
            raise RuntimeError(f"block {i} does not expose norm and mlp submodules")
        if type(blocks[i].mlp).__name__ != "GatedMLP":
            raise RuntimeError(f"block {i} mlp is {type(blocks[i].mlp).__name__}")
    return blocks


def watch_forward(model, cache, token, log):
    orig = model.forward

    def wrapped(input_ids, params=None):
        if params is None:
            raise RuntimeError("decode forward was called without generator params")
        if tuple(input_ids.shape) != (1, 1):
            raise RuntimeError(f"expected a single token, got shape {tuple(input_ids.shape)}")
        got = int(input_ids.reshape(-1)[0].item())
        if got != token:
            raise RuntimeError(f"forward token {got} != prompt token {token}")
        seqlens = params["cache_seqlens"].detach().cpu().tolist()
        states = params.get("recurrent_states")
        if not states or len(states) != 1:
            raise RuntimeError(f"expected one recurrent state, got {states!r}")
        state = states[0]
        if state.position != 0:
            raise RuntimeError(f"recurrent position is {state.position}, expected 0")
        positions = params.get("positions")
        pos_list = None if positions is None else positions.detach().cpu().tolist()
        nz_state = state_nonzero(cache, state.slot)
        nz_kv = kv_nonzero(cache)
        log.append({
            "token": got,
            "cache_seqlens": seqlens,
            "recurrent_position": state.position,
            "positions": pos_list,
            "gdn_nonzero": nz_state,
            "kv_nonzero": nz_kv,
        })
        if seqlens != [0] or nz_state != 0 or nz_kv != 0:
            raise RuntimeError(
                f"decode is not empty-cache position 0: seqlens={seqlens} "
                f"gdn_nonzero={nz_state} kv_nonzero={nz_kv}"
            )
        return orig(input_ids, params)

    model.forward = wrapped
    return orig


def run_job(gen, ids, capture, specs):
    prefill_called = {"n": 0}
    orig_prefill = gen.model.prefill

    def bomb(*args, **kwargs):
        prefill_called["n"] += 1
        raise RuntimeError("model.prefill ran; a 1-token prompt must skip it")

    gen.model.prefill = bomb
    stages = []
    try:
        if capture is not None:
            capture.install(specs)
        gen.enqueue(Job(
            input_ids=ids,
            max_new_tokens=1,
            sampler=ArgmaxSampler(),
            token_healing=False,
            seed=0,
        ))
        while gen.num_remaining_jobs():
            for result in gen.iterate():
                stages.append(result.get("stage"))
    finally:
        gen.model.prefill = orig_prefill
        if capture is not None:
            capture.uninstall()
    torch.cuda.synchronize()
    if prefill_called["n"]:
        raise RuntimeError("model.prefill was invoked")
    if "prefill" in stages:
        raise RuntimeError(f"iterate reported a prefill stage: {stages}")
    if stages.count("streaming") != 1:
        raise RuntimeError(f"expected one streaming result, got stages {stages}")
    return stages


def bc_snapshot(counts, blocks):
    """How many times each BC entry has been called, for the layers that have one."""
    out = {"gdn": {}, "attn": {}, "mlp": {}}
    for block in blocks:
        i = block.layer_idx
        if type(block.attn).__name__ == "GatedDeltaNet":
            out["gdn"][i] = counts["gdn"][i]
        else:
            out["attn"][i] = counts["attn"][i]
        out["mlp"][i] = counts["mlp"][i]
    return out


def assert_replay_ready(before, entered, blocks):
    """The saved step must be the third BC entry on every layer, and that entry must be the BC call."""
    for block in blocks:
        i = block.layer_idx
        kind = "gdn" if type(block.attn).__name__ == "GatedDeltaNet" else "attn"
        if before[kind][i] != 2:
            raise RuntimeError(f"layer {i} {kind} had {before[kind][i]} BC calls before the saved step, expected 2")
        if before["mlp"][i] != 2:
            raise RuntimeError(f"layer {i} mlp had {before['mlp'][i]} BC calls before the saved step, expected 2")
        if entered[kind][i] is not True:
            raise RuntimeError(f"layer {i} {kind} did not enter its BC call on the saved step")
        if entered["mlp"][i] is not True:
            raise RuntimeError(f"layer {i} mlp did not enter BC_GatedMLP on the saved step")


def compare_named(a, b):
    am = {n: t for n, _k, t in a}
    bm = {n: t for n, _k, t in b}
    if list(am) != list(bm):
        return [f"name order {list(am)} != {list(bm)}"]
    bad = []
    for name, ta in am.items():
        tb = bm[name]
        if ta.dtype != tb.dtype or tuple(ta.shape) != tuple(tb.shape):
            bad.append(f"{name}: {dtype_name(ta)} {tuple(ta.shape)} vs {dtype_name(tb)} {tuple(tb.shape)}")
            continue
        n = differing_elements(ta, tb)
        if n:
            bad.append(f"{name}: {n} elements differ")
    return bad


def records_finite(records):
    bad = []
    for name, _key, t in records:
        if not torch.isfinite(t).all():
            bad.append(name)
    return bad


def main():
    if sys.byteorder != "little":
        raise SystemExit("this dump writes little-endian raw files")
    if serve_listening():
        raise SystemExit("port 8889 is listening; refusing to use the GPU")

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "testdata/decode_ref")
    out.mkdir(parents=True, exist_ok=True)

    print("loading target (no draft)", file=sys.stderr, flush=True)
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
    blocks = assert_modules(model)
    tokenizer = Tokenizer.from_config(cfg)
    gen = Generator(model, cache, tokenizer)
    # No draft model. Generator.__init__ leaves draft_model unset, so iterate()
    # takes the plain iterate_gen branch.

    ids = tokenizer.encode(PROMPT_TEXT, add_bos=False, add_eos=False)
    if tuple(ids.shape) != (1, 1):
        raise SystemExit(
            f"{PROMPT_TEXT!r} encodes to shape {tuple(ids.shape)}; the prompt must be one token"
        )
    token = int(ids.reshape(-1)[0].item())
    print(f"prompt token {token} ({PROMPT_TEXT!r})", file=sys.stderr)

    counts = {"gdn": [0] * 64, "attn": [0] * 64, "mlp": [0] * 64}
    entered = {"gdn": [None] * 64, "attn": [None] * 64, "mlp": [None] * 64}
    install_bc_counters(blocks, counts, entered)
    specs = capture_specs(model.modules)
    names = expected_names(64)
    if [name for _m, name, _k in specs] != names:
        raise RuntimeError("capture spec order does not match the expected forward order")

    forwards = []
    watch_forward(model, cache, token, forwards)

    capture = ActivationCapture()

    def one(label, save):
        print(f"{label}: start", file=sys.stderr, flush=True)
        zero_kv(cache)
        torch.cuda.synchronize()
        n_fwd = len(forwards)
        run_job(gen, ids, capture if save else None, specs)
        if len(forwards) != n_fwd + 1:
            raise RuntimeError(f"{label}: expected one model.forward, log={forwards}")
        print(
            f"{label}: seqlens={forwards[-1]['cache_seqlens']} "
            f"positions={forwards[-1]['positions']}",
            file=sys.stderr,
            flush=True,
        )
        if not save:
            return None
        got = capture.to_cpu()
        got_names = [name for name, _key, _t in got]
        if got_names != names:
            raise RuntimeError(f"{label}: hook order {got_names} != forward order {names}")
        return got

    # Call 1 is the eager _gr warmup inside each BC object. Saved only to
    # compare against the replay, then dropped.
    eager = one("bc call 1 eager warmup", save=True)
    # Call 2 captures the CUDA graph and launches it. No activation clones
    # are installed, so the capture records only the fork's own kernels.
    one("bc call 2 capture", save=False)
    before_save = bc_snapshot(counts, blocks)
    # entered[] still describes call 2 until call 3 overwrites it.
    saved = one("bc call 3 graph replay", save=True)
    assert_replay_ready(before_save, entered, blocks)
    repeated = one("bc call 4 graph replay", save=True)

    eager_diff = compare_named(eager, saved)
    replay_diff = compare_named(saved, repeated)
    nonfinite = records_finite(saved)
    if nonfinite:
        raise RuntimeError(f"non-finite tensors: {nonfinite}")
    if replay_diff:
        # Still write the files so the mismatch can be inspected, then fail.
        print("replay did not repeat bit-exactly:", file=sys.stderr)
        for line in replay_diff:
            print(f"  {line}", file=sys.stderr)

    tensors = []
    total = 0
    for name, key, t in saved:
        path = out / f"{name}.bin"
        arr = raw_numpy(t)
        arr.tofile(path)
        nbytes = path.stat().st_size
        total += nbytes
        tensors.append({
            "name": name,
            "file": path.name,
            "key": key,
            "dtype": dtype_name(t),
            "shape": list(t.shape),
            "bytes": nbytes,
        })
        print(
            f"{name} {key} {dtype_name(t)} {tuple(t.shape)} {nbytes}",
            file=sys.stderr,
        )

    logits = next(t for name, _key, t in saved if name == "logits")
    if logits.numel() != logits.shape[-1]:
        raise RuntimeError(f"logits are not a single row: {tuple(logits.shape)}")
    argmax = int(logits.reshape(-1).argmax().item())

    # Remove bins from a previous dump that this run no longer writes.
    keep = {item["file"] for item in tensors} | {"meta.json"}
    for stale in out.glob("*"):
        if stale.name not in keep:
            stale.unlink()

    meta = {
        "target": TARGET,
        "fork_commit": git("rev-parse", "HEAD"),
        "ext_tree": git("rev-parse", "HEAD:exllamav3/exllamav3_ext"),
        "torch": torch.__version__,
        "token": token,
        "token_text": PROMPT_TEXT,
        "argmax_token": argmax,
        "names": [item["name"] for item in tensors],
        "tensors": tensors,
        "total_bytes": total,
        "env": {k: os.environ.get(k) for k in ENV_FLAGS},
        "cache": {
            "layer_type": "CacheLayer_quant",
            "k_bits": 3,
            "v_bits": 3,
            "max_batch_size": 1,
            "max_num_tokens": CACHE_TOKENS,
            "draft": False,
        },
        "position": forwards[2]["recurrent_position"],
        "cache_seqlens": forwards[2]["cache_seqlens"],
        "positions": forwards[2]["positions"],
        "path": {
            "saved_step": "BC CUDA graph replay",
            "bc_calls_before_saved_step": 2,
            "mixers": {
                "block_0": "BC_GatedDeltaNetSplit::run_bszN",
                "block_3": "BC_Attention::run",
                "mlp": "BC_GatedMLP::run_bszN",
            },
            "layers_entered_bc": {
                "gdn": sum(1 for i, b in enumerate(blocks) if type(b.attn).__name__ == "GatedDeltaNet" and entered["gdn"][i] is True),
                "attn": sum(1 for i, b in enumerate(blocks) if type(b.attn).__name__ == "Attention" and entered["attn"][i] is True),
                "mlp": sum(1 for i in range(64) if entered["mlp"][i] is True),
            },
            "eager_warmup_bit_identical": not eager_diff,
            "replay_repeat_bit_identical": not replay_diff,
            "eager_warmup_differences": eager_diff,
            "replay_repeat_differences": replay_diff,
            "note": (
                "Calls 1 and 2 of each (bsz=1, q_len=1) slot were discarded "
                "one-token decodes at position 0 (eager _gr warmup, then CUDA "
                "graph capture+launch). The files are call 3, graph replay only. "
                "Embedding, RMSNorm, residual add, and lm_head are direct launches "
                "outside those graphs. Activation clones run after each wrapped "
                "forward returns and are not inside the capture."
            ),
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"wrote {out} ({total} bytes), argmax token {argmax}, "
        f"eager==replay {not eager_diff}, replay repeated {not replay_diff}",
        file=sys.stderr,
    )
    if total > 3 * 1024 * 1024:
        print(
            "total is over 3 MB; logits were stored as produced by lm_head",
            file=sys.stderr,
        )
    if replay_diff:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

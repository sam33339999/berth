#!/usr/bin/env python3
"""300-step greedy decode reference for the 27B target. Needs the GPU.

Do not run while the :8889 serve is up: this loads the 27B (~16 GB). Do not
run it at the same time as a Rust 27B load. The script waits while
``nvidia-smi`` reports 4000 MiB or more, up to 20 minutes, then refuses.

Same setup as ``dump_27b_decode_reference.py``
-----------------------------------------------
One-token prompt ``"A"`` (token 32). exllamav3 does not prefill that token, so
the first ``model.forward`` is a pure decode at position 0 with an empty cache
and a zeroed GDN state. Target only, no draft. ``CacheLayer_quant`` cq3,
``max_batch_size`` 1, 4096 cache tokens. Serve env flags are set before import.

The serve's decode does not run the Python eager bodies. Each ``(bsz=1,
q_len=1)`` slot of ``BC_GatedDeltaNetSplit``, ``BC_Attention`` and
``BC_GatedMLP`` does this (``BC_Attention::run``, ``attention.cpp``):

- call 1: eager ``*_gr`` warmup (``runs == 0``)
- call 2: CUDA graph capture, then launch (``runs`` becomes 2)
- every later call: replay, patching parameters with ``record_param``

This script discards two identical one-token jobs (eager, then capture) and
saves a third job of 300 argmax steps. Those 300 forwards are calls 3..302,
graph replay only. Norms, residual adds, the embedding lookup and ``lm_head``
stay direct launches between the per-layer graphs. Activation clones run after
each wrapped forward returns and are not inside the capture. Hooks are not
installed on the capture job.

What is saved
-------------
Per step, blake2b-256 of the little-endian bytes of ``block_00`` .. ``block_63``
(fp32 ``[1,1,5120]``), ``final_norm`` (fp16 ``[1,1,5120]``) and ``logits``
(fp16 ``[1,1,248320]``). Step 0 must match ``testdata/decode_ref``'s ``.bin``
files. Full logits are written only for steps 0, 1, 255, 256 and 299.

``num_splits`` is ``split_config`` (``attention.cpp``) on the block-table
width actually passed into ``BC_Attention::run``. That width is the pinned
staging buffer ``iterate_gen`` builds, padded up to a multiple of 16 pages
(``generator.py``). It is not the count of pages the sequence has touched.
``page_boundary_positions`` are the cache positions that open a new 256-token
page (``position > 0`` and ``position % 256 == 0``).

Run with the serve's environment (``scripts/serve-host.sh`` exports)::

    PYTHONPATH=/home/sam/qwen38-exl3-dflash2/exllamav3 \\
    TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \\
    LD_LIBRARY_PATH=<torch nvidia libs>:$LD_LIBRARY_PATH \\
    EXL3_INT8_GEMV=2 EXL3_INT8_GEMV_MAX_K=5 EXL3_QC_STAGING=1 \\
    EXL3_HGEMM_F16ACC=auto \\
    /home/sam/qwen38-exl3-dflash2/env/bin/python \\
        scripts/dump_27b_decode_sequence.py [OUT_DIR]
"""

import hashlib
import importlib.util
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

ROOT = Path(__file__).resolve().parent.parent
REF_DIR = ROOT / "testdata" / "decode_ref"
STEPS = 300
LOGIT_STEPS = (0, 1, 255, 256, 299)
PAGE_SIZE = 256
HIDDEN = 5120
VOCAB = 248320
MEM_LIMIT_MIB = 4000
MEM_WAIT_S = 60
MEM_WAIT_BUDGET_S = 20 * 60


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


def load_ref_dump():
    path = Path(__file__).resolve().parent / "dump_27b_decode_reference.py"
    spec = importlib.util.spec_from_file_location("dump_27b_decode_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def blake2b_256(raw):
    return hashlib.blake2b(raw, digest_size=32).hexdigest()


def split_config(bt_width, page_size, q_len, block_n, splits_cap):
    """``split_config`` in ``libtorch/attention.cpp``.

    ``CEIL_DIVIDE`` is ``((x) + (size) - 1) / (size)`` (``util.h``).
    ``bound = bt_width * page_size + q_len``
    ``num_splits = max(1, min(splits_cap, ceil(bound / (4 * block_n))))``
    ``split_len = ceil(ceil(bound / num_splits) / block_n) * block_n``
    """
    if min(bt_width, page_size, q_len, block_n, splits_cap) < 1:
        raise RuntimeError(
            f"split_config arguments must be positive: bt_width={bt_width} "
            f"page_size={page_size} q_len={q_len} block_n={block_n} "
            f"splits_cap={splits_cap}"
        )
    bound = bt_width * page_size + q_len
    denom = 4 * block_n
    num_splits = max(1, min(splits_cap, (bound + denom - 1) // denom))
    per = (bound + num_splits - 1) // num_splits
    split_len = ((per + block_n - 1) // block_n) * block_n
    return num_splits, split_len, bound


def attn_slot_consts(attn):
    """``block_n`` and ``splits_cap`` from ``BCAttn._configure`` (``bc_attn.py``).

    Decode slot is ``bsz=1``, ``q_len=1``. ``programs = bsz * kv_heads * h_blocks``
    with ``block_h = max(16 / block_m, 1)`` and ``h_blocks = cdiv(group, block_h)``.
    ``splits_cap = max(1, min(2 * SMs / programs, 128))``.
    ``block_n = max(16, 8192 / next_power_of_two(head_dim))``.
    """
    import triton

    from exllamav3.modules.attention_fn.bc_attn import _get_sm_count

    hd = int(attn.head_dim)
    hd_pad = triton.next_power_of_2(hd)
    qh = int(attn.num_q_heads)
    kvh = int(attn.num_kv_heads)
    if qh % kvh != 0:
        raise RuntimeError(f"layer {attn.layer_idx}: {qh} q heads not divisible by {kvh} kv heads")
    group = qh // kvh
    q_len = 1
    bsz = 1
    block_n = max(16, 8192 // hd_pad)
    block_m = triton.next_power_of_2(q_len)
    block_h = max(16 // block_m, 1)
    h_blocks = triton.cdiv(group, block_h)
    programs = bsz * kvh * h_blocks
    sm = _get_sm_count(attn.device)
    splits_cap = max(1, min((2 * sm) // programs, 128))
    return {
        "head_dim": hd,
        "hd_pad": hd_pad,
        "num_q_heads": qh,
        "num_kv_heads": kvh,
        "block_n": block_n,
        "programs": programs,
        "sm_count": sm,
        "splits_cap": splits_cap,
        "page_size": PAGE_SIZE,
        "q_len": q_len,
    }


def expected_names():
    return [f"block_{i:02d}" for i in range(64)] + ["final_norm", "logits"]


def capture_specs(modules, names):
    specs = []
    for mod in modules:
        cls = type(mod).__name__
        if cls == "Embedding":
            continue
        if cls == "TransformerBlock":
            specs.append((mod, f"block_{mod.layer_idx:02d}"))
        elif cls == "RMSNorm":
            specs.append((mod, "final_norm"))
        elif cls == "Linear" and mod.key == "lm_head":
            specs.append((mod, "logits"))
        else:
            raise RuntimeError(f"unexpected module {cls} key={getattr(mod, 'key', None)}")
    got = [name for _mod, name in specs]
    if got != names:
        raise RuntimeError(f"capture order {got} != {names}")
    return specs


class ActivationCapture:
    """Clone module outputs after the original forward returns.

    The clone is a later kernel on the same stream. It is not part of the
    captured graph, and the tensor passed downstream is the original return.
    """

    def __init__(self):
        self.saved = []
        self.records = []

    def install(self, specs):
        if self.saved:
            raise RuntimeError("activation capture is already installed")
        self.records = []
        for mod, name in specs:
            orig = mod.forward

            def wrapped(x, params=None, *args, _orig=orig, _name=name, **kwargs):
                y = _orig(x, params, *args, **kwargs)
                self.records.append((_name, y.detach().clone()))
                return y

            mod.forward = wrapped
            self.saved.append((mod, orig))

    def uninstall(self):
        for mod, orig in self.saved:
            mod.forward = orig
        self.saved = []

    def take_cpu(self):
        import torch

        torch.cuda.synchronize()
        out = []
        for name, t in self.records:
            out.append((name, t.detach().cpu()))
            del t
        self.records = []
        return out


def install_bc_counters(blocks, counts, hits, entered, bc_widths, gdn_uses_bc, mlp_uses_bc):
    """Count entries into the BC calls and record the block table each attention sees.

    ``counts`` increments on every call. ``hits`` increments only when that call
    actually entered the BC object, so a Python fallback cannot hide inside the
    total.
    """
    for block in blocks:
        layer = block.layer_idx
        attn = block.attn
        if type(attn).__name__ == "GatedDeltaNet":
            orig = attn.forward

            def wrapped(
                x, params=None, *args,
                _orig=orig, _mod=attn, _layer=layer, _uses=gdn_uses_bc, **kwargs,
            ):
                used = _uses(_mod, x, params)
                y = _orig(x, params, *args, **kwargs)
                counts["gdn"][_layer] += 1
                entered["gdn"][_layer] = used
                if used:
                    hits["gdn"][_layer] += 1
                return y

            attn.forward = wrapped
        elif type(attn).__name__ == "Attention":
            orig = attn.bc_attn_step

            def wrapped(*args, _orig=orig, _layer=layer, **kwargs):
                # Instance attribute: self is already bound in _orig. args[3] is block_table.
                bt = args[3] if len(args) > 3 else kwargs.get("block_table")
                x = args[0] if args else kwargs.get("x")
                if bt is None or x is None:
                    raise RuntimeError(f"layer {_layer}: bc_attn_step args do not carry x and block_table")
                if int(x.shape[0]) != 1 or int(x.shape[1]) != 1:
                    raise RuntimeError(
                        f"layer {_layer}: BC attention q shape {tuple(x.shape)}, expected q_len 1"
                    )
                y = _orig(*args, **kwargs)
                used = y is not None
                counts["attn"][_layer] += 1
                entered["attn"][_layer] = used
                if used:
                    hits["attn"][_layer] += 1
                bc_widths.append((int(_layer), int(bt.shape[1])))
                return y

            attn.bc_attn_step = wrapped
        else:
            raise RuntimeError(f"layer {layer}: unexpected mixer {type(attn).__name__}")

        mlp = block.mlp
        orig_mlp = mlp.forward

        def wrapped_mlp(
            x, params=None, *args,
            _orig=orig_mlp, _mod=mlp, _layer=layer, _uses=mlp_uses_bc, **kwargs,
        ):
            used = _uses(_mod, x)
            y = _orig(x, params, *args, **kwargs)
            counts["mlp"][_layer] += 1
            entered["mlp"][_layer] = used
            if used:
                hits["mlp"][_layer] += 1
            return y

        mlp.forward = wrapped_mlp


def watch_forward(model, cache, ref, log, bc_widths):
    """Record the token, cache length and block-table width of every forward."""
    import torch

    orig = model.forward

    def wrapped(input_ids, params=None):
        if params is None:
            raise RuntimeError("decode forward was called without generator params")
        if tuple(input_ids.shape) != (1, 1):
            raise RuntimeError(f"expected a single token, got shape {tuple(input_ids.shape)}")
        token = int(input_ids.reshape(-1)[0].item())
        seqlens_t = params["cache_seqlens"]
        seqlens = [int(v) for v in seqlens_t.detach().cpu().reshape(-1).tolist()]
        bt = params["block_table"]
        if bt.dim() != 2 or int(bt.shape[0]) != 1:
            raise RuntimeError(f"block_table shape {tuple(bt.shape)}, expected (1, pages)")
        staged_width = int(bt.shape[1])
        positions = params.get("positions")
        pos_list = None if positions is None else [int(v) for v in positions.detach().cpu().reshape(-1).tolist()]
        # Checked before the forward. The forward itself writes K/V and GDN state.
        if log["require_empty"]:
            state = params.get("recurrent_states")
            if not state or len(state) != 1 or state[0].position != 0:
                raise RuntimeError(f"recurrent state is not a fresh slot at position 0: {state!r}")
            nz_state = ref.state_nonzero(cache, state[0].slot)
            nz_kv = ref.kv_nonzero(cache)
            if seqlens != [0] or nz_state != 0 or nz_kv != 0:
                raise RuntimeError(
                    f"decode is not empty-cache position 0: seqlens={seqlens} "
                    f"gdn_nonzero={nz_state} kv_nonzero={nz_kv}"
                )
            if pos_list not in (None, [0]):
                raise RuntimeError(f"position 0 forward has positions={pos_list}")
        n_before = len(bc_widths)
        y = orig(input_ids, params)
        seen = bc_widths[n_before:]
        del bc_widths[n_before:]
        if not seen:
            raise RuntimeError("forward did not enter BC_Attention")
        widths = {w for _layer, w in seen}
        if widths != {staged_width}:
            raise RuntimeError(
                f"BC block_table widths {sorted(widths)} != staged width {staged_width}"
            )
        if len(seen) != 16:
            raise RuntimeError(f"expected 16 full-attention layers, saw {len(seen)}")
        log["rows"].append({
            "token": token,
            "cache_seqlens": seqlens,
            "block_table_width": staged_width,
            "positions": pos_list,
        })
        return y

    model.forward = wrapped
    return orig


def assert_replay_ready(before, entered, blocks):
    for block in blocks:
        i = block.layer_idx
        kind = "gdn" if type(block.attn).__name__ == "GatedDeltaNet" else "attn"
        if before[kind][i] != 2:
            raise RuntimeError(
                f"layer {i} {kind} had {before[kind][i]} BC calls before the saved steps, expected 2"
            )
        if before["mlp"][i] != 2:
            raise RuntimeError(
                f"layer {i} mlp had {before['mlp'][i]} BC calls before the saved steps, expected 2"
            )
        if entered[kind][i] is not True:
            raise RuntimeError(f"layer {i} {kind} did not enter its BC call on the capture step")
        if entered["mlp"][i] is not True:
            raise RuntimeError(f"layer {i} mlp did not enter BC_GatedMLP on the capture step")


def check_step_tensor(name, tensor):
    import torch

    if name.startswith("block_"):
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != (1, 1, HIDDEN):
            raise RuntimeError(f"{name}: {tensor.dtype} {tuple(tensor.shape)}, expected fp32 [1,1,{HIDDEN}]")
    elif name == "final_norm":
        if tensor.dtype != torch.float16 or tuple(tensor.shape) != (1, 1, HIDDEN):
            raise RuntimeError(f"{name}: {tensor.dtype} {tuple(tensor.shape)}, expected fp16 [1,1,{HIDDEN}]")
    elif name == "logits":
        if tensor.dtype != torch.float16 or tuple(tensor.shape) != (1, 1, VOCAB):
            raise RuntimeError(f"{name}: {tensor.dtype} {tuple(tensor.shape)}, expected fp16 [1,1,{VOCAB}]")
    else:
        raise RuntimeError(f"unexpected capture name {name}")
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} is not finite")


def page_boundary_positions(positions, page_size):
    """Cache positions that open a new page.

    ``Job.receive_sample`` (``job.py``) hashes a page when
    ``page_after > page_before``, with ``page_before = kv_position // PAGE_SIZE``
    before the accept and ``page_after`` after ``kv_position += 1``. The next
    forward's ``cache_seqlens`` is that new position. For a step whose
    ``cache_seqlens`` is ``p``, the cross is visible when ``p > 0`` and
    ``p % PAGE_SIZE == 0``.
    """
    return [p for p in positions if p > 0 and p % page_size == 0]


def assert_step0(digests, names):
    if not REF_DIR.is_dir():
        raise RuntimeError(f"missing {REF_DIR}")
    for index, name in enumerate(names):
        path = REF_DIR / f"{name}.bin"
        raw = path.read_bytes()
        expect = blake2b_256(raw)
        got = digests[0][index]
        if got != expect:
            raise RuntimeError(
                f"step 0 {name} digest {got} != blake2b of {path.name} ({expect}), "
                f"file is {len(raw)} bytes"
            )


def main():
    if sys.byteorder != "little":
        raise SystemExit("this dump writes little-endian raw files")
    # Port check before torch / the extension, which can touch the GPU on import.
    if serve_listening():
        raise SystemExit("port 8889 is listening; refusing to use the GPU")

    out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "testdata" / "decode_seq")
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    wait_for_gpu()

    ref = load_ref_dump()
    import torch

    from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.generator.sampler.presets import ArgmaxSampler

    print("loading target (no draft)", file=sys.stderr, flush=True)
    wait_for_gpu()
    cfg = Config.from_directory(ref.TARGET)
    model = Model.from_config(cfg)
    cache = Cache(
        model,
        max_num_tokens=ref.CACHE_TOKENS,
        layer_type=CacheLayer_quant,
        k_bits=3,
        v_bits=3,
        max_batch_size=1,
    )
    model.load(progressbar=False)
    blocks = ref.assert_modules(model)
    tokenizer = Tokenizer.from_config(cfg)
    gen = Generator(model, cache, tokenizer)

    ids = tokenizer.encode(ref.PROMPT_TEXT, add_bos=False, add_eos=False)
    if tuple(ids.shape) != (1, 1):
        raise SystemExit(
            f"{ref.PROMPT_TEXT!r} encodes to shape {tuple(ids.shape)}; the prompt must be one token"
        )
    token = int(ids.reshape(-1)[0].item())
    if token != 32:
        raise SystemExit(f"prompt token is {token}, expected 32")
    print(f"prompt token {token} ({ref.PROMPT_TEXT!r})", file=sys.stderr, flush=True)

    attn_layers = [b for b in blocks if type(b.attn).__name__ == "Attention"]
    if len(attn_layers) != 16:
        raise RuntimeError(f"expected 16 full-attention layers, got {len(attn_layers)}")
    slot = attn_slot_consts(attn_layers[0].attn)
    for block in attn_layers[1:]:
        other = attn_slot_consts(block.attn)
        if other != slot:
            raise RuntimeError(f"layer {block.layer_idx} slot consts {other} != layer {attn_layers[0].layer_idx} {slot}")
    print(
        f"split_config consts block_n={slot['block_n']} splits_cap={slot['splits_cap']} "
        f"programs={slot['programs']} sms={slot['sm_count']} page_size={slot['page_size']}",
        file=sys.stderr,
        flush=True,
    )

    names = expected_names()
    specs = capture_specs(model.modules, names)
    counts = {"gdn": [0] * 64, "attn": [0] * 64, "mlp": [0] * 64}
    hits = {"gdn": [0] * 64, "attn": [0] * 64, "mlp": [0] * 64}
    entered = {"gdn": [None] * 64, "attn": [None] * 64, "mlp": [None] * 64}
    bc_widths = []
    install_bc_counters(
        blocks, counts, hits, entered, bc_widths, ref.gdn_uses_bc, ref.mlp_uses_bc,
    )
    log = {"rows": [], "require_empty": True}
    watch_forward(model, cache, ref, log, bc_widths)

    def one_token(label):
        print(f"{label}: start", file=sys.stderr, flush=True)
        ref.zero_kv(cache)
        torch.cuda.synchronize()
        n_fwd = len(log["rows"])
        ref.run_job(gen, ids, None, [])
        if len(log["rows"]) != n_fwd + 1:
            raise RuntimeError(f"{label}: expected one model.forward, got {len(log['rows']) - n_fwd}")
        row = log["rows"][-1]
        print(
            f"{label}: token={row['token']} seqlens={row['cache_seqlens']} "
            f"bt_width={row['block_table_width']}",
            file=sys.stderr,
            flush=True,
        )

    # Call 1 eager warmup, call 2 capture+launch. Both discarded. Hooks stay off
    # so the capture records only the fork's own kernels.
    log["require_empty"] = True
    one_token("bc call 1 eager warmup")
    one_token("bc call 2 capture")
    before_save = ref.bc_snapshot(counts, blocks)
    assert_replay_ready(before_save, entered, blocks)
    for block in blocks:
        i = block.layer_idx
        kind = "gdn" if type(block.attn).__name__ == "GatedDeltaNet" else "attn"
        if hits[kind][i] != 2 or hits["mlp"][i] != 2:
            raise RuntimeError(
                f"layer {i} entered BC on {hits[kind][i]} mixer calls and "
                f"{hits['mlp'][i]} mlp calls during warmup, expected 2 and 2"
            )

    capture = ActivationCapture()
    digests = []
    argmax = []
    input_tokens = []
    num_splits = []
    split_lens = []
    positions = []
    bt_widths = []
    logit_raw = {}

    print(f"bc calls 3..{2 + STEPS}: {STEPS}-step graph replay", file=sys.stderr, flush=True)
    ref.zero_kv(cache)
    torch.cuda.synchronize()
    log["require_empty"] = True
    prefill_called = {"n": 0}
    orig_prefill = gen.model.prefill

    def bomb(*args, **kwargs):
        prefill_called["n"] += 1
        raise RuntimeError("model.prefill ran; a 1-token prompt must skip it")

    gen.model.prefill = bomb
    capture.install(specs)
    try:
        gen.enqueue(Job(
            input_ids=ids,
            max_new_tokens=STEPS,
            min_new_tokens=STEPS,
            sampler=ArgmaxSampler(),
            stop_conditions=(),
            token_healing=False,
            seed=0,
        ))
        while gen.num_remaining_jobs():
            got_stream = False
            stages = []
            for result in gen.iterate():
                stage = result.get("stage")
                stages.append(stage)
                if stage == "prefill":
                    raise RuntimeError("iterate reported a prefill stage")
                if stage != "streaming":
                    if stage != "started":
                        raise RuntimeError(f"unexpected iterate stage {stage}")
                    continue
                if got_stream:
                    raise RuntimeError(f"two streaming results in one iterate: {stages}")
                got_stream = True
                step = len(digests)
                cpu = capture.take_cpu()
                got_names = [name for name, _t in cpu]
                if got_names != names:
                    raise RuntimeError(f"step {step}: hook order {got_names} != {names}")
                if len(log["rows"]) != 2 + step + 1:
                    raise RuntimeError(
                        f"step {step}: forward log has {len(log['rows'])} rows, expected {2 + step + 1}"
                    )
                row = log["rows"][-1]
                if row["cache_seqlens"] != [step]:
                    raise RuntimeError(f"step {step}: cache_seqlens {row['cache_seqlens']}")
                if row["positions"] not in (None, [step]):
                    raise RuntimeError(f"step {step}: positions {row['positions']}")
                if row["token"] != (token if step == 0 else argmax[-1]):
                    raise RuntimeError(
                        f"step {step}: fed token {row['token']} != previous argmax"
                    )
                ns, sl, _bound = split_config(
                    row["block_table_width"],
                    slot["page_size"],
                    slot["q_len"],
                    slot["block_n"],
                    slot["splits_cap"],
                )
                step_digests = []
                for name, tensor in cpu:
                    check_step_tensor(name, tensor)
                    raw = ref.raw_numpy(tensor).tobytes()
                    step_digests.append(blake2b_256(raw))
                    if name == "logits":
                        if int(tensor.reshape(-1).argmax().item()) < 0:
                            raise RuntimeError("logits argmax underflow")
                        argmax.append(int(tensor.reshape(-1).argmax().item()))
                        if step in LOGIT_STEPS:
                            logit_raw[step] = raw
                if len(argmax) != step + 1:
                    raise RuntimeError(f"step {step}: logits hook missing")
                digests.append(step_digests)
                input_tokens.append(row["token"])
                num_splits.append(ns)
                split_lens.append(sl)
                positions.append(step)
                bt_widths.append(row["block_table_width"])
                if step == 0:
                    # The next forward is position 1 and must not be required to see an empty cache.
                    log["require_empty"] = False
                if step in LOGIT_STEPS or step % 50 == 0 or step + 1 == STEPS:
                    print(
                        f"step {step}: token {row['token']} argmax {argmax[-1]} "
                        f"bt_width {row['block_table_width']} num_splits {ns} split_len {sl}",
                        file=sys.stderr,
                        flush=True,
                    )
            if not got_stream:
                raise RuntimeError(f"iterate produced no token: {stages}")
    finally:
        gen.model.prefill = orig_prefill
        capture.uninstall()

    if prefill_called["n"]:
        raise RuntimeError("model.prefill was invoked")
    if len(digests) != STEPS:
        raise RuntimeError(f"saved {len(digests)} steps, expected {STEPS}")
    if input_tokens[0] != 32 or input_tokens != [token, *argmax[:-1]]:
        raise RuntimeError("input_tokens are not prompt token then the previous argmax")
    for i in range(STEPS - 1):
        if input_tokens[i + 1] != argmax[i]:
            raise RuntimeError(f"input_tokens[{i + 1}]={input_tokens[i + 1]} != argmax[{i}]={argmax[i]}")

    # Every saved forward entered the BC call. Counters include the two warmup calls.
    for block in blocks:
        i = block.layer_idx
        kind = "gdn" if type(block.attn).__name__ == "GatedDeltaNet" else "attn"
        if counts[kind][i] != 2 + STEPS or hits[kind][i] != 2 + STEPS:
            raise RuntimeError(
                f"layer {i} {kind} BC calls {counts[kind][i]}, entries {hits[kind][i]}, "
                f"expected {2 + STEPS} of each"
            )
        if counts["mlp"][i] != 2 + STEPS or hits["mlp"][i] != 2 + STEPS:
            raise RuntimeError(
                f"layer {i} mlp BC calls {counts['mlp'][i]}, entries {hits['mlp'][i]}, "
                f"expected {2 + STEPS} of each"
            )
        if entered[kind][i] is not True or entered["mlp"][i] is not True:
            raise RuntimeError(f"layer {i} left the BC path on the last saved step")

    assert_step0(digests, names)
    print("step 0 digests match testdata/decode_ref", file=sys.stderr, flush=True)

    boundaries = page_boundary_positions(positions, PAGE_SIZE)
    changes = [positions[i] for i in range(1, STEPS) if num_splits[i] != num_splits[i - 1]]
    len_changes = [positions[i] for i in range(1, STEPS) if split_lens[i] != split_lens[i - 1]]
    width_changes = [positions[i] for i in range(1, STEPS) if bt_widths[i] != bt_widths[i - 1]]

    for step, raw in logit_raw.items():
        path = out / f"logits_step_{step:03d}.bin"
        path.write_bytes(raw)
        if path.stat().st_size != len(raw):
            raise RuntimeError(f"{path.name} size {path.stat().st_size} != {len(raw)}")
    if step0_logits := logit_raw.get(0):
        ref_logits = (REF_DIR / "logits.bin").read_bytes()
        if step0_logits != ref_logits:
            raise RuntimeError("logits_step_000.bin bytes differ from testdata/decode_ref/logits.bin")

    keep = {f"logits_step_{step:03d}.bin" for step in LOGIT_STEPS} | {"meta.json"}
    for stale in out.glob("*"):
        if stale.name not in keep:
            stale.unlink()

    entered_counts = {
        "gdn": sum(
            1 for b in blocks
            if type(b.attn).__name__ == "GatedDeltaNet" and entered["gdn"][b.layer_idx] is True
        ),
        "attn": sum(
            1 for b in blocks
            if type(b.attn).__name__ == "Attention" and entered["attn"][b.layer_idx] is True
        ),
        "mlp": sum(1 for i in range(64) if entered["mlp"][i] is True),
    }
    unique_widths = sorted(set(bt_widths))
    unique_splits = sorted(set(num_splits))
    unique_lens = sorted(set(split_lens))
    bound0 = bt_widths[0] * slot["page_size"] + slot["q_len"]
    meta = {
        "prompt_token": token,
        "steps": STEPS,
        "input_tokens": input_tokens,
        "argmax": argmax,
        "hash": "blake2b-256",
        "names": names,
        "digests": digests,
        "num_splits": num_splits,
        "page_boundary_positions": boundaries,
        "target": ref.TARGET,
        "fork_commit": ref.git("rev-parse", "HEAD"),
        "ext_tree": ref.git("rev-parse", "HEAD:exllamav3/exllamav3_ext"),
        "torch": torch.__version__,
        "env": {k: os.environ.get(k) for k in ref.ENV_FLAGS},
        "cache": {
            "layer_type": "CacheLayer_quant",
            "k_bits": 3,
            "v_bits": 3,
            "max_batch_size": 1,
            "max_num_tokens": ref.CACHE_TOKENS,
            "draft": False,
        },
        "path": {
            "saved_steps": "BC CUDA graph replay",
            "bc_calls_before_saved_steps": 2,
            "saved_call_range": [3, 2 + STEPS],
            "mixers": {
                "block_0": "BC_GatedDeltaNetSplit::run_bszN",
                "block_3": "BC_Attention::run",
                "mlp": "BC_GatedMLP::run_bszN",
            },
            "layers_entered_bc": entered_counts,
            "split_config": {
                "source": "exllamav3/exllamav3_ext/libtorch/attention.cpp:388",
                "replay_patch": "exllamav3/exllamav3_ext/libtorch/attention.cpp:998",
                "page_size": slot["page_size"],
                "q_len": slot["q_len"],
                "block_n": slot["block_n"],
                "splits_cap": slot["splits_cap"],
                "programs": slot["programs"],
                "sm_count": slot["sm_count"],
                "block_table_width": unique_widths if len(unique_widths) > 1 else unique_widths[0],
                "block_table_width_change_positions": width_changes,
                "bound_at_step_0": bound0,
                "num_splits_unique": unique_splits,
                "split_len_unique": unique_lens,
                "split_len_change_positions": len_changes,
            },
            "num_splits_change_positions": changes,
            "note": (
                "Calls 1 and 2 of each (bsz=1, q_len=1) slot were discarded "
                "one-token decodes at position 0 (eager _gr warmup, then CUDA "
                "graph capture+launch). The 300 saved steps are calls 3..302, "
                "graph replay only. Embedding, RMSNorm, residual add, and lm_head "
                "are direct launches outside those graphs. Activation clones run "
                "after each wrapped forward returns and are not inside the capture. "
                "num_splits is split_config(block_table.size(1), page_size=256, "
                "q_len=1, block_n, splits_cap) with block_n and splits_cap from "
                "BCAttn._configure. block_table.size(1) is the width iterate_gen "
                "stages, padded up to a multiple of 16 pages (generator.py). "
                "page_boundary_positions are cache positions that open a new "
                "256-token page (position > 0 and position % 256 == 0), the "
                "cache_seqlens of the first forward after Job.receive_sample "
                "sees page_after > page_before. num_splits_change_positions lists "
                "those cache positions where num_splits differs from the previous step."
            ),
        },
    }
    text = json.dumps(meta, separators=(",", ":")) + "\n"
    (out / "meta.json").write_text(text)
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(
        f"wrote {out} ({total} bytes), steps {STEPS}, "
        f"num_splits {unique_splits} changes at {changes or 'none'}, "
        f"page boundaries {boundaries}, bt_width {unique_widths}",
        file=sys.stderr,
        flush=True,
    )
    if total > 3 * 1024 * 1024:
        print(
            f"testdata/decode_seq is {total} bytes, over 3 MiB. "
            "Five fp16 logits rows are 2483200 bytes and 300x66 blake2b-256 "
            "hex digests are about 1.27 MB; both are required by the format.",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()

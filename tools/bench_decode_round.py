"""Decode rounds of a real-shape synthetic Qwen3.8-Flash-Next on one CUDA GPU (production MultiDecoder rounds).

How to run (dev-01: through ~/tensorfold-bw/bin/gpu-run.sh, which holds the shared GPU lock, runs the pinned
verification image on a copy of the worktree source and saves logs and host-side nvidia-smi samples in the run dir;
B = python3 tools/bench_decode_round.py):

  gpu-run.sh base-n1-150k $B --contexts 150000 --out /runs/base-n1-150k/bench.json
  gpu-run.sh base-n4 $B --contexts 30000,30000,150000,150000 --out /runs/base-n4/bench.json
  # nsys over 10 rounds only (cudaProfilerStart/Stop bracket --profile-rounds), graph kernels as nodes:
  gpu-run.sh nsys-n1 /opt/nsys/bin/nsys profile -t cuda,nvtx -s none --cpuctxsw none --cuda-graph-trace node \
      --capture-range cudaProfilerApi --capture-range-end stop -o /runs/nsys-n1/trace \
      $B --contexts 150000 --rounds 0 --profile-rounds 10 --out /runs/nsys-n1/bench.json
  # then on the host: /opt/nvidia/nsight-systems/2026.3.2/bin/nsys stats --report cuda_gpu_kern_sum,cuda_gpu_trace \
  #     --format csv -o /home/kklouzal/tensorfold-bw/runs/nsys-n1/stats .../nsys-n1/trace.nsys-rep
  # ncu on the profiled rounds' kernels (graph kernel nodes profiled one by one):
  gpu-run.sh ncu-n1 /opt/ncu/ncu --profile-from-start off --graph-profiling node --launch-count 30 \
      --metrics gpu__time_duration.sum,dram__bytes.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum \
      -o /runs/ncu-n1/prof $B --contexts 150000 --rounds 0 --profile-rounds 1 --out /runs/ncu-n1/bench.json

Defaults are production's decode settings (GB10 compose: --parallel 4, --mtp-drafts 4, --kv-dtype rotorquant6-norm,
--yarn-factor 4 with a 1,048,576-token window, --decode-share 0.5, --ple-on-ssd with 32 native read threads,
TENSORFOLD_PREFILL_ROWS 2048, captured rounds within TENSORFOLD_MULTI_GRAPH_MIB 512, PDL by chip policy) except:
greedy sampling, copy drafts off, confidence 0 (every chain runs to --depth, so rows/round = streams x (depth+1)) and
a fixed accepted path (``--accept``, below); no vision tower (production's ``--vision`` serves image requests
only). ``--help`` lists every knob; the JSON records all of them, the environment's TENSORFOLD_/CUDA_/PYTORCH_/
TORCH_/TRITON_/NVIDIA_/BENCH_ variables and the runtime versions.

What is real: every kernel, launch geometry, packing and code path of a decode round. The weights are random MLX
affine 4-bit/group-32 arrays at the production dimensions, assembled by ``flashnext_synthetic.build`` exactly as
``weights.load`` assembles a checkpoint (hidden 2560, 512 routed experts top-10 + the shared expert in one grouped
table, expert width 640, vocab 248,320 with the packaged 79,591-id draft head, heads 24 / KV 2 / head_dim 256, partial
rotary 64 under the production YaRN-4 policy, DeltaNet nk 16 / nv 48 / dk = dv = 128 / conv 4, 4 hyper-connection
streams of low rank 320, indexer 4 x 128 with budget 2048 and ratio 4, the MTP head with its own MoE layer). The
decoder is ``multi.MultiDecoder`` built as ``engine.FlashNextEngine`` builds it for ``--parallel 4`` (4 slots, KEEP 8,
the RQ6 pair, prefill rows, share), warmed and graph-captured by its own ``warm()``; rounds are its ``round()``.
Each stream's state is built by the real prompt path (``admit``, then the decoder's prompt passes, ``_fill``, over a
random-token prompt of the stream's context, all of them before any decode round, so every stream starts the rounds
at exactly its context): the KV/indexer caches, pooled keys, DeltaNet/conv states and MTP cache hold what that path
computes. A stream whose prompt ends samples its first token and drafts as in production. (Production runs a filling
prompt's passes inside the live streams' rounds; those mixed rounds are not measured here.)

What is synthetic or not represented:
- Layers: default one real period, 3 DeltaNet + 1 full-attention layer (``--layers LLLA``), not 48 (36 + 12): the
  real model does not fit dev-01. Per-round costs of the real model are ~12 x the per-period layer kernels (nsys
  per-kernel sums) plus the once-per-forward parts (embedding, final mixer, LM head; the MTP step and draft head);
  the per-layer split comes from the kernel trace (graph replays keep no NVTX inside), or from ``--nvtx-layers``
  eager runs (``--graph-mib 0``).
- PLE: the real model's one n-gram layer (``ple_layer_ids [2]``: zero-indexed layer 1, a DeltaNet layer) is
  included, its MLX 4-bit table read from a local file at each lookup through the production reader (``open_table``
  with ``ssd=True``, native reader, read-ahead, late reads during the forward). The table is synthetic and smaller
  (``--ngram-base`` 500,000 -> 8.0M rows, 0.8 GB, page-cache resident after first use; production: 20M base, 320M
  rows, 32 GB on NVMe), so host read latency is optimistic; the GPU work (16 rows of 160 values per token) is real.
- MTP experts: by default (``--mtp-experts shared``) the MTP layer reads the last decoder layer's expert table, not a
  table of its own (one 1.47 GiB table less: the default plan then fits the ~10.3 GiB the card has free beside the
  embedding server). The bytes each MTP step reads are the same; between the two layers' reads the LM head
  (397 MB) or the draft head (127 MB) streams through the 32 MiB L2, so the sharing cannot turn DRAM reads into
  cache hits. ``--mtp-experts own`` gives it its own table where memory allows (e.g. smaller --experts).
- Routing: random router rows over random hidden states pick experts nearly uniformly, so a window's rows share
  fewer experts than production's (~89% of the picks unique at n=1, telemetry): expert bytes per round are
  over-estimated. ``--routing-rounds`` (default 20) records it in every report: untimed eager rounds after the
  measured ones (the same kernels and bits as the replays) copy each MoE layer's picks and count unique routed
  experts per verify layer and MTP step. ``--router-skew S`` (default 0: off) scales each routed expert's router
  row by exp(S z), z ~ N(0, 1), so popular experts win more rows and windows share experts. Measured on dev-01
  (2026-10-14; unique routed experts / picks in the verify forward, uniform expectation 0.962 at 5 rows, 0.834 at
  20): S = 0: 0.949 (n=1, 5 rows), 0.806 (n=4, 20 rows); S = 0.3: 0.891, 0.641; S = 0.6: 0.795, 0.473. S = 0.3
  matches production's n=1 figure; production's n=4 overlap is unmeasured.
- Acceptance: random weights accept almost no drafts (the real rule keeps 1 row a round). ``--accept N`` (default 3,
  close to production's n=1 mean of ~3.0-3.2 accepted rows a round) keeps the first N rows of each window (the
  committed path, the DeltaNet rows folded next round and the MTP absorb all have N rows, as in production); a
  window shorter than N (a chain ``--confidence`` stopped early) keeps all its rows: each sample's ``committed``
  counts the rows kept. ``--no-mtp`` windows hold one row (``--accept 1`` or ``model``). ``--accept model`` uses
  the real rule. Either way the emitted tokens are a deterministic function of the arithmetic (``tokens_sha256`` in
  the JSON: equal across bit-identical builds).
- Sampling: greedy (production samples at temperature 1, top-k 20, top-p 0.95: same verify/draft kernels, the
  sampler's host packets differ slightly). Copy drafts off (production TENSORFOLD_MTP_COPY=1 proposes repeated text
  instead of MTP rows on repetitive contexts; random prompts would never match).
- Memory admission: the decoder's memory gate is replaced by an unbounded one (production's gate keeps 2 GiB + the
  startup reserve free, which a 16 GiB card shared with the embedding server cannot); the tool enforces its own
  ``--vram-budget-gib`` instead and prints the byte plan first. No kernel depends on the gate.
- Prompt resume points: none (``points=None``; the chat-template markers never occur in random prompts).

Measurement: ``--warmup`` rounds, then ``--rounds`` timed rounds: CUDA events around each ``round()`` (the round's
GPU span, host gaps included) and wall time, events around the verify forward and the MTP chain, NVTX ranges per
round, verify forward ("verify[graph]"/"verify[eager]"), MTP chain and each MTP step; per-round rows, MTP steps,
graph replays vs eager forwards. The phase events and NVTX ranges are recorded inside the round's span (their host
time is in ``round_ms``; every run of this tool carries the same ones). ``--profile-rounds K`` runs K more rounds
between cudaProfilerStart/Stop for ``nsys --capture-range cudaProfilerApi`` or ``ncu --profile-from-start off``
(their timings are listed apart); the routing rounds come last. Every stream decodes in every round: a stream that
ends, fails or waits for memory stops the run, and so does a cache resize during the rounds (all are grown first).
``tokens_sha256`` hashes every stream's tokens up to the end of the measured rounds: equal for equal settings on
bit-identical builds (cross-run determinism on one build and GPU; checked 2026-10-14).
Both expert kernel forms run as in production: the verify window's buffers use the experts' prompt-form kernels
(``moe_prefill=True``), the MTP steps' the decode-form ones.

ncu on sm_120 (GB206): ``dram__bytes_read.sum`` is not available and unknown ``--metrics`` names are dropped
silently; use ``dram__bytes.sum``, ``dram__bytes_op_read.sum`` / ``dram__bytes_op_write.sum``. ncu 2026.3 writes
its report as ``.ncu-repz``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

TOOL = "bench_decode_round"
VERSION = 1
# the environment a report records: the decoder's, the runtimes' and the wrapper's settings (no secrets live there)
ENV_PREFIXES = ("TENSORFOLD_", "CUDA_", "PYTORCH_", "TORCH_", "TRITON_", "NVIDIA_", "BENCH_")


def _args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--contexts", default="150000",
                   help="each stream's prompt length, comma separated (1 to --slots streams)")
    p.add_argument("--layers", default="LLLA", help="layer pattern: L = DeltaNet (linear), A = full attention")
    p.add_argument("--experts", type=int, default=512, help="routed experts per MoE layer (plus the shared one)")
    p.add_argument("--slots", type=int, default=4, help="decoder slots (production --parallel 4)")
    p.add_argument("--depth", type=int, default=4, help="MTP drafts a round (production --mtp-drafts 4)")
    p.add_argument("--confidence", type=float, default=0.0,
                   help="MTP chain stop probability (0: every chain runs to --depth; production 0.5)")
    p.add_argument("--accept", default="3",
                   help="rows each window keeps: N (fixed accepted path) or 'model' (the real accept rule)")
    p.add_argument("--kv-dtype", default="rotorquant6-norm")
    p.add_argument("--graph-mib", type=int, default=512,
                   help="captured-round budget (TENSORFOLD_MULTI_GRAPH_MIB); 0 = eager rounds")
    p.add_argument("--pdl", choices=("chip", "on", "off"), default="chip",
                   help="programmatic dependent launch: chip policy (on for GB10 only), forced on, forced off")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--profile-rounds", type=int, default=0,
                   help="rounds run between cudaProfilerStart/Stop after the timed rounds")
    p.add_argument("--prefill-rows", type=int, default=2048, help="prompt pass rows (TENSORFOLD_PREFILL_ROWS)")
    p.add_argument("--capacity", type=int, default=1048576, help="each slot's window (production 1,048,576)")
    p.add_argument("--yarn-factor", type=float, default=4.0, help="static YaRN factor (0: native RoPE)")
    p.add_argument("--decode-share", type=float, default=0.5)
    p.add_argument("--no-mtp", action="store_true", help="no MTP head (one row a round, serial)")
    p.add_argument("--no-ple", action="store_true", help="no n-gram layer")
    p.add_argument("--ple-in-memory", action="store_true",
                   help="map the n-gram table in host memory (production reads it from SSD: --ple-on-ssd)")
    p.add_argument("--ssd-threads", default="32", help="TENSORFOLD_SSD_NATIVE_THREADS ('' = the Python reader)")
    p.add_argument("--ngram-base", type=int, default=500_000,
                   help="n-gram vocabulary base (table rows ~16x this; production 20,000,000)")
    p.add_argument("--draft-vocab", default="default", help="'default' (packaged ids), 'none', or a count")
    p.add_argument("--mtp-experts", choices=("shared", "own"), default="shared",
                   help="the MTP layer's expert table: the last layer's (saves 1.47 GiB; docstring) or its own")
    p.add_argument("--router-skew", type=float, default=0.0,
                   help="lognormal sigma of the routed experts' router-row scales (0: off; docstring table)")
    p.add_argument("--routing-rounds", type=int, default=20,
                   help="untimed eager rounds after the measured ones that record each MoE layer's routed experts")
    p.add_argument("--nvtx-layers", action="store_true",
                   help="NVTX range per decoder layer (eager forwards only: graph replays carry none)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--config", default="", help="the real model's config.json (default: the embedded copy)")
    p.add_argument("--data-dir", default=os.environ.get("BENCH_DATA_DIR", "/data"),
                   help="where the synthetic n-gram table is made once and reused")
    p.add_argument("--vram-budget-gib", type=float, default=10.5,
                   help="refuse a plan above this; report measured use against it")
    p.add_argument("--out", default="", help="JSON report path")
    a = p.parse_args(argv)
    for name, low in (("experts", 1), ("slots", 1), ("depth", 0), ("warmup", 0), ("rounds", 0), ("profile_rounds", 0),
                      ("prefill_rows", 1), ("capacity", 1), ("graph_mib", 0), ("ngram_base", 1),
                      ("routing_rounds", 0)):
        if getattr(a, name) < low:
            p.error(f"--{name.replace('_', '-')} must be at least {low}")
    for name, value in (("confidence", a.confidence), ("decode-share", a.decode_share),
                        ("yarn-factor", a.yarn_factor), ("router-skew", a.router_skew)):
        if not value >= 0 or (name == "confidence" and value > 1):      # NaN fails too
            p.error(f"--{name} {value} is out of range")
    if not a.vram_budget_gib > 0:
        p.error("--vram-budget-gib must be positive")
    if a.accept != "model" and not a.accept.isdecimal():
        p.error(f"--accept: a row count or 'model', not {a.accept!r}")
    return a


def _pct(values, q):
    import numpy as np

    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else None


def _summary(values) -> dict:
    if not values:
        return {}
    return {"n": len(values), "median": _pct(values, 50), "p10": _pct(values, 10), "p90": _pct(values, 90),
            "mean": statistics.fmean(values), "min": min(values), "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}


def _source_sha() -> dict:
    """The tensorfold package this process imports (sha256 over its .py/.cu/.cpp/.cuh/.c/.h/.txt files; each binary
    in it, e.g. the wrapper's _fd_owner overlay, apart), this tool's own two files, and the git sha the wrapper
    recorded on the host (BENCH_GIT_SHA; the container holds a copy without .git)."""

    import tensorfold

    root = Path(tensorfold.__file__).parent
    h = hashlib.sha256()
    n = 0
    binaries = {}
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix in (".py", ".cu", ".cpp", ".cuh", ".c", ".h", ".txt"):
            h.update(str(f.relative_to(root)).encode())
            h.update(f.read_bytes())
            n += 1
        elif f.is_file() and f.suffix == ".so":
            binaries[str(f.relative_to(root))] = hashlib.sha256(f.read_bytes()).hexdigest()
    here = Path(__file__).resolve().parent
    tools = {name: hashlib.sha256((here / name).read_bytes()).hexdigest()
             for name in ("bench_decode_round.py", "flashnext_synthetic.py")}
    return {"git": os.environ.get("BENCH_GIT_SHA", "unknown"), "git_dirty": os.environ.get("BENCH_GIT_DIRTY", ""),
            "package_root": str(root), "package_files": n, "package_sha256": h.hexdigest(),
            "package_binaries_sha256": binaries, "tools_sha256": tools}


def _gpu(torch) -> dict:
    props = torch.cuda.get_device_properties(0)
    out = {"name": props.name, "capability": list(torch.cuda.get_device_capability(0)),
           "sms": props.multi_processor_count, "total_bytes": props.total_memory,
           "l2_bytes": getattr(props, "L2_cache_size", None), "torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        import triton

        out["triton"] = triton.__version__
    except Exception as exc:                           # noqa: BLE001  (provenance only: recorded, not fatal)
        out["triton"] = f"unavailable: {type(exc).__name__}"
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            hnd = pynvml.nvmlDeviceGetHandleByIndex(0)     # NVML's index 0 (NVML ignores CUDA_VISIBLE_DEVICES)
            out["nvml_device"] = pynvml.nvmlDeviceGetName(hnd)         # ... so its name shows which device it read
            out["clocks_mhz"] = {k: pynvml.nvmlDeviceGetClockInfo(hnd, v) for k, v in
                                 (("sm", pynvml.NVML_CLOCK_SM), ("mem", pynvml.NVML_CLOCK_MEM))}
            out["max_clocks_mhz"] = {k: pynvml.nvmlDeviceGetMaxClockInfo(hnd, v) for k, v in
                                     (("sm", pynvml.NVML_CLOCK_SM), ("mem", pynvml.NVML_CLOCK_MEM))}
            out["driver"] = pynvml.nvmlSystemGetDriverVersion()
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:                           # noqa: BLE001  (the wrapper samples nvidia-smi on the host)
        out["nvml"] = f"unavailable: {type(exc).__name__}"
    return out


class Probe:
    """Per-round counters, CUDA events and NVTX ranges around the decoder's verify forward, MTP chain and MTP steps,
    installed as wrappers on the decoder instance and on ``multi.compute`` (no source edits). Events, counters and
    these ranges are recorded only while ``on`` (``--nvtx-layers``' per-layer ranges: in every eager forward from
    here on, the routing rounds' included); the wrappers never change arguments or results."""

    def __init__(self, dec, multi_mod, torch, nvtx_layers: bool) -> None:
        self.on = False
        self.count: Counter = Counter()
        self.marks: dict[str, list] = {}
        nvtx = torch.cuda.nvtx

        def event():
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e

        def span(name, fn):
            def wrapped(*a, **kw):
                if not self.on:
                    return fn(*a, **kw)
                nvtx.range_push(name)
                start = event()
                try:
                    return fn(*a, **kw)
                finally:
                    self.marks.setdefault(name, []).append((start, event()))
                    nvtx.range_pop()
            return wrapped

        g = dec.graphs
        if g is not None:
            real_verify, real_step = g.verify_forward, g.mtp_step

            def verify_forward(segs):
                if not self.on:
                    return real_verify(segs)
                nvtx.range_push("verify[graph]")
                start = event()
                out = real_verify(segs)
                if out is not None:
                    self.marks.setdefault("verify", []).append((start, event()))
                    self.count["verify_replays"] += 1
                nvtx.range_pop()
                return out

            def mtp_step(segs):
                out = real_step(segs)
                if self.on:
                    self.count["mtp_replays" if out is not None else "mtp_graph_misses"] += 1
                return out

            g.verify_forward, g.mtp_step = verify_forward, mtp_step
        real_compute = multi_mod.compute

        def compute(w, segs, b, *a, **kw):
            if not self.on or b is not dec.buf:
                return real_compute(w, segs, b, *a, **kw)
            self.count["verify_eager"] += 1
            nvtx.range_push("verify[eager]")
            start = event()
            try:
                return real_compute(w, segs, b, *a, **kw)
            finally:
                self.marks.setdefault("verify", []).append((start, event()))
                nvtx.range_pop()

        multi_mod.compute = compute
        timed_mtp = span("mtp_step", dec._mtp)

        def mtp(segs):
            if self.on:
                self.count["mtp_steps"] += 1
                self.count["mtp_rows"] += segs[-1][2]
            return timed_mtp(segs)

        dec._mtp = mtp
        dec._draft_all = span("mtp_chain", dec._draft_all)
        if nvtx_layers:
            from tensorfold.families.qwen4_exp.cuda import forward as fwd, mtp as mtp_mod

            real_layer = fwd.layer_forward

            def layer_forward(layer, *a, **kw):
                nvtx.range_push(f"layer {layer.index}")
                try:
                    return real_layer(layer, *a, **kw)
                finally:
                    nvtx.range_pop()

            fwd.layer_forward = mtp_mod.layer_forward = layer_forward

    def begin(self) -> None:
        self.count.clear()
        self.marks.clear()
        self.on = True

    def end(self) -> dict:
        """The round's counters and phase milliseconds (call after the round's end event completed)."""

        self.on = False
        out = dict(self.count)
        for name, key in (("verify", "verify_ms"), ("mtp_chain", "chain_ms"), ("mtp_step", "mtp_step_ms")):
            pairs = self.marks.get(name, [])
            out[key] = sum(a.elapsed_time(b) for a, b in pairs) if pairs else 0.0
        return out


def _reader(layer) -> str:
    """The n-gram reader's class and, for a read-ahead wrapper, its table's (the native and Python SSD readers share
    the wrapper class)."""

    table = layer.ple.table
    inner = getattr(table, "table", None)
    return type(table).__name__ + (f"({type(inner).__name__})" if inner is not None else "")


def _check_streams(streams, done, where: str) -> None:
    """Every stream decodes in every round this tool runs: one that ended (failed or not) or waits for memory stops
    the run (its error chained), as its absence would silently change the measured rounds."""

    if done:
        error = next((s.error for s in done if s.error is not None), None)
        raise RuntimeError(f"{len(done)} stream(s) ended {where}") from error
    waiting = [s.sid for s in streams if s.waiting]
    if waiting:
        raise RuntimeError(f"stream(s) {waiting} wait for memory {where}")


def _routing(dec, cfg, torch, streams, count: int, log) -> dict:
    """``count`` more rounds, eager (graphs detached: the same kernels and bits as their replays) and untimed, recording
    every MoE layer's routed picks (``moe.pick[:R, :top_k]``, copied after each ``forward.moe_block``): unique routed
    experts per layer of each verify forward and per MTP step (step 0 of a round is the absorb), against the
    uniform-routing expectation E (1 - (1 - k/E)^R)."""

    from tensorfold.families.qwen4_exp.cuda import forward as fwd

    real = fwd.moe_block
    seen: list = []
    E, K = cfg.experts, cfg.top_k

    def moe_block(layer, w, b, R):
        out = real(layer, w, b, R)
        if b is not dec.buf and b is not dec.mbuf:                  # rounds without a prompt pass use no other
            raise RuntimeError("a routing round ran an MoE layer on a buffer other than the verify/MTP windows'")
        seen.append(("verify" if b is dec.buf else "mtp", layer.index, R, b.moe.pick[:R, :K].clone()))
        return out

    graphs, dec.graphs = dec.graphs, None
    fwd.moe_block = moe_block
    per_round = []
    try:
        for i in range(count):
            seen.clear()
            _check_streams(streams, dec.round(), f"in routing round {i}")
            torch.cuda.synchronize()
            rec = {"verify": [], "mtp": []}
            for kind, index, R, pick in seen:
                rec[kind].append([index, R, int(torch.unique(pick).numel())])
            per_round.append(rec)
    finally:
        fwd.moe_block = real
        dec.graphs = graphs

    def expect(R):
        return E * (1 - (1 - K / E) ** R)

    def fraction(entries):
        return [u / (R * K) for _, R, u in entries]

    verify = [e for r in per_round for e in r["verify"]]
    absorb = [r["mtp"][0] for r in per_round if r["mtp"]]
    chain = [e for r in per_round for e in r["mtp"][1:]]
    layers = sorted({e[0] for e in verify})
    out = {"rounds": count, "experts": E, "top_k": K, "note": "unique routed experts (shared expert excluded); "
           "fraction = unique / (rows x top_k); uniform = the expectation under uniform independent routing",
           "verify": {"rows": _summary([e[1] for e in verify]), "unique": _summary([e[2] for e in verify]),
                      "fraction": _summary(fraction(verify)),
                      "uniform_fraction": (statistics.fmean(expect(e[1]) / (e[1] * K) for e in verify)
                                           if verify else None),
                      "per_layer_unique_mean": {str(li): statistics.fmean(e[2] for e in verify if e[0] == li)
                                                for li in layers}},
           "mtp_absorb": {"rows": _summary([e[1] for e in absorb]), "unique": _summary([e[2] for e in absorb]),
                          "fraction": _summary(fraction(absorb))},
           "mtp_chain": {"rows": _summary([e[1] for e in chain]), "unique": _summary([e[2] for e in chain]),
                         "fraction": _summary(fraction(chain))},
           "samples": per_round}
    v = out["verify"]
    if verify:
        log(f"[bench] routing ({count} eager rounds): verify {v['unique']['mean']:.1f} unique routed experts a layer "
            f"of {v['rows']['mean']:.1f} rows x {K} ({v['fraction']['mean']:.3f}; uniform "
            f"{v['uniform_fraction']:.3f}); "
            f"MTP absorb {out['mtp_absorb']['unique'].get('mean', 0):.1f} of "
            f"{out['mtp_absorb']['rows'].get('mean', 0):.1f} rows, chain step "
            f"{out['mtp_chain']['unique'].get('mean', 0):.1f} of {out['mtp_chain']['rows'].get('mean', 0):.1f} rows")
    return out


def main(argv=None) -> int:
    args = _args(argv)
    parts = [x.strip() for x in args.contexts.split(",") if x.strip()]
    if not all(x.isdecimal() and int(x) >= 1 for x in parts) or not 1 <= len(parts) <= args.slots:
        raise SystemExit(f"--contexts: 1 to --slots ({args.slots}) positive prompt lengths, not {args.contexts!r}")
    contexts = [int(x) for x in parts]
    accept = None if args.accept == "model" else int(args.accept)
    rows_max = (args.depth if not args.no_mtp else 0) + 1                # a window's rows
    if accept is not None and not 1 <= accept <= rows_max:
        raise SystemExit(f"--accept: 1 to {rows_max} rows (depth + 1; 1 with --no-mtp), or 'model'")
    # production's environment, read by the decoder and the n-gram reader at construction
    os.environ["TENSORFOLD_MULTI_GRAPH_MIB"] = str(args.graph_mib)
    os.environ["TENSORFOLD_MTP_COPY"] = "0"
    os.environ["TENSORFOLD_PREFILL_ROWS"] = str(args.prefill_rows)
    if args.ssd_threads:
        os.environ["TENSORFOLD_SSD_NATIVE_THREADS"] = args.ssd_threads
    else:
        os.environ.pop("TENSORFOLD_SSD_NATIVE_THREADS", None)

    import numpy as np
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import flashnext_synthetic as synth

    from tensorfold.cuda import pdl
    from tensorfold.cuda.memory_gate import MemoryGate
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda import multi as multi_mod
    from tensorfold.families.qwen4_exp.cuda.engine import KEEP
    from tensorfold.families.qwen4_exp.kv_formats import get_pair

    t_start = time.perf_counter()
    log = lambda *a: print(*a, flush=True)                                     # noqa: E731
    report: dict = {"tool": TOOL, "version": VERSION, "argv": sys.argv if argv is None else [sys.argv[0], *argv],
                    "args": vars(args), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "host": platform.node(), "source": _source_sha(),
                    "runtime": {"python": platform.python_version(), "numpy": np.__version__},
                    "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)}}
    torch.cuda.init()
    free0, total = torch.cuda.mem_get_info()
    report["gpu"] = _gpu(torch)
    log(f"[bench] {report['gpu']['name']} sm_{''.join(map(str, report['gpu']['capability']))}, "
        f"{report['gpu']['sms']} SMs; {free0 / 2**30:.2f} of {total / 2**30:.2f} GiB free at init")

    # -- config and byte plan --------------------------------------------------------------------------------------
    real = json.loads(Path(args.config).read_text()) if args.config else synth.REAL_CONFIG
    if args.config:   # the embedded copy must be the real one (text config)
        given, embedded = synth.text(real), synth.text(synth.REAL_CONFIG)
        mismatch = sorted(k for k in set(given) | set(embedded) if given.get(k) != embedded.get(k))
        report["config_check"] = {"file": args.config, "text_config_keys_differing_from_embedded": mismatch,
                                  "sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest()}
        log(f"[bench] --config text_config vs embedded copy: {mismatch or 'identical'}")
    raw = synth.synthetic_config(real, args.layers, args.experts, args.ngram_base, not args.no_ple)
    with tempfile.TemporaryDirectory(prefix="flashnext-synth-") as cdir:   # read now: Config keeps no path
        cfg, rope = synth.read_config(synth.write_config(Path(cdir), raw), args.yarn_factor or None)
    mtp = not args.no_mtp
    draft = None if args.draft_vocab == "none" else (int(args.draft_vocab) if args.draft_vocab.isdecimal()
                                                     else args.draft_vocab)
    from tensorfold.families.qwen4_exp.cuda.weights import draft_token_ids

    draft_rows = head_rows = 0          # the draft logits' rows; the rows of a draft head of their own
    if mtp:
        ids = draft_token_ids(draft, cfg.vocab)
        draft_rows = cfg.vocab if ids is None else int(ids.size)
        head_rows = 0 if ids is None else draft_rows        # no draft ids: the drafts read the LM head itself
    wplan = synth.plan_weight_bytes(cfg, mtp=mtp, draft_rows=head_rows,
                                    share_mtp_experts=args.mtp_experts == "shared")
    # rows a stream may take in every round this tool runs (the routing rounds' too), grown before the first one
    extra_rows = (args.warmup + args.rounds + args.profile_rounds + args.routing_rounds + 4) * (args.depth + 2) + 64
    step = multi_mod.STEP                               # caches grow by this many rows; an idle slot keeps FIRST
    splan = [synth.plan_state_bytes(cfg, args.kv_dtype, -(-(n + extra_rows) // step) * step, mtp=mtp)
             for n in contexts]
    idle = synth.plan_state_bytes(cfg, args.kv_dtype, multi_mod.FIRST, mtp=mtp) * (args.slots - len(contexts))
    # buffers: decode/MTP windows (small), the prompt pass (its rows' activations and its attention block scores
    # over the window), measured after construction; this is the estimate used for the refusal only
    pass_rows = args.prefill_rows + args.slots * (args.depth + 1)
    # (+480 MiB: the rest, calibrated on dev-01: 1.334 GiB measured at 2,048 prefill rows and a 1M window)
    buffers = int(pass_rows * cfg.streams * cfg.hidden * 2 * 14 + min(pass_rows, 256) * (args.capacity // 4) * 4
                  + pass_rows * (cfg.top_k + 1) * cfg.moe_width * 2 * 3 + 480 * 2**20)
    graph_cap = args.graph_mib << 20
    context_overhead = 600 * 2**20                     # CUDA context, cuBLAS/Triton modules (measured below)
    plan_total = wplan["total"] + sum(splan) + idle + buffers + graph_cap + context_overhead
    report["plan_bytes"] = {"weights": wplan, "stream_states": splan, "idle_slots": idle, "buffers_estimate": buffers,
                            "graph_budget": graph_cap, "context_overhead_estimate": context_overhead,
                            "total": plan_total, "budget": int(args.vram_budget_gib * 2**30)}
    log("[bench] byte plan (GiB): " + ", ".join(f"{k} {v / 2**30:.3f}" for k, v in wplan.items()))
    log(f"[bench]   stream states {[round(s / 2**30, 3) for s in splan]} + idle slots {idle / 2**30:.3f}, buffers "
        f"~{buffers / 2**30:.3f}, graphs <= {graph_cap / 2**30:.3f}, context ~{context_overhead / 2**30:.2f}: "
        f"total {plan_total / 2**30:.3f} of budget {args.vram_budget_gib:.2f} (free at init {free0 / 2**30:.2f})")
    report["plan_bytes"]["fits_free_at_init"] = plan_total - context_overhead <= free0
    if plan_total > args.vram_budget_gib * 2**30 or plan_total - context_overhead > free0:
        raise SystemExit("the plan exceeds --vram-budget-gib or the free memory: try fewer --experts, smaller "
                         "--prefill-rows or contexts, or --mtp-experts shared")

    # -- model -------------------------------------------------------------------------------------------------------
    synth.check_chunked_pack()
    data_dir = Path(args.data_dir)
    built = synth.build(cfg, rope, seed=args.seed, mtp=mtp, draft_vocab=draft,
                        share_mtp_experts=args.mtp_experts == "shared", data_dir=data_dir if cfg.ple_layers else None,
                        ple_on_ssd=not args.ple_in_memory, router_skew=args.router_skew, log=log)
    w = built.w
    mem = {"weights_nbytes": w.nbytes(), "after_weights_allocated": torch.cuda.memory_allocated(),
           "after_weights_free": torch.cuda.mem_get_info()[0]}
    report["model"] = {"layers": args.layers, "layer_types": cfg.layer_types, "ple_layers": cfg.ple_layers,
                       "experts": cfg.experts, "top_k": cfg.top_k, "hidden": cfg.hidden, "vocab": cfg.vocab,
                       "moe_width": cfg.moe_width, "heads": cfg.heads, "kv_heads": cfg.kv_heads,
                       "head_dim": cfg.head_dim, "rotary_dim": cfg.rotary_dim, "rope": cfg.rope.metadata(),
                       "gdn": [cfg.nk, cfg.nv, cfg.dk, cfg.dv, cfg.conv_kernel], "hc": [cfg.streams, cfg.low],
                       "indexer": [cfg.index_heads, cfg.index_dim, cfg.index_budget, cfg.index_ratio],
                       "mtp": mtp, "draft_rows": draft_rows, "mtp_experts": args.mtp_experts,
                       "router_skew": args.router_skew,
                       "ngram": ({"rows": cfg.ngram(0).rows, "dims": cfg.ngram(0).dims, "shards": cfg.ngram_shards,
                                  "table": str(built.table_path), "table_record": built.table_record,
                                  "on_ssd": not args.ple_in_memory, "reader": _reader(w.layers[cfg.ple_layers[0]])}
                                 if cfg.ple_layers else None),
                       "build_s": round(built.seconds, 2)}

    # -- the decoder, as FlashNextEngine builds it for --parallel > 1 ------------------------------------------------
    real_select = pdl.select
    if args.pdl != "chip":
        pdl.select = (lambda device=0: True) if args.pdl == "on" else (lambda device=0: False)
    try:
        dec = multi_mod.MultiDecoder(w, slots=args.slots, capacity=args.capacity, depth=args.depth if mtp else 0,
                                     confidence=args.confidence, keep=KEEP, points=None,
                                     kv_pair=get_pair(args.kv_dtype), share=args.decode_share,
                                     prefill_rows=args.prefill_rows, workspace_bytes=0, graphs=args.graph_mib > 0)
    finally:
        pdl.select = real_select
    dec.memory_gate = MemoryGate(1 << 62, reserve=0, live=None)     # admission is this tool's budget (docstring)
    if accept is not None:
        def fixed_accept(tokens, parents, sampled, room, eos=()):
            """The window's first k rows (a chain, as round() builds it) and the last one's sample. End tokens are
            never looked for: the measured streams take none (stop_eos=False, ``eos`` empty) and warm()'s request
            (which does) keeps one row (room 1), where ``accept`` returns the same."""

            k = max(1, min(accept, len(tokens), room))
            if any(parents[r] != r - 1 for r in range(1, k)):
                raise RuntimeError("--accept N needs chain windows (row r's parent is r - 1)")
            if eos and k > 1:
                raise RuntimeError("--accept N: a stream that stops at end tokens takes the real rule")
            return list(range(k)), sampled[k - 1]

        multi_mod.accept = fixed_accept
    mem["after_decoder_allocated"] = torch.cuda.memory_allocated()
    mem["decoder_buffers"] = mem["after_decoder_allocated"] - mem["after_weights_allocated"]
    t0 = time.perf_counter()
    dec.warm()                                        # a synthetic request through prefill and a round; captures
    warm_s = time.perf_counter() - t0
    torch.cuda.synchronize()
    mem["after_warm_allocated"] = torch.cuda.memory_allocated()
    mem["after_warm_reserved"] = torch.cuda.memory_reserved()
    mem["after_warm_free"] = torch.cuda.mem_get_info()[0]
    report["decoder"] = {"pdl": bool(dec.pdl), "pdl_policy": args.pdl, "graph_budget": dec.graph_budget,
                         "graphs": {k: v for k, v in dec.graph_record.items() if k != "keys"},
                         "graph_keys": dec.graph_record.get("keys"), "warm_s": round(warm_s, 2),
                         "converged": bool(dec.converged), "rotor_lookup_policy": [dec.rotor_decoder_policy.serial,
                                                                                   dec.rotor_decoder_policy.mtp],
                         "slots": args.slots, "capacity": args.capacity, "kv_pair": [dec.free[0].kv_key_dtype,
                                                                                     dec.free[0].kv_value_dtype]}
    rec = dec.graph_record
    log(f"[bench] decoder warm {warm_s:.1f}s; pdl {dec.pdl}; graphs: " +
        (f"{rec['verify_keys']} verify + {rec['mtp_keys']} MTP keys, {rec['bytes'] / 2**20:.0f} MiB "
         f"(room {rec['room_bytes'] / 2**20:.0f} MiB), {rec['capture_s']:.1f}s, stopped: {rec['stopped']}"
         if rec.get("captured") else f"none ({rec.get('reason')})"))

    # -- streams at their contexts through the real prompt path ------------------------------------------------------
    eos = set(cfg.eos) | {cfg.ple_eos}
    streams = []
    for i, n in enumerate(contexts):
        rng = np.random.default_rng(args.seed + 1000 + i)
        prompt = rng.integers(0, 248000, n).tolist()      # ordinary (non-special) ids
        prompt = [t if t not in eos else t + 1 for t in prompt]
        streams.append(Stream(prompt, 1 << 40, None, draft=mtp, stop_eos=False))
    t0 = time.perf_counter()
    for s in streams:
        dec.admit(s)
    joined: dict[int, float] = {}
    passes = Counter()
    real_pass = dec._pass

    def counted_pass():
        passes["passes"] += 1
        return real_pass()

    dec._pass = counted_pass
    while dec.filling:          # prompt passes only (no decode round between them: each stream joins at its context)
        _check_streams(streams, dec._fill(), "while filling")
        for s in streams:
            if s.sid in dec.streams and s.sid not in joined:
                joined[s.sid] = time.perf_counter() - t0
    del dec._pass
    torch.cuda.synchronize()
    fill_s = time.perf_counter() - t0
    if any(dec.streams.get(s.sid) is not s or s.done for s in streams):
        raise RuntimeError("the prompt passes ended without every stream decoding")
    for s in streams:                                   # room for every round this tool runs: no resize in them
        need = max(s.st.pos, s.st.mtp_len) + extra_rows
        if not dec._grow(s.st, need) or s.st.capacity < need:
            raise RuntimeError(f"stream {s.sid}: no room for {need} cache rows (its window holds {s.st.limit}): "
                               "fewer rounds or a larger --capacity")
    grown = [s.st.capacity for s in streams]
    torch.cuda.synchronize()
    mem["after_fill_allocated"] = torch.cuda.memory_allocated()
    mem["after_fill_reserved"] = torch.cuda.memory_reserved()
    mem["after_fill_free"] = torch.cuda.mem_get_info()[0]
    report["streams"] = [{"context": n, "prompt_tokens": len(s.prompt), "joined_s": round(joined[s.sid], 2),
                          "prefill_pass_s": round(s.prefill_s, 2), "pos_after_fill": s.st.pos,
                          "capacity_rows": s.st.capacity, "cache_bytes": s.st.cache_bytes()}
                         for n, s in zip(contexts, streams, strict=True)]
    report["fill"] = {"seconds": round(fill_s, 2), "passes": passes["passes"],
                      "tokens_per_s": round(sum(contexts) / fill_s, 1)}
    log(f"[bench] prompts filled through the prompt path in {fill_s:.1f}s ({sum(contexts) / fill_s:.0f} tok/s, "
        f"{passes['passes']} passes); joined at " + ", ".join(f"{joined[s.sid]:.1f}s" for s in streams))

    # -- rounds ------------------------------------------------------------------------------------------------------
    probe = Probe(dec, multi_mod, torch, args.nvtx_layers)
    nvtx = torch.cuda.nvtx

    def rounds(count: int, label: str) -> list[dict]:
        samples = []
        for i in range(count):
            live = [s for s in streams if not s.done and not s.waiting]
            rows = sum(1 + len(s.drafts) for s in live)
            out0 = sum(len(s.out) for s in streams)
            pos0 = [s.st.pos for s in streams]
            probe.begin()
            nvtx.range_push(f"{label} round {i}")
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            w0 = time.perf_counter()
            done = dec.round()
            e1.record()
            e1.synchronize()
            w1 = time.perf_counter()
            nvtx.range_pop()
            got = probe.end()
            _check_streams(streams, done, f"in {label} round {i}")
            got.update(round_ms=e0.elapsed_time(e1), wall_ms=(w1 - w0) * 1e3, rows=rows, streams=len(live),
                       tokens=sum(len(s.out) for s in streams) - out0,
                       committed=sum(s.st.pos for s in streams) - sum(pos0))
            got["other_ms"] = got["round_ms"] - got["verify_ms"] - got["chain_ms"]
            samples.append(got)
        return samples

    t0 = time.perf_counter()
    warm = rounds(args.warmup, "warmup")
    timed = rounds(args.rounds, "timed")
    profiled = []
    if args.profile_rounds:
        cudart = torch.cuda.cudart()
        torch.cuda.synchronize()
        cudart.cudaProfilerStart()
        try:
            profiled = rounds(args.profile_rounds, "profiled")
        finally:
            torch.cuda.synchronize()
            cudart.cudaProfilerStop()
    rounds_s = time.perf_counter() - t0
    tok = hashlib.sha256(json.dumps([s.out for s in streams]).encode()).hexdigest()
    report["tokens_sha256"] = tok                       # the measured rounds' tokens (before the routing rounds)
    report["tokens_out"] = [len(s.out) for s in streams]
    report["routing"] = _routing(dec, cfg, torch, streams, args.routing_rounds, log) if args.routing_rounds else None
    if [s.st.capacity for s in streams] != grown:
        raise RuntimeError(f"stream caches resized during the rounds: {grown} -> {[s.st.capacity for s in streams]}")
    mem["peak_allocated"] = torch.cuda.max_memory_allocated()
    mem["end_free"] = torch.cuda.mem_get_info()[0]
    mem["process_device_bytes_estimate"] = free0 - mem["end_free"]
    mem["within_budget"] = mem["process_device_bytes_estimate"] <= report["plan_bytes"]["budget"]
    mem["note"] = ("process_device_bytes_estimate = free at init - free at end: this process's allocations after "
                   "CUDA init if no other process's use changed meanwhile (its context's own bytes come before init's "
                   "reading; the host pmon log has fb MB per process); peak_allocated: torch's peak over the whole "
                   "process (model build and prompt passes included)")
    report["memory"] = mem

    def stats(samples: list[dict]) -> dict:
        keys = ("round_ms", "wall_ms", "verify_ms", "chain_ms", "other_ms", "mtp_step_ms", "rows", "tokens",
                "mtp_steps", "mtp_rows", "verify_replays", "verify_eager", "mtp_replays", "mtp_graph_misses")
        return {k: _summary([float(s.get(k, 0)) for s in samples]) for k in keys}

    report["rounds"] = {"warmup": len(warm), "timed": len(timed), "profiled": len(profiled),
                        "seconds": round(rounds_s, 2), "summary": stats(timed),
                        "profiled_summary": stats(profiled) if profiled else None,
                        "samples": timed, "warmup_samples": warm, "profiled_samples": profiled}
    report["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    if timed:
        st = report["rounds"]["summary"]
        log(f"[bench] {len(timed)} timed rounds: round {st['round_ms']['median']:.3f} ms median "
            f"(p10 {st['round_ms']['p10']:.3f}, p90 {st['round_ms']['p90']:.3f}); wall {st['wall_ms']['median']:.3f}; "
            f"verify {st['verify_ms']['median']:.3f}, MTP chain {st['chain_ms']['median']:.3f}; rows/round "
            f"{st['rows']['mean']:.2f}, MTP steps/round {st['mtp_steps']['mean']:.2f}, verify replays "
            f"{st['verify_replays']['mean']:.2f}, eager {st['verify_eager']['mean']:.2f}; MTP replays "
            f"{st['mtp_replays']['mean']:.2f} (misses {st['mtp_graph_misses']['mean']:.2f}); tokens/round "
            f"{st['tokens']['mean']:.2f}")
    log(f"[bench] memory: weights {mem['weights_nbytes'] / 2**30:.3f} GiB, peak allocated "
        f"{mem['peak_allocated'] / 2**30:.3f} GiB, process ~{mem['process_device_bytes_estimate'] / 2**30:.3f} GiB "
        f"after init (--vram-budget-gib {args.vram_budget_gib:.2f}{'' if mem['within_budget'] else ': EXCEEDED'}); "
        f"tokens sha256 {tok[:16]}")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(report, indent=1))
        os.replace(tmp, out)
        log(f"[bench] wrote {out}")
    w.meta["ple_tables"].close(torch.cuda.synchronize)
    return 0


if __name__ == "__main__":
    sys.exit(main())

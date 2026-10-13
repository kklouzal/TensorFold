"""Time Flash Next's QSA block selection: today's one-CTA-a-row kernels ("old": ``_select`` within its registers,
``_select_tiles`` past them) against the multi-CTA select ("T", ``attention._select_multi``) and the launcher's policy
("policy"), and, for several streams, each stream's own select ("old") against one select over every stream's rows
("new": the policy with per-row positions). With --unit each call scores the blocks first (several streams: each
stream's own ``_scores`` against one ``attn_multi._scores_multi``), so the select reads them from L2 as in decode.
Every rep runs every implementation interleaved, from the same scores, and checks their whole id, length and flag
buffers are equal (any difference aborts).

Per (blocks, rows, streams, pattern) it records, per implementation and rep:
  eager   CUDA-event time of one eager call from an idle GPU (device time plus any exposed launch time)
  host    host time to enqueue the call (no sync)
  graph   CUDA-event time of one replay of the call captured in a CUDA graph (device time alone)
Then median, p10, p90 and a bootstrap 95% CI of the median paired difference (impl - old), as JSON with the raw
samples, each case's seed, skipped combinations, and the environment: versions, the measured files' SHA-256, the
git revision where readable, and host load and GPU clocks, throttle reasons, temperatures and power before and after.

    python tools/bench_qsa_select.py --blocks 32768,75000,150000,262144 --rows 1,3,5,20 --out qsa.json
    python tools/bench_qsa_select.py --blocks 150000 --rows 20 --streams 4 --unit --out qsa-b.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import triton

from tensorfold.cuda import ab_switch, geometry
from tensorfold.families.qwen4_exp.cuda import attention as att
from tensorfold.families.qwen4_exp.cuda import attn_multi

DEV = "cuda"
RATIO, TOP, DI, HI = 4, 512, 128, 4
IMPLS = ("old", "T", "policy", "new")
PATTERNS = ("real", "tied", "flat")


def ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x]


class Case:
    """One measured configuration: a scratch, the streams' pooled keys and index queries, positions."""

    def __init__(self, blocks: int, rows: int, streams: int, pattern: str, seed: int) -> None:
        self.blocks, self.rows, self.streams = blocks, rows, streams
        self.each = rows // streams
        self.sc = att.AttnScratch(rows, 8, 64, RATIO * blocks + 8, DEV)
        g = torch.Generator(device=DEV).manual_seed(seed)
        self.pooled = [(torch.randn((self.sc.nb, DI), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
                       for _ in range(streams)]
        self.iq = (torch.randn((rows, HI, DI), generator=g, device=DEV) * 0.5).to(torch.bfloat16)
        # each stream's rows end at 4 * blocks - each + 1 .. 4 * blocks, the last at the launch bound
        end = RATIO * blocks
        self.firsts = [end - self.each] * streams
        self.pos = torch.tensor([end - self.each], dtype=torch.int32, device=DEV)
        self.stream_pos = [torch.tensor([p], dtype=torch.int32, device=DEV) for p in self.firsts]
        self.posr = torch.tensor([p + i for p in self.firsts for i in range(self.each)], dtype=torch.int32, device=DEV)
        self.sid = torch.tensor([s for s in range(streams) for _ in range(self.each)], dtype=torch.int32, device=DEV)
        self.cp = torch.zeros((attn_multi.PTRS * streams,), dtype=torch.int64, device=DEV)
        self.cp[5 * streams:] = torch.tensor([t.data_ptr() for t in self.pooled], dtype=torch.int64)
        self.score_all()
        if pattern == "tied":
            self.sc.scores.copy_(torch.floor(self.sc.scores * 2) / 2)
        elif pattern == "flat":
            self.sc.scores.zero_()
        elif pattern != "real":
            raise SystemExit(f"unknown pattern {pattern}")
        self.fixed = self.sc.scores.clone()

    def score_all(self) -> None:
        for s in range(self.streams):
            self.scores_old(s)

    def scores_old(self, s: int) -> None:
        a0, sc = s * self.each, self.sc
        att._scores[(self.each, triton.cdiv(self.blocks, 64))](self.iq[a0:], self.pooled[s], self.stream_pos[s],
                                                             sc.scores[a0:], sc.nb, HI=HI, DI=DI, RATIO=RATIO,
                                                             TOP=TOP, BB=64, num_warps=4)

    def select_old(self, s: int) -> None:
        """Today's kernel choice for one stream's rows (``_launch_select`` before T): one stream launches on the
        scratch itself (``qsa_rows``), several each on their rows' views (``attn_multi._rows_from``)."""

        a0, sc, n = s * self.each, self.sc, self.each
        width = triton.next_power_of_2(self.blocks)
        if self.streams == 1:
            args = (sc.scores, self.stream_pos[s], sc.ids, sc.nk, sc.sparse, sc.nb)
        else:
            args = (sc.scores[a0:], self.stream_pos[s], sc.ids[a0:], sc.nk[a0:], sc.sparse[a0:], sc.nb)
        if width <= att.SELECT_REGS:
            att._select[(n,)](*args, RATIO=RATIO, TOP=TOP, IDW=sc.idw, BLOCK=width, num_warps=16)
        else:
            tb, warps = (4096, 8) if n >= 64 else (8192, 16)
            att._select_tiles[(n,)](*args, RATIO=RATIO, TOP=TOP, IDW=sc.idw, TB=tb, num_warps=warps)

    def impls(self, names: list[str], unit: bool) -> dict:
        """Each implementation's call: selection alone, or with ``unit`` the scores first."""

        sc, out = self.sc, {}
        if self.streams == 1:
            out["old"] = lambda: self.select_old(0)
            out["T"] = lambda: att._select_multi(sc, self.pos, self.rows, self.blocks, False)
            out["policy"] = lambda: att._launch_select(sc, self.pos, self.rows, self.blocks)
            if unit:
                out = {k: (lambda f=f: (self.scores_old(0), f())) for k, f in out.items()}
        else:
            def old():
                for s in range(self.streams):
                    if unit:
                        self.scores_old(s)
                    self.select_old(s)

            def batched(select):
                def call():
                    if unit:
                        attn_multi._scores_multi[(self.rows, triton.cdiv(self.blocks, 64))](
                            self.iq, self.cp, self.posr, self.sid, sc.scores, sc.nb, self.streams, HI=HI, DI=DI,
                            RATIO=RATIO, TOP=TOP, BB=64, num_warps=4)
                    select()
                return call
            out = {"old": old, "new": batched(lambda: att._launch_select(sc, self.posr, self.rows, self.blocks,
                                                                         per_row=True))}
        return {k: out[k] for k in names if k in out}


def lists(sc: att.AttnScratch) -> tuple:
    return sc.ids.clone(), sc.nk.clone(), sc.sparse.clone()


def reset(case: Case) -> None:
    case.sc.ids.fill_(-7)
    case.sc.nk.fill_(-1)
    case.sc.sparse.fill_(-1)
    case.sc.scores.copy_(case.fixed)


def graph_of(fn) -> torch.cuda.CUDAGraph:
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def summary(xs: list[float]) -> dict:
    a = np.asarray(xs)
    return {"median": float(np.median(a)), "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "n": int(a.size)}


def paired(xs: list[float], base: list[float], rng: np.random.Generator) -> dict:
    """Median of the per-rep differences (interleaved reps pair them) with a 95% percentile-bootstrap CI."""

    d = np.asarray(xs) - np.asarray(base)
    boots = np.median(d[rng.integers(0, d.size, (2000, d.size))], axis=1)
    return {"median": float(np.median(d)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def conditions() -> dict:
    """Host load and the GPUs' clocks, clock-event (throttle) reasons, temperatures and power, now."""

    try:
        smi = subprocess.run(["nvidia-smi", "-q", "-d", "CLOCK,PERFORMANCE,TEMPERATURE,POWER"], capture_output=True,
                             text=True, timeout=20, check=False).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        smi = f"unavailable: {exc}"
    return {"unix_time": time.time(), "loadavg": list(os.getloadavg()), "nvidia_smi": smi}


def revision() -> dict:
    """The checkout's commit and its changed tracked files, where git can read it (a mounted worktree may not be)."""

    top = str(Path(__file__).resolve().parent.parent)
    try:
        head = subprocess.run(["git", "-C", top, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20,
                              check=True).stdout.strip()
        changed = subprocess.run(["git", "-C", top, "status", "--porcelain", "--untracked-files=no"],
                                 capture_output=True, text=True, timeout=20, check=True).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        return {"unavailable": str(exc)}
    return {"head": head, "changed": changed}


def environment(args) -> dict:
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    # every file whose bytes define what is measured: the kernels and their launchers, the scratch's constants, the
    # A/B switch the policy reads, and this tool
    files = [att.__file__, attn_multi.__file__, geometry.__file__, ab_switch.__file__, __file__]
    return {"device": p.name, "sms": p.multi_processor_count, "l2_bytes": p.L2_cache_size,
            "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__,
            "triton": triton.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
            "host": platform.node(), "cpu": platform.processor(),
            "source_sha256": {Path(f).name: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
            "revision": revision(), "source_rev": args.source_rev, "argv": sys.argv, "before": conditions(),
            "policy": {"MULTI_ROWS": att.MULTI_ROWS, "MULTI_MIN_BLOCKS": att.MULTI_MIN_BLOCKS,
                       "SELECT_TB": att.SELECT_TB, "SELECT_FILL": att.SELECT_FILL,
                       "SELECT_SLICES": att.SELECT_SLICES, "SELECT_SPAN": att.SELECT_SPAN,
                       "qsa": "new" if att.qsa_new() else "old"}}


def run_case(case: Case, names: list[str], args, flush: torch.Tensor | None, rng) -> dict:
    fns = case.impls(names, args.unit)
    graphs = {}
    for k, fn in fns.items():
        reset(case)
        graphs[k] = graph_of(fn)
    samples = {k: {"eager": [], "host": [], "graph": []} for k in fns}
    order = list(fns)
    for rep in range(args.warmup + args.reps):
        want = None
        for k in order if rep % 2 == 0 else order[::-1]:          # alternate the order rep to rep
            for metric in ("eager", "graph"):
                reset(case)
                if flush is not None:
                    flush.add_(1)
                torch.cuda.synchronize()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                t0 = time.perf_counter()
                if metric == "eager":
                    fns[k]()
                else:
                    graphs[k].replay()
                host = time.perf_counter() - t0
                b.record()
                torch.cuda.synchronize()
                got = lists(case.sc)
                if want is None:
                    want = got
                elif not all(torch.equal(x, y) for x, y in zip(got, want)):
                    raise SystemExit(f"MISMATCH: {k} {metric} at blocks {case.blocks} rows {case.rows} "
                                     f"streams {case.streams} rep {rep}")
                if rep >= args.warmup:
                    samples[k][metric].append(a.elapsed_time(b) * 1000)
                    if metric == "eager":
                        samples[k]["host"].append(host * 1e6)
    result = {k: {m: summary(v) for m, v in s.items()} for k, s in samples.items()}
    base = samples.get("old")
    if base is not None:
        for k, s in samples.items():
            if k != "old":
                result[k]["vs_old"] = {m: paired(s[m], base[m], rng) for m in ("eager", "graph", "host")}
    return {"blocks": case.blocks, "rows": case.rows, "streams": case.streams, "summary": result, "raw": samples}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", type=ints, default=ints("32768,75000,150000,262144"))
    ap.add_argument("--rows", type=ints, default=ints("1,3,5,20"))
    ap.add_argument("--streams", type=ints, default=[1])
    ap.add_argument("--pattern", default="real", help="comma list of " + ", ".join(PATTERNS))
    ap.add_argument("--impl", default="old,T,policy,new",
                    help="comma list of " + ", ".join(IMPLS) + " (each case runs those that apply to it)")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--l2", choices=("warm", "cold"), default="warm")
    ap.add_argument("--unit", action="store_true", help="time the block scores and the select together")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--source-rev", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    names, patterns = args.impl.split(","), args.pattern.split(",")
    unknown = sorted(set(names) - set(IMPLS)) + sorted(set(patterns) - set(PATTERNS))
    if unknown:
        ap.error(f"unknown --impl or --pattern: {', '.join(unknown)}")
    flush = None
    if args.l2 == "cold":                                         # a write of twice the L2 between reps
        flush = torch.zeros((2 * torch.cuda.get_device_properties(torch.cuda.current_device()).L2_cache_size // 4,),
                            device=DEV)
    rng = np.random.default_rng(args.seed)
    record = {"environment": environment(args), "args": vars(args), "cases": [], "skipped": []}
    for blocks in args.blocks:
        for rows in args.rows:
            for streams in args.streams:
                if rows % streams:
                    record["skipped"].append({"blocks": blocks, "rows": rows, "streams": streams,
                                              "reason": "rows are not a multiple of streams"})
                    continue
                for pattern in patterns:
                    seed = args.seed + blocks + rows
                    case = Case(blocks, rows, streams, pattern, seed)
                    out = run_case(case, names, args, flush, rng)
                    out["pattern"], out["seed"] = pattern, seed
                    record["cases"].append(out)
                    line = [f"blocks {blocks:7d} rows {rows:3d} streams {streams} {pattern:5s}"]
                    for k, s in out["summary"].items():
                        line.append(f"{k}: eager {s['eager']['median']:7.1f} graph {s['graph']['median']:7.1f} "
                                    f"host {s['host']['median']:6.1f}")
                    print(" | ".join(line), flush=True)
                    del case
                    torch.cuda.empty_cache()
    record["environment"]["after"] = conditions()
    Path(args.out).write_text(json.dumps(record))


if __name__ == "__main__":
    main()

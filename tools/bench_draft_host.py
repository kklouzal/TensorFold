"""Host (CPU) cost of Flash Next's drafting decisions: the session index (``drafters.session_index``), today's copy
index (``CopyIndex``, the A/B baseline) and the cost-aware rule's per-round arithmetic (``drafters.draft_rule``).

    python tools/bench_draft_host.py [--sizes 50000,200000,800000] [--rounds 2000] [--repeats 5] [--json OUT]

Contexts are synthetic code-like token sequences (random ids with frequent copies of earlier spans, vocabulary
248,320). Per size: the session index's prompt build on the calling thread, and in the background beside a simulated
engine thread (1 ms GPU waits, then 50 us of host work) with how late it wakes from its waits (against the same
waits with no build); per-round ``sync`` + ``propose`` (1-5 tokens a round, half of them copies of recent text) as
p50/p99/max; a heavily repeated gram's query; memory per token; and ``CopyIndex``'s first (bulk) and per-round
proposals. Rule: one round's host decisions for 4 streams x depth 4 (chances, chain steps, text-versus-chain values,
the trim and the cost model's update). numpy only: run it in the target's runtime image for its pinned numpy and CPU
(GB10's Grace cores).
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Sequence
import json
from pathlib import Path
import platform
import random
import sys
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from tensorfold.drafters import draft_rule, session_index  # noqa: E402
from tensorfold.drafters.session_index import Builder, SessionIndex  # noqa: E402

VOCAB = 248_320


def copy_index_class():
    source = ROOT / "src/tensorfold/families/qwen3_5/cuda/decode.py"
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "CopyIndex")
    namespace = {"Sequence": Sequence}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["CopyIndex"]


def corpus(n: int, rng: random.Random) -> list[int]:
    ctx = [rng.randrange(VOCAB) for _ in range(2000)]
    while len(ctx) < n:
        if rng.random() < 0.6:
            at = rng.randrange(len(ctx) - 400)
            ctx += ctx[at:at + rng.randint(8, 400)]
        else:
            ctx += [rng.randrange(VOCAB) for _ in range(rng.randint(5, 200))]
    return ctx[:n]


def quantiles(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {"p50_us": round(xs[len(xs) // 2] * 1e6, 2), "p99_us": round(xs[int(len(xs) * 0.99)] * 1e6, 2),
            "max_us": round(xs[-1] * 1e6, 2)}


def rounds(rng: random.Random, ctx: list[int], count: int) -> list[list[int]]:
    """Each round's new tokens: half the time 1-5 tokens copied from recent text, else one fresh token."""

    out, seq = [], list(ctx[-2000:])
    for _ in range(count):
        if rng.random() < 0.5:
            k = rng.randint(1, 5)
            at = rng.randrange(len(seq) - 60)
            new = seq[at:at + k]
        else:
            new = [rng.randrange(VOCAB)]
        seq += new
        out.append(new)
    return out


def bench_session(n: int, args, rng: random.Random) -> dict:
    prompt = corpus(n, rng)
    builds = []
    for _ in range(args.repeats):
        t = time.perf_counter()
        index = SessionIndex(prompt)                    # synchronous: the whole build on this thread
        builds.append(time.perf_counter() - t)
    stall, late = Builder(), []
    try:                                 # an engine thread meanwhile: 1 ms GPU waits (GIL free), then 50 us host work
        t0 = time.perf_counter()
        background = SessionIndex(prompt, stall)
        while not background.future.done():
            t = time.perf_counter()
            time.sleep(0.001)
            t1 = time.perf_counter()                     # back from the wait: the GIL reacquired
            while time.perf_counter() - t1 < 50e-6:
                pass
            late.append(max(0.0, t1 - t - 0.001))
        background_s = time.perf_counter() - t0
        background.ready()
    finally:
        stall.close()
    idle = []
    for _ in range(200):                                 # the same waits with no build: the timer's own overshoot
        t = time.perf_counter()
        time.sleep(0.001)
        idle.append(max(0.0, time.perf_counter() - t - 0.001))
    news = rounds(rng, prompt, args.rounds)
    out, per = [], []
    for new in news:
        t = time.perf_counter()
        index.sync(out, new)
        index.propose(4)
        per.append(time.perf_counter() - t)
        out += new
    repeated = SessionIndex([1, 2, 3, 4] * (n // 8) + [rng.randrange(VOCAB) for _ in range(n // 2)] + [1, 2, 3, 4])
    t = time.perf_counter()
    repeated.propose(4)
    hot = time.perf_counter() - t
    return {"tokens": n, "build_ms_min": round(min(builds) * 1e3, 2), "build_ms_median": round(sorted(builds)[
        len(builds) // 2] * 1e3, 2), "background_ms": round(background_s * 1e3, 2),
        "engine_wake_late": quantiles(late), "idle_wake_late": quantiles(idle),
        "round": quantiles(per),
        "repeated_gram_query_us": round(hot * 1e6, 1),
        "bytes_per_prompt_token": round((index.keys.nbytes + 4 * n) / n, 2),
        "resident_bytes": int(index.keys.nbytes + index.tokens.nbytes)}


def bench_copy(n: int, args, rng: random.Random) -> dict:
    CopyIndex = copy_index_class()
    ctx = corpus(n, rng)
    copies = CopyIndex(8)
    t = time.perf_counter()
    copies.propose(ctx, 8)
    first = time.perf_counter() - t
    per = []
    for new in rounds(rng, ctx, args.rounds):
        ctx += new
        t = time.perf_counter()
        copies.propose(ctx, 8)
        per.append(time.perf_counter() - t)
    return {"tokens": n, "first_propose_ms": round(first * 1e3, 2), "round": quantiles(per)}


def bench_rule(args, rng: random.Random) -> dict:
    """One round's host arithmetic for 4 streams at depth 4: thresholds, 4 chain steps a stream, a text-versus-chain
    value for two of them, the round-start trim and the cost model's update."""

    pred, costs = draft_rule.Predictor.prior(0.5), draft_rule.CostModel()
    per = []
    for _ in range(args.rounds):
        t = time.perf_counter()
        lam = costs.lam(4, False, 200_000)
        ths = [costs.thresholds(lam, 200_000 + s, False) for s in range(4)]
        chains = []
        for s, th in enumerate(ths):
            c, chain = 1.0, []
            for j in range(4):
                c, include, full, shared = draft_rule.chain_step(c, pred.mtp(s % 2 == 0, j, rng.random(), None),
                                                                 pred.abar("mtp", s % 2 == 0, j + 1), j < 3, th)
                if include:
                    chain.append(c)
                if not (full or shared):
                    break
            chains.append(chain)
            if s < 2:
                given = [pred.suffix(False, k, 12) for k in range(4)]
                abar = [pred.abar("mtp", False, j) for j in range(4)]
                draft_rule.suffix_plan(given, th)
                draft_rule.mtp_value(0.9, abar, 4, th)
        draft_rule.trim(chains, [th.row for th in ths], 16)
        costs.observe(rows=1 + sum(map(len, chains)) + 3, ctx_rows=800_000, pass_rows=0, mixed=False,
                      ms=50.0 + rng.random(), chain_ms=4.5, chain_steps=3, tokens=6, live=4, context=200_000)
        per.append(time.perf_counter() - t)
    return {"streams": 4, "depth": 4, "round": quantiles(per)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", default="50000,200000,800000", help="context sizes in tokens")
    ap.add_argument("--rounds", type=int, default=2000, help="rounds measured a size (default 2000)")
    ap.add_argument("--repeats", type=int, default=5, help="builds timed a size (default 5)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", help="write the results here")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    sizes = [int(x) for x in args.sizes.split(",") if x]
    if any(n < session_index.BULK for n in sizes):           # shorter prompts build on the calling thread
        ap.error(f"--sizes: each at least {session_index.BULK} tokens (a background build to measure)")
    result = {"host": {"machine": platform.machine(), "processor": platform.processor(),
                       "python": sys.version.split()[0], "numpy": np.__version__,
                       "switch_interval_s": sys.getswitchinterval()},
              "params": {"sizes": sizes, "rounds": args.rounds, "repeats": args.repeats, "seed": args.seed,
                         "anchor": session_index.ANCHOR, "recent": session_index.RECENT,
                         "longest": session_index.LONGEST},
              "session": [bench_session(n, args, rng) for n in sizes],
              "copy": [bench_copy(n, args, rng) for n in sizes],
              "rule": bench_rule(args, rng), "threads": threading.active_count()}
    print(json.dumps(result, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Host (CPU) cost of Flash Next's drafting decisions: the session index (``drafters.session_index``), today's copy
index (``CopyIndex``, the A/B baseline) and the cost-aware rule's per-round arithmetic (``drafters.draft_rule``).

    python tools/bench_draft_host.py [--sizes 50000,200000,800000] [--rounds 2000] [--repeats 5] [--reply 100000]
        [--json OUT]

Contexts are synthetic code-like token sequences (random ids with frequent copies of earlier spans, vocabulary 248,320).
Per size: the session index's prompt build on the calling thread, and in the background beside a simulated engine thread
(1 ms GPU waits, then 50 us of host work) with how late it wakes from its waits (against the same waits with no build);
per-round ``sync`` + ``propose`` (1-5 tokens a round, half of them copies of recent text) as p50/p99/max; a heavily
repeated gram's query; memory (``tracemalloc``, numpy's arrays and Python's objects, in a pass of its own: the build's
peak, what stays a prompt token, and what a reply token adds at worst, every reply gram new); and ``CopyIndex``'s first
(bulk) and per-round proposals. Rule (``bench_rule``): one round's host decisions on the decoder's critical path for 1
and 4 streams at depth 4, the confidence rule's against the cost-aware rule's. numpy only: run it in the target's
runtime image for its pinned Python, numpy and CPU (GB10's Grace cores).
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
import tracemalloc

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


class _Head:
    """The first ``n`` tokens of a list, sliced without copying the rest (a reply for ``SessionIndex.sync``)."""

    def __init__(self, seq: list[int], n: int) -> None:
        self.seq, self.n = seq, n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, key: slice) -> list[int]:
        start, stop, _ = key.indices(self.n)
        return self.seq[start:stop]


def session_memory(prompt: list[int], reply: list[int]) -> dict:
    """The session index's traced bytes: its prompt build's peak, what stays after it, and what ``reply`` (fresh ids:
    every gram new, the worst case) adds a token, synced 4 tokens a round."""

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        base = tracemalloc.get_traced_memory()[0]
        index = SessionIndex(prompt)
        built, peak = tracemalloc.get_traced_memory()
        for at in range(0, len(reply), 4):
            index.sync(_Head(reply, at), reply[at:at + 4])
        grown = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    return {"build_peak_bytes": peak - base, "built_bytes": built - base,
            "built_bytes_per_prompt_token": round((built - base) / len(prompt), 2),
            "reply_bytes_per_token_worst": round((grown - built) / max(1, len(reply)), 1)}


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
        "resident_bytes": int(index.keys.nbytes + index.tokens.nbytes),
        "memory": session_memory(prompt, [rng.randrange(VOCAB) for _ in range(args.reply)])}


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


def _fitted_like(rng: random.Random) -> draft_rule.Predictor:
    """A predictor with fitted-like MTP tables (12 score bins, sample counts a bin) for the rule's lookups."""

    data = draft_rule.Predictor.prior(0.5).as_dict()
    edges = [round(0.2 + 0.065 * k, 3) for k in range(11)]
    for policy in draft_rule.POLICIES:
        data["tables"]["mtp"][policy] = {
            "feature": "p", "depth_edges": [0, 1, 2, 3, 5, 8], "score_edges": edges,
            "table": [[round(min(0.99, 0.3 + 0.06 * k - 0.02 * d), 4) for k in range(12)] for d in range(6)],
            "abar": [0.97] * 6, "count": [[rng.randint(50, 900) for _ in range(12)] for _ in range(6)]}
    data["source"] = {"tool": "bench"}
    return draft_rule.Predictor.from_dict(data)


def _picks(rng: random.Random, n: int, depth: int) -> list[list[float]]:
    """Each stream's draft probabilities a pick: mostly confident, as the calibration log's (70% at 0.95 or more)."""

    return [[rng.uniform(0.95, 1.0) if rng.random() < 0.7 else rng.uniform(0.0, 0.95) for _ in range(depth)]
            for _ in range(n)]


def confidence_round(picks: list[list[float]], confidence: float, depth: int) -> int:
    """The A/B baseline's host decisions in a round (``MultiDecoder._chain``, rule "old"): a comparison a pick."""

    active, drafts, steps = list(range(len(picks))), [[] for _ in picks], 0
    for j in range(depth):
        go = []
        for i in active:
            p = picks[i][j]
            low = confidence > 0 and p < confidence
            if not low or j == 0:
                drafts[i].append(j)
            if low:
                continue
            if j + 1 < depth:
                go.append(i)
        if not go:
            break
        active, steps = go, steps + 1
    return steps


def rule_round(costs: draft_rule.CostModel, pred: draft_rule.Predictor, picks: list[list[float]], depth: int,
               ctx: list[int], chances: list[list[float]]) -> int:
    """The cost-aware rule's host decisions in a round, as ``MultiDecoder`` runs them on the critical path: the
    round-start trim, the chain's costs, each pick's chance, chain step and joint step, and the cost observation."""

    n = len(picks)
    lam = costs.lam(n, False, max(ctx))                                  # _trim
    draft_rule.trim(chances, [lam * costs.row_ms(c, False) for c in ctx])
    lam = costs.lam(n, False, max(ctx))                                  # _chain
    step, extra = lam * costs.step_ms(False), lam * costs.step_stream_ms(False)
    rows = {i: lam * costs.row_ms(c, False) for i, c in enumerate(ctx)}
    tables = {i: pred.mtp_tables[draft_rule.greedy(None)] for i in range(n)}
    chance = dict.fromkeys(rows, 1.0)
    active, drafts, steps, stream_steps = list(range(n)), [[] for _ in picks], 0, 0
    for j in range(depth):
        candidates = []
        for i in active:
            table = tables[i]
            c, include, value = draft_rule.chain_step(chance[i], table.chance(j, picks[i][j]), table, j,
                                                      j + 1 < depth, rows[i])
            chance[i] = c
            if include:
                drafts[i].append(c)
            if value is not None:
                candidates.append((i, value))
        active = [i for (i, _), on in zip(candidates, draft_rule.joint_step([v for _, v in candidates], step, extra))
                  if on]
        if not active:
            break
        steps, stream_steps = steps + 1, stream_steps + len(active)
    rows_all = n + sum(map(len, drafts))
    costs.observe(rows=rows_all, live=n, ctx_rows=float(rows_all * max(ctx)), pass_rows=0, mixed=False,
                  ms=50.0 + 10.0 * n, chain_ms=2.2 * steps, chain_steps=steps, chain_rows=stream_steps, tokens=n + 2,
                  context=max(ctx))
    return steps


def bench_rule(args, rng: random.Random) -> dict:
    """One round's host decisions at depth 4 for 1 and 4 streams: the confidence rule (the A/B baseline) and the
    cost-aware rule with the prior's tables and with fitted-like ones (sample counts: expected next-draft values),
    as p50/p99/max microseconds a round (rounds interleaved; picks drawn as the calibration log's)."""

    out = {}
    fitted = _fitted_like(rng)
    for n in (1, 4):
        ctx = [30_000 + 70_000 * i for i in range(n)]
        variants = {"confidence": [], "rule-prior": [], "rule-fitted": []}
        models = {"rule-prior": (draft_rule.CostModel(), draft_rule.Predictor.prior(0.5)),
                  "rule-fitted": (draft_rule.CostModel(), fitted)}
        for _ in range(args.rounds):
            picks = _picks(rng, n, 4)
            chances = [[0.9, 0.81, 0.73][:rng.randint(0, 3)] for _ in range(n)]
            for name in variants:                        # interleaved: the same picks, back to back
                t = time.perf_counter()
                if name == "confidence":
                    confidence_round(picks, 0.5, 4)
                else:
                    rule_round(*models[name], picks, 4, ctx, chances)
                variants[name].append(time.perf_counter() - t)
        out[f"streams{n}"] = {name: quantiles(xs) for name, xs in variants.items()}
    return {"depth": 4, "rounds": args.rounds, **out}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sizes", default="50000,200000,800000", help="context sizes in tokens")
    ap.add_argument("--rounds", type=int, default=2000, help="rounds measured a size (default 2000)")
    ap.add_argument("--repeats", type=int, default=5, help="builds timed a size (default 5)")
    ap.add_argument("--reply", type=int, default=100_000, help="reply tokens of the memory pass (default 100000)")
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
              "params": {"sizes": sizes, "rounds": args.rounds, "repeats": args.repeats, "reply": args.reply,
                         "seed": args.seed, "anchor": session_index.ANCHOR, "recent": session_index.RECENT,
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

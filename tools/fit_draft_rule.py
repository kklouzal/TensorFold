"""Fit the cost-aware draft rule's tables (``tensorfold.drafters.draft_rule``) from a server's draft telemetry.

    python tools/fit_draft_rule.py --log DRAFT_LOG.jsonl... [--traces TRACE_DIR...] --output rule.json
        [--source provenance.json] [--confidence 0.5] [--suffix-depth 8]

Inputs: the ``TENSORFOLD_DRAFT_LOG`` lines of a run (with ``TENSORFOLD_DRAFT_PROBE`` for chains past the rule) and,
optionally, its ``TENSORFOLD_DRAFT_TRACE`` files (``tensorfold.cuda.draft_log``).

- MTP tables (``mtp``, per policy): every logged MTP draft whose chain reached it (the drafts before it were kept) is
  a sample of its conditional acceptance: its ``hit``. Labels are exact (a reply does not depend on its drafts), and
  shadow drafts (dropped by the rule, or chained past it in probes) count too. Each candidate score feature (``p``,
  the draft head's probability; ``m``, the keyed margin) gets a monotone table (``drafters.calibration.fit``: pooled
  adjacent violators by depth row over quantile score bins); the feature with the lower held-out log-loss is kept.
- Suffix tables (``suffix``): with traces, every reply position is a round start the session index could draft from;
  its continuation's tokens are labelled against the reply (exact), by continuation index and match length. Without
  traces, the logged suffix and copy drafts are used.
- Cost prior (``cost``): an MTP chain step's mean logged time, and the logged rounds' milliseconds less their chain
  steps' fitted to ``t0 + row * (R - 1) + ctx * sum(rows * context) / 1e5 + pass * prompt_rows`` (least squares
  anchored at the design's prior, as the server's online model), with and without a prompt pass, and lambda* = all
  rounds' tokens over their milliseconds.

Requests are split by a hash of their random id into a fitting half and a held-out half for the report (log-loss,
Brier score and reliability by predicted band, against the confidence rule's prior too); the written tables are then
fitted on every sample. The file holds numbers only: no token ids or text. Install it with
``TENSORFOLD_DRAFT_RULE=<path>`` (the server validates it at startup).
"""

from __future__ import annotations

import argparse
import bisect
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from tensorfold.drafters import calibration, draft_rule, session_index  # noqa: E402

DEPTH_EDGES = (0, 1, 2, 3, 5, 8)
L_EDGES = (5.5, 7.5, 11.5, 15.5, 23.5, 31.5, 47.5, 63.5)      # match lengths: <=5, 6-7, 8-11, ... 48-63, 64
SCORE_BINS = 12                                                # quantile bins of a probability or margin
EPS = 1e-6


def _score(x):
    return float(x) if isinstance(x, str) else x


class _Prefix:
    """The first ``n`` tokens of a list, sliced without copying the rest (``SessionIndex.sync`` reads a reply)."""

    def __init__(self, seq: list[int], n: int) -> None:
        self.seq, self.n = seq, n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, key: slice) -> list[int]:
        start, stop, _ = key.indices(self.n)
        return self.seq[start:stop]


def held(rid: str) -> bool:
    """The held-out half: a stable hash of the request's random id."""

    return hashlib.sha256(rid.encode()).digest()[0] & 1 == 1


def mtp_samples(lines: Iterable[dict]) -> dict[str, list[tuple[int, float | None, float | None, int, bool]]]:
    """(depth, p, m, hit, held out) for each MTP draft its chain reached, by policy."""

    out: dict[str, list] = {"greedy": [], "sampled": []}
    for line in lines:
        for st in line["streams"]:
            if st["src"] != "mtp" or "greedy" not in st:
                continue
            policy = "greedy" if st["greedy"] else "sampled"
            for j, d in enumerate(st["drafts"]):
                if d["hit"] is None or (j and st["drafts"][j - 1]["label"] != 1):
                    break
                out[policy].append((d["j"], d["p"], _score(d["m"]), d["hit"], held(st["rid"])))
    return out


def suffix_samples_from_traces(traces: Sequence[dict], depth: int) -> dict[str, list[tuple[int, int, int, bool]]]:
    """(continuation index, match length, hit, held out) at every reply position of each drafting request."""

    out: dict[str, list] = {"greedy": [], "sampled": []}
    for tr in traces:
        if not tr["draft"] or tr["constrained"]:
            continue
        decoder = tr["decoder"]
        index = session_index.SessionIndex(tr["prompt"], k=int(decoder.get("anchor", session_index.ANCHOR)),
                                           recent=int(decoder.get("recent", session_index.RECENT)),
                                           longest=int(decoder.get("longest", session_index.LONGEST)))
        sampling, reply = tr["sampling"], tr["out"]
        policy = "greedy" if sampling is None or sampling["temperature"] <= 0 else "sampled"
        h = held(tr["rid"])
        for i in range(1, len(reply)):                   # a round starting with i reply tokens out
            index.sync(_Prefix(reply, i))
            length, tokens = index.propose(min(depth, len(reply) - i))
            for k, t in enumerate(tokens):
                hit = t == reply[i + k]
                out[policy].append((k, length, hit, h))
                if not hit:
                    break
    return out


def suffix_samples_from_log(lines: Iterable[dict]) -> dict[str, list[tuple[int, int, int, bool]]]:
    out: dict[str, list] = {"greedy": [], "sampled": []}
    for line in lines:
        for st in line["streams"]:
            if st["src"] not in ("suffix", "copy") or "greedy" not in st:
                continue
            policy = "greedy" if st["greedy"] else "sampled"
            for j, d in enumerate(st["drafts"]):
                if d["hit"] is None or d["L"] is None or (j and st["drafts"][j - 1]["label"] != 1):
                    break
                out[policy].append((d["j"], d["L"], d["hit"], held(st["rid"])))
    return out


def edges(scores: Sequence[float]) -> list[float]:
    """Quantile bin edges of the finite scores (strictly ascending)."""

    finite = np.asarray([s for s in scores if s is not None and math.isfinite(s)], dtype=np.float64)
    if not len(finite):
        return []
    qs = np.quantile(finite, np.linspace(0, 1, SCORE_BINS + 1)[1:-1])
    out: list[float] = []
    for q in qs:
        q = float(q)
        if not out or q > out[-1]:
            out.append(q)
    return out


def fit_table(rows: Sequence[tuple[int, float, bool]], feature: str, score_edges: Sequence[float],
              depth_edges: Sequence[int] = DEPTH_EDGES) -> draft_rule.Table:
    """A monotone table and each depth row's optimistic chance (the 75th percentile of its samples' chances)."""

    cal = calibration.fit(rows, depth_edges, score_edges)
    count = [[0] * (len(score_edges) + 1) for _ in depth_edges]
    chances: list[list[float]] = [[] for _ in depth_edges]
    for depth, score, _ in rows:
        r = max(0, bisect.bisect_right(depth_edges, int(depth)) - 1)
        count[r][bisect.bisect_left(cal.score_edges, float(score))] += 1
        chances[r].append(cal.probability(depth, score))
    abar = [float(np.percentile(c, 75)) if c else max(cal.table[r]) for r, c in enumerate(chances)]
    table = [[round(p, 6) for p in row] for row in cal.table]
    return draft_rule.Table(feature, calibration.Calibration(table, depth_edges, score_edges),
                            [round(a, 6) for a in abar], count)


def metrics(predict, rows: Sequence[tuple[int, float, bool]]) -> dict:
    """Log-loss, Brier score and reliability (predicted mean / observed rate / count by tenth) of ``predict``."""

    if not rows:
        return {"n": 0}
    loss = brier = 0.0
    bands: dict[int, list[float]] = {}
    for depth, score, hit in rows:
        p = min(1 - EPS, max(EPS, predict(depth, score)))
        loss -= math.log(p if hit else 1 - p)
        brier += (p - hit) ** 2
        band = bands.setdefault(min(9, int(p * 10)), [0.0, 0.0, 0])
        band[0] += p
        band[1] += hit
        band[2] += 1
    return {"n": len(rows), "log_loss": round(loss / len(rows), 5), "brier": round(brier / len(rows), 5),
            "reliability": [[b, round(v[0] / v[2], 4), round(v[1] / v[2], 4), v[2]] for b, v in sorted(bands.items())]}


def fit_mtp(samples, prior: draft_rule.Predictor) -> tuple[dict, dict]:
    """Per policy: the feature with the lower held-out log-loss, its table fitted on every sample; and the report."""

    tables, report = {}, {}
    for policy, rows in samples.items():
        candidates = {}
        for feature, at in (("p", 1), ("m", 2)):
            usable = [(d, s[at], s[3], s[4]) for s in rows for d in (s[0],) if s[at] is not None
                      and not (isinstance(s[at], float) and math.isnan(s[at]))]
            if len(usable) < 2 * SCORE_BINS:
                continue
            score_edges = edges([u[1] for u in usable if not u[3]])
            train = [(d, s, h) for d, s, h, out in usable if not out]
            test = [(d, s, h) for d, s, h, out in usable if out]
            fitted = fit_table(train, feature, score_edges)
            candidates[feature] = {"held_out": metrics(fitted.chance, test), "samples": len(usable),
                                   "edges": score_edges, "rows": usable}
        baseline = prior.tables["mtp", policy]
        test_p = [(s[0], s[1], s[3]) for s in rows if s[4] and s[1] is not None]
        entry = {"prior": metrics(baseline.chance, test_p)}
        if candidates:
            best = min(candidates, key=lambda f: candidates[f]["held_out"]["log_loss"])
            chosen = candidates[best]
            tables[policy] = fit_table([(d, s, h) for d, s, h, _ in chosen["rows"]], best, chosen["edges"])
            entry.update({f: {k: v for k, v in c.items() if k not in ("rows", "edges")} for f, c in candidates.items()})
            entry["selected"] = best
        report[policy] = entry
    return tables, report


def fit_suffix(samples, prior: draft_rule.Predictor) -> tuple[dict, dict]:
    tables, report = {}, {}
    for policy, rows in samples.items():
        entry = {"prior": metrics(prior.tables["suffix", policy].chance, [(k, L, h) for k, L, h, out in rows if out])}
        if len(rows) >= 2 * SCORE_BINS:
            train = [(k, L, h) for k, L, h, out in rows if not out]
            entry["fit"] = {"held_out": metrics(fit_table(train, "L", L_EDGES).chance,
                                                [(k, L, h) for k, L, h, out in rows if out]), "samples": len(rows)}
            tables[policy] = fit_table([(k, L, h) for k, L, h, _ in rows], "L", L_EDGES)
        report[policy] = entry
    return tables, report


def fit_cost(lines: Sequence[dict]) -> tuple[dict, dict]:
    """The cost prior: each regime's MTP chain step (the logged steps' mean time), its rounds less their chain steps
    by least squares anchored at the design's prior (the server's online model, without forgetting), and lambda*."""

    prior = {**draft_rule.COST_PRIOR, "lambda": draft_rule.LAMBDA_PRIOR}
    out, report, tokens, ms = {}, {}, 0, 0.0
    for name, mixed in (("unmixed", False), ("mixed", True)):
        fit = draft_rule.CostFit(prior[name])
        x, y, chain_ms, chain_steps = [], [], 0.0, 0
        for line in lines:
            if bool(line["mixed"]) != mixed or not line["streams"]:
                continue
            ctx_rows = sum((st["verified"] + 1) * st["ctx"] for st in line["streams"])
            x.append(draft_rule.features(line["rows"], ctx_rows, line["pass_rows"]))
            y.append(line["ms"] - line["chain_ms"])
            chain_ms += line["chain_ms"]
            chain_steps += line["chain_steps"]
        if x:
            a, b = np.asarray(x), np.asarray(y)
            fit.theta = np.linalg.solve(a.T @ a + fit.precision0, a.T @ b + fit.precision0 @ fit.theta0)
            residual = b - a @ fit.theta
            report[name] = {"rounds": len(x), "rms_ms": round(float(np.sqrt(np.mean(residual ** 2))), 4),
                            "chain_steps": chain_steps}
        else:
            report[name] = {"rounds": 0}
        out[name] = {t: round(float(v), 6) for t, v in zip(draft_rule.COST_TERMS, fit.theta)}
        out[name]["step"] = round(chain_ms / chain_steps, 6) if chain_steps else prior[name]["step"]
    for line in lines:
        tokens += sum(st["accepted"] + 1 for st in line["streams"])
        ms += line["ms"]
    lam = tokens / ms if ms > 0 else draft_rule.LAMBDA_PRIOR
    out["lambda"] = round(min(draft_rule.LAMBDA[1], max(draft_rule.LAMBDA[0], lam)), 6)
    report["lambda"] = {"tokens": tokens, "ms": round(ms, 3)}
    return out, report


def _inputs(paths: Sequence[Path]) -> list[dict]:
    return [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
            for p in paths]


def run(args) -> dict:
    log_paths = [Path(p) for p in args.log]
    lines = [json.loads(t) for p in log_paths for t in p.read_text().splitlines() if t.strip()]
    trace_paths: list[Path] = []
    for p in map(Path, args.traces):
        trace_paths += sorted(p.glob("*.json")) if p.is_dir() else [p]
    traces = [json.loads(p.read_text()) for p in trace_paths]
    prior = draft_rule.Predictor.prior(args.confidence)
    mtp_tables, mtp_report = fit_mtp(mtp_samples(lines), prior)
    suffix = suffix_samples_from_traces(traces, args.suffix_depth) if traces else suffix_samples_from_log(lines)
    suffix_tables, suffix_report = fit_suffix(suffix, prior)
    cost, cost_report = fit_cost(lines)
    tables = {}
    for source, fitted in (("mtp", mtp_tables), ("suffix", suffix_tables)):
        for policy in draft_rule.POLICIES:
            table = fitted.get(policy) or prior.tables[source, policy]   # too few samples: the prior's, said so
            tables.setdefault(source, {})[policy] = table.as_dict()
    unfitted = [f"{s}.{p}" for s, f in (("mtp", mtp_tables), ("suffix", suffix_tables))
                for p in draft_rule.POLICIES if p not in f]
    extra = json.loads(Path(args.source).read_text()) if args.source else {}
    source = {"tool": "tools/fit_draft_rule.py", "created": datetime.datetime.now(datetime.timezone.utc)
              .isoformat(timespec="seconds"), "logs": _inputs(log_paths), "traces": len(trace_paths),
              "trace_inputs": _inputs(trace_paths), "suffix_from": "traces" if traces else "log",
              "confidence_prior": args.confidence, "prior_tables": unfitted,
              "held_out": {"mtp": mtp_report, "suffix": suffix_report, "cost": cost_report}, **extra}
    data = {"v": draft_rule.VERSION, "kind": draft_rule.KIND, "source": source, "tables": tables, "cost": cost}
    draft_rule.Predictor.from_dict(data)                 # what the server will accept
    return data


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", nargs="+", required=True, help="TENSORFOLD_DRAFT_LOG files")
    ap.add_argument("--traces", nargs="*", default=[], help="TENSORFOLD_DRAFT_TRACE files or directories")
    ap.add_argument("--output", required=True, help="the calibration file to write (TENSORFOLD_DRAFT_RULE)")
    ap.add_argument("--source", help="a JSON object of provenance to add (model, image, commands, prompts)")
    ap.add_argument("--confidence", type=float, default=0.5, help="the prior's confidence (default 0.5, production)")
    ap.add_argument("--suffix-depth", type=int, default=8, help="continuation tokens labelled a position (8)")
    args = ap.parse_args(argv)
    data = run(args)
    Path(args.output).write_text(json.dumps(data, indent=1) + "\n")
    print(json.dumps(data["source"]["held_out"], indent=1))
    if data["source"]["prior_tables"]:
        print(f"too few samples, the prior kept: {data['source']['prior_tables']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

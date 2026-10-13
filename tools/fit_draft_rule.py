"""Fit the cost-aware draft rule's tables (``tensorfold.drafters.draft_rule``) from a server's draft telemetry.

    python tools/fit_draft_rule.py --log DRAFT_LOG.jsonl... [--traces TRACE_DIR...] --output rule.json
        [--source provenance.json] [--confidence 0.5] [--suffix-depth 8] [--fit-half 0|1]

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
- Cost prior (``cost``): the online model's two fits (``draft_rule.CostModel``), with and without a prompt pass, by
  least squares anchored at the server's prior with each round held within its outlier bound of its prediction (as
  online: a paused round moves the fit boundedly; without the online forgetting): the logged rounds' milliseconds
  less their chain steps' against ``t0 + stream * (live - 1) + row * (R - live) + ctx * sum(rows * context) / 1e5 +
  pass * prompt_rows``, and their chain steps' milliseconds against ``step * steps + step_stream * (stream_steps -
  steps)``; and lambda* = all rounds' tokens over their milliseconds. A round's stream_steps (the streams its steps
  drafted for, summed) is its line's ``chain_rows``; a log written before that field derives it from the round that
  verified those drafts (the next round by ``t``: each drafting call's streams took part in one step a draft past
  their first), and leaves a round out of the step fit where that does not account for its steps. A fit of fewer
  rounds than the online fit's memory (MIN_COST_ROUNDS) keeps the server's prior, said so (``prior_costs``).

Requests are split by a hash of their random id (``draft_rule.half``) into a fitting half and a held-out half for the
report (log-loss, Brier score and reliability by predicted band, against the confidence rule's prior too); the
written tables are then fitted on every sample, or with ``--fit-half H`` on half H's requests alone (the report's
held-out half is then the other: two such files cross-fit an evaluation, ``tools/draft_replay.py --cross-tables``). The file holds numbers only: no token ids or text. Install it with
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
FIT_ITERATIONS = 1000                                          # the bounded cost fit's fixed-point iterations at most
FIT_TOLERANCE = 1e-9                                           # ... until no coefficient moves by more (relative)
# a regime's cost terms are fitted from at least this many rounds (the online fit's memory, 1 / (1 - FORGET): a prior
# should hold no less than the server weighs), else the server's prior stays (said so: ``prior_costs``)
MIN_COST_ROUNDS = round(1 / (1 - draft_rule.FORGET))


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


def mtp_samples(lines: Iterable[dict], fit_half: int = 0) -> dict[str, list[tuple[int, float | None, float | None,
                                                                                  int, bool]]]:
    """(depth, p, m, hit, held out: not in ``fit_half``) for each MTP draft its chain reached, by policy."""

    out: dict[str, list] = {"greedy": [], "sampled": []}
    for line in lines:
        for st in line["streams"]:
            if st["src"] != "mtp" or "greedy" not in st:
                continue
            policy = "greedy" if st["greedy"] else "sampled"
            for j, d in enumerate(st["drafts"]):
                if d["hit"] is None or (j and st["drafts"][j - 1]["label"] != 1):
                    break
                out[policy].append((d["j"], d["p"], _score(d["m"]), d["hit"], draft_rule.half(st["rid"]) != fit_half))
    return out


def suffix_samples_from_traces(traces: Sequence[dict], depth: int, fit_half: int = 0) -> dict[str, list[tuple[int,
                                                                                                         int, int,
                                                                                                         bool]]]:
    """(continuation index, match length, hit, held out: not in ``fit_half``) at every reply position of each
    drafting request."""

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
        h = draft_rule.half(tr["rid"]) != fit_half
        for i in range(1, len(reply)):                   # a round starting with i reply tokens out
            index.sync(_Prefix(reply, i))
            length, tokens = index.propose(min(depth, len(reply) - i))
            for k, t in enumerate(tokens):
                hit = t == reply[i + k]
                out[policy].append((k, length, hit, h))
                if not hit:
                    break
    return out


def suffix_samples_from_log(lines: Iterable[dict], fit_half: int = 0) -> dict[str, list[tuple[int, int, int,
                                                                                            bool]]]:
    out: dict[str, list] = {"greedy": [], "sampled": []}
    for line in lines:
        for st in line["streams"]:
            if st["src"] not in ("suffix", "copy") or "greedy" not in st:
                continue
            policy = "greedy" if st["greedy"] else "sampled"
            for j, d in enumerate(st["drafts"]):
                if d["hit"] is None or d["L"] is None or (j and st["drafts"][j - 1]["label"] != 1):
                    break
                out[policy].append((d["j"], d["L"], d["hit"], draft_rule.half(st["rid"]) != fit_half))
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


def fit_mtp(samples, prior: draft_rule.Predictor, train_only: bool = False) -> tuple[dict, dict]:
    """Per policy: the feature with the lower held-out log-loss, its table fitted on every sample (``train_only``:
    the fitting half's); and the report."""

    tables, report = {}, {}
    for policy, rows in samples.items():
        candidates = {}
        for feature, at in (("p", 1), ("m", 2)):
            usable = [(s[0], s[at], s[3], s[4]) for s in rows
                      if s[at] is not None and not (isinstance(s[at], float) and math.isnan(s[at]))]
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
            tables[policy] = fit_table([(d, s, h) for d, s, h, out in chosen["rows"] if not (train_only and out)],
                                       best, chosen["edges"])
            entry.update({f: {k: v for k, v in c.items() if k not in ("rows", "edges")} for f, c in candidates.items()})
            entry["selected"] = best
        report[policy] = entry
    return tables, report


def fit_suffix(samples, prior: draft_rule.Predictor, train_only: bool = False) -> tuple[dict, dict]:
    tables, report = {}, {}
    for policy, rows in samples.items():
        entry = {"prior": metrics(prior.tables["suffix", policy].chance, [(k, L, h) for k, L, h, out in rows if out])}
        if len(rows) >= 2 * SCORE_BINS:
            train = [(k, L, h) for k, L, h, out in rows if not out]
            entry["fit"] = {"held_out": metrics(fit_table(train, "L", L_EDGES).chance,
                                                [(k, L, h) for k, L, h, out in rows if out]), "samples": len(rows)}
            tables[policy] = fit_table([(k, L, h) for k, L, h, out in rows if not (train_only and out)], "L",
                                       L_EDGES)
        report[policy] = entry
    return tables, report


def bounded_fit(fit: draft_rule.CostFit, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int, int]:
    """The anchored least squares of ``a @ theta ~ b`` with every round held within the online model's outlier bound
    of its own prediction (``draft_rule.OUTLIER_MS``, ``OUTLIER_SHARE``; ``CostFit.observe``), from the prior's
    coefficients (``fit``'s) to the fixed point: (coefficients, iterations, rounds held to the bound). Raises if it
    does not settle within FIT_ITERATIONS."""

    precision = np.asarray(fit.precision0)
    lhs, anchor = a.T @ a + np.diag(precision), precision * np.asarray(fit.theta0)
    theta = np.asarray(fit.theta0)
    for iteration in range(1, FIT_ITERATIONS + 1):
        predicted = a @ theta
        bound = draft_rule.OUTLIER_MS + draft_rule.OUTLIER_SHARE * np.abs(predicted)
        new = np.linalg.solve(lhs, a.T @ (predicted + np.clip(b - predicted, -bound, bound)) + anchor)
        if np.max(np.abs(new - theta)) <= FIT_TOLERANCE * (1.0 + np.max(np.abs(new))):
            predicted = a @ new
            held = np.abs(b - predicted) > draft_rule.OUTLIER_MS + draft_rule.OUTLIER_SHARE * np.abs(predicted)
            return new, iteration, int(np.count_nonzero(held))
        theta = new
    raise ValueError(f"the bounded cost fit did not settle in {FIT_ITERATIONS} iterations")


def stream_steps(ordered: Sequence[dict], i: int) -> int | None:
    """Round ``i``'s chain steps' streams, summed (``ordered``: a run's lines by ``t``, the server's round order): its
    ``chain_rows``, else derived from the next round, which verified what those steps drafted (each drafting call's
    streams, joined prompts' apart, took part in one step a draft past their first); None where that does not
    account for the round's steps."""

    line = ordered[i]
    if "chain_rows" in line:
        return line["chain_rows"]
    if i + 1 >= len(ordered):
        return None
    steps = total = 0
    for join in (False, True):
        drafted = [len(st["drafts"]) for st in ordered[i + 1]["streams"] if st["src"] == "mtp" and st["join"] == join]
        if drafted:
            steps += max(drafted) - 1
            total += sum(n - 1 for n in drafted)
    return total if steps == line["chain_steps"] else None


def _settle(fit: draft_rule.CostFit, x: list, y: list) -> tuple[list[float], dict]:
    if len(x) < MIN_COST_ROUNDS:
        return fit.theta0, {"rounds": len(x), "prior_kept": True}
    a, b = np.asarray(x), np.asarray(y)
    theta, iterations, held = bounded_fit(fit, a, b)
    residual = b - a @ theta
    return [float(v) for v in theta], {"rounds": len(x), "bounded": held, "iterations": iterations,
                                       "rms_ms": round(float(np.sqrt(np.mean(residual ** 2))), 4)}


def fit_cost(lines: Sequence[dict]) -> tuple[dict, dict]:
    """The cost prior (the module's description): each regime's rounds less their chain steps and its chain steps, by
    least squares anchored at the server's prior with its outlier bound (``bounded_fit``; without its forgetting),
    and lambda*; a fit of fewer than MIN_COST_ROUNDS rounds keeps the server's prior (``prior_kept``). The report
    gives each regime's rounds, those held to the bound (``bounded``), the fits' iterations and root-mean-square
    residuals, and its chain steps' (``steps``), with the rounds whose steps' streams could not be derived
    (``underived``)."""

    prior = {**draft_rule.COST_PRIOR, "lambda": draft_rule.LAMBDA_PRIOR}
    ordered = sorted(lines, key=lambda line: line["t"])
    out, report, tokens, ms = {}, {}, 0, 0.0
    for name, mixed in (("unmixed", False), ("mixed", True)):
        rounds = draft_rule.CostFit([prior[name][t] for t in draft_rule.COST_TERMS], draft_rule.COST_SD)
        steps = draft_rule.CostFit([prior[name][t] for t in draft_rule.STEP_TERMS], draft_rule.STEP_SD)
        x, y, sx, sy, underived = [], [], [], [], 0
        for i, line in enumerate(ordered):
            if bool(line["mixed"]) != mixed or not line["streams"]:
                continue
            ctx_rows = sum((st["verified"] + 1) * st["ctx"] for st in line["streams"])
            x.append(draft_rule.features(line["rows"], line["live"], ctx_rows, line["pass_rows"]))
            y.append(line["ms"] - line["chain_ms"])
            if line["chain_steps"]:
                rows = stream_steps(ordered, i)
                if rows is None:
                    underived += 1
                else:
                    sx.append(draft_rule.step_features(line["chain_steps"], rows))
                    sy.append(line["chain_ms"])
        coef, report[name] = _settle(rounds, x, y)
        step_coef, report[name]["steps"] = _settle(steps, sx, sy)
        report[name]["steps"]["underived"] = underived
        out[name] = {t: round(v, 6) for t, v in zip((*draft_rule.COST_TERMS, *draft_rule.STEP_TERMS),
                                                     (*coef, *step_coef))}
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
    half, train_only = args.fit_half or 0, args.fit_half is not None
    mtp_tables, mtp_report = fit_mtp(mtp_samples(lines, half), prior, train_only)
    suffix = suffix_samples_from_traces(traces, args.suffix_depth, half) if traces else \
        suffix_samples_from_log(lines, half)
    suffix_tables, suffix_report = fit_suffix(suffix, prior, train_only)
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
              "confidence_prior": args.confidence, "fit_half": args.fit_half, "prior_tables": unfitted,
              "prior_costs": [f"{name}{'.steps' if part == 'steps' else ''}" for name in ("unmixed", "mixed")
                              for part, entry in (("rounds", cost_report[name]), ("steps", cost_report[name]["steps"]))
                              if entry.get("prior_kept")],
              "held_out": {"mtp": mtp_report, "suffix": suffix_report, "cost": cost_report}}
    if not isinstance(extra, dict) or set(extra) & set(source):     # added provenance never replaces the fit's own
        raise ValueError(f"--source: a JSON object without the fit's own fields {sorted(source)}")
    source.update(extra)
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
    ap.add_argument("--fit-half", type=int, choices=(0, 1),
                    help="fit the written tables on this half's requests alone (cross-fitting; default: all)")
    args = ap.parse_args(argv)
    data = run(args)
    Path(args.output).write_text(json.dumps(data, indent=1) + "\n")
    print(json.dumps(data["source"]["held_out"], indent=1))
    if data["source"]["prior_tables"]:
        print(f"too few samples, the prior kept: {data['source']['prior_tables']}")
    if data["source"]["prior_costs"]:
        print(f"too few rounds, the server's cost prior kept: {data['source']['prior_costs']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

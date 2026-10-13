"""Cost-aware drafting (design 3): acceptance chances, an online round-cost model, the stop rule and the trim.

Drafts never change a reply (keyed sampling: design 2), only how many of its tokens a round verifies, so these
decide nothing but speed. A round's long-run throughput is best when each round maximizes expected tokens minus
lambda* times its expected milliseconds (lambda*: the achievable tokens a millisecond), so a draft is verified when
its chance of being kept, ``c`` (the product of its chain's conditional chances), pays for its row::

    include draft j  <=>  c_j >= lambda* * row_ms                       (j = 0 too: a round may verify no draft)
    chain on        <=>  c_j * abar[j + 1] >= lambda* * (row_ms + step_ms)       (a step it forces alone)
                     or  c_j * abar[j + 1] >= lambda* * (row_ms + STEP_SHARED * step_ms) while another stream forces it

``abar[j]`` is a table row's optimistic (75th percentile) chance, the option value of chaining on. Everything here is
host arithmetic on values the decoder already read back.

``Predictor``: monotone tables (``drafters.calibration.Calibration``: non-decreasing in score within a depth row) by
source (``mtp``: score ``p``, the draft head's probability, or ``m``, the keyed margin; ``suffix``: score ``L``, the
match length, depth = the continuation index) and policy (``greedy``, ``sampled``). A calibration file
(``TENSORFOLD_DRAFT_RULE=<path>``, written by ``tools/fit_draft_rule.py``) holds fitted tables, their provenance and
optionally a fitted cost prior; it holds no token ids or text. Without one, ``Predictor.prior`` builds tables from the
confidence the server was started with that reproduce the confidence rule's decisions at the default costs (see its
docstring).

``CostModel``: a round's milliseconds ``t0 + row * (R - 1) + ctx * sum(rows * context) / 1e5 + step * steps +
pass * prompt_rows``, fitted online for rounds with and without a prompt pass by exponentially forgetting least
squares anchored at a prior (recursive least squares with forgetting; the prior keeps unexcited directions), and
lambda* per region (live streams, a prompt pass, the longest context's bucket) as a ratio of exponential averages of
tokens and milliseconds. Estimates are clamped to sane ranges. Determinism: decisions follow measured wall times, so
a request's round structure is not reproducible run to run; its tokens are.
"""

from __future__ import annotations

import bisect
import heapq
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tensorfold.drafters.calibration import Calibration

ENV = "TENSORFOLD_DRAFT_RULE"
KIND = "tensorfold-draft-rule"
VERSION = 1
FEATURES = {"mtp": ("p", "m"), "suffix": ("L",)}
POLICIES = ("greedy", "sampled")

# The prior's chances (design 3.3, until a fit exists): see ``Predictor.prior``.
PRIOR_HIGH = 0.9            # an MTP draft at or above the confidence
PRIOR_LOW_FIRST = 0.35      # a first MTP draft below it: verified, ends its chain (at the default costs)
PRIOR_LOW_LATER = 0.2       # a later one below it: dropped
PRIOR_COPY = 0.94           # a suffix token after a match of COPY_MATCH+ tokens (as family_depth's copy_rate)
COPY_MATCH = 8              # the needle today's copy drafts need; shorter suffix matches get chance 0 in the prior

STEP_SHARED = 0.15          # the share of an MTP step's time a stream adds to a step another stream forces
BUCKETS = ((50_000, "<50k"), (300_000, "50-300k"), (600_000, "300-600k"), (None, ">600k"))

# Cost prior (ms; design 3.5 defaults): t0 = 32 ms + 2.6 ms host, a verify row 6.5 ms, an MTP step 1.6 ms, the
# indexer ~0.22 ms a row per 100k tokens of context; a round with a prompt pass reads the experts for it anyway, so
# its window rows are cheap, and each prompt row costs about a prefill row (~580 tok/s measured).
COST_PRIOR = {"unmixed": {"t0": 34.6, "row": 6.5, "ctx": 0.22, "step": 1.6, "pass": 0.0},
              "mixed": {"t0": 34.6, "row": 1.0, "ctx": 0.22, "step": 1.6, "pass": 1.7}}
COST_TERMS = ("t0", "row", "ctx", "step", "pass")
COST_SD = (20.0, 5.0, 0.5, 2.0, 2.0)  # the prior's standard deviations (ms): weak, so a few rounds outweigh it
LAMBDA_PRIOR = 2.33 / 50.0  # tokens a ms: production's 2.33 tokens a round at ~50 ms
PRIOR_MS = 50.0             # the lambda prior's weight, in milliseconds of rounds
FORGET = 0.995              # least squares' forgetting a round (memory ~200 rounds)
LAMBDA_RATE = 0.05          # lambda*'s exponential averages: the newest round's weight
ROW_MS = (0.01, 200.0)      # clamps: a verify row's marginal ms
STEP_MS = (0.0, 100.0)      # an MTP step's ms
LAMBDA = (0.002, 2.0)       # tokens a ms


def bucket(context: int) -> str:
    return next(name for limit, name in BUCKETS if limit is None or context < limit)


def greedy(sampling) -> bool:
    """The policy a draft's table follows: greedy as ``MultiDecoder._picks`` decides it."""

    return sampling is None or sampling.temperature <= 0


# -- predictor -----------------------------------------------------------------------------------------------------

class Table:
    """One source and policy's chances: a ``Calibration`` over (depth, score) with its score ``feature`` and each
    depth row's optimistic chance ``abar``."""

    __slots__ = ("feature", "cal", "abar", "count")

    def __init__(self, feature: str, cal: Calibration, abar: Sequence[float], count=None) -> None:
        self.feature, self.cal, self.abar, self.count = feature, cal, tuple(float(a) for a in abar), count

    def chance(self, depth: int, score: float) -> float:
        return self.cal.probability(depth, score)

    def optimistic(self, depth: int) -> float:
        return self.abar[max(0, bisect.bisect_right(self.cal.depth_edges, int(depth)) - 1)]

    def as_dict(self) -> dict:
        out = {"feature": self.feature, "depth_edges": list(self.cal.depth_edges),
               "score_edges": list(self.cal.score_edges), "table": [list(r) for r in self.cal.table],
               "abar": list(self.abar)}
        if self.count is not None:
            out["count"] = self.count
        return out


def _number(x, where: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{where}: a finite number, not {x!r}")
    return float(x)


def _table(data, source: str, policy: str) -> Table:
    where = f"tables.{source}.{policy}"
    if not isinstance(data, dict):
        raise ValueError(f"{where}: an object")
    unknown = set(data) - {"feature", "depth_edges", "score_edges", "table", "abar", "count"}
    if unknown:
        raise ValueError(f"{where}: unknown fields {sorted(unknown)}")
    feature = data.get("feature")
    if feature not in FEATURES[source]:
        raise ValueError(f"{where}.feature: one of {FEATURES[source]}, not {feature!r}")
    depth = data.get("depth_edges")
    if not isinstance(depth, list) or not depth or depth[0] != 0 or \
            any(isinstance(d, bool) or not isinstance(d, int) for d in depth) or \
            any(b <= a for a, b in zip(depth, depth[1:])):
        raise ValueError(f"{where}.depth_edges: ascending integers from 0")
    scores = data.get("score_edges")
    if not isinstance(scores, list):
        raise ValueError(f"{where}.score_edges: a list")
    scores = [_number(x, f"{where}.score_edges") for x in scores]
    if any(b <= a for a, b in zip(scores, scores[1:])):
        raise ValueError(f"{where}.score_edges: strictly ascending finite numbers")
    rows = data.get("table")
    if not isinstance(rows, list) or len(rows) != len(depth) or \
            any(not isinstance(r, list) or len(r) != len(scores) + 1 for r in rows):
        raise ValueError(f"{where}.table: a row a depth bin, a column a score bin")
    table = [[_number(p, f"{where}.table") for p in r] for r in rows]
    for r in table:
        if any(not 0.0 <= p <= 1.0 for p in r):
            raise ValueError(f"{where}.table: chances from 0 to 1")
        if any(b < a for a, b in zip(r, r[1:])):
            raise ValueError(f"{where}.table: each row non-decreasing in score")
    abar = data.get("abar")
    if not isinstance(abar, list) or len(abar) != len(depth):
        raise ValueError(f"{where}.abar: one chance a depth bin")
    abar = [_number(a, f"{where}.abar") for a in abar]
    if any(not 0.0 <= a <= 1.0 for a in abar):
        raise ValueError(f"{where}.abar: chances from 0 to 1")
    count = data.get("count")
    if count is not None and (not isinstance(count, list) or len(count) != len(table) or any(
            not isinstance(r, list) or len(r) != len(scores) + 1 or
            any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in r) for r in count)):
        raise ValueError(f"{where}.count: the fit's samples a bin (non-negative integers, the table's shape)")
    return Table(feature, Calibration(table, depth, scores), abar, count)


def _cost(data) -> dict:
    if not isinstance(data, dict) or set(data) != {"unmixed", "mixed", "lambda"}:
        raise ValueError("cost: {unmixed, mixed, lambda}")
    out = {}
    for name in ("unmixed", "mixed"):
        terms = data[name]
        if not isinstance(terms, dict) or set(terms) != set(COST_TERMS):
            raise ValueError(f"cost.{name}: {{{', '.join(COST_TERMS)}}}")
        out[name] = {t: _number(terms[t], f"cost.{name}.{t}") for t in COST_TERMS}
    lam = _number(data["lambda"], "cost.lambda")
    if not LAMBDA[0] <= lam <= LAMBDA[1]:
        raise ValueError(f"cost.lambda: tokens a ms from {LAMBDA[0]} to {LAMBDA[1]}")
    out["lambda"] = lam
    return out


class Predictor:
    """Acceptance chances by source and policy (a fitted file's, or the confidence rule's prior)."""

    def __init__(self, tables: dict[tuple[str, str], Table], source: dict, cost: dict | None = None) -> None:
        self.tables, self.source, self.cost = tables, source, cost
        self.needs_margins = any(t.feature == "m" for (src, _), t in tables.items() if src == "mtp")

    @classmethod
    def prior(cls, confidence: float) -> Predictor:
        """Tables that reproduce the confidence rule's decisions (design 3.4) at the default costs, until a fit.

        An MTP draft at or above ``confidence`` lands with chance PRIOR_HIGH, so a chain of them is verified to its
        room (up to 9 drafts at the default thresholds 0.30 a row, 0.38 a row and step: lambda* 0.0466 tokens/ms x
        6.5 ms a row, 1.6 ms a step); a first draft below it has PRIOR_LOW_FIRST, verified but ending the chain, and
        a later one PRIOR_LOW_LATER, dropped. A stream alone in its steps keeps exactly those decisions while
        lambda* x row_ms is in (0.253, 0.35] (lambda* 0.039-0.054 at 6.5 ms: production's 0.0466 and session 1's
        0.0526). Elsewhere the rule follows the design: deeper chains where rows are cheap (prompt-pass rounds), a
        low first draft dropped where they are dear, and a low first draft chaining on in a step another stream
        forces (STEP_SHARED) where lambda* x row_ms <= 0.30. A suffix (or copy) token after a match of COPY_MATCH+
        tokens lands with PRIOR_COPY > PRIOR_HIGH, so it wins over any MTP chain, as copies do today; a shorter match
        gets 0 (never verified). ``confidence`` 0 (no rule): every MTP draft gets PRIOR_HIGH."""

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence: a probability from 0 to 1, not {confidence}")
        tables = {}
        for policy in POLICIES:
            if confidence > 0:        # p < confidence <=> p <= the float below it (bisect_left: that column)
                edges, rows = [math.nextafter(confidence, -math.inf)], [[PRIOR_LOW_FIRST, PRIOR_HIGH],
                                                                       [PRIOR_LOW_LATER, PRIOR_HIGH]]
            else:
                edges, rows = [], [[PRIOR_HIGH], [PRIOR_HIGH]]
            tables["mtp", policy] = Table("p", Calibration(rows, [0, 1], edges), [PRIOR_HIGH, PRIOR_HIGH])
            tables["suffix", policy] = Table("L", Calibration([[0.0, PRIOR_COPY]], [0], [COPY_MATCH - 1]),
                                             [PRIOR_COPY])
        return cls(tables, {"prior": "confidence rule", "confidence": confidence})

    @classmethod
    def from_dict(cls, data: Any) -> Predictor:
        if not isinstance(data, dict):
            raise ValueError("a draft rule calibration is a JSON object")
        unknown = set(data) - {"v", "kind", "source", "tables", "cost"}
        if unknown:
            raise ValueError(f"unknown fields {sorted(unknown)}")
        if data.get("v") != VERSION or data.get("kind") != KIND:
            raise ValueError(f'not a version {VERSION} "{KIND}" file')
        if not isinstance(data.get("source"), dict):
            raise ValueError("source: an object (the fit's provenance)")
        tables = data.get("tables")
        if not isinstance(tables, dict) or set(tables) != set(FEATURES):
            raise ValueError(f"tables: one object a source {sorted(FEATURES)}")
        out = {}
        for source in FEATURES:
            if not isinstance(tables[source], dict) or set(tables[source]) != set(POLICIES):
                raise ValueError(f"tables.{source}: one table a policy {list(POLICIES)}")
            for policy in POLICIES:
                out[source, policy] = _table(tables[source][policy], source, policy)
        cost = _cost(data["cost"]) if "cost" in data else None
        return cls(out, data["source"], cost)

    @classmethod
    def load(cls, path: str | Path) -> Predictor:
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except ValueError as error:
            raise ValueError(f"{ENV}={path}: {error}") from None

    @classmethod
    def from_env(cls, confidence: float) -> Predictor:
        path = os.environ.get(ENV, "")
        return cls.load(path) if path else cls.prior(confidence)

    def as_dict(self) -> dict:
        tables: dict = {}
        for (source, policy), table in self.tables.items():
            tables.setdefault(source, {})[policy] = table.as_dict()
        out = {"v": VERSION, "kind": KIND, "source": self.source, "tables": tables}
        if self.cost is not None:
            out["cost"] = self.cost
        return out

    def mtp(self, is_greedy: bool, depth: int, p: float, m: float | None) -> float:
        """An MTP draft's chance given the drafts before it were kept."""

        table = self.tables["mtp", POLICIES[0] if is_greedy else POLICIES[1]]
        score = p if table.feature == "p" else m
        if score is None:
            raise ValueError(f"an MTP draft's chance needs its {table.feature!r}")
        return table.chance(depth, score)

    def suffix(self, is_greedy: bool, k: int, length: int) -> float:
        """A suffix token's chance (the ``k``-th of a match of ``length`` tokens) given the ones before it."""

        return self.tables["suffix", POLICIES[0] if is_greedy else POLICIES[1]].chance(k, length)

    def abar(self, source: str, is_greedy: bool, depth: int) -> float:
        return self.tables[source, POLICIES[0] if is_greedy else POLICIES[1]].optimistic(depth)


# -- cost model ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Thresholds:
    """A stream's thresholds in tokens: a verify row's cost, and a row plus an MTP step forced alone or shared."""

    row: float
    full: float
    shared: float
    step: float                                          # an MTP step alone, in tokens (full - row)


def features(rows: int, ctx_rows: float, steps: int, pass_rows: int) -> np.ndarray:
    return np.asarray([1.0, rows - 1.0, ctx_rows / 1e5, float(steps), float(pass_rows)])


class CostFit:
    """Exponentially forgetting least squares anchored at a prior (a 5-term round cost; ``tools/fit_draft_rule.py``
    solves the same anchored problem over logged rounds without forgetting)."""

    def __init__(self, prior: dict) -> None:
        self.theta0 = np.asarray([prior[t] for t in COST_TERMS], dtype=np.float64)
        self.precision0 = np.diag([1.0 / sd ** 2 for sd in COST_SD])
        self.a = np.zeros((len(COST_TERMS), len(COST_TERMS)))
        self.b = np.zeros(len(COST_TERMS))
        self.theta = self.theta0.copy()
        self.rounds = 0

    def observe(self, x: np.ndarray, ms: float) -> None:
        predicted = float(x @ self.theta)
        bound = 5.0 + 0.5 * abs(predicted)               # robust: an outlier round (a pause) moves it boundedly
        y = predicted + min(bound, max(-bound, ms - predicted))
        self.a = FORGET * self.a + np.outer(x, x)
        self.b = FORGET * self.b + x * y
        self.theta = np.linalg.solve(self.a + self.precision0, self.b + self.precision0 @ self.theta0)
        self.rounds += 1


class CostModel:
    """Round costs and lambda* per region, online (``observe`` once a round)."""

    def __init__(self, prior: dict | None = None) -> None:
        prior = prior or {**COST_PRIOR, "lambda": LAMBDA_PRIOR}
        self.fits = {False: CostFit(prior["unmixed"]), True: CostFit(prior["mixed"])}
        self.lambda0 = prior["lambda"]
        self.regions: dict[tuple[int, bool, str], list[float]] = {}     # region -> [tokens EMA, ms EMA]

    @staticmethod
    def region(live: int, mixed: bool, context: int) -> tuple[int, bool, str]:
        return min(max(1, live), 4), bool(mixed), bucket(context)

    def observe(self, *, rows: int, ctx_rows: float, steps: int, pass_rows: int, mixed: bool, ms: float,
                tokens: int, live: int, context: int) -> None:
        """A round's verify rows, the sum of its rows' contexts, its MTP steps, prompt rows, wall milliseconds and
        emitted tokens (``live`` streams, the longest at ``context``)."""

        if not (math.isfinite(ms) and ms > 0) or rows < 1 or tokens < 0:
            raise ValueError(f"a round's cost observation: rows {rows}, {ms} ms, {tokens} tokens")
        self.fits[bool(mixed)].observe(features(rows, ctx_rows, steps, pass_rows), ms)
        ema = self.regions.setdefault(self.region(live, mixed, context), [self.lambda0 * PRIOR_MS, PRIOR_MS])
        ema[0] += LAMBDA_RATE * (tokens - ema[0])
        ema[1] += LAMBDA_RATE * (ms - ema[1])

    def lam(self, live: int, mixed: bool, context: int) -> float:
        ema = self.regions.get(self.region(live, mixed, context))
        value = self.lambda0 if ema is None else ema[0] / ema[1]
        return min(LAMBDA[1], max(LAMBDA[0], value))

    def row_ms(self, context: int, mixed: bool) -> float:
        theta = self.fits[bool(mixed)].theta
        return min(ROW_MS[1], max(ROW_MS[0], float(theta[1] + theta[2] * context / 1e5)))

    def step_ms(self, mixed: bool) -> float:
        return min(STEP_MS[1], max(STEP_MS[0], float(self.fits[bool(mixed)].theta[3])))

    def thresholds(self, lam: float, context: int, mixed: bool) -> Thresholds:
        row, step = lam * self.row_ms(context, mixed), lam * self.step_ms(mixed)
        return Thresholds(row, row + step, row + STEP_SHARED * step, step)


# -- decisions -----------------------------------------------------------------------------------------------------

def chain_step(c_prev: float, a: float, abar_next: float, more: bool, th: Thresholds) -> tuple[float, bool, bool, bool]:
    """Draft j of a chain: its cumulative chance, whether it is verified, and whether the chain goes on (a step the
    stream would force alone, or one it joins). ``more``: a further draft fits the stream's room."""

    c = c_prev * a
    include = c >= th.row
    ahead = c * abar_next if include and more else -1.0
    return c, include, ahead >= th.full, ahead >= th.shared


def suffix_plan(chances: Sequence[float], th: Thresholds) -> tuple[list[float], float]:
    """A candidate's verified prefix (its tokens' cumulative chances while they pay for their rows) and its value in
    tokens (expected tokens less their rows' cost). It needs no MTP step."""

    kept, c, value = [], 1.0, 0.0
    for a in chances:
        c *= a
        if c < th.row:
            break
        kept.append(c)
        value += c - th.row
    return kept, value


def mtp_value(c0: float, abar: Sequence[float], room: int, th: Thresholds) -> float:
    """The MTP chain's value from its first draft (chance ``c0``), deeper drafts at their optimistic chances, each
    paying a row and the step that drafts it."""

    if c0 < th.row:
        return 0.0
    value, c = c0 - th.row, c0
    for j in range(1, room):
        c *= abar[j]
        if c < th.full:
            break
        value += c - th.full
    return value


def trim(chances: Sequence[Sequence[float]], thresholds: Sequence[float], budget: int) -> list[int]:
    """How many drafts each stream keeps at a round's start: those whose chance pays for its row at the round's
    actual threshold, then the highest chances while they fit ``budget`` rows. Chances never rise along a chain, so
    each stream keeps a prefix (ties: the shallower first, then the earlier stream)."""

    counts = []
    for chain, th in zip(chances, thresholds):
        n = 0
        while n < len(chain) and chain[n] >= th:
            n += 1
        counts.append(n)
    if sum(counts) <= budget:
        return counts
    best = heapq.nsmallest(max(0, budget), ((-chain[j], j, s) for s, chain in enumerate(chances)
                                            for j in range(counts[s])))
    kept = [0] * len(chances)
    for _, j, s in best:
        kept[s] = max(kept[s], j + 1)
    return kept


__all__ = ["COPY_MATCH", "CostFit", "CostModel", "ENV", "Predictor", "STEP_SHARED", "Table", "Thresholds", "bucket",
           "chain_step", "greedy", "mtp_value", "suffix_plan", "trim"]

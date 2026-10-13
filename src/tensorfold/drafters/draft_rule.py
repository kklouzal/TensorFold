"""Cost-aware drafting (design 3): acceptance chances, an online round-cost model, the stop rule and the trim.

Drafts never change a reply (keyed sampling: design 2), only how many of its tokens a round verifies, so these
decide nothing but speed. A round's long-run throughput is best when each round maximizes expected tokens minus
lambda* times its expected milliseconds (lambda*: the achievable tokens a millisecond). Both add up over a round's
parts, so each decision takes what pays at the margin.

A verified draft j of a stream adds ``c_j`` expected tokens (the chance it is kept: the product of its chain's
conditional chances) and one verify row, ``row_ms`` (a draft row's marginal milliseconds at the stream's context)::

    include draft j  <=>  c_j >= lambda* * row_ms                       (j = 0 too: a round may verify no draft)

A batched MTP chain step drafts the next draft of every stream it takes and costs ``step_ms + step_stream_ms * (n -
1)`` milliseconds for n streams (measured). For stream s, the step reveals its next draft's conditional chance ``a``,
and that draft is verified only if it then pays for its row, so the step is worth ``v_s = E[max(0, c_j * a - lambda* *
row_ms(s))]`` tokens to it, over the chances the table gives its next depth (``Table.value``: its fit's samples a
score bin; a table without them, as the prior's, puts all on that depth's optimistic chance ``abar``, so ``v_s =
max(0, c_j * abar[j + 1] - lambda* * row_ms(s))``). A draft the stream does not verify, or one past its room, has no
next. Chances never rise along a chain, so a step that does not pay now pays no better later: this one-step
lookahead is the stopping rule (a monotone problem). Running the step for a set P of streams adds ``sum_P v_s -
lambda* * (step_ms + step_stream_ms * (|P| - 1))``: each member adds ``v_s - lambda* * step_stream_ms`` beyond the
step's fixed ``lambda* * (step_ms - step_stream_ms)``, so the best set takes exactly the streams with ``v_s >= lambda*
* step_stream_ms`` (and ``v_s > 0``), and the step runs when their gains pay the fixed part (``joint_step``)::

    the step runs  <=>  sum over those streams of (v_s - lambda* * step_stream_ms) >= lambda* * (step_ms - step_stream_ms)

Alone (one stream) that is ``v_s >= lambda* * step_ms``: the step pays for itself (with the prior's tables, ``c_j *
abar[j + 1] >= lambda* * (row_ms + step_ms)``). With more streams, a step that no single stream pays for runs when its
streams' values add up to its cost.

``Predictor``: monotone tables (``drafters.calibration.Calibration``: non-decreasing in score within a depth row) by
source (``mtp``: score ``p``, the draft head's probability, or ``m``, the keyed margin; ``suffix``: score ``L``, the
match length, depth = the continuation index) and policy (``greedy``, ``sampled``). A calibration file
(``TENSORFOLD_DRAFT_RULE=<path>``, written by ``tools/fit_draft_rule.py``) holds fitted tables, their provenance and
optionally a fitted cost prior; it holds no token ids or text. Without one, ``Predictor.prior`` builds tables from the
confidence the server was started with that reproduce the confidence rule's decisions at the default costs (see its
docstring).

``CostModel``: two fits a regime (rounds with and without a prompt pass), each exponentially forgetting least squares
anchored at a prior (the prior keeps unexcited directions): a round's milliseconds less its MTP chain steps',
``t0 + stream * (live - 1) + row * (R - live) + ctx * sum(rows * context) / 1e5 + pass * prompt_rows`` for ``live``
streams with ``R`` verify rows in all (a stream's own row and work cost ``stream``, each draft row ``row``: one slope
for every row would fold the streams' fixed cost into the draft rows' price), and its chain steps' milliseconds,
``step * steps + step_stream * (stream_steps - steps)`` (stream_steps: the streams the steps drafted for, summed; each
step timed from its launch to its pick's readback, which it needs anyway). Timing steps apart keeps their cost out of
the round's regression, where it is collinear with the rows the same decisions draft. Each fit holds an observation
within an outlier bound of its prediction (OUTLIER_MS, OUTLIER_SHARE), so a paused round moves it boundedly. lambda* is
kept per region (live streams, a prompt pass, the longest context's bucket) as a ratio of exponential averages of
tokens and milliseconds; a region not seen yet takes its nearest seen region's (as streams join or finish), then every
round's, then the prior. Estimates are clamped to sane ranges. Determinism: decisions follow measured wall times, so a
request's round structure is not reproducible run to run; its tokens are.

Host cost: the decoder runs this between a pick's readback and the next launch, so it is plain Python floats and lists
(a fit's update solves its small system by Cholesky, without numpy's per-call overheads).

Failure: everything here is total over validated state and its callers' contract (tables checked at load, a round's
observation checked before it changes anything, every estimate finite and clamped, a margin given where a table reads
one), so the only exception a decision or an estimate can raise is the interpreter's own exhaustion.
``MultiDecoder`` still contains a failure in one stream's decisions to that stream (``_drafting_failed``).
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

from tensorfold.drafters.calibration import Calibration

ENV = "TENSORFOLD_DRAFT_RULE"
KIND = "tensorfold-draft-rule"
VERSION = 2                 # version 1's cost block had one row slope and no per-stream step cost
MAX_FILE = 16 << 20         # a calibration file's bytes at most (a fit's are kilobytes): bounds the startup's parse
MAX_DEPTH_EDGE = 64         # a table's last depth edge at most (lookups hold a row a depth up to it)
FEATURES = {"mtp": ("p", "m"), "suffix": ("L",)}
POLICIES = ("greedy", "sampled")

# The prior's chances (design 3.3, until a fit exists): see ``Predictor.prior``.
PRIOR_HIGH = 0.9            # an MTP draft at or above the confidence
PRIOR_LOW_FIRST = 0.35      # a first MTP draft below it: verified, ends its chain (at the default costs)
PRIOR_LOW_LATER = 0.2       # a later one below it: dropped
PRIOR_COPY = 0.94           # a suffix token after a match of COPY_MATCH+ tokens (as family_depth's copy_rate)
COPY_MATCH = 8              # the needle today's copy drafts need; shorter suffix matches get chance 0 in the prior

BUCKETS = ((50_000, "<50k"), (300_000, "50-300k"), (600_000, "300-600k"), (None, ">600k"))

# Cost prior (ms) and LAMBDA_PRIOR: tools/fit_draft_rule.py over the calibration draft log of 2026-10-13 (GB10,
# Qwen3.8 Flash Next rotorquant6-norm, build 263d7203, --mtp-drafts 4, --mtp-confidence 0.5, TENSORFOLD_MTP_COPY=1,
# both "draft_rule" arms; 26,859 rounds of 1-4 streams at 30k-300k tokens, 20,154 of them with chain steps; log sha256
# d994602a...97b220c89): unmixed rounds' terms (rms residual 3.42 ms a round, 0.35 ms a round's chain steps). Its
# prompts were cached, so 3 rounds had a prompt pass, too few to fit: mixed rounds' terms are the unmixed fit's, but a
# round with a prompt pass reads the experts for it anyway, so its draft rows are cheap (design 3.5), a stream's own
# work (attention, recurrent state, sampling) is the unmixed stream cost less a draft row's, and a prompt row costs
# about a prefill row.
COST_PRIOR = {"unmixed": {"t0": 38.8, "stream": 9.33, "row": 4.17, "ctx": 0.118, "pass": 0.0,
                          "step": 2.18, "step_stream": 0.263},
              "mixed": {"t0": 38.8, "stream": 5.16, "row": 1.0, "ctx": 0.118, "pass": 1.7,
                        "step": 2.18, "step_stream": 0.263}}
COST_TERMS = ("t0", "stream", "row", "ctx", "pass")   # a round's regression (less its chain steps)
STEP_TERMS = ("step", "step_stream")                  # its chain steps' regression
COST_SD = (20.0, 10.0, 5.0, 0.5, 2.0)  # the priors' standard deviations (ms): weak, so a few rounds outweigh them
STEP_SD = (2.0, 1.0)
# an observation counts at most OUTLIER_MS + OUTLIER_SHARE x |its prediction| off that prediction, online
# (``CostFit.observe``) and offline (tools/fit_draft_rule.py): a paused round moves a fit boundedly
OUTLIER_MS, OUTLIER_SHARE = 5.0, 0.5
COST_LIMIT = 1e6            # a cost file's terms: at most this many ms in magnitude (a fit's predictions stay finite)
LAMBDA_PRIOR = 0.070        # tokens a ms: the calibration log's rounds' (lone streams' 0.056, four streams' 0.10)
PRIOR_MS = 50.0             # the lambda prior's weight, in milliseconds of rounds
FORGET = 0.995              # least squares' forgetting an observation (memory ~200 rounds)
LAMBDA_RATE = 0.05          # lambda*'s exponential averages: the newest round's weight (after the first rounds')
ROW_MS = (0.01, 200.0)      # clamps: a draft row's marginal ms
STEP_MS = (0.0, 100.0)      # an MTP step's ms (a further stream's share of one: from 0 to the step's)
LAMBDA = (0.002, 2.0)       # tokens a ms


def bucket(context: int) -> str:
    for limit, name in BUCKETS[:-1]:
        if context < limit:
            return name
    return BUCKETS[-1][1]


def half(rid: str) -> int:
    """A request's half, 0 or 1, by a stable hash of its random id (the draft log's ``rid``): a fit's held-out report
    (tools/fit_draft_rule.py) and cross-fitted evaluations (tools/draft_replay.py) split requests by it."""

    return hashlib.sha256(rid.encode()).digest()[0] & 1


def greedy(sampling) -> bool:
    """The policy a draft's table follows: greedy as ``MultiDecoder._picks`` decides it."""

    return sampling is None or sampling.temperature <= 0


# -- predictor -----------------------------------------------------------------------------------------------------

class Table:
    """One source and policy's chances: a ``Calibration`` over (depth, score) with its score ``feature``, each depth
    row's optimistic chance ``abar`` and optionally its fit's samples a bin (``count``). Lookups read a depth's row
    directly (``rows``, ``abars``, ``spreads``: one a depth up to the last depth edge, whose row serves every deeper
    one)."""

    __slots__ = ("feature", "cal", "abar", "count", "edges", "rows", "abars", "spreads")

    def __init__(self, feature: str, cal: Calibration, abar: Sequence[float], count=None) -> None:
        self.feature, self.cal, self.abar, self.count = feature, cal, tuple(float(a) for a in abar), count
        self.edges = cal.score_edges
        spreads = []                                     # a depth row's chances, ascending, and the weights from each
        for r, row in enumerate(cal.table):              # on: (chances, sum of weights, sum of weight x chance)
            counts = count[r] if count is not None else None
            total = sum(counts) if counts else 0
            pairs = sorted((p, n / total) for p, n in zip(row, counts) if n) if total else [(self.abar[r], 1.0)]
            weight, mass = [0.0] * (len(pairs) + 1), [0.0] * (len(pairs) + 1)
            for k in range(len(pairs) - 1, -1, -1):
                weight[k] = weight[k + 1] + pairs[k][1]
                mass[k] = mass[k + 1] + pairs[k][1] * pairs[k][0]
            spreads.append((tuple(p for p, _ in pairs), tuple(weight), tuple(mass)))
        index = [max(0, bisect.bisect_right(cal.depth_edges, d) - 1) for d in range(int(cal.depth_edges[-1]) + 1)]
        self.rows = tuple(cal.table[i] for i in index)
        self.abars = tuple(self.abar[i] for i in index)
        self.spreads = tuple(spreads[i] for i in index)

    def chance(self, depth: int, score: float) -> float:
        rows = self.rows
        return (rows[depth] if depth < len(rows) else rows[-1])[bisect.bisect_left(self.edges, score)]

    def optimistic(self, depth: int) -> float:
        abars = self.abars
        return abars[depth] if depth < len(abars) else abars[-1]

    def value(self, depth: int, c: float, row: float) -> float:
        """E[max(0, c * a - row)] for a draft at ``depth`` whose chain's chance so far is ``c``, over the chances
        ``a`` its depth row gives (weighted by the fit's samples, else all on ``abar``): its expected tokens net of
        its row (``row`` tokens) when it is verified only if it pays."""

        if c <= 0.0:
            return 0.0
        spreads = self.spreads
        chances, weight, mass = spreads[depth] if depth < len(spreads) else spreads[-1]
        k = bisect.bisect_left(chances, row / c)        # the chances that pay: a >= row / c
        return max(0.0, c * mass[k] - row * weight[k])

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
            any(b <= a for a, b in zip(depth, depth[1:])) or depth[-1] > MAX_DEPTH_EDGE:
        raise ValueError(f"{where}.depth_edges: ascending integers from 0 to at most {MAX_DEPTH_EDGE}")
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
    terms = (*COST_TERMS, *STEP_TERMS)
    for name in ("unmixed", "mixed"):
        given = data[name]
        if not isinstance(given, dict) or set(given) != set(terms):
            raise ValueError(f"cost.{name}: {{{', '.join(terms)}}}")
        out[name] = {t: _number(given[t], f"cost.{name}.{t}") for t in terms}
        if any(abs(v) > COST_LIMIT for v in out[name].values()) or any(out[name][t] < 0 for t in STEP_TERMS):
            raise ValueError(f"cost.{name}: milliseconds of at most {COST_LIMIT:g} in magnitude, a step's from 0")
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
        # by policy (greedy: True, sampled: False), as the decoder reads them a pick
        self.mtp_tables = {g: tables["mtp", POLICIES[0] if g else POLICIES[1]] for g in (True, False)}
        self.suffix_tables = {g: tables["suffix", POLICIES[0] if g else POLICIES[1]] for g in (True, False)}

    @classmethod
    def prior(cls, confidence: float) -> Predictor:
        """Tables that reproduce the confidence rule's decisions (design 3.4) at the default costs, until a fit.

        An MTP draft at or above ``confidence`` lands with chance PRIOR_HIGH, so a chain of them is verified to its
        room; a first draft below it has PRIOR_LOW_FIRST, verified but ending the chain, and a later one
        PRIOR_LOW_LATER, dropped. A stream alone in its steps keeps exactly those decisions while lambda* x row_ms <=
        PRIOR_LOW_FIRST (0.35) < lambda* x (row_ms + step_ms) / PRIOR_HIGH, for rooms up to 6 drafts (more at lower
        lambda*): at the cost prior's draft row and step (COST_PRIOR, at 30k tokens), lambda* above 0.049 and up to
        0.083 tokens a ms (the calibration log's lone streams: 0.056). Elsewhere the rule follows the design: a low
        first draft chaining on where rows and steps are cheap for the tokens they buy (prompt-pass rounds, long
        contexts' lower lambda*), dropped where rows are dear (several streams' higher lambda*), and chains that pay
        only together going on in a step their streams share (``joint_step``). A suffix (or copy) token after a match
        of COPY_MATCH+ tokens lands with PRIOR_COPY > PRIOR_HIGH, so it wins over any MTP chain, as copies do today; a
        shorter match gets 0 (never verified). ``confidence`` 0 (no rule): every MTP draft gets PRIOR_HIGH."""

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
        """A calibration file, validated (at most MAX_FILE bytes); any failure names the file."""

        try:
            with open(path, "rb") as f:
                raw = f.read(MAX_FILE + 1)
            if len(raw) > MAX_FILE:
                raise ValueError(f"larger than {MAX_FILE} bytes")
            return cls.from_dict(json.loads(raw.decode("utf-8")))
        except (OSError, ValueError, RecursionError) as error:      # RecursionError: nesting past the parser's
            raise ValueError(f"{ENV}={path}: {error}") from error

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

        table = self.mtp_tables[bool(is_greedy)]
        score = p if table.feature == "p" else m
        if score is None:
            raise ValueError(f"an MTP draft's chance needs its {table.feature!r}")
        return table.chance(depth, score)

    def suffix(self, is_greedy: bool, k: int, length: int) -> float:
        """A suffix token's chance (the ``k``-th of a match of ``length`` tokens) given the ones before it."""

        return self.suffix_tables[bool(is_greedy)].chance(k, length)

    def abar(self, source: str, is_greedy: bool, depth: int) -> float:
        return self.tables[source, POLICIES[0] if is_greedy else POLICIES[1]].optimistic(depth)


# -- cost model ----------------------------------------------------------------------------------------------------

def features(rows: int, live: int, ctx_rows: float, pass_rows: int) -> list[float]:
    """A round's terms (COST_TERMS): ``live`` streams with ``rows`` verify rows in all."""

    return [1.0, live - 1.0, float(rows - live), ctx_rows / 1e5, float(pass_rows)]


def step_features(steps: int, stream_steps: int) -> list[float]:
    """A round's chain-step terms (STEP_TERMS): its steps, and the streams they drafted for beyond one a step."""

    return [float(steps), float(stream_steps - steps)]


def solve(m: Sequence[Sequence[float]], diag: Sequence[float], v: Sequence[float]) -> list[float] | None:
    """x with (m + diag(``diag``)) x = v, for a symmetric positive semidefinite ``m`` (its lower triangle is read) and
    a positive ``diag``: Cholesky on the system scaled to a unit diagonal. None when rounding leaves a pivot that is
    not positive or a result that is not finite (singular to working precision: inputs far beyond any round's)."""

    k = len(v)
    scale = []
    for i in range(k):
        d = m[i][i] + diag[i]
        if not 0.0 < d < math.inf:
            return None
        scale.append(1.0 / math.sqrt(d))
    low: list[list[float]] = []
    for i in range(k):
        row, mi, si = [0.0] * k, m[i], scale[i]
        for j in range(i):
            lj = low[j]
            s = mi[j] * si * scale[j]
            for q in range(j):
                s -= row[q] * lj[q]
            row[j] = s / lj[j]
        s = (mi[i] + diag[i]) * si * si
        for q in range(i):
            s -= row[q] * row[q]
        if not s > 0.0:
            return None
        row[i] = math.sqrt(s)
        low.append(row)
    z = []
    for i in range(k):
        s, li = v[i] * scale[i], low[i]
        for q in range(i):
            s -= li[q] * z[q]
        z.append(s / li[i])
    x = [0.0] * k
    for i in range(k - 1, -1, -1):
        s = z[i]
        for q in range(i + 1, k):
            s -= low[q][i] * x[q]
        x[i] = s / low[i][i]
    out = [xi * si for xi, si in zip(x, scale)]
    return out if all(math.isfinite(c) for c in out) else None


class CostFit:
    """Exponentially forgetting least squares anchored at a prior (coefficients ``prior`` with standard deviations
    ``sd``), each observation held within the outlier bound of its prediction; ``tools/fit_draft_rule.py`` solves the
    same anchored, bounded problem over logged rounds, without forgetting. ``coef``: the estimate. An update singular
    to working precision (``solve``; no round's inputs make one) keeps the estimate and is counted (``singular``)."""

    __slots__ = ("theta0", "precision0", "anchor", "a", "b", "coef", "rounds", "singular")

    def __init__(self, prior: Sequence[float], sd: Sequence[float]) -> None:
        k = len(prior)
        self.theta0 = [float(v) for v in prior]
        self.precision0 = [1.0 / s ** 2 for s in sd]
        self.anchor = [p * t for p, t in zip(self.precision0, self.theta0)]     # the prior's pull
        self.a = [[0.0] * k for _ in range(k)]          # the data's forgotten normal equations (lower triangle)
        self.b = [0.0] * k
        self.coef = list(self.theta0)
        self.rounds = 0
        self.singular = 0

    def predict(self, x: Sequence[float]) -> float:
        return sum(c * v for c, v in zip(self.coef, x))

    def observe(self, x: Sequence[float], y: float) -> None:
        predicted = self.predict(x)
        bound = OUTLIER_MS + OUTLIER_SHARE * abs(predicted)    # robust: an outlier (a pause) moves it boundedly
        y = predicted + min(bound, max(-bound, y - predicted))
        a, b = self.a, self.b
        for i, xi in enumerate(x):
            ai = a[i]
            for j in range(i + 1):
                ai[j] = FORGET * ai[j] + xi * x[j]
            b[i] = FORGET * b[i] + xi * y
        coef = solve(a, self.precision0, [bi + pull for bi, pull in zip(b, self.anchor)])
        if coef is None:
            self.singular += 1
        else:
            self.coef = coef
        self.rounds += 1


class CostModel:
    """Round and chain-step costs and lambda* per region, online (``observe`` once a round)."""

    def __init__(self, prior: dict | None = None) -> None:
        prior = prior or {**COST_PRIOR, "lambda": LAMBDA_PRIOR}
        regimes = ((False, "unmixed"), (True, "mixed"))
        self.fits = {mixed: CostFit([prior[name][t] for t in COST_TERMS], COST_SD) for mixed, name in regimes}
        self.steps = {mixed: CostFit([prior[name][t] for t in STEP_TERMS], STEP_SD) for mixed, name in regimes}
        self.lambda0 = prior["lambda"]
        self.regions: dict[tuple[int, bool, str], list[float]] = {}     # region -> [tokens, ms averages, rounds]
        self.overall: list[float] | None = None                         # every round's

    @staticmethod
    def region(live: int, mixed: bool, context: int) -> tuple[int, bool, str]:
        return min(max(1, live), 4), bool(mixed), bucket(context)

    def observe(self, *, rows: int, live: int, ctx_rows: float, pass_rows: int, mixed: bool, ms: float,
                chain_ms: float, chain_steps: int, chain_rows: int, tokens: int, context: int) -> None:
        """A round of ``live`` streams: its verify rows (``rows`` in all), the sum of its rows' contexts, its prompt
        rows, wall milliseconds (``chain_ms`` of them in its ``chain_steps`` MTP chain steps, which drafted for
        ``chain_rows`` streams in all) and emitted tokens (the longest stream at ``context``)."""

        if not (math.isfinite(ms) and ms > 0 and math.isfinite(chain_ms) and 0 <= chain_ms <= ms) or live < 1 or \
                rows < live or tokens < 0 or chain_steps < 0 or chain_rows < chain_steps or \
                (chain_rows and not chain_steps) or not (math.isfinite(ctx_rows) and ctx_rows >= 0) or pass_rows < 0:
            raise ValueError(f"a round's cost observation: {live} streams, rows {rows}, context rows {ctx_rows}, "
                             f"prompt rows {pass_rows}, {ms} ms ({chain_ms} in {chain_steps} steps for {chain_rows} "
                             f"streams), {tokens} tokens")
        mixed = bool(mixed)
        self.fits[mixed].observe(features(rows, live, ctx_rows, pass_rows), ms - chain_ms)
        if chain_steps:
            self.steps[mixed].observe(step_features(chain_steps, chain_rows), chain_ms)
        region = self.region(live, mixed, context)
        ema = self.regions.get(region)
        if ema is None:                                  # a new region starts from its nearest estimate's lambda*
            ema = self.regions[region] = [self._estimate(region) * PRIOR_MS, PRIOR_MS, 0]
        if self.overall is None:
            self.overall = [self.lambda0 * PRIOR_MS, PRIOR_MS, 0]
        for e in (ema, self.overall):                    # the start counts as one round: averages of the first
            e[2] += 1                                    # rounds, then exponential (LAMBDA_RATE)
            rate = max(LAMBDA_RATE, 1.0 / (e[2] + 1))
            e[0] += rate * (tokens - e[0])
            e[1] += rate * (ms - e[1])

    def _estimate(self, region: tuple[int, bool, str]) -> float:
        """lambda* of ``region``: its own, else the nearest live count's with the same prompt pass and context
        bucket (fewer streams on ties), else every round's, else the prior."""

        ema = self.regions.get(region)
        if ema is None:
            live, mixed, name = region
            near = [r for r in self.regions if r[1:] == (mixed, name)]
            if near:
                ema = self.regions[min(near, key=lambda r: (abs(r[0] - live), r[0]))]
            else:
                ema = self.overall
        return self.lambda0 if ema is None else ema[0] / ema[1]

    def lam(self, live: int, mixed: bool, context: int) -> float:
        return min(LAMBDA[1], max(LAMBDA[0], self._estimate(self.region(live, mixed, context))))

    def row_ms(self, context: int, mixed: bool) -> float:
        """A draft row's marginal milliseconds at ``context``."""

        coef = self.fits[bool(mixed)].coef
        return min(ROW_MS[1], max(ROW_MS[0], coef[2] + coef[3] * context / 1e5))

    def step_ms(self, mixed: bool) -> float:
        """An MTP chain step's milliseconds for one stream."""

        return min(STEP_MS[1], max(STEP_MS[0], self.steps[bool(mixed)].coef[0]))

    def step_stream_ms(self, mixed: bool) -> float:
        """The milliseconds each further stream adds to a chain step: from 0 to a step's."""

        return min(self.step_ms(mixed), max(STEP_MS[0], self.steps[bool(mixed)].coef[1]))


# -- decisions -----------------------------------------------------------------------------------------------------

def chain_step(c_prev: float, a: float, table: Table, depth: int, more: bool,
               row: float) -> tuple[float, bool, float | None]:
    """Draft ``depth`` of a chain (conditional chance ``a``; ``table``: its source and policy's): its cumulative
    chance, whether it is verified (``row``: a draft row's cost in tokens), and the next step's value to the stream
    in tokens (``Table.value``; None: the chain cannot go on, because this draft is not verified, or ``more`` says no
    further draft fits the stream's room). ``joint_step`` decides the step."""

    c = c_prev * a
    if c < row:
        return c, False, None
    return c, True, (table.value(depth + 1, c, row) if more else None)


def joint_step(values: Sequence[float], step: float, extra: float) -> list[bool]:
    """Which streams a batched MTP step drafts for, given each candidate's value for it ``values`` (tokens, net of
    its row): those worth something and at least ``extra`` (a further stream's share of a step, in tokens), when their
    gains beyond it pay the rest of the step (``step`` for its first stream); else none (the module's derivation).
    With ``extra`` at most ``step`` (``CostModel.step_stream_ms``) and ``step`` > 0, a candidate alone goes on exactly
    when its value pays ``step``."""

    gain, members = 0.0, []
    for v in values:
        joins = v > 0.0 and v >= extra
        members.append(joins)
        if joins:
            gain += v - extra
    if gain >= step - extra and any(members):
        return members
    return [False] * len(members)


def suffix_plan(chances: Sequence[float], row: float) -> tuple[list[float], float]:
    """A candidate's verified prefix (its tokens' cumulative chances while they pay for their rows, ``row`` tokens
    each) and its value in tokens (expected tokens less their rows' cost). It needs no MTP step."""

    kept, c, value = [], 1.0, 0.0
    for a in chances:
        c *= a
        if c < row:
            break
        kept.append(c)
        value += c - row
    return kept, value


def mtp_value(c0: float, abar: Sequence[float], room: int, row: float, step: float) -> float:
    """The MTP chain's value from its first draft (chance ``c0``), deeper drafts at their optimistic chances, each
    paying a row and the step that drafts it as if alone (``row``, ``step``: tokens; a text needs no step, and
    whether other streams will share the chain's steps is not known when a stream chooses its text)."""

    if c0 < row:
        return 0.0
    full = row + step
    value, c = c0 - row, c0
    for j in range(1, room):
        c *= abar[j]
        if c < full:
            break
        value += c - full
    return value


def trim(chances: Sequence[Sequence[float]], thresholds: Sequence[float]) -> list[int]:
    """How many drafts each stream keeps at a round's start: the leading ones whose chance pays for its row at the
    round's actual threshold (chances never rise along a chain, so that is every one at or above it). No row budget:
    each stream drafts at most its room (the depth at most), so a round's window always fits the buffers, and
    ``stage`` rejects one that would not."""

    counts = []
    for chain, th in zip(chances, thresholds):
        n = 0
        while n < len(chain) and chain[n] >= th:
            n += 1
        counts.append(n)
    return counts


__all__ = ["COPY_MATCH", "CostFit", "CostModel", "ENV", "OUTLIER_MS", "OUTLIER_SHARE", "Predictor", "Table",
           "bucket", "chain_step", "features", "greedy", "half", "joint_step", "mtp_value", "solve", "step_features",
           "suffix_plan", "trim"]

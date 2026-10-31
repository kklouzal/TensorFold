"""Request-keyed standard speculative rejection sampling over immutable categorical rows.

Proposal, acceptance and residual noise have distinct domains. A request's logical proposal
windows belong to the request, never the scheduler. Target probabilities must describe the
actual masked/penalized/filtered sampler; proposal probabilities describe the actual draw.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .exact_sampling import Sampling, uniform

PROPOSAL = 0x3C6EF372FE94F82B
ACCEPT = 0x6A09E667F3BCC909
RESIDUAL = 0xBB67AE8584CAA73B
_MASK = (1 << 64) - 1


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ProbabilityRow:
    """Unique global IDs and rounded normalized FP64 weights, owned by immutable byte buffers.

    Input weights need not sum to one; zero weights are removed. Their order is retained
    for deterministic ties. The categorical law divides stored weights by their exact sum,
    which need not be exactly one. No caller-owned mutable storage survives construction.
    """

    ids: np.ndarray
    probs: np.ndarray

    def __init__(self, ids, probs):
        tokens, weights = np.asarray(ids), np.asarray(probs, dtype=np.float64)
        if (tokens.ndim != 1 or weights.shape != tokens.shape or not tokens.size
                or tokens.dtype.kind not in "iu" or np.any(tokens < 0)
                or np.any(tokens > np.iinfo(np.int64).max)):
            raise ValueError("a probability row needs nonnegative signed64 token IDs and matching weights")
        tokens = tokens.astype(np.int64, copy=False)
        if len(np.unique(tokens)) != len(tokens) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("probability rows need unique IDs and finite nonnegative weights")
        keep = weights > 0
        total = weights.sum(dtype=np.float64)
        if not np.isfinite(total) or total <= 0:
            raise ValueError("a probability row needs positive finite mass")
        tokens, weights = tokens[keep], weights[keep] / total
        represented = weights > 0  # division may underflow a positive input to zero
        tokens, weights = tokens[represented], weights[represented]
        self._store(tokens, weights)

    def _store(self, tokens, weights):
        # Trusted construction only: bytes-backed arrays cannot be made writable again through setflags.
        object.__setattr__(self, "ids", np.frombuffer(tokens.tobytes(), dtype=np.int64))
        object.__setattr__(self, "probs", np.frombuffer(weights.tobytes(), dtype=np.float64))

    def probability(self, token: int) -> float:
        where = np.flatnonzero(self.ids == token)
        return float(self.probs[where[0]]) if len(where) else 0.0

    def __eq__(self, other):
        return isinstance(other, ProbabilityRow) and np.array_equal(self.ids, other.ids) and np.array_equal(self.probs, other.probs)


def proposal_row(values, ids, sampling: Sampling) -> ProbabilityRow:
    """The actual proposal distribution over available candidates (a sparse proposal is valid).

    Unlike target extraction, top-p here is normalized over this explicit candidate set.
    Every proposed token must be drawn from the returned row, not from nominal logits.
    """

    values, ids = np.asarray(values, dtype=np.float64), np.asarray(ids)
    if values.ndim != 1 or values.shape != ids.shape or not values.size:
        raise ValueError("one proposal score per candidate required")
    if np.isnan(values).any() or np.isposinf(values).any() or not np.isfinite(values).any():
        raise ValueError("proposal scores need a finite candidate and may contain only finite scores or -inf")
    if sampling is None or sampling.temperature <= 0:
        raise ValueError("stochastic proposals require positive-temperature sampling")
    order = np.lexsort((ids, -values))
    count = min(len(order), sampling.top_k) if sampling.top_k else len(order)
    order = order[:max(1, count)]
    with np.errstate(over="ignore"):
        scaled = (values[order] - values[order[0]]) / max(float(sampling.temperature), 1e-6)
    weights = np.exp(scaled)
    weights /= weights.sum()
    keep = len(weights)
    if 0 < sampling.top_p < 1:
        keep = int(np.searchsorted(np.cumsum(weights), sampling.top_p)) + 1
    if sampling.min_p > 0:
        keep = min(keep, int((scaled >= scaled[0] + sampling.min_log).sum()))
    return ProbabilityRow(ids[order[:keep]], weights[:keep])


def draw(row: ProbabilityRow, seed: int, position: int, *, domain: int = PROPOSAL) -> int:
    """One categorical Gumbel draw under a stable request/position/domain key."""

    noise = -np.log(-np.log(uniform((seed ^ domain) & _MASK, position, row.ids)))
    return int(row.ids[int(np.argmax(np.log(row.probs) + noise))])


@dataclass(frozen=True, slots=True)
class ChainPacket:
    """Tokens and their actual q rows, each at first_position+j; no mutable launch data."""

    tokens: tuple[int, ...]
    q_rows: tuple[ProbabilityRow, ...]
    first_position: int
    scores: tuple[float, ...] = ()

    def __post_init__(self):
        tokens, rows, scores = tuple(self.tokens), tuple(self.q_rows), tuple(self.scores)
        if len(tokens) != len(rows) or (scores and len(scores) != len(tokens)):
            raise ValueError("one proposal probability row and optional score per token required")
        if (type(self.first_position) is not int or not 0 <= self.first_position <= _MASK
                or len(tokens) > _MASK - self.first_position):
            raise ValueError("proposal positions must fit unsigned64")
        if any(type(t) is not int or t < 0 or not isinstance(q, ProbabilityRow) or q.probability(t) <= 0
               for t, q in zip(tokens, rows)):
            raise ValueError("every proposal must lie in its actual probability row")
        if scores and not np.isfinite(scores).all():
            raise ValueError("proposal scores must be finite")
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "q_rows", rows)
        object.__setattr__(self, "scores", scores)


def node(p: ProbabilityRow, q: ProbabilityRow, children, seed: int, position: int) -> tuple[int, int]:
    """Recursive rejection for children sampled without replacement from q, in draw order.

    Callers own the proposal-order/shape proof. An impossible empty residual fails explicitly;
    it is never replaced with the previous distribution. Chains use exactly one child.
    """

    residual, proposal = p, q
    for index, child in enumerate(children):
        qc, pc = proposal.probability(child), residual.probability(child)
        if qc <= 0:
            raise ValueError("a child must lie in the proposal distribution")
        u = float(uniform((seed ^ ACCEPT) & _MASK, position, np.array([index], dtype=np.int64))[0])
        accepted, grid = _accept(residual, proposal, pc, qc, u)
        if accepted:
            return index, int(child)
        residual = _residual(residual, proposal, grid)
        if index + 1 < len(children):
            proposal = ProbabilityRow(proposal.ids, np.where(proposal.ids == child, 0.0, proposal.probs))
    return -1, draw(residual, seed, position, domain=RESIDUAL)


def _sum_bounds(weights):
    """Outward bounds for a positive binary64 sum under any addition tree.

    At most n rounded additions give relative error gamma_n=n*u/(1-n*u), u=2^-53.
    Adding nonnegative subnormals is exact until the sum is normal. Every bound operation
    is rounded outward; an unusable gamma sends the decision to exact integer arithmetic.
    """
    total = float(weights.sum(dtype=np.float64))
    nu = math.nextafter(len(weights) * 2.**-53, math.inf)
    denominator = math.nextafter(1. - nu, -math.inf)
    if denominator <= 0:
        return 0., math.inf
    gamma = math.nextafter(nu / denominator, math.inf)
    if gamma >= 1:
        return 0., math.inf
    lower_denominator = math.nextafter(1. + gamma, math.inf)
    upper_denominator = math.nextafter(1. - gamma, -math.inf)
    return (math.nextafter(total / lower_denominator, -math.inf),
            math.nextafter(total / upper_denominator, math.inf))


def _product_bounds(bounds, factor):
    if factor == 0:
        return 0., 0.
    lo, hi = bounds[0] * factor, bounds[1] * factor
    return (math.nextafter(lo, -math.inf) if lo > 0 else 0., math.nextafter(hi, math.inf))


def _units(weight, exponent):
    numerator, denominator = float(weight).as_integer_ratio()
    return numerator << (exponent - (denominator.bit_length() - 1))


def _grid(p, q):
    """Exact dyadic integer units/totals; no per-weight Python object list is retained."""
    exponent = max(float(w).as_integer_ratio()[1].bit_length() - 1 for row in (p, q) for w in row.probs)
    return exponent, sum(_units(w, exponent) for w in p.probs), sum(_units(w, exponent) for w in q.probs)


def _accept(p, q, pc, qc, u):
    if pc == 0:  # u and qc are strictly positive, including subnormal qc
        return False, None
    if p is q:
        return True, None
    left = _product_bounds(_product_bounds(_sum_bounds(p.probs), qc), u)
    right = _product_bounds(_sum_bounds(q.probs), pc)
    if left[1] <= right[0]:
        return True, None
    if left[0] > right[1]:
        return False, None
    grid = exponent, pt, qt = _grid(p, q)
    un, ud = u.as_integer_ratio()
    return un * _units(qc, exponent) * pt <= ud * _units(pc, exponent) * qt, grid


def _residual(p, q, grid=None):
    """Condition the exact positive crossproduct residual before rounding to binary64.

    Each normalized integer ratio rounds once, with error at most half its binary64 ulp
    (half the least subnormal for underflow). Tiny coordinates may round to zero, as in
    ordinary FP64 softmax; the largest coordinate is >=1/support and cannot vanish.
    No second floating normalization can erase a valid conditioned residual.
    """
    exponent, pt, qt = _grid(p, q) if grid is None else grid
    # Align sparse q once; full-support fallback never scans q separately for every target token.
    order = np.argsort(q.ids)
    keys = q.ids[order]
    at = np.searchsorted(keys, p.ids).clip(max=len(keys) - 1)
    probability = np.where(keys[at] == p.ids, q.probs[order[at]], 0.0)
    del order, keys, at  # sorting scratch need not overlap the exact integer residual/conditioning
    integers = [0] * len(p.ids)
    for i, (pw, qw) in enumerate(zip(p.probs, probability)):
        integers[i] = max(_units(pw, exponent) * qt - _units(qw, exponent) * pt, 0)
    total = sum(integers)
    if total <= 0:
        raise ValueError("a rejected categorical proposal must have positive exact residual mass")
    weights = np.fromiter((value / total for value in integers), dtype=np.float64, count=len(integers))
    represented = weights > 0
    result = object.__new__(ProbabilityRow)
    result._store(p.ids[represented], weights[represented])
    return result


def residual_workspace(support: int, *, proposal_support: int | None = None) -> int:
    """Additional host bytes for one visited row's exact integer residual, never all packet rows.

    Binary64 weights <=1 use at most2^1074 units. Totals add support.bit_length() bits;
    a sum of crossproducts uses at most2148+2*support.bit_length()+1 bits. Add53 bits
    for an exact acceptance numerator and one runtime digit for arithmetic allocation.
    Sixteen scalar integer slots cover grid/ratio operands, two product chains, subtraction,
    accumulation and division temporaries; these phases do not stack over nodes/rows.
    Existing caller charges cover alignment/FP64 arrays; this adds only bigint/list/row metadata.
    """
    import sys

    if proposal_support is None:
        proposal_support = support
    if any(type(n) is not int or n < 1 for n in (support, proposal_support)):
        raise ValueError("positive exact residual support required")
    bits = 2148 + 2 * max(support, proposal_support).bit_length() + 1 + 53 + sys.int_info.bits_per_digit
    integer = sys.getsizeof((1 << bits) - 1)
    pointer = sys.getsizeof((None,)) - sys.getsizeof(())
    row = ProbabilityRow([0], [1.])
    header = sys.getsizeof(row) + sum(sys.getsizeof(a) + sys.getsizeof(a.base) - a.nbytes for a in (row.ids, row.probs))
    return sys.getsizeof([]) + support * (pointer + integer) + 16 * integer + header


def verify(chain: ChainPacket, p_rows, seed: int, *, room: int | None = None, eos=()) -> tuple[list[int], int]:
    """Accepted target input path (root0 included), plus the pending terminal output token.

    Room truncates the defined rejection output, never proposal existence. An accepted child
    in the last output slot is returned as the pending terminal, excluded from the commit path.
    None p rows are permitted only at descendants unreachable because grammar
    pruned the input path or output room terminates verification first.
    """

    if len(p_rows) != len(chain.tokens) + 1:
        raise ValueError("one target probability row per anchor/proposal input required")
    if room is not None and (type(room) is not int or room < 1):
        raise ValueError("rejection output room must be a positive integer")
    path = [0]
    for index, (token, q) in enumerate(zip(chain.tokens, chain.q_rows)):
        p = p_rows[index]
        if p is None:
            raise ValueError("a visited rejection row needs its target distribution")
        accepted, terminal = node(p, q, [token], seed, chain.first_position + index)
        if accepted < 0 or token in eos or (room is not None and len(path) >= room):
            return path, terminal
        path.append(index + 1)
    p = p_rows[len(chain.tokens)]
    if p is None:
        raise ValueError("a visited bonus row needs its target distribution")
    return path, draw(p, seed, chain.first_position + len(chain.tokens), domain=0)

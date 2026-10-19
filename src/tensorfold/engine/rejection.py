"""Request-keyed standard speculative rejection sampling over immutable categorical rows.

Proposal, acceptance and residual noise have distinct domains. A request's logical proposal
windows belong to the request, never the scheduler. Target probabilities must describe the
actual masked/penalized/filtered sampler; proposal probabilities describe the actual draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exact_sampling import Sampling, uniform

PROPOSAL = 0x3C6EF372FE94F82B
ACCEPT = 0x6A09E667F3BCC909
RESIDUAL = 0xBB67AE8584CAA73B
_MASK = (1 << 64) - 1


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ProbabilityRow:
    """Unique global IDs and positive normalized FP64 weights, owned by immutable byte buffers.

    Input weights need not sum to one; zero weights are removed. Their order is retained
    for deterministic ties. No caller-owned mutable storage survives construction.
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
        # bytes-backed arrays cannot be made writable again through setflags.
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
        # Avoid underflow in u*qc falsely accepting p(child)==0 for subnormal q.
        if pc >= qc or float(uniform((seed ^ ACCEPT) & _MASK, position, np.array([index], dtype=np.int64))[0]) <= pc / qc:
            return index, int(child)
        # Align sparse q once. A per-target-token scan of q would make complete-support fallback O(P*Q).
        order = np.argsort(proposal.ids)
        keys = proposal.ids[order]
        at = np.searchsorted(keys, residual.ids).clip(max=len(keys) - 1)
        probability = np.where(keys[at] == residual.ids, proposal.probs[order[at]], 0.0)
        weights = np.maximum(residual.probs - probability, 0.0)
        residual = ProbabilityRow(residual.ids, weights)
        if index + 1 < len(children):
            proposal = ProbabilityRow(proposal.ids, np.where(proposal.ids == child, 0.0, proposal.probs))
    return -1, draw(residual, seed, position, domain=RESIDUAL)


def verify(chain: ChainPacket, p_rows, seed: int, *, room: int | None = None, eos=()) -> tuple[list[int], int]:
    """Accepted target input path (root0 included), plus the pending terminal output token.

    Room truncates the defined rejection output, never proposal existence. An accepted child
    in the last output slot is returned as the pending terminal, excluded from the commit path.
    None p rows are permitted only at unreachable grammar-pruned descendants.
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

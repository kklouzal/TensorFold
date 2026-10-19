"""The DFlash2 tree policy: which of the drafter's candidates a round verifies, and where they hang."""

from __future__ import annotations

import heapq
from typing import NamedTuple

import numpy as np

from tensorfold.engine.exact_sampling import Sampling, uniform_rows

# the Metal engine's values; the target's keyed Gumbel noise is known ahead, so ``noise`` weighs it into draft scores
POLICY = {"edge": 0.6, "noise": 0.7, "scale": 1.5, "branch": 4, "nucleus": False}
HOT_SPANS = ((0, 98304), (248032, 248320))


def draft_spans(vocab: int, policy: str = "truncated") -> tuple[tuple[int, int], ...]:
    """Output-token ranges for the original truncated proposal head or the full vocabulary."""

    if type(vocab) is not int or vocab < 1 or policy not in ("truncated", "full"):
        raise ValueError("draft vocabulary needs a positive vocabulary size and truncated or full policy")
    return ((0, vocab),) if policy == "full" else tuple((a, min(b, vocab)) for a, b in HOT_SPANS if a < vocab)


class RejectionResult(NamedTuple):
    path: list[int]
    terminal: int


def draft_limit(metadata: dict) -> int | None:
    """Only an explicitly declared positive companion position limit is authoritative; target YaRN cannot extend it."""

    if "max_position_embeddings" not in metadata:
        return None
    value = metadata["max_position_embeddings"]
    if type(value) is not int or value < 1:
        raise ValueError("DFlash2 max_position_embeddings must be a positive integer when declared")
    return value


def draft_block(metadata: dict) -> int:
    """The declared training width controls stochastic shape/admission; only a positive exact integer is valid."""

    value = metadata["dflash_config"].get("block_size", 8)
    if type(value) is not int or value < 1:
        raise ValueError("DFlash2 dflash_config.block_size must be a positive integer when declared")
    return value


def copy_packet(tokens, position):
    """Deterministic copy proposals have point-mass q, not the companion model's distribution."""

    from tensorfold.engine.rejection import ChainPacket, ProbabilityRow

    return ChainPacket(tuple(tokens), tuple(ProbabilityRow(np.array([t], dtype=np.int64),
                                                          np.array([1.0], dtype=np.float64)) for t in tokens), position)


def chain_probabilities(logits, sampling, rows, *, plan=None):
    """Original-chain target row alignment after physical output-limit/grammar pruning.

    Missing descendants are unreachable: an invalid child is rejected by its retained parent's masked p, or the
    final output slot returns that child as pending without visiting its absent input row. Never shorten q.
    """

    from tensorfold.cuda.sampling import effective_rows, effective_shards

    result = effective_rows(logits, sampling) if plan is None else effective_shards(logits, sampling, plan=plan)
    if len(result) > rows:
        raise ValueError("target rows exceed their original stochastic chain")
    return [*result, *([None] * (rows - len(result)))]

def best_first(cands: np.ndarray, unary: np.ndarray, projected: np.ndarray,
               pred: np.ndarray, succ: np.ndarray, anchor: int, max_nodes: int,
               sampling: Sampling | None, first_position: int,
               history=None) -> tuple[list[int], list[int], list[float]]:
    """Metal's best-first lattice policy on the host: nodes, parents and path scores in pop order, parents first.

    With a penalty ``history`` (tensorfold.engine.penalties.History) a node's candidate scores are penalized as the
    target's verify row will be: by the reply and the tokens on the node's path."""

    depth_count = cands.shape[0]
    temp = max(float(sampling.temperature), 1e-6) if sampling is not None else 1.0
    noise = None
    if sampling is not None:
        positions = first_position + np.arange(depth_count)
        noise = -np.log(-np.log(uniform_rows(sampling.seed, positions, cands)))
    successor = [succ[cands[d]].astype(np.float64) for d in range(depth_count)]
    heap: list[tuple[float, int, int, int]] = []
    tokens: list[int] = []
    parents: list[int] = []
    paths: list[list[int]] = []                     # each node's tokens from the first depth down to it

    def expand(token: int, depth: int, parent: int, path_score: float) -> None:
        edge = successor[depth] @ (pred[token].astype(np.float64) * projected[depth])
        values = unary[depth] + POLICY["edge"] * edge
        if history is not None:
            values = history.candidates(values[None], cands[depth][None],
                                        paths[parent] if parent >= 0 else ())[0].astype(np.float64)
        values = values / temp
        if noise is not None:
            if POLICY["nucleus"] and sampling.top_p < 1:
                probs = np.exp(values - values.max())
                probs /= probs.sum()
                order = np.argsort(-probs)
                keep = int(np.searchsorted(np.cumsum(probs[order]), sampling.top_p)) + 1
                values = np.full_like(values, -np.inf)
                values[order[:keep]] = ((unary[depth] + POLICY["edge"] * edge) / temp)[order[:keep]]
            values = values + POLICY["noise"] * noise[depth]
        values = values / POLICY["scale"]
        values -= values.max()
        logp = values - np.log(np.exp(values).sum())
        for i in np.argsort(-logp)[:POLICY["branch"]]:
            if not np.isfinite(logp[i]):
                break
            heapq.heappush(heap, (path_score - float(logp[i]), parent,
                                  int(cands[depth, i]), depth))

    expand(anchor, 0, -1, 0.0)
    scores: list[float] = []
    while heap and len(tokens) < max_nodes:
        score, parent, token, depth = heapq.heappop(heap)
        me = len(tokens)
        tokens.append(token)
        parents.append(parent)
        scores.append(score)
        if history is not None:
            paths.append((paths[parent] if parent >= 0 else []) + [token])
        if depth + 1 < depth_count:
            expand(token, depth + 1, me, score)
    return tokens, parents, scores


def allocate(scores: list[list[float]], rows: int, tokens: float, cost, overhead: float = 0.0) -> list[int]:
    """How many of each tree's pop-order nodes (at least one) a round verifies, for the most expected tokens per ms."""

    counts = [1 if s else 0 for s in scores]
    rows += sum(counts)
    tokens += sum(float(np.exp(-s[0])) for s in scores if s)
    heap = [(s[1], i, 1) for i, s in enumerate(scores) if len(s) > 1]
    heapq.heapify(heap)
    taken: list[int] = []
    best, best_k = tokens / (cost(rows) + overhead), 0
    while heap:
        score, i, k = heapq.heappop(heap)
        taken.append(i)
        tokens += float(np.exp(-score))
        rate = tokens / (cost(rows + len(taken)) + overhead)
        if rate > best:
            best, best_k = rate, len(taken)
        if k + 1 < len(scores[i]):
            heapq.heappush(heap, (scores[i][k + 1], i, k + 1))
    for i in taken[:best_k]:
        counts[i] += 1
    return counts

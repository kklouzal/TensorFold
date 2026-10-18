"""Draft proposals under the target's keyed sampling noise (Gumbel-coupled drafting).

The target samples the token at absolute position ``n`` as the argmax of its filtered, temperature-scaled scores plus
the keyed Gumbel noise ``-log(-log(uniform(seed, n, id)))`` (``exact_sampling.choose_rows``). A drafter that knows the
request's ``Sampling`` can apply the same rule to its own scores with the same noise: its proposal for ``n`` is the
target's own choice whenever the two filtered distributions agree, and agrees with it with probability at least
``(1 - TV) / (1 + TV)`` otherwise (Gumbel-max coupling without communication: Daliri, Musco and Suresh 2024,
arXiv 2408.07978). Further branches of a tree node are the next-best perturbed scores: the Gumbel-top-k trick samples
the drafter's distribution without replacement (Kool, van Hoof and Welling 2019), so a node's ``b`` branches hold the
target's choice with the list-matching probability of Rowan, Phan and Khisti 2025 (arXiv 2506.05632).

Verification still accepts a draft only when it equals the target's keyed choice at its row, so drafted output is the
serial output token for token; coupling changes acceptance only. All math is FP64 on the host over the candidates the
GPU read back (``tensorfold.cuda.speculative``); drafts need not match the target's rounding, only its rule.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from tensorfold.engine.exact_sampling import Sampling, uniform_rows

# Temperature of the softmax over a node's perturbed draft scores that estimates how likely each branch is the
# target's choice (the DFlash2 tree's calibrated scale); it orders and budgets draft rows, never output.
CONFIDENCE_SCALE = 1.5


def filtered(values: np.ndarray, ids: np.ndarray, s: Sampling,
             norms: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The target's rule on candidate rows: ``(ids, scaled)`` [rows, k] in (value desc, id asc) order.

    ``scaled`` is value / temperature in FP64, ``-inf`` where top-k, top-p or min-p drops the candidate, exactly as
    ``choose_rows`` cuts a row. With top-k off the nucleus is a share of the complete row's mass: ``norms`` holds each
    row's log-sum-exp of (scores / temperature) over the whole vocabulary (required when top_p < 1). Candidates beyond
    the read count are dropped, so a nucleus wider than the candidates is truncated (a draft-only approximation).
    """

    if values.shape != ids.shape or values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("one candidate id per score, in nonempty rows, required")
    rows, width = ids.shape
    order = np.lexsort((ids, -values), axis=-1)
    k = max(1, min(int(s.top_k) if s.top_k else width, width))
    ids = np.take_along_axis(ids, order, axis=-1)[:, :k]
    scaled = np.take_along_axis(values, order, axis=-1)[:, :k].astype(np.float64) / max(float(s.temperature), 1e-6)
    if 0.0 < s.top_p < 1.0:
        if s.top_k:
            probs = np.exp(scaled - scaled[:, :1])
            probs /= probs.sum(axis=-1, keepdims=True)
        else:
            if norms is None or np.shape(norms) != (rows,):
                raise ValueError("top_k off with top_p < 1 needs each complete row's log-sum-exp")
            probs = np.exp(scaled - np.asarray(norms, dtype=np.float64)[:, None])
        keep = (np.cumsum(probs, axis=-1) < s.top_p).sum(axis=-1) + 1
        scaled[np.arange(k)[None, :] >= keep[:, None]] = -np.inf
    if s.min_p > 0.0:
        scaled[scaled < scaled[:, :1] + s.min_log] = -np.inf
    return ids, scaled


def coupled(values: np.ndarray, ids: np.ndarray, positions: Sequence[int], s: Sampling, *, branch: int = 1,
            norms: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Each row's ``branch`` best draft tokens under the target's keyed noise at that row's position.

    Returns ``(picks, logconf)`` [rows, branch]: picks in decreasing perturbed score (-1 past the row's kept
    candidates) and the log of each pick's share of a softmax of the perturbed scores at ``CONFIDENCE_SCALE``
    (-inf for -1). Row ``r``'s first pick is ``choose_rows`` of the same candidates at ``positions[r]``.
    """

    if type(branch) is not int or branch < 1:
        raise ValueError("a positive branch count is required")
    if len(positions) != ids.shape[0]:
        raise ValueError("one keyed position per candidate row required")
    ids, scaled = filtered(values, ids, s, norms)
    score = scaled - np.log(-np.log(uniform_rows(s.seed, np.asarray(positions, dtype=np.uint64), ids)))
    order = np.argsort(-score, axis=-1, kind="stable")[:, :branch]   # first maximum, as choose_rows' argmax
    top = np.take_along_axis(score, order, axis=-1)
    picks = np.where(np.isfinite(top), np.take_along_axis(ids, order, axis=-1), -1)
    finite = np.isfinite(score)
    tempered = np.where(finite, score / CONFIDENCE_SCALE, -np.inf)
    peak = tempered.max(axis=-1, keepdims=True)
    log_norm = peak + np.log(np.where(finite, np.exp(tempered - peak), 0.0).sum(axis=-1, keepdims=True))
    logconf = np.where(picks >= 0, top / CONFIDENCE_SCALE - log_norm, -np.inf)
    if branch > picks.shape[1]:
        pad = branch - picks.shape[1]
        picks = np.pad(picks, ((0, 0), (0, pad)), constant_values=-1)
        logconf = np.pad(logconf, ((0, 0), (0, pad)), constant_values=-np.inf)
    return picks.astype(np.int64), logconf


__all__ = ["CONFIDENCE_SCALE", "coupled", "filtered"]

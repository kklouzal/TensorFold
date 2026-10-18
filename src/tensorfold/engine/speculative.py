"""Draft proposals under the target's keyed sampling noise (Gumbel-coupled drafting).

The target samples the token at absolute position ``n`` as the argmax of its filtered, temperature-scaled scores plus
the keyed Gumbel noise ``-log(-log(uniform(seed, n, id)))`` (``exact_sampling.choose_rows``). A drafter that knows the
request's ``Sampling`` can apply the same rule to its own scores with the same noise: its proposal for ``n`` is the
target's own choice whenever the two filtered distributions agree, and agrees with it with probability at least
``(1 - TV) / (1 + TV)`` otherwise (Gumbel-max coupling without communication: Daliri, Musco and Suresh 2024,
arXiv 2408.07978). A drafter whose distribution is miscalibrated can be sharpened before the noise is added; at the
limit its proposals are its argmax.

Verification still accepts a draft only when it equals the target's keyed choice at its row, so drafted output is the
serial output token for token; coupling changes acceptance only. All math is FP64 on the host over candidate rows
(values and ids, the complete row's log-sum-exp for a top-k-off nucleus); drafts need not match the target's rounding,
only its rule. MTP drafters draw with the target's rule itself (``exact_sampling.choose_rows``); GB10 card-sampling
sweeps (2026-10-18) measured the DFlash block drafters' argmax at least as accepted as these picks at any sharpening,
so those propose their argmax (``tensorfold.cuda.speculative``) and this rule serves measurements of other drafters.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from tensorfold.engine.exact_sampling import Sampling, uniform_rows

# Temperature of the softmax over a row's perturbed draft scores that estimates how likely the pick is the target's
# choice (the DFlash2 tree's calibrated scale); it orders and budgets draft rows, never output.
CONFIDENCE_SCALE = 1.5


def filtered(values: np.ndarray, ids: np.ndarray, s: Sampling,
             norms: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The target's rule on candidate rows: ``(ids, scaled)`` [rows, k] in (value desc, id asc) order.

    ``scaled`` is value / temperature in FP64, ``-inf`` where top-p or min-p drops the candidate (top-k keeps the first
    k columns), exactly as ``choose_rows`` cuts a row. With top-k off the nucleus is a share of the complete row's
    mass: ``norms`` holds each row's log-sum-exp of (scores / temperature) over the whole vocabulary (required when
    top_p < 1). Candidates beyond the read count are dropped, so a nucleus wider than the candidates is truncated (a
    draft-only approximation).
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


def coupled(values: np.ndarray, ids: np.ndarray, positions: Sequence[int], s: Sampling, *,
            norms: np.ndarray | None = None, sharpen: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Each row's draft token under the target's keyed noise at that row's position.

    Returns ``(picks, logconf)`` [rows]: the picks and the log of each pick's share of a softmax of the row's perturbed
    scores at ``CONFIDENCE_SCALE``. With ``sharpen`` 1 and top-k on, row ``r``'s pick is ``choose_rows`` of the same
    candidates at ``positions[r]`` (top-k off: the nucleus is cut by the complete-row ``norms``). ``sharpen`` < 1
    divides the kept scaled scores by it before the noise is added: a drafter calibration (its distribution raised to
    the power 1 / sharpen within the target's kept set; toward 0 the picks approach the drafter's argmax). It changes
    proposals only, never output.
    """

    if len(positions) != ids.shape[0]:
        raise ValueError("one keyed position per candidate row required")
    if not (isinstance(sharpen, (int, float)) and math.isfinite(sharpen) and sharpen > 0):
        raise ValueError("draft sharpening must be a positive finite number")
    ids, scaled = filtered(values, ids, s, norms)
    if sharpen != 1.0:
        scaled = scaled / float(sharpen)
    score = scaled - np.log(-np.log(uniform_rows(s.seed, np.asarray(positions, dtype=np.uint64), ids)))
    best = np.argmax(score, axis=-1)                    # the first maximum, as choose_rows' argmax
    rows = np.arange(len(best))
    # every row keeps its top candidate, so the maximum is finite; dropped candidates add nothing to the softmax
    tempered = score / CONFIDENCE_SCALE
    peak = tempered[rows, best]
    log_norm = peak + np.log(np.exp(tempered - peak[:, None]).sum(axis=-1))
    return ids[rows, best].astype(np.int64), tempered[rows, best] - log_norm


__all__ = ["CONFIDENCE_SCALE", "coupled", "filtered"]

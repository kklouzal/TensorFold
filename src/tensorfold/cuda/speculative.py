"""The GPU half of coupled drafting: one candidate packet per drafter launch, read back once.

A block drafter's logits rows go through ``launch``: the rows of sampled streams get their top candidates (and, for a
top-k-off nucleus, each row's log-sum-exp at its stream's temperature) packed into one int64 tensor on the drafter's
stream, with no host wait. ``chain`` turns a stream's read-back rows into its coupled chain of proposals with
``tensorfold.engine.speculative.coupled``. Greedy streams keep their callers' argmax proposals unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, Sampling
from tensorfold.engine.speculative import coupled

NUCLEUS_DRAFT = 256     # candidates a top_k-off draft reads per row: a nucleus wider than this is truncated


def sampled(sampling: Sampling | None) -> bool:
    return sampling is not None and sampling.temperature > 0


def count(samplings: Sequence[Sampling | None], width: int) -> int:
    """Candidates per row covering every sampled stream's rule: top_k + MARGIN (MARGIN keeps boundary ties), or
    ``NUCLEUS_DRAFT`` with top_k off; 0 when every stream is greedy."""

    need = [int(s.top_k) + MARGIN if s.top_k else NUCLEUS_DRAFT for s in samplings if sampled(s)]
    return min(width, max(need)) if need else 0


def launch(logits: torch.Tensor, samplings: Sequence[Sampling | None], rows: int) -> torch.Tensor | None:
    """Queue the candidate packet of ``logits`` [len(samplings) * rows, V] (stream-major rows), or None if all greedy.

    The packet is [N, 2C + 1] int64: C raw float32 score words, C column ids, then the raw float64 word of the row's
    log-sum-exp of (scores / temperature) where its stream's nucleus needs one (0 otherwise).
    """

    if logits.ndim != 2 or logits.shape[0] != len(samplings) * rows:
        raise ValueError("draft candidates need stream-major rows, one block of rows per stream")
    c = count(samplings, logits.shape[1])
    if not c:
        return None
    scores = logits.float()
    values, cols = torch.topk(scores, c, dim=-1, sorted=False)
    norms = torch.zeros((scores.shape[0], 1), dtype=torch.float64, device=scores.device)
    nucleus = [i for i, s in enumerate(samplings) if sampled(s) and not s.top_k and 0.0 < s.top_p < 1.0]
    if nucleus:
        inverse = torch.tensor([1.0 / max(float(samplings[i].temperature), 1e-6) for i in nucleus],
                               dtype=torch.float64, device=scores.device).repeat_interleave(rows)[:, None]
        at = torch.tensor([i * rows + r for i in nucleus for r in range(rows)], device=scores.device)
        norms[at] = torch.logsumexp(scores[at].double() * inverse, dim=-1, keepdim=True)
    return torch.cat([values.contiguous().view(torch.int32).to(torch.int64), cols, norms.view(torch.int64)], dim=1)


def read(packet: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The host copy of a launched packet: (values float32, column ids int64, norms float64)."""

    words = packet.cpu().numpy()
    c = (words.shape[1] - 1) // 2
    values = words[:, :c].astype(np.int32).view(np.float32)
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("draft scores contain NaN or positive infinity")
    return values, words[:, c:2 * c], np.ascontiguousarray(words[:, 2 * c]).view(np.float64)


def chain(host: tuple[np.ndarray, np.ndarray, np.ndarray], start: int, rows: int, first_position: int,
          sampling: Sampling, id_map: np.ndarray | None = None) -> tuple[list[int], list[float]]:
    """A stream's coupled chain: row ``start + j`` proposes the token at absolute position ``first_position + j``.

    Returns (tokens, path scores): scores are cumulative ``-log`` confidences, nondecreasing like the argmax path's.
    ``id_map`` maps head columns to global token ids (a drafter head over a vocabulary subset).
    """

    values, cols, norms = (part[start:start + rows] for part in host)
    ids = id_map[cols] if id_map is not None else cols
    picks, logconf = coupled(values, ids, first_position + np.arange(rows, dtype=np.uint64), sampling,
                             norms=norms)
    return [int(t) for t in picks[:, 0]], np.cumsum(-logconf[:, 0]).tolist()


__all__ = ["NUCLEUS_DRAFT", "chain", "count", "launch", "read", "sampled"]

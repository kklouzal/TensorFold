"""Private teacher-forced scoring: target log-probabilities from (tensor-parallel) vocabulary-shard logits.

Only a request carrying ``tensorfold.engine.probabilities.TeacherForcing`` reaches this module (the server's
private ``/v1/tensorfold/score`` route, off by default). Every rank calls :func:`shard_logprobs` with the same rows
in the same order: it is one all-gather of a fixed per-row packet. Each rank packs its shard's FP32 maximum, its
FP32 sum of ``exp(x - max)``, the target's logit when this rank owns the target ID (else -inf) and its top
``top`` scores with their global IDs; the host combines the packets in float64. The scores are the head's own FP32
logits as served, so a numerics change anywhere before the head shows up here.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch

HEAD = 3  # max, sum of exp(x - max), owned target logit


@torch.no_grad()
def shard_logprobs(w, logits: torch.Tensor, targets: Sequence[int], top: int, rows: Sequence[int] | None = None):
    """Rows of FP32 shard logits -> (target logprobs, top IDs, top logprobs), identical on every rank."""

    if logits.dtype != torch.float32 or not logits.is_cuda or logits.ndim != 2:
        raise ValueError("teacher scoring needs CUDA FP32 head logits")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("teacher scoring reads logits on the host and cannot run inside a CUDA graph capture")
    comm = w.comm
    offset = int(w.meta.get("vocab_offset", 0)) if comm is not None else 0
    if rows is not None:
        logits = logits.index_select(0, torch.tensor(list(rows), dtype=torch.long, device=logits.device))
    n, width = logits.shape
    if n != len(targets) or n == 0:
        raise ValueError("teacher scoring needs one target per scored row")
    if type(top) is not int or not 0 <= top <= 20:
        raise ValueError("teacher scoring returns at most 20 alternatives")
    device = logits.device
    local = torch.tensor([int(t) for t in targets], dtype=torch.long, device=device) - offset
    owned = (local >= 0) & (local < width)
    packet = torch.full((n, HEAD + 2 * top), -math.inf, dtype=torch.float32, device=device)
    peak = logits.amax(dim=1)
    packet[:, 0] = peak
    packet[:, 1] = torch.exp(logits - peak[:, None]).sum(dim=1)
    target = logits.gather(1, local.clamp(0, width - 1)[:, None])[:, 0]
    packet[:, 2] = torch.where(owned, target, torch.full_like(target, -math.inf))
    k = min(top, width)
    if k:
        values, cols = logits.topk(k, dim=1, sorted=True)
        packet[:, HEAD:HEAD + k] = values
        ids = torch.full((n, top), -1, dtype=torch.int32, device=device)
        ids[:, :k] = (cols + offset).to(torch.int32)
        packet[:, HEAD + top:] = ids.view(torch.float32)
    if comm is None:
        gathered = packet[None]
    else:
        flat = packet.reshape(-1).contiguous()
        got = torch.empty((comm.world * flat.numel(),), dtype=torch.float32, device=device)
        comm.all_gather(flat, got)
        gathered = got.view(comm.world, n, HEAD + 2 * top)
    host = gathered.cpu().numpy()
    peaks, masses, owners = host[:, :, 0].astype(np.float64), host[:, :, 1].astype(np.float64), host[:, :, 2]
    if not np.isfinite(peaks).all() or not np.isfinite(masses).all() or (masses <= 0).any():
        raise ValueError("teacher scoring found non-finite shard logits")
    maximum = peaks.max(axis=0)
    lse = maximum + np.log((masses * np.exp(peaks - maximum)).sum(axis=0))
    holders = np.isfinite(owners)
    if not (holders.sum(axis=0) == 1).all():
        raise ValueError("each scored target must be owned by exactly one vocabulary shard")
    chosen = (np.where(holders, owners, 0.0).astype(np.float64).sum(axis=0) - lse).tolist()
    alternatives, scores = [], []
    if top:
        tvals = host[:, :, HEAD:HEAD + top].astype(np.float64)
        tids = np.ascontiguousarray(host[:, :, HEAD + top:]).view(np.int32).astype(np.int64)
        for row in range(n):
            pairs = sorted((-float(v), int(i)) for v, i in zip(tvals[:, row].ravel(), tids[:, row].ravel())
                           if i >= 0)[:top]
            alternatives.append([i for _, i in pairs])
            scores.append([-v - float(lse[row]) for v, _ in pairs])     # plain floats: owned JSON values
    else:
        alternatives, scores = [[] for _ in range(n)], [[] for _ in range(n)]
    return chosen, alternatives, scores


def capture(w, logits, tokens, positions, probabilities, rows=None, origins=None) -> None:
    """``tensorfold.cuda.logprobs.capture`` for shard logits: the emitted tokens' rows into the request's sink."""

    if probabilities is None or not tokens:
        return
    values, ids, scores = shard_logprobs(w, logits, tokens, probabilities.top, rows)
    probabilities.add(positions, tokens, values, ids, scores)
    if origins is not None:
        probabilities.origins(positions, origins)


def forced(stream) -> list[int] | None:
    """The teacher-forced reply of a scoring request, else None."""

    return getattr(stream.probabilities, "forced", None)

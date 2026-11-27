"""Bounded device counts and asynchronous snapshots of logical expert routes."""

import torch

from .kernels.index_topk import _masked_histogram
from .kernels.triton_utils import jit, tl, triton


PREFILL = 0
DECODE = 1
_LAYERS = 40
_EXPERTS = 384
_TOP_K = 6
_TILE = 512


@jit(do_not_specialize=["n_routes"])
def _count_routes(ids, live, counts, n_routes, HAS_LIVE: tl.constexpr, TOP_K: tl.constexpr, EXPERTS: tl.constexpr,
                  TILE: tl.constexpr):
    route = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = route < n_routes
    if HAS_LIVE:
        valid &= tl.load(live + route // TOP_K, mask=valid, other=False)
    expert = tl.load(ids + route, mask=valid, other=-1)
    valid &= (expert >= 0) & (expert < EXPERTS)
    histogram = _masked_histogram(expert.to(tl.int32), valid, TILE)
    bins = tl.arange(0, TILE)
    tl.atomic_add(counts + bins, histogram.to(tl.int64), mask=(bins < EXPERTS) & (histogram != 0), sem="relaxed")


class ExpertUsage:
    """One rank's rolling counts, owned by its scheduler thread and stream.

    Graphs may borrow ``counts`` until they are destroyed. All count writes and
    snapshot requests must use the same ordered CUDA stream after startup. The
    single pinned landing buffer is never exposed to readers or shared with a
    second snapshot in flight.
    """

    def __init__(self, device):
        self.counts = torch.zeros((2, _LAYERS, _EXPERTS), dtype=torch.int64, device=device)
        self._host = torch.empty(self.counts.shape, dtype=torch.int64, pin_memory=True)
        self._ready = torch.cuda.Event()
        self._pending = False

    def count_routes(self, ids, live=None, *, layer, phase):
        """Count a sampled whole-rank copy of the global choices before local masking.

        An absent live mask means every row in ``ids`` is logical. A supplied
        mask excludes staged padding. Off-range IDs, including sentinels, never
        contribute. The engine owns sampling and which rank records the routes.
        """
        if phase not in (PREFILL, DECODE) or not 0 <= layer < _LAYERS:
            raise ValueError("Invalid expert usage phase or layer")
        if (ids.ndim != 2 or ids.shape[1] != _TOP_K or ids.device != self.counts.device
                or ids.dtype not in (torch.int32, torch.int64) or not ids.is_contiguous()):
            raise ValueError("Expert usage requires contiguous global route IDs on the counter device")
        if live is not None and (live.shape != (ids.shape[0],) or live.dtype != torch.bool
                                 or live.device != ids.device or not live.is_contiguous()):
            raise ValueError("Expert usage requires one contiguous boolean validity value per route row")
        n_routes = ids.numel()
        if n_routes:
            _count_routes[(triton.cdiv(n_routes, _TILE),)](
                ids, ids if live is None else live, self.counts[phase, layer], n_routes, live is not None,
                _TOP_K, _EXPERTS, _TILE,
            )

    def request_snapshot(self):
        """Queue a copy and reset, or return False and retain counts if one is pending."""
        if self._pending:
            return False
        self._host.copy_(self.counts, non_blocking=True)
        self._ready.record(torch.cuda.current_stream(self.counts.device))
        self._pending = True
        self.counts.zero_()
        return True

    def poll_snapshot(self):
        """Return an independent CPU snapshot when its asynchronous copy is ready."""
        if not self._pending or not self._ready.query():
            return None
        result = self._host.clone()
        self._pending = False
        return result

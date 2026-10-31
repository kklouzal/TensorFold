"""Native grouped plan readback and slot-window publication ownership."""
from __future__ import annotations

from copy import copy

import torch

from . import experts as grouped


class ExpertWindow:
    """One MoE buffer's bounded plan snapshot and reusable item-window uploads.

    The original plan owns members and pair order. A window changes only item
    expert IDs and the valid item count; its consumers keep the original tile
    arithmetic. Copy completion protects pinned host data before it is reused.
    Like the owning MoE buffers, this scratch serves one CUDA stream at a time.
    """

    def __init__(self, source: grouped.Plan) -> None:
        self.plan = copy(source)
        # These copy destinations outlive the caller's inference/ordinary mode.
        with torch.inference_mode(False):
            self.plan.items = torch.empty_like(source.items)
            self.plan.counts = torch.empty_like(source.counts)
            self.report = torch.empty((source.items.numel() + 2,), dtype=torch.int32, device=source.items.device)
            self.host = torch.empty(self.report.shape, dtype=torch.int32, device="cpu", pin_memory=True)
        self.ready = torch.cuda.Event()
        self.uploaded = torch.cuda.Event()
        self.pending = False

    def read(self, source: grouped.Plan) -> torch.Tensor:
        # One bounded D2H transfer obtains both the valid count and item IDs.
        self.report[:2].copy_(source.counts)
        self.report[2:].copy_(source.items.reshape(-1))
        self.host.copy_(self.report, non_blocking=True)
        self.ready.record(torch.cuda.current_stream(source.items.device))
        self.ready.synchronize()
        count = int(self.host[0])
        if not 0 <= count <= source.items.shape[0]:
            raise RuntimeError(f"expert plan has {count} valid items outside its bounded scratch")
        self.plan.tile = source.tile
        return self.host[2:].view(-1, 3)[:count]

    def stage(self, items: torch.Tensor, mapping: dict[int, int], capacity: int) -> grouped.Plan:
        if self.pending:
            self.uploaded.synchronize()
        ids = items[:, 0].numpy()
        ids[:] = [mapping[int(expert)] for expert in ids]
        self.host[0], self.host[1] = items.shape[0], len(mapping)
        self.plan.experts = capacity
        self.plan.items[:items.shape[0]].copy_(items, non_blocking=True)
        self.plan.counts.copy_(self.host[:2], non_blocking=True)
        self.uploaded.record(torch.cuda.current_stream(self.plan.items.device))
        self.pending = True
        return self.plan

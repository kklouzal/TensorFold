"""Completed-HC feature capture with bounded, transaction-owned frame storage."""
from __future__ import annotations

import torch


class FeatureFrame:
    def __init__(self, config, rows, device, *, target_dtype=torch.float16):
        if type(rows) is not int or rows < 1:
            raise ValueError("DFlash2 feature frame rows must be positive")
        self.config, self.rows = config, rows
        self.features = torch.empty((rows, len(config.taps), config.hidden), dtype=torch.float16, device=device)
        self.completed = torch.empty((rows, config.hc_streams * config.hidden), dtype=target_dtype, device=device)
        self.pss = torch.empty((rows, config.hidden // 256, config.hc_streams), dtype=torch.float32, device=device)
        self.readers = 0
        self.generation = 0

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.features, self.completed, self.pss))

    def begin(self, segs, buffers):
        if self.readers:
            raise ValueError("DFlash2 feature frame still has an unresolved accepted-prefix reader")
        rows = segs[-1][2]
        if rows > self.rows:
            raise ValueError("DFlash2 target frame exceeds allocated feature capture rows")
        self.generation += 1
        return Capture(self, tuple(segs), buffers, self.generation)


class Capture:
    def __init__(self, owner, segs, buffers, generation):
        self.owner, self.segs, self.buffers, self.generation = owner, segs, buffers, generation
        self.seen = []
        self.published = False

    def completed_layer(self, layer, weights, pending):
        config = self.owner.config
        if layer not in config.taps:
            return
        if self.owner.generation != self.generation or self.seen != list(config.taps[:len(self.seen)]):
            raise ValueError("DFlash2 target feature capture is stale or out of order")
        if len(self.seen) >= len(config.taps) or layer != config.taps[len(self.seen)]:
            raise ValueError("DFlash2 target feature tap order differs from the declared manifest")
        from .forward import _writeback

        for _target, lo, hi in self.segs:
            rows = hi - lo
            completed = self.owner.completed[lo:hi]
            completed.copy_(self.buffers.h[lo:hi])
            # Capturing a mixed frame must not perform draft feature work for
            # serial-only rows. The HC operator requires contiguous gathered
            # partials; only this enabled span is materialized, in rank order.
            if pending is not None:
                mode, values, weights0, inject = pending
                partial = values[:, lo:hi].contiguous() if mode in (3, 4) else values[lo:hi]
                part = (mode, partial, None if weights0 is None else weights0[lo:hi], inject[lo:hi])
                _writeback(completed, self.owner, rows, weights.cfg, part, pss=self.owner.pss[lo:hi])
            mean = completed.reshape(rows, config.hc_streams, config.hidden).float().mean(1).half()
            self.owner.features[lo:hi, len(self.seen)].copy_(mean)
        self.seen.append(layer)

    def finish(self):
        if self.published:
            raise ValueError("DFlash2 target feature frame was already published")
        if self.owner.generation != self.generation or tuple(self.seen) != self.owner.config.taps:
            raise ValueError("DFlash2 completed-layer feature capture is incomplete")
        self.published = True
        return self.owner.features[:self.segs[-1][2]].flatten(1)

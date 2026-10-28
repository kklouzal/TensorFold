"""Frozen original indexed preparation oracle (c928); no candidate predicates."""

import torch
from tensorfold.cuda.sm70 import family_ops as ops
from tensorfold.families.qwen4_exp.cuda.compact_indexer import RATIO


def indexed_pool(self, pooled, weight, inv_freq, eps, sections, scale):
    span = self.span
    span.check()
    first, end = span.start // RATIO, span.end // RATIO
    if first == end:
        return
    with self.reading((self.protected, self.protected_coordinates, span.raw, span.coordinates), span=span) as reader:
        positions = torch.arange(first * RATIO, end * RATIO, device=self.raw.device, dtype=torch.int64)
        previous = positions < span.start
        source = (positions - span.start).clamp(0, len(span.raw) - 1)
        raw = torch.where(
            previous[:, None], self.protected.index_select(0, positions % RATIO), span.raw.index_select(0, source)
        )
        raw = raw.reshape(end - first, RATIO, self.dim).float()
        total = raw[:, 0]
        for member in range(1, RATIO):
            total = total + raw[:, member]
        value = (total / RATIO).half()
        normalization_rows = max(len(value), min(self.capacity // RATIO, 16)) if value.is_cuda else len(value)
        if normalization_rows != len(value):
            padded = torch.zeros((normalization_rows, self.dim), dtype=torch.float16, device=self.raw.device)
            padded[: len(value)].copy_(value)
            value = ops.rms(padded, weight, eps)[: len(value)]
        else:
            value = ops.rms(value, weight, eps)
        member = positions.reshape(-1, RATIO)[:, 0]
        old = member < span.start
        at = (member - span.start).clamp(0, len(span.raw) - 1)
        coordinates = torch.where(
            old[:, None],
            self.protected_coordinates.index_select(0, member % RATIO),
            span.coordinates.index_select(0, at),
        )
        value = ops.rotary(
            value,
            inv_freq,
            torch.arange(len(value), device=value.device),
            rope=coordinates,
            length=len(value),
            sections=sections,
            scale=scale,
        )
        value = ops._guard(value, self.status, 16, self.layer)
        self._copy(value, pooled, span.expected, first)
        reader[0] = True

"""Private captured target-feature projection; only accepted target context persists."""
from __future__ import annotations

import torch

from . import dspark_reference as ref
from .dspark_cuda import DeviceContextUpdate, _event, _record, context_norm

_ENABLED = False


class ContextWorkspace:
    """Bound verify buffer owner, explicit graph anchor and independent exports."""
    def __init__(self, provider, rows, device):
        self.provider, self.rows = provider, rows
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device != provider.device or type(rows) is not int or not 1 <= rows <= 2048:
            raise ValueError("DSpark context graphs require 1..2048 admitted target rows")
        self.owned = {}
        self.layout = {}
        self.closed = self.graph_active = self.recorded = False
        self.pending_done = False
        self.done = _event(self.device)
        self.frames = self.exported_rows = self.readbacks = 0

    def allocate(self):
        if self.closed or self.owned:
            raise ValueError("DSpark context storage allocation is one-shot")
        c, kv = self.provider.config, self.provider.local["kv_heads"]
        for name, shape, dtype in (
            ("keys", (c.layers, self.rows, kv, c.head_dim), torch.float16),
            ("values", (c.layers, self.rows, kv, c.head_dim), torch.float16),
            ("positions", (self.rows,), torch.int64),
            ("offsets", (self.rows,), torch.int64),
            ("finite", (self.rows,), torch.bool),
        ):
            value = torch.empty(shape, dtype=dtype, device=self.device)
            self.owned[name] = value
            self.layout[name] = (tuple(value.shape), value.dtype, value.data_ptr())
            setattr(self, name, value)
        torch.arange(self.rows, out=self.offsets)

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.owned.values())

    @staticmethod
    def device_bytes(config, rows, kv_heads):
        return config.layers * rows * kv_heads * config.head_dim * 4 + rows * 17

    def prepare_replay(self):
        self._live()
        if self.pending_done:
            torch.cuda.current_stream(self.device).wait_event(self.done)
            self.pending_done = False

    def _live(self):
        if self.closed or len(self.owned) != 5 or any(getattr(self, name, None) is not value
                or (tuple(value.shape), value.dtype, value.data_ptr()) != self.layout[name]
                or value.device != self.device or not value.is_contiguous()
                for name, value in self.owned.items()):
            raise ValueError("DSpark captured context storage is closed or changed outside its owner")

    def compute(self, features, segments):
        self._live()
        if (self.closed or features.dtype != torch.float16 or features.device != self.device
                or features.ndim != 2 or features.shape[1] != len(self.provider.config.taps) * self.provider.config.hidden
                or not features.is_contiguous() or not 1 <= features.shape[0] <= self.rows):
            raise ValueError("DSpark captured context features exceed their admitted buffer")
        if any(torch._C._overlaps(features, t) for t in self.owned.values()):
            raise ValueError("DSpark context features must not alias their owned projection buffers")
        self.graph_active, self.recorded = True, False
        p, c, rows = self.provider, self.provider.config, features.shape[0]
        end = 0
        for state, a0, a1 in segments:
            if (a0 != end or not a0 < a1 <= rows or state.pos_dev.device != self.device
                    or state.pos_dev.dtype != torch.int32 or state.pos_dev.shape != (1,)):
                raise ValueError("DSpark captured positions require compact State device cursors")
            self.positions[a0:a1].copy_(state.pos_dev.to(torch.int64) + self.offsets[:a1-a0])
            end = a1
        if end != rows:
            raise ValueError("DSpark captured positions must cover every physical target row")
        self.finite[:rows].copy_(torch.isfinite(features).all(-1))
        tensors = p.weights.tensors
        projected = context_norm(p._linear(features, "fc.weight"), tensors["hidden_norm.weight"], c.eps)
        for layer in range(c.layers):
            base = f"layers.{layer}.self_attn."
            k = p._linear(projected, base + "k_proj.weight").view(rows, p.local["kv_heads"], c.head_dim)
            k = context_norm(k, tensors[base + "k_norm.weight"], c.eps)
            self.keys[layer, :rows].copy_(ref.rope(k, self.positions[:rows], c.theta))
            self.values[layer, :rows].copy_(p._linear(projected, base + "v_proj.weight").view_as(k))
        self.recorded = True

    def stage(self, states, segments, index):
        """One cohort finite consensus, independent contiguous update per State."""
        self._live()
        if (not self.recorded or not self.graph_active or not isinstance(index, torch.Tensor)
                or index.dtype != torch.int64 or index.device != self.device or index.ndim != 1
                or not index.is_contiguous() or not 1 <= index.numel() <= self.rows
                or len(states) != len(segments) or len({id(s) for s in states}) != len(states)):
            raise ValueError("DSpark context export requires its trusted bound row map and original States")
        p, end = self.provider, 0
        for aux, (st, a0, a1) in zip(states, segments, strict=True):
            p._check(aux, aligned=False)
            if (aux.target is not st or a0 != end or not a0 < a1 <= index.numel()
                    or aux.pending is not None or aux.length != st.pos
                    or a1-a0 > p.prefill_rows or aux.length + a1-a0 > aux.capacity):
                raise ValueError("DSpark context export lost its committed cursor/owner or staging admission")
            end = a1
        if end != index.numel():
            raise ValueError("DSpark context export segments must cover all real target rows")
        if any(torch._C._overlaps(index, t) for t in self.owned.values()):
            raise ValueError("DSpark context row map must not alias owned projection buffers")
        try:
            good = self.finite.index_select(0, index).all()
            pieces = []
            for aux, (_, a0, a1) in zip(states, segments, strict=True):
                indices = index[a0:a1]
                k = self.keys.index_select(1, indices)
                v = self.values.index_select(1, indices)
                pieces.append((aux, k, v))
                good = good & torch.isfinite(k).all() & torch.isfinite(v).all()
        finally:
            # Exported per-State tensors own their storage; graph inputs may be
            # reused after these reads even while finite consensus is finishing.
            _record(self.done, self.device)
            self.pending_done = self.done is not None
        self.readbacks += 1
        if not bool(p._gather(good.to(torch.int32).reshape(1)).all().cpu()):
            raise ValueError("DSpark nonfinite captured context; no auxiliary prefix published")
        updates = []
        for aux, k, v in pieces:
            update = DeviceContextUpdate(aux.length, k, v, aux.version, aux, k.shape[1], _event(self.device))
            _record(update.ready, self.device)
            updates.append(update)
        for aux, update in zip(states, updates, strict=True):
            aux.pending = update
        self.frames += 1
        self.exported_rows += index.numel()
        return tuple(updates)

    def retire_graph(self):
        self.graph_active = False

    def close(self):
        if self.graph_active:
            raise ValueError("DSpark context graph still borrows its projection buffers")
        if self.pending_done:
            self.done.synchronize()
        self.owned.clear()
        self.layout.clear()
        for name in ("keys", "values", "positions", "offsets", "finite"):
            setattr(self, name, None)
        self.closed = True
        self.provider = self.done = None

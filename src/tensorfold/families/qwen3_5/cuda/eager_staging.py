"""Private one-request eager-plan prototype; engine activation awaits admission.

Only topology tables are retained. Tokens and rotary positions keep the normal
forward's transfers, and each hit creates a fresh four-int attention stream.
There is no reusable pinned mirror, graph, KV owner or recurrent-state binding.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass

import torch

from tensorfold.cleanup import rollback
from tensorfold.cuda.kernels import attention


@dataclass
class _Entry:
    parents: tuple[int, ...]
    geometry: tuple
    boundary: tuple[int, int]
    owner: object
    caches: tuple
    origin: torch.Tensor | None
    origin_layout: tuple | None
    plan: object
    aplan: object
    aoffs: dict
    windows: torch.Tensor

    def matches(self, w, st, parents, softmax, origin) -> bool:
        c = w.config
        if (self.parents != tuple(parents)
                or self.geometry != (c.conv_kernel, c.heads, c.kv_heads, softmax)
                or self.boundary != (st.pos // attention.CHUNK, attention.slots(st.pos, len(parents)))
                or self.owner is not st.kv_owner or self.origin is not origin
                or self.origin_layout != _layout(origin)):
            return False
        # Weak referent identity prevents address reuse from validating another
        # allocation. Metadata/address checks also reject in-place view changes.
        for layer, pair in zip(softmax, self.caches):
            for tensor, (reference, layout) in zip(st.kv[layer], pair):
                if reference() is not tensor or _layout(tensor) != layout:
                    return False
        return True


def _layout(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device) if tensor is not None else None


class _RequestStaging:
    """One default-stream request's settled topology entry, never an engine cache.

    The caller owns this context through every forward and calls ``completed``
    only after its existing successful verification synchronization. Reuse or
    eviction before that acknowledgement is an invariant error. On failure,
    close synchronizes the actual producer stream before releasing tables; a
    failed close retains this owner on the primary error using ``rollback``.

    Native default streams live with the CUDA context also owned by the request's
    model tensors. Other streams use fresh plans, so an external stream handle
    cannot become a persistent cache identity. Model identity is weak; KV views
    are weak and must remain exact current views. Growth therefore safely misses
    when it publishes a replacement view, without retaining an old arena.
    """

    def __init__(self, w):
        self.model = weakref.ref(w)
        self.default = torch.cuda.default_stream(w.norm.device)
        self.entry: _Entry | None = None
        self.producer = None
        self.pending = False
        self.closed = False
        self.cacheable = False

    def begin(self, w, st, parents, chain, softmax, origin):
        if self.closed or self.pending:
            raise RuntimeError("eager staging requires an open, completed request entry")
        if self.model() is not w:
            raise ValueError("eager staging belongs to another model")
        stream = torch.cuda.current_stream(w.norm.device)
        self.cacheable = chain and stream == self.default
        self.producer, self.pending = stream, True
        entry = self.entry
        if self.cacheable and entry is not None and entry.matches(w, st, parents, softmax, origin):
            return entry
        # Drop a settled miss before preparing its replacement: at most one
        # retained plan, with no old/new cache-table or arena overlap.
        self.entry = None
        return None

    def capture(self, w, st, parents, softmax, origin, plan, aplan, aoffs, windows):
        if not self.cacheable:
            return
        c = w.config
        caches = tuple(tuple((weakref.ref(t), _layout(t))
                             for t in st.kv[i]) for i in softmax)
        self.entry = _Entry(tuple(parents), (c.conv_kernel, c.heads, c.kv_heads, softmax),
                            (st.pos // attention.CHUNK, attention.slots(st.pos, len(parents))),
                            st.kv_owner, caches, origin, _layout(origin),
                            plan, aplan, aoffs, windows)

    def completed(self):
        """Acknowledge the caller's successful fence; this method submits no work."""
        if self.closed or not self.pending:
            raise RuntimeError("eager staging has no pending verification")
        self.pending = False

    def close(self):
        if self.closed:
            return
        if self.pending:
            self.producer.synchronize()
            self.pending = False
        self.entry = None
        self.producer = None
        self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError("eager staging is closed")
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            rollback(self, error, self.close)
        self.close()

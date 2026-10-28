"""Private pure-MoE capture component; serving integration follows actual oracles.

No State, sampling, partial reduction or collective is inside this region.
The caller owns process-wide CUDA quiescence during prepare and fixed buffer/
weight bindings until close. Replay is serialized on its original CUDA stream.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from tensorfold.cleanup import raise_failures
from .capture_lifetime import on_stream


def region_tensors(buffers):
    return (buffers.mixed, buffers.moe.logits, buffers.moe.pick,
            buffers.moe.wts, buffers.moe.y)


def validate_region(weights, layer, buffers, rows):
    if type(rows) is not int or not 1 <= rows <= 32:
        raise ValueError("MoE graph region needs an exact row count in1..32")
    if buffers.prefill or buffers.mixed.dtype != torch.float16 or not buffers.mixed.is_cuda:
        raise ValueError("MoE graph region requires canonical FP16 CUDA decode buffers")
    from tensorfold.cuda.sm70.execution_policy import SHARED_FUSION_TRIAL, HC_SMALLROW_VENDOR_TRIAL

    profile = getattr(weights, "meta", {}).get("sm70_profile")
    expected = "packed_blocks2w" if profile in (SHARED_FUSION_TRIAL, HC_SMALLROW_VENDOR_TRIAL) else "packed_exact"
    expert = layer.moe.experts
    if (getattr(expert, "kernel", None) != "gguf"
            or expert.implementation != expected or expert.packed_rows < rows
            or expert.count != 513 or expert.dims != 2560 or expert.width not in (80, 160)
            or weights.cfg.experts != 512 or weights.cfg.top_k != 10):
        raise ValueError("MoE graph region requires the qualified actual513-router/11-selected-slot GGUF bank")
    if buffers.mixed.shape[0] < rows or buffers.moe.y.shape[0] < rows:
        raise ValueError("MoE graph region exceeds its external prepared rows")
    for tensor in region_tensors(buffers):
        if tensor.device != buffers.mixed.device or not tensor.is_contiguous():
            raise ValueError("MoE graph operands require canonical contiguous device storage")


class MoERegion:
    """One exact layer/row graph over externally prepaid persistent operands.

    The component keeps an independent pool by default. A later bounded serving
    catalogue may share one pool per exactR only if every layer replays in its
    capture order, no partial failed cycle is reused, and consumers retire.
    """
    def __init__(self, weights, layer, buffers, rows, *, side=None, pool=None, capture_router=None):
        self.weights, self.layer, self.buffers, self.rows = weights, layer, buffers, rows
        self.graph = None
        self.side, self.pool = side, pool
        self.capture_router, self.local_buffers = capture_router, None
        self.failure = None
        self.prepared = self.closed = self.retiring = False
        self.hits = 0
        self.counter_overflowed = False
        self.replay_stream = None
        self.tensors = ()
        validate_region(weights, layer, buffers, rows)
        self.tensors = region_tensors(buffers)  # Strong storage owners, not transient addresses.

    def _body(self):
        from .forward import moe_block

        buffers = self.buffers if self.local_buffers is None else self.local_buffers
        moe_block(self.layer, self.weights, buffers, self.rows, _local_only=True)

    def _warm(self):
        if getattr(self.buffers, "router_blas", None) is not None:
            from tensorfold.cuda.sm70.router_blas import RolePlans

            # Capture must not borrow the main-stream role. Publish the partial
            # side-stream owner before its constructor can allocate/submit.
            if self.capture_router is None:
                self.capture_router = object.__new__(RolePlans)
                self.capture_router.__init__((self.layer.moe.router,), 32)
            elif not any(weight is self.layer.moe.router for weight in self.capture_router.owners):
                raise ValueError("shared capture router does not own this exact layer weight")
            self.local_buffers = SimpleNamespace(prefill=self.buffers.prefill,
                mixed=self.buffers.mixed, moe=self.buffers.moe, router_blas=self.capture_router)
        self._body()

    def _on_side(self, operation):
        on_stream(torch.cuda, self.side, operation)

    def _capture(self):
        secondary = [None]  # No outcome allocation after accepted capture_begin.
        self.graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")
        primary = None
        try:
            self._body()
        except BaseException as error:
            primary = error
        try:
            self.graph.capture_end()  # Exactly once after an accepted begin.
        except BaseException as error:
            secondary[0] = error
        if secondary[0] is not None:
            raise_failures(primary, secondary)
        if primary is not None:
            raise primary

    def prepare(self):
        if self.prepared or self.graph is not None or self.closed or self.retiring or self.failure is not None:
            raise RuntimeError("MoE graph component cannot be prepared twice or after failure")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("MoE region preparation requires process-wide capture quiescence")
        device = self.buffers.mixed.device
        current = torch.cuda.current_stream(device)
        self.replay_stream = current.cuda_stream
        try:
            if self.side is None:
                self.side = torch.cuda.Stream(device=device)
            self.side.wait_stream(current)
            self._on_side(self._warm)  # Exact kernels/modules and dedicated capture role.
            self.side.synchronize()
            self.graph = torch.cuda.CUDAGraph()  # Publish ownership before accepted capture.
            self._on_side(self._capture)
            current.wait_stream(self.side)
            self.prepared = True
        except BaseException as error:
            self.failure = error
            raise

    def replay(self):
        if not self.prepared or self.closed or self.retiring or self.failure is not None:
            raise RuntimeError("MoE region is not an admitted prepared component") from self.failure
        try:
            # Component boundary validation. A later actor may hoist this only
            # with a proved immutable canonical-buffer lifetime contract.
            current = region_tensors(self.buffers)
            if any(value is not original for value, original in zip(current, self.tensors)):
                raise RuntimeError("MoE region lost its original external buffer owners")
            stream = torch.cuda.current_stream(self.buffers.mixed.device)
            if stream.cuda_stream != self.replay_stream:
                raise RuntimeError("MoE region replay requires its original serialized stream")
            self.graph.replay()
            if self.hits < (1 << 63) - 1:
                self.hits += 1
            else:
                self.counter_overflowed = True
        except BaseException as error:
            self.failure = error
            raise

    def close(self):
        if self.closed:
            return
        # Do not clear references on uncertain retirement. Caller cleanup must
        # preserve a previously propagated primary if this fence/reset fails.
        self.retiring = True  # Refuse replay even if fence/reset leaves retirement uncertain.
        try:
            torch.cuda.synchronize(self.buffers.mixed.device)
        except BaseException as error:
            if self.failure is None:
                self.failure = error
            raise
        try:
            self._release_fenced()
        except BaseException as error:
            if self.failure is None:
                self.failure = error
            raise

    def _release_fenced(self):
        """Catalogue-only: caller already positively fenced and closed replay admission."""
        self.retiring = True
        if self.graph is not None:
            self.graph.reset()
        self.graph = self.side = self.pool = None
        # A graph may retain captured pointer-table/scratch addresses. Release
        # these external owners only after positive fence AND graph reset.
        self.capture_router = self.local_buffers = None
        self.tensors = ()
        self.closed = True

    def record(self):
        return {"rows": self.rows, "layer": self.layer.index, "prepared": self.prepared,
                "closed": self.closed, "retiring": self.retiring, "replays": None if self.counter_overflowed else self.hits,
                "scope": "private component only; no integrated performance/capacity qualification"}

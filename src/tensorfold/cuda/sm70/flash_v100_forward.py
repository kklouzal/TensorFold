"""Unselected exact ai-bond FlashAttention-V10026.10 forward-only component.

This boundary consumes an explicitly built/verified native module, not a package
installation or a serving selector. Native sources require immutable upstream
commit0ababf82e3836c1786d7937c6002670ca4ecda62 and its retained BSD notice.
The controlled component build records external source/header/SDK identities.

Q/Out[B,M,Hq,256] and KV[B,N,Hkv,256] are contiguous,16-byte aligned Half;
LSE[B,Hq,M] is contiguous Float32. N/M/B are positive. No dropout, causal,
window, ALiBi or softcap is selected. QSA visibility/ownership filtering, empty
shards and distributed LSE combination remain separate owning operations.
Inputs are finite Half values from their validated producer. Output is locally
normalized Half with tile-rounded Half probabilities and Float32 accumulation;
it does not preserve TensorFold's globally normalized probability rounding.
This SC117/head256 component binds the exact1/16 score scale. Other scales are
outside this component's declared numerical domain.

Prepare overwrites Out/LSE and fences once after warmup. Run borrows operands
on the current stream; callers retain them and this owner/native module until
completion and graph retirement, and order work when changing streams. All
operations are serialized on one host owner; explicit cold handoff fences before
publishing a new Thread object. A drained control thread may close once after
graph retirement. No runtime allocations, attribute updates or synchronization.
"""
import threading

import torch

from tensorfold.cleanup import raise_failures
from .prefill_dense import _plain

SOURCE_COMMIT = '0ababf82e3836c1786d7937c6002670ca4ecda62'


def _publish(out, lse):
    for value in (out, lse):
        if not value.is_inference():
            torch.autograd.graph.increment_version(value)


class Prepared:
    def __init__(self, module, q, k, v, out, lse, *, scale, _valid=None, _credit=None):
        self._thread = threading.current_thread()
        self._closed = self._failed = self._retirement_failed = False
        self._module, self._native = module, None
        if type(scale) is not float or scale != 0.0625:
            raise ValueError('FA26.10 head256 component requires its exact1/16 score scale')
        if any(not _plain(t) for t in (q, k, v, out, lse)) or (_valid is not None and not _plain(_valid)):
            raise ValueError('FA26.10 requires plain non-grad operands')
        if _valid is None:
            self._native = module.Prepared(q, k, v, out, lse, scale)
        else:
            needed = compacted_workspace_bytes(q.shape[0] // 12)
            if type(_credit) is not int or not needed <= _credit <= (1 << 63) - 1:
                raise MemoryError('FA26.10 compacted tensor byte credit exhausted')
            self._native = module.Prepared(q, k, v, out, lse, scale, _valid, _credit)
        try:
            self._native.configure()
            self._native.warm()
            _publish(out, lse)
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                raise_failures(primary, [cleanup])
            raise

    def _check(self):
        if self._closed or self._failed or self._retirement_failed:
            raise RuntimeError('FA26.10 owner is closed or failed')
        if threading.current_thread() is not self._thread:
            raise RuntimeError('FA26.10 owner belongs to another host Thread')

    def run(self, q, k, v, out, lse):
        self._check()
        if any(not _plain(t) for t in (q, k, v, out, lse)):
            raise ValueError('FA26.10 requires plain non-grad operands')
        return self._invoke(self._native.run, (q, k, v, out, lse), out, lse)

    def _invoke(self, call, operands, out, lse):
        try:
            call(*operands)
            _publish(out, lse)
        except BaseException:
            self._failed = True
            raise
        return out, lse

    def handoff_to_current_thread(self):
        if self._closed or self._failed or self._retirement_failed:
            raise RuntimeError('FA26.10 owner is closed or failed')
        try:
            self._native.handoff_to_current_thread()
        except BaseException:
            self._retirement_failed = True
            raise
        self._thread = threading.current_thread()

    def settings(self):
        self._check()
        return self._native.settings()

    def close(self):
        if self._closed:
            return
        if self._retirement_failed:
            raise RuntimeError('FA26.10 retirement uncertain; retry forbidden')
        if self._native is not None:
            try:
                self._native.close()
            except BaseException:
                self._retirement_failed = True
                raise
        self._closed = True
        self._native = self._module = None

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, kind, primary, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if primary is None:
                raise
            raise_failures(primary, [cleanup])
        return False


def prepare(module, q, k, v, out, lse, *, scale=0.0625):
    """Bind an explicitly verified native artifact; no implicit download/build."""
    return Prepared(module, q, k, v, out, lse, scale=scale)


def compacted_workspace_bytes(batch):
    """Extra fixed-slot KV and private length payload, excluding allocator/module effects."""
    if type(batch) is not int or not 1 <= batch <= 256:
        raise ValueError('FA26.10 compacted rows require1..256')
    return 2 * batch * 2051 * 256 * 2 + 4 * (3 * batch + 2)


class CompactedPrepared(Prepared):
    """Fixed12 head-as-row Q; stable2051-slot compaction and GPU-only lengths.

    Q/Out[12*B,1,256], decoded KV[B,2051,1,256], Bool valid[B,1,2051],
    Float LSE[1,12*B]. Preparation funds and owns compacted KV/private offsets
    and counts. Input holes are omitted stably, without reading their payload.
    Empty output is zero/-inf; the adapter applies no repair to nonempty
    foreign results. Inputs retain the finite-valid-producer contract; upstream
    softmax need not propagate every NaN Q/K pattern. This is a distinct
    unqualified profile with the known Float32LSE mass limitation.
    Prepare overwrites Out/LSE using initialized one-key private warm data;
    serving run consumes the actual supplied mask. Existing borrowing, Thread,
    current-stream and positive retirement contracts apply without runtime
    allocation, host length readback, attribute changes or synchronization.
    """
    def __init__(self, module, q, k, v, valid, out, lse, *, scale, tensor_budget_bytes):
        if not _plain(valid):
            raise ValueError('FA26.10 requires a plain non-grad validity tensor')
        super().__init__(module, q, k, v, out, lse, scale=scale,
                         _valid=valid, _credit=tensor_budget_bytes)

    def run(self, q, k, v, valid, out, lse):
        self._check()
        if any(not _plain(t) for t in (q, k, v, valid, out, lse)):
            raise ValueError('FA26.10 requires plain non-grad operands')
        return self._invoke(self._native.run_compacted, (q, k, v, valid, out, lse), out, lse)


def prepare_compacted(module, q, k, v, valid, out, lse, *, tensor_budget_bytes, scale=0.0625):
    return CompactedPrepared(module, q, k, v, valid, out, lse,
        scale=scale, tensor_budget_bytes=tensor_budget_bytes)

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
    def __init__(self, module, q, k, v, out, lse, *, scale):
        self._thread = threading.current_thread()
        self._closed = self._failed = self._retirement_failed = False
        self._module, self._native = module, None
        if type(scale) is not float or scale != 0.0625:
            raise ValueError('FA26.10 head256 component requires its exact1/16 score scale')
        if any(not _plain(t) for t in (q, k, v, out, lse)):
            raise ValueError('FA26.10 requires plain non-grad operands')
        self._native = module.Prepared(q, k, v, out, lse, scale)
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
        try:
            self._native.run(q, k, v, out, lse)
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

"""Unselected prepared-geometry SM70 dense projection candidates.

Half X[M,K], contiguous Half W[N,K], Half/Float output[M,N]; row
padding is allowed on X/output. No M32 input padding or output copy is used.
All candidates accumulate Float32 but reduction ordering differs from the
incumbent: numerical/model and integrated performance qualification is required.

Preparation warms the supplied operands outside capture and overwrites output.
Each owner belongs to one host thread; an explicit cold handoff can transfer
ownership after the prior host owner drains. Run accepts different pointers
on the current CUDA stream with exactly the prepared geometry/strides/dtypes.
The caller owns every operand through stream completion and graph retirement,
and must retain this owner until every graph using it has retired. Explicit
Context.close fences the device once, then releases every registered plan.
A failed retirement is terminal; abandoned native resources are retained with
a diagnostic. Initial operand references are released after positive warmup.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import threading
import weakref

import torch

from tensorfold.cleanup import raise_failures
from .prefill_dense import _plain


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .provider import _native
    from .foundation_headers import selected_headers
    from .source_cache import stage_sources

    headers = selected_headers()
    if headers is None:
        raise RuntimeError('dense candidates require the protected CUTLASS foundation')
    root = Path(__file__).parent
    sources = [root / 'dense_backends.cpp', root / 'dense_backends.cu']
    included = [root / 'dense_backends.cuh', root / 'prefill_dense_contracts.cuh']
    identity = hashlib.sha256(_native().__name__.encode())
    identity.update(headers.headers_sha256.encode())
    for path in [*sources, *included]:
        identity.update(path.name.encode() + b'\0' + path.read_bytes())
    return build.load('tf_sm70_dense_' + identity.hexdigest()[:20],
                      stage_sources(sources, headers=included), need=(7, 0),
                      extra_cflags=['-O3'], extra_cuda_cflags=['-O3'],
                      extra_ldflags=['-lcublas', '-lcublasLt'])


class Context:
    """One explicit device/backend owner; max_plans is caller-admitted capacity.

    Close this context after graphs retire. It fences once and retires all
    registered plans, including partially prepared plans. No per-call loans or
    unbounded geometry cache are kept. One owner thread serializes launches.
    After draining/joining the previous host owner, handoff_to_current_thread
    fences the device and transfers ownership. No capture may be active during
    handoff or close. A control thread may close after workers/graphs retire.
    """
    def __init__(self, device, *, backend, max_plans):
        # Retain the actual Thread owner: native/Python numeric thread IDs can
        # be reused after startup or scheduler thread exit.
        self._thread = threading.current_thread()
        self._closed = False
        self._retirement_failed = False
        self._native = None
        self._plans = []
        if backend not in ('cublas', 'cublaslt', 'cutlass'):
            raise ValueError('unknown dense backend')
        if type(max_plans) is not int or max_plans <= 0:
            raise ValueError('dense max_plans requires a positive caller-admitted integer')
        self._capacity = max_plans
        self.backend = backend
        device = torch.device(device)
        if device.type != 'cuda' or device.index is None:
            raise ValueError('dense context requires an explicit CUDA device index')
        self._native = extension().Context(device.index, backend)
        try:
            self._native.configure()
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                raise_failures(primary, [cleanup])
            raise

    def _check(self):
        if threading.current_thread() is not self._thread:
            raise RuntimeError('dense context belongs to its current owner host thread')
        if self._closed or self._retirement_failed:
            raise RuntimeError('dense context is closed or retirement uncertain')

    def handoff_to_current_thread(self):
        """Cold ownership transfer; caller has drained the prior host owner."""
        if self._closed or self._retirement_failed:
            raise RuntimeError('dense context is closed or retirement uncertain')
        try:
            self._native.handoff_to_current_thread()
        except BaseException:
            self._retirement_failed = True
            raise
        self._thread = threading.current_thread()

    def close(self):
        if self._closed:
            return
        if self._retirement_failed:
            raise RuntimeError('dense context retirement is uncertain; retry is forbidden')
        # Reserve failure slots before retirement; every independent plan is
        # attempted even when an earlier descriptor retirement failed.
        errors = [None] * len(self._plans)
        primary = None
        try:
            self._native.close()
        except BaseException as error:
            self._retirement_failed = True
            if not self._native.retirement_fenced():
                raise
            primary = error
        failed = False
        for index, plan in enumerate(self._plans):
            try:
                plan.close()
            except BaseException as error:
                errors[index] = error
                failed = True
        if primary is not None or failed:
            self._retirement_failed = True
            raise_failures(primary, [error for error in errors if error is not None])
        self._closed = True
        self._native = None
        self._plans.clear()

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


class Prepared:
    """Geometry handle; its Context owns retirement, including failed setup.

    Plans are not standalone context managers. Retire graphs/consumers, then
    close the Context, which fences once and closes all registered plans.
    """
    def __init__(self, context, x, weight, out):
        self._native = None
        self._closed = False
        self._failed = False
        self._retirement_failed = False
        if type(context) is not Context:
            raise ValueError('dense plan requires its explicit Context')
        context._check()
        if len(context._plans) >= context._capacity:
            raise RuntimeError('dense prepared plan capacity exhausted')
        if any(not _plain(t) for t in (x, weight, out)):
            raise ValueError('dense candidates require plain non-grad tensors')
        # Journal before acquisitions/warmup: failed preparation remains owned
        # by Context.close, including uncertain native work or acquisition.
        self._context = weakref.ref(context)
        self._native = extension().Prepared(context._native, x, weight, out)
        try:
            context._plans.append(self)
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                raise_failures(primary, [cleanup])
            raise
        self._native.configure()
        self._native.warm(x, weight, out)
        if not out.is_inference():
            torch.autograd.graph.increment_version(out)

    def _check(self):
        if self._closed or self._failed:
            raise RuntimeError('dense owner is closed or failed')
        context = self._context()
        if context is None:
            raise RuntimeError('dense context owner has retired')
        context._check()

    def run(self, x, weight, out):
        self._check()
        if any(not _plain(t) for t in (x, weight, out)):
            raise ValueError('dense candidates require plain non-grad tensors')
        try:
            self._native.run(x, weight, out)
        except BaseException:
            self._failed = True
            raise
        if not out.is_inference():
            torch.autograd.graph.increment_version(out)
        return out

    def settings(self):
        self._check()
        return self._native.settings()

    def close(self):
        if self._closed:
            return
        if self._retirement_failed:
            raise RuntimeError('dense retirement is uncertain; retry is forbidden')
        if self._native is not None:
            try:
                self._native.close()
            except BaseException:
                self._retirement_failed = True
                self._failed = True
                raise
        self._closed = True
        self._native = None

def prepare(context, x, weight, out):
    """No runtime fallback, algorithm search, workspace allocation or sync in run."""
    return Prepared(context, x, weight, out)

"""Component-only larger-M pedantic FP32 GEMM; no runtime caller selects it.

One prepared owner borrows a fixed device/stream and caller workspace. Each
run retains all operands until finish/close positively fences that stream.
Only one run may be pending. Explicit close (normally a with block) is required;
an abandoned or uncertain native State is retained until process exit rather
than freeing CUDA borrowers (with a bounded diagnostic). Workspace must not be
resized or aliased for other work while open; operand loans must not be mutated
until finish/close. Uncertain retirement is terminal, never retried. Execution
is uncaptured and restricted to the preparing host thread/current CUDA stream.
No vendor-unconditional gamma/error certificate or model qualification is implied.
"""
from functools import lru_cache
import hashlib
from pathlib import Path
import threading

import torch

from tensorfold.cleanup import raise_failures


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .provider import _native
    from .source_cache import stage_sources

    prepared = _native()
    root = Path(__file__).parent
    sources = [root / 'prefill_dense.cpp', root / 'prefill_dense.cu']
    header = root / 'prefill_dense_contracts.cuh'
    identity = hashlib.sha256(prepared.__name__.encode())
    for source in [*sources, header]:
        identity.update(source.name.encode() + b'\0' + source.read_bytes())
    staged = stage_sources(sources, headers=[header])
    return build.load('tf_sm70_prefill_dense_' + identity.hexdigest()[:20], staged, need=(7, 0),
        extra_cuda_cflags=['-O3', '--fmad=false'], extra_cflags=['-O3'], extra_ldflags=['-lcublas'])


def _plain(tensor):
    return (type(tensor) is torch.Tensor and not tensor.requires_grad
            and not tensor.is_neg() and not tensor.is_conj())


class Prepared:
    __slots__ = ('_native', '_workspace', '_operands', '_closed', '_failure', '_retirement_failed', '_thread',
                 '_cleanup_one', '_cleanup_two', '_native_chain')

    def __init__(self, workspace):
        # Reserve Python state before the native handle is acquired.
        self._native = None
        self._workspace = workspace
        self._operands = None
        self._closed = False
        self._failure = None
        self._retirement_failed = False
        self._thread = threading.get_ident()
        self._cleanup_one = [None]
        self._cleanup_two = [None, None]
        self._native_chain = [None, None]
        if not _plain(workspace):
            raise ValueError('prefill Dense workspace requires a plain non-grad tensor')
        try:
            self._native = self._make_native(workspace)
            self._native.configure()
        except BaseException as primary:
            self._rollback(primary)

    def _make_native(self, workspace):
        return extension().Prepared(workspace)

    def _rollback(self, primary):
        # These status slots predate handle acquisition. No reporting or status
        # allocation precedes the independently attempted native retirement.
        errors = ()
        try:
            Prepared.close(self)
        except BaseException as cleanup:
            self._cleanup_one[0] = cleanup
            errors = self._cleanup_one
            try:
                namespace = BaseException.__dict__['__dict__'].__get__(primary)
                dict.__setitem__(namespace, '_tensorfold_prefill_dense_owner', self)
            except BaseException as retention:
                self._cleanup_two[0] = cleanup
                self._cleanup_two[1] = retention
                errors = self._cleanup_two
        try:
            self._native_chain[0] = BaseException.__cause__.__get__(primary)
            self._native_chain[1] = BaseException.__context__.__get__(primary)
            raise_failures(primary, errors)
        except BaseException as transport:
            if transport is primary:
                raise
            # Primary, owner and prepaid statuses remain traceback-owned even
            # if diagnostic transport itself cannot allocate.
            raise primary from transport

    def _thread_check(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError('prefill Dense owner belongs to its preparing host thread')

    def run(self, x, weight, accumulation, out):
        return self._execute((x, weight, accumulation, out), (accumulation, out))

    def _execute(self, operands, outputs):
        self._thread_check()
        if self._closed or self._failure is not None or self._operands is not None:
            raise RuntimeError('prefill Dense owner is closed, failed or has pending work')
        if any(not _plain(t) for t in operands):
            raise ValueError('prefill Dense operands require plain non-grad tensors')
        self._operands = operands
        try:
            self._native.run(*operands)
            # Each raw output publishes its ordinary owner/view mutation exactly once.
            for tensor in outputs:
                if not tensor.is_inference():
                    torch.autograd.graph.increment_version(tensor)
        except BaseException as primary:
            self._failure = primary
            raise
        return outputs[-1]

    def finish(self):
        self._thread_check()
        if self._closed:
            raise RuntimeError('prefill Dense owner is closed')
        if self._retirement_failed:
            raise RuntimeError('prefill Dense retirement is uncertain; retry is forbidden')
        if self._native is not None:
            try:
                self._native.finish()
            except BaseException as primary:
                self._failure = primary
                self._retirement_failed = True
                raise
        self._operands = None

    def close(self):
        self._thread_check()
        if self._closed:
            return
        if self._retirement_failed:
            raise RuntimeError('prefill Dense retirement is uncertain; retry is forbidden')
        if self._native is not None:
            try:
                self._native.close()
            except BaseException as primary:
                self._failure = primary
                self._retirement_failed = True
                raise
        self._closed = True
        self._native = self._workspace = self._operands = None

    def __enter__(self):
        if self._closed or self._failure is not None:
            raise RuntimeError('prefill Dense owner is closed or failed')
        return self

    def settings(self):
        self._thread_check()
        if self._closed:
            raise RuntimeError('prefill Dense owner is closed')
        return self._native.settings()

    def __exit__(self, kind, primary, traceback):
        if primary is None:
            self.close()
        else:
            self._rollback(primary)
        return False


def prepare(workspace):
    """Caller supplies CUDA byte workspace; no fixed4096 or runtime budget claim."""
    return Prepared(workspace)

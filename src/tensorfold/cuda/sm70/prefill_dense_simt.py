"""Unselected pinned CUTLASS SIMT/Float32/direct-Half prefill component.

Only finite supplied Half operands with K<=10240 enter the proposed gamma_K*S
proof; actual emitted RN32 DAG and RN16 epilogue still require target inspection.
Half midpoint/overflow intervals do not certify equal bits/finite output. NaNs,
Infs and signedzero have separate functional controls, not a finite certificate.
No runtime caller selects this component; no model quality/performance claim.

Qualification region: exact protected SDK receipt below, SM70, R1..4096,
N/K1..10240, Half row-major X/output (contiguous columns, disjoint rows),
contiguous Half W[N,K]. Preparation binds actual pointers/strides and resolves
can_implement/workspace/function residency; live values may change only after
positive finish. Bound storage/metadata must remain valid until close. The
inherited owner uses the existing X tensor as its lifetime anchor, not a new
workspace. No Float32 result tensor is allocated. Uncaptured, one pending call,
fixed preparing host thread/current stream; terminal cleanup is never retried.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

from .prefill_dense import Prepared as _Owner, _plain

FOUNDATION_MANIFEST_SHA256 = '491f93dab1f592da675221d60552e19f0ed1aa6b4e803f2eacd082fad19c0e53'
CUTLASS_SOURCE_COMMIT = '2c85f497547eb441f0fe80bfb0fb2e6f9add1719'


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .provider import _native
    from .foundation_headers import selected_headers
    from .source_cache import stage_sources

    selected = selected_headers()
    if selected is None:
        raise RuntimeError('SIMT component requires the protected selected CUTLASS foundation')
    if selected.manifest_sha256 != FOUNDATION_MANIFEST_SHA256:
        raise RuntimeError('SIMT numerical source contract requires the qualified CUTLASS2c85 receipt')
    prepared = _native()
    root = Path(__file__).parent
    sources = [root / 'prefill_dense_simt.cpp', root / 'prefill_dense_simt.cu']
    headers = [root / 'prefill_dense_simt.cuh', root / 'prefill_dense_contracts.cuh']
    identity = hashlib.sha256(prepared.__name__.encode())
    identity.update(selected.headers_sha256.encode())
    for source in [*sources, *headers]:
        identity.update(source.name.encode() + b'\0' + source.read_bytes())
    staged = stage_sources(sources, headers=headers)
    return build.load('tf_sm70_prefill_simt_' + identity.hexdigest()[:20], staged, need=(7, 0),
        extra_cuda_cflags=['-O3', '--fmad=false', '--ftz=false'], extra_cflags=['-O3'])


class Prepared(_Owner):
    __slots__ = ('_binding',)

    def __init__(self, x, weight, out):
        if any(not _plain(t) for t in (x, weight, out)):
            raise ValueError('prefill SIMT requires plain non-grad tensors')
        self._binding = (x, weight, out)
        super().__init__(x)
        self._binding = None  # Native State retains all bound tensors until positive close.

    def _make_native(self, x):
        return extension().Prepared(*self._binding)

    def run(self, x, weight, out):
        return self._execute((x, weight, out), (out,))


def prepare(x, weight, out):
    """Bind actual geometry/strides/pointers and load the exact function cold."""
    return Prepared(x, weight, out)

"""Private pointer-batched cuBLAS GEMV experiment; no production selector change.

The shared matrix is never replicated. The plan owns fixed FP32 staging and
pointer tables before use, and only one serial writer on its construction
stream may use it. Numerical equivalence to canonical row GEMV is UNVERIFIED.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

MAX_ROWS = 32
HIDDEN = 2560
EXPERTS = 513
OUTPUT_STRIDE = 516
HOST_BYTES_PER_PLAN = 16 << 10


def workspace_bytes(rows):
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS:
        raise ValueError("private router batch capacity must be1..32")
    return rows * ((HIDDEN + OUTPUT_STRIDE) * 4 + 3 * 8)


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .provider import _native
    from .source_cache import stage_sources
    prepared = _native()
    source = Path(__file__).with_suffix(".cpp")
    key = hashlib.sha256(prepared.__name__.encode() + source.read_bytes()).hexdigest()[:20]
    return build.load("tf_router_blas_" + key, stage_sources([source]), need=(7,0),
                      extra_cflags=["-O3"], extra_ldflags=["-lcublas"], with_cuda=True)


def prepare(weight, *, capacity=MAX_ROWS):
    """Operator prototype only. Model admission/selection remains untouched."""
    workspace_bytes(capacity)
    return extension().RouterPlan(weight, capacity)

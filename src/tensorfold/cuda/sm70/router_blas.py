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


PROFILE = "source-batch-router-blas-trial"
MAX_WEIGHTS = 48
HOST_BASE_BYTES = 64 << 10


def role_bytes(rows, weights):
    """Persistent GPU payload and conservative host construction/retention bound.

    One FP32 X/Y and two pointer tables are shared by all serial layer calls.
    Each immutable weight owns only its repeated-A table and retained handles.
    Host allowance includes Python/C++ owners, dictionaries, tensor/view headers,
    temporary pointer-table construction and bounded diagnostic integer updates.
    """
    workspace_bytes(rows)
    if type(weights) is not int or not 1 <= weights <= MAX_WEIGHTS:
        raise ValueError("router role must own1..48 weights")
    def rounded(size):
        return ((size + 511) // 512) * 512
    return (rounded(rows * HIDDEN * 4) + rounded(rows * OUTPUT_STRIDE * 4)
            + (weights + 2) * rounded(rows * 8),
            HOST_BASE_BYTES + weights * HOST_BYTES_PER_PLAN)


def role_payload_bytes(rows, weights):
    """Raw tensor payload; admission separately charges each512-byte granule."""
    role_bytes(rows, weights)
    return rows * ((HIDDEN + OUTPUT_STRIDE) * 4 + (weights + 2) * 8)



class RolePlans:
    """One Buffers-owned serial writer, on its construction CUDA stream.

    Weight objects and original storages stay retained through graph retirement.
    Prefill stays canonical. This explicit private numeric implementation has
    different FP32 logit bits; model quality is unqualified, not assumed equal.

    A graph owner may exclusively borrow a dedicated capture-stream role:
    ``run`` still executes only on that construction stream while recording.
    Its captured kernels may then replay on the graph owner's ordered stream,
    retaining this role's tensors until positive graph retirement. No host run
    may concurrently reuse that borrowed scratch; ordinary eager roles are not
    transferred to graph capture.
    """

    def __init__(self, weights, rows):
        gpu, host = role_bytes(rows, len(weights))
        self.plans = {}
        self.owners = tuple(weights)
        self.gpu_bytes, self.host_bytes = gpu, host
        self.gpu_payload_bytes = role_payload_bytes(rows, len(weights))
        self.calls = 0
        self.overflowed = False
        shared = None
        for weight in self.owners:
            key = id(weight)
            if key in self.plans:
                raise ValueError("router role weights must be distinct owned tensors")
            plan = extension().RouterPlan(weight, rows, shared)
            self.plans[key] = plan
            if shared is None:
                shared = plan

    def run(self, weight, value, out):
        plan = self.plans.get(id(weight))
        if plan is None:
            raise ValueError("router weight was not admitted into this role")
        result = plan.run(value, out)
        if self.calls < (1 << 63) - 1:
            self.calls += 1
        else:
            self.overflowed = True
        return result


def create_role(w, rows, *, prefill, role, eager=True):
    """Graph-only buffers and prefill keep their original numerical route."""
    if not eager or prefill or w.meta.get("sm70_profile") != PROFILE:
        return None
    if type(rows) is not int or not 1 <= rows <= MAX_ROWS:
        raise ValueError("private router requires the admitted row32 envelope")
    weights = [w.mtp.layer.moe.router] if role == "mtp" else [layer.moe.router for layer in w.layers]
    return RolePlans(weights, MAX_ROWS)

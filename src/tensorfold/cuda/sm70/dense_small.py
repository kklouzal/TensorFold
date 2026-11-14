"""SM70 weight-streaming tensor-core GEMM for FP16 dense projections with 1..32 rows.

``out[:rows] = x[:rows] @ w.T`` with FP32 accumulation and one rounding to the
output dtype (FP16, or FP32 for tensor-parallel partials and logits). Each
weight element is read once per call regardless of the row count. Small-N
shapes split K across CTAs and reduce deterministically (fixed split order)
through a caller-owned FP32 workspace and self-resetting int32 arrival
counters; the launch plan depends only on (rows, N, K). Reduction
order differs from cuBLAS. Capturable: no allocation, no host synchronization.

Workspace contract: one ``Workspace`` serves calls issued in one stream order
(a Buffers' exclusive CUDA lifetime). Concurrent launches on different streams
must not share it. Counters start zero and every completed launch leaves them
zero.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch

POLICY = "sm70-small-m-wmma-stream-v1"
MAX_ROWS = 32
CUDA_FLAGS = ("-O3", "--fmad=false")


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .source_cache import stage_sources

    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("small dense GEMM requires an SM70 device")
    root = Path(__file__).parent
    sources = [root / "dense_small_bindings.cpp", root / "dense_small.cu"]
    digest = hashlib.sha256(repr(CUDA_FLAGS).encode())
    for source in sources:
        digest.update(source.name.encode() + b"\0" + source.read_bytes())
    staged = stage_sources(sources, headers=[], source_root=root.parent)
    return build.load("tf_sm70_dense_small_" + digest.hexdigest()[:20], staged, need=(7, 0),
                      extra_cuda_cflags=list(CUDA_FLAGS), extra_cflags=["-O3"])


# Fixed bound from the native plan (two CTAs per SM target on 80-SM V100):
# 2 * 160 CTAs * 32 rows * 64 columns FP32 partials and 160 int32 counters.
WORKSPACE_FLOATS = 2 * 160 * 32 * 64
WORKSPACE_COUNTERS = 160


def workspace_bytes():
    return 4 * (WORKSPACE_FLOATS + WORKSPACE_COUNTERS)


@lru_cache(maxsize=None)
def _plan(rows, n, k):
    return tuple(extension().requirement(rows, n, k))


def selected(x, weight, out):
    """Admissible layout and measured faster than the incumbent fixed-M32 cuBLAS call.

    Rows 17..32 with a K split and FP32 (tensor-parallel partial) output measured
    slower than cuBLAS on V100 (D2560/D768 TP8 out and HC down projections), so
    those calls keep the incumbent path.
    """
    if not (x.dtype == torch.float16 and weight.dtype == torch.float16 and x.ndim == 2 and weight.ndim == 2
            and 1 <= x.shape[0] <= MAX_ROWS and x.shape[1] == weight.shape[1] and x.shape[1] % 8 == 0
            and x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
            and weight.is_contiguous() and weight.data_ptr() % 16 == 0
            and out.dtype in (torch.float16, torch.float32) and out.ndim == 2
            and out.shape == (x.shape[0], weight.shape[0]) and out.stride(1) == 1
            and out.stride(0) >= out.shape[1] and x.device == weight.device == out.device and x.is_cuda):
        return False
    rows = x.shape[0]
    return rows <= 16 or out.dtype != torch.float32 or _plan(rows, weight.shape[0], weight.shape[1])[3] == 1


class Workspace:
    """Owned split-K scratch, allocated before capture; also loads the extension."""

    def __init__(self, device):
        device = torch.device(device)
        if device.type == "cuda" and tuple(extension().bound()) != (WORKSPACE_FLOATS, WORKSPACE_COUNTERS):
            raise RuntimeError("small dense workspace bound differs from its native plan")
        self.partials = torch.empty(WORKSPACE_FLOATS, dtype=torch.float32, device=device)
        self.counters = torch.zeros(WORKSPACE_COUNTERS, dtype=torch.int32, device=device)

    def tensors(self):
        return (self.partials, self.counters)


def matmul(x, weight, out, workspace):
    """Caller checked ``selected``; native validation still rejects bad layouts."""
    extension().dense_small(x, weight, out, workspace.partials, workspace.counters)
    return out

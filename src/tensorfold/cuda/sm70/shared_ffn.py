"""SM70 decode shared-expert SwiGLU FFN for 1..64 FP16 rows: two launches instead of the padded-M32 reference
(eight launches: pad, GEMM, split-K reduce and copy per projection, SiLU, product, cast).

``out = down @ fp16(silu(gate @ x) * (up @ x))`` with the reference program's operation boundaries (FP32 sums,
FP32 SiLU and product, one FP16 rounding, FP32 output); the summation order differs from cuBLAS and is fixed per
row, independent of the call's row count. Capturable: no workspace beyond the caller's (or a fresh) FP16 activation.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch

# Every call of 1..64 rows takes this program: a lower row limit would compute a token's shared output with
# different arithmetic depending on how many rows share its call (decode/verify row counts vary run to run).
# V100, 48 distinct layers in one graph, us/layer new vs padded-M32 reference (two M32 panels above 32 rows):
# width 80 (SC117 TP8, the only width the owned ETP4 path binds) 6.5/10.9/22.0/38.2 vs 42.8/38.3/38.0/38.9 at
# 1/5/16/32 rows and 39.2/46.0/53.8/69.6 vs 77.4/77.4/78.0/77.8 at 33/40/48/64 rows (error vs FP64 equal to the
# reference's at every row count); width 128 8.7/14.3/31.5/56.5 vs 43.7/39.9/38.5/39.1 at 1/5/16/32 rows.
MAX_ROWS = 64


CUDA_FLAGS = ("-O3", "--fmad=false")


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .provider import _native
    from .source_cache import stage_sources

    prepared = _native()  # The selected SM70/toolkit/ABI admission and identity, as gdn_native.
    root = Path(__file__).parent
    sources = [root / "shared_ffn_bindings.cpp", root / "shared_ffn.cu"]
    header = root.parent / "kernels/tensor_contracts.cuh"
    digest = hashlib.sha256(prepared.__name__.encode() + repr(CUDA_FLAGS).encode())
    for source in [*sources, header]:
        digest.update(source.name.encode() + b"\0" + source.read_bytes())
    staged = stage_sources(sources, headers=[header])
    return build.load("tf_sm70_shared_ffn_" + digest.hexdigest()[:20], [str(p) for p in staged], need=(7, 0),
                      extra_cuda_cflags=list(CUDA_FLAGS), extra_cflags=["-O3"])


def selected(x, gate, up, down):
    """Decode rows on SM70 with the supported geometry; other calls keep the reference program."""
    return (x.is_cuda and x.dtype == torch.float16 and x.ndim == 2 and 1 <= x.shape[0] <= MAX_ROWS
            and x.shape[1] % 256 == 0 and x.shape[1] <= 2560 and x.stride(1) == 1 and x.stride(0) % 8 == 0
            and x.data_ptr() % 16 == 0 and gate.shape == up.shape == (down.shape[1], x.shape[1])
            and 2 <= gate.shape[0] <= 256 and gate.shape[0] % 2 == 0 and down.shape[0] % 32 == 0
            and all(t.dtype == torch.float16 and t.is_contiguous() and t.device == x.device and t.data_ptr() % 16 == 0
                    for t in (gate, up, down))
            and torch.cuda.get_device_capability(x.device) == (7, 0))


def run(x, gate, up, down):
    """FP32 [rows, N] shared-expert output; caller checked ``selected``."""
    act = torch.empty((x.shape[0], gate.shape[0]), dtype=torch.float16, device=x.device)
    out = torch.empty((x.shape[0], down.shape[0]), dtype=torch.float32, device=x.device)
    extension().run(x, gate, up, down, act, out)
    return out

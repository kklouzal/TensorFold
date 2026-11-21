"""DeepSeek-V4.1 routed-expert prefill for SM70: FP16 dequantized tiles and cuBLAS tensor-op GEMMs.

The operands are ``onecat_ds41.v41_experts``'s (one EP/ETP rank's original GGUF
expert bytes: gate/up [E, 576, 20*block], down [E, 5120, 3*block] with ``left``
in 0/64/128/192; FP16 x [M, 5120]; int32 local ids and FP32 probabilities
[M, 6]) for any M >= 1, and the output is the same FP16 [M, 5120] partial.

Arithmetic (see ds41_prefill_moe.cu): routes with probability 0 are dropped
(exactly +/-0 in the reference). Weights are ggml-quants.c FP32 products rounded
once to FP16; activations stay FP16 (no Q8_1); gate/up accumulate in FP32 and
stay FP32 into the clamped SwiGLU times the probability, which rounds to FP16;
down accumulates in FP32 (no reduced-precision split-K) and rounds per route to
FP16; the token output is the FP32 sum of its routes in top-k order, rounded
to FP16. A routed id outside [0, E) raises after the call's one host
synchronization (before any expert work is enqueued).

Not graph-capturable: one stream synchronization per call and current-stream
caching-allocator workspace, released at return. Workspace per call:
``workspace_bytes``. The caller retains operands until current-stream
completion.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch

HIDDEN, ROWS, TOP = 5120, 576, 6
EXPERT_TYPES = (16, 17, 29)
TILE, ROUTE_CAP = 32, 16384


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .source_cache import stage_sources

    if torch.cuda.get_device_capability() != (7, 0) or "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError("DeepSeek-V4.1 prefill experts require a Volta-enabled PyTorch build and SM70 device")
    here = Path(__file__).parent
    sources = [here / "ds41_prefill_moe.cpp", here / "ds41_prefill_moe.cu"]
    headers = [here / "ds41_ggml_decode.cuh", here / "onecat_ds41_codebooks.cuh"]
    flags = ["-O3", "--fmad=false"]
    identity = hashlib.sha256(repr(flags).encode())
    for path in [*sources, *headers]:
        identity.update(path.name.encode() + b"\0" + path.read_bytes())
    staged = stage_sources(sources, headers=headers)
    return build.load("tf_ds41_prefill_moe_" + identity.hexdigest()[:20], staged, need=(7, 0),
                      extra_cflags=["-O3"], extra_cuda_cflags=flags, extra_ldflags=["-lcublas"])


def prepare():
    """Build/load the extension before serving (otherwise the first call does)."""
    return extension()


def workspace_bytes(routes, tile=TILE, route_cap=ROUTE_CAP, max_expert_routes=None):
    """Upper bound of one call's device workspace for ``routes`` nonzero-probability routes."""
    largest = routes if max_expert_routes is None else max_expert_routes
    tile_routes = min(routes, max(route_cap, largest))
    weights = tile * (2 * ROWS * HIDDEN + HIDDEN * ROWS) * 2
    per_tile_route = HIDDEN * 2 + 2 * ROWS * 4 + ROWS * 2
    return weights + tile_routes * per_tile_route + routes * (HIDDEN * 2 + 24)


def experts(x, ids, probabilities, gate, up, down, gate_type, down_type, left, *, out=None,
            rows=64, tile=TILE, route_cap=ROUTE_CAP):
    """Routed experts for FP16 x [M, 5120], top-6 local ids/probabilities -> FP16 [M, 5120].

    ``rows``: 64 or 128 routes per fused-GEMM tile; 0 selects the dequantized-tile cuBLAS path
    (``tile`` experts per FP16 weight tile, ``route_cap`` routes per tile).
    """
    if gate_type not in EXPERT_TYPES or down_type not in EXPERT_TYPES:
        raise ValueError("V4.1 prefill experts require IQ2_XXS/IQ2_XS/IQ1_M carriers")
    if out is None:
        out = torch.empty_like(x)
    extension().experts(out, x, ids, probabilities, gate, up, down, int(gate_type), int(down_type), int(left),
                        int(tile), int(route_cap), int(rows))
    return out

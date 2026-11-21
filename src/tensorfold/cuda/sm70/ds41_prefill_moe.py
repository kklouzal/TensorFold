"""DeepSeek-V4.1 routed-expert prefill for SM70: grouped WMMA GEMMs decoding the IQ rows in their K loops.

The operands are ``onecat_ds41.v41_experts``'s (one EP/ETP rank's original GGUF
expert bytes: gate/up [E, 576, 20*block], down [E, 5120, 3*block] with ``left``
in 0/64/128/192; FP16 x [M, 5120]; int32 local ids and FP32 probabilities
[M, 6]) for any M >= 1, and the output is the same FP16 [M, 5120] partial.

Arithmetic (see ds41_prefill_moe.cu): routes with probability 0 are dropped
(exactly +/-0 in the reference). Weights are ggml-quants.c FP32 products rounded
once to FP16; activations stay FP16 (no Q8_1); products accumulate in FP32;
gate/up stay FP32 into the clamped SwiGLU times the probability, which rounds to
FP16; down rounds per route to FP16; the token output is the FP32 sum of its
routes in top-k order, rounded to FP16. Deterministic (fixed reduction orders).
A routed id outside [0, E) raises after the call's one host synchronization,
before any expert work is enqueued.

Not graph-capturable: one stream synchronization per call and current-stream
caching-allocator workspace (``workspace_bytes``), released at return. The
caller retains operands until current-stream completion.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch

HIDDEN, ROWS, TOP = 5120, 576, 6
EXPERT_TYPES = (16, 17, 29)
# Fewest tokens for which this op replaces the decode op (ds41_decode_moe). Set when the decode op was the
# 6-row loop (crossover blk.27 between 65 and 256 tokens). The expert-major decode op is faster still at 384
# tokens (blk.0: 4.07 vs 6.93 ms, gpuq/runs/em2-3), but this op is the more accurate one (FP16 activations,
# FP32 gate/up; no Q8_1), so the threshold stays an accuracy boundary until a perplexity A/B decides it.
MIN_TOKENS = 128


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
                      extra_cflags=["-O3"], extra_cuda_cflags=flags)


def prepare():
    """Build/load the extension before serving (otherwise the first call does)."""
    return extension()


def workspace_bytes(tokens, routes):
    """Upper bound of one call's transient device workspace for ``routes`` nonzero-probability routes."""
    route_tables = tokens * TOP * (4 + 8 + 8 + 8 + 4 + 8)  # bucket, sort keys/order, counts input, position
    tiles = (routes // 64 + 1 + 384) * 16
    return routes * (ROWS + HIDDEN) * 2 + routes * (8 + 4) + route_tables + tiles


def experts(x, ids, probabilities, gate, up, down, gate_type, down_type, left, *, out=None):
    """Routed experts for FP16 x [M, 5120], top-6 local ids/probabilities -> FP16 [M, 5120]."""
    if gate_type not in EXPERT_TYPES or down_type not in EXPERT_TYPES:
        raise ValueError("V4.1 prefill experts require IQ2_XXS/IQ2_XS/IQ1_M carriers")
    if out is None:
        out = torch.empty_like(x)
    extension().experts(out, x, ids, probabilities, gate, up, down, int(gate_type), int(down_type), int(left))
    return out

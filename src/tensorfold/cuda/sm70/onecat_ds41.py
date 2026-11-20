"""DeepSeek-V4.1 original-GGUF routed-expert operators for SM70 (1Cat-vLLM 4d57473db67e).

The integer-dot pipeline is upstream's non-v41k ``_v41_experts`` path, with
the same arithmetic: Q8_1 activations (an explicit activation quantization,
NOT an FP16-exact product), FP32 integer-dot accumulation in a fixed order,
FP16 gate/up rounding, FP32 clamped SwiGLU times the route probability,
an FP16 boundary, Q8_1 again, then the down product with FP16 per-route
rounding and an FP32 route sum. Expert carriers are the original GGUF bytes of
one TP rank: gate/up [E, 576, 20*block], down [E, 5120, 3*block] with ``left``
in 0/64/128/192. E is any 1..65535; a route id outside [0, E) traps the device.
A route with probability 0 contributes exactly (+/-)0 to the down sum.

All operands are CUDA, contiguous and on one device. ``out`` tensors are
caller-owned and fully written; passing them (and preallocating every scratch)
keeps a call graph-capturable. The M6 IQ1_M adaptive gate/up allocates two
tiny device tensors per call from the current-stream caching allocator.
The caller retains operands until current-stream completion.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch

PIN = "4d57473db67eb77a13018d179e4d95169e82044f"
EXPERT_TYPES = (16, 17, 29)  # IQ2_XXS, IQ2_XS, IQ1_M
BLOCK_BYTES = {16: 66, 17: 74, 29: 56}
DEQUANT_LAYOUT = {8: (32, 34), 13: (256, 176), 16: (256, 66), 17: (256, 74), 20: (32, 18), 29: (256, 56)}
HIDDEN, ROWS, TOP = 5120, 576, 6


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .source_cache import stage_sources

    if torch.cuda.get_device_capability() != (7, 0) or "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError("DeepSeek-V4.1 dp4a operators require a Volta-enabled PyTorch build and SM70 device")
    here = Path(__file__).parent
    sources = [here / "onecat_ds41.cpp", here / "onecat_ds41.cu"]
    headers = [here / "onecat_ds41_dp4a.cuh", here / "onecat_ds41_codebooks.cuh"]
    # Upstream's compile options (torch's defaults, -O3; FMA contraction on).
    flags = ["-O3"]
    identity = hashlib.sha256(repr(flags).encode())
    for path in [*sources, *headers]:
        identity.update(path.name.encode() + b"\0" + path.read_bytes())
    staged = stage_sources(sources, headers=headers)
    return build.load("tf_onecat_ds41_" + identity.hexdigest()[:20], staged, need=(7, 0),
                      extra_cflags=["-O3"], extra_cuda_cflags=flags)


def prepare():
    """Build/load the extension outside capture (otherwise the first call does)."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare the DeepSeek-V4.1 extension before graph capture")
    return extension()


def _empty(out, shape, dtype, like):
    if out is None:
        return torch.empty(shape, dtype=dtype, device=like.device)
    return out


def dequantize(packed, ggml_type, cols, out=None):
    """Original GGML rows uint8 [rows, cols/block*bytes] -> FP16 [rows, cols].

    Types 8 Q8_0, 13 Q5_K, 16 IQ2_XXS, 17 IQ2_XS, 20 IQ4_NL, 29 IQ1_M. Values are
    ggml-quants.c's FP32 reference products rounded once to FP16. ``out`` must be
    16-byte aligned (any torch allocation or a whole-row slice of one).
    """
    if ggml_type not in DEQUANT_LAYOUT:
        raise ValueError(f"unsupported GGML dequantize type {ggml_type}")
    out = _empty(out, (packed.shape[0], cols), torch.float16, packed)
    extension().dequantize(out, packed, int(ggml_type))
    return out


def quantize_q8_1(x, out=None):
    """FP16 [M, K] (M <= 20, K % 256 == 0) -> Q8_1 blocks uint8 [M, K/32, 36]."""
    out = _empty(out, (x.shape[0], x.shape[1] // 32, 36), torch.uint8, x)
    extension().quantize_q8_1(out, x)
    return out


def v41_gate_up(q8, ids, gate, up, ggml_type, out=None):
    """Q8_1 [M<=6,160,36] x routed gate/up rows -> FP16 [M, 6, 2, 576] (gate, up).

    M == 6 with IQ1_M uses upstream's adaptive schedule (a device-side choice
    between per-route CTAs and per-expert reuse; identical per-route arithmetic).
    """
    out = _empty(out, (q8.shape[0], TOP, 2, ROWS), torch.float16, q8)
    if q8.shape[0] == 6 and ggml_type == 29:
        extension().gate_adaptive(out, q8, ids, gate, up, int(ggml_type))
    else:
        extension().gate_up(out, q8, ids, gate, up, int(ggml_type))
    return out


def v41_swiglu_q8(projections, probabilities, out=None):
    """FP16 [M,6,2,576] and FP32 [M,6] -> Q8_1 [M,6,18,36] of fp16(silu(g)*u*p)."""
    out = _empty(out, (projections.shape[0], TOP, 18, 36), torch.uint8, projections)
    extension().swiglu_q8(out, projections, probabilities)
    return out


def v41_down(down_q8, ids, down, ggml_type, left, out=None):
    """Q8_1 [M,6,18,36] x routed down rows [E,5120,3*block] -> FP16 [M, 5120].

    Upstream's q8_down_tiled with unit route weights (the probabilities were
    applied before the Q8_1 boundary) and its default tile choice.
    """
    out = _empty(out, (down_q8.shape[0], HIDDEN), torch.float16, down_q8)
    extension().down(out, down_q8, ids, down, int(ggml_type), int(left))
    return out


def shared_swiglu(gate, up, out=None):
    """Shared expert FP16 [M, N] gate/up -> fp16(silu(min(g,10)) * clamp(u,-10,10))."""
    out = _empty(out, tuple(gate.shape), torch.float16, gate)
    extension().shared_swiglu(out, gate, up)
    return out


def v41_experts(x, ids, probabilities, gate, up, down, gate_type, down_type, left):
    """Routed experts for FP16 x [M<=6, 5120], top-6 ids/probabilities -> FP16 [M, 5120]."""
    if (x.dtype != torch.float16 or x.ndim != 2 or not 1 <= x.shape[0] <= 6 or x.shape[1] != HIDDEN
            or tuple(ids.shape) != (x.shape[0], TOP) or tuple(probabilities.shape) != tuple(ids.shape)
            or probabilities.dtype != torch.float32
            or gate_type not in EXPERT_TYPES or down_type not in EXPERT_TYPES):
        raise ValueError("V4.1 integer experts require FP16 M1..6 x5120, top-6 and IQ1_M/IQ2 carriers")
    q8 = quantize_q8_1(x)
    projections = v41_gate_up(q8, ids, gate, up, gate_type)
    down_q8 = v41_swiglu_q8(projections, probabilities)
    return v41_down(down_q8, ids, down, down_type, left)

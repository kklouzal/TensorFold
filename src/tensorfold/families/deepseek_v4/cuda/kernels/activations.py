# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Provenance: 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f
#   vllm/models/deepseek_v41/activations.py (blob 434d2af62cf3a7d4f5394bbf0e75399439403000), which registers
#   torch.ops.vllm.deepseek_v41_swiglu -> torch.ops._C.deepseek_v41_swiglu_sm70_out, implemented in
#   csrc/sm70_turbomind/ops/gguf_dp4a_sm70.cu (blob daf0385c453de5b778f790c6c6b14de098ef22d6 at the same commit):
#     v41_swiglu(gate, up, 1): gate = gate > 10 ? 10 : gate; up = up > 10 ? 10 : up; up = up < -10 ? -10 : up;
#       activated = __fdiv_rn(gate, __fadd_rn(1, expf(-gate))); return __fmul_rn(__fmul_rn(activated, up), 1)
#     v41_shared_swiglu: out[i] = __float2half_rn(v41_swiglu(half2float(gate[i]), half2float(up[i]), 1))
#   built into vLLM's _C extension without --use_fast_math (expf is the full-precision CUDA expf).
# TensorFold changes:
# - @triton.jit -> triton_utils.jit: the same JITFunction; launches add the SM70 Triton 3.2 profile options
#   (ptx_version=83, needed with CUDA 12.9 ptxas). Kernel arithmetic unchanged.
# - The C op and its custom-op registration are replaced by one Triton kernel with the same FP32 operations:
#   NaN-preserving clamps, libdevice expf (tldevice.exp, CUDA's expf implementation), round-to-nearest div_rn,
#   no FMA contraction (enable_fp_fusion=False), the exact *1 probability multiply, and an RNE FP16 store.
#   tests/cuda/deepseek_v41/test_deepseek_v41_activations.py checks bit identity against the verbatim CUDA source.
"""FP32 clamped SwiGLU with the original FP16 output boundary."""

import torch

from .triton_utils import jit, on_triton_device, tl, tldevice, triton

_BLOCK = 1024


@jit
def _swiglu(gate, up, out, COUNT, BLOCK: tl.constexpr):
    at = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    live = at < COUNT
    g = tl.load(gate + at, live, 0.0).to(tl.float32)
    u = tl.load(up + at, live, 0.0).to(tl.float32)
    g = tl.where(g > 10.0, 10.0, g)
    u = tl.where(u > 10.0, 10.0, u)
    u = tl.where(u < -10.0, -10.0, u)
    activated = tl.div_rn(g, 1.0 + tldevice.exp(-g))
    tl.store(out + at, ((activated * u) * 1.0).to(tl.float16), live)


def swiglu(gate, up):
    """gate, up: matching contiguous FP16 [rows, cols] on one device; returns FP16 [rows, cols]."""
    if not (
        on_triton_device(gate)
        and gate.is_contiguous()
        and gate.dim() == 2
        and gate.dtype == torch.float16
        and up.device == gate.device
        and up.is_contiguous()
        and up.dtype == torch.float16
        and up.shape == gate.shape
    ):
        raise ValueError("V4.1 shared SwiGLU requires matching contiguous FP16 matrices")
    output = torch.empty_like(gate)
    if gate.numel():
        _swiglu[(triton.cdiv(gate.numel(), _BLOCK),)](
            gate, up, output, gate.numel(), BLOCK=_BLOCK, num_warps=4, enable_fp_fusion=False
        )
    return output

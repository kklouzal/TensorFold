"""V4.1 shared SwiGLU: the Triton replacement is bit-identical to the upstream CUDA kernel.

Independent oracle: the upstream device function and kernel from 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f
csrc/sm70_turbomind/ops/gguf_dp4a_sm70.cu (v41_swiglu, v41_shared_swiglu) compiled verbatim with nvcc and the
extension's flags (no --use_fast_math). Every FP16 gate value is paired with a spread of up values, including the
clamp boundaries, signed zeros, infinities and NaN.
"""

import pytest
import torch

from tensorfold.families.deepseek_v4.cuda.kernels.activations import swiglu

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_UPSTREAM = r"""
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ float v41_swiglu(float gate, float up, float probability) {
  gate = gate > 10.f ? 10.f : gate;
  up = up > 10.f ? 10.f : up;
  up = up < -10.f ? -10.f : up;
  const float activated = __fdiv_rn(gate, __fadd_rn(1.f, expf(-gate)));
  return __fmul_rn(__fmul_rn(activated, up), probability);
}

__global__ void v41_shared_swiglu(half* out, const half* gate, const half* up,
                                  int64_t count) {
  const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count)
    out[index] = __float2half_rn(
        v41_swiglu(__half2float(gate[index]), __half2float(up[index]), 1.f));
}

void upstream_swiglu(torch::Tensor out, torch::Tensor gate, torch::Tensor up) {
  v41_shared_swiglu<<<(gate.numel() + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<half*>(out.data_ptr()), reinterpret_cast<const half*>(gate.data_ptr()),
      reinterpret_cast<const half*>(up.data_ptr()), gate.numel());
}
"""


def _upstream():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        "tf_v41_swiglu_oracle",
        cpp_sources="void upstream_swiglu(torch::Tensor out, torch::Tensor gate, torch::Tensor up);",
        cuda_sources=_UPSTREAM,
        functions=["upstream_swiglu"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"],
    )


def _same(actual, expected):
    # Bit identity for every non-NaN result; NaN results must be NaN at the same places (payloads are not part of
    # the contract).
    nan = torch.isnan(expected)
    return torch.equal(torch.isnan(actual), nan) and torch.equal(
        actual.view(torch.int16)[~nan], expected.view(torch.int16)[~nan]
    )


def test_swiglu_bit_identical_to_upstream_cuda():
    oracle = _upstream()
    gate = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.float16)
    ups = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, 0.5, 3.0, -7.25, 9.99, 10.0, 10.01, -10.0, -10.01, 1e4, -1e4, 6e-8,
         float("inf"), float("-inf"), float("nan")],
        dtype=torch.float16,
    ).tolist() + torch.randn(46, generator=torch.Generator().manual_seed(41)).mul(4).tolist()
    for value in ups:
        up = torch.full_like(gate, value)
        expected = torch.empty_like(gate)
        oracle.upstream_swiglu(expected, gate, up)
        actual = swiglu(gate.view(256, 256), up.view(256, 256)).view(-1)
        assert _same(actual, expected), value
    # Every up value against a spread of gates, as a [rows, cols] matrix.
    up = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.float16)
    for value in (-12.0, -1.5, -1e-3, 0.0, 0.7, 4.0, 10.0, 11.0):
        g = torch.full_like(up, value)
        expected = torch.empty_like(up)
        oracle.upstream_swiglu(expected, g, up)
        actual = swiglu(g.view(512, 128), up.view(512, 128)).view(-1)
        assert _same(actual, expected), value


def test_swiglu_rejects_unqualified_inputs():
    a = torch.ones(4, 8, dtype=torch.float16, device="cuda")
    for gate, up in ((a.float(), a), (a, a[:, :4]), (a.t(), a.t()), (a[0], a[0])):
        with pytest.raises(ValueError, match="shared SwiGLU"):
            swiglu(gate, up)

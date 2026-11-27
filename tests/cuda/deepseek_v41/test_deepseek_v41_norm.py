# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f tests/models/test_deepseek_v41_norm.py (blob 295961d36bbf1dd65a2e7b8f3bddc8f8ce3197bf).
# TensorFold changes: vendored imports and wide row and validation coverage.
"""Fused V4.1 RMS preserves FP32 arithmetic and the FP16 store boundary."""

import pytest
import torch

from tensorfold.families.deepseek_v4.cuda.kernels.norm import rms_norm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("hidden", [128, 512, 1280, 5120])
@pytest.mark.parametrize("magnitude", [0.0, 2**-20, 0.03125, 1.37, 100.0, 10000.0])
def test_short_rms_fp32_reference_row_invariance_and_graph(hidden, magnitude):
    generator = torch.Generator(device="cuda").manual_seed(4190)
    x = (torch.randn(6, hidden, generator=generator, device="cuda") * magnitude).half()
    weight = torch.rand(hidden, generator=generator, device="cuda") + 0.5
    eps = 1e-20
    rms_norm(x, weight, eps)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = rms_norm(x, weight, eps)
    for multiplier in (-0.75, 1.0):
        x.mul_(multiplier)
        weight.mul_(0.9)
        graph.replay()
        values = x.float()
        expected = (
            values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps) * weight
        ).half()
        torch.testing.assert_close(output, expected, rtol=0.001, atol=2e-6)
        for rows in range(1, 7):
            partial = rms_norm(x[:rows], weight, eps)
            torch.testing.assert_close(partial, output[:rows], rtol=0, atol=0)


def test_prefill_rms_preserves_other_layouts_and_dtypes():
    weight = torch.ones(512, dtype=torch.float32, device="cuda")
    x = torch.arange(1024, dtype=torch.float16, device="cuda").repeat((65, 1))
    for values, w in ((x[:, ::2], weight), (x[:, :512].float(), weight),
                      (x[:, :512].contiguous(), weight.half()),
                      (x[:, :512].contiguous(), weight[:1]), (x[:0, :512], weight)):
        source = values.float()
        expected = (source * torch.rsqrt(source.square().mean(-1, keepdim=True) + 1e-20) * w).to(values.dtype)
        torch.testing.assert_close(rms_norm(values, w, 1e-20), expected, rtol=0, atol=0)


@pytest.mark.parametrize("rows,hidden", [
    (65, 128), (65, 512), (65, 1280), (65, 5120),
    (257, 128), (257, 512), (257, 1280),
    (8192, 128), (8192, 512), (8192, 1280),
])
def test_prefill_rms_preserves_torch_rounding(rows, hidden):
    generator = torch.Generator(device="cuda").manual_seed(4210 + hidden)
    x = torch.randn(rows, hidden, generator=generator, dtype=torch.float16, device="cuda")
    x[0].zero_()
    x[64].mul_(2.0 ** -19)
    x[-1, :4] = torch.tensor([-65504, 65504, 2.0 ** -24, -0.0], device="cuda")
    weight = torch.rand(hidden, generator=generator, device="cuda") + 0.5
    actual = rms_norm(x, weight, 1e-20)
    source = x.float()
    expected_torch = (source * torch.rsqrt(source.square().mean(-1, keepdim=True) + 1e-20) * weight).half()
    assert torch.equal(actual.view(torch.int16), expected_torch.view(torch.int16))
    for current in (slice(0, 6), slice(64, 67), slice(-5, None)):
        values = x[current].double()
        expected = (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-20) * weight).half()
        torch.testing.assert_close(actual[current], expected, rtol=0.001, atol=2e-6)

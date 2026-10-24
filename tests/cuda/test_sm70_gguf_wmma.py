"""Optional SM70 WMMA dense provider has fixed row arithmetic and FP32 accumulation."""

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 WMMA provider")


@pytest.mark.parametrize("n,k", [(4, 2560), (80, 2560), (320, 10240), (10240, 320),
                                (1280, 2560), (2560, 768), (1536, 2560), (512, 2560),
                                (324, 10240), (2060, 2560), (2688, 2560)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_wmma_actual_projection_geometries_match_independent_fp64_and_serial_rows(n, k, dtype):
    torch.manual_seed(2517)
    weight = torch.randn((n, k), device="cuda", dtype=torch.float16) * .02
    linear = GGUFDense(weight)
    linear.implementation = "wmma"
    x = torch.randn((18, k), device="cuda", dtype=torch.float16)
    output = torch.empty((18, n), device="cuda", dtype=dtype)
    assert linear.matmul(x, out=output) is output
    expected = x.cpu().double() @ weight.cpu().double().T
    torch.testing.assert_close(output.cpu().double(), expected,
                               rtol=.001 if dtype == torch.float16 else .0003, atol=5e-5)
    serial = torch.cat([linear.matmul(x[row:row + 1], f32=dtype == torch.float32) for row in range(18)])
    torch.testing.assert_close(output, serial, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [0, 1, 3, 6, 18, 32, 33])
def test_wmma_row_controls_and_large_prompt_fallback(rows):
    torch.manual_seed(2518)
    linear = GGUFDense(torch.randn((65, 256), device="cuda", dtype=torch.float16) * .02)
    linear.implementation = "wmma"
    x = torch.randn((rows, 256), device="cuda", dtype=torch.float16)
    output = linear.matmul(x, f32=True)
    assert output.shape == (rows, 65)
    if rows == 33:
        torch.testing.assert_close(output, linear.matmul_reference(x, f32=True), rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="at most32"):
            linear.matmul_wmma(x)


def test_wmma_strided_concat_output_graph_replay_and_fixed_provider_identity():
    torch.manual_seed(2519)
    linear = GGUFDense(torch.randn((65, 256), device="cuda", dtype=torch.float16) * .02)
    assert linear.implementation == "reference"
    linear.implementation = "wmma"
    identity = linear.wmma_policy
    x = torch.randn((18, 512), device="cuda", dtype=torch.float16)[:, ::2]
    backing = torch.full((18, 97), -998, device="cuda", dtype=torch.float32)
    output = backing[:, 5:70]
    linear.matmul(x, out=output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        linear.matmul(x, out=output)
    x.mul_(.5)
    graph.replay()
    torch.cuda.synchronize()
    expected = linear.matmul_wmma(x, f32=True)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert torch.all(backing[:, :5] == -998) and torch.all(backing[:, 70:] == -998)
    assert linear.wmma_policy == identity


def test_wmma_keeps_head_and_unaligned_shapes_on_reference():
    head = GGUFDense(torch.zeros((16385, 16), device="cuda", dtype=torch.float16))
    head.implementation = "wmma"
    x = torch.ones((1, 16), device="cuda", dtype=torch.float16)
    torch.testing.assert_close(head.matmul(x), head.matmul_reference(x), rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="non-head"):
        head.matmul_wmma(x)
    odd = GGUFDense(torch.zeros((8, 33), device="cuda", dtype=torch.float16))
    odd.implementation = "wmma"
    y = torch.ones((1, 33), device="cuda", dtype=torch.float16)
    torch.testing.assert_close(odd.matmul(y), odd.matmul_reference(y), rtol=0, atol=0)


def test_wmma_vector_staging_preserves_scalar_fallback_bits_for_unaligned_storage():
    torch.manual_seed(2520)
    x = torch.randn((18, 256), device="cuda", dtype=torch.float16)
    weight = torch.randn((65, 256), device="cuda", dtype=torch.float16) * .02
    shifted_x = torch.empty((18, 257), device="cuda", dtype=torch.float16)[:, 1:]
    shifted_x.copy_(x)
    shifted_weight = torch.empty(weight.numel() + 1, device="cuda", dtype=torch.float16)[1:].reshape(weight.shape)
    shifted_weight.copy_(weight)
    aligned = GGUFDense(weight).matmul_wmma(x, f32=True)
    scalar = GGUFDense(shifted_weight).matmul_wmma(shifted_x, f32=True)
    torch.testing.assert_close(aligned, scalar, rtol=0, atol=0)

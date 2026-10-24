"""Native staging retains the exact fixed-M32 ATen dense arithmetic."""

import pytest
import torch

from tensorfold.cuda.sm70.gguf_ops import dense32
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 dense32 staging")


@pytest.mark.parametrize("rows", [0, 1, 3, 18, 32, 33])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_native_dense_staging_retains_reference_and_serial_bits(rows, dtype):
    torch.manual_seed(719)
    linear = GGUFDense(torch.randn((81, 256), device="cuda", dtype=torch.float16) * .02)
    linear.implementation = "native32"
    x = torch.randn((rows, 256), device="cuda", dtype=torch.float16)
    out = torch.empty((rows, 81), device="cuda", dtype=dtype)
    expected = linear.matmul_reference(x, f32=dtype == torch.float32)
    assert linear.matmul(x, out=out) is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if rows:
        serial = torch.cat([linear.matmul(x[row:row + 1], f32=dtype == torch.float32) for row in range(rows)])
        torch.testing.assert_close(out, serial, rtol=0, atol=0)
    if rows == 33:
        with pytest.raises(RuntimeError, match="at most32"):
            linear.matmul_native32(x, out=out)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_dense32_handles_strided_inputs_and_concat_output_without_overwriting_neighbors(dtype):
    torch.manual_seed(720)
    linear = GGUFDense(torch.randn((80, 256), device="cuda", dtype=torch.float16) * .02)
    x = torch.randn((18, 512), device="cuda", dtype=torch.float16)[:, ::2]
    backing = torch.full((18, 173), -4321, device="cuda", dtype=dtype)
    out = backing[:, 5:85]
    expected = linear.matmul_reference(x, f32=dtype == torch.float32)
    linear.matmul_native32(x, out=out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert torch.all(backing[:, :5] == -4321) and torch.all(backing[:, 85:] == -4321)


def test_dense32_force_f32_retains_stock_mtp_storage_boundary_and_graph_replay():
    torch.manual_seed(721)
    weight = torch.randn((80, 256), device="cuda", dtype=torch.float16) * .02
    x = torch.randn((3, 256), device="cuda", dtype=torch.float16)
    output = torch.empty((3, 80), device="cuda", dtype=torch.float16)
    dense32(x, weight, output, force_f32=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dense32(x, weight, output, force_f32=True)
    x.mul_(.75)
    graph.replay()
    torch.cuda.synchronize()
    padded = torch.zeros((32, 256), device="cuda", dtype=torch.float16)
    padded[:3].copy_(x)
    expected = torch.mm(padded, weight.T, out_dtype=torch.float32)[:3].half()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    oracle = x.cpu().double() @ weight.cpu().double().T
    torch.testing.assert_close(output.cpu().double(), oracle, rtol=.001, atol=2e-5)


def test_dense32_refuses_invalid_output_layout_before_launch():
    linear = GGUFDense(torch.zeros((80, 256), device="cuda", dtype=torch.float16))
    x = torch.zeros((3, 256), device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="output"):
        linear.matmul_native32(x, out=torch.zeros((80, 3), device="cuda", dtype=torch.float16).T)

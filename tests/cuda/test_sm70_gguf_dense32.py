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


def _frozen_m32(linear, x, dtype):
    """Independent incumbent, never calls a candidate dispatch method."""
    out = torch.empty((x.shape[0], linear.n), device=x.device, dtype=dtype)
    for start in range(0, x.shape[0], 32):
        count = min(32, x.shape[0] - start)
        padded = torch.zeros((32, linear.k), device=x.device, dtype=torch.float16)
        padded[:count].copy_(x[start:start + count])
        value = (torch.mm(padded, linear.weight.T, out_dtype=torch.float32) if dtype == torch.float32
                 else torch.mm(padded, linear.weight.T))
        out[start:start + count].copy_(value[:count])
    return out


def _aligned(shape, dtype, *, sentinel=0):
    elements = 1
    for extent in shape:
        elements *= extent
    guard = 512 // torch.empty((), dtype=dtype).element_size()
    owner = torch.full((elements + 3 * guard,), sentinel, dtype=dtype, device="cuda")
    offset = guard + ((-owner.data_ptr()) % 512) // owner.element_size()
    return owner[offset:offset + elements].view(shape), owner, offset


@pytest.mark.parametrize("rows", [1, 31, 32, 33, 63, 64, 65, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("columns,outputs", [(2560, 80), (80, 2560), (320, 10240),
                                           (10240, 320), (2560, 31040), (256, 81)])
def test_native32_fulltiles_and_tails_match_frozen_incumbent_bits(rows, dtype, columns, outputs):
    torch.manual_seed(722)
    linear = GGUFDense(torch.randn((outputs, columns), device="cuda", dtype=torch.float16) * .02)
    linear.implementation = "native32"
    x, _, _ = _aligned((rows, columns), torch.float16)
    x.normal_()
    out, owner, offset = _aligned((rows, outputs), dtype, sentinel=-437)
    expected = _frozen_m32(linear, x, dtype)
    assert linear.matmul(x, out=out) is out
    bits = torch.int16 if dtype == torch.float16 else torch.int32
    assert torch.equal(out.view(bits), expected.view(bits))
    assert torch.all(owner[:offset] == -437) and torch.all(owner[offset + out.numel():] == -437)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_native32_multitile_fallback_views_preserve_bits_and_neighbors(dtype):
    torch.manual_seed(723)
    linear = GGUFDense(torch.randn((81, 256), device="cuda", dtype=torch.float16) * .02)
    linear.implementation = "native32"
    x = torch.randn((65, 512), device="cuda", dtype=torch.float16)[:, ::2]
    owner = torch.full((65, 93), -931, device="cuda", dtype=dtype)
    out = owner[:, 3:84]
    expected = _frozen_m32(linear, x, dtype)
    linear.matmul(x, out=out)
    bits = torch.int16 if dtype == torch.float16 else torch.int32
    assert torch.equal(out.contiguous().view(bits), expected.view(bits))
    assert torch.all(owner[:, :3] == -931) and torch.all(owner[:, 84:] == -931)


def test_native32_multitile_graph_replays_changed_inputs_without_replacing_output():
    torch.manual_seed(724)
    linear = GGUFDense(torch.randn((80, 2560), device="cuda", dtype=torch.float16) * .02)
    linear.implementation = "native32"
    x, _, _ = _aligned((65, 2560), torch.float16)
    out, _, _ = _aligned((65, 80), torch.float32)
    x.normal_()
    linear.matmul(x, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        linear.matmul(x, out=out)
    identity = out.data_ptr()
    for factor in (.5, -.75):
        x.mul_(factor)
        graph.replay()
        torch.cuda.synchronize()
        assert out.data_ptr() == identity
        assert torch.equal(out.view(torch.int32), _frozen_m32(linear, x, torch.float32).view(torch.int32))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_native32_multitile_cancellation_subnormals_and_signed_zeros(dtype):
    x, _, _ = _aligned((65, 256), torch.float16)
    values = torch.tensor([0., -0., 2. ** -24, -(2. ** -24), 65504., -65504., 1., -1.],
                          device="cuda", dtype=torch.float16)
    x.copy_(values.repeat(65, 32))
    weight = values.roll(3).repeat(80, 32) * (2. ** -14)
    linear = GGUFDense(weight.contiguous())
    linear.implementation = "native32"
    out, _, _ = _aligned((65, 80), dtype)
    linear.matmul(x, out=out)
    bits = torch.int16 if dtype == torch.float16 else torch.int32
    assert torch.equal(out.view(bits), _frozen_m32(linear, x, dtype).view(bits))

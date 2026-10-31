"""Optional fused stacks share one retained allocation with their reference views."""

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFConcat, GGUFDense

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 projection stacks")


@pytest.mark.parametrize("counts", [(32, 4), (160, 96, 6, 6), (192, 32, 32, 64, 16)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_projection_stack_reference_views_preserve_bits_and_fused_matches_fp64(counts, dtype):
    torch.manual_seed(1627)
    weights = [torch.randn((n, 256), device="cuda", dtype=torch.float16) * .02 for n in counts]
    originals = [GGUFDense(w.clone()) for w in weights]
    stack = GGUFConcat([GGUFDense(w) for w in weights])
    assert stack.fused is not None
    assert len(stack.device_tensors()) == 1
    retained = stack.device_tensors()[0]
    assert stack.nbytes() == sum(w.numel() * w.element_size() for w in weights)
    assert all(part.weight.untyped_storage().data_ptr() == retained.untyped_storage().data_ptr()
               for part in stack.parts)
    assert all(part.weight.data_ptr() % 512 == 0 for part in stack.parts)
    x = torch.randn((18, 256), device="cuda", dtype=torch.float16)
    expected = torch.cat([part.matmul_reference(x, f32=dtype == torch.float32) for part in originals], dim=1)
    reference = stack.matmul_reference(x, f32=dtype == torch.float32)
    torch.testing.assert_close(reference, expected, rtol=0, atol=0)
    stack.implementation = "fused"
    backing = torch.full((18, stack.n + 19), -999, device="cuda", dtype=dtype)
    actual = stack.matmul(x, out=backing[:, 7:7 + stack.n])
    oracle = x.cpu().double() @ torch.cat(weights).cpu().double().T
    torch.testing.assert_close(actual.cpu().double(), oracle, rtol=.001, atol=3e-5)
    assert torch.all(backing[:, :7] == -999) and torch.all(backing[:, 7 + stack.n:] == -999)
    serial = torch.cat([stack.matmul(x[row:row + 1], f32=dtype == torch.float32) for row in range(18)])
    torch.testing.assert_close(actual, serial, rtol=0, atol=0)


def test_fused_projection_stack_graph_tracks_live_inputs_and_keeps_its_weight_owner():
    torch.manual_seed(1628)
    stack = GGUFConcat([GGUFDense(torch.randn((n, 256), device="cuda", dtype=torch.float16) * .02)
                        for n in [32, 4]])
    stack.implementation = "fused"
    x = torch.randn((3, 256), device="cuda", dtype=torch.float16)
    stack.matmul(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = stack.matmul(x)
    x.mul_(.5)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(result, stack.matmul(x), rtol=0, atol=0)


def test_unaligned_generic_stack_retains_its_reference_owners_and_refuses_fused_selection():
    stack = GGUFConcat([GGUFDense(torch.zeros((n, 33), device="cuda", dtype=torch.float16)) for n in [1, 2]])
    assert stack.fused is None and len(stack.device_tensors()) == 2
    stack.implementation = "fused"
    with pytest.raises(ValueError, match="aligned"):
        stack.matmul(torch.zeros((1, 33), device="cuda", dtype=torch.float16))


@pytest.mark.parametrize("counts", [(32, 4), (160, 96, 6, 6), (192, 32, 32, 64, 16)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows", [0, 1, 3, 18, 32, 33])
def test_shared_native32_pad_and_fused_copy_preserve_original_part_bits(counts, dtype, rows):
    torch.manual_seed(1629)
    stack = GGUFConcat([GGUFDense(torch.randn((n, 256), device="cuda", dtype=torch.float16) * .02)
                        for n in counts])
    stack.implementation = "native32"
    x = torch.randn((rows, 256), device="cuda", dtype=torch.float16)
    expected = torch.cat([part.matmul_reference(x, f32=dtype == torch.float32) for part in stack.parts], dim=1)
    actual = stack.matmul(x, f32=dtype == torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if rows:
        serial = torch.cat([stack.matmul(x[row:row + 1], f32=dtype == torch.float32) for row in range(rows)])
        torch.testing.assert_close(actual, serial, rtol=0, atol=0)
    if rows == 33:
        with pytest.raises(RuntimeError, match="at most32"):
            stack.matmul_native32(x)


def test_native32_stack_replays_dynamic_inputs_with_strided_concat_outputs():
    torch.manual_seed(1630)
    stack = GGUFConcat([GGUFDense(torch.randn((n, 256), device="cuda", dtype=torch.float16) * .02)
                        for n in [32, 4]])
    stack.implementation = "native32"
    x = torch.randn((18, 512), device="cuda", dtype=torch.float16)[:, ::2]
    backing = torch.full((18, stack.n + 19), -997, device="cuda", dtype=torch.float32)
    output = backing[:, 5:5 + stack.n]
    stack.matmul(x, out=output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        stack.matmul(x, out=output)
    x.mul_(.5)
    graph.replay()
    torch.cuda.synchronize()
    expected = torch.cat([part.matmul_reference(x, f32=True) for part in stack.parts], dim=1)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert torch.all(backing[:, :5] == -997) and torch.all(backing[:, 5 + stack.n:] == -997)

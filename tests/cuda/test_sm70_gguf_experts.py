"""Independent packed-expert numeric, route, row-isolation and replay controls."""

import numpy as np
import pytest
import torch

from gguf_fixture import blocks, scalar_reference
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFExperts, GGUFPacked
from tensorfold.families.qwen4_exp.gguf import FORMATS

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 packed experts")


def assert_bits(actual, expected):
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)


def expert_fixture(kind, *, rows=3, dims=256, width=80, offset=16, down_kind=None, up_kind=None):
    down_kind = kind if down_kind is None else down_kind
    up_kind = kind if up_kind is None else up_kind
    experts = 2
    graw = blocks(kind, experts * width, dims, seed=411)
    uraw = blocks(up_kind, experts * width, dims, seed=412)
    block = FORMATS[down_kind][1]
    physical = -(-(offset + width) // block) * block
    draw = blocks(down_kind, experts * dims, physical, seed=413)
    packed = [GGUFPacked(torch.tensor(raw, device="cuda"), typ, raw.shape[0], k, crop)
              for raw, typ, k, crop in [(graw, kind, dims, 0), (uraw, up_kind, dims, 0),
                                        (draw, down_kind, width, offset)]]
    torch.manual_seed(414)
    shared = [GGUFDense(torch.randn(shape, device="cuda", dtype=torch.float16) * .02)
              for shape in [(width, dims), (width, dims), (dims, width)]]
    bank = GGUFExperts(*packed, shared, count=experts, width=width, dims=dims)
    x = torch.randn((rows, dims), device="cuda", dtype=torch.float16) * .1
    picks = torch.tensor([[0, 1, 2], [2, 1, 1], [1, 2, 0]], dtype=torch.int64, device="cuda")[:rows].contiguous()
    cpu = [torch.from_numpy(scalar_reference(raw, typ, columns).astype(np.float16)).double()
           for raw, typ, columns in [(graw, kind, dims), (uraw, up_kind, dims), (draw, down_kind, physical)]]
    gate, up = [value.reshape(experts, width, dims) for value in cpu[:2]]
    down = cpu[2][:, offset:offset + width].reshape(experts, dims, width)
    return bank, x, picks, (gate, up, down)


def fp64_oracle(bank, x, picks, cpu):
    result = torch.empty((*picks.shape, bank.dims), dtype=torch.float64)
    shared = [part.weight.cpu().double() for part in bank.shared]
    for row in range(x.shape[0]):
        xx = x[row].cpu().double()
        for slot, expert in enumerate(picks[row].cpu().tolist()):
            gate, up, down = shared if expert == bank.count - 1 else [part[expert] for part in cpu]
            activation = (torch.nn.functional.silu(gate @ xx) * (up @ xx)).half().double()
            result[row, slot] = down @ activation
    return result


@pytest.mark.parametrize("kind", [0, 1, 30, 8, 7, 12, 13, 20])
@pytest.mark.parametrize("method", ["forward_packed", "forward_splitk4", "forward_blocks"])
def test_packed_experts_match_independent_fp64_and_original_coefficient_rounding(kind, method):
    bank, x, picks, cpu = expert_fixture(kind)
    run = getattr(bank, method)
    actual = run(x, picks)
    if method == "forward_blocks":
        assert_bits(actual, bank.forward_packed(x, picks))
    expected = fp64_oracle(bank, x, picks, cpu)
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=.002, atol=5e-5)
    half = torch.empty_like(actual, dtype=torch.float16)
    assert run(x, picks.int(), half) is half
    torch.testing.assert_close(half, actual.half(), rtol=0, atol=0)
    serial = torch.cat([run(x[row:row + 1], picks[row:row + 1]) for row in range(x.shape[0])])
    torch.testing.assert_close(actual, serial, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["forward_packed", "forward_splitk4", "forward_blocks"])
def test_mixed_packed_experts_capture_dynamic_routes_and_caller_output(method):
    bank, x, picks, _ = expert_fixture(13, down_kind=7)
    run = getattr(bank, method)
    run(x, picks)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run(x, picks)
    picks.copy_(torch.tensor([[2, 0, 1], [0, 2, 0], [1, 0, 2]], device="cuda"))
    x.mul_(.5)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(result, run(x, picks), rtol=0, atol=0)
    bank.implementation = {"forward_packed": "packed", "forward_splitk4": "packed_splitk4",
                           "forward_blocks": "packed_blocks"}[method]
    torch.testing.assert_close(bank.forward(x, picks), result, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["forward_packed", "forward_splitk4", "forward_blocks"])
def test_eighteen_packed_rows_preserve_serial_reductions_with_live_six_slot_geometry(method):
    bank, x, picks, _ = expert_fixture(12, down_kind=7)
    run = getattr(bank, method)
    x, picks = x.repeat(6, 1), picks.repeat(6, 1)
    packed = run(x, picks)
    serial = torch.cat([run(x[row:row + 1], picks[row:row + 1]) for row in range(18)])
    torch.testing.assert_close(packed, serial, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["forward_packed", "forward_splitk4", "forward_blocks"])
def test_packed_expert_bounds_are_checked_before_launch_and_invalid_routes_do_not_read_banks(method):
    bank, x, picks, _ = expert_fixture(12, down_kind=7)
    run = getattr(bank, method)
    with pytest.raises(RuntimeError, match="FP16 CUDA"):
        run(x.float(), picks)
    with pytest.raises(RuntimeError, match="contiguous"):
        run(x, picks[:, ::2])
    with pytest.raises(RuntimeError, match="overlap"):
        run(x, picks[:, :1].contiguous(), x[:, None])
    picks[0] = torch.tensor([-1, 3, 2**40], device="cuda")
    result = run(x, picks)
    assert torch.isnan(result[0]).all()
    assert torch.isfinite(result[1:]).all()


@pytest.fixture(scope="module", params=[(12, 12), (13, 13), (12, 13), (13, 12)])
def multi_block_bank(request):
    kind, up_kind = request.param
    bank, x, picks, cpu = expert_fixture(kind, dims=768, down_kind=7, up_kind=up_kind)
    return bank, x.repeat(6, 1), picks.repeat(6, 1), cpu


@pytest.mark.parametrize("rows", [1, 3, 6, 18])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_header_reuse_retains_exact_v1_math_across_blocks_rows_and_mixed_kinds(multi_block_bank, rows, dtype):
    bank, xx, ids, _ = multi_block_bank
    x, picks = xx[:rows], ids[:rows]
    expected = bank.forward_packed(x, picks).to(dtype)
    output = torch.empty_like(expected)
    assert bank.forward_blocks(x, picks, output) is output
    assert_bits(output, expected)
    serial = torch.cat([bank.forward_blocks(x[row:row + 1], picks[row:row + 1]).to(dtype)
                        for row in range(rows)])
    assert_bits(output, serial)


@pytest.mark.parametrize("kind", [12, 13])
def test_unaligned_header_views_and_cropped_columns_keep_the_original_fallback(kind):
    bank, x, picks, _ = expert_fixture(kind, down_kind=7)
    for part in [bank.gate, bank.up]:
        owner = torch.empty(part.data.numel() + 1, device="cuda", dtype=torch.uint8)
        unaligned = owner[1:].reshape_as(part.data)
        unaligned.copy_(part.data)
        part.data = unaligned
        assert part.data.data_ptr() % 16 == 1
    assert_bits(bank.forward_blocks(x, picks), bank.forward_packed(x, picks))
    dims = 240
    gate = GGUFPacked(bank.gate.data.clone(), kind, bank.gate.n, dims, 16)
    up = GGUFPacked(bank.up.data.clone(), kind, bank.up.n, dims, 16)
    data = torch.cat([bank.down.data[:dims], bank.down.data[256:256 + dims]])
    down = GGUFPacked(data, 7, 2 * dims, bank.width, bank.down.column_offset)
    shared = [GGUFDense(part.weight[:, 16:].contiguous()) for part in bank.shared[:2]]
    shared.append(GGUFDense(bank.shared[2].weight[:dims].contiguous()))
    cropped = GGUFExperts(gate, up, down, shared, count=2, width=bank.width, dims=dims)
    xx = x[:, :dims].contiguous()
    assert_bits(cropped.forward_blocks(xx, picks), cropped.forward_packed(xx, picks))

"""Two projection warps retain the qualified packed-block arithmetic bytes."""

import pytest
import torch

from test_sm70_gguf_experts import assert_bits, expert_fixture, fp64_oracle

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 projection warps")


@pytest.mark.parametrize("kind", [0, 1, 30, 8, 7, 12, 13, 20])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_projection_warps_keep_all_format_bits_signed_zeros_and_fp64_boundary(kind, dtype):
    bank, x, picks, cpu = expert_fixture(kind)
    x[0] = torch.copysign(torch.zeros_like(x[0]), x[0])
    expected = bank.forward_blocks(x, picks).to(dtype)
    output = torch.empty_like(expected)
    assert bank.forward_blocks2w(x, picks.int(), output) is output
    assert_bits(output, expected)
    oracle = fp64_oracle(bank, x, picks, cpu)
    torch.testing.assert_close(output.cpu().double(), oracle, rtol=.002, atol=5e-5)
    serial = torch.cat([bank.forward_blocks2w(x[row:row + 1], picks[row:row + 1]).to(dtype)
                        for row in range(x.shape[0])])
    assert_bits(output, serial)


@pytest.fixture(scope="module", params=[(12, 12), (13, 13), (12, 13), (13, 12)])
def wide_bank(request):
    kind, up_kind = request.param
    bank, x, picks, _ = expert_fixture(kind, dims=768, down_kind=7, up_kind=up_kind)
    return bank, x.repeat(6, 1), picks.repeat(6, 1)


@pytest.mark.parametrize("rows", [1, 3, 6, 18])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_projection_warps_preserve_three_superblocks_mixed_kinds_and_row_placement(wide_bank, rows, dtype):
    bank, xx, ids = wide_bank
    x, picks = xx[:rows], ids[:rows]
    output = torch.empty((*picks.shape, bank.dims), device="cuda", dtype=dtype)
    bank.forward_blocks2w(x, picks, output)
    assert_bits(output, bank.forward_blocks(x, picks).to(dtype))
    serial = torch.cat([bank.forward_blocks2w(x[row:row + 1], picks[row:row + 1]).to(dtype)
                        for row in range(rows)])
    assert_bits(output, serial)


@pytest.mark.parametrize("width", [1, 3, 5])
def test_inactive_output_pairs_still_join_the_barrier_and_invalid_routes_do_not_read_banks(width):
    bank, x, picks, _ = expert_fixture(12, width=width, down_kind=7)
    assert_bits(bank.forward_blocks2w(x, picks), bank.forward_blocks(x, picks))
    picks[0] = torch.tensor([-1, 3, 2**40], device="cuda")
    actual = bank.forward_blocks2w(x, picks)
    assert_bits(actual, bank.forward_blocks(x, picks))
    assert torch.isnan(actual[0]).all() and torch.isfinite(actual[1:]).all()
    with pytest.raises(RuntimeError, match="FP16 CUDA"):
        bank.forward_blocks2w(x.float(), picks)
    with pytest.raises(RuntimeError, match="overlap"):
        bank.forward_blocks2w(x, picks[:, :1].contiguous(), x[:, None])


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_projection_warps_capture_live_routes_with_no_retained_shared_scratch(dtype):
    bank, x, picks, _ = expert_fixture(13, down_kind=7)
    output = torch.empty((*picks.shape, bank.dims), device="cuda", dtype=dtype)
    bank.forward_blocks2w(x, picks, output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bank.forward_blocks2w(x, picks, output)
    x.mul_(.5)
    picks.copy_(picks.roll(1, dims=1))
    graph.replay()
    torch.cuda.synchronize()
    assert_bits(output, bank.forward_blocks(x, picks).to(dtype))
    bank.implementation = "packed_blocks2w"
    assert_bits(bank.forward(x, picks), bank.forward_blocks(x, picks))


@pytest.mark.parametrize("kind", [12, 13])
def test_projection_warps_reuse_the_unaligned_header_fallback(kind):
    bank, x, picks, _ = expert_fixture(kind, down_kind=7)
    owner = torch.empty(bank.gate.data.numel() + 1, device="cuda", dtype=torch.uint8)
    data = owner[1:].reshape_as(bank.gate.data)
    data.copy_(bank.gate.data)
    bank.gate.data = data
    assert_bits(bank.forward_blocks2w(x, picks), bank.forward_blocks(x, picks))

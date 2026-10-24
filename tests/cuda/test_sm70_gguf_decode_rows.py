"""Larger native decode windows retain the existing per-row packed algorithm."""

import gc

import pytest
import torch

from gguf_fixture import blocks
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFExperts, GGUFPacked, DECODE_BYTES
from tensorfold.families.qwen4_exp.gguf import FORMATS

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 packed decode rows")


def bits(actual, expected):
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)


def bank_fixture(kind, *, dims=256, width=80, count=2, up_kind=None, down_kind=None):
    up_kind = kind if up_kind is None else up_kind
    down_kind = kind if down_kind is None else down_kind
    gate = blocks(kind, count * width, dims, seed=5741)
    up = blocks(up_kind, count * width, dims, seed=5742)
    offset = 16
    block = FORMATS[down_kind][1]
    physical = -(-(offset + width) // block) * block
    down = blocks(down_kind, count * dims, physical, seed=5743)
    packed = [
        GGUFPacked(torch.tensor(raw, device="cuda"), typ, raw.shape[0], k, crop)
        for raw, typ, k, crop in [(gate, kind, dims, 0), (up, up_kind, dims, 0), (down, down_kind, width, offset)]
    ]
    torch.manual_seed(5744)
    shared = [
        GGUFDense((torch.randn(shape, device="cuda") * 0.02).half())
        for shape in [(width, dims), (width, dims), (dims, width)]
    ]
    return GGUFExperts(*packed, shared, count=count, width=width, dims=dims)


@pytest.mark.parametrize("kind", [0, 1, 30, 8, 7, 12, 13, 20])
@pytest.mark.parametrize(
    "implementation,method", [("packed", "forward_packed"), ("packed_blocks2w", "forward_blocks2w")]
)
def test_r19_r24_r32_all_formats_and_both_outputs_match_native_serial_rows(kind, implementation, method):
    bank = bank_fixture(kind)
    bank.implementation = implementation
    torch.manual_seed(5745)
    xx = (torch.randn(32, 256, device="cuda") * 0.1).half()
    picks = torch.arange(32 * 11, device="cuda").reshape(32, 11).remainder(3).long().contiguous()
    one = getattr(bank, method)
    for rows in (19, 24, 32):
        for dtype in (torch.float16, torch.float32):
            expected = torch.cat([one(xx[r : r + 1], picks[r : r + 1]).to(dtype) for r in range(rows)])
            out = torch.empty_like(expected)
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            assert bank.forward(xx[:rows], picks[:rows], out) is out
            torch.cuda.synchronize()
            assert torch.cuda.max_memory_allocated() - baseline <= DECODE_BYTES
            bits(out, expected)
    del bank, xx, picks, expected, out
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("rows", [19, 24, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_larger_decode_capture_replays_global_slot_maps_and_preserves_output_sentinels(rows, dtype):
    bank = bank_fixture(12, up_kind=13, down_kind=7)
    bank.implementation = "packed_blocks2w"
    x = (torch.randn(rows, 256, device="cuda") * 0.1).half()
    picks = torch.arange(rows * 11, device="cuda").reshape(rows, 11).remainder(3).long().contiguous()
    storage = torch.full((rows * 11 * 256 + 2,), -97, dtype=dtype, device="cuda")
    out = storage[1:-1].reshape(rows, 11, 256)
    bank.forward(x, picks, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bank.forward(x, picks, out)
    picks.add_(1).remainder_(3)
    x.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    serial = torch.cat([bank.forward(x[r : r + 1], picks[r : r + 1]).to(dtype) for r in range(rows)])
    bits(out, serial)
    assert storage[0] == storage[-1] == -97
    picks[0, :3] = torch.tensor([-1, 3, 2**40], device="cuda")
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isnan(out[0, :3]).all() and torch.isfinite(out[1:]).all()
    assert storage[0] == storage[-1] == -97


@pytest.mark.parametrize("implementation", ["packed", "packed_blocks2w"])
def test_r32_output_shape_alias_and_pair_budget_checks_happen_before_launch(implementation):
    bank = bank_fixture(12, down_kind=7)
    bank.implementation = implementation
    x = torch.randn(32, 256, device="cuda", dtype=torch.float16)
    picks = torch.zeros((32, 11), device="cuda", dtype=torch.int64)
    with pytest.raises(RuntimeError, match="rows,slots,hidden"):
        bank.forward(x, picks, torch.empty((31, 11, 256), device="cuda"))
    with pytest.raises(RuntimeError, match="overlap"):
        bank.forward(x, picks[:, :1].contiguous(), x[:, None])
    excessive = torch.zeros((32, 2048), device="cuda", dtype=torch.int64)
    out = torch.empty((32, 2048, 256), device="cuda")
    with pytest.raises(RuntimeError, match="bounded launch"):
        bank.forward(x, excessive, out)

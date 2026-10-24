"""Reference-prefill batching changes only the independent BMM batch axis."""

import gc

import pytest
import torch

from gguf_fixture import blocks, tiny_model
from tensorfold.families.qwen4_exp.cuda.gguf_prefill import PrefillPlan
from tensorfold.families.qwen4_exp.cuda.gguf_weights import (
    GGUFDense, GGUFExperts, GGUFPacked, load_gguf, weight_byte_plan,
)
from tensorfold.families.qwen4_exp.gguf import FORMATS

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 GGUF prefill")


def assert_bits(actual, expected):
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)


def fixture(kind, width, *, dims=2560, rows=33, slots=11):
    count, offset = 2, 16
    gate = blocks(kind, count * width, dims, seed=5401)
    up = blocks(kind, count * width, dims, seed=5402)
    block = FORMATS[kind][1]
    physical = -(-(offset + width) // block) * block
    down = blocks(kind, count * dims, physical, seed=5403)
    parts = [GGUFPacked(torch.tensor(raw, device="cuda"), kind, raw.shape[0], k, crop)
             for raw, k, crop in [(gate, dims, 0), (up, dims, 0), (down, width, offset)]]
    torch.manual_seed(5407)
    shared = [GGUFDense(torch.randn(shape, device="cuda", dtype=torch.float16) * .02)
              for shape in [(width, dims), (width, dims), (dims, width)]]
    bank = GGUFExperts(*parts, shared, count=count, width=width, dims=dims)
    x = torch.randn(rows, dims, device="cuda", dtype=torch.float16) * .1
    ids = torch.arange(rows * slots, device="cuda").reshape(rows, slots)
    picks = ((ids * 7 + ids // slots) % 3).long().contiguous()
    return bank, x, picks


@pytest.mark.parametrize("kind", [0, 1, 30, 8, 7, 12, 13, 20])
@pytest.mark.parametrize("width", [80, 160])
def test_all_formats_actual_tp8_tp4_geometry_and_partial_pair_tails(kind, width):
    bank, x, picks = fixture(kind, width)
    original = bank.pairs
    for dtype in (torch.float16, torch.float32):
        expected = torch.empty((*picks.shape, bank.dims), dtype=dtype, device=x.device)
        bank.forward_reference(x, picks, expected)
        actual = torch.empty_like(expected)
        for pairs in (32, 64, 128):
            prepared = GGUFExperts(bank.gate, bank.up, bank.down, bank.shared,
                                   count=bank.count - 1, width=width, dims=bank.dims)
            plan = PrefillPlan(width, bank.dims, x.shape[0], picks.shape[1], pairs, 1 << 30)
            prepared.prepare_prefill(plan)
            torch.cuda.synchronize()
            baseline_bytes = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            assert prepared.forward_prefill(x, picks, actual) is actual
            torch.cuda.synchronize()
            assert torch.cuda.max_memory_allocated() - baseline_bytes <= plan.workspace_bytes
            assert_bits(actual, expected)
            assert bank.pairs == original
    del bank, prepared, x, picks, actual, expected
    gc.collect()
    torch.cuda.empty_cache()


def test_short_cached_tails_zero_rows_and_changed_routes_graph_replay():
    bank, x, picks = fixture(13, 80, dims=256, rows=7)
    bank.prepare_prefill(PrefillPlan(80, 256, 7, 11, 32, 1 << 30))
    for rows in (0, 1, 3, 6, 7):
        assert_bits(bank.forward_prefill(x[:rows], picks[:rows]), bank.forward_reference(x[:rows], picks[:rows]))
    actual = bank.forward_prefill(x, picks)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bank.forward_prefill(x, picks, actual)
    picks.add_(1).remainder_(3)
    x.mul_(.5)
    graph.replay()
    torch.cuda.synchronize()
    assert_bits(actual, bank.forward_reference(x, picks))


def test_loader_binds_the_same_admitted_plan_without_retaining_additional_weights(tmp_path):
    path = tiny_model(tmp_path / "tiny.gguf")
    plan = PrefillPlan(80, 256, 7, 2, 128, 128 << 20)
    weights = load_gguf(path, tp=(3, 8), _prefill_plan=plan)
    try:
        forecast = weight_byte_plan(path, (3, 8), prefill_plan=plan)
        assert weights.nbytes() == forecast["resident_bytes"]
        assert forecast["resident_bytes"] == weight_byte_plan(path, (3, 8))["resident_bytes"]
        assert weights.meta["prefill_experts"] == plan.receipt()
        x = torch.randn((7, 256), device="cuda", dtype=torch.float16) * .1
        picks = torch.tensor([[0, 2], [1, 2], [1, 0], [0, 1], [2, 0], [1, 1], [2, 2]], device="cuda")
        for layer in weights.layers:
            bank = layer.moe.experts
            assert bank._prefill_plan is plan
            assert bank.pairs == 16
            assert_bits(bank.forward_prefill(x, picks), bank.forward_reference(x, picks))
    finally:
        weights.meta["ple_tables"].close(lambda: torch.cuda.synchronize())


@pytest.mark.parametrize("kind", [0, 1, 30, 8, 7, 12, 13, 20])
@pytest.mark.parametrize("width", [80, 160])
def test_short_adaptive_calls_preserve_original_bytes_for_every_storage_type(kind, width):
    bank, x, picks = fixture(kind, width, rows=18)
    prepared = GGUFExperts(bank.gate, bank.up, bank.down, bank.shared,
                           count=bank.count - 1, width=width, dims=bank.dims)
    prepared.prepare_prefill(PrefillPlan(width, bank.dims, 18, 11, 128, 1 << 30))
    for rows in (1, 3, 6, 18):
        for dtype in (torch.float16, torch.float32):
            shape = (rows, picks.shape[1], bank.dims)
            expected = torch.empty(shape, dtype=dtype, device=x.device)
            actual = torch.empty_like(expected)
            bank.forward_reference(x[:rows], picks[:rows], expected)
            prepared.forward_prefill(x[:rows], picks[:rows], actual)
            assert_bits(actual, expected)
    del bank, prepared, x, picks, expected, actual
    gc.collect()
    torch.cuda.empty_cache()

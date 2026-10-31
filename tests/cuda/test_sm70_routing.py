"""Exact selection and unchanged weight arithmetic on the actual Volta provider."""
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda import moe
from tensorfold.cuda.sm70 import family_ops
from tensorfold.cuda.sm70.routing_native import indices

pytestmark = pytest.mark.skipif(not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0), reason="requires native SM70")


@pytest.mark.parametrize("rows,count", [(0, 10), (1, 1), (3, 10), (18, 32), (129, 10)])
def test_indices_match_stable_sort_with_ties_nan_infinity_signed_zero(rows, count):
    logits = torch.randn(rows, 520, device="cuda", generator=torch.Generator(device="cuda").manual_seed(17))
    if rows:
        logits[0, :512] = -1.0
        logits[:, 2:8] = 1.0
        logits[:, 10] = float("inf")
        logits[:, 11] = float("-inf")
        logits[:, 12:15] = float("nan")
        logits[:, 20] = 0.0
        logits[:, 21] = -0.0
        logits[:, 30] = torch.finfo(torch.float32).tiny
        logits[:, 31] = -torch.finfo(torch.float32).tiny
    expected = logits[:, :512].argsort(dim=-1, descending=True, stable=True)[:, :count]
    assert torch.equal(indices(logits, count), expected)


def buffer(rows):
    return SimpleNamespace(act=torch.empty((), device="cuda", dtype=torch.float16),
        pick=torch.empty((rows, 11), device="cuda", dtype=torch.int32),
        wts=torch.empty((rows, 11), device="cuda", dtype=torch.float32))


def test_router_weights_and_dynamic_graph_replay_are_exact(monkeypatch):
    logits = torch.randn(3, 513, device="cuda")
    reference, candidate = buffer(3), buffer(3)
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", False)
    moe.select_rows(logits, reference, 10, 512)
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
    moe.select_rows(logits, candidate, 10, 512)
    assert torch.equal(candidate.pick, reference.pick)
    assert torch.equal(candidate.wts, reference.wts)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe.select_rows(logits, candidate, 10, 512)
    for seed in (3, 11, 45):
        logits.copy_(torch.randn(logits.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(seed)))
        monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", False)
        moe.select_rows(logits, reference, 10, 512)
        monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
        graph.replay()
        assert torch.equal(candidate.pick, reference.pick)
        assert torch.equal(candidate.wts, reference.wts)


@pytest.mark.parametrize("count", [1, 10, 32])
def test_full_tie_groups_keep_input_id_order(count):
    rows = 33
    # Many duplicates spanning all four items/thread and all 128 threads.
    # This checks stable radix pairing independently of floating-point ordering.
    generator = torch.Generator(device="cuda").manual_seed(739)
    logits = torch.randint(-3, 4, (rows, 519), device="cuda", generator=generator).float()
    logits[0, :512] = 0.0
    logits[1, :512] = float("nan")
    logits[2, :512] = float("inf")
    logits[3, :512] = float("-inf")
    logits[4, :512:2] = -0.0
    logits[4, 1:512:2] = 0.0
    expected = logits[:, :512].argsort(dim=-1, descending=True, stable=True)[:, :count]
    assert torch.equal(indices(logits, count), expected)


@pytest.mark.parametrize("rows", [1, 2, 3, 8, 18])
def test_decode_router_preserves_each_serial_target_row(rows):
    # Seed 113 crosses an FP16 routing-weight boundary with a batched GEMM.
    # Compare every row against the original serial operator, including storage bits.
    generator = torch.Generator(device="cuda").manual_seed(113)
    x = (torch.randn((rows, 2560), device="cuda", generator=generator) * .7).half()
    weight = (torch.randn((513, 2560), device="cuda", generator=generator) * .03).half()
    expected = torch.cat([moe.router(row[None], weight) for row in x])
    actual = moe.router(x, weight, independent_rows=True)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    expected_routes, actual_routes = buffer(rows), buffer(rows)
    moe.select_rows(expected, expected_routes, 10, 512)
    moe.select_rows(actual, actual_routes, 10, 512)
    assert torch.equal(actual_routes.pick, expected_routes.pick)
    assert torch.equal(actual_routes.wts.view(torch.int32), expected_routes.wts.view(torch.int32))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        moe.router(x, weight, out=actual, independent_rows=True)
    x.mul_(.5)
    expected = torch.cat([moe.router(row[None], weight) for row in x])
    graph.replay()
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_prompt_router_retains_batched_operator():
    x = torch.randn((19, 2560), device="cuda").half()
    weight = torch.randn((513, 2560), device="cuda").half()
    expected = x.float() @ weight.float().T
    assert torch.equal(moe.router(x, weight).view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("rows", [2, 3])
def test_seed113_exposes_old_batched_weight_bin_and_canonical_router_matches_serial(rows):
    # Preserve the original counterexample's rounding order: random FP16
    # storage first, followed by scalar multiplication rounded back to FP16.
    generator = torch.Generator(device="cuda").manual_seed(113)
    x = torch.randn((3, 2560), device="cuda", dtype=torch.float16, generator=generator) * .7
    weight = torch.randn((513, 2560), device="cuda", dtype=torch.float16, generator=generator) * .03
    expected = torch.cat([moe.router(row[None], weight) for row in x[:rows]])
    old_batched = moe.router(x[:rows], weight)
    canonical = moe.router(x[:rows], weight, independent_rows=True)
    serial_routes, old_routes, canonical_routes = [buffer(rows) for _ in range(3)]
    for logits, route in ((expected, serial_routes), (old_batched, old_routes), (canonical, canonical_routes)):
        moe.select_rows(logits, route, 10, 512)
    assert torch.equal(old_routes.pick, serial_routes.pick)
    assert serial_routes.wts[1, 0].item() == .1453857421875
    assert old_routes.wts[1, 0].item() == .145263671875
    assert not torch.equal(old_routes.wts.view(torch.int32), serial_routes.wts.view(torch.int32))
    assert torch.equal(canonical.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(canonical_routes.pick, serial_routes.pick)
    assert torch.equal(canonical_routes.wts.view(torch.int32), serial_routes.wts.view(torch.int32))

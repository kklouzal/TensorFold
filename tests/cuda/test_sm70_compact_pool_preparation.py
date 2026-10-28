"""Actual V100 strict original-pool oracle; no approximate threshold."""

import pytest
import torch
from tensorfold.cuda.sm70.execution_policy import _family_runtime
from test_compact_pool_preparation import make_case, assert_bits
from compact_pool_reference import indexed_pool


@pytest.fixture(autouse=True)
def actual_target():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires actual pinned SM70")
    _family_runtime(torch)


@pytest.mark.parametrize("start", [0, 1, 2, 3])
@pytest.mark.parametrize("count", [1, 3, 7, 16, 32])
@pytest.mark.parametrize("mtp", [False, True])
@torch.no_grad()
def test_actual_pool_preparation_exact_old_Torch_and_RMS16_padding(start, count, mtp):
    cache, out, w, f = make_case(start, count, device="cuda", mtp=mtp)
    reference, expected, _, _ = make_case(start, count, device="cuda", mtp=mtp)
    assert cache._pool_slices_supported(cache.span)
    cache.pool(out, w, f, 1e-6, (11, 11, 10), 0.83)
    indexed_pool(reference, expected, w, f, 1e-6, (11, 11, 10), 0.83)
    assert_bits(out, expected)
    assert torch.equal(cache.status, reference.status)
    torch.cuda.synchronize()


@pytest.mark.parametrize("exceptional", [float("nan"), float("inf"), -float("inf"), -0.0])
@torch.no_grad()
def test_actual_nonfinite_and_signedzero_guard_same_original(exceptional):
    cache, out, w, f = make_case(1, 7, device="cuda")
    reference, expected, _, _ = make_case(1, 7, device="cuda")
    for entry in (cache, reference):
        entry.span.raw[0, 0] = exceptional
    cache.pool(out, w, f, 1e-6, (11, 11, 10), 1.0)
    indexed_pool(reference, expected, w, f, 1e-6, (11, 11, 10), 1.0)
    assert_bits(out, expected)
    assert torch.equal(cache.status, reference.status)
    torch.cuda.synchronize()


@torch.no_grad()
def test_actual_stale_generation_keeps_status_and_unpublished_pool():
    cache, out, w, f = make_case(3, 9, device="cuda")
    reference, expected, _, _ = make_case(3, 9, device="cuda")
    for entry in (cache, reference):
        entry.generations[0] = 1
    cache.pool(out, w, f, 1e-6, (11, 11, 10), 1.0)
    indexed_pool(reference, expected, w, f, 1e-6, (11, 11, 10), 1.0)
    assert_bits(out, expected)
    assert torch.equal(cache.status, reference.status)
    assert cache.status.tolist() == [32, 3]
    assert torch.all(out == 37)

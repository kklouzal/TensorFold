"""Pinned SM70 progressive-pool arithmetic versus the old full-row reduction."""

from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70 import family_ops as ops
from tests.test_sm70_progressive_pool import _legacy_pool


@pytest.mark.parametrize("dims", [64, 128, 256])
@pytest.mark.parametrize("rows", [1, 5, 37])
def test_progressive_gpu_pool_keeps_full_history_rms_mode_and_future_mask(dims, rows, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires the pinned SM70 foundation")
    monkeypatch.setattr(ops, "_fused", lambda value: None)
    capacity, start, ratio = 262148, 262003, 4
    raw = torch.full((capacity, dims), float("nan"), device="cuda", dtype=torch.float16)
    first = start // ratio * ratio
    generator = torch.Generator(device="cuda").manual_seed(701 + dims + rows)
    raw[first:start + rows].copy_(torch.randn((start + rows - first, dims), generator=generator,
                                             device="cuda").half())
    pooled = torch.full(((capacity + 3) // 4, dims), float("nan"), device="cuda", dtype=torch.float16)
    expected = pooled.clone()
    weight = torch.linspace(0.5, 1.5, dims, device="cuda")
    frequency = torch.tensor([1e-4, 3e-5, 9e-7, 7e-8], device="cuda")
    position, delta = torch.tensor([start], device="cuda"), torch.tensor([11], device="cuda", dtype=torch.int32)
    _legacy_pool(raw, expected, position, weight, frequency, 1e-6, ratio, rows, delta=delta, scale=1.25)
    ops.qsa_pool(raw, pooled, position, weight, frequency, 1e-6, NS(ratio=ratio), rows,
                 delta=delta, rope_scale=1.25)
    torch.cuda.synchronize()
    assert torch.equal(pooled.view(torch.uint8), expected.view(torch.uint8))

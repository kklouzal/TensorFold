"""Fused original-byte K160 WMMA candidates against the original B8 BMM."""

import gc
import numpy as np
import pytest
import torch

from test_sm70_gguf_exact import exact_fixture as fixture, bits
from test_sm70_gguf_exact_actual import MODEL, make
from gguf_fixture import scalar_reference
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFPacked

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 original K160 expert arithmetic",
)


@pytest.fixture(autouse=True)
def cleanup():
    yield
    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module", params=[0, 1, 30, 8, 7, 12, 13, 20])
def bank(request):
    return fixture(request.param, width=160), request.param


@pytest.mark.parametrize("rows", [1, 3, 6, 18, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_all_formats_original_b8_outputs_and_serial_bits(bank, rows, dtype):
    (b, _, _), _ = bank
    torch.manual_seed(581 + rows)
    x = (torch.randn(rows, 2560, device="cuda") * 0.1).half()
    x[0].copy_(torch.copysign(torch.zeros_like(x[0]), x[0]))
    ids = torch.arange(rows * 3, device="cuda").reshape(rows, 3).remainder(3)
    original = b.forward_reference(x, ids).to(dtype)
    out = torch.empty_like(original)
    assert b.forward_exact(x, ids.int(), out) is out
    bits(out, original)
    serial = torch.cat(
        [b.forward_exact(x[r : r + 1], ids[r : r + 1]).to(dtype) for r in range(rows)]
    )
    bits(out, serial)


def test_independent_scalar_down_coefficients_and_fp64_error(bank):
    (b, raw, physical), kind = bank
    # A single exactly represented coefficient fixes all native G/U values to
    # x0=.25; the independent oracle covers all160 packed down coefficients.
    sparse = np.zeros((320, 2560), np.float32)
    sparse[:, 0] = 1.0
    packed = torch.tensor(sparse.view(np.uint8), device="cuda")
    g = GGUFPacked(packed, 0, 320, 2560)
    old_gate, old_up = b.gate, b.up
    b.gate = b.up = g
    try:
        x = torch.zeros((1, 2560), dtype=torch.float16, device="cuda")
        x[0, 0] = 0.25
        ids = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
        actual = b.forward_exact(x, ids)[0, 0, :8].cpu().double().numpy()
        activation = np.float16((0.25 / (1.0 + np.exp(-0.25))) * 0.25)
        stored = (
            torch.nn.functional.silu(torch.tensor(0.25, device="cuda")) * 0.25
        ).half()
        assert activation.view(np.uint16) == stored.cpu().numpy().view(np.uint16)
        weight = (
            scalar_reference(raw[2][:8], kind, physical)
            .astype(np.float16)[:, 16:176]
            .astype(np.float64)
        )
        exact = (weight * float(activation)).sum(1)
        bound = (
            np.finfo(np.float32).eps * 160 * (np.abs(weight * float(activation)).sum(1))
        )
        assert np.all(np.abs(actual - exact) <= bound)
    finally:
        b.gate, b.up = old_gate, old_up


def test_capture_dynamic_routes_bounds_and_tails():
    b, _, _ = fixture(13, width=160)
    x = (torch.randn(24, 2560, device="cuda") * 0.1).half()
    ids = torch.arange(24 * 3, device="cuda").reshape(24, 3).remainder(3)
    out = torch.empty((24, 3, 2560), dtype=torch.float16, device="cuda")
    b.forward_exact(x, ids, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        b.forward_exact(x, ids, out)
    x.mul_(0.5)
    ids.copy_(ids.roll(1, 1))
    graph.replay()
    bits(out, b.forward_reference(x, ids).half())
    with pytest.raises(RuntimeError, match="overlap"):
        b.forward_exact(x, ids[:, :1].contiguous(), x[:, None])
    ids[0] = torch.tensor([-1, 3, 2**40], device="cuda")
    graph.replay()
    assert torch.isnan(out[0]).all() and torch.isfinite(out[1:]).all()


@pytest.fixture(scope="module", params=[0, 2])
def actual(request):
    if not MODEL.is_file():
        pytest.skip("optional actual Flash Next GGUF")
    bank, metadata = make(request.param, 4)
    return bank, metadata


@pytest.mark.parametrize("rows", [1, 3, 6, 18, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_actual_q4_q5_complete_original_b8_slot_maps(actual, rows, dtype):
    b, _ = actual
    torch.manual_seed(812 + rows)
    x = (torch.randn(rows, 2560, device="cuda") * 0.1).half()
    ids = (torch.arange(rows * 11, device="cuda").reshape(rows, 11) + rows).remainder(
        10
    )
    ids[:, -1] = 10
    original = b.forward_reference(x, ids).to(dtype)
    out = torch.empty_like(original)
    b.forward_exact(x, ids, out)
    bits(out, original)

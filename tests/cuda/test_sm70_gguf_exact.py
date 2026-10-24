"""Packed exact provider: original BMM bytes, independent coefficients and State-safe captures."""

import numpy as np
import pytest
import torch

from gguf_fixture import blocks, scalar_reference
from tensorfold.families.qwen4_exp.cuda.gguf_weights import (
    GGUFPacked,
    GGUFDense,
    GGUFExperts,
)
from tensorfold.families.qwen4_exp.gguf import FORMATS

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 original expert arithmetic",
)


def bits(a, b):
    torch.testing.assert_close(
        a.contiguous().view(torch.uint8),
        b.contiguous().view(torch.uint8),
        rtol=0,
        atol=0,
    )


def exact_fixture(kind, width=80, dims=2560, offset=16, experts=2):
    block = FORMATS[kind][1]
    physical = -(-(offset + width) // block) * block
    raw = [
        blocks(kind, experts * width, dims, seed=821),
        blocks(kind, experts * width, dims, seed=822),
        blocks(kind, experts * dims, physical, seed=823),
    ]
    packed = [
        GGUFPacked(torch.tensor(v, device="cuda"), kind, v.shape[0], k, crop)
        for v, k, crop in zip(raw, [dims, dims, width], [0, 0, offset])
    ]
    torch.manual_seed(824)
    shared = [
        GGUFDense(torch.randn(shape, device="cuda", dtype=torch.float16) * 0.02)
        for shape in [(width, dims), (width, dims), (dims, width)]
    ]
    bank = GGUFExperts(*packed, shared, count=experts, width=width, dims=dims)
    bank.packed_rows = 32
    return bank, raw, physical


@pytest.fixture(scope="module", params=[0, 1, 30, 8, 7, 12, 13, 20])
def bank(request):
    return exact_fixture(request.param), request.param


@pytest.mark.parametrize("rows", [1, 3, 18, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_all_formats_recover_original_routed_and_shared_bytes(bank, rows, dtype):
    (b, _, _), _kind = bank
    torch.manual_seed(1851 + rows)
    x = (torch.randn(rows, b.dims, device="cuda") * 0.1).half()
    ids = torch.arange(rows * 3, device="cuda").reshape(rows, 3).remainder(3)
    ids = ids.roll(1, 1).contiguous()
    expected = b.forward_reference(x, ids).to(dtype)
    out = torch.empty_like(expected)
    assert b.forward_exact(x, ids, out) is out
    bits(out, expected)
    serial = torch.cat(
        [b.forward_exact(x[r : r + 1], ids[r : r + 1]).to(dtype) for r in range(rows)]
    )
    bits(out, serial)


def test_independent_scalar_half_coefficients_and_fp32_contiguous_chain(bank):
    (b, raw, physical), kind = bank
    torch.manual_seed(715)
    act = (torch.randn(1, 80, device="cuda") * 0.1).half()
    # Independently establish the pinned original BMM's coefficient/storage and
    # reduction contract. The end-to-end cases above link the fused packed
    # operator to that exact BMM without a debug-only production entrypoint.
    decoded = torch.zeros((16, 2560, 80), dtype=torch.float16, device="cuda")
    decoded[0, :8].copy_(b.down.decode(torch.arange(8, device="cuda")))
    padded_act = torch.zeros((16, 80), dtype=torch.float16, device="cuda")
    padded_act[:1].copy_(act)
    actual = (
        torch.bmm(
            padded_act[:, None], decoded.transpose(1, 2), out_dtype=torch.float32
        )[0, 0, :8]
        .cpu()
        .numpy()
    )
    weight = (
        scalar_reference(raw[2][:8], kind, physical)
        .astype(np.float16)[:, 16:96]
        .astype(np.float32)
    )
    x = act[0].cpu().numpy().astype(np.float32)
    partials = np.zeros((8, 4), np.float32)
    for i in range(20):
        for part in range(4):
            partials[:, part] = (
                partials[:, part] + weight[:, part * 20 + i] * x[part * 20 + i]
            )
    expected = (partials[:, 0] + partials[:, 2]) + (partials[:, 1] + partials[:, 3])
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
    exact = (weight.astype(np.float64) * x.astype(np.float64)).sum(1)
    bound = (
        np.finfo(np.float32).eps
        * 24
        * (np.abs(weight.astype(np.float64) * x.astype(np.float64)).sum(1))
    )
    assert np.all(np.abs(actual.astype(np.float64) - exact) <= bound)


def test_original_shared_is_required_and_fallback_boundaries_are_exact():
    b, _, _ = exact_fixture(12)
    for rows in (0, 1, 33):
        x = torch.randn(rows, b.dims, device="cuda", dtype=torch.float16) * 0.1
        ids = torch.full((rows, 3), 2, device="cuda", dtype=torch.int32)
        bits(b.forward_exact(x, ids), b.forward_reference(x, ids))
    b, _, _ = exact_fixture(12, width=320)
    x = torch.randn(3, b.dims, device="cuda", dtype=torch.float16) * 0.1
    ids = torch.tensor([[0, 1, 2], [2, 0, 1], [1, 2, 0]], device="cuda")
    bits(b.forward_exact(x, ids), b.forward_reference(x, ids))


def test_recovered_graph_has_dynamic_routes_and_no_alias_or_invalid_reads():
    b, _, _ = exact_fixture(13)
    x = torch.randn(3, b.dims, device="cuda", dtype=torch.float16) * 0.1
    ids = torch.tensor([[0, 1, 2], [2, 0, 1], [1, 2, 0]], device="cuda")
    out = torch.empty((3, 3, b.dims), device="cuda", dtype=torch.float16)
    b.forward_exact(x, ids, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        b.forward_exact(x, ids, out)
    ids.copy_(ids.roll(1, 1))
    x.mul_(0.5)
    graph.replay()
    bits(out, b.forward_reference(x, ids).half())
    with pytest.raises(RuntimeError, match="overlap"):
        b.forward_exact(x, ids[:, :1].contiguous(), x[:, None])
    ids[0] = torch.tensor([-1, 3, 2**40], device="cuda")
    result = b.forward_exact(x, ids)
    assert torch.isnan(result[0]).all()
    assert torch.isfinite(result[1:]).all()

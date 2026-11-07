# ruff: noqa: E402 - optional Torch is checked before importing its CUDA orchestration.
"""Distinct fused/separate MoE arithmetic and selected dispatch; CPU controls only."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.cuda.sm70 import family_native, family_ops
from tensorfold.families.qwen4_exp.cuda import glue


def partial_case(dtype, rows=3, width=7):
    """Stored-input fixtures include cancellation, non-FMA and the shared slot."""
    count, slots = rows + 2, 11
    values = ((np.arange(count * slots * width).reshape(count, slots, width) % 31) - 15).astype(np.float32) / 64
    weights = ((np.arange(count * slots).reshape(count, slots) % 9) - 4).astype(np.float32) / 8
    if rows:
        values[0] = 0
        weights[0] = 1
        large = 2**12 if dtype == torch.float16 else 2**24
        values[0, 0], values[0, 1], values[0, 2], values[0, 3], values[0, -1] = large, 1, -large, 2, .5
    if rows > 1:
        values[1] = 0
        weights[1] = 1
        values[1, 0], values[1, 1] = -1, np.float32(1 + 2**-23)
        weights[1, 1] = np.float32(1 - 2**-23)
    values[rows:] = np.nan
    weights[rows:] = np.nan
    return values.astype(np.float16 if dtype == torch.float16 else np.float32), weights


def ordered_partial(values, weights, rows):
    """Independent scalar FP32 multiplication then addition, in slot order."""
    result = np.empty((rows, values.shape[2]), dtype=np.float32)
    for row in range(rows):
        for col in range(values.shape[2]):
            total = np.float32(0)
            for slot in range(values.shape[1]):
                product = np.float32(np.float32(values[row, slot, col]) * np.float32(weights[row, slot]))
                total = np.float32(total + product)
            result[row, col] = total
    return result


def fused_partial(values, weights, rows):
    """Independent C99 fmaf oracle: one FP32 rounding per slot, not FP64 emulation."""
    import ctypes
    from ctypes.util import find_library

    name = find_library("m")
    if name is None:
        raise RuntimeError("the fused oracle requires the target's C99 math library")
    library = ctypes.CDLL(name)
    fmaf = library.fmaf
    fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
    fmaf.restype = ctypes.c_float
    result = np.empty((rows, values.shape[2]), dtype=np.float32)
    for row in range(rows):
        for col in range(values.shape[2]):
            total = np.float32(0)
            for slot in range(values.shape[1]):
                total = np.float32(fmaf(float(values[row, slot, col]), float(weights[row, slot]), float(total)))
            result[row, col] = total
    return result


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_independent_oracles_distinguish_fusion_from_slot_reordering(dtype):
    values, weights = partial_case(dtype)
    fused = fused_partial(values, weights, 3)
    separate = ordered_partial(values, weights, 3)
    if dtype == torch.float32:
        assert np.all(separate[1] == 0)
        assert np.all(fused[1] == np.float32(-2**-46))
        assert not np.array_equal(fused.view(np.uint32), separate.view(np.uint32))
        order = [0, 2, 1, *range(3, 11)]
        assert np.all(fused[0] == np.float32(2.5))
        assert np.all(fused_partial(values[:, order], weights[:, order], 1)[0] == np.float32(3.5))
    else:
        np.testing.assert_array_equal(fused.view(np.uint32), separate.view(np.uint32))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows", [0, 1, 3])
@pytest.mark.parametrize("enabled", [False, True])
def test_cpu_family_partial_preserves_fp32_slot_order_and_live_row_bounds(monkeypatch, dtype, rows, enabled):
    values, weights = partial_case(dtype, rows)
    expected = ordered_partial(values, weights, rows)
    out = torch.full((rows + 2, values.shape[2]), 913.25, dtype=torch.float32)
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", enabled)
    monkeypatch.setattr(family_native, "extension", lambda: pytest.fail("CPU reference reached native compilation"))
    family_ops.moe_partial(torch.from_numpy(values), torch.from_numpy(weights), out, rows)
    np.testing.assert_array_equal(out[:rows].numpy().view(np.uint32), expected.view(np.uint32))
    assert out.dtype == torch.float32 and torch.equal(out[rows:], torch.full_like(out[rows:], 913.25))
    if rows and dtype == torch.float32:
        assert np.all(expected[0] == np.float32(2.5))
        order = [0, 2, 1, *range(3, 11)]
        assert np.all(ordered_partial(values[:, order], weights[:, order], 1)[0] == np.float32(3.5))
    if rows > 1 and dtype == torch.float32:
        assert np.all(expected[1] == 0)  # Fusing multiplication/addition would retain -2**-46.


@pytest.mark.parametrize("dtype,cuda,capability,expected", [
    (torch.float16, False, None, "family"),
    (torch.float16, True, (7, 0), "family"),
    (torch.float16, True, (8, 0), "family"),
    (torch.float32, True, (7, 0), "triton"),
    (torch.float32, True, (8, 0), "triton"),
    (torch.float32, True, (9, 0), "triton"),
    (torch.float32, True, (12, 1), "triton"),
    (torch.bfloat16, True, (8, 0), "triton"),
    (torch.bfloat16, True, (12, 1), "triton"),
])
def test_glue_retains_selected_float32_kernel_and_existing_half_routes(monkeypatch, dtype, cuda, capability, expected):
    y = NS(dtype=dtype, is_cuda=cuda, device=torch.device("cuda:2" if cuda else "cpu"))
    wts, out = NS(shape=(4, 11)), NS(shape=(4, 2560))
    queries, calls = [], []
    def query(device):
        queries.append(device)
        return capability
    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append(("triton", grid, args, kwargs))
    monkeypatch.setattr(torch.cuda, "get_device_capability", query)
    monkeypatch.setattr(glue._half_ops, "moe_partial", lambda *args: calls.append(("family", args)))
    monkeypatch.setattr(glue, "_moe_partial", Kernel())
    glue.moe_partial(y, wts, out, 3)
    if expected == "family":
        assert calls == [("family", (y, wts, out, 3))]
    else:
        assert calls == [("triton", (3, 10), (y, wts, out),
                          {"D": 2560, "TOPK": 10, "SLOTS": 11, "BLOCK": 256, "num_warps": 2})]
    assert queries == []  # Original dispatch adds no per-call capability query.


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("enabled", [False, True])
def test_existing_family_partial_selector_retains_both_supported_storage_types(monkeypatch, dtype, enabled):
    y, wts = NS(dtype=dtype), object()
    out = NS(dtype=torch.float32, is_cuda=True)
    calls = []
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", enabled)
    monkeypatch.setattr(family_native, "extension", lambda: NS(partial=lambda *args: calls.append(("native", args))))
    monkeypatch.setitem(family_ops._reference, "moe_partial", lambda *args: calls.append(("reference", args)))
    family_ops.moe_partial(y, wts, out, 3)
    assert calls == [("native" if enabled else "reference", (y, wts, out, 3))]

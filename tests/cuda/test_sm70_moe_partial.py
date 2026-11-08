"""Actual SM70 original F32 fused dispatch and Half family numerical controls."""
import numpy as np
import pytest
import torch

from tensorfold.cuda.sm70 import family_ops
from tensorfold.families.qwen4_exp.cuda import glue
from test_flashnext_moe_partial import fused_partial, ordered_partial, partial_case

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="requires an actual SM70 device")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows", [1, 7])
def test_sm70_selected_partial_obeys_its_fused_or_separate_policy_and_row_bounds(monkeypatch, dtype, rows):
    # The runner owns writable TRITON_CACHE_DIR/CPP caches before import; do not
    # patch compiler flags, force another architecture or alter native selection.
    values, weights = partial_case(dtype, rows, width=2560)
    oracle = fused_partial if dtype == torch.float32 else ordered_partial
    expected = oracle(values, weights, rows)
    y, wts = torch.from_numpy(values).cuda(), torch.from_numpy(weights).cuda()
    out = torch.full((rows + 2, 2560), 913.25, device="cuda", dtype=torch.float32)
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", False)
    glue.moe_partial(y, wts, out, rows)
    result = out.cpu().numpy()
    np.testing.assert_array_equal(result[:rows].view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(result[rows:], np.full((2, 2560), 913.25, dtype=np.float32))
    assert out.dtype == torch.float32


@pytest.mark.parametrize("rows", [0, 1, 7, 32, 256])
def test_private_native_half_partial_matches_separate_oracle_without_touching_guards(monkeypatch, rows):
    from test_flashnext_moe_partial import half_partial_adverse_case

    values, weights = half_partial_adverse_case(rows)
    expected = ordered_partial(values, weights, rows)
    y, wts = torch.from_numpy(values).cuda(), torch.from_numpy(weights).cuda()
    out = torch.full((rows + 2, 2560), 913.25, device="cuda", dtype=torch.float32)
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", False)
    monkeypatch.setattr(family_ops, "_PRIVATE_HALF_PARTIAL", "native-half-v1")
    glue.moe_partial(y, wts, out, rows)
    result = out.cpu().numpy()
    np.testing.assert_array_equal(result[:rows].view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(result[rows:].view(np.uint32),
                                  np.full((2, 2560), 913.25, dtype=np.float32).view(np.uint32))
    assert not family_ops._NATIVE_FAMILY


@pytest.mark.parametrize("invalid", ["strided_values", "strided_weights", "half_out", "too_many_rows"])
def test_private_half_partial_rejects_invalid_native_operands_before_writing(monkeypatch, invalid):
    y = torch.ones((2, 11, 2560), dtype=torch.float16, device="cuda")
    wts = torch.ones((2, 11), dtype=torch.float32, device="cuda")
    out = torch.full((2, 2560), 913.25, dtype=torch.float32, device="cuda")
    rows = 2
    if invalid == "strided_values":
        y = torch.ones((2, 11, 5120), dtype=torch.float16, device="cuda")[..., ::2]
    elif invalid == "strided_weights":
        wts = torch.ones((2, 22), dtype=torch.float32, device="cuda")[:, ::2]
    elif invalid == "half_out":
        out = out.half()
    else:
        rows = 3
    before = out.clone()
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", False)
    monkeypatch.setattr(family_ops, "_PRIVATE_HALF_PARTIAL", "native-half-v1")
    with pytest.raises(RuntimeError):
        glue.moe_partial(y, wts, out, rows)
    assert torch.equal(out, before)

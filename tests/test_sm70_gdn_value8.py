"""Private value8 resource/selection boundaries; no native execution claim."""
import pytest
import torch
from types import SimpleNamespace

from tensorfold.cuda.sm70 import gdn_value8


@pytest.mark.parametrize("rows,preload", [(0, 0), (1, 1), (5, 5), (32, 0), (120, 0), (65535, 0)])
def test_finite_geometry_and_additive_q_credit(rows, preload):
    facts = gdn_value8.layout(rows)
    assert facts["q_shape"] == (rows, 2, 128)
    assert facts["additional_device_bytes"] == rows * 1024
    assert facts["core_ctas_per_sequence"] == 6 * 16
    assert facts["preload_rows"] == preload


@pytest.mark.parametrize("rows", [True, -1, 65536, 1.0, None])
def test_refuses_invalid_or_hardware_unrepresentable_rows(rows):
    with pytest.raises(ValueError):
        gdn_value8.layout(rows)


def test_workspace_requires_credit_before_any_allocation(monkeypatch):
    monkeypatch.setattr(torch, "empty", lambda *args, **kwargs: pytest.fail("unfunded allocation"))
    for credit in (1023, True, None):
        with pytest.raises(MemoryError):
            gdn_value8.allocate_workspace(1, "cpu", device_budget_bytes=credit)


def test_funded_workspace_has_exact_payload_and_no_state_mirror():
    workspace = gdn_value8.allocate_workspace(5, "cpu", device_budget_bytes=5120)
    assert workspace.capacity == 5 and workspace.q.shape == (5, 2, 128)
    assert workspace.q.dtype == torch.float32 and workspace.q.is_contiguous()
    assert workspace.q.numel() * workspace.q.element_size() == 5120
    assert set(vars(workspace)) == {"q", "capacity"}


def test_lazy_negative_prefix_control_refused_before_native_dispatch():
    operand = torch.zeros(1)
    keep = torch._neg_view(torch.tensor([1], dtype=torch.int32))
    assert keep.is_neg() and keep.is_contiguous()
    scratch = SimpleNamespace(k=operand, v=operand, g=operand, b=operand, keep_rows=keep)
    module = SimpleNamespace(chain=lambda *args: pytest.fail("invalid prefix reached native code"))
    prepared = gdn_value8.Prepared(module, torch.device("cpu"))
    with pytest.raises(TypeError, match="plain same-device"):
        prepared.chain(operand, operand, operand, operand, operand, operand, operand,
                       1e-6, 1, scratch, operand, operand, operand,
                       gdn_value8.Workspace(operand, 1))

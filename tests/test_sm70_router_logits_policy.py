"""Exact router qualification stays explicit, decode-only and on matching storage."""
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch

from tensorfold.cuda import moe  # noqa: E402
from tensorfold.cuda.sm70 import routing_native  # noqa: E402


def operands(monkeypatch):
    device = torch.device("cuda:0")

    def tensor(shape, dtype, address):
        return NS(shape=shape, dtype=dtype, device=device, is_cuda=True, ndim=len(shape),
                  stride=lambda dimension: 2560 if dimension == 0 else 1,
                  is_contiguous=lambda: True, data_ptr=lambda: address)

    x = tensor((3, 2560), torch.float16, 512)
    weight = tensor((513, 2560), torch.float32, 1024)
    output = tensor((3, 513), torch.float32, 1536)
    monkeypatch.setattr(routing_native, "_NATIVE_LOGITS", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 0))
    monkeypatch.setattr(torch._C, "_overlaps", lambda *args: False)
    return x, weight, output


def test_exact_router_policy_is_disabled_by_default_and_does_not_probe_cuda(monkeypatch):
    monkeypatch.setattr(routing_native, "_NATIVE_LOGITS", False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: pytest.fail("disabled policy probes CUDA"))
    assert not routing_native.supports_logits(NS(), NS(), NS())


def test_matching_serial_decode_geometry_selects_without_retained_workspace(monkeypatch):
    assert routing_native.supports_logits(*operands(monkeypatch))


@pytest.mark.parametrize("mismatch", ["zero", "one", "large", "width", "bf16", "cpu", "input_stride",
    "input_row_overlap", "half_weight", "weight_shape", "weight_stride", "weight_alignment", "weight_device",
    "output_shape", "output_type", "output_stride", "output_device", "new_gpu", "alias"])
def test_unqualified_storage_or_device_retains_original_router(monkeypatch, mismatch):
    x, weight, output = operands(monkeypatch)
    if mismatch in ("zero", "one", "large"):
        rows = {"zero": 0, "one": 1, "large": 33}[mismatch]
        x.shape, output.shape = (rows, 2560), (rows, 513)
    elif mismatch == "width":
        x.shape = (3, 2559)
    elif mismatch == "bf16":
        x.dtype = torch.bfloat16
    elif mismatch == "cpu":
        x.is_cuda = False
    elif mismatch in ("input_stride", "input_row_overlap"):
        x.stride = lambda dim: (2559 if mismatch == "input_row_overlap" else 2560) if dim == 0 else (
            2 if mismatch == "input_stride" else 1)
    elif mismatch == "half_weight":
        weight.dtype = torch.float16
    elif mismatch == "weight_shape":
        weight.shape = (512, 2560)
    elif mismatch == "weight_stride":
        weight.is_contiguous = lambda: False
    elif mismatch == "weight_alignment":
        weight.data_ptr = lambda: 1028
    elif mismatch in ("weight_device", "output_device"):
        (weight if mismatch == "weight_device" else output).device = torch.device("cuda:1")
    elif mismatch == "output_shape":
        output.shape = (3, 512)
    elif mismatch == "output_type":
        output.dtype = torch.float16
    elif mismatch == "output_stride":
        output.is_contiguous = lambda: False
    elif mismatch == "new_gpu":
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: (8, 0))
    else:
        monkeypatch.setattr(torch._C, "_overlaps", lambda *args: True)
    assert not routing_native.supports_logits(x, weight, output)


def test_router_phase_gate_passes_original_half_input_to_selected_operator(monkeypatch):
    torch.manual_seed(728)
    x = torch.randn(3, 32).half()
    weight = torch.randn(5, 32)
    output = torch.empty(3, 5)
    calls = []
    monkeypatch.setattr(routing_native, "supports_logits", lambda *args: True)

    def selected(source, coefficients, out):
        assert source is x and coefficients is weight and out is output
        calls.append("decode")
        return out.fill_(7)

    monkeypatch.setattr(routing_native, "logits", selected)
    expected = x.float() @ weight.T
    assert moe.router(x, weight, output, independent_rows=False) is output
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert calls == []
    assert moe.router(x, weight, output, independent_rows=True) is output
    assert calls == ["decode"] and torch.all(output == 7)


def test_enabled_policy_keeps_cpu_and_half_weight_fallback_arithmetic(monkeypatch):
    monkeypatch.setattr(routing_native, "_NATIVE_LOGITS", True)
    monkeypatch.setattr(routing_native, "logits", lambda *args: pytest.fail("CPU routing selected CUDA"))
    torch.manual_seed(729)
    x, weight = torch.randn(3, 32).half(), torch.randn(5, 32).half()
    expected = torch.cat([row[None].float() @ weight.float().T for row in x])
    torch.testing.assert_close(moe.router(x, weight, independent_rows=True), expected, rtol=0, atol=0)

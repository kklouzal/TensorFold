"""Actual load-scoped packing boundary; CPU controls do not qualify CUDA policy."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from test_nvfp4_loader_phase_control import fixture

from tensorfold.cuda.nvfp4 import fragments, linear
from tensorfold.families.qwen3_5.cuda import exl3_load, nvfp4_load, weights


@pytest.mark.parametrize("selected", [False, True])
def test_weight_dispatch_passes_only_the_selected_native_load_callback(tmp_path, monkeypatch, selected):
    calls = []
    monkeypatch.setattr(exl3_load, "quant_config", lambda directory: None)
    monkeypatch.setattr(nvfp4_load, "quantized", lambda directory: True)
    monkeypatch.setattr(nvfp4_load, "load_nvfp4", lambda *a, **kw: calls.append(kw) or "loaded")
    options = {"_pack_fp8": fragments.fragment_order} if selected else {}
    assert weights.load(tmp_path, eos_token_ids=(7, 9), **options) == "loaded"
    assert calls == [{"eos_token_ids": (7, 9), **options}]


@pytest.mark.parametrize("kind", ["affine", "exl3"])
def test_selected_private_callback_cannot_silently_enter_another_format(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(exl3_load, "quant_config", lambda directory: {} if kind == "exl3" else None)
    monkeypatch.setattr(nvfp4_load, "quantized", lambda directory: False)
    def payload(*args, **kwargs):
        pytest.fail("incompatible explicit packing reached payload loading")
    monkeypatch.setattr(exl3_load, "load_exl3", payload)
    monkeypatch.setattr(weights.Config, "read", payload)
    with pytest.raises(ValueError, match="private FP8 packing"):
        weights.load(tmp_path, _pack_fp8=fragments.fragment_order)


@pytest.mark.parametrize("kind", ["fp8", "fp8channel"])
@pytest.mark.parametrize("selected", [False, True])
def test_actual_native_loader_binds_both_original_checkpoint_fp8_branches(kind, selected):
    load, _, state, modules = fixture()
    headers = modules["tensorfold.cuda.capacity"].headers(Path("opaque"))
    for name in list(headers):
        if "_proj.weight" in name or name == "lm_head.weight":
            headers[name[:-7] + ".weight_scale"] = {"dtype": "F32", "shape": [1]}
    modules["tensorfold.cuda.capacity"].headers = lambda directory: headers
    modules["tensorfold.cuda.nvfp4"].format = SimpleNamespace(
        LinearMetadata=lambda block: SimpleNamespace(validate=lambda *a, **kw: (kind, None)),
        calibration=lambda *a, **kw: 1.0,
    )
    built = []
    modules["tensorfold.cuda.nvfp4.linear"].Fp8Linear = SimpleNamespace(
        from_checkpoint=lambda *a, **kw: built.append(kw) or SimpleNamespace())
    load.__globals__["SUFFIXES"] = ("weight", "weight_scale")
    packing = fragments.fragment_order if selected else None
    with patch.dict("sys.modules", modules), contextlib.redirect_stdout(io.StringIO()):
        result = load(Path("opaque"), _pack_fp8=packing)
    assert len(built) == 8 and all(call["_packing"] is packing for call in built)
    assert all(call.get("token", False) is False and call.get("act") is None for call in built)
    assert state["closes"] == 1 and result.precision == "full"


@pytest.mark.parametrize("channel", [False, True])
def test_checkpoint_constructor_selects_exact_bytes_without_retaining_policy(monkeypatch, channel):
    codes = torch.arange(65 * 128, dtype=torch.int64).to(torch.uint8).reshape(65, 128)
    weight = codes.view(torch.float8_e4m3fn)
    scale = torch.ones(65) if channel else .5
    original = linear._fragment_order
    baseline = linear.Fp8Linear.from_checkpoint(weight, scale, token=channel)
    def forbidden(*args, **kwargs):
        pytest.fail("explicit callback fell through to default packing")
    monkeypatch.setattr(linear, "_fragment_order", forbidden)
    selected = linear.Fp8Linear.from_checkpoint(weight, scale, token=channel, _packing=fragments.fragment_order)
    assert torch.equal(selected.w8, baseline.w8)
    assert selected.scale == baseline.scale and selected.token == baseline.token
    if channel:
        assert torch.equal(selected.cols, baseline.cols)
    assert not any(callable(value) for value in vars(selected).values())
    calls = []
    monkeypatch.setattr(linear, "_fragment_order", lambda *args: calls.append(args) or original(*args))
    linear.Fp8Linear.from_checkpoint(weight, scale, token=channel)
    assert len(calls) == 1


def test_selected_packing_failure_stays_primary_and_never_retries_default(monkeypatch):
    primary = RuntimeError("selected packing failed")
    def failed(*args, **kwargs):
        raise primary
    def forbidden(*args, **kwargs):
        pytest.fail("failed selected packing retried the default")
    monkeypatch.setattr(linear, "_fragment_order", forbidden)
    with pytest.raises(RuntimeError) as caught:
        linear.Fp8Linear.from_checkpoint(torch.zeros((1, 64), dtype=torch.float8_e4m3fn), 1.0, _packing=failed)
    assert caught.value is primary

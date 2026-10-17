"""Actual Fp4Linear admission on CPU tensors; native packing is an isolated dependency."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.cuda.kernels import qmm
from tensorfold.cuda.nvfp4 import checkpoint, format
from tensorfold.cuda.nvfp4.linear import Fp4Linear


def scales(dtype, values):
    raw = torch.tensor(values, dtype=torch.uint8).repeat(-(-256 // len(values)))[:256].reshape(64, 4)
    return raw if dtype == "u8" else raw.view(torch.float8_e4m3fn)


def original_scales(linear):
    return linear.bs.view(1, 1, 64, 4).permute(0, 2, 1, 3).reshape(64, 4)


@pytest.mark.parametrize("dtype", ["u8", "e4m3"])
@pytest.mark.parametrize("bad", [0x7F, 0x80, 0x81, 0xFE, 0xFF])
def test_native_invalid_scale_payload_stops_before_packing_without_repair(dtype, bad, monkeypatch):
    value = scales(dtype, [0, 0x38, 0x7E])
    value.view(torch.uint8)[-1, -1] = bad
    before = value.view(torch.uint8).clone()
    calls = []
    validate = format.validate_native_a4_scales

    def checked(scale, *, name):
        calls.append((scale, name))
        validate(scale, name=name)

    monkeypatch.setattr(format, "validate_native_a4_scales", checked)
    monkeypatch.setattr(checkpoint, "pack4", lambda *args: pytest.fail("invalid scales reached native packing"))
    with pytest.raises(ValueError, match="finite unsigned E4M3"):
        Fp4Linear.from_checkpoint(torch.zeros((64, 32), dtype=torch.uint8), value, 1.0, act=.125)
    assert len(calls) == 1 and calls[0][0] is value
    assert torch.equal(value.view(torch.uint8), before)


@pytest.mark.parametrize("dtype", ["u8", "e4m3"])
def test_native_valid_zero_through_max_scales_are_admitted_once_and_kept_exactly(dtype, monkeypatch):
    value = scales(dtype, list(range(0x7F)))
    before = value.view(torch.uint8).clone()
    calls = []
    validate = format.validate_native_a4_scales

    def checked(scale, *, name):
        calls.append((scale, name))
        validate(scale, name=name)

    def pack(weight, npad):
        calls.append(("pack", npad))
        return torch.zeros((1, 1, 8, 32, 2), dtype=torch.int32)

    monkeypatch.setattr(format, "validate_native_a4_scales", checked)
    monkeypatch.setattr(checkpoint, "pack4", pack)
    linear = Fp4Linear.from_checkpoint(torch.zeros((64, 32), dtype=torch.uint8), value, 1.0, act=.125)
    assert len(calls) == 2 and calls[0][0] is value and calls[1] == ("pack", 64)
    assert torch.equal(value.view(torch.uint8), before)
    assert torch.equal(original_scales(linear), before)


@pytest.mark.parametrize("dtype", ["u8", "e4m3"])
def test_w4a16_does_not_acquire_the_unsigned_native_scale_contract(dtype, monkeypatch):
    value = scales(dtype, list(range(256)))
    before = value.view(torch.uint8).clone()
    monkeypatch.setattr(format, "validate_native_a4_scales", lambda *args, **kwargs:
                        pytest.fail("W4A16 entered native scale admission"))
    monkeypatch.setattr(qmm, "pack", lambda *args: SimpleNamespace(
        weight=torch.zeros((1, 1, 8, 32, 2), dtype=torch.int32)))
    linear = Fp4Linear.from_checkpoint(torch.zeros((64, 32), dtype=torch.uint8), value, 1.0)
    assert torch.equal(value.view(torch.uint8), before)
    assert torch.equal(original_scales(linear), before)


def test_original_ct_multiplier_survives_native_preparation_and_subhead_views(monkeypatch):
    original = float(torch.tensor(0x3FB7F13D, dtype=torch.int32).view(torch.float32))
    act = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(original, dtype=torch.float32))
    roundtrip = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(act, dtype=torch.float32))
    assert original != roundtrip
    monkeypatch.setattr(checkpoint, "pack4", lambda *args: torch.zeros((1, 1, 8, 32, 2), dtype=torch.int32))
    linear = Fp4Linear.from_checkpoint(torch.zeros((64, 32), dtype=torch.uint8), scales("u8", [0x38]),
                                     1.0, act=act, act_inverse=original)
    assert linear.act == act and linear.act_inverse == original
    sliced = linear.tiles(0, 1)
    assert sliced.act == act and sliced.act_inverse == original
    assert sliced.words.untyped_storage().data_ptr() == linear.words.untyped_storage().data_ptr()


@pytest.mark.parametrize("inverse", [0, -1, float("nan"), float("inf"), 1e100, 1e-100])
def test_invalid_explicit_multiplier_fails_before_native_packing(inverse, monkeypatch):
    monkeypatch.setattr(checkpoint, "pack4", lambda *args: pytest.fail("bad activation inverse reached native pack"))
    with pytest.raises(ValueError, match="calibration"):
        Fp4Linear.from_checkpoint(torch.zeros((64, 32), dtype=torch.uint8), scales("u8", [0x38]),
                                 1.0, act=.125, act_inverse=inverse)


def test_legacy_constructor_positions_keep_the_original_activation_and_no_explicit_multiplier():
    linear = Fp4Linear(object(), object(), 1.0, 64, 64, "nvfp4", None, .125)
    assert linear.act == .125 and linear.act_inverse is None


def test_explicit_multiplier_without_native_activation_is_refused_before_packing(monkeypatch):
    monkeypatch.setattr(qmm, "pack", lambda *args: pytest.fail("unused native metadata reached W4A16 pack"))
    with pytest.raises(ValueError, match="requires native activation"):
        Fp4Linear.from_checkpoint(torch.zeros((64, 32), dtype=torch.uint8), scales("u8", [0x38]),
                                 1.0, act_inverse=2.0)

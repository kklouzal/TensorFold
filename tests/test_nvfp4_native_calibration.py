"""Prepared native factors: exact math, scalar mutation and bounded ownership."""

import gc
import struct
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.cuda.nvfp4 import checkpoint as ck
from tensorfold.cuda.nvfp4 import format as fmt
from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8Linear
from tensorfold.cuda.requant import prefix


def bits(value):
    return struct.pack("<f", value)


def projection(prepared, kind="fp4"):
    if kind == "fp4":
        value = Fp4Linear(torch.empty((2, 1, 8, 32, 2), dtype=torch.int32),
                         torch.empty((2, 1, 64, 4), dtype=torch.uint8), prepared.weight_scale, 128, 64,
                         act=prepared.act, act_inverse=prepared.act_inverse)
    else:
        value = Fp8Linear(torch.empty((128 * 64,), dtype=torch.uint8), prepared.weight_scale, 128, 64, 128,
                         act=prepared.act)
    value._native_calibration = prepared
    return value


@pytest.mark.parametrize("act,scale", [(1., .5), (.037, .13), (1e-30, 1e-8),
                                      (float.fromhex("0x1p-127"), 1.5),
                                      (float.fromhex("0x1.fffffep125"), 1.5)])
def test_prepared_factors_keep_exact_existing_fp32_operations(act, scale):
    prepared = fmt.prepare_native_calibration(act, scale, name="projection")
    assert prepared.act is act and prepared.weight_scale is scale
    assert prepared.act_inverse is None
    assert bits(prepared.alpha) == bits(ck.alpha(act, scale))
    assert bits(prepared.inverse) == bits(ck._inv(act))
    with pytest.raises(FrozenInstanceError):
        prepared.alpha = 0.


def test_ct_preparation_keeps_original_multiplier_without_reciprocal_roundtrip():
    original = struct.unpack("<f", struct.pack("<I", 0x3FB7F13D))[0]
    act = 1. / original
    prepared = fmt.prepare_native_calibration(act, .037, name="ct", act_inverse=original)
    assert bits(prepared.act_inverse) == bits(original)
    assert bits(prepared.inverse) == bits(original)
    assert bits(prepared.alpha) == bits(ck.alpha(act, .037))
    assert bits(ck._inv(act)) != bits(original)
    # Preparation retains the factory's old narrowing for explicit Python
    # values that were not already widened from original checkpoint F32.
    wider = float.fromhex("0x1.0000001p0")
    other = fmt.prepare_native_calibration(.5, .25, name="ct", act_inverse=wider)
    assert other.act_inverse == float(torch.tensor(wider, dtype=torch.float32))


def test_snapshot_refuses_mutable_or_nonfloat_factor_fields():
    with pytest.raises(TypeError, match="immutable built-in float"):
        fmt.NativeCalibration(.5, .125, None, torch.tensor(2.), .0625)


@pytest.mark.parametrize("act,scale", [(1e-40, 1.), (1e30, 1e30), (1e-30, 1e-30)])
def test_preparation_retains_native_inverse_and_product_exhaustion_refusal(act, scale):
    with pytest.raises(ValueError, match="native FP32"):
        fmt.prepare_native_calibration(act, scale, name="projection")


@pytest.mark.parametrize("kind", ["fp4", "fp8"])
def test_prepared_factory_consumes_admitted_factors_without_repeating_torch_math(monkeypatch, kind):
    calls = []
    alpha, inverse = ck.alpha, ck._inv
    monkeypatch.setattr(ck, "alpha", lambda *args: calls.append("alpha") or alpha(*args))
    monkeypatch.setattr(ck, "_inv", lambda *args: calls.append("inverse") or inverse(*args))
    prepared = fmt.prepare_native_calibration(.5, .125, name=kind)
    assert calls == ["inverse", "alpha"]
    monkeypatch.setattr(ck, "alpha", lambda *args: pytest.fail("factory/runtime repeated prepared alpha"))
    monkeypatch.setattr(ck, "_inv", lambda *args: pytest.fail("factory/runtime repeated prepared inverse"))
    if kind == "fp4":
        monkeypatch.setattr(ck, "pack4", lambda *args: torch.zeros((1, 1, 8, 32, 2), dtype=torch.int32))
        value = Fp4Linear.from_checkpoint(torch.zeros((64, 32), dtype=torch.uint8),
                                          torch.full((64, 4), 0x38, dtype=torch.uint8), .125,
                                          act=.5, prepared=prepared)
    else:
        value = Fp8Linear.from_checkpoint(torch.zeros((64, 64), dtype=torch.uint8).view(torch.float8_e4m3fn),
                                          .125, act=.5, prepared=prepared)
    assert ck._prepared_calibration(value) is not None
    assert ck._scales(ck.A4 if kind == "fp4" else ck.A8, (), value) == (prepared.alpha, None, None)
    assert ck._input_inverse(value) == prepared.inverse


def test_prepared_ct_factory_does_not_repeat_multiplier_narrowing(monkeypatch):
    prepared = fmt.prepare_native_calibration(.5, .125, name="ct", act_inverse=2.)
    monkeypatch.setattr(ck, "pack4", lambda *args: torch.zeros((1, 1, 8, 32, 2), dtype=torch.int32))
    original = torch.tensor

    def tensor(*args, **kwargs):
        pytest.fail("prepared CT factory repeated a scalar tensor construction")

    monkeypatch.setattr(torch, "tensor", tensor)
    value = Fp4Linear.from_checkpoint(torch.zeros((64, 32), dtype=torch.uint8),
                                      torch.full((64, 4), 0x38, dtype=torch.uint8), .125, act=.5,
                                      act_inverse=prepared.act_inverse, prepared=prepared)
    assert value._native_calibration is prepared
    monkeypatch.setattr(torch, "tensor", original)


@pytest.mark.parametrize("field", ["act", "scale", "act_inverse"])
def test_any_different_float_binding_invalidates_before_factor_use(field, monkeypatch):
    prepared = fmt.prepare_native_calibration(.5, .125, name="ct", act_inverse=2.)
    value = projection(prepared)
    original = getattr(value, field)
    changed = float.fromhex(original.hex())
    assert changed is not original and changed == original
    setattr(value, field, changed)
    calls = []
    alpha = ck.alpha
    monkeypatch.setattr(ck, "alpha", lambda *args: calls.append(args) or alpha(*args))
    assert ck._scales(ck.A4, (), value)[0] == alpha(value.act, value.scale)
    assert value._native_calibration is None
    assert calls == [(value.act, value.scale)]


def test_mutable_scalar_rebinding_and_inplace_change_remain_uncached():
    prepared = fmt.prepare_native_calibration(.5, .125, name="projection")
    value = projection(prepared)
    value.scale = torch.tensor(.25)
    with pytest.warns(UserWarning, match="To copy construct"):
        assert ck._scales(ck.A4, (), value)[0] == .125
    assert value._native_calibration is None
    value.scale.fill_(.5)
    with pytest.warns(UserWarning, match="To copy construct"):
        assert ck._scales(ck.A4, (), value)[0] == .25


@pytest.mark.parametrize("kind", ["fp4", "fp8"])
def test_tiles_share_factors_without_retaining_parent_or_cross_invalidating(kind):
    prepared = fmt.prepare_native_calibration(.5, .125, name=kind)
    parent = projection(prepared, kind)
    tile = prefix(parent, 128)
    reference = weakref.ref(parent)
    assert tile._native_calibration is prepared
    parent.scale = .25
    assert ck._prepared_calibration(parent) is None
    assert ck._prepared_calibration(tile) is prepared
    del parent
    gc.collect()
    assert reference() is None


def test_original_ct_none_versus_value_grouping_key_is_unchanged():
    direct = projection(fmt.prepare_native_calibration(.5, .125, name="modelopt"))
    ct = projection(fmt.prepare_native_calibration(.5, .125, name="ct", act_inverse=2.))
    assert ck._quantization_key(direct) == (.5, None)
    assert ck._quantization_key(ct) == (.5, 2.)
    assert ck._input_inverse(direct) == ck._input_inverse(ct) == 2.


def test_per_token_fp8_keeps_its_existing_route_without_snapshot_lookup(monkeypatch):
    value = Fp8Linear(torch.empty((128 * 64,), dtype=torch.uint8), 1., 64, 64, 128,
                      cols=torch.ones(128), token=True)
    row_scales = torch.ones(2)
    monkeypatch.setattr(ck, "_prepared_calibration", lambda *args: pytest.fail("dynamic FP8 consulted static factors"))
    factor, selected_rows, selected_columns = ck._scales(ck.A8, (None, row_scales), value)
    assert factor == 1. and selected_rows is row_scales and selected_columns is value.cols
    captured = []
    monkeypatch.setattr(ck, "_rowsq", lambda *args, **kwargs: captured.append(kwargs))
    ck._rows_of(ck.A8, torch.ones(2, 64), value)
    assert captured == [{"inverse": None, "token": True}]


def test_quant8_accepts_prepared_inverse_but_standalone_calls_keep_the_oracle(monkeypatch):
    seen = []
    monkeypatch.setattr(ck, "_ext", lambda: SimpleNamespace(quant8=lambda x, inverse, out, tb: seen.append(inverse)))
    x = torch.ones((1, 64), dtype=torch.bfloat16)
    inverse = ck._inv(.037)
    monkeypatch.setattr(ck, "_inv", lambda act: seen.append("oracle") or inverse)
    ck.quant8(x, .037, inverse=inverse)
    ck.quant8(x, .037)
    assert seen == [inverse, "oracle", inverse]


def test_general_helpers_remain_context_sensitive_after_warm_calls():
    ck.alpha(.5, .125)
    ck._inv(.5)
    with torch.device("meta"):
        with pytest.raises(RuntimeError, match="meta"):
            ck.alpha(.5, .125)
        with pytest.raises(RuntimeError, match="meta"):
            ck._inv(.5)


def test_prepared_and_oracle_bits_match_in_controlled_consumer_threads():
    prepared = fmt.prepare_native_calibration(float.fromhex("0x1p-127"), 1.5, name="subnormal")
    value = projection(prepared)

    def consume():
        return (bits(ck._scales(ck.A4, (), value)[0]), bits(ck._input_inverse(value)),
                bits(ck.alpha(value.act, value.scale)), bits(ck._inv(value.act)), str(torch.get_default_device()))

    expected = consume()
    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(lambda _: consume(), range(8))) == [expected] * 8
    assert expected[0] == expected[2] and expected[1] == expected[3]

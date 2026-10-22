"""Admission-only results keep the actual native factor operations and errors."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.cuda.nvfp4 import checkpoint, format


@pytest.mark.parametrize("inverse", [None, 2.000000238418579])
def test_optional_error_preserves_helper_order_and_public_activation(inverse, monkeypatch):
    act, weight = float("0.037"), .13
    original_inverse, original_alpha = checkpoint._inv, checkpoint.alpha
    original_calibration = format._calibration_result
    calls = []

    def reciprocal(value):
        calls.append(("inverse", value))
        return original_inverse(value)

    def product(left, right):
        calls.append(("alpha", left, right))
        return original_alpha(left, right)

    def calibration(value, **options):
        calls.append(("calibration", value, options))
        return original_calibration(value, **options)

    monkeypatch.setattr(checkpoint, "_inv", reciprocal)
    monkeypatch.setattr(checkpoint, "alpha", product)
    monkeypatch.setattr(format, "_calibration_result", calibration)
    expected = [("inverse", act)] if inverse is None else [("calibration", inverse, {})]
    expected.append(("alpha", act, weight))
    assert format._native_calibration_error(act, weight, name="projection", act_inverse=inverse) is None
    assert calls == expected
    calls.clear()
    assert format.native_calibration(act, weight, name="projection", act_inverse=inverse) is act
    assert calls == expected


@pytest.mark.parametrize("inverse", [0., -0., -1., float("nan"), float("inf"), 1e100, 1e-100])
def test_invalid_original_multiplier_keeps_exact_error_before_alpha(inverse, monkeypatch):
    monkeypatch.setattr(checkpoint, "_inv", lambda _value: pytest.fail("reconstructed an explicit multiplier"))
    monkeypatch.setattr(checkpoint, "alpha", lambda *_values: pytest.fail("invalid multiplier reached alpha"))
    message = ("calibration must remain finite and positive in FP32" if inverse in (1e100, 1e-100)
               else "calibration must be finite and positive")
    expected = "projection.input_global_scale: " + message
    assert format._native_calibration_error(.5, .125, name="projection", act_inverse=inverse) == expected
    with pytest.raises(ValueError) as caught:
        format.native_calibration(.5, .125, name="projection", act_inverse=inverse)
    assert type(caught.value) is ValueError and str(caught.value) == expected


@pytest.mark.parametrize("inverse,product", [(float("inf"), 1.), (0., 1.), (1., float("nan")), (1., -0.)])
def test_native_factor_refusal_keeps_both_operations_and_exact_public_error(inverse, product, monkeypatch):
    calls = []

    def reciprocal(value):
        calls.append(("inverse", value))
        return inverse

    def joint(left, right):
        calls.append(("alpha", left, right))
        return product

    monkeypatch.setattr(checkpoint, "_inv", reciprocal)
    monkeypatch.setattr(checkpoint, "alpha", joint)
    expected = "projection: native FP32 inverse and joint calibration must remain finite and positive"
    assert format._native_calibration_error(.5, .125, name="projection") == expected
    assert calls == [("inverse", .5), ("alpha", .5, .125)]
    with pytest.raises(ValueError) as caught:
        format.native_calibration(.5, .125, name="projection")
    assert type(caught.value) is ValueError and str(caught.value) == expected


@pytest.mark.parametrize("stage", ["inverse", "explicit_inverse", "alpha"])
@pytest.mark.parametrize("failure", [MemoryError("factor resource failure"), ValueError("factor helper defect"),
                                     KeyboardInterrupt("factor cancellation")])
def test_unexpected_factor_failure_propagates_the_exact_root(stage, failure, monkeypatch):
    def fail(*_args, **_options):
        raise failure

    if stage == "inverse":
        monkeypatch.setattr(checkpoint, "_inv", fail)
    elif stage == "explicit_inverse":
        monkeypatch.setattr(format, "_calibration_result", fail)
    else:
        # Invalid implicit inverse must still reach alpha before the represented
        # refusal. It cannot hide an actual subsequent helper failure.
        monkeypatch.setattr(checkpoint, "_inv", lambda _value: float("inf"))
        monkeypatch.setattr(checkpoint, "alpha", fail)
    options = {"act_inverse": 2.} if stage == "explicit_inverse" else {}
    with pytest.raises(type(failure)) as caught:
        format.native_calibration(.5, .125, name="projection", **options)
    assert caught.value is failure

"""Actual CPU Torch storage checks for the unchanged CUDA-loader frequency formula."""
import math
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen3_5.cuda.weights import _rotary_frequencies


def config(theta, pairs, active):
    return SimpleNamespace(rope_theta=theta, rope_dims=2 * pairs, layers=1, is_linear=lambda index: not active)


def legacy(cfg):
    half = cfg.rope_dims // 2
    return (cfg.rope_theta ** (-torch.arange(half, dtype=torch.float64) / half)).to(torch.float32)


@pytest.mark.parametrize("theta", [1., 10000., 10000000.])
def test_normal_theta_retains_exact_legacy_bytes_without_finite_tensor_work(theta, monkeypatch):
    cfg = config(theta, 32, True)
    expected = legacy(cfg)
    monkeypatch.setattr(torch, "isfinite", lambda value: pytest.fail("normal metadata added finite tensor work"))
    assert torch.equal(_rotary_frequencies(cfg, "cpu").view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("pairs,active", [(0, True), (1, True), (4, False)])
def test_unused_or_constant_frequency_roles_retain_original_math(pairs, active, monkeypatch):
    cfg = config(1e-100, pairs, active)
    expected = legacy(cfg)
    monkeypatch.setattr(torch, "isfinite", lambda value: pytest.fail("unused/constant role added finite reduction"))
    assert torch.equal(_rotary_frequencies(cfg, "cpu").view(torch.uint8), expected.view(torch.uint8))


def test_actual_torch_cast_is_guarded_on_both_sides_of_fp32_overflow():
    # Predeclared 129 adjacent FP64 theta values surround the FP32 rounding
    # overflow threshold for the final pair. Python/libm is only the case
    # generator: the independent per-value finite predicate inspects actual
    # Torch-produced FP32 scalars rather than assuming libm/Torch pow parity.
    pairs = 4
    threshold = 2.**128 - 2.**103
    center = math.pow(threshold, -pairs / (pairs - 1))
    seen = set()
    for offset in range(-64, 65):
        theta = center
        for _ in range(abs(offset)):
            theta = math.nextafter(theta, 0. if offset < 0 else math.inf)
        cfg = config(theta, pairs, True)
        expected = legacy(cfg)
        finite = all(math.isfinite(float(value)) for value in expected)
        seen.add(finite)
        if finite:
            assert torch.equal(_rotary_frequencies(cfg, "cpu").view(torch.uint8), expected.view(torch.uint8))
        else:
            with pytest.raises(ValueError, match="overflow FP32"):
                _rotary_frequencies(cfg, "cpu")
    assert seen == {False, True}

"""The new component provider must never weaken ordinary CUDA engine admission."""

import pytest

from tensorfold.cuda import build
from tensorfold.cuda.sm70 import SM70Provider
from tensorfold.cuda.sm70.provider import _native

pytestmark = pytest.mark.torch


@pytest.fixture(autouse=True)
def native_cache():
    _native.cache_clear()
    yield
    _native.cache_clear()


def test_sm70_precision_is_explicit_and_immutable():
    from dataclasses import FrozenInstanceError

    provider = SM70Provider()
    assert (provider.activation_dtype, provider.accumulator_dtype, provider.state_dtype) == (
        "float16", "float32", "float32")
    with pytest.raises(FrozenInstanceError):
        provider.state_dtype = "float16"
    with pytest.raises(ValueError, match="precision policy"):
        SM70Provider(activation_dtype="bfloat16")


def test_modern_provider_still_refuses_volta(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "V100")
    assert build.MIN_CAPABILITY == (8, 9)
    with pytest.raises(RuntimeError, match="FP8 MMA"):
        build.arch_flags()
    assert build.arch_flags(need=(7, 0)) == ["-gencode=arch=compute_70,code=sm_70"]


@pytest.mark.parametrize("capability,architectures,cuda,reason", [
    ((8, 9), ["sm_70", "sm_89"], "12.6", "requires a Volta"),
    ((7, 0), ["sm_75", "sm_80"], "12.6", "no sm_70 kernels"),
    ((7, 0), ["sm_70"], "13.0", "CUDA 12.x"),
])
def test_incompatible_foundation_fails_before_compilation(monkeypatch, capability, architectures, cuda, reason):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: architectures)
    monkeypatch.setattr(torch.version, "cuda", cuda)
    with pytest.raises(RuntimeError, match=reason):
        SM70Provider().prepare()

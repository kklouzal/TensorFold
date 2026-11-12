"""Owned compiler-option selection; no GPU or vendor implementation mutation."""
import os
from types import SimpleNamespace
import sys

import pytest

from tensorfold.cuda.sm70 import triton_options
from tensorfold.families.qwen4_exp.cuda import glue


def test_default_compiler_options_are_empty_immutable_and_need_no_provider_probe(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_SM70_TRITON_PROFILE", raising=False)
    monkeypatch.setattr(triton_options.importlib.metadata, "version", lambda name: pytest.fail("unused provider probe"))
    options = triton_options.compiler_options()
    assert dict(options) == {}
    with pytest.raises(TypeError):
        options["ptx_version"] = 83


def test_explicit_profile_selects_supported_isa_without_rewriting_toolkit_version(monkeypatch):
    monkeypatch.setenv("TENSORFOLD_SM70_TRITON_PROFILE", "sm70-triton32-v1")
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__="3.2.0"))
    monkeypatch.setattr(triton_options.importlib.metadata, "version", lambda name: "3.2.0")
    monkeypatch.setenv("CUDA_VERSION", "12.9.2")
    before = dict(os.environ)
    options = triton_options.compiler_options()
    assert dict(options) == {"ptx_version": 83}
    assert dict(os.environ) == before
    with pytest.raises(TypeError):
        options["ptx_version"] = 88


@pytest.mark.parametrize("profile,module,metadata", [("unknown", "3.2.0", "3.2.0"),
    ("sm70-triton32-v1", "3.9.0", "3.2.0"), ("sm70-triton32-v1", "3.2.0", "3.9.0")])
def test_explicit_profile_and_actual_provider_must_agree(monkeypatch, profile, module, metadata):
    monkeypatch.setenv("TENSORFOLD_SM70_TRITON_PROFILE", profile)
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(__version__=module))
    monkeypatch.setattr(triton_options.importlib.metadata, "version", lambda name: metadata)
    with pytest.raises(ValueError):
        triton_options.compiler_options()


def test_live_moe_launch_uses_the_selected_compiler_options_and_original_math_entry(monkeypatch):
    import torch

    calls = []
    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append((grid, args, kwargs))
    monkeypatch.setattr(glue, "_moe_partial", Kernel())
    monkeypatch.setattr(glue, "_MOE_COMPILER_OPTIONS", {"ptx_version": 83})
    y, weights, out = SimpleNamespace(dtype=torch.float32), SimpleNamespace(shape=(7, 11)), SimpleNamespace(shape=(7, 2560))
    glue.moe_partial(y, weights, out, 7)
    assert calls == [((7, 10), (y, weights, out),
                      {"D": 2560, "TOPK": 10, "SLOTS": 11, "BLOCK": 256, "num_warps": 2, "ptx_version": 83})]

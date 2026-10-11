"""Fork CUDA capability admission and Flash Next delegation without GPU or weight loads."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from tensorfold.cuda import build, capacity, precision
from tensorfold.families import qwen4_exp


def _fake_torch(monkeypatch, capability, *, available=True, name="Test CUDA GPU"):
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: available, get_device_capability=lambda *a: capability,
                                 get_device_name=lambda *a: name)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


@pytest.mark.parametrize("capability", [(8, 9), (9, 0), (10, 0), (12, 0), (12, 1)])
def test_flash_next_delegates_admitted_bf16_devices_to_format_specific_engine(tmp_path, monkeypatch, capability):
    """The family wrapper must preserve the engine's declared 8.9+ format paths; this does not run those kernels."""

    _fake_torch(monkeypatch, capability)
    package = ModuleType("tensorfold.cuda.exl3")
    package.__path__ = []
    format_module = ModuleType("tensorfold.cuda.exl3.format")
    format_module.is_exl3 = lambda path: False
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, format_module.__name__, format_module)
    engine_module = ModuleType("tensorfold.families.qwen4_exp.cuda.engine")
    calls, result = [], object()

    def engine(path, **kwargs):
        calls.append((path, kwargs))
        return result

    engine_module.FlashNextEngine = engine
    monkeypatch.setitem(sys.modules, engine_module.__name__, engine_module)
    assert qwen4_exp.cuda_engine(tmp_path, no_drafts=True, kv_dtype="bf16") is result
    assert len(calls) == 1 and calls[0][0] == tmp_path
    assert calls[0][1]["depth"] == 0 and calls[0][1]["kv_dtype"] == "bf16"


@pytest.mark.parametrize("capability", [(7, 5), (8, 0), (8, 6)])
def test_unsupported_gpu_refuses_before_checkpoint_metadata_or_weights(tmp_path, monkeypatch, capability):
    torch = _fake_torch(monkeypatch, capability)

    def no_read(*args, **kwargs):
        pytest.fail("unsupported CUDA admission reached checkpoint metadata or weight estimation")

    monkeypatch.setattr(capacity, "config", no_read)
    monkeypatch.setattr(capacity, "estimate_weights", no_read)
    with pytest.raises(ValueError, match=r"compute capability 8\.9 or newer.*Test CUDA GPU"):
        capacity.admit(tmp_path, None, None, torch, no_read, no_read)


@pytest.mark.parametrize("capability", [(8, 9), (9, 0), (10, 0), (12, 0), (12, 1)])
def test_format_math_and_compilation_flags_preserve_existing_capability_selection(monkeypatch, capability):
    _fake_torch(monkeypatch, capability)
    assert build.MIN_CAPABILITY == (8, 9) and build.CLUSTERS == (9, 0)
    own = precision.own_math(capability)
    assert own == {"nvfp4": capability[0] == 12, "fp8": True}
    suffix = "a" if own["nvfp4"] else ""
    major, minor = capability
    assert build.arch_flags(arch_specific=own["nvfp4"]) == [
        f"-gencode=arch=compute_{major}{minor}{suffix},code=sm_{major}{minor}{suffix}"]


@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 9)])
def test_architecture_specific_build_requires_cluster_generation(monkeypatch, capability):
    _fake_torch(monkeypatch, capability)
    with pytest.raises(RuntimeError, match=r"compute capability 9\.0 or newer.*thread-block clusters"):
        build.arch_flags(arch_specific=True)


@pytest.mark.parametrize("capability", [(9, 0), (10, 0), (12, 0), (12, 1)])
def test_explicit_cluster_floor_preserves_flags_on_supported_generations(monkeypatch, capability):
    _fake_torch(monkeypatch, capability)
    major, minor = capability
    assert build.arch_flags(build.CLUSTERS, arch_specific=True) == [
        f"-gencode=arch=compute_{major}{minor}a,code=sm_{major}{minor}a"]


@pytest.mark.parametrize("raw", ["", "8.0", "8.6", "9.0", "12.0", "invalid", "1.2.3"])
def test_unselected_environment_override_cannot_change_the_supported_envelope(monkeypatch, raw):
    """A fresh import preserves the fork's fixed floor even when an upstream-only variable is inherited."""

    monkeypatch.setenv("TENSORFOLD_MIN_CAPABILITY", raw)
    path = Path(build.__file__)
    spec = importlib.util.spec_from_file_location("tensorfold_build_capability_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _fake_torch(monkeypatch, (8, 9))
    assert module.arch_flags() == ["-gencode=arch=compute_89,code=sm_89"]
    _fake_torch(monkeypatch, (8, 6))
    with pytest.raises(ValueError, match=r"compute capability 8\.9 or newer"):
        module.refuse_old_gpu()


def test_machine_without_cuda_leaves_device_requirement_to_engine(monkeypatch):
    _fake_torch(monkeypatch, (8, 0), available=False)
    build.refuse_old_gpu()

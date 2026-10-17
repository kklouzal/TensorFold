"""The plain-CUDA launcher admits actual Driver API providers, never markers."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "deploy/gb10"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("test_cuda_entrypoint", directory / "cuda_entrypoint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import foundation_guard
    monkeypatch.setattr(foundation_guard, "verify", lambda state: None)  # Driver boundary control; guard has separate real-file tests.
    compat = tmp_path / "cuda/compat"
    compat.mkdir(parents=True)
    real = compat / "libcuda.so.615.71.09"
    real.write_bytes(b"sealed driver fixture")
    (compat / "libcuda.so.1").symlink_to(real.name)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"toolchain": {"header_version": 13040, "cuda_home": str(compat.parent)},
                                 "native_files": [str(real)]}))
    monkeypatch.setenv("TF_FOUNDATION_STATE", str(state))
    monkeypatch.setenv("LD_LIBRARY_PATH", str(compat) + ":/existing/libs")
    monkeypatch.setenv("LD_PRELOAD", "/unrelated/preload")
    monkeypatch.setenv("_CUDA_COMPAT_STATUS", "stale marker")
    return module, compat


def result(arguments, *, library="/host/libcuda.so.615.71.09", version=13040, devices=1):
    return subprocess.CompletedProcess(arguments, 0, json.dumps({"library": library, "driver_api": version,
                                                                "devices": devices}), "")


def test_native_provider_is_freshly_probed_without_fc(launcher, monkeypatch):
    module, compat = launcher
    calls = []

    def run(arguments, **options):
        calls.append((arguments, options))
        return result(arguments)

    monkeypatch.setattr(module.subprocess, "run", run)
    env = module.prepare_environment()
    assert json.loads(env["TF_CUDA_DRIVER_PROOF"])["mode"] == "native"
    assert env["LD_LIBRARY_PATH"] == "/existing/libs"
    assert "_CUDA_COMPAT_STATUS" not in env
    assert env["LD_PRELOAD"] == "/unrelated/preload"
    args, options = calls[0]
    assert args[-4:] == ["--library", "libcuda.so.1", "--required-api", "13040"]
    assert options["env"]["LD_PRELOAD"] == ""
    assert str(compat) not in options["env"]["LD_LIBRARY_PATH"]
    assert options["timeout"] == 35


def test_native_refusal_selects_only_sealed_compat_provider(launcher, monkeypatch):
    module, compat = launcher
    calls = []

    def run(arguments, **options):
        calls.append((arguments, options))
        if len(calls) == 1:
            return subprocess.CompletedProcess(arguments, 1, "", "driver API 13000 below selected SDK")
        return result(arguments, library=str((compat / "libcuda.so.1").resolve()))

    monkeypatch.setattr(module.subprocess, "run", run)
    env = module.prepare_environment()
    assert json.loads(env["TF_CUDA_DRIVER_PROOF"])["mode"] == "forward-compatible"
    assert env["LD_LIBRARY_PATH"] == str(compat) + ":/existing/libs"
    assert calls[1][0][-3] == str(compat / "libcuda.so.1")
    assert calls[1][1]["env"]["LD_PRELOAD"] == ""


@pytest.mark.parametrize("bad", ["wrong-library", "unsupported", "timeout"])
def test_compatibility_failure_never_inherits_a_marker(launcher, monkeypatch, bad):
    module, _ = launcher
    count = 0

    def run(arguments, **_):
        nonlocal count
        count += 1
        if count == 1:
            return subprocess.CompletedProcess(arguments, 1, "", "native refusal")
        if bad == "timeout":
            raise subprocess.TimeoutExpired(arguments, 35)
        if bad == "unsupported":
            return subprocess.CompletedProcess(arguments, 1, "", "cuInit CUDA error 804")
        return result(arguments, library="/wrong/provider.so")

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(RuntimeError, match=r"native CUDA probe.*compatibility probe"):
        module.prepare_environment()


def test_missing_compatibility_provider_preserves_native_refusal(launcher, monkeypatch):
    module, compat = launcher
    (compat / "libcuda.so.1").unlink()
    monkeypatch.setattr(module.subprocess, "run", lambda args, **_: subprocess.CompletedProcess(args, 1, "", "native refused"))
    with pytest.raises(RuntimeError, match="sealed compatibility provider is missing") as caught:
        module.prepare_environment()
    assert "native refused" in str(caught.value.__cause__)


@pytest.mark.parametrize("version,devices", [(True, 1), (13040, True), (13000, 1), (13040, 0)])
def test_invalid_probe_results_refuse(launcher, monkeypatch, version, devices):
    module, _ = launcher
    monkeypatch.setattr(module.subprocess, "run", lambda args, **_: result(args, version=version, devices=devices))
    with pytest.raises(RuntimeError):
        module.prepare_environment()


def test_missing_sealed_state_refuses(launcher, monkeypatch):
    module, _ = launcher
    monkeypatch.delenv("TF_FOUNDATION_STATE")
    with pytest.raises(RuntimeError, match="sealed foundation state"):
        module.prepare_environment()


def test_vendor_entrypoint_precedes_final_provider_proof(launcher, monkeypatch):
    module, _ = launcher
    env = {"TF_FOUNDATION_STATE": "/sealed/state", "LD_LIBRARY_PATH": "/selected/compat"}
    monkeypatch.setattr(module, "prepare_environment", env.copy)
    monkeypatch.setattr(module.subprocess, "run", lambda args, **_: subprocess.CompletedProcess(args, 0, '{"CC":"/selected/cc"}', ""))
    monkeypatch.setattr(module.sys, "argv", ["launcher", "python3", "serve.py", "--help"])
    calls = []
    monkeypatch.setattr(module.os, "execve", lambda *args: calls.append(args))
    module.main()
    executable, arguments, environment = calls[0]
    assert executable == module.NVIDIA_ENTRYPOINT
    assert arguments[:2] == [module.NVIDIA_ENTRYPOINT, sys.executable]
    assert arguments[3:] == ["checked-run", "--", "python3", "serve.py", "--help"]
    assert environment["LD_LIBRARY_PATH"] == "/selected/compat"
    assert environment["CC"] == "/selected/cc"


def test_unverified_foundation_never_reaches_driver_loader(launcher, monkeypatch):
    module, _ = launcher
    import foundation_guard

    def refused(state):
        raise ValueError("foundation state identity changed")

    monkeypatch.setattr(foundation_guard, "verify", refused)
    monkeypatch.setattr(module.subprocess, "run", lambda *_, **__: pytest.fail("unverified driver load"))
    with pytest.raises(ValueError, match="state identity changed"):
        module.prepare_environment()


def test_implicit_fc_mapping_is_reprobed_with_explicit_sealed_provider(launcher, monkeypatch):
    module, compat = launcher
    calls = []

    def run(arguments, **_):
        calls.append(arguments)
        return result(arguments, library=str((compat / "libcuda.so.1").resolve()))

    monkeypatch.setattr(module.subprocess, "run", run)
    env = module.prepare_environment()
    assert len(calls) == 2
    assert calls[1][-3] == str(compat / "libcuda.so.1")
    assert json.loads(env["TF_CUDA_DRIVER_PROOF"])["mode"] == "forward-compatible"

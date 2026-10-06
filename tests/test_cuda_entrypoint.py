"""The deployment launcher validates FC instead of trusting NGC's cached marker."""

import importlib.util
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "deploy" / "gb10" / "cuda_entrypoint.py"
    spec = importlib.util.spec_from_file_location("test_cuda_entrypoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    driver = tmp_path / "driver-version"
    driver.write_text("NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  580.178.04  Release Build\n")
    libraries = tmp_path / "lib.real"
    libraries.mkdir()
    monkeypatch.setattr(module, "DRIVER_VERSION_FILE", driver)
    monkeypatch.setattr(module, "COMPAT_LIBRARY", libraries)
    monkeypatch.setenv("CUDA_DRIVER_VERSION", "615.71.09")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/libs")
    monkeypatch.setenv("LD_PRELOAD", "/unrelated/preload")
    monkeypatch.setenv("_CUDA_COMPAT_STATUS", "CUDA Driver OK")
    return module


def test_fresh_probe_and_process_environment(launcher, monkeypatch):
    calls = []

    def probe(arguments, **options):
        calls.append((arguments, options))
        return subprocess.CompletedProcess(arguments, 0, "CUDA Driver OK\n", "")

    monkeypatch.setattr(launcher.subprocess, "run", probe)
    environment = launcher.prepare_environment()
    assert environment["_CUDA_COMPAT_STATUS"] == "CUDA Driver OK"
    assert environment["LD_LIBRARY_PATH"] == str(launcher.COMPAT_LIBRARY) + ":/existing/libs"
    assert environment["LD_PRELOAD"] == "/unrelated/preload"
    arguments, options = calls[0]
    assert arguments == [launcher.COMPAT_PROBE]
    assert options["env"]["LD_LIBRARY_PATH"] == str(launcher.COMPAT_LIBRARY)
    assert options["env"]["LD_PRELOAD"] == ""
    assert "_CUDA_COMPAT_STATUS" not in options["env"]
    assert options["timeout"] == 35


@pytest.mark.parametrize("code,status", [(1, "CUDA Driver OK"), (0, "CUDA Driver UNAVAILABLE"), (0, "")])
def test_failed_probe_cannot_inherit_cached_success(launcher, monkeypatch, code, status):
    monkeypatch.setattr(launcher.subprocess, "run",
                        lambda arguments, **_: subprocess.CompletedProcess(arguments, code, status, "probe failed"))
    with pytest.raises(RuntimeError, match="forward compatibility probe failed"):
        launcher.prepare_environment()


def test_native_driver_needs_no_forward_compatibility(launcher, monkeypatch):
    launcher.DRIVER_VERSION_FILE.write_text("NVRM version: NVIDIA UNIX Open Kernel Module for aarch64  615.71.09\n")
    monkeypatch.setattr(launcher.subprocess, "run", lambda *_, **__: pytest.fail("unexpected FC probe"))
    environment = launcher.prepare_environment()
    assert "_CUDA_COMPAT_STATUS" not in environment
    assert environment["LD_LIBRARY_PATH"] == "/existing/libs"


def test_unknown_driver_is_refused(launcher):
    launcher.DRIVER_VERSION_FILE.write_text("unknown driver")
    with pytest.raises(RuntimeError, match="cannot determine"):
        launcher.prepare_environment()


def test_missing_compatibility_libraries_are_refused(launcher):
    launcher.COMPAT_LIBRARY.rmdir()
    with pytest.raises(RuntimeError, match="libraries are missing"):
        launcher.prepare_environment()


def test_timeout_refuses_launch(launcher, monkeypatch, capsys):
    def probe(*_, **__):
        raise subprocess.TimeoutExpired(launcher.COMPAT_PROBE, 35)

    monkeypatch.setattr(launcher.subprocess, "run", probe)
    monkeypatch.setattr(launcher.os, "execve", lambda *_, **__: pytest.fail("failed probe must not launch"))
    assert launcher.main() == 1
    assert "startup refused" in capsys.readouterr().err

"""Source-build input boundaries; no GPU or source-built qualification implied."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v100_source_foundation", ROOT / "deploy/v100/source_foundation.py")
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)


def report():
    names = ("scikit-build-core", "cmake", "ninja", "numpy", "setuptools", "packaging", "pip", "pyyaml", "six", "typing-extensions")
    return {"environment": {"sys_platform": "linux", "platform_machine": "x86_64",
                            "implementation_name": "cpython", "python_version": "3.12"},
            "install": [{"metadata": {"name": name, "version": "1.0"}, "download_info": {
                "url": f"https://files.pythonhosted.org/packages/{name.replace('-', '_')}-1.0-py3-none-any.whl",
                "archive_info": {"hashes": {"sha256": "a" * 64}}}} for name in names]}


def test_source_build_tools_retain_exact_artifact_bytes_and_resolver_provenance():
    raw = report()
    lock, receipt = SOURCE.render_build_report(raw, "b" * 64)
    assert "numpy @ https://files.pythonhosted.org/packages/numpy-1.0-py3-none-any.whl --hash=sha256:" + "a" * 64 in lock
    assert receipt["resolver_report_sha256"] == "b" * 64
    assert receipt["lock_sha256"] == hashlib.sha256(lock.encode()).hexdigest()
    reordered = copy.deepcopy(raw)
    reordered["install"].reverse()
    assert SOURCE.render_build_report(reordered, "b" * 64) == (lock, receipt)


@pytest.mark.parametrize("provider", ["Torch", "torchvision", "triton", "nvidia_nccl_cu12", "cuda-bindings"])
def test_binary_accelerator_providers_cannot_enter_source_build_tools(provider):
    data = report()
    data["install"][0]["metadata"]["name"] = provider
    with pytest.raises(ValueError, match="accelerator provider"):
        SOURCE.render_build_report(data, "b" * 64)


@pytest.mark.parametrize("field,value", [("sys_platform", "win32"), ("platform_machine", "aarch64"),
                                         ("implementation_name", "pypy"), ("python_version", "3.14")])
def test_source_build_report_must_match_declared_native_target(field, value):
    data = report()
    data["environment"][field] = value
    with pytest.raises(ValueError, match="native Linux AMD64 CPython"):
        SOURCE.render_build_report(data, "b" * 64)


@pytest.mark.parametrize("broken", ["duplicate", "missing", "foreign-url", "wrong-version", "wrong-name", "bad-sha"])
def test_malformed_or_incomplete_source_package_identity_is_rejected(broken):
    data = report()
    first = data["install"][0]
    if broken == "duplicate":
        data["install"].append(copy.deepcopy(first))
    elif broken == "missing":
        data["install"].pop()
    elif broken == "foreign-url":
        first["download_info"]["url"] = "https://unqualified.invalid/scikit_build_core-1.0-py3-none-any.whl"
    elif broken == "wrong-version":
        first["metadata"]["version"] = "2.0"
    elif broken == "wrong-name":
        first["download_info"]["url"] = "https://files.pythonhosted.org/packages/other-1.0-py3-none-any.whl"
    else:
        first["download_info"]["archive_info"]["hashes"]["sha256"] = "not-a-whole-file-hash"
    with pytest.raises(ValueError):
        SOURCE.render_build_report(data, "b" * 64)


def test_native_package_bytes_are_verified_before_installation(tmp_path):
    artifact = tmp_path / "library.deb"
    artifact.write_bytes(b"owned exact package payload")
    pins = {"native_packages": [{"artifact": artifact.name, "bytes": artifact.stat().st_size,
                                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}]}
    SOURCE.verify_native(pins, tmp_path)
    artifact.write_bytes(b"changed exact package payload")
    with pytest.raises(ValueError, match="identity mismatch"):
        SOURCE.verify_native(pins, tmp_path)
    pins["native_packages"][0]["artifact"] = "../library.deb"
    with pytest.raises(ValueError, match="declared local"):
        SOURCE.verify_native(pins, tmp_path)


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'])
def test_metadata_reader_rejects_ambiguous_or_nonfinite_data(tmp_path, raw):
    path = tmp_path / "report.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        SOURCE.read_json(path)


@pytest.mark.skipif(os.name != "posix", reason="the source-build producer targets native Linux")
def test_source_metadata_fifo_is_refused_before_blocking(tmp_path):
    fifo = tmp_path / "metadata"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        SOURCE.read_json(fifo)


def test_recorded_source_stack_is_distinct_from_the_historical_cu126_wheel_recipe():
    pins = json.loads((ROOT / "deploy/v100/source-pins.json").read_text())
    assert pins["targets"] == {"cuda": "12.9.2", "nvcc": "12.9.86", "cublas": "12.9.2.10",
        "pytorch": "2.14.1", "cccl": "3.5.0", "nccl": "2.32.3", "cutlass": "4.8.1",
        "cudnn": "9.10.2.21", "nsight_compute": "2025.2.1.3", "nsight_systems": "2025.3.2.474",
        "compute_sanitizer_package": "12.9.79"}
    assert "ubuntu24.04@sha256:" in pins["base"]
    assert all(len(source["commit"]) == 40 for source in pins["sources"].values())
    assert "candidate" in pins["status"] and "unrun" in pins["status"]

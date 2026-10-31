"""Owned wheel-producer boundaries; the upstream backend itself is not tested."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "deploy/gb10/calibration"


@pytest.fixture
def producer(monkeypatch):
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    role = load("build_test_role", TOOLS / "role.py")
    monkeypatch.setitem(sys.modules, "role", role)
    preflight = load("build_test_preflight", TOOLS / "wheel_preflight.py")
    monkeypatch.setattr(preflight, "verify_git_role", lambda: None)
    monkeypatch.setitem(sys.modules, "wheel_preflight", preflight)
    consumer = load("build_test_consumer", TOOLS / "build_wheel.py")
    # Mock the public backend object, not an alternate production build path.
    consumer.backend = importlib.import_module("setuptools.build_meta")
    return consumer


def arguments(producer, tmp_path, monkeypatch, head=None):
    source = tmp_path / "source"
    source.mkdir()
    (source / ".git").mkdir()
    output = tmp_path / "output"
    executable = tmp_path / "git"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    monkeypatch.setattr(sys.modules["wheel_preflight"].shutil, "which", lambda name: str(executable))
    monkeypatch.setattr(sys, "argv", ["build_wheel.py", "--source", str(source),
        "--output", str(output), "--source-archive-sha256", "a" * 64])
    def git(argv, **kwargs):
        if argv[1] == "--version":
            return SimpleNamespace(stdout="git version 2.43.0")
        command = argv[3]
        text = {"rev-parse": head or producer.MODEL_OPT_COMMIT, "status": "",
                "describe": "genuine-pinned-description", "diff": ""}[command]
        return SimpleNamespace(stdout=text)
    monkeypatch.setattr(producer.subprocess, "run", git)
    return source, output


def test_wrong_source_stops_before_backend_or_output(producer, tmp_path, monkeypatch):
    _, output = arguments(producer, tmp_path, monkeypatch, head="wrong")
    with pytest.raises(ValueError, match="exact clean"):
        producer.main()
    assert not output.exists()


def test_scm_override_is_rejected(producer, tmp_path, monkeypatch):
    _, output = arguments(producer, tmp_path, monkeypatch)
    monkeypatch.setenv("SETUPTOOLS_SCM_PRETEND_VERSION", "invented")
    with pytest.raises(ValueError, match="overrides"):
        producer.main()
    assert not output.exists()


@pytest.mark.parametrize("requirement", ["setuptools>=80", "torch>=2.10"])
def test_backend_version_and_wheel_hash_are_recorded_without_override(producer, tmp_path, monkeypatch, requirement):
    source, output = arguments(producer, tmp_path, monkeypatch)
    (source / "pyproject.toml").write_text(f'[build-system]\nrequires=["{requirement}"]\nbuild-backend="setuptools.build_meta"\n')
    previous = Path.cwd()
    def backend(destination):
        assert Path.cwd() == source
        artifact = Path(destination) / "nvidia_modelopt-0.0.0-py3-none-any.whl"
        with zipfile.ZipFile(artifact, "w") as wheel:
            wheel.writestr("nvidia_modelopt-0.0.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: nvidia-modelopt\nVersion: 0.0.0\n")
            wheel.writestr("modelopt/__init__.py", "# fixture")
        return artifact.name
    monkeypatch.setattr(producer.backend, "build_wheel", backend)
    monkeypatch.setattr(producer.backend, "get_requires_for_build_wheel", lambda: [])
    monkeypatch.setattr(producer.md, "version", lambda name: "2.16.0.dev20261014+cu134" if name == "torch" else "81.0.0")
    producer.main()
    receipt = json.loads((output / "source.json").read_text())
    assert Path.cwd() == previous
    assert receipt["wheel_version"] == "0.0.0"
    assert receipt["source_commit"] == producer.MODEL_OPT_COMMIT
    assert receipt["source_type"] == "incremental-git-bundle"
    assert receipt["upstream_repository"] == producer.MODEL_OPT_UPSTREAM_REPOSITORY
    assert receipt["upstream_commit"] == producer.MODEL_OPT_UPSTREAM_COMMIT
    assert receipt["source_repository"] == "https://github.com/kklouzal/TensorFold"
    assert receipt["source_bundle_path"] == producer.MODEL_OPT_BUNDLE_PATH
    assert receipt["source_bundle_sha256"] == producer.MODEL_OPT_BUNDLE_SHA256
    assert receipt["wheel_sha256"] == producer.sha(output / "nvidia_modelopt-0.0.0-py3-none-any.whl")
    assert receipt["git_describe"] == "genuine-pinned-description"


def test_unsatisfied_dynamic_backend_dependency_stops_before_build(producer, tmp_path, monkeypatch):
    source, output = arguments(producer, tmp_path, monkeypatch)
    (source / "pyproject.toml").write_text('[build-system]\nrequires=["setuptools>=80"]\nbuild-backend="setuptools.build_meta"\n')
    previous = Path.cwd()
    monkeypatch.setattr(producer.backend, "get_requires_for_build_wheel", lambda: ["missing-builder>=100"])
    def version(name):
        if name == "setuptools":
            return "81.0.0"
        raise producer.md.PackageNotFoundError(name)
    monkeypatch.setattr(producer.md, "version", version)
    def forbidden(*args):
        pytest.fail("build started with an unsatisfied dynamic requirement")
    monkeypatch.setattr(producer.backend, "build_wheel", forbidden)
    with pytest.raises(producer.md.PackageNotFoundError):
        producer.main()
    assert Path.cwd() == previous
    assert not list(output.glob("*.whl"))

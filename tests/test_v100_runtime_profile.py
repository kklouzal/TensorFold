"""Owned runtime revision boundaries; no native or hardware qualification."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("runtime_profile_source", ROOT / "deploy/v100/source_foundation.py")
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)


def inputs(tmp_path):
    profile = json.loads((ROOT / "deploy/v100/source-runtime-triton32-input.json").read_text())
    wheel = tmp_path / profile["triton"]["artifact"]
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("triton-3.2.0.dist-info/METADATA", "Name: triton\nVersion: 3.2.0\nRequires-Dist: filelock\n\n")
    profile["triton"].update(SOURCE.file_identity(wheel))
    row = {"version": "3.9.0", "url": "https://files.pythonhosted.org/packages/triton-3.9.0-py3-none-any.whl",
           "sha256": "a" * 64, "requires_dist": []}
    base = {"schema": "tensorfold.v100-source-runtime", "version": 2, "environment": {},
            "lock_sha256": "b" * 64, "resolver_report_sha256": "c" * 64,
            "packages": {"triton": row, "filelock": {**row, "version": "4.2.10"}},
            "source_distributions": {"torch": {"version": "2.16.0+owned", "requires_dist": []},
                                     "tensorfold": {"version": "0.6.6", "requires_dist": []}},
            "triton_execution": "original restricted FMA evidence"}
    raw = SOURCE.canonical(base)
    report = {"environment": {"sys_platform": "linux", "platform_machine": "x86_64",
                              "implementation_name": "cpython", "python_version": "3.12"},
              "install": [{"is_direct": True, "metadata": {"name": "triton", "version": "3.2.0", "requires_dist": ["filelock"]},
                           "download_info": {"url": "file:///wheel/" + wheel.name,
                               "archive_info": {"hashes": {"sha256": profile["triton"]["sha256"]}}}}]}
    return base, raw, report, profile, wheel


def candidate(values):
    base, raw, report, profile, wheel = values
    return SOURCE.render_runtime_profile(base, hashlib.sha256(raw).hexdigest(), report, "d" * 64, profile, wheel)


def test_renderer_retains_full_base_metadata_and_records_actual_single_delta(tmp_path):
    values = inputs(tmp_path)
    base, raw, _, profile, _ = values
    before = copy.deepcopy(base)
    lock, receipt = candidate(values)
    assert base == before
    assert receipt["packages"]["filelock"] == base["packages"]["filelock"]
    assert receipt["source_distributions"] == base["source_distributions"]
    assert receipt["base_resolver_report_sha256"] == base["resolver_report_sha256"]
    assert receipt["runtime_profile"]["base_runtime_sha256"] == hashlib.sha256(raw).hexdigest()
    assert "triton @ " + profile["triton"]["url"] in lock
    assert receipt["packages"]["triton"]["requires_dist"] == ["filelock"]
    assert receipt["lock_sha256"] == hashlib.sha256(lock.encode()).hexdigest()
    SOURCE.validate_runtime_profile_delta(base, raw, receipt, profile)


@pytest.mark.parametrize("fault", ["other-package", "requires", "sha", "foreign-url", "environment", "version", "old-base"])
def test_renderer_rejects_undeclared_or_non_genuine_inputs(tmp_path, fault):
    base, raw, report, profile, wheel = values = inputs(tmp_path)
    item = report["install"][0]
    if fault == "other-package":
        report["install"].append(copy.deepcopy(item))
    elif fault == "requires":
        item["metadata"]["requires_dist"] = []
    elif fault == "sha":
        item["download_info"]["archive_info"]["hashes"]["sha256"] = "e" * 64
    elif fault == "foreign-url":
        profile["triton"]["url"] = profile["triton"]["url"].replace("files.pythonhosted.org", "invalid.example")
    elif fault == "environment":
        report["environment"]["platform_machine"] = "aarch64"
    elif fault == "version":
        profile["version"] = True
    else:
        base["packages"]["triton"]["version"] = "3.8.0"
    with pytest.raises(ValueError):
        candidate(values)


@pytest.mark.parametrize("fault", ["dependency", "source", "lock", "unknown", "base-hash"])
def test_revision_cannot_hide_non_triton_or_cache_binding_changes(tmp_path, fault):
    values = inputs(tmp_path)
    base, raw, _, profile, _ = values
    _, receipt = candidate(values)
    if fault == "dependency":
        receipt["packages"]["filelock"] = {**receipt["packages"]["filelock"], "version": "unexpected"}
    elif fault == "source":
        receipt["source_distributions"] = {"torch": {"version": "different"}}
    elif fault == "lock":
        receipt["lock_sha256"] = "e" * 64
    elif fault == "base-hash":
        receipt["runtime_profile"]["base_runtime_sha256"] = "e" * 64
    else:
        receipt["unknown"] = "not a declared role"
    with pytest.raises(ValueError):
        SOURCE.validate_runtime_profile_delta(base, raw, receipt, profile)


def test_generated_recipe_audits_before_and_after_and_never_rebuilds_framework(tmp_path):
    values = inputs(tmp_path)
    _, receipt = candidate(values)
    recipe = SOURCE.runtime_profile_dockerfile({"unchanged": "pins"}, receipt, values[3])
    assert recipe.index(" audit-process ") < recipe.index("prepare-runtime-profile") < recipe.index("pip install")
    assert recipe.index("pip install") < recipe.index("apply-runtime-profile") < recipe.rindex(" audit-process")
    assert "--require-hashes" in recipe and "--no-deps" in recipe and "pip check" in recipe
    assert "framework --" not in recipe and "pip wheel" not in recipe
    assert "/" + receipt["lock_sha256"] + "/triton" in recipe
    assert "TENSORFOLD_SM70_TRITON_PROFILE=sm70-triton32-v1" in recipe
    assert recipe.index("chmod 0555 /runtime-profile") < recipe.index("apply-runtime-profile")
    assert recipe.index("chmod 0444 /runtime-profile/revision-before.json") < recipe.index("apply-runtime-profile")
    assert recipe.index("os.setgroups([])") < recipe.index("os.setgid(1000)") < recipe.index("os.setuid(1000)")
    assert recipe.index("os.setuid(1000)") < recipe.index("unprivileged_artifact_access") < recipe.index("issue_source_binding")
    assert "verify_source_binding(authority)" in recipe


@pytest.mark.skipif(os.getuid() != 1000 or os.getgid() != 1000, reason="requires actual unprivileged worker UID/GID1000")
def test_unprivileged_gate_finds_all_unsearchable_parents_and_recovers_after_owned_fix(tmp_path):
    entries = []
    directories = []
    for index in range(2):
        directory = tmp_path / str(index)
        directory.mkdir()
        path = directory / "sealed.json"
        path.write_text("immutable metadata")
        path.chmod(0o444)
        entries.append({"path": str(path)})
        directories.append(directory)
    for directory in directories:
        directory.chmod(0o444)
    try:
        with pytest.raises(PermissionError) as caught:
            SOURCE.unprivileged_artifact_access({"artifacts": entries})
        summary = json.loads(str(caught.value).split(": ", 1)[1])
        assert summary["unsearchable_parents"] == summary["unreadable_artifacts"] == 2
    finally:
        for directory in directories:
            directory.chmod(0o555)
    try:
        summary = SOURCE.unprivileged_artifact_access({"artifacts": entries})
        assert summary["uid"] == summary["gid"] == 1000 and summary["artifacts_checked"] == 2
        assert summary["unsearchable_parents"] == summary["unreadable_artifacts"] == 0
    finally:
        for directory in directories:
            directory.chmod(0o755)


def test_post_install_revision_proves_original_bytes_and_keeps_build_provenance(tmp_path, monkeypatch):
    base, raw, report, profile, wheel = values = inputs(tmp_path)
    _, runtime = candidate(values)
    directory, prior = tmp_path / "pins", tmp_path / "prior"
    directory.mkdir()
    prior.mkdir()
    pins = {"immutable": "source providers"}
    old_producer = prior / "source_foundation.py"
    old_producer.write_bytes(b"original producer, no new framework compilation")
    (directory / "source_foundation.py").write_bytes(b"new runtime auditor")
    (prior / "source-runtime-amd64.json").write_bytes(raw)
    (directory / "source-runtime-amd64.json").write_bytes(SOURCE.canonical(runtime))
    torch = tmp_path / "libtorch.so"
    torch.write_bytes(b"immutable original Torch/native bytes")
    build = tmp_path / "source-build.json"
    build.write_bytes(SOURCE.canonical({"producer_inputs": {"deploy/v100/source_foundation.py": SOURCE.file_identity(old_producer)}}))
    identity = {"torch": "2.16.0+owned", "triton": {"version": "3.9.0"}, "mapped_libraries": [str(torch)]}
    old_path = "/opt/venv/lib/python3.12/site-packages/triton/_removed_fixture.py"
    removed = {"path": old_path, "bytes": 1, "sha256": "a" * 64}
    artifacts = [{"path": str(directory / "source_foundation.py"), **SOURCE.file_identity(old_producer)},
                 {"path": str(directory / "source-runtime-amd64.json"), **SOURCE.file_identity(prior / "source-runtime-amd64.json")},
                 {"path": str(torch), **SOURCE.file_identity(torch)}, {"path": str(build), **SOURCE.file_identity(build)}, removed]
    previous = {"schema": "tensorfold.v100-source-foundation-install", "version": 2,
        "pins_sha256": hashlib.sha256(SOURCE.canonical(pins)).hexdigest(), "release_qualified": False,
        "runtime": identity, "build_receipt": {"path": str(build), **SOURCE.file_identity(build)}, "artifacts": artifacts}
    (prior / "v100-source-foundation.json").write_bytes(SOURCE.canonical(previous))
    (prior / "audit-process.json").write_text('{"scope":"historical prior check"}')
    profile_path = tmp_path / "profile.json"
    profile_path.write_bytes(SOURCE.canonical(profile))
    monkeypatch.setattr(SOURCE, "distribution_artifacts", lambda name: [removed])
    monkeypatch.setattr(SOURCE.importlib.metadata, "version", lambda name: "3.9.0")
    prepared = SOURCE.prepare_runtime_profile(pins, prior, directory / "source-runtime-amd64.json", profile_path, wheel,
                                               directory=directory)
    prepared_path = tmp_path / "before.json"
    prepared_path.write_bytes(SOURCE.canonical(prepared))
    monkeypatch.setattr(SOURCE, "distribution_artifacts", lambda name: [])
    monkeypatch.setattr(SOURCE.importlib.metadata, "version", lambda name: "3.2.0")
    class Distribution:
        def __init__(self, name):
            wanted = {**runtime["packages"], **runtime["source_distributions"]}[name]
            self.version, self.requires = wanted["version"], wanted["requires_dist"]
    monkeypatch.setattr(SOURCE.importlib.metadata, "distribution", Distribution)
    monkeypatch.setattr(SOURCE, "runtime_identity", lambda p: {**identity, "triton": {"version": "3.2.0"}})
    monkeypatch.setattr(SOURCE, "validate_runtime", lambda *args: None)
    monkeypatch.setattr(SOURCE, "audit_install", lambda *args: None)
    original_build = build.read_bytes()
    result = SOURCE.apply_runtime_profile(pins, prior, prepared_path, profile_path, directory=directory)
    assert build.read_bytes() == original_build
    assert result["runtime_dependency_revision"]["original_build_producer"] == SOURCE.file_identity(old_producer)
    assert result["runtime_dependency_revision"]["current_runtime_auditor"] == SOURCE.file_identity(Path(SOURCE.__file__))
    assert any(p["path"] == str(old_producer) for p in result["artifacts"])
    torch.write_bytes(b"changed bytes must never be excused by a Triton revision")
    with pytest.raises(ValueError, match="artifact changed"):
        SOURCE.apply_runtime_profile(pins, prior, prepared_path, profile_path, directory=directory)

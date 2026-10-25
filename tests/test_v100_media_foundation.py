"""Owned media acquisition boundaries; no processor/model/codec qualification."""

import hashlib
import importlib.util
import copy
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy/v100"))
try:
    SPEC = importlib.util.spec_from_file_location("v100_media_foundation", ROOT / "deploy/v100/media_foundation.py")
    MEDIA = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MEDIA)
finally:
    sys.path.pop(0)


def pin(payload=b"exact source archive"):
    return {"name": "owned", "artifact": "owned.tar.gz", "url": "https://github.com/owner/project/source.tar.gz",
            "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def test_partial_transfer_is_resumed_and_exact_bytes_publish_atomically(tmp_path, monkeypatch):
    payload = b"exact source archive"
    partial = tmp_path / "owned.tar.gz.part"
    partial.write_bytes(payload[:5])
    seen = []

    def transfer(command, **kwargs):
        seen.append((command, kwargs))
        assert partial.read_bytes() == payload[:5]
        partial.write_bytes(payload)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(MEDIA.subprocess, "run", transfer)
    assert MEDIA.fetch_artifact(pin(payload), tmp_path, deadline_s=60) == "downloaded"
    assert (tmp_path / "owned.tar.gz").read_bytes() == payload and not partial.exists()
    assert seen[0][0][seen[0][0].index("--continue-at") + 1] == "-"
    assert seen[0][0][seen[0][0].index("--max-filesize") + 1] == str(len(payload))
    assert seen[0][1]["timeout"] == 90
    assert MEDIA.fetch_artifact(pin(payload), tmp_path) == "reused" and len(seen) == 1


@pytest.mark.parametrize("failure", ["transport", "identity"])
def test_failed_or_changed_download_retains_partial_without_publishing(tmp_path, monkeypatch, failure):
    def transfer(command, **kwargs):
        (tmp_path / "owned.tar.gz.part").write_bytes(b"wrong source payload")
        return subprocess.CompletedProcess(command, 28 if failure == "transport" else 0, "", "")

    monkeypatch.setattr(MEDIA.subprocess, "run", transfer)
    with pytest.raises((RuntimeError, ValueError), match="partial retained"):
        MEDIA.fetch_artifact(pin(), tmp_path)
    assert (tmp_path / "owned.tar.gz.part").exists() and not (tmp_path / "owned.tar.gz").exists()


def test_dangling_partial_alias_is_refused_before_transport(tmp_path, monkeypatch):
    (tmp_path / "owned.tar.gz.part").symlink_to(tmp_path / "outside")
    monkeypatch.setattr(MEDIA.subprocess, "run", lambda *args, **kwargs: pytest.fail("transport reached"))
    with pytest.raises(ValueError, match="bounded regular"):
        MEDIA.fetch_artifact(pin(), tmp_path)


@pytest.mark.parametrize("change", ["traversal", "credentials", "origin", "hash", "bytes"])
def test_malformed_pins_refuse_before_transport(tmp_path, monkeypatch, change):
    row = pin()
    if change == "traversal":
        row["artifact"] = "../outside"
    elif change == "credentials":
        row["url"] = "https://token@github.com/owner/source"
    elif change == "origin":
        row["url"] = "https://unapproved.invalid/source"
    elif change == "hash":
        row["sha256"] = "short"
    else:
        row["bytes"] = True
    monkeypatch.setattr(MEDIA.subprocess, "run", lambda *args, **kwargs: pytest.fail("transport reached"))
    with pytest.raises(ValueError):
        MEDIA.fetch_artifact(row, tmp_path)


def test_source_extraction_uses_exact_archive_and_cumulative_budget(tmp_path):
    downloads, sources = tmp_path / "downloads", tmp_path / "sources"
    downloads.mkdir()
    initial = tmp_path / "root"
    initial.mkdir()
    (initial / "source.txt").write_bytes(b"canonical source")
    archive = downloads / "owned.tar.gz"
    with tarfile.open(archive, "w:gz") as writer:
        writer.add(initial, arcname="selected-source")
    row = pin(archive.read_bytes())
    row["extraction_max_bytes"] = 1000
    target = MEDIA.extract_source(row, downloads, sources)
    assert (target / "source.txt").read_bytes() == b"canonical source"
    assert MEDIA.extract_source(row, downloads, sources) == target
    (target / "source.txt").write_bytes(b"unrecorded source modification")
    with pytest.raises(ValueError, match="source input changed"):
        MEDIA.extract_source(row, downloads, sources)
    row["name"] = "bounded"
    row["extraction_max_bytes"] = 1
    with pytest.raises(ValueError, match="extraction budget"):
        MEDIA.extract_source(row, downloads, sources)


def test_source_cache_accepts_only_original_or_exact_completed_generated_inventory(tmp_path):
    downloads, sources = tmp_path / "downloads", tmp_path / "sources"
    downloads.mkdir()
    initial = tmp_path / "root"
    initial.mkdir()
    (initial / "source.py").write_text("immutable = True\n")
    archive = downloads / "owned.tar.gz"
    with tarfile.open(archive, "w:gz") as writer:
        writer.add(initial, arcname="selected-source")
    row = pin(archive.read_bytes())
    row["extraction_max_bytes"] = 1000
    target = MEDIA.extract_source(row, downloads, sources)
    generated = target / "generated.o"
    generated.write_bytes(b"completed build output")
    with pytest.raises(ValueError, match="unrecorded generated"):
        MEDIA.verify_source(row, downloads, sources)
    receipt = tmp_path / "completed.json"
    receipt.write_bytes(MEDIA.canonical(MEDIA.source_inventory(row, downloads, sources)))
    MEDIA.verify_source(row, downloads, sources, receipt)
    generated.write_bytes(b"substituted cache bytes")
    with pytest.raises(ValueError, match="last completed stage"):
        MEDIA.verify_source(row, downloads, sources, receipt)


def test_source_cache_refuses_new_external_symlink(tmp_path):
    downloads, sources = tmp_path / "downloads", tmp_path / "sources"
    downloads.mkdir()
    initial = tmp_path / "root"
    initial.mkdir()
    (initial / "source.py").write_text("immutable = True\n")
    archive = downloads / "owned.tar.gz"
    with tarfile.open(archive, "w:gz") as writer:
        writer.add(initial, arcname="selected-source")
    row = pin(archive.read_bytes())
    row["extraction_max_bytes"] = 1000
    target = MEDIA.extract_source(row, downloads, sources)
    (target / "external").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="introduce aliases"):
        MEDIA.source_inventory(row, downloads, sources)


def media_manifest(tmp_path, monkeypatch):
    artifact = tmp_path / "owned-media.so"
    artifact.write_bytes(b"actual installed media input")
    runtime = {"actual": "CPU processor and decoder identity"}
    pins = {"schema": "tensorfold.v100-media-inputs", "version": 1}
    monkeypatch.setattr(MEDIA, "runtime_inventory", lambda: runtime)
    monkeypatch.setattr(MEDIA, "installed_paths", lambda *args: {str(artifact)})
    manifest = {"schema": "tensorfold.v100-media-install", "version": 1,
                "pins_sha256": hashlib.sha256(MEDIA.canonical(pins)).hexdigest(),
                "packages": ["av"], "runtime": runtime,
                "artifacts": [{"path": str(artifact), **MEDIA.file_identity(artifact)}],
                "release_qualified": False, "pending_gates": ["V100", "compiler-6111901"]}
    return manifest, pins, artifact


def test_media_audit_binds_actual_files_runtime_and_pending_gates(tmp_path, monkeypatch):
    manifest, pins, _ = media_manifest(tmp_path, monkeypatch)
    result = MEDIA.audit_install(manifest, pins)
    assert result["software_integrity"] == "passed" and result["release_qualified"] is False
    assert result["pending_gates"] == ["V100", "compiler-6111901"]


@pytest.mark.parametrize("change", ["bytes", "runtime", "extra", "duplicate", "schema", "version", "pins", "qualified"])
def test_media_audit_refuses_changed_or_forged_inputs(tmp_path, monkeypatch, change):
    manifest, pins, artifact = media_manifest(tmp_path, monkeypatch)
    if change == "bytes":
        artifact.write_bytes(b"substituted media library")
    elif change == "runtime":
        monkeypatch.setattr(MEDIA, "runtime_inventory", lambda: {"actual": "different"})
    elif change == "extra":
        monkeypatch.setattr(MEDIA, "installed_paths", lambda *args: {str(artifact), "/extra-input"})
    elif change == "duplicate":
        manifest["artifacts"].append(copy.deepcopy(manifest["artifacts"][0]))
    elif change == "schema":
        manifest["schema"] = "unknown"
    elif change == "version":
        manifest["version"] = True
    elif change == "pins":
        pins["unexpected"] = "changed"
    else:
        manifest["release_qualified"] = True
    with pytest.raises(ValueError):
        MEDIA.audit_install(manifest, pins)

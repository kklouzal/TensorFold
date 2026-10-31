"""OS-addon boundaries; no package manager installation or native build runs."""
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location("git_os_control",
    Path(__file__).resolve().parents[1] / "deploy/gb10/calibration/git_os.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_plan_reads_actual_tab_protocol_and_keeps_control_metadata(tmp_path, monkeypatch):
    metadata, debs = tmp_path / "metadata", tmp_path / "debs"
    metadata.mkdir()
    debs.mkdir()
    raw = b"old:arm64" + bytes([9]) + b"1\n"
    for name in ("dpkg-before.stdout", "dpkg-after.stdout"):
        (metadata / name).write_bytes(raw)
    (metadata / "summary.json").write_text(json.dumps({"dpkg_unchanged": True,
        "new_packages": [{"package": "git", "version": "2", "metadata_file": "git.stdout"}]}))
    artifact = debs / "git.deb"
    artifact.write_bytes(b"fixture archive")
    checksum = module.digest(artifact)
    (metadata / "git.stdout").write_text(f"Package: git\nVersion: 2\nArchitecture: arm64\nSHA256: {checksum}\n")
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(json.dumps({"packages": [{"name": "git", "version": "2", "architecture": "arm64",
        "filename": artifact.name, "sha256": checksum, "size": artifact.stat().st_size}]}))
    old = {"old:arm64": {"version": "1", "status": "install ok installed"}}
    monkeypatch.setattr(module, "PACKAGES", {"git"})
    monkeypatch.setattr(module, "dpkg_state", lambda: old)
    monkeypatch.setattr(module, "dpkg_config", lambda: {"files": {}, "filters": [list(rule) for rule in module.FILTERS]})
    monkeypatch.setattr(module, "run", lambda *args: "arm64\n")
    monkeypatch.setattr(module, "inspect_deb", lambda path: {"control": {"Package": "git", "Version": "2", "Architecture": "arm64"},
        "maintainers": {"postinst": "# inspected fixture"}, "payload": []})
    output = tmp_path / "plan.json"
    module.plan(metadata, debs, acquisition, output)
    plan = module.load(output)
    assert plan["baseline"] == old
    assert plan["packages"][0]["maintainers"]["postinst"] == "# inspected fixture"
    assert plan["inspector_sha256"] == module.digest(module.__file__)


@pytest.mark.parametrize("path", ["../old", "/usr/bin/git"])
def test_deb_member_cannot_escape_its_relative_archive_namespace(path):
    with pytest.raises(ValueError, match="unsafe"):
        module.payload_path(path)
    assert module.payload_path("./usr/bin/git") == Path("/usr/bin/git")


@pytest.mark.parametrize("changed", ["version", "status"])
def test_all_old_versions_and_configuration_states_are_preserved(changed):
    original = {"old": {"version": "1", "status": "install ok installed"}}
    after = {"old": {**original["old"], changed: "changed"}}
    with pytest.raises(ValueError, match="existing dpkg"):
        module.old_unchanged(original, after)


def installer(tmp_path, monkeypatch, *, collision=False, failed=False, filtered_doc=False, missing_runtime=False):
    old = {"old": {"version": "1", "status": "install ok installed"}}
    current = dict(old)
    artifact = tmp_path / "git.deb"
    artifact.write_bytes(b"fixture")
    target = tmp_path / "git"
    details = {"control": {}, "maintainers": {}, "payload": [
        {"path": str(target), "type": "file", "old_path_collision": collision}]}
    config = {"files": {}, "filters": [list(rule) for rule in module.FILTERS]}
    if filtered_doc:
        details["payload"].append({"path": "/usr/share/doc/git-policy-fixture-neverinstalled/README.Debian",
            "type": "file", "mode": 0o644, "old_path_collision": False})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 2, "inspector_sha256": module.digest(module.__file__),
        "dpkg_config": config, "filtered_paths": module.filter_omissions([details], config),
        "baseline": old, "packages": [{"name": "git", "version": "2", "filename": artifact.name,
        "size": artifact.stat().st_size, "sha256": module.digest(artifact), **details}]}))
    monkeypatch.setattr(module, "PACKAGES", {"git"})
    monkeypatch.setattr(module, "dpkg_state", lambda: current.copy())
    monkeypatch.setattr(module, "dpkg_config", lambda: config)
    monkeypatch.setattr(module, "inspect_deb", lambda path: details)
    calls = []
    def run(argv, **kwargs):
        calls.append(argv[0])
        if argv[0] == "dpkg":
            if failed:
                raise subprocess.CalledProcessError(7, argv)
            if not missing_runtime:
                target.write_bytes(b"actual installed bytes")
            current["git"] = {"version": "2", "status": "install ok installed"}
    monkeypatch.setattr(module.subprocess, "run", run)
    return manifest, target, calls


def test_successful_materialization_seals_and_rechecks_new_files(tmp_path, monkeypatch):
    manifest, target, calls = installer(tmp_path, monkeypatch)
    state = tmp_path / "state.json"
    module.install(manifest, tmp_path, state)
    module.verify(state)
    assert calls == [module.sys.executable, "dpkg", module.sys.executable]
    target.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="file/link changed"):
        module.verify(state)


def test_install_failure_still_audits_and_never_publishes_state(tmp_path, monkeypatch):
    manifest, _, calls = installer(tmp_path, monkeypatch, failed=True)
    state = tmp_path / "state.json"
    with pytest.raises(ExceptionGroup) as failure:
        module.install(manifest, tmp_path, state)
    assert isinstance(failure.value.exceptions[0], subprocess.CalledProcessError)
    assert calls[-1] == module.sys.executable
    assert not state.exists()


def test_old_payload_collision_stops_before_package_manager(tmp_path, monkeypatch):
    manifest, _, calls = installer(tmp_path, monkeypatch, collision=True)
    with pytest.raises(ValueError, match="existing non-directory"):
        module.install(manifest, tmp_path, tmp_path / "state.json")
    assert "dpkg" not in calls


def test_actual_base_dpkg_filter_policy_is_captured():
    contract = module.dpkg_config()
    assert contract["filters"] == [list(rule) for rule in module.FILTERS]
    assert "/etc/dpkg/dpkg.cfg.d/excludes" in contract["files"]


def test_ordered_filters_reinclude_copyright_and_changelogs():
    names = ["README.Debian", "copyright", "changelog.Debian.gz"]
    package = {"payload": [{"path": f"/usr/share/doc/git/{name}", "type": "file", "mode": 0o644} for name in names]}
    result = module.filter_omissions([package], {"filters": module.FILTERS})
    assert set(result) == {"/usr/share/doc/git/README.Debian"}


@pytest.mark.parametrize("extra", [{"mode": 0o755}, {"mode": 0o644, "elf": {"soname": None}}])
def test_exclusion_cannot_hide_runtime_executable_or_library(extra):
    package = {"payload": [{"path": "/usr/share/doc/git/binary", "type": "file", **extra}]}
    with pytest.raises(ValueError, match="runtime executable/library"):
        module.filter_omissions([package], {"filters": module.FILTERS})


def test_declared_doc_omission_is_recorded_not_silently_skipped(tmp_path, monkeypatch):
    manifest, _, _ = installer(tmp_path, monkeypatch, filtered_doc=True)
    state = tmp_path / "state.json"
    module.install(manifest, tmp_path, state)
    body = module.verify(state)
    assert body["omitted_by_dpkg"] == {"/usr/share/doc/git-policy-fixture-neverinstalled/README.Debian": {
        "rule": "/usr/share/doc/*", "type": "file"}}


def test_missing_unfiltered_runtime_still_fails_and_publishes_no_seal(tmp_path, monkeypatch):
    manifest, _, _ = installer(tmp_path, monkeypatch, missing_runtime=True)
    state = tmp_path / "state.json"
    with pytest.raises(FileNotFoundError):
        module.install(manifest, tmp_path, state)
    assert not state.exists()


def test_changed_filter_policy_is_refused_before_install(tmp_path, monkeypatch):
    manifest, _, calls = installer(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "dpkg_config", lambda: {"files": {"changed": "sha"}, "filters": []})
    with pytest.raises(ValueError, match="filter configuration changed"):
        module.install(manifest, tmp_path, tmp_path / "state.json")
    assert calls == []

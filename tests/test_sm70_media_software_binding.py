"""Combined authority boundary controls with owned fake producers, not SDK proof."""

import hashlib
import json
import os
import sys
import types

import pytest

from tensorfold.cuda.sm70 import media_software_binding as media
from tensorfold.cuda.sm70 import software_binding as source

SOURCE_PRODUCER = b'''import os,hashlib,json
def application_nccl_identity(): return {"path":"source-provider","version":23203}
def runtime_identity():
    path=os.environ.get("TF_TEST_PROCESS_PATH")
    return {"mapped_libraries":[path] if path else [],"host_driver_libraries":[]}
def audit_process(manifest,pins):
    actual={"marker":os.environ.get("TF_TEST_SOURCE_RUNTIME","a"*64),"paths":runtime_identity()}
    return {"schema":"tensorfold.v100-source-foundation-process-audit","version":1,
            "software_integrity":"passed","release_qualified":False,
            "runtime_sha256":hashlib.sha256(json.dumps(actual,sort_keys=True).encode()).hexdigest(),
            "pending_gates":["compiler-6111901"],"additional_process_libraries":[],"host_driver_artifacts":[]}
'''

MEDIA_PRODUCER = b'''import os
from source_foundation import runtime_identity
CALLS=0
def audit_install(manifest,pins):
    global CALLS
    CALLS+=1
    runtime_identity()
    if pins.get("late_source_runtime") and CALLS>1: os.environ["TF_TEST_SOURCE_RUNTIME"]="e"*64
    if pins.get("fail"): raise ValueError("owned media refusal")
    if pins.get("mutate"):
        with open(pins["mutate"],"wb") as out: out.write(b"changed in media audit")
    return {"schema":pins.get("result_schema","tensorfold.v100-media-audit"),"version":1,
            "software_integrity":"passed","release_qualified":False,"artifact_count":len(manifest["artifacts"]),
            "runtime_sha256":os.environ.get("TF_TEST_MEDIA_RUNTIME","d"*64),
            "pending_gates":["service-input-bounds-and-format-parity"]}
'''


def identity(path):
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def protected(tmp_path, monkeypatch):
    base_root, media_root = tmp_path / "source", tmp_path / "media"
    base_root.mkdir()
    media_root.mkdir()
    base_paths = tuple(base_root / name for name in (
        "v100-source-foundation.json", "source-pins.json", "source_foundation.py"))
    producer = base_root / "media_foundation.py"
    media_paths = tuple(media_root / name for name in (
        "media-install.json", "media-pins.json", "media-build.json", "preprocessor_config.json",
        "video_preprocessor_config.json")) + (producer,)
    monkeypatch.setattr(source, "_ROOT", base_root)
    monkeypatch.setattr(source, "_PROTECTED", base_paths)
    monkeypatch.setattr(source, "_ADMIN_UID", os.getuid())
    monkeypatch.setattr(source, "_protected_location", lambda root=None: None)
    monkeypatch.setattr(media, "_ROOT", media_root)
    monkeypatch.setattr(media, "_PRODUCER", producer)
    monkeypatch.setattr(media, "_PROTECTED", media_paths)
    monkeypatch.setenv("TF_TEST_SOURCE_RUNTIME", "a" * 64)
    monkeypatch.setenv("TF_TEST_MEDIA_RUNTIME", "d" * 64)
    monkeypatch.setenv("TF_TEST_PROCESS_PATH", "")
    return base_paths, media_paths


def install(protected, **options):
    base, files = protected
    base[2].write_bytes(SOURCE_PRODUCER)
    base_data = base[0].parent / "source.data"
    base_data.write_bytes(b"original source input")
    base[0].write_bytes(source._canonical({"artifacts": [
        {"path": str(path), **identity(path)} for path in (base[2], base_data)]}))
    base[1].write_bytes(source._canonical({"targets": {"torch": "source"}, "specification_sha256": "b" * 64}))
    files[-1].write_bytes(MEDIA_PRODUCER)
    files[3].write_bytes(b'{"role":"official image"}')
    files[4].write_bytes(b'{"role":"official video"}')
    pins = {"schema": "tensorfold.v100-media-inputs", "version": 1, "specification_sha256": "c" * 64, **options}
    files[1].write_bytes(source._canonical(pins))
    files[2].write_bytes(source._canonical({
        "schema": "tensorfold.v100-media-build", "version": 1,
        "pins_sha256": hashlib.sha256(source._canonical(pins)).hexdigest(),
        "producer_inputs": {"source_foundation.py": identity(base[2]), "media_foundation.py": identity(files[-1])}}))
    media_data = files[0].parent / "media.data"
    media_data.write_bytes(b"original media input")
    files[0].write_bytes(source._canonical({"schema": "tensorfold.v100-media-install", "version": 1,
        "runtime": {"image_backend": "torchvision", "decoder_backend": "pyav", "packages": {"torch": "source"}},
        "artifacts": [{"path": str(path), **identity(path)} for path in (*files[1:], media_data)]}))
    for path in (*base, *files, base_data, media_data):
        path.chmod(0o644)
    return media_data


def test_combined_authority_is_process_issued_and_software_only(protected):
    install(protected)
    authority = media.issue_media_binding()
    assert getattr(authority._media, "states", None) is None
    record = media.verify_media_binding(authority)
    assert record["status"] == "verified-source-and-media-software"
    assert record["software_only"] is True and record["release_qualified"] is False
    assert len(record["protected_media"]) == 6
    assert {"compiler-6111901", "SC117-mmproj-model-equivalence"} <= set(record["pending_gates"])
    record["pending_gates"].clear()
    assert authority.record()["pending_gates"]
    for wrong in (record, source.issue_source_binding(required=True), source.ReferenceSoftwareStatus(),
                  object.__new__(media.MediaSoftwareBinding), None):
        with pytest.raises(source.SoftwareBindingError, match="forged"):
            media.verify_media_binding(wrong)
        with pytest.raises(source.SoftwareBindingError, match="forged"):
            media.reissue_media_binding(wrong)
    with pytest.raises(TypeError, match="process-issued"):
        media.MediaSoftwareBinding(None, None, None, None, None)


@pytest.mark.parametrize("missing", ["source-all", "media-all", 0, 1, 2, 3, 4, 5])
def test_missing_or_partial_protected_media_never_downgrades_to_reference(protected, missing):
    install(protected)
    base, files = protected
    for path in base if missing == "source-all" else files if missing == "media-all" else (files[missing],):
        path.unlink()
    with pytest.raises(source.SoftwareBindingError, match="absent|incomplete"):
        media.issue_media_binding()


def test_exact_private_source_dependency_ignores_and_preserves_ambient_import(protected, monkeypatch):
    install(protected)
    ambient = types.ModuleType("source_foundation")
    ambient.runtime_identity = lambda: pytest.fail("ambient source helper executed")
    monkeypatch.setitem(sys.modules, "source_foundation", ambient)
    authority = media.issue_media_binding()
    assert authority._media.module.runtime_identity is authority._source._module.runtime_identity
    assert sys.modules["source_foundation"] is ambient
    media.verify_media_binding(authority)


def test_producer_substitution_is_refused_before_execution(protected):
    install(protected)
    protected[1][-1].write_bytes(b'raise AssertionError("untrusted media producer executed")\n')
    with pytest.raises(source.SoftwareBindingError, match="sealed artifact"):
        media.issue_media_binding()


def test_media_build_must_bind_the_actual_source_audit_producer(protected):
    install(protected)
    files = protected[1]
    build = json.loads(files[2].read_bytes())
    build["producer_inputs"]["source_foundation.py"]["sha256"] = "f" * 64
    files[2].write_bytes(source._canonical(build))
    manifest = json.loads(files[0].read_bytes())
    for row in manifest["artifacts"]:
        if row["path"] == str(files[2]):
            row.update(identity(files[2]))
    files[0].write_bytes(source._canonical(manifest))
    with pytest.raises(source.SoftwareBindingError, match="actual source and media"):
        media.issue_media_binding()


def test_malformed_configuration_refuses_even_with_a_matching_artifact_hash(protected):
    install(protected)
    files = protected[1]
    files[3].write_bytes(b'{"duplicate":1,"duplicate":2}')
    manifest = json.loads(files[0].read_bytes())
    for row in manifest["artifacts"]:
        if row["path"] == str(files[3]):
            row.update(identity(files[3]))
    files[0].write_bytes(source._canonical(manifest))
    with pytest.raises(source.SoftwareBindingError, match="duplicate"):
        media.issue_media_binding()


@pytest.mark.parametrize("which", ["source", "media", "config", "package"])
def test_changed_immutable_inputs_cannot_be_verified_or_repaired(protected, which):
    data = install(protected)
    authority = media.issue_media_binding()
    path = {"source": protected[0][1], "media": protected[1][1], "config": protected[1][3], "package": data}[which]
    path.write_bytes(path.read_bytes() + b"changed")
    for operation in (media.verify_media_binding, media.reissue_media_binding):
        with pytest.raises(source.SoftwareBindingError, match="immutable.*changed"):
            operation(authority)


@pytest.mark.parametrize("runtime", ["source", "media"])
def test_runtime_drift_requires_explicit_reissue_with_same_immutable_anchor(protected, monkeypatch, runtime):
    install(protected)
    old = media.issue_media_binding()
    monkeypatch.setenv("TF_TEST_" + runtime.upper() + "_RUNTIME", "e" * 64)
    with pytest.raises(source.SoftwareBindingError, match="runtime changed"):
        media.verify_media_binding(old)
    new = media.reissue_media_binding(old)
    assert new.record()["identity_sha256"] != old.record()["identity_sha256"]
    assert new.record()["protected_media"] == old.record()["protected_media"]
    media.verify_media_binding(new)
    with pytest.raises(source.SoftwareBindingError, match="runtime changed"):
        media.verify_media_binding(old)


def test_reissue_retains_files_that_are_no_longer_mapped(protected, monkeypatch, tmp_path):
    install(protected)
    retired = tmp_path / "previously-mapped.so"
    retired.write_bytes(b"original observed process code")
    monkeypatch.setenv("TF_TEST_PROCESS_PATH", str(retired))
    old = media.issue_media_binding()
    monkeypatch.delenv("TF_TEST_PROCESS_PATH")
    replacement = media.reissue_media_binding(old)
    assert retired in replacement._states
    retired.write_bytes(b"changed retired process code")
    for operation in (media.verify_media_binding, media.reissue_media_binding):
        with pytest.raises(source.SoftwareBindingError, match="immutable.*changed"):
            operation(replacement)


def test_accumulated_anchor_has_a_bound_and_never_evicts_to_fit(protected, monkeypatch, tmp_path):
    install(protected)
    old = media.issue_media_binding()
    before = dict(old._states)
    monkeypatch.setattr(media, "_MAX_INPUTS", len(before))
    extra = tmp_path / "newly-mapped.so"
    extra.write_bytes(b"new process observation")
    monkeypatch.setenv("TF_TEST_PROCESS_PATH", str(extra))
    with pytest.raises(source.SoftwareBindingError, match="inventory exceeds"):
        media.reissue_media_binding(old)
    assert old._states == before


def test_base_runtime_change_inside_media_audit_cannot_escape_verification(protected):
    install(protected, late_source_runtime=True)
    authority = media.issue_media_binding()
    with pytest.raises(source.SoftwareBindingError, match="runtime changed"):
        media.verify_media_binding(authority)


def test_mutation_during_media_audit_is_not_issued(protected):
    data = install(protected)
    install(protected, mutate=str(data))
    with pytest.raises(source.SoftwareBindingError, match="changed during"):
        media.issue_media_binding()


def test_expected_media_refusal_is_not_source_only_success(protected):
    install(protected, fail=True)
    with pytest.raises(ValueError, match="owned media refusal"):
        media.issue_media_binding()


def test_unretired_metadata_failure_preserves_original_scope(protected, monkeypatch):
    install(protected)
    primary = OSError("unretired media metadata")
    scope = object()
    primary._tensorfold_checkpoint_file_scope = scope
    actual = source._read_protected

    def read(path):
        if path == protected[1][0]:
            raise primary
        return actual(path)

    monkeypatch.setattr(source, "_read_protected", read)
    with pytest.raises(OSError) as caught:
        media.issue_media_binding()
    assert caught.value is primary and caught.value._tensorfold_checkpoint_file_scope is scope


def test_foreign_pid_refuses_before_directories_or_audits(protected, monkeypatch):
    install(protected)
    authority = media.issue_media_binding()
    pid = os.getpid()
    monkeypatch.setattr(media.os, "getpid", lambda: pid + 1)
    monkeypatch.setattr(media, "_locations", lambda: pytest.fail("foreign PID reached protected IO"))
    for operation in (media.verify_media_binding, media.reissue_media_binding):
        with pytest.raises(source.SoftwareBindingError, match="another process"):
            operation(authority)


def install_policy_rebind(protected):
    install(protected)
    base, _files = protected
    original = base[2].read_bytes()
    history = base[0].parent / "audit-history/build-host-v1/source_foundation.py"
    history.parent.mkdir(parents=True)
    history.write_bytes(original)
    base[2].write_bytes(original + b"\n# reviewed runtime-policy audit revision\n")
    manifest = json.loads(base[0].read_bytes())
    manifest.update(version=2, runtime_policy="torch2141-build-host-v1", audit_policy_revision={
        "original_build_producer": identity(history), "original_build_producer_path": str(history),
        "current_audit_producer": identity(base[2])})
    for row in manifest["artifacts"]:
        if row["path"] == str(base[2]):
            row.update(identity(base[2]))
    manifest["artifacts"].append({"path": str(history), **identity(history)})
    base[0].write_bytes(source._canonical(manifest))
    return history


def test_policy_rebind_preserves_original_media_build_producer_role(protected):
    history = install_policy_rebind(protected)
    original_build = protected[1][2].read_bytes()
    authority = media.issue_media_binding()
    assert history in authority._source._states
    assert media.verify_media_binding(authority)["status"] == "verified-source-and-media-software"
    assert protected[1][2].read_bytes() == original_build
    history.write_bytes(b"changed original source build producer")
    with pytest.raises(source.SoftwareBindingError, match="inputs changed"):
        media.verify_media_binding(authority)


@pytest.mark.parametrize("fault", ["version", "policy", "current", "original", "outside-history", "unsealed", "malformed-revision"])
def test_policy_rebind_requires_exact_current_and_original_source_roles(protected, fault):
    history = install_policy_rebind(protected)
    path = protected[0][0]
    value = json.loads(path.read_bytes())
    revision = value["audit_policy_revision"]
    if fault == "malformed-revision":
        value["audit_policy_revision"] = None
    elif fault == "version":
        value["version"] = 1
    elif fault == "policy":
        value["runtime_policy"] = "unknown"
    elif fault == "current":
        revision["current_audit_producer"]["sha256"] = "a" * 64
    elif fault == "original":
        revision["original_build_producer"]["sha256"] = "b" * 64
    elif fault == "outside-history":
        revision["original_build_producer_path"] = str(protected[0][2])
    else:
        value["artifacts"] = [row for row in value["artifacts"] if row["path"] != str(history)]
    path.write_bytes(source._canonical(value))
    with pytest.raises(source.SoftwareBindingError):
        media.issue_media_binding()

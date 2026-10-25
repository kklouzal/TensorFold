"""Process authority controls with an owned fake audit boundary, not SDK qualification."""

import hashlib
import json
import os

import pytest

from tensorfold.cuda.sm70 import software_binding as binding


PRODUCER = b'''RUNTIME = "a" * 64
def application_nccl_identity():
    return {"path":"source-provider","version":23203}
def runtime_identity():
    return {"mapped_libraries":[],"host_driver_libraries":[]}
def audit_process(manifest,pins):
    if pins.get("fail"): raise ValueError("owned audit refusal")
    if pins.get("mutate"):
        with open(pins["mutate"],"wb") as output: output.write(b"changed during audit")
    return {"schema":"tensorfold.v100-source-foundation-process-audit","version":1,
            "software_integrity":"passed","release_qualified":False,"runtime_sha256":RUNTIME,
            "pending_gates":["compiler-6111901"],"additional_process_libraries":[],"host_driver_artifacts":[]}
'''


@pytest.fixture
def protected(tmp_path, monkeypatch):
    root = tmp_path / "protected"
    root.mkdir()
    paths = tuple(root / name for name in ("manifest.json", "pins.json", "producer.py"))
    monkeypatch.setattr(binding, "_ROOT", root)
    monkeypatch.setattr(binding, "_PROTECTED", paths)
    monkeypatch.setattr(binding, "_ADMIN_UID", os.getuid())
    # Unit fixtures intentionally live under /tmp; the actual selected-image
    # controls separately exercise the fixed administrator-owned hierarchy.
    monkeypatch.setattr(binding, "_protected_location", lambda: None)
    return paths


def install_fixture(paths, *, fail=False, mutate=None):
    manifest, pins, producer = paths
    producer.write_bytes(PRODUCER)
    data = manifest.parent / "sealed.data"
    data.write_bytes(b"sealed compiler input")
    rows = [{"path": str(path), "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in (producer, data)]
    manifest.write_text(json.dumps({"artifacts": rows}))
    pins.write_text(json.dumps({"fail": fail, "mutate": str(mutate) if mutate else None,
                               "targets": {"torch": "source"}, "specification_sha256": "b" * 64}))
    for path in (*paths, data):
        path.chmod(0o644)
    return data


def test_only_complete_absence_can_be_an_unverified_reference(protected):
    result = binding.issue_source_binding()
    assert isinstance(result, binding.ReferenceSoftwareStatus)
    assert result.record()["status"] == "unverified-reference"
    assert result.record()["release_qualified"] is False
    with pytest.raises(binding.SoftwareBindingError, match="required.*absent"):
        binding.issue_source_binding(required=True)
    with pytest.raises(binding.SoftwareBindingError, match="forged, unverified"):
        binding.verify_source_binding(result)


@pytest.mark.parametrize("count", [1, 2])
def test_partial_protected_artifacts_never_downgrade_to_reference(protected, count):
    for path in protected[:count]:
        path.write_bytes(b"partial")
        path.chmod(0o644)
    with pytest.raises(binding.SoftwareBindingError, match="incomplete"):
        binding.issue_source_binding(required=False)


@pytest.mark.parametrize("bad", [b'{"artifacts":[],"artifacts":[]}', b'{"artifacts":NaN}', b'not json'])
def test_malformed_protected_metadata_refuses(protected, bad):
    install_fixture(protected)
    protected[0].write_bytes(bad)
    with pytest.raises((binding.SoftwareBindingError, ValueError)):
        binding.issue_source_binding()


def test_producer_bytes_must_match_sealed_identity_before_execution(protected):
    install_fixture(protected)
    protected[2].write_bytes(b'raise AssertionError("untrusted producer executed")\n')
    with pytest.raises(binding.SoftwareBindingError, match="sealed artifact identity"):
        binding.issue_source_binding()


def test_audit_refusal_propagates_instead_of_issuing_reference(protected):
    install_fixture(protected, fail=True)
    with pytest.raises(ValueError, match="owned audit refusal"):
        binding.issue_source_binding()


def test_issued_record_is_software_only_and_cannot_recreate_authority(protected):
    install_fixture(protected)
    authority = binding.issue_source_binding(required=True)
    record = binding.verify_source_binding(authority)
    assert record["status"] == "verified-source-software"
    assert record["software_only"] is True and record["release_qualified"] is False
    assert "image-video-processor-stack" in record["pending_gates"]
    assert set(record["protected"]) == {path.name for path in protected}
    with pytest.raises(binding.SoftwareBindingError, match="forged"):
        binding.verify_source_binding(record)
    with pytest.raises(TypeError, match="process-issued"):
        binding.SourceSoftwareBinding(None, None, None, None, None, None, None)
    forged = object.__new__(binding.SourceSoftwareBinding)
    with pytest.raises(binding.SoftwareBindingError, match="forged"):
        binding.verify_source_binding(forged)
    record["pending_gates"].clear()
    assert authority.record()["pending_gates"]


@pytest.mark.parametrize("which", [0, 1, 2, "sealed"])
def test_changed_protected_or_immutable_inputs_invalidate_authority(protected, which):
    data = install_fixture(protected)
    authority = binding.issue_source_binding(required=True)
    path = data if which == "sealed" else protected[which]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(binding.SoftwareBindingError, match="inputs changed"):
        binding.verify_source_binding(authority)


def test_runtime_change_requires_explicit_reissue(protected):
    install_fixture(protected)
    authority = binding.issue_source_binding(required=True)
    authority._module.RUNTIME = "c" * 64
    with pytest.raises(binding.SoftwareBindingError, match="runtime changed"):
        binding.verify_source_binding(authority)


def test_verification_rechecks_protected_hierarchy(protected, monkeypatch):
    install_fixture(protected)
    authority = binding.issue_source_binding(required=True)

    def changed():
        raise binding.SoftwareBindingError("protected hierarchy is writable")

    monkeypatch.setattr(binding, "_protected_location", changed)
    with pytest.raises(binding.SoftwareBindingError, match="hierarchy is writable"):
        binding.verify_source_binding(authority)


def test_input_mutation_during_audit_cannot_be_issued(protected):
    data = install_fixture(protected)
    install_fixture(protected, mutate=data)
    with pytest.raises(binding.SoftwareBindingError, match="changed during audit"):
        binding.issue_source_binding(required=True)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="selected source runtime is native Linux")
def test_authority_cannot_cross_an_actual_process_fork(protected):
    install_fixture(protected)
    authority = binding.issue_source_binding(required=True)
    child = os.fork()
    if child == 0:
        try:
            binding.verify_source_binding(authority)
        except binding.SoftwareBindingError:
            os._exit(0)
        os._exit(1)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0


def test_failed_metadata_close_preserves_primary_and_cleanup_roots(protected, monkeypatch):
    from tensorfold import file_io

    install_fixture(protected)
    actual = protected[0].open("rb")
    primary, cleanup = ValueError("primary metadata read"), OSError("metadata close refusal")

    class Failed:
        def fileno(self):
            return actual.fileno()

        def read(self, size):
            raise primary

    class Scope:
        retired = False

        def __init__(self, **kwargs):
            self.held = actual

        def open(self, *args):
            return type("Record", (), {"stream": Failed()})(), None

        def drain(self):
            return [cleanup]

    monkeypatch.setattr(file_io, "FileStreams", Scope)
    try:
        with pytest.raises(ValueError) as caught:
            binding._read_protected(protected[0])
        assert caught.value is primary
        assert cleanup is primary.__cause__ or isinstance(primary.__cause__, BaseExceptionGroup)
        assert isinstance(primary.__dict__["_tensorfold_checkpoint_file_scope"], Scope)
    finally:
        actual.close()

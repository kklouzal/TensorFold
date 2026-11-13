"""CPU failure/retention contracts; these controls do not simulate a GPU pass."""
import hashlib
import json
import sys

import pytest

from tools import v100_gdn_native_envelope as envelope


def test_all_retirement_stages_preserve_the_native_primary():
    primary = RuntimeError("kernel failed")
    failures = [OSError("snapshot"), ValueError("cache"), PermissionError("publication")]
    calls = []

    def fail(name, error):
        def callback(*_):
            calls.append(name)
            raise error
        return callback

    with pytest.raises(RuntimeError) as caught:
        envelope.complete_trial(fail("action", primary), {},
                                after=fail("after", failures[0]),
                                retain=fail("retain", failures[1]),
                                publish=fail("publish", failures[2]))
    assert caught.value is primary
    assert calls == ["action", "after", "retain", "publish"]
    assert isinstance(primary.__cause__, BaseExceptionGroup)
    assert all(any(error is retained for retained in primary.__cause__.exceptions) for error in failures)


def test_successful_kernel_does_not_hide_a_retirement_failure():
    failure = OSError("cache not retained")
    published = []

    def retain():
        raise failure

    with pytest.raises(OSError) as caught:
        envelope.complete_trial(lambda: None, {}, after=lambda: {"ready": True},
                                retain=retain, publish=published.append)
    assert caught.value is failure
    assert published[0]["primary"] is None
    assert published[0]["cleanup"] == [{"stage": "retained_cache", "type": "OSError"}]


def test_base_exception_and_exclusive_publication_failure_are_both_retained():
    primary, publication = KeyboardInterrupt(), FileExistsError("retain original report")

    def action():
        raise primary

    def publish(_):
        raise publication

    with pytest.raises(KeyboardInterrupt) as caught:
        envelope.complete_trial(action, {}, after=dict, retain=dict, publish=publish)
    assert caught.value is primary and primary.__cause__ is publication


def test_success_publishes_both_completed_retirement_results():
    published = []
    envelope.complete_trial(lambda: None, {}, after=lambda: {"resource": 1},
                            retain=lambda: {"logical_bytes": 3}, publish=published.append)
    assert published == [{"primary": None, "cleanup": [], "after": {"resource": 1},
                          "retained_cache": {"logical_bytes": 3}}]


@pytest.mark.parametrize("code", [None, 0])
def test_zero_procedure_exit_cannot_hide_failed_retirement(code):
    previous = sys.argv
    publication = PermissionError("report cannot be published")

    def action():
        envelope.invoke_procedure(f"raise SystemExit({code!r})".encode(), "selected.py", [])

    def publish(_):
        raise publication

    with pytest.raises(RuntimeError, match="before returning") as caught:
        envelope.complete_trial(action, {}, after=dict, retain=dict, publish=publish)
    assert sys.argv is previous
    causes = caught.value.__cause__.exceptions
    assert any(isinstance(error, SystemExit) and error.code == code for error in causes)
    assert any(error is publication for error in causes)


def test_procedure_normal_return_restores_process_arguments():
    previous = sys.argv
    envelope.invoke_procedure(b"import sys; assert sys.argv == ['selected.py', '--flag']",
                              "selected.py", ["--flag"])
    assert sys.argv is previous


def test_cache_copy_retains_links_without_reading_their_targets(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.cu").write_bytes(b"abc")
    outside = tmp_path / "SDK-library"
    outside.write_bytes(b"not charged or copied through the link")
    (source / "SDK-link").symlink_to(outside)
    target = tmp_path / "retained"
    report = envelope.retain_cache(source, target, entry_limit=2, byte_limit=3)
    assert report["logical_bytes"] == 3 and report["entry_count"] == 2
    assert (target / "SDK-link").is_symlink()
    assert report["files"]["module.cu"]["sha256"] == hashlib.sha256(b"abc").hexdigest()


@pytest.mark.parametrize("entries,bytes_limit", [(1, 9), (2, 2)])
def test_cache_exhaustion_stops_before_copying(tmp_path, entries, bytes_limit):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a").write_bytes(b"ab")
    (source / "b").write_bytes(b"c")
    target = tmp_path / "retained"
    with pytest.raises(RuntimeError, match="grant exhausted"):
        envelope.retain_cache(source, target, entry_limit=entries, byte_limit=bytes_limit)
    assert not target.exists()


def test_selected_input_requires_the_complete_byte_and_component_identity(tmp_path):
    path = tmp_path / "data.json"
    data = {"schema": "tensorfold.g3-native-data-v2",
            "component_hashes": {name: "a" * 64 for name in envelope.COMPONENT_FILES}}
    raw = json.dumps(data).encode()
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    assert envelope.selected_data(path, len(raw), sha) == data
    with pytest.raises(ValueError, match="selected identity"):
        envelope.selected_data(path, len(raw), "0" * 64)
    with pytest.raises(ValueError, match="selected identity"):
        envelope.selected_data(path, len(raw) - 1, sha)
    data["component_hashes"].pop("gdn_chunk64_artifact.py")
    raw = json.dumps(data).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="four installed"):
        envelope.selected_data(path, len(raw), hashlib.sha256(raw).hexdigest())

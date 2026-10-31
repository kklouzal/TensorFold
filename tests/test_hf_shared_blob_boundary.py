"""HF shared-cache support admits managed payloads without authorizing the hub."""

import json
import os
import struct

import pytest

from tensorfold.cuda import capacity
from tensorfold.cuda.tensor_file import checkpoint_path


@pytest.fixture
def shared_model(tmp_path):
    hub = tmp_path / "hub"
    repository = hub / "models--owner--model"
    snapshot = repository / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (repository / "blobs").mkdir()
    shared = hub / "blobs"
    (shared / "ab").mkdir(parents=True)
    (shared / ".huggingface-shared-blobs").write_bytes(b"1\n")
    payload = shared / "ab" / ("ab" + "0" * 62)
    header = json.dumps({"x": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}).encode()
    payload.write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
    blob = repository / "blobs" / "etag"
    blob.symlink_to("../../blobs/ab/" + payload.name)
    (snapshot / "model.safetensors").symlink_to("../../blobs/etag")
    return snapshot, blob, payload


def test_managed_shared_payload_passes_path_and_header_admission(shared_model):
    snapshot, _, payload = shared_model
    assert checkpoint_path(snapshot, "model.safetensors") == payload
    (snapshot / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "model.safetensors"}}))
    assert capacity.headers(snapshot)["x"]["shape"] == [4]


def test_nested_snapshot_link_and_absolute_managed_blob_links_are_supported(shared_model):
    snapshot, blob, payload = shared_model
    (snapshot / "nested").mkdir()
    (snapshot / "nested" / "model.safetensors").symlink_to(blob)
    blob.unlink()
    blob.symlink_to(payload)
    assert checkpoint_path(snapshot, "nested/model.safetensors") == payload


@pytest.mark.parametrize("content", [None, b"", b"1", b"2\n", b"1\nextra", b"\xff\n"])
def test_missing_unknown_or_malformed_shared_marker_is_refused(shared_model, content):
    snapshot, _, payload = shared_model
    marker = payload.parent.parent / ".huggingface-shared-blobs"
    marker.unlink()
    if content is not None:
        marker.write_bytes(content)
    with pytest.raises(ValueError, match="authorized roots"):
        checkpoint_path(snapshot, "model.safetensors")


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo"])
def test_shared_marker_must_be_an_actual_regular_file(shared_model, tmp_path, kind):
    snapshot, _, payload = shared_model
    marker = payload.parent.parent / ".huggingface-shared-blobs"
    marker.unlink()
    if kind == "symlink":
        other = tmp_path / "other-marker"
        other.write_bytes(b"1\n")
        marker.symlink_to(other)
    elif kind == "directory":
        marker.mkdir()
    elif hasattr(os, "mkfifo"):
        os.mkfifo(marker)
    else:
        pytest.skip("POSIX FIFO fixture")
    with pytest.raises(ValueError, match="authorized roots"):
        checkpoint_path(snapshot, "model.safetensors")


@pytest.mark.parametrize("relative", ["aa/" + "ab" + "0" * 62, "ab/" + "ab" + "0" * 61,
                                     "AB/" + "AB" + "0" * 62, "ag/" + "ag" + "0" * 62,
                                     "ab/nested/" + "ab" + "0" * 62])
def test_shared_payload_requires_canonical_prefix_and_lowercase_hash(shared_model, relative):
    snapshot, blob, payload = shared_model
    target = payload.parent.parent / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload.read_bytes())
    blob.unlink()
    blob.symlink_to("../../blobs/" + relative)
    with pytest.raises(ValueError, match="authorized roots"):
        checkpoint_path(snapshot, "model.safetensors")


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo", "missing"])
def test_shared_payload_must_be_an_actual_regular_file(shared_model, tmp_path, kind):
    snapshot, _, payload = shared_model
    payload.unlink()
    if kind == "symlink":
        other = tmp_path / "other-payload"
        other.write_bytes(b"payload")
        payload.symlink_to(other)
    elif kind == "directory":
        payload.mkdir()
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("POSIX FIFO fixture")
        os.mkfifo(payload)
    with pytest.raises(ValueError, match="authorized roots"):
        checkpoint_path(snapshot, "model.safetensors")


@pytest.mark.parametrize("component", ["shared", "prefix"])
def test_shared_directory_symlinks_cannot_expand_authority(shared_model, tmp_path, component):
    snapshot, _, payload = shared_model
    source = payload.parent.parent if component == "shared" else payload.parent
    moved = tmp_path / "moved"
    source.rename(moved)
    source.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="authorized roots"):
        checkpoint_path(snapshot, "model.safetensors")


@pytest.mark.parametrize("kind", ["direct_shared", "other_repository", "unrelated_hub_file"])
def test_shared_marker_does_not_authorize_unmanaged_snapshot_targets(shared_model, kind):
    snapshot, blob, payload = shared_model
    linked = snapshot / "model.safetensors"
    linked.unlink()
    if kind == "direct_shared":
        linked.symlink_to(payload)
    elif kind == "other_repository":
        other = payload.parent.parent.parent / "models--other--model" / "blobs"
        other.mkdir(parents=True)
        (other / "etag").symlink_to(payload)
        linked.symlink_to(other / "etag")
    else:
        unrelated = payload.parent.parent.parent / "config.json"
        unrelated.write_text("{}")
        blob.unlink()
        blob.symlink_to(unrelated)
        linked.symlink_to(blob)
    with pytest.raises(ValueError, match="authorized roots"):
        checkpoint_path(snapshot, "model.safetensors")


def test_shared_authority_is_revalidated_after_marker_or_link_changes(shared_model, tmp_path):
    snapshot, blob, payload = shared_model
    assert checkpoint_path(snapshot, "model.safetensors") == payload
    marker = payload.parent.parent / ".huggingface-shared-blobs"
    marker.write_bytes(b"2\n")
    with pytest.raises(ValueError, match="authorized roots"):
        checkpoint_path(snapshot, "model.safetensors")
    marker.write_bytes(b"1\n")
    assert checkpoint_path(snapshot, "model.safetensors") == payload
    unrelated = tmp_path / "unrelated"
    unrelated.write_bytes(b"payload")
    blob.unlink()
    blob.symlink_to(unrelated)
    with pytest.raises(ValueError, match="authorized roots"):
        checkpoint_path(snapshot, "model.safetensors")


@pytest.mark.parametrize("failure", [PermissionError("marker access denied"), OSError(5, "marker IO failed")])
def test_marker_access_and_io_failures_keep_their_primary_exception(shared_model, monkeypatch, failure):
    import tensorfold.cuda.tensor_file as files

    snapshot, _, _ = shared_model

    def broken_stream(path):
        raise failure

    monkeypatch.setattr(files, "_regular_stream", broken_stream)
    with pytest.raises(type(failure)) as caught:
        checkpoint_path(snapshot, "model.safetensors")
    assert caught.value is failure


def test_marker_retirement_failure_preserves_owned_scope_context(shared_model, monkeypatch):
    from contextlib import contextmanager

    import tensorfold.cuda.tensor_file as files

    snapshot, _, _ = shared_model
    failure = OSError(5, "marker retirement failed")
    owner = object()
    failure._tensorfold_checkpoint_file_scope = owner

    @contextmanager
    def broken_retirement(path):
        with open(path, "rb") as stream:
            yield stream
        raise failure

    monkeypatch.setattr(files, "_regular_stream", broken_retirement)
    with pytest.raises(OSError) as caught:
        checkpoint_path(snapshot, "model.safetensors")
    assert caught.value is failure
    assert caught.value._tensorfold_checkpoint_file_scope is owner

"""Immutable standard item dependencies and the schema-compatible local video adapter."""
import base64
import hashlib
import json
import os
import threading
from types import SimpleNamespace

import pytest

from tensorfold.server.api_records import ApiRecordStore, RecordCapacity, RecordNotFound
from tensorfold.server.api_record_media import video_capable
from tensorfold.server.errors import RequestError
from tensorfold.server.responses_translate import messages, translate
from tensorfold.vision.images import ImageLimits, split_images
from tensorfold.vision.media_frontend import MediaExecutor
from tensorfold.vision.replay import ReplayLease, ReplayStore
from tensorfold.vision.videos import VideoSource
from tests.test_api_records import begin, limits, response, store as _store
from tests.test_media_replay import envelope, video_source

store = _store


def remote(store):
    return store.normalize_items("alice", [{"role": "user", "content": [
        {"type": "input_text", "text": "original"},
        {"type": "input_image", "image_url": "https://example.com/image.png", "detail": "high"}]}])


def generic_app():
    return SimpleNamespace(vision=SimpleNamespace(allow_urls=True),
        image_limits=ImageLimits(max_encoded_bytes=2048, max_total_encoded_bytes=4096))


def test_remote_capture_preserves_wire_and_rebuild_never_refetches_mutated_url(store, monkeypatch):
    calls = []
    raw = b"immutable-encoded-image-bytes"
    def fetch(url, **limits):
        calls.append((url, limits))
        return raw, "image/png"
    monkeypatch.setattr("tensorfold.vision.images_http.fetch_image", fetch)
    original = remote(store)
    with begin(store, added=original) as operation:
        operation.capture_media(generic_app())
        encoded = operation.execution_input()[0]["content"][1]["image_url"]
        assert base64.b64decode(encoded.partition(",")[2]) == raw
        assert operation.added[0]["content"][1]["image_url"] == "https://example.com/image.png"
        operation.commit(response("resp_one"))
    assert len(calls) == 1 and calls[0][1]["max_bytes"] <= 2048
    assert store.response_items("alice", "resp_one")["data"][0]["content"][1]["image_url"] == "https://example.com/image.png"
    monkeypatch.setattr("tensorfold.vision.images_http.fetch_image", lambda *_a, **_kw: pytest.fail("old URL refetched"))
    with begin(store, "resp_branch", parent="resp_one") as operation:
        assert operation.execution_history[0]["content"][1]["image_url"] == encoded
        operation.commit(response("resp_branch"))
    store.delete("resp_one", principal="alice")
    with begin(store, "resp_next", added=[{"type": "item_reference", "id": original[0]["id"]}]) as operation:
        operation.capture_media(generic_app())
        assert operation.execution_input()[-1]["content"][1]["image_url"] == encoded


def test_bounded_capture_reuses_actual_stage_and_retires_replay_on_conversion_failure(store, tmp_path, monkeypatch):
    directory = tmp_path / "replays"
    directory.mkdir(mode=0o700)
    replay = ReplayStore(directory, envelope())
    raw = b"worker-owned-encoded-input"
    calls = []
    def staged(self, record, output, cancelled, **kwargs):
        calls.append(record)
        os.write(output.fileno(), raw)
        return {"sha256": hashlib.sha256(raw).hexdigest(), "byte_length": len(raw)}
    monkeypatch.setattr(MediaExecutor, "_run", staged)
    app = SimpleNamespace(vision=SimpleNamespace(is_media_runtime=True, allow_urls=True, videos=True,
        media_envelope=replay.envelope, store=replay, frontend=SimpleNamespace(cancelled=lambda: False),
        check_media_health=replay._check_owner))
    with begin(store, added=remote(store)) as operation:
        operation.capture_media(app)
        assert base64.b64decode(operation.execution_input()[0]["content"][1]["image_url"].partition(",")[2]) == raw
        assert replay._bytes == 0 and replay._leases == 0 and not list(directory.iterdir())
    assert len(calls) == 1 and calls[0]["operation"] == "stage"
    with begin(store, added=remote(store)) as operation:
        operation.byte_reservation = 1
        with pytest.raises(RecordCapacity):
            operation.capture_media(app)
    assert replay._bytes == 0 and replay._leases == 0 and not list(directory.iterdir())
    primary = MemoryError("after acquired replay, before first item")
    interrupted = False
    def fail_refs(_lease):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise primary
        return _lease._refs
    monkeypatch.setattr(ReplayLease, "refs", property(fail_refs))
    with begin(store, added=remote(store)) as operation:
        with pytest.raises(MemoryError) as caught:
            operation.capture_media(app)
    assert caught.value is primary
    assert replay._bytes == 0 and replay._leases == 0 and not list(directory.iterdir())


def test_inline_video_file_uses_identical_internal_video_source_and_declared_capability(store):
    source = video_source()
    standard = {"role": "user", "content": [{"type": "input_text", "text": "before"},
        {"type": "input_file", "filename": "clip.mkv", "file_data": source.url},
        {"type": "input_text", "text": "after"}]}
    configured = SimpleNamespace(vision=SimpleNamespace(is_media_runtime=True, videos=True, media_envelope=envelope()))
    assert video_capable(configured)
    request = translate({"input": [standard]}, store, principal="alice", history=[], allow_video=True)
    template, sources = split_images(request.chat["messages"], allow_videos=True)
    assert type(sources[0]) is VideoSource and sources[0].url == source.url
    expected = [{"role": "user", "content": [{"type": "text", "text": "before"},
                {"type": "video_url", "video_url": {"url": source.url}}, {"type": "text", "text": "after"}]}]
    assert (template, sources) == split_images(expected, allow_videos=True)
    assert "before" in json.dumps(template) and "after" in json.dumps(template)
    with begin(store, added=[standard]) as operation:
        operation.capture_media(configured)
        assert messages(list(operation.execution_input()), allow_video=True) == request.chat["messages"]
        assert operation.added[0]["content"][1]["type"] == "input_file"
    with pytest.raises(RequestError, match="input_file"):
        translate({"input": [standard]}, store, principal="alice", history=[])
    with begin(store, added=[standard]) as operation:
        with pytest.raises(RequestError, match="SM70"):
            operation.capture_media(generic_app())


def test_mutation_capture_reserves_and_publishes_snapshot_independently_of_object_lifetime(store, monkeypatch):
    monkeypatch.setattr("tensorfold.vision.images_http.fetch_image", lambda *_a, **_kw: (b"snapshot", "image/png"))
    values = remote(store)
    with store.begin_mutation("alice", values, create=True) as operation:
        operation.capture_media(generic_app())
        conversation = store.create_conversation("alice", metadata={}, items=values, operation=operation)
    store.delete_conversation("alice", conversation["id"])
    reference = [{"type": "item_reference", "id": values[0]["id"]}]
    with begin(store, added=reference) as operation:
        operation.capture_media(generic_app())
        assert operation.execution_input()[0]["content"][1]["image_url"].startswith("data:")
    assert store.get_item("alice", values[0]["id"])["content"][1]["image_url"].startswith("https:")


def test_accepted_added_reference_pins_snapshot_until_capture_and_final_owner_retirement(store, monkeypatch):
    calls = []
    raw = b"original immutable media"
    def fetch(*args, **kwargs):
        calls.append(1)
        return (raw if len(calls) == 1 else b"changed URL"), "image/png"
    monkeypatch.setattr("tensorfold.vision.images_http.fetch_image", fetch)
    values = remote(store)
    with store.mutation("alice", values, create=True) as operation:
        operation.capture_media(generic_app())
        expected = operation.execution_input()[0]["content"][1]["image_url"]
        conversation = store.create_conversation("alice", metadata={}, items=values, operation=operation)
    with begin(store, "resp_child", added=[{"type": "item_reference", "id": values[0]["id"]}]) as operation:
        store.delete_item("alice", conversation["id"], values[0]["id"])
        operation.capture_media(generic_app())
        assert operation.execution_input()[0]["content"][1]["image_url"] == expected
        assert base64.b64decode(expected.partition(",")[2]) == raw
    assert len(calls) == 1
    with pytest.raises(RequestError, match="unavailable"):
        store.get_item("alice", values[0]["id"])


@pytest.mark.parametrize("mutation", [False, True])
def test_reference_resolution_and_lease_publication_exclude_last_dependency_deletion(store, monkeypatch, mutation):
    calls = []
    def fetch(*args, **kwargs):
        calls.append(1)
        return (b"original" if len(calls) == 1 else b"changed"), "image/png"
    monkeypatch.setattr("tensorfold.vision.images_http.fetch_image", fetch)
    values = remote(store)
    with store.mutation("alice", values, create=True) as setup:
        setup.capture_media(generic_app())
        expected = setup.execution_input()[0]["content"][1]["image_url"]
        conversation = store.create_conversation("alice", metadata={}, items=setup.added, operation=setup)["id"]
    reference = [{"type": "item_reference", "id": values[0]["id"]}]
    translated = translate({"input": reference}, store, principal="alice", history=[])
    assert translated.items == reference  # preserve reference semantics until atomic admission
    attempted, deleted, errors = threading.Event(), threading.Event(), []
    def delete_last_link():
        acquired = store._lock.acquire(blocking=False)
        try:
            if acquired:
                store.delete_item("alice", conversation, values[0]["id"])
            attempted.set()
            if not acquired:
                store.delete_item("alice", conversation, values[0]["id"])
        except BaseException as error:
            errors.append(error)
        finally:
            if acquired:
                store._lock.release()
            attempted.set()
            deleted.set()
    worker = threading.Thread(target=delete_last_link)
    original = store.normalize_items
    started = False
    def resolve(*args, **kwargs):
        nonlocal started
        result = original(*args, **kwargs)
        if not started:
            started = True
            worker.start()
            assert attempted.wait(5)
        return result
    monkeypatch.setattr(store, "normalize_items", resolve)
    scope = (store.mutation("alice", reference, create=True) if mutation else
             store.response_operation("alice", "resp_new", previous_response_id=None, conversation_id=None,
                                      added=reference, stored=True))
    try:
        with scope as operation:
            assert deleted.wait(5) and not errors
            operation.capture_media(generic_app())
            assert operation.execution_input()[0]["content"][1]["image_url"] == expected
    finally:
        worker.join(5)
    assert not worker.is_alive() and len(calls) == 1
    with pytest.raises(RequestError, match="unavailable"):
        store.get_item("alice", values[0]["id"])


def test_expiry_cannot_destroy_a_resolved_reference_before_claim_publication(tmp_path, monkeypatch):
    source = ApiRecordStore(tmp_path / "expiry.sqlite", limits=limits(response_retention_seconds=1.0))
    fetched = []
    def fetch(*args, **kwargs):
        fetched.append(1)
        return (b"original immutable media" if len(fetched) == 1 else b"changed URL bytes"), "image/png"
    monkeypatch.setattr("tensorfold.vision.images_http.fetch_image", fetch)
    try:
        values = remote(source)
        with begin(source, "resp_original", added=values) as operation:
            operation.capture_media(generic_app())
            expected = operation.execution_input()[0]["content"][1]["image_url"]
            operation.commit(response("resp_original"))
        expires = source._connection.execute("SELECT expires FROM objects WHERE id='resp_original'").fetchone()[0]
        ticks = 0
        def clock():
            nonlocal ticks
            ticks += 1
            return expires - .5 if ticks == 1 else expires + .5
        monkeypatch.setattr("tensorfold.server.api_records.time.time", clock)
        try:
            with source.response_operation("alice", "resp_new", previous_response_id=None, conversation_id=None,
                added=[{"type": "item_reference", "id": values[0]["id"]}], stored=True) as operation:
                operation.capture_media(generic_app())
                assert operation.execution_input()[0]["content"][1]["image_url"] == expected
        except RecordNotFound:
            pass  # Expiry before acceptance may refuse; it cannot refresh the content.
        assert len(fetched) == 1 and not source._operations
        assert source.get("resp_original", principal="alice") is None
        with pytest.raises(RecordNotFound):
            source.get_item("alice", values[0]["id"])
    finally:
        source.close()

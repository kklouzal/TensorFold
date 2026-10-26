"""Durable logical records, independent of GPU/session cache ownership."""
from dataclasses import replace
import json
import os
import time

import pytest

from tensorfold.server.api_records import (ApiRecordLimits, ApiRecordStore, RecordCapacity,
                                           RecordConflict, RecordNotFound, RecordOperation, current_operation)
from tensorfold.server.errors import RequestError


def limits(**changes):
    # Deliberate fixture capacities, not service defaults or qualification budgets.
    value = ApiRecordLimits(max_responses=20, max_conversations=10, max_items=200,
        max_total_bytes=2_000_000, max_record_bytes=80_000, max_input_items=20, max_output_items=8,
        max_history_items=100, max_history_bytes=300_000, max_active_operations=8,
        max_principal_responses=15, max_principal_conversations=8, max_principal_items=150,
        max_principal_bytes=1_000_000, max_principal_operations=4, max_materialized_bytes=1_000_000,
        max_database_bytes=4_194_304, max_operation_bytes=100_000, response_retention_seconds=None)
    return replace(value, **changes)


@pytest.fixture
def store(tmp_path):
    value = ApiRecordStore(tmp_path / "records.sqlite", limits=limits())
    try:
        yield value
    finally:
        value.close()


def item(store, text="hello", principal="alice"):
    return store.normalize_items(principal, [{"role": "user", "content": text}])[0]


def response(identifier, text="answer"):
    return {"id": identifier, "object": "response", "created_at": int(time.time()), "status": "completed",
            "output": [{"id": "msg_" + identifier, "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]}]}


def begin(store, identifier="resp_one", *, principal="alice", parent=None, conversation=None, added=None, stored=True):
    return store.begin_response(principal, identifier, previous_response_id=parent,
        conversation_id=conversation, added=[item(store, principal=principal)] if added is None else added, stored=stored)


def test_restart_durability_and_conversation_delete_preserves_independent_items(tmp_path):
    path = tmp_path / "records.sqlite"
    store = ApiRecordStore(path, limits=limits())
    original = item(store)
    conversation = store.create_conversation("alice", metadata={"title": "one"}, items=[original])
    with begin(store, conversation=conversation["id"]) as operation:
        operation.commit(response("resp_one"))
    store.close()
    assert os.stat(path).st_mode & 0o777 == 0o600
    store = ApiRecordStore(path, limits=limits())
    try:
        assert store.get_conversation("alice", conversation["id"]) == conversation
        assert len(store.list_items("alice", conversation["id"])["data"]) == 3
        assert store.delete_conversation("alice", conversation["id"])["deleted"] is True
        assert store.get_item("alice", original["id"]) == original
        with pytest.raises(RecordNotFound):
            store.conversation_item("alice", conversation["id"], original["id"])
        assert store.get("resp_one", principal="alice")["status"] == "completed"
    finally:
        store.close()


def test_branches_own_input_snapshot_after_deleted_parent_and_are_principal_scoped(store):
    with begin(store) as operation:
        operation.commit(response("resp_one"))
    with begin(store, "resp_branch", parent="resp_one") as operation:
        assert len(operation.history) == 2
        assert operation.parent_generation == 1 and operation.logical_retained
        operation.commit(response("resp_branch", "branch"))
    store.delete("resp_one", principal="alice")
    with begin(store, "resp_third", parent="resp_branch") as operation:
        assert [part["content"][0]["text"] for part in operation.history] == ["hello", "answer", "hello", "branch"]
    assert store.get("resp_branch", principal="bob") is None
    with pytest.raises(RecordNotFound):
        begin(store, principal="bob", parent="resp_branch")
    with pytest.raises(RecordNotFound):
        store.delete("resp_branch", principal="bob")


def test_store_false_and_conversation_retention_are_separate(store):
    with begin(store, stored=False) as operation:
        operation.commit(response("resp_one"))
    assert store.get("resp_one", principal="alice") is None
    assert store._usage() == (0, 0, 0, 0)
    conversation = store.create_conversation("alice", metadata=None, items=[])
    with begin(store, stored=False, conversation=conversation["id"]) as operation:
        assert operation.logical_retained
        operation.commit(response("resp_one"))
    assert store.get("resp_one", principal="alice") is None
    assert len(store.list_items("alice", conversation["id"])["data"]) == 2


def test_mutation_lease_serializes_conversation_and_resets_operation_context(store):
    conversation = store.create_conversation("alice", metadata={}, items=[])["id"]
    primary = RuntimeError("cancelled logical operation")
    with pytest.raises(RuntimeError) as caught:
        with begin(store, conversation=conversation) as operation, operation.execution_scope():
            assert current_operation() is operation
            with pytest.raises(RecordConflict):
                begin(store, "resp_other", conversation=conversation)
            for mutation in (lambda: store.update_conversation("alice", conversation, {}),
                             lambda: store.delete_conversation("alice", conversation),
                             lambda: store.add_items("alice", conversation, [])):
                with pytest.raises(RecordConflict):
                    mutation()
            with pytest.raises(RuntimeError, match="still own"):
                store.close()
            raise primary
    assert caught.value is primary and current_operation() is None and not store._operations
    assert store.update_conversation("alice", conversation, {"after": "cancel"})["metadata"] == {"after": "cancel"}
    operation.close()


def test_item_pagination_delete_and_immutable_reference_isolation(store):
    values = [item(store, str(i)) for i in range(4)]
    conversation = store.create_conversation("alice", metadata={}, items=values)["id"]
    first = store.list_items("alice", conversation, limit=2)
    assert [i["id"] for i in first["data"]] == [values[3]["id"], values[2]["id"]] and first["has_more"]
    rest = store.list_items("alice", conversation, after=first["last_id"], limit=2)
    assert [i["id"] for i in rest["data"]] == [values[1]["id"], values[0]["id"]] and not rest["has_more"]
    reference = store.normalize_items("alice", [{"type": "item_reference", "id": values[0]["id"]}])[0]
    reference["content"][0]["text"] = "caller mutated"
    assert store.get_item("alice", values[0]["id"])["content"][0]["text"] == "0"
    with pytest.raises(RecordNotFound):
        store.normalize_items("bob", [{"type": "item_reference", "id": values[0]["id"]}])
    assert store.delete_item("alice", conversation, values[1]["id"])["id"] == conversation
    with pytest.raises(RecordNotFound):
        store.get_item("alice", values[1]["id"])


def test_global_principal_and_active_reservations_release_on_failure(tmp_path):
    store = ApiRecordStore(tmp_path / "records.sqlite", limits=limits(max_principal_operations=1))
    try:
        with begin(store):
            with pytest.raises(RecordCapacity):
                begin(store, "resp_overlap")
            with begin(store, "resp_other", principal="bob"):
                pass
        with begin(store, "resp_retry") as operation:
            with pytest.raises(RecordCapacity):
                operation.commit(response("resp_retry", "x" * 80_000))
        assert store._usage() == (0, 0, 0, 0) and not store._operations
        with begin(store, "resp_retry") as operation:
            operation.commit(response("resp_retry"))
    finally:
        store.close()


def test_quota_transaction_rolls_back_and_existing_items_cannot_change(tmp_path):
    store = ApiRecordStore(tmp_path / "records.sqlite", limits=limits(max_conversations=1))
    try:
        original = item(store)
        first = store.create_conversation("alice", metadata={}, items=[original])
        usage = store._usage()
        with pytest.raises(RecordCapacity):
            store.create_conversation("alice", metadata={}, items=[item(store, "unpublished")])
        assert store._usage() == usage
        changed = {**original, "content": [{"type": "input_text", "text": "changed"}]}
        with pytest.raises(RequestError, match="cannot change"):
            store.add_items("alice", first["id"], [changed])
        assert store.get_item("alice", original["id"]) == original
    finally:
        store.close()


def test_expiry_reclaims_only_expired_response_not_descendant_snapshot(tmp_path, monkeypatch):
    store = ApiRecordStore(tmp_path / "records.sqlite", limits=limits(response_retention_seconds=10))
    now = time.time()
    try:
        with begin(store) as operation:
            operation.commit(response("resp_one"))
        monkeypatch.setattr("tensorfold.server.api_records.time.time", lambda: now + 5)
        with begin(store, "resp_child", parent="resp_one") as operation:
            operation.commit(response("resp_child"))
        monkeypatch.setattr("tensorfold.server.api_records.time.time", lambda: now + 11)
        assert store.get("resp_one", principal="alice") is None
        with begin(store, "resp_next", parent="resp_child") as operation:
            assert len(operation.history) == 4
        assert store._usage()[0] == 1
    finally:
        store.close()


def test_private_exclusive_database_owner_and_explicit_limits(tmp_path):
    path = tmp_path / "records.sqlite"
    store = ApiRecordStore(path, limits=limits())
    try:
        with pytest.raises(BlockingIOError):
            ApiRecordStore(path, limits=limits())
        record = {"schema": "tensorfold.api-record-limits", "version": 1, **limits().__dict__}
        assert ApiRecordLimits.from_record(json.loads(json.dumps(record))) == limits()
        for invalid in ([], {**record, "version": True}, {**record, "unknown": 1}, {k: v for k, v in record.items() if k != "max_items"}):
            with pytest.raises(ValueError):
                ApiRecordLimits.from_record(invalid)
        for field, value in (("max_items", 0), ("max_database_bytes", True), ("response_retention_seconds", float("inf")),
                             ("response_retention_seconds", 10**400)):
            with pytest.raises(ValueError):
                limits(**{field: value})
        with pytest.raises(RequestError):
            begin(store, parent="resp_one", conversation="conv_one")
    finally:
        store.close()
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        ApiRecordStore(path, limits=limits())


@pytest.mark.parametrize("mutation", [False, True])
def test_context_owned_acquisition_survives_failure_after_claim_publication(store, monkeypatch, mutation):
    primary = MemoryError("after logical claim publication")
    name = "begin_mutation" if mutation else "begin_response"
    original = getattr(store, name)
    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise primary
    monkeypatch.setattr(store, name, interrupted)
    scope = (store.mutation("alice", [item(store)], create=True) if mutation else
             store.response_operation("alice", "resp_one", previous_response_id=None, conversation_id=None,
                                      added=[item(store)], stored=True))
    with pytest.raises(MemoryError) as caught, scope:
        pytest.fail("unpublished operation reached caller")
    assert caught.value is primary and not store._operations and store._usage() == (0, 0, 0, 0)


def test_materialized_page_guard_is_aggregate_not_only_per_item(tmp_path):
    store = ApiRecordStore(tmp_path / "records.sqlite", limits=limits(max_materialized_bytes=3000))
    try:
        values = [item(store, "x" * 400) for _ in range(4)]
        conversation = store.create_conversation("alice", metadata={}, items=values)
        assert len(store.list_items("alice", conversation["id"], limit=1)["data"]) == 1
        with pytest.raises(RecordCapacity, match="materialized"):
            store.list_items("alice", conversation["id"], limit=4)
    finally:
        store.close()


def test_logical_catalog_notifications_use_committed_generation_and_independent_aliases(store):
    events = []
    def committed(principal, sequence, *, generation):
        events.append(("committed", principal, sequence, generation))
        if sequence.startswith("response:"):
            assert store.get(sequence.split(":", 1)[1], principal=principal) is not None
    def unavailable(principal, sequence, *, generation, reason):
        events.append((reason, principal, sequence, generation))
    store.bind_reference_listeners(unavailable=unavailable, committed=committed)
    conversation = store.create_conversation("alice", metadata={}, items=[])["id"]
    with begin(store, conversation=conversation) as operation:
        assert operation.base_generation == 0 and operation.target_generation == operation.generation == 1
        operation.commit(response("resp_one"))
    assert events == [("committed", "alice", "conversation:" + conversation, 1),
                      ("committed", "alice", "response:resp_one", 1)]
    with begin(store, "resp_abort", conversation=conversation) as operation:
        assert operation.base_generation == 1 and operation.target_generation == 2
    assert events[-2:] == [("operation_aborted", "alice", "conversation:" + conversation, 2),
                           ("operation_aborted", "alice", "response:resp_abort", 2)]
    store.delete("resp_one", principal="alice")
    assert events[-1] == ("deleted", "alice", "response:resp_one", 1)
    count = len(events)
    store.delete_conversation("alice", conversation)
    assert len(events) == count
    assert store.get_item("alice", "msg_resp_one")["role"] == "assistant"
    with pytest.raises(RuntimeError, match="once"):
        store.bind_reference_listeners(unavailable=unavailable, committed=committed)


def test_expiry_notifies_only_its_response_alias_after_durable_removal(tmp_path, monkeypatch):
    events = []
    store = ApiRecordStore(tmp_path / "records.sqlite", limits=limits(response_retention_seconds=5),
        reference_unavailable=lambda *args, **kwargs: events.append((args, kwargs)),
        reference_committed=lambda *_args, **_kwargs: None)
    now = time.time()
    try:
        with begin(store) as operation:
            operation.commit(response("resp_one"))
        monkeypatch.setattr("tensorfold.server.api_records.time.time", lambda: now + 6)
        assert store.get("resp_one", principal="alice") is None
        assert events == [(("alice", "response:resp_one"), {"generation": 1, "reason": "expired"})]
        assert store.get("resp_one", principal="alice") is None and len(events) == 1
    finally:
        store.close()


def test_failed_abort_listener_retains_original_primary_and_operation_owner(store):
    primary, cleanup = RuntimeError("generation failed"), RequestError("catalog retirement unproved")
    def failure(*args, **kwargs):
        raise cleanup
    store.bind_reference_listeners(unavailable=failure, committed=lambda *_args, **_kwargs: None)
    operation = begin(store)
    try:
        with pytest.raises(RuntimeError) as caught, operation:
            raise primary
        assert caught.value is primary and store._failure is cleanup
        assert store._operations.get(operation.token) is operation
        assert operation in primary._tensorfold_retained_owners
        with pytest.raises(RuntimeError, match="containment"):
            store.check_health()
    finally:
        store._reference_unavailable = lambda *_args, **_kwargs: None
        operation.close()


def test_item_identity_is_unique_per_collection_and_commit_converts_its_reservation(store):
    original = item(store)
    conversation = store.create_conversation("alice", metadata={}, items=[original])["id"]
    with pytest.raises(RequestError, match="only once"):
        store.add_items("alice", conversation, [original])
    assert len(store.list_items("alice", conversation)["data"]) == 1
    with begin(store, added=[original]) as operation:
        operation.commit(response("resp_one"))
        # Occupied records replace the reservation at successful publication.
        # A live publisher still counts toward the separate operation limit.
        store._check_capacity("alice", response=store.limits.max_principal_responses - store._usage("alice")[0])
        assert operation.committed and store._operations[operation.token] is operation
    with pytest.raises(RequestError, match="already present"):
        begin(store, "resp_duplicate", parent="resp_one", added=[original])
    assert not store._operations


@pytest.mark.parametrize("value", [None, 0, False, {"text": "object"}])
def test_tool_output_text_and_function_arguments_refuse_nonstring_without_coercion(store, value):
    for candidate in ({"type": "function_call_output", "call_id": "call_one",
                       "output": [{"type": "input_text", "text": value}]},
                      {"type": "function_call", "call_id": "call_one", "name": "tool", "arguments": value},
                      {"type": "reasoning", "content": [{"type": "reasoning_text", "text": value}]}):
        with pytest.raises(RequestError):
            store.normalize_items("alice", [candidate])
    assert not store._operations and store._usage() == (0, 0, 0, 0)


@pytest.mark.parametrize("tag", [False, 0, "", [], None])
def test_explicit_invalid_item_tag_is_not_the_omitted_message_shorthand(store, tag):
    with pytest.raises(RequestError, match="item type"):
        store.normalize_items("alice", [{"type": tag, "role": "user", "content": "valid text"}])
    assert not store._operations and store._usage() == (0, 0, 0, 0)
    for valid in ({"role": "user", "content": "valid text"},
                  {"type": "message", "role": "user", "content": "valid text"}):
        normalized = store.normalize_items("alice", [valid])
        assert normalized[0]["type"] == "message"
        assert normalized[0]["content"] == [{"type": "input_text", "text": "valid text"}]


def test_supplied_item_id_does_not_require_unneeded_uuid_acquisition(store, monkeypatch):
    conversation = store.create_conversation("alice", metadata={}, items=[])["id"]
    def unavailable():
        raise OSError("new UUID acquisition unavailable")
    monkeypatch.setattr("tensorfold.server.api_records.uuid.uuid4", unavailable)
    supplied = {"id": "msg_supplied/日本", "role": "user", "content": "preserved"}
    with store.mutation("alice", [supplied], conversation=conversation) as operation:
        store.add_items("alice", conversation, operation.added, operation=operation)
    persisted = store.conversation_item("alice", conversation, supplied["id"])
    assert persisted["id"] == supplied["id"]
    assert persisted["content"] == [{"type": "input_text", "text": "preserved"}]


def test_valid_empty_tool_text_arguments_and_nullable_request_flags_are_preserved(store):
    from tensorfold.server.responses_translate import messages, translate
    items = store.normalize_items("alice", [
        {"type": "function_call", "call_id": "call_one", "name": "tool", "arguments": ""},
        {"type": "function_call_output", "call_id": "call_one", "output": [
            {"type": "input_text", "text": ""}, {"type": "input_text", "text": "preserved"}]}])
    converted = messages(items)
    assert converted[0]["tool_calls"][0]["function"]["arguments"] == ""
    assert converted[1]["content"] == "preserved"
    request = translate({"input": "hello", "store": None, "stream": None}, store, principal="alice", history=[])
    assert request.store is True and request.stream is False


def test_generation_and_item_ordinals_refuse_exhaustion_before_sql_float_promotion(store):
    events = []
    store.bind_reference_listeners(unavailable=lambda *args, **kwargs: events.append((args, kwargs)),
                                  committed=lambda *args, **kwargs: events.append((args, kwargs)))
    original = item(store)
    conversation = store.create_conversation("alice", metadata={"before": "max"}, items=[original])["id"]
    maximum = (1 << 63) - 1
    store._connection.execute("UPDATE objects SET generation=? WHERE id=?", (maximum - 1, conversation))
    assert store.update_conversation("alice", conversation, {"before": "max"})["metadata"] == {"before": "max"}
    for operation in (lambda: store.update_conversation("alice", conversation, {"changed": "no"}),
                      lambda: store.add_items("alice", conversation, [item(store, "not appended")]),
                      lambda: store.delete_item("alice", conversation, original["id"]),
                      lambda: begin(store, conversation=conversation)):
        with pytest.raises(RecordCapacity, match="generation"):
            operation()
    generation, kind = store._connection.execute("SELECT generation,typeof(generation) FROM objects WHERE id=?", (conversation,)).fetchone()
    assert generation == maximum and kind == "integer" and events == []
    assert len(store.list_items("alice", conversation)["data"]) == 1
    store._connection.execute("UPDATE objects SET generation=0 WHERE id=?", (conversation,))
    store._connection.execute("UPDATE links SET ordinal=? WHERE parent=?", (maximum, conversation))
    with pytest.raises(RecordCapacity, match="ordinal"):
        store.add_items("alice", conversation, [item(store, "overflow")])
    with begin(store, conversation=conversation) as operation:
        with pytest.raises(RecordCapacity, match="ordinal"):
            operation.commit(response("resp_one"))
    assert store.get("resp_one", principal="alice") is None
    ordinal, kind = store._connection.execute("SELECT ordinal,typeof(ordinal) FROM links WHERE parent=?", (conversation,)).fetchone()
    assert ordinal == maximum and kind == "integer"
    assert store.get_item("alice", original["id"])["content"][0]["text"] == "hello"
    assert events == [(("alice", "conversation:" + conversation), {"generation": 1, "reason": "operation_aborted"}),
                      (("alice", "response:resp_one"), {"generation": 1, "reason": "operation_aborted"})]


def test_mutation_reserves_complete_canonical_items_snapshots_and_object_bytes(tmp_path):
    store = ApiRecordStore(tmp_path / "records.sqlite", limits=limits(max_operation_bytes=512))
    try:
        values = store.normalize_items("alice", [{"role": "user", "content": "é" * 150}])
        # UTF-8 wire input can fit while canonical ASCII JSON expands beyond
        # the storage reservation; refuse before creating persistent rows.
        assert len(json.dumps({"items": [{"role": "user", "content": "é" * 150}]}, ensure_ascii=False).encode()) < 512
        with pytest.raises(RecordCapacity, match="operation byte"):
            with store.mutation("alice", values, create=True):
                pytest.fail("canonical input exceeded the admitted operation")
        assert store._usage() == (0, 0, 0, 0) and not store._operations
        small = [item(store, "x" * 150)]
        with pytest.raises(RecordCapacity, match="operation byte"):
            with store.mutation("alice", small, create=True) as operation:
                store.create_conversation("alice", metadata={"large": "y" * 250}, items=small, operation=operation)
        assert store._usage() == (0, 0, 0, 0)
        conversation = store.create_conversation("alice", metadata={}, items=[])["id"]
        usage = store._usage()
        with pytest.raises(RecordCapacity, match="operation byte"):
            store.update_conversation("alice", conversation, {"value": "é" * 150})
        assert store._usage() == usage and store.get_conversation("alice", conversation)["metadata"] == {}
        with pytest.raises(RecordCapacity, match="operation byte"):
            with store.mutation("alice", small, conversation=conversation) as operation:
                operation.snapshots[small[0]["id"]] = {**small[0], "content": [{"type": "input_text", "text": "s" * 400}]}
                store.add_items("alice", conversation, small, operation=operation)
        assert store._usage() == usage and not store.list_items("alice", conversation)["data"]
    finally:
        store.close()


def test_unstored_conversation_abort_notifies_but_read_only_snapshot_has_no_compute_target(store):
    events = []
    store.bind_reference_listeners(unavailable=lambda *args, **kwargs: events.append((args, kwargs)),
                                  committed=lambda *args, **kwargs: events.append((args, kwargs)))
    conversation = store.create_conversation("alice", metadata={}, items=[])["id"]
    with begin(store, stored=False, conversation=conversation):
        pass
    assert events == [(("alice", "conversation:" + conversation), {"generation": 1, "reason": "operation_aborted"})]
    events.clear()
    with store.response_operation("alice", "resp_count", previous_response_id=None, conversation_id=conversation,
                                   added=[], stored=False, read_only=True) as counting:
        assert counting.target_generation is None and counting.base_generation == 1
        assert not counting.logical_retained
        with begin(store, conversation=conversation) as generated:
            generated.commit(response("resp_one"))
        with pytest.raises(RuntimeError, match="read-only"):
            counting.commit(response("resp_count"))
    assert all(event[1].get("reason") != "operation_aborted" for event in events)
    assert len(events) == 2 and not store._operations


@pytest.mark.parametrize("after_retirement", [False, True])
def test_context_cleanup_failure_contains_only_an_authoritatively_held_claim(store, monkeypatch, after_retirement):
    failure = MemoryError("controlled operation cleanup boundary")
    original = RecordOperation.close
    def close(operation):
        if after_retirement:
            original(operation)
        raise failure
    monkeypatch.setattr(RecordOperation, "close", close)
    operation = None
    try:
        with pytest.raises(MemoryError) as caught:
            with store.response_operation("alice", "resp_one", previous_response_id=None, conversation_id=None,
                                           added=[item(store)], stored=True) as operation:
                pass
        assert caught.value is failure
        assert (store._operations.get(operation.token) is operation) is not after_retirement
        assert (store._failure is failure) is not after_retirement
    finally:
        monkeypatch.setattr(RecordOperation, "close", original)
        if operation is not None:
            original(operation)


def test_failed_response_is_durable_but_abort_generation_is_never_reused(tmp_path):
    path = tmp_path / "records.sqlite"
    store = ApiRecordStore(path, limits=limits())
    try:
        conversation = store.create_conversation("alice", metadata={}, items=[])["id"]
        with begin(store, conversation=conversation) as operation:
            failed = {**response("resp_one", "partial"), "status": "failed", "error": {"code": "server_error", "message": "model failed"}}
            operation.fail(failed)
            assert operation.published and not operation.committed
        with begin(store, "resp_next", conversation=conversation) as operation:
            assert operation.base_generation == 1 and operation.target_generation == 2
        assert store._connection.execute("SELECT generation FROM objects WHERE id=?", (conversation,)).fetchone()[0] == 2
    finally:
        store.close()
    store = ApiRecordStore(path, limits=limits())
    try:
        assert store.get("resp_one", principal="alice") == failed
        assert not store.list_items("alice", conversation)["data"]
        with begin(store, "resp_branch", parent="resp_one") as operation:
            assert operation.history[-1]["content"][0]["text"] == "partial"
    finally:
        store.close()


def test_unpaired_surrogate_identifiers_refuse_before_acceptance(store):
    from tensorfold.server.responses_translate import translate
    with pytest.raises(RequestError, match="UTF-8"):
        store.normalize_items("alice", [{"id": "msg_\ud800", "role": "user", "content": "text"}])
    for field in ("conversation", "previous_response_id"):
        with pytest.raises(RequestError, match="UTF-8"):
            translate({"input": "text", field: "\ud800"}, store, principal="alice", history=[])

# ruff: noqa: E402 - optional Torch must be checked before backend imports.
"""Actual CPU delivery/detok/stop owners across real RAM state transfer.

The optional native xgrammar plugin is an opaque constructor boundary here;
this verifies ownership/no-reset, not native grammar execution or CUDA fences.
"""

import threading
from types import SimpleNamespace as NS

import pytest
torch = pytest.importorskip("torch")
from tokenizers import Tokenizer, models

from tensorfold.cuda.output_mailbox import OutputMailbox
from tensorfold.cuda.scheduler import DeliveryOwner
from tensorfold.cuda.streams import Stream
from tensorfold.engine.grammar import Constraint
from tensorfold.server.cancellation import RequestCancelled
from tensorfold.server.request_limits import RequestLimit
from tensorfold.server.stopping import StopPolicy
from tensorfold.server.text import IncrementalText
from tensorfold.families.qwen4_exp.cuda.retained_host import RetainedHost, RetentionIntent
from tensorfold.families.qwen4_exp.cuda.retained_catalog import LogicalReference, RetentionMismatch
from test_retained_catalog import owner

pytest_plugins = ("test_state_retention",)


@pytest.mark.parametrize("delivered_before", [False, True], ids=["undelivered-prefix", "delivered-prefix"])
def test_actual_cpu_delivery_and_matcher_owner_survive_ram_without_reemit(materialized, delivered_before, monkeypatch):
    catalog, future, _wake = owner(materialized)
    limit, box, cancelled = RequestLimit(1), OutputMailbox(8), threading.Event()
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "STOP": 2, "world": 3}, unk_token="[UNK]"))
    lock = threading.RLock()
    text, stops = IncrementalText(tok, lock), StopPolicy({"stop": "STOP"}, tok, lock, frozenset())
    observed = []
    def emit(values):
        observed.append(stops.visible(text.extend(values), partial=True))
        return stops(text.tokens)
    native = object()  # Library-owned opaque state; never copied/reset/inspected by the adapter.
    constraint = Constraint(NS(GrammarMatcher=lambda _compiled: native), object(), 4)
    constraint._shifts[torch.device("cpu")] = torch.arange(8, dtype=torch.uint8)
    constraint._pinned = torch.empty((8, 1), dtype=torch.int32)
    monkeypatch.setattr(constraint, "fresh", lambda: pytest.fail("retained matcher was reset"))
    monkeypatch.setattr(constraint, "advance", lambda *_args: pytest.fail("retention advanced grammar/RNG"))
    stream = Stream([3, 7, 11], 8, draft=False, st=materialized, constraint=constraint, cancel=cancelled)
    stream.out, stream.context = [1], [3, 7, 11, 1]
    stream.committed_output_tokens = 1
    stream.emit = lambda values: box.put(("tokens", values))
    host = request = handle = None
    try:
        with limit.request():
            limit.borrow(box)
            stream.delivery_owner = DeliveryOwner(limit, box, cancelled, emit, None)
            box.put(("tokens", [1]))
            if delivered_before:
                kind, values = box.get_nowait()
                assert kind == "tokens"
                emit(values)
            before = box.record()
            host = RetainedHost.capture(stream, leader=True)
            assert stream.constraint is constraint and constraint.m is native
            assert constraint._shifts == {} and constraint._pinned is constraint._landed is None
            reference = LogicalReference("local-unauthenticated", "response:host", 1)
            handle = catalog.reserve(reference, future_bytes=future)
            catalog.capture_idle(stream, handle=handle, host=host)
            host.detach(materialized)
            assert stream.st is None
            transfer = catalog.suspend(handle)
            assert transfer.retired.wait(5) and catalog.saved(handle), transfer.failure
            destination = catalog.states[1]
            destination.begin_sequence_generation()
            destination.set_pos(0)
            request = catalog.borrow(handle)
            request.claim(destination, stream=stream)
            transfer = handle.transfer
            assert transfer.retired.wait(5)
            handoff = catalog.restored(request)
            resumed, fill = host.attach(destination, handoff=handoff)
            assert resumed is stream and fill is None and resumed.constraint.m is native
            assert box.record() == before  # Capture/restore neither dequeues nor republishes output.
            if not delivered_before:
                kind, values = box.get_nowait()
                assert kind == "tokens" and values == [1]
                emit(values)
            for token in (3, 2):
                stream.emit([token])
                kind, values = box.get_nowait()
                assert kind == "tokens"
                emit(values)
            reference_text = IncrementalText(tok, lock)
            expected = [stops.visible(reference_text.extend([token]), partial=True) for token in (1, 3, 2)]
            assert observed == expected and stops(text.tokens)
            assert box.record()["published_token_ids"] == box.record()["consumer_dequeued_token_ids"] == 3
            assert stream.out == [1] and stream.committed_output_tokens == 1  # No synthetic generation/RNG bookkeeping.
            limit.release(box)
            with pytest.raises(RequestCancelled):
                stream.delivery_owner.require_live()
    finally:
        if request is not None:
            request.close()
        if handle is not None:
            catalog.release_owner(handle)
        catalog.close(timeout=5)
        limit.release(box)
        if host is not None:
            host.close()


def test_retained_host_rejects_missing_delivery_and_foreign_handoff(materialized):
    stream = Stream([3, 7, 11], 8, draft=False, st=materialized, cancel=threading.Event())
    with pytest.raises(RetentionMismatch):
        RetainedHost.capture(stream, leader=True)
    host = RetainedHost.capture(stream, leader=False)
    host.detach(materialized)
    with pytest.raises(RetentionMismatch, match="actual completed catalog handoff"):
        host.attach(materialized, handoff={})
    host.close()


def test_pending_backend_replacement_requires_actual_worker_retirement(materialized):
    limit, box, cancelled = RequestLimit(1), OutputMailbox(8), threading.Event()
    stream = Stream([3, 7, 11], 8, draft=False, st=materialized, cancel=cancelled)
    with limit.request():
        limit.borrow(box)
        stream.delivery_owner = DeliveryOwner(limit, box, cancelled, lambda _tokens: None, None)
        host = RetainedHost.capture(stream, leader=True)
        cancelled.set()  # A cancellation request alone is never completion proof.
        with pytest.raises(RuntimeError, match="still owns"):
            host.retire_backend()
        assert host.stream is stream and stream.st is materialized
        limit.release(box)
        host.retire_backend()
        assert host.closed and host.stream is None and stream.st is None


def test_typed_retention_intent_roundtrip_has_only_copied_standard_ids():
    target = LogicalReference("trusted", "conversation:one", 8)
    parent = LogicalReference("trusted", "conversation:one", 7)
    alias = LogicalReference("trusted", "response:resp_one", 8)
    intent = RetentionIntent(target, parent, (alias,), "resp_one")
    restored = RetentionIntent.from_record(intent.record())
    assert restored == intent and set(restored.__dict__) == {"target", "parent", "aliases", "operation"}
    bad = intent.record()
    bad["parent"][0] = "foreign"
    with pytest.raises(ValueError, match="cross authenticated"):
        RetentionIntent.from_record(bad)


def test_actual_api_operation_context_is_copied_before_generation_without_live_db_owner(tmp_path):
    from tensorfold.server.api_records import ApiRecordStore
    from test_api_records import begin, limits, response

    store = ApiRecordStore(tmp_path / "logical.sqlite", limits=limits())
    try:
        with begin(store, "resp_parent") as operation:
            operation.commit(response("resp_parent"))
        with begin(store, "resp_child", parent="resp_parent") as operation, operation.execution_scope():
            first, second = RetentionIntent.current(), RetentionIntent.current()
            assert first == second  # Thinking-budget backend subruns retain one pending logical identity.
            assert first.parent == LogicalReference("alice", "response:resp_parent", 1)
            assert first.target == LogicalReference("alice", "response:resp_child", 1)
            operation.commit(response("resp_child"))
        conversation = store.create_conversation("alice", metadata={}, items=[])["id"]
        with begin(store, "resp_conv_one", conversation=conversation) as operation:
            operation.commit(response("resp_conv_one"))
        with begin(store, "resp_conv_two", conversation=conversation) as operation, operation.execution_scope():
            intent = RetentionIntent.current()
            assert intent.parent == LogicalReference("alice", "conversation:" + conversation, 1)
            assert intent.target == LogicalReference("alice", "conversation:" + conversation, 2)
            assert intent.aliases == (LogicalReference("alice", "response:resp_conv_two", 2),)
            operation.commit(response("resp_conv_two"))
        with begin(store, "resp_disposable", parent="resp_parent", stored=False) as operation, operation.execution_scope():
            disposable = RetentionIntent.current()
            assert disposable.target is None and disposable.parent == first.parent
            operation.commit(response("resp_disposable"))
    finally:
        store.close()
    assert RetentionIntent.current() is None
    assert RetentionIntent.from_record(intent.record()) == intent


def test_host_detaches_only_after_gpu_snapshot_owners_are_excluded_and_keeps_actual_cpu_copy_index(materialized):
    from tensorfold.families.qwen3_5.cuda.decode import CopyIndex

    stream = Stream([3, 7, 11], 8, draft=False, st=materialized, cancel=threading.Event())
    stream.snap = {"opaque_gpu_snapshot": materialized.rec}
    with pytest.raises(RetentionMismatch, match="unsupported opaque"):
        RetainedHost.capture(stream, leader=False)
    stream.snap = None
    stream.copies = object()
    with pytest.raises(RetentionMismatch, match="actual bounded CPU"):
        RetainedHost.capture(stream, leader=False)
    stream.copies = CopyIndex(1)
    stream.copies.update([1, 2, 1])
    index = stream.copies
    host = RetainedHost.capture(stream, leader=False)
    host.detach(materialized)
    assert stream.st is stream.snap is None and stream.copies is index
    assert host._source() is materialized  # Weak identity only; catalog owns actual State lifetime.
    host.close()


def test_completed_tp_capture_uses_worker_proof_not_concurrent_raw_disconnect(materialized):
    from tensorfold.families.qwen4_exp.cuda.tp_scheduler import FrozenCancellation

    limit, box, raw = RequestLimit(1), OutputMailbox(8), threading.Event()
    frozen = FrozenCancellation(raw)
    frozen.active = True
    stream = Stream([3, 7, 11], 1, st=materialized, cancel=frozen, done=True)
    with limit.request():
        limit.borrow(box)
        stream.delivery_owner = DeliveryOwner(limit, box, raw, lambda _tokens: None, None)
        raw.set()  # The round already froze False identically on every rank.
        host = RetainedHost.capture(stream, leader=True, completed=True)
        assert host.stream is stream
        limit.release(box)
        host.retire_backend()
        assert host.closed

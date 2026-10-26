# ruff: noqa: E402 - optional Torch must be checked before backend imports.
"""Actual live CPU State/host transfer; no V100/model-quality qualification."""

import threading
from types import SimpleNamespace as NS

import pytest
torch = pytest.importorskip("torch")

from tensorfold.cuda.output_mailbox import OutputMailbox
from tensorfold.cuda.scheduler import DeliveryOwner
from tensorfold.cuda.streams import Stream
from tensorfold.engine.grammar import Constraint
from tensorfold.families.qwen4_exp.cuda.multi import _slot
from tensorfold.families.qwen4_exp.cuda.tp_scheduler import FrozenCancellation
from tensorfold.server.request_limits import RequestLimit
from test_retained_multi_caller import decoder_for, owner

pytest_plugins = ("test_state_retention",)


@pytest.mark.parametrize("phase", ["partial-prefill", "output-blocked"])
def test_actual_live_snapshot_restores_same_cpu_frontier_and_different_slot(materialized, monkeypatch, phase):
    catalog, _future, _wake = owner(materialized)
    catalog.configure_admission(max_waiting_input_bytes=4096, max_pending_requests=2)
    decoder = decoder_for(catalog, monkeypatch)
    decoder.passed, decoder.failed_slots = {}, []
    source = materialized
    raw, limit, box = threading.Event(), RequestLimit(2), OutputMailbox(4, pending_tokens=1)
    cancelled = FrozenCancellation(raw)
    cancelled.active = True
    filling = phase == "partial-prefill"
    stream = Stream([3, 7, 11, 9, 5] if filling else [3, 7, 11], 2, draft=False,
                    st=source, cancel=cancelled, sid=7)
    stream.principal, stream.admission_ticket = None, 11
    stream.blocked_input = "client_input" if filling else "output"
    native = object()
    grammar = Constraint(NS(GrammarMatcher=lambda _compiled: native), object(), 512)
    stream.constraint = grammar
    monkeypatch.setattr(grammar, "fresh", lambda: pytest.fail("live resume reset its native matcher"))
    monkeypatch.setattr(grammar, "advance", lambda *_args: pytest.fail("retention advanced grammar/RNG/output"))
    token_chunk = [9]
    if filling:
        engine = _slot(decoder.w, source, decoder.buf, decoder.mbuf, decoder.pbuf, 8, 4)
        engine.stops = [4]
        decoder.fills[stream.sid] = [engine, False, 3, [], None]
        decoder.filling.append(stream)
    else:
        stream.out, stream.context = [9], [3, 7, 11, 9]
        stream.committed_output_tokens = 1
        decoder.streams[stream.sid] = stream
        box.put(("tokens", token_chunk))
        assert box.get_nowait() == ("tokens", token_chunk)
        assert box.reserve(1) is None  # Actual callback still owns the only production credit.
    before = source.rec[source.cur[0]].clone(), source.conv.clone(), source.ple_tail.clone()
    handle = None
    try:
        with limit.request():
            limit.borrow(box)
            stream.delivery_owner = DeliveryOwner(limit, box, raw, lambda _values: None, None)
            output_before = box.record()
            assert decoder.suspend_request(stream)
            handle, host = stream.retention_handle, stream.retained_host
            assert handle.role == "temporary" and not handle.owners
            assert stream.st is None and stream.sid not in decoder.streams and not decoder.filling
            assert host.stream is stream and (host.fill is not None) is filling
            assert handle.transfer.retired.wait(5)
            catalog.progress()
            decoder._release_retained_states()
            decoder.free.reverse()  # Relocate into the other actual admitted State.
            assert decoder.admit(stream) is False
            destination = stream.retention_window
            assert destination is catalog.states[1]
            assert stream.retention_request.restore_handle.transfer.retired.wait(5)
            catalog.progress()
            assert decoder.admit(stream) is None
            assert stream.st is destination and stream.retained_host is None
            assert stream.admission_ticket == 11 and stream.constraint is grammar and grammar.m is native
            assert box.record() == output_before
            assert torch.equal(destination.rec[destination.cur[0]], before[0])
            assert torch.equal(destination.conv, before[1]) and torch.equal(destination.ple_tail, before[2])
            if filling:
                assert decoder.filling == [stream] and decoder.fills[stream.sid][2] == 3
                assert decoder.fills[stream.sid][0].stops == [4]
                decoder.filling.clear()
                decoder.fills.clear()
            else:
                assert decoder.streams[stream.sid] is stream and stream.out == [9] and stream.context == [3, 7, 11, 9]
                box.complete_delivery(token_chunk)
                lease = box.reserve(1)
                assert lease is not None
                lease.close()
                decoder.streams.clear()
            stream.done = True
            assert decoder.retire_retained_request(stream)
            limit.release(box)
            stream.retirement_host.retire_backend()
            assert stream.retirement_host.closed
        assert not catalog.has_retained
    finally:
        limit.release(box)
        if handle is not None and handle.internal_owner:
            if stream.retention_request is not None:
                stream.retention_request.close()
            catalog.release_temporary(handle)
        catalog.close(timeout=5)


@pytest.mark.parametrize("role", ["temporary", "logical"])
@pytest.mark.parametrize("ending", ["cancel", "drop"])
def test_aborted_live_backend_retires_reset_target_once_and_preserves_readonly_parent(materialized, monkeypatch, role, ending):
    from tensorfold.families.qwen4_exp.cuda.retained_catalog import LOCAL_PRINCIPAL, LogicalReference
    from tensorfold.families.qwen4_exp.cuda.retained_host import RetentionIntent

    catalog, future, _wake = owner(materialized)
    catalog.configure_admission(max_waiting_input_bytes=4096, max_pending_requests=2)
    decoder = decoder_for(catalog, monkeypatch)
    decoder.passed, decoder.failed_slots = {}, []
    parent_state = catalog.states[1]
    parent_ref = LogicalReference(LOCAL_PRINCIPAL, "response:protected", 1)
    parent = catalog.reserve(parent_ref, future_bytes=future)
    catalog.capture_idle(Stream([3, 7, 11, 9], 2, draft=False, st=parent_state), handle=parent)
    catalog.logical_reference_committed(parent_ref.principal, parent_ref.sequence, generation=1)
    catalog.progress()
    parent_bytes = parent_state.rec.clone()
    decoder.free.clear()  # Both actual State claims are owned: active request and protected parent.
    raw, limit, box = threading.Event(), RequestLimit(2), OutputMailbox(2)
    cancelled = FrozenCancellation(raw)
    cancelled.active = True
    stream = Stream([3, 7, 11, 9, 5], 2, draft=False, st=materialized, sid=5, cancel=cancelled)
    stream.principal, stream.admission_ticket, stream.blocked_input = None, 19, "client_input"
    target_ref = LogicalReference(LOCAL_PRINCIPAL, "response:unfinished", 1)
    stream.retention_intent = RetentionIntent(target_ref, None, (), "unfinished") if role == "logical" else None
    engine = _slot(decoder.w, materialized, decoder.buf, decoder.mbuf, decoder.pbuf, 8, 4)
    engine.stops = []
    decoder.fills[stream.sid] = [engine, False, 3, [], None]
    decoder.filling.append(stream)
    target = None
    original = ValueError("controlled retired backend refusal")
    try:
        with limit.request():
            limit.borrow(box)
            stream.delivery_owner = DeliveryOwner(limit, box, raw, lambda _values: None, None)
            if role == "logical":
                decoder.enqueue_retained(stream)
            assert decoder.suspend_request(stream)
            target = stream.retention_handle
            assert target.transfer.retired.wait(5)
            catalog.progress()
            decoder._release_retained_states()
            assert decoder.admit(stream) is False and stream.retention_request.restore_handle.transfer.retired.wait(5)
            catalog.progress()
            decoder.admit(stream)
            state, generation = stream.st, stream.st.slot_generation
            stream.error = original
            if ending == "cancel":
                decoder.cancel([stream])
                assert decoder.retire_retained_request(stream)
            else:
                assert decoder.drop(original) == [stream]
            assert state.pos == state.mtp_len == state.target_tentative_length == 0
            assert state.slot_generation == generation + 1  # Exactly one reset/release, not a duplicate free.
            assert sum(item is state for item in decoder.free) == 1
            assert catalog.lookup(parent_ref) is parent and parent.state is parent_state
            assert torch.equal(parent_state.rec, parent_bytes)
            if role == "logical":
                assert target.phase == "failed" and target.state is None and target.failure is original
                assert catalog.record()["future_backing_bytes"] == 2 * future
                catalog.logical_reference_unavailable(target_ref.principal, target_ref.sequence, generation=1, reason="aborted")
                catalog.progress()
            else:
                assert not target.internal_owner
            limit.release(box)
            stream.retirement_host.retire_backend()
        assert catalog.record()["future_backing_bytes"] == future
    finally:
        limit.release(box)
        if target is not None and target.internal_owner:
            catalog.release_temporary(target)
        if role == "logical" and target is not None and target_ref in target.owners:
            catalog.logical_reference_unavailable(target_ref.principal, target_ref.sequence, generation=1, reason="aborted")
            catalog.progress()
        catalog.release_owner(parent)
        catalog.close(timeout=5)

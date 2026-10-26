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

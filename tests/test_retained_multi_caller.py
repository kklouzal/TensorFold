# ruff: noqa: E402 - optional Torch must be checked before backend imports.
"""Actual Multi/catalog/State ownership with CPU tensors; no model/GPU qualification."""

import threading
from types import SimpleNamespace as NS

import pytest
torch = pytest.importorskip("torch")

from tensorfold.cuda.output_mailbox import OutputMailbox
from tensorfold.cuda.scheduler import DeliveryOwner
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
from tensorfold.families.qwen4_exp.cuda.retained_catalog import LogicalReference, RetentionCorrupt
from tensorfold.families.qwen4_exp.cuda.retained_host import RetentionIntent
from tensorfold.server.request_limits import RequestLimit
from test_dspark_taps_forward import cpu_weights
from test_flashnext_mtp_inputs import family as _mtp_family  # noqa: F401 - fixture registration
from test_retained_catalog import captured
from tensorfold.families.qwen4_exp.cuda.retained_catalog import make_retention_catalog
from tensorfold.families.qwen4_exp.cuda.retention_policy import RetentionPolicy

pytest_plugins = ("test_state_retention",)


def owner(state):
    # Two full future promises coexist in these branch/caller controls. The
    # explicit tiny-fixture budget includes replicated bounded manifests;
    # this is not a measured production RAM capacity or a default.
    policy = RetentionPolicy("ram", 2, 10, 1 << 30, 37, 74, 0)
    wake = threading.Event()
    catalog = make_retention_catalog(policy, NS(meta={"rank": 0}, comm=None),
        [state, state.clone()], prefix_limit=2, recovery_versions=2, wake=wake.set)
    return catalog, catalog.future_claim(), wake


def decoder_for(catalog, monkeypatch):
    # Real admission, prefix/reset and restore code; only unrelated buffer
    # construction uses maintained tiny CPU weights rather than CUDA startup.
    from tensorfold.families.qwen4_exp.cuda.state import Buffers

    value = object.__new__(MultiDecoder)
    value.w = cpu_weights(monkeypatch)
    value.w.comm = None
    value.states, value.free = catalog.states, [catalog.states[1]]
    value.retention_catalog = catalog
    value.admission_policy, value.capacity, value.depth = "fifo", 8, 0
    value.logical_context = catalog.states[0].logical_context
    value.kv_pair, value.prefill_rows, value.next_id = catalog.states[0].kv_pair, 4, 0
    value.buf = Buffers(value.w, 4, 8)
    value.pbuf = Buffers(value.w, 4, 8, prefill=True)
    value.mbuf, value._draft_map, value.vision, value.media_runtime = None, None, None, None
    value.controller, value.copy, value.points, value.graphs = None, False, None, None
    value.streams, value.filling, value.fills, value.held = {}, [], {}, {}
    value.keep, value.clock, value.used = 2, 0, {}
    value.memory_gate = NS(held=0)
    value.poison_control = lambda error: (_ for _ in ()).throw(error)
    return value


def incoming(*, target="child", parent=None):
    stream = Stream([3, 7, 11, 9, 5], 1, draft=True, cancel=threading.Event())
    stream.principal, stream.admission_ticket = "local-unauthenticated", 7
    stream.retention_intent = RetentionIntent(
        None if target is None else LogicalReference(stream.principal, "response:" + target, 1), parent, (), target or "temp")
    return stream


def publish(catalog, reference):
    catalog.logical_reference_committed(reference.principal, reference.sequence, generation=reference.generation)
    catalog.progress()


def test_actual_multi_parks_ram_restore_then_admits_from_completed_same_ticket(materialized, monkeypatch):
    catalog, future, _wake = owner(materialized)
    parent, handle = captured(catalog, materialized, future)
    publish(catalog, parent)
    save = catalog.suspend(handle)
    assert save.retired.wait(5) and catalog.saved(handle)
    decoder = decoder_for(catalog, monkeypatch)
    catalog.take_released_states()  # Its initial source is intentionally the frozen parent fixture.
    stream = incoming(parent=parent)
    request = None
    try:
        decoder.enqueue_retained(stream, parent=parent)
        request = stream.retention_request
        assert request.handle is handle and len(catalog.protected_states()) == 0
        assert decoder.admit(stream) is False
        destination = stream.retention_window
        assert destination is catalog.states[1] and destination not in decoder.free
        assert decoder.live() == 0 and decoder.filling == []
        assert stream.admission_ticket == 7 and request.restore_handle.transfer.retired.wait(5)
        catalog.progress()
        assert decoder.admit(stream) is None
        assert stream.cached == 3 and decoder.fills[stream.sid][2] == 3
        assert stream.st is destination and stream.retention_window is None
        assert request.handle.phase == "ram"  # New branch never mutates the original required generation.
        assert decoder.filling == [stream]
    finally:
        if request is not None:
            request.close()
        decoder.filling.clear()
        for owned in (stream.retention_handle, handle):
            catalog.release_owner(owned)
        catalog.close(timeout=5)


def test_real_corrupt_restore_retires_actual_destination_and_preserves_source_failure(materialized, monkeypatch):
    catalog, future, _wake = owner(materialized)
    parent, handle = captured(catalog, materialized, future)
    publish(catalog, parent)
    save = catalog.suspend(handle)
    assert save.retired.wait(5) and catalog.saved(handle)
    decoder = decoder_for(catalog, monkeypatch)
    catalog.take_released_states()
    stream = incoming(parent=parent)
    key = next(key for key in handle.keys if catalog._ram.descriptor(key)["bytes"])
    catalog._ram._objects[key].data[0] ^= 1  # Actual backing fault, not a synthetic exception classifier.
    try:
        decoder.enqueue_retained(stream, parent=parent)
        assert decoder.admit(stream) is False
        destination = stream.retention_window
        before = destination.rec.clone()
        transfer = stream.retention_request.restore_handle.transfer
        assert transfer.retired.wait(5)
        with pytest.raises(RetentionCorrupt) as original:
            catalog.progress()
        stream.error = original.value
        assert torch.equal(destination.rec, before)
        assert decoder.retire_retained_request(stream)
        assert stream.st is None and stream.retention_request is None
        assert handle.failure is original.value and handle.phase == "failed"
        assert all(state is not destination for state in catalog.protected_states())
        assert any(state is destination for state in decoder.free)
        assert catalog.record()["future_backing_bytes"] > 0  # Pending target survives until actual logical abort.
        assert original.value.__traceback__ is None  # Safe actor/worker frames retired after the actual fence.
        catalog.check_health()
    finally:
        catalog.release_owner(stream.retention_handle)
        catalog.release_owner(handle)
        catalog.close(timeout=5)


def test_actual_completed_capture_finish_and_fresh_subrun_retire_host_before_replacing(materialized, monkeypatch):
    catalog, _future, _wake = owner(materialized)
    decoder = decoder_for(catalog, monkeypatch)
    stream = incoming()
    stream.prompt = [3, 7, 11, 9]
    stream.st, stream.sid = materialized, 3
    limit, box = RequestLimit(1), OutputMailbox(4)
    try:
        with limit.request():
            limit.borrow(box)
            stream.delivery_owner = DeliveryOwner(limit, box, stream.cancel, lambda _tokens: None, None)
            decoder.enqueue_retained(stream)
            stream.done = True
            decoder.streams[stream.sid] = stream
            decoder._capture_retained(stream)
            target = stream.retention_handle
            assert target.state is materialized and target.host.stream is stream
            decoder.finish([stream])
            assert stream.st is None and target.state is materialized
            assert all(state is not materialized for state in decoder.free)
            next_subrun = incoming()
            next_subrun.prompt = list(stream.prompt)
            with pytest.raises(RuntimeError, match="still owns"):
                decoder.enqueue_retained(next_subrun)
            assert target.host.stream is stream
            limit.release(box)
            decoder.enqueue_retained(next_subrun)
            assert next_subrun.retention_handle is target and target.phase == "reserved"
            assert target.host is None and next_subrun.retention_window is None  # RAM tier releases idle GPU.
            assert catalog.record()["sessions"] == 1
    finally:
        limit.release(box)
        if getattr(stream, "retention_handle", None) is not None:
            catalog.release_owner(stream.retention_handle)
        catalog.close(timeout=5)


def test_disposable_ram_prefix_consumer_retires_after_active_owner_is_removed(materialized, monkeypatch):
    catalog, future, _wake = owner(materialized)
    parent, handle = captured(catalog, materialized, future)
    publish(catalog, parent)
    save = catalog.suspend(handle)
    assert save.retired.wait(5) and catalog.saved(handle)
    catalog.configure_admission(max_waiting_input_bytes=4096, max_pending_requests=2)
    decoder = decoder_for(catalog, monkeypatch)
    catalog.take_released_states()
    stream = incoming(target=None, parent=parent)
    try:
        decoder.enqueue_retained(stream, parent=parent)
        request = stream.retention_request
        transient = request.target_handle
        assert transient.transient and transient.future_bytes == 0
        assert decoder.admit(stream) is False and transient.transfer.retired.wait(5)
        catalog.progress()
        decoder.admit(stream)
        decoder.filling.remove(stream)
        decoder.fills.pop(stream.sid)
        decoder.streams[stream.sid] = stream
        stream.done = True
        assert decoder.retire_retained_request(stream) and not request.closed
        assert transient.state is stream.st
        destination = stream.st
        decoder.finish([stream])
        assert request.closed and stream.retention_request is None
        assert any(state is destination for state in decoder.free)
        assert catalog.lookup(parent) is handle and handle.phase == "ram"
        assert catalog.record()["sessions"] == 1  # Disposable cache hit made no retained promise.
    finally:
        if getattr(stream, "retention_request", None) is not None:
            stream.retention_request.close()
        catalog.release_owner(handle)
        catalog.close(timeout=5)


@pytest.mark.parametrize("marker_failure", [False, True], ids=["retained-marker", "marker-allocation-failed"])
def test_real_failed_destination_cleanup_latches_tp_before_diagnostic_marker_allocation(materialized, monkeypatch, marker_failure):
    import tensorfold.cleanup as cleanup
    from tensorfold.families.qwen4_exp.cuda.execution_identity import _unretired
    from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder

    catalog, future, _wake = owner(materialized)
    parent, handle = captured(catalog, materialized, future)
    publish(catalog, parent)
    save = catalog.suspend(handle)
    assert save.retired.wait(5) and catalog.saved(handle)
    catalog.take_released_states()
    decoder = decoder_for(catalog, monkeypatch)
    decoder._pass_rows = lambda: 4
    decoder.points, decoder.arrived = None, lambda: False
    peer = TensorParallelDecoder(decoder, NS(_abort_comm=False))
    stream = incoming(parent=parent)
    key = next(key for key in handle.keys if catalog._ram.descriptor(key)["bytes"])
    catalog._ram._objects[key].data[0] ^= 1
    try:
        decoder.enqueue_retained(stream, parent=parent)
        assert decoder.admit(stream) is False
        request = stream.retention_request
        assert request.restore_handle.transfer.retired.wait(5)
        with pytest.raises(RetentionCorrupt) as original:
            catalog.progress()
        stream.error = original.value
        secondary = OSError("injected actor destination retirement failure before mutation")
        actual_retire = catalog.retire_failed_destination
        def failed_retire(*_args, **_kwargs):
            raise secondary
        class FailedMarker(dict):
            @staticmethod
            def __setitem__(mapping, key, value):
                if key == "_tensorfold_retained_owners":
                    raise MemoryError("injected diagnostic marker allocation failure")
                dict.__setitem__(mapping, key, value)
        with monkeypatch.context() as fault:
            fault.setattr(catalog, "retire_failed_destination", failed_retire)
            if marker_failure:
                fault.setattr(cleanup, "dict", FailedMarker, raising=False)
            with pytest.raises(RetentionCorrupt) as reported:
                peer._retention_cpu(lambda: decoder.retire_retained_request(stream))
        assert reported.value is original.value
        assert peer.failure is original.value and peer.engine._abort_comm
        assert _unretired(original.value) is (not marker_failure)
        assert request.restore_handle.failed_transfer is not None
        assert request.restore_handle.state is stream.retention_window
        with pytest.raises(RuntimeError, match="poisoned"):
            peer.drop()  # No comm/model publication follows uncertain cleanup, even without a retained marker.
        # This controlled CPU fault happened before mutation. Test teardown
        # proves the real destination fence/retirement explicitly, outside the
        # poisoned serving actor; it is not a production recovery path.
        actual_retire(request, fenced=True)
        request.close()
        stream.retention_request = None
        decoder._release_retained_states()
    finally:
        catalog.release_owner(stream.retention_handle)
        catalog.release_owner(handle)
        catalog.close(timeout=5)


def test_actual_multi_restored_text_hc_conditions_next_input_once(request, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import forward, mtp
    from tensorfold.families.qwen4_exp.cuda.state import State
    from test_flashnext_mtp_inputs import buffers

    weights = request.getfixturevalue("_mtp_family")
    source, destination = [State(weights, 8, 4, "fp16", limit=8) for _ in range(2)]
    catalog = make_retention_catalog(RetentionPolicy("ram", 2, 10, 1 << 30, 37, 74, 0),
        NS(meta={"rank": 0}, comm=None), [source, destination], prefix_limit=2, recovery_versions=2)
    prompt, target = [3, 7, 11, 9, 5], buffers(weights)
    target.ids[:3].copy_(torch.tensor(prompt[:3], dtype=torch.int32))
    target.ple_v.fill_(.125)
    import numpy as np
    source.ple_last = (source.ple_history.copy(), np.asarray(prompt[:3], dtype=np.int64))
    forward.compute(weights, [(source, 0, 3)], target)
    hidden = target.streams[:3].clone()
    forward.commit(weights, source, target, 3, 3)
    mtp.mtp_forward(weights, source, buffers(weights), prompt[1:3], hidden[:2], logits=False)
    source.set_mtp_len(2)
    source.media_mtp_enabled = True
    expected = source.clone()
    mtp.mtp_forward(weights, expected, buffers(weights), [prompt[3]], hidden[2:3], logits=False)
    expected.set_mtp_len(3)
    parent = LogicalReference("local-unauthenticated", "response:mtp", 1)
    handle = catalog.reserve(parent, future_bytes=catalog.future_claim())
    catalog.capture_idle(Stream(prompt, 1, draft=True, st=source), pending_hc=hidden[2:3], handle=handle)
    publish(catalog, parent)
    save = catalog.suspend(handle)
    assert save.retired.wait(5) and catalog.saved(handle)
    decoder = decoder_for(catalog, monkeypatch)
    decoder.w, decoder.depth = weights, 1
    decoder.buf, decoder.pbuf, decoder.mbuf = (buffers(weights) for _ in range(3))
    catalog.take_released_states()
    stream = incoming(parent=parent)
    try:
        decoder.enqueue_retained(stream, parent=parent)
        assert decoder.admit(stream) is False and stream.retention_request.restore_handle.transfer.retired.wait(5)
        catalog.progress()
        decoder.admit(stream)
        assert stream.st.pos == 3 and stream.st.mtp_len == 3 and stream.st.mtp_drafted == 0
        assert stream.out == stream.drafts == []
        for name in ("k", "v", "ks", "vs"):
            actual, baseline = getattr(destination.mtp_kc, name), getattr(expected.mtp_kc, name)
            if actual is not None:
                assert torch.equal(actual[:3], baseline[:3]), name
        assert torch.equal(destination.mtp_pooled[:3 // destination.ratio], expected.mtp_pooled[:3 // expected.ratio])
        assert torch.equal(hidden[2:3], catalog.restored(stream.retention_request)["pending_hc"])
    finally:
        stream.retention_request.close()
        decoder.filling.clear()
        catalog.release_owner(stream.retention_handle)
        catalog.release_owner(handle)
        catalog.close(timeout=5)


@pytest.mark.parametrize("boundary", ["stage_request", "request_blocked_reason", "retention_boundary", "retention_deadline"])
def test_actual_catalog_failure_contains_each_public_tp_cpu_boundary_before_publication(materialized, monkeypatch, boundary):
    from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder

    catalog, _future, _wake = owner(materialized)
    decoder = decoder_for(catalog, monkeypatch)
    decoder._pass_rows, decoder.points, decoder.arrived = lambda: 4, None, lambda: False
    peer = TensorParallelDecoder(decoder, NS(_abort_comm=False))
    monkeypatch.setattr(peer, "_invoke", lambda *_args, **_kwargs: pytest.fail("poisoned CPU boundary published a TP round"))
    failure = OSError("injected actual catalog ownership failure")
    catalog._latch_failure(failure)
    stream = incoming()
    callback = {"stage_request": lambda: peer.stage_request(stream, wake=lambda: None,
                    max_waiting_input_bytes=4096, max_pending_requests=2),
                "request_blocked_reason": lambda: peer.request_blocked_reason(stream),
                "retention_boundary": lambda: peer.retention_boundary(idle=True),
                "retention_deadline": peer.retention_deadline}[boundary]
    try:
        with pytest.raises(RuntimeError) as reported:
            callback()
        assert reported.value.__cause__ is failure
        assert peer.failure is reported.value and peer.engine._abort_comm
    finally:
        catalog.close(timeout=5)

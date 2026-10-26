# ruff: noqa: E402 - optional Torch precedes the actual TP caller.
"""Exact streamed digest, bounded keys and NEXT-counter terminal behavior."""
import hashlib
import json
from types import SimpleNamespace as NS

import pytest
pytest.importorskip("torch")

from tensorfold.cuda import rank_protocol
from tensorfold.cuda.store_fields import validate_store_key
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda import tp_scheduler
from tensorfold.families.qwen4_exp.cuda.tp_scheduler import MAX_PROTOCOL_SEQUENCE, TensorParallelDecoder
from test_tp_ack_envelope import Store, actor


def recorded_state(value):
    """Independent pre-change record contract; retain every row and field."""
    state = [(wire, s.sid, s.count, s.cached, len(s.out), s.out[-8:], s.drafts, s.owed,
              bool(s.done), bool(s.waiting), None if s.st is None else [
                  s.st.pos, getattr(s.st, "mtp_len", None), getattr(s.st, "mtp_drafted", None),
                  getattr(s.st, "cur", None), getattr(s.st, "slot_generation", None),
                  getattr(s.st, "history_generation", None)],
              None if s.error is None else type(s.error).__name__) for wire, s in value.requests.items()]
    c = value.decoder.controller
    record = [state, None if c is None else [c.epoch, c.effective, c.last_served, c.effective_family],
              None if value.engine.serving_profiles is None else [
                  value.engine.serving_profiles.gate.epoch, value.engine.serving_profiles.gate.available]]
    if value.admission_policy == "fifo":
        record.append(["fifo", [(wire, s.admission_ticket, s.principal) for wire, s in value.requests.items()]])
    return record


@pytest.mark.parametrize("count,fifo,controller,profiles", [
    (0, False, False, False), (1, False, True, True), (3, True, False, True), (64, True, True, False)])
def test_streamed_digest_matches_all_legacy_utf8_bytes_without_joined_encoder_output(
        monkeypatch, count, fifo, controller, profiles):
    value = TensorParallelDecoder.__new__(TensorParallelDecoder)
    value.requests = {}
    for index in range(count):
        stream = Stream([1, 2], 20, st=None if index % 2 else NS(
            pos=19, mtp_len=18, mtp_drafted=2, cur=[2, 1], slot_generation=7, history_generation=3),
            sid=index, out=list(range(12)), drafts=[3, 5], owed=[8, 9])
        stream.cached, stream.done, stream.waiting = index, bool(index % 3), bool(index % 5)
        stream.admission_ticket, stream.principal = index + 11, "\u96ea\U0001f600" * 17
        stream.error = ValueError("existing failure") if index % 7 else None
        value.requests[count - index + 4] = stream  # Preserve insertion order; do not sort rows.
    control = NS(epoch=8, effective={"z": -0.0, "a": [1.25e-9, float("inf"), float("nan")]},
                 last_served={"z": 4, "a": 1}, effective_family={"z": "\u96ea", "a": True}) if controller else None
    value.decoder = NS(controller=control)
    value.engine = NS(serving_profiles=NS(gate=NS(epoch=5, available=False)) if profiles else None)
    value.admission_policy = "fifo" if fifo else "legacy"
    expected = json.dumps(recorded_state(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    real_sha, chunks = hashlib.sha256, []
    class Incremental:
        def __init__(self):
            self.digest = real_sha()
        def update(self, chunk):
            assert type(chunk) is bytes
            chunks.append(chunk)
            self.digest.update(chunk)
        def hexdigest(self):
            return self.digest.hexdigest()
    monkeypatch.setattr(tp_scheduler.hashlib, "sha256", Incremental)
    monkeypatch.setattr(tp_scheduler.json, "dumps", lambda *_a, **_k: pytest.fail("digest joined whole JSON"))
    monkeypatch.setattr(rank_protocol, "MAX_MESSAGE_BYTES", 1)  # Digest input has NO RPC/ACK cap.
    assert value._state_digest() == real_sha(expected).hexdigest()
    assert len(chunks) > 1 and b"".join(chunks) == expected


@pytest.mark.parametrize("method,part", [("_key", "rank"), ("_ack_key", "ack")])
def test_each_protocol_key_is_byte_identical_and_shared_validation_runs_once(monkeypatch, method, part):
    calls = []
    def validate(key):
        calls.append(key)
        validate_store_key(key)
    monkeypatch.setattr(tp_scheduler, "validate_store_key", validate)
    value = TensorParallelDecoder.__new__(TensorParallelDecoder)
    expected = f"tensorfold/flashnext/round/31/{part}/7"
    assert getattr(value, method)(31, 7) == expected and calls == [expected]
    with pytest.raises(ValueError, match="key field"):
        getattr(value, method)("x" * 8192, 7)


def test_last_operation_id_is_max_minus_one_then_exhaustion_is_pre_control_publication():
    status = {"error": None, "state": "same"}
    store = Store(json.dumps(status).encode())
    value = actor(0, store, status)
    value.sequence = MAX_PROTOCOL_SEQUENCE - 1
    assert value._invoke({"kind": "round"}) == "completed"
    assert value.sequence == MAX_PROTOCOL_SEQUENCE
    assert store.sent[0][0] == f"tensorfold/flashnext/round/{MAX_PROTOCOL_SEQUENCE - 1}/rank/1"
    before = list(store.sent), list(store.deleted)
    value._controls = lambda *_a: pytest.fail("exhaustion mutated controller state")
    body = {"kind": "round"}
    with pytest.raises(OverflowError, match="operation sequence exhausted"):
        value._invoke(body)
    assert body == {"kind": "round"} and value.sequence == MAX_PROTOCOL_SEQUENCE
    assert (store.sent, store.deleted) == before


def test_last_stop_id_preserves_bytes_and_refuses_next_before_publication():
    store = Store()
    value = actor(0, store, {})
    value.sequence = MAX_PROTOCOL_SEQUENCE - 1
    value.shutdown()
    assert store.sent == [(f"tensorfold/flashnext/round/{MAX_PROTOCOL_SEQUENCE - 1}/rank/1",
                           '{"kind": "stop", "abort": false}')]
    with pytest.raises(OverflowError, match="stop sequence exhausted"):
        value.shutdown()
    assert value.sequence == MAX_PROTOCOL_SEQUENCE and len(store.sent) == 1


@pytest.mark.parametrize("stop", [False, True])
def test_receive_reserves_last_id_then_refuses_before_wait_get_or_reuse(stop):
    body = {"kind": "stop", "abort": False} if stop else {"kind": "round", "control": {"admission_policy": "fifo"}}
    store = Store(json.dumps(body).encode())
    value = actor(1, store, {})
    value.sequence = MAX_PROTOCOL_SEQUENCE - 1
    received = value.receive()
    assert received is None if stop else received["sequence"] == MAX_PROTOCOL_SEQUENCE - 1
    assert value.sequence == MAX_PROTOCOL_SEQUENCE
    store.wait = lambda *_a: pytest.fail("exhausted receive waited for a reused key")
    store.get = lambda *_a: pytest.fail("exhausted receive fetched a reused key")
    with pytest.raises(OverflowError, match="receive sequence exhausted"):
        value.receive()


@pytest.mark.parametrize("rank", [0, 1])
def test_prefix_last_step_and_key_validation_precede_terminal_refusal(monkeypatch, rank):
    decisions = [[1, False]]
    store = Store(json.dumps({"decisions": decisions, "failed": False}).encode())
    value = actor(rank, store, {})
    value.prefix_operation, value.prefix_sequence = 3, MAX_PROTOCOL_SEQUENCE - 1
    calls = []
    def validate(key):
        calls.append(key)
        validate_store_key(key)
    monkeypatch.setattr(tp_scheduler, "validate_store_key", validate)
    assert value._prefix_decisions(decisions) == decisions
    key = f"tensorfold/flashnext/prefix/3/{MAX_PROTOCOL_SEQUENCE - 1}/rank/1"
    assert calls == [key] and value.prefix_sequence == MAX_PROTOCOL_SEQUENCE
    before = list(store.sent), list(store.deleted)
    with pytest.raises(OverflowError, match="prefix sequence exhausted"):
        value._prefix_decisions(decisions)
    assert calls == [key] and (store.sent, store.deleted) == before


@pytest.mark.parametrize("invalid", [None, True, -1])
def test_invalid_next_counter_refuses_without_formatting_or_publication(invalid):
    value = actor(0, Store(), {})
    value.sequence = invalid
    with pytest.raises(ValueError, match="nonnegative integer"):
        value._invoke({"kind": "round"})
    assert value.sequence is invalid and value.comm.store.sent == []


def wire_caller(monkeypatch, kind):
    """Keep allocation/identity code real; observe its existing handoff boundary."""
    value = actor(0, Store(), {})
    value.next_wire, value.wires, value.requests, value._points = MAX_PROTOCOL_SEQUENCE - 1, {}, {}, None
    value.decoder = NS(vision=NS(is_media_runtime=True) if kind == "media" else None, w=object(), kv_pair=None)
    calls = []
    value._pack = lambda stream: calls.append(("pack", stream)) or {"kept": True}
    value._invoke = lambda body, source=None: calls.append(("invoke", body, source))
    value._media_cpu = lambda operation: operation()
    monkeypatch.setattr("tensorfold.families.qwen4_exp.cuda.prefix_policy.reuse_allowed", lambda *_a, **_k: False)
    monkeypatch.setattr(tp_scheduler.media_requests, "prepare", lambda stream, owner, wire, wake:
        calls.append(("media", stream, wire, wake)) or True)
    catalog = NS(set_wake=lambda wake: calls.append(("wake", wake)))
    def wake():
        pass
    def stream():
        result = Stream([1, 2], 3, vision=object() if kind == "media" else None)
        result.retention_intent = None
        result.principal = None  # Scheduler preparation supplies the trusted NoAuth principal domain.
        return result
    def run(item):
        if kind == "stage":
            return value._stage_request(item, catalog, wake, 4096, 2)
        if kind == "admit":
            return value.admit(item)
        return value.prepare_request(item, wake=wake)
    return value, calls, stream, run


@pytest.mark.parametrize("kind", ["stage", "admit", "media"])
def test_all_wire_allocation_sites_issue_last_id_then_refuse_before_claim_map_or_publication(monkeypatch, kind):
    value, calls, new_stream, run = wire_caller(monkeypatch, kind)
    accepted = new_stream()
    run(accepted)
    if kind == "media":
        assert calls[-1][0] == "media" and calls[-1][2] == MAX_PROTOCOL_SEQUENCE - 1
    else:
        body = next(call[1] for call in calls if call[0] == "invoke")
        assert body["wire"] == MAX_PROTOCOL_SEQUENCE - 1 and body["stream"] == {"kept": True}
    assert value.next_wire == MAX_PROTOCOL_SEQUENCE
    before = list(calls), dict(value.wires), dict(value.requests)
    refused = new_stream()
    with pytest.raises(OverflowError, match="wire sequence exhausted"):
        run(refused)
    assert not hasattr(refused, "_media_wire") and value.next_wire == MAX_PROTOCOL_SEQUENCE
    assert (calls, value.wires, value.requests) == before


@pytest.mark.parametrize("kind", ["stage", "admit", "media"])
def test_existing_zero_wire_can_continue_without_new_allocation_at_exhaustion(monkeypatch, kind):
    value, calls, new_stream, run = wire_caller(monkeypatch, kind)
    existing = new_stream()
    value.next_wire = MAX_PROTOCOL_SEQUENCE
    if kind == "admit":
        value.wires[id(existing)] = 0
    else:
        existing._media_wire = 0
    run(existing)
    if kind == "media":
        assert calls[-1][2] == 0
    else:
        assert next(call[1] for call in calls if call[0] == "invoke")["wire"] == 0
    assert value.next_wire == MAX_PROTOCOL_SEQUENCE

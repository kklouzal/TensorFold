# ruff: noqa: E402 - optional Torch is required by the maintained TP caller.
"""ACK publication/decoding boundaries without oversized native Store writes."""
import json
from types import SimpleNamespace as NS

import pytest
pytest.importorskip("torch")

from tensorfold.cuda import rank_protocol
from tensorfold.families.qwen4_exp.cuda import tp_scheduler
from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder


class Store:
    def __init__(self, payload=None, *, batch=False):
        self.payload, self.sent, self.deleted = payload, [], []
        self.multi_set = self._multi_set if batch else None
        self.multi_get = self._multi_get if batch else None

    def set(self, key, value):
        self.sent.append((key, value))

    def _multi_set(self, keys, values):
        self.sent.extend(zip(keys, values))

    def wait(self, _keys, _timeout):
        pass

    def get(self, _key):
        return self.payload

    def _multi_get(self, keys):
        return [self.get(key) for key in keys]

    def delete_key(self, key):
        self.deleted.append(key)


def actor(rank, store, status):
    value = TensorParallelDecoder.__new__(TensorParallelDecoder)
    value.comm, value.engine = NS(rank=rank, world=2, store=store), NS(_abort_comm=False)
    value.failure, value.sequence, value.admission_policy = None, 0, "fifo"
    value._controls = lambda _execution: {}
    value._execute = lambda *_args: ("completed", None, status)
    def poison(error):
        value.failure, value.engine._abort_comm = error, True
    value._poison_control = poison
    return value


def at_limit(limit):
    empty = {"error": None, "state": ""}
    width = limit - len(json.dumps(empty, separators=(",", ":")))
    return {"error": None, "state": "x" * width}


def test_real_rank_transport_contract_remains_96_mib():
    assert rank_protocol.MAX_MESSAGE_BYTES == 96 << 20


def test_ack_send_accepts_exact_envelope_and_rejects_next_byte_before_store_set(monkeypatch):
    # A smaller TEST-ONLY cap exercises the same arithmetic without allocating
    # or publishing a native payload near the real 96MiB deployment contract.
    limit = 128
    monkeypatch.setattr(rank_protocol, "MAX_MESSAGE_BYTES", limit)
    status, store = at_limit(limit), Store()
    peer = actor(1, store, status)
    peer.apply({"kind": "round", "sequence": 3})
    assert len(store.sent) == 1
    key, text = store.sent[0]
    assert key == peer._ack_key(3, 1) and len(text.encode("ascii")) == limit
    assert json.loads(text) == status and peer.failure is None
    status["state"] += "x"
    with pytest.raises(ValueError, match="96 MiB transport envelope") as caught:
        peer.apply({"kind": "round", "sequence": 4})
    assert len(store.sent) == 1 and peer.failure is caught.value
    assert peer.engine._abort_comm


def test_ack_codec_preserves_nested_unicode_status_and_recoverable_error_contents():
    error = ValueError("controlled refusal")
    status = {"error": "ValueError", "state": "digest", "complete": {"changed": False, "reason": "\u96ea\n"}}
    store = Store()
    peer = actor(1, store, status)
    peer._execute = lambda *_args: (None, error, status)
    peer._recoverable_operation = lambda actual, kind: actual is error and kind == "retention_boundary"
    peer.apply({"kind": "retention_boundary", "sequence": 9})
    assert json.loads(store.sent[0][1]) == status and peer.failure is None
    assert store.sent[0][1].isascii()


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("payload", [None, "{}", bytearray(b"{}"), memoryview(b"{}"), b"x" * 129])
def test_bad_ack_type_or_size_refuses_before_decode_and_retirement(monkeypatch, batch, payload):
    monkeypatch.setattr(rank_protocol, "MAX_MESSAGE_BYTES", 128)
    store = Store(payload, batch=batch)
    leader = actor(0, store, {"error": None, "state": "same"})
    monkeypatch.setattr(tp_scheduler.json, "loads", lambda *_a, **_k: pytest.fail("unbounded ACK reached decoding"))
    with pytest.raises(ValueError, match="acknowledgment requires at most 96 MiB of bytes") as caught:
        leader._invoke({"kind": "round"})
    assert leader.failure is caught.value and leader.engine._abort_comm
    assert store.deleted == []  # Uncertain remote execution never takes normal ACK reclamation.


@pytest.mark.parametrize("batch", [False, True])
def test_ack_at_exact_limit_preserves_comparison_and_normal_reclamation(monkeypatch, batch):
    limit = 128
    monkeypatch.setattr(rank_protocol, "MAX_MESSAGE_BYTES", limit)
    status = at_limit(limit)
    wire = json.dumps(status, separators=(",", ":")).encode("ascii")
    assert len(wire) == limit
    store = Store(wire, batch=batch)
    leader = actor(0, store, status)
    assert leader._invoke({"kind": "round"}) == "completed"
    assert leader.failure is None
    assert store.deleted == [leader._ack_key(0, 1), leader._key(0, 1)]


def test_invalid_ack_keeps_original_source_failure_and_poisons_without_decode(monkeypatch):
    monkeypatch.setattr(rank_protocol, "MAX_MESSAGE_BYTES", 128)
    source = ValueError("original request failure")
    store = Store(b"x" * 129)
    leader = actor(0, store, {"error": None, "state": "same"})
    with pytest.raises(ValueError) as caught:
        leader._invoke({"kind": "drop"}, source=source)
    assert caught.value is source and leader.failure is source
    assert store.deleted == []

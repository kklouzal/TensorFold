"""Real TP RPC methods retire only redundant Python owners after last use."""
import json
import threading
from types import SimpleNamespace as NS
import weakref

import pytest

pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder


class Body(dict):
    pass


class Prompt(list):
    pass


class Encoded(str):
    pass


def follower(*, apply_error=None, cleanup_error=None):
    engine = FlashNextEngine.__new__(FlashNextEngine)
    engine._lifecycle = threading.Condition()
    engine._calls = {}
    engine._closing = engine._closed = engine._abort_comm = False
    engine._receiving = False
    engine.scheduler = None
    oracle = NS(receives=0, retirements=0, stream=None, body=None, prompt=None, carry=None)
    unpacker = TensorParallelDecoder.__new__(TensorParallelDecoder)
    unpacker.admission_policy = "fifo"

    def receive():
        oracle.receives += 1
        if oracle.receives == 2:
            assert oracle.body() is None
            assert oracle.prompt() is None  # Original pointer list, not its shared integers.
            assert oracle.stream.carry is oracle.carry()
            assert oracle.stream.prompt == [987654]
            return None
        prompt, carry = Prompt([int("987654")]), Body(tag="owned-child")
        body = Body(stream={"prompt": prompt, "count": 2, "sampling": None,
            "draft": False, "stop_eos": True, "background": False, "grammar": [],
            "owed": [], "carry": carry, "profile_epoch": 0, "committed_output_tokens": 0,
            "admission_ticket": 7, "principal": "trusted", "output_capacity": 2, "retention": None})
        oracle.body, oracle.prompt, oracle.carry = weakref.ref(body), weakref.ref(prompt), weakref.ref(carry)
        return body

    def apply(body):
        assert body is oracle.body()
        if apply_error is not None:
            raise apply_error
        oracle.stream = unpacker._unpack(body["stream"])
        assert oracle.stream.prompt is not body["stream"]["prompt"]
        assert oracle.stream.prompt[0] is body["stream"]["prompt"][0]
        assert oracle.stream.carry is body["stream"]["carry"]

    def retire(request, *, failed=False):
        oracle.retirements += 1
        if oracle.retirements == 1:
            assert oracle.body() is not None  # Retirement remains a body last-use boundary.
            assert request.locked()
            if cleanup_error is not None:
                raise cleanup_error
        FlashNextEngine._end_request(engine, request, failed=failed)

    engine.tp_decoder = NS(receive=receive, apply=apply)
    engine._end_request = retire
    return engine, oracle


def test_successful_follower_retires_outer_body_before_next_receive_but_keeps_children():
    engine, oracle = follower()
    engine._follow_concurrent()
    assert oracle.receives == oracle.retirements == 2
    assert engine._calls == {} and engine._receiving is False
    assert oracle.stream.carry["tag"] == "owned-child"


@pytest.mark.parametrize("apply_fails,cleanup_fails", [(True, False), (False, True), (True, True)])
def test_follower_failures_preserve_primary_and_body_diagnostic_frame(apply_fails, cleanup_fails):
    primary, cleanup = RuntimeError("dispatch-primary"), RuntimeError("retirement-secondary")
    engine, oracle = follower(apply_error=primary if apply_fails else None,
                              cleanup_error=cleanup if cleanup_fails else None)
    with pytest.raises(RuntimeError) as caught:
        engine._follow_concurrent()
    assert caught.value is (primary if apply_fails else cleanup)
    if apply_fails and cleanup_fails:
        assert primary.__cause__ is cleanup
    frames = []
    trace = caught.value.__traceback__
    while trace is not None:
        if trace.tb_frame.f_code is FlashNextEngine._follow_concurrent.__code__:
            frames.append(trace.tb_frame)
        trace = trace.tb_next
    assert frames and all(frame.f_locals["body"] is oracle.body() for frame in frames)
    assert oracle.body() is not None and oracle.receives == 1 and oracle.retirements == 1


class Store:
    def __init__(self, oracle, *, batch, failure=None):
        self.oracle, self.failure = oracle, failure
        self.publications, self.acks, self.deleted = 0, 0, []
        self.multi_set = self._multi_set if batch else None
        self.multi_get = self._multi_get if batch else None

    def publish(self, values):
        self.publications += 1
        assert values and all(value is self.oracle.text() for value in values)
        # Actual last-use observation: payload survives the complete sending
        # callback. Native copied storage is a separate, unchanged contract.
        assert len(self.oracle.text()) == len("encoded-command")
        if self.failure is not None:
            raise self.failure

    def _multi_set(self, keys, values):
        assert len(keys) == len(values) == 2
        assert values[0] is values[1]  # Publication list contains shared references.
        self.publish(values)

    def set(self, key, value):
        self.publish([value])

    def wait(self, keys, timeout):
        assert self.oracle.text() is None
        assert len(keys) == 2

    def _multi_get(self, keys):
        return [self.get(key) for key in keys]

    def get(self, key):
        assert self.oracle.text() is None
        self.acks += 1
        return json.dumps(self.oracle.status).encode()

    def delete_key(self, key):
        assert self.oracle.text() is None
        self.deleted.append(key)
        return True


def caller(monkeypatch, *, batch, failure=None):
    oracle = NS(text=None, executes=0, status={"error": None, "state": "same"})
    def encode(body, **_kwargs):
        text = Encoded("encoded-command")
        oracle.text = weakref.ref(text)
        return text
    monkeypatch.setattr("tensorfold.cuda.rank_protocol.bounded_json", encode)
    store = Store(oracle, batch=batch, failure=failure)
    decoder = TensorParallelDecoder.__new__(TensorParallelDecoder)
    decoder.comm = NS(rank=0, world=3, store=store)
    decoder.engine = NS(_abort_comm=False)
    decoder.failure, decoder.sequence, decoder.admission_policy = None, 0, "fifo"
    decoder._controls = lambda _execution: {}
    decoder._poison_control = lambda error: None
    def execute(body, source):
        assert oracle.text() is None
        assert body["kind"] == "round" and source is None
        oracle.executes += 1
        return "completed", None, oracle.status
    decoder._execute = execute
    return decoder, oracle, store


@pytest.mark.parametrize("batch", [False, True])
def test_publisher_keeps_text_through_send_but_retires_it_before_execution_and_acks(monkeypatch, batch):
    decoder, oracle, store = caller(monkeypatch, batch=batch)
    assert decoder._invoke({"kind": "round"}) == "completed"
    assert oracle.text() is None and oracle.executes == 1
    assert store.publications == (1 if batch else 2) and store.acks == 2
    assert len(store.deleted) == 4  # ACK and command keys retain their existing retirement.


@pytest.mark.parametrize("batch", [False, True])
def test_publication_failure_preserves_primary_and_serialized_diagnostic_frame(monkeypatch, batch):
    primary = RuntimeError("publication-primary")
    decoder, oracle, store = caller(monkeypatch, batch=batch, failure=primary)
    with pytest.raises(RuntimeError) as caught:
        decoder._invoke({"kind": "round"})
    assert caught.value is primary and oracle.executes == store.acks == 0
    trace = primary.__traceback__
    frames = []
    while trace is not None:
        if trace.tb_frame.f_code is TensorParallelDecoder._invoke.__code__:
            frames.append(trace.tb_frame)
        trace = trace.tb_next
    assert frames and all(frame.f_locals["text"] is oracle.text() for frame in frames)
    assert oracle.text() is not None

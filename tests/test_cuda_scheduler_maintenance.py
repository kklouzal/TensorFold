"""Owner maintenance holds admission while ordinary request work can retire."""

from types import SimpleNamespace
import threading

import pytest

from tensorfold.cuda.scheduler import Scheduler
from tensorfold.server.cancellation import RequestCancelled
from tensorfold.server.errors import CapacityError


def wait(event):
    assert event.wait(3), "scheduler did not reach the planned boundary"


def call(function):
    result = SimpleNamespace(done=threading.Event(), error=None)

    def invoke():
        try:
            function()
        except BaseException as error:
            result.error = error
        finally:
            result.done.set()

    result.thread = threading.Thread(target=invoke, daemon=True)
    result.thread.start()
    return result


def finished(result, error=None):
    wait(result.done)
    result.thread.join(3)
    assert not result.thread.is_alive()
    if error is None:
        assert result.error is None
    else:
        assert isinstance(result.error, error)


class Decoder:
    def __init__(self):
        self.arrived = lambda: False
        self.streams = {}
        self.filling = []
        self.admitted = []
        self.first_round = threading.Event()
        self.allow_rounds = threading.Event()

    def live(self):
        return len(self.streams)

    def owns_admission(self, stream):
        return stream.prompt[0] == 2

    def admit(self, stream):
        stream.sid = stream.prompt[0]
        stream.st = SimpleNamespace()
        self.streams[stream.sid] = stream
        self.admitted.append(stream.sid)

    def round(self):
        if self.streams:
            self.first_round.set()
            wait(self.allow_rounds)
        for stream in self.streams.values():
            if not stream.done:
                stream.take([stream.prompt[0] + 10])
        return [stream for stream in self.streams.values() if stream.done]

    def finish(self, streams):
        for stream in streams:
            self.streams.pop(stream.sid, None)

    def drop(self):
        streams = list(self.streams.values())
        self.streams.clear()
        return streams


def scheduler(policy, decoder, maintenance, monkeypatch, *, queue_timeout_s=None):
    options = {"max_waiting_input_bytes": 4096} if policy == "fifo" else {}
    owner = Scheduler(decoder, max_streams=3, max_pending_requests=8, admission_policy=policy,
                      queue_timeout_s=queue_timeout_s, worker_maintenance=maintenance, **options)
    events = {}
    method = "publish" if policy == "fifo" else "put"
    original = getattr(owner.waiting, method)

    def publish(item, *args, **kwargs):
        result = original(item, *args, **kwargs)
        events[item[0].prompt[0]].set()
        return result

    monkeypatch.setattr(owner.waiting, method, publish)
    return owner.start(), events


def submit(owner, events, token, count=1):
    output = []
    events[token] = threading.Event()
    result = call(lambda: owner.submit([token], count, None, False, lambda values: output.extend(values)))
    result.output = output
    wait(events[token])
    return result


@pytest.mark.parametrize("policy", ["legacy", "fifo"])
def test_maintenance_keeps_queue_order_and_finishes_active_request(policy, monkeypatch):
    decoder = Decoder()
    pending, draining, applying, applied = (threading.Event() for _ in range(4))
    allow_apply = threading.Event()

    def maintenance(closing, occupied):
        if closing or not pending.is_set():
            return False
        assert threading.current_thread() is owner.thread
        if occupied:
            draining.set()
            return True
        applying.set()
        wait(allow_apply)
        pending.clear()
        applied.set()
        return False

    owner, events = scheduler(policy, decoder, maintenance, monkeypatch)
    if policy == "fifo":
        suspend = owner._suspend_blocked

        def suspension():
            assert not pending.is_set(), "maintenance must not park an already admitted stream"
            suspend()

        monkeypatch.setattr(owner, "_suspend_blocked", suspension)
    try:
        active = submit(owner, events, 1, count=3)
        wait(decoder.first_round)
        owner.request_runtime(pending.set)
        first = submit(owner, events, 2)  # FIFO's claimed path still must wait
        second = submit(owner, events, 3)
        decoder.allow_rounds.set()
        wait(draining)
        wait(applying)
        finished(active)
        assert active.output == [11, 11, 11] and decoder.admitted == [1]
        assert not first.done.is_set() and not second.done.is_set()
        # Another producer can publish while the maintenance owner waits.
        later = submit(owner, events, 4)
        allow_apply.set()
        wait(applied)
        for result in (first, second, later):
            finished(result)
        assert decoder.admitted == [1, 2, 3, 4]
    finally:
        decoder.allow_rounds.set()
        allow_apply.set()
        owner.close()


@pytest.mark.parametrize("policy", ["legacy", "fifo"])
def test_maintenance_preserves_queue_deadline(policy, monkeypatch):
    decoder = Decoder()
    decoder.allow_rounds.set()
    pending, held = threading.Event(), threading.Event()

    def maintenance(closing, occupied):
        if not closing and pending.is_set():
            assert occupied == 0
            held.set()
            return True
        return False

    owner, events = scheduler(policy, decoder, maintenance, monkeypatch, queue_timeout_s=0.2)
    try:
        owner.request_runtime(pending.set)
        wait(held)
        queued = submit(owner, events, 1)
        finished(queued, CapacityError)
        assert decoder.admitted == []
        owner.request_runtime(pending.clear)
    finally:
        owner.close()


@pytest.mark.parametrize("policy", ["legacy", "fifo"])
def test_close_settles_waiting_maintenance_and_queued_work(policy, monkeypatch):
    decoder = Decoder()
    decoder.allow_rounds.set()
    pending, held, cancelled = (threading.Event() for _ in range(3))

    def maintenance(closing, occupied):
        if closing and pending.is_set():
            pending.clear()
            cancelled.set()
        elif pending.is_set():
            held.set()
        return pending.is_set()

    owner, events = scheduler(policy, decoder, maintenance, monkeypatch)
    try:
        owner.request_runtime(pending.set)
        wait(held)
        queued = submit(owner, events, 1)
        closing = call(owner.close)
        finished(closing)
        wait(cancelled)
        finished(queued, RequestCancelled if policy == "fifo" else None)
        assert decoder.admitted == ([] if policy == "fifo" else [1])
    finally:
        owner.close()

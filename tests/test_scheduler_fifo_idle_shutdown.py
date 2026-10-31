"""Real FIFO actor retirement, independent of the logical caller's input credit."""

from queue import Empty, Queue
import threading
from types import SimpleNamespace

import pytest

from tensorfold.cuda.scheduler import Scheduler
from tensorfold.server.cancellation import RequestCancelled


def wait(event):
    if not event.wait(3):
        raise AssertionError('controlled actor did not reach its lifecycle boundary')


def join(thread):
    thread.join(3)
    if thread.is_alive():
        raise AssertionError('controlled actor did not retire')


class Controls:
    def __init__(self, events):
        self.pending = Queue()
        self.events = events

    def admission_limit(self):
        return 1

    def boundary(self):
        while True:
            try:
                value = self.pending.get_nowait()
            except Empty:
                return
            if isinstance(value, BaseException):
                raise value
            self.events.append(('control', value))


class Decoder:
    def __init__(self, *, paused=False):
        self.arrived = lambda: False
        self.streams = {}
        self.filling = []
        self.events = []
        self.controller = Controls(self.events)
        self.entered = threading.Event()
        self.release = threading.Event()
        if not paused:
            self.release.set()
        self.retired = []
        self.next_id = 0

    def live(self):
        return len(self.streams)

    def admit(self, stream):
        stream.sid, self.next_id = self.next_id, self.next_id + 1
        stream.st = SimpleNamespace(pos=len(stream.prompt))
        self.streams[stream.sid] = stream
        self.events.append(('admitted', stream.prompt[0]))

    def round(self):
        self.entered.set()
        wait(self.release)
        for stream in self.streams.values():
            if not stream.done:
                stream.st.pos += 1
                stream.take([stream.prompt[0] + 1])
        return [stream for stream in self.streams.values() if stream.done]

    def cancel(self, streams):
        for stream in streams:
            stream.done = True
        return streams

    def retire_request(self, stream):
        self.retired.append(stream.prompt[0])
        return True

    def finish(self, streams):
        for stream in streams:
            self.events.append(('finished', stream.prompt[0]))
            self.streams.pop(stream.sid, None)

    def drop(self):
        streams = list(self.streams.values())
        self.streams.clear()
        return streams


def scheduler(decoder):
    return Scheduler(decoder, max_streams=1, max_pending_requests=4,
                     admission_policy='fifo', max_waiting_input_bytes=4096,
                     output_queue_tokens=1).start()


def call(function):
    result = SimpleNamespace(done=threading.Event(), value=None, error=None)
    def run():
        try:
            result.value = function()
        except BaseException as error:
            result.error = error
        finally:
            result.done.set()
    result.thread = threading.Thread(target=run, daemon=True)
    result.thread.start()
    return result


def submitted(scheduling, token, count=1, **kwargs):
    output = []
    result = call(lambda: scheduling.submit([token], count, None, False,
                                           lambda values: output.extend(values), **kwargs))
    result.output = output
    return result


def stop_published(scheduling):
    event = threading.Event()
    original = scheduling.waiting.stop
    def publish():
        original()
        event.set()
    scheduling.waiting.stop = publish
    return event


def test_idle_fifo_close_wakes_worker_and_is_repeatable(monkeypatch):
    decoder = Decoder()
    scheduling = Scheduler(decoder, max_pending_requests=2, admission_policy='fifo',
                           max_waiting_input_bytes=4096)
    entered = threading.Event()
    original = scheduling.waiting.get_control
    def parked(**kwargs):
        entered.set()
        return original(**kwargs)
    monkeypatch.setattr(scheduling.waiting, 'get_control', parked)
    original_arrival = decoder.arrived
    scheduling.start()
    wait(entered)
    closing = call(scheduling.close)
    wait(closing.done)
    join(closing.thread)
    assert closing.error is None
    assert not scheduling.thread.is_alive() and scheduling._worker_done.is_set()
    assert scheduling.decoder is None and decoder.arrived is original_arrival
    scheduling.close()


def test_completed_backend_does_not_wait_for_its_logical_caller_or_release_credit():
    decoder = Decoder()
    scheduling = scheduler(decoder)
    closing = None
    closed_inside = False
    try:
        with scheduling._request_limit.request():
            scheduling.submit([1], 1, None, False, lambda _values: None)
            before = scheduling.queue_record()
            assert before['reserved_live_requests'] == 1
            assert before['charged_live_input_bytes'] > 0
            scheduling.request_runtime(lambda: decoder.controller.pending.put('accepted-before-close'))
            closing = call(scheduling.close)
            closed_inside = closing.done.wait(1)
            # Always leave the caller scope, even on the old deadlock path.
            if closed_inside:
                assert closing.error is None
                assert scheduling.pending_requests == 1
                assert scheduling.queue_record()['charged_live_input_bytes'] == before['charged_live_input_bytes']
                with pytest.raises(RuntimeError, match='closed|closing'):
                    scheduling.submit([2], 1, None, False, lambda _values: None)
        wait(closing.done)
        join(closing.thread)
        assert closed_inside, 'FIFO close waited for an already-retired backend caller'
        assert ('control', 'accepted-before-close') in decoder.events
        assert scheduling.pending_requests == 0
        assert scheduling.queue_record()['charged_live_input_bytes'] == 0
        assert scheduling.decoder is None and not scheduling.thread.is_alive()
    finally:
        decoder.release.set()
        if closing is not None:
            join(closing.thread)
        scheduling.close()


def test_control_accepted_after_a_boundary_is_drained_before_close(monkeypatch):
    decoder = Decoder()
    scheduling = Scheduler(decoder, max_streams=1, max_pending_requests=2,
                           admission_policy='fifo', max_waiting_input_bytes=4096)
    armed, parked, release = threading.Event(), threading.Event(), threading.Event()
    original = scheduling._execution_boundary
    def boundary():
        original()
        if armed.is_set() and not parked.is_set():
            parked.set()
            wait(release)
    monkeypatch.setattr(scheduling, '_execution_boundary', boundary)
    scheduling.start()
    closing = None
    try:
        with scheduling._request_limit.request():
            scheduling.submit([1], 1, None, False, lambda _values: None)
            armed.set()
            scheduling.waiting.wake()
            wait(parked)
            scheduling.request_runtime(lambda: decoder.controller.pending.put('last-accepted-control'))
            stopped = stop_published(scheduling)
            closing = call(scheduling.close)
            wait(stopped)
            release.set()
            closed_inside = closing.done.wait(1)
        wait(closing.done)
        join(closing.thread)
        assert closed_inside and closing.error is None
        assert ('control', 'last-accepted-control') in decoder.events
        assert decoder.controller.pending.empty()
        assert scheduling.pending_requests == 0
    finally:
        release.set()
        if closing is not None:
            join(closing.thread)
        scheduling.close()


def test_close_drains_active_work_and_retires_queued_cancellation():
    decoder = Decoder(paused=True)
    scheduling = scheduler(decoder)
    active = submitted(scheduling, 1, 3)
    wait(decoder.entered)
    queued = submitted(scheduling, 2, eligibility=lambda: 'client_input')
    with scheduling.waiting.not_empty:
        published = scheduling.waiting.not_empty.wait_for(lambda: scheduling.waiting.revision >= 2, 3)
    if not published:
        raise AssertionError('second accepted request was not published')
    stopped = stop_published(scheduling)
    closing = call(scheduling.close)
    try:
        wait(stopped)
        assert not closing.done.is_set()
        rejected = submitted(scheduling, 3)
        wait(rejected.done)
        join(rejected.thread)
        assert isinstance(rejected.error, RuntimeError)
    finally:
        decoder.release.set()
    for result in (active, queued, closing):
        wait(result.done)
        join(result.thread)
    assert active.error is None and active.output == [2, 2, 2]
    assert isinstance(queued.error, RequestCancelled) and queued.output == []
    assert closing.error is None
    assert ('admitted', 2) not in decoder.events and 2 in decoder.retired
    assert scheduling.pending_requests == 0
    assert not scheduling.waiting.requests(include_reserved=True)[1]


def test_caller_cancellation_retires_before_idle_close():
    decoder = Decoder()
    scheduling = scheduler(decoder)
    observed = threading.Event()
    output = []
    def emit(values):
        output.extend(values)
        if values:
            observed.set()
            return True
        return False
    request = call(lambda: scheduling.submit([4], 5, None, False, emit))
    try:
        wait(observed)
        closing = call(scheduling.close)
        for result in (request, closing):
            wait(result.done)
            join(result.thread)
        assert request.error is None and closing.error is None
        assert output == [5] and 4 in decoder.retired
        assert scheduling.pending_requests == 0
        assert not scheduling.thread.is_alive()
    finally:
        scheduling.close()


def test_concurrent_close_preserves_a_live_logical_credit_until_caller_exit():
    scheduling = scheduler(Decoder())
    closers = []
    completed_inside = False
    try:
        with scheduling._request_limit.request():
            scheduling.submit([1], 1, None, False, lambda _values: None)
            closers = [call(scheduling.close) for _ in range(2)]
            completed_inside = all(result.done.wait(1) for result in closers)
            if completed_inside:
                assert scheduling.pending_requests == 1
                assert scheduling.queue_record()['reserved_live_requests'] == 1
        for result in closers:
            wait(result.done)
            join(result.thread)
            assert result.error is None
        assert completed_inside
        scheduling.close()
        assert scheduling.pending_requests == 0
    finally:
        for result in closers:
            join(result.thread)
        scheduling.close()


def test_worker_failure_is_preserved_by_close_and_repeated_close():
    decoder = Decoder()
    scheduling = scheduler(decoder)
    failure = RuntimeError('controlled FIFO boundary failure')
    scheduling.request_runtime(lambda: decoder.controller.pending.put(failure))
    wait(scheduling._worker_done)
    for _ in range(2):
        with pytest.raises(RuntimeError, match='worker failed') as error:
            scheduling.close()
        assert error.value.__cause__ is failure
    assert scheduling.decoder is None and not scheduling.thread.is_alive()

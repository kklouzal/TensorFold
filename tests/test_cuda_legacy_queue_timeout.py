"""Legacy admission (Gemma 4 on eight V100s) bounds a request's wait for a free --parallel lane: one no lane took
within queue_timeout_s fails alone with a capacity refusal (HTTP 503), the lanes keep running, and a later request
is still served."""

import threading
import time

import pytest

from tensorfold.cleanup import finish
from tensorfold.cuda import scheduler as scheduling
from tensorfold.cuda.scheduler import Scheduler
from tensorfold.server.errors import CapacityError


class Busy:
    """One lane whose prompts fill until their caller leaves."""

    def __init__(self):
        self.filling, self.streams, self.admitted = [], {}, []

    def live(self):
        return len(self.filling)

    def admit(self, s):
        self.admitted.append(s.prompt)
        self.filling.append(s)

    def round(self):
        time.sleep(0.005)
        return []

    def cancel(self, streams):
        for s in streams:
            self.filling.remove(s)
            s.done = True
        return streams

    def finish(self, done):
        pass

    def drop(self):
        live, self.filling = self.filling, []
        return live


@pytest.fixture
def owner(monkeypatch):
    monkeypatch.setattr(scheduling, "TICK_S", 0.02)
    owners = []

    def start(decoder, **kw):
        owners.append(Scheduler(decoder, **kw).start())
        return owners[-1]

    yield start
    finish([o.close for o in owners])


def test_a_request_no_lane_takes_within_the_queue_timeout_gets_a_capacity_refusal(owner):
    dec = Busy()
    sched = owner(dec, max_streams=1, queue_timeout_s=0.2)
    leave = threading.Event()
    first = threading.Thread(target=sched.submit, args=([1], 4, None, True, lambda new: leave.is_set()))
    first.start()
    while not dec.admitted:
        time.sleep(0.005)
    began = time.monotonic()
    with pytest.raises(CapacityError, match="queue timeout"):
        sched.submit([2], 4, None, True, lambda new: False)
    assert 0.15 <= time.monotonic() - began < 2.0 and dec.admitted == [[1]]     # never took the lane
    leave.set()
    first.join(5)
    assert not first.is_alive()
    stats = sched.submit([3], 4, None, True, lambda new: True)                    # the lane is free again
    assert dec.admitted == [[1], [3]] and isinstance(stats, dict)


def test_legacy_admission_takes_a_queue_timeout_but_not_fifo_byte_limits():
    Scheduler(object(), max_streams=1, queue_timeout_s=900)
    with pytest.raises(ValueError, match="require FIFO admission"):
        Scheduler(object(), max_streams=1, max_waiting_input_bytes=1 << 20)


def test_gemma4_tp8_serve_options_take_only_the_queue_timeout():
    from types import SimpleNamespace as NS

    from tensorfold.serve_options import _fifo_options

    gemma = NS(model_type="gemma4")

    def args(**kw):
        base = dict(tp=8, max_waiting_input_bytes=None, max_waiting_input_bytes_per_principal=None,
                    queue_timeout_s=None, output_queue_tokens=None, retention_policy=None)
        return NS(**{**base, **kw})

    _fifo_options(args(queue_timeout_s=900.0), gemma, "cuda", None)
    with pytest.raises(ValueError, match="--queue-timeout-s only"):
        _fifo_options(args(max_waiting_input_bytes=1 << 20), gemma, "cuda", None)
    with pytest.raises(ValueError, match="FIFO input/deadline"):
        _fifo_options(args(tp=1, queue_timeout_s=900.0), gemma, "cuda", None)

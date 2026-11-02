"""Queue keepalive is outside the scheduler lock; no GPU/model/server work."""
import threading

import pytest

from tensorfold.cuda.turns import Turns
from tensorfold.server.cancellation import RequestCancelled


def test_wait_tick_does_not_hold_queue_lock_and_can_cancel(monkeypatch):
    turns = Turns()
    turns.busy = True
    events = []
    monkeypatch.setattr(turns.cv, 'wait', lambda timeout: events.append(('wait', timeout)))
    def tick():
        acquired = []
        def other_thread():
            with turns.cv:
                acquired.append(True)
        worker = threading.Thread(target=other_thread)
        worker.start()
        worker.join(timeout=1)
        assert not worker.is_alive() and acquired == [True]
        events.append('tick')
        return True
    with pytest.raises(RequestCancelled):
        turns.take(False, tick=tick)
    assert events == [('wait', 1.0), 'tick']
    assert turns.parked == turns.waiting == 0 and turns.busy


def test_available_turn_has_no_unnecessary_tick():
    turns = Turns()
    turns.take(False, tick=lambda: pytest.fail('IO on an immediately available turn'))
    assert turns.busy and turns.parked == turns.waiting == 0
    turns.give()
    assert not turns.busy

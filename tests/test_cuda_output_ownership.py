"""Real CPU producer/caller lifetimes, without model or GPU substitutes for numerics."""

import queue
import threading
from collections import deque

import pytest

from tensorfold.cuda.output_mailbox import OutputMailbox
from tensorfold.cuda.scheduler import Scheduler, Waiting
from tensorfold.cuda.streams import Stream
from tensorfold.server.errors import CapacityError
from tensorfold.server.request_limits import RequestLimit


def test_full_reply_backlog_keeps_progress_and_terminal_in_reserved_cells():
    mailbox = OutputMailbox(5)
    mailbox.put(("tokens", [1, 2]))
    mailbox.put(("tokens", [3, 4, 5]))
    for progress in range(1000):
        mailbox.put(("filled", progress))
    mailbox.put(("done", {"canonical": 5}))
    assert mailbox.record()["queued_token_ids"] == 5
    assert mailbox.record()["progress_cells"] == mailbox.record()["terminal_cells"] == 1
    assert mailbox.get_nowait() == ("tokens", [1, 2])
    assert mailbox.get_nowait() == ("tokens", [3, 4, 5])
    assert mailbox.get_nowait() == ("filled", 999)
    assert mailbox.get_nowait() == ("done", {"canonical": 5})
    with pytest.raises(queue.Empty):
        mailbox.get_nowait()
    assert mailbox.record()["consumer_dequeued_token_ids"] == 5


def test_coalesced_progress_cannot_overtake_previously_published_tokens():
    mailbox = OutputMailbox(3)
    mailbox.put(("filled", 1))
    mailbox.put(("tokens", [7]))
    mailbox.put(("filled", 2))
    assert mailbox.get_nowait() == ("tokens", [7])
    assert mailbox.get_nowait() == ("filled", 2)


def test_token_allowance_is_total_publication_not_released_consumer_credit():
    mailbox = OutputMailbox(2)
    mailbox.put(("tokens", [1, 2]))
    assert mailbox.get_nowait() == ("tokens", [1, 2])
    with pytest.raises(RuntimeError, match="reply token allowance"):
        mailbox.put(("tokens", [3]))
    failure = RuntimeError("owned failure after full output")
    mailbox.put(("error", failure))
    assert mailbox.get_nowait() == ("error", failure)


@pytest.mark.parametrize("after", [False, True])
def test_interrupted_token_publication_preserves_exact_owned_payload_and_failure(after):
    primary = KeyboardInterrupt("controlled output publication")
    class Interrupted(deque):
        def append(self, value):
            if after:
                super().append(value)
            raise primary
    mailbox = OutputMailbox(2)
    mailbox._chunks = Interrupted()
    with pytest.raises(KeyboardInterrupt) as caught:
        mailbox.put(("tokens", [7]))
    assert caught.value is primary
    assert mailbox.record()["published_token_ids"] == int(after)
    mailbox.put(("error", primary))
    if after:
        assert mailbox.get_nowait() == ("tokens", [7])
    assert mailbox.get_nowait() == ("error", primary)
    assert mailbox.record()["queued_token_ids"] == 0


def test_canonical_commit_precedes_callback_and_replay_does_not_recount():
    observed = []
    stream = Stream([1], 8, emit=lambda new: observed.append((list(new), stream.committed_output_tokens)))
    stream.take([2, 3])
    assert observed == [([2, 3], 2)]
    continued = stream.continued()
    continued.emit = lambda new: observed.append((list(new), continued.committed_output_tokens))
    continued.take([2])
    assert continued.committed_output_tokens == 2
    continued.take([3, 4])
    assert continued.committed_output_tokens == 3
    assert observed[-1] == ([4], 3)
    assert continued.stats()["committed_output_tokens"] == 3


class ImmediateDecoder:
    def __init__(self):
        self.streams, self.admitted = {}, []

    def live(self):
        return len(self.streams)

    def admit(self, stream):
        stream.sid = id(stream)
        self.streams[stream.sid] = stream
        self.admitted.append(stream)
        stream.take([stream.prompt[0]])

    def round(self):
        done = []
        for stream in list(self.streams.values()):
            if not stream.done:
                stream.take([stream.prompt[0]])
                if stream.done:
                    done.append(stream)
        return done

    def finish(self, done):
        for stream in done:
            self.streams.pop(stream.sid)

    def drop(self):
        result, self.streams = list(self.streams.values()), {}
        return result


@pytest.mark.parametrize("maximum,healthy_allowed", [(1, False), (2, True)])
def test_slow_caller_cannot_block_coordinator_or_escape_finite_owner_count(maximum, healthy_allowed):
    decoder = ImmediateDecoder()
    scheduler = Scheduler(decoder, max_streams=2, max_pending_requests=maximum).start()
    entered, release, retired = threading.Event(), threading.Event(), threading.Event()
    mailboxes, result, failures = [], [], []
    terminal = scheduler._terminal
    def finish(box, kind, value):
        mailboxes.append(box)
        terminal(box, kind, value)
        retired.set()
    scheduler._terminal = finish
    def slow(tokens):
        entered.set()
        assert release.wait(5), "CPU control actor was not released"
        return False
    def caller():
        try:
            result.append(scheduler.submit([11], 8, None, False, slow))
        except BaseException as error:
            failures.append(error)
    thread = threading.Thread(target=caller)
    try:
        thread.start()
        assert entered.wait(5) and retired.wait(5)
        assert thread.is_alive() and scheduler.pending_requests == 1
        assert decoder.admitted[0].committed_output_tokens == 8
        state = mailboxes[0].record()
        assert state["published_token_ids"] == 8 and state["queued_token_ids"] == 7
        assert state["consumer_dequeued_token_ids"] == 1
        if healthy_allowed:
            output = []
            stats = scheduler.submit([22], 3, None, False, lambda new: output.extend(new))
            assert output == [22] * 3 and stats["committed_output_tokens"] == 3
            assert scheduler.pending_requests == 1
        else:
            with pytest.raises(CapacityError):
                scheduler.submit([22], 3, None, False, lambda new: False)
        assert scheduler.decoder is decoder  # The blocked socket/callback owns CPU output, not the worker.
    finally:
        release.set()
        thread.join(5)
        scheduler.close()
    assert not thread.is_alive() and not failures
    assert scheduler.pending_requests == 0
    assert result[0]["output_ownership"]["published_token_ids"] == 8


def test_app_scope_is_shared_but_nested_direct_calls_get_independent_owners():
    limit = RequestLimit(2)
    with limit.request():
        with limit.inherited_request():
            assert limit.used == 1
    with limit.inherited_request():
        assert limit.used == 1
        with limit.inherited_request():
            assert limit.used == 2
    assert limit.used == 0


def test_profile_epoch_mismatch_is_rejected_before_any_decoder_slot():
    decoder = ImmediateDecoder()
    scheduler = Scheduler(decoder).start()
    scheduler._serving_profile_epoch = 2
    try:
        for epoch in (None, 0, 1):
            with pytest.raises(CapacityError, match="profile lease"):
                scheduler.submit([1], 1, None, False, lambda new: False, profile_epoch=epoch)
        assert decoder.admitted == []
        scheduler.submit([2], 1, None, False, lambda new: False, profile_epoch=2)
        assert decoder.admitted[0].profile_epoch == 2
    finally:
        scheduler.close()


def test_queued_old_epoch_is_rejected_before_decoder_admission():
    decoder = ImmediateDecoder()
    scheduler = Scheduler(decoder)
    stream, mailbox = Stream([1], 1, profile_epoch=0), OutputMailbox(1)
    scheduler.waiting.put((stream, mailbox))
    scheduler._serving_profile_epoch = 1
    assert scheduler._admit() == []
    kind, error = mailbox.get_nowait()
    assert kind == "error" and isinstance(error, CapacityError)
    assert decoder.admitted == [] and not scheduler.boxes
    scheduler.close()


@pytest.mark.parametrize("control", ["wake", "stop"])
@pytest.mark.parametrize("after", [False, True])
def test_control_markers_remain_bounded_across_interrupted_publication(control, after):
    waiting = Waiting()
    primary = KeyboardInterrupt("controlled queue publication")
    original_order, original_notify = waiting._order, waiting.not_empty.notify

    def interrupted(*_args):
        raise primary

    class InterruptedOrder:
        __next__ = staticmethod(interrupted)

    if after:
        waiting.not_empty.notify = interrupted
    else:
        waiting._order = InterruptedOrder()
    with pytest.raises(KeyboardInterrupt) as caught:
        getattr(waiting, control)()
    assert caught.value is primary
    waiting._order, waiting.not_empty.notify = original_order, original_notify
    for _ in range(100):
        getattr(waiting, control)()
    assert waiting.qsize() == 0  # Neither control is an admitted request/sequence.
    assert waiting._qsize() == 1
    assert not waiting.queue  # Two fixed cells cannot accumulate heap nodes after failure.
    item = waiting.get_nowait()
    assert waiting.is_wake(item) if control == "wake" else item is None
    if control == "stop":
        waiting.stop()
        assert waiting.empty()


def test_notification_failure_retry_wakes_the_actual_idle_consumer():
    waiting = Waiting()
    started, completed = threading.Event(), threading.Event()
    original_wait, original_notify = waiting.not_empty.wait, waiting.not_empty.notify
    received = []

    def idle_wait(*args, **kwargs):
        started.set()
        return original_wait(*args, **kwargs)

    def consume():
        received.append(waiting.get())
        completed.set()

    def failed_notify(*_args):
        raise MemoryError("controlled condition notification failure")

    waiting.not_empty.wait = idle_wait
    worker = threading.Thread(target=consume, daemon=True)
    worker.start()
    try:
        assert started.wait(2)
        waiting.not_empty.notify = failed_notify
        with pytest.raises(MemoryError):
            waiting.wake()
        assert not completed.is_set()
        waiting.not_empty.notify = original_notify
        waiting.wake()
        assert completed.wait(2)
        assert len(received) == 1 and waiting.is_wake(received[0])
    finally:
        waiting.not_empty.notify = original_notify
        waiting.stop()
        worker.join(2)
        assert not worker.is_alive()


def test_fixed_control_cells_preserve_request_and_wake_priority_order():
    waiting = Waiting()
    first = (Stream([1], 1, background=True), object())
    later = (Stream([2], 1, background=True), object())
    foreground = (Stream([3], 1), object())
    waiting.put(first)
    waiting.wake()
    waiting.put(later)
    waiting.put(foreground)
    waiting.stop()
    assert waiting.qsize() == 3 and waiting._qsize() == 5
    assert waiting.get_nowait() is foreground
    assert waiting.get_nowait() is first
    assert waiting.is_wake(waiting.get_nowait())
    assert waiting.get_nowait() is later
    assert waiting.get_nowait() is None

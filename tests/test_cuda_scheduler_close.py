"""Scheduler.close: the worker stops once idle and lets go of its decoder, so an engine's GPU memory can be freed."""

import gc
import weakref
import threading
from unittest.mock import patch

from tensorfold.cuda.memory_gate import NoRoom

from tensorfold.cuda.scheduler import Scheduler


class Idle:
    """A decoder with nothing to decode."""

    def live(self):
        return 0

    def round(self):
        return []

    def finish(self, done):
        pass

    def drop(self):
        return []


def test_close_stops_the_worker_and_frees_the_decoder():
    decoder = Idle()
    ref = weakref.ref(decoder)
    scheduler = Scheduler(decoder)
    scheduler.start()
    del decoder
    gc.collect()
    assert ref() is not None                             # the running worker holds it
    scheduler.close()
    assert not scheduler.thread.is_alive()
    gc.collect()
    assert ref() is None


def test_invalid_lane_counts_reject_before_hook_changes_or_actor_creation():
    decoder = Idle()
    original = decoder.arrived = object()
    for count in (0, -1, True, False, 1.5, "2", None):
        with patch.object(threading, "Thread", side_effect=AssertionError("invalid capacity created an actor")):
            try:
                Scheduler(decoder, max_streams=count)
            except ValueError as error:
                assert "positive integer" in str(error)
            else:
                raise AssertionError(f"invalid lane count accepted: {count!r}")
        assert decoder.arrived is original


def test_constructed_owner_does_not_launch_or_change_hook_and_closes_without_join():
    class Hook(Idle):
        def arrived(self):
            return False

    decoder = Hook()
    ref = weakref.ref(decoder)
    scheduler = Scheduler(decoder)
    assert not scheduler._start_attempted and not scheduler._arrival_applied
    assert decoder.arrived.__self__ is decoder
    try:
        scheduler.submit([1], 1, None, False, lambda new: False)
    except RuntimeError as error:
        assert "new requests are rejected" in str(error)
    else:
        raise AssertionError("an unstarted owner accepted a request")
    with patch.object(scheduler.thread, "join", side_effect=AssertionError("never-started actor joined")):
        scheduler.close()
    assert scheduler.decoder is None and scheduler._worker_done.is_set()
    del decoder
    gc.collect()
    assert ref() is None
    try:
        scheduler.start()
    except RuntimeError:
        pass
    else:
        raise AssertionError("a closed scheduler restarted")


def test_thread_construction_failure_has_no_decoder_side_effects():
    decoder = Idle()
    original = decoder.arrived = object()
    primary = KeyboardInterrupt("controlled actor construction failure")
    with patch.object(threading, "Thread", side_effect=primary):
        try:
            Scheduler(decoder)
        except BaseException as caught:
            assert caught is primary
        else:
            raise AssertionError("actor construction failure hidden")
    assert decoder.arrived is original


class Unused(Idle):
    def __init__(self):
        self.arrived = object()

    def live(self):
        raise AssertionError("aborted startup accessed the decoder")

    def drop(self):
        raise AssertionError("aborted startup cleaned an unused decoder")


def test_failed_start_before_spawn_retains_journal_until_actor_completion_is_proven():
    decoder = Unused()
    original = decoder.arrived
    scheduler = Scheduler(decoder)
    primary = KeyboardInterrupt("controlled pre-spawn interruption")
    with patch.object(scheduler.thread, "start", side_effect=primary):
        try:
            scheduler.start()
        except BaseException as caught:
            assert caught is primary
        else:
            raise AssertionError("startup failure hidden")
    try:
        scheduler.close()
    except RuntimeError:
        pass
    else:
        raise AssertionError("unsettled startup claimed joined completion")
    assert scheduler.decoder is decoder and scheduler._arrival_applied
    assert scheduler._closing and not scheduler._started_ok
    # The fixture settles the ambiguous launch. It must abort without touching
    # decoder state; the production owner stays retained until a real join.
    scheduler.thread.start()
    assert scheduler._worker_entered.wait(5)
    scheduler.close()
    assert scheduler.decoder is None and decoder.arrived is original
    assert scheduler._worker_done.is_set() and not scheduler.thread.is_alive()


def test_failed_start_after_actual_spawn_aborts_and_joins_without_decoder_access():
    decoder = Unused()
    original = decoder.arrived
    scheduler = Scheduler(decoder)
    primary = KeyboardInterrupt("controlled accepted startup interruption")
    real_start = scheduler.thread.start

    def accepted_then_raised():
        real_start()
        raise primary

    with patch.object(scheduler.thread, "start", accepted_then_raised):
        try:
            scheduler.start()
        except BaseException as caught:
            assert caught is primary
        else:
            raise AssertionError("accepted startup failure hidden")
    assert scheduler._worker_entered.wait(5)
    scheduler.close()
    assert scheduler.decoder is None and decoder.arrived is original
    assert scheduler._worker_done.is_set() and not scheduler.thread.is_alive()
    assert scheduler._start_error is primary and primary.__traceback__ is None


def test_native_spawn_before_thread_started_publication_retains_owner_until_retry_join():
    decoder = Unused()
    original = decoder.arrived
    scheduler = Scheduler(decoder)
    release = threading.Event()
    primary = KeyboardInterrupt("controlled interruption before native bootstrap publication")
    native_start = threading._start_new_thread

    def delayed_bootstrap(function, arguments):
        def bootstrap():
            if not release.wait(5):
                raise AssertionError("native startup release deadline")
            function(*arguments)
        native_start(bootstrap, ())
        raise primary

    try:
        with patch.object(threading, "_start_new_thread", delayed_bootstrap):
            try:
                scheduler.start()
            except BaseException as caught:
                assert caught is primary
            else:
                raise AssertionError("native startup interruption hidden")
        try:
            scheduler.close()
        except RuntimeError:
            pass
        else:
            raise AssertionError("an unpublished native actor was treated as absent")
        assert scheduler.decoder is decoder and scheduler._arrival_applied
        assert not scheduler._worker_done.is_set()
    finally:
        release.set()
        assert scheduler._worker_done.wait(5), "actual aborted native actor did not complete"
        scheduler.close()
    assert scheduler.decoder is None and decoder.arrived is original
    assert not scheduler.thread.is_alive()


def test_partial_hook_assignment_and_failed_restoration_remain_owned_for_retry():
    primary, cleanup = KeyboardInterrupt("hook assignment interrupted"), RuntimeError("hook restore failed")

    class Hooked(Idle):
        def __init__(self):
            self.value = object()
            self.failure = primary

        @property
        def arrived(self):
            return self.value

        @arrived.setter
        def arrived(self, value):
            self.value = value
            if self.failure is not None:
                raise self.failure

    decoder = Hooked()
    original = decoder.arrived
    scheduler = Scheduler(decoder)
    try:
        scheduler.start()
    except BaseException as caught:
        assert caught is primary
    else:
        raise AssertionError("partial hook assignment failure hidden")
    assert not scheduler._start_attempted and scheduler._arrival_applied
    decoder.failure = cleanup
    try:
        scheduler.close()
    except BaseException as caught:
        assert caught is cleanup
    else:
        raise AssertionError("failed hook restoration hidden")
    assert scheduler.decoder is decoder and scheduler._arrival_applied
    decoder.failure = None
    scheduler.close()
    assert scheduler.decoder is None and decoder.arrived is original


class Controlled(Idle):
    """Bounded CPU decoder; pause one round to control admission/close races."""

    def __init__(self, *, hold=False, fail_finish=False):
        self.streams = {}
        self.entered = threading.Event()
        self.release = threading.Event()
        self.first = True
        self.hold = hold
        self.fail_finish = fail_finish
        self.admitted = []

    def live(self):
        return sum(not s.done for s in self.streams.values())

    def admit(self, stream):
        if self.hold and self.live():
            raise NoRoom("controlled memory hold")
        stream.sid = len(self.admitted)
        self.admitted.append(stream.prompt[0])
        self.streams[stream.sid] = stream

    def round(self):
        if self.first:
            self.first = False
            self.entered.set()
            if not self.release.wait(5):
                raise RuntimeError("test decoder release deadline")
        done = []
        for stream in list(self.streams.values()):
            stream.take([stream.prompt[0]])
            if stream.done:
                done.append(stream)
        return done

    def finish(self, done):
        if self.fail_finish:
            raise RuntimeError("controlled finish failure")
        for stream in done:
            self.streams.pop(stream.sid)

    def drop(self):
        streams = list(self.streams.values())
        self.streams.clear()
        return streams


def client(scheduler, token, results, *, emit=None, background=False):
    output = []

    def run():
        try:
            stats = scheduler.submit([token], 3, None, False, emit or (lambda new: output.extend(new)),
                                     background=background)
            results[token] = (output, stats)
        except BaseException as error:
            results[token] = error

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def closer(scheduler, errors):
    started = threading.Event()
    original_stop = scheduler.waiting.stop

    def stop():
        original_stop()
        started.set()

    scheduler.waiting.stop = stop

    def run():
        try:
            scheduler.close()
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert started.wait(5), "close did not close admission at its controlled boundary"
    return thread


def join(thread):
    thread.join(5)
    assert not thread.is_alive(), "bounded CPU lifecycle did not complete"


def test_close_consumes_marker_while_live_then_drains_and_rejects_new_submissions():
    decoder = Controlled()
    scheduler = Scheduler(decoder, max_streams=2)
    scheduler.start()
    results, errors = {}, []
    running = client(scheduler, 1, results)
    try:
        assert decoder.entered.wait(5)
        closing = closer(scheduler, errors)
        with scheduler._state_lock:
            assert scheduler._closing
        assert closing.is_alive() and scheduler.decoder is decoder
        rejected = client(scheduler, 2, results)
        join(rejected)
        assert isinstance(results[2], RuntimeError) and "closing" in str(results[2])
    finally:
        decoder.release.set()
        scheduler.close()
    join(running)
    join(closing)
    assert not errors and results[1][0] == [1, 1, 1]
    assert scheduler._stopping and scheduler._worker_done.is_set()
    assert scheduler.decoder is None and not scheduler.boxes and scheduler.held is None
    scheduler.close()


def test_close_drains_held_and_queued_requests_in_priority_arrival_order():
    decoder = Controlled(hold=True)
    scheduler = Scheduler(decoder, max_streams=2)
    scheduler.start()
    results, errors = {}, []
    threads = [client(scheduler, 1, results)]
    assert decoder.entered.wait(5)
    accepted = threading.Event()
    original_put = scheduler.waiting.put

    def put(item, *args, **kwargs):
        original_put(item, *args, **kwargs)
        accepted.set()

    scheduler.waiting.put = put
    try:
        for token, background in ((2, True), (3, False), (4, False)):
            accepted.clear()
            threads.append(client(scheduler, token, results, background=background))
            assert accepted.wait(5)
        closing = closer(scheduler, errors)
    finally:
        decoder.release.set()
        scheduler.close()
    for thread in threads:
        join(thread)
    join(closing)
    assert not errors and decoder.admitted == [1, 3, 4, 2]
    assert all(results[token][0] == [token]*3 for token in (1, 2, 3, 4))
    assert scheduler.held is None and not scheduler.boxes


def test_interrupted_emitter_cancels_at_next_round_and_close_still_drains():
    class CancelDecoder(Controlled):
        def __init__(self):
            super().__init__()
            self.next_round = threading.Event()
            self.next_release = threading.Event()
            self.calls = 0

        def round(self):
            self.calls += 1
            if self.calls == 2:
                self.next_round.set()
                assert self.next_release.wait(5)
            return super().round()

    decoder = CancelDecoder()
    scheduler = Scheduler(decoder)
    scheduler.start()
    results = {}

    def emit(new):
        raise KeyboardInterrupt("controlled caller interruption")

    running = client(scheduler, 1, results, emit=emit)
    assert decoder.entered.wait(5)
    decoder.release.set()
    join(running)
    assert decoder.next_round.wait(5)
    decoder.next_release.set()
    scheduler.close()
    assert isinstance(results[1], KeyboardInterrupt)
    assert not scheduler.thread.is_alive() and not decoder.streams
    assert decoder.calls == 2


def test_interrupted_accepted_publication_cancels_before_caller_waits():
    class Accepted(Controlled):
        def admit(self, stream):
            self.accepted_stream = stream
            super().admit(stream)

    decoder = Accepted()
    scheduler = Scheduler(decoder)
    scheduler.start()
    original = scheduler.waiting.put
    primary = KeyboardInterrupt("interrupted after accepted publication")

    def published_then_interrupted(item, *args, **kwargs):
        original(item, *args, **kwargs)
        raise primary

    scheduler.waiting.put = published_then_interrupted
    caught = None
    try:
        scheduler.submit([71], 100, None, False, lambda new: False)
    except BaseException as error:
        caught = error
    finally:
        decoder.release.set()
        scheduler.close()
    assert caught is primary
    assert decoder.admitted == [71]
    assert len(decoder.accepted_stream.out) == 1
    assert not scheduler.thread.is_alive() and not decoder.streams


def test_worker_finish_failure_replies_to_accepted_callers_and_close_reports_primary():
    decoder = Controlled(fail_finish=True)
    scheduler = Scheduler(decoder)
    scheduler.start()
    results, errors = {}, []
    running = client(scheduler, 1, results)
    assert decoder.entered.wait(5)
    closing = closer(scheduler, errors)
    decoder.release.set()
    join(running)
    join(closing)
    assert isinstance(results[1], RuntimeError) and "finish failure" in str(results[1])
    assert len(errors) == 1 and errors[0].__cause__ is scheduler._failure
    assert scheduler._worker_done.is_set() and not scheduler.boxes and scheduler.held is None


def test_clean_fatal_shutdown_drops_foreign_decoder_traceback_owners():
    decoder = Controlled(fail_finish=True)
    ref = weakref.ref(decoder)
    scheduler = Scheduler(decoder)
    scheduler.start()
    results = {}
    running = client(scheduler, 1, results)
    assert decoder.entered.wait(5)
    decoder.release.set()
    join(running)
    try:
        scheduler.close()
    except RuntimeError as error:
        assert error.__cause__ is scheduler._failure
    else:
        raise AssertionError("fatal shutdown was reported as success")
    assert scheduler._failure.__traceback__ is None
    assert scheduler.decoder is None
    del decoder
    gc.collect()
    assert ref() is None


def test_failed_background_handoff_and_foreign_cleanup_still_reply_to_every_owner():
    class Primary(RuntimeError):
        def add_note(self, note):
            raise AssertionError("foreign add_note must not run")

    class Cleanup(RuntimeError):
        def __repr__(self):
            raise AssertionError("foreign cleanup repr must not run")

    class YieldFailure(Controlled):
        def finish(self, done):
            if any(s.background and not s.done for s in done):
                raise Primary("controlled preemption failure")
            super().finish(done)

        def drop(self):
            super().drop()
            raise Cleanup("controlled cleanup failure")

    decoder = YieldFailure()
    scheduler = Scheduler(decoder, max_streams=1)
    scheduler.start()
    results, errors = {}, []
    background = client(scheduler, 1, results, background=True)
    assert decoder.entered.wait(5)
    accepted = threading.Event()
    original_put = scheduler.waiting.put

    def put(item, *args, **kwargs):
        original_put(item, *args, **kwargs)
        accepted.set()

    scheduler.waiting.put = put
    foreground = client(scheduler, 2, results)
    assert accepted.wait(5)
    closing = closer(scheduler, errors)
    decoder.release.set()
    for thread in (background, foreground, closing):
        join(thread)
    assert all(isinstance(results[token], Primary) for token in (1, 2))
    assert isinstance(scheduler._cleanup_failure, Cleanup)
    assert scheduler.decoder is decoder              # failed cleanup keeps resource ownership explicit
    assert len(errors) == 1 and errors[0].__cause__ is scheduler._failure
    assert "cleanup also failed" in scheduler._failure.__notes__[0]
    assert scheduler._worker_done.is_set() and not scheduler.boxes and scheduler.held is None


def _stop_publication_failure(after_insert):
    decoder = Controlled()
    scheduler = Scheduler(decoder, max_streams=1)
    scheduler.start()
    results, errors = {}, []
    running = client(scheduler, 1, results)
    assert decoder.entered.wait(5)
    original_stop = scheduler.waiting.stop
    primary = RuntimeError("controlled stop publication failure")

    def interrupted():
        if after_insert:
            original_stop()
        raise primary

    scheduler.waiting.stop = interrupted
    try:
        try:
            scheduler.close()
        except RuntimeError as caught:
            assert caught is primary
        else:
            raise AssertionError("stop publication failure was hidden")
        assert scheduler._closing and not scheduler._stop_published
        scheduler.waiting.stop = original_stop
        decoder.release.set()

        def retry():
            try:
                scheduler.close()
            except BaseException as error:
                errors.append(error)

        closing = threading.Thread(target=retry, daemon=True)
        closing.start()
        join(closing)
        join(running)
        assert not errors and results[1][0] == [1, 1, 1]
        assert scheduler._worker_done.is_set() and scheduler.decoder is None
    finally:
        decoder.release.set()
        scheduler.waiting.stop = original_stop
        original_stop()                       # bounded fixture cleanup even if retry assertion failed
        scheduler.thread.join(5)


def test_failed_stop_publication_before_insert_remains_retryable():
    _stop_publication_failure(False)


def test_ambiguous_stop_publication_after_insert_retains_drain_semantics():
    _stop_publication_failure(True)

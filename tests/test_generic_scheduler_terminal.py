"""Production Python ownership protocol; explicit fake GPU drain, no native imports."""

import gc
import importlib.util
from pathlib import Path
import queue
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref


ROOT = Path(__file__).parents[1]
lane = ModuleType("tensorfold.engine.lane_engine")
lane.LaneStream = SimpleNamespace


def load(name, relative, replacements):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module, **replacements}):
        spec.loader.exec_module(module)
    return module


prompt = load("terminal_prompt_fill", "src/tensorfold/server/prompt_fill.py", {lane.__name__: lane})
source = load(
    "terminal_scheduler",
    "src/tensorfold/server/scheduler.py",
    {lane.__name__: lane, "tensorfold.server.prompt_fill": prompt},
)
Scheduler, ChatJob = source.Scheduler, source.ChatJob


class Fatal(BaseException):
    pass


class Engine:
    active_count = 0
    prefill_chunks = 0

    def __init__(self):
        self.events = []
        self.prefill_guard = None
        self.drain_error = None

    def drain(self):
        self.events.append("drain")
        if self.drain_error is not None:
            raise self.drain_error

    def reset(self):
        assert "drain" in self.events
        self.events.append("reset")

    def discard_stream(self, stream):
        stream.finished = True


def job(name):
    return ChatJob(name, [1, 2], 3, 0.0)


def run_thread(fn):
    result = []

    def run():
        try:
            result.append(fn())
        except BaseException as error:
            result.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, result


def joined(thread):
    thread.join(3)
    assert not thread.is_alive(), "bounded fixture thread did not finish"


class TerminalProtocol(unittest.TestCase):
    def scheduler(self, engine=None):
        return Scheduler(engine or Engine(), lanes=3, eos_ids=frozenset())

    def terminal_once(self, value):
        self.assertTrue(value.done.is_set())
        self.assertIsNotNone(value.error)
        self.assertIsNone(value.chunks.get_nowait())
        with self.assertRaises(queue.Empty):
            value.chunks.get_nowait()

    def test_stop_before_start_finishes_accepted_jobs_and_saves_after_drain(self):
        scheduler = self.scheduler()
        values = [job("a"), job("b")]
        for value in values:
            scheduler.submit(value)
        scheduler.on_stop = lambda: scheduler.engine.events.append("save")
        scheduler.stop(3)
        self.assertEqual(scheduler.engine.events, ["drain", "drain", "reset", "save"])
        for value in values:
            self.terminal_once(value)
        self.assertFalse(scheduler._accepted)
        self.assertFalse(scheduler._shutdown_owners)
        scheduler.stop(3)
        with self.assertRaises(RuntimeError):
            scheduler.submit(job("late"))

    def test_fatal_handoffs_fail_queued_held_active_filling_and_lost_job(self):
        scheduler = self.scheduler()
        values = [job(str(i)) for i in range(5)]
        for value in values:
            scheduler.submit(value)
        saved = []
        scheduler.on_stop = lambda: saved.append(True)
        fatal = Fatal("controlled admission failure")

        def loop():
            scheduler._queue.get_nowait()  # lost in a fallible admission handoff
            scheduler._held = scheduler._queue.get_nowait()
            active = scheduler._queue.get_nowait()
            active.stream = SimpleNamespace(finished=False, history_checkpoints=[object()], proposer=object())
            scheduler._jobs[active.job_id] = active
            filling = prompt.Filling(scheduler._queue.get_nowait())
            scheduler._fills.append(filling)
            scheduler._owned_fills[id(filling)] = filling
            raise fatal

        scheduler._loop = loop
        scheduler.start()
        with self.assertRaises(RuntimeError) as outcome:
            scheduler.stop(3)
        self.assertIs(outcome.exception.__cause__, fatal)
        for value in values:
            self.terminal_once(value)
            self.assertEqual(str(value.error), "controlled admission failure")
        self.assertFalse(saved)
        self.assertEqual(scheduler.engine.events, ["drain", "drain", "reset"])
        self.assertFalse(scheduler._jobs or scheduler._fills or scheduler._owned_fills)

    def test_stop_timeout_retains_live_owner_and_later_reaps(self):
        scheduler = self.scheduler()
        entered, release = threading.Event(), threading.Event()
        value = job("blocked")
        scheduler.submit(value)
        scheduler._loop = lambda: (entered.set(), release.wait(3))
        scheduler.start()
        self.assertTrue(entered.wait(3))
        with self.assertRaises(TimeoutError):
            scheduler.stop(0.01)
        self.assertTrue(scheduler._thread.is_alive())
        self.assertIs(scheduler._accepted[id(value)], value)
        self.assertFalse(scheduler.engine.events)
        release.set()
        scheduler.stop(3)
        self.terminal_once(value)

    def test_acceptance_is_atomic_with_stop_and_rejects_late_work(self):
        scheduler = self.scheduler()
        publication, release, loop_release = (threading.Event() for _ in range(3))
        put = scheduler._queue.put

        def blocked_put(value, *args, **kwargs):
            publication.set()
            assert release.wait(3)
            put(value, *args, **kwargs)

        scheduler._queue.put = blocked_put
        scheduler._loop = lambda: loop_release.wait(3)
        scheduler.start()
        value = job("accepted")
        submitting, submitted = run_thread(lambda: scheduler.submit(value))
        self.assertTrue(publication.wait(3))
        stopping, stopped = run_thread(lambda: scheduler.stop(3))
        self.assertFalse(scheduler._stop.is_set())  # acceptance still holds the state lock
        release.set()
        joined(submitting)
        loop_release.set()
        joined(stopping)
        self.assertEqual(submitted, [None])
        self.assertEqual(stopped, [None])
        self.terminal_once(value)
        with self.assertRaises(RuntimeError):
            scheduler.submit(job("rejected"))

    def test_active_rpc_baseexception_finishes_rpc_and_jobs(self):
        scheduler = self.scheduler()
        accepted = threading.Event()
        put = scheduler._engine_calls.put
        scheduler._engine_calls.put = lambda item: (put(item), accepted.set())
        scheduler._loop = lambda: (accepted.wait(3), scheduler._run_engine_call())
        value = job("queued")
        scheduler.submit(value)
        scheduler.start()

        def callback(_):
            raise Fatal("RPC failure")

        calling, called = run_thread(lambda: scheduler.on_engine(callback, timeout=3))
        joined(calling)
        with self.assertRaises(RuntimeError):
            scheduler.stop(3)
        self.assertIsInstance(called[0], source.RoundError)
        self.assertEqual(str(called[0]), "RPC failure")
        self.terminal_once(value)
        self.assertFalse(scheduler._engine_waiters)

    def test_drain_failure_retains_owner_and_bypasses_foreign_format_hooks(self):
        class Foreign(Fatal):
            def __str__(self):
                raise AssertionError("foreign formatter invoked")

            def add_note(self, note):
                raise AssertionError("foreign note hook invoked")

        engine = Engine()
        engine.drain_error = RuntimeError("cannot drain")
        scheduler = self.scheduler(engine)
        value = job("owned")
        scheduler.submit(value)
        failure = Foreign("primary failure")

        def loop():
            raise failure

        scheduler._loop = loop
        scheduler.start()
        with self.assertRaises(RuntimeError) as outcome:
            scheduler.stop(3)
        self.assertIs(outcome.exception.__cause__, failure)
        self.assertIs(scheduler._cleanup_failure, engine.drain_error)
        self.assertTrue(scheduler._shutdown_owners)
        self.assertEqual(engine.events, ["drain"])
        self.assertEqual(str(value.error), "primary failure")
        self.assertIn("owned state remains retained", failure.__notes__[0])

    def test_successful_cleanup_releases_worker_failure_frames(self):
        class Payload:
            pass

        references = []
        failure = Fatal("retire frame")
        scheduler = self.scheduler()

        def loop():
            payload = Payload()
            references.append(weakref.ref(payload))
            raise failure

        scheduler._loop = loop
        scheduler.start()
        with self.assertRaises(RuntimeError):
            scheduler.stop(3)
        gc.collect()
        self.assertIsNone(references[0]())
        self.assertIsNone(failure.__traceback__)

    def test_open_fill_failure_retains_and_releases_reservation_after_drain(self):
        scheduler = self.scheduler()
        held = object()
        ended = []
        scheduler.prompt_memory = SimpleNamespace(begin=lambda *a, **k: held, end=lambda value: ended.append(value))

        def fail(_):
            raise Fatal("fill admission failure")

        scheduler.engine.prompt_chunks = fail
        value = job("opening")
        scheduler.submit(value)
        scheduler._loop = lambda: scheduler._open_job(scheduler._queue.get_nowait())
        scheduler.start()
        with self.assertRaises(RuntimeError):
            scheduler.stop(3)
        self.terminal_once(value)
        self.assertEqual(ended, [held])
        self.assertFalse(scheduler._owned_fills)

    def test_failed_preemption_never_saves_or_publishes_success(self):
        scheduler = self.scheduler()
        value = job("preempting")
        scheduler.submit(value)
        saved, closed = [], []

        class Steps:
            def close(self):
                closed.append(True)
                if len(closed) == 1:
                    raise Fatal("generator close failure")

        filling = prompt.Filling(value, steps=Steps())
        scheduler._fills.append(filling)
        scheduler._owned_fills[id(filling)] = filling
        scheduler._keep_checkpoints = lambda *args: saved.append(True)
        scheduler._loop = lambda: scheduler._preempt_filling(filling)
        scheduler.start()
        with self.assertRaises(RuntimeError):
            scheduler.stop(3)
        self.terminal_once(value)
        self.assertFalse(value.preempted or saved)
        self.assertEqual(closed, [True, True])

    def test_priority_and_exactly_once_completion_remain(self):
        scheduler = self.scheduler()
        values = [job("background"), job("first"), job("second")]
        values[0].background = True
        for value in values:
            scheduler.submit(value)
        self.assertEqual([scheduler._queue.get_nowait().job_id for _ in values], ["first", "second", "background"])
        for value in values:
            scheduler._finish_cancelled(value)
            scheduler._finish(value)
            self.terminal_once(value)
        scheduler.stop(3)

    def test_generator_cleanup_enqueues_work_that_drains_before_reset(self):
        class CleanupEngine(Engine):
            pending = False

            def drain(self):
                super().drain()
                self.pending = False

            def reset(self):
                self.assertion = not self.pending
                assert self.assertion, "cleanup copy was released before its native work ended"
                super().reset()

        engine = CleanupEngine()
        scheduler = self.scheduler(engine)
        value = job("copy-on-close")
        scheduler.submit(value)

        class Steps:
            def close(self):
                engine.pending = True
                engine.events.append("cleanup_enqueue")

        filling = prompt.Filling(value, steps=Steps())
        scheduler._owned_fills[id(filling)] = filling

        def loop():
            raise Fatal("terminal failure before cleanup")

        scheduler._loop = loop
        scheduler.start()
        with self.assertRaises(RuntimeError):
            scheduler.stop(3)
        self.assertEqual(engine.events, ["drain", "cleanup_enqueue", "drain", "reset"])
        self.assertFalse(engine.pending or scheduler._shutdown_owners)

    def test_corrupted_public_join_state_never_releases_live_engine(self):
        scheduler = self.scheduler()
        entered, release = threading.Event(), threading.Event()
        value = job("independent-completion")
        scheduler.submit(value)
        scheduler._loop = lambda: (entered.set(), release.wait(3))
        scheduler.start()
        self.assertTrue(entered.wait(1))
        try:
            with (
                patch.object(scheduler._thread, "is_alive", return_value=False),
                patch.object(scheduler._thread, "join"),
            ):
                with self.assertRaises(TimeoutError):
                    scheduler.stop(0.01)
            self.assertFalse(scheduler.engine.events)
            self.assertIs(scheduler._accepted[id(value)], value)
            self.assertFalse(scheduler._work.retired.is_set())
        finally:
            release.set()
            scheduler.stop(3)
        self.terminal_once(value)

    def test_start_accepted_then_interrupted_is_cancelled_before_engine_entry(self):
        scheduler = self.scheduler()
        value = job("startup-owner")
        scheduler.submit(value)
        old_thread = scheduler._thread
        start = old_thread.start
        primary = KeyboardInterrupt("after native bootstrap")

        def interrupted():
            start()
            raise primary

        with patch.object(old_thread, "start", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt) as result:
                scheduler.start()
        self.assertIs(result.exception, primary)
        self.assertTrue(scheduler._work.cancelled_before_entry)
        self.assertFalse(scheduler.engine.events)
        with self.assertRaises(RuntimeError):
            scheduler.start()
        scheduler.stop(3)
        old_thread.join(3)
        self.assertFalse(old_thread.is_alive())
        self.assertIsNot(old_thread, scheduler._thread)
        self.terminal_once(value)
        self.assertEqual(scheduler.engine.events, ["drain", "drain", "reset"])

    def test_malformed_failure_notes_preserve_primary_and_cleanup(self):
        engine = Engine()
        cleanup = engine.drain_error = RuntimeError("drain failed")
        scheduler = self.scheduler(engine)
        primary = Fatal("primary")
        primary.__notes__ = 123

        def fail():
            raise primary

        scheduler._loop = fail
        scheduler.start()
        with self.assertRaises(RuntimeError) as result:
            scheduler.stop(3)
        self.assertIs(result.exception.__cause__, primary)
        self.assertIsInstance(scheduler._cleanup_failure, BaseExceptionGroup)
        self.assertIs(scheduler._cleanup_failure.exceptions[0], cleanup)
        self.assertIsInstance(scheduler._cleanup_failure.exceptions[1], TypeError)
        self.assertTrue(scheduler._shutdown_owners)
        self.assertTrue(scheduler._work.retired.is_set())


if __name__ == "__main__":
    unittest.main()

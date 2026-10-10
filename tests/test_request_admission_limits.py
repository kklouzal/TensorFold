"""Real Python admission/worker lifetimes; numerical engine boundaries are explicit fixtures."""

import importlib.util
import gc
import inspect
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from tensorfold.cuda.scheduler import Scheduler as CudaScheduler
from tensorfold.server.errors import CapacityError
from tensorfold.server.request_limits import RequestLimit, optional_limit
from tensorfold.serve_options import check, check_numbers


ROOT = Path(__file__).resolve().parents[1]


def generic_source():
    """Execute the complete production scheduler with only its numerical LaneStream import replaced."""
    lane = ModuleType("tensorfold.engine.lane_engine")
    lane.LaneStream = SimpleNamespace
    name = "request_limit_scheduler_control"
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/tensorfold/server/scheduler.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {lane.__name__: lane, name: module}):
        # PromptFill imports the same numerical boundary; retain its real implementation.
        spec.loader.exec_module(module)
    return module


GENERIC = generic_source()


def actor(call):
    results = []

    def run():
        try:
            results.append(call())
        except BaseException as error:
            results.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, results


def joined(thread):
    thread.join(5)
    if thread.is_alive():
        raise AssertionError("control actor did not retire")


class FailingDict(dict):
    """One deterministic before/after publication failure, then ordinary operations."""
    def __init__(self, *, after):
        super().__init__()
        self.after, self.error = after, KeyboardInterrupt("controlled registry publication")

    def __setitem__(self, key, value):
        error, self.error = self.error, None
        if error is not None and not self.after:
            raise error
        super().__setitem__(key, value)
        if error is not None:
            raise error

    def setdefault(self, key, value):
        error, self.error = self.error, None
        if error is not None and not self.after:
            raise error
        result = super().setdefault(key, value)
        if error is not None:
            raise error
        return result


class FailingSet(set):
    def __init__(self, *, after):
        super().__init__()
        self.after, self.error = after, KeyboardInterrupt("controlled set publication")

    def add(self, value):
        error, self.error = self.error, None
        if error is not None and not self.after:
            raise error
        super().add(value)
        if error is not None:
            raise error


class Decoder:
    """Controlled decoder boundary, never imports or exercises a numerical SDK."""
    def __init__(self):
        self.streams = {}
        self.entered, self.release_round = threading.Event(), threading.Event()
        self.failure = None
        self.cleaned = threading.Event()

    def live(self):
        return len(self.streams)

    def admit(self, stream):
        self.streams[id(stream)] = stream

    def round(self):
        self.entered.set()
        for stream in list(self.streams.values()):
            stream.take([7])
        if not self.release_round.wait(5):
            raise RuntimeError("control round release deadline")
        if self.failure is not None:
            raise self.failure
        return [stream for stream in self.streams.values() if stream.done]

    def finish(self, streams):
        for stream in streams:
            self.streams.pop(id(stream), None)

    def drop(self):
        streams = list(self.streams.values())
        self.streams.clear()
        self.cleaned.set()
        return streams


class RequestLimitTests(unittest.TestCase):
    def test_configured_method_has_no_app_cycle_and_releases_model_without_gc(self):
        class Model:
            pass

        class App:
            def run(self, body, *, prepared=None):
                return body

        enabled = gc.isenabled()
        gc.disable()
        try:
            owner, model = App(), Model()
            owner._model, owner._request_limit = model, RequestLimit(1)
            original = inspect.signature(owner.run)
            owner.run = owner._request_limit.wrap(owner.run)
            self.assertEqual(inspect.signature(owner.run), original)
            self.assertIs(owner.run.__wrapped__, App.run)
            self.assertEqual(owner.run("exact"), "exact")
            app_ref, model_ref, borrowed = weakref.ref(owner), weakref.ref(model), owner.run
            del model, owner
            self.assertIsNone(app_ref())
            self.assertIsNone(model_ref())
            with self.assertRaisesRegex(RuntimeError, "retired"):
                borrowed("late")
        finally:
            if enabled:
                gc.enable()

    def test_wrapped_active_call_keeps_app_strong_until_its_scope_retires(self):
        entered, leave = threading.Event(), threading.Event()

        class App:
            def chat(self):
                entered.set()
                if not leave.wait(5):
                    raise RuntimeError("active wrapper control deadline")
                return "exact"

        enabled = gc.isenabled()
        gc.disable()
        try:
            owner = App()
            limit = owner._request_limit = RequestLimit(1)
            owner.chat = limit.wrap(owner.chat)
            app_ref, borrowed = weakref.ref(owner), owner.chat
            thread, result = actor(borrowed)
            self.assertTrue(entered.wait(5))
            del owner
            self.assertIsNotNone(app_ref())
            self.assertEqual(limit.used, 1)
            leave.set()
            joined(thread)
            self.assertEqual(result, ["exact"])
            self.assertEqual(limit.used, 0)
            self.assertIsNone(app_ref())
        finally:
            leave.set()
            if enabled:
                gc.enable()

    def test_strict_counts_and_early_cuda_scope(self):
        for name in ("max_http_connections", "max_pending_requests", "max_engine_calls"):
            for value in (0, -1, True, False, 1.5, "2"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    check_numbers(SimpleNamespace(**{name: value}))
            for value in (None, 1, 20):
                check_numbers(SimpleNamespace(**{name: value}))
                self.assertEqual(optional_limit(value, name), value)
        with self.assertRaises(ValueError):
            RequestLimit(None)
        family = SimpleNamespace(package=SimpleNamespace())
        with self.assertRaisesRegex(ValueError, "MLX only"):
            check(SimpleNamespace(max_engine_calls=1), family, "cuda")

    def test_caller_cancellation_keeps_worker_and_releases_exactly_once(self):
        limit, work = RequestLimit(1), object()
        with self.assertRaises(KeyboardInterrupt):
            with limit.request():
                limit.borrow(work)
                raise KeyboardInterrupt("client canceled")
        self.assertEqual(limit.used, 1)
        with self.assertRaises(CapacityError):
            with limit.request():
                pass
        limit.release(work)
        limit.release(work)
        self.assertEqual(limit.used, 0)

    def test_request_publication_failures_roll_back_before_and_after_mutation(self):
        for field, replacement in (("_owners", FailingSet), ("_callers", FailingDict)):
            for after in (False, True):
                with self.subTest(field=field, after=after):
                    limit = RequestLimit(1)
                    setattr(limit, field, replacement(after=after))
                    with self.assertRaises(KeyboardInterrupt):
                        with limit.request():
                            self.fail("failed publication entered the request")
                    self.assertEqual(limit.used, 0)
                    self.assertFalse(limit._callers)
                    with limit.request():
                        self.assertEqual(limit.used, 1)

    def test_request_append_failure_and_interrupted_cleanup_reclaim_native_scope(self):
        class Stack(list):
            failure = "append"

            def append(self, owner):
                super().append(owner)
                if self.failure == "append":
                    self.failure = None
                    raise KeyboardInterrupt("stack append interrupted after publication")

            def pop(self):
                if self.failure == "pop":
                    self.failure = None
                    raise KeyboardInterrupt("retirement interrupted before stack removal")
                return super().pop()

        limit, identity = RequestLimit(1), threading.get_ident()
        limit._callers[identity] = Stack()
        with self.assertRaises(KeyboardInterrupt):
            with limit.request():
                pass
        self.assertEqual(limit.used, 0)
        self.assertFalse(limit._callers)
        stack = Stack()
        stack.failure = "pop"
        limit._callers[identity] = stack
        with self.assertRaises(KeyboardInterrupt):
            with limit.request():
                pass
        self.assertEqual(limit.used, 0)
        self.assertFalse(limit._callers)
        with limit.request():
            self.assertEqual(limit.used, 1)

    def test_admission_lock_exit_interruption_cannot_lose_published_owner(self):
        class Lock:
            def __init__(self):
                self.lock, self.fail = threading.RLock(), True

            def __enter__(self):
                return self.lock.__enter__()

            def __exit__(self, *args):
                self.lock.__exit__(*args)
                if self.fail:
                    self.fail = False
                    raise KeyboardInterrupt("interrupt after registration lock exit")

        limit = RequestLimit(1)
        limit._lock = Lock()
        with self.assertRaises(KeyboardInterrupt):
            with limit.request():
                self.fail("interrupted request entered")
        self.assertEqual(limit.used, 0)
        self.assertFalse(limit._callers)
        limit = RequestLimit(1)
        limit._lock = Lock()
        with self.assertRaises(KeyboardInterrupt):
            limit.borrow(object())
        self.assertEqual(limit.used, 0)
        self.assertFalse(limit._workers)

    def test_worker_registration_rolls_back_partial_owner_and_worker_mutations(self):
        for field, replacement in (("_owners", FailingSet), ("_workers", FailingDict)):
            for after in (False, True):
                with self.subTest(field=field, after=after):
                    limit = RequestLimit(1)
                    setattr(limit, field, replacement(after=after))
                    with self.assertRaises(KeyboardInterrupt):
                        limit.borrow(object())
                    self.assertEqual(limit.used, 0)
                    self.assertFalse(limit._workers)
        limit = RequestLimit(1)
        with limit.request():
            owner = limit._callers[threading.get_ident()][-1]
            owner.workers = FailingSet(after=True)
            with self.assertRaises(KeyboardInterrupt):
                limit.borrow(object())
            self.assertEqual(limit.used, 1)
            self.assertFalse(owner.workers)
            self.assertFalse(limit._workers)

    def test_worker_retirement_interruption_keeps_retryable_journal(self):
        class Registry(dict):
            failed = False

            def pop(self, *args):
                if not self.failed:
                    self.failed = True
                    raise KeyboardInterrupt("retirement journal interrupted")
                return super().pop(*args)

        limit, work = RequestLimit(1), object()
        limit.borrow(work)
        limit._workers = Registry(limit._workers)
        with self.assertRaises(KeyboardInterrupt):
            limit.release(work)
        self.assertEqual(limit.used, 0)
        self.assertFalse(limit._workers)
        limit.release(work)

    def test_request_cleanup_failure_preserves_original_error_and_native_scope_retirement(self):
        class Stack(list):
            def pop(self):
                raise OSError("controlled request cleanup failure")

        limit, primary = RequestLimit(1), KeyboardInterrupt("original caller error")
        limit._callers[threading.get_ident()] = Stack()
        with self.assertRaises(KeyboardInterrupt) as caught:
            with limit.request():
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertIsInstance(primary.__cause__, OSError)
        self.assertEqual(limit.used, 0)
        self.assertFalse(limit._callers)

    def test_malformed_notes_cannot_replace_primary_or_drop_prior_and_cleanup_statuses(self):
        class Primary(BaseException):
            def add_note(self, _):
                raise AssertionError("foreign note hook called")

        for malformed in (object(), 17):
            limit, primary = RequestLimit(1), Primary("original")
            before_cause, before_context, cleanup = OSError("prior cause"), ValueError("prior context"), OSError("cleanup")
            primary.__notes__ = malformed
            BaseException.__cause__.__set__(primary, before_cause)
            BaseException.__context__.__set__(primary, before_context)

            class Stack(list):
                def pop(self):
                    BaseException.__cause__.__set__(primary, RuntimeError("forged during retirement"))
                    BaseException.__context__.__set__(primary, None)
                    raise cleanup

            limit._callers[threading.get_ident()] = Stack()
            with self.assertRaises(Primary) as caught:
                with limit.request():
                    raise primary
            self.assertIs(caught.exception, primary)
            group = BaseException.__cause__.__get__(primary)
            self.assertIsInstance(group, BaseExceptionGroup)
            self.assertTrue(all(error in group.exceptions for error in (before_cause, before_context, cleanup)))
            self.assertTrue(any(isinstance(error, TypeError) for error in group.exceptions))
            self.assertEqual(limit.used, 0)

    def test_successful_retirement_restores_native_primary_roots(self):
        limit, primary = RequestLimit(1), KeyboardInterrupt("original")
        cause, context = OSError("cause"), ValueError("context")
        BaseException.__cause__.__set__(primary, cause)
        BaseException.__context__.__set__(primary, context)
        BaseException.__suppress_context__.__set__(primary, False)

        class Stack(list):
            def pop(self):
                BaseException.__cause__.__set__(primary, RuntimeError("forged cause"))
                BaseException.__context__.__set__(primary, None)
                BaseException.__suppress_context__.__set__(primary, True)
                return super().pop()

        limit._callers[threading.get_ident()] = Stack()
        with self.assertRaises(KeyboardInterrupt) as caught:
            with limit.request():
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertIs(BaseException.__cause__.__get__(primary), cause)
        self.assertIs(BaseException.__context__.__get__(primary), context)
        self.assertFalse(BaseException.__suppress_context__.__get__(primary))

    def test_borrow_rollback_failure_uses_thread_safe_closed_signal_and_preserves_status(self):
        limit = RequestLimit(1)
        primary, cleanup, root = KeyboardInterrupt("registration"), OSError("rollback"), ValueError("prior")
        primary.__notes__ = object()
        BaseException.__cause__.__set__(primary, root)

        class Registry(dict):
            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                raise primary

            def pop(self, *args):
                BaseException.__cause__.__set__(primary, RuntimeError("forged during rollback"))
                raise cleanup

        limit._workers = Registry()
        with self.assertRaises(KeyboardInterrupt) as caught:
            limit.borrow(object())
        self.assertIs(caught.exception, primary)
        self.assertTrue(limit._closed.is_set())
        group = BaseException.__cause__.__get__(primary)
        self.assertTrue(all(error in group.exceptions for error in (root, cleanup)))
        with self.assertRaisesRegex(RuntimeError, "closed"):
            with limit.request():
                pass

    def test_nested_calls_own_independent_slots_and_restore_outer(self):
        limit, one, two = RequestLimit(2), object(), object()
        with limit.request():
            with limit.request():
                self.assertEqual(limit.used, 2)
                limit.borrow(two)
                limit.release(two)
            limit.borrow(one)
            limit.release(one)
            self.assertEqual(limit.used, 1)
        self.assertEqual(limit.used, 0)

    def test_admitted_continuations_are_not_rejected_when_capacity_is_full(self):
        limit = RequestLimit(1)
        with limit.request():
            for _ in range(40):
                work = object()
                limit.borrow(work)
                self.assertEqual(limit.used, 1)
                limit.release(work)
                self.assertEqual(limit.used, 1)
        self.assertEqual(limit.used, 0)

    def test_concurrent_internal_job_is_invariant_failure_without_new_owner(self):
        limit, one = RequestLimit(2), object()
        with limit.request():
            limit.borrow(one)
            with self.assertRaisesRegex(RuntimeError, "concurrent"):
                limit.borrow(object())
            self.assertEqual(limit.used, 1)
            limit.release(one)
        self.assertEqual(limit.used, 0)

    def test_closed_owner_rejects_new_callers_without_discarding_old_work(self):
        limit, work = RequestLimit(1), object()
        limit.borrow(work)
        limit.close()
        self.assertEqual(limit.used, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            limit.borrow(object())
        limit.release(work)
        self.assertEqual(limit.used, 0)

    def test_racing_admission_never_exceeds_configured_count(self):
        limit, entered, leave = RequestLimit(4), threading.Barrier(13), threading.Event()
        accepted, refused = [], []
        lock = threading.Lock()

        def try_one():
            entered.wait(timeout=5)
            try:
                with limit.request():
                    with lock:
                        accepted.append(limit.used)
                    if not leave.wait(5):
                        raise RuntimeError("admission race deadline")
            except CapacityError:
                with lock:
                    refused.append(True)

        threads = [actor(try_one) for _ in range(12)]
        entered.wait(timeout=5)
        # Lock-protected completion count, with an explicit event from the final refusal.
        for _ in range(10000):
            with lock:
                settled = len(accepted) + len(refused) == 12
            if settled:
                break
            threading.Event().wait(.001)
        self.assertTrue(settled)
        self.assertEqual(len(accepted), 4)
        self.assertEqual(len(refused), 8)
        self.assertLessEqual(max(accepted), 4)
        leave.set()
        for thread, result in threads:
            joined(thread)
            self.assertEqual(result, [None])
        self.assertEqual(limit.used, 0)


class CudaLimitTests(unittest.TestCase):
    def test_finalizer_attempts_every_orphan_reply_when_first_publication_fails(self):
        import queue

        decoder = Decoder()
        scheduler = CudaScheduler(decoder, max_pending_requests=2)
        one, two = queue.Queue(), queue.Queue()
        for box in (one, two):
            scheduler._request_limit.borrow(box)
        scheduler._request_boxes = [one, two]  # deterministic failure order, same journal iteration contract
        scheduler._started_ok = True
        original = scheduler._terminal
        calls, primary = [], OSError("first orphan reply failed")

        def terminal(box, kind, value):
            calls.append(box)
            if box is one:
                raise primary
            scheduler._request_limit.release(box)
            box.put((kind, value))
            scheduler._request_boxes.remove(box)

        with patch.object(scheduler, "_run", lambda: None), patch.object(scheduler, "_terminal", terminal):
            scheduler._loop()
        self.assertEqual(calls, [one, two])
        self.assertIs(scheduler._cleanup_failure, primary)
        self.assertTrue(scheduler._worker_done.is_set())
        self.assertIsInstance(two.get_nowait()[1], RuntimeError)
        self.assertEqual(scheduler.pending_requests, 1)
        scheduler._request_boxes = {one}
        original(one, "error", primary)
        self.assertEqual(scheduler.pending_requests, 0)

    def test_failed_registration_rollback_closes_and_retires_orphan_journal(self):
        decoder = Decoder()
        scheduler = CudaScheduler(decoder, max_pending_requests=1).start()
        primary, cleanup = KeyboardInterrupt("registration"), OSError("rollback")
        primary.__notes__ = object()

        class Registry(set):
            def add(self, box):
                super().add(box)
                raise primary

            def discard(self, box):
                if not getattr(self, "failed", False):
                    self.failed = True
                    raise cleanup
                super().discard(box)

        scheduler._request_boxes = Registry()
        with self.assertRaises(KeyboardInterrupt) as caught:
            scheduler.submit([1], 1, None, False, lambda _: False)
        self.assertIs(caught.exception, primary)
        self.assertTrue(scheduler._request_limit._closed.is_set())
        self.assertTrue(scheduler._closing)
        self.assertIn(cleanup, BaseException.__cause__.__get__(primary).exceptions)
        scheduler.close()
        self.assertEqual(scheduler.pending_requests, 0)
        self.assertFalse(scheduler._request_boxes)

    def test_failed_wakeup_preserves_opaque_primary_and_close_can_retry(self):
        decoder = Decoder()
        scheduler = CudaScheduler(decoder, max_pending_requests=1).start()
        primary, cleanup = KeyboardInterrupt("queue publication"), OSError("wakeup")
        primary.__notes__ = object()
        with patch.object(scheduler.waiting, "put", side_effect=primary), patch.object(scheduler.waiting, "stop", side_effect=cleanup):
            with self.assertRaises(KeyboardInterrupt) as caught:
                scheduler.submit([1], 1, None, False, lambda _: False)
        self.assertIs(caught.exception, primary)
        self.assertIn(cleanup, BaseException.__cause__.__get__(primary).exceptions)
        with self.assertRaises(RuntimeError):
            scheduler.close()
        self.assertEqual(scheduler.pending_requests, 0)
        self.assertFalse(scheduler.thread.is_alive())

    def test_request_registry_failure_rolls_back_without_queue_publication(self):
        for after in (False, True):
            decoder = Decoder()
            decoder.release_round.set()
            scheduler = CudaScheduler(decoder, max_pending_requests=1).start()
            scheduler._request_boxes = FailingSet(after=after)
            try:
                with self.assertRaises(KeyboardInterrupt):
                    scheduler.submit([1], 1, None, False, lambda _: False)
                self.assertEqual(scheduler.pending_requests, 0)
                self.assertTrue(scheduler.waiting.empty())
                self.assertFalse(scheduler._request_boxes)
                self.assertIsInstance(scheduler.submit([2], 1, None, False, lambda _: False), dict)
            finally:
                scheduler.close()

    def test_canceled_caller_retains_active_worker_capacity_until_round_finishes(self):
        decoder = Decoder()
        scheduler = CudaScheduler(decoder, max_pending_requests=1).start()
        primary = KeyboardInterrupt("controlled canceled caller")

        def emit(_):
            raise primary

        thread, result = actor(lambda: scheduler.submit([1], 1, None, False, emit))
        try:
            self.assertTrue(decoder.entered.wait(5))
            joined(thread)
            self.assertIs(result[0], primary)
            self.assertEqual(scheduler.pending_requests, 1)
            with self.assertRaises(CapacityError):
                scheduler.submit([2], 1, None, False, lambda _: False)
        finally:
            decoder.release_round.set()
            scheduler.close()
        self.assertEqual(scheduler.pending_requests, 0)
        self.assertFalse(scheduler._request_boxes)

    def test_queued_and_running_jobs_share_cap_and_close_drains_both(self):
        decoder = Decoder()
        scheduler = CudaScheduler(decoder, max_streams=1, max_pending_requests=2).start()
        first = actor(lambda: scheduler.submit([1], 1, None, False, lambda _: False))
        self.assertTrue(decoder.entered.wait(5))
        second = actor(lambda: scheduler.submit([2], 1, None, False, lambda _: False))
        try:
            for _ in range(5000):
                if scheduler.pending_requests == 2:
                    break
                threading.Event().wait(.001)
            self.assertEqual(scheduler.pending_requests, 2)
            with self.assertRaises(CapacityError):
                scheduler.submit([3], 1, None, False, lambda _: False)
        finally:
            decoder.release_round.set()
            scheduler.close()
            for thread, _ in (first, second):
                joined(thread)
        self.assertTrue(all(isinstance(result[0], dict) for _, result in (first, second)))
        self.assertEqual(scheduler.pending_requests, 0)

    def test_app_owner_is_shared_across_two_actual_scheduler_submissions(self):
        decoder = Decoder()
        decoder.release_round.set()
        limit, scheduler = RequestLimit(1), CudaScheduler(decoder)
        scheduler.configure_requests(limit)
        scheduler.start()
        try:
            with limit.request():
                for _ in range(2):
                    scheduler.submit([1], 1, None, False, lambda _: False)
                    self.assertEqual(scheduler.pending_requests, 1)
            self.assertEqual(scheduler.pending_requests, 0)
        finally:
            scheduler.close()

    def test_publication_failure_before_or_after_enqueue_retains_then_retires(self):
        for enqueue in (False, True):
            with self.subTest(enqueue=enqueue):
                decoder = Decoder()
                decoder.release_round.set()
                scheduler = CudaScheduler(decoder, max_pending_requests=1).start()
                primary = KeyboardInterrupt("controlled publication failure")
                put = scheduler.waiting.put

                def failed(item, *args, **kwargs):
                    if enqueue:
                        put(item, *args, **kwargs)
                    raise primary

                with patch.object(scheduler.waiting, "put", failed):
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        scheduler.submit([1], 1, None, False, lambda _: False)
                self.assertIs(caught.exception, primary)
                with self.assertRaisesRegex(RuntimeError, "worker failed"):
                    scheduler.close()
                self.assertEqual(scheduler.pending_requests, 0)
                self.assertFalse(scheduler._request_boxes)

    def test_background_handoff_reuses_box_and_preserves_logical_owners(self):
        from tensorfold.cuda.streams import Stream
        import queue

        decoder = Decoder()
        scheduler = CudaScheduler(decoder, max_streams=1, max_pending_requests=2)
        background, foreground = Stream([1], 3, background=True), Stream([2], 1)
        box, other = queue.Queue(), queue.Queue()
        scheduler._request_limit.borrow(box)
        scheduler._request_limit.borrow(other)
        scheduler._request_boxes.update((box, other))
        decoder.streams[id(background)] = background
        scheduler.boxes[id(background)] = box
        scheduler.waiting.put((foreground, other))
        scheduler._yield()
        self.assertEqual(scheduler.pending_requests, 2)
        self.assertEqual(scheduler.yields, 1)
        self.assertEqual(scheduler.waiting.get_nowait(), (foreground, other))
        continued, retained = scheduler.waiting.get_nowait()
        self.assertIs(retained, box)
        self.assertIsNot(continued, background)
        self.assertTrue(continued.background)
        scheduler._terminal(box, "done", {})
        scheduler._terminal(other, "done", {})
        self.assertEqual(scheduler.pending_requests, 0)
        scheduler.close()

    def test_worker_failure_retires_admission_only_after_successful_cleanup(self):
        for cleanup_fails in (False, True):
            with self.subTest(cleanup_fails=cleanup_fails):
                decoder = Decoder()
                decoder.release_round.set()
                primary = decoder.failure = KeyboardInterrupt("controlled worker failure")
                if cleanup_fails:
                    def drop():
                        raise OSError("controlled native-boundary cleanup failure")
                    decoder.drop = drop
                scheduler = CudaScheduler(decoder, max_pending_requests=1).start()
                with self.assertRaises(KeyboardInterrupt) as caught:
                    scheduler.submit([1], 1, None, False, lambda _: False)
                self.assertIs(caught.exception, primary)
                with self.assertRaises(RuntimeError):
                    scheduler.close()
                self.assertEqual(scheduler.pending_requests, int(cleanup_fails))
                self.assertIs(scheduler.decoder, decoder if cleanup_fails else None)

    def test_opaque_worker_notes_and_drop_failure_notify_all_capped_callers(self):
        decoder = Decoder()
        primary, cleanup = KeyboardInterrupt("worker failed"), OSError("drop failed")
        primary.__notes__ = object()
        decoder.failure = primary

        def drop():
            raise cleanup

        decoder.drop = drop
        scheduler = CudaScheduler(decoder, max_streams=1, max_pending_requests=2).start()
        first = actor(lambda: scheduler.submit([1], 3, None, False, lambda _: False))
        self.assertTrue(decoder.entered.wait(5))
        second = actor(lambda: scheduler.submit([2], 3, None, False, lambda _: False))
        for _ in range(5000):
            if scheduler.pending_requests == 2:
                break
            threading.Event().wait(.001)
        self.assertEqual(scheduler.pending_requests, 2)
        decoder.release_round.set()
        for thread, result in (first, second):
            joined(thread)
            self.assertIs(result[0], primary)
        with self.assertRaises(RuntimeError):
            scheduler.close()
        self.assertIn(cleanup, BaseException.__cause__.__get__(primary).exceptions)
        self.assertEqual(scheduler.pending_requests, 2)
        self.assertIs(scheduler.decoder, decoder)


class GenericLimitTests(unittest.TestCase):
    def scheduler(self, **kwargs):
        engine = SimpleNamespace(drain=lambda: None, reset=lambda: None, active_count=0, prefill_guard=None)
        return GENERIC.Scheduler(engine, lanes=1, eos_ids=frozenset(), **kwargs)

    def job(self, name):
        return GENERIC.ChatJob(name, [1], 1, 0.)

    def test_failed_registration_rollback_closes_admission_and_retains_owned_job(self):
        scheduler = self.scheduler(max_pending_requests=1)
        primary, cleanup = KeyboardInterrupt("registration"), OSError("rollback")
        primary.__notes__ = object()

        class Registry(dict):
            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                raise primary

            def pop(self, *args):
                if not getattr(self, "failed", False):
                    self.failed = True
                    raise cleanup
                return super().pop(*args)

        scheduler._accepted = Registry()
        with self.assertRaises(KeyboardInterrupt) as caught:
            scheduler.submit(self.job("failed"))
        self.assertIs(caught.exception, primary)
        self.assertIn(cleanup, BaseException.__cause__.__get__(primary).exceptions)
        self.assertTrue(scheduler._stop.is_set())
        self.assertTrue(scheduler._request_limit._closed.is_set())
        self.assertEqual(scheduler.pending_requests, 1)
        scheduler._finalize(None)
        self.assertEqual(scheduler.pending_requests, 0)
        self.assertFalse(scheduler._accepted)

    def test_accepted_registry_failure_rolls_back_unpublished_job(self):
        for after in (False, True):
            scheduler = self.scheduler(max_pending_requests=1)
            scheduler._accepted = FailingDict(after=after)
            with self.assertRaises(KeyboardInterrupt):
                scheduler.submit(self.job("failed"))
            self.assertTrue(scheduler._queue.empty())
            self.assertFalse(scheduler._accepted)
            self.assertEqual(scheduler.pending_requests, 0)
            job = self.job("next")
            scheduler.submit(job)
            scheduler._finish(job)
            self.assertEqual(scheduler.pending_requests, 0)

    def test_job_publication_failure_before_and_after_enqueue_drains_journal(self):
        for enqueue in (False, True):
            scheduler = self.scheduler(max_pending_requests=1)
            put = scheduler._queue.put
            primary = KeyboardInterrupt("controlled job queue publication")

            def failed(item):
                if enqueue:
                    put(item)
                raise primary

            with patch.object(scheduler._queue, "put", failed):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    scheduler.submit(self.job("failed"))
            self.assertIs(caught.exception, primary)
            self.assertEqual(scheduler.pending_requests, 1)
            self.assertTrue(scheduler._stop.is_set())
            scheduler._finalize(primary)
            self.assertEqual(scheduler.pending_requests, 0)
            self.assertTrue(scheduler._queue.empty())

    def test_rpc_registry_failure_rolls_back_before_any_callback_publication(self):
        for after in (False, True):
            scheduler = self.scheduler(max_engine_calls=1)
            scheduler._engine_waiters = FailingDict(after=after)
            with self.assertRaises(KeyboardInterrupt):
                scheduler.on_engine(lambda _: None, timeout=0)
            self.assertEqual(scheduler.engine_calls, 0)
            self.assertTrue(scheduler._engine_calls.empty())

    def test_rpc_queue_failure_retains_ambiguous_work_until_stop(self):
        for enqueue in (False, True):
            scheduler = self.scheduler(max_engine_calls=1)
            calls, primary, put = [], KeyboardInterrupt("RPC queue publication"), scheduler._engine_calls.put

            def failed(item):
                if enqueue:
                    put(item)
                raise primary

            with patch.object(scheduler._engine_calls, "put", failed):
                with self.assertRaises(KeyboardInterrupt):
                    scheduler.on_engine(lambda _: calls.append("must not execute"), timeout=0)
            self.assertEqual(scheduler.engine_calls, 1)
            self.assertTrue(scheduler._stop.is_set())
            scheduler._finalize(primary)
            self.assertEqual(scheduler.engine_calls, 0)
            self.assertTrue(scheduler._engine_calls.empty())
            self.assertFalse(calls)

    def test_default_policy_keeps_admission_uncapped(self):
        scheduler = self.scheduler()
        jobs = [self.job(str(index)) for index in range(30)]
        for job in jobs:
            scheduler.submit(job)
        self.assertIsNone(scheduler._request_limit)
        self.assertEqual(len(scheduler._accepted), 30)
        for job in jobs:
            scheduler._finish(job)
        self.assertIsNone(scheduler.pending_requests)

    def test_jobs_reject_then_reuse_capacity_at_terminal_completion(self):
        scheduler = self.scheduler(max_pending_requests=1)
        one, two = self.job("one"), self.job("two")
        scheduler.submit(one)
        with self.assertRaises(CapacityError):
            scheduler.submit(two)
        scheduler._finish(one)
        scheduler.submit(two)
        scheduler._finish(two)
        self.assertEqual(scheduler.pending_requests, 0)
        self.assertFalse(scheduler._accepted)

    def test_replayed_job_borrows_same_live_request(self):
        limit = RequestLimit(1)
        scheduler = self.scheduler(max_pending_requests=1, request_limit=limit)
        with limit.request():
            for name in ("original", "background-replay", "gate-continuation"):
                job = self.job(name)
                scheduler.submit(job)
                scheduler._finish(job)
                self.assertEqual(limit.used, 1)
        self.assertEqual(limit.used, 0)

    def test_queued_cancel_releases_job_and_active_caller_owns_until_return(self):
        limit = RequestLimit(1)
        scheduler = self.scheduler(max_pending_requests=1, request_limit=limit)
        with limit.request():
            job = self.job("cancel")
            scheduler.submit(job)
            scheduler.cancel(job.cancellation)
            self.assertTrue(job.done.is_set())
            self.assertEqual(scheduler.pending_requests, 1)
        self.assertEqual(scheduler.pending_requests, 0)

    def test_rpc_timeout_keeps_queued_callback_cap_until_execution(self):
        scheduler = self.scheduler(max_engine_calls=1)
        calls = []
        with self.assertRaises(TimeoutError):
            scheduler.on_engine(lambda _: calls.append("first"), timeout=0)
        self.assertEqual(scheduler.engine_calls, 1)
        with self.assertRaises(CapacityError):
            scheduler.on_engine(lambda _: calls.append("rejected"), timeout=0)
        self.assertTrue(scheduler._run_engine_call())
        self.assertEqual(calls, ["first"])
        self.assertEqual(scheduler.engine_calls, 0)
        with self.assertRaises(TimeoutError):
            scheduler.on_engine(lambda _: calls.append("second"), timeout=0)
        scheduler._run_engine_call()
        self.assertEqual(calls, ["first", "second"])

    def test_running_rpc_retains_cap_and_callback_error_retires_it(self):
        scheduler = self.scheduler(max_engine_calls=1)
        entered, leave = threading.Event(), threading.Event()
        primary = ValueError("controlled RPC failure")

        def callback(_):
            entered.set()
            if not leave.wait(5):
                raise RuntimeError("RPC control deadline")
            raise primary

        caller, result = actor(lambda: scheduler.on_engine(callback))
        for _ in range(5000):
            if scheduler.engine_calls:
                break
            threading.Event().wait(.001)
        worker, worker_result = actor(scheduler._run_engine_call)
        try:
            self.assertTrue(entered.wait(5))
            with self.assertRaises(CapacityError):
                scheduler.on_engine(lambda _: None, timeout=0)
        finally:
            leave.set()
            joined(caller)
            joined(worker)
        self.assertIs(result[0], primary)
        self.assertEqual(worker_result, [True])
        self.assertEqual(scheduler.engine_calls, 0)

    def test_rpc_reply_failure_before_or_after_enqueue_preserves_waiter_for_shutdown(self):
        import queue

        for enqueue in (False, True):
            scheduler = self.scheduler(max_engine_calls=1)
            with self.assertRaises(TimeoutError):
                scheduler.on_engine(lambda _: "computed", timeout=0)
            fn, done = scheduler._engine_calls.get_nowait()
            scheduler._engine_calls.put((fn, done))
            put, primary = done.put_nowait, KeyboardInterrupt("reply publication")

            def fail_once(value):
                if enqueue:
                    put(value)
                raise primary

            with patch.object(done, "put_nowait", fail_once):
                with self.assertRaises(KeyboardInterrupt):
                    scheduler._run_engine_call()
            self.assertEqual(scheduler.engine_calls, 1)
            scheduler._finalize(primary)
            self.assertEqual(scheduler.engine_calls, 0)
            reply = done.get_nowait()
            self.assertEqual(reply, "computed") if enqueue else self.assertIsInstance(reply, BaseException)
            with self.assertRaises(queue.Empty):
                done.get_nowait()

    def test_rpc_shutdown_attempts_remaining_replies_and_drains_after_publication_failure(self):
        scheduler = self.scheduler(max_engine_calls=2)
        for _ in range(2):
            with self.assertRaises(TimeoutError):
                scheduler.on_engine(lambda _: None, timeout=0)
        first, second = tuple(scheduler._engine_waiters.values())
        error, observations = OSError("reply refused"), []
        scheduler.engine.drain = lambda: observations.append("drain")
        with patch.object(first, "put_nowait", side_effect=error):
            scheduler._finalize(None)
        self.assertEqual(observations, ["drain", "drain"])
        self.assertIs(scheduler._cleanup_failure, error)
        self.assertEqual(scheduler.engine_calls, 1)
        self.assertIsInstance(second.get_nowait(), BaseException)
        self.assertIn(id(first), scheduler._engine_waiters)

    def test_failed_shutdown_drain_keeps_request_reservation_and_closes_admission(self):
        scheduler = self.scheduler(max_pending_requests=1)
        job = self.job("live")
        scheduler.submit(job)
        primary = OSError("drain failed")

        def failed():
            raise primary

        scheduler.engine.drain = failed
        scheduler._finalize(None)
        self.assertIs(scheduler._cleanup_failure, primary)
        self.assertEqual(scheduler.pending_requests, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            scheduler.submit(self.job("rejected"))

    def test_successful_shutdown_drains_before_reservation_release(self):
        scheduler = self.scheduler(max_pending_requests=1, max_engine_calls=1)
        scheduler.submit(self.job("queued"))
        observations = []
        scheduler.engine.drain = lambda: observations.append(scheduler.pending_requests)
        with self.assertRaises(TimeoutError):
            scheduler.on_engine(lambda _: None, timeout=0)
        scheduler._finalize(None)
        self.assertEqual(observations, [1, 1])
        self.assertEqual(scheduler.pending_requests, 0)
        self.assertEqual(scheduler.engine_calls, 0)


if __name__ == "__main__":
    unittest.main()

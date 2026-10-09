"""Maintained Engine boundaries, exercised without CUDA or model imports."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from tensorfold.cuda.scheduler import Scheduler


SOURCE = Path(__file__).resolve().parents[1] / "src/tensorfold/families/qwen4_exp/cuda/engine.py"


def engine_type():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FlashNextEngine")
    keep = {"close", "_close_resources", "_begin_request", "_prune_requests", "_end_request", "_retire_request", "generate", "follow",
            "_limit", "_resume", "supports_logprobs"}
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in keep]
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                           ast.ClassDef(name="Engine", bases=[], keywords=[], body=body, decorator_list=[])],
                      type_ignores=[])
    ast.fix_missing_locations(unit)
    namespace = {"threading": threading}
    exec(compile(unit, str(SOURCE), "exec"), namespace)
    return namespace["Engine"]


Engine = engine_type()


class Resource:
    def __init__(self, events, phase):
        self.events, self.phase = events, phase

    def close(self):
        self.events.append(self.phase)


def engine(scheduler=None):
    e = Engine()
    e._lifecycle = threading.Condition()
    e._closing = e._closed = e._close_running = False
    e._calls = {}
    e.events = []
    e.scheduler = scheduler
    e.w = SimpleNamespace(device="owned-device", meta={"expert_cache": Resource(e.events, "cache-close")})
    plan = Resource(e.events, "plan-close")
    e.multi = SimpleNamespace(buf=SimpleNamespace(hc_plans=plan), mbuf=None)
    e.e = e.serial = None
    e.tp, e.vision, e.depth, e.max_len, e.cache = 1, None, 4, 2053, []
    e._serial = lambda *args, **kwargs: {"serial": True}
    e._decode = lambda *args, **kwargs: {"draft": True}
    return e


def spawn(function, outcomes, name):
    def run():
        try:
            outcomes[name] = function()
        except BaseException as error:
            outcomes[name] = error
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def join(t):
    t.join(5)
    assert not t.is_alive(), "owned CPU worker exceeded deadline"


def closing(e):
    with e._lifecycle:
        assert e._lifecycle.wait_for(lambda: e._closing, 5), "close did not stop acceptance"


class LifetimeControls(unittest.TestCase):
    def setUp(self):
        self.runtime = ModuleType("torch")
        self.events = []
        self.runtime.cuda = SimpleNamespace(synchronize=lambda device: self.events.append(("fence", device)))
        self.stub = patch.dict(sys.modules, {"torch": self.runtime})
        self.stub.start()

    def tearDown(self):
        self.stub.stop()

    def test_postclose_rejects_before_limits_or_decoder_dispatch_and_keeps_fixed_owner(self):
        e = engine()
        weight = e.w
        e.close()
        e._limit = lambda *args: self.fail("closed request reached validation")
        for draft in (False, True):
            with self.subTest(draft=draft), self.assertRaisesRegex(RuntimeError, "closing or closed"):
                e.generate([], 1, None, lambda _: False, draft=draft)
        self.assertIs(e.w, weight)
        self.assertEqual(e.events, ["plan-close", "cache-close"])
        self.assertEqual(self.events, [("fence", "owned-device")])
        e.close()
        self.assertEqual(e.events, ["plan-close", "cache-close"])

    def test_captured_scheduler_cannot_fall_through_when_close_wins_before_submit(self):
        began, release = threading.Event(), threading.Event()
        outcomes = {}

        class DelayedScheduler:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

            def submit(self, *args, **kwargs):
                if self.closed:
                    raise RuntimeError("captured scheduler rejected")
                return {"accepted": True}

        e = engine(DelayedScheduler())
        e._serial = e._decode = lambda *args, **kwargs: self.fail("request changed decoder mode")

        def limit(*args):
            began.set()
            if not release.wait(5):
                raise RuntimeError("fixture release deadline")
            return 1

        e._limit = limit
        caller = spawn(lambda: e.generate([1], 1, None, lambda _: False), outcomes, "caller")
        self.assertTrue(began.wait(5))
        closer = spawn(e.close, outcomes, "close")
        try:
            closing(e)
            self.assertTrue(closer.is_alive())
            self.assertEqual(e.events, [])
        finally:
            release.set()
            join(caller)
            join(closer)
        self.assertRegex(str(outcomes["caller"]), "captured scheduler rejected")
        self.assertIsNone(outcomes["close"])
        self.assertFalse(e._calls)

    def test_serial_close_drains_request_and_rejects_overlapping_serial_call(self):
        began, release = threading.Event(), threading.Event()
        e, outcomes = engine(), {}

        def serial(*args, **kwargs):
            began.set()
            if not release.wait(5):
                raise RuntimeError("fixture release deadline")
            return {"serial": True}

        e._serial = serial
        caller = spawn(lambda: e.generate([1], 1, None, lambda _: False, draft=False), outcomes, "caller")
        self.assertTrue(began.wait(5))
        with self.assertRaisesRegex(RuntimeError, "already owns"):
            e.generate([2], 1, None, lambda _: False, draft=False)
        closer = spawn(e.close, outcomes, "close")
        try:
            closing(e)
            self.assertTrue(closer.is_alive())
            self.assertEqual(e.events, [])
            with self.assertRaisesRegex(RuntimeError, "closing or closed"):
                e.generate([2], 1, None, lambda _: False)
        finally:
            release.set()
            join(caller)
            join(closer)
        self.assertEqual(outcomes, {"caller": {"serial": True}, "close": None})
        self.assertFalse(e._calls)

    def test_primary_error_and_validation_error_retire_caller_owners(self):
        e = engine()
        primary = RuntimeError("owned primary")

        def serial(*args, **kwargs):
            raise primary

        e._serial = serial
        with self.assertRaises(RuntimeError) as caught:
            e.generate([1], 1, None, lambda _: False, draft=False)
        self.assertIs(caught.exception, primary)
        with self.assertRaises(ValueError):
            e.generate([1] * 2053, 1, None, lambda _: False)
        self.assertFalse(e._calls)
        e.close()

    def test_request_callback_and_decoding_worker_close_reject_before_mutation(self):
        e = engine()

        def serial(*args, **kwargs):
            with self.assertRaisesRegex(RuntimeError, "active request"):
                e.close()
            self.assertFalse(e._closing)
            return {"valid": True}

        e._serial = serial
        self.assertEqual(e.generate([1], 1, None, lambda _: False, draft=False), {"valid": True})
        e.scheduler = SimpleNamespace(thread=threading.current_thread())
        with self.assertRaisesRegex(RuntimeError, "decoding worker"):
            e.close()
        self.assertFalse(e._closing)
        e.scheduler = None
        e.close()

    def test_failed_fence_retains_owners_and_retry_keeps_admission_closed(self):
        e = engine()
        primary = RuntimeError("controlled failed fence")

        def fence(device):
            raise primary

        self.runtime.cuda.synchronize = fence
        with self.assertRaises(RuntimeError) as caught:
            e.close()
        self.assertIs(caught.exception, primary)
        self.assertTrue(e._closing)
        self.assertFalse(e._closed or e._close_running)
        self.assertEqual(e.events, [])
        with self.assertRaisesRegex(RuntimeError, "closing or closed"):
            e.generate([1], 1, None, lambda _: False)
        self.runtime.cuda.synchronize = lambda _: None
        e.close()
        self.assertEqual(e.events, ["plan-close", "cache-close"])
        self.assertTrue(e._closed)

    def test_concurrent_closes_share_one_cleanup(self):
        e, outcomes = engine(), {}
        began, release = threading.Event(), threading.Event()
        original = e._close_resources

        def resources():
            began.set()
            if not release.wait(5):
                raise RuntimeError("fixture release deadline")
            original()

        e._close_resources = resources
        first = spawn(e.close, outcomes, "first")
        self.assertTrue(began.wait(5))
        second = spawn(e.close, outcomes, "second")
        try:
            self.assertTrue(second.is_alive())
        finally:
            release.set()
            join(first)
            join(second)
        self.assertEqual(outcomes, {"first": None, "second": None})
        self.assertEqual(e.events, ["plan-close", "cache-close"])

    def test_partial_lifecycle_initialization_closes_without_missing_attribute(self):
        for partial in (False, True):
            e = Engine()
            if partial:
                e._lifecycle = threading.Condition()
            e.close()
            e.close()

    def test_rank_one_follow_uses_the_same_terminal_boundary(self):
        e = engine()
        e.served = 0
        requests = iter([([1], 1, None, False, 0, [], True, []), None])
        e._receive = lambda: next(requests)
        e.follow()
        self.assertEqual(e.served, 1)
        self.assertFalse(e._calls)
        e.close()
        e._receive = lambda: ([1], 1, None, False, 0, [], True, [])
        with self.assertRaisesRegex(RuntimeError, "closing or closed"):
            e.follow()

    def test_real_scheduler_worker_drain_keeps_callback_owner_until_it_retires(self):
        began, release = threading.Event(), threading.Event()
        outcomes = {}

        class OneToken:
            def __init__(self):
                self.streams = {}

            def live(self):
                return len(self.streams)

            def admit(self, stream):
                self.streams[id(stream)] = stream

            def round(self):
                done = list(self.streams.values())
                for stream in done:
                    stream.take([17])
                return done

            def finish(self, done):
                for stream in done:
                    self.streams.pop(id(stream))

        scheduler = Scheduler(OneToken(), max_streams=4)
        scheduler.start()
        e = engine(scheduler)

        def emitted(tokens):
            self.assertEqual(tokens, [17])
            began.set()
            if not release.wait(5):
                raise RuntimeError("fixture release deadline")
            return False

        caller = spawn(lambda: e.generate([1], 1, None, emitted), outcomes, "caller")
        self.assertTrue(began.wait(5))
        closer = spawn(e.close, outcomes, "close")
        try:
            closing(e)
            self.assertTrue(scheduler._worker_done.wait(5))
            self.assertFalse(scheduler.thread.is_alive())
            self.assertTrue(closer.is_alive())
            self.assertEqual(e.events, [])
        finally:
            release.set()
            join(caller)
            join(closer)
            if scheduler.thread.is_alive():
                scheduler.close()
        self.assertIsNone(outcomes["close"])
        self.assertIsInstance(outcomes["caller"], dict)
        self.assertFalse(e._calls)

    def test_trace_interrupts_registration_return_and_caller_handoff_retire_journal(self):
        for target, event in (("_begin_request", "return"), ("generate", "line"), ("follow", "line")):
            e = engine()
            e.served = 0
            e._receive = lambda: ([1], 1, None, False, 0, [], True, [])
            primary = KeyboardInterrupt("controlled registration cancellation")
            injected = False

            def tracer(frame, observed, arg):
                nonlocal injected
                if (not injected and frame.f_code.co_name == target and observed == event
                        and frame.f_locals.get("self") is e and e._calls):
                    injected = True
                    raise primary
                return tracer

            original = sys.gettrace()
            try:
                sys.settrace(tracer)
                with self.assertRaises(KeyboardInterrupt) as caught:
                    if target == "follow":
                        e.follow()
                    else:
                        e.generate([1], 1, None, lambda _: False, draft=False)
                self.assertIs(caught.exception, primary)
            finally:
                sys.settrace(original)
            self.assertTrue(injected)
            self.assertFalse(e._calls)
            e.close()

    def test_trace_interrupts_retirement_is_reported_after_bounded_owner_repair(self):
        for fail_request in (False, True):
            e = engine()
            primary = RuntimeError("owned request failure") if fail_request else None
            cancellation = KeyboardInterrupt("controlled retirement cancellation")
            injected = False

            def serial(*args, **kwargs):
                if primary is not None:
                    raise primary
                return {"serial": True}

            e._serial = serial

            def tracer(frame, event, arg):
                nonlocal injected
                if (not injected and frame.f_code.co_name == "_end_request" and event == "line"
                        and frame.f_locals.get("self") is e and e._calls):
                    injected = True
                    raise cancellation
                return tracer

            original = sys.gettrace()
            try:
                sys.settrace(tracer)
                with self.assertRaises(BaseException) as caught:
                    e.generate([1], 1, None, lambda _: False, draft=False)
                self.assertIs(caught.exception, primary if fail_request else cancellation)
                if primary is not None:
                    self.assertIs(primary.__cause__, cancellation)
            finally:
                sys.settrace(original)
            self.assertTrue(injected)
            self.assertFalse(e._calls)
            e.close()

    def test_trace_interrupt_after_close_owner_claim_allows_retry(self):
        e = engine()
        primary = KeyboardInterrupt("controlled close publication cancellation")
        injected = False

        def tracer(frame, event, arg):
            nonlocal injected
            if (not injected and frame.f_code.co_name == "close" and event == "line"
                    and frame.f_locals.get("self") is e and e._close_running):
                injected = True
                raise primary
            return tracer

        original = sys.gettrace()
        try:
            sys.settrace(tracer)
            with self.assertRaises(KeyboardInterrupt) as caught:
                e.close()
            self.assertIs(caught.exception, primary)
        finally:
            sys.settrace(original)
        self.assertTrue(injected)
        self.assertFalse(e._close_running)
        self.assertTrue(e._closing)
        self.assertEqual(e.events, [])
        e.close()

    def test_trace_interrupt_before_python_retirement_still_releases_native_scope(self):
        for target in ("_retire_request", "close"):
            e = engine()
            primary = KeyboardInterrupt("controlled before-retirement cancellation")
            injected = False

            def tracer(frame, event, arg):
                nonlocal injected
                if (not injected and frame.f_code.co_name == target and frame.f_locals.get("self") is e
                        and ((target == "_retire_request" and event == "call")
                             or (target == "close" and event == "line" and e._closed))):
                    injected = True
                    raise primary
                return tracer

            original = sys.gettrace()
            try:
                sys.settrace(tracer)
                with self.assertRaises(KeyboardInterrupt) as caught:
                    if target == "close":
                        e.close()
                    else:
                        e.generate([1], 1, None, lambda _: False, draft=False)
                self.assertIs(caught.exception, primary)
            finally:
                sys.settrace(original)
            self.assertTrue(injected)
            self.assertTrue(all(not request.locked() for request in e._calls))
            e.close()
            self.assertFalse(e._calls)


if __name__ == "__main__":
    unittest.main()

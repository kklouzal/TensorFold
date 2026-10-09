"""Real callback lifetime controls; no numerical or foreign runtime imports."""

import gc
import importlib.util
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
import unittest
import weakref

spec = importlib.util.spec_from_file_location(
    "owned_thread_work", Path(__file__).parents[1] / "src/tensorfold/thread_work.py"
)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
ThreadWork = source.ThreadWork


class ThreadWorkLifetime(unittest.TestCase):
    def test_entry_timeout_retains_callback_until_real_retirement(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        work = ThreadWork(lambda: (entered.set(), release.wait(3), calls.append("retired")))
        thread = threading.Thread(target=work.run)
        work.launch(thread)
        self.assertTrue(entered.wait(3))
        try:
            with self.assertRaises(TimeoutError):
                work.drain(thread, timeout=0.01)
            self.assertIsNotNone(work._callback)
            self.assertFalse(work.retired.is_set())
        finally:
            release.set()
            work.drain(thread, timeout=3)
        self.assertEqual(calls, ["retired"])
        self.assertIsNone(work._callback)

    def test_post_native_start_failure_never_enters_callback(self):
        calls = []
        work = ThreadWork(lambda: calls.append("unsafe"))

        class Interrupted(threading.Thread):
            def start(self):
                super().start()
                raise KeyboardInterrupt("after bootstrap")

        thread = Interrupted(target=work.run)
        with self.assertRaises(KeyboardInterrupt):
            work.launch(thread)
        work.drain(thread, timeout=3)
        self.assertTrue(work.cancelled_before_entry)
        self.assertFalse(calls)
        self.assertFalse(thread.is_alive())

    def test_pending_native_bootstrap_only_retains_empty_controller(self):
        called = []
        work = ThreadWork(lambda: called.append("unsafe"))

        class Pending:
            def start(self):
                raise KeyboardInterrupt("accepted by a native bootstrap seam")

            def join(self, timeout):
                raise RuntimeError("cannot join thread before it is started")

        thread = Pending()
        with self.assertRaises(KeyboardInterrupt):
            work.launch(thread)
        work.drain(thread, timeout=0)
        work.run()  # eventual bootstrap can enter only an empty controller
        self.assertEqual(called, [])
        self.assertIsNone(work._callback)

    def test_unstarted_cancellation_releases_callback_owner(self):
        class Payload:
            pass

        payload = Payload()
        ref = weakref.ref(payload)
        work = ThreadWork(lambda value=payload: value)
        thread = threading.Thread(target=work.run)
        del payload
        work.drain(thread, timeout=0)
        gc.collect()
        self.assertIsNone(ref())
        with self.assertRaises(RuntimeError):
            work.launch(thread)

    def test_failure_is_observed_without_foreign_format_hooks(self):
        class Opaque(BaseException):
            def __str__(self):
                raise AssertionError("formatter")

        error = Opaque()
        entered = threading.Event()

        def fail():
            entered.set()
            raise error

        work = ThreadWork(fail)
        thread = threading.Thread(target=work.run)
        work.launch(thread)
        self.assertTrue(entered.wait(3))
        with self.assertRaises(RuntimeError) as result:
            work.drain(thread, timeout=3)
        self.assertIs(result.exception.__cause__, error)
        self.assertTrue(work.retired.is_set())

    def test_invalid_timeout_does_not_cancel_owned_work(self):
        for timeout in (True, -1, float("nan"), float("inf"), "1"):
            work = ThreadWork(lambda: None)
            with self.assertRaises(ValueError):
                work.drain(threading.Thread(target=work.run), timeout=timeout)
            self.assertFalse(work.retired.is_set())

    @unittest.skipUnless(sys.platform == "linux", "isolated POSIX signal schedule")
    def test_actual_interrupted_join_cannot_retire_callback_resources(self):
        # Separate process owns the actual SIGINT handler and bounded workers.
        program = textwrap.dedent("""
            import importlib.util, os, signal, sys, threading, time
            spec = importlib.util.spec_from_file_location("w", sys.argv[1])
            source = importlib.util.module_from_spec(spec); spec.loader.exec_module(source)
            entered, release = threading.Event(), threading.Event()
            work = source.ThreadWork(lambda: (entered.set(), release.wait(4)))
            thread = threading.Thread(target=work.run)
            work.launch(thread); assert entered.wait(2)
            trigger = threading.Thread(target=lambda: (time.sleep(.05), os.kill(os.getpid(), signal.SIGINT)))
            trigger.start()
            try:
                thread.join(2)
                raise AssertionError("signal missed")
            except KeyboardInterrupt:
                pass
            trigger.join(2)
            assert not work.retired.is_set()
            try:
                work.drain(thread, timeout=.01)
                raise AssertionError("unfinished callback was released")
            except TimeoutError:
                pass
            finally:
                release.set()
            work.drain(thread, timeout=2)
            assert work.retired.is_set() and work._callback is None
            print("callback retirement independently proven")
        """)
        result = subprocess.run(
            [sys.executable, "-W", "error", "-c", program, str(Path(source.__file__))],
            capture_output=True,
            text=True,
            timeout=8,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("independently proven", result.stdout)


if __name__ == "__main__":
    unittest.main()

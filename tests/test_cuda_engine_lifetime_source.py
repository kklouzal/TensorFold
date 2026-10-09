"""Actual stdlib threads/signals for mutable CUDA request lifetime, no SDK."""
from __future__ import annotations

import os
import signal
import threading
import unittest

from tensorfold.cuda.engine_lifetime import EngineLifetime


class EngineLifetimeTests(unittest.TestCase):
    def test_close_waits_accepted_request_and_rejects_pending_and_future_work(self):
        scope = EngineLifetime()
        entered, release, done = threading.Event(), threading.Event(), threading.Event()
        retired = []
        def request():
            try:
                with scope.request():
                    entered.set()
                    if not release.wait(5):
                        raise AssertionError("fixture request deadline")
            finally:
                done.set()
        worker = threading.Thread(target=request)
        worker.start()
        self.assertTrue(entered.wait(5))
        closer = threading.Thread(target=lambda: scope.close(lambda: retired.append(done.is_set())))
        closer.start()
        self.assertTrue(scope.closing.wait(5))
        self.assertEqual(retired, [])
        release.set()
        self.assertTrue(done.wait(5))
        worker.join(5)
        closer.join(5)
        self.assertEqual(retired, [True])
        with self.assertRaises(RuntimeError):
            with scope.request():
                self.fail("closed request entered")
        scope.close(lambda: self.fail("retirement repeated"))

    def test_reentry_rejected_and_callback_close_cannot_deadlock(self):
        scope = EngineLifetime()
        with scope.request():
            with self.assertRaises(RuntimeError):
                with scope.request():
                    self.fail("reentrant request entered")
            with self.assertRaises(RuntimeError):
                scope.close(lambda: self.fail("live callback retired"))
        scope.close(lambda: None)
        self.assertTrue(scope.retired)

    def test_failure_closes_admission_and_failed_retirement_can_retry(self):
        scope, primary, fence = EngineLifetime(), KeyboardInterrupt(), OSError()
        with self.assertRaises(KeyboardInterrupt) as caught:
            with scope.request():
                raise primary
        self.assertIs(caught.exception, primary)
        with self.assertRaises(RuntimeError) as caught:
            with scope.request():
                self.fail("failed engine entered")
        self.assertIs(caught.exception.__cause__, primary)
        with self.assertRaises(OSError):
            scope.close(lambda: (_ for _ in ()).throw(fence))
        self.assertFalse(scope.retired)
        scope.close(lambda: None)
        self.assertTrue(scope.retired)

    def test_invalid_admission_does_not_poison_a_valid_engine(self):
        scope = EngineLifetime()
        with self.assertRaises(ValueError):
            with scope.request(lambda: (_ for _ in ()).throw(ValueError("invalid request"))):
                self.fail("invalid request accepted")
        self.assertIsNone(scope.failed)
        with scope.request(lambda: 7) as accepted:
            self.assertEqual(accepted, 7)

    def test_actual_sigint_close_retry_waits_request_completion(self):
        scope = EngineLifetime()
        entered, release, done, sent = (threading.Event() for _ in range(4))
        failures = []
        def request():
            try:
                with scope.request():
                    entered.set()
                    if not release.wait(5):
                        raise AssertionError("fixture request deadline")
            except BaseException as error:
                failures.append(error)
            finally:
                done.set()
        def interrupt():
            if not scope.closing.wait(5):
                failures.append(AssertionError("close did not enter"))
                return
            os.kill(os.getpid(), signal.SIGINT)
            sent.set()
        worker, sender = threading.Thread(target=request), threading.Thread(target=interrupt)
        previous = signal.signal(signal.SIGINT, signal.default_int_handler)
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            sender.start()
            with self.assertRaises(KeyboardInterrupt):
                scope.close(lambda: self.fail("request is still live"))
            self.assertTrue(sent.wait(5))
            self.assertFalse(scope.retired)
            self.assertFalse(done.is_set())
            release.set()
            scope.close(lambda: self.assertTrue(done.is_set()))
        finally:
            release.set()
            self.assertTrue(done.wait(5))
            worker.join(5)
            if sender.ident is not None:
                sender.join(5)
            signal.signal(signal.SIGINT, previous)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()

"""Telemetry completion releases per-thread request/model references, preserving counters."""
import gc
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from tensorfold.server import metrics


class App:
    pass


class Job:
    def __init__(self):
        self.stream = SimpleNamespace(drafted=7, accepted=3)


class RequestLifetime(unittest.TestCase):
    def setUp(self):
        self.locals = patch.dict(metrics._local.__dict__, {}, clear=True)
        self.locals.start()
        self.addCleanup(self.locals.stop)

    def begin(self, app, job=None):
        metrics.begin(app, 4, time.perf_counter())
        if job is not None:
            metrics.bind(job)
        metrics.tokens(2, time.perf_counter())

    def test_success_releases_completed_refs_and_counts_once(self):
        app, job = App(), Job()
        app_ref, job_ref = weakref.ref(app), weakref.ref(job)
        self.begin(app, job)
        metrics.finish_request()
        counters = app.metrics
        self.assertEqual((counters.prompt, counters.generation, counters.drafted, counters.accepted), (4, 2, 7, 3))
        self.assertEqual((counters.latency.n, counters.ttft.n), (1, 1))
        self.assertIsNone(metrics._local.app)
        self.assertIsNone(metrics._local.job)
        del app, job
        gc.collect()
        self.assertIsNone(app_ref())
        self.assertIsNone(job_ref())
        metrics.finish_request()
        self.assertEqual(counters.latency.n, 1)

    def test_failure_does_not_retain_TLS_refs_or_replace_primary(self):
        app, job = App(), Job()
        self.begin(app, job)
        primary = RuntimeError("controlled telemetry failure")

        def fail(*args, **kwargs):
            raise primary

        with patch.object(metrics, "note", fail), self.assertRaises(RuntimeError) as caught:
            metrics.finish_request()
        self.assertIs(caught.exception, primary)
        self.assertFalse(metrics._local.armed)
        self.assertIsNone(metrics._local.app)
        self.assertIsNone(metrics._local.job)
        metrics.finish_request()  # failed completion remains exactly one attempted fold
        self.assertFalse(hasattr(app, "metrics"))

    def test_next_request_owns_only_its_new_job_and_app(self):
        old_app, old_job = App(), Job()
        old = weakref.ref(old_job)
        self.begin(old_app, old_job)
        metrics.finish_request()
        del old_job
        new_app, new_job = App(), Job()
        self.begin(new_app, new_job)
        self.assertIs(metrics._local.app, new_app)
        self.assertIs(metrics._local.job, new_job)
        metrics.finish_request()
        gc.collect()
        self.assertIsNone(old())
        self.assertEqual(old_app.metrics.latency.n, 1)
        self.assertEqual(new_app.metrics.latency.n, 1)

    def test_reentrant_new_request_is_not_cleared_after_the_fold(self):
        app, next_app = App(), App()
        self.begin(app, Job())
        original = metrics.note

        def note_then_begin(target, **values):
            original(target, **values)
            metrics.begin(next_app, 9, time.perf_counter())

        with patch.object(metrics, "note", note_then_begin):
            metrics.finish_request()
        self.assertTrue(metrics._local.armed)
        self.assertIs(metrics._local.app, next_app)
        metrics.finish_request()
        self.assertEqual((app.metrics.prompt, next_app.metrics.prompt), (4, 9))


if __name__ == "__main__":
    unittest.main()

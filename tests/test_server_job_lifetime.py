"""Request/background job ownership with native-free scheduler boundary fixtures."""

import ast
from concurrent.futures import Future
import importlib.util
from pathlib import Path
import queue
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from tensorfold.server.cancellation import Cancellation, RequestCancelled
from tensorfold.server.errors import RequestError, RoundError
from tensorfold.server import http
from tensorfold.thread_work import ThreadWork


class Terminal:
    def __init__(self):
        self.finished = False
        self.waits = []

    def is_set(self):
        return self.finished

    def set(self):
        self.finished = True

    def wait(self, timeout):
        self.waits.append(timeout)
        return self.finished


class Job:
    def __init__(self, **values):
        self.__dict__.update(values)
        self.cancellation = values.get("cancellation", Cancellation())
        self.done, self.chunks = Terminal(), queue.Queue()
        self.error, self.preempted, self.cached_tokens = None, False, 0
        self.scored = ([1.0, 2.0], 3.0)


def load(name):
    """Load the production request module; replace only the native scheduler boundary."""
    source = Path(__file__).parents[1] / "src/tensorfold/server" / (name + ".py")
    spec = importlib.util.spec_from_file_location("job_lifetime_" + name, source)
    module = importlib.util.module_from_spec(spec)
    scheduler = ModuleType("tensorfold.server.scheduler")
    scheduler.ChatJob = Job
    tree = ast.parse((source.parent / "scheduler.py").read_bytes())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Scheduler")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_terminal_error")
    namespace = {"RoundError": RoundError, "RequestCancelled": RequestCancelled}
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    scheduler.Scheduler = SimpleNamespace(_terminal_error=namespace["_terminal_error"])
    with patch.dict(sys.modules, {scheduler.__name__: scheduler}):
        spec.loader.exec_module(module)
    return module


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, tokens):
        return "".join(map(chr, tokens))

    def apply_chat_template(self, messages, **kwargs):
        text = messages[0]["content"] + "\n"
        return text if kwargs.get("tokenize", True) is False else self.encode(text)


class DecisionsTests(unittest.TestCase):
    def app(self, submit):
        module = load("decision_requests")
        owner = type("Owner", (module.DecisionRequests,), {})()
        owner.tokenizer, owner.tokenizer_lock, owner.context_window = Tokenizer(), threading.Lock(), 0
        owner._decision_keep = lambda *args: (0, (), 0)
        jobs, cancelled = [], []

        def accept(job):
            jobs.append(job)
            submit(job, len(jobs))

        def cancel(value):
            cancelled.append(value)
            value.cancel()
            for job in jobs:
                if job.cancellation is value:
                    job.error = RequestCancelled("request cancelled")
                    job.done.set()

        owner.scheduler = SimpleNamespace(submit=accept, cancel=cancel)
        body = {
            "input": "Input",
            "questions": [
                {
                    "id": str(i),
                    "type": "choice",
                    "question": "Which?",
                    "options": [{"name": "first"}, {"name": "second"}],
                }
                for i in range(3)
            ],
        }
        return owner, body, jobs, cancelled

    def test_success_keeps_completed_scores_and_does_not_cancel(self):
        owner, body, jobs, cancelled = self.app(lambda job, _: job.done.set())
        result = owner.decisions(body)
        self.assertEqual(list(result["answers"]), ["0", "1", "2"])
        self.assertTrue(all(job.done.is_set() for job in jobs))
        self.assertEqual(cancelled, [])

    def test_timeout_cancels_every_unfinished_owned_job(self):
        owner, body, jobs, cancelled = self.app(lambda *_: None)
        with self.assertRaisesRegex(TimeoutError, "did not score"):
            owner.decisions(body)
        self.assertEqual(cancelled, [job.cancellation for job in jobs])
        self.assertTrue(all(job.done.is_set() and job.cancellation.cancelled for job in jobs))
        self.assertEqual(jobs[0].done.waits, [600.0])

    def test_scoring_error_preserves_primary_and_cancels_later_jobs(self):
        primary = ValueError("failed scoring")

        def submit(job, index):
            if index == 1:
                job.error = primary
                job.done.set()

        owner, body, jobs, cancelled = self.app(submit)
        with self.assertRaisesRegex(RequestError, "question '0': failed scoring") as raised:
            owner.decisions(body)
        self.assertIs(raised.exception.__cause__, primary)
        self.assertEqual(cancelled, [job.cancellation for job in jobs[1:]])

    def test_partial_accept_failure_cancels_accepted_and_failing_job(self):
        primary = RuntimeError("submit failed after acceptance")

        def submit(job, index):
            if index == 2:
                raise primary

        owner, body, jobs, cancelled = self.app(submit)
        with self.assertRaises(RuntimeError) as raised:
            owner.decisions(body)
        self.assertIs(raised.exception, primary)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(cancelled, [job.cancellation for job in jobs])

    def test_cancel_failure_is_attached_and_remaining_jobs_still_cancel(self):
        owner, body, jobs, cancelled = self.app(lambda *_: None)
        cancel = owner.scheduler.cancel

        def fail_once(value):
            if value is jobs[0].cancellation:
                raise OSError("cancel boundary failed")
            cancel(value)

        owner.scheduler.cancel = fail_once
        with self.assertRaises(TimeoutError) as raised:
            owner.decisions(body)
        self.assertIn("failed: OSError", raised.exception.__notes__[0])
        self.assertEqual(cancelled, [job.cancellation for job in jobs[1:]])

    def test_overridden_note_and_cleanup_formatting_never_mask_primary(self):
        class Primary(BaseException):
            def add_note(self, text):
                raise AssertionError("foreign add_note hook was invoked")

        class Cleanup(Exception):
            def __str__(self):
                raise AssertionError("foreign cleanup formatting was invoked")

        primary = Primary("original failure")

        def submit(*_):
            raise primary

        owner, body, jobs, cancelled = self.app(submit)

        def fail(*_):
            raise Cleanup()

        owner.scheduler.cancel = fail
        with self.assertRaises(Primary) as raised:
            owner.decisions(body)
        self.assertIs(raised.exception, primary)
        self.assertEqual(len(jobs), 1)
        self.assertIn("failed: Cleanup", primary.__notes__[0])


class WarmupTests(unittest.TestCase):
    def warm(self, submit):
        module = load("prompt_blocks")
        owner = type("Owner", (module.PromptBlocks,), {})()
        owner.tokenizer = Tokenizer()
        owner.engine = SimpleNamespace(prefill_plan=object())
        owner.snapshot_registry = object()
        jobs = []

        def accept(job):
            jobs.append(job)
            submit(job, len(jobs))
            job.done.set()
            job.chunks.put(None)

        owner.scheduler = SimpleNamespace(submit=accept)
        plan = ModuleType("tensorfold.engine.prefill_plan")
        plan.block_jobs = lambda *_: [([1, 2, 0], 2), ([1, 2, 3, 4, 0], 4)]
        disk = ModuleType("tensorfold.engine.prefix_snapshots")
        disk.blocks_to_warm = lambda *_, **unused: [[1, 2, 3, 4]]
        with patch.dict(sys.modules, {plan.__name__: plan, disk.__name__: disk}):
            owner._warm_known_blocks(Path("unused-fixture-directory"), "fixture-model")
            owner.warmup_work.drain(owner.warmup_thread, timeout=2, cancel_unentered=False)
        self.assertFalse(owner.warmup_thread.is_alive())
        self.assertFalse(owner.warming)
        return owner, jobs

    def test_success_has_terminal_future_and_all_chunks(self):
        owner, jobs = self.warm(lambda *_: None)
        self.assertIsNone(owner.warmup_result.result())
        self.assertEqual([job.history_len for job in jobs], [2, 4])
        self.assertTrue(all(job.cancellation is owner.warmup_cancellation for job in jobs))

    def test_failure_stops_warmup_and_preserves_error_without_frames(self):
        def fail(job, _):
            job.error = ValueError("private failure details")
            job.preempted = True  # failure takes precedence over the normal retry policy

        owner, jobs = self.warm(fail)
        self.assertEqual(len(jobs), 1)
        error = owner.warmup_result.exception()
        self.assertIsInstance(error, RoundError)
        self.assertEqual((error.error_type, str(error)), ("ValueError", "private failure details"))
        self.assertIsNone(error.__traceback__)

    def test_foreign_error_format_cannot_strand_warmup_future(self):
        class Foreign(BaseException):
            def __str__(self):
                raise AssertionError("foreign formatter")

        def fail(job, _):
            job.error = Foreign("owned failure")

        owner, jobs = self.warm(fail)
        error = owner.warmup_result.exception(timeout=1)
        self.assertEqual(error.error_type, "Foreign")
        self.assertEqual(str(error), "owned failure")
        self.assertEqual(len(jobs), 1)
        self.assertTrue(owner.warmup_work.retired.is_set())

    def test_preempted_success_retries_same_chunk_then_advances(self):
        def preempt(job, index):
            job.preempted = index == 1

        owner, jobs = self.warm(preempt)
        self.assertIsNone(owner.warmup_result.result())
        self.assertEqual([job.history_len for job in jobs], [2, 2, 4])

    def test_health_exposes_only_sanitized_terminal_error(self):
        result = Future()
        result.set_exception(RoundError("ValueError", "private prompt/traceback text"))
        app = SimpleNamespace(
            served_name="fixture", model_ids=["fixture"], max_batch_size=4, warming=False, warmup_result=result
        )
        handler = http.make_handler(app).__new__(http.make_handler(app))
        handler.path = "/health"
        from http.client import HTTPMessage
        from io import BytesIO

        handler.headers, handler.rfile = HTTPMessage(), BytesIO()
        found = []
        handler._send_json = found.append
        with patch.object(http, "_memory", return_value={}):
            handler.do_GET()
        self.assertEqual(found[0]["warmup_error"], "saved-prefix warmup failed: ValueError")
        self.assertNotIn("private", str(found))

    def test_close_cancels_warmup_before_stopping_and_joins_worker(self):
        # Evaluate only the production close method; importing the model app would
        # import native runtimes outside this scheduler-boundary fixture's scope.
        source = Path(__file__).parents[1] / "src/tensorfold/server/app.py"
        cls = next(
            n for n in ast.parse(source.read_bytes()).body if isinstance(n, ast.ClassDef) and n.name == "ChatApp"
        )
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "close")
        namespace = {"threading": threading}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        entered, cancelled = threading.Event(), threading.Event()

        def worker():
            entered.set()
            if not cancelled.wait(2):
                raise TimeoutError("owned warmup fixture not cancelled")

        work = ThreadWork(worker)
        thread = threading.Thread(target=work.run)
        work.launch(thread)
        self.assertTrue(entered.wait(1))
        order = []
        app = SimpleNamespace(
            warmup_thread=thread, warmup_work=work, warmup_cancellation=object(), save_sessions=lambda: None
        )

        def cancel(value):
            self.assertIs(value, app.warmup_cancellation)
            order.append("cancel")
            cancelled.set()

        def stop(timeout):
            self.assertEqual(timeout, 120)
            order.append("stop")

        app.scheduler = SimpleNamespace(cancel=cancel, stop=stop, _thread=SimpleNamespace(is_alive=lambda: False))
        try:
            namespace["close"](app)
        finally:
            cancelled.set()
            thread.join(2)
        self.assertEqual(order, ["cancel", "stop"])
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()

"""Maintained front-end replay and request lifetime, without native model imports."""
import ast
from contextlib import redirect_stdout
import hashlib
import http.client
import io
import json
from pathlib import Path
import queue
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import weakref
import gc

from tensorfold.server.cancellation import Cancellation
from tensorfold.server.errors import CONTEXT_LIMIT, ContextLengthError, RequestError, RoundError
from tensorfold.server.http import Server, make_handler
from tensorfold.server.text import (CHANNEL_MARKERS, IncrementalText, is_title_request, parse_harmony_output,
                                    reasoning_count, split_thinking, streaming_visible_text, strip_trailing_stops)


ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    def decode(self, ids, **unused):
        return "".join(chr(token) for token in ids)

    def encode(self, text, **unused):
        return list(map(ord, text))


class Job:
    def __init__(self, **values):
        self.__dict__.update(values)
        self.chunks = queue.Queue()
        self.error = None
        self.cached_tokens, self.preempted = 0, False
        self.submitted_at, self.prefilled_at, self.finished_at = 1., 2., 3.
        self.started_at = 1.
        self.stream = SimpleNamespace(finish_reason="length", rounds=1, drafted=0, accepted=0, proposer=None)


class Stops:
    ignore_eos, strings, eos_ids = False, (), ()

    def __init__(self, *unused):
        pass

    def visible(self, text, **unused):
        return text

    def flush(self, *unused):
        pass


class Fields(dict):
    pass


def maintained():
    path = ROOT / "src/tensorfold/server/app.py"
    original = next(node for node in ast.parse(path.read_bytes()).body
                    if isinstance(node, ast.ClassDef) and node.name == "ChatApp")
    selected = ast.ClassDef(name="RequestApp", bases=[], keywords=[], decorator_list=[], body=[
        node for node in original.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in ("chat", "_chat_prepared", "_Preparing")])
    source = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), selected],
                        type_ignores=[])
    ast.fix_missing_locations(source)
    local, counts = threading.local(), []
    metrics = SimpleNamespace(begin=lambda *a: None, bind=lambda *a: None,
                              tokens=lambda *a: counts.append(a), finish_request=lambda: None)
    namespace = dict(threading=threading, time=time, queue=queue, uuid=uuid, _REQUEST=local, metrics=metrics,
                     Cancellation=Cancellation, ChatJob=Job, StopPolicy=Stops, RoundError=RoundError,
                     RequestError=RequestError, CONTEXT_LIMIT=CONTEXT_LIMIT, ContextLengthError=ContextLengthError,
                     is_title_request=is_title_request, IncrementalText=IncrementalText, CHANNEL_MARKERS=CHANNEL_MARKERS,
                     split_thinking=split_thinking, parse_harmony_output=parse_harmony_output,
                     strip_trailing_stops=strip_trailing_stops, reasoning_count=reasoning_count,
                     streaming_visible_text=streaming_visible_text,
                     _token_sha=lambda ids: hashlib.sha256(",".join(str(t) for t in ids).encode()).hexdigest()[:12],
                     grammar=SimpleNamespace(FIELDS=(), request_constraint=lambda *a: None))
    exec(compile(source, str(path), "exec"), namespace)
    return namespace["RequestApp"], local, metrics, counts


def fixture(second, *, second_error=None):
    cls, local, metrics, counts = maintained()
    app = cls()
    app.tokenizer, app.tokenizer_lock = Tokenizer(), threading.Lock()
    app.default_max_tokens, app.context_window = 32, 0
    app.enable_thinking, app.use_proposer = False, False
    app.thinking_budget, app.min_match = 0, 4
    app.stop_ids, app.think_markers = [], ("", "</think>")
    app.system_prefix_len = lambda *a, **k: 0
    app._resolve_sampling = lambda *a: None
    app._call_gate = lambda *a: None
    app._round_profile = lambda *a: ""
    app.effort_for = lambda *a: None
    app.exact_mode, app.checkpoints = {"mode": "exact", "engine": "lanes"}, None
    app.requests_completed, app._preparing = 0, 0
    app._preparing_lock = threading.Lock()
    jobs, cancelled = [], []

    def submit(job):
        jobs.append(job)
        chunks = ([97, 98], None) if len(jobs) == 1 else second
        job.preempted = len(jobs) == 1
        job.error = second_error if len(jobs) > 1 else None
        for chunk in chunks:
            job.chunks.put(chunk)

    def cancel(value):
        cancelled.append(value)
        value.cancel()

    app.scheduler = SimpleNamespace(submit=submit, cancel=cancel, engine=SimpleNamespace(round_stats=[]))
    prompts = ModuleType("tensorfold.server.prompts")
    prompts.prepare_prompt = lambda *a: SimpleNamespace(tokens=[10, 11], history_len=0, vision=None)
    return app, local, metrics, counts, jobs, cancelled, prompts


class Replay(unittest.TestCase):
    def run_request(self, second, **kwargs):
        app, local, metrics, counts, jobs, cancelled, prompts = fixture(second, **kwargs)
        emitted = []
        with patch.dict(sys.modules, {prompts.__name__: prompts}), redirect_stdout(io.StringIO()):
            try:
                result = app.chat([{"role": "user", "content": "x"}], on_delta=emitted.append)
            except BaseException as error:
                result = error
        return result, emitted, app, local, jobs, cancelled

    def test_matching_replay_preserves_text_usage_and_drops_only_owed_tokens(self):
        result, emitted, app, local, jobs, cancelled = self.run_request(([97], [98, 99], None))
        self.assertEqual((result["content"], result["completion_tokens"], result["prompt_tokens"]), ("abc", 3, 2))
        self.assertEqual("".join(emitted), "abc")
        self.assertEqual(app.requests_completed, 1)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(cancelled, [])
        self.assertIsNone(local.sampling)

    def test_changed_replay_refuses_before_changed_or_new_bytes(self):
        result, emitted, app, local, jobs, cancelled = self.run_request(([120, 98, 99], None))
        self.assertIsInstance(result, RoundError)
        self.assertEqual(result.error_type, "ReplayDivergence")
        self.assertEqual(emitted, ["ab"])
        self.assertEqual(app.requests_completed, 0)
        self.assertEqual(len(cancelled), 1)
        self.assertTrue(jobs[-1].cancellation.cancelled)
        self.assertIsNone(local.sampling)

    def test_early_end_refuses_after_matching_partial_replay(self):
        for chunks in ((None,), ([97], None)):
            result, emitted, app, local, jobs, cancelled = self.run_request(chunks)
            self.assertIsInstance(result, RoundError)
            self.assertEqual(result.error_type, "ReplayDivergence")
            self.assertEqual(emitted, ["ab"])
            self.assertEqual(app.requests_completed, 0)
            self.assertEqual(len(cancelled), 1)

    def test_terminal_backend_error_keeps_precedence_over_unfinished_replay(self):
        primary = RoundError("BackendFailure", "owned error")
        result, emitted, *_ = self.run_request(([97], None), second_error=primary)
        self.assertIs(result, primary)
        self.assertEqual(emitted, ["ab"])

    def test_partial_http_stream_publishes_error_done_and_no_success_shape(self):
        app, _, _, _, _, _, prompts = fixture(([120, 98, 99], None))
        app.accepts_sampling, app.accepts_cancellation = True, True
        app.accepts_raw_prompt = app.streams_prose_with_tools = True
        app.served_name, app.model_ids = "fixture", ["fixture"]
        app.vision = None
        server = Server(("127.0.0.1", 0), make_handler(app))
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        connection = http.client.HTTPConnection(*server.server_address, timeout=3)
        try:
            with patch.dict(sys.modules, {prompts.__name__: prompts}), redirect_stdout(io.StringIO()):
                body = json.dumps({"model": "fixture", "messages": [{"role": "user", "content": "x"}], "stream": True})
                connection.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                raw = response.read().decode()
            rows = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
            text = "".join(row.get("choices", [{}])[0].get("delta", {}).get("content", "") for row in rows)
            self.assertEqual(text, "ab")
            self.assertEqual(rows[-1]["error"]["type"], "server_error")
            self.assertTrue(raw.endswith("data: [DONE]\n\n"))
            self.assertFalse(any(row.get("choices", [{}])[0].get("finish_reason") is not None for row in rows))
            self.assertEqual(app.requests_completed, 0)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(3)
        self.assertFalse(thread.is_alive())


class Lifetime(unittest.TestCase):
    def owner(self):
        cls, local, metrics, _ = maintained()
        app = cls()
        app.default_max_tokens, app._preparing = 32, 0
        app._preparing_lock = threading.Lock()
        app.scheduler = SimpleNamespace(cancel=lambda value: value.cancel())
        return app, local, metrics

    def test_completed_success_and_failure_release_sampling_fields(self):
        for fail in (False, True):
            app, local, metrics = self.owner()
            fields = Fields(temperature=0)
            ref = weakref.ref(fields)
            primary = RoundError("owned", "request error")

            def operation(*a, **k):
                self.assertIs(local.sampling, ref())
                if fail:
                    raise primary
                return "ok"

            app._chat_prepared = operation
            if fail:
                with self.assertRaises(RoundError) as caught:
                    app.chat([], sampling=fields)
                self.assertIs(caught.exception, primary)
                primary.__traceback__ = None
            else:
                self.assertEqual(app.chat([], sampling=fields), "ok")
            del fields
            gc.collect()
            self.assertIsNone(ref())
            self.assertIsNone(local.sampling)
            self.assertEqual(app._preparing, 0)

    def test_nested_requests_restore_outer_fields_then_release(self):
        app, local, metrics = self.owner()
        outer, inner = Fields(outer=True), Fields(inner=True)
        calls = []

        def operation(*a, **k):
            calls.append(local.sampling)
            if local.sampling is outer:
                self.assertEqual(app.chat([], sampling=inner), "inner")
                self.assertIs(local.sampling, outer)
                return "outer"
            return "inner"

        app._chat_prepared = operation
        self.assertEqual(app.chat([], sampling=outer), "outer")
        self.assertEqual(calls, [outer, inner])
        self.assertIsNone(local.sampling)

    def test_cleanup_failures_preserve_primary_and_attempt_all_owners(self):
        app, local, metrics = self.owner()

        class Primary(BaseException):
            def add_note(self, text):
                raise AssertionError("foreign note hook invoked")

        primary = Primary()
        cancellation_failure, metrics_failure = RuntimeError("cancel failed"), RuntimeError("fold failed")
        order = []

        def operation(*a, **k):
            raise primary

        def cancel(value):
            order.append("cancel")
            raise cancellation_failure

        def finish():
            order.append("metrics")
            raise metrics_failure

        app._chat_prepared = operation
        app.scheduler.cancel, metrics.finish_request = cancel, finish
        with self.assertRaises(Primary) as caught:
            app.chat([], sampling=Fields())
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, cancellation_failure)
        self.assertEqual(order, ["cancel", "metrics"])
        self.assertEqual(len(primary.__notes__), 2)
        self.assertEqual(app._preparing, 0)
        self.assertIsNone(local.sampling)


if __name__ == "__main__":
    unittest.main()

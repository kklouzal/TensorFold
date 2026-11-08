"""Scenario protocol/lifecycle controls with HTTP fixtures; no model qualification."""
import copy
import hashlib
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import urllib.error

import pytest

from tools import v100_adaptive_exercise as exercise
from tools import v100_rotorquant_live as live


def request(kind="short", maximum=3):
    result = {"model": "target", "messages": [{"role": "user", "content": kind}],
              "max_tokens": maximum, "temperature": 0, "seed": 17}
    if kind == "stop":
        result["stop"] = "END"
    return result


def statistics(capacity=2):
    return {"object": "tensorfold.stats", "server": {"instance_id": "unchanged"},
            "model": {"id": "target", "aliases": [], "context_length": 32, "kv_dtype": "int8"},
            "totals": {"cancelled": 0},
            "load": {"requests_in_flight": 0, "requests_running": 0, "requests_waiting": 0},
            "capacity_plan": {
                "session_capacity": {"resident_capacity": capacity, "context_tokens": 32,
                                     "full_context_commitments": capacity},
                "resource_ledger_bytes": {"migration_overlap_limit": 64 << 20},
                "speculation": {"method": "mtp"},
                "kv_full_windows": {"materialized": True, "reserved": capacity, "stream_slots": capacity,
                    "ranks": [{"rank": rank, "materialized": True, "startup_capacities": [36] * capacity,
                               "allocated_cache_bytes": rank + 1, "torch_allocated_bytes": 100,
                               "torch_reserved_bytes": 200} for rank in range(8)]}}}


class Service:
    """Explicit endpoint responses; runtime active/retained state is independent of allocated slots."""
    counts = {"short": 4, "long": 20, "intermediate": 12, "stop": 5, "exact": 29, "overrun": 31}
    keep = True
    shrink_tokenizer_context = False
    refusal_code = "context_length_exceeded"
    last = None

    def __init__(self, _url, out, **_kwargs):
        Service.last = self
        self.out, self.lock = out, threading.Lock()
        self.active = self.prefill = self.decode = self.cancelled = self.generations = 0
        self.kept = set()
        self.capacity = 2
        self.degree, self.pending_degree = 4, None
        self.pending_reads = 0
        self.settings = {"prefill_tokens": 16, "proposal_limit": 3, "row_budget": 16, "session_ceiling": 2}
        self.original = copy.deepcopy(self.settings)
        self.seen = []

    def runtime(self):
        if self.pending_degree is not None:
            self.pending_reads += 1
            if self.pending_reads >= 2:
                self.degree, self.pending_degree = self.pending_degree, None
        migration = {"requested_degree": self.pending_degree or self.degree,
                     "pending": self.pending_degree is not None, "blocked": None}
        return {"object": "tensorfold.runtime", "context_tokens": 32,
                "capacity_plan": statistics()["capacity_plan"],
                "execution": {"epoch": self.generations, "effective": copy.deepcopy(self.settings), "migration": migration},
                "sessions": {"allocated_slots": self.capacity, "retained": min(self.capacity, len(self.kept)) if self.keep else 0,
                             "active": self.active, "decode": self.decode, "prefill": self.prefill,
                             "admission_ceiling": self.settings["session_ceiling"]},
                "placement": {"slots": [{"slot": index, "slot_generation": self.generations,
                    "target_cached_length": 4 if self.kept else 0, "history_poisoned": False,
                    "attention_degree": self.degree, "pending_attention_degree": self.pending_degree,
                    "migration_bytes": 1024 if self.pending_degree else 0} for index in range(self.capacity)]}}

    def call(self, label, body=None, *, path=None, cancel=False, observer=None, **_kwargs):
        directory = self.out / label
        directory.mkdir()
        with self.lock:
            self.seen.append((label, copy.deepcopy(body), path))
            if path == "/v1/runtime":
                if body is not None:
                    if "attention_degree" in body:
                        self.pending_degree, self.pending_reads = body["attention_degree"], 0
                    else:
                        self.settings.update(body)
                    result = {"object": "tensorfold.runtime.update"}
                else:
                    result = self.runtime()
            elif path == "/v1/tokenize":
                count = self.counts[body["messages"][0]["content"]]
                result = {"count": count, "tokens": list(range(count)),
                          "max_model_len": 31 if self.shrink_tokenizer_context else 32}
            elif body is None:
                result = statistics()
                result["totals"]["cancelled"] = self.cancelled
                result["load"].update(requests_in_flight=self.active, requests_running=self.active)
            else:
                result = None
        if result is not None:
            (directory / "response.raw").write_text(json.dumps(result))
            return result
        name = body["messages"][0]["content"]
        if name == "overrun":
            raw = json.dumps({"error": {"code": self.refusal_code}}).encode()
            (directory / "response.raw").write_bytes(raw)
            raise urllib.error.HTTPError("http://unused", 400, "context", {}, io.BytesIO(raw))
        with self.lock:
            self.active += 1
            self.prefill += 1
            self.kept.add(name)
            self.generations += 1
        try:
            time.sleep(.015 if name == "long" else .002)
            with self.lock:
                self.prefill -= 1
                self.decode += 1
            if observer is not None:
                observer({"choices": [{"index": 0, "delta": {"content": "answer"}}]}, ["answer"])
            time.sleep(.03)
            if cancel:
                with self.lock:
                    self.cancelled += 1
                return {"intentionally_incomplete": True}
            count = self.counts[name]
            result = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"},
                                    "finish_reason": "stop" if name == "stop" else "length"}],
                      "usage": {"prompt_tokens": count, "completion_tokens": body["max_tokens"],
                                "total_tokens": count + body["max_tokens"]},
                      "tensorfold": {"token_ids": list(range(body["max_tokens"])), "cached": max(0, count - 1)}}
            (directory / "response.raw").write_text(json.dumps(result))
            return result
        finally:
            with self.lock:
                self.active -= 1
                self.decode -= 1


def arguments(tmp_path, scenarios, names=("short", "long", "intermediate", "stop", "exact", "overrun")):
    named = []
    for name in names:
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(request(name)))
        named.append(name + "=" + str(path))
    return SimpleNamespace(url="http://unused", out=tmp_path / "run", request=named, scenarios=scenarios,
                           request_mib=1, response_mib=1, sse_line_kib=4, max_inflight=8,
                           batch_response_mib=8, max_calls=1000, max_cleanup_calls=100, max_events=100,
                           timeout=1., operation_timeout=1., poll_timeout=.3, poll_interval=.001, cycles=2)


@pytest.fixture(autouse=True)
def service(monkeypatch):
    monkeypatch.setattr(exercise.live, "Client", Service)
    monkeypatch.setattr(Service, "keep", True)
    monkeypatch.setattr(Service, "shrink_tokenizer_context", False)
    monkeypatch.setattr(Service, "refusal_code", "context_length_exceeded")


def test_explicit_request_lifecycle_counts_full_context_and_restores_live_settings(tmp_path):
    args = arguments(tmp_path, ["requests", "mixed", "burst", "cancel", "reuse", "retention",
                                "migration", "oscillation", "limits", "zero-proposal", "stop", "context"])
    assert exercise.run(args) == 0
    report = json.loads((args.out / "suite.json").read_text())
    assert report["complete"] and report["passed"]
    assert report["qualification"] == "not-granted-by-this-tool"
    assert report["inputs"]["long"]["prompt_tokens"] == 20
    assert report["inputs"]["exact"]["prompt_tokens"] + report["inputs"]["exact"]["max_tokens"] == 32
    assert report["inputs"]["overrun"]["fits"] is False
    assert all(row["status"] == "passed" for row in report["scenarios"])
    assert Service.last.settings == Service.last.original and Service.last.degree == 4
    outcomes = {row["scenario"]: row["result"] for row in report["scenarios"]}
    assert outcomes["retention"]["retained_before"] == outcomes["retention"]["retained_after"] == 2
    assert outcomes["migration"]["partial_migration_observed"]
    assert outcomes["burst"]["observed"]["peak_active"] == 2
    assert any(row["status"] == "expected-refusal" for row in report["calls"])
    assert list(args.out.glob("*/timing.json"))


def test_allocated_slots_never_prove_retained_capacity(monkeypatch, tmp_path):
    monkeypatch.setattr(Service, "keep", False)
    args = arguments(tmp_path, ["retention"])
    assert exercise.run(args) == 2
    result = json.loads((args.out / "suite.json").read_text())
    assert result["scenarios"][0]["status"] == "unsupported"
    assert "does not retain" in result["scenarios"][0]["reason"]
    assert Service.last.capacity == 2


def test_actual_tokenizer_context_mismatch_stops_before_generation(monkeypatch, tmp_path):
    monkeypatch.setattr(Service, "shrink_tokenizer_context", True)
    args = arguments(tmp_path, ["requests"])
    with pytest.raises(ValueError, match="exact IDs/count"):
        exercise.run(args)
    assert all(body is None or path == "/v1/tokenize" for _label, body, path in Service.last.seen)
    result = json.loads((args.out / "suite.json").read_text())
    assert not result["complete"] and result["unrun_scenarios"] == ["requests"]


def test_missing_explicit_long_request_is_unsupported_without_synthesizing_prompts(tmp_path):
    args = arguments(tmp_path, ["mixed", "migration"], names=("short",))
    assert exercise.run(args) == 2
    report = json.loads((args.out / "suite.json").read_text())
    assert all(row["status"] == "unsupported" for row in report["scenarios"])
    generation = [body for _label, body, path in Service.last.seen if body is not None and path is None]
    assert not generation


def test_cumulative_input_budget_and_boolean_single_reply_are_rejected(tmp_path):
    path = tmp_path / "body.json"
    path.write_text(json.dumps(request()))
    with pytest.raises(ValueError, match="cumulative byte"):
        exercise.load_requests(["one=" + str(path), "two=" + str(path)], budget_bytes=path.stat().st_size)
    body = request()
    body["n"] = True
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="single text-only"):
        exercise.load_requests(["one=" + str(path)], budget_bytes=4096)


def test_cleanup_has_separate_bounded_budget_and_preserves_original_settings(tmp_path):
    args = arguments(tmp_path, ["limits"])
    args.max_calls = 10
    with pytest.raises(RuntimeError, match="HTTP-call budget"):
        exercise.run(args)
    assert Service.last.settings == Service.last.original
    result = json.loads((args.out / "suite.json").read_text())
    assert result["scenarios"][0]["status"] == "failed"
    assert not result["complete"]


def test_unrelated_http400_never_passes_context_refusal_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(Service, "refusal_code", "unrelated_validation_error")
    args = arguments(tmp_path, ["context"])
    with pytest.raises(urllib.error.HTTPError):
        exercise.run(args)
    report = json.loads((args.out / "suite.json").read_text())
    assert not report["complete"]
    assert report["scenarios"][0]["status"] == "failed"


def test_early_eos_never_claims_actual_context_exhaustion(monkeypatch, tmp_path):
    original = Service.call

    def early(self, label, body=None, **kwargs):
        value = original(self, label, body, **kwargs)
        if body is not None and kwargs.get("path") is None and body["messages"][0]["content"] == "exact":
            value["choices"][0]["finish_reason"] = "stop"
            value["usage"].update(completion_tokens=2, total_tokens=31)
            value["tensorfold"]["token_ids"] = [0, 1]
            (self.out / label / "response.raw").write_text(json.dumps(value))
        return value

    monkeypatch.setattr(Service, "call", early)
    args = arguments(tmp_path, ["context"])
    assert exercise.run(args) == 2
    report = json.loads((args.out / "suite.json").read_text())
    scenario = report["scenarios"][0]
    assert scenario["status"] == "unsupported" and "before context exhaustion" in scenario["reason"]
    assert scenario["observed_evidence"]["exact_window_requests"][0]["tokens"] == 2


def test_partial_task_start_failure_releases_waiting_callbacks_without_http(monkeypatch, tmp_path):
    args = arguments(tmp_path, ["burst"], names=("short",))
    args.out.mkdir()
    rows = exercise.load_requests(args.request, budget_bytes=1 << 20)
    rows["short"].update(fits=True, prompt_tokens=4)
    suite = exercise.Suite(args, rows)
    suite.effective = exercise.qualification.profile(statistics())
    original, calls = exercise.Task.start, 0
    primary = RuntimeError("second worker acquisition failed")

    def start(task):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise primary
        return original(task)

    monkeypatch.setattr(exercise.Task, "start", start)
    with pytest.raises(RuntimeError) as caught:
        suite.concurrent(["short", "short"])
    assert caught.value is primary
    assert Service.last.active == 0 and not Service.last.seen


def test_cleanup_failure_does_not_replace_primary_scenario_failure(monkeypatch, tmp_path):
    args = arguments(tmp_path, ["limits"])
    args.max_calls = 10
    original = Service.call

    def call(self, label, body=None, **kwargs):
        if body is not None and set(body) == {"prefill_tokens", "proposal_limit", "row_budget", "session_ceiling"}:
            raise OSError("restoration transport failed")
        return original(self, label, body, **kwargs)

    monkeypatch.setattr(Service, "call", call)
    with pytest.raises(RuntimeError, match="HTTP-call budget") as caught:
        exercise.run(args)
    assert isinstance(caught.value.__cause__, OSError)
    report = json.loads((args.out / "suite.json").read_text())
    assert not report["complete"] and report["restoration_error"]["type"] == "OSError"


class Response(io.BytesIO):
    status = 200

    def __init__(self, raw, kind):
        super().__init__(raw)
        self.kind = kind

    def getheader(self, _name, _default=""):
        return self.kind

    def read1(self, amount):
        return self.read(amount)

    @property
    def headers(self):
        return {"Content-Type": self.kind}


def test_recording_client_supports_generic_objects_without_generation_usage(monkeypatch, tmp_path):
    # Fixture replaces the shared module's Client; recover its implementation
    # from the isolated import so the actual HTTP transport is tested here.
    import importlib.util
    spec = importlib.util.spec_from_file_location("tools._exercise_client_test", live.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    response = Response(b'{"count":3,"tokens":[1,2,3],"max_model_len":32}', "application/json")
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *_args, **_kwargs: response)
    result = module.Client("http://unused", tmp_path).call("tokenize", {"messages": []},
                                                          path="/v1/tokenize", object_response=True)
    assert result["tokens"] == [1, 2, 3] and response.closed
    assert json.loads((tmp_path / "tokenize/request.json").read_text())["body"] == {"messages": []}


def test_observer_failure_is_primary_when_exchange_receipt_also_fails(monkeypatch, tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("tools._exercise_client_failure_test", live.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = b'data: {"choices":[{"index":0,"delta":{"content":"a"}}]}\n\n'
    response = Response(raw, "text/event-stream")
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *_args, **_kwargs: response)
    original = module.write_json
    primary = RuntimeError("observer failed")

    def publish(path, value):
        if path.name == "exchange.json":
            raise OSError("receipt failed")
        return original(path, value)

    def observe(_event, _pieces):
        raise primary

    monkeypatch.setattr(module, "write_json", publish)
    with pytest.raises(RuntimeError, match="observer failed") as caught:
        module.Client("http://unused", tmp_path).call("failed", {"max_tokens": 3, "stream": True}, observer=observe)
    assert caught.value is primary and isinstance(caught.value.__cause__, OSError)
    assert response.closed and (tmp_path / "failed/response.raw").read_bytes() == raw


@contextmanager
def dribbling_service(*, stall_after=None, expected=1):
    stop, entered = threading.Event(), threading.Event()
    lock, started, finished = threading.Lock(), [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            size = int(self.headers["Content-Length"])
            if not 0 < size < 65536:
                raise ValueError("fixture request budget exceeded")
            self.rfile.read(size)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            with lock:
                started.append(self)
                if len(started) >= expected:
                    entered.set()
            began = time.monotonic()
            try:
                while not stop.is_set():
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    if stall_after is not None and time.monotonic() - began >= stall_after:
                        stop.wait(2)
                        return
                    stop.wait(.01)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with lock:
                    finished.append(self)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    owner = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": .01}, name="adaptive-http-fixture")
    owner.start()
    try:
        yield "http://127.0.0.1:" + str(httpd.server_port), entered
    finally:
        stop.set()
        httpd.shutdown()
        httpd.server_close()
        owner.join(timeout=3)
        deadline = time.monotonic() + 3
        while len(finished) != len(started) and time.monotonic() < deadline:
            time.sleep(.001)
        assert not owner.is_alive() and len(finished) == len(started)


def actual_client_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tools._exercise_deadline_transport", live.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def correctness_service(fault=None, *, long_capacity=0, allowance=64):
    """Independent HTTP answers; this fixture is not a model/tokenizer oracle."""
    state = SimpleNamespace(calls=[], stats_calls=0, lock=threading.Lock(), active=set(), samples=0)

    def answer(body):
        prompt = body["messages"][-1]["content"]
        if prompt.startswith("long-"):
            text = "SENTINEL-" + prompt.split("-")[-1]
            if fault == "answer":
                return "WRONG"
            return " \n" + text + "\t" if fault == "whitespace" else text
        if prompt.startswith("Calculate"):
            return str(17 * 23 - 19)
        if prompt.startswith("Start x"):
            if fault == "loop-31":
                return "31"  # Retained actual matching serial/MTP semantic failure; the oracle remains43.
            value = 3
            for number in [2, 5, 1]:
                value = 2 * value + number
            return str(value)
        if prompt.startswith("For the integers"):
            values = [7, 3, 11, 5]
            result = dict(count=len(values), sum=sum(values), max=max(values))
            if fault == "json-type":
                result["sum"] = float(result["sum"])
            return json.dumps(result)
        if prompt.startswith("From [pear"):
            return ",".join(sorted(word for word in ["pear", "fig", "apple", "kiwi"] if len(word) == 4))
        if prompt.startswith("Track these labels"):
            return "MOSS-2846"
        if prompt.startswith("Update beta"):
            return "KITE-7319|LIME-5082"
        return "AURORA-6197|VIOLET-2831"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, body, kind="application/json"):
            raw = json.dumps(body).encode() if isinstance(body, dict) else body
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass  # A negative client test may cancel a diagnostic mate.
            finally:
                if hasattr(self, "long_index"):
                    with state.lock:
                        state.active.discard(self.long_index)

        def do_GET(self):
            with state.lock:
                state.calls.append((self.path, None))
            if self.path == "/v1/runtime":
                if long_capacity:
                    with state.lock:
                        active = set(state.active)
                        state.samples += 1
                        selected = state.samples % long_capacity
                    floor = 4096 - allowance
                    slots = []
                    for index in range(long_capacity):
                        length = floor if index in active else 0
                        if fault == "underfilled" or fault == "union-only" and index != selected:
                            length = max(0, length - 1)
                        slots.append({"slot": index, "slot_generation": 1, "request_active": index in active,
                                      "target_cached_length": length, "history_poisoned": False})
                    return self.respond({"object": "tensorfold.runtime", "context_tokens": 4096,
                        "reading_contract": "advisory_host_snapshot_no_gpu_synchronization",
                        "sessions": {"retained": 0, "active": len(active), "decode": len(active), "prefill": 0},
                        "placement": {"slots": slots, "active_request_slots": len(active)},
                        "execution": {"effective": {"proposal_limit": 4}}})
                return self.respond({"object": "tensorfold.runtime", "context_tokens": 4096,
                    "sessions": {"retained": 0, "active": 0, "decode": 0, "prefill": 0},
                    "placement": {"slots": []}, "execution": {"effective": {}}})
            row = statistics(capacity=long_capacity or 1)
            row["model"]["context_length"] = 4096
            row["capacity_plan"]["session_capacity"]["context_tokens"] = 4096
            for rank in row["capacity_plan"]["kv_full_windows"]["ranks"]:
                rank["startup_capacities"] = [4101] * (long_capacity or 1)
            row["totals"]["failed"] = 0
            with state.lock:
                state.stats_calls += 1
                if fault == "identity" and state.stats_calls > 1:
                    row["server"]["instance_id"] = "another-process"
            self.respond(row)

        def do_POST(self):
            count = int(self.headers["Content-Length"])
            if not 0 < count <= 65536:
                raise ValueError("fixture request boundary")
            body = json.loads(self.rfile.read(count))
            with state.lock:
                state.calls.append((self.path, copy.deepcopy(body)))
            if self.path == "/v1/tokenize":
                if long_capacity:
                    index = int(body["messages"][-1]["content"].split("-")[-1])
                    ids = [100 + index, *range(4095 - allowance)]
                    if fault == "token-collision":
                        ids[0] = 100
                    return self.respond({"count": len(ids), "tokens": ids, "max_model_len": 4096})
                return self.respond({"count": 12, "tokens": list(range(12)), "max_model_len": 4096})
            if long_capacity:
                self.long_index = int(body["messages"][-1]["content"].split("-")[-1])
                with state.lock:
                    state.active.add(self.long_index)
                time.sleep(.10)  # Make same-snapshot overlap observable, not an inference performance fixture.
            text = answer(body)
            if fault == "answer" and body["messages"][-1]["content"].startswith("Calculate"):
                text = "wrong"
            ids = [ord(char) for char in text]
            if fault == "ids" and body["draft"] and not body["stream"]:
                ids[0] += 1
            sha = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:12]
            cached = 1 if fault == "cold" and not body["draft"] else 0
            stats = {"token_ids": ids, "token_sha": "bad" if fault == "sha" else sha,
                     "drafts": body["draft"], "drafted": int(body["draft"] and fault != "no-proposals"),
                     "accepted": 0, "cached": cached}
            prompt_count = 4096 - allowance if long_capacity else 12
            usage = {"prompt_tokens": prompt_count, "completion_tokens": len(ids), "total_tokens": prompt_count + len(ids),
                     "prompt_tokens_details": {"cached_tokens": cached},
                     "completion_tokens_details": {"reasoning_tokens": 0}}
            finish = "length" if fault == "finish" else "stop"
            if body["stream"]:
                if fault == "sse-dispatch":
                    stats["drafts"] = False
                frame = {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}],
                         "usage": usage, "tensorfold": stats}
                if fault == "sse-parity":
                    frame["choices"][0]["delta"]["content"] = "different"
                raw = b": retained comment\n\ndata: " + json.dumps(frame).encode() + b"\n\n"
                if fault != "sse-missing-done":
                    raw += b"data: [DONE]\n\n"
                if fault == "sse-duplicate-done":
                    raw += b"data: [DONE]\n\n"
                return self.respond(raw, "text/event-stream")
            self.respond({"choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                       "finish_reason": finish}], "usage": usage, "tensorfold": stats})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
    worker.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port), state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
        assert not worker.is_alive()


def test_correctness_uses_one_live_service_and_preserves_seven_case_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(exercise.live, "Client", actual_client_module().Client)
    args = arguments(tmp_path, ["correctness"], names=())
    with correctness_service() as (args.url, state):
        assert exercise.run(args) == 0
    report = json.loads((args.out / "suite.json").read_text())
    result = report["correctness"]
    assert result["passed"] and result["requests"] == 15 and len(result["cases"]) == 7
    assert result["drafted_total"] > 0
    generations = [body for path, body in state.calls if path == "/v1/chat/completions"]
    assert len(generations) == 15
    assert [body["draft"] for body in generations[:4]] == [True, False, False, True]
    assert all(body["max_tokens"] == 64 and body["temperature"] == 0 for body in generations)
    history = result["cases"][5]["messages"]
    assert history[-2] == {"role": "assistant", "content": result["cases"][4]["mtp"]["content"]}
    assert sum(line.startswith("noise entry ") for line in result["cases"][6]["messages"][-1]["content"].splitlines()) == 32
    assert result["stream"]["token_ids"] == result["cases"][2]["mtp"]["token_ids"]
    assert any(b"retained comment" in path.read_bytes() for path in args.out.glob("*/response.raw"))
    assert report["tool_sha256"] and result["contract_sha256"]
    assert result["runtime_identity"]["instance_id"] == "unchanged"
    for path in args.out.glob("*-stats/exchange.json"):
        exchange = json.loads(path.read_text())
        assert 0 < exchange["read_control"]["deadline_monotonic_s"] - exchange["acquired_monotonic_s"] <= args.operation_timeout


@pytest.mark.parametrize("fault", ["answer", "json-type", "ids", "sha", "finish", "cold",
                                  "sse-missing-done", "sse-duplicate-done", "sse-parity", "sse-dispatch",
                                  "identity", "no-proposals", "loop-31"])
def test_correctness_rejects_false_answers_parity_protocol_and_identity(monkeypatch, tmp_path, fault):
    monkeypatch.setattr(exercise.live, "Client", actual_client_module().Client)
    args = arguments(tmp_path, ["correctness"], names=())
    with correctness_service(fault) as (args.url, _state), pytest.raises(ValueError):
        exercise.run(args)
    report = json.loads((args.out / "suite.json").read_text())
    assert not report["passed"] and report["scenarios"][0]["status"] == "failed"
    assert list(args.out.glob("*/response.raw"))
    if fault in {"finish", "sha"}:
        assert report["correctness"]["requests"] == 1
    if fault == "loop-31":
        loop = report["correctness"]["cases"][1]
        assert loop["mtp"]["content"] == loop["serial"]["content"] == "31"
        assert loop["mtp"]["token_ids"] == loop["serial"]["token_ids"]
        assert {(row["case"], row["phase"]) for row in report["correctness"]["failures"]} == {
            ("loop-reasoning", "mtp-answer"), ("loop-reasoning", "serial-answer")}


@pytest.mark.parametrize("option,value", [("response_mib", 2), ("sse_line_kib", 65),
                                         ("timeout", 301), ("operation_timeout", 301)])
def test_correctness_budget_refusal_precedes_http_client_construction(monkeypatch, tmp_path, option, value):
    args = arguments(tmp_path, ["correctness"], names=())
    setattr(args, option, value)
    monkeypatch.setattr(exercise.live, "Client", lambda *_a, **_k: pytest.fail("invalid budget reached HTTP"))
    with pytest.raises(ValueError, match="correctness requires"):
        exercise.run(args)


def test_correctness_cli_is_opt_in_and_uses_no_request_files(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(exercise, "run", lambda args: seen.append(args) or 0)
    base = ["--url", "http://unused", "--out", str(tmp_path / "unused")]
    assert exercise.main([*base, "--scenarios", "correctness", "--response-mib", "1",
                          "--timeout", "300", "--operation-timeout", "300", "--cycles", "1"]) == 0
    assert seen[-1].request == [] and seen[-1].scenarios == ["correctness"]
    with pytest.raises(SystemExit) as missing:
        exercise.main(base)
    assert missing.value.code == 2
    assert exercise.main([*base, "--request", "short=example.json"]) == 0
    assert "correctness" not in seen[-1].scenarios


def test_actual_keepalives_cannot_keep_response_body_alive_past_cooperative_deadline(tmp_path):
    module = actual_client_module()
    with dribbling_service() as (base, _entered):
        control = module.ReadControl(time.monotonic() + .1)
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="deadline"):
            module.Client(base, tmp_path, timeout=.04).call("dribble", {**request(), "stream": True}, read_control=control)
        elapsed = time.monotonic() - started
    receipt = json.loads((tmp_path / "dribble/exchange.json").read_text())
    assert not receipt["complete"] and receipt["error"]["type"] == "TimeoutError"
    assert elapsed < .6  # scheduling allowance; no inference performance claim
    assert b"keepalive" in (tmp_path / "dribble/response.raw").read_bytes()
    assert "no whole-wall guarantee" in receipt["read_control"]["contract"]


def test_stalled_read_near_deadline_reports_and_obeys_the_real_additional_read_bound(tmp_path):
    module = actual_client_module()
    with dribbling_service(stall_after=.09) as (base, _entered):
        deadline = time.monotonic() + .12
        with pytest.raises(TimeoutError, match="deadline"):
            module.Client(base, tmp_path, timeout=.08).call("stalled", {**request(), "stream": True},
                                                          read_control=module.ReadControl(deadline))
        finished = time.monotonic()
    receipt = json.loads((tmp_path / "stalled/exchange.json").read_text())
    declared = receipt["read_control"]
    assert declared["socket_inactivity_timeout_s"] == pytest.approx(.08)
    assert declared["body_completion_bound_after_acquisition_monotonic_s"] == pytest.approx(deadline + .08)
    assert finished >= deadline  # this read really overlaps the nominal deadline
    assert finished <= deadline + .08 + .5  # fixture scheduling allowance, separately visible from the read bound


def test_group_failure_cancels_real_keepalive_readers_and_drains_before_releasing_owner(monkeypatch, tmp_path):
    module = actual_client_module()
    monkeypatch.setattr(exercise.live, "Client", module.Client)
    args = arguments(tmp_path, ["burst"], names=("short",))
    args.timeout = .08
    args.out.mkdir()
    rows = exercise.load_requests(args.request, budget_bytes=1 << 20)
    rows["short"].update(fits=True, prompt_tokens=4)
    primary = RuntimeError("control-plane failure while streams are live")
    with dribbling_service(expected=2) as (base, entered):
        args.url = base
        suite = exercise.Suite(args, rows)
        suite.effective = exercise.qualification.profile(statistics())

        def fail(_label):
            assert entered.wait(1)
            raise primary

        monkeypatch.setattr(suite, "runtime", fail)
        with pytest.raises(RuntimeError) as caught:
            suite.concurrent(["short", "short"])
        assert caught.value is primary and suite.active_read_control is None
    exchanges = [json.loads(path.read_text()) for path in args.out.glob("*/exchange.json")]
    assert len(exchanges) == 2 and all(row["complete"] is False for row in exchanges)
    assert all(row["error"]["type"] == "ReadCancelled" for row in exchanges)


def supplied_long_arguments(tmp_path, *, capacity=2, allowance=64):
    args = arguments(tmp_path, ["supplied-long"], names=())
    args.oracle = []
    for index in range(capacity):
        name = f"case-{index}"
        body = request(f"long-{index}", maximum=allowance)
        path, gold = tmp_path / f"{name}.json", tmp_path / f"{name}-gold.json"
        path.write_text(json.dumps(body))
        gold.write_text(json.dumps({"expected_text": f"SENTINEL-{index}"}))
        args.request.append(name + "=" + str(path))
        args.oracle.append(name + "=" + str(gold))
    return args


@pytest.mark.parametrize("capacity", [2, 3])
def test_supplied_long_derives_capacity_and_accepts_semantic_eos_without_claiming_exhaustion(monkeypatch, tmp_path, capacity):
    monkeypatch.setattr(exercise.live, "Client", actual_client_module().Client)
    args = supplied_long_arguments(tmp_path, capacity=capacity)
    originals = [json.loads(Path(item.split("=", 1)[1]).read_text()) for item in args.request]
    with correctness_service(long_capacity=capacity) as (args.url, state):
        assert exercise.run(args) == 2  # Semantics pass; strict simultaneous-owner proof remains unsupported.
    report = json.loads((args.out / "suite.json").read_text())
    result = report["supplied_long"]
    assert result["resident_capacity"] == capacity and result["prompt_tokens"] == 4032
    assert result["declared_reply_tokens"] == 64 and result["context_tokens"] == 4096
    assert result["context_exhaustion"].startswith("UNRUN")
    assert result["semantic_parity_passed"] and result["sse_nonstream_parity_passed"]
    assert result["mtp_work"] == "observed" and result["drafted_total"] > 0
    assert result["strict_simultaneous_history"].startswith("unsupported")
    witness = result["concurrent"]["observed"]["advisory_all_active_histories"]
    assert witness["reading_contract"] == "advisory_host_snapshot_no_gpu_synchronization"
    assert witness["active"] == capacity
    assert len({slot["slot"] for slot in witness["slots"]}) == capacity
    assert all(slot["request_active"] and slot["target_cached_length"] >= 4032 for slot in witness["slots"])
    generations = [body for path, body in state.calls if path == "/v1/chat/completions"]
    assert len(generations) == 3 * capacity
    for body in generations:
        original = next(row for row in originals if row["messages"] == body["messages"])
        assert all(body[key] == value for key, value in original.items())
    assert sum(path == "/v1/tokenize" for path, _body in state.calls) == capacity
    for pair in result["cases"].values():
        assert pair["serial"]["choices"][0]["finish_reason"] == "stop"
        assert pair["serial"]["usage"]["completion_tokens"] < 64
        assert pair["serial"]["tensorfold"]["cached"] == 0
        assert len(pair["prompt_token_ids_sha256"]) == 64
    assert all(row["sha256"] for row in result["oracles"].values())


@pytest.mark.parametrize("fault", ["underfilled", "union-only", "no-proposals"])
def test_supplied_long_does_not_infer_coverage_from_capacity_peaks_or_requested_mode(monkeypatch, tmp_path, fault):
    monkeypatch.setattr(exercise.live, "Client", actual_client_module().Client)
    args = supplied_long_arguments(tmp_path)
    with correctness_service(fault, long_capacity=2) as (args.url, _state):
        assert exercise.run(args) == 2
    report = json.loads((args.out / "suite.json").read_text())
    assert report["scenarios"][0]["status"] == "unsupported" and not report["passed"]
    result = report["supplied_long"]
    assert result["semantic_parity_passed"] and result["context_exhaustion"].startswith("UNRUN")
    if fault == "no-proposals":
        assert result["drafted_total"] == 0 and result["mtp_work"] == "unsupported-no-proposals"
    else:
        assert result["concurrent"]["observed"]["peak_active"] == 2
        assert result["concurrent"]["observed"]["advisory_all_active_histories"] is None
    assert list(args.out.glob("*concurrent-runtime/response.raw"))


def test_supplied_long_uses_actual_token_ids_for_distinct_histories(monkeypatch, tmp_path):
    monkeypatch.setattr(exercise.live, "Client", actual_client_module().Client)
    args = supplied_long_arguments(tmp_path)
    with correctness_service("token-collision", long_capacity=2) as (args.url, state), pytest.raises(ValueError, match="distinct actual tokenizer"):
        exercise.run(args)
    assert not any(path == "/v1/chat/completions" for path, _body in state.calls)
    report = json.loads((args.out / "suite.json").read_text())
    assert len({row["prompt_token_ids_sha256"] for row in report["inputs"].values()}) == 1


def test_supplied_long_reports_large_reply_budget_as_explicit_floor_not_generic_nearfull(monkeypatch, tmp_path):
    monkeypatch.setattr(exercise.live, "Client", actual_client_module().Client)
    args = supplied_long_arguments(tmp_path, allowance=4095)
    with correctness_service(long_capacity=2, allowance=4095) as (args.url, _state):
        assert exercise.run(args) == 2
    result = json.loads((args.out / "suite.json").read_text())["supplied_long"]
    assert result["prompt_tokens"] == 1 and result["declared_reply_tokens"] == 4095
    assert "no generic near-full" in result["scope"]
    assert result["context_exhaustion"].startswith("UNRUN")


def test_supplied_long_text_gold_uses_existing_response_whitespace_contract(monkeypatch, tmp_path):
    monkeypatch.setattr(exercise.live, "Client", actual_client_module().Client)
    args = supplied_long_arguments(tmp_path)
    with correctness_service("whitespace", long_capacity=2) as (args.url, _state):
        assert exercise.run(args) == 2
    record = json.loads((args.out / "suite.json").read_text())["supplied_long"]
    assert record["cases"]["case-0"]["mtp"]["choices"][0]["message"]["content"].startswith(" \n")
    assert record["oracles"]["case-0"]["expected"]["expected_text"] == "SENTINEL-0"


@pytest.mark.parametrize("payload", [{"expected_text": 3}, {"expected_json": {"sum": True}},
                                     {"expected_text": "answer", "code": "execute"}])
def test_supplied_oracles_are_bounded_typed_answer_data_only(tmp_path, payload):
    path = tmp_path / "gold.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="oracle requires"):
        exercise.load_oracles(["one=" + str(path)], budget_bytes=1024)
    with pytest.raises(ValueError, match="byte budget"):
        exercise.load_oracles(["one=" + str(path)], budget_bytes=1)


@pytest.mark.parametrize("fault", ["answer", "ids", "sha", "cold", "sse-missing-done", "sse-parity", "identity"])
def test_supplied_long_rejects_wrong_gold_ids_protocol_and_service_identity(monkeypatch, tmp_path, fault):
    monkeypatch.setattr(exercise.live, "Client", actual_client_module().Client)
    args = supplied_long_arguments(tmp_path)
    with correctness_service(fault, long_capacity=2) as (args.url, _state), pytest.raises(ValueError):
        exercise.run(args)
    report = json.loads((args.out / "suite.json").read_text())
    assert not report["passed"] and report["scenarios"][0]["status"] == "failed"
    assert list(args.out.glob("*/response.raw"))


def test_supplied_long_does_not_normalize_the_declared_gold(monkeypatch, tmp_path):
    monkeypatch.setattr(exercise.live, "Client", actual_client_module().Client)
    args = supplied_long_arguments(tmp_path)
    Path(args.oracle[0].split("=", 1)[1]).write_text(json.dumps({"expected_text": " SENTINEL-0"}))
    with correctness_service("whitespace", long_capacity=2) as (args.url, _state), pytest.raises(ValueError, match="known answer"):
        exercise.run(args)


def test_supplied_long_prompt_fit_alone_does_not_prove_selected_boundary(monkeypatch, tmp_path):
    monkeypatch.setattr(exercise.live, "Client", actual_client_module().Client)
    args = supplied_long_arguments(tmp_path, allowance=63)
    with correctness_service(long_capacity=2, allowance=64) as (args.url, state):
        assert exercise.run(args) == 2
    report = json.loads((args.out / "suite.json").read_text())
    assert all(row["fits"] for row in report["inputs"].values())
    assert report["scenarios"][0]["status"] == "unsupported"
    assert not any(path == "/v1/chat/completions" for path, _body in state.calls)

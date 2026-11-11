"""Owned client contracts; all HTTP responses are synthetic local fixtures."""
import contextlib
import hashlib
import json
import runpy
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import gemma_shared_prefix_qualify as q


def event(value, ending="\n"):
    return ("data: " + json.dumps(value) + ending + ending).encode()


def chat_wire(content="ok", cached=0):
    return (event({"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
            + event({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            + event({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 1,
                                             "prompt_tokens_details": {"cached_tokens": cached}}})
            + b"data: [DONE]\n\n")


@contextlib.contextmanager
def server(payload, status=200):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def do_GET(self):
            self.send_response(status)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
        def log_message(self, *args):
            pass
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(httpd.server_port)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(2)
        assert not thread.is_alive()


def test_import_and_spawn_import_have_no_network_or_argument_effects(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("import attempted network")
    monkeypatch.setattr(q.http.client, "HTTPConnection", forbidden)
    monkeypatch.syspath_prepend(str(Path(q.__file__).parent))
    module = runpy.run_path(q.__file__, run_name="__mp_main__")
    assert callable(module["main"])


def test_sse_usage_only_done_and_exact_tool_argument_bytes():
    raw = event({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_wire",
        "function": {"name": "f", "arguments": '{"x":'}}]}, "finish_reason": None}]})
    raw += event({"choices": [{"delta": {"tool_calls": [{"index": 0,
        "function": {"arguments": "1}"}}]}, "finish_reason": "tool_calls"}]})
    raw += event({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 5,
                                          "prompt_tokens_details": {"cached_tokens": 8}}})
    raw += b"data: [DONE]\n\n"
    result = q.normalize(raw, True)
    assert result["message"]["tool_calls"] == [{"id": "call_wire", "type": "function",
        "function": {"name": "f", "arguments": '{"x":1}'}}]
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 8


@pytest.mark.parametrize("raw", [b"data: {}", event({"choices": []}) + b"data: [DONE]\n\n",
                                 chat_wire().replace(b"data: [DONE]\n\n", b"")])
def test_incomplete_sse_or_usage_refuses(raw):
    with pytest.raises(ValueError):
        q.normalize(raw, True)


def test_crlf_sse_and_transport_phase_observations(tmp_path):
    payload = chat_wire().replace(b"\n", b"\r\n")
    assert q.normalize(payload, True)["message"]["content"] == "ok"
    with server(payload) as endpoint:
        client = q.Client(endpoint, q.Evidence(tmp_path / "evidence"), "baseline", timeout=2)
        raw, _, timing = client.call("phase", "/fixture")
    assert raw == payload
    assert 0 <= timing["first_byte_s"] <= timing["first_sse_data_s"]
    assert timing["first_sse_data_s"] <= timing["first_visible_s"] <= timing["last_visible_s"]
    assert timing["last_visible_s"] <= timing["terminal_s"] <= timing["full_wall_s"]
    assert timing["last_visible_to_terminal_s"] >= 0


def test_http_failure_preserves_actual_raw_error_body(tmp_path):
    payload = b'{"error":{"message":"fixture failure"}}'
    out = tmp_path / "evidence"
    with server(payload, 503) as endpoint:
        client = q.Client(endpoint, q.Evidence(out), "baseline", timeout=2)
        with pytest.raises(ValueError, match="503"):
            client.call("failure", "/fixture")
    assert (out / "001-failure-response.raw").read_bytes() == payload
    assert json.loads((out / "001-failure-transport.json").read_text())["failure"] == "ValueError"


def test_response_cap_records_bounded_partial_and_does_not_retry(tmp_path):
    out = tmp_path / "evidence"
    with server(b"x" * 100) as endpoint:
        client = q.Client(endpoint, q.Evidence(out), "baseline", timeout=2, cap=16)
        with pytest.raises(ValueError, match="cap"):
            client.call("cap", "/fixture")
    assert len((out / "001-cap-response.raw").read_bytes()) == 17
    assert len(list(out.glob("*request.json"))) == 1


def test_authentic_request_failure_survives_connection_close_failure(tmp_path, monkeypatch):
    primary, cleanup = RuntimeError("request"), RuntimeError("close")
    class Broken:
        def __init__(self, *args, **kwargs):
            pass
        def request(self, *args, **kwargs):
            raise primary
        def close(self):
            raise cleanup
    monkeypatch.setattr(q.http.client, "HTTPConnection", Broken)
    client = q.Client("http://127.0.0.1:1234", q.Evidence(tmp_path / "evidence"), "baseline")
    with pytest.raises(RuntimeError) as caught:
        client.call("broken", "/fixture")
    assert caught.value is primary and caught.value.__cause__ is cleanup
    assert (tmp_path / "evidence/001-broken-response.raw").exists()


def test_fresh_output_cannot_overwrite_previous_evidence(tmp_path):
    root = tmp_path / "evidence"
    q.Evidence(root).save("retained.json", {"retained": True})
    with pytest.raises(FileExistsError):
        q.Evidence(root)
    assert json.loads((root / "retained.json").read_text()) == {"retained": True}


def test_raw_public_tapes_and_actual_output_continuation_have_exact_geometry():
    class Raw:
        def __init__(self):
            self.out = SimpleNamespace(save=lambda name, value: None)
            self.bodies = []
        def tokens(self, name, body):
            if name == "raw-seed":
                return [11, 12] + [30] * 39996 + [13, 14]
            if name == "raw-empty-frame":
                return [11, 12, 13, 14]
            if name == "raw-ordinary":
                return [40]
            if name == "raw-suffix-closed":
                return [11, 12, 50, 13, 14]
            return [11, 12, 50, 13, 14, 20, 21]
        def json(self, *args):
            return {"prompt": " different"}
        def generate(self, name, path, body):
            self.bodies.append((name, body))
            cached = {"raw-A-first": 0, "raw-B": 4000, "raw-A-repeat": 24000,
                      "raw-A-output-continuation": 24001}[name]
            return {"message": {"content": '{"ok":true}'}, "finish": "stop",
                    "tensorfold": {"token_ids": [99]}, "prompt": len(body["prompt"]), "cached": cached}
        def feature(self, value, name):
            q.require(value, name)
    client = Raw()
    q.run_raw(client, {"model": "fixture", "raw": {"seed_scenario": "six", "ordinary_text": "different",
                      "suffix_messages": [{}, {}, {}]}},
              {"six": {"long": [{"role": "user", "content": "public record"}]}})
    a, b, repeat, grown = [body["prompt"] for _, body in client.bodies]
    assert len(a) == 24000 and len(b) == 12000
    assert a[:4000] == b[:4000] and a[4000] != b[4000]
    assert a[:2] == b[:2] == [11, 12] and a[-2:] == b[-2:] == [13, 14]
    assert a == repeat
    assert grown == a + [99] + [20, 21]


@pytest.mark.parametrize("candidate", [False, True])
def test_baseline_feature_miss_is_recorded_but_candidate_requires_it(tmp_path, candidate):
    client = q.Client("http://127.0.0.1:1", q.Evidence(tmp_path / "out"),
                      "candidate" if candidate else "baseline")
    if candidate:
        with pytest.raises(ValueError, match="new feature"):
            client.feature(False, "new feature")
    else:
        assert client.feature(False, "new feature") is False


def test_common_quality_failure_is_never_a_baseline_waiver():
    with pytest.raises(ValueError, match="gold"):
        q.json_ok({"message": {"content": '{"ok":false}'}})


def test_actual_output_ids_are_required_for_deterministic_raw_comparison():
    a = {"message": {"content": "same"}, "finish": "stop", "tensorfold": {"token_ids": [1]}}
    b = {**a, "tensorfold": {"token_ids": [2]}}
    with pytest.raises(ValueError, match="IDs"):
        q.same(a, b)


def test_strict_40k_refuses_39999_and_stops_before_generation():
    class Client:
        def tokens(self, *args):
            return [1] * 39999
        def generate(self, *args):
            raise AssertionError("generation after failed input alignment")
    cases = {name: {"long": [{"role": "user", "content": "frozen"}], "short": []} for name in ("six", "ten")}
    with pytest.raises(ValueError, match="40000|40K"):
        q.run_retention(Client(), {"model": "fixture"}, cases)


def test_responses_terminal_and_call_id_are_actual_wire_authority():
    final = {"status": "completed", "output": [{"type": "function_call", "id": "fc_unrelated",
             "call_id": "call_actual", "name": "f", "arguments": "{}"}],
             "usage": {"input_tokens": 4, "output_tokens": 1, "input_tokens_details": {"cached_tokens": 2}}}
    result = q.normalize(event({"type": "response.completed", "response": final}), True, True)
    assert result["message"]["tool_calls"][0]["id"] == "call_actual"
    assert result["response"]["output"][0]["id"] == "fc_unrelated"


def test_input_identity_and_oversize_refuse_before_parsing(tmp_path):
    path = tmp_path / "manifest"
    path.write_text("{}")
    with pytest.raises(ValueError, match="identity"):
        q.load_inputs(path, "0" * 64)
    large = tmp_path / "large"
    with large.open("wb") as stream:
        stream.truncate(q.CAP + 1)
    with pytest.raises(ValueError, match="bounded"):
        q.load_inputs(large, hashlib.sha256(b"").hexdigest())


def test_usage_cached_bounds_are_not_silently_coerced(tmp_path, monkeypatch):
    body = {"stream": True}
    client = q.Client("http://127.0.0.1:1", q.Evidence(tmp_path / "out"), "baseline")
    bad = chat_wire(cached=11)
    monkeypatch.setattr(client, "call", lambda *args: (bad, "bad", {}))
    with pytest.raises(ValueError, match="usage"):
        client.generate("bad", "/v1/chat/completions", body)


def test_candidate_responses_header_hit_is_not_complete_history_reuse():
    client, spec, requests = canonical_scenario(response_cached=2)
    with pytest.raises(ValueError, match="complete materialized"):
        q.run_canonical(client, spec, requests)



@pytest.mark.parametrize("arm", ["baseline", "candidate"])
def test_old_cache_schema_is_reference_behavior_only(tmp_path, monkeypatch, arm):
    client = q.Client("http://127.0.0.1:1", q.Evidence(tmp_path / "out"), arm)
    old = {"model": {"context_length": 262144}, "server": {"instance_id": "fixed"},
           "prefix_cache": {"entries": 4, "capacity": 4, "max_entries": 4, "tokens": 100,
                            "charged_bytes": 20, "byte_budget": 100}}
    monkeypatch.setattr(client, "json", lambda *args: old)
    if arm == "candidate":
        with pytest.raises(ValueError, match="schema"):
            client.stats("old")
    else:
        assert client.stats("old") == old


def test_first_cleanup_survives_compaction_exhaustion_with_original_statuses():
    first, secondary, exhausted = RuntimeError("first cleanup"), ValueError("second cleanup"), MemoryError("compact")
    prior = LookupError("prior native cause")
    first.__cause__ = prior
    class Statuses:
        passes = 0
        def __iter__(self):
            self.passes += 1
            if self.passes == 2:
                raise exhausted
            return iter((None, first, secondary))
    statuses = Statuses()
    with pytest.raises(RuntimeError) as caught:
        q.finish_errors(None, statuses)
    assert caught.value is first and first.__cause__ is prior
    assert first.__context__ is exhausted
    trace = first.__traceback__
    frames = []
    while trace is not None:
        frames.append(trace.tb_frame)
        trace = trace.tb_next
    assert any(frame.f_locals.get("errors") is statuses for frame in frames)


def test_first_cleanup_survives_error_transport_exhaustion(monkeypatch):
    first, secondary, exhausted = RuntimeError("first cleanup"), ValueError("second cleanup"), MemoryError("transport")
    prior, context = LookupError("native cause"), ArithmeticError("native context")
    first.__cause__, first.__context__ = prior, context
    statuses = [None, first, secondary]
    def fail(*args):
        raise exhausted
    monkeypatch.setattr(q, "raise_failures", fail)
    with pytest.raises(RuntimeError) as caught:
        q.finish_errors(None, statuses)
    assert caught.value is first and first.__cause__ is prior and first.__context__ is exhausted
    trace = first.__traceback__
    frames = []
    while trace is not None:
        frames.append(trace.tb_frame)
        trace = trace.tb_next
    assert any(frame.f_locals.get("errors") is statuses and frame.f_locals.get("native_context") is context
               for frame in frames)


def test_harness_token_ids_require_explicit_opt_in_without_mutating_input(tmp_path, monkeypatch):
    client = q.Client("http://127.0.0.1:1", q.Evidence(tmp_path / "out"), "baseline")
    original = {"model": "fixture", "messages": [{"role": "user", "content": "public"}]}
    observed = []
    def harness(name, route, body):
        observed.append((route, body))
        reply = {"count": 2, "harness_tokenizer_version": 1, "provenance": {"source": "fixture"}}
        if body.get("return_tokens") is True:
            reply["tokens"] = [11, 12]
        return reply
    monkeypatch.setattr(client, "json", harness)
    assert client.tokens("open", original) == [11, 12]
    assert observed[0][0] == "/v1/tokenize" and observed[0][1]["return_tokens"] is True
    assert "return_tokens" not in original


@pytest.mark.parametrize("body", [
    {"model": "fixture", "prompt": "ordinary", "add_special_tokens": False},
    {"model": "fixture", "messages": [{"role": "user", "content": "public"}], "add_generation_prompt": False},
])
def test_raw_and_closed_inspection_bind_declared_native_alias(tmp_path, monkeypatch, body):
    client = q.Client("http://127.0.0.1:1", q.Evidence(tmp_path / "out"), "baseline")
    seen = []
    def native(name, route, inspection):
        assert route == "/tokenize"  # Harness is chat-only and always generation-open.
        seen.append(inspection)
        return {"count": 2, "tokens": [21, 22], "max_model_len": 262144}
    monkeypatch.setattr(client, "json", native)
    assert client.tokens("native", body) == [21, 22]
    assert seen[0]["return_tokens"] is True
    for key, value in body.items():
        assert seen[0][key] == value


def test_missing_id_array_remains_a_failure_without_local_or_count_fallback(tmp_path, monkeypatch):
    client = q.Client("http://127.0.0.1:1", q.Evidence(tmp_path / "out"), "baseline")
    attempts = []
    def count_only(*args):
        attempts.append(args)
        return {"count": 40000, "harness_tokenizer_version": 1}
    monkeypatch.setattr(client, "json", count_only)
    with pytest.raises(ValueError, match="native token IDs missing"):
        client.tokens("unsupported", {"messages": [{"role": "user", "content": "public"}]})
    assert len(attempts) == 1


def test_boolean_tokenizer_count_cannot_masquerade_as_one_id(tmp_path, monkeypatch):
    client = q.Client("http://127.0.0.1:1", q.Evidence(tmp_path / "out"), "baseline")
    monkeypatch.setattr(client, "json", lambda *args: {"count": True, "tokens": [1]})
    with pytest.raises(ValueError, match="tokenizer count"):
        client.tokens("bad", {"messages": [{"role": "user", "content": "public"}]})


def canonical_scenario(*, response_cached=100, bad_prefix=False, auto_cached=190):
    tools = [{"type": "function", "function": {"name": "get_time"}}]
    class Scenario:
        def __init__(self):
            self.saved, self.generated = {}, []
            self.out = SimpleNamespace(save=lambda name, value: self.saved.__setitem__(name, value))
        def tokens(self, name, body):
            if name == "chat-next-authority":
                return [1] * 11
            if name == "chat-canonical":
                return [1] * 10
            if name.startswith("tool-canonical"):
                return [10] * 190
            if name.startswith("tool-none-authority"):
                assert body["tool_choice"] == "none"
                return [10] * 6 + [20] * 168
            if name.startswith("tool-compatible-authority"):
                assert body["tool_choice"] == "auto" and body["tools"] == tools
                return ([99] if bad_prefix else [10]) + [10] * 189 + [30] * 26
            raise AssertionError(name)
        def feature(self, passed, name):
            q.require(passed, name)
        def generate(self, name, path, body):
            self.generated.append((name, body))
            calls, content, stats, prompt, cached = [], "same", {}, 11, 10
            if name.startswith("tool-call"):
                calls = [{"id": "call_" + str(body["stream"]), "type": "function",
                          "function": {"name": "get_time", "arguments": '{"zone":"UTC"}'}}]
                content = ""
            if name.startswith("tool-none-follow"):
                content, prompt, cached = "12:34", 174, 174 if body["stream"] else 6
            if name.startswith("tool-compatible-follow"):
                content, prompt, cached = "12:34", 216, auto_cached
            if name.startswith("response-first"):
                stats = {"cache_publication": {"mode": "frontend", "complete": True, "materialized_tokens": 100}}
            if name.startswith("response-next"):
                cached = response_cached
            return {"message": {"content": content, "tool_calls": calls}, "finish": "stop",
                    "tensorfold": stats, "cached": cached, "prompt": prompt, "response": {"id": "resp_actual"}}
    user = {"role": "user", "content": "public"}
    requests = {"cache-A-first": {"messages": [user]}, "cache-A-extended": {"messages": [user]},
                "clock-call-stream": {"messages": [user], "tools": tools,
                    "tool_choice": {"type": "function", "function": {"name": "get_time"}}},
                "clock-followup-stream": {"messages": [user, {"role": "tool", "content": "12:34"}],
                    "tools": tools, "tool_choice": "none"}}
    return Scenario(), {"responses_body": {"input": "public"}, "responses_followup": "next"}, requests


def test_tool_none_miss_and_warmed_branch_are_distinguished_from_strict_compatible_reuse():
    client, spec, requests = canonical_scenario()
    summary = q.run_canonical(client, spec, requests)
    assert summary["chat_canonical_tokens"] == 10
    first, second = client.saved["tool-context-proof-False.json"], client.saved["tool-context-proof-True.json"]
    assert first["closed_tokens"] == second["closed_tokens"] == 190
    assert first["forced_closed_vs_none_lcp"] == second["forced_closed_vs_none_lcp"] == 6
    assert first["expected_none_cached"] == 6 and second["expected_none_cached"] == 174
    assert first["compatible_tokens"] == second["compatible_tokens"] == 216
    assert requests["clock-followup-stream"]["tool_choice"] == "none"
    none = [body for name, body in client.generated if name.startswith("tool-none-follow")]
    auto = [body for name, body in client.generated if name.startswith("tool-compatible-follow")]
    assert len(none) == len(auto) == 2
    assert all(body["tool_choice"] == "none" for body in none)
    assert all(body["tool_choice"] == "auto" and body["tools"] == requests["clock-call-stream"]["tools"] for body in auto)


def test_incompatible_auto_ids_stop_before_any_follow_generation():
    client, spec, requests = canonical_scenario(bad_prefix=True)
    with pytest.raises(ValueError, match="exact closed canonical history"):
        q.run_canonical(client, spec, requests)
    assert not any("follow" in name for name, _ in client.generated)


def test_compatible_cache_cut_stays_full190_not_a_lowered_threshold():
    client, spec, requests = canonical_scenario(auto_cached=189)
    with pytest.raises(ValueError, match="compatible tool-call history"):
        q.run_canonical(client, spec, requests)

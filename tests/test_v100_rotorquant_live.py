"""Offline live-service qualification controls; no model, GPU or HTTP service."""

import copy
import io
import json
import threading
import urllib.error

import pytest

from tools import v100_rotorquant_live as live


def response(tokens=3):
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 31, "completion_tokens": tokens, "total_tokens": 31 + tokens},
            "tensorfold": {"token_ids": list(range(tokens)), "cached": 30}}


def stats(parallel=4):
    window = dict(requested=parallel, reserved=parallel, stream_slots=parallel, context_window=262144,
                  cache_slots_each=262148, materialized=True, startup_capacities=[262148] * parallel,
                  allocated_cache_bytes=2000, workspace_bytes=2 << 30,
                  ranks=[dict(rank=rank, materialized=True, allocated_cache_bytes=2000,
                              torch_allocated_bytes=4000, torch_reserved_bytes=5000,
                              startup_capacities=[262148] * parallel) for rank in range(8)])
    return dict(object="tensorfold.stats", model=dict(id="test", aliases=[], kv_dtype=live.Q4_FORMATS[1], context_length=262144),
                server=dict(instance_id="fixed-instance", source_revision="fixed-source"), limits=dict(lanes=parallel),
                load=dict(requests_in_flight=0, requests_running=0, requests_waiting=0),
                totals=dict(cancelled=0, failed=0), capacity_plan=dict(context_window=262144, kv_full_windows=window))


@pytest.fixture
def case_data(tmp_path):
    path = tmp_path / "cases"
    path.mkdir()
    (path / "baseline-ready.json").write_text(json.dumps(dict(passed=True, cases=list(live.CASES))))
    for name in live.CASES:
        (path / (name + "-request.json")).write_text(json.dumps(dict(
            model="old-int8-name", messages=[dict(role="user", content=name)], max_tokens=32, temperature=0)))
        # Deliberately invalid old response data must never be read or compared.
        (path / (name + "-response.json")).write_text("old INT8 IDs are not an authority for Q4")
    return path


def test_loads_only_twelve_request_data_files_and_hashes(case_data):
    cases, hashes = live.load_cases(case_data)
    assert [name for name, _ in cases] == list(live.CASES) and len(hashes) == 13
    assert all(body["max_tokens"] == 32 for _, body in cases)


@pytest.mark.parametrize("mutation", ["missing", "index", "symlink", "budget", "tools"])
def test_invalid_case_corpus_refuses_before_any_service_request(case_data, mutation):
    path = case_data / "multiply-request.json"
    if mutation == "missing":
        path.unlink()
    elif mutation == "symlink":
        path.unlink()
        path.symlink_to(case_data / "modular-request.json")
    elif mutation == "index":
        (case_data / "baseline-ready.json").write_text(json.dumps(dict(passed=True, cases=["../escape"])))
    else:
        body = json.loads(path.read_text())
        body["max_tokens" if mutation == "budget" else "tools"] = True if mutation == "budget" else [{"function": {}}]
        path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        live.load_cases(case_data)


def test_all_nine_seeded_profiles_preserve_original_sampling_and_thinking():
    rows = live.profiles("new-model", 128)
    assert len(rows) == 9
    for index, (name, body) in enumerate(rows):
        assert body["model"] == "new-model" and body["max_tokens"] == 128 and body["return_token_ids"]
        assert body["seed"] == (17, 19, 47)[index % 3]
        assert body["temperature"] == (0, 0.7, 0.7)[index % 3]
        assert body["top_k"] == (0, 40, 40)[index % 3] and body["top_p"] == (1, 0.9, 0.9)[index % 3]
        assert body["chat_template_kwargs"]["enable_thinking"] == name.startswith("reason-")


def test_complete_messages_and_ids_are_both_required():
    reference = response()
    assert live.compare(reference, copy.deepcopy(reference), 3)["passed"]
    for field in ("ids", "message", "finish"):
        candidate = copy.deepcopy(reference)
        if field == "ids":
            candidate["tensorfold"]["token_ids"][-1] = 17
        elif field == "message":
            candidate["choices"][0]["message"]["content"] = "different"
        else:
            candidate["choices"][0]["finish_reason"] = "stop"
        assert not live.compare(reference, candidate, 3)["passed"]


@pytest.mark.parametrize("mutation", ["ids-missing", "ids-short", "ids-bool", "usage-bool", "usage-total", "message-content", "unfinished", "index-bool", "error"])
def test_partial_or_malformed_replies_never_compare_equal(mutation):
    value = response()
    if mutation == "ids-missing":
        value["tensorfold"].pop("token_ids")
    elif mutation == "ids-short":
        value["tensorfold"]["token_ids"].pop()
    elif mutation == "ids-bool":
        value["tensorfold"]["token_ids"][0] = False
    elif mutation == "usage-bool":
        value["usage"]["completion_tokens"] = True
    elif mutation == "usage-total":
        value["usage"]["total_tokens"] += 1
    elif mutation == "message-content":
        value["choices"][0]["message"].pop("content")
    elif mutation == "unfinished":
        value["choices"][0]["finish_reason"] = None
    elif mutation == "index-bool":
        value["choices"][0]["index"] = False
    else:
        value = {"error": {"message": "refused"}}
    with pytest.raises(ValueError):
        live.compare(value, value, 3)


def test_all_eight_public_physical_full_window_receipts_are_required():
    result = live.capacity_proof(stats(), parallel=4, kv_dtype=live.Q4_FORMATS[1], context=262144)
    assert result["capacity_each"] == 262148 and len(result["ranks"]) == 8


@pytest.mark.parametrize("mutation", ["int8", "context", "lanes", "reserved", "not-materialized", "rank-missing", "rank-order", "capacity", "bytes", "workspace", "torch"])
def test_allocation_promise_or_wrong_format_is_not_a_physical_receipt(mutation):
    value = stats()
    window = value["capacity_plan"]["kv_full_windows"]
    if mutation == "int8":
        value["model"]["kv_dtype"] = "int8"
    elif mutation == "context":
        value["model"]["context_length"] = 8192
    elif mutation == "lanes":
        value["limits"]["lanes"] = 3
    elif mutation == "reserved":
        window["reserved"] = 1
    elif mutation == "not-materialized":
        window["materialized"] = False
    elif mutation == "rank-missing":
        window["ranks"].pop()
    elif mutation == "rank-order":
        window["ranks"].reverse()
    elif mutation == "capacity":
        window["ranks"][3]["startup_capacities"][0] = 256
    elif mutation == "bytes":
        window["ranks"][5]["allocated_cache_bytes"] = 1999
    elif mutation == "workspace":
        window["workspace_bytes"] = (2 << 30) - 1
    else:
        window["ranks"][7]["torch_allocated_bytes"] = 1999
    with pytest.raises(ValueError):
        live.capacity_proof(value, parallel=4, kv_dtype=live.Q4_FORMATS[1], context=262144)


class Response(io.BytesIO):
    status = 200

    def __init__(self, raw, media="application/json"):
        super().__init__(raw)
        self.headers = {"Content-Type": media}

    def getheader(self, name, default=None):
        return self.headers.get(name, default)


def sse(*events, done=True):
    return b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + (b"data: [DONE]\n\n" if done else b"")


def stream_events():
    value = response()
    return [dict(choices=[dict(index=0, delta=dict(role="assistant"), finish_reason=None)]),
            dict(choices=[dict(index=0, delta=dict(content="answer"), finish_reason=None)]),
            dict(choices=[dict(index=0, delta={}, finish_reason="length")]),
            dict(choices=[], usage=value["usage"], tensorfold=value["tensorfold"])]


def test_canonical_sse_parser_requires_done_usage_and_exact_full_reply(monkeypatch, tmp_path):
    raw = sse(*stream_events())
    monkeypatch.setattr(live.urllib.request, "urlopen", lambda *a, **kw: Response(raw, "text/event-stream"))
    client = live.Client("http://unused", tmp_path)
    result = client.call("sse", dict(max_tokens=3, stream=True))
    assert live.compare(response(), result, 3)["passed"]
    assert result["sse_done"] and (tmp_path / "sse/response.raw").read_bytes() == raw
    assert json.loads((tmp_path / "sse/exchange.json").read_text())["done"]


@pytest.mark.parametrize("failure", ["done", "usage", "ids", "oversized"])
def test_sse_errors_retain_raw_and_fail_instead_of_partial_success(monkeypatch, tmp_path, failure):
    events = stream_events()
    if failure == "usage":
        events[-1].pop("usage")
    elif failure == "ids":
        events[-1]["tensorfold"]["token_ids"] = []
    raw = sse(*events, done=failure != "done")
    monkeypatch.setattr(live.urllib.request, "urlopen", lambda *a, **kw: Response(raw, "text/event-stream"))
    client = live.Client("http://unused", tmp_path, limits=live.protocol.Limits(response_bytes=32) if failure == "oversized" else None)
    with pytest.raises(ValueError):
        client.call("failed", dict(max_tokens=3, stream=True))
    metadata = json.loads((tmp_path / "failed/exchange.json").read_text())
    assert not metadata["complete"] and "error" in metadata and (tmp_path / "failed/response.raw").exists()


def test_http_refusal_is_saved_and_raised_without_retry(monkeypatch, tmp_path):
    raw = b'{"error":{"message":"budget refused"}}'
    error = urllib.error.HTTPError("http://unused", 503, "refused", {"Content-Type": "application/json"}, io.BytesIO(raw))
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        raise error

    monkeypatch.setattr(live.urllib.request, "urlopen", fail)
    with pytest.raises(urllib.error.HTTPError):
        live.Client("http://unused", tmp_path).call("refusal", dict(max_tokens=3))
    assert len(calls) == 1 and (tmp_path / "refusal/response.raw").read_bytes() == raw
    assert json.loads((tmp_path / "refusal/exchange.json").read_text())["status"] == 503


def test_cancellation_closes_connection_and_retains_explicit_partial_exchange(monkeypatch, tmp_path):
    source = Response(sse(*stream_events()), "text/event-stream")
    monkeypatch.setattr(live.urllib.request, "urlopen", lambda *a, **kw: source)
    result = live.Client("http://unused", tmp_path).call("cancel", dict(max_tokens=128, stream=True), cancel=True)
    assert result["intentionally_incomplete"] and source.closed
    metadata = json.loads((tmp_path / "cancel/exchange.json").read_text())
    assert metadata["cancelled_by_client"] and not metadata["complete"] and not metadata["done"]


class MockClient:
    last = None
    mutation = None

    def __init__(self, *args, **kwargs):
        self.calls, self.cancelled = [], False
        self.release = threading.Event()
        MockClient.last = self

    def call(self, label, body=None, **kwargs):
        self.calls.append((label, copy.deepcopy(body), kwargs))
        if label.startswith("concurrent-stats-"):
            self.release.set()
            value = stats()
            value["load"]["requests_in_flight"] = 4
            return value
        if body is None:
            if label.startswith("health-"):
                return dict(ok=True)
            value = stats()
            value["totals"]["cancelled"] = int(self.cancelled)
            if self.mutation == "capacity":
                value["capacity_plan"]["kv_full_windows"]["materialized"] = False
            return value
        if label.startswith("concurrent-") and label.endswith("-draft"):
            assert self.release.wait(2)
        if kwargs.get("cancel"):
            self.cancelled = True
            return dict(intentionally_incomplete=True)
        value = response(body["max_tokens"])
        if self.mutation == "cold":
            value["tensorfold"]["cached"] = 0
        if self.mutation == "mismatch" and label == "profiles-code-s19-draft":
            value["tensorfold"]["token_ids"][-1] = 999
        return value


def arguments(case_data, output):
    return ["--url", "http://unused", "--cases-dir", str(case_data), "--output-dir", str(output), "--tokens", "128"]


@pytest.mark.parametrize("mutation,exit_code,complete", [(None, 0, True), ("mismatch", 2, True), ("capacity", 1, False)])
def test_full_mocked_procedure_uses_fresh_same_format_references_and_preserves_failed_gates(
    monkeypatch, case_data, tmp_path, mutation, exit_code, complete
):
    monkeypatch.setattr(live, "Client", MockClient)
    monkeypatch.setattr(MockClient, "mutation", mutation)
    output = tmp_path / "result"
    assert live.main(arguments(case_data, output)) == exit_code
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete"] == complete and summary["passed"] == (exit_code == 0)
    if complete:
        assert summary["prefix_reuse"] == "qualified" and summary["prefix_reused"]
        assert len(summary["checks"]) == 12 + 9 + 4 + 1 + 2
        assert summary["cancellation_cleanup"] and summary["prefix_recovered"]
        pairs = [body["draft"] for label, body, _ in MockClient.last.calls if label.startswith("requests-")]
        assert pairs == [False, True] * 12
        assert all(body["max_tokens"] == 32 for label, body, _ in MockClient.last.calls if label.startswith("requests-"))
        assert all(body["model"] == "test" for _, body, _ in MockClient.last.calls if body)
    else:
        assert all(body is None for _, body, _ in MockClient.last.calls)


def test_existing_output_refused_before_any_network(monkeypatch, case_data, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "earlier.json").write_text("preserve")
    monkeypatch.setattr(live, "Client", lambda *a, **kw: pytest.fail("network client created"))
    with pytest.raises(ValueError, match="empty output"):
        live.main(arguments(case_data, output))


@pytest.mark.parametrize("cached", [None, False, -1, 1, 127])
def test_explicit_cold_mode_rejects_cached_or_missing_complete_reply_metadata(cached):
    value = response()
    value["tensorfold"]["cached"] = cached
    with pytest.raises(ValueError, match="cold prefix mode"):
        live.prefix_receipt(value, "cold")


def test_qualified_recovery_still_refuses_cold_reply():
    value = response()
    value["tensorfold"]["cached"] = 0
    with pytest.raises(ValueError, match="did not reuse"):
        live.prefix_receipt(value, "qualified", recovery=True)


def test_explicit_cold_full_procedure_preserves_identity_sse_cancellation_and_recovery(monkeypatch, case_data, tmp_path):
    monkeypatch.setattr(live, "Client", MockClient)
    monkeypatch.setattr(MockClient, "mutation", "cold")
    output = tmp_path / "cold"
    assert live.main([*arguments(case_data, output), "--prefix-reuse", "cold"]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["prefix_reuse"] == "cold" and not summary["prefix_reused"]
    assert summary["complete"] and summary["passed"] and summary["cancellation_cleanup"] and summary["prefix_recovered"]
    assert len(summary["checks"]) == 28 and all(row["cached"] == 0 and row["prefix_reuse"] == "cold" for row in summary["checks"])


@pytest.mark.parametrize("label", ["requests-multiply-draft", "profiles-code-s19-draft", "concurrent-2-draft", "complete-sse", "prefix-recovery-prime", "prefix-recovery-reuse"])
def test_cold_mode_rejects_cached_reply_in_every_completed_drafted_phase(monkeypatch, case_data, tmp_path, label):
    class CachedLeak(MockClient):
        mutation = "cold"

        def call(self, current, body=None, **kwargs):
            value = super().call(current, body, **kwargs)
            if current == label:
                value["tensorfold"]["cached"] = 127
            return value

    monkeypatch.setattr(live, "Client", CachedLeak)
    output = tmp_path / "cached-leak"
    assert live.main([*arguments(case_data, output), "--prefix-reuse", "cold"]) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert not summary["passed"] and not summary["complete"] and summary["prefix_reuse"] == "cold"
    assert "cold prefix mode" in summary["error"]["message"]


@pytest.mark.parametrize("body", [None, {"input": "Describe this recording."}])
def test_recorded_responses_object_accepts_null_error_for_post_and_stored_get(monkeypatch, tmp_path, body):
    value = {"id": "resp_fixture", "object": "response", "status": "incomplete", "error": None,
             "incomplete_details": {"reason": "max_output_tokens"}, "output": [],
             "usage": {"input_tokens": 31, "output_tokens": 3, "total_tokens": 34}}
    raw = json.dumps(value).encode()
    monkeypatch.setattr(live.urllib.request, "urlopen", lambda *a, **kw: Response(raw))
    got = live.Client("http://unused", tmp_path).call("stored-response", body,
                     path="/v1/responses/resp_fixture" if body is None else "/v1/responses", object_response=True)
    assert got == value
    assert (tmp_path / "stored-response/response.raw").read_bytes() == raw
    assert json.loads((tmp_path / "stored-response/exchange.json").read_text())["complete"] is True


@pytest.mark.parametrize("error", [{"message": "failed"}, {}, [], False, 0, ""])
def test_recorded_responses_object_still_rejects_nonnull_error(monkeypatch, tmp_path, error):
    raw = json.dumps({"object": "response", "error": error}).encode()
    monkeypatch.setattr(live.urllib.request, "urlopen", lambda *a, **kw: Response(raw))
    with pytest.raises(ValueError, match="SSE error or invalid object"):
        live.Client("http://unused", tmp_path).call("error-response", {"input": "x"},
                                                  path="/v1/responses", object_response=True)
    assert (tmp_path / "error-response/response.raw").read_bytes() == raw
    assert json.loads((tmp_path / "error-response/exchange.json").read_text())["complete"] is False

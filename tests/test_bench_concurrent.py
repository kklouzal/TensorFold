"""HTTP/SSE and exactness evidence boundaries; no server or GPU is needed."""
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def bench():
    path = Path(__file__).resolve().parents[1] / "tools/bench_concurrent.py"
    spec = importlib.util.spec_from_file_location("bench_concurrent_boundary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("left,right,want", [
    ({}, {}, None), ({"token_sha": None}, {"token_sha": None}, None),
    ({"token_sha": ""}, {"token_sha": ""}, None),
    ({"token_sha": 123}, {"token_sha": 123}, None),
    ({"token_sha": "abc"}, {"token_sha": "abc"}, True),
    ({"token_sha": "abc"}, {"token_sha": "def"}, False),
    ({"token_sha": "abc", "error": "failed"}, {"token_sha": "abc"}, None),
])
def test_missing_or_failed_hash_evidence_never_proves_equality(bench, left, right, want):
    assert bench.compare_hash(left, right) is want


def sse(*events, done=True):
    result = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    return result + (b"data: [DONE]\n\n" if done else b"")


@pytest.mark.parametrize("payload,reason", [
    (sse({"error": {"message": "GPU rejected invalid state"}}), "SSE error"),
    (sse({"error": None}), "SSE error"),
    (sse(["invalid event"]), "invalid object"),
    (sse({"usage": {"completion_tokens": 4}}, done=False), "before [DONE]"),
    (sse({"usage": {"completion_tokens": False}}), "bounded positive"),
    (sse({"usage": {"completion_tokens": 0}}), "bounded positive"),
    (sse({"usage": {"completion_tokens": 17}}), "bounded positive"),
    (sse({}), "bounded positive"),
    (b"data: " + b"x" * 65536 + b"\n", "bounded size"),
])
def test_sse_error_incomplete_or_invalid_usage_is_a_failed_stream(bench, monkeypatch, payload, reason):
    monkeypatch.setattr(bench.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(payload))
    row = bench.stream("http://unused", "fixture", bench.PROMPTS[0], 16, 0, 123)
    assert row["tokens"] == 0 and reason in row["error"]
    assert row["token_sha"] is None and row["decode_tps"] is None


def test_valid_sse_keeps_optional_missing_hash_explicit(bench, monkeypatch):
    payload = sse(*({"choices": [{"text": "x"}]} for _ in range(4)),
                  {"usage": {"completion_tokens": 4}})
    monkeypatch.setattr(bench.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(payload))
    row = bench.stream("http://unused", "fixture", bench.PROMPTS[0], 16, 0, 123)
    assert "error" not in row and row["tokens"] == 4 and len(row["pieces"]) == 4
    assert row["token_sha"] is None


def test_usage_and_hash_without_text_pieces_cannot_claim_measured_rates(bench, monkeypatch):
    payload = sse({"usage": {"completion_tokens": 4}, "tensorfold": {"token_sha": "abc"}})
    monkeypatch.setattr(bench.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(payload))
    row = bench.stream("http://unused", "fixture", bench.PROMPTS[0], 16, 0, 123)
    assert "error" not in row and row["tokens"] == 4 and row["token_sha"] == "abc"
    assert row["unmeasured"] is True and row["decode_tps"] is None


@pytest.mark.parametrize("kind,strict,want", [
    ("valid", True, 0), ("missing", False, 0), ("missing", True, 1),
    ("unequal", False, 1), ("unequal", True, 1),
    ("error", False, 1), ("unmeasured", False, 0), ("unmeasured", True, 1),
])
def test_cli_retains_result_and_exits_by_verified_cells(bench, monkeypatch, tmp_path, kind, strict, want):
    def row(item, seed, tokens, draft=True):
        digest = None if kind == "missing" else f"{item['name']}-{seed}-{draft if kind == 'unequal' else True}"
        result = {"prompt": item["name"], "seed": seed, "tokens": tokens,
                  "first": 1.0, "last": 2.0, "sent": 0.0, "pieces": [(1.0, 1), (2.0, 1)],
                  "ttft_s": 1.0, "decode_tps": tokens - 1, "token_sha": digest}
        if kind == "error":
            result.update(error="SSE error: GPU failed", decode_tps=None)
        elif kind == "unmeasured":
            result.update(unmeasured=True, decode_tps=None)
        return result

    monkeypatch.setattr(bench, "together", lambda base, model, specs, tokens, temp, *rest:
                        [row(item, seed, tokens) for item, seed in specs])
    monkeypatch.setattr(bench, "stream", lambda base, model, item, tokens, temp, seed, draft=True:
                        row(item, seed, tokens, draft))
    output = tmp_path / "retained.json"
    argv = ["bench_concurrent.py", "http://unused", "fixture", "--levels", "1,4", "--reps", "1",
            "--temperatures", "0", "--alone", "--serial", "--output", str(output)]
    if strict:
        argv.append("--strict")
    monkeypatch.setattr(sys, "argv", argv)
    assert bench.main() == want
    report = json.loads(output.read_text())
    assert report["passed"] is (want == 0) and report["strict"] is strict
    assert len(report["cells"]) == 4
    if kind == "missing":
        assert all(cell["alone"]["equal"] == 0 and cell["alone"]["unverified"] > 0
                   and cell["serial"]["equal"] == 0 and cell["serial"]["unverified"] > 0
                   for cell in report["cells"])
    if kind == "unequal":
        assert all(cell["serial"]["unequal"] > 0 for cell in report["cells"])


def test_strict_no_seed_is_refused_before_requests(bench, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench", "http://unused", "fixture", "--strict", "--no-seed"])
    monkeypatch.setattr(bench, "together", lambda *a, **kw: pytest.fail("invalid gate made a request"))
    with pytest.raises(SystemExit) as caught:
        bench.main()
    assert caught.value.code == 2

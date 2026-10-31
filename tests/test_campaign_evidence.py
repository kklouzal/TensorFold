"""Full needle evidence and context-bound next-turn control; CPU fixtures only."""
import json
import sys
import traceback
from types import SimpleNamespace

import pytest
from test_campaign_tools import PROFILE, _write_inputs, count_tokens, fake_server, gates


def needle_args(corpus, **overrides):
    return SimpleNamespace(corpus=str(corpus), seed=2024, needle_fractions="0.05,0.3,0.55,0.8,0.95",
                           needle_depths="512", needle_tokens=256, needle_thinking="off",
                           min_needle_recall=None, allow_long_prefill=False, **overrides)


def test_needle_retains_one_complete_request_reply_without_repeating_inference(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("haystack " * 4000)
    evidence, seen = tmp_path / "evidence", []
    with fake_server(seen) as url:
        result = gates.gate_needle(url, {**PROFILE, "context": 4096}, needle_args(corpus), evidence_out=evidence)
    assert len(seen) == len(result["rows"]) == 1
    index = json.loads((evidence / "index.json").read_text())
    assert index["cases"] == ["needle-0000.json"] and not index["held"]
    record = json.loads((evidence / index["cases"][0]).read_text())
    assert record["status"] == "complete" and record["request"] == seen[0]
    assert record["expected_codes"] == {"AMBER": "492925", "BIRCH": "190532", "CEDAR": "763640",
                                        "DELTA": "606446", "EMBER": "318506"}
    reply = record["reply"]
    assert len(reply["content"]) > 600 and result["rows"][0]["answer"] == reply["content"][:600]
    assert reply["finish"] == "length" and reply["usage"]["completion_tokens"] == 256
    assert reply["stats"]["token_ids"] == list(range(10, 266))
    assert 0 <= record["sizing"]["upper_chars"] - record["sizing"]["chars"] <= 64
    # The shared HTTP fixture intentionally uses different fake token IDs from its quarter-character count
    # route. Verify the retained upper endpoint against that route's contract, not its unrelated usage fixture.
    upper = gates.needle_prompt(corpus.read_text(), record["sizing"]["upper_chars"], index["fractions"],
                                record["expected_codes"])
    assert record["sizing"]["upper_tokens"] == count_tokens(upper)
    assert result["status"] == "report" and result["verdict"] is None  # full evidence is no new quality verdict


def test_needle_retains_held_work_without_requests(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("haystack " * 4000)
    args = needle_args(corpus)
    args.needle_depths = str(gates.LONG_PREFILL_TOKENS + 1)
    monkeypatch.setattr(gates, "stream_chat", lambda *_a, **_kw: pytest.fail("held needle must not infer"))
    evidence = tmp_path / "evidence"
    result = gates.gate_needle("http://unused", PROFILE, args, evidence_out=evidence)
    index = json.loads((evidence / "index.json").read_text())
    assert not index["cases"] and index["held"] == result["held"] and result["status"] == "incomplete"


@pytest.mark.parametrize("publication_failure", [False, True])
def test_needle_error_preserves_request_and_primary_failure(tmp_path, monkeypatch, publication_failure):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("haystack " * 100)
    monkeypatch.setattr(gates, "post_json", lambda *_a: {"count": 1})
    calls = []
    transport = OSError("nested transport failure")

    def request(_url, body, **_kwargs):
        calls.append(body)
        raise RuntimeError("inference failed") from transport

    write = gates.write_json

    def publish(path, value):
        if value.get("status") == "error" and publication_failure:
            raise OSError("evidence disk failed")
        write(path, value)

    monkeypatch.setattr(gates, "stream_chat", request)
    monkeypatch.setattr(gates, "write_json", publish)
    evidence = tmp_path / "evidence"
    expected_error = BaseExceptionGroup if publication_failure else RuntimeError
    with pytest.raises(expected_error) as caught:
        gates.gate_needle("http://fixture", PROFILE, needle_args(corpus), evidence_out=evidence)
    assert len(calls) == 1
    record = json.loads((evidence / "needle-0000.json").read_text())
    assert record["request"]["messages"] == calls[0]["messages"]
    assert record["status"] == ("requested" if publication_failure else "error")
    if publication_failure:
        primary, publication = caught.value.exceptions
        assert isinstance(primary, RuntimeError) and primary.__cause__ is transport
        assert isinstance(publication, OSError) and str(publication) == "evidence disk failed"
        trace = "".join(traceback.format_exception(caught.value))
        assert all(message in trace for message in ("inference failed", "nested transport failure", "evidence disk failed"))
    else:
        assert caught.value.__cause__ is transport


def test_existing_needle_evidence_is_never_overwritten(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("haystack")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    sentinel = evidence / "old.json"
    sentinel.write_text("historical failure")
    monkeypatch.setattr(gates, "stream_chat", lambda *_a, **_kw: pytest.fail("no request before evidence ownership"))
    with pytest.raises(FileExistsError):
        gates.gate_needle("http://unused", PROFILE, needle_args(corpus), evidence_out=evidence)
    assert sentinel.read_text() == "historical failure"


def test_needle_cli_passes_the_optional_evidence_directory(tmp_path, monkeypatch):
    _write_inputs(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("haystack " * 1000)
    evidence, out = tmp_path / "evidence", tmp_path / "result.json"
    with fake_server() as url:
        monkeypatch.setattr(sys, "argv", ["campaign_gates.py", "needle", "--url", url,
                                          "--profile", str(tmp_path / "profile.json"), "--out", str(out),
                                          "--corpus", str(corpus), "--needle-depths", "512", "--needle-tokens", "16",
                                          "--needle-evidence-out", str(evidence)])
        assert gates.main() == 0
    assert json.loads(out.read_text())["status"] == "report"
    assert len(json.loads((evidence / "index.json").read_text())["cases"]) == 1


def test_next_turn_context_hold_prevents_invalid_request_without_weakening_extension(tmp_path):
    _write_inputs(tmp_path)
    args = SimpleNamespace(prompts=str(tmp_path / "prompts.json"), gate_prompts="short-code-nothink", thinking="off",
                           max_tokens=6, seed=2024, evict_requests=1, allow_long_prefill=True,
                           allow_nondeterminism=False)
    seen = []
    with fake_server(seen) as url:
        result = gates.gate_determinism(url, {**PROFILE, "context": 32}, args)
    assert result["status"] == "incomplete" and result["verdict"] is None
    assert result["exercised"] == {"warm": 2, "recold": 2, "extended": 2}
    assert len(result["held"]) == 2 and all(row["reported_only"] for row in result["held"])
    assert all(row["next_turn_prompt_tokens"] == 31 and row["reply_budget"] == 6 and row["context"] == 32
               and row["status"] == "incomplete (context)" for row in result["held"])
    assert not any("Continue." in json.dumps(body.get("messages")) for body in seen)
    extensions = [body for body in seen if "prompt" in body]
    assert len(extensions) == 4 and all(len(body["prompt"]) + body["max_tokens"] <= 32 for body in extensions)

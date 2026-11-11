"""Observed-active cancellation boundaries retain the historical work/oracles."""
import pytest

from tools import gemma_shared_prefix_cancel_qualify as q


def health(*, tokens=11, running=1, total=3):
    return {"ok": True, "busy": bool(running), "requests_running": running,
            "requests_total": total, "completion_tokens_total": tokens}


def test_work_limits_preserved_and_decode_requires_early_visible_active_work():
    assert q.BUDGET == 4096 and q.MAX_OBSERVED_BEFORE_CLOSE == 256
    assert q.MAX_EVENTS == 1024 and q.STREAM_BYTES == 4 << 20
    before = health(tokens=10, running=0)
    result = q.validate_active("decode", before, health(), {"choices": []}, ["1"])
    assert result["visible_text"] and result["server_generated_tokens_since_before"] == 1
    with pytest.raises(q.Inconclusive):
        q.validate_active("decode", before, health(tokens=267), {}, ["late"])
    with pytest.raises(q.Inconclusive):
        q.validate_active("decode", before, health(running=0), {}, ["ended"])


def test_prefill_requires_actual65536_interior_progress_before_output():
    before = health(tokens=10, running=0)
    event = {"prompt_progress": {"total": 65536, "cache": 0, "processed": 4096}}
    result = q.validate_active("prefill", before, health(tokens=10), event, [])
    assert result["native_prompt_progress"]["total"] == 65536
    with pytest.raises(q.Inconclusive):
        q.validate_active("prefill", before, health(tokens=11), event, [])
    with pytest.raises(ValueError, match="65536"):
        q.validate_active("prefill", before, health(tokens=10),
                          {"prompt_progress": {"total": 32768, "cache": 0, "processed": 4096}}, [])


def test_completed_before_disconnect_and_invalid_progress_are_not_passes():
    with pytest.raises(q.Inconclusive):
        q.trigger_event("decode", {"usage": {}}, ["already done"])
    with pytest.raises(ValueError, match="ordering"):
        q.trigger_event("prefill", {"prompt_progress": {"total": 65536, "cache": 10, "processed": 2}}, [])


def test_cancellation_ack_requires_increment_and_all_load_idle():
    before = {"totals": {"cancelled": 2}}
    after = {"totals": {"cancelled": 3},
             "load": {"requests_in_flight": 0, "requests_running": 0, "requests_waiting": 0}}
    assert q.retirement_state(before, after)
    after["load"]["requests_waiting"] = 1
    assert not q.retirement_state(before, after)
    after["totals"]["cancelled"] = 4
    with pytest.raises(ValueError, match="extra cancellation"):
        q.retirement_state(before, after)


def test_recovery_gold_and_normal_eos_are_preserved():
    q.validate_recovery({"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}]})
    for content, finish in (("four", "stop"), ("4", "length")):
        with pytest.raises(ValueError, match="recovery"):
            q.validate_recovery({"choices": [{"message": {"content": content}, "finish_reason": finish}]})


def test_candidate_logical_cache_obligations_must_settle_without_physical_claim():
    cache = {"charged_bytes": 20, "byte_budget": 100, "store_schema": "gemma-canonical-block-archive-v1",
             "entries": None, "max_entries": None, "capacity": None, "tokens": None, "metadata_status": "ready",
             **dict.fromkeys(("pins", "pending_batches", "reserved_payload_bytes", "uncertain_payload_bytes",
                              "temporary_bytes", "metadata_owners", "host_metadata_claim_bytes"), 0)}
    assert q.validate_cache_settled({"prefix_cache": cache}, "candidate") is cache
    cache["uncertain_payload_bytes"] = 1
    with pytest.raises(ValueError, match="settled"):
        q.validate_cache_settled({"prefix_cache": cache}, "candidate")
    q.validate_cache_settled({"prefix_cache": {"charged_bytes": 20, "byte_budget": 100}}, "baseline")


def test_old_production_endpoint_is_rejected_before_any_output_or_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network reached")
    monkeypatch.setattr(q.urllib.request, "urlopen", forbidden)
    with pytest.raises(ValueError, match="private"):
        q.main(["--url", "http://127.0.0.1:8888", "--output", str(tmp_path / "out"),
                "--expected-source-uid", "a" * 64, "--phase", "prefill", "--arm", "candidate"])
    assert not (tmp_path / "out").exists()


def test_declared_deadline_remains_finite_and_strict():
    q.check_deadline(10.0, 9.9)
    with pytest.raises(TimeoutError):
        q.check_deadline(10.0, 10.0)


def test_stream_primary_and_native_cause_survive_close_and_publication_without_installed_package(tmp_path, monkeypatch):
    import builtins
    from types import SimpleNamespace
    primary, close_error, publication = RuntimeError("stream read"), RuntimeError("response close"), RuntimeError("publish")
    prior = LookupError("native prior cause")
    primary.__cause__ = prior
    close_calls = []
    class Response:
        headers = {}
        def close(self):
            close_calls.append("close")
            raise close_error
    class BrokenEvents:
        def __iter__(self):
            raise primary
    protocol = SimpleNamespace(response_type=lambda *args: None, Limits=lambda *args: None,
                               sse_objects=lambda *args, **kwargs: BrokenEvents())
    monkeypatch.setattr(q.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    imported = builtins.__import__
    def no_package(name, *args, **kwargs):
        if name.startswith("tensorfold"):
            raise AssertionError("undeclared installed package import")
        return imported(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_package)
    def publish(*args):
        raise publication
    evidence = SimpleNamespace(raw=lambda *args: None)
    args = SimpleNamespace(url="http://127.0.0.1:18888", output=tmp_path, observe_seconds=1,
                           socket_timeout=1, phase="decode")
    with pytest.raises(RuntimeError) as caught:
        q.observe_and_disconnect(args, {}, "/fixture", {}, protocol,
            lambda *args: Response(), q.ReadControl, publish, evidence)
    assert caught.value is primary and close_calls == ["close"]
    causes = primary.__cause__.exceptions
    assert any(error is prior for error in causes)
    assert any(error is close_error for error in causes)
    assert any(error is publication for error in causes)


def test_main_body_primary_survives_summary_publication_without_package_dependency(tmp_path, monkeypatch):
    from types import SimpleNamespace
    primary, publication = RuntimeError("stats acquisition"), RuntimeError("summary publication")
    def call(*args, **kwargs):
        raise primary
    monkeypatch.setattr(q, "Client", lambda *args, **kwargs: SimpleNamespace(call=call))
    def publish(*args, **kwargs):
        raise publication
    monkeypatch.setattr(q, "write_json", publish)
    with pytest.raises(RuntimeError) as caught:
        q.main(["--url", "http://127.0.0.1:18888", "--output", str(tmp_path / "out"),
                "--expected-source-uid", "a" * 64, "--phase", "decode", "--arm", "candidate"])
    assert caught.value is primary and primary.__cause__ is publication

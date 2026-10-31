"""Measurement boundaries: cache truth, work counts, failures and independent paired units."""
import json

import pytest

from tools import gemma_e2b_bench as bench


def response(*, cached=0, prompt=512, count=4):
    return {
        "usage": {"prompt_tokens": prompt, "completion_tokens": count},
        "stats": {"token_ids": list(range(count)), "cached": cached,
                  "prefill_s": 0.5, "decode_s": 0.25, "decode_tps": 12.0},
        "sent": 10.0, "headers": 10.1, "first": 10.5, "last": 10.75, "done": 10.8,
        "content": "actual response", "reasoning": "", "finish": "length",
        "tool_call_deltas": [], "pieces": count,
    }


def test_uncached_tokens_and_complete_body_are_not_decode_wall_estimates():
    out = bench.measured_record(response(), 512, 4, "fresh")
    assert out["metrics"]["prefill_tps"] == 1024.0
    assert out["metrics"]["decode_tps"] == 12.0
    assert out["metrics"]["complete_body_tps"] == pytest.approx(5.0)
    assert out["metrics"]["ttft_client_s"] == 0.5


@pytest.mark.parametrize("phase,cached", [("fresh", 1), ("warm", 0)])
def test_requested_cache_phase_must_be_exercised(phase, cached):
    with pytest.raises(ValueError, match="prefix cache state"):
        bench.measured_record(response(cached=cached), 512, 4, phase)


def test_timer_zero_on_full_cache_hit_is_not_invented_prefill_work():
    raw = response(cached=512)
    raw["stats"]["prefill_s"] = 0.0
    out = bench.measured_record(raw, 512, 4, "warm")
    assert out["metrics"]["prefill_tps"] is None


def test_short_response_or_changed_prompt_cannot_count_as_faster():
    with pytest.raises(ValueError, match="token counts"):
        bench.measured_record(response(count=3), 512, 4, "fresh")
    with pytest.raises(ValueError, match="token counts"):
        bench.measured_record(response(prompt=511), 512, 4, "fresh")


def test_invalid_raw_result_and_error_are_retained(tmp_path, monkeypatch):
    manifest = {"model": "test", "model_revision": "fixed", "cycles": [["a", "b", "b", "a"]],
                "reply_tokens": [4], "samples_per_start": 1, "native_sampling": {"temperature": 1},
                "prompts": [{"id": "p", "prompt_tokens": 512, "depth": 512, "task": "code",
                             "thinking": True, "messages": [{"role": "user", "content": "fixed"}]}]}
    metadata = {"image_id": "sha256:image", "source_revision": "source",
                "model_revision": "fixed", "launch": ["serve"], "instance_id": "one-start"}
    source, meta, out = tmp_path / "manifest.json", tmp_path / "meta.json", tmp_path / "run.json"
    source.write_text(json.dumps(manifest))
    meta.write_text(json.dumps(metadata))
    monkeypatch.setattr(bench, "json_endpoint", lambda *args, **kwargs: {})
    monkeypatch.setattr(bench.campaign, "host_snapshot", lambda **kwargs: {})
    replies = iter([response(), response(), response(), response(cached=512)])
    calls = []
    def stream(*args, **kwargs):
        calls.append((args, kwargs))
        return next(replies)
    monkeypatch.setattr(bench.campaign, "stream_chat", stream)
    with pytest.raises(ValueError, match="prefix cache state"):
        bench.run("http://localhost", source, meta, out, cycle=0, position=0, timeout=10)
    saved = json.loads(out.read_text())
    assert saved["status"] == "error"
    assert saved["records"][-1]["status"] == "error"
    assert saved["records"][-1]["raw_stream_result"]["stats"]["cached"] == 512
    eviction_args, eviction_kwargs = calls[2]
    assert eviction_kwargs["route"] == "/v1/completions"
    assert eviction_args[1]["prompt"] == [1000]
    assert "messages" not in eviction_args[1]


def test_analysis_uses_cycles_not_correlated_samples_as_ci_units(tmp_path):
    manifest = {"baseline": "a", "candidate": "b", "cycles": [["a", "b", "b", "a"]] * 2,
                "prompts": [{"id": "p"}], "samples_per_start": 1, "reply_tokens": [8, 32]}
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(manifest))
    runs = []
    for cycle, order in enumerate(manifest["cycles"]):
        for position, label in enumerate(order):
            records = []
            for budget in manifest["reply_tokens"]:
                for sampling in ("greedy", "native"):
                    for phase in ("fresh", "warm"):
                        records.append({"prompt": "p", "depth": 512, "task": "code", "thinking": True,
                                        "sampling": sampling, "reply_tokens": budget, "phase": phase, "sample": 0, "status": "complete",
                                        "metrics": {"decode_tps": 100.0 if label == "a" else 110.0,
                                                    "complete_body_s": 1.0 if label == "a" else (0.9 if budget == 8 else 1.1)}})
            row = {"manifest_sha256": bench.digest(source), "tool_sha256": "same",
                   "cycle": cycle, "position": position, "variant": label,
                   "records": records, "status": "complete",
                   "meta": {"instance_id": f"{cycle}-{position}", "image_id": "fixed-image",
                            "source_revision": "fixed-source", "model_revision": "fixed-model",
                            "launch": ["serve", label]}}
            path = tmp_path / f"{cycle}-{position}.json"
            path.write_text(json.dumps(row))
            runs.append(path)
    out = bench.analyze(source, runs)
    assert out["complete"]
    assert out["objective_order"] == ["complete_body_s", "ttft_client_s", "decode_tps"]
    assert all("/32/" in item["region"] for item in out["regional_regressions"])
    assert len(out["regional_regressions"]) == 4
    result = next(iter(out["cells"].values()))["decode_tps"]
    assert result["independent_cycles"] == 2
    assert result["candidate_over_baseline"] == pytest.approx(1.1)
    assert result["preferred_direction"] == "higher"
    with pytest.raises(ValueError, match="duplicate independent"):
        bench.analyze(source, runs + [runs[0]])
    changed = json.loads(runs[-1].read_text())
    changed["meta"]["launch"].append("--different-runtime")
    runs[-1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="material configuration changed"):
        bench.analyze(source, runs)


def test_missing_start_is_incomplete_not_equivalence(tmp_path):
    manifest = {"cycles": [["a", "b", "b", "a"]], "prompts": [], "samples_per_start": 1, "reply_tokens": [8]}
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(manifest))
    out = bench.analyze(source, [])
    assert not out["complete"]
    assert out["missing_start_positions"] == [(0, 0), (0, 1), (0, 2), (0, 3)]


def test_changed_workload_cannot_be_pooled(tmp_path):
    source, run = tmp_path / "manifest.json", tmp_path / "run.json"
    source.write_text(json.dumps({"cycles": []}))
    run.write_text(json.dumps({"manifest_sha256": "other"}))
    with pytest.raises(ValueError, match="different frozen workload"):
        bench.analyze(source, [run])


def test_request_body_keeps_each_fixed_output_budget():
    manifest = {"model": "test", "native_sampling": {"temperature": 1, "top_k": 64, "top_p": 0.95}}
    item = {"messages": [{"role": "user", "content": "fixed"}], "thinking": True}
    for budget in (8, 32, 256):
        body = bench.body_for(manifest, item, "native", 0, budget)
        assert body["max_tokens"] == budget
        assert body["ignore_eos"] is True
        assert body["top_k"] == 64


def test_missing_card_uses_actual_conditional_hf_defaults_with_greedy_provenance(tmp_path):
    from transformers import Gemma4Config
    Gemma4Config().save_pretrained(tmp_path)
    card, source, label = bench.sampling_policy(tmp_path, 'hf-defaults')
    assert card == {'temperature': 1.0, 'top_k': 50, 'top_p': 1.0}
    assert source['do_sample'] is False
    assert source['native_omitted_api_policy'].startswith('greedy')
    assert label == 'sampled_hf_defaults'
    item = {'messages': [{'role': 'user', 'content': 'fixed'}], 'thinking': True}
    body = bench.body_for({'model': 'test', 'native_sampling': card}, item, label, 0, 8)
    assert body['temperature'] == 1.0
    assert body['top_k'] == 50


def test_existing_generation_card_cannot_be_replaced_with_defaults(tmp_path):
    (tmp_path / 'generation_config.json').write_text('{"temperature":0.7,"top_k":64,"top_p":0.9}')
    with pytest.raises(ValueError, match='retain checkpoint authority'):
        bench.sampling_policy(tmp_path, 'hf-defaults')
    card, source, label = bench.sampling_policy(tmp_path, 'checkpoint')
    assert card['temperature'] == 0.7
    assert source['sha256'] == bench.digest(tmp_path / 'generation_config.json')
    assert label == 'native'

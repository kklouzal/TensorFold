"""Benchmark registration/identity/analysis controls; no live service calls."""
import copy
import json
from types import SimpleNamespace

import pytest

from tools import v100_indexer_archive_bench as bench


def body(index=0):
    return {"model": "model", "messages": [{"role": "user", "content": str(index)}],
            "max_tokens": 512, "temperature": 0, "top_p": 1, "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False}}


def response(cached=0):
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"},
                         "finish_reason": "length"}],
            "usage": {"prompt_tokens": 128, "completion_tokens": 512, "total_tokens": 640},
            "tensorfold": {"token_ids": list(range(512)), "drafts": True, "cached": cached,
                           "decode_tps": 50., "decode_s": 10., "prefill_s": .5, "ttft_s": .6}}


def reports(tmp_path):
    paths = []
    for phase, rate in (("ram4-a1", 50.), ("gpu3-b", 25.), ("ram4-a2", 50.)):
        rows = []
        for name in ("one", "two", "three"):
            for cache in ("cold", "warm"):
                for block in range(2):
                    row = bench.measurement(response(0 if cache == "cold" else 127), cache, 11.)
                    rows.append({"body": name, "cache": cache, "block": block, **row, "decode_tps": rate})
        data = {"phase": phase, "complete": True, "passed": True, "image": "sha256:" + "a" * 64,
                "reps": 2, "tool_sha256": "b" * 64, "bodies": dict.fromkeys(("one", "two", "three"), "hash"),
                "rows": rows}
        path = tmp_path / (phase + ".json")
        path.write_text(json.dumps(data))
        paths.append(path)
    return paths


def test_loads_exact_three_frozen_body_data_without_editing(tmp_path):
    for index in range(3):
        (tmp_path / f"{index}-p128-t{index}-o512-request.json").write_text(json.dumps(body(index)))
    rows, hashes = bench.load_bodies(tmp_path)
    assert len(rows) == len(hashes) == 3
    assert [entry[1] for entry in rows] == [body(index) for index in range(3)]
    bad = tmp_path / "extra-p128-t3-o512-request.json"
    bad.write_text(json.dumps(body(4)))
    with pytest.raises(ValueError, match="exactly three"):
        bench.load_bodies(tmp_path)


@pytest.mark.parametrize("changes", [{"max_tokens": 128}, {"temperature": .7}, {"stream": True},
                                     {"return_token_ids": False}, {"draft": False}])
def test_changed_frozen_sampling_or_budget_is_refused(tmp_path, changes):
    named = []
    for index in range(3):
        path = tmp_path / str(index)
        value = body(index)
        if index == 0:
            value.update(changes)
        path.write_text(json.dumps(value))
        named.append(f"case{index}={path}")
    with pytest.raises(ValueError, match="frozen greedy"):
        bench.load_bodies(named=named)


def test_actual_prefix_condition_and_full_usage_are_hard_gates():
    assert bench.measurement(response(0), "cold", 11.)["usage"]["completion_tokens"] == 512
    assert bench.measurement(response(127), "warm", 11.)["runtime"]["cached"] == 127
    for label, changed in (("cold", response(127)), ("warm", response(0))):
        with pytest.raises(ValueError, match="prefix condition"):
            bench.measurement(changed, label, 11.)
    short = response()
    short["usage"]["completion_tokens"] = 511
    with pytest.raises(ValueError):
        bench.measurement(short, "cold", 11.)


def test_aba_analysis_retains_cells_ranges_and_no_equivalence_claim(tmp_path):
    result = bench.analyze(reports(tmp_path))
    assert result["complete"] and len(result["cells"]) == 6
    assert "not isolated-storage" in result["scope"]
    for cell in result["cells"]:
        assert cell["descriptive_bracket_ram4_over_gpu3_decode_ratio"] == pytest.approx(2.)
        assert len(cell["distributions"]["ram4-a1"]["decode_tps"]["all"]) == 2


def test_late_id_change_and_missing_cell_cannot_pass(tmp_path):
    paths = reports(tmp_path)
    value = json.loads(paths[1].read_text())
    value["rows"][-1]["completion"]["ids"][-1] = 999
    paths[1].write_text(json.dumps(value))
    with pytest.raises(ValueError, match="IDs/messages"):
        bench.analyze(paths)
    paths = reports(tmp_path)
    for path in paths:
        value = json.loads(path.read_text())
        value["rows"] = []
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="three bodies"):
        bench.analyze(paths)


def test_phase_order_and_source_identity_must_match(tmp_path):
    paths = reports(tmp_path)
    with pytest.raises(ValueError, match="fixed RAM4"):
        bench.analyze(list(reversed(paths)))
    value = json.loads(paths[1].read_text())
    value["image"] = "sha256:" + "c" * 64
    paths[1].write_text(json.dumps(value))
    with pytest.raises(ValueError, match="provenance"):
        bench.analyze(paths)


def test_cpu_only_data_and_analysis_do_not_open_http(monkeypatch, tmp_path):
    monkeypatch.setattr(bench.live, "Client", lambda *args, **kwargs: pytest.fail("unexpected HTTP construction"))
    paths = reports(tmp_path)
    unchanged = copy.deepcopy(json.loads(paths[0].read_text()))
    bench.analyze(paths)
    assert json.loads(paths[0].read_text()) == unchanged


def gpu_stats():
    ranks = [{"rank": rank, "materialized": True, "startup_capacities": [262148] * 3,
              "allocated_cache_bytes": 10, "torch_allocated_bytes": 20, "torch_reserved_bytes": 30}
             for rank in range(8)]
    return {"server": {"instance_id": "fixed", "backend": "cuda"},
            "model": {"id": "model", "kv_dtype": "int8", "mtp_drafts": 3, "context_length": 262144},
            "limits": {"lanes": 3}, "load": dict.fromkeys(
                ("requests_in_flight", "requests_running", "requests_waiting"), 0),
            "capacity_plan": {"context_window": 262144, "kv_full_windows": {
                "requested": 3, "reserved": 3, "stream_slots": 3, "materialized": True,
                "cache_slots_each": 262148, "startup_capacities": [262148] * 3,
                "workspace_bytes": 2 << 30, "allocated_cache_bytes": 10, "ranks": ranks}}}


def test_boolean_false_is_not_a_cold_prefix_count():
    with pytest.raises(ValueError, match="prefix condition"):
        bench.measurement(response(False), "cold", 11.)


@pytest.mark.parametrize("bad", [None, 0, False])
def test_missing_or_nonpositive_gpu_allocation_is_rejected(bad):
    value = gpu_stats()
    value["capacity_plan"]["kv_full_windows"]["allocated_cache_bytes"] = bad
    for row in value["capacity_plan"]["kv_full_windows"]["ranks"]:
        row["allocated_cache_bytes"] = bad
    with pytest.raises(ValueError, match="positive GPU"):
        bench.capacity(value, "gpu", 3)


def test_rank_actual_gpu_sizes_must_cover_cache():
    value = gpu_stats()
    value["capacity_plan"]["kv_full_windows"]["ranks"][4]["torch_allocated_bytes"] = 0
    with pytest.raises(ValueError, match="actual GPU"):
        bench.capacity(value, "gpu", 3)


@pytest.mark.parametrize("field", ["server", "model", "history", "slots", "context", "missing"])
def test_static_deployment_drift_is_rejected(field):
    value = gpu_stats()
    identity = bench.static_identity(value)
    if field == "server":
        value["server"]["instance_id"] = "restarted"
    elif field == "model":
        value["model"]["id"] = "different"
    elif field == "history":
        value["capacity_plan"]["indexer_history"] = {"mode": "ram"}
    elif field == "slots":
        value["limits"]["lanes"] = 4
    elif field == "context":
        value["model"]["context_length"] = value["capacity_plan"]["context_window"] = 8192
    else:
        del value["server"]["instance_id"]
    with pytest.raises(ValueError):
        bench.verify_static(value, identity)


def test_summary_write_failure_keeps_original_http_failure(monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for index in range(3):
        (data / f"{index}-p128-t{index}-o512-request.json").write_text(json.dumps(body(index)))
    primary, cleanup = RuntimeError("original HTTP failure"), OSError("summary storage failed")

    def fail_call(*args, **kwargs):
        raise primary

    def write(path, value):
        if path.name == "summary.json":
            raise cleanup

    monkeypatch.setattr(bench.live, "Client", lambda *args, **kwargs: SimpleNamespace(call=fail_call))
    monkeypatch.setattr(bench.live, "write_json", write)
    args = SimpleNamespace(bodies_dir=data, request_json=[], reps=2, image="sha256:" + "a" * 64,
                           out=tmp_path / "output", url="http://unused", timeout=900, phase="gpu3-b",
                           response_mib=16, sse_line_kib=64)
    with pytest.raises(RuntimeError, match="original HTTP") as caught:
        bench.run_phase(args)
    assert caught.value is primary
    assert caught.value.__cause__ is not None

"""Tool boundaries and evidence accounting; no serving or GPU qualification."""
import copy
import io
import json
import sys
from types import SimpleNamespace

import pytest

from tools import v100_adaptive_qualification as qualification


def stats(capacity=18, context=262144):
    return {"object": "tensorfold.stats", "server": {"instance_id": "same-process"},
            "model": {"id": "target-revision", "context_length": context, "kv_dtype": "int8"},
            "capacity_plan": {"session_capacity": {"resident_capacity": capacity, "context_tokens": context,
                                                   "full_context_commitments": capacity},
                              "kv_full_windows": {"materialized": True, "reserved": capacity, "stream_slots": capacity,
                                  "ranks": [{"rank": rank, "materialized": True,
                                             "startup_capacities": [context + 4] * int(capacity),
                                             "allocated_cache_bytes": rank + 1,
                                             "torch_allocated_bytes": 100, "torch_reserved_bytes": 200}
                                            for rank in range(8)]}},
            "load": {"requests_in_flight": 0, "requests_running": 0, "requests_waiting": 0},
            "totals": {"prompt_tokens": 100, "cached_prompt_tokens": 20, "generated_tokens": 5}}


def test_command_exit_failure_missing_and_output_budget():
    command = [sys.executable, "-c"]
    successful = qualification.capture([*command, "print('actual output')"])
    assert successful["status"] == "ok" and successful["exit_code"] == 0
    assert successful["output_utf8"] == "actual output\n"
    failed = qualification.capture([*command, "raise SystemExit(7)"])
    assert failed["status"] == "failed" and failed["exit_code"] == 7
    assert qualification.capture(["/path/that/does/not/exist"])["status"] == "unavailable"
    limited = qualification.capture([*command, "print('x' * 10000)"], limit=37)
    assert limited["status"] == "output-limit" and len(limited["output_utf8"]) == 37


def test_command_timeout_also_retires_stdout_closing_parent():
    result = qualification.capture([sys.executable, "-c", "import os,time; os.close(1);os.close(2);time.sleep(20)"],
                                   timeout=.1)
    assert result["status"] == "timeout" and result["exit_code"] is not None
    assert result["elapsed_s"] < 5


def test_group_is_retired_before_the_only_reaping_wait(monkeypatch):
    events = []
    original_wait = qualification.subprocess.Popen.wait
    original_kill = qualification.os.killpg

    def wait(process, *args, **kwargs):
        events.append("reap")
        return original_wait(process, *args, **kwargs)

    def kill(group, sig):
        events.append("retire-group")
        return original_kill(group, sig)

    monkeypatch.setattr(qualification.subprocess.Popen, "wait", wait)
    monkeypatch.setattr(qualification.os, "killpg", kill)
    result = qualification.capture([sys.executable, "-c", "print('done')"])
    assert result["status"] == "ok" and result["exit_code"] == 0
    assert events == ["retire-group", "reap"]


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_nonfinite_or_unbounded_command_timeout_is_rejected(timeout):
    with pytest.raises(ValueError, match="timeout"):
        qualification.capture([sys.executable, "-c", "pass"], timeout=timeout)


def test_inventory_has_explicit_gpu_rank_identity_and_missing_measurements(monkeypatch):
    def capture(argv, **_kwargs):
        if "--query-gpu=" in " ".join(argv):
            return {"status": "ok", "output_utf8": "0, GPU-a, V100, 0000:01:00.0, 32768, Enabled, 580\n"
                    "1, GPU-b, V100, 0000:02:00.0, 32768, Enabled, 580\n"}
        return {"status": "unavailable", "argv": argv, "exit_code": None}
    monkeypatch.setattr(qualification, "capture", capture)
    result = qualification.inventory(rank_devices=["GPU-b", "0"])
    assert result["evidence"] == "inventory-only"
    assert [row["logical_rank"] for row in result["devices"]] == [1, 0]
    assert result["devices"][0]["memory_mib"] == 32768
    assert result["commands"]["nvlink_state"]["status"] == "unavailable"
    assert "pairwise-transfer-bandwidth" in result["pending"]
    with pytest.raises(ValueError, match="at most once"):
        qualification.inventory(rank_devices=["0", "GPU-a"])


@pytest.mark.parametrize("capacity", [1, 18, 40])
def test_matrix_covers_every_real_count_and_exact_non_threshold_endpoint(capacity):
    value = qualification.matrix(stats(capacity))
    assert value["concurrency"] == list(range(1, capacity + 1))
    assert value["contexts"] == [4096, 65536, 262144]
    assert value["evidence"] == "test-plan-unrun"
    assert "partial-migration" in value["axes"]["transition"]
    assert "all-resident-one-active" in value["axes"]["retention"]
    assert value["capacity_candidates"]["status"].endswith("not-measured-maxima")


def test_matrix_rejects_missing_commitments_mismatched_context_and_tool_overflow():
    value = stats()
    value["capacity_plan"]["session_capacity"]["full_context_commitments"] = False
    with pytest.raises(ValueError, match="full-context"):
        qualification.matrix(value)
    value = stats()
    value["model"]["context_length"] -= 1
    with pytest.raises(ValueError, match="matching public context"):
        qualification.matrix(value)
    with pytest.raises(ValueError, match="case budget"):
        qualification.matrix(stats(40), max_cases=39)
    value = stats(True)
    with pytest.raises(ValueError, match="positive integer"):
        qualification.matrix(value)


def test_token_deltas_do_not_fabricate_missing_counters_or_hide_regressions():
    before, after = stats(), stats()
    after["totals"].update(prompt_tokens=125, cached_prompt_tokens=25, generated_tokens=35)
    assert qualification.counter_deltas(before, after) == {
        "prompt_tokens": 25, "cached_prompt_tokens": 5, "generated_tokens": 30}
    after["totals"].pop("prompt_tokens")
    with pytest.raises(ValueError, match="missing or regressing"):
        qualification.counter_deltas(before, after)
    after = stats()
    after["totals"]["generated_tokens"] = 4
    with pytest.raises(ValueError, match="regressing"):
        qualification.counter_deltas(before, after)


def run_result(*, error=False):
    return {"sent": 1., "first": None if error else 2., "last": None if error else 4.,
            "tokens": 0 if error else 4, "ttft_s": None if error else 1.,
            "decode_tps": None if error else 1.5, "pieces": [] if error else [(2., 1), (3., 1), (4., 1)],
            **({"error": "connection reset"} if error else {})}


def test_every_offered_failure_remains_visible_in_summary():
    summary = qualification.summarize([run_result(), run_result(error=True)])
    assert summary["offered"] == 2 and summary["completed"] == 1
    assert summary["failed_indexes"] == [1]
    assert summary["aggregate"]["failed"] == 1
    assert summary["inter_piece_s"] == [[1., 1.]]


class StatsResponse(io.BytesIO):
    status = 200

    def getheader(self, _key, _default=""):
        return "application/json"


def test_benchmark_preserves_all_runs_and_does_not_claim_matrix_axes(monkeypatch, tmp_path):
    snapshot = stats(2)
    counter = 0

    def read(_url, **_kwargs):
        nonlocal counter
        counter += 1
        current = copy.deepcopy(snapshot)
        current["totals"]["generated_tokens"] += counter * 4
        current["totals"]["prompt_tokens"] += counter * 16
        return StatsResponse(json.dumps(current).encode())

    def together(_url, _model, specs, *_args, **_kwargs):
        return [run_result(error=i == 1) for i in range(len(specs))]

    monkeypatch.setattr("urllib.request.urlopen", read)
    monkeypatch.setattr(qualification.bench, "together", together)
    args = SimpleNamespace(url="http://unused", out=tmp_path / "run", reps=1, warmup=0,
                           tokens=4, max_inflight=8, max_cases=100, seed=17, temperature=0,
                           response_mib=1, sse_line_kib=1, batch_response_mib=8)
    assert qualification.benchmark(args) == 2
    report = json.loads((args.out / "report.json").read_text())
    assert report["complete"] is True and report["passed"] is False
    assert len(report["cells"]) == 2
    cell = json.loads((args.out / report["cells"][1]["artifact"]).read_text())
    assert cell["runs"][1]["error"] == "connection reset"
    assert report["pending_axes"] == qualification.MATRIX_AXES
    assert report["evidence"] == "short-prompt-sse-workload"


def test_json_input_byte_limit_and_cli_matrix_publication(tmp_path):
    source, output = tmp_path / "stats.json", tmp_path / "matrix.json"
    source.write_text(json.dumps(stats(18)))
    with pytest.raises(ValueError, match="byte limit"):
        qualification.load_json(source, limit=8)
    assert qualification.main(["matrix", "--stats", str(source), "--out", str(output)]) == 0
    assert json.loads(output.read_text())["concurrency"][-1] == 18


def test_late_counter_failure_preserves_finished_batch_evidence(monkeypatch, tmp_path):
    snapshot = stats(1)
    calls = 0

    def read(_url, **_kwargs):
        nonlocal calls
        calls += 1
        value = copy.deepcopy(snapshot)
        if calls == 3:
            value["totals"]["generated_tokens"] += 99
        return StatsResponse(json.dumps(value).encode())

    monkeypatch.setattr("urllib.request.urlopen", read)
    monkeypatch.setattr(qualification.bench, "together", lambda *_args, **_kwargs: [run_result()])
    args = SimpleNamespace(url="http://unused", out=tmp_path / "run", reps=1, warmup=0,
                           tokens=4, max_inflight=8, max_cases=100, seed=17, temperature=0,
                           response_mib=1, sse_line_kib=1, batch_response_mib=8)
    with pytest.raises(ValueError, match="differs from measured completed"):
        qualification.benchmark(args)
    report = json.loads((args.out / "report.json").read_text())
    assert report["complete"] is False and report["passed"] is False
    assert len(report["cells"]) == 1
    cell = json.loads((args.out / report["cells"][0]["artifact"]).read_text())
    assert cell["runs"][0]["tokens"] == 4
    assert cell["stats_after"]["totals"]["generated_tokens"] == 104


def test_aggregate_batch_manifest_budget_checked_before_any_requests(monkeypatch, tmp_path):
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: StatsResponse(json.dumps(stats(2)).encode()))
    monkeypatch.setattr(qualification.bench, "together", lambda *_args, **_kwargs: pytest.fail("unbudgeted request"))
    args = SimpleNamespace(url="http://unused", out=tmp_path / "run", reps=51, warmup=0,
                           tokens=4, max_inflight=8, max_cases=100, seed=17, temperature=0,
                           response_mib=1, sse_line_kib=1, batch_response_mib=8)
    with pytest.raises(ValueError, match="total timed and warmup"):
        qualification.benchmark(args)


def test_idle_requirement_rejects_missing_or_boolean_load_counters():
    value = stats()
    qualification.require_idle(value)
    value["load"]["requests_in_flight"] = False
    with pytest.raises(ValueError, match="exclusive idle"):
        qualification.require_idle(value)


def test_materialization_requires_actual_each_rank_and_preserves_unequal_owned_bytes():
    value = stats(18)
    receipt = qualification.materialization(value, qualification.profile(value))
    assert len({rank["allocated_cache_bytes"] for rank in receipt["ranks"]}) == 8
    receipt["ranks"][3]["startup_capacities"][-1] = 128
    with pytest.raises(ValueError, match="every promised full-window"):
        qualification.materialization(value, qualification.profile(value))
    value = stats()
    value["capacity_plan"]["kv_full_windows"]["materialized"] = False
    with pytest.raises(ValueError, match="header-only"):
        qualification.materialization(value, qualification.profile(value))


def test_transfer_inputs_rejected_before_importing_optional_framework():
    with pytest.raises(ValueError, match="positive integer"):
        qualification.transfers(payload_bytes=0)
    with pytest.raises(ValueError, match="positive integer"):
        qualification.transfers(reps=False)

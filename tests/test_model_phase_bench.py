"""Phase-tool boundaries, complete measurements and failure lifetime; no CUDA/model substitute claims."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
try:
    spec = importlib.util.spec_from_file_location("model_phase_bench", ROOT / "tools/model_phase_bench.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
finally:
    sys.path.pop(0)


def fixture_plan(tmp_path, warm=4):
    metadata = tmp_path / "config.json"
    metadata.write_text('{"fixture": true}')
    body = {"messages": [{"role": "user", "content": "text"}], "ignore_eos": True,
            "max_tokens": 4, "draft": True, "temperature": 0}
    return {"schema": 1, "artifact_id": "sha256:" + "0" * 64,
            "source_sha256": {"tool": hashlib.sha256(Path(tool.__file__).read_bytes()).hexdigest(),
                              "tensorfold.cuda.direct_read": hashlib.sha256(metadata.read_bytes()).hexdigest()},
            "input_files": [{"path": "config.json", "sha256": hashlib.sha256(metadata.read_bytes()).hexdigest()}],
            "serve_args": ["serve", "/fixture/model"], "port": 8899,
            "cache_state": "CPU fixture only; no real model/image qualification",
            "cases": [{"id": "cold", "phase": "cold", "body": body, "prompt_tokens": 5, "cached_tokens": 0},
                      {"id": "warm", "phase": "warm", "body": body, "prompt_tokens": 5, "cached_tokens": warm}]}


def save_plan(tmp_path, plan):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path


def install_runtime(monkeypatch, tmp_path, *, warm=4, fault=None, native_error=None, close_error=None):
    state = {"constructed": 0, "closed": False, "calls": 0, "syncs": 0}
    native_error = RuntimeError("native partial failure") if native_error is None else native_error
    close_error = OSError("native retirement failure") if close_error is None else close_error

    class Engine:
        context_window = 4096

        def generate(self, prompt, count, sampling, receive, *, draft, stop_eos):
            assert prompt == [1, 2, 3, 4, 5] and count == 4 and draft and not stop_eos
            state["calls"] += 1
            receive([20, 21])
            if fault == "partial":
                raise native_error
            receive([22, 23] if fault != "short" else [22])
            if fault == "stats-list":
                return ["invalid statistics type"]
            if fault == "stats-repr":
                return state["stats"]
            return {"cached": (0 if state["calls"] == 1 else warm) if fault != "cached" else True,
                    "prefill_s": .1, "decode_s": float("nan") if fault == "nan" else .2,
                    "rounds": 2, "drafted": 2, "accepted": 1}

    app = SimpleNamespace(engine=Engine(), served="fixture")

    def prepare(body, chat):
        assert chat
        return SimpleNamespace(prompt=[1, 2, 3, 4, 5], max_tokens=3 if fault == "prepared" else 4,
                               ignore_eos=True, vision=None, grammar=None, calls=None, stop=(),
                               think_budget=0, penalties=None, sampling=None)
    app.prepare = prepare

    def with_app(serve_args, port, work):
        assert serve_args == ["serve", "/fixture/model"] and port == 8899
        state["constructed"] += 1
        try:
            work(app)
        finally:
            state["closed"] = True
        if fault == "close":
            raise close_error

    def synchronize():
        state["syncs"] += 1
        if fault == "settle" and state["calls"]:
            raise close_error
    torch = SimpleNamespace(__version__="fixture", version=SimpleNamespace(cuda="fixture"),
                            cuda=SimpleNamespace(synchronize=synchronize, get_device_name=lambda: "CPU fixture",
                                                 get_device_capability=lambda: (12, 1)))
    monkeypatch.setattr(tool, "with_app", with_app)
    monkeypatch.setattr(tool, "host_snapshot", lambda: {"fixture": True})
    monkeypatch.setattr(tool.HotGate, "wait", lambda: 0.)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "tensorfold.cuda.direct_read",
                        SimpleNamespace(__file__=str(tmp_path / "config.json")))
    return state, native_error, close_error


@pytest.mark.parametrize("warm", [4, 5], ids=["last-token-left", "full-prompt-cached"])
def test_complete_typed_cold_warm_outcomes_and_cli_ownership(monkeypatch, tmp_path, warm):
    plan = fixture_plan(tmp_path, warm)
    state, _, _ = install_runtime(monkeypatch, tmp_path, warm=warm)
    path = save_plan(tmp_path, plan)
    report = tool.run(path, tmp_path / "out")
    assert state["closed"] and state["constructed"] == 1 and state["calls"] == 2 and state["syncs"] == 4
    assert report["status"] == "COMPLETE" and report["startup_s"] >= 0
    assert report["cli_lifetime_s"] >= report["startup_s"]
    assert (tmp_path / "out/plan.json").read_bytes() == path.read_bytes()
    for row in report["records"]:
        assert row["status"] == "COMPLETE" and row["native_settled"]
        assert row["token_ids"] == [20, 21, 22, 23] and type(row["response_tokens"]) is int
        assert row["first_s"] <= row["last_s"] <= row["generated_s"] <= row["settled_wall_s"]
        assert row["prepare_s"] <= row["first_from_prepare_s"] <= row["prepared_request_wall_s"]
    assert report["records"][0]["prompt_sha256"] == report["records"][1]["prompt_sha256"]
    assert json.loads((tmp_path / "out/report.json").read_text())["status"] == "COMPLETE"
    with pytest.raises(FileExistsError):
        tool.run(path, tmp_path / "out")


@pytest.mark.parametrize("field,value", [("schema", True), ("port", False), ("artifact_id", "latest"),
                                        ("cache_state", ""), ("serve_args", ["serve", "/model", "--api-key=secret"])])
def test_unsealed_or_ambiguous_plan_rejected_before_model(monkeypatch, tmp_path, field, value):
    plan = fixture_plan(tmp_path)
    plan[field] = value
    state, _, _ = install_runtime(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        tool.run(save_plan(tmp_path, plan), tmp_path / "out")
    assert not state["constructed"] and not (tmp_path / "out").exists()


def test_changed_input_or_tool_hash_rejected_before_model(monkeypatch, tmp_path):
    plan = fixture_plan(tmp_path)
    state, _, _ = install_runtime(monkeypatch, tmp_path)
    path = save_plan(tmp_path, plan)
    (tmp_path / "config.json").write_text("changed")
    with pytest.raises(ValueError, match="input file hash changed"):
        tool.run(path, tmp_path / "out")
    plan = fixture_plan(tmp_path)
    plan["source_sha256"]["tool"] = "f" * 64
    with pytest.raises(ValueError, match="tool source differs"):
        tool.run(save_plan(tmp_path, plan), tmp_path / "out")
    assert state["constructed"] == 0


@pytest.mark.parametrize("fault,error_text", [("prepared", "prepared request"), ("short", "fewer"),
                                            ("cached", "nonnegative integer"), ("nan", "JSON compliant")])
def test_incomplete_or_invalid_measurement_retains_failure_and_no_success(monkeypatch, tmp_path, fault, error_text):
    plan = fixture_plan(tmp_path)
    state, _, _ = install_runtime(monkeypatch, tmp_path, fault=fault)
    with pytest.raises(ValueError, match=error_text):
        tool.run(save_plan(tmp_path, plan), tmp_path / "out")
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert state["closed"] and report["status"] == "FAILED"
    assert report["records"][0]["status"] == "FAILED" and report["records"][1]["status"] == "NOT_RUN"
    assert report["records"][0]["native_settled"]
    if fault == "nan":
        assert "nan" in report["records"][0]["invalid_stats_repr"]


def test_partial_native_failure_preserves_exact_primary_and_ids(monkeypatch, tmp_path):
    plan = fixture_plan(tmp_path)
    state, native, _ = install_runtime(monkeypatch, tmp_path, fault="partial")
    with pytest.raises(RuntimeError) as caught:
        tool.run(save_plan(tmp_path, plan), tmp_path / "out")
    assert caught.value is native and state["closed"]
    row = json.loads((tmp_path / "out/report.json").read_text())["records"][0]
    assert row["token_ids"] == [20, 21] and row["response_tokens"] == 2 and row["native_settled"]


def test_nonobject_stats_raise_type_error_and_retain_complete_consumed_output(monkeypatch, tmp_path):
    plan = fixture_plan(tmp_path)
    state, _, _ = install_runtime(monkeypatch, tmp_path, fault="stats-list")
    with pytest.raises(TypeError, match="statistics must be an object"):
        tool.run(save_plan(tmp_path, plan), tmp_path / "out")
    assert state["closed"]
    row = json.loads((tmp_path / "out/report.json").read_text())["records"][0]
    assert row["stats"] == ["invalid statistics type"] and row["token_ids"] == [20, 21, 22, 23]
    assert row["status"] == "FAILED" and row["native_settled"]


@pytest.mark.parametrize("fault", ["settle", "close"])
def test_deferred_completion_and_cli_close_are_required_outcomes(monkeypatch, tmp_path, fault):
    plan = fixture_plan(tmp_path)
    state, _, retirement = install_runtime(monkeypatch, tmp_path, fault=fault)
    with pytest.raises(OSError) as caught:
        tool.run(save_plan(tmp_path, plan), tmp_path / "out")
    assert caught.value is retirement and state["closed"]
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["status"] == "FAILED"
    if fault == "settle":
        assert not report["records"][0]["native_settled"]
    else:
        assert all(row["status"] == "COMPLETE" for row in report["records"])


def test_installed_source_mismatch_never_generates(monkeypatch, tmp_path):
    plan = fixture_plan(tmp_path)
    plan["source_sha256"]["tensorfold.cuda.direct_read"] = "f" * 64
    state, _, _ = install_runtime(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="installed source differs"):
        tool.run(save_plan(tmp_path, plan), tmp_path / "out")
    assert state["closed"] and state["calls"] == 0
    assert json.loads((tmp_path / "out/report.json").read_text())["status"] == "FAILED"


def test_progress_publication_failure_retains_partial_inference_primary(monkeypatch, tmp_path):
    plan = fixture_plan(tmp_path)
    state, native, _ = install_runtime(monkeypatch, tmp_path, fault="partial")
    original = tool.Progress.update

    def update(phase, **fields):
        if phase == "idle" and state["calls"]:
            raise OSError("heartbeat publication failed")
        return original(phase, **fields)
    monkeypatch.setattr(tool.Progress, "update", update)
    with pytest.raises(RuntimeError) as caught:
        tool.run(save_plan(tmp_path, plan), tmp_path / "out")
    assert caught.value is native and state["closed"]
    report = json.loads((tmp_path / "out/report.json").read_text())
    row = report["records"][0]
    assert row["token_ids"] == [20, 21] and row["error"]["message"] == "native partial failure"
    assert row["progress_error"]["message"] == "heartbeat publication failed"
    assert report["status"] == "FAILED" and report["progress_error"]


def test_report_publication_failure_retains_inference_primary(monkeypatch, tmp_path):
    plan = fixture_plan(tmp_path)
    state, native, _ = install_runtime(monkeypatch, tmp_path, fault="partial")
    original = tool.write_json
    publications = []

    def write(path, document):
        publications.append(document["status"])
        if state["calls"] and document["status"] == "RUNNING":
            raise OSError("report publication failed")
        return original(path, document)
    monkeypatch.setattr(tool, "write_json", write)
    with pytest.raises(RuntimeError) as caught:
        tool.run(save_plan(tmp_path, plan), tmp_path / "out")
    assert caught.value is native and state["closed"] and "FAILED" in publications
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["records"][0]["token_ids"] == [20, 21] and report["status"] == "FAILED"


def failure_roots(error):
    pending, seen = [error], []
    while pending:
        item = pending.pop()
        if item is None or any(item is prior for prior in seen):
            continue
        seen.append(item)
        pending.extend((item.__cause__, item.__context__))
        if isinstance(item, BaseExceptionGroup):
            pending.extend(item.exceptions)
        pending.extend(getattr(item, "_tensorfold_retained_failures", ()))
    return seen


def test_settlement_attempts_both_operations_and_retains_both_raw_failures():
    drained, synchronized = OSError("drain failed"), MemoryError("synchronize failed")
    original = LookupError("original drain cause")
    drained.__cause__ = original
    calls = []

    def drain():
        calls.append("drain")
        raise drained

    def synchronize():
        calls.append("synchronize")
        raise synchronized

    with pytest.raises(OSError) as caught:
        tool._settle(SimpleNamespace(drain=drain), SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize)))
    assert caught.value is drained and calls == ["drain", "synchronize"]
    roots = failure_roots(drained)
    assert all(any(root is item for root in roots) for item in (original, synchronized))


def test_settlement_lookup_failure_still_attempts_device_completion():
    lookup = MemoryError("drain binding exhausted")
    calls = []

    class Engine:
        @property
        def drain(self):
            raise lookup

    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: calls.append("synchronize")))
    with pytest.raises(MemoryError) as caught:
        tool._settle(Engine(), torch)
    assert caught.value is lookup and calls == ["synchronize"]


def test_formatter_exhaustion_cannot_skip_retirement_progress_or_partial_report(monkeypatch, tmp_path):
    plan = fixture_plan(tmp_path)
    state, native, _ = install_runtime(monkeypatch, tmp_path, fault="partial")
    prior_cause, prior_context = ValueError("original cause"), LookupError("original context")
    native.__cause__, native.__context__ = prior_cause, prior_context
    formatting = MemoryError("formatting exhausted")
    publications, idle = [], []
    original_write, original_progress = tool.write_json, tool.Progress.update

    def failure(error):
        raise formatting

    def write(path, document):
        publications.append((document["status"], state["closed"]))
        return original_write(path, document)

    def progress(phase, **fields):
        if phase == "idle":
            idle.append((state["calls"], state["closed"]))
        return original_progress(phase, **fields)

    monkeypatch.setattr(tool, "_failure", failure)
    monkeypatch.setattr(tool, "write_json", write)
    monkeypatch.setattr(tool.Progress, "update", progress)
    with pytest.raises(RuntimeError) as caught:
        tool.run(save_plan(tmp_path, plan), tmp_path / "out")
    assert caught.value is native and state["closed"] and state["syncs"] == 2
    # configure publishes an initial idle frame before the request and both
    # mandatory retirement tails publish their own idle frame afterward.
    assert idle == [(0, False), (1, False), (1, True)] and ("FAILED", True) in publications
    roots = failure_roots(native)
    assert all(any(root is item for root in roots) for item in (prior_cause, prior_context, formatting))
    report = json.loads((tmp_path / "out/report.json").read_text())
    row = report["records"][0]
    assert report["status"] == row["status"] == "FAILED"
    assert report["diagnostic_failed"] and row["diagnostic_failed"]
    assert row["token_ids"] == [20, 21] and row["response_tokens"] == 2 and row["native_settled"]


@pytest.mark.parametrize("fault", ["partial", "close"])
def test_bad_exception_string_preserves_native_primary_and_cli_close(monkeypatch, tmp_path, fault):
    plan = fixture_plan(tmp_path)
    formatting = MemoryError("exception string exhausted")

    class BadStringError(RuntimeError):
        def __str__(self):
            raise formatting

    native = BadStringError()
    state, _, _ = install_runtime(monkeypatch, tmp_path, fault=fault,
                                  native_error=native, close_error=native)
    with pytest.raises(BadStringError) as caught:
        tool.run(save_plan(tmp_path, plan), tmp_path / "out")
    assert caught.value is native and state["closed"]
    assert any(root is formatting for root in failure_roots(native))
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["status"] == "FAILED" and report["diagnostic_failed"]
    assert all(row["native_settled"] for row in report["records"] if row["status"] != "NOT_RUN")
    assert state["syncs"] == (2 if fault == "partial" else 4)


def test_bad_stats_representation_keeps_json_error_and_consumed_ids(monkeypatch, tmp_path):
    plan = fixture_plan(tmp_path)
    state, _, _ = install_runtime(monkeypatch, tmp_path, fault="stats-repr")
    formatting = MemoryError("statistics representation exhausted")

    class BadStats:
        def __repr__(self):
            raise formatting

    state["stats"] = BadStats()
    with pytest.raises(TypeError, match="JSON serializable") as caught:
        tool.run(save_plan(tmp_path, plan), tmp_path / "out")
    assert state["closed"] and state["syncs"] == 2
    assert any(root is formatting for root in failure_roots(caught.value))
    row = json.loads((tmp_path / "out/report.json").read_text())["records"][0]
    assert row["token_ids"] == [20, 21, 22, 23] and row["native_settled"] and row["diagnostic_failed"]


@pytest.mark.parametrize("retirement", [False, True], ids=["heartbeat-tail", "all-five-request-slots"])
def test_progress_formatter_and_publication_failures_preserve_all_raw_roots(monkeypatch, tmp_path, retirement):
    plan = fixture_plan(tmp_path)
    state, native, _ = install_runtime(monkeypatch, tmp_path, fault="partial")
    progress_error, formatting, publication = OSError("progress failure"), MemoryError("format failure"), OSError("publish failure")
    retirement_error = OSError("retirement failure")
    original_failure, original_write, original_progress = tool._failure, tool.write_json, tool.Progress.update
    original_settle = tool._settle
    publication_attempts = []

    def progress(phase, **fields):
        if phase == "idle" and state["calls"]:
            raise progress_error
        return original_progress(phase, **fields)

    def failure(error):
        if error is progress_error or error is retirement_error:
            raise formatting
        return original_failure(error)

    def settle(engine, torch):
        original_settle(engine, torch)
        if retirement and state["calls"]:
            raise retirement_error

    def write(path, document):
        if state["calls"]:
            publication_attempts.append(state["closed"])
            if not state["closed"]:
                raise publication
        return original_write(path, document)

    monkeypatch.setattr(tool.Progress, "update", progress)
    monkeypatch.setattr(tool, "_failure", failure)
    monkeypatch.setattr(tool, "write_json", write)
    monkeypatch.setattr(tool, "_settle", settle)
    with pytest.raises(RuntimeError) as caught:
        tool.run(save_plan(tmp_path, plan), tmp_path / "out")
    assert caught.value is native and state["closed"] and state["syncs"] == 2
    assert publication_attempts == [False, True]
    roots = failure_roots(native)
    assert all(any(root is item for root in roots) for item in (progress_error, formatting, publication))
    if retirement:
        assert any(root is retirement_error for root in roots)
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["status"] == "FAILED" and report["diagnostic_failed"]
    assert report["records"][0]["native_settled"] is not retirement

"""Private selector and publication boundaries; no CUDA/model qualification."""
from types import SimpleNamespace as NS

import pytest

from tools import v100_owned_qualification as tool


def arguments(**changes):
    values = dict(onecat_gu_method=None, onecat_down_method="tensorfold", onecat_iq2_source="active-frame",
                  onecat_prefill_method="tensorfold",
                  onecat_device_budget_bytes=None,
                  onecat_global_host_bytes=None, grouped_gate_up_rows=None,
                  prepared_feature_receipts=False)
    values.update(changes)
    return NS(**values)


def test_baseline_does_not_import_new_runtime(monkeypatch):
    def forbidden():
        pytest.fail("baseline imported the new capability")
    monkeypatch.setattr(tool, "_onecat_qualification_module", forbidden)
    assert tool._onecat_capability_from_args(arguments()) is None


@pytest.mark.parametrize("field", ["onecat_device_budget_bytes", "onecat_global_host_bytes"])
def test_unselected_credit_is_rejected_before_runtime_import(monkeypatch, field):
    monkeypatch.setattr(tool, "_onecat_qualification_module", lambda: pytest.fail("late validation"))
    with pytest.raises(ValueError, match="explicit private method"):
        tool._onecat_capability_from_args(arguments(**{field: 1}))


def test_selected_capability_receives_exact_explicit_credits(monkeypatch):
    calls, capability = [], object()
    def create(method, **credits):
        calls.append((method, credits))
        return capability
    monkeypatch.setattr(tool, "_onecat_qualification_module", lambda: NS(create_capability=create))
    selected = arguments(onecat_gu_method="onecat-half-epilogue",
                         onecat_device_budget_bytes=4056689664,
                         onecat_global_host_bytes=1073741824)
    assert tool._onecat_capability_from_args(selected) is capability
    assert calls == [("onecat-half-epilogue", dict(device_budget_bytes=4056689664,
                                                 global_host_bytes=1073741824))]


def test_target_down_selection_is_explicit_and_independent_of_gu(monkeypatch):
    calls = []
    monkeypatch.setattr(tool, "_onecat_qualification_module", lambda: NS(
        create_capability=lambda method, **kwargs: calls.append((method, kwargs))))
    tool._onecat_capability_from_args(arguments(onecat_gu_method="tensorfold",
        onecat_down_method="onecat-q8-slots", onecat_device_budget_bytes=1, onecat_global_host_bytes=2))
    assert calls == [("tensorfold", {"device_budget_bytes": 1, "global_host_bytes": 2,
                                     "down_method": "onecat-q8-slots"})]
    with pytest.raises(ValueError, match="explicit private method"):
        tool._onecat_capability_from_args(arguments(onecat_down_method="onecat-q8-slots"))


def test_raw_iq2_source_is_forwarded_without_changing_down_selection(monkeypatch):
    calls = []
    monkeypatch.setattr(tool, "_onecat_qualification_module", lambda: NS(
        create_capability=lambda method, **kwargs: calls.append((method, kwargs))))
    tool._onecat_capability_from_args(arguments(onecat_gu_method="tensorfold-fp32-epilogue",
        onecat_iq2_source="raw", onecat_device_budget_bytes=1, onecat_global_host_bytes=2))
    assert calls == [("tensorfold-fp32-epilogue", {"device_budget_bytes": 1, "global_host_bytes": 2,
                                                "iq2_source": "raw"})]
    with pytest.raises(ValueError, match="explicit private method"):
        tool._onecat_capability_from_args(arguments(onecat_iq2_source="raw"))


def test_prefill_selection_is_independent_and_requires_explicit_capability(monkeypatch):
    calls = []
    monkeypatch.setattr(tool, "_onecat_qualification_module", lambda: NS(
        create_capability=lambda method, **kwargs: calls.append((method, kwargs))))
    tool._onecat_capability_from_args(arguments(onecat_gu_method="tensorfold-fp32-epilogue",
        onecat_prefill_method="candidate", onecat_device_budget_bytes=1, onecat_global_host_bytes=2))
    assert calls == [("tensorfold-fp32-epilogue", {"device_budget_bytes": 1, "global_host_bytes": 2,
                                                "prefill_method": "candidate"})]
    with pytest.raises(ValueError, match="explicit private method"):
        tool._onecat_capability_from_args(arguments(onecat_prefill_method="candidate"))


@pytest.mark.parametrize("changes", [{"grouped_gate_up_rows": 4}, {"prepared_feature_receipts": True}])
def test_unqualified_selector_intersections_fail_before_capability(monkeypatch, changes):
    monkeypatch.setattr(tool, "_onecat_qualification_module", lambda: pytest.fail("late validation"))
    with pytest.raises(ValueError):
        tool._onecat_capability_from_args(arguments(onecat_gu_method="onecat-half-epilogue", **changes))


def test_parent_rejects_credit_before_any_worker_spawn(tmp_path):
    with pytest.raises(ValueError, match="explicit private method"):
        tool.main(["--gpus", "0,1,2,3,4,5,6,7", "--out", str(tmp_path / "unused"),
                   "--onecat-device-budget-bytes", "1", "--", "/unused/model"])
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("flag", ["--prepared-feature-receipts", "--normalization-parity",
                                  "--gdn-batch-parity", "--q8-preparation-parity"])
def test_graph_use_refuses_incompatible_close_observers_before_effects(tmp_path, flag):
    with pytest.raises(ValueError, match="exclude parity/prepared-feature close observers"):
        tool.main(["--gpus", "0,1,2,3,4,5,6,7", "--out", str(tmp_path / "unused"),
                   "--graph-use-receipts", flag, "--", "/unused/model"])
    assert not (tmp_path / "unused").exists()


@pytest.fixture
def publisher(monkeypatch, tmp_path):
    monkeypatch.setattr(tool, "bound_receipt", lambda engine: {"rank": engine.rank})
    monkeypatch.setattr(tool, "_owned_q8_preparation_receipt", lambda engine: None)
    monkeypatch.setattr(tool, "_reference_tile_receipt", lambda *args: {})
    for name in ("_half_policy", "_router_policy", "_empty_prefix_policy"):
        monkeypatch.setattr(tool, name, lambda *args: {})
    monkeypatch.setattr(tool, "notify_failure", lambda: True)
    ready = []
    def mark_ready(name):
        assert (tmp_path / "rank-0.json").is_file()
        ready.append(name)
    engine = NS(rank=0, comm=NS(ready=mark_ready))
    return engine, ready, tmp_path


def test_selected_publication_uses_actual_binding_and_distinct_field(monkeypatch, publisher):
    import json
    engine, ready, output = publisher
    capability, calls = object(), []
    record = {"status": "prepared-execution-eligible", "qualification": "unqualified"}
    def bound(actual, selected):
        calls.append((actual, selected))
        return record
    monkeypatch.setattr(tool, "_onecat_qualification_module", lambda: NS(bound_receipt=bound))
    monkeypatch.setattr(tool, "_grouped_gate_up_receipt", lambda *args: pytest.fail("old selector used"))
    tool.publish_receipt(engine, output, onecat_gu=capability)
    result = json.loads((output / "rank-0.json").read_text())
    assert result["onecat_experts"] == record and "grouped_gate_up" not in result
    assert calls == [(engine, capability)] and ready == ["owned_expert_qualification"]


def test_baseline_publication_preserves_grouped_receipt_without_new_import(monkeypatch, publisher):
    import json
    engine, ready, output = publisher
    record = {"requested_rows": 4, "actual_rows": 4}
    def old_binding(actual, requested):
        assert actual is engine and requested == 4
        return record
    monkeypatch.setattr(tool, "_grouped_gate_up_receipt", old_binding)
    monkeypatch.setattr(tool, "_onecat_qualification_module", lambda: pytest.fail("baseline imported runtime"))
    tool.publish_receipt(engine, output, grouped_gate_up=4)
    result = json.loads((output / "rank-0.json").read_text())
    assert result["grouped_gate_up"] == record and "onecat_experts" not in result
    assert ready == ["owned_expert_qualification"]


def test_failed_binding_cannot_publish_or_mark_ready(monkeypatch, publisher):
    engine, ready, output = publisher
    primary = RuntimeError("provider binding mismatch")
    def fail(*args):
        raise primary
    monkeypatch.setattr(tool, "_onecat_qualification_module", lambda: NS(bound_receipt=fail))
    with pytest.raises(RuntimeError) as caught:
        tool.publish_receipt(engine, output, onecat_gu=object())
    assert caught.value is primary and engine._abort_comm is True
    assert not ready and not (output / "rank-0.json").exists()


def graph_retirement_fixture(monkeypatch, tmp_path):
    import json
    events = []
    state = {"serving_hits": {"verify": 0, "mtp": 0}, "serving_misses": {"verify": 0, "mtp": 0}}
    engine = NS(rank=0, tp=8, depth=4, _closed=False, _abort_comm=False, drained=False,
                w=NS(mtp=object()), multi=NS(), source="a" * 64)
    graph = NS()
    def record():
        assert engine.drained
        events.append("snapshot")
        return state
    def graph_close():
        events.append("native-close")
        state["serving_hits"].update(verify=0, mtp=0)
    graph.record, graph.close = record, graph_close
    engine.multi.graphs = graph
    def close():
        events.append("drain")
        engine.drained = True
        graph.close()
        engine._closed = True
        events.append("engine-closed")
    engine.close = close
    monkeypatch.setattr(tool, "_onecat_qualification_module", lambda: NS(
        _graph_receipt=lambda actual: {"requested": True, "prepared": True, "replay_hits": 1000}))
    monkeypatch.setattr(tool, "_graph_use_sources", lambda *args: {"engine": engine.source})
    (tmp_path / "rank-0.json").write_text(json.dumps({"rank": 0}))
    return engine, state, events, close, graph_close


def test_graph_use_finalization_snapshots_after_drain_before_graph_clear(monkeypatch, tmp_path):
    import json
    engine, state, events, original_close, original_graph_close = graph_retirement_fixture(monkeypatch, tmp_path)
    tool._attach_graph_use_observer(engine, tmp_path, None, 4)
    state["serving_hits"].update(verify=2, mtp=4)
    engine.close()
    report = json.loads((tmp_path / "graph-use-rank-0.json").read_text())
    assert events == ["drain", "snapshot", "native-close", "engine-closed"]
    assert report["closed"] is True and report["failed"] is False and report["mtp_depth"] == 4
    assert report["selector"]["gu"] == "incumbent"
    assert report["graph_use"]["serving_hits"] == {"verify": 2, "mtp": 4}
    assert report["graph_use"]["serving_misses"] == {"verify": 0, "mtp": 0}
    assert engine.close is original_close and engine.multi.graphs.close is original_graph_close


def test_graph_use_counter_failure_still_retires_graphs_and_preserves_failure(monkeypatch, tmp_path):
    import json
    engine, state, events, _, _ = graph_retirement_fixture(monkeypatch, tmp_path)
    tool._attach_graph_use_observer(engine, tmp_path, None, 4)
    state["serving_hits"]["mtp"] = True
    with pytest.raises(ValueError, match="per-role schema"):
        engine.close()
    assert events == ["drain", "snapshot", "native-close"]
    report = json.loads((tmp_path / "graph-use-rank-0.json").read_text())
    assert report["closed"] is False and report["failed"] is True
    assert report["snapshot_error_type"] == "ValueError" and report["graph_use"] is None


def test_graph_use_publication_failure_propagates_after_actual_retirement(monkeypatch, tmp_path):
    engine, _, events, _, _ = graph_retirement_fixture(monkeypatch, tmp_path)
    tool._attach_graph_use_observer(engine, tmp_path, None, 4)
    primary = OSError("injected graph-use publication failure")
    def refuse(*args, **kwargs):
        raise primary
    monkeypatch.setattr(tool, "write_json", refuse)
    with pytest.raises(OSError) as caught:
        engine.close()
    assert caught.value is primary and engine._closed
    assert events[-1] == "engine-closed" and not (tmp_path / "graph-use-rank-0.json").exists()


def test_graph_use_source_change_is_not_a_positive_closed_receipt(monkeypatch, tmp_path):
    import json
    engine, _, _, _, _ = graph_retirement_fixture(monkeypatch, tmp_path)
    tool._attach_graph_use_observer(engine, tmp_path, None, 4)
    engine.source = "b" * 64
    with pytest.raises(ValueError, match="identity changed"):
        engine.close()
    report = json.loads((tmp_path / "graph-use-rank-0.json").read_text())
    assert engine._closed and report["failed"] and not report["closed"]


def test_graph_use_preserves_original_close_failure(monkeypatch, tmp_path):
    import json
    engine, _, _, _, _ = graph_retirement_fixture(monkeypatch, tmp_path)
    primary = RuntimeError("injected underlying close failure")
    def failed_close():
        raise primary
    engine.close = failed_close
    tool._attach_graph_use_observer(engine, tmp_path, None, 4)
    with pytest.raises(RuntimeError) as caught:
        engine.close()
    assert caught.value is primary
    report = json.loads((tmp_path / "graph-use-rank-0.json").read_text())
    assert report["failed"] and not report["closed"] and report["graph_use"] is None


def test_eager_arm_receipt_does_not_invent_graph_replays(monkeypatch, tmp_path):
    import json
    engine, _, _, _, _ = graph_retirement_fixture(monkeypatch, tmp_path)
    engine.multi.graphs = None
    def close():
        engine._closed = True
    engine.close = close
    monkeypatch.setattr(tool, "_onecat_qualification_module", lambda: NS(
        _graph_receipt=lambda actual: {"requested": False, "prepared": False}))
    tool._attach_graph_use_observer(engine, tmp_path, None, 4)
    engine.close()
    report = json.loads((tmp_path / "graph-use-rank-0.json").read_text())
    assert report["closed"] and report["graph_use"] == {
        "requested": False, "prepared": False, "serving_hits": None, "serving_misses": None}

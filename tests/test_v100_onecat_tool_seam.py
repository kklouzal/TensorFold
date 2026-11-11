"""Private selector and publication boundaries; no CUDA/model qualification."""
from types import SimpleNamespace as NS

import pytest

from tools import v100_owned_qualification as tool


def arguments(**changes):
    values = dict(onecat_gu_method=None, onecat_device_budget_bytes=None,
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

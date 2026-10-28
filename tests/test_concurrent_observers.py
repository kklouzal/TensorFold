"""Finite diagnostics preserve real calls and bound retained observation data."""
from contextlib import nullcontext
import json
from types import SimpleNamespace as NS

import pytest
pytest.importorskip("torch")

from tools import concurrent_observers as diagnostic  # noqa: E402


def test_counter_payload_is_bounded_and_overflow_is_explicit():
    owner = diagnostic._Counters()
    for values in (owner.round_live, owner.ready_segments, owner.selected_segments, owner.selected_rows,
                   owner.compute_rows, owner.compute_segments, owner.owner_rows):
        owner.bucket(values, 1000000)
        assert values[-1] == 1
    owner.calls[0] = (1 << 63) - 1
    owner.add(owner.calls, 0)
    assert owner.overflowed and owner.calls[0] == (1 << 63) - 1
    assert len(json.dumps(owner.snapshot()).encode()) < 8192


def test_wrappers_preserve_inputs_results_failures_and_actual_aliases(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import (
        multi, forward, multi_graphs, bound_graphs, history_owner, adaptive_controller,
    )
    monkeypatch.setattr(diagnostic, "_owner", None)
    calls, labels = [], []
    def original_round(decoder):
        calls.append(("round", decoder))
        return decoder
    def original_select(controller, work, plan):
        calls.append(("select", controller, work, plan))
        return [work[0][0], work[1][0]]
    sentinel = object()
    def original_compute(weights, segments, buffers, *args, **kwargs):
        calls.append(("compute", weights, segments, buffers, args, kwargs))
        return sentinel
    error = RuntimeError("original pending write failed")
    def original_prepare(*args, **kwargs):
        calls.append(("prepare", args, kwargs))
        raise error
    monkeypatch.setattr(multi.MultiDecoder, "_round", original_round)
    monkeypatch.setattr(adaptive_controller.AdaptiveController, "select", original_select)
    for module in (forward, multi, multi_graphs, bound_graphs):
        monkeypatch.setattr(module, "compute", original_compute)
    monkeypatch.setattr(history_owner, "prepare", original_prepare)
    def record(label):
        labels.append(label)
        return nullcontext()
    plan = diagnostic.observer_plan(record)
    for owner, name, old, wrapper in plan:
        assert getattr(owner, name) is old
        monkeypatch.setattr(owner, name, wrapper)
    decoder = NS(streams={0: NS(done=False), 1: NS(done=True)})
    assert multi.MultiDecoder._round(decoder) is decoder
    work, execution = [(4, 3), (7, 2)], {"row_budget": 32}
    assert adaptive_controller.AdaptiveController.select(sentinel, work, execution) == [4, 7]
    segments = [(object(), 0, 3), (object(), 3, 5)]
    buffer = NS(prefill=False)
    for module in (forward, multi, multi_graphs, bound_graphs):
        assert module.compute(sentinel, segments, buffer, check=False) is sentinel
    with pytest.raises(RuntimeError) as caught:
        history_owner.prepare(sentinel, sentinel, sentinel, sentinel, buffer, 2, 5, mtp=True)
    assert caught.value is error
    stats = diagnostic.observer_snapshot()
    assert stats["round_live"][1] == 1
    assert stats["ready_segments"][2] == stats["selected_segments"][2] == stats["selected_rows"][5] == 1
    assert stats["compute_rows"][5] == stats["compute_segments"][2] == 4
    assert stats["owner_rows"][3] == 1
    assert stats["phases"]["mtp_owner_prepare"]["calls"] == 1
    assert all(label.startswith("tensorfold/body/concurrent/") for label in labels)
    assert len(calls) == 7
    with pytest.raises(RuntimeError, match="already owned"):
        diagnostic.observer_plan(record)


@pytest.mark.parametrize("failure", ["label_exit", "timer"])
def test_observer_cleanup_failure_keeps_original_operation_primary(monkeypatch, failure):
    from tensorfold.families.qwen4_exp.cuda import multi
    monkeypatch.setattr(diagnostic,"_owner",None)
    primary=ValueError("original decode failure")
    secondary=RuntimeError("diagnostic cleanup failure")
    def operation(_decoder):
        raise primary
    monkeypatch.setattr(multi.MultiDecoder,"_round",operation)
    class Label:
        def __enter__(self):
            return self
        def __exit__(self,*_args):
            if failure == "label_exit":
                raise secondary
    plan=diagnostic.observer_plan(lambda _name:Label())
    wrapper=next(wrapper for owner,name,_original,wrapper in plan if owner is multi.MultiDecoder and name=="_round")
    calls=[]
    def clock():
        calls.append(True)
        if failure == "timer" and len(calls)>1:
            raise secondary
        return 1
    monkeypatch.setattr(diagnostic.time,"perf_counter_ns",clock)
    with pytest.raises(ValueError) as caught:
        wrapper(NS(streams={0:NS(done=False)}))
    assert caught.value is primary and primary.__cause__ is secondary


def test_diagnostic_argument_allocation_preserves_original_operation_failure(monkeypatch):
    import sys
    import inspect
    from tensorfold.families.qwen4_exp.cuda import multi
    monkeypatch.setattr(diagnostic,"_owner",None)
    primary=ValueError("original decode failure")
    cleanup=RuntimeError("label exit failed")
    allocation=MemoryError("diagnostic tuple allocation")
    def operation(_decoder):
        raise primary
    monkeypatch.setattr(multi.MultiDecoder,"_round",operation)
    class Label:
        def __enter__(self):
            return self
        def __exit__(self,*_args):
            raise cleanup
    plan=diagnostic.observer_plan(lambda _name:Label())
    wrapper=next(wrapper for owner,name,_original,wrapper in plan if owner is multi.MultiDecoder and name=="_round")
    source,first=inspect.getsourcelines(diagnostic.observer_plan)
    target=first+next(index for index,line in enumerate(source) if "errors = (diagnostic,)" in line)
    fired=[]
    def trace(frame,event,_arg):
        if event=="line" and frame.f_code.co_name=="raise_diagnostic" and frame.f_lineno==target:
            fired.append(True)
            raise allocation
        return trace
    previous=sys.gettrace()
    try:
        sys.settrace(trace)
        with pytest.raises(ValueError) as caught:
            wrapper(NS(streams={0:NS(done=False)}))
    finally:
        sys.settrace(previous)
    assert fired==[True] and caught.value is primary and primary.__cause__ is allocation


def test_actual_full24_depth2_row_histograms_keep_48_to72_distinct():
    owner = diagnostic._Counters()
    for rows in (48, 56, 64, 72):
        owner.bucket(owner.selected_rows, rows)
        owner.bucket(owner.compute_rows, rows)
    owner.bucket(owner.compute_rows, 73)
    assert len(owner.selected_rows) == 74
    assert [owner.selected_rows[i] for i in (48, 56, 64, 72)] == [1, 1, 1, 1]
    assert owner.compute_rows[73] == 1 and owner.selected_rows[73] == 0
    for name in ("round_live", "ready_segments", "selected_segments", "selected_rows",
                 "compute_rows", "compute_segments", "owner_rows"):
        getattr(owner, name)[:] = [diagnostic._LIMIT] * len(getattr(owner, name))
    assert len(json.dumps(owner.snapshot()).encode()) < 16384


def test_owner_rows_retain_actual_prefill256_separately_from_decode_geometry():
    owner = diagnostic._Counters()
    owner.bucket(owner.owner_rows, 72)
    owner.bucket(owner.owner_rows, 256)
    owner.bucket(owner.owner_rows, 257)
    assert owner.owner_rows[72] == owner.owner_rows[256] == owner.owner_rows[257] == 1
    assert not owner.overflowed and len(owner.owner_rows) == 258
    assert len(owner.compute_rows) == 74 and len(owner.selected_rows) == 74

"""CPU controls for graph dispatch, temporary ownership and failure boundaries."""

import dis
import sys
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.gemma4.cuda import graphs as module


def _runner(monkeypatch, *, fail_at=None, enabled=True):
    events = []
    failure = KeyboardInterrupt(fail_at)

    class State:
        position = 0
        poisoned = False
        pos = torch.zeros((), dtype=torch.int32)

        def check(self, rows):
            assert not self.poisoned and rows <= 4

        def advance(self, rows):
            events.append("advance")
            self.position += rows

    class GC:
        def isenabled(self):
            return self.enabled

        def disable(self):
            events.append("disable")
            self.enabled = False

        def enable(self):
            events.append("enable")
            self.enabled = True

    class Graph:
        def __init__(self):
            events.append("construct")
            if fail_at == "construct":
                raise failure

        def replay(self):
            events.append("replay")
            if fail_at == "replay":
                raise failure

    class Capture:
        def __init__(self, graph, *, pool, capture_error_mode):
            assert isinstance(graph, Graph) and pool is runner.pool
            assert capture_error_mode == "thread_local"

        def __enter__(self):
            events.append("capture")

        def __exit__(self, *_args):
            events.append("capture-end")

    class Entries(dict):
        def __setitem__(self, key, value):
            events.append("publish")
            if fail_at == "publish":
                raise failure
            super().__setitem__(key, value)

    def staged(ids, state, *, chunks, output_logits, output_taps):
        assert torch.is_inference_mode_enabled()
        assert state is runner.state and chunks == 1 and output_logits is True
        assert ids.data_ptr() == runner.ids.data_ptr()
        stage = "warm" if "construct" not in events else "captured-work"
        events.append(stage)
        if fail_at == ("warm" if stage == "warm" else "capture"):
            raise failure
        return object(), object(), {1: object()} if output_taps else {}

    runner = module.Graphs.__new__(module.Graphs)
    runner.state, runner.target = State(), SimpleNamespace(forward_staged=staged)
    runner.max_rows, runner.span, runner.limit = 4, 1, 4
    runner.ids, runner.pool, runner.graphs = torch.zeros(4, dtype=torch.int32), object(), Entries()
    gc = GC()
    gc.enabled = enabled
    monkeypatch.setattr(module, "gc", gc)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", Capture)
    return runner, events, failure, gc


@pytest.mark.parametrize("taps", [False, True])
def test_cached_replay_borrows_outputs_and_stages_new_inputs_without_capture(monkeypatch, taps):
    runner, events, _failure, gc = _runner(monkeypatch)
    first = runner.forward([4], runner.state, output_taps=taps)
    assert events == ["warm", "construct", "disable", "capture", "captured-work", "capture-end",
                      "enable", "publish", "replay", "advance"]
    events.clear()
    second = runner.forward([9], runner.state, output_taps=taps)
    assert events == ["replay", "advance"]
    assert runner.ids[0].item() == 9 and runner.state.pos.item() == 1
    assert runner.state.position == 2 and gc.enabled
    assert second.hidden is first.hidden and second.logits is first.logits and second.taps is first.taps
    assert second.start_position == 1 and first.start_position == 0


def test_miss_preserves_each_boolean_conversion_and_dynamic_target_binding(monkeypatch):
    runner, events, _failure, _gc = _runner(monkeypatch)
    calls = []

    class Taps:
        def __bool__(self):
            calls.append("bool")
            return True

    original = runner.target.forward_staged
    capture = runner._capture

    def capture_defaults(run):
        assert run.__closure__ is None
        assert run.__defaults__[0] is runner
        assert run.__defaults__[1].data_ptr() == runner.ids.data_ptr()
        assert run.__defaults__[2:] == (runner.state, 1, taps)
        return capture(run)

    def warm(*args, **kwargs):
        result = original(*args, **kwargs)
        runner.target = SimpleNamespace(forward_staged=original)
        return result

    runner.target.forward_staged = warm
    runner._capture = capture_defaults
    taps = Taps()
    runner.forward([4], runner.state, output_taps=taps)
    assert len(calls) == 3 and events.count("warm") == events.count("captured-work") == 1
    runner.forward([5], runner.state, output_taps=Taps())
    assert len(calls) == 4


@pytest.mark.parametrize("stage", ["warm", "construct", "capture", "publish", "replay"])
def test_failed_staged_work_preserves_primary_poison_and_publication_boundary(monkeypatch, stage):
    runner, _events, failure, gc = _runner(monkeypatch, fail_at=stage)
    with pytest.raises(KeyboardInterrupt) as raised:
        runner.forward([4], runner.state, output_taps=True)
    assert raised.value is failure
    assert runner.state.poisoned and runner.state.position == 0 and gc.enabled
    assert len(runner.graphs) == (1 if stage == "replay" else 0)


def test_capture_keeps_preexisting_disabled_gc_disabled(monkeypatch):
    runner, events, _failure, gc = _runner(monkeypatch, enabled=False)
    runner.forward([4], runner.state, output_taps=False)
    assert not gc.enabled and "disable" in events and "enable" not in events


@pytest.mark.skipif(sys.implementation.name != "cpython", reason="CPython opcode fault boundary")
def test_function_construction_failure_precedes_poison_boundary(monkeypatch):
    runner, events, _failure, gc = _runner(monkeypatch)
    code = module.Graphs.forward.__wrapped__.__code__
    offsets = {item.offset for item in dis.get_instructions(code) if item.opname == "MAKE_FUNCTION"}
    assert len(offsets) == 1 and code.co_cellvars == ()
    failure = MemoryError("miss function construction")

    def trace(frame, event, _arg):
        if frame.f_code is code:
            frame.f_trace_opcodes = True
            if event == "opcode" and frame.f_lasti in offsets:
                raise failure
        return trace

    previous = sys.gettrace()
    owner = sys._getframe()
    previous_opcodes = owner.f_trace_opcodes
    try:
        # Python3.12 chooses instruction events when settrace is installed.
        owner.f_trace_opcodes = True
        sys.settrace(trace)
        with pytest.raises(MemoryError) as raised:
            runner.forward([4], runner.state, output_taps=True)
    finally:
        sys.settrace(previous)
        owner.f_trace_opcodes = previous_opcodes
        del owner
    assert raised.value is failure
    assert not runner.state.poisoned and runner.state.position == 0 and gc.enabled
    assert not events and not runner.graphs

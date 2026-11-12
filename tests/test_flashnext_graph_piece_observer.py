"""Startup allocation observations must stay outside capture and preserve failures."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.cuda import multi_graphs


@pytest.fixture
def capture(monkeypatch):
    state = SimpleNamespace(active=False, free=1000, allocated=100, reserved=200, ends=0)
    graphs = []

    class Graph:
        def __init__(self):
            assert not state.active
            state.free -= 1
            graphs.append(self)

        def capture_begin(self, **kwargs):
            assert not state.active
            state.active = True

        def capture_end(self):
            assert state.active
            state.active = False
            state.free -= 2
            state.ends += 1

    def outside(value):
        assert not state.active
        return getattr(state, value)

    cuda = SimpleNamespace(CUDAGraph=Graph, stream=lambda _: nullcontext(),
                           memory_allocated=lambda: outside("allocated"),
                           memory_reserved=lambda: outside("reserved"))
    monkeypatch.setattr(multi_graphs, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(multi_graphs, "free_bytes", lambda: outside("free"))
    owner = object.__new__(multi_graphs.RoundGraphs)
    owner.side, owner.pool, owner.pending = object(), object(), []
    return owner, SimpleNamespace(split=None), state, graphs


def test_signed_piece_deltas_exclude_observer_work_and_end_before_next_begin(capture):
    owner, buffers, state, graphs = capture
    insert = ("owner_attention", object(), 5, False)
    observations = []

    def observe(index, ending, values):
        assert not state.active and state.ends == index + 1 and len(graphs) == index + 1
        observations.append((index, ending, values))
        state.free += 50
        state.allocated -= 1
        state.reserved -= 2

    def run():
        state.free -= 9
        state.allocated += 3
        state.reserved += 7
        buffers.split(insert)
        state.free += 20
        state.allocated -= 4
        state.reserved -= 5
        return "output"

    pieces = owner._pieces(run, buffers, observer=observe)
    assert pieces.out == "output" and pieces.inserts == [insert]
    assert [row[:2] for row in observations] == [(0, insert), (1, None)]
    assert [(row[2]["physical_delta_bytes"], row[2]["allocated_delta_bytes"],
             row[2]["reserved_delta_bytes"]) for row in observations] == [(12, 3, 7), (-17, -4, -5)]
    assert all(type(value) in (int, str) for _, _, values in observations for value in values.values())
    assert buffers.split is None and not state.active and owner.pending == []


def test_disabled_observer_performs_no_allocation_queries(capture, monkeypatch):
    owner, buffers, state, _ = capture
    monkeypatch.setattr(multi_graphs, "free_bytes", lambda: pytest.fail("disabled physical query"))
    for name in ("memory_allocated", "memory_reserved"):
        monkeypatch.setattr(multi_graphs.torch.cuda, name, lambda: pytest.fail("disabled allocator query"))
    pieces = owner._pieces(lambda: buffers.split(("select", None)), buffers)
    assert len(pieces.graphs) == 2 and not state.active


@pytest.mark.parametrize("at_split", [True, False])
def test_observer_failure_leaves_closed_capture_and_retained_native_owners(capture, at_split):
    owner, buffers, state, graphs = capture
    failure = RuntimeError("observer publication failed")

    def observe(*args):
        assert not state.active
        raise failure

    with pytest.raises(RuntimeError) as caught:
        owner._pieces(lambda: buffers.split(("select", None)) if at_split else None,
                      buffers, observer=observe)
    assert caught.value is failure
    assert len(graphs) == state.ends == 1 and not state.active and buffers.split is None
    assert len(owner.pending) == 1 and owner.pending[0].graphs == graphs


def test_body_failure_stays_primary_when_cleanup_observer_also_fails(capture):
    owner, buffers, state, _ = capture
    primary, cleanup = ValueError("forward failed"), RuntimeError("observer failed")

    def run():
        raise primary

    def observe(*args):
        assert not state.active
        raise cleanup

    with pytest.raises(ValueError) as caught:
        owner._pieces(run, buffers, observer=observe)
    assert caught.value is primary and caught.value.__cause__ is cleanup
    assert buffers.split is None and not state.active and state.ends == 1


def test_failed_capture_end_is_not_observed_or_retried(capture, monkeypatch):
    owner, buffers, state, _ = capture
    graph = multi_graphs.torch.cuda.CUDAGraph
    original = graph.capture_end
    primary, cleanup = ValueError("forward failed"), RuntimeError("end failed")

    def end(self):
        original(self)
        raise cleanup

    def run():
        raise primary

    monkeypatch.setattr(graph, "capture_end", end)
    with pytest.raises(ValueError) as caught:
        owner._pieces(run, buffers, observer=lambda *args: pytest.fail("failed end observed"))
    assert caught.value is primary and caught.value.__cause__ is cleanup
    assert buffers.split is None and not state.active and state.ends == 1

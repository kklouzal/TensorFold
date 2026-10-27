"""Real State/CompactIndexer guards join transfer events before graph capture."""

from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def capture_runtime(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import bound_graphs, multi_graphs
    from tensorfold.families.qwen4_exp.cuda.compact_indexer import CompactIndexer
    from tensorfold.families.qwen4_exp.cuda.state import State

    events, graphs = [], []
    runtime = NS(capturing=False, wait_failure=None, body_failure=None)

    class Stream:
        def __init__(self, name, number):
            self.name, self.cuda_stream = name, number
            self.joined = None

        def wait_stream(self, producer):
            assert not runtime.capturing
            self.joined = producer
            events.append(("producer", self.name, producer.name))

        def wait_event(self, event):
            events.append(("transfer", self.name, runtime.capturing, event))
            if runtime.capturing:
                raise RuntimeError("905: external event dependency during capture")
            if runtime.wait_failure is not None:
                raise runtime.wait_failure

    producer, side = Stream("producer", 1), Stream("side", 2)
    runtime.current = producer

    @contextmanager
    def on_stream(stream):
        previous, runtime.current = runtime.current, stream
        try:
            yield
        finally:
            runtime.current = previous

    class Graph:
        def __init__(self):
            graphs.append(self)

        def capture_begin(self, **kwargs):
            assert runtime.current is side and not runtime.capturing
            runtime.capturing = True
            events.append(("begin",))

        def capture_end(self):
            runtime.capturing = False
            events.append(("end",))

    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: runtime.current)
    monkeypatch.setattr(torch.cuda, "stream", on_stream)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: events.append(("fence",)))
    monkeypatch.setattr(multi_graphs, "free_bytes", lambda: 4096)
    compact = object.__new__(CompactIndexer)
    compact.empty, compact.entries = NS(device="cuda:0"), []
    compact._transfer_done, compact._transfer_stream = object(), producer.cuda_stream
    compact._transfer_error, compact._transfer_borrowers = None, ()
    state = object.__new__(State)
    state.compact_indexer, state._history_error = compact, None
    state._resize_error = state._kv_error = None
    state.kv_status, state._kv_pending = torch.tensor([0, -2], dtype=torch.int32), False
    segments = [(state, 0, 1)]
    decoder = NS(w=NS(device="cuda:0", comm=None))
    owner = object.__new__(multi_graphs.RoundGraphs)
    owner.dec, owner.side, owner.pool = decoder, side, (1, 0)
    owner.room_bytes, owner.bytes, owner.budget = 0, 0, 4096
    owner.measured, owner.verify, owner.mtp, owner.pending = [], {}, {}, []
    owner._has_room = lambda: True
    bound = object.__new__(bound_graphs.BoundStateGraphs)
    bound.unique, bound.owner, bound.dec = True, owner, decoder
    bound._agree, bound.bindings = bool, {}
    buffers = NS(attn_o=torch.ones(1, 3), split=None)

    def run():
        bound_graphs._kv_begin(segments)  # Shipping guard, including the real CompactIndexer.check/_wait_transfer.
        if runtime.capturing:
            assert side.joined is producer
            if runtime.body_failure is not None:
                raise runtime.body_failure
        return buffers.attn_o

    return NS(runtime=runtime, events=events, graphs=graphs, producer=producer, side=side,
              compact=compact, state=state, bound=bound, owner=owner, buffers=buffers,
              segments=segments, run=run)


def capture(fixture):
    return fixture.bound._capture("verify", ("slot", 1, None), [], fixture.buffers, fixture.segments, fixture.run)


def test_transfer_frontier_joins_side_before_capture_without_changing_the_event(capture_runtime):
    f = capture_runtime
    event = f.compact._transfer_done
    pieces = capture(f)
    transfers = [row for row in f.events if row[0] == "transfer"]
    assert transfers == [("transfer", "side", False, event)]
    assert f.events.index(transfers[0]) < f.events.index(("begin",))
    assert f.compact._transfer_done is event and f.compact._transfer_error is None
    assert f.compact._transfer_stream == f.side.cuda_stream
    assert len(pieces.graphs) == 1 and f.owner.pending == []


def test_transfer_join_failure_is_latched_before_capture_and_preserves_primary(capture_runtime):
    f = capture_runtime
    primary = RuntimeError("prefix transfer join failed")
    f.runtime.wait_failure = primary
    with pytest.raises(RuntimeError) as failure:
        capture(f)
    assert failure.value is primary and f.compact._transfer_error is primary
    assert f.compact._transfer_stream == f.producer.cuda_stream
    assert not f.graphs and f.owner.pending == [] and f.buffers.split is None


def test_existing_transfer_failure_is_not_cleared_or_treated_as_completed(capture_runtime):
    f = capture_runtime
    primary = RuntimeError("older transfer retirement failed")
    f.compact._transfer_error = primary
    with pytest.raises(RuntimeError) as failure:
        capture(f)
    assert failure.value is primary and f.compact._transfer_error is primary
    assert not f.graphs and not any(row[0] == "producer" for row in f.events)


def test_capture_body_failure_retains_graph_owner_and_restores_stream_callback(capture_runtime):
    f = capture_runtime
    primary = RuntimeError("captured kernel submission failed")
    f.runtime.body_failure = primary
    with pytest.raises(RuntimeError) as failure:
        capture(f)
    assert failure.value is primary
    assert len(f.graphs) == 1 and len(f.owner.pending) == 1
    assert not f.runtime.capturing and f.runtime.current is f.producer and f.buffers.split is None

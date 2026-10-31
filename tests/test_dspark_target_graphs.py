"""Target graph tap ownership and accepted-prefix publication contracts."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import forward
from tensorfold.families.qwen4_exp.cuda.dspark_taps import TapWorkspace
from tensorfold.families.qwen4_exp.cuda.state import Buffers, KVNumericError, State
from test_dspark_runtime import assert_target, bits, setup as runtime_fixture
from test_dspark_taps import config, state


@pytest.fixture
def setup(tmp_path, monkeypatch):
    yield from runtime_fixture.__wrapped__(tmp_path, monkeypatch)


def sink(owner, rows, h):
    frame = owner.begin_graph(rows)
    for layer in owner.before:
        frame.before_layer(layer, h)
    frame.finish_graph_sink()


def test_six_lane_padded_export_owns_only_real_rows_and_drops_nan_padding():
    width, counts = 8, (1, 2, 3, 4, 6, 8)
    graph = TapWorkspace(config(), 48, "cpu")
    source = TapWorkspace(config(), 24, "cpu")
    h = torch.full((48, 1024), float("nan"), dtype=torch.float16)
    states, segments, mapping, end = [state(i) for i in range(6)], [], [], 0
    for lane in (5, 2, 0, 4, 1, 3):
        count = counts[lane]
        h[lane * width:lane * width + count].fill_(lane + 0.25)
        segments.append((states[lane], end, end + count))
        mapping.extend(range(lane * width, lane * width + count))
        end += count
    sink(graph, 48, h)
    index = torch.tensor(mapping, dtype=torch.int64)
    batch = graph.export_graph(source, segments, index)
    expected = torch.tensor([i // width + 0.25 for i in mapping]).half()
    bits(batch.features[:, 0], expected)
    assert torch.isfinite(batch.features).all()
    h.fill_(9)
    sink(graph, 48, h)  # copied feature lease no longer borrows graph storage
    bits(batch.features[:, 0], expected)
    with pytest.raises(ValueError, match="active"):
        source.close()
    with pytest.raises(ValueError, match="graph plan"):
        graph.close()
    batch.release()
    source.close()
    graph.retire_graph()
    graph.close()


def test_graph_sink_completeness_alias_and_retirement_are_explicit():
    owner = TapWorkspace(config(), 3, "cpu")
    frame = owner.begin_graph(3)
    frame.before_layer(owner.before[0], torch.zeros(3, 1024, dtype=torch.float16))
    with pytest.raises(ValueError, match="every configured"):
        frame.finish_graph_sink()
    with pytest.raises(ValueError, match="active"):
        owner.retire_graph()
    frame.abort()
    frame = owner.begin_graph(3)
    with pytest.raises(ValueError, match="alias"):
        frame.before_layer(owner.before[0], owner.materialized)
    frame.abort()
    with pytest.raises(ValueError, match="graph plan"):
        owner.close()
    owner.retire_graph()
    owner.close()


@pytest.mark.parametrize("kind", ("dtype", "shape", "empty", "alias", "unrecorded", "busy"))
def test_graph_export_refuses_invalid_ownership_without_touching_output(kind):
    graph, source = TapWorkspace(config(), 3, "cpu"), TapWorkspace(config(), 3, "cpu")
    h = torch.ones(3, 1024, dtype=torch.float16)
    if kind != "unrecorded":
        sink(graph, 3, h)
    index = torch.tensor([0, 1], dtype=torch.int64)
    if kind == "dtype":
        index = index.int()
    elif kind == "shape":
        index = index.view(1, 2)
    elif kind == "empty":
        index = index[:0]
    elif kind == "alias":
        source = graph
    elif kind == "busy":
        active = source.begin(1)
    source.features.fill_(7)
    with pytest.raises(ValueError):
        graph.export_graph(source, [(state(), 0, 2)], index)
    assert torch.all(source.features == 7)
    if kind == "busy":
        active.abort()
    graph.retire_graph()
    graph.close()
    if source is not graph:
        source.close()


def inputs(w, b, st, ids=(4, 7, 19), check=False):
    b.ids[:len(ids)].copy_(torch.tensor(ids, dtype=torch.int32))
    b.ple_v.fill_(0.125)
    return forward.compute(w, [(st, 0, len(ids))], b, check=check).clone()


@pytest.mark.parametrize("cache", ("fp16", "int8"))
def test_target_graph_sink_leaves_real_target_bytes_and_stages_accepted_context_only(setup, cache):
    w, baseline, runtime, _ = setup
    a, ref = State(w, 16, 8, cache), State(baseline, 16, 8, cache)
    graph, source, control = Buffers(w, 8, 16), Buffers(w, 8, 16), Buffers(baseline, 8, 16)
    graph.dspark_graph = True
    segs = [(a, 0, 3)]
    try:
        runtime.check_graph(segs, graph)
        actual = inputs(w, graph, a)
        expected = inputs(baseline, control, ref)
        bits(actual, expected)
        assert a.dspark.pending is None and a.dspark.length == 0
        runtime.stage_graph(source, graph, segs, torch.arange(3))
        assert a._kv_pending == (cache == "int8")
        assert a.dspark.length == 0 and a.dspark.pending.rows == 3
        keys = a.dspark.pending.keys[:, :2].clone()
        forward._kv_check(segs)
        forward.commit(w, a, graph, 3, 2)
        forward.commit(baseline, ref, control, 3, 2)
        assert_target(a, ref)
        assert a.dspark.length == a.pos == 2
        bits(a.dspark.keys[:, :2], keys)
        runtime.check_graph([(a, 0, 1)], graph)
    finally:
        runtime.retire_graph(graph)


def test_target_graph_codec_failure_never_publishes_context_and_reset_replays_healthy(setup):
    w, _, runtime, _ = setup
    a = State(w, 16, 8, "int8")
    graph, source = Buffers(w, 8, 16), Buffers(w, 8, 16)
    graph.dspark_graph = True
    segs = [(a, 0, 3)]
    try:
        runtime.check_graph(segs, graph)
        inputs(w, graph, a)
        runtime.stage_graph(source, graph, segs, torch.arange(3))
        a.kv_status[0].fill_(4)
        with pytest.raises(KVNumericError):
            forward._kv_check(segs)
        assert a.dspark.length == 0
        runtime.abort(None, segs)
        assert a.dspark.pending is None
        a.reset(w)
        runtime.check_graph(segs, graph)
        inputs(w, graph, a)
        runtime.stage_graph(source, graph, segs, torch.arange(3))
        forward._kv_check(segs)
        forward.commit(w, a, graph, 3, 2)
        assert a.dspark.length == a.pos == 2
    finally:
        runtime.retire_graph(graph)


@pytest.mark.parametrize("fault", ("cursor", "pending", "poison", "identity", "unvalidated"))
def test_graph_preflight_rejects_stale_or_uncommitted_actual_state(setup, fault):
    w, _, runtime, _ = setup
    a = State(w, 16, 8, "int8")
    graph = Buffers(w, 8, 16)
    graph.dspark_graph = True
    if fault == "cursor":
        a.pos += 1
    elif fault == "pending":
        a.dspark.pending = NS()
    elif fault == "poison":
        a._kv_error = ValueError("poison")
    elif fault == "identity":
        a.version += 1
    elif fault == "unvalidated":
        a._kv_pending = True
    with pytest.raises(ValueError):
        runtime.check_graph([(a, 0, 1)], graph)
    a.dspark.pending = None
    a._kv_error = None
    runtime.retire_graph(graph)


def test_partial_sink_cannot_export_uninitialized_features_or_finish_as_ordinary_capture():
    graph, source = TapWorkspace(config(), 3, "cpu"), TapWorkspace(config(), 3, "cpu")
    ordinary = graph.begin(1)
    for index in graph.before:
        ordinary.before_layer(index, torch.ones(1, 1024, dtype=torch.float16))
    with pytest.raises(ValueError, match="every configured"):
        ordinary.finish_graph_sink()
    ordinary.abort()
    partial = graph.begin_graph(3)
    partial.before_layer(graph.before[0], torch.ones(3, 1024, dtype=torch.float16))
    partial.abort()
    with pytest.raises(ValueError, match="owned row map"):
        graph.export_graph(source, [(state(), 0, 1)], torch.zeros(1, dtype=torch.int64))
    graph.retire_graph()
    graph.close()
    source.close()


def test_failed_pending_graph_is_reset_before_bound_tap_storage_retirement(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.multi_graphs import RoundGraphs

    tap = TapWorkspace(config(), 1, "cpu")
    sink(tap, 1, torch.ones(1, 1024, dtype=torch.float16))
    sequence = []
    owner = RoundGraphs.__new__(RoundGraphs)
    owner.dec = NS(w=NS(device=torch.device("cpu"), dspark=object()))
    owner.pending = [NS(graphs=[NS(reset=lambda: sequence.append("reset-failed-partial"))])]

    def close_bound():
        assert not owner.pending, "partial native graph still borrows tap storage"
        sequence.append("retire-bound-features")
        tap.retire_graph()
        tap.close()

    owner.bound = NS(close=close_bound)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    owner.close()
    assert sequence == ["reset-failed-partial", "retire-bound-features"]

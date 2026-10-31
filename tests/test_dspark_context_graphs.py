"""Canonical target-context projection regrouping and atomic graph export on CPU."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.dspark_context_graphs import ContextWorkspace
from test_dspark_cuda import absorb, bits, features, owner, provider


def state(p, length=4):
    target = owner()
    target.pos_dev = torch.tensor([0], dtype=torch.int32)
    st = p.make_state(target, capacity=32)
    if length:
        absorb(p, st, features(p.config, length))
    target.pos_dev.fill_(length)
    return st


@pytest.mark.parametrize("counts", ((1,), (3,), (1, 2, 3, 4, 6, 8)))
def test_projected_context_keeps_per_state_original_fixed32_arithmetic_at_all_physical_rows(counts):
    p, width = provider(), 8
    states = [state(p, i % 4) for i in range(len(counts))]
    rows = width * len(states)
    ws = ContextWorkspace(p, rows, "cpu")
    ws.allocate()
    assert ws.nbytes == ws.device_bytes(p.config, rows, p.local["kv_heads"])
    h = torch.full((rows, len(p.config.taps) * p.config.hidden), float("nan"), dtype=torch.float16)
    expected, physical, compact, mapping, end = [], [], [], [], 0
    for i, (st, count) in enumerate(zip(states, counts)):
        frame = features(p.config, count) + i / 64
        h[i*width:i*width+count].copy_(frame)
        expected.append(p._context(st, frame))
        physical.append((st.target, i*width, (i+1)*width))
        compact.append((st.target, end, end+count))
        mapping.extend(range(i*width, i*width+count))
        end += count
    ws.compute(h, physical)
    updates = ws.stage(states, compact, torch.tensor(mapping))
    assert ws.readbacks == ws.frames == 1 and ws.exported_rows == sum(counts)
    for st, update, reference in zip(states, updates, expected):
        bits(update.keys, reference.keys)
        bits(update.values, reference.values)
        assert update.keys.is_contiguous() and update.values.is_contiguous()
        assert not torch._C._overlaps(update.keys, ws.keys)
        assert not torch._C._overlaps(update.values, ws.values)
        assert st.length == st.target.pos and st.pending is update
    with pytest.raises(ValueError, match="still borrows"):
        ws.close()
    # A following captured write must not change any pending independent update.
    ws.keys.fill_(7)
    ws.values.fill_(8)
    for st, update, reference in zip(states, updates, expected):
        keep = min(2, update.rows)
        st.target.pos += keep
        p.commit(st, update, keep)
        bits(st.keys[:, st.length-keep:st.length], reference.keys[:, :keep])
        bits(st.values[:, st.length-keep:st.length], reference.values[:, :keep])
    ws.retire_graph()
    ws.close()


@pytest.mark.parametrize("remote", (False, True))
def test_nonfinite_real_context_or_remote_rejection_stages_no_lane_and_healthy_retry_succeeds(remote):
    p = provider()
    states = [state(p), state(p)]
    ws = ContextWorkspace(p, 4, "cpu")
    ws.allocate()
    h = features(p.config, 4)
    segs = [(states[0].target, 0, 2), (states[1].target, 2, 4)]
    if not remote:
        h[2, 0] = float("nan")
    ws.compute(h, segs)
    normal = p._gather
    if remote:
        p._gather = lambda flags: torch.stack((flags, torch.zeros_like(flags)))
    with pytest.raises(ValueError, match="nonfinite captured context"):
        ws.stage(states, segs, torch.arange(4))
    assert all(st.pending is None and st.length == st.target.pos == 4 for st in states)
    p._gather = normal
    h.zero_()
    ws.compute(h, segs)
    ws.stage(states, segs, torch.arange(4))
    assert all(st.pending is not None and st.length == 4 for st in states)
    for st in states:
        p.discard(st)
    ws.retire_graph()
    ws.close()


@pytest.mark.parametrize("fault", ("pending", "cursor", "coverage", "dtype", "replaced_storage"))
def test_projection_export_metadata_and_ownership_failure_precedes_all_pending_publication(fault):
    p = provider()
    st = state(p)
    ws = ContextWorkspace(p, 3, "cpu")
    ws.allocate()
    segs, index = [(st.target, 0, 3)], torch.arange(3)
    ws.compute(features(p.config, 3), segs)
    if fault == "pending":
        st.pending = NS()
    elif fault == "cursor":
        st.target.pos += 1
    elif fault == "coverage":
        segs = [(st.target, 0, 2)]
    elif fault == "dtype":
        index = index.int()
    elif fault == "replaced_storage":
        ws.keys = torch.empty_like(ws.keys)
    with pytest.raises(ValueError):
        ws.stage([st], segs, index)
    if fault == "pending":
        assert isinstance(st.pending, NS)
        st.pending = None
    else:
        assert st.pending is None
    ws.retire_graph()
    ws.close()


def test_projection_rejects_owned_input_alias_and_tracks_live_device_position_without_host_cursor():
    p = provider()
    c = p.config
    st = state(p)
    ws = ContextWorkspace(p, 3, "cpu")
    ws.allocate()
    alias = ws.keys.reshape(3, len(c.taps) * c.hidden)
    with pytest.raises(ValueError, match="alias"):
        ws.compute(alias, [(st.target, 0, 3)])
    h = features(c, 3)
    ws.compute(h, [(st.target, 0, 3)])
    first = ws.keys.clone()
    st.target.pos_dev.fill_(5)
    ws.compute(h, [(st.target, 0, 3)])
    assert not torch.equal(ws.keys, first)
    assert st.target.pos == st.length == 4  # compute never publishes a host cursor
    ws.retire_graph()
    ws.close()


@pytest.mark.parametrize("cache", ("fp16", "int8"))
def test_runtime_forward_graph_projection_preserves_target_and_ordinary_context_bytes(tmp_path, monkeypatch, cache):
    from test_dspark_runtime import assert_target, setup as make_setup
    from tensorfold.families.qwen4_exp.cuda import dspark_context_graphs as module, forward
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    fixture = make_setup.__wrapped__(tmp_path, monkeypatch)
    w, _, runtime, _ = next(fixture)
    graph, source, ref = Buffers(w, 8, 16), Buffers(w, 8, 16), Buffers(w, 8, 16)
    graph.dspark_graph = True
    a, b = State(w, 16, 8, cache), State(w, 16, 8, cache)
    monkeypatch.setattr(module, "_ENABLED", True)
    runtime.prepare_graph_context(graph, NS(_agree=lambda enabled: enabled))
    try:
        outputs = []
        for st, buf in ((a, graph), (b, ref)):
            buf.ids[:3].copy_(torch.tensor([4, 7, 19], dtype=torch.int32))
            buf.ple_v.fill_(0.125)
            outputs.append(forward.compute(w, [(st, 0, 3)], buf, check=False).clone())
        bits(*outputs)
        assert a.dspark.pending is None and b.dspark.pending is not None
        runtime.stage_graph(source, graph, [(a, 0, 3)], torch.arange(3))
        bits(a.dspark.pending.keys, b.dspark.pending.keys)
        bits(a.dspark.pending.values, b.dspark.pending.values)
        forward._kv_check([(a, 0, 3), (b, 3, 6)])
        forward.commit(w, a, graph, 3, 2)
        forward.commit(w, b, ref, 3, 2)
        assert_target(a, b)
        bits(a.dspark.keys[:, :2], b.dspark.keys[:, :2])
        assert graph.dspark_context.readbacks == graph.dspark_context.frames == 1
        with pytest.raises(BaseExceptionGroup):
            runtime.close()
        assert runtime.provider is not None and not runtime.closed
    finally:
        runtime.retire_graph(graph)
        try:
            next(fixture)
        except StopIteration:
            pass

"""Captured fixed32 context projections match each original real target segment."""
from types import SimpleNamespace as NS

import pytest
import torch

from test_flashnext_dspark_proposal_graphs import provider
from tensorfold.families.qwen4_exp.cuda.dspark_context_graphs import ContextWorkspace

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="native DSpark context projection")


def exact(a, b):
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def state(p):
    target = NS(pos=0, capacity=64, version=0, kv_identity=("int8", "int8"), cur=[0, 0],
                _kv_error=None, _resize_error=None, _kv_pending=False,
                pos_dev=torch.zeros(1, dtype=torch.int32, device="cuda"))
    return p.make_state(target, capacity=64)


@pytest.mark.parametrize("counts", ((1,), (3,), (1, 2, 3, 4, 6, 8)))
def test_native_captured_projection_is_per_segment_original_exact_with_padding_and_dynamic_positions(counts):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 context projection scope")
    p, width = provider(), 8
    states = [state(p) for _ in counts]
    rows = len(states)*width
    ws = ContextWorkspace(p, rows, "cuda")
    ws.allocate()
    assert ws.nbytes == ws.device_bytes(p.config, rows, p.local["kv_heads"])
    gen = torch.Generator(device="cuda").manual_seed(4001)
    h = torch.randn(rows, len(p.config.taps)*p.config.hidden, device="cuda", generator=gen).half()*0.01
    physical = [(st.target, i*width, (i+1)*width) for i, st in enumerate(states)]
    segs, mapping, end = [], [], 0
    for i, (st, count) in enumerate(zip(states, counts)):
        segs.append((st.target, end, end+count))
        mapping.extend(range(i*width, i*width+count))
        h[i*width+count:(i+1)*width].fill_(float("nan"))
        end += count
    index = torch.tensor(mapping, dtype=torch.int64, device="cuda")
    ws.compute(h, physical)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ws.compute(h, physical)
    try:
        for iteration in range(2):
            h.mul_(0.5)
            expected = [p._context(st, h[i*width:i*width+count])
                        for i, (st, count) in enumerate(zip(states, counts))]
            ws.prepare_replay()
            graph.replay()
            updates = ws.stage(states, segs, index)
            assert all(st.length == st.target.pos and st.pending is update for st, update in zip(states, updates))
            for st, update, reference in zip(states, updates, expected):
                exact(update.keys, reference.keys)
                exact(update.values, reference.values)
                assert update.keys.is_contiguous() and update.values.is_contiguous()
                assert not torch._C._overlaps(update.keys, ws.keys)
                assert not torch._C._overlaps(update.values, ws.values)
            ws.keys.fill_(7)
            ws.values.fill_(8)
            for st, update, reference in zip(states, updates, expected):
                keep = min(2, update.rows)
                start = st.length
                st.target.pos += keep
                st.target.pos_dev.fill_(st.target.pos)
                p.commit(st, update, keep)
                exact(st.keys[:, start:st.length], reference.keys[:, :keep])
                exact(st.values[:, start:st.length], reference.values[:, :keep])
        assert ws.frames == ws.readbacks == 2 and ws.exported_rows == 2*sum(counts)
        print({"physical_rows": rows, "real_rows": sum(counts), "layers": p.config.layers,
               "context_device_bytes": ws.nbytes, "readbacks": ws.readbacks}, flush=True)
    finally:
        graph.reset()
        ws.retire_graph()
        ws.close()
        for st in states:
            p.close_state(st)


def test_native_nonfinite_context_publishes_no_update_then_same_capture_recovers():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 context projection scope")
    p, st = provider(), None
    st = state(p)
    ws = ContextWorkspace(p, 3, "cuda")
    ws.allocate()
    h = torch.zeros(3, 2560, dtype=torch.float16, device="cuda")
    segs, index = [(st.target, 0, 3)], torch.arange(3, device="cuda")
    ws.compute(h, segs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ws.compute(h, segs)
    try:
        h[1, 0].fill_(float("nan"))
        graph.replay()
        with pytest.raises(ValueError, match="nonfinite captured context"):
            ws.stage([st], segs, index)
        assert st.pending is None and st.length == st.target.pos == 0
        h.zero_()
        ws.prepare_replay()
        graph.replay()
        update, = ws.stage([st], segs, index)
        assert update.rows == 3 and st.length == 0
        p.discard(st)
    finally:
        graph.reset()
        ws.retire_graph()
        ws.close()
        p.close_state(st)


@pytest.mark.parametrize("cache", ("fp16", "int8"))
def test_bound_context_owner_matches_original_state_and_context_then_retires_graph_borrowers(cache, monkeypatch):
    from test_flashnext_dspark_target_graphs import (
        test_six_lane_target_bound_replay_matches_eager_context_and_partial_commit_with_auxiliary as qualify,
    )
    from tensorfold.cuda.sm70 import family_ops
    from tensorfold.families.qwen4_exp.cuda import dspark_context_graphs as module
    from tensorfold.families.qwen4_exp.cuda.dspark_runtime import DSparkRuntime

    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
    # Keep this short-context ownership fixture on the original Torch operator.
    # Qualified native attention (65..8192) has its independent exact oracle.
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", False)
    monkeypatch.setattr(module, "_ENABLED", True)
    prepared, exports = [], []
    prepare, stage = DSparkRuntime.prepare_graph_context, DSparkRuntime.stage_graph

    def tracked_prepare(self, buffers, bound):
        result = prepare(self, buffers, bound)
        assert result and buffers.dspark_context.nbytes > 0
        prepared.append(buffers.dspark_context)
        return result

    def projected_stage(self, source, bound, segs, index):
        assert bound.dspark_context is not None
        before = bound.dspark_context.frames
        old = self.provider._context

        def forbidden(*args, **kwargs):
            raise AssertionError("captured context must not reproject eagerly after replay")

        self.provider._context = forbidden
        try:
            result = stage(self, source, bound, segs, index)
        finally:
            self.provider._context = old
        assert bound.dspark_context.frames == before + 1
        exports.append(sum(a1-a0 for _, a0, a1 in segs))
        return result

    monkeypatch.setattr(DSparkRuntime, "prepare_graph_context", tracked_prepare)
    monkeypatch.setattr(DSparkRuntime, "stage_graph", projected_stage)
    qualify(cache)
    assert len(prepared) == 1 and len(exports) >= 8
    assert all(ws.closed and ws.nbytes == 0 and not ws.graph_active for ws in prepared)
    print({"cache": cache, "captured_context_frames": len(exports), "real_rows": sum(exports),
           "one_consensus_per_frame": prepared[0].readbacks == prepared[0].frames,
           "retired_owned_context_bytes": prepared[0].nbytes}, flush=True)

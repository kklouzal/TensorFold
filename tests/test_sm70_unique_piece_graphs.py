"""Exact N1 capture geometry and the eager unique-history ownership cuts."""

from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda import bound_graphs, multi_graphs  # noqa: E402 - Torch is optional.


def planner():
    graph = object.__new__(bound_graphs.BoundStateGraphs)
    graph.unique = True
    graph.native_owner = False
    graph.dec = NS(depth=2, w=NS())
    states = [NS(pos=17, mtp_len=13, logical_context=262144, capacity=262147,
                 kv_identity=("int8",), version=0, cur=[0]) for _ in range(2)]
    graph.states, graph.lanes = tuple(states), {}
    graph.indices = {id(state): index for index, state in enumerate(states)}
    graph._admit_lanes = lambda _ordered: True
    graph._lane = lambda state: NS(index=graph.indices[id(state)], identity=(state.kv_identity, state.version))
    return graph, states


def test_unique_capture_requires_its_own_explicit_contract_and_provider_readiness():
    weights = NS(meta={"history_layout": "unique", "execution_provider": "sm70"}, comm=None,
                 layers=[], mtp=None)
    assert "unique history" in multi_graphs.capturable(weights)
    weights.meta["unique_capture_contract"] = multi_graphs.UNIQUE_CAPTURE_CONTRACT
    assert "capture-safe" in multi_graphs.capturable(weights)
    weights.meta["capture_ready"] = True
    assert multi_graphs.capturable(weights) is None
    weights.meta["unique_capture_contract"] = "unqualified-successor"
    assert "unique history" in multi_graphs.capturable(weights)


@pytest.mark.parametrize("kind", ["verify", "mtp"])
def test_exact_live_widths_reuse_only_their_own_state_and_never_borrow_legacy_attention_scope(kind, monkeypatch):
    graph, (state, other) = planner()
    from tensorfold.cuda.sm70 import attention_qualification

    monkeypatch.setattr(attention_qualification, "allows", lambda *_a: pytest.fail("legacy candidate queried"))
    keys = []
    for rows in (1, 2, 3):
        plan = graph._plan(kind, [(state, 0, rows)])
        assert plan[0][1:] == (rows, None) and plan[4:] == ((rows,), list(range(rows)))
        keys.append(plan[0])
        state.pos, state.mtp_len = 120000, 119997
        assert graph._plan(kind, [(state, 0, rows)])[0] == plan[0]
        assert graph._plan(kind, [(other, 0, rows)])[0] != plan[0]
    assert len(set(keys)) == 3


def test_unique_capture_declines_multiple_participants_and_unqualified_device_proposals():
    graph, (state, other) = planner()
    assert graph._plan("verify", [(state, 0, 1), (other, 1, 2)]) is None
    assert graph._plan("verify", [(state, 0, 4)]) is None
    assert graph._plan("mtp", [(state, 0, 1)], mtp_positions=[14]) is None
    assert graph._plan("mtp", [(state, 0, 1)], device_tokens=torch.tensor([1])) is None
    assert not graph.mtp_ready(state, 1, 13)


def test_same_marker_cannot_capture_an_unqualified_fourth_row_at_greater_depth():
    graph, (state, _) = planner()
    graph.dec.depth = 3
    assert graph._plan("verify", [(state, 0, 3)]) is not None
    assert graph._plan("verify", [(state, 0, 4)]) is None
    assert graph._plan("mtp", [(state, 0, 4)]) is None


def test_physical_guard_rows_do_not_extend_the_unique_capture_logical_domain():
    graph, (state, _) = planner()
    state.pos = 262143
    assert graph._plan("verify", [(state, 0, 1)]) is not None
    assert graph._plan("verify", [(state, 0, 2)]) is None
    assert state.capacity == 262147 and state.logical_context == 262144


def test_exact_lane_keeps_live_rows_unpadded_and_preserves_recurrent_ownership():
    state = NS(rec=torch.arange(16, dtype=torch.float32).reshape(2, 1, 2, 4), cur=[1],
               scratch=[NS()], mtp_pos=torch.tensor([13]), mtp_len=13, kv_identity=("int8",), version=0)
    lane = bound_graphs.BoundLane(state, 0, exact_rows=True)
    view = lane.refresh(2)
    assert not hasattr(view, "graph_rows") and view.history_rows == 2
    assert view.rec is lane.recurrent and view.scratch[0].keep_rows.item() == 2
    assert torch.equal(view.rec[0], state.rec[1])
    unchanged = state.rec[1].clone()
    view.rec[1].fill_(23)
    lane.publish()
    assert torch.equal(state.rec[1], unchanged) and bool((state.rec[0] == 23).all())
    state.version += 1
    with pytest.raises(ValueError, match="original cache storage"):
        lane.refresh(2)


def test_replay_runs_unique_attention_between_the_graphs_with_this_frames_exact_segments(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import history_owner

    events, layer = [], object()
    segs = [(object(), 0, 2)]
    buffers, weights = NS(), NS()
    owner = object.__new__(multi_graphs.RoundGraphs)
    owner.dec = NS(w=weights)
    pieces = NS(graphs=["before", "after"], inserts=[("owner_attention", layer)])
    monkeypatch.setattr(multi_graphs, "profile_replay", lambda graph, _comm: events.append(graph))
    weights.comm = None

    def attention(actual_layer, actual_w, actual_segs, actual_b, rows, *, mtp):
        assert (actual_layer, actual_w, actual_segs, actual_b, rows, mtp) == (
            layer, weights, segs, buffers, 2, True)
        events.append("live-attention")

    monkeypatch.setattr(history_owner, "attention_layer", attention)
    owner._replay_pieces(pieces, segs, buffers, mtp=True)
    assert events == ["before", "live-attention", "after"]


def test_failed_kernel_warmup_restores_the_capture_callback_without_running_attention(monkeypatch):
    from contextlib import nullcontext

    graph = object.__new__(bound_graphs.BoundStateGraphs)
    graph.unique = True
    graph.owner = NS(room_bytes=0, budget=1, _has_room=lambda: True, side=NS(wait_stream=lambda _stream: None))
    graph.dec = NS(w=NS(device="cpu"))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: object())
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    graph._agree = bool
    buffers = NS(attn_o=torch.ones(2, 3), split=None)
    primary = RuntimeError("kernel preparation failed")

    def run():
        assert buffers.split is not None and not bool(buffers.attn_o.any())
        buffers.split(("owner_attention", object()))
        raise primary

    with pytest.raises(RuntimeError) as failure:
        graph._capture("verify", None, [], buffers, [], run)
    assert failure.value is primary and buffers.split is None


@pytest.mark.parametrize("depth", [2, 3])
def test_unique_warmup_validates_and_retires_every_temporary_raw_span_before_reuse(monkeypatch, depth):
    events = []

    class State:
        pending = False

        def reset(self, _weights, *, new_sequence):
            assert not self.pending and new_sequence is False
            events.append("reset")

    state = State()
    graph = object.__new__(bound_graphs.BoundStateGraphs)
    graph.unique = True
    graph.dec = NS(depth=depth, w=NS(dspark=None, comm=None),
                   buf=NS(rows=6, streams=torch.zeros(6, 4)), mbuf=NS())

    def stage(_weights, _buffers, windows, **_kwargs):
        assert len(windows) == 1
        return [(windows[0][0], 0, len(windows[0][1]))]

    def execute(kind, segments):
        assert not state.pending
        state.pending = True
        events.append((kind, segments[0][2]))

    def check(_segments, **_kwargs):
        assert state.pending
        state.pending = False
        events.append("checked")

    monkeypatch.setattr(bound_graphs, "stage", stage)
    monkeypatch.setattr(bound_graphs, "mtp_stage", lambda buffers, windows: stage(None, buffers, windows))
    monkeypatch.setattr(bound_graphs, "_kv_check", check)
    graph._prepare_greedy = lambda: pytest.fail("ordinary proposal protocol changed")
    graph.forward = execute
    graph.warm([State(), state])
    expected = []
    for rows in (1, 2, 3):
        for kind in ("verify", "mtp"):
            expected.extend([(kind, rows), "checked", "reset"])
    assert events == expected and not state.pending


@pytest.mark.parametrize("unique,rows,segments", [(True, 3, 1), (False, 32, 26)])
@pytest.mark.parametrize("kind", ["verify", "mtp"])
def test_private_buffer_extent_reuse_and_retirement_preserve_original_full_window_owners(
        monkeypatch, unique, rows, segments, kind):
    import gc
    import weakref
    from tensorfold.cuda.sm70 import family_ops

    graph = object.__new__(bound_graphs.BoundStateGraphs)
    graph.unique, graph.closed, graph.native_owner = unique, False, False
    original = NS(rows=32, moe=NS(plan=NS(prefill=True)))
    weights = NS(device="cpu", dspark=None)
    graph.dec = NS(w=weights, buf=original, mbuf=original, capacity=262147)
    graph.states = tuple(NS(capacity=262147, logical_context=262144) for _ in range(26))
    original_states = graph.states
    graph.buffers, graph.row_maps, graph.segment_maps, graph.map_events = {}, {}, {}, {}
    graph.indices, graph.lanes, graph.bindings = {}, {}, {}
    graph._resident = lambda size: None
    allocations = []

    class Buffers:
        def __init__(self, w, actual_rows, capacity, **kwargs):
            assert w is weights
            allocations.append((actual_rows, capacity, kwargs))
            self.rows = actual_rows

    empty = torch.empty
    monkeypatch.setattr(bound_graphs, "Buffers", Buffers)
    monkeypatch.setattr(bound_graphs.torch, "empty", lambda shape, *, dtype, **_kw: empty(shape, dtype=dtype))
    monkeypatch.setattr(bound_graphs.torch.cuda, "memory_allocated", lambda _device: 0)
    recorded = []
    monkeypatch.setattr(bound_graphs.torch.cuda, "Event", lambda: NS(record=lambda: recorded.append("record")))
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", False)
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", False)
    private = graph._buffer(kind)
    assert allocations == [(rows, 262147, {"moe_prefill": True, "owner_role": "mtp" if kind == "mtp" else None})]
    assert private is graph._buffer(kind) and len(allocations) == 1
    if unique:
        assert graph.row_maps[kind][0] is graph.segment_maps[kind][0] is None
        assert torch.equal(graph.row_maps[kind][1], torch.arange(rows))
        assert torch.equal(graph.segment_maps[kind][1], torch.zeros(segments, dtype=torch.int64))
        assert recorded == ["record"] and graph.map_events[kind][1]
    else:
        assert all(mapping.numel() == rows for mapping in graph.row_maps[kind])
        assert all(mapping.numel() == segments for mapping in graph.segment_maps[kind])
        assert recorded == [] and not graph.map_events[kind][1]
    assert original.rows == 32 and graph.states is original_states and len(graph.states) == 26
    assert all(state.capacity == 262147 and state.logical_context == 262144 for state in graph.states)
    retained = weakref.ref(private)
    del private
    gc.collect()
    assert retained() is not None
    graph.close()
    gc.collect()
    assert retained() is None and graph.closed
    assert graph.states is original_states and original.rows == 32


@pytest.mark.parametrize("kind", ["verify", "mtp"])
def test_unique_static_maps_match_actual_plans_across_widths_and_state_slot_reuse(monkeypatch, kind):
    graph, states = planner()
    row_device, segment_device = torch.arange(3), torch.zeros(1, dtype=torch.int64)
    event = NS(synchronize=lambda: pytest.fail("immutable maps must not block on the CPU"))
    graph.row_maps = {kind: (None, row_device)}
    graph.segment_maps = {kind: (None, segment_device)}
    graph.map_events = {kind: [event, True]}
    waits = []
    monkeypatch.setattr(bound_graphs.torch.cuda, "current_stream",
        lambda device: NS(wait_event=lambda joined: waits.append((device, joined))))
    row_pointer, segment_pointer = row_device.data_ptr(), segment_device.data_ptr()
    # A one-lane canonical segment remains segment0 even for a different decoder
    # slot; exact row widths retain identity, with no old-slot or upload state.
    for repetition in range(2):
        for state in states:
            state.pos, state.mtp_len = 100000 + repetition, 99997 + repetition
            for rows in (1, 2, 3):
                plan = graph._plan(kind, [(state, 0, rows)])
                assert plan is not None and plan[4] == (rows,)
                inverse = plan[5]
                order = [graph.indices[id(state)]]
                segment_indices = [sorted(order).index(index) for index in order]
                assert inverse == list(range(rows)) and segment_indices == [0]
                actual_rows, actual_segments = graph._maps(kind, inverse, segment_indices)
                assert actual_rows.tolist() == list(range(rows))
                assert actual_segments.tolist() == [0]
                assert actual_rows.data_ptr() == row_pointer
                assert actual_segments.data_ptr() == segment_pointer
    assert len(waits) == 12 and all(device == row_device.device and joined is event for device, joined in waits)
    assert row_device.tolist() == [0, 1, 2] and segment_device.tolist() == [0]

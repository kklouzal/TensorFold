"""Bound graph geometry/ownership contracts that do not require CUDA."""

from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.cuda.bound_graphs import BoundStateGraphs
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
from tensorfold.families.qwen4_exp.cuda.service_memory import release_uncaptured_graph_plan
from tensorfold.cuda.sm70 import attention_qualification


def planner():
    graph = object.__new__(BoundStateGraphs)
    # This geometry-only oracle uses the ordinary reference provider and
    # substitutes its allocation work below; it does not claim SM70 admission.
    graph.dec = SimpleNamespace(depth=2, w=SimpleNamespace(
        cfg=SimpleNamespace(execution_dtype="bfloat16")))
    states = [SimpleNamespace(pos=9, mtp_len=7, cur=[index % 2]) for index in range(3)]
    graph.indices = {id(state): index for index, state in enumerate(states)}
    graph._admit_lanes = lambda ordered: True  # geometry oracle substitutes all allocation work below
    graph._lane = lambda state: SimpleNamespace(index=graph.indices[id(state)], identity=(("fp16",), 0))
    return graph, states


def test_padded_graph_key_ignores_request_order_live_counts_and_physical_parity():
    graph, (a, b, c) = planner()
    first = graph._plan("verify", [(c, 0, 1), (a, 1, 4), (b, 4, 6)])
    b.cur = [1 - b.cur[0]]
    second = graph._plan("verify", [(b, 0, 1), (c, 1, 3), (a, 3, 6)])
    assert first[0] == second[0]
    assert first[4] == (3, 2, 1)
    assert first[5] == [6, 0, 1, 2, 3, 4]
    assert second[5] == [3, 6, 7, 0, 1, 2]


def test_chained_mtp_captures_exact_width_one_and_absorbs_keep_the_speculative_width():
    graph, (a, b, c) = planner()
    segments = [(c, 0, 1), (a, 1, 2), (b, 2, 3)]
    chained = graph._plan("mtp", segments)
    assert chained[0][1] == 1 and chained[5] == [2, 0, 1]
    assert graph._plan("verify", segments)[0][1] == 3
    absorb = graph._plan("mtp", [(c, 0, 1), (a, 1, 3), (b, 3, 4)])
    assert absorb[0][1] == 3 and absorb[5] == [6, 0, 1, 3]


def qualified_planner(monkeypatch):
    graph, states = planner()
    graph.dec.w = SimpleNamespace(device="cuda:0", cfg=SimpleNamespace(
        execution_dtype="float16", heads=3, kv_heads=1, head_dim=256))
    graph.states, graph.lanes = tuple(states), {}
    for state in states:
        state.kv_identity, state.version, state.capacity = ("fp16",), 0, 20000
        state.pos = state.mtp_len = 128
    monkeypatch.setattr(attention_qualification, "active", lambda: True)
    monkeypatch.setattr(attention_qualification, "runtime_identity", lambda device:
                        dict(attention_qualification.REQUIRED_IDENTITY))
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 0))
    return graph, states


@pytest.mark.parametrize("position,heads", [(20, 3), (8192, 3), (128, 6)])
def test_unqualified_plan_declines_before_lane_admission_or_allocation(monkeypatch, position, heads):
    graph, states = qualified_planner(monkeypatch)
    state = states[0]
    state.pos = position
    graph.dec.w.cfg.heads = heads
    before = dict(vars(state))

    def forbidden(*args):
        pytest.fail("unqualified plan reached lane admission/allocation")

    graph._admit_lanes = graph._lane = forbidden
    assert graph._plan("verify", [(state, 0, 1)]) is None
    assert vars(state) == before and graph.lanes == {}


def test_unqualified_plan_keeps_invalid_owner_error_before_fallback(monkeypatch):
    graph, _ = qualified_planner(monkeypatch)
    foreign = SimpleNamespace(pos=20, mtp_len=20)
    with pytest.raises(ValueError, match="original State slots"):
        graph._plan("verify", [(foreign, 0, 1)])


def test_cached_mtp_key_does_not_override_real_or_provisional_scope(monkeypatch):
    graph, states = qualified_planner(monkeypatch)
    state = states[0]
    index = graph.indices[id(state)]
    lane = SimpleNamespace(index=index, state=state, identity=(state.kv_identity, state.version),
                           mtp_position=object())
    graph.lanes[index] = lane
    graph.closed = False
    keys = {(((index, lane.identity),), 1, context): object() for context in (256, 16384)}
    graph.owner = SimpleNamespace(failure=None, mtp=keys)
    assert graph.mtp_ready(state, 1, 128)             # Qualified cached-key control.
    state.mtp_len = 20
    assert not graph.mtp_ready(state, 1, state.mtp_len)
    state.mtp_len = 128
    assert not graph.mtp_ready(state, 1, 20)          # Provisional position wins over committed length.
    assert not graph.mtp_ready(state, 1, 8192)        # Even a matching cached large bucket declines.
    graph.dec.w.cfg.heads = 6
    assert not graph.mtp_ready(state, 1, 128)


@pytest.mark.parametrize("rows", [[(1, 2)], [(0, 4)], [(0, 1), (2, 3)], []])
def test_invalid_segment_layout_refuses_before_capture_or_device_work(rows):
    graph, states = planner()
    with pytest.raises(ValueError, match="rows|live"):
        graph._plan("verify", [(states[index], a0, a1) for index, (a0, a1) in enumerate(rows)])


def engine_with_graph(graph, events):
    engine = object.__new__(FlashNextEngine)
    engine.e = SimpleNamespace(graphs=graph, buf=None, mbuf=None)
    engine.comm = SimpleNamespace(close=lambda **kwargs: events.append("comm"))
    engine.w = engine.multi = engine.serial = None
    return engine


def test_native_graph_borrowers_retire_before_their_communicator():
    events = []
    graph = SimpleNamespace(close=lambda: events.append("graphs"))
    engine_with_graph(graph, events)._close_resources()
    assert events == ["graphs", "comm"]


def test_failed_graph_retirement_retains_the_native_communicator():
    events = []

    def fail():
        raise RuntimeError("graph borrower still alive")

    engine = engine_with_graph(SimpleNamespace(close=fail), events)
    owner = engine.comm
    with pytest.raises(RuntimeError, match="borrower"):
        engine._close_resources()
    assert events == [] and engine.comm is owner


def floor_decoder(header):
    class Slot:
        capacity, limit = 256, 20000

        def cache_bytes(self, capacity=None):
            return 10 * (self.capacity if capacity is None else capacity)

    dec = object.__new__(MultiDecoder)
    dec.capacity, dec.free = 20000, [Slot() for _ in range(6)]
    dec.graph_header_room, dec.graph_budget = header, 123
    dec.memory_gate = SimpleNamespace(held=0)

    def grow(state, rows):
        dec.memory_gate.held += state.cache_bytes(rows) - state.cache_bytes()
        state.capacity = rows
        return True

    dec._grow = grow
    return dec


def test_bound_floors_charge_all_slots_but_header_counts_one_full_window_once():
    delta = (8192 - 256) * 10
    dec = floor_decoder(5 * delta + 1000)
    assert dec._prepare_graph_floor()
    assert dec.graph_floor == 8192 and all(state.capacity == 8192 for state in dec.free)
    assert dec.memory_gate.held == 6 * delta
    assert dec.graph_floor_extra_bytes == 5 * delta and dec.graph_header_room == 1000


def test_bound_floor_refuses_insufficient_header_before_any_growth():
    dec = floor_decoder(1)
    assert not dec._prepare_graph_floor()
    assert dec.memory_gate.held == 0 and all(state.capacity == 256 for state in dec.free)
    assert dec.graph_budget == 0 and "header" in dec.graph_record["reason"]


def test_only_uncaptured_reserved_bytes_can_be_released_without_startup_credit():
    plan = {"cache_workspace_bytes_estimate": 50, "serving_peak_bytes_estimate": 100,
            "startup_peak_bytes_estimate": 90, "mapped_table_bytes": 12}
    release_uncaptured_graph_plan(plan, 30)
    assert plan["cache_workspace_bytes_estimate"] == 20
    assert plan["serving_peak_bytes_estimate"] == 70 and plan["total_bytes_estimate"] == 90
    assert plan["full_mapped_working_set_bytes_estimate"] == 102
    before = dict(plan)
    with pytest.raises(RuntimeError):
        release_uncaptured_graph_plan(plan, 21)
    assert plan == before

"""Equal resident target/draft context promises, independent CPU byte oracles."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda.capacity import Weights, make_plan
from tensorfold.cuda.geometry import indexed_stream_geometry
from tensorfold.cuda.memory_gate import MemoryGate
from tensorfold.families.qwen4_exp.ram_experts import check_kv_full_windows
from tensorfold.families.qwen4_exp.cuda.service_memory import (
    automatic_graph_bytes,
    cache_resize_overlap,
    context_reservation,
    future_service,
)
from tests.test_flashnext_graph_budget_host import decoder, source_class

GIB = 1 << 30
TEXT = dict(
    hidden_size=2560,
    num_attention_heads=24,
    num_key_value_heads=8,
    head_dim=256,
    num_hidden_layers=4,
    layer_types=["linear_attention", "full_attention"] * 2,
    linear_num_key_heads=16,
    linear_num_value_heads=48,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    vocab_size=248320,
    hc_count=4,
    num_experts_per_tok=10,
    moe_intermediate_size=640,
    indexer_head_dim=128,
    indexer_compress_ratio=4,
)


@pytest.mark.parametrize("streams", list(range(2, 9)))
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_resident_validation_reuses_existing_count_and_topology_contract(streams, tp):
    check_kv_full_windows(streams, streams=streams, vram_experts=None, tp=tp, resident_gguf=True)
    with pytest.raises(ValueError, match="requires"):
        check_kv_full_windows(streams, streams=streams, vram_experts=None, tp=tp)


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "2"])
def test_resident_invalid_counts_are_never_coerced(value):
    with pytest.raises(ValueError, match="positive whole"):
        check_kv_full_windows(value, streams=6, vram_experts=None, tp=8, resident_gguf=True)


@pytest.mark.parametrize("windows", [1, 2, 4, 6, 8, 99])
@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("pair", [((8, 32, 2), (8, 32, 2)), ((16, 32, 2), (8, 32, 2)), ((16, 32, 2), (16, 32, 2))])
def test_full_geometry_charges_each_window_initial_floor_resize_and_workspace(windows, mtp, pair):
    streams = 8
    count = 2 + int(mtp)
    actual = min(streams, windows)
    args = dict(streams=streams, each=4, keep=8, mtp=mtp, world=8, prefill_rows=64, kv_pair=pair)
    old = indexed_stream_geometry(TEXT, **args)
    new = indexed_stream_geometry(TEXT, **args, full_windows=windows, workspace_bytes=2 * GIB)

    # Independent KV row oracle: FP16 has one two-byte dummy scale, INT8
    # scales one half per32 coefficients. Per-slot index and pool stay separate.
    def side(bits):
        return 256 * bits // 8 + (256 // 32 * 2 if bits == 8 else 0)

    row = sum(side(bits) for bits, _, _ in pair)
    dummies = sum(2 for bits, _, _ in pair if bits == 16)

    def layer(n):
        return n * row + dummies + (n + (n + 3) // 4) * 128 * 2

    for capacity in (128, 256, 8193, 237602):
        expected = (actual - 1) * count * (layer(capacity) - layer(min(256, capacity))) + layer(capacity) + 2 * GIB
        assert new.bytes_at(capacity) - old.bytes_at(capacity) == expected
        assert new.reserve == old.reserve == 4
        status = indexed_stream_geometry(TEXT, **args, full_windows=windows, workspace_bytes=2 * GIB, kv_status=True)
        assert status.bytes_at(capacity) - new.bytes_at(capacity) == 8 * streams


class Array:
    def numel(self):
        return 8

    def element_size(self):
        return 4

    def __getitem__(self, index):
        return self


def slot(capacity=256, limit=8196, aux=0):
    return NS(
        capacity=capacity,
        limit=limit,
        rec=Array(),
        conv=Array(),
        ple_tail=Array(),
        cache_bytes=lambda rows=None: (capacity if rows is None else rows) * (100 + aux),
        layer_bytes=lambda rows: rows * 10,
        dspark_runtime=NS(plan=NS(cache_row_bytes=aux)) if aux else None,
    )


@pytest.mark.parametrize("slots", list(range(2, 9)))
@pytest.mark.parametrize("aux", [0, 2560])
def test_all_slots_fit_full_growth_and_one_resize_without_borrowing_workspace(slots, aux):
    states = [slot(aux=aux) for _ in range(slots)]
    p = context_reservation(states, slots)
    assert p["full_windows_reserved"] == slots
    assert p["growth_bytes"] == slots * (8196 - 256) * (100 + aux)
    assert p["copy_overlap_bytes"] == 8196 * max(10, aux)
    room = p["growth_bytes"] + p["copy_overlap_bytes"] + 2 * GIB
    gate = MemoryGate(room, reserve=2 * GIB, live=lambda: room, allow_lone_growth=False)
    each = (8196 - 256) * (100 + aux)
    for st in states:
        assert gate.fits(each + cache_resize_overlap(st, st.limit, auxiliary=True))
        gate.take(each)
    assert gate.held == p["growth_bytes"]
    assert gate.room - gate.held >= gate.reserve + p["copy_overlap_bytes"]
    assert not gate.fits(p["copy_overlap_bytes"] + 1)


def test_reservation_uses_largest_windows_without_mutating_state_or_precision():
    states = [slot(limit=n, aux=100) for n in (8192, 32768, 16384, 8192)]
    old = context_reservation(states)
    shared = context_reservation(states, 2)
    assert old["growth_bytes"] == (32768 - 256) * 200
    assert shared["growth_bytes"] == (32768 + 16384 - 512) * 200
    assert old["copy_overlap_bytes"] == 32768 * 10
    assert shared["copy_overlap_bytes"] == 32768 * 100
    assert [s.capacity for s in states] == [256] * 4


def test_future_graph_allowance_protects_target_auxiliary_snapshots_and_workspace():
    states = [slot(aux=2560) for _ in range(6)]
    p = context_reservation(states, 6)
    w = NS(cfg=NS(streams=4, hidden=2560), head=NS(n=31040), mtp=None)
    future = future_service(
        w,
        states,
        streams=6,
        retained=20,
        growth_bytes=p["growth_bytes"],
        copy_overlap_bytes=p["copy_overlap_bytes"],
        minimum_workspace_bytes=2 * GIB,
    )
    assert future["snapshot_bytes_each"] == 3 * 8 * 4  # DSpark retained snapshots are prefix markers.
    assert future["future_bytes"] == p["growth_bytes"] + p["copy_overlap_bytes"] + 20 * 96 + 2 * GIB
    assert automatic_graph_bytes(future["future_bytes"] + 123456, future["future_bytes"]) == 123456
    assert automatic_graph_bytes(future["future_bytes"] - 1, future["future_bytes"]) == 0


def test_graph_publication_cannot_subtract_its_pool_from_an_already_bounded_promise():
    dec, live, _ = decoder(live_bytes=9000, header=1000)
    dec.full_windows = 2
    dec.memory_gate.room = 3000
    dec.capture_graphs()
    assert dec.memory_gate.room == 3000
    assert dec.graph_record["future_service"]["growth_bytes"] == 1350
    assert dec.graph_record["future_service"]["copy_overlap_bytes"] == 200


def test_graph_publication_refuses_a_pool_that_consumed_promised_windows():
    dec, live, _ = decoder(live_bytes=9000, header=1000)
    dec.full_windows = 2
    dec.memory_gate.room = 8900
    with pytest.raises(RuntimeError, match="promised full KV"):
        dec.capture_graphs()
    assert dec.graphs is None and dec.memory_gate.room == 8900
    assert dec._memory_error is not None


@pytest.mark.parametrize("slots", list(range(2, 9)))
def test_full_context_capacity_is_a_real_ceiling_not_one_large_window(slots):
    old = indexed_stream_geometry(TEXT, slots, 4, 8, mtp=True, world=8, prefill_rows=64)
    full = indexed_stream_geometry(
        TEXT, slots, 4, 8, mtp=True, world=8, prefill_rows=64, full_windows=slots, workspace_bytes=2 * GIB
    )
    weights = Weights(13 * GIB, 32 << 20)
    budget = weights.resident + full.needed(32768)
    p = make_plan(262144, None, False, budget, weights, full)
    assert weights.resident + full.needed(p.fitting) <= budget
    if p.fitting < 262144:
        assert weights.resident + full.needed(p.fitting + 1) > budget
    assert p.fitting < make_plan(262144, None, False, budget, weights, old).fitting


def test_engine_contract_uses_reservation_for_both_target_and_draft_before_loading():
    root = Path(os.environ.get("TENSORFOLD_PRIVATE_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
    tree = ast.parse((root / "src/tensorfold/families/qwen4_exp/cuda/engine.py").read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    geometry = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "indexed_stream_geometry")
    assert {k.arg for k in geometry.keywords} >= {"full_windows", "workspace_bytes"}
    text = (root / "src/tensorfold/families/qwen4_exp/cuda/engine.py").read_text()
    assert "total * dspark_cache_copies" in text
    assert "streams - dspark_cache_copies" in text
    assert "max(0, total * dspark_plan.cache_row_bytes - target_copy)" in text
    assert 'getattr(self, "kv_full_windows", None)' in text


@pytest.mark.parametrize("value", [-1, True, 1.0, "1"])
def test_copy_overlap_cannot_reduce_or_coerce_the_promised_service(value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        future_service(NS(), [], streams=6, retained=20, growth_bytes=0, copy_overlap_bytes=value)


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "2"])
def test_geometry_rejects_invalid_full_windows_before_sizing(value):
    with pytest.raises(ValueError, match="positive integer"):
        indexed_stream_geometry(TEXT, 6, 4, 8, mtp=True, world=8, full_windows=value)


@pytest.mark.parametrize("slots", range(2, 9))
@pytest.mark.parametrize("selected", [None, 1, 2, 4, 8])
def test_graph_cache_floors_do_not_double_charge_selected_full_windows(slots, selected):
    cls = source_class("multi.py", "MultiDecoder", {"_prepare_graph_floor"}, {"FIRST": 256, "NoRoom": RuntimeError})

    class Cache:
        def __init__(self):
            self.capacity = 256

        def cache_bytes(self, rows=None):
            return (self.capacity if rows is None else rows) * 100

    dec = cls()
    dec.free = [Cache() for _ in range(slots)]
    dec.full_windows = selected
    dec.capacity = 32768
    dec.graph_header_room = 10**10
    dec.graph_budget = None
    dec.graph_floor_extra_bytes = 0

    def grow(state, rows):
        state.capacity = rows
        return True

    dec._grow = grow
    assert dec._prepare_graph_floor()
    expected = (slots - min(slots, selected or 1)) * (8192 - 256) * 100
    assert dec.graph_floor_extra_bytes == expected
    assert dec.graph_header_room == 10**10 - expected
    assert all(st.capacity == 8192 for st in dec.free)


def test_all_full_windows_can_prepare_floors_even_with_zero_extra_graph_header_room():
    cls = source_class("multi.py", "MultiDecoder", {"_prepare_graph_floor"}, {"FIRST": 256, "NoRoom": RuntimeError})
    states = [NS(cache_bytes=lambda rows=None: 100 * (rows if rows is not None else 256)) for _ in range(6)]
    dec = cls()
    dec.free, dec.full_windows, dec.capacity = states, 6, 262144
    dec.graph_header_room, dec.graph_budget = 0, None
    dec._grow = lambda state, rows: True
    assert dec._prepare_graph_floor()
    assert dec.graph_floor_extra_bytes == dec.graph_header_room == 0

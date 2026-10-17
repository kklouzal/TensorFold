"""Explicit shared KV budgets preserve memory bounds, progress and failure outcomes."""

from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda.memory_gate import MemoryGate, NoRoom
from tensorfold.families.qwen4_exp.ram_experts import check_kv_full_windows, reserved_growth
from tests.test_cuda_geometry import allocations  # noqa: F401 - pytest fixture
from tests.test_flashnext_absolute_grow_host import AbsoluteGrowthTests as _GrowthCase


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "2"])
def test_invalid_window_count_fails_before_checkpoint_access(value):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    with pytest.raises(ValueError, match="--kv-full-windows.*positive whole"):
        FlashNextEngine(Path("/unavailable-checkpoint"), streams=4, vram_experts="auto", kv_full_windows=value)


@pytest.mark.parametrize("options", [{"streams": 1, "vram_experts": "auto"}, {"streams": 4, "vram_experts": 2},
                                     {"streams": 4, "vram_experts": None}, {"streams": 4, "vram_experts": "auto", "tp": 2}])
def test_irrelevant_policy_is_refused_before_checkpoint_access(options):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    with pytest.raises(ValueError, match="--kv-full-windows requires"):
        FlashNextEngine(Path("/unavailable-checkpoint"), kv_full_windows=1, **options)


@pytest.mark.parametrize("windows,expected", [(None, (17, 4)), (1, (5, 1)), (2, (10, 2)), (3, (14, 3)),
                                               (4, (17, 4)), (9, (17, 4))])
def test_reservation_keeps_largest_windows_and_leaves_input_unchanged(windows, expected):
    each = [3, 5, 4, 5]
    check_kv_full_windows(windows, streams=4, vram_experts="auto")
    assert reserved_growth(each, windows) == expected
    assert each == [3, 5, 4, 5]


@pytest.fixture
def growth():
    case = _GrowthCase()
    case.setUp()
    try:
        yield case
    finally:
        case.doCleanups()


@pytest.mark.parametrize("capped", [False, True])
def test_isolated_shared_budget_never_escapes_gate_or_allocates_before_refusal(growth, monkeypatch, capped):
    from tensorfold.cuda.capacity import LIMIT_ENV

    if not capped:
        monkeypatch.delenv(LIMIT_ENV)
    dec, alloc = growth.dec, growth.alloc
    st = dec.free[0]
    before = alloc.allocated
    dec.memory_gate = MemoryGate(0, 0, live=lambda: 1 << 62, allow_lone_growth=False)
    with pytest.raises(NoRoom, match="shared KV reservation"):
        dec._grow(st, 300, alone=True)
    assert st.capacity == 256 and alloc.allocated == alloc.peak == before
    assert dec.memory_gate.held == 0
    # A complete one-window reservation admits the maximum window and its copy.
    dec.memory_gate.room = st.cache_bytes(st.limit) - st.cache_bytes() + st.layer_bytes(st.limit)
    assert dec._grow(st, st.limit, alone=True)
    assert st.capacity == st.limit and dec.memory_gate.held <= dec.memory_gate.room


def test_a_decoder_waits_for_filling_owner_then_recovers_without_gate_escape(growth, monkeypatch):
    from tensorfold.cuda.capacity import LIMIT_ENV

    filling, decoding = growth.waiting_and_decoding()
    monkeypatch.delenv(LIMIT_ENV)
    dec, alloc = growth.dec, growth.alloc
    held = dec.memory_gate.held
    dec.memory_gate = MemoryGate(held, 0, live=lambda: 1 << 62, allow_lone_growth=False)
    dec.memory_gate.take(held)
    decoding.st.pos = decoding.st.capacity - 1
    before = alloc.allocated
    assert dec._make_room() == []
    assert decoding.waiting and not decoding.done and not filling.done
    assert alloc.allocated == before and decoding.st.capacity == 24576
    dec.filling.remove(filling)
    dec.finish([filling])
    assert dec._make_room() == []
    assert not decoding.waiting and not decoding.done and decoding.st.capacity == 32768
    assert dec.memory_gate.held <= dec.memory_gate.room


def test_all_blocked_decoders_end_newest_explicitly_and_oldest_finishes_its_full_window(growth):
    from tensorfold.cuda.streams import Stream

    dec = growth.dec
    old, new = [Stream([1], 20) for _ in range(2)]
    for sid, (stream, st) in enumerate(zip((old, new), dec.free)):
        stream.sid, stream.st = sid, st
        stream.out = [7, 9]
        st.limit, st.pos, st.mtp_len = 16384, 255, 255
    probe = old.st
    room = probe.cache_bytes(probe.limit) - probe.cache_bytes() + probe.layer_bytes(probe.limit)
    dec.free = []
    dec.streams = {0: old, 1: new}
    dec.memory_gate = MemoryGate(room, 0, live=lambda: 1 << 62, allow_lone_growth=False)
    assert dec._make_room() == []
    assert old.st.capacity == new.st.capacity == 8192
    old.st.pos = old.st.mtp_len = 8191
    assert dec._make_room() == [new]
    assert new.done and isinstance(new.error, RuntimeError) and "stopped" in str(new.error)
    assert new.out == [7, 9] and new.st in dec.free
    assert old.st.capacity == old.st.limit and not old.waiting and not old.done
    assert dec.memory_gate.ends == 1 and dec.memory_gate.held <= room


@pytest.mark.parametrize("windows,selected", [(None, 4), (1, 1), (2, 2), (9, 4)])
def test_final_auto_pool_receipt_and_gate_preserve_all_other_reservations(windows, selected, capsys):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.cuda.prompt_plan import POINTS
    from tensorfold.families.qwen4_exp.ram_experts import AUTO_HEADROOM_BYTES

    gib, entry = 2**30, 2**20
    allocated = [gib]
    cache = NS(entry_bytes=entry, score_history_entries=32768, host_bytes=4 * gib)

    def configure(nbytes):
        allocated[0] += nbytes - entry
        cache.gpu_bytes = nbytes
        cache.capacity = cache.resident_capacity = nbytes // entry

    cache.configure_before_use = configure

    class Array:
        def numel(self):
            return 1024

        def element_size(self):
            return 4

        def __getitem__(self, index):
            return self

    st = NS(limit=65536, cache_bytes=lambda rows=None: entry + (gib if rows is not None else 0),
            layer_bytes=lambda rows: gib // 4, rec=Array(), conv=Array(), ple_tail=Array())
    engine = FlashNextEngine.__new__(FlashNextEngine)
    engine.kv_full_windows = windows
    engine.w = NS(meta={"expert_cache": cache}, cfg=NS(streams=4, hidden=64), mtp=object(), head=NS(n=4096), x3=None)
    engine.multi = NS(free=[st] * 4, keep=8, pbuf=NS(rows=16))
    engine.capacity_plan = {"vram_experts": {"gpu_bytes": entry, "control_device_bytes": 0},
                            "weight_bytes_estimate": gib, "startup_peak_bytes_estimate": gib,
                            "mapped_table_bytes": 0}
    runtime = NS(cuda=NS(synchronize=lambda: None, empty_cache=lambda: None,
                         mem_get_info=lambda: (16 * gib - allocated[0], 16 * gib),
                         memory_allocated=lambda: allocated[0], memory_reserved=lambda: allocated[0]))
    vision = 7 * entry
    engine._configure_auto_experts(runtime, 4, vision_workspace_bytes=vision)
    result = engine.capacity_plan["vram_experts_auto"]
    snapshots = 8 + 4 * POINTS
    snapshot_bytes = 3 * 1024 * 4 + 4 * 64 * 2
    transient = 4 * (4096 * 4 + 4 * 64 * 2) + vision
    assert result["full_context_growth_bytes"] == 4 * gib
    assert result["reserved_context_growth_bytes"] == selected * gib
    assert result["kv_full_windows_requested"] == windows and result["kv_full_windows_reserved"] == selected
    assert result["kv_stream_slots"] == 4
    assert result["layer_resize_overlap_bytes"] == gib // 4
    assert result["kept_snapshot_count"] == snapshots and result["snapshot_bytes"] == snapshots * snapshot_bytes
    assert result["unallocated_workspace_bytes"] == result["prompt_transient_bytes"] == transient
    assert result["reserved_bytes"] == AUTO_HEADROOM_BYTES + selected * gib + gib // 4 + snapshots * snapshot_bytes + transient
    assert engine.multi.memory_gate.room == selected * gib + gib // 4
    assert engine.multi.memory_gate.allow_lone_growth is (windows is None)
    assert result["policy"] == ("remaining-device-bytes-after-fixed-service-state-v1" if windows is None else
                                "remaining-device-bytes-after-fixed-service-state-shared-kv-v1")
    assert result["physical_free_bytes_after_pool"] >= result["reserved_bytes"]
    assert engine.capacity_plan["serving_peak_bytes_estimate"] == allocated[0] + result["reserved_bytes"] - AUTO_HEADROOM_BYTES
    assert ("request errors possible" in capsys.readouterr().out) is (windows is not None)


@pytest.mark.torch
@pytest.mark.usefixtures("allocations")
@pytest.mark.parametrize("failed_call", [1, 3, 5])
@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_failed_admission_retires_partial_storage_and_preserves_following_request(
        monkeypatch, failed_call, cleanup_failure):
    import importlib

    import torch

    from tensorfold.cuda.streams import Stream
    from tests.test_cuda_growing_caches import decoder, weights

    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    dec = decoder(multi, state, torch, room=1 << 62)
    dec.failed_slots = []
    dec.next_id = 0
    dec.capacity, dec.copy, dec.points, dec.vision = 65536, False, None, None
    dec._draft_map, dec.buf, dec.mbuf, dec.pbuf, dec.prefill_rows = None, None, None, None, 16
    st = state.State(weights(), 256, 4, "int4", limit=65536)
    dec.free = [st]
    before = st.cache_bytes()
    original = state._rows
    primary, cleanup = MemoryError("controlled admission resize failure"), RuntimeError("controlled retirement reset failure")
    calls = 0

    def failed_rows(array, rows, keep):
        nonlocal calls
        calls += 1
        if calls == failed_call:
            raise primary
        return original(array, rows, keep)

    with monkeypatch.context() as patch:
        patch.setattr(state, "_rows", failed_rows)
        if cleanup_failure:
            patch.setattr(st, "reset", lambda _: (_ for _ in ()).throw(cleanup))
        with pytest.raises(MemoryError) as caught:
            dec.admit(Stream([1] * 300, 8, draft=False))
    assert caught.value is primary
    assert not dec.live() and not dec.kept and not dec.fills
    assert st.cache_bytes() == before and dec.memory_gate.held == 0
    if cleanup_failure:
        assert dec.free == [] and dec.failed_slots == [st] and st._resize_error is cleanup
        assert primary.__cause__ is cleanup
        # drop() and close() retry only exclusive failed-slot ownership.
        assert dec.drop() == []
    assert dec.free == [st] and dec.failed_slots == []
    assert st._resize_error is None and st.pos == st.mtp_len == 0
    monkeypatch.setattr(multi, "_slot", lambda *args: NS())
    monkeypatch.setattr(multi, "prefill_begin", lambda *args, **kwargs: 0)
    monkeypatch.setattr(multi.image_rows, "begin", lambda *args: None)
    following = Stream([2, 3, 4], 1, draft=False)
    dec.admit(following)
    assert dec.filling == [following] and following.st is st and not following.done
    assert dec.memory_gate.held == 0

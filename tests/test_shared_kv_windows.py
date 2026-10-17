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
@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("phase", ["during", "after_success"])
def test_interrupted_grow_charges_actual_partial_or_completed_replacement(monkeypatch, failure_type, phase):
    import importlib

    import torch

    from tests.test_cuda_growing_caches import decoder, weights
    from tests.test_flashnext_resize_failure import attached_bytes

    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    dec = decoder(multi, state, torch, room=1 << 62)
    dec.failed_slots = []
    st = state.State(weights(), 256, 4, "int4", limit=65536)
    before = attached_bytes(st)
    original, resize = state._rows, st.resize
    primary = failure_type("interrupted replacement")
    calls = 0

    def failed_rows(array, rows, keep):
        nonlocal calls
        calls += 1
        if calls == 3:                           # one complete layer has already replaced its old storage
            raise primary
        return original(array, rows, keep)

    def after_success(rows):
        resize(rows)
        raise primary

    with monkeypatch.context() as patch:
        if phase == "during":
            patch.setattr(state, "_rows", failed_rows)
        else:
            patch.setattr(st, "resize", after_success)
        with pytest.raises(failure_type) as caught:
            dec._grow(st, 300)
    added = attached_bytes(st) - before
    assert caught.value is primary and added > 0
    assert st._resize_error is (primary if phase == "during" else None)
    assert st.cache_bytes() == before + added
    assert dec.memory_gate.held == added
    dec._quarantine(st)
    dec.recover_failed_slots()
    assert dec.memory_gate.held == 0 and dec.free == [st] and not dec.failed_slots


@pytest.mark.torch
@pytest.mark.usefixtures("allocations")
@pytest.mark.parametrize("resize_failure", [False, True])
@pytest.mark.parametrize("after_publication", [False, True])
@pytest.mark.parametrize("operation", ["grow", "shrink"])
def test_accounting_failure_before_or_after_publication_contains_the_decoder(
        monkeypatch, resize_failure, after_publication, operation):
    import importlib

    import torch

    from tensorfold.cuda.streams import Stream
    from tests.test_cuda_growing_caches import decoder, weights
    from tests.test_flashnext_resize_failure import attached_bytes

    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    dec = decoder(multi, state, torch, room=1 << 62)
    dec.failed_slots = []
    st = state.State(weights(), 256, 4, "int4", limit=65536)
    initial, others = attached_bytes(st), 12345
    if operation == "shrink":
        st.resize(8192)
    before = attached_bytes(st)
    primary, cleanup = KeyboardInterrupt("resize interruption"), RuntimeError("accounting publication failure")

    class PublicationFailure(MemoryGate):
        failure = None

        @property
        def held(self):
            return self._held

        @held.setter
        def held(self, value):
            if self.failure is not None:
                failure, self.failure = self.failure, None
                if after_publication:
                    self._held = value
                raise failure
            self._held = value

    gate = dec.memory_gate = PublicationFailure(1 << 62, 0)
    gate.take(others + before - initial)
    gate.failure = cleanup
    resize = st.resize

    def interrupted(rows):
        resize(rows)
        raise primary

    with monkeypatch.context() as patch:
        if resize_failure:
            patch.setattr(st, "resize", interrupted)
        with pytest.raises(KeyboardInterrupt if resize_failure else RuntimeError) as caught:
            if operation == "grow":
                dec._grow(st, 300)
            else:
                dec._shrink(st)
    after = attached_bytes(st)
    assert caught.value is (primary if resize_failure else cleanup) and after != before
    assert gate.held == others + (after if after_publication else before) - initial
    if resize_failure:
        assert primary.__cause__ is cleanup
    for operation in (lambda: dec.admit(Stream([1], 1)), dec.round):
        with pytest.raises(RuntimeError) as contained:
            operation()
        assert contained.value is cleanup
    gate.held = others + after - initial         # only repair the injected test gate for exclusive resource retirement
    dec._quarantine(st)
    dec.recover_failed_slots()
    assert gate.held == others and dec.free == [st] and not dec.failed_slots
    with pytest.raises(RuntimeError) as contained:
        dec.admit(Stream([1], 1))                # successful retirement never clears an unproven model budget
    assert contained.value is cleanup


@pytest.mark.torch
@pytest.mark.usefixtures("allocations")
@pytest.mark.parametrize("operation", ["grow", "shrink"])
@pytest.mark.parametrize("diagnostic", ["wrapper", "traceback"])
def test_traceback_held_storage_blocks_inference_until_model_retirement(monkeypatch, operation, diagnostic):
    import importlib
    import traceback
    import weakref

    import torch

    from tensorfold.cuda.streams import Stream
    from tests.test_cuda_growing_caches import decoder, weights

    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    dec = decoder(multi, state, torch, room=1 << 62)
    dec.failed_slots = []
    dec._memory_error = None
    st = state.State(weights(), 256, 4, "int4", limit=65536)
    initial = st.cache_bytes()
    if operation == "shrink":
        st.resize(8192)
        dec.memory_gate.take(st.cache_bytes() - initial)
    primary, secondary = OSError("replacement failed"), MemoryError("failure diagnostic failed")
    retained = []

    def failed(*args):
        uncommitted = torch.zeros(16)
        retained.append(weakref.ref(uncommitted))
        raise primary

    def diagnostic_failed(*args):
        raise secondary

    with monkeypatch.context() as patch:
        patch.setattr(state, "_rows", failed)
        patch.setattr(state if diagnostic == "wrapper" else traceback,
                      "KVResizeError" if diagnostic == "wrapper" else "clear_frames", diagnostic_failed)
        with pytest.raises(OSError) as caught:
            if operation == "grow":
                dec._grow(st, 300)
            else:
                dec._shrink(st)
        assert caught.value is primary
        if diagnostic == "traceback":
            assert st._resize_frames_pending is not None and retained[0]() is not None
            assert dec._memory_error is primary
            for use in (lambda: dec.admit(Stream([1], 1)), dec.round):
                with pytest.raises(OSError) as contained:
                    use()
                assert contained.value is primary
        else:
            assert st._resize_frames_pending is None and retained[0]() is None
            assert getattr(dec, "_memory_error", None) is None
    dec._quarantine(st)
    dec.recover_failed_slots()
    assert not dec.failed_slots and dec.free == [st] and dec.memory_gate.held == 0
    assert st._resize_frames_pending is None and retained[0]() is None
    assert getattr(dec, "_memory_error", None) is (primary if diagnostic == "traceback" else None)


@pytest.mark.torch
@pytest.mark.usefixtures("allocations")
@pytest.mark.parametrize("failure_type", [MemoryError, KeyboardInterrupt])
def test_failed_slot_publication_restores_quarantine_before_retry(failure_type):
    import importlib

    import torch

    from tests.test_cuda_growing_caches import decoder, weights

    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    multi = importlib.import_module("tensorfold.families.qwen4_exp.cuda.multi")
    dec = decoder(multi, state, torch, room=1 << 62)
    st = state.State(weights(), 256, 4, "int4", limit=65536)
    dec.failed_slots = [st]
    failure = failure_type("interrupted free-slot publication")

    class InterruptedPublication(list):
        def append(self, item):
            super().append(item)
            raise failure

    dec.free = InterruptedPublication()
    with pytest.raises(failure_type) as caught:
        dec.recover_failed_slots()
    assert caught.value is failure and dec.free == [] and dec.failed_slots == [st]
    dec.recover_failed_slots()
    assert dec.free == [st] and not dec.failed_slots and dec.memory_gate.held == 0


@pytest.mark.torch
@pytest.mark.usefixtures("allocations")
@pytest.mark.parametrize("failed_call", [1, 3, 5])
@pytest.mark.parametrize("cleanup_failure", [False, True])
@pytest.mark.parametrize("failure_type", [MemoryError, KeyboardInterrupt, SystemExit])
def test_failed_admission_retires_partial_storage_and_preserves_following_request(
        monkeypatch, failed_call, cleanup_failure, failure_type):
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
    primary = failure_type("controlled admission resize failure")
    cleanup = RuntimeError("controlled retirement reset failure")
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
        with pytest.raises(failure_type) as caught:
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

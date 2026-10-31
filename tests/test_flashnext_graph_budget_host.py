"""Captured-round preparation and live-resource failures without an accelerator runtime."""

import ast
import gc
import traceback
import weakref
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tensorfold.cleanup import raise_failures
from tensorfold.cuda import capacity, geometry
from tensorfold.cuda.memory_gate import MemoryGate, torch_live
from tensorfold.families.qwen4_exp.cuda.service_memory import (
    automatic_graph_bytes,
    context_reservation,
    failure_frame_values,
    future_service,
    private_pool_spare,
    reserve_graph_plan,
)

ROOT = Path(__file__).resolve().parents[1]
GIB = 1 << 30


def source_class(filename, name, methods, namespace):
    path = ROOT / "src/tensorfold/families/qwen4_exp/cuda" / filename
    parsed = ast.parse(path.read_text())
    cls = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    for method in cls.body:
        method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class Array:
    def __init__(self, count, size, first=None):
        self.count, self.size, self.first = count, size, first

    def numel(self):
        return self.count

    def element_size(self):
        return self.size

    def __getitem__(self, index):
        assert index == 0
        return self.first


def states():
    return [NS(limit=limit, rec=Array(16, 4, Array(8, 4)), conv=Array(3, 2), ple_tail=Array(5, 2),
               cache_bytes=lambda rows=None, limit=limit: 5 * (limit if rows is not None else limit // 10),
               layer_bytes=lambda rows, limit=limit: limit)
            for limit in (100, 200)]


def weights(mtp=True):
    return NS(cfg=NS(streams=2, hidden=4), head=NS(n=10), mtp=object() if mtp else None)


@pytest.mark.parametrize("value,want", [(None, None), ("", None), (" ", None), ("0", 0),
                                      ("512", 512 << 20), ("2048", 2048 << 20)])
def test_request_preserves_automatic_vs_explicit_zero_and_exact_numeric(monkeypatch, value, want):
    if value is None:
        monkeypatch.delenv(geometry.MULTI_GRAPH_ENV, raising=False)
    else:
        monkeypatch.setenv(geometry.MULTI_GRAPH_ENV, value)
    assert geometry.multi_graph_request() == want
    assert geometry.multi_graph_bytes() == (want or 0)


def test_service_formula_shares_overlapping_workspace_without_reserving_all_resident_windows():
    w, slots = weights(), states()
    each = [slot.cache_bytes(slot.limit) - slot.cache_bytes() for slot in slots]
    one = future_service(w, slots, streams=2, retained=9, growth_bytes=max(each),
                         vision_workspace_bytes=17, minimum_workspace_bytes=120)
    # One complete window:900; old/new layer overlap:200. Each snapshot
    # contains8FP32+3BF16+5BF16+8BF16 values=64bytes; nine snapshots=576.
    # Head/hidden clones112 plus vision17 exceed the overlapping120 allowance.
    assert one == {"growth_bytes": 900, "copy_overlap_bytes": 200, "snapshot_bytes_each": 64,
                   "snapshot_bytes": 576, "transient_bytes": 129, "workspace_bytes": 129,
                   "future_bytes": 1805}
    fixed_pool = future_service(w, slots, streams=2, retained=9, growth_bytes=sum(each),
                               vision_workspace_bytes=17)
    assert fixed_pool["future_bytes"] - one["future_bytes"] == 450
    larger_workspace = future_service(w, slots, streams=2, retained=9, growth_bytes=900,
                                     vision_workspace_bytes=17, minimum_workspace_bytes=300)
    assert larger_workspace["future_bytes"] == 1976


def test_no_mtp_or_growth_removes_only_the_corresponding_allowances():
    result = future_service(weights(False), states(), streams=2, retained=1, growth_bytes=0)
    assert result["snapshot_bytes_each"] == 48
    assert result["copy_overlap_bytes"] == 0
    assert result["future_bytes"] == 48 + 112


@pytest.mark.parametrize("live,future,header,want", [
    (10, 12, None, 0), (10, 10, None, 0), (100, 20, None, 80),
    (100, 20, 40, 40), (100, 20, 90, 80), (200 * GIB, 20 * GIB, None, 180 * GIB),
])
def test_automatic_ceiling_is_only_live_service_and_header_room(live, future, header, want):
    assert automatic_graph_bytes(live, future, header) == want


def test_unified_floor_and_reusable_allocator_bytes_are_charged_once(monkeypatch):
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "8")
    monkeypatch.delenv(capacity.LIMIT_ENV, raising=False)
    monkeypatch.setattr(capacity, "_meminfo", lambda: {"MemTotal": 128 * GIB, "MemAvailable": 30 * GIB})
    torch = NS(cuda=NS(mem_get_info=lambda: (25 * GIB, 128 * GIB),
                       get_device_properties=lambda device: NS(is_integrated=True),
                       memory_allocated=lambda: 60 * GIB, memory_reserved=lambda: 62 * GIB))
    live = torch_live(torch, capacity.available_bytes)
    assert live() == 24 * GIB
    assert automatic_graph_bytes(live(), 4 * GIB) == 20 * GIB
    monkeypatch.setenv(capacity.LIMIT_ENV, "70")
    assert automatic_graph_bytes(live(), 4 * GIB) == 6 * GIB


def test_owned_private_pool_spare_is_not_general_service_room(monkeypatch):
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "8")
    monkeypatch.delenv(capacity.LIMIT_ENV, raising=False)
    monkeypatch.setattr(capacity, "_meminfo", lambda: {"MemTotal": 128 * GIB, "MemAvailable": 20 * GIB})
    torch = NS(cuda=NS(mem_get_info=lambda: (15 * GIB, 128 * GIB),
                       get_device_properties=lambda device: NS(is_integrated=True),
                       memory_allocated=lambda: 60 * GIB, memory_reserved=lambda: 68 * GIB))
    owner = object()
    room = torch_live(torch, capacity.available_bytes, unavailable_spare_bytes=6 * GIB, owner=owner)
    assert room() == 14 * GIB  #12physical +2ordinary cached, never the6private cached
    assert room.__defaults__ == (owner,)
    monkeypatch.setenv(capacity.LIMIT_ENV, "70")
    assert room() == 10 * GIB  # exclusions precede the absolute-cap min, not a second cap charge


def test_quiescent_private_pool_receipt_counts_only_owned_inactive_blocks():
    segments = [{"device": 0, "segment_pool_id": (0, 7), "total_size": 128, "allocated_size": 32, "active_size": 32},
                {"device": 0, "segment_pool_id": (0, 7), "total_size": 64, "allocated_size": 0, "active_size": 0}]
    assert private_pool_spare(segments, (0, 7), 0) == 160


@pytest.mark.parametrize("field,value", [("device", 1), ("segment_pool_id", (0, 8)),
                                       ("segment_pool_id", 7), ("allocated_size", -1),
                                       ("allocated_size", 129), ("active_size", 64), ("total_size", True)])
def test_private_pool_receipt_refuses_foreign_or_unsettled_storage(field, value):
    segment = {"device": 0, "segment_pool_id": (0, 7), "total_size": 128, "allocated_size": 32, "active_size": 32}
    segment[field] = value
    with pytest.raises(ValueError):
        private_pool_spare([segment], (0, 7), 0)


def receipt():
    return {"context_window": 262144, "cache_slots": 262148, "budget_bytes": 100,
            "cache_workspace_bytes_estimate": 30, "serving_peak_bytes_estimate": 70,
            "startup_peak_bytes_estimate": 90, "total_bytes_estimate": 90,
            "mapped_table_bytes": 12, "full_mapped_working_set_bytes_estimate": 102}


def test_resolved_plan_retains_initial_context_loading_and_exact_graph_ceiling():
    plan = receipt()
    before = deepcopy(plan)
    reserve_graph_plan(plan, 25)
    assert plan["initial_graph_admission"] == before
    assert plan["context_window"] == 262144 and plan["cache_slots"] == 262148
    assert plan["startup_peak_bytes_estimate"] == 90
    assert plan["serving_peak_bytes_estimate"] == plan["total_bytes_estimate"] == 95
    assert plan["cache_workspace_bytes_estimate"] == 55
    assert plan["full_mapped_working_set_bytes_estimate"] == 107


def test_unadmitted_resolved_budget_is_rejected_before_plan_mutation():
    for budget in (-1, 31):
        plan = receipt()
        before = deepcopy(plan)
        with pytest.raises(RuntimeError, match="admitted service plan"):
            reserve_graph_plan(plan, budget)
        assert plan == before


def decoder(request=None, live_bytes=5000, header=1000):
    live = [live_bytes]
    captured = []
    class Graphs:
        def __init__(self, dec, budget, *, spare):
            self.budget, self.spare, self.room_bytes = budget, spare, 300
            self.private_spare_bytes = 70
            self.admitted = True

        def retain_failure(self, primary):
            self.failure = primary

        def capture(self, *, prepared):
            assert prepared
            captured.append((self.budget, self.spare()))
            live[0] -= self.room_bytes

        def record(self):
            return {"budget_bytes": self.budget, "bytes": 300, "room_bytes": 300}
    cls = source_class("multi.py", "MultiDecoder", {"capture_graphs", "_publish_graphs"},
                       {"multi_graphs": NS(RoundGraphs=Graphs, prepare_capture=lambda: None,
                                           live_bytes=lambda: live[0], service_live_bytes=lambda pool: lambda: live[0]),
                        "POINTS": 4,
                        "future_service": future_service, "automatic_graph_bytes": automatic_graph_bytes,
                        "context_reservation": context_reservation,
                        "raise_failures": raise_failures})
    dec = cls()
    dec.w, dec.free = weights(), states()
    dec.streams, dec.filling, dec.kept, dec.keep = {}, [], [], 1
    dec.round_tables = NS(slots=2)
    dec.graphs, dec.graph_record = None, {"captured": False, "reason": None}
    dec._failed_graphs = dec._memory_error = None
    dec.graph_request, dec.graph_budget = request, request or 0
    dec.graph_header_room, dec.vision_workspace_bytes = header, 17
    dec.memory_gate = MemoryGate(live_bytes, 120, live=lambda: live[0])
    return dec, live, captured


def test_capture_resolves_automatic_once_and_debits_only_measured_graph_memory():
    dec, live, captured = decoder()
    dec.capture_graphs()
    assert captured == [(1000, 3195)]
    assert dec.graph_budget == 1000
    assert dec.graph_record["automatic"] and dec.graph_record["future_service"]["growth_bytes"] == 900
    assert dec.graph_record["initial_live_bytes"] == 5000 and live == [4700]
    assert dec.memory_gate.room == 4700


@pytest.mark.parametrize("budget", [512, 2048])
def test_numeric_budget_is_not_replaced_with_an_automatic_or_header_ceiling(budget):
    dec, _, captured = decoder(request=budget, header=1)
    dec.capture_graphs()
    assert dec.graph_budget == budget and captured[0][0] == budget
    assert not dec.graph_record["automatic"]


def test_no_automatic_room_is_reported_without_capturing_or_changing_cache_room():
    dec, _, captured = decoder(live_bytes=1805)
    dec.capture_graphs()
    assert not captured and dec.graphs is None and dec.memory_gate.room == 1805
    assert dec.graph_budget == 0 and "no room" in dec.graph_record["reason"]


def test_disabled_or_host_uncapturable_decoder_remains_graph_free():
    for reason in ("not requested", "TENSORFOLD_MULTI_GRAPH_MIB=0", "host-backed experts"):
        dec, _, captured = decoder()
        dec.graph_record["reason"] = reason
        dec.capture_graphs()
        assert not captured and dec.graphs is None and dec.memory_gate.room == 5000


def test_busy_decoder_refuses_before_budget_resolution_or_capture():
    dec, _, captured = decoder()
    dec.streams[1] = object()
    with pytest.raises(RuntimeError, match="idle slots"):
        dec.capture_graphs()
    assert not captured and dec.graph_budget == 0


def test_successful_pool_cannot_be_recaptured_after_its_constant_receipt_is_published():
    dec, _, captured = decoder()
    dec.capture_graphs()
    with pytest.raises(RuntimeError, match="immutable"):
        dec.capture_graphs()
    assert len(captured) == 1


def test_settled_capture_failure_preserves_primary_and_accounts_retained_pool_for_eager_recovery():
    dec, _, _ = decoder()
    boom = RuntimeError("native capture failed")
    native_cause = OSError("native cause")
    boom.__cause__ = native_cause
    namespace = dec.capture_graphs.__func__.__globals__
    namespace["multi_graphs"].RoundGraphs.capture = lambda self, **kwargs: (_ for _ in ()).throw(boom)
    with pytest.raises(RuntimeError) as caught:
        dec.capture_graphs()
    assert caught.value is boom and boom.__cause__ is native_cause
    assert dec.graphs is None and dec.memory_gate.room == 4700 and dec._failed_graphs is not None
    assert dec._memory_error is None and dec.graph_record["captured"] is False
    assert "decoder remains eager" in dec.graph_record["reason"]
    with pytest.raises(RuntimeError, match="immutable"):
        dec.capture_graphs()


def test_unadmitted_capture_memory_is_latched_before_any_request_can_borrow_decoder():
    dec, _, _ = decoder()
    boom = MemoryError("unadmitted retained graph pool")
    namespace = dec.capture_graphs.__func__.__globals__
    def unadmitted(self, **kwargs):
        self.admitted = False
        raise boom
    namespace["multi_graphs"].RoundGraphs.capture = unadmitted
    with pytest.raises(MemoryError) as caught:
        dec.capture_graphs()
    assert caught.value is dec._memory_error is boom
    assert dec.graphs is None and dec.memory_gate.room == 5000


def test_failed_pool_publication_preserves_capture_primary_and_cleanup_then_contains_decoder():
    dec, _, _ = decoder()
    primary, cleanup = RuntimeError("native capture"), OSError("pool accounting")
    namespace = dec.capture_graphs.__func__.__globals__
    namespace["multi_graphs"].RoundGraphs.capture = lambda self, **kwargs: (_ for _ in ()).throw(primary)
    namespace["multi_graphs"].service_live_bytes = lambda pool: (_ for _ in ()).throw(cleanup)
    with pytest.raises(RuntimeError) as caught:
        dec.capture_graphs()
    assert caught.value is primary and primary.__cause__ is cleanup
    assert dec._memory_error is primary and dec._failed_graphs is not None and dec.graphs is None


def test_successful_capture_publication_failure_retains_its_owner_and_contains_reuse():
    dec, _, _ = decoder()
    cleanup = OSError("live provider preparation")
    namespace = dec.capture_graphs.__func__.__globals__
    namespace["multi_graphs"].service_live_bytes = lambda pool: (_ for _ in ()).throw(cleanup)
    with pytest.raises(OSError) as caught:
        dec.capture_graphs()
    assert caught.value is dec._memory_error is cleanup
    assert dec._failed_graphs is not None and dec.graphs is None and dec.memory_gate.room == 5000


def round_graphs():
    free = [100000]
    cls = source_class("multi_graphs.py", "RoundGraphs", {"_keys", "_has_room", "capture", "_account_pool", "retain_failure"},
                       {"pdl": NS(ON=False), "time": NS(perf_counter=lambda: 0), "MIB": 1 << 20,
                        "free_bytes": lambda: free[0],
                        "prepare_capture": lambda: None,
                        "live_bytes": lambda: free[0], "private_pool_spare": private_pool_spare,
                        "failure_frame_values": failure_frame_values,
                        "torch": NS(cuda=NS(synchronize=lambda: None, empty_cache=lambda: None,
                                            memory_snapshot=lambda **kwargs: [
                                                {"device": 0, "segment_pool_id": (0, 7), "total_size": 128,
                                                 "allocated_size": 32, "active_size": 32}],
                                            memory_reserved=lambda: 500, memory_allocated=lambda: 372)),
                        "gc": NS(collect=lambda: None, isenabled=lambda: True, disable=lambda: None,
                                 enable=lambda: None), "raise_failures": raise_failures})
    graphs = cls()
    graphs.dec = NS(depth=3, pdl=False, mbuf=object())
    graphs.budget, graphs.bytes, graphs.verify, graphs.mtp = 1 << 40, 0, {}, {}
    graphs.measured, graphs.stopped, graphs.rejected = [], None, None
    graphs.pool, graphs.admitted = (0, 7), False
    graphs.allocator_backend = "native"
    graphs.private_spare_bytes = None
    graphs.failure = graphs.failure_traceback = graphs.failure_cause = graphs.failure_context = None
    graphs.failure_frames = None
    graphs.spare = lambda: free[0] - 1000
    graphs._warm = lambda *args: (None, None)
    def pieces(*args):
        free[0] -= 10
        return NS(out=object(), graphs=[object()])
    graphs._pieces = pieces
    return graphs, free


def test_other_allocator_backend_uses_physical_room_without_interpreting_empty_native_stats():
    graphs, _ = round_graphs()
    graphs.allocator_backend = "cudaMallocAsync"
    graphs._account_pool.__func__.__globals__["torch"].cuda.memory_snapshot = (
        lambda **kwargs: pytest.fail("non-native snapshot has no proved segment semantics"))
    graphs._account_pool()
    assert graphs.private_spare_bytes is None
    assert graphs.allocator_record["accounting"] == "physical-only"
    assert graphs.allocator_record["backend"] == "cudaMallocAsync"
    path = ROOT / "src/tensorfold/families/qwen4_exp/cuda/multi_graphs.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "service_live_bytes")
    namespace = {"live_bytes": lambda: 123}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    live = namespace["service_live_bytes"](graphs)
    assert live is namespace["live_bytes"] and live() == 123


def test_failed_pool_keeps_tensor_bearing_frames_after_caller_clears_exception_traceback():
    pool, _ = round_graphs()
    live, probes = [], []
    class PrivateTensor:
        def __init__(self):
            live.append(True)

        def __del__(self):
            live.pop()
    primary = RuntimeError("failed captured run")
    native_cause, native_context = OSError("native cause"), KeyError("native context")
    primary.__cause__, primary.__context__ = native_cause, native_context
    def fail():
        temporary = PrivateTensor()
        probes.append(weakref.ref(temporary))
        raise primary
    try:
        fail()
    except RuntimeError as caught:
        pool.retain_failure(caught)
    pool.dec.w = NS(device=NS(index=0))
    pool._account_pool()
    traceback.clear_frames(pool.failure_traceback)
    primary.with_traceback(None)
    gc.collect()
    assert probes[0]() is not None and live == [True]
    assert pool.failure is primary and pool.failure_traceback is not None
    assert pool.failure_cause is native_cause
    # A later propagation must not replace the original tensor-bearing trace.
    saved = pool.failure_traceback
    pool.retain_failure(primary)
    assert pool.failure_traceback is saved
    pool.failure_traceback = pool.failure_frames = None
    del saved
    gc.collect()
    assert probes[0]() is None and live == []


def test_failed_frame_journal_exhaustion_keeps_primary_and_poisons_admission():
    graphs, _ = round_graphs()
    graphs.dec.free = [NS(reset=lambda w: None)]
    graphs.dec.w, graphs.dec.gdn = NS(device=NS(index=0)), NS(parity=0)
    graphs.dec.buf, graphs.dec.mbuf = NS(gdn_tables=None, attn_step=None), None
    primary, cleanup = RuntimeError("capture body"), MemoryError("failure frame journal")
    graphs._keys = lambda states: (_ for _ in ()).throw(primary)
    graphs._account_pool.__func__.__globals__["failure_frame_values"] = lambda *args: (_ for _ in ()).throw(cleanup)
    with pytest.raises(RuntimeError) as caught:
        graphs.capture()
    assert caught.value is primary and primary.__cause__ is cleanup
    assert not graphs.admitted and graphs.failure is primary and graphs.failure_traceback is not None


@pytest.mark.parametrize("link", ["cause", "context", "group"])
def test_frame_journal_covers_causal_traces_and_visits_cycles_once(link):
    probes = []
    class PrivateTensor:
        pass
    def fail():
        temporary = PrivateTensor()
        probes.append(weakref.ref(temporary))
        raise OSError("native capture")
    try:
        fail()
    except OSError as caught:
        native = caught
    primary = RuntimeError("capture wrapper")
    if link == "context":
        primary.__context__ = native
    else:
        primary.__cause__ = ExceptionGroup("native failures", [native]) if link == "group" else native
    native.__context__ = primary  # a repeated causal edge cannot loop or duplicate the journal forever
    journal = failure_frame_values(primary, None, primary.__cause__, primary.__context__)
    traceback.clear_frames(native.__traceback__)
    native.with_traceback(None)
    gc.collect()
    assert probes[0]() is not None
    del journal
    gc.collect()
    assert probes[0]() is None


@pytest.mark.parametrize("secondary", [None, "snapshot", "budget"])
def test_native_capture_primary_retains_accounted_pool_or_secondary_failure(secondary):
    graphs, free = round_graphs()
    reset = []
    graphs.dec.free = [NS(reset=reset.append)]
    graphs.dec.w, graphs.dec.gdn = NS(device=NS(index=0)), NS(parity=0)
    graphs.dec.buf, graphs.dec.mbuf = NS(gdn_tables=None, attn_step=None), None
    primary, native_cause, cleanup = RuntimeError("capture body"), KeyError("native cause"), OSError("snapshot")
    primary.__cause__ = native_cause
    def capture(states):
        free[0] -= 128
        raise primary
    graphs._keys = capture
    if secondary == "snapshot":
        graphs._account_pool = lambda: (_ for _ in ()).throw(cleanup)
    elif secondary == "budget":
        graphs.budget = 100
    with pytest.raises(RuntimeError) as caught:
        graphs.capture()
    assert caught.value is primary and reset == [graphs.dec.w]
    assert graphs.room_bytes == graphs.settled_bytes == 128
    if secondary is None:
        assert graphs.admitted and graphs.private_spare_bytes == 96
        assert primary.__cause__ is native_cause
        assert graphs.allocator_record["reserved_bytes"] == 500
    else:
        assert not graphs.admitted and isinstance(primary.__cause__, BaseExceptionGroup)
        causes = primary.__cause__.exceptions
        assert native_cause in causes
        assert cleanup in causes if secondary == "snapshot" else any(isinstance(e, MemoryError) for e in causes)


def test_finite_mtp3_parallel4_keys_include_both_n4_parities_and_all_mtp_rows():
    graphs, _ = round_graphs()
    graphs._keys([object()] * 4)
    assert len(graphs.verify) == 68 and len(graphs.mtp) == 34 and len(graphs.measured) == 102
    assert all((False, 4, rows, parity) in graphs.verify for rows in range(4, 17) for parity in (0, 1))
    assert all((False, 4, rows) in graphs.mtp for rows in range(4, 17))
    assert graphs.bytes == 1020 and graphs.stopped is None


def test_fresh_room_checks_stop_before_warmup_and_after_warmup():
    graphs, free = round_graphs()
    free[0] = 1000
    graphs._warm = lambda *args: pytest.fail("must not warm without live room")
    graphs._keys([object()])
    assert not graphs.measured and "live memory" in graphs.stopped
    graphs, free = round_graphs()
    def warm(*args):
        free[0] = 1000
        return None, None
    graphs._warm = warm
    graphs._pieces = lambda *args: pytest.fail("must not capture after warmup consumed room")
    graphs._keys([object()])
    assert not graphs.measured and "live memory" in graphs.stopped


@pytest.mark.parametrize("constraint", ["budget", "future"])
def test_candidate_that_exceeds_a_reservation_is_recorded_and_not_published(constraint):
    graphs, free = round_graphs()
    if constraint == "budget":
        graphs.budget = 5
    else:
        free[0] = 1005
    graphs._keys([object()])
    assert not graphs.verify and not graphs.mtp and graphs.bytes == 0
    assert graphs.rejected["key"] == ["verify", False, 1, 1, 0]
    assert graphs.rejected["bytes"] == 10 and graphs.stopped


@pytest.mark.parametrize("constraint", ["service", "admitted_budget"])
def test_settled_capture_cannot_silently_consume_service_or_exceed_header_budget(constraint):
    graphs, free = round_graphs()
    graphs.dec.free = [NS(reset=lambda w: None)]
    graphs.dec.w, graphs.dec.gdn = object(), NS(parity=0)
    graphs.dec.buf = NS(gdn_tables=None, attn_step=None)
    graphs.dec.mbuf = None
    graphs._account_pool = lambda: None  # pool schema/lifetime is exercised separately; isolate the settled guard
    def capture(states):
        free[0] = 999 if constraint == "service" else 98500
    graphs._keys = capture
    if constraint == "admitted_budget":
        graphs.budget = 1000
    with pytest.raises(MemoryError, match="required native window" if constraint == "service" else "admitted budget"):
        graphs.capture()
    assert graphs.settled_bytes == (99001 if constraint == "service" else 1500)

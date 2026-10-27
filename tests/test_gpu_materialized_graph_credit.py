"""Fixed-GPU graph admission: full ownership, unchanged promises and rank agreement."""

import copy
from types import SimpleNamespace as NS

import pytest

from tensorfold.families.qwen4_exp.cuda import service_memory as memory
from tests.test_ram_graph_credit import decoder, fake_runtime, header  # noqa: F401


GIB = 1 << 30


def gpu_states(count=26):
    placement = object()
    seq = []
    for slot in range(count):
        caches = [NS(logical_capacity=262147, placement=placement, slot=slot, rank=0) for _ in range(13)]
        seq.append(NS(capacity=262147, limit=262147, indexer_history_mode="gpu", indexer_history_owner=None,
            history_placement=placement, history_slot=slot, history_rank=0,
            compact_indexer=NS(capacity=262147, placement=placement), kc=caches[:-1], mtp_kc=caches[-1],
            cache_bytes=lambda: 1000))
    return seq


def exact_future():
    return dict(growth_bytes=0, copy_overlap_bytes=0, snapshot_bytes=1639350272,
        workspace_bytes=2147483648, future_bytes=3786833920)


def all_windows(count=26):
    return dict(growth_bytes=0, full_windows_reserved=count, copy_overlap_bytes=0)


def test_actual_observation_numbers_keep_all_26_windows_and_complete_future_under_same_budget():
    states, required = gpu_states(), exact_future()
    room = memory.materialized_gpu_graph_room(states, 262147, 4079484928, all_windows(), required)
    assert room == 292651008 and room > 169869312
    assert len(states) == 26 and all(state.capacity == state.limit == 262147 for state in states)
    receipt = header()
    receipt.update(budget_bytes=29073342464, serving_peak_bytes_estimate=28923635136,
        total_bytes_estimate=28923635136, cache_slots=262147, context_window=262144)
    original = copy.deepcopy(receipt)
    proof = dict(policy=memory.GPU_UNIQUE_GRAPH_ADMISSION_VERSION, agreed_graph_bytes=room,
        future_service=required, private_spare_credit_bytes=0, full_slots=26)
    memory.refine_materialized_graph_plan(receipt, proof)
    assert receipt["observed_graph_admission"]["original_header"]["serving_peak_bytes_estimate"] == 28923635136
    assert receipt["serving_peak_bytes_estimate"] == 28780691456
    memory.reserve_graph_plan(receipt, room)
    assert receipt["total_bytes_estimate"] == original["budget_bytes"]
    assert receipt["cache_slots"] == 262147 and receipt["context_window"] == 262144
    assert receipt["observed_graph_admission"]["future_service"] == required


@pytest.mark.parametrize("broken", ["partial", "limit", "placement", "indexer", "indexer-capacity",
    "indexer-owner", "kv-capacity", "kv-owner", "slot", "rank", "migration", "growth", "overlap",
    "windows", "future-overlap", "workspace", "sum"])
def test_incomplete_or_unowned_windows_and_reduced_future_cannot_earn_credit(broken):
    seq, promise, required = gpu_states(), all_windows(), exact_future()
    state = seq[0]
    if broken == "partial":
        state.capacity -= 1
    elif broken == "limit":
        state.limit -= 1
    elif broken == "placement":
        state.history_placement = None
    elif broken == "indexer":
        state.compact_indexer = None
    elif broken == "indexer-capacity":
        state.compact_indexer.capacity -= 1
    elif broken == "indexer-owner":
        state.compact_indexer.placement = object()
    elif broken == "kv-capacity":
        state.mtp_kc.logical_capacity -= 1
    elif broken == "kv-owner":
        state.kc[0].placement = object()
    elif broken == "slot":
        state.kc[0].slot += 1
    elif broken == "rank":
        state.kc[0].rank += 1
    elif broken == "migration":
        state._history_migration = {"target": object()}
    elif broken == "growth":
        promise["growth_bytes"] = 1
    elif broken == "overlap":
        promise["copy_overlap_bytes"] = 1
    elif broken == "windows":
        promise["full_windows_reserved"] = 24
    elif broken == "future-overlap":
        required["copy_overlap_bytes"] = 1
        required["future_bytes"] += 1
    elif broken == "workspace":
        required["workspace_bytes"] -= 1
        required["future_bytes"] -= 1
    else:
        required["future_bytes"] -= 1
    with pytest.raises(ValueError):
        memory.materialized_gpu_graph_room(seq, 262147, 4079484928, promise, required)


def unique_decoder(base, monkeypatch):
    multi, dec, calls, captures = base
    seq, required = gpu_states(8), exact_future()
    dec.free, dec.capacity = seq, 262147
    dec.round_tables.slots = 8
    dec.w.meta.update(indexer_history="gpu", history_layout="unique", world=8,
        unique_capture_contract=multi.multi_graphs.UNIQUE_CAPTURE_CONTRACT)
    monkeypatch.setattr(multi, "context_reservation", lambda *args, **kwargs: all_windows(8))
    monkeypatch.setattr(multi, "future_service", lambda *args, **kwargs: dict(required))
    monkeypatch.setattr(multi.multi_graphs, "live_bytes", lambda: required["future_bytes"] + 500)

    def gather(runtime, collective, packet, *, world):
        calls.append(list(packet))
        return [packet] + [[1, 200 + rank, required["future_bytes"] + 200 + rank, required["future_bytes"]] for rank in range(7)]
    monkeypatch.setattr(multi, "gather_ints", gather)
    return multi, dec, calls, captures


def test_real_orchestration_uses_smallest_rank_and_records_static_and_materialized_proofs(decoder, monkeypatch):  # noqa: F811
    _, dec, calls, captures = unique_decoder(decoder, monkeypatch)
    dec.capture_graphs()
    proof = dec.graph_record["observed_graph_admission"]
    assert calls == [[1, 500, exact_future()["future_bytes"] + 500, exact_future()["future_bytes"]]]
    assert captures == [(200, True)] and dec.graph_budget == 200
    assert proof["policy"] == memory.GPU_UNIQUE_GRAPH_ADMISSION_VERSION
    assert proof["header_room_bytes"] == 31 and proof["private_spare_credit_bytes"] == 0
    assert proof["full_slots"] == 8 and proof["future_service"] == exact_future()


@pytest.mark.parametrize("broken", ["peer-failure", "peer-future", "peer-sum", "local-owner"])
def test_peer_or_local_proof_failure_precedes_any_graph_capture(decoder, monkeypatch, broken):  # noqa: F811
    multi, dec, calls, captures = unique_decoder(decoder, monkeypatch)
    if broken == "local-owner":
        dec.free[0].kc[0].logical_capacity -= 1

    def gather(runtime, collective, packet, *, world):
        calls.append(list(packet))
        peer = [1, 200, exact_future()["future_bytes"] + 200, exact_future()["future_bytes"]]
        if broken == "peer-failure":
            peer[0] = 0
        elif broken == "peer-future":
            peer[2] += 1
            peer[3] += 1
        elif broken == "peer-sum":
            peer[1] += 1
        return [packet, peer] + [list(packet) for _ in range(6)]
    monkeypatch.setattr(multi, "gather_ints", gather)
    with pytest.raises((RuntimeError, ValueError)):
        dec.capture_graphs()
    assert len(calls) == 1 and not captures


def test_absolute_cap_debits_external_mappings_without_allocator_reclaim_credit(monkeypatch):
    from tensorfold.cuda import capacity, kv_arena
    from tensorfold.families.qwen4_exp.cuda import multi_graphs

    runtime = NS(cuda=NS(mem_get_info=lambda: (8 * GIB, 32 * GIB), memory_allocated=lambda: 22 * GIB,
        get_device_properties=lambda device: NS(is_integrated=False),
        memory_reserved=lambda: pytest.fail("allocator spare cannot be borrowed")))
    monkeypatch.setattr(multi_graphs, "torch", runtime)
    monkeypatch.setattr(kv_arena, "mapped_bytes", lambda: GIB)
    monkeypatch.setenv(capacity.LIMIT_ENV, "24")
    monkeypatch.delenv("TENSORFOLD_MEMORY_RESERVE_GIB", raising=False)
    monkeypatch.delenv("TENSORFOLD_CUDA_MEMORY_RESERVE_GIB", raising=False)
    assert multi_graphs.live_bytes() == GIB
    monkeypatch.setattr(kv_arena, "mapped_bytes", lambda: 3 * GIB)
    assert multi_graphs.live_bytes() == 0


def test_gpu_graph_policy_version_is_part_of_the_existing_rank_configuration_digest(fake_runtime, monkeypatch):  # noqa: F811
    import torch
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.kv_formats import get_pair

    class Comm:
        def __init__(self, peer=None):
            self.peer = peer
        def all_gather(self, source, output):
            self.sent = source.clone()
            output.copy_(torch.cat([source, source if self.peer is None else self.peer]))

    def participant(comm):
        engine = FlashNextEngine.__new__(FlashNextEngine)
        engine.depth, engine.confidence, engine.max_len = 2, .7, 262147
        engine.streams, engine.graphs_enabled, engine.prefill_rows, engine.tp = 26, True, 256, 2
        engine.rope = NS(metadata=lambda: {"rope_type": "default"})
        engine.kv_pair, engine.comm = get_pair("int8"), comm
        engine.execution_provider, engine.indexer_history = "sm70", "gpu"
        engine.sm70_profile = "source-packed-graphs-trial"
        return engine

    old = Comm()
    monkeypatch.setattr(memory, "GPU_UNIQUE_GRAPH_ADMISSION_VERSION", "header-graph-policy-v0")
    participant(old)._same_settings(torch, None)
    new = Comm(old.sent)
    monkeypatch.setattr(memory, "GPU_UNIQUE_GRAPH_ADMISSION_VERSION", "gpu-unique-full-resident-observed-v1")
    with pytest.raises(RuntimeError, match="different settings"):
        participant(new)._same_settings(torch, None)
    assert new.sent[9] != old.sent[9]
    assert new.sent.numel() == old.sent.numel() == 12

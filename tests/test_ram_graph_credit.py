"""Observed graph-credit contracts, using real policy/orchestration without CUDA."""

import copy
from types import SimpleNamespace as NS

import pytest

from tensorfold.families.qwen4_exp.cuda import service_memory as memory
from tests.test_cuda_capacity import fake_runtime  # noqa: F401


GIB = 1 << 30


def future():
    return dict(growth_bytes=0, copy_overlap_bytes=40, snapshot_bytes=80,
                workspace_bytes=2 * GIB, future_bytes=2 * GIB + 120)


def states():
    return [NS(capacity=10000, limit=10000, indexer_history_owner=NS(limit=10000),
               cache_bytes=lambda: 1000) for _ in range(2)]


def promise():
    return dict(growth_bytes=0, full_windows_reserved=2, copy_overlap_bytes=40)


def test_observed_spare_protects_the_entire_existing_future_and_accepts_zero_only_when_it_fits():
    required = future()
    assert memory.materialized_graph_room(states(), 10000, required['future_bytes'] + 667740672,
                                           promise(), required) == 667740672
    assert memory.materialized_graph_room(states(), 10000, required['future_bytes'], promise(), required) == 0
    with pytest.raises(MemoryError, match='required service promises'):
        memory.materialized_graph_room(states(), 10000, required['future_bytes'] - 1, promise(), required)


@pytest.mark.parametrize('broken', ['partial', 'owner', 'growth', 'selected', 'workspace', 'sum', 'bool'])
def test_unproven_materialization_or_reduced_promises_cannot_earn_credit(broken):
    seq, keep, required = states(), promise(), future()
    if broken == 'partial':
        seq[0].capacity -= 1
    elif broken == 'owner':
        seq[0].indexer_history_owner = None
    elif broken == 'growth':
        keep['growth_bytes'] = 1
    elif broken == 'selected':
        keep['full_windows_reserved'] = 1
    elif broken == 'workspace':
        required['workspace_bytes'] -= 1
        required['future_bytes'] -= 1
    elif broken == 'sum':
        required['future_bytes'] -= 1
    else:
        required['snapshot_bytes'] = True
    with pytest.raises(ValueError):
        memory.materialized_graph_room(seq, 10000, 4 * GIB, keep, required)


def header():
    return dict(native_window=262144, requested_context=260060, context_window=260060,
                cache_slots=260064, budget_bytes=30 * GIB, weight_bytes_estimate=14 * GIB,
                loading_bytes_estimate=32 << 20, cache_workspace_bytes_estimate=16 * GIB - 31056,
                startup_peak_bytes_estimate=14 * GIB + (32 << 20), serving_peak_bytes_estimate=30 * GIB - 31056,
                total_bytes_estimate=30 * GIB - 31056, mapped_table_bytes=0,
                full_mapped_working_set_bytes_estimate=30 * GIB - 31056)


def test_refinement_preserves_header_authority_and_does_not_debit_its_estimate_twice():
    receipt = header()
    original = copy.deepcopy(receipt)
    proof = dict(policy='ram-full-resident-observed-v1', agreed_graph_bytes=667740672,
                 future_service=future(), private_spare_credit_bytes=0)
    memory.refine_materialized_graph_plan(receipt, proof)
    assert receipt['observed_graph_admission']['original_header']['serving_peak_bytes_estimate'] == original['serving_peak_bytes_estimate']
    assert receipt['weight_bytes_estimate'] == original['weight_bytes_estimate']
    assert receipt['cache_slots'] == original['cache_slots'] and receipt['context_window'] == original['context_window']
    assert receipt['serving_peak_bytes_estimate'] == original['budget_bytes'] - proof['agreed_graph_bytes']
    memory.reserve_graph_plan(receipt, proof['agreed_graph_bytes'])
    assert receipt['total_bytes_estimate'] == receipt['budget_bytes']
    assert receipt['initial_graph_admission']['observed_graph_admission']['original_header'] == receipt['observed_graph_admission']['original_header']
    with pytest.raises(RuntimeError, match='repeated'):
        memory.refine_materialized_graph_plan(receipt, proof)


@pytest.fixture
def decoder(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import multi

    required, seq = future(), states()
    dec = multi.MultiDecoder.__new__(multi.MultiDecoder)
    dec.streams, dec.filling, dec.kept = {}, {}, []
    dec.free, dec.capacity = seq, 10000
    dec.round_tables = NS(slots=2)
    dec.graphs = dec._failed_graphs = None
    dec.graph_record = {'reason': None}
    dec.w = NS(meta={'execution_provider': 'sm70', 'indexer_history': 'ram', 'world': 2}, comm=NS())
    dec.full_resident, dec.full_windows, dec.keep = True, 2, 8
    dec.graph_request, dec.graph_budget, dec.graph_header_room = None, 0, 31
    dec.graph_floor, dec.graph_floor_extra_bytes, dec.vision_workspace_bytes = 8192, 0, 0
    dec.memory_gate = NS(reserve=2 * GIB)
    dec._prepare_graph_floor = lambda: True
    monkeypatch.setattr(multi.multi_graphs, 'prepare_capture', lambda: None)
    monkeypatch.setattr(multi.multi_graphs, 'live_bytes', lambda: required['future_bytes'] + 500)
    monkeypatch.setattr(multi, 'context_reservation', lambda *a, **kw: promise())
    monkeypatch.setattr(multi, 'future_service', lambda *a, **kw: dict(required))
    calls, captures = [], []

    def gather(runtime, collective, packet, *, world):
        calls.append(list(packet))
        peer = list(packet)
        peer[1], peer[2] = 200, required['future_bytes'] + 200
        return [packet, peer]

    monkeypatch.setattr(multi, 'gather_ints', gather)
    dec.w.comm.all_gather = object()

    class Graphs:
        def __init__(self, owner, budget, *, spare):
            self.budget = budget
            assert spare() == 500
        def capture(self, *, prepared):
            captures.append((self.budget, prepared))

    monkeypatch.setattr(multi.multi_graphs, 'RoundGraphs', Graphs)
    def publish(graphs, resolution, *, captured):
        dec.graph_record = {**resolution, 'captured': captured, 'budget_bytes': graphs.budget}
        dec.graphs = graphs
    dec._publish_graphs = publish
    return multi, dec, calls, captures


def test_actual_orchestrator_uses_all_rank_minimum_and_records_unchanged_future(decoder):
    _, dec, calls, captures = decoder
    dec.capture_graphs()
    assert calls == [[1, 500, future()['future_bytes'] + 500, future()['future_bytes']]]
    assert dec.graph_budget == 200 and captures == [(200, True)]
    proof = dec.graph_record['observed_graph_admission']
    assert proof['agreed_graph_bytes'] == 200 and proof['future_service'] == future()
    assert proof['private_spare_credit_bytes'] == 0 and proof['header_room_bytes'] == 31


@pytest.mark.parametrize('region', ['gpu', 'nonfull', 'explicit'])
def test_existing_gpu_nonfull_and_explicit_ceiling_paths_are_unchanged(decoder, region):
    _, dec, calls, captures = decoder
    if region == 'gpu':
        dec.w.meta['indexer_history'] = 'gpu'
    elif region == 'nonfull':
        dec.full_resident = False
    else:
        dec.graph_request = dec.graph_budget = 123
    dec.capture_graphs()
    expected = 123 if region == 'explicit' else 31  # old min(live-future, header_room)
    assert not calls and captures == [(expected, True)] and dec.graph_budget == expected
    assert 'observed_graph_admission' not in dec.graph_record


@pytest.mark.parametrize('failed_rank', ['local', 'peer'])
def test_failed_observation_is_shared_before_any_capture_or_failure(decoder, monkeypatch, failed_rank):
    multi, dec, calls, captures = decoder
    if failed_rank == 'local':
        monkeypatch.setattr(multi.multi_graphs, 'live_bytes', lambda: (_ for _ in ()).throw(RuntimeError('live observation failed')))
    def gather(runtime, collective, packet, *, world):
        calls.append(packet)
        return [packet, [0 if failed_rank == 'peer' else 1, 0, 0, future()['future_bytes']]]
    monkeypatch.setattr(multi, 'gather_ints', gather)
    with pytest.raises(RuntimeError):
        dec.capture_graphs()
    assert len(calls) == 1 and not captures
    assert calls[0][0] == (0 if failed_rank == 'local' else 1)


def test_original_live_reader_keeps_device_floor_absolute_cap_and_no_allocator_spare(monkeypatch):
    from tensorfold.cuda import capacity
    from tensorfold.families.qwen4_exp.cuda import multi_graphs

    runtime = NS(cuda=NS(mem_get_info=lambda: (8 * GIB, 32 * GIB), memory_allocated=lambda: 22 * GIB,
        get_device_properties=lambda device: NS(is_integrated=False),
        memory_reserved=lambda: pytest.fail('private allocator spare must not be credited')))
    monkeypatch.setattr(multi_graphs, 'torch', runtime)
    monkeypatch.delenv('TENSORFOLD_MEMORY_RESERVE_GIB', raising=False)
    monkeypatch.delenv('TENSORFOLD_CUDA_MEMORY_RESERVE_GIB', raising=False)
    monkeypatch.delenv(capacity.LIMIT_ENV, raising=False)
    assert multi_graphs.live_bytes() == 4 * GIB
    monkeypatch.setenv(capacity.LIMIT_ENV, '24')
    assert multi_graphs.live_bytes() == 2 * GIB


@pytest.mark.parametrize('invalid', ['negative', 'inconsistent', 'short', 'bool'])
def test_malformed_peer_observation_never_allocates_a_graph(decoder, monkeypatch, invalid):
    multi, dec, calls, captures = decoder
    def gather(runtime, collective, packet, *, world):
        calls.append(packet)
        peer = [1, 200, future()['future_bytes'] + 200, future()['future_bytes']]
        if invalid == 'negative':
            peer[1] = -1
        elif invalid == 'inconsistent':
            peer[1] += 1
        elif invalid == 'short':
            peer.pop()
        else:
            peer[0] = True
        return [packet, peer]
    monkeypatch.setattr(multi, 'gather_ints', gather)
    with pytest.raises(RuntimeError, match='observation|proof'):
        dec.capture_graphs()
    assert len(calls) == 1 and not captures


@pytest.mark.parametrize('mode', ['gpu', 'ram'])
def test_ram_graph_policy_version_uses_existing_rank_digest_without_changing_gpu_bytes(fake_runtime, monkeypatch, mode):  # noqa: F811
    import torch
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.cuda.indexer_history import HistoryPlan
    from tensorfold.families.qwen4_exp.kv_formats import get_pair

    class Comm:
        def __init__(self, peer=None):
            self.peer, self.sent = peer, None
        def all_gather(self, source, output):
            self.sent = source.clone()
            output.copy_(torch.cat([source, source if self.peer is None else self.peer]))

    def rank(comm):
        engine = FlashNextEngine.__new__(FlashNextEngine)
        engine.depth, engine.confidence, engine.max_len = 3, 0.0, 8192
        engine.streams, engine.graphs_enabled, engine.prefill_rows, engine.tp = 2, True, 2048, 2
        engine.rope = NS(metadata=lambda: {'rope_type': 'default'})
        engine.kv_pair, engine.comm = get_pair('int8'), comm
        engine.execution_provider, engine.indexer_history = 'sm70', mode
        engine._indexer_history_plan = HistoryPlan(13, 128, 4, 2048, 4)
        return engine

    old = Comm()
    monkeypatch.setattr(memory, 'RAM_GRAPH_ADMISSION_VERSION', 'header-graph-policy-v0')
    rank(old)._same_settings(torch, None)
    new = Comm(old.sent)
    monkeypatch.setattr(memory, 'RAM_GRAPH_ADMISSION_VERSION', 'observed-graph-admission-v1')
    if mode == 'ram':
        with pytest.raises(RuntimeError, match='different settings'):
            rank(new)._same_settings(torch, None)
        assert new.sent[9] != old.sent[9]
    else:
        rank(new)._same_settings(torch, None)
        assert torch.equal(new.sent, old.sent)
    assert new.sent.numel() == old.sent.numel() == 12

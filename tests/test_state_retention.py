# ruff: noqa: E402 - optional torch capability must be checked before CUDA-family imports.
"""Real family State required-payload inventory; CPU software scope only."""

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda import forward
from tensorfold.families.qwen4_exp.cuda.state import Buffers, State
from tensorfold.families.qwen4_exp.cuda.state_retention import state_inventory
from test_dspark_taps_forward import cpu_weights


@pytest.fixture
def materialized_family(monkeypatch):
    weights = cpu_weights(monkeypatch)
    state, buffers = State(weights, 8, 4, "fp16", limit=8), Buffers(weights, 4, 8, prefill=True)
    tokens = [3, 7, 11]
    buffers.ids[:3].copy_(torch.tensor(tokens, dtype=torch.int32))
    buffers.ple_v.fill_(.125)
    state.ple_last = (state.ple_history.copy(), np.asarray(tokens, dtype=np.int64))
    forward.compute(weights, [(state, 0, 3)], buffers)
    forward.commit(weights, state, buffers, 3, 3)
    return weights, state, buffers


@pytest.fixture
def materialized(materialized_family):
    return materialized_family[1]


def test_complete_inventory_includes_cache_rows_beyond_state_snapshot_without_cloning(materialized):
    state = materialized
    inventory = state_inventory(state, recovery_versions=0)
    parts = {part.name: part for part in inventory.parts}
    assert {"state.rec", "state.conv", "state.ple_tail", "history.kv.0.0.0.k", "history.kv.0.0.0.v"} <= parts.keys()
    cache = parts["history.kv.0.0.0.k"]
    assert cache.tensor.shape[0] == state.pos == 3 and cache.tensor.data_ptr() == state.kc[0].k.data_ptr()
    assert parts["state.rec"].tensor.data_ptr() == state.rec[state.cur[0]].data_ptr()
    assert all("scratch" not in part.name for part in inventory.parts)
    assert sum(spec.nbytes for spec in inventory.specs) <= inventory.future_bytes


def test_actual_recovery_versions_are_inventoried_or_refused(materialized):
    snapshot = materialized.snapshot()
    with pytest.raises(ValueError, match="version bound"):
        state_inventory(materialized, recoveries=(snapshot,), recovery_versions=0)
    inventory = state_inventory(materialized, recoveries=(snapshot,), recovery_versions=1)
    assert any(part.name == "recovery.0.rec" for part in inventory.parts)


def test_restore_resource_projection_matches_actual_new_payloads_and_excludes_full_windows(materialized, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.state_retention import state_restore_plan, state_restore_resources

    state = materialized
    inventory = state_inventory(state, recoveries=(state.snapshot(),), recovery_versions=1)
    records = ((inventory.metadata, inventory.specs),)
    expected = sum(part.tensor.nbytes for part in inventory.parts if part.name.startswith("recovery."))
    resources = state_restore_resources(state, records, continuation="same-request")
    assert resources["recovery_device_bytes"] == expected
    assert resources["pending_hc_device_bytes"] == resources["phase_device_bytes"] == 0
    assert resources["host_destination_bytes"] == 0  # No Stream/canonical IDs in this inventory.
    original = torch.empty
    allocated = []
    def counted(*args, **kwargs):
        value = original(*args, **kwargs)
        allocated.append(value.nbytes)
        return value
    monkeypatch.setattr(torch, "empty", counted)
    state_restore_plan(state, records, continuation="same-request", recovery_versions=1)
    assert sum(allocated) == expected


def test_new_turn_resource_projection_drops_parent_recovery_allocations(materialized):
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda.state_retention import state_restore_resources

    inventory = state_inventory(materialized, recoveries=(materialized.snapshot(),), recovery_versions=1)
    value = state_restore_resources(materialized, ((inventory.metadata, inventory.specs),),
        stream=Stream([0] * (materialized.pos + 1), 2, draft=False), continuation="new-turn")
    assert value["recovery_device_bytes"] == value["pending_hc_device_bytes"] == value["phase_device_bytes"] == 0


def test_physical_rebind_reuses_owned_canonical_bytes_without_a_python_token_archive(materialized, monkeypatch):
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda import state_retention

    stream = Stream([1, 2, 3, 4], 3, draft=False)
    old = state_inventory(materialized, stream=stream, recovery_versions=0)
    canonical = next(part for part in old.parts if part.name == "state.canonical_ids")
    before = next(part for part in old.parts if part.name == "history.kv.0.0.0.k")
    # A controlled replacement models the physical-identity boundary; values
    # and semantic frontier remain unchanged. Device migration has its own tests.
    materialized.kc[0].k = materialized.kc[0].k.clone()
    materialized.history_generation += 1
    monkeypatch.setattr(state_retention, "canonical_tokens", lambda *_a, **_kw: pytest.fail("canonical input was rebuilt"))
    rebound = state_retention.rebind_state_inventory(materialized, old, recovery_versions=0)
    after = next(part for part in rebound.parts if part.name == "history.kv.0.0.0.k")
    assert after.tensor.data_ptr() != before.tensor.data_ptr()
    assert torch.equal(after.tensor, before.tensor)
    assert next(part for part in rebound.parts if part.name == "state.canonical_ids") is canonical
    assert rebound.metadata["request"] == old.metadata["request"]
    assert rebound.metadata["method"]["draws"] == old.metadata["method"]["draws"]
    assert rebound.metadata["source_history_generation"] == materialized.history_generation
    materialized.set_pos(materialized.pos - 1)
    with pytest.raises(ValueError, match="semantic generation"):
        state_retention.rebind_state_inventory(materialized, rebound, recovery_versions=0)


def test_unresolved_target_rows_cannot_be_published_as_retained_state(materialized):
    materialized.target_tentative_length += 1
    with pytest.raises(ValueError, match="resolved target"):
        state_inventory(materialized, recovery_versions=0)


def test_actual_eight_rank_unique_history_relocates_all_bytes_to_different_slot():
    from types import SimpleNamespace

    from tensorfold.families.qwen4_exp.cuda.state_retention import commit_state_restore, state_restore_plan
    from test_flashnext_history_placement import _seed_zero_indexer_prefix
    from test_sm70_rotorquant_state import model

    sources, destinations = [], []
    for rank in range(8):
        weights = model(8, mtp=False)
        weights.meta.update(full=SimpleNamespace(kv_heads=2), rank=rank, execution_provider="sm70",
                            history_layout="unique", attention_degree=2)
        source = State(weights, 517, 4, "fp16", limit=517, slot=0)
        destination = State(weights, 517, 4, "fp16", limit=517, slot=5)
        _seed_zero_indexer_prefix(source, 257)
        for ordinal, cache in enumerate(source.kc):
            for _consumed, offset, count in cache._spans(0, source.pos):
                cache.k[offset:offset + count].fill_(rank + ordinal * 8 + 1)
                cache.v[offset:offset + count].fill_(-rank - ordinal * 8 - 1)
        for ordinal, value in enumerate(source.pooled):
            value[:source.pos // 4].fill_(ordinal + 1)
        sources.append(source)
        destinations.append(destination)
    inventories = tuple(state_inventory(state, recovery_versions=0) for state in sources)
    records = tuple((item.metadata, item.specs) for item in inventories)
    payloads = {(rank, number): part.tensor.clone() for rank, item in enumerate(inventories)
                for number, part in enumerate(item.parts)}
    moved = 0
    for rank, state in enumerate(destinations):
        plan = state_restore_plan(state, records, continuation="same-request", recovery_versions=0)
        for key, part in plan.targets.items():
            part.tensor.copy_(payloads[key])
            moved += key[0] != rank
        commit_state_restore(plan)
        assert state.pos == state.pos_dev.item() == 257
    assert moved > 0
    # Logical descriptors identify source bytes independently of physical rank
    # and offset. Every history component is present exactly once globally.
    before = {part.name: part.tensor for item in inventories for part in item.parts if part.category == "history"
              or part.name.startswith("history.kv.")}
    after = {part.name: part.tensor for state in destinations
             for part in state_inventory(state, recovery_versions=0).parts if part.category == "history"
             or part.name.startswith("history.kv.")}
    assert before.keys() == after.keys()
    assert all(torch.equal(value, after[name]) for name, value in before.items())


def test_tentative_mtp_pages_and_pools_are_never_marked_immutable_backing():
    from tensorfold.families.qwen4_exp.cuda.state_retention import _history_parts
    from test_sm70_rotorquant_state import model

    value = State(model(), 256, 4, "fp16", limit=256)
    value.pos = 126
    value.mtp_len, value.mtp_drafted = 129, 3
    ordinal = len(value.kc)
    parts = {part.name: part for part in _history_parts(value)}
    assert parts[f"history.kv.{ordinal}.0.0.k"].category == "state"
    assert parts[f"history.pool.{ordinal}.0"].category == "state"
    # Once all of the first128 rows are committed, that full page is immutable;
    # the incomplete tail remains a mutable state object.
    value.mtp_drafted = 1
    parts = {part.name: part for part in _history_parts(value)}
    assert parts[f"history.kv.{ordinal}.0.0.k"].category == "history"
    assert parts[f"history.kv.{ordinal}.0.1.k"].category == "state"


def test_new_turn_rewinds_restored_mtp_proposal_base_and_device_counter():
    from types import SimpleNamespace

    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda.compact_indexer import Staging
    from tensorfold.families.qwen4_exp.cuda.state_retention import commit_state_restore, state_restore_plan
    from test_compact_indexer import begin
    from test_flashnext_history_placement import _seed_zero_indexer_prefix
    from test_sm70_rotorquant_state import model

    sources, destinations, bases, requests = [], [], {}, []
    for rank in range(8):
        weights = model(8)
        weights.meta.update(full=SimpleNamespace(kv_heads=2), rank=rank, execution_provider="sm70",
                            history_layout="unique", attention_degree=2)
        source, destination = [State(weights, 16, 4, "fp16", limit=16, slot=slot) for slot in (0, 5)]
        _seed_zero_indexer_prefix(source, 5)
        source.mtp_len = 5
        source.mtp_pos.fill_(5)
        entry = source.compact_indexer.entries[-1]
        if entry is not None:
            entry.n = 5
            entry.raw.fill_(3)
            entry.coordinates.fill_(4)
            bases[len(source.kc)] = entry.raw.clone()
        source.begin_mtp_proposals()
        if entry is not None:
            stage = Staging(4, len(source.kc) + 1, "cpu", mtp=True)
            begin(entry, stage, torch.full((2, 128), 9, dtype=torch.float16),
                  torch.tensor([[5, 5, 5], [6, 6, 6]], dtype=torch.int64), mtp=True)
            entry.publish_commit(entry.prepare_commit(7, (0, 0)), 7, (0, 0))
        source.mtp_len, source.mtp_drafted = 7, 2
        source.mtp_pos.fill_(7)
        stream = Stream([1] * 5, 8, st=source)
        stream.out, stream.context, stream.drafts = [1], [1] * 6, [2, 3]
        sources.append(source)
        destinations.append(destination)
        requests.append(stream)
    inventories = [state_inventory(state, stream=stream, recovery_versions=0)
                   for state, stream in zip(sources, requests)]
    records = tuple((item.metadata, item.specs) for item in inventories)
    for destination in destinations:
        plan = state_restore_plan(destination, records, stream=Stream([1] * 8, 8),
                                  continuation="new-turn", recovery_versions=0)
        for (rank, number), part in plan.targets.items():
            part.tensor.copy_(inventories[rank].parts[number].tensor)
        commit_state_restore(plan)
        assert destination.mtp_len == destination.mtp_pos.item() == 5 and destination.mtp_drafted == 0
        entry = destination.compact_indexer.entries[-1]
        if entry is not None:
            assert entry.base is None and entry.proposal_count == 0 and entry.n == 5
            assert torch.equal(entry.raw, bases[len(destination.kc)])

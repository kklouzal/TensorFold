"""Pointer construction, ownership, physical pages and exact native byte domains.

These CPU controls do not execute native device pointers or qualify a kernel.
"""

import gc
from types import SimpleNamespace as NS
import weakref

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.history_owner import (
    HistoryArena, HistoryPageClass, OwnerBuffers, allocate_target_caches,
)
from tensorfold.families.qwen4_exp.cuda.history_placement import (
    HistoryPlacement, native_owner_workspace_bytes, owner_workspace_bytes,
)
from tensorfold.families.qwen4_exp.cuda.state import State, Buffers, KVNumericError
from tensorfold.families.qwen4_exp.kv_formats import get_pair
from tests.test_sm70_rotorquant_state import model

pytestmark = pytest.mark.torch


def _weights(rank=0):
    weights = model(8)
    weights.cfg.heads, weights.cfg.index_heads = 3, 4
    weights.cfg.index_budget = 2048
    weights.layers = [NS(index=layer, linear=False) for layer in range(12)]
    weights.meta.update(full=NS(kv_heads=2), rank=rank, execution_provider="sm70", history_layout="unique")
    weights.comm = NS(world=8, rank=rank)
    return weights


def test_header_only_native_workspace_has_exact_bytes_and_safe_inactive_views():
    weights = _weights()
    workspace = OwnerBuffers(weights, 7, NS(idw=2052))
    before = workspace.nbytes
    workspace.prepare_native(pair=get_pair("int8"))
    expected = native_owner_workspace_bytes(7)
    assert workspace.nbytes - before == expected["device_bytes"] == 228 * 7 + 4
    assert workspace.descriptors_host.nbytes == expected["host_staging_bytes"] == 128 * 7
    assert workspace.native_states is None
    workspace.clear_native_inputs(7)
    views = workspace.native_views(7)
    assert torch.count_nonzero(views["row_table"]) == 0
    assert torch.count_nonzero(views["q_all"]) == torch.count_nonzero(views["selected_all"]) == 0
    assert views["q_all"].shape == (8, 7, 3, 256)
    assert views["out"].data_ptr() == views["q_all"].data_ptr()
    assert views["local_stats"].data_ptr() == views["local_values"].data_ptr()
    assert views["all_stats"].data_ptr() == views["all_values"].data_ptr()
    assert all(tensor.is_contiguous() for tensor in views.values())


def test_uncaptured_native_calls_use_only_the_finite_startup_bucket_catalogue():
    workspace = OwnerBuffers(_weights(), 7, NS(idw=2052))
    workspace.prepare_native(pair=get_pair("int8"), buckets=(1, 4, 7))
    assert workspace.native_buckets == (1, 4, 7)
    assert [workspace.native_bucket(count) for count in range(1, 8)] == [1, 4, 4, 4, 7, 7, 7]
    with pytest.raises(ValueError, match="not prepared"):
        workspace.native_run(2)
    with pytest.raises(ValueError, match="cannot change"):
        workspace.prepare_native(pair=get_pair("int8"), buckets=(1, 2, 4, 7))
    workspace.prepare_native(pair=get_pair("int8"))
    assert workspace.native_buckets == (1, 4, 7)


def test_native_workspace_plan_includes_actual_tables_and_pinned_packet_once():
    text = NS(heads=3, head_dim=256, index_heads=4, index_dim=128,
              index_budget=2048, index_ratio=4, context=262148)
    reference = owner_workspace_bytes(text, 32)
    native = owner_workspace_bytes(text, 32, native=True)
    expected = native_owner_workspace_bytes(32)
    assert native["persistent_bytes"] - reference["persistent_bytes"] == expected["device_bytes"]
    assert native["host_staging_bytes"] == expected["host_staging_bytes"]
    assert native["temporary_bytes"] == reference["temporary_bytes"]


@pytest.mark.parametrize("prefill,rows", [(False, 32), (True, 4096)])
def test_header_only_factory_and_receipt_match_actual_tensor_domains(prefill, rows):
    text = {"hidden_size": 2560, "num_attention_heads": 24, "num_key_value_heads": 2,
            "head_dim": 256, "indexer_budget": 2048, "indexer_compress_ratio": 4}
    comm = NS(rank=3, world=8)
    workspace = OwnerBuffers.from_text(text, rows=rows, prefill=prefill, idw=2052,
                                       world=8, rank=3, device="cpu", comm=comm)
    before = workspace.allocation_record()
    workspace.prepare_native(pair=get_pair("int8"))
    after = workspace.allocation_record()
    assert workspace.rows == min(rows, 256)
    assert after["prefill"] == prefill and after["rank"] == 3
    assert before["native_tensor_bytes"] == 0
    assert after["communication_tensor_bytes"] == before["device_tensor_bytes"]
    extra = native_owner_workspace_bytes(rows)
    assert after["native_tensor_bytes"] == extra["device_bytes"]
    assert after["host_staging_bytes"] == extra["host_staging_bytes"]
    assert after["device_tensor_bytes"] == sum(t.nbytes for name, t in vars(workspace).items()
                                               if isinstance(t, torch.Tensor) and name != "descriptors_host")


def test_factory_journals_canonical_partial_owner_before_allocation_failure(monkeypatch):
    text = {"hidden_size": 2560, "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256}
    registered, order = [], []
    def register(owner):
        order.append("register")
        assert owner.queries is None and not owner._construction_started
        registered.append(owner)
    original, calls = torch.empty, 0
    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        order.append("allocate")
        assert len(registered) == 1
        if calls == 2:
            raise MemoryError("injected second owner allocation failure")
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "empty", fail_second)
    with pytest.raises(MemoryError, match="second owner allocation"):
        OwnerBuffers.from_text(text, rows=4, prefill=False, idw=2052, world=8, rank=0,
                                device="cpu", comm=NS(rank=0, world=8), register=register)
    owner = registered[0]
    assert order[0] == "register" and not owner._base_ready
    assert owner.allocation_record()["device_tensor_bytes"] > 0
    owner.close()
    assert owner.allocation_record()["device_tensor_bytes"] == 0
    assert owner.closed


def test_descriptor_words_name_retained_storage_and_zero_every_inactive_row():
    weights = _weights()
    states = [State(weights, 257, 4, "int8", slot=slot) for slot in range(2)]
    workspace = OwnerBuffers(weights, 8, NS(idw=2052))
    workspace.prepare_native(states)
    layer = NS(index=0)
    workspace.stage_native_descriptors([(states[0], 0, 3), (states[1], 3, 5)], layer, 0, 5, bucket_rows=8)
    for row, (state, offset) in enumerate([(states[0], 0), (states[0], 1), (states[0], 2),
                                           (states[1], 0), (states[1], 1)]):
        cache = state.kc[0]
        words = workspace.descriptors[row].tolist()
        assert words[:8] == [cache.k.data_ptr(), cache.v.data_ptr(), cache.ks.data_ptr(), cache.vs.data_ptr(),
                             cache.page_offsets.data_ptr(), state.history_generations.data_ptr(),
                             state.kv_status.data_ptr(), state.pos_dev.data_ptr()]
        assert words[9:14] == [state.slot_generation, state.history_generation, 257,
                               cache.capacity, cache.page_offsets.numel()]
        assert words[14:] == [offset, state.history_placement.indexer_owner(state.history_slot, 0)]
    assert workspace.descriptors[0, 8] == workspace.descriptors[2, 8]
    assert workspace.descriptors[0, 8] != workspace.descriptors[3, 8]
    assert workspace.lane_rows[:2].tolist() == [3, 2]
    assert torch.count_nonzero(workspace.descriptors[5:]) == 0
    addresses = workspace.descriptors.data_ptr(), workspace.global_stats.data_ptr()
    workspace.stage_native_descriptors([(states[0], 0, 3)], layer, 0, 1, bucket_rows=8)
    assert torch.count_nonzero(workspace.descriptors[1:]) == 0
    assert addresses == (workspace.descriptors.data_ptr(), workspace.global_stats.data_ptr())


def test_catalogue_retains_pointer_owners_until_explicit_retirement():
    weights = _weights()
    state = State(weights, 257, 4, "int8")
    state_reference, data_reference = weakref.ref(state), weakref.ref(state.kc[0].k)
    workspace = OwnerBuffers(weights, 4, NS(idw=2052))
    workspace.prepare_native([state])
    del state
    gc.collect()
    assert state_reference() is not None and data_reference() is not None
    workspace.retire_native()
    gc.collect()
    assert state_reference() is None and data_reference() is None


def test_storage_replacement_needs_retirement_before_catalogue_rebind():
    weights = _weights()
    state = State(weights, 257, 4, "int8")
    workspace = OwnerBuffers(weights, 4, NS(idw=2052))
    workspace.prepare_native([state])
    state.kc[0].page_offsets = state.kc[0].page_offsets.clone()
    before = workspace.descriptors.clone()
    with pytest.raises(RuntimeError, match="canonical storage catalogue"):
        workspace.stage_native_descriptors([(state, 0, 1)], NS(index=0), 0, 1)
    assert torch.equal(before, workspace.descriptors)
    with pytest.raises(RuntimeError, match="retire"):
        workspace.bind_native_states([state])
    workspace.retire_native()
    workspace.bind_native_states([state])
    workspace.stage_native_descriptors([(state, 0, 1)], NS(index=0), 0, 1)


def test_slot_generation_has_exact_device_words_and_no_signed_wrap():
    weights = _weights()
    state = State(weights, 257, 4, "int8")
    assert state.history_generations.nbytes == 16 and state.history_generations.tolist() == [0, 0]
    state.reset(weights)
    assert state.slot_generation == 1 and state.history_generations.tolist() == [1, 0]
    state.slot_generation = 2**63 - 1
    state.history_generations[0].fill_(state.slot_generation)
    before = state.conv.clone()
    with pytest.raises(RuntimeError, match="invalid or exhausted"):
        state.reset(weights)
    assert state.slot_generation == 2**63 - 1 and state.history_generations[0] == 2**63 - 1
    assert torch.equal(state.conv, before)


def test_new_admission_retires_registered_native_readers_before_word_publication(monkeypatch):
    weights = _weights()
    state = State(weights, 257, 4, "int8")
    workspace = OwnerBuffers(weights, 4, NS(idw=2052))
    workspace.prepare_native([state])
    events = []
    def retire():
        assert state.slot_generation == 0 and state.history_generations[0] == 0
        events.append("retire")
    original = torch.Tensor.fill_
    def publish(tensor, value):
        if tensor.data_ptr() == state.history_generations.data_ptr() and tensor.numel() == 1:
            events.append("publish")
        return original(tensor, value)
    monkeypatch.setattr(workspace, "wait_native_reuse", retire)
    monkeypatch.setattr(torch.Tensor, "fill_", publish)
    assert state.begin_sequence_generation() == 1
    assert events == ["retire", "publish"]
    state.reset(weights, new_sequence=False)
    assert state.slot_generation == 1
    workspace.retire_native()
    assert workspace not in state.history_native_readers
    assert set(state.history_native_readers) == {state.compact_indexer}


def test_ambiguous_generation_publication_cannot_be_erased_by_reset(monkeypatch):
    weights = _weights()
    weights.layers = model(8).layers
    state = State(weights, 257, 4, "int8")
    state.conv.fill_(17)
    before = state.conv.clone()
    original = torch.Tensor.fill_
    def fail_after_write(tensor, value):
        result = original(tensor, value)
        if tensor.data_ptr() == state.history_generations.data_ptr() and tensor.numel() == 1:
            raise RuntimeError("injected ambiguous generation publication")
        return result
    monkeypatch.setattr(torch.Tensor, "fill_", fail_after_write)
    with pytest.raises(RuntimeError, match="injected ambiguous"):
        state.begin_sequence_generation()
    assert state.history_generations[0] == 1 and state.slot_generation == 0
    assert torch.equal(state.conv, before)
    for operation in (state.kv_begin, state.begin_sequence_generation,
                      lambda: state.reset(weights), lambda: state.reset(weights, new_sequence=False)):
        with pytest.raises(RuntimeError, match="exclusive retirement"):
            operation()


def test_generation_poison_survives_diagnostic_allocation_failure(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import state as state_module

    weights = _weights()
    state = State(weights, 257, 4, "int8")
    primary = RuntimeError("actual ambiguous generation write")
    original = torch.Tensor.fill_

    def fail_after_write(tensor, value):
        result = original(tensor, value)
        if tensor.data_ptr() == state.history_generations.data_ptr() and tensor.numel() == 1:
            raise primary
        return result

    def no_diagnostic(*_args):
        raise MemoryError("diagnostic allocation failed")

    monkeypatch.setattr(torch.Tensor, "fill_", fail_after_write)
    monkeypatch.setattr(state_module, "RuntimeError", no_diagnostic, raising=False)
    with pytest.raises(type(primary)) as caught:
        state.begin_sequence_generation()
    assert caught.value is primary
    assert state._history_error is state._kv_error is primary
    assert state.history_generations[0] == 1 and state.slot_generation == 0
    with pytest.raises(type(primary)) as repeated:
        state.reset(weights, new_sequence=False)
    assert repeated.value is primary


def test_generated_descriptor_order_change_fails_before_pointer_table_creation(monkeypatch):
    from tensorfold.cuda.sm70 import paged_attention_descriptor

    workspace = OwnerBuffers(_weights(), 4, NS(idw=2052))
    monkeypatch.setattr(paged_attention_descriptor, "FIELDS", tuple(reversed(paged_attention_descriptor.FIELDS)))
    with pytest.raises(RuntimeError, match="generated schema"):
        workspace.prepare_native(pair=get_pair("int8"))
    assert workspace.descriptors is None


def test_descriptor_error_is_named_and_rejected_before_state_publication():
    state = State(_weights(), 257, 4, "int8")
    state.kv_begin()
    state.kv_status.copy_(torch.tensor([32, 3], dtype=torch.int32))
    with pytest.raises(KVNumericError, match="descriptor"):
        state.kv_check()
    assert state.pos == 0


def test_same_preallocated_native_pipeline_runs_each_phase_in_dependency_order(monkeypatch):
    from tensorfold.cuda.sm70 import paged_attention

    weights = _weights()
    state = State(weights, 257, 4, "int8")
    events = []
    def gather(send, receive):
        events.append("gather")
        receive.copy_(send.expand(8, *send.shape))
    weights.comm.all_gather = gather
    workspace = OwnerBuffers(weights, 4, NS(idw=2052))
    workspace.prepare_native([state])
    workspace.stage_native_descriptors([(state, 0, 4)], NS(index=0), 0, 4)
    views = workspace.native_views(4)
    def statistics(query, selection, table, out, **kwargs):
        events.append("statistics")
        assert query.data_ptr() == views["q_all"].data_ptr()
        assert kwargs["layer"] is workspace.native_layer
        out.fill_(1)
    def merge(statistics, table, out, **kwargs):
        events.append("merge_statistics")
        assert torch.all(statistics == 1)
        out.fill_(2)
    def values(query, selection, table, normalizers, out, **kwargs):
        events.append("normalized_values")
        assert torch.all(normalizers == 2)
        out.fill_(3)
    def output(values, table, out, **kwargs):
        events.append("merge_values")
        assert torch.all(values == 3)
        out.fill_(4)
    monkeypatch.setattr(paged_attention, "local_statistics", statistics)
    monkeypatch.setattr(paged_attention, "merge_statistics", merge)
    monkeypatch.setattr(paged_attention, "normalized_partial", values)
    monkeypatch.setattr(paged_attention, "merge_values", output)
    assert torch.all(workspace.native_run(4) == 4)
    assert events == ["statistics", "gather", "merge_statistics", "normalized_values", "gather", "merge_values"]


def test_ownership_generation_exhaustion_precedes_every_page_copy_or_publication():
    weights = _weights()
    weights.comm.broadcast = lambda *args: pytest.fail("exhausted generation copied a page")
    weights.comm.barrier = lambda: pytest.fail("exhausted generation entered transfer rendezvous")
    weights.comm.all_gather = lambda *args: pytest.fail("exhausted generation entered a control vote")
    state = State(weights, 257, 4, "int8")
    state.history_generation = 2**63 - 1
    state.history_generations[1].fill_(state.history_generation)
    before = [cache.page_offsets.clone() for cache in state.kc]
    state.begin_history_migration(4)
    with pytest.raises(OverflowError, match="ownership generation exhausted"):
        state.migrate_history_bundle()
    assert state.history_generation == 2**63 - 1
    assert state.history_generations[1] == 2**63 - 1
    assert all(torch.equal(cache.page_offsets, prior) for cache, prior in zip(state.kc, before))


def test_invalid_descriptor_lane_frame_does_not_modify_an_already_staged_table():
    weights = _weights()
    state = State(weights, 257, 4, "int8")
    workspace = OwnerBuffers(weights, 4, NS(idw=2052))
    workspace.prepare_native([state])
    workspace.stage_native_descriptors([(state, 0, 4)], NS(index=0), 0, 4)
    before = workspace.descriptors.clone()
    for segments in ([(state, 0, 1), (state, 2, 4)], [(state, 0, 3), (state, 2, 4)]):
        with pytest.raises(ValueError, match="gap or overlap"):
            workspace.stage_native_descriptors(segments, NS(index=0), 0, 4)
        assert torch.equal(workspace.descriptors, before)


@pytest.mark.parametrize("actual_rows", [3, 4])
def test_actual_buffers_adopt_prepared_role_workspaces_without_reallocation(monkeypatch, actual_rows):
    weights = _weights()
    weights.device = torch.device("cpu")
    weights.cfg.low, weights.cfg.experts, weights.cfg.top_k = 32, 8, 2
    weights.cfg.moe_width, weights.cfg.shared_width = 512, 512
    weights.cfg.heads_per_ngram, weights.cfg.ple_dim = 8, 512
    weights.head = NS(n=128)
    prepared = {role: OwnerBuffers(weights, 4, NS(idw=2052), prefill=role == "prefill")
                for role in ("decode", "mtp", "prefill")}
    pointers = {role: workspace.queries.data_ptr() for role, workspace in prepared.items()}
    weights.meta["owner_workspaces"] = prepared
    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("a prepared owner workspace was allocated twice")
    monkeypatch.setattr(OwnerBuffers, "__init__", unexpected_allocation)
    for role in prepared:
        buffers = Buffers(weights, actual_rows, 257, prefill=role == "prefill", owner_role=role)
        assert buffers.history_owner is prepared[role]
        assert buffers.history_owner.queries.data_ptr() == pointers[role]


@pytest.mark.parametrize("rows", [1, 127, 128, 129, 257, 517])
@pytest.mark.parametrize("degree", [1, 2, 4])
def test_full_typed_page_reservations_keep_logical_tail_bounds(rows, degree):
    placement, pair = HistoryPlacement(8, 2, 12, degree=degree), get_pair("int8")
    page_class = HistoryPageClass(pair, 256, 128)
    for rank in range(8):
        caches = allocate_target_caches(rows, 256, "cpu", pair=pair, placement=placement, slot=2, rank=rank)
        arena = caches[0].arena
        assert isinstance(arena, HistoryArena) and arena.rank == rank
        assert arena.capacity % 128 == 0 and arena.page_class == page_class
        assert arena.free_pages == 0 and arena.reserved_pages == arena.page_capacity
        assert arena.nbytes == arena.page_capacity * page_class.nbytes
        assert sum(cache.reserved_rows for cache in caches) == arena.capacity
        with pytest.raises(ValueError, match="unavailable"):
            arena.reserve(page_class, 1)
        for cache in caches:
            assert cache.logical_capacity == rows
            offsets, valid = cache.local_indices(torch.tensor([rows - 1, rows, (rows + 127) // 128 * 128 - 1]))
            assert not valid[1]
            if rows % 128:
                assert not valid[2]
            else:
                assert valid[2] == valid[0]
            assert (offsets >= 0).all()
            with pytest.raises(ValueError, match="logical history"):
                cache.append(rows, torch.zeros((1, 1, 256), dtype=torch.int8),
                             torch.zeros((1, 1, 256), dtype=torch.int8),
                             torch.zeros((1, 1, 8), dtype=torch.float16),
                             torch.zeros((1, 1, 8), dtype=torch.float16))

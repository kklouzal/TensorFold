"""Head-group transport preserves global-indexed buffers and phase ordering."""

from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70 import paged_attention
from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers
from tensorfold.families.qwen4_exp.cuda.owner_execution import FULL_COLLECTIVE, HEAD_COLLECTIVE
from tensorfold.families.qwen4_exp.kv_formats import get_pair
from tests.test_native_owner_adapter import _weights


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("policy", [FULL_COLLECTIVE, HEAD_COLLECTIVE])
def test_native_head_transport_keeps_global_rank_slices_and_original_merge_inputs(monkeypatch, rank, policy):
    weights = _weights(rank)
    owner = OwnerBuffers(weights, 3, NS(idw=2052))
    owner.prepare_native(pair=get_pair("int8"))
    first = rank // 4 * 4
    world = 4 if policy == HEAD_COLLECTIVE else 8
    members = range(first, first + 4) if world == 4 else range(8)
    events = []

    def gather(send, target):
        assert target.is_contiguous() and target.shape[0] == world
        # Rank values are distinct: an accidental compact-vs-global reindex
        # changes the receiver's observed merge input on the second KV head.
        for local, global_rank in enumerate(members):
            target[local].fill_(global_rank + 1)
        events.append(("gather", tuple(target.shape)))

    comm = NS(world=world, rank=rank % world, all_gather=gather)
    if world == 8:
        comm = weights.comm
        comm.all_gather = gather
    owner.configure_native_transport(policy, comm)
    views = owner.native_views(3)
    owner.queries.fill_(float("nan"))
    send = views["q_all"][rank]
    send.fill_(rank + 1)
    owner.gather_native(send, views["q_all"])
    assert send.data_ptr() == views["q_all"][first:first + 4][rank % 4].data_ptr()
    assert [int(views["q_all"][peer, 0, 0, 0]) for peer in members] == [peer + 1 for peer in members]
    if world == 4:
        assert torch.isnan(views["q_all"][4 if first == 0 else 0]).all()
    owner.partials.fill_(float("nan"))

    def local(query, selection, table, out, **kwargs):
        assert kwargs["group_start"] == first and kwargs["layer"] is owner.native_layer
        events.append(("statistics",))
        out.fill_(rank + 1)

    def merge(statistics, table, out, **kwargs):
        assert statistics.shape[0] == 8 and kwargs["group_start"] == first
        assert [int(statistics[peer, 0, 0, 0]) for peer in range(first, first + 4)] == list(range(first + 1, first + 5))
        if world == 4:
            assert torch.isnan(statistics[4 if first == 0 else 0]).all()
        events.append(("merge_statistics",))
        out.fill_(1)

    def values(query, selection, table, normalizers, out, **kwargs):
        assert kwargs["group_start"] == first and torch.all(normalizers == 1)
        events.append(("values",))
        out.fill_(rank + 1)

    def output(values, table, out, **kwargs):
        assert values.shape[0] == 8 and kwargs["rank"] == rank
        assert [int(values[peer, 0, 0, 0]) for peer in range(first, first + 4)] == list(range(first + 1, first + 5))
        events.append(("merge_values",))
        out.fill_(sum(range(first + 1, first + 5)))

    monkeypatch.setattr(paged_attention, "local_statistics", local)
    monkeypatch.setattr(paged_attention, "merge_statistics", merge)
    monkeypatch.setattr(paged_attention, "normalized_partial", values)
    monkeypatch.setattr(paged_attention, "merge_values", output)
    assert torch.all(owner.native_run(3) == sum(range(first + 1, first + 5)))
    assert [event[0] for event in events] == ["gather", "statistics", "gather", "merge_statistics", "values", "gather", "merge_values"]
    with pytest.raises(RuntimeError, match="immutable"):
        owner.configure_native_transport(policy, comm)


@pytest.mark.parametrize("world,rank,policy", [(8, 4, HEAD_COLLECTIVE), (4, 4, HEAD_COLLECTIVE),
                                               (4, 0, FULL_COLLECTIVE), (4, 0, "unknown")])
def test_mismatched_group_is_rejected_before_any_collective(world, rank, policy):
    owner = OwnerBuffers(_weights(4), 3, NS(idw=2052))
    with pytest.raises(ValueError, match="fixed head/rank"):
        owner.configure_native_transport(policy, NS(world=world, rank=rank))


@pytest.mark.parametrize("rank", [0, 4, 7])
def test_startup_warms_head_queries_and_full_canonical_selection_in_the_used_types(rank):
    weights = _weights(rank)
    events = []
    weights.comm.all_gather = lambda send, target: events.append(("selection", send.dtype, tuple(target.shape)))
    group = NS(world=4, rank=rank % 4,
               all_gather=lambda send, target: events.append(("query", send.dtype, tuple(target.shape))))
    owner = OwnerBuffers(weights, 3, NS(idw=2052))
    owner.prepare_native(pair=get_pair("int8"))
    owner.configure_native_transport(HEAD_COLLECTIVE, group)
    for rows in owner.native_buckets:
        owner.clear_native_inputs(rows)
        owner.warm_native_transport(rows)
    assert events == [entry for rows in (1, 2, 3) for entry in
                      (("query", torch.float16, (4, rows, 3, 256)),
                       ("selection", torch.int32, (8, rows, 2054)))]
    owner.native_states = {}
    with pytest.raises(RuntimeError, match="unbound startup"):
        owner.warm_native_transport(3)


def test_all_eligible_reference_counts_and_types_are_warmed_on_full8_even_when_head4_is_selected():
    weights = _weights(4)
    events = []
    def gather(send, target):
        assert torch.count_nonzero(send) == 0 and target.shape[0] == 8
        assert send.untyped_storage().data_ptr() != target.untyped_storage().data_ptr()
        events.append((send.dtype, tuple(send.shape), tuple(target.shape)))
    weights.comm.all_gather = gather
    group = NS(world=4, rank=0, all_gather=lambda *args: pytest.fail("reference warmup used subgroup"))
    owner = OwnerBuffers(weights, 5, NS(idw=2052))
    owner.prepare_native(pair=get_pair("int8"), buckets=(1, 4, 5))
    owner.configure_native_transport(HEAD_COLLECTIVE, group)
    owner.warm_reference_transport()
    expected = [(dtype, shape, (8, *shape)) for rows in range(1, 6) for dtype, shape in (
        (torch.float16, (rows, 3, 256)), (torch.int32, (rows, 2054)),
        (torch.float32, (rows, 12, 2)), (torch.float32, (rows, 12, 256)))]
    assert events == expected and owner.reference_transport_rows == 5
    with pytest.raises(RuntimeError, match="untouched startup"):
        owner.warm_reference_transport()

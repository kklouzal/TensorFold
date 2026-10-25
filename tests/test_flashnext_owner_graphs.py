"""Bounded catalogue/ownership controls; CUDA numerical qualification is separate."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.cuda import owner_graphs


@pytest.mark.parametrize("rows,expected", [(1, (1,)), (18, (1, 2, 4, 8, 16, 18)),
                                           (40, (1, 2, 4, 8, 16, 32, 40))])
def test_row_catalogue_has_no_state_or_context_cartesian_product(rows, expected):
    assert owner_graphs.row_buckets(rows) == expected


@pytest.mark.parametrize("rows", [0, -1, True, 1.5])
def test_invalid_row_envelope_is_rejected(rows):
    with pytest.raises(ValueError):
        owner_graphs.row_buckets(rows)


@pytest.fixture
def prepared_owner(monkeypatch):
    import torch
    from tensorfold.cuda import cohort

    status = SimpleNamespace(free=1 << 20, active=False, notifications=[], ends=0)
    graphs = []

    class Graph:
        def __init__(self):
            self.replays = self.resets = 0
            graphs.append(self)

        def capture_begin(self, *, pool, capture_error_mode):
            assert not status.active and capture_error_mode == "global"
            status.active = True

        def capture_end(self):
            assert status.active
            status.active = False
            status.ends += 1
            status.free -= 16

        def replay(self):
            self.replays += 1

        def reset(self):
            self.resets += 1

    stream = SimpleNamespace(wait_stream=lambda other: None)
    cuda = SimpleNamespace(Stream=lambda **kwargs: stream, current_stream=lambda device: stream,
                           graph_pool_handle=lambda: (1, 0), CUDAGraph=Graph,
                           stream=lambda other: nullcontext(), synchronize=lambda device: None,
                           empty_cache=lambda: None, mem_get_info=lambda device: (status.free, 2 << 20))
    fake = SimpleNamespace(cuda=cuda, int64=torch.int64,
                           empty=lambda shape, **kwargs: torch.empty(shape, dtype=kwargs["dtype"]))
    monkeypatch.setattr(owner_graphs, "torch", fake)
    monkeypatch.setattr(cohort, "notify_failure", lambda: status.notifications.append(True))
    calls = []
    output = torch.zeros((6, 3, 256))
    side = SimpleNamespace(bits=8, codec=0)
    owner = SimpleNamespace(queries=torch.zeros((8, 6, 3, 256)), heads=3, head_dim=256,
                            selection=torch.zeros((6, 4), dtype=torch.int32), native_page_tokens=128,
                            native_pair=SimpleNamespace(key=side, value=side),
                            clear_native_inputs=lambda rows: calls.append(("clear", rows)),
                            wait_native_inputs=lambda: calls.append(("wait",)),
                            native_run=lambda rows: calls.append(("native", rows)),
                            mark_native_submitted=lambda: calls.append(("submitted",)),
                            native_views=lambda rows: {"out": output[:rows]})
    comm = SimpleNamespace(world=8, all_gather=lambda source, target: target.fill_(source.item()))
    return owner, comm, status, calls, graphs


def test_fixed_catalogue_replays_without_python_recapture_and_eager_uses_same_body(prepared_owner):
    owner, comm, status, calls, graphs = prepared_owner
    catalogue = owner_graphs.OwnerAttentionGraphs(owner, rows=6, budget_bytes=256, comm=comm)
    catalogue.prepare(quiescent=lambda: True)
    assert list(catalogue.graphs) == [1, 2, 4, 6] and len(graphs) == 4
    assert [call for call in calls if call[0] == "native"] == [
        ("native", rows) for rows in (1, 1, 2, 2, 4, 4, 6, 6)]
    before = list(calls)
    assert catalogue.run(3, graph=True).shape == (3, 3, 256)
    assert calls[len(before):] == [("wait",)]
    assert catalogue.graphs[4].replays == 1 and len(graphs) == 4
    assert catalogue.run(3, graph=False).shape == (3, 3, 256)
    assert calls[-2:] == [("wait",), ("native", 4)]
    assert catalogue.record()["vote_workspace_bytes"] == 72
    catalogue.close()
    assert all(graph.resets == 1 for graph in graphs)
    assert not status.notifications


def test_quiescence_is_required_before_capture_and_failure_keeps_owners(prepared_owner):
    owner, comm, status, calls, graphs = prepared_owner
    catalogue = owner_graphs.OwnerAttentionGraphs(owner, rows=6, budget_bytes=256, comm=comm)
    with pytest.raises(RuntimeError, match="quiescent"):
        catalogue.prepare(quiescent=lambda: False)
    assert not graphs and not calls
    assert status.notifications == [True] and catalogue.failure is not None


def test_exceeded_graph_budget_is_a_failed_startup_not_an_unaccounted_eager_fallback(prepared_owner):
    owner, comm, status, calls, graphs = prepared_owner
    catalogue = owner_graphs.OwnerAttentionGraphs(owner, rows=6, budget_bytes=8, comm=comm)
    with pytest.raises(MemoryError, match="budget"):
        catalogue.prepare(quiescent=lambda: True)
    assert not catalogue.prepared and len(catalogue.graphs) == 1
    assert catalogue.room_bytes == 16 and status.notifications == [True]
    with pytest.raises(RuntimeError, match="terminal"):
        catalogue.close()
    assert all(graph.resets == 0 for graph in graphs)


def test_failed_capture_ends_once_and_retains_the_unpublished_graph(prepared_owner):
    owner, comm, status, calls, graphs = prepared_owner
    original = owner.native_run

    def fail_in_capture(rows):
        if status.active:
            raise OSError("capture-time native submission failed")
        original(rows)

    owner.native_run = fail_in_capture
    catalogue = owner_graphs.OwnerAttentionGraphs(owner, rows=6, budget_bytes=256, comm=comm)
    with pytest.raises(OSError):
        catalogue.prepare(quiescent=lambda: True)
    assert status.ends == 1 and not status.active
    assert len(catalogue.pending) == 1 and catalogue.pending[0] is graphs[0]
    assert not catalogue.graphs and status.notifications == [True]


def test_rank_geometry_disagreement_is_refused_before_any_shape_dependent_capture(prepared_owner):
    owner, comm, status, calls, graphs = prepared_owner

    def gather(source, target):
        value = int(source.item())
        target.fill_(value)
        if value == 6:
            target[3] = 8  # One peer requested a different maximum execution catalogue.

    comm.all_gather = gather
    catalogue = owner_graphs.OwnerAttentionGraphs(owner, rows=6, budget_bytes=256, comm=comm)
    with pytest.raises(ValueError, match="differ across"):
        catalogue.prepare(quiescent=lambda: True)
    assert not calls and not graphs and status.notifications == [True]


def test_an_explicit_measured_bucket_set_covers_intermediate_rows_without_extra_captures(prepared_owner):
    owner, comm, status, calls, graphs = prepared_owner
    catalogue = owner_graphs.OwnerAttentionGraphs(owner, rows=6, budget_bytes=256, comm=comm, buckets=[2, 6])
    catalogue.prepare(quiescent=lambda: True)
    assert list(catalogue.graphs) == [2, 6] and len(graphs) == 2
    assert catalogue.bucket(1) == 2 and catalogue.bucket(3) == 6
    catalogue.run(3, graph=True)
    assert len(graphs) == 2 and catalogue.graphs[6].replays == 1


@pytest.mark.parametrize("buckets", [[1, 4], [1, 6, 6], [6, 2], [True, 6], []])
def test_invalid_bucket_policy_is_rejected_before_native_capture(prepared_owner, buckets):
    owner, comm, status, calls, graphs = prepared_owner
    with pytest.raises(ValueError, match="buckets"):
        owner_graphs.OwnerAttentionGraphs(owner, rows=6, budget_bytes=256, comm=comm, buckets=buckets)
    assert not calls and not graphs


def test_settled_private_pool_spare_is_measured_without_exposing_addresses(prepared_owner):
    owner, comm, status, calls, graphs = prepared_owner
    cuda = owner_graphs.torch.cuda
    cuda.memory = SimpleNamespace(get_allocator_backend=lambda: "native")
    cuda.memory_snapshot = lambda **kwargs: [{"device": 0, "segment_pool_id": (1, 0),
                                              "total_size": 32, "allocated_size": 16, "active_size": 16}]
    catalogue = owner_graphs.OwnerAttentionGraphs(owner, rows=6, budget_bytes=256, comm=comm)
    catalogue.prepare(quiescent=lambda: True)
    record = catalogue.record()
    assert record["private_pool_spare_bytes"] == 16
    assert record["allocator"] == {"backend": "native", "accounting": "settled-native-private-pool",
                                     "private_reserved_bytes": 32, "private_allocated_bytes": 16}


def test_unsettled_private_pool_is_a_failed_memory_proof_not_zero_spare(prepared_owner):
    owner, comm, status, calls, graphs = prepared_owner
    cuda = owner_graphs.torch.cuda
    cuda.memory = SimpleNamespace(get_allocator_backend=lambda: "native")
    cuda.memory_snapshot = lambda **kwargs: [{"device": 0, "segment_pool_id": (1, 0),
                                              "total_size": 32, "allocated_size": 8, "active_size": 16}]
    catalogue = owner_graphs.OwnerAttentionGraphs(owner, rows=6, budget_bytes=256, comm=comm)
    with pytest.raises(ValueError, match="settled"):
        catalogue.prepare(quiescent=lambda: True)
    assert catalogue.private_spare_bytes is None and catalogue.failure is not None

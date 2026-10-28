"""Bounded catalogue/ownership controls; CUDA numerical qualification is separate."""

from contextlib import contextmanager, nullcontext
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
    owner = SimpleNamespace(queries=torch.zeros((8, 6, 3, 256)), rows=6, rank=0, heads=3, head_dim=256,
                            native_buckets=(1, 2, 4, 6),
                            selection=torch.zeros((6, 4), dtype=torch.int32), native_page_tokens=128,
                            native_pair=SimpleNamespace(key=side, value=side),
                            clear_native_inputs=lambda rows: calls.append(("clear", rows)),
                            warm_native_transport=lambda rows: calls.append(("transport", rows)),
                            wait_native_inputs=lambda: calls.append(("wait",)),
                            native_run=lambda rows: calls.append(("native", rows)),
                            mark_native_submitted=lambda: calls.append(("submitted",)),
                            native_views=lambda rows: {"out": output[:rows]})
    comm = SimpleNamespace(world=8, all_gather=lambda source, target: target.fill_(source.item()))
    comm.rank, comm.device, comm.closed, comm.comm = 0, owner.queries.device, False, SimpleNamespace(value=123)
    owner.comm = owner.native_comm = comm
    owner.native_collective = "tp8-allgather-v1"
    owner.graph_tensors = lambda: {"queries": owner.queries, "selection": owner.selection}
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


def test_transport_and_native_warmup_join_the_side_stream_before_capture(prepared_owner, monkeypatch):
    owner, comm, status, _calls, _graphs = prepared_owner
    events = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            events.append(("dependency", self.name, other.name))

    main, side = Stream("main"), Stream("side")
    current = [main]

    @contextmanager
    def on_stream(stream):
        previous, current[0] = current[0], stream
        try:
            yield
        finally:
            current[0] = previous

    cuda = owner_graphs.torch.cuda
    monkeypatch.setattr(cuda, "Stream", lambda **_kwargs: side)
    monkeypatch.setattr(cuda, "current_stream", lambda _device: current[0])
    monkeypatch.setattr(cuda, "stream", on_stream)
    monkeypatch.setattr(cuda, "synchronize", lambda _device: events.append(("fence", current[0].name)))

    def observe(name, operation):
        def run(*args):
            events.append((name, current[0].name, status.active))
            return operation(*args)
        return run

    for name in ("clear_native_inputs", "warm_native_transport", "wait_native_inputs", "native_run"):
        monkeypatch.setattr(owner, name, observe(name, getattr(owner, name)))
    catalogue = owner_graphs.OwnerAttentionGraphs(owner, rows=6, budget_bytes=256, comm=comm)
    catalogue.prepare(quiescent=lambda: True)
    native = [index for index, event in enumerate(events) if event[0] == "native_run"]
    for eager, capture in zip(native[::2], native[1::2], strict=True):
        assert events[eager] == ("native_run", "side", False)
        assert events[capture] == ("native_run", "side", True)
        assert ("fence", "main") in events[eager + 1:capture]
    assert len(native) == 2 * len(catalogue.buckets)
    assert all(event[1:] == ("side", False) for event in events
               if event[0] in ("clear_native_inputs", "warm_native_transport", "wait_native_inputs"))
    assert events.count(("dependency", "side", "main")) == len(catalogue.buckets)
    assert current[0] is main and not status.active
    catalogue.close()


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


@pytest.mark.parametrize("mutation", ["tensor", "storage", "shape", "communicator", "policy", "codec"])
def test_final_adoption_refuses_rebound_graph_resources_and_retains_old_storage(prepared_owner, mutation):
    import torch

    owner, comm, status, calls, graphs = prepared_owner
    catalogue = owner_graphs.OwnerAttentionGraphs(owner, rows=6, budget_bytes=256, comm=comm)
    catalogue.prepare(quiescent=lambda: True)
    old_storage = catalogue.tensor_leases["queries"][1]
    old_pointer, old_bytes = old_storage.data_ptr(), old_storage.nbytes()
    if mutation == "tensor":
        owner.queries = torch.zeros_like(owner.queries)
    elif mutation == "storage":
        owner.queries.set_(torch.zeros_like(owner.queries).untyped_storage())
    elif mutation == "shape":
        owner.queries.transpose_(0, 1)
    elif mutation == "communicator":
        comm.comm.value = 456
    elif mutation == "policy":
        owner.native_collective = "head4-allgather-v1"
    else:
        owner.native_pair = SimpleNamespace(key=SimpleNamespace(bits=4, codec=10), value=owner.native_pair.value)
    with pytest.raises(RuntimeError, match="contract changed"):
        catalogue.validate_adoption()
    assert old_storage.data_ptr() == old_pointer and old_storage.nbytes() == old_bytes
    assert status.notifications == [True] and catalogue.failure is not None
    assert all(graph.resets == 0 for graph in graphs)


def test_partial_graph_constructor_releases_lease_only_after_retirement(prepared_owner, monkeypatch):
    owner, comm, status, calls, graphs = prepared_owner
    partial_owner = object.__new__(owner_graphs.OwnerAttentionGraphs)

    def fail_stream(**kwargs):
        raise MemoryError("stream allocation")

    monkeypatch.setattr(owner_graphs.torch.cuda, "Stream", fail_stream)
    with pytest.raises(MemoryError, match="stream allocation"):
        partial_owner.__init__(owner, rows=6, budget_bytes=256, comm=comm)
    assert partial_owner.tensor_leases and partial_owner.comm_leases
    partial_owner.close()
    assert partial_owner.closed and not partial_owner.tensor_leases and not partial_owner.comm_leases


def test_construction_hooks_cover_capture_and_replay_and_retire_before_clearing_maps(prepared_owner):
    owner, comm, status, calls, graphs = prepared_owner
    bindings, events = [], []

    def bind_protocol(target, attribute, label, *, refusal):
        assert refusal == () and label == "owner/capture"
        original = getattr(target, attribute)
        def capture(graph, run):
            assert graph in target.pending and not status.active  # Graph owner exists before GO.
            events.append(("protocol", label))
            return original(graph, run)
        bindings.append((target, attribute, original))
        setattr(target, attribute, capture)

    def bind_mapping(target, key, label):
        original = target[key]
        def replay():
            events.append(("submit", label))
            return original()
        bindings.append((target, key, original))
        target[key] = replay

    def retire(target):
        for entry in tuple(reversed(bindings)):
            if entry[0] is target:
                if isinstance(target, dict):
                    assert entry[1] in target
                    target[entry[1]] = entry[2]
                else:
                    setattr(target, entry[1], entry[2])
                bindings.remove(entry)
        events.append(("retire", id(target)))

    comm._profile_construction = SimpleNamespace(bind_protocol=bind_protocol,
        bind_mapping_submit=bind_mapping, retire_bindings=retire)
    catalogue = owner_graphs.OwnerAttentionGraphs(owner, rows=6, budget_bytes=256, comm=comm)
    catalogue.prepare(quiescent=lambda: True)
    assert events == [("protocol", "owner/capture")] * 4
    catalogue.run(3, graph=True)
    assert events[-1] == ("submit", "owner/replay")
    replay_map = catalogue.replays
    for graph in graphs:
        original = graph.reset
        def reset(original=original):
            assert not bindings  # Early retirement handles constructor rollback before context exit.
            events.append(("reset",))
            original()
        graph.reset = reset
    catalogue.close()
    assert not replay_map and not bindings and all(graph.resets == 1 for graph in graphs)
    assert events[5:7] == [("retire", id(catalogue)), ("retire", id(replay_map))]


@pytest.mark.parametrize("run_fails,end_fails", [(False, False), (True, False), (False, True), (True, True)])
def test_actual_capture_outcome_preserves_primary_and_always_ends_accepted_begin(monkeypatch, run_fails, end_fails):
    from tensorfold.families.qwen4_exp.cuda import owner_graphs
    owner = object.__new__(owner_graphs.OwnerAttentionGraphs)
    owner.pool = object()
    calls = []
    primary, secondary = RuntimeError("run"), RuntimeError("end")
    class Graph:
        def capture_begin(self, **kwargs):
            calls.append("begin")
        def capture_end(self):
            calls.append("end")
            if end_fails:
                raise secondary
    graph = Graph()
    def run():
        calls.append("run")
        if run_fails:
            raise primary
    if not run_fails and not end_fails:
        monkeypatch.setattr(owner_graphs, "raise_failures", lambda *args: pytest.fail("success cannot transport None primary"))
        assert owner._capture(graph, run) is graph
    else:
        with pytest.raises(RuntimeError) as caught:
            owner._capture(graph, run)
        assert caught.value is (primary if run_fails else secondary)
        if run_fails and end_fails:
            assert caught.value.__cause__ is secondary
    assert calls == ["begin", "run", "end"]


def test_capture_outcome_allocation_failure_precedes_any_native_acquisition(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import owner_graphs
    owner = object.__new__(owner_graphs.OwnerAttentionGraphs)
    owner.pool = object()
    calls = []
    error = MemoryError("outcome")
    def fail(*args):
        raise error
    monkeypatch.setattr(owner_graphs, "_outcome_slot", fail)
    graph = SimpleNamespace(capture_begin=lambda **kwargs: calls.append("begin"),
                            capture_end=lambda: calls.append("end"))
    with pytest.raises(MemoryError) as caught:
        owner._capture(graph, lambda: calls.append("run"))
    assert caught.value is error and not calls


@pytest.mark.parametrize("allocation_fails", [False, True])
def test_actual_poison_retains_operation_primary_when_notification_or_outcome_allocation_fails(monkeypatch, allocation_fails):
    from tensorfold.cuda import cohort
    from tensorfold.families.qwen4_exp.cuda import owner_graphs
    owner = object.__new__(owner_graphs.OwnerAttentionGraphs)
    owner.failure = None
    primary, notification, allocation = RuntimeError("operation"), RuntimeError("notify"), MemoryError("outcome")
    calls = []
    def notify():
        calls.append("notify")
        raise notification
    monkeypatch.setattr(cohort, "notify_failure", notify)
    if allocation_fails:
        def fail(error=None):
            assert error is notification
            raise allocation
        monkeypatch.setattr(owner_graphs, "_outcome_slot", fail)
    with pytest.raises(RuntimeError) as caught:
        owner._poison(primary)
    assert caught.value is primary and owner.failure is primary and calls == ["notify"]
    if allocation_fails:
        assert primary.__context__ is allocation and allocation.__context__ is notification
    else:
        assert primary.__cause__ is notification
    owner._poison(RuntimeError("later"))
    assert owner.failure is primary and calls == ["notify"]

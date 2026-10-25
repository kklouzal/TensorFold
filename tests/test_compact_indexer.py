"""Compact tails/pools against the independent full-history reference domain."""

from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70 import family_ops as ops
from tensorfold.cuda.sm70.compact_indexer import publish
from tensorfold.families.qwen4_exp.cuda.compact_indexer import (
    Cache, CompactIndexer, Span, Staging, stage_mtp_indexer_frame, validate_mtp_indexer_frame,
    stage_target_indexer_frame, validate_target_indexer_frame,
)
from tensorfold.families.qwen4_exp.cuda.history_placement import compact_cache_bytes, compact_stage_bytes


def cache(*, capacity=128, proposals=0):
    generations = torch.zeros(2, dtype=torch.int64)
    status = torch.tensor([0, -2], dtype=torch.int32)
    return Cache(capacity, 128, "cpu", generations, status, layer=3,
                 proposal_rows=proposals, mtp=proposals > 0)


def begin(entry, stage, raw, coords, *, mtp=False):
    stage.begin(mtp=mtp)
    target, axes, epoch = stage.span(0, 0, len(raw), mtp=mtp)
    span = Span(entry.n, target, axes, stage, epoch, mtp, (0, 0))
    entry.begin(span, raw, coords)
    return span


def expected_tail(raw, coords, length):
    values = torch.zeros((4, 128), dtype=torch.float16)
    axes = torch.zeros((4, 3), dtype=torch.int64)
    for position in range(length - length % 4, length):
        values[position % 4] = raw[position]
        axes[position % 4] = coords[position]
    return values, axes


@pytest.mark.parametrize("chunks", [[1] * 27, [3, 7, 1, 9, 7], [8, 8, 8, 3]])
def test_all_group_splits_and_every_cut_match_full_history(chunks):
    torch.manual_seed(917)
    entry, stage = cache(), Staging(max(chunks), 12, "cpu", mtp=False)
    raw = torch.randn((128, 128)).half()
    coords = torch.stack((torch.arange(128) * 5 + 7, torch.arange(128) * 2 + 9,
                          torch.arange(128) * 3 + 11), 1).long()
    gamma, frequencies = torch.linspace(0.7, 1.3, 128), torch.linspace(0.01, 0.2, 32)
    pooled, reference = torch.zeros((32, 128), dtype=torch.float16), torch.zeros((32, 128), dtype=torch.float16)
    full = torch.zeros_like(raw)
    scratch = NS(ratio=4)
    for count in chunks:
        start = entry.n
        begin(entry, stage, raw[start:start + count], coords[start:start + count])
        entry.pool(pooled, gamma, frequencies, 1e-6, (11, 11, 10), 1.0)
        full[start:start + count].copy_(raw[start:start + count])
        ops.qsa_pool(full, reference, torch.tensor([start], dtype=torch.int32), gamma, frequencies,
                     1e-6, scratch, count, rope=coords, length=len(coords))
        for cut in range(start, start + count + 1):
            snapshot = entry.snapshot(cut)
            expected, axes = expected_tail(raw, coords, cut)
            assert torch.equal(snapshot["raw"], expected)
            assert torch.equal(snapshot["coordinates"], axes)
            assert snapshot["pooled_valid"] == cut // 4
        entry.publish_commit(entry.prepare_commit(start + count, (0, 0)), start + count, (0, 0))
        assert torch.equal(pooled[:entry.n // 4], reference[:entry.n // 4])
        assert entry.raw.shape == (4, 128)


@pytest.mark.parametrize("accepted", [0, 1, 3, 4, 8])
def test_rejected_members_never_become_valid_and_rollback_preserves_pending_tail(accepted):
    torch.manual_seed(12)
    entry, stage = cache(), Staging(9, 12, "cpu", mtp=False)
    raw, coords = torch.randn(32, 128).half(), torch.arange(96).reshape(32, 3).long()
    pooled, gamma, frequencies = torch.zeros(32, 128).half(), torch.ones(128), torch.tensor([0.3])
    begin(entry, stage, raw[:3], coords[:3])
    entry.pool(pooled, gamma, frequencies, 1e-6, (1, 1, 1), 1.0)
    entry.publish_commit(entry.prepare_commit(3, (0, 0)), 3, (0, 0))
    before = entry.snapshot(3)
    begin(entry, stage, raw[3:12], coords[3:12])
    entry.pool(pooled, gamma, frequencies, 1e-6, (1, 1, 1), 1.0)
    if accepted:
        entry.publish_commit(entry.prepare_commit(3 + accepted, (0, 0)), 3 + accepted, (0, 0))
        with pytest.raises(ValueError, match="rejected"):
            entry.snapshot(12)
    else:
        entry.span = None
    entry.restore(before, 3, (0, 0))
    assert entry.n == 3 and torch.equal(entry.raw, before["raw"])
    assert torch.equal(entry.coordinates, before["coordinates"])
    replacement = raw[3:8].neg()
    begin(entry, stage, replacement, coords[3:8])
    entry.pool(pooled, gamma, frequencies, 1e-6, (1, 1, 1), 1.0)
    expected = torch.cat((raw[:3], replacement), 0)
    full, oracle = torch.zeros(128, 128).half(), torch.zeros_like(pooled)
    full[:8].copy_(expected)
    ops.qsa_pool(full, oracle, torch.tensor([0], dtype=torch.int32), gamma, frequencies,
                 1e-6, NS(ratio=4), 8, rope=coords, length=len(coords), sections=(1, 1, 1))
    assert torch.equal(pooled[:2], oracle[:2])


def test_mtp_base_and_every_intermediate_cut_survive_shared_stage_reuse():
    entry, stage = cache(proposals=8), Staging(8, 13, "cpu", mtp=True)
    raw = torch.arange(128 * 10).reshape(10, 128).half()
    coords = torch.arange(30).reshape(10, 3).long() + (1 << 33)
    pooled, gamma, freq = torch.zeros(32, 128).half(), torch.ones(128), torch.tensor([0.1])
    begin(entry, stage, raw[:3], coords[:3], mtp=True)
    entry.pool(pooled, gamma, freq, 1e-6, (1, 1, 1), 1.0)
    entry.publish_commit(entry.prepare_commit(3, (0, 0)), 3, (0, 0))
    base = entry.snapshot(3)
    entry.begin_proposals((0, 0))
    for position in range(3, 10):
        begin(entry, stage, raw[position:position + 1], coords[position:position + 1], mtp=True)
        entry.pool(pooled, gamma, freq, 1e-6, (1, 1, 1), 1.0)
        entry.publish_commit(entry.prepare_commit(position + 1, (0, 0)), position + 1, (0, 0))
        stage.mtp.fill_(-777)  # Another request is free to use the shared execution arena.
    for cut in range(3, 11):
        expected, axes = expected_tail(raw, coords, cut)
        assert torch.equal(entry.snapshot(cut)["raw"], expected)
        assert torch.equal(entry.snapshot(cut)["coordinates"], axes)
    entry.publish_commit(entry.prepare_commit(3, (0, 0)), 3, (0, 0))
    assert entry.base is None and torch.equal(entry.raw, base["raw"])


def test_owned_bytes_are_context_independent_and_match_actual_tensor_extents():
    for capacity in (32, 262144, 1048576):
        assert cache(capacity=capacity).nbytes == compact_cache_bytes(128)
        assert cache(capacity=capacity, proposals=4).nbytes == compact_cache_bytes(128, mtp=True, proposal_rows=4)
    for mtp, only in ((False, False), (True, False), (True, True)):
        stage = Staging(32, 13, "cpu", mtp=mtp, mtp_only=only)
        assert stage.nbytes == compact_stage_bytes(32, 13, mtp=mtp, mtp_only=only)


def test_stale_generation_does_not_publish_model_bytes():
    source, destination = torch.ones(4, 128).half(), torch.full((4, 128), 7).half()
    status = torch.tensor([0, -2], dtype=torch.int32)
    publish(source, destination, 0, torch.tensor([4, 9]), (3, 9), status, 7)
    assert torch.all(destination == 7) and status.tolist() == [32, 7]


@pytest.mark.parametrize("failure", [None, "operation", "event", "operation_event", "allocator_record"])
def test_prefix_transfer_retires_kv_pool_uses_on_both_states_even_without_owned_indexer(monkeypatch, failure):
    placement = NS(target_layers=1, indexer_owner=lambda *args, **kwargs: 0)
    owners = [CompactIndexer(placement, slot, 7, 32, 1, 128, "cpu",
                             torch.zeros(2, dtype=torch.int64), torch.tensor([0, -2], dtype=torch.int32),
                             proposal_rows=0, layer_ids=(3,)) for slot in (0, 1)]
    assert all(owner.entries == [None] for owner in owners)
    events, current = [], [2]
    primary, retirement = MemoryError("partial pooled transfer"), ValueError("transfer retirement")
    class Event:
        def __init__(self, label):
            self.label, self.stream = label, 1
        def record(self, stream):
            events.append(("record", self.label, stream.cuda_stream))
            if failure in ("event", "operation_event") and self.label == "destination":
                raise retirement
            self.stream = stream.cuda_stream
    class Stream:
        def __init__(self, number):
            self.cuda_stream = number
        def wait_event(self, event):
            events.append(("wait", self.cuda_stream, event.label, event.stream))
    for owner, label in zip(owners, ("source", "destination")):
        owner._transfer_done, owner._transfer_stream = Event(label), 1
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: Stream(current[0]))
    source_kv, destination_pool = torch.zeros(3, 256), torch.ones(3, 128)
    borrowers = (source_kv, destination_pool)
    def record(tensor, stream):
        events.append(("borrow", id(tensor), stream.cuda_stream))
        if failure == "allocator_record" and tensor is destination_pool:
            raise retirement
    monkeypatch.setattr(torch.Tensor, "record_stream", record)
    def transfer():
        with owners[0].transferring(owners[1], borrowers):
            events.append(("issued",))
            if failure in ("operation", "operation_event"):
                raise primary
    if failure is None:
        transfer()
    else:
        with pytest.raises(BaseException) as caught:
            transfer()
        assert caught.value is (primary if failure in ("operation", "operation_event") else retirement)
        if failure == "operation_event":
            assert caught.value.__cause__ is retirement
    assert events[:3] == [("wait", 2, "source", 1), ("wait", 2, "destination", 1), ("issued",)]
    assert ("borrow", id(source_kv), 2) in events and ("borrow", id(destination_pool), 2) in events
    current[0] = 3
    if failure in ("event", "operation_event", "allocator_record"):
        expected = primary if failure == "operation_event" else retirement
        for owner in owners:
            assert owner.retirement_failure is expected and owner._transfer_borrowers is borrowers
            with pytest.raises(BaseException) as caught:
                owner.check()
            assert caught.value is expected
            with pytest.raises(BaseException) as caught:
                owner.wait_native_reuse()
            assert caught.value is expected
    else:
        from tensorfold.families.qwen4_exp.cuda.state import State

        for owner in owners:
            assert owner._transfer_borrowers == () and owner.retirement_failure is None
            state = NS(_history_error=None, _kv_error=None, _resize_error=None,
                       compact_indexer=owner, dflash2=None, kv_status=None)
            State.kv_begin(state)  # The real boundary before subsequent persistent writes.
            owner.wait_native_reuse()  # Existing registered reader boundary for reset/new admission.
        assert events[-2:] == [("wait", 3, "source", 2), ("wait", 3, "destination", 2)]


def test_prefix_transfer_wait_failure_cannot_be_reclassified_as_fresh_exhaustion(monkeypatch):
    placement = NS(target_layers=1, indexer_owner=lambda *args, **kwargs: 0)
    owner = CompactIndexer(placement, 0, 7, 32, 1, 128, "cpu", torch.zeros(2, dtype=torch.int64),
                           torch.tensor([0, -2], dtype=torch.int32), proposal_rows=0, layer_ids=(3,))
    failure = MemoryError("failed transfer wait")
    class Stream:
        cuda_stream = 2
        def wait_event(self, event):
            raise failure
    owner._transfer_done, owner._transfer_stream = object(), 1
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: Stream())
    with pytest.raises(MemoryError) as caught:
        owner.check()
    assert caught.value is owner.retirement_failure is failure
    with pytest.raises(MemoryError) as caught:
        owner.wait_native_reuse()
    assert caught.value is failure


@pytest.mark.parametrize("control", ["generations", "status"])
def test_publication_rejects_destination_control_alias_before_any_write(control):
    generations, status = torch.tensor([3, 5], dtype=torch.int64), torch.tensor([0, -2], dtype=torch.int32)
    destination = generations if control == "generations" else status
    before = destination.clone()
    with pytest.raises(ValueError, match="aliases"):
        publish(torch.zeros_like(destination), destination, 0, generations, (3, 5), status, 7)
    assert torch.equal(destination, before)


@pytest.mark.parametrize("span_read,failure", [(False, None), (True, None), (False, "clone"),
                                               (True, "pool"), (True, "pool_mark")])
def test_cross_stream_cut_reads_extend_cache_and_shared_staging_retirement(monkeypatch, span_read, failure):
    entry, staging = cache(), Staging(4, 12, "cpu", mtp=False)
    events, current = [], [1]
    class Event:
        def __init__(self, label):
            self.label, self.stream = label, 1
        def record(self, stream):
            if failure == "pool_mark" and self.label == "staging":
                raise RuntimeError("staging event publication")
            self.stream = stream.cuda_stream
            events.append(("record", self.label, self.stream))
    class Stream:
        def __init__(self, number):
            self.cuda_stream = number
        def wait_event(self, event):
            events.append(("wait", self.cuda_stream, event.label, event.stream))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: Stream(current[0]))
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda tensor: True))
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: None)
    entry.done, entry.pending, entry._stream = Event("cache"), True, 1
    staging._events[0], staging._streams[0] = Event("staging"), 1
    if span_read:
        staging.target_epoch = 1
        staging.target[0].fill_(7)
        staging.coordinates.fill_(99)
        entry.span = Span(0, staging.target[0], staging.coordinates, staging, 1, False, (0, 0))
        length = 2
    else:
        entry.n = length = 2
        entry.raw[:2].fill_(7)
        entry.coordinates[:2].fill_(99)
    current[0] = 2
    if failure == "clone":
        original = torch.Tensor.clone
        def clone(tensor, *args, **kwargs):
            if tensor is entry.coordinates:
                raise MemoryError("second tail clone allocation")
            return original(tensor, *args, **kwargs)
        monkeypatch.setattr(torch.Tensor, "clone", clone)
        with pytest.raises(MemoryError, match="second tail clone"):
            entry.tail(length)
    elif failure in ("pool", "pool_mark"):
        def failed_norm(*args, **kwargs):
            raise MemoryError("normalization allocation after raw gathers")
        monkeypatch.setattr(ops, "rms", failed_norm)
        with pytest.raises(MemoryError, match="normalization allocation") as caught:
            entry.pool(torch.zeros(32, 128).half(), torch.ones(128), torch.ones(1), 1e-6, (1, 1, 1), 1.)
        if failure == "pool_mark":
            assert isinstance(caught.value.__cause__, RuntimeError)
            assert entry._retirement_error is caught.value
            assert staging._errors[0] is caught.value.__cause__
            current[0] = 3
            with pytest.raises(RuntimeError, match="staging event publication"):
                staging.begin(mtp=False)
            assert staging.target_epoch == 1
            return
    else:
        raw, coordinates = entry.tail(length)
    assert ("wait", 2, "cache", 1) in events and ("record", "cache", 2) in events
    current[0] = 3
    entry.wait_native_reuse()
    assert events[-1] == ("wait", 3, "cache", 2)
    if span_read:
        assert ("record", "staging", 2) in events
        staging.begin(mtp=False)
        assert events[-1] == ("wait", 3, "staging", 2)
        staging.target.fill_(-1)
        if failure is None:
            assert torch.all(raw[:2] == 7) and torch.all(coordinates[:2] == 99)


def test_numeric_latch_diagnostic_exhaustion_cannot_make_failed_state_healthy(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import state as state_module
    from tests.test_native_owner_adapter import _weights

    state = state_module.State(_weights(), 32, 4, "int8")
    state.kv_status.copy_(torch.tensor([32, 3], dtype=torch.int32))
    state.kv_begin()
    original = state_module.KVNumericError
    def exhausted(*args):
        raise MemoryError("numeric diagnostic constructor")
    with monkeypatch.context() as fault:
        fault.setattr(state_module, "KVNumericError", exhausted)
        for _ in range(2):
            with pytest.raises(MemoryError, match="diagnostic constructor"):
                state.kv_check()
            assert state._kv_pending and state._kv_error is None
            assert state.kv_status.tolist() == [32, 3]
    with pytest.raises(original, match="descriptor"):
        state.kv_check()
    assert state._kv_error is not None


def test_issued_writer_with_failed_fence_blocks_reset_and_other_stream_reuse(monkeypatch):
    entry = cache()
    error = MemoryError("writer fence publication")
    waits = []
    class Stream:
        cuda_stream = 2
        def wait_event(self, event):
            waits.append(event)
    class Event:
        def record(self, stream):
            raise error
    entry.done, entry.pending, entry._stream = Event(), True, 1
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: Stream())
    with pytest.raises(MemoryError) as caught:
        entry._copy(torch.ones_like(entry.raw), entry.raw, (0, 0))
    assert caught.value is error and torch.all(entry.raw == 1)
    assert entry._retirement_error is error and len(waits) == 1
    with pytest.raises(MemoryError) as reused:
        entry.reset((0, 0))
    assert reused.value is error and len(waits) == 1 and torch.all(entry.raw == 1)


@pytest.mark.parametrize("failure", [False, True])
def test_explicit_coordinate_consumer_on_non_indexer_owner_retires_shared_role(monkeypatch, failure):
    from tensorfold.families.qwen4_exp.cuda import history_owner

    stage, current, events = Staging(4, 13, "cpu", mtp=True), [2], []
    stage.coordinates.fill_(91)
    class Event:
        stream = 1
        def record(self, stream):
            self.stream = stream.cuda_stream
            events.append(("record", self.stream))
    class Stream:
        def __init__(self, number):
            self.cuda_stream = number
        def wait_event(self, event):
            events.append(("wait", self.cuda_stream, event.stream))
    stage._events[0], stage._streams[0] = Event(), 1
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: Stream(current[0]))
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: None)
    buffers = NS(target_indexer_coordinates=stage.coordinates, compact_indexer_stage=stage)
    state = NS(history_rank=7)
    issued = []
    def consumer(layer, weights, segments, buffers, rows, *, mtp):
        assert segments[0][0].history_rank == 7  # This cache's canonical indexer is another rank.
        issued.append(buffers.target_indexer_coordinates.clone())
        if failure:
            raise MemoryError("after explicit coordinate read")
        return issued[-1]
    monkeypatch.setattr(history_owner, "_attention_layer", consumer)
    if failure:
        with pytest.raises(MemoryError, match="coordinate read"):
            history_owner.attention_layer(None, None, [(state, 0, 4)], buffers, 4)
    else:
        history_owner.attention_layer(None, None, [(state, 0, 4)], buffers, 4)
    assert events == [("wait", 2, 1), ("record", 2)]
    current[0] = 3
    stage.begin(mtp=False)
    assert events[-1] == ("wait", 3, 2)
    stage.coordinates.zero_()
    assert torch.all(issued[0] == 91)


@pytest.mark.parametrize("mtp", [False, True])
def test_coordinate_frame_binds_own_rows_generations_and_coordinate_storage(mtp):
    prefix = "mtp" if mtp else "target"
    staging = Staging(4, 13, "cpu", mtp=True)
    storage = staging.mtp_coordinates if mtp else staging.coordinates
    buffers = NS(rows=4, compact_indexer_stage=staging, **{prefix + "_indexer_coordinate_storage": storage,
                            prefix + "_indexer_coordinates": None, prefix + "_indexer_frame": None,
                            prefix + "_indexer_frame_id": 0})
    state = NS(pos=19, mtp_len=18, slot_generation=2, history_generation=3)
    segs, coords = [(state, 0, 2)], torch.tensor([[100, 20, 30], [200, 40, 60]], dtype=torch.int64)
    stage_fn, validate = ((stage_mtp_indexer_frame, validate_mtp_indexer_frame) if mtp else
                          (stage_target_indexer_frame, validate_target_indexer_frame))
    stage_fn(buffers, segs, coords)
    coords.zero_()
    assert validate(buffers, segs).tolist() == [[100, 20, 30], [200, 40, 60]]
    state.slot_generation += 1
    with pytest.raises(ValueError, match="generation"):
        validate(buffers, segs)
    stage_fn(buffers, segs, None)
    assert validate(buffers, segs) is None


def test_unique_prefix_relocates_captured_tail_after_shared_staging_was_reused():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from tensorfold.families.qwen4_exp.cuda.state import State
    from tests.test_native_owner_adapter import _weights

    barrier = Barrier(8, timeout=15)
    packets = [None] * 8
    payload = [None]

    def communicator(rank):
        def gather(source, target):
            packets[rank] = source.clone()
            barrier.wait()
            target.copy_(torch.stack(packets))
            barrier.wait()
        def broadcast(value, root):
            if rank == root:
                payload[0] = value.clone()
            barrier.wait()
            value.copy_(payload[0])
            barrier.wait()
        return NS(rank=rank, world=8, all_gather=gather, broadcast=broadcast)

    raw = torch.arange(16 * 128).reshape(16, 128).half()
    coordinates = torch.stack((torch.arange(16) + 10, torch.arange(16) + 90, torch.arange(16) + 130), 1).long()
    key = torch.arange(16 * 256).reshape(16, 1, 256).to(torch.int8)
    scale = torch.ones((16, 1, 8), dtype=torch.float16)

    def worker(rank):
        weights = _weights(rank)
        weights.comm = communicator(rank)
        source, target = State(weights, 32, 4, "int8", slot=0), State(weights, 32, 4, "int8", slot=1)
        stage = Staging(16, 13, "cpu", mtp=True)
        stage.begin(mtp=False)
        for ordinal, kv in enumerate(source.kc):
            kv.append(0, key[:13], -key[:13], scale[:13], scale[:13])
            entry = source.compact_indexer.entries[ordinal]
            if entry is not None:
                source.compact_indexer.append(ordinal, 0, raw[:13] + ordinal, coordinates[:13],
                    stage, 0, 13, mtp=False, expected=(0, 0))
                source.pooled[ordinal].fill_(ordinal + 0.25)
        source.set_pos(13)
        stage.begin(mtp=True)
        source.mtp_kc.append(0, key[:9], -key[:9], scale[:9], scale[:9])
        if source.compact_indexer.entries[-1] is not None:
            source.compact_indexer.append(12, 0, raw[:9] + 12, coordinates[:9],
                stage, 0, 9, mtp=True, expected=(0, 0))
            source.mtp_pooled.fill_(12.25)
        source.set_mtp_len(9)
        snapshot = {**source.snapshot(), "pos": 10, "mtp_len": 6,
                    **source.indexer_snapshot_metadata(10, 6)}
        # A real kept snapshot must own its tail, not a reused execution view.
        stage.begin(mtp=False)
        stage.begin(mtp=True)
        stage.target.fill_(-900)
        stage.mtp.fill_(-900)
        target.begin_sequence_generation()
        relocated = target.copy_prefix(source, 10, 6, indexer_snapshot=snapshot["compact_indexer"])
        target.restore({**snapshot, "compact_indexer": relocated})
        assert target.pos == 10 and target.mtp_len == 6
        for ordinal, entry in enumerate(target.compact_indexer.entries):
            if entry is None:
                continue
            length = 6 if ordinal == 12 else 10
            expected, axes = expected_tail(raw + ordinal, coordinates, length)
            assert torch.equal(entry.raw, expected) and torch.equal(entry.coordinates, axes)
            pool = target.mtp_pooled if ordinal == 12 else target.pooled[ordinal]
            assert torch.all(pool[:length // 4] == ordinal + 0.25)
        for kv in target.kc:
            local, owned = kv.local_indices(torch.arange(10, dtype=torch.int64))
            if bool(owned.any()):
                assert torch.equal(kv.k[local[owned]], key[:10][owned])
        return True

    with ThreadPoolExecutor(max_workers=8) as workers:
        assert all(workers.map(worker, range(8)))


@pytest.mark.parametrize("failure", ["staging", "snapshot", "transfer", "retired_clone", "unretired_clone",
                                    "unretired_value", "reader_wait"])
def test_prefix_cohort_refuses_before_mutation_or_contains_uncertain_transfer(monkeypatch, failure):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, local
    from tensorfold.cuda.memory_gate import NoRoom
    from tensorfold.families.qwen4_exp.cuda import history_owner
    from tensorfold.families.qwen4_exp.cuda.state import State
    from tests.test_native_owner_adapter import _weights

    barrier, thread = Barrier(8, timeout=10), local()
    packets, payload, broadcasts, poisoned = [None] * 8, [None], [], []
    def communicator(rank):
        def gather(source, target):
            packets[rank] = source.clone()
            barrier.wait()
            target.copy_(torch.stack(packets))
            barrier.wait()
        def broadcast(value, root):
            broadcasts.append(rank)
            if failure == "transfer" and rank == 3:
                raise RuntimeError("uncertain prefix publication")
            if rank == root:
                payload[0] = value.clone()
            barrier.wait()
            value.copy_(payload[0])
            barrier.wait()
        return NS(rank=rank, world=8, all_gather=gather, broadcast=broadcast)
    states = []
    for rank in range(8):
        weights = _weights(rank)
        weights.comm = communicator(rank)
        source, target = State(weights, 32, 4, "int8", slot=0), State(weights, 32, 4, "int8", slot=1)
        snapshot = source.indexer_snapshot_metadata(0, 0)["compact_indexer"]
        if failure == "snapshot" and rank == 3:
            tail = next(value for value in snapshot["tails"] if value is not None)
            tail["raw"] = torch.zeros(3, 128).half()
        states.append((source, target, snapshot))
    original_clone = torch.Tensor.clone
    cause = MemoryError("second snapshot allocation")
    if failure in ("retired_clone", "unretired_clone"):
        entry = next(value for value in states[3][0].compact_indexer.entries if value is not None)
        def clone(tensor, *args, **kwargs):
            if getattr(thread, "rank", None) == 3 and tensor is entry.coordinates:
                raise cause
            return original_clone(tensor, *args, **kwargs)
        monkeypatch.setattr(torch.Tensor, "clone", clone)
        if failure == "unretired_clone":
            def failed_retirement(*args, **kwargs):
                raise ValueError("snapshot reader retirement failed")
            monkeypatch.setattr(entry, "_read_done", failed_retirement)
    elif failure == "unretired_value":
        entry = next(value for value in states[3][0].compact_indexer.entries if value is not None)
        cause = ValueError("previous event retirement failed")
        entry._retirement_error = cause
    elif failure == "reader_wait":
        cause = MemoryError("previous prefix reader wait failed")
        class Reader:
            def wait_native_reuse(self):
                raise cause
        reader = Reader()
        states[3][0].history_native_readers.add(reader)
    original = history_owner.KVCache
    def allocate(*args, **kwargs):
        if failure == "staging" and thread.rank == 3:
            raise MemoryError("one-rank staging exhaustion")
        return original(*args, **kwargs)
    monkeypatch.setattr(history_owner, "KVCache", allocate)
    def worker(rank):
        thread.rank = rank
        source, target, snapshot = states[rank]
        def poison(error):
            poisoned.append(rank)
            barrier.abort()  # Models the managed supervisor, no normal ACK waits.
        try:
            target.copy_prefix(source, 0, 0,
                indexer_snapshot=None if failure in ("retired_clone", "unretired_clone") else snapshot,
                poison_control=poison)
        except BaseException as error:
            return error
        pytest.fail("a partial cohort published a successful prefix")
    with ThreadPoolExecutor(max_workers=8) as workers:
        errors = list(workers.map(worker, range(8)))
    if failure in ("staging", "snapshot", "retired_clone"):
        assert not broadcasts and not poisoned
        expected = ValueError if failure == "snapshot" else NoRoom
        assert all(isinstance(error, expected) for error in errors)
        assert all(target._history_error is None and target.pos == 0 for _, target, _ in states)
        assert all(source.compact_indexer.retirement_failure is None for source, _, _ in states)
        if failure == "retired_clone":
            monkeypatch.setattr(torch.Tensor, "clone", original_clone)
            def retry(rank):
                source, target, snapshot = states[rank]
                return target.copy_prefix(source, 0, 0, indexer_snapshot=snapshot)
            with ThreadPoolExecutor(max_workers=8) as workers:
                assert all(result["schema"] == "tensorfold.compact-indexer-v1" for result in workers.map(retry, range(8)))
    else:
        assert sorted(poisoned) == list(range(8))
        assert bool(broadcasts) == (failure == "transfer")
        assert all(target._history_error is error for (_, target, _), error in zip(states, errors))
        if failure in ("unretired_clone", "unretired_value", "reader_wait"):
            assert errors[3] is cause

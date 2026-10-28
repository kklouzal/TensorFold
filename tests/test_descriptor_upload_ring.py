"""DMA host leases are independently retired before pinned-slot reuse."""
from types import SimpleNamespace as NS

import pytest
torch = pytest.importorskip("torch")
from tensorfold.families.qwen4_exp.cuda.descriptor_upload import (  # noqa: E402
    DescriptorUploadRing, selected_slots, host_commitment,
)


class Event:
    def __init__(self):
        self.ready = False
        self.records = self.queries = self.waits = 0
        self.error = None
    def record(self, stream):
        if self.error is not None:
            raise self.error
        self.records += 1
        self.ready = False
    def query(self):
        self.queries += 1
        if self.error is not None:
            raise self.error
        return self.ready
    def synchronize(self):
        self.waits += 1
        if self.error is not None:
            raise self.error
        self.ready = True


@pytest.mark.parametrize("slots", [2, 4, 8])
def test_full_ring_uses_each_disjoint_host_slab_before_waiting(slots):
    ring = DescriptorUploadRing(3, slots, "cpu", event_factory=Event)
    stream = NS(cuda_stream=7)
    snapshots = []
    for index in range(slots):
        host = ring.acquire()
        host.fill_(index + 1)
        snapshots.append(host.clone())
        assert host.data_ptr() == ring.views[index].data_ptr()
        ring.publish(stream)
    assert all(event.waits == 0 for event in ring.events)
    for actual, expected in zip(ring.views, snapshots):
        assert torch.equal(actual, expected)
    first = ring.acquire()
    assert first.data_ptr() == ring.views[0].data_ptr()
    assert ring.events[0].waits == 1 and sum(event.waits for event in ring.events) == 1
    assert host_commitment(3, slots) >= ring.hosts.nbytes


def test_completed_slot_query_avoids_wait_and_cross_stream_joins_latest_write():
    ring = DescriptorUploadRing(3, 2, "cpu", event_factory=Event)
    waits = []
    first = NS(cuda_stream=7, wait_event=lambda event: waits.append(event))
    for _ in range(2):
        ring.acquire()
        ring.publish(first)
    ring.events[1].ready = True
    host = ring.acquire()
    assert host.data_ptr() == ring.views[1].data_ptr()
    assert sum(event.waits for event in ring.events) == 0
    ring.before_gpu_write(first)
    assert waits == []
    ring.before_gpu_write(NS(cuda_stream=9, wait_event=lambda event: waits.append(event)))
    assert waits == [ring.latest_event]


@pytest.mark.parametrize("failure", ["query", "record"])
def test_unknown_event_failure_poison_retains_host_and_refuses_reuse(failure):
    ring = DescriptorUploadRing(3, 2, "cpu", event_factory=Event)
    stream, error = NS(cuda_stream=7), RuntimeError("unknown upload")
    if failure == "query":
        for _ in range(2):
            ring.acquire()
            ring.publish(stream)
        ring.events[0].error = error
        action = ring.acquire
    else:
        ring.acquire()
        ring.events[0].error = error
        def action():
            return ring.publish(stream)
    host = ring.hosts
    with pytest.raises(RuntimeError) as caught:
        action()
    assert caught.value is error and ring.error is error and ring.hosts is host
    with pytest.raises(RuntimeError, match="poisoned"):
        ring.acquire()
    with pytest.raises(RuntimeError, match="global fence"):
        ring.drain()
    ring.drain(globally_fenced=True)
    assert not any(ring.pending) and ring.hosts is host
    with pytest.raises(RuntimeError, match="poisoned"):
        ring.acquire()


def test_retirement_waits_every_pending_slot_and_failure_retains_all_borrowers():
    ring = DescriptorUploadRing(3, 4, "cpu", event_factory=Event)
    for _ in range(4):
        ring.acquire()
        ring.publish(NS(cuda_stream=7))
    ring.drain()
    assert [event.waits for event in ring.events] == [1, 1, 1, 1]
    assert not any(ring.pending)


@pytest.mark.parametrize("slots", ["2", "4", "8"])
def test_explicit_profile_selects_only_prepared_bounded_slot_counts(monkeypatch, slots):
    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", "source-owner-descriptor-ring-trial")
    monkeypatch.setenv("TENSORFOLD_OWNER_DESCRIPTOR_UPLOAD_SLOTS", slots)
    assert selected_slots() == int(slots)


def test_legacy_default_and_invalid_selection(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_OWNER_DESCRIPTOR_UPLOAD_SLOTS", raising=False)
    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", "reference")
    assert selected_slots() == 1
    monkeypatch.setenv("TENSORFOLD_OWNER_DESCRIPTOR_UPLOAD_SLOTS", "4")
    with pytest.raises(ValueError, match="explicit ring"):
        selected_slots()
    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", "source-owner-descriptor-ring-trial")
    monkeypatch.setenv("TENSORFOLD_OWNER_DESCRIPTOR_UPLOAD_SLOTS", "16")
    with pytest.raises(ValueError, match="2,4or8"):
        selected_slots()


@pytest.mark.parametrize("slots", [2, 4, 8])
def test_actual_owner_cpu_staging_keeps_gpu_table_and_all_state_guards(monkeypatch, slots):
    from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers
    from tensorfold.families.qwen4_exp.cuda.state import State
    from tests.test_native_owner_adapter import _weights
    from tensorfold.families.qwen4_exp.cuda.history_placement import native_owner_workspace_bytes

    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", "source-owner-descriptor-ring-trial")
    monkeypatch.setenv("TENSORFOLD_OWNER_DESCRIPTOR_UPLOAD_SLOTS", str(slots))
    weights = _weights()
    state = State(weights, 128, 4, "int8", slot=0)
    owner = OwnerBuffers(weights, 4, NS(idw=2052))
    owner.prepare_native((state,))
    descriptor_pointer = owner.descriptors.data_ptr()
    expected = None
    for index in range(3 * slots):
        owner.stage_native_descriptors([(state, 0, 1)], NS(index=0), 0, 1)
        actual = owner.descriptors.clone()
        if expected is None:
            expected = actual
        assert torch.equal(actual, expected)
        assert owner.descriptors.data_ptr() == descriptor_pointer
        assert owner.descriptors_host.data_ptr() == owner.descriptor_ring.views[index % slots].data_ptr()
    record = owner.native_record()
    assert record["descriptor_upload_slots"] == slots
    assert record["host_descriptor_bytes"] == native_owner_workspace_bytes(4)["host_staging_bytes"]
    assert host_commitment(4, slots) == record["host_descriptor_bytes"] + record["host_ring_metadata_bytes_bound"]
    with pytest.raises(ValueError, match="cannot change"):
        owner.prepare_native((state,), descriptor_upload_slots=2 if slots != 2 else 4)
    owner.clear_native_inputs(4)
    assert torch.count_nonzero(owner.descriptors) == 0
    owner.close()
    assert owner.descriptor_ring is None


def test_partial_host_or_event_acquisition_remains_retirable(monkeypatch):
    ring = object.__new__(DescriptorUploadRing)
    original = torch.zeros
    with monkeypatch.context() as patch:
        patch.setattr(torch, "zeros", lambda *args, **kwargs: (_ for _ in ()).throw(MemoryError("pinned")))
        with pytest.raises(MemoryError, match="pinned"):
            ring.__init__(3, 4, "cpu")
    ring.drain(globally_fenced=True)
    assert ring.hosts is None and not any(ring.pending)
    calls = []
    def event():
        if len(calls) == 2:
            raise MemoryError("event")
        value = Event()
        calls.append(value)
        return value
    with pytest.raises(MemoryError, match="event"):
        ring.__init__(3, 4, "cpu", event_factory=event)
    assert ring.hosts is not None and ring.events == calls
    ring.drain(globally_fenced=True)
    assert ring.hosts.nbytes == original((4, 3, 16), dtype=torch.int64).nbytes


def test_wait_failure_latches_and_retains_every_slot():
    ring = DescriptorUploadRing(3, 2, "cpu", event_factory=Event)
    for _ in range(2):
        ring.acquire()
        ring.publish(NS(cuda_stream=7))
    error = RuntimeError("DMA fence failed")
    # All queries report pending, so reuse must fence the oldest slot.
    def fail():
        raise error
    ring.events[0].synchronize = fail
    with pytest.raises(RuntimeError) as caught:
        ring.acquire()
    assert caught.value is error and ring.error is error
    assert all(ring.pending) and ring.hosts is not None
    with pytest.raises(RuntimeError, match="global fence"):
        ring.drain()
    ring.drain(globally_fenced=True)


def test_ring_profile_has_separate_state_epoch_and_unchanged_native_graph_contract(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import prefix_policy
    from tensorfold.families.qwen4_exp.cuda.owner_preparation import OwnerPreparation
    from tests.test_owner_preparation import specification
    monkeypatch.setenv("TENSORFOLD_SM70_PROFILE", "source-owner-descriptor-ring-trial")
    monkeypatch.delenv("TENSORFOLD_OWNER_DESCRIPTOR_UPLOAD_SLOTS", raising=False)
    prepared = OwnerPreparation(specification(), role_rows={"decode": 32, "mtp": 32, "prefill": 256},
                                manifest_sha256="a" * 64)
    assert prepared.descriptor_upload_slots == prepared.identity()["descriptor_upload_slots"] == 4
    assert prefix_policy.state_epoch(prefix_policy.OWNER_DESCRIPTOR_RING_PROFILE) != prefix_policy.OWNER_TARGET_PRELUDE_STATE

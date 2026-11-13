"""CPU ownership controls; no CUDA timing or DMA correctness qualification."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import descriptor_upload as upload
from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers
from tensorfold.families.qwen4_exp.cuda.owner_preparation import parse_preparation
from tensorfold.families.qwen4_exp.kv_formats import get_pair
from tests.test_native_owner_adapter import _weights


class Stream:
    def __init__(self):
        self.work, self.values, self.done = [], [], 0
        self.waits = 0

    def enqueue(self, host):
        self.work.append(host)

    def retire(self, stop):
        while self.done < stop:
            self.values.append(self.work[self.done].clone())
            self.done += 1

    def synchronize(self):
        self.waits += 1
        self.retire(len(self.work))


class Event:
    def __init__(self):
        self.stream, self.stop, self.waits = None, 0, 0
        self.record_error = self.wait_error = None

    def record(self, stream):
        if self.record_error is not None:
            raise self.record_error
        self.stream, self.stop = stream, len(stream.work)

    def synchronize(self):
        self.waits += 1
        if self.wait_error is not None:
            raise self.wait_error
        self.stream.retire(self.stop)


def ring(monkeypatch):
    stream, events = Stream(), [Event(), Event()]
    zeros = torch.zeros
    pinned = []

    def allocate(*args, **kwargs):
        pinned.append(kwargs.pop('pin_memory'))
        return zeros(*args, **kwargs)

    monkeypatch.setattr(upload.torch, 'zeros', allocate)
    monkeypatch.setattr(upload.torch.cuda, 'Event', lambda: events[1])
    monkeypatch.setattr(upload.torch.cuda, 'current_stream', lambda device: stream)
    host = zeros((3, 16), dtype=torch.int64)
    result = upload.DescriptorUploadRing(host, host.numpy(), events[0], 'cuda:0')
    assert pinned == [True] and [e.waits for e in events] == [1, 1]
    return result, stream, events


def enqueue(owner, stream, value):
    host, words, event = owner.acquire()
    words[:] = value
    owner.begin(stream)
    stream.enqueue(host)
    try:
        owner.submitted(stream)
    except BaseException as error:
        owner.fail(error)
        raise
    return host, event


def test_two_uploads_need_no_host_wait_then_wrapped_slot_has_backpressure(monkeypatch):
    owner, stream, events = ring(monkeypatch)
    first, _ = enqueue(owner, stream, 11)
    second, _ = enqueue(owner, stream, 22)
    assert first is not second and [e.waits for e in events] == [1, 1]
    assert not stream.values
    third, _ = enqueue(owner, stream, 33)
    assert third is first and [e.waits for e in events] == [2, 1]
    assert [int(v[0, 0]) for v in stream.values] == [11]
    enqueue(owner, stream, 44)
    owner.drain()
    assert [int(v[0, 0]) for v in stream.values] == [11, 22, 33, 44]
    assert owner.pending == [False, False]
    assert owner.payload_bytes == 2 * first.nbytes and owner.metadata_bytes > 0


def test_record_failure_retains_source_and_drains_its_actual_stream(monkeypatch):
    owner, stream, events = ring(monkeypatch)
    error = RuntimeError('event record failed after DMA enqueue')
    events[0].record_error = error
    with pytest.raises(RuntimeError) as caught:
        enqueue(owner, stream, 7)
    assert caught.value is error and owner.pending == [True, False]
    with pytest.raises(RuntimeError) as retired:
        owner.drain()
    assert retired.value is error and stream.waits == 1
    assert int(stream.values[0][0, 0]) == 7 and owner.pending == [False, False]
    assert len(owner.hosts) == 2
    with pytest.raises(RuntimeError) as poisoned:
        owner.acquire()
    assert poisoned.value is error


def roots(error):
    seen, stack = {}, [error]
    while stack:
        item = stack.pop()
        if item is None or id(item) in seen:
            continue
        seen[id(item)] = item
        stack.extend((item.__cause__, item.__context__, *getattr(item, 'exceptions', ())))
    return tuple(seen.values())


def test_drain_attempts_both_slots_and_preserves_primary_and_native_roots(monkeypatch):
    owner, stream, events = ring(monkeypatch)
    enqueue(owner, stream, 1)
    enqueue(owner, stream, 2)
    primary, prior = ValueError('copy failure'), LookupError('original root')
    primary.__cause__ = prior
    owner.fail(primary)
    events[0].wait_error, events[1].wait_error = RuntimeError('slot zero'), RuntimeError('slot one')
    with pytest.raises(ValueError) as caught:
        owner.drain()
    assert caught.value is primary and [e.waits for e in events] == [2, 2]
    retained = roots(primary)
    assert all(error in retained for error in (prior, events[0].wait_error, events[1].wait_error))
    assert owner.pending == [True, True] and len(owner.hosts) == 2


def test_partial_cold_allocation_stays_owned_and_cannot_be_acquired(monkeypatch):
    host = torch.empty((2, 16), dtype=torch.int64)
    owner = object.__new__(upload.DescriptorUploadRing)
    error = MemoryError('second pinned host slot')
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(upload.torch, 'zeros', fail)
    with pytest.raises(MemoryError) as caught:
        owner.__init__(host, host.numpy(), None, 'cpu')
    assert caught.value is error and owner.hosts == [host]
    owner.drain()
    with pytest.raises(RuntimeError, match='construction'):
        owner.acquire()


def test_reentrant_writer_is_rejected_before_any_slot_changes(monkeypatch):
    owner, _stream, _events = ring(monkeypatch)
    host, _, _ = owner.acquire()
    with pytest.raises(RuntimeError, match='host writer'):
        owner.acquire()
    assert host is owner.hosts[0] and not any(owner.pending)
    owner.drain()


def test_manifest_requires_explicit_versioned_trial_and_omission_is_unchanged():
    spec = dict(kernel_family='sm70-paged-owner-rn-ex2-p2051-v1', graph_family='uncaptured',
                graph_budget_bytes=0, trial=True, rn_heads_per_block=12)
    assert parse_preparation(spec) == parse_preparation({**spec, 'descriptor_upload': None})
    assert parse_preparation({**spec, 'descriptor_upload': upload.POLICY})['descriptor_upload'] == upload.POLICY
    for bad in (False, True, 'ring2', 2):
        with pytest.raises(ValueError, match='descriptor upload'):
            parse_preparation({**spec, 'descriptor_upload': bad})
    with pytest.raises(ValueError, match='trial'):
        parse_preparation({**spec, 'trial': False, 'descriptor_upload': upload.POLICY})


@pytest.mark.parametrize('policy', [None, upload.POLICY])
def test_real_owner_keeps_gpu_slab_addresses_generations_and_inactive_tails(policy):
    from tensorfold.families.qwen4_exp.cuda.state import State
    weights = _weights()
    state = State(weights, 257, 4, 'int8')
    owner = OwnerBuffers(weights, 8, NS(idw=2052))
    owner.prepare_native([state], **({'descriptor_upload_policy': policy} if policy else {}))
    pointers = owner.descriptors.data_ptr(), owner.global_stats.data_ptr(), owner.lane_rows.data_ptr()
    hosts = set()
    for count in (3, 1, 2, 1):
        owner.stage_native_descriptors([(state, 0, count)], NS(index=0), 0, count, bucket_rows=8)
        hosts.add(owner.descriptors_host.data_ptr())
        assert pointers == (owner.descriptors.data_ptr(), owner.global_stats.data_ptr(), owner.lane_rows.data_ptr())
        assert owner.descriptors[:count, 9].tolist() == [state.slot_generation] * count
        assert owner.descriptors[:count, 10].tolist() == [state.history_generation] * count
        assert torch.count_nonzero(owner.descriptors[count:]) == 0
    assert len(hosts) == (1 if policy is None else 2)
    record = owner.allocation_record()
    assert record['host_staging_bytes'] == record['host_metadata_bytes'] + (1 if policy is None else 2) * 8 * 128
    if policy is None:
        assert 'descriptor_upload' not in owner.native_record()
        with pytest.raises(RuntimeError, match='fixed'):
            owner.prepare_native(descriptor_upload_policy=upload.POLICY)
    owner.close()
    assert owner.closed and owner.descriptor_upload_ring is None


def test_failed_device_fence_still_attempts_both_host_slot_completions(monkeypatch):
    owner, stream, events = ring(monkeypatch)
    enqueue(owner, stream, 1)
    enqueue(owner, stream, 2)
    class CudaTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True
    workspace = object.__new__(OwnerBuffers)
    workspace._empty_resources()
    workspace.queries = torch.empty(0).as_subclass(CudaTensor)
    workspace.device, workspace.descriptor_upload_ring = 'cuda:0', owner
    primary = RuntimeError('device fence failed')
    def fail(device):
        raise primary
    monkeypatch.setattr(torch.cuda, 'synchronize', fail)
    with pytest.raises(RuntimeError) as caught:
        workspace.retire_native()
    assert caught.value is primary and [e.waits for e in events] == [2, 2]
    assert workspace.descriptor_upload_ring is owner and not workspace.closed

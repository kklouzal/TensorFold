"""Exercise real private protocol/state logic with deterministic CUDA boundary doubles."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import threading

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import pipeline_transfer as p

SHA = 'a' * 64


class Store:
    def __init__(self):
        self.values = {}
        self.seen_keys = set()
        self.wire = {}
        self.condition = threading.Condition()

    def set(self, key, value):
        with self.condition:
            self.seen_keys.add(key)
            self.values[key] = value.encode() if isinstance(value, str) else value
            self.condition.notify_all()

    def wait(self, keys, timeout):
        with self.condition:
            self.seen_keys.update(keys)
            if not self.condition.wait_for(lambda: all(k in self.values for k in keys), timeout=3):
                raise RuntimeError('test Store peer stranded')

    def get(self, key):
        with self.condition:
            return self.values[key]

    def delete_key(self, key):
        with self.condition:
            self.values.pop(key, None)


class Stream:
    def __init__(self):
        self.waits = []
        self.operations = []

    def wait_event(self, event):
        self.waits.append(event)
        self.operations.append(('wait', event))


class Event:
    def __init__(self):
        self.complete = False
        self.recorded = None
        self.fail_record = self.fail_query = False

    def record(self, stream):
        if self.fail_record:
            raise RuntimeError('event record failure')
        self.recorded = stream
        stream.operations.append(('record', self))

    def query(self):
        if self.fail_query:
            raise RuntimeError('event query failure')
        return self.complete


class CUDA:
    capture = False

    def __init__(self, device):
        self.device = device

    def uncaptured(self):
        if self.capture:
            raise p._CaptureRefusal('capture forbidden')

    def stream(self):
        return Stream()

    def event(self):
        return Event()

    def empty(self, shape, dtype):
        return torch.empty(shape, dtype=dtype)

    def validate_stream(self, stream):
        if not isinstance(stream, Stream):
            raise ValueError('stream required')

    def validate(self, tensor, shape, dtype):
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError('bad payload')

    def scope(self, stream):
        return nullcontext()


class Comm:
    world = 2
    device = 'cuda:0'

    def __init__(self, rank, store):
        self.rank, self.store = rank, store
        self.calls = []
        self.route = False
        self.fail_native = self.fail_check = self.fail_close = False

    def send(self, tensor, peer):
        self.calls.append(('send', tuple(tensor.shape), tensor.dtype, tensor.numel() * tensor.element_size(), peer))
        if self.fail_native:
            raise RuntimeError('native send failure')
        if self.route:
            with self.store.condition:
                self.store.wire[len(self.calls) - 1] = tensor.clone()
                self.store.condition.notify_all()

    def recv(self, tensor, peer):
        self.calls.append(('recv', tuple(tensor.shape), tensor.dtype, tensor.numel() * tensor.element_size(), peer))
        if self.fail_native:
            raise RuntimeError('native receive failure')
        if self.route:
            index = len(self.calls) - 1
            with self.store.condition:
                if not self.store.condition.wait_for(lambda: index in self.store.wire, timeout=3):
                    raise RuntimeError('test payload peer stranded')
                tensor.copy_(self.store.wire.pop(index))

    def check(self):
        if self.fail_check:
            raise RuntimeError('native async failure')

    def close(self, *, abort):
        self.calls.append(('close', abort))
        if self.fail_close:
            raise RuntimeError('close failure')


@pytest.fixture
def pair(monkeypatch):
    monkeypatch.setattr(p, '_CUDA', CUDA)
    monkeypatch.setattr(p, 'notify_failure', lambda: False)
    store = Store()
    return tuple(p.HCTransport(Comm(i, store), profile_sha=SHA, epoch=2, max_rows=4, slots=2) for i in range(2))


def ticket(**changes):
    return p.HCTicket(**(dict(profile_sha=SHA, epoch=2, sequence=10, generation=3, round=4, rows=2,
                              token_identity='b' * 64, media_identity='c' * 64, position_identity='d' * 64) | changes))


def source(rows=2):
    return dict(hc=torch.empty((rows, 4, 2560), dtype=torch.float16),
                positions=torch.empty((rows, 3), dtype=torch.int32),
                token_ids=torch.empty((rows,), dtype=torch.int32), producer_stream=Stream(), source_owner=object())


def together(left, right):
    def call(fn):
        try:
            return fn()
        except BaseException as error:
            return error
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(call, fn) for fn in (left, right)]
        return tuple(f.result(timeout=8) for f in futures)


def submit(pair, t=None, **kwargs):
    t = ticket() if t is None else t
    return together(lambda: pair[0].submit(t, **(source() | kwargs)), lambda: pair[1].submit(t))


def finish(pair, leases):
    for transport, lease in zip(pair, leases):
        transport._slots[lease.slot].done.complete = True
        lease.cancel()
        lease.release()


def test_payload_order_exact_bytes_and_source_readiness(pair):
    args = source()
    a, b = together(lambda: pair[0].submit(ticket(), **args), lambda: pair[1].submit(ticket()))
    assert isinstance(a, p.HCLease) and isinstance(b, p.HCLease)
    assert pair[1].device_payload_bytes == 4 * 2 * (20480 + 12 + 4)
    assert pair[0].device_payload_bytes == 0
    assert [c[3] for c in pair[0].comm.calls] == [40960, 24, 8]
    assert [c[1:4] for c in pair[0].comm.calls] == [c[1:4] for c in pair[1].comm.calls]
    slot = pair[0]._slots[a.slot]
    assert slot.ready.recorded is args['producer_stream']
    assert pair[0]._stream.waits == [slot.ready]
    assert not a.producer_reusable
    assert slot.roots[0] is args['hc']
    slot.done.complete = True
    assert a.producer_reusable
    assert slot.roots == ()
    assert pair[0]._store.values == {}
    finish(pair, (a, b))


@pytest.mark.parametrize('bad', ['profile', 'epoch', 'identity', 'shape', 'dtype', 'cancel', 'capture', 'scalar'])
def test_matched_refusal_before_native_and_next_attempt(pair, bad):
    left, right = ticket(), ticket()
    args = source()
    if bad == 'profile':
        left = ticket(profile_sha='f' * 64)
    elif bad == 'epoch':
        left = ticket(epoch=1)
    elif bad == 'identity':
        right = ticket(position_identity='f' * 64)
    elif bad == 'shape':
        args['hc'] = torch.empty((2, 10240), dtype=torch.float16)
    elif bad == 'dtype':
        args['positions'] = args['positions'].to(torch.int64)
    elif bad == 'cancel':
        args['cancelled'] = True
    elif bad == 'capture':
        pair[0]._cuda.capture = True
    elif bad == 'scalar':
        left = ticket(rows=True)
    results = together(lambda: pair[0].submit(left, **args), lambda: pair[1].submit(right))
    assert all(isinstance(x, p._ManifestRefusal) for x in results)
    assert all(not t.comm.calls and t._attempt == 1 for t in pair)
    assert pair[0]._store.values == {}
    pair[0]._cuda.capture = False
    leases = submit(pair)
    assert all(isinstance(x, p.HCLease) for x in leases)
    finish(pair, leases)


def test_receive_completion_does_not_release_consumer_and_stale_lease(pair):
    a, b = submit(pair)
    slot = pair[1]._slots[b.slot]
    slot.done.complete = True
    assert b.poll()
    with pytest.raises(ValueError, match='consumer completion'):
        b.release()
    consumer = Stream()
    with b.consume(consumer) as values:
        assert [tuple(t.shape) for t in values] == [(2, 4, 2560), (2, 3), (2,)]
        assert consumer.waits == [slot.done]
        with pytest.raises(ValueError, match='active CPU'):
            pair[1].close(abort=True)
    with pytest.raises(ValueError, match='consumer still pending'):
        b.release()
    slot.consumed.complete = True
    b.release()
    pair[0]._slots[a.slot].done.complete = True
    a.release()
    new = submit(pair, ticket(round=5))
    assert new[1].slot == b.slot and new[1].generation != b.generation
    with pytest.raises(ValueError, match='stale'):
        b.release()
    finish(pair, new)


def test_late_cancel_drain_and_credit_exhaustion(pair):
    first = submit(pair)
    for lease in first:
        lease.cancel()
        with pytest.raises(ValueError, match='still pending'):
            lease.release()
    second = submit(pair, ticket(sequence=11))
    third = submit(pair, ticket(sequence=12))
    assert all(isinstance(x, p._ManifestRefusal) for x in third)
    assert all(len(t.comm.calls) == 6 for t in pair)
    finish(pair, first)
    finish(pair, second)


def test_one_unresolved_round_per_sequence(pair):
    first = submit(pair)
    refused = submit(pair, ticket(round=5))
    assert all(isinstance(x, p._ManifestRefusal) for x in refused)
    finish(pair, first)


@pytest.mark.parametrize('failure', ['native', 'record', 'query', 'check'])
def test_uncertain_failures_quarantine_and_retain_until_abort(pair, failure):
    sender = pair[0]
    if failure == 'native':
        sender.comm.fail_native = True
    if failure == 'record':
        sender._slots[0].done.fail_record = True
    results = submit(pair)
    if failure in ('query', 'check'):
        lease = results[0]
        if failure == 'query':
            sender._slots[lease.slot].done.fail_query = True
        else:
            sender.comm.fail_check = True
        with pytest.raises(RuntimeError):
            lease.poll()
    else:
        assert isinstance(results[0], RuntimeError)
    assert sender._poison is not None and sender._slots[0].roots[0] is not None
    with pytest.raises(ValueError, match='quiescent'):
        sender.close()
    with pytest.raises(RuntimeError, match='quarantined'):
        sender.submit(ticket(), **source())
    sender.close(abort=True)
    assert sender._closed and sender._slots == []
    pair[1].close(abort=True)


def test_close_failure_preserves_all_roots_and_never_retries(pair):
    leases = submit(pair)
    sender = pair[0]
    roots, slots, comm = sender._slots[0].roots, sender._slots, sender.comm
    comm.fail_close = True
    with pytest.raises(RuntimeError, match='close failure') as caught:
        sender.close(abort=True)
    assert caught.value._tensorfold_retained_owners == [sender]
    assert sender._slots is slots and sender._slots[0].roots is roots and sender.comm is comm
    with pytest.raises(RuntimeError, match='cannot retry'):
        sender.close(abort=True)
    assert sum(c[0] == 'close' for c in comm.calls) == 1
    pair[1].close(abort=True)
    assert isinstance(leases[0], p.HCLease)


def test_receive_pending_and_completed_async_error_never_enter_consumer(pair):
    a, b = submit(pair)
    entered = []
    with pytest.raises(ValueError, match='receive still pending'):
        with b.consume(Stream()):
            entered.append(True)
    assert not entered and not pair[1]._slots[b.slot].consumer and pair[1]._poison is None
    pair[1]._slots[b.slot].done.complete = True
    pair[1].comm.fail_check = True
    with pytest.raises(RuntimeError, match='async'):
        with b.consume(Stream()):
            entered.append(True)
    assert not entered and pair[1]._poison is not None
    for transport in pair:
        transport.close(abort=True)
    assert isinstance(a, p.HCLease)


def test_consumer_fault_preserves_primary_and_avoids_event_record(pair):
    _, b = submit(pair)
    slot = pair[1]._slots[b.slot]
    slot.done.complete = True
    primary = RuntimeError('consumer CUDA fault')
    slot.consumed.fail_record = True
    with pytest.raises(RuntimeError) as caught:
        with b.consume(Stream()):
            raise primary
    assert caught.value is primary
    assert slot.consumed.recorded is None and slot.state == 'received'
    assert pair[1]._poison is primary and slot.views
    for transport in pair:
        transport.close(abort=True)


def test_real_cuda_scope_restores_success_and_makes_no_fault_cleanup(monkeypatch):
    source = type('S', (), {'device': torch.device('cuda:1')})()
    destination = type('S', (), {'device': torch.device('cuda:0')})()
    selected = type('S', (), {'device': torch.device('cuda:0')})()
    adapter = object.__new__(p._CUDA)
    adapter.device = torch.device('cuda:0')
    actions = []
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device=None: source if device is None else destination)
    monkeypatch.setattr(torch.cuda, 'set_stream', lambda stream: actions.append(stream))
    with adapter.scope(selected):
        actions.append('body')
    assert actions == [selected, 'body', destination, source]
    actions.clear()
    primary = RuntimeError('device fault')
    with pytest.raises(RuntimeError) as caught:
        with adapter.scope(selected):
            actions.append('body')
            raise primary
    assert caught.value is primary and actions == [selected, 'body']


def test_consumer_restore_failure_quarantines_without_record(pair):
    _, lease = submit(pair)
    transport = pair[1]
    slot = transport._slots[lease.slot]
    slot.done.complete = True
    class FailRestore:
        def __enter__(self):
            return self
        def __exit__(self, kind, value, tb):
            assert kind is None
            raise RuntimeError('restore failure')
    transport._cuda.scope = lambda stream: FailRestore()
    with pytest.raises(RuntimeError, match='restore failure'):
        with lease.consume(Stream()):
            pass
    assert transport._poison is not None and slot.consumed.recorded is None
    for transport in pair:
        transport.close(abort=True)


@pytest.mark.parametrize('field,value', [('slot', -1), ('slot', True), ('slot', 2),
                                       ('generation', True), ('generation', 0), ('generation', -1)])
def test_malformed_lease_fields_are_rejected(pair, field, value):
    leases = submit(pair)
    lease = leases[0]
    forged = p.HCLease(pair[0], **({'slot': lease.slot, 'generation': lease.generation} | {field: value}))
    with pytest.raises(ValueError):
        forged.release()
    finish(pair, leases)


def test_repeated_attempts_use_fixed_store_keys_and_exact_ordinal(pair):
    for ordinal in range(40):
        results = submit(pair, ticket(round=ordinal), cancelled=True)
        assert all(isinstance(x, p._ManifestRefusal) for x in results)
        assert pair[0]._store.values == {}
    assert len(pair[0]._store.seen_keys) == 12  # preflight + retirement, two banks
    assert all('/bank/0/' in key or '/bank/1/' in key for key in pair[0]._store.seen_keys)
    for endpoint in pair:
        endpoint._attempt = p._LIMIT
    leases = submit(pair)
    finish(pair, leases)
    assert all(endpoint._attempt == p._LIMIT + 1 for endpoint in pair)
    for endpoint in pair:
        with pytest.raises(RuntimeError, match='ordinal exhausted'):
            endpoint.submit(ticket())
        endpoint.close()


def test_full_attempt_ordinal_is_agreed_in_same_bank(pair):
    pair[1]._attempt = 2
    refused = submit(pair)
    assert all(isinstance(x, p._ManifestRefusal) for x in refused)
    assert not any(t.comm.calls for t in pair)
    assert pair[0]._store.values == {}


def test_slot_generation_overflow_is_matched_refusal(pair):
    pair[0]._slots[0].generation = p._LIMIT
    refused = submit(pair)
    assert all(isinstance(x, p._ManifestRefusal) for x in refused)
    assert not any(t.comm.calls for t in pair)


def test_payload_sentinel_routing_uses_exact_destination_views(pair):
    for transport in pair:
        transport.comm.route = True
    args = source()
    args['hc'].fill_(12.5)
    args['positions'].copy_(torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32))
    args['token_ids'].copy_(torch.tensor([17, 91], dtype=torch.int32))
    leases = together(lambda: pair[0].submit(ticket(), **args), lambda: pair[1].submit(ticket()))
    slot = pair[1]._slots[leases[1].slot]
    slot.done.complete = True
    with leases[1].consume(Stream()) as received:
        for actual, expected in zip(received, (args['hc'], args['positions'], args['token_ids'])):
            assert torch.equal(actual, expected)
            assert actual.data_ptr() != expected.data_ptr()
    slot.consumed.complete = True
    leases[1].release()
    pair[0]._slots[leases[0].slot].done.complete = True
    leases[0].release()


def failure_tree(error):
    """Integration assertion: retain primary and every independently raised root."""
    pending, seen, result = [error], set(), []
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        result.append(current)
        pending.extend((current.__cause__, current.__context__))
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
    return result


def test_poison_notifies_once_preserves_prior_owners_and_primary(pair, monkeypatch):
    endpoint = pair[0]
    calls = []
    notification = RuntimeError('supervisor notification failed')
    def fail_notify():
        calls.append(True)
        raise notification
    monkeypatch.setattr(p, 'notify_failure', fail_notify)
    class Hostile(RuntimeError):
        def __setattr__(self, name, value):
            raise AssertionError('must use native exception namespace')
    primary = Hostile('primary')
    prior = object()
    namespace = BaseException.__dict__['__dict__'].__get__(primary)
    namespace['_tensorfold_retained_owners'] = [prior]
    with pytest.raises(Hostile) as caught:
        endpoint.poison_control(primary)
    assert caught.value is primary
    assert any(root is notification for root in failure_tree(primary))
    assert namespace['_tensorfold_retained_owners'] == [prior, endpoint]
    endpoint.poison_control(primary)
    assert calls == [True]
    endpoint.close(abort=True)
    pair[1].close()


def test_refusal_retirement_failure_preserves_refusal_and_retained_owner(pair, monkeypatch):
    original = p._retire_store
    cleanup = OSError('retirement cleanup failed')
    def fail_retire(engine, channel, *, stages):
        original(engine, channel, stages=stages)
        if engine.rank == 0:
            raise cleanup
    monkeypatch.setattr(p, '_retire_store', fail_retire)
    refused = submit(pair, cancelled=True)
    assert all(isinstance(x, p._ManifestRefusal) for x in refused)
    assert any(root is cleanup for root in failure_tree(refused[0]))
    assert any(type(root) is ValueError and 'cancellation' in str(root)
               for root in failure_tree(refused[0]))
    from tensorfold.families.qwen4_exp.cuda.execution_identity import _unretired
    assert _unretired(refused[0]) and pair[0]._poison is cleanup
    for endpoint in pair:
        endpoint.close(abort=True)


@pytest.mark.parametrize('kind,nth', [('stream', 1), ('event', 1), ('event', 2), ('event', 3),
                                     ('empty', 1), ('empty', 2), ('empty', 3)])
def test_partial_constructor_publishes_every_acquired_owner(monkeypatch, kind, nth):
    acquired = []
    primary = RuntimeError('injected acquisition failure')
    class FailCUDA(CUDA):
        def __init__(self, device):
            super().__init__(device)
            self.counts = {}
        def acquire(self, name, operation):
            self.counts[name] = self.counts.get(name, 0) + 1
            if name == kind and self.counts[name] == nth:
                raise primary
            value = operation()
            acquired.append(value)
            return value
        def stream(self):
            return self.acquire('stream', super().stream)
        def event(self):
            return self.acquire('event', super().event)
        def empty(self, shape, dtype):
            return self.acquire('empty', lambda: super(FailCUDA, self).empty(shape, dtype))
    monkeypatch.setattr(p, '_CUDA', FailCUDA)
    monkeypatch.setattr(p, 'notify_failure', lambda: False)
    comm = Comm(1, Store())
    with pytest.raises(RuntimeError) as caught:
        p.HCTransport(comm, profile_sha=SHA, epoch=2, max_rows=4, slots=2)
    assert caught.value is primary
    owner = caught.value._tensorfold_retained_owners[0]
    roots = [owner._stream]
    for slot in owner._slots:
        roots.extend((slot.ready, slot.done, slot.consumed, *slot.buffers))
    assert all(any(actual is root for root in roots) for actual in acquired)
    owner.close(abort=True)
    assert comm.calls == [('close', True)]


def test_delayed_retirement_cannot_reuse_old_bank_early(pair, monkeypatch):
    original = pair[0]._store.delete_key
    paused = threading.Event()
    proceed = threading.Event()
    first = True
    def delayed(key):
        nonlocal first
        if first:
            first = False
            paused.set()
            if not proceed.wait(3):
                raise RuntimeError('test retirement gate timed out')
        return original(key)
    pair[0]._store.delete_key = delayed
    def attempts(endpoint, args):
        results = []
        for _ in range(3):
            try:
                endpoint.submit(ticket(), cancelled=True, **args)
            except p._ManifestRefusal as error:
                results.append(error)
        return results
    with ThreadPoolExecutor(2) as pool:
        sender = pool.submit(attempts, pair[0], source())
        receiver = pool.submit(attempts, pair[1], {})
        assert paused.wait(3)
        assert pair[0]._attempt == 0 and pair[1]._attempt <= 1
        # Rank1 may enter bank1 but cannot finish it while rank0 retires bank0.
        assert not sender.done() and not receiver.done()
        proceed.set()
        assert len(sender.result(timeout=8)) == len(receiver.result(timeout=8)) == 3
    assert pair[0]._store.values == {}



def test_consumer_cuda_precheck_fault_quarantines_without_restore(monkeypatch):
    real_adapter = p._CUDA
    monkeypatch.setattr(p, '_CUDA', CUDA)
    notifications = []
    monkeypatch.setattr(p, 'notify_failure', lambda: notifications.append(True))
    store = Store()
    pair = tuple(p.HCTransport(Comm(i, store), profile_sha=SHA, epoch=2, max_rows=4, slots=2)
                 for i in range(2))
    _, lease = submit(pair)
    endpoint = pair[1]
    slot = endpoint._slots[lease.slot]
    slot.done.complete = True
    adapter = object.__new__(real_adapter)
    adapter.device = torch.device('cuda:0')
    actions = []
    primary = RuntimeError('capture query CUDA fault')
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 1)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: actions.append(device))
    def fail_query():
        raise primary
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', fail_query)
    endpoint._cuda.uncaptured = lambda: adapter.uncaptured()
    entered = []
    stream = Stream()
    with pytest.raises(RuntimeError) as caught:
        with lease.consume(stream):
            entered.append(True)
    assert caught.value is primary and endpoint._poison is primary
    assert primary._tensorfold_retained_owners == [endpoint]
    assert notifications == [True] and not entered
    assert actions == [0] and not stream.operations and slot.consumed.recorded is None
    assert slot.views and not slot.consumer
    for endpoint in pair:
        endpoint.close(abort=True)


def test_consumer_known_capture_refusal_is_unpoisoned(pair):
    leases = submit(pair)
    receiver = pair[1]
    receiver._cuda.capture = True
    with pytest.raises(p._CaptureRefusal):
        with leases[1].consume(Stream()):
            pytest.fail('capture refusal entered consumer')
    assert receiver._poison is None and not receiver._slots[leases[1].slot].consumer
    receiver._cuda.capture = False
    finish(pair, leases)

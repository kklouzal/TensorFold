"""Private, uncaptured two-leader HC transfer, not a PP2 serving profile.

The caller transfers exclusive ownership of the dedicated NCCL communicator and
its Store. Never pass the model's TP8 communicator. There is no CPU transport.
This does not implement TP8 fanout, owner services, or distributed round commit.

Sender storage (including the A6 workspace) MUST remain read-only and unreused
until producer_reusable is true. Strong references alone do not enforce this.
Receive buffers remain borrowed until a consumer-stream completion event retires.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import threading

import torch

from tensorfold.cleanup import raise_failures
from tensorfold.cuda.cohort import notify_failure

from .execution_identity import _ManifestRefusal, _phase, _retire_store
from .retention_resources import StoreFieldGuard

_LIMIT = 2**63 - 1
_ROW_BYTES = 4 * 2560 * 2 + 3 * 4 + 4


def _integer(value, name, *, minimum=0, maximum=_LIMIT):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} outside bounded integer contract")


def _sha(value, name):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f"{name} requires lowercase SHA256")


@dataclass(frozen=True)
class HCTicket:
    """Immutable identity references; hashes bind canonical token/media/position data."""
    profile_sha: str
    epoch: int
    sequence: int
    generation: int
    round: int
    rows: int
    token_identity: str
    media_identity: str
    position_identity: str

    def validate(self):
        for name in ('profile_sha', 'token_identity', 'media_identity', 'position_identity'):
            _sha(getattr(self, name), name)
        for name in ('epoch', 'sequence', 'generation', 'round'):
            _integer(getattr(self, name), name)
        _integer(self.rows, 'rows', minimum=1)


class _CaptureRefusal(ValueError):
    """Known pre-enqueue graph admission refusal, not an unknown CUDA fault."""


class _CUDA:
    """Narrow CUDA boundary, replaced only by deterministic unit-test doubles."""
    def __init__(self, device):
        if device is None or torch.device(device).type != 'cuda':
            raise ValueError('HC transfer requires communicator CUDA device')
        self.device = device

    def uncaptured(self):
        previous = torch.cuda.current_device()
        target = torch.device(self.device).index
        if previous != target:
            torch.cuda.set_device(target)
        captured = torch.cuda.is_current_stream_capturing()
        if previous != target:
            torch.cuda.set_device(previous)
        if captured:
            raise _CaptureRefusal('HC transport CUDA graph capture is unqualified')

    def stream(self):
        return torch.cuda.Stream(device=self.device)

    def event(self):
        # CUDA events initialize on their first explicit stream record.
        return torch.cuda.Event(enable_timing=False)

    def empty(self, shape, dtype):
        return torch.empty(shape, dtype=dtype, device=self.device)

    def validate_stream(self, stream):
        if not isinstance(stream, torch.cuda.Stream) or stream.device != self.device:
            raise ValueError('explicit stream on communicator device required')

    def validate(self, tensor, shape, dtype):
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                or tensor.dtype != dtype or tensor.device != self.device
                or tensor.requires_grad or not tensor.is_contiguous()):
            raise ValueError('exact contiguous HC/metadata dtype, shape and device required')

    @contextmanager
    def scope(self, stream):
        # The SDK StreamContext unconditionally restores streams even after an
        # unknown CUDA fault. Only a successful body may make further CUDA calls.
        # On fault the caller must quarantine and contain the cohort; its current
        # device/stream is deliberately not restored. On success both the source
        # device stream and destination device stream retain their entry values.
        source = torch.cuda.current_stream()
        destination = torch.cuda.current_stream(self.device)
        torch.cuda.set_stream(stream)
        yield
        if destination.device != source.device:
            torch.cuda.set_stream(destination)
        torch.cuda.set_stream(source)


@dataclass
class _Slot:
    generation: int = 0
    state: str = 'free'
    buffers: list = field(default_factory=lambda: [None, None, None])
    done: object = None
    ready: object = None
    consumed: object = None
    roots: tuple = ()
    views: tuple = ()
    discarded: bool = False
    consumer: bool = False
    ticket: object = None


@dataclass(frozen=True)
class HCLease:
    _transport: object
    slot: int
    generation: int

    def poll(self):
        return self._transport._poll(self)

    @property
    def producer_reusable(self):
        return self._transport._producer_reusable(self)

    def cancel(self):
        """Late cancel discards the payload only after its agreed sends/receives drain."""
        self._transport._cancel(self)

    def consume(self, stream):
        """Bounded CPU scope; enqueue all consumer work on stream inside this scope."""
        return self._transport._consume(self, stream)

    def release(self):
        self._transport._release(self)


class HCTransport:
    """Single CPU operation at a time; consumer scopes exclude close/submission.

    Payload reservation is exact device tensor storage only. NCCL internal memory,
    CUDA events/streams, allocator overhead and GPUDirect registration remain
    separately unqualified. Rank0 borrows payloads; rank1 reserves the ring.
    Constructor failure retains the partially built owner on the exception.
    Without a launcher failure descriptor, fatal cross-server containment remains
    the external supervisor's responsibility (local notification is no substitute).
    A CUDA failure leaves caller device/stream restoration unspecified until abort.
    """
    def __init__(self, comm, *, profile_sha, epoch, max_rows, slots):
        self.comm = comm
        self._store = comm.store
        self._lock = threading.RLock()
        self._slots = []
        self._stream = None
        self._cuda = None
        self._poison = None
        self._notified = False
        self._closed = False
        self._close_entered = False
        self._attempt = 0
        self.rank, self.tp = comm.rank, 2
        self.multi = self
        self.store_fields = StoreFieldGuard(16384)
        self.profile_sha, self.epoch = profile_sha, epoch
        self.max_rows, self.slot_count = max_rows, slots
        try:
            if type(comm.world) is not int or comm.world != 2 or type(comm.rank) is not int or comm.rank not in (0, 1):
                raise ValueError('dedicated two-leader NCCL communicator required')
            _sha(profile_sha, 'profile_sha')
            _integer(epoch, 'epoch')
            _integer(max_rows, 'max_rows', minimum=1, maximum=_LIMIT // _ROW_BYTES)
            _integer(slots, 'slots', minimum=1, maximum=_LIMIT // (max_rows * _ROW_BYTES))
            self._cuda = _CUDA(comm.device)
            self._cuda.uncaptured()
            self._stream = self._cuda.stream()
            for _ in range(slots):
                slot = _Slot()
                self._slots.append(slot)  # Publish each owner before the next acquisition.
                slot.ready = self._cuda.event()
                slot.done = self._cuda.event()
                slot.consumed = self._cuda.event()
                if self.rank == 1:
                    for index, (shape, dtype) in enumerate((((max_rows, 4, 2560), torch.float16),
                                                           ((max_rows, 3), torch.int32),
                                                           ((max_rows,), torch.int32))):
                        slot.buffers[index] = self._cuda.empty(shape, dtype)
        except BaseException as error:
            self.poison_control(error)
            raise

    @property
    def device_payload_bytes(self):
        return self.max_rows * self.slot_count * _ROW_BYTES if self.rank == 1 else 0

    def poison_control(self, error):
        if self._poison is None:
            self._poison = error
        errors = []
        try:
            namespace = BaseException.__dict__['__dict__'].__get__(error, type(error))
            retained = dict.get(namespace, '_tensorfold_retained_owners', [])
            if type(retained) is not list:
                raise ValueError('retained HC owners must remain a list')
            if all(owner is not self for owner in retained):
                dict.__setitem__(namespace, '_tensorfold_retained_owners', [*retained, self])
        except BaseException as retention:
            errors.append(retention)
        if not self._notified:
            self._notified = True  # An entered notification is never retried.
            try:
                notify_failure()
            except BaseException as notification:
                errors.append(notification)
        if errors:
            raise_failures(error, errors)

    def _live(self):
        if self._closed or self._close_entered or self._poison is not None:
            raise RuntimeError('HC transport is closed or quarantined') from self._poison

    def _lease(self, lease):
        if type(lease) is not HCLease or lease._transport is not self:
            raise ValueError('foreign HC lease')
        _integer(lease.slot, 'lease slot', maximum=len(self._slots) - 1)
        _integer(lease.generation, 'lease generation', minimum=1)
        slot = self._slots[lease.slot]
        if slot.generation != lease.generation or slot.state == 'free':
            raise ValueError('stale HC lease generation')
        return slot

    def submit(self, ticket, *, hc=None, positions=None, token_ids=None,
               producer_stream=None, source_owner=None, cancelled=False):
        """Both endpoints MUST call once per attempt, including local refusal/cancel.

        Fixed namespace/monotonic ordinal avoids identity-dependent peer waits.
        A peer must continue calling for refusal; this is not a progress service.
        Post-admission cancel uses lease.cancel and cannot skip native operations.
        """
        with self._lock:
            self._live()
            if self._attempt > _LIMIT:
                raise RuntimeError('HC attempt ordinal exhausted; close this transport')
            channel = f'tf_private_hc_v1/bank/{self._attempt % 2}'
            chosen = None
            def preflight():
                nonlocal chosen
                _integer(self._attempt, 'attempt ordinal')
                self._cuda.uncaptured()
                if any(slot.consumer for slot in self._slots):
                    raise ValueError('active CPU consumer scope')
                if type(cancelled) is not bool or cancelled:
                    raise ValueError('pre-enqueue cancellation or invalid cancellation scalar')
                if type(ticket) is not HCTicket:
                    raise ValueError('typed HC ticket required')
                ticket.validate()
                if ticket.profile_sha != self.profile_sha or ticket.epoch != self.epoch or ticket.rows > self.max_rows:
                    raise ValueError('HC ticket profile/epoch/row envelope mismatch')
                if any(slot.state != 'free' and slot.ticket is not None
                       and slot.ticket.sequence == ticket.sequence
                       and slot.ticket.generation == ticket.generation for slot in self._slots):
                    raise ValueError('sequence already has an unresolved round')
                chosen = next((i for i, slot in enumerate(self._slots) if slot.state == 'free'), None)
                if chosen is None:
                    raise ValueError('HC ring credit exhausted')
                _integer(self._slots[chosen].generation, 'slot generation', maximum=_LIMIT - 1)
                if self.rank == 0:
                    self._cuda.validate_stream(producer_stream)
                    if source_owner is None:
                        raise ValueError('explicit source owner required')
                    for value, shape, dtype in ((hc, (ticket.rows, 4, 2560), torch.float16),
                                                (positions, (ticket.rows, 3), torch.int32),
                                                (token_ids, (ticket.rows,), torch.int32)):
                        self._cuda.validate(value, shape, dtype)
                elif any(value is not None for value in (hc, positions, token_ids, producer_stream, source_owner)):
                    raise ValueError('receiver cannot supply source storage')
                return dict(attempt=self._attempt, ticket=asdict(ticket), max_rows=self.max_rows, slots=self.slot_count)
            stages = ['preflight']
            try:
                values = _phase(self, channel, 'preflight', preflight)
                stages.append('agreement')
                def agree():
                    if values[0] != values[1]:
                        raise ValueError('HC endpoints disagree on exact ticket/configuration')
                    return True
                _phase(self, channel, 'agreement', agree)
            except _ManifestRefusal as primary:
                try:
                    _retire_store(self, channel, stages=stages)
                except BaseException as error:
                    try:
                        self.poison_control(error)
                    except BaseException as notification:
                        if notification is not error:
                            raise_failures(primary, [error, notification])
                    raise_failures(primary, [error])
                self._attempt += 1
                raise
            except BaseException as error:
                self.poison_control(error)
                raise
            try:
                _retire_store(self, channel, stages=stages)
                self._attempt += 1
                slot = self._slots[chosen]
                slot.generation += 1
                slot.ticket = ticket
                slot.state = 'submitted'
                slot.discarded = False
                slot.roots = (hc, positions, token_ids, source_owner, producer_stream)
                slot.views = ((hc, positions, token_ids) if self.rank == 0 else
                              tuple(buffer[:ticket.rows] for buffer in slot.buffers))
                # Dependency is enqueued BEFORE NCCL reads the borrowed workspace.
                if self.rank == 0:
                    slot.ready.record(producer_stream)
                    self._stream.wait_event(slot.ready)
                with self._cuda.scope(self._stream):
                    for tensor in slot.views:
                        (self.comm.send if self.rank == 0 else self.comm.recv)(tensor, 1 - self.rank)
                    slot.done.record(self._stream)
                return HCLease(self, chosen, slot.generation)
            except BaseException as error:
                self.poison_control(error)
                raise

    def _complete(self, slot):
        try:
            complete = slot.done.query()
            self.comm.check()
            if complete:
                slot.state = 'received' if self.rank == 1 else 'sent'
                if self.rank == 0:
                    slot.roots = ()
                    slot.views = ()
            return complete
        except BaseException as error:
            self.poison_control(error)
            raise

    def _poll(self, lease):
        with self._lock:
            self._live()
            slot = self._lease(lease)
            if slot.state == 'submitted':
                return self._complete(slot)
            return True

    def _producer_reusable(self, lease):
        with self._lock:
            if self.rank != 0:
                raise ValueError('receiver has no producer borrow')
            self._live()
            slot = self._lease(lease)
            return slot.state == 'sent' or self._complete(slot)

    def _cancel(self, lease):
        with self._lock:
            self._live()
            slot = self._lease(lease)
            if slot.consumer or slot.state == 'consumed':
                raise ValueError('consumer already owns payload')
            slot.discarded = True

    @contextmanager
    def _consume(self, lease, stream):
        # Lock covers CPU scope as well as submission, including other threads.
        with self._lock:
            self._live()
            slot = self._lease(lease)
            if self.rank != 1 or slot.discarded or slot.consumer or slot.state not in ('submitted', 'received'):
                raise ValueError('HC receive lease unavailable to consumer')
            try:
                self._cuda.uncaptured()
            except _CaptureRefusal:
                raise
            except BaseException as error:
                self.poison_control(error)
                raise
            self._cuda.validate_stream(stream)
            if slot.state == 'submitted' and not self._complete(slot):
                raise ValueError('HC receive still pending')
            try:
                # A prior poll is not an async-error check at this handoff.
                self.comm.check()
                stream.wait_event(slot.done)
                slot.consumer = True
                with self._cuda.scope(stream):
                    yield slot.views
                slot.consumed.record(stream)
                slot.state = 'consumed'
            except BaseException as error:
                # The body may have queued work or faulted CUDA. Make no further
                # CUDA calls (including stream restore/event record) on that path.
                # Abort then device-fence is the only permitted retirement.
                self.poison_control(error)
                raise
            finally:
                slot.consumer = False

    def _release(self, lease):
        with self._lock:
            self._live()
            slot = self._lease(lease)
            if slot.consumer:
                raise ValueError('active consumer scope')
            if slot.state == 'submitted' and not self._complete(slot):
                raise ValueError('HC transfer still pending')
            if self.rank == 1:
                if slot.state == 'consumed':
                    try:
                        complete = slot.consumed.query()
                        self.comm.check()
                    except BaseException as error:
                        self.poison_control(error)
                        raise
                    if not complete:
                        raise ValueError('HC consumer still pending')
                elif not slot.discarded:
                    raise ValueError('HC payload requires consumer completion or discard')
            slot.roots, slot.views = (), ()
            slot.ticket = None
            slot.state = 'free'

    def close(self, *, abort=False):
        with self._lock:
            if type(abort) is not bool:
                raise ValueError('typed abort policy required')
            if self._closed:
                return
            if self._close_entered:
                raise RuntimeError('entered communicator close cannot retry') from self._poison
            if any(slot.consumer for slot in self._slots):
                raise ValueError('active CPU consumer scope')
            if not abort and (self._poison is not None or any(slot.state != 'free' for slot in self._slots)):
                raise ValueError('ordinary close requires quiescent healthy ring')
            self._close_entered = True
            try:
                # NCCL abort precedes its device fence. No roots are released on failure.
                self.comm.close(abort=abort)
            except BaseException as error:
                self.poison_control(error)
                raise
            self._closed = True
            self._slots.clear()
            self._stream = self._cuda = self._store = self.comm = None

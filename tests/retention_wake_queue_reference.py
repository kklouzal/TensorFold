"""Bounded CPU transfer-worker wakes for one current retained-state cohort.

The catalog retires local data workers, fences this sender connection, then
agrees ALL sender retirements through its main-store phase before close.
Only this helper's receiver consumes its distinct queue; queue keys never enter
set/get/delete operations. A receiver ACK precedes its stable scheduler wake.
"""

from datetime import timedelta
import os
import threading
import time
import uuid

from tensorfold.cleanup import raise_failures, rollback
from tensorfold.cuda.store_wait import timed_out
from tensorfold.thread_work import ThreadWork

MAX_WORD = (1 << 63) - 1
MAX_PACKET_BYTES = 64
# The actual Store transport preserves this ONE native queue across fully
# retired catalog activations. Its native backend does not erase empty queues.
TRANSPORT_QUEUE = "tf_retention/wake/v1/queue"
TRANSPORT_LEASE = "tf_retention/wake/v1/receiver-owner"


class RetentionWakeRelay:
    @classmethod
    def create(cls, comm, namespace, *, wake, fatal, register):
        """Publish the partial controller before cloned sockets or a thread."""
        rank, world = (0, 1) if comm is None else (comm.rank, comm.world)
        if (type(rank) is not int or type(world) is not int or not 1 <= world <= MAX_WORD
                or not 0 <= rank < world or type(namespace) is not str or not namespace or "\0" in namespace
                or any(not callable(value) for value in (wake, fatal, register))):
            raise ValueError("wake relay needs an actual rank cohort, activation namespace and owned callbacks")
        owner = cls.__new__(cls)
        owner.rank, owner.world, owner.namespace = rank, world, namespace
        owner._pid, owner._lock, owner._sending, owner._closing_lock = os.getpid(), threading.Lock(), threading.Lock(), threading.Lock()
        owner._wake, owner._fatal = wake, fatal
        owner._control = owner._sender = owner._receiver = owner._worker = owner._thread = None
        owner._started = owner._closing = owner._closed = owner._claim_attempted = owner._stop_attempted = False
        owner._claim_owned = False
        owner._transport_owned = owner._transport_attempted = owner._transport_retire_attempted = False
        owner._fenced = owner._senders_fenced = owner._receiver_fenced = False
        owner._failure, owner._fatal_reported = None, False
        owner._generation, owner._received, owner._notifications = 0, {}, 0
        owner._queue = TRANSPORT_QUEUE
        owner._claim = namespace + f"/wake-owner/{rank}"
        owner._nonce = uuid.uuid4().hex.encode("ascii")
        owner.timeout = None
        base = None if comm is None else comm.store
        comm = None  # The relay never retains its model/communicator owner.
        try:
            register(owner)
            if world > 1:
                owner.timeout = base.timeout
                if not isinstance(owner.timeout, timedelta) or owner.timeout <= timedelta(0):
                    raise ValueError("wake relay requires a positive inherited Store timeout")
                owner._control = base.clone()
                owner._control.set_timeout(owner.timeout)
                owner._claim_attempted = True  # compare_set can publish before a transport failure.
                if owner._control.compare_set(owner._claim, b"", owner._nonce) != owner._nonce:
                    raise ValueError("wake relay activation/rank is already owned")
                owner._claim_owned = True
                if rank:
                    owner._sender = base.clone()
                    owner._sender.set_timeout(owner.timeout)
                else:
                    owner._transport_attempted = True
                    if owner._control.compare_set(TRANSPORT_LEASE, b"", owner._nonce) != owner._nonce:
                        raise ValueError("the actual Store transport already owns a retention wake receiver")
                    owner._transport_owned = True
                    if owner._control.queue_len(owner._queue):
                        raise RuntimeError("the reusable transport wake queue has not retired")
                    owner._receiver = base.clone()
                    owner._receiver.set_timeout(owner.timeout)
                    owner._worker = ThreadWork(owner._receive)
                    owner._thread = threading.Thread(target=owner._worker.run, name="tensorfold-retention-wake", daemon=False)
            return owner
        except BaseException as primary:
            rollback(owner, primary, lambda: owner.close(timeout=0 if owner.timeout is None else owner.timeout.total_seconds()))
        finally:
            base = None

    def _process(self):
        if os.getpid() != self._pid:
            raise RuntimeError("wake relay belongs to another process")

    def _ack(self, rank, generation):
        return self.namespace + f"/wake-ack/{rank}/{generation}"

    def _fence_key(self, rank):
        return self.namespace + f"/wake-fenced/{rank}"

    def _report(self, primary):
        with self._lock:
            if self._failure is None:
                self._failure = primary
            if self._fatal_reported:
                return
            self._fatal_reported = True
        try:
            self._fatal(primary)
        except BaseException as secondary:
            raise_failures(primary, [secondary])

    def start(self):
        self._process()
        with self._lock:
            if self._started or self._closing or self._closed:
                raise RuntimeError("wake relay starts once before closure")
            self._started = True  # Retain launch attempts even if native bootstrap raises.
        if self._worker is not None:
            try:
                self._worker.launch(self._thread)
            except BaseException as primary:
                self._report(primary)
                raise

    def fence(self):
        """After local producers quiesce, prove this socket's sends processed.

        queue_push returns after flushing bytes, without a server reply. The
        same sender's GET reply proves its earlier commands were processed;
        only then may a separately connected leader observe our fence marker.
        Call before the catalog's all-rank senders-retired agreement. Failure
        forbids further notifications and retains ownership for explicit retry.
        """
        self._process()
        with self._sending:
            with self._lock:
                if self._closed or self._fenced:
                    return
                self._closing = True
            try:
                if self._sender is not None:
                    if self._sender.get(self._claim) != self._nonce:
                        raise RuntimeError("wake relay sender activation ownership changed")
                    marker = self._fence_key(self.rank)
                    self._sender.set(marker, b"1")
                    if self._sender.get(marker) != b"1":
                        raise ValueError("wake relay sender retirement marker is invalid")
                with self._lock:
                    self._fenced = True
            except BaseException as primary:
                self._report(primary)
                raise

    def notify(self):
        """Called only by the catalog's serialized CPU transfer worker."""
        self._process()
        with self._sending:
            with self._lock:
                if not self._started or self._closing or self._closed:
                    raise RuntimeError("wake relay is not open for transfer-worker notifications")
                if self._failure is not None:
                    raise RuntimeError("wake relay is failed") from self._failure
            try:
                if self.rank == 0:
                    self._wake()
                    return
                if self._generation:
                    previous = self._ack(self.rank, self._generation)
                    if not self._sender.check([previous]):
                        return  # This worker's pre-ACK publication is covered by that wake.
                    if self._sender.get(previous) != b"1":
                        raise ValueError("wake relay received an invalid generation ACK")
                    self._sender.delete_key(previous)
                if self._generation == MAX_WORD:
                    raise OverflowError("wake relay generation exhausted")
                with self._lock:
                    self._generation += 1  # queue_push may commit before reporting a transport error.
                    generation = self._generation
                self._sender.queue_push(self._queue, f"{self.rank}:{generation}")
                with self._lock:
                    self._notifications += 1
            except BaseException as primary:
                self._report(primary)
                raise

    def _receive(self):
        try:
            while True:
                try:
                    payload = self._receiver.queue_pop(self._queue)
                except Exception as error:
                    if timed_out(error, [self._queue], self.timeout):
                        continue  # No scheduler/business wake on an acknowledged transport timeout.
                    raise
                if type(payload) is not bytes or not 0 < len(payload) <= MAX_PACKET_BYTES:
                    raise ValueError("wake relay packet exceeds its bounded schema")
                try:
                    words = payload.decode("ascii").split(":")
                    if len(words) != 2:
                        raise ValueError
                    rank, generation = map(int, words)
                except (ValueError, UnicodeDecodeError):
                    raise ValueError("wake relay packet is not a rank/generation pair") from None
                if payload != f"{rank}:{generation}".encode("ascii"):
                    raise ValueError("wake relay packet is not canonical")
                if rank == generation == 0:
                    if not self._stop_attempted:
                        raise ValueError("wake relay received an unissued stop")
                    return
                if not 0 < rank < self.world or not 1 <= generation <= MAX_WORD:
                    raise ValueError("wake relay rank/generation is outside the cohort")
                with self._lock:
                    if generation != self._received.get(rank, 0) + 1:
                        raise ValueError("wake relay generation is stale or skipped")
                    self._received[rank] = generation  # Journal a possibly committed ACK.
                self._receiver.set(self._ack(rank, generation), b"1")
                self._wake()
        except BaseException as primary:
            self._report(primary)
            raise

    def _control_budget(self, deadline):
        self._control.set_timeout(min(self.timeout, timedelta(seconds=max(0., deadline - time.monotonic()))))

    def _transport_lease(self, deadline):
        if not self._transport_attempted:
            return False
        self._control_budget(deadline)
        present = self._control.check([TRANSPORT_LEASE])
        if not present:
            if self._transport_owned and not self._transport_retire_attempted:
                raise RuntimeError("retention wake transport lease disappeared before retirement")
            self._transport_owned = False
            return False
        self._control_budget(deadline)
        actual = self._control.get(TRANSPORT_LEASE)
        if actual == self._nonce:
            self._transport_owned = True
            return True
        if self._transport_owned:
            raise RuntimeError("retention wake transport lease changed owners")
        return False  # A refused constructor must never touch another activation's queue.

    def close(self, timeout):
        """Retire after the all-rank fenced-sender agreement; uncertainty keeps owners."""
        self._process()
        if type(timeout) not in (int, float) or not 0 <= timeout <= threading.TIMEOUT_MAX:
            raise ValueError("wake relay close timeout must fit the runtime's nonnegative wait range")
        if not self._closing_lock.acquire(blocking=False):
            raise RuntimeError("wake relay close already has an owner")
        try:
            if self._closed:
                return
            deadline = time.monotonic() + timeout
            with self._lock:
                self._closing = True
                if self._started and not self._fenced:
                    raise_failures(self._failure, [RuntimeError("wake relay requires its sender fence before close")])
            errors = []
            if self.rank == 0 and self._fenced and not self._senders_fenced and self._control is not None:
                # A different connection's barrier cannot order sender bytes.
                # These markers are published only after each sender's readback.
                for rank in range(1, self.world):
                    self._control_budget(deadline)
                    if self._control.get(self._fence_key(rank)) != b"1":
                        raise ValueError("wake relay peer sender retirement marker is invalid")
                self._senders_fenced = True
            if self._worker is not None:
                if self._started and not self._worker.retired.is_set() and not self._stop_attempted:
                    self._stop_attempted = True
                    try:
                        self._control_budget(deadline)
                        self._control.queue_push(self._queue, "0:0")
                    except BaseException as error:
                        errors.append(error)
                try:
                    self._worker.drain(self._thread, timeout=max(0., deadline - time.monotonic()),
                                       cancel_unentered=not self._started)
                except BaseException as error:
                    if self._worker.retired.is_set() and self._worker.failure is not None:
                        if self._failure is None:
                            self._failure = self._worker.failure
                    else:
                        errors.append(error)
                if not self._worker.retired.is_set():
                    raise_failures(None, errors)  # Its callback/socket references remain owned for retry.
            if self._control is not None:
                try:
                    if self._receiver is not None and self._started and not self._receiver_fenced:
                        # A terminal wake callback can throw after a reply-free
                        # ACK set. The retired worker now hands off its clone;
                        # fence those ACK bytes before deleting their keys.
                        self._receiver.set_timeout(min(self.timeout, timedelta(
                            seconds=max(0., deadline - time.monotonic()))))
                        if self._receiver.get(self._claim) != self._nonce:
                            raise RuntimeError("wake relay receiver activation ownership changed")
                        self._receiver_fenced = True
                    self._control_budget(deadline)
                    transport_owned = self.rank == 0 and self._transport_lease(deadline)
                    if transport_owned:
                        pending = self._control.queue_len(self._queue)
                        if pending:
                            if self._failure is None or pending > self.world:
                                raise RuntimeError("wake relay closed before its bounded queue/data workers retired")
                            # Terminal receiver failure: no callback can consume
                            # these owned notifications and no data worker can append.
                            for _ in range(pending):
                                self._control_budget(deadline)
                                self._control.queue_pop(self._queue, block=False)
                        for rank, generation in self._received.items():
                            self._control_budget(deadline)
                            self._control.delete_key(self._ack(rank, generation))
                        if self._senders_fenced:
                            for rank in range(1, self.world):
                                self._control_budget(deadline)
                                self._control.delete_key(self._fence_key(rank))
                    self._control_budget(deadline)
                    if self._claim_attempted and self._control.check([self._claim]):
                        self._control_budget(deadline)
                        if self._control.get(self._claim) == self._nonce:
                            self._control_budget(deadline)
                            self._control.delete_key(self._claim)
                        elif self._claim_owned:
                            raise RuntimeError("wake relay activation ownership changed")
                    if transport_owned:
                        if errors:
                            raise_failures(self._failure, errors)  # Uncertain earlier cleanup keeps exclusivity.
                        self._control_budget(deadline)
                        if self._control.queue_len(self._queue):
                            raise RuntimeError("retention wake queue is not empty after receiver retirement")
                        self._transport_retire_attempted = True
                        self._control_budget(deadline)
                        self._control.delete_key(TRANSPORT_LEASE)
                        self._transport_owned = False
                except BaseException as error:
                    errors.append(error)
                    raise_failures(self._failure, errors)
            primary = self._failure
            self._sender = self._receiver = self._control = self._worker = self._thread = None
            self._wake = self._fatal = None
            self._closed = True
            raise_failures(primary, errors)
        finally:
            self._closing_lock.release()

    def record(self):
        with self._lock:
            return {"rank": self.rank, "world": self.world, "started": self._started, "closed": self._closed,
                    "cpu_receiver_workers": int(self._worker is not None),
                    "transport_queue_name_bound": int(self.world > 1),
                    "sender_fenced": self._fenced,
                    "posted_generations": self._notifications, "failed": self._failure is not None,
                    "qualification": "CPU store protocol only; V100 progress qualification pending"}

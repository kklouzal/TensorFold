"""Requests submit from any thread; one worker thread runs the rounds, and a slow client only fills its own queue."""
# Background requests go last; one decoding yields its lane to a waiting request and re-queues to replay later.

from __future__ import annotations

import itertools
from collections import OrderedDict
from dataclasses import dataclass, replace
import queue
import sys
import threading
import time
from typing import Any, Callable

from tensorfold.cleanup import finish, raise_failures, rollback
from tensorfold.server.errors import CapacityError, RequestError
from tensorfold.server.request_limits import RequestLimit, optional_limit

from .memory_gate import NoRoom
from .output_mailbox import OutputMailbox
from .streams import Stream, mark_error_state_retired

TICK_S = 15.0           # a request waiting this long for tokens (queued or prefilling) ticks ``emit([])``: a keepalive


@dataclass(frozen=True)
class AdmissionStatus:
    """One immutable observation of a waiting request; durations use monotonic time."""
    ticket: int
    enqueued_at: float
    input_bytes: int
    deadline: float | None
    blocked_reason: str | None = None
    changed_at: float | None = None
    ready_at: float | None = None
    terminal_reason: str | None = None
    issued_at: float | None = None
    principal: str | None = None
    phase: str = "waiting"
    gpu_admitted: bool = False
    readmission_issued_at: float | None = None
    actual_input_bytes: int | None = None


@dataclass(frozen=True)
class AdmissionOutcome:
    ticket: int
    reason: str
    waited_s: float
    at: float
    input_bytes: int
    admission_s: float | None


@dataclass(frozen=True)
class DeliveryOwner:
    """Actual CPU callback/output owner, using the existing logical-request lifetime proof."""
    limit: RequestLimit
    mailbox: OutputMailbox
    cancelled: Any
    emit: Callable
    progress: Callable | None

    def require_live(self):
        from tensorfold.server.cancellation import RequestCancelled

        if self.cancelled.is_set() or not self.limit.delivery_active(self.mailbox):
            raise RequestCancelled("the original retained request delivery owner has retired")


class InputClaim:
    """The Waiting entry and input credit owned by one actual logical request."""
    def __init__(self, authority, ticket, payload, source, *, logical):
        self.authority, self.ticket, self.payload, self.source = authority, ticket, payload, source
        self.closed, self.logical = False, logical

    def __getitem__(self, index):
        return (0, self.ticket, self.payload)[index]

    def __iter__(self):
        return iter((0, self.ticket, self.payload))

    def backend_retired(self, worker):
        with self.authority.not_empty:
            stream, box = self.payload
            if box is worker:
                host = getattr(stream, "retirement_host", None)
                if host is not None:
                    host.retire_backend()
                    stream.retirement_host = None
                stream.admission_status = replace(stream.admission_status, phase="between_subruns")
                self.authority.revision += 1
                self.authority.not_empty.notify()
            elif stream.admission_status.phase != "between_subruns":
                raise RuntimeError("input credit belongs to another live backend worker")

    def close(self):
        with self.authority.not_full:
            if self.closed:
                return
            current = self.authority.queue.get(self.ticket)
            if current is not None and current is not self:
                raise RuntimeError("logical input credit changed its authoritative queue entry")
            if current is self:
                del self.authority.queue[self.ticket]
            self.closed = True
            self.source = None
            self.authority.revision += 1
            self.authority.not_full.notify()
            self.authority.not_empty.notify()


class Waiting(queue.PriorityQueue):
    """One owned request queue: legacy background priority or immutable FIFO tickets."""

    def __init__(self, *, policy="legacy", max_input_bytes=None, max_principal_bytes=None, queue_timeout_s=None) -> None:
        self.policy, self.max_input_bytes = policy, max_input_bytes
        self.max_principal_bytes = max_principal_bytes
        self.queue_timeout_s = queue_timeout_s
        super().__init__()
        self.revision = 0
        self._order = itertools.count()
        self._wake_marker = object()
        self._wake_pending = False
        self._wake_entry = self._stop_entry = None
        self._stop_published = self._stop_queued = False

    def _init(self, maxsize):
        self.queue = OrderedDict() if self.policy == "fifo" else []

    def _put(self, item):
        if self.policy == "fifo":
            self.queue[item[1]] = item
        else:
            super()._put(item)

    def put(self, item, block: bool = True, timeout: float | None = None) -> None:
        if self.policy == "fifo":
            self.publish(item)
            return
        super().put((1 if item[0].background else 0, next(self._order), item), block, timeout)

    def publish(self, item, *, claim=None, register=None, reserved_bytes=None, source=None):
        stream, _box = item
        if self.policy != "fifo":
            raise RuntimeError("logical input credit requires the FIFO queue role")
        amount = stream.admission_status.input_bytes if reserved_bytes is None else reserved_bytes
        if type(amount) is not int or amount < stream.admission_status.input_bytes:
            raise ValueError("input reservation must cover the actual request and its supported future growth")
        if claim is not None:
            with self.not_full:
                if (type(claim) is not InputClaim or claim.closed or claim.authority is not self
                        or self.queue.get(claim.ticket) is not claim):
                    raise RuntimeError("logical input reservation is not owned by this queue")
                previous = claim.payload[0].admission_status
                if previous.phase != "between_subruns" or previous.principal != stream.principal:
                    raise RuntimeError("logical input reservation still owns another backend or principal")
                if source is not claim.source or amount > previous.input_bytes:
                    raise ValueError("logical input changed its fixed replay or complete future input commitment")
                stream.admission_status = replace(previous, phase="waiting", blocked_reason=None,
                    terminal_reason=None, readmission_issued_at=None,
                    actual_input_bytes=stream.admission_status.input_bytes)
                stream.admission_ticket = claim.ticket
                claim.payload = item
                self.revision += 1
                self.not_empty.notify()
            return claim
        with self.not_full:
            total, principal_total = 0, 0
            for entry in self.queue.values():
                status = entry[2][0].admission_status
                total += status.input_bytes
                if status.principal == stream.admission_status.principal:
                    principal_total += status.input_bytes
            if amount > self.max_input_bytes - total:
                raise CapacityError("waiting request input-byte capacity reached (includes reserved live credits); retry later")
            if self.max_principal_bytes is not None and amount > self.max_principal_bytes - principal_total:
                raise CapacityError("principal waiting-input byte capacity reached (includes reserved live credits); retry later")
            ticket = next(self._order)
            if ticket >= 1 << 63:
                raise RuntimeError("FIFO admission ticket identity exhausted")
            now = time.monotonic()
            stream.admission_status = replace(stream.admission_status, ticket=ticket, enqueued_at=now,
                input_bytes=amount, actual_input_bytes=stream.admission_status.input_bytes,
                deadline=None if self.queue_timeout_s is None else now + self.queue_timeout_s)
            stream.admission_ticket = ticket
            claim = InputClaim(self, ticket, item, source, logical=register is not None)
            if register is not None:
                register(claim)
            self._put(claim)
            self.unfinished_tasks += 1
            self.revision += 1
            self.not_empty.notify()
            return claim

    def get(self, block: bool = True, timeout: float | None = None):
        return super().get(block, timeout)[2]

    def get_control(self, *, since=None, deadline=None):
        """A full media cohort waits for completion without consuming queued requests."""
        with self.not_empty:
            while (not self._wake_pending and not (self._stop_queued and (self.policy == "fifo" or not self.queue))
                   and (since is None or self.revision == since)):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return self._wake_marker
                self.not_empty.wait(None if remaining is None else min(remaining, threading.TIMEOUT_MAX))
            if self._wake_pending:
                self._wake_pending = False
                return self._wake_marker
            if since is not None and self.revision != since:
                return self._wake_marker
            self._stop_queued = False
            return None

    def requests(self, *, include_reserved=False):
        """Bounded ordered references; eligibility never runs under the queue lock."""
        with self.mutex:
            return self.revision, (tuple(entry for entry in self.queue.values() if include_reserved or
                entry[2][0].admission_status.phase == "waiting") if self.policy == "fifo" else tuple(sorted(self.queue)))

    def activate(self, entry):
        with self.mutex:
            if self.queue.get(entry[1]) is not entry:
                raise RuntimeError("GPU admission lost its owned FIFO input entry")
            stream = entry[2][0]
            stream.admission_status = replace(stream.admission_status, phase="active", gpu_admitted=True)

    def requeue(self, stream, *, reason):
        with self.not_empty:
            entry = self.queue.get(stream.admission_ticket)
            if entry is None or entry[2][0] is not stream or stream.admission_status.phase != "active":
                raise RuntimeError("same-request suspension lost its original FIFO input credit")
            stream.admission_status = replace(stream.admission_status, phase="waiting", blocked_reason=reason,
                                               readmission_issued_at=None)
            stream.queued = time.perf_counter()
            self.revision += 1
            self.not_empty.notify()

    def remove_request(self, entry):
        """The serialized worker removes a selected/retired request without changing its ticket."""
        with self.not_full:
            if self.policy != "fifo":
                raise RuntimeError("FIFO request removal requires its FIFO queue role")
            if self.queue.get(entry[1]) is not entry:
                raise RuntimeError("FIFO selected request lost its authoritative queue entry")
            del self.queue[entry[1]]
            self.not_full.notify()

    def _qsize(self):
        # Controls have fixed owner cells rather than heap/task entries. A
        # notification interruption cannot grow the request heap on retry.
        size = super()._qsize()
        return size + int(self._wake_pending) + int(self._stop_queued)

    def _get(self):
        if self._wake_pending and (not self.queue or self.policy != "fifo" and self._wake_entry < self.queue[0]):
            item = self._wake_entry
            self._wake_pending = False
            return item
        if self.queue:
            return self.queue.popitem(last=False)[1] if self.policy == "fifo" else super()._get()
        if self._stop_queued:
            self._stop_published = True
            self._stop_queued = False
            return self._stop_entry
        raise RuntimeError("waiting queue has no owned request or control")

    def qsize(self) -> int:
        with self.mutex:
            return (sum(entry[2][0].admission_status.phase == "waiting" for entry in self.queue.values())
                    if self.policy == "fifo" else super()._qsize())

    def is_wake(self, value) -> bool:
        return value is self._wake_marker

    def wake(self) -> None:
        """One coalesced owner-control marker; ordinary queued work keeps priority."""
        self._control(1, self._wake_marker, wake=True)

    def _control(self, priority, marker, *, wake):
        with self.not_full:
            if self._wake_pending if wake else self._stop_published or self._stop_queued:
                if not wake:
                    self._stop_published = True
                # A prior publication can precede a failed notification. A
                # retry must wake the idle owner even when no new cell is needed.
                self.not_empty.notify()
                return
            entry = (priority, next(self._order), marker)
            if wake:
                self._wake_entry = entry
                self._wake_pending = True
            else:
                self._stop_entry = entry
                self._stop_queued = True
                self._stop_published = True
            self.not_empty.notify()

    def stop(self) -> None:
        """Wake an idle worker to stop: None comes after every waiting request."""

        self._control(2, None, wake=False)

    def foreground(self) -> bool:
        """Whether a foreground request waits."""

        with self.mutex:
            return (any(entry[2][0].admission_status.phase == "waiting" for entry in self.queue.values())
                    if self.policy == "fifo" else bool(self.queue) and self.queue[0][0] == 0)


class Scheduler:
    """Construct, publish the owner, then call ``start`` before submitting.

    A failed start can have launched a native actor. The published scheduler
    retains it and its decoder until ``close`` acquires its retired work scope
    and joins;
    an unresolved start remains owned for retry or process containment.
    """

    def __init__(self, decoder: Any, *, max_streams: int = 4,
                 max_pending_requests: int | None = None, admission_policy: str = "legacy",
                 max_waiting_input_bytes: int | None = None,
                 max_waiting_input_bytes_per_principal: int | None = None, queue_timeout_s: float | None = None,
                 output_queue_tokens: int | None = None) -> None:
        if type(max_streams) is not int or max_streams <= 0:
            raise ValueError("max_streams must be a positive integer")
        if admission_policy not in ("legacy", "fifo"):
            raise ValueError("admission_policy must be legacy or fifo")
        maximum_bytes = optional_limit(max_waiting_input_bytes, "max_waiting_input_bytes")
        principal_bytes = optional_limit(max_waiting_input_bytes_per_principal, "max_waiting_input_bytes_per_principal")
        output_tokens = optional_limit(output_queue_tokens, "output_queue_tokens")
        if queue_timeout_s is not None and (type(queue_timeout_s) not in (int, float)
                or not 0 < queue_timeout_s <= threading.TIMEOUT_MAX):
            raise ValueError("queue_timeout_s must be positive, finite and representable by the wait runtime")
        if admission_policy == "fifo" and maximum_bytes is None:
            raise ValueError("FIFO admission requires an explicit max_waiting_input_bytes limit")
        if admission_policy == "legacy" and (maximum_bytes is not None or principal_bytes is not None
                                            or queue_timeout_s is not None or output_tokens is not None):
            raise ValueError("waiting-input and queue-timeout limits require FIFO admission")
        self.admission_policy = admission_policy
        self.max_waiting_input_bytes = maximum_bytes
        self.max_waiting_input_bytes_per_principal = principal_bytes
        self.queue_timeout_s = None if queue_timeout_s is None else float(queue_timeout_s)
        self.output_queue_tokens = output_tokens
        self.decoder = decoder
        self.max_streams = max_streams
        self.max_pending_requests = optional_limit(max_pending_requests, "max_pending_requests")
        self._request_limit = RequestLimit(max_pending_requests) if max_pending_requests is not None else None
        self._request_boxes: set[OutputMailbox] = set()
        self._error_retirements = {}
        self._submitted = False
        self.waiting = Waiting(policy=admission_policy, max_input_bytes=maximum_bytes,
                               max_principal_bytes=principal_bytes, queue_timeout_s=self.queue_timeout_s)
        self._media_wake = self.waiting.wake
        self.held: tuple | None = None               # a request waiting for memory, admitted before any other
        self._media_held_wait = False
        self._retiring: list[Stream] = []            # completed device work whose CPU consumers still retire
        self._queue_outcomes = (None, (0, 0, 0, 0))  # last observation + admitted/cancelled/expired/refused counts
        self.boxes: dict[int, OutputMailbox] = {}
        self._state_lock = threading.Lock()
        self._closing = True                     # admission opens only after explicit startup commits
        self._stop_published = False
        self._stopping = False                    # worker consumed the marker; accepted work still drains
        self._failure: BaseException | None = None
        self.serving_profiles = None
        self._pending_manifest = None
        self._manifest_sequence = 0
        self._manifest_status = {"configured": None, "pending": False, "effective": None,
                                 "request_epoch": 0, "error": None}
        self._cleanup_failure: BaseException | None = None
        self._worker_done = threading.Event()
        self._worker_entered = threading.Event()
        self._worker_lease = threading.Lock()
        self._start_attempted = False
        self._started_ok = False
        self._start_error: BaseException | None = None
        self._arrival_applied = False
        self._serving_profile_epoch: int | None = None
        self._serving_profile_available = True
        self._arrival_hook = self.waiting.foreground
        try:
            self._arrival_original = decoder.arrived
        except AttributeError:
            self._has_arrival = False
        else:
            self._has_arrival = True
        self.yields = 0                              # background streams that gave up their lane
        configure = getattr(decoder, "configure_admission_policy", None)
        if configure is not None:
            configure(admission_policy)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        if self._request_limit is not None:
            self.submit = self._request_limit.wrap(self.submit, inherit_scope=True)

    def configure_requests(self, limit: RequestLimit) -> None:
        """Share the App's logical owner before the endpoint publishes any work."""
        with self._state_lock:
            if self._submitted or self._worker_done.is_set():
                raise RuntimeError("request admission must be configured before the first submission")
            if self._request_limit is not None:
                raise RuntimeError("CUDA request admission was already configured")
            wrapped = limit.wrap(self.submit, inherit_scope=True)
            self.max_pending_requests = limit.maximum
            self._request_limit = limit
            self.submit = wrapped

    def request_runtime(self, publish: Callable[[], Any]) -> Any:
        """Publish a bounded host control and wake idle retained work without polling."""
        with self._state_lock:
            if self._closing:
                raise RuntimeError("CUDA scheduler is closing; runtime updates are rejected")
            if self.serving_profiles is not None:
                profile = self.serving_profiles.record()
                if not profile["available"] or profile["pending"] is not None:
                    raise ValueError("execution controls require a ready serving profile with no pending structural transition")
            result = publish()
            self.waiting.wake()
            return result

    def request_execution_manifest(self, relative_name: str) -> dict:
        """Queue verified profile replacement; file reads belong to the TP worker."""
        from pathlib import PurePosixPath

        if (not isinstance(relative_name, str) or not 1 <= len(relative_name) <= 1024
                or "\\" in relative_name or "\x00" in relative_name
                or PurePosixPath(relative_name).is_absolute()
                or any(part in ("", ".", "..") for part in relative_name.split("/"))):
            raise ValueError("profile_manifest must be a relative file within the startup evidence directory")
        with self._state_lock:
            if self._closing:
                raise RuntimeError("CUDA scheduler is closing; runtime updates are rejected")
            if self.serving_profiles is not None:
                profile = self.serving_profiles.record()
                if not profile["available"] or profile["pending"] is not None:
                    raise ValueError("execution controls require a ready serving profile with no pending structural transition")
            if getattr(self.decoder, "execution_manifest_installer", None) is None:
                raise ValueError("execution profile verification was not enabled at startup")
            self._manifest_sequence += 1
            self._pending_manifest = (self._manifest_sequence, relative_name)
            self._manifest_status = {**self._manifest_status, "configured": relative_name, "pending": True,
                                     "request_epoch": self._manifest_sequence, "error": None}
            self.waiting.wake()
            return dict(self._manifest_status)

    def execution_manifest_record(self) -> dict:
        with self._state_lock:
            return dict(self._manifest_status)

    @property
    def pending_requests(self) -> int | None:
        """Configured logical ownership count; uncapped mode has no such registry."""
        return self._request_limit.used if self._request_limit is not None else None

    def _terminal(self, box: OutputMailbox, kind: str, value: Any, *, retired=False) -> None:
        if kind == "error" and self.serving_profiles is not None:
            if not retired:
                value = self.serving_profiles.error_outcome(value)
            if self.serving_profiles.failure is None:
                signal = self._error_retirements.get(box)
                if signal is not None:
                    signal.set()
        if self._request_limit is not None:
            with self._state_lock:
                try:
                    self._request_limit.release(box)
                except BaseException as error:
                    if self.admission_policy == "fifo":
                        self._admission_failure(error)
                    raise
            # Keep the reply owner journaled through a fallible publication.
            box.put((kind, value))
            with self._state_lock:
                self._request_boxes.discard(box)
        else:
            box.put((kind, value))
        self._error_retirements.pop(box, None)

    def _unregister_unpublished(self, box, signal=None):
        finish((lambda: self._error_retirements.pop(box, None), lambda: self._request_boxes.discard(box)))
        if signal is not None:
            signal.set()  # No model work was published; a completed rollback is safe for this caller.

    def _stop_requests(self) -> None:
        """Caller holds the publication lock; wake a draining worker after closing admission."""
        self._closing = True
        self.waiting.stop()
        self._stop_published = True

    def start(self) -> "Scheduler":
        """Start once, after the caller has journaled this scheduler.

        The acceptance lock gates decoder access until startup commits. Even when
        ``Thread.start`` spawns before raising, that actor cannot use the
        decoder. A failed hook assignment also remains owned for restoration.
        """

        with self._state_lock:
            if self._worker_done.is_set() or self._start_attempted or self._arrival_applied:
                raise RuntimeError("CUDA scheduler has closed or startup was already attempted")
            try:
                if self._has_arrival:
                    self._arrival_applied = True
                    self.decoder.arrived = self._arrival_hook
                self._start_attempted = True
                self.thread.start()
                self._worker_entered.wait()
                self._started_ok = True
                self._closing = False
            except BaseException as error:
                self._closing = True
                self._started_ok = False
                self._start_error = error
                raise
        return self

    def close(self) -> None:
        """Reject new submissions, drain accepted work, then join and release the decoder.

        Acceptance and closing share a linearization lock. The marker comes
        after every accepted request, and may be consumed while lanes are live.
        Concurrent/repeated closes join the same worker and publish its failure.
        """

        if threading.current_thread() is self.thread:
            raise RuntimeError("the decoding worker cannot join itself")
        with self._state_lock:
            self._closing = True
            if self._request_limit is not None:
                self._request_limit.close()
            attempted = self._start_attempted
            if self._started_ok and not self._stop_published and not self._worker_done.is_set():
                self.waiting.stop()
                self._stop_published = True
        if attempted:
            if not self._worker_entered.is_set():
                raise RuntimeError("CUDA scheduler start did not settle; decoder and actor owners remain retained") from self._start_error
            # Wait for the target-owned scope before joining. CPython can mark
            # a Thread stopped when an asynchronous exception interrupts join;
            # its bookkeeping alone cannot prove decoder work has retired.
            with self._worker_lease:
                pass
            self.thread.join()
        else:
            self._worker_done.set()              # construction never published an actor
        self._clear_failure_frames()
        with self._state_lock:
            if self._arrival_applied:
                self.decoder.arrived = self._arrival_original
                self._arrival_applied = False
            if self._cleanup_failure is None:
                self.decoder = None
                self._arrival_original = None
        if self._failure is not None:
            raise RuntimeError("CUDA scheduler worker failed during shutdown") from self._failure

    def submit(self, prompt: list[int], count: int, sampling: Any, draft: bool,
               emit: Callable[[list[int]], bool | None], stop_eos: bool = True, *, vision: Any = None,
               constraint: Any = None, background: bool = False, probabilities: Any = None,
               progress: Callable[[int, int, int, float], bool] | None = None, penalties: Any = None,
               token_limit: Callable[[list[int], list[int]], tuple[int, bool]] | None = None,
               profile_epoch: int | None = None, _error_retired=None,
               eligibility: Callable[[], str | None] | None = None) -> dict:
        """Decode one request; ``emit`` runs on the calling thread and returns True to stop. Returns its stats.
        While no tokens come (queued, or a long prompt filling) ``emit([])`` ticks every TICK_S: the caller keeps its
        connection alive and says whether its client left, and a request whose client left stops even mid-prompt.
        ``progress(total, cached, processed, ms since admission)``, also on the calling thread, receives the latest
        reported prompt progress (``Stream.on_fill``). Intermediate updates may coalesce while the caller handles
        output; True stops it."""

        if type(count) is not int:
            raise ValueError("a CUDA sequence requires an integer reply token allowance")
        if profile_epoch is not None and (type(profile_epoch) is not int or profile_epoch < 0):
            raise ValueError("a serving profile lease requires a nonnegative integer epoch")
        if eligibility is not None and (self.admission_policy != "fifo" or not callable(eligibility)):
            raise ValueError("nonblocking request eligibility requires FIFO admission")
        if self.admission_policy == "fifo":
            if self._request_limit is None:
                raise ValueError("FIFO admission requires a finite logical request count before submission")
            if type(prompt) not in (list, tuple):
                raise ValueError("FIFO prompt IDs must be a bounded list/tuple of signed-int32 nonnegative token IDs")
            if len(prompt) * sys.getsizeof(0) > self.max_waiting_input_bytes:
                raise CapacityError("waiting request input-byte capacity reached; retry later")
            if any(type(token) is not int or not 0 <= token < 1 << 31 for token in prompt):
                raise ValueError("FIFO prompt IDs must be a bounded list/tuple of signed-int32 nonnegative token IDs")
        box = OutputMailbox(max(1, count), pending_tokens=self.output_queue_tokens)
        if self.admission_policy == "fifo":
            box.set_wake(self._media_wake)
        cancel = threading.Event()
        stream = Stream(list(prompt), max(1, count), sampling, draft=draft, stop_eos=stop_eos, vision=vision,
                        constraint=constraint, background=background, probabilities=probabilities, cancel=cancel,
                        penalties=penalties, token_limit=token_limit, profile_epoch=profile_epoch)
        if self.admission_policy == "fifo":
            stream.principal = self._request_limit.current_principal()
            if self.max_waiting_input_bytes_per_principal is not None and stream.principal is None:
                raise ValueError("per-principal waiting-input bytes require an authenticated request principal")
            prepare_enqueue = getattr(self.decoder, "prepare_enqueue", None)
            if prepare_enqueue is not None:
                prepare_enqueue(stream)
            else:
                validate = getattr(self.decoder, "validate_request", None)
                if validate is not None:
                    validate(stream)
            source_bytes = 0
            if vision is not None:
                from tensorfold.vision.media_frontend import PreparedReplayMedia

                if type(vision) is not PreparedReplayMedia:
                    raise ValueError("FIFO media input requires its actual immutable encoded replay owner")
                source_bytes = sum(ref.byte_length for ref in vision.replay.refs)
            size = sys.getsizeof(stream.prompt) + sum(sys.getsizeof(token) for token in stream.prompt) + source_bytes
            now = time.monotonic()
            stream.admission_status = AdmissionStatus(-1, now, size,
                None if self.queue_timeout_s is None else now + self.queue_timeout_s, principal=stream.principal)
            stream.admission_eligibility = eligibility
            stream.delivery_owner = DeliveryOwner(self._request_limit, box, cancel, emit, progress)
            stream._fifo_started = False
            stream.output_capacity = box.production_capacity
        stream.emit = lambda new: (box.put(("tokens", new)), cancel.is_set())[1]
        if progress is not None:
            stream.on_fill = lambda rows: box.put(("filled", rows))
        try:
            # Publication can succeed before an interrupt is delivered. Own
            # cancellation before entering the acceptance scope, so accepted
            # work never loses its stop signal when its caller cannot wait.
            with self._state_lock:
                if self._closing:
                    raise RuntimeError("CUDA scheduler is closing; new requests are rejected")
                if self.serving_profiles is not None:
                    binding = self.serving_profiles.gate.current()
                    if binding is None or not binding.scope.locked() or binding.epoch != profile_epoch:
                        from .profile_binding import ProfileUnavailable

                        raise ProfileUnavailable("scheduler submission requires an admitted model-profile lease")
                self._check_profile(stream.profile_epoch)
                try:
                    if self._request_limit is not None:
                        self._request_limit.borrow(box)
                        self._request_boxes.add(box)
                    if _error_retired is not None:
                        self._error_retirements[box] = _error_retired
                except BaseException as primary:
                    if self._request_limit is not None:
                        self._request_limit.rollback_registration(
                            self, box, primary, lambda: self._unregister_unpublished(box, _error_retired), self._stop_requests)
                    else:
                        rollback(self, primary, lambda: self._unregister_unpublished(box, _error_retired))
                self._submitted = True
                try:
                    if self.admission_policy == "fifo":
                        reserve = getattr(self.decoder, "input_reservation_bytes", None)
                        amount = size if reserve is None else reserve(stream)
                        source = None if vision is None else vision.replay
                        self._request_limit.publish_inputs(box, self.waiting, lambda claim, register:
                            self.waiting.publish((stream, box), claim=claim, register=register,
                                                 reserved_bytes=amount, source=source))
                    else:
                        self.waiting.put((stream, box))
                except BaseException as error:
                    if self.admission_policy == "fifo" and isinstance(error, CapacityError):
                        # Waiting's FIFO capacity refusal is specified before
                        # ticket/heap publication. Retire only this clean
                        # registration; the healthy endpoint remains open.
                        self._request_limit.rollback_registration(self, box, error,
                            lambda: self._unregister_unpublished(box, _error_retired), self._stop_requests)
                    if self._request_limit is not None:
                        # Publication may have completed. Stop accepting new
                        # owners, drain the worker, then retire the journal.
                        self._closing = True
                        self._failure = error
                        rollback(self, error, lambda: finish((self._request_limit.close, self._stop_requests)))
                    raise
            while True:
                try:
                    kind, value = box.get(timeout=TICK_S)
                except queue.Empty:
                    if not cancel.is_set() and emit([]):
                        self._request_cancel(cancel)  # the client left: retire at the next owned boundary
                    continue
                if kind == "tokens":
                    try:
                        if not cancel.is_set() and emit(value):
                            self._request_cancel(cancel)
                    finally:
                        box.complete_delivery(value)
                elif kind == "filled":
                    if not cancel.is_set() and progress(len(stream.prompt), stream.cached, value,
                                                         (time.perf_counter() - stream.admitted) * 1000.0):
                        self._request_cancel(cancel)
                elif kind == "error":
                    raise value
                else:
                    if isinstance(value, dict):
                        value = {**value, "output_ownership": box.record()}
                    return value
        except BaseException as primary:
            rollback(cancel, primary, lambda: self._request_cancel(cancel))

    def _request_cancel(self, cancel):
        cancel.set()
        if self.admission_policy == "fifo" or getattr(self.decoder, "prepare_request", None) is not None:
            self.waiting.wake()  # CPU readiness can leave the coordinator asleep on its existing queue.

    @staticmethod
    def _queue_state(stream, reason, *, input_ready=False):
        status = stream.admission_status
        if status.blocked_reason != reason or status.changed_at is None or input_ready and status.ready_at is None:
            now = time.monotonic()
            stream.admission_status = replace(status, blocked_reason=reason, changed_at=now,
                ready_at=now if input_ready and status.ready_at is None else status.ready_at)

    def queue_record(self):
        """Administrative FIFO observations, without new generation events or client fields."""
        revision, entries = self.waiting.requests()
        _revision, charged = self.waiting.requests(include_reserved=True)
        now = time.monotonic()
        rows = []
        for _priority, _ticket, (stream, _box) in entries:
            status = getattr(stream, "admission_status", None)
            if status is not None:
                rows.append({"ticket": status.ticket, "age_s": now - status.enqueued_at,
                    "queue_wait_s": (now if status.issued_at is None else status.issued_at) - status.enqueued_at,
                    "phase": "waiting" if status.readmission_issued_at is None and status.issued_at is None else "admitting",
                    "input_bytes": status.input_bytes, "blocked_reason": status.blocked_reason,
                    "actual_input_bytes": status.actual_input_bytes,
                    "pending_terminal_reason": status.terminal_reason,
                    "admission_issued_at_monotonic_s": status.issued_at,
                    "readmission_issued_at_monotonic_s": status.readmission_issued_at,
                    "enqueued_at_monotonic_s": status.enqueued_at,
                    "first_observed_ready_at_monotonic_s": status.ready_at,
                    "status_changed_at_monotonic_s": status.changed_at,
                    "deadline_monotonic_s": status.deadline})
        outcome, counts = self._queue_outcomes
        return {"policy": self.admission_policy, "revision": revision, "waiting": rows,
                "waiting_input_bytes": sum(row["input_bytes"] for row in rows),
                "charged_live_input_bytes": sum(entry[2][0].admission_status.input_bytes for entry in charged),
                "reserved_live_requests": len(charged) - len(entries),
                "max_waiting_input_bytes": self.max_waiting_input_bytes, "queue_timeout_s": self.queue_timeout_s,
                "max_waiting_input_bytes_per_principal": self.max_waiting_input_bytes_per_principal,
                "output_queue_tokens": self.output_queue_tokens,
                "input_charge": "complete supported future prompt list/int extents and fixed replay; held through logical caller and backend retirement",
                "outcomes": dict(zip(("admitted", "cancelled", "expired", "refused"), counts)),
                "last_outcome": None if outcome is None else {
                    "ticket": outcome.ticket, "reason": outcome.reason, "waited_s": outcome.waited_s,
                    "at_monotonic_s": outcome.at, "input_bytes": outcome.input_bytes,
                    "admission_s": outcome.admission_s}}

    def _queue_outcome(self, stream, reason):
        index = ("admitted", "cancelled", "expired", "refused").index(reason)
        _previous, counts = self._queue_outcomes
        now = time.monotonic()
        status = stream.admission_status
        until = now if status.issued_at is None else status.issued_at
        outcome = AdmissionOutcome(status.ticket, reason, until - status.enqueued_at, now, status.input_bytes,
                                   None if status.issued_at is None else now - status.issued_at)
        self._queue_outcomes = (outcome, tuple(value + (offset == index) for offset, value in enumerate(counts)))

    def _eligibility(self, stream):
        check = stream.admission_eligibility
        try:
            reason = None if check is None else check()
            if reason not in (None, "client_input", "tier_unavailable", "restoring", "sequence_busy"):
                raise RuntimeError("request eligibility returned an unsupported lifecycle reason")
            return reason
        except RequestError:
            raise
        except BaseException as failure:
            self._admission_failure(failure)

    def _admission_failure(self, failure):
        try:
            poison = getattr(self.decoder, "poison_control", None)
            if poison is not None:
                poison(failure)
        except BaseException as containment:
            raise_failures(failure, (containment,))
        raise failure

    def _retire_waiting(self, entry, error=None):
        stream, box = entry[2]
        if stream.error is None and error is not None:
            stream.error = error
            if stream.admission_status.terminal_reason is None:
                stream.admission_status = replace(stream.admission_status, terminal_reason="refused")
        retire = getattr(self.decoder, "retire_request", None)
        if retire is not None and not retire(stream):
            self._queue_state(stream, "retiring")
            return False
        self.boxes[id(stream)] = box
        if not isinstance(entry, InputClaim) or not entry.logical:
            self.waiting.remove_request(entry)
        self._queue_outcome(stream, stream.admission_status.terminal_reason or "cancelled")
        stream.queue_s += time.perf_counter() - stream.queued
        if stream.error is not None:
            mark_error_state_retired(stream.error)
            if self.serving_profiles is not None:
                self._retire_admission_error(stream.error)
        self._reply(stream, *(("error", stream.error) if stream.error is not None else ("done", stream.stats())),
                    retired=self.serving_profiles is not None and stream.error is not None)
        return True

    def _admit_fifo(self, entries):
        from tensorfold.server.cancellation import RequestCancelled

        done, capacity_blocked = [], False
        for entry in entries:
            stream, box = entry[2]
            status = stream.admission_status
            if self._stopping and stream.error is None:
                stream.error = RequestCancelled("CUDA scheduler stopped before request admission")
                stream.cancel.set()
                stream.admission_status = replace(status, terminal_reason="cancelled")
            if stream.error is not None or stream.cancel.is_set():
                self._retire_waiting(entry)
                continue
            if not status.gpu_admitted and status.issued_at is None and status.deadline is not None and time.monotonic() >= status.deadline:
                stream.cancel.set()
                stream.admission_status = replace(status, terminal_reason="expired")
                self._retire_waiting(entry, CapacityError("request queue timeout elapsed before GPU admission; retry later"))
                continue
            try:
                self._check_profile(stream.profile_epoch)
                stage = getattr(self.decoder, "stage_request", None)
                if stage is not None:
                    reason = stage(stream, wake=self._media_wake, max_waiting_input_bytes=self.max_waiting_input_bytes,
                                   max_pending_requests=self._request_limit.maximum)
                    if reason is not None:
                        self._queue_state(stream, reason)
                        continue
                reason = self._eligibility(stream)
                retained_reason = getattr(self.decoder, "request_blocked_reason", None)
                if reason is None and retained_reason is not None:
                    reason = retained_reason(stream)
                if reason is None and getattr(stream, "retained_host", None) is not None:
                    need = self.decoder.output_need(stream)
                    if not box.available(need):
                        reason = "output"
                if reason is None:
                    prepare = getattr(self.decoder, "prepare_request", None)
                    if prepare is not None:
                        prepared = prepare(stream, wake=self._media_wake)
                        if type(prepared) is not bool:
                            self._admission_failure(RuntimeError("request preparation must report explicit nonblocking readiness"))
                        if not prepared:
                            reason = "media"
                if reason is not None:
                    self._queue_state(stream, reason)
                    continue
            except (ValueError, RequestError) as error:
                self._retire_waiting(entry, error)
                continue
            owns = getattr(self.decoder, "owns_admission", None)
            claimed = owns is not None and owns(stream)
            if capacity_blocked or not claimed and self._occupied() >= self._stream_limit():
                self._queue_state(stream, "gpu_capacity", input_ready=True)
                continue
            self.boxes[id(stream)] = box
            self._queue_state(stream, None, input_ready=True)
            if stream.admission_status.gpu_admitted:
                if stream.admission_status.readmission_issued_at is None:
                    stream.admission_status = replace(stream.admission_status, readmission_issued_at=time.monotonic())
            elif stream.admission_status.issued_at is None:
                stream.admission_status = replace(stream.admission_status, issued_at=time.monotonic())
            try:
                admitted = self.decoder.admit(stream)
                if admitted is False:
                    stream.admission_status = replace(stream.admission_status, gpu_admitted=True)
                    self._queue_state(stream, "restoring", input_ready=True)
                    self.boxes.pop(id(stream), None)
                    continue
            except NoRoom as error:
                catalog = getattr(self.decoder, "retention_catalog", None)
                if self._occupied() or catalog is not None and catalog.has_retained:
                    if self.serving_profiles is not None:
                        self._retire_admission_error(error)
                    capacity_blocked = True  # An eligible head never loses priority to a cheaper newer request.
                    stream.admission_status = replace(stream.admission_status,
                        readmission_issued_at=None) if stream.admission_status.gpu_admitted else replace(
                            stream.admission_status, issued_at=None)
                    self._queue_state(stream, "gpu_capacity", input_ready=True)
                    self.boxes.pop(id(stream), None)
                else:
                    self._retire_waiting(entry, error)
                continue
            except Exception as error:
                self._retire_waiting(entry, error)
                continue
            self.waiting.activate(entry)
            self._queue_state(stream, None, input_ready=True)
            self._queue_outcome(stream, "admitted")
            now = time.perf_counter()
            stream.queue_s += now - stream.queued
            resumed = getattr(stream, "_fifo_started", False)
            if not resumed:
                stream.admitted = now
            stream._fifo_started = True
            if not resumed and stream.on_fill is not None:
                stream.on_fill(stream.cached)
            if stream.done:
                done.append(stream)
        return done

    def _admit(self, first=None) -> list[Stream]:
        done = []
        while self._occupied() < self._stream_limit():
            if first is not None:
                item, first = first, None
            elif self.held is not None:
                item, self.held = self.held, None
            else:
                try:
                    item = self.waiting.get_nowait()
                except queue.Empty:
                    break
            if item is None:
                self._stopping = True
                break
            if self.waiting.is_wake(item):
                continue
            stream, box = item
            self.boxes[id(stream)] = box
            retire = getattr(self.decoder, "retire_request", None)
            if stream.error is not None or stream.cancel is not None and stream.cancel.is_set():
                if retire is not None and not retire(stream):
                    self.held, self._media_held_wait = (stream, box), True
                    self.boxes.pop(id(stream), None)
                    break
                self._media_held_wait = False
                self._reply(stream, *(('error', stream.error) if stream.error is not None else ('done', stream.stats())))
                continue
            try:
                # Profile replacement and admission share this serialized GPU
                # worker. Do not reacquire the producer publication lock: a
                # published request can be admitted before its producer returns.
                self._check_profile(stream.profile_epoch)
                prepare = getattr(self.decoder, "prepare_request", None)
                if prepare is not None and not prepare(stream, wake=self._media_wake):
                    self.held, self._media_held_wait = (stream, box), True
                    self.boxes.pop(id(stream), None)
                    break
                self._media_held_wait = False
                self.decoder.admit(stream)
                stream.admitted = time.perf_counter()             # one clock read an admission, not a round
                stream.queue_s += stream.admitted - stream.queued
                if stream.on_fill is not None:                    # its prefill starts from the rows it resumed
                    stream.on_fill(stream.cached)
            except NoRoom as exc:
                if self.serving_profiles is not None:
                    self._retire_admission_error(exc)
                if self.decoder.live():              # waits, first in line, until a live stream finishes
                    self.held = (stream, box)
                    self.boxes.pop(id(stream), None)
                    break
                if retire is not None and not retire(stream):
                    stream.error = exc
                    self.held, self._media_held_wait = (stream, box), True
                    self.boxes.pop(id(stream), None)
                    break
                self._terminal(box, "error", exc, retired=self.serving_profiles is not None)
                self.boxes.pop(id(stream), None)
                continue
            except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
                if self.serving_profiles is not None:
                    self._retire_admission_error(exc)
                if retire is not None and not retire(stream):
                    stream.error = exc
                    self.held, self._media_held_wait = (stream, box), True
                    self.boxes.pop(id(stream), None)
                    break
                box = self.boxes[id(stream)]
                self._terminal(box, "error", exc, retired=self.serving_profiles is not None)
                self.boxes.pop(id(stream), None)
                continue
            if stream.done:
                done.append(stream)
        return done

    def _occupied(self):
        # A cancelled filling request leaves decoder.live before its State can
        # be reused. Decoding requests stay in streams until finish, so count
        # only detached retirement owners in addition to the decoder's live set.
        claims = getattr(self.decoder, "admission_claims", None)
        return self.decoder.live() + (0 if claims is None else claims()) + sum(
            stream.st is not None and stream.sid not in getattr(self.decoder, "streams", {}) for stream in self._retiring)

    def _finish_ready(self, done, *, finish_empty=True):
        """Retain each request and mailbox until both CPU and rank consumers finish."""
        retire = getattr(self.decoder, "retire_request", None)
        ready, pending = [], []
        for stream in [*self._retiring, *done]:
            if retire is None or retire(stream):
                ready.append(stream)
            else:
                pending.append(stream)
        self._retiring = pending
        if ready or finish_empty:
            self.decoder.finish(ready)
        if self.serving_profiles is not None:
            for stream in ready:
                if stream.error is not None and not getattr(stream, "_media_outcome_retired", False):
                    self.serving_profiles.error_outcome(stream.error, state_retired=True,
                        states=() if stream.st is None else (stream.st,))
                stream.st = stream.snap = stream.copies = stream.vision = None
        for stream in ready:
            self._reply(stream, *(("error", stream.error) if stream.error is not None else ("done", stream.stats())),
                        retired=self.serving_profiles is not None and stream.error is not None)

    def _retire_admission_error(self, error):
        coordinated = getattr(self.decoder, "retire_admission_error", None)
        if coordinated is not None and getattr(self.decoder, "_operation_error", None) is error:
            coordinated()
        else:
            self.serving_profiles.error_outcome(error)

    def _check_profile(self, epoch):
        """Acceptance lock or serialized GPU worker; an epoch supplies no ownership.

        Draining does not invalidate admitted leases. Availability changes only
        after their queued/model work retires, and every replacement gets a new
        epoch even when artifact identities match an earlier profile.
        """
        from .profile_binding import ProfileUnavailable

        current = self._serving_profile_epoch
        if current is None:
            if epoch is not None:
                raise mark_error_state_retired(ProfileUnavailable("serving profile leases are not enabled for this scheduler"))
            return
        if not self._serving_profile_available or epoch != current:
            raise mark_error_state_retired(ProfileUnavailable(
                "the requested serving profile lease is unavailable or obsolete; retry preparation"))

    def _yield(self) -> None:
        """Lanes full, a foreground request waiting: the newest background stream (no images) re-queues."""

        if self.decoder.live() < self._stream_limit() or not self.waiting.foreground():
            return
        live = list(getattr(self.decoder, "streams", {}).values())
        stream = next((s for s in reversed(live) if s.background and not s.done
                       and s.vision is None and len(s.out) < s.count), None)
        if stream is None:
            return
        box = self.boxes[id(stream)]
        stream.finished = time.perf_counter()     # close this decode segment before continued() carries its timings
        self.decoder.finish([stream])
        self.waiting.put((stream.continued(), box))
        self.boxes.pop(id(stream))
        self.yields += 1

    def _cancelled(self) -> list[Stream]:
        """Prompts still filling whose clients left: a decoder that can drops them, keeping the state they reached."""

        drop = getattr(self.decoder, "cancel", None)
        candidates = list(getattr(self.decoder, "filling", ()))
        if self.admission_policy == "fifo":
            candidates += list(getattr(self.decoder, "streams", {}).values())
        gone = [s for s in candidates if not s.done and s.cancel is not None and s.cancel.is_set()]
        return drop(gone) if drop is not None and gone else []

    def _production_plan(self, media):
        """Reserve actual CPU output credit before any selected GPU window."""
        leases, runnable = [], False
        _revision, entries = self.waiting.requests(include_reserved=True)
        for entry in entries:
            stream, box = entry[2]
            if stream.admission_status.phase != "active" or stream.done:
                continue
            reason = self._eligibility(stream) or media.get(stream.sid)
            need = getattr(self.decoder, "output_need", None)
            width = (min(stream.count - len(stream.out), 1 + len(stream.drafts)) if need is None else need(stream))
            lease = None if reason is not None else box.reserve(width)
            if reason is None and lease is None:
                reason = "output"
            stream.blocked_input = reason
            self._queue_state(stream, reason)
            stream.production_budget = 0 if lease is None else width
            if lease is not None:
                leases.append(lease)
                runnable = True
        return leases, runnable

    def _suspend_blocked(self):
        _revision, queued = self.waiting.requests()
        if not any(entry[2][0].admission_status.blocked_reason == "gpu_capacity" for entry in queued):
            return
        suspend = getattr(self.decoder, "suspend_requests", None)
        if suspend is None:
            return
        _revision, all_entries = self.waiting.requests(include_reserved=True)
        blocked = [entry[2][0] for entry in all_entries if entry[2][0].admission_status.phase == "active"
                   and not entry[2][0].done and entry[2][0].blocked_input is not None]
        for stream in suspend(blocked):
            self.waiting.requeue(stream, reason=stream.blocked_input)
            self.boxes.pop(id(stream), None)

    def _reply(self, s: Stream, kind: str, value: Any, *, retired=False) -> None:
        box = self.boxes.get(id(s))                 # Journal until fallible outcome publication completes.
        if box is not None:
            self._terminal(box, kind, value, retired=retired)
            self.boxes.pop(id(s), None)

    def _clear_failure_frames(self) -> None:
        """Keep failure objects/context without retaining foreign decoder frames.

        Native BaseException descriptors bypass foreign overrides. Request
        callers may add their own traceback when receiving an error; the
        worker's and cleanup's frames are removed before publication and again
        at joined shutdown. Failed cleanup still retains its explicit decoder.
        """
        if self.serving_profiles is not None and self.serving_profiles.failure is not None:
            return  # Unknown profile owners stay intact until fenced retirement/process containment.
        pending = [self._failure, self._cleanup_failure, self._start_error]
        seen = set()
        while pending:
            error = pending.pop()
            if error is None or id(error) in seen:
                continue
            seen.add(id(error))
            BaseException.__traceback__.__set__(error, None)
            pending.append(BaseException.__cause__.__get__(error))
            pending.append(BaseException.__context__.__get__(error))

    def _loop(self) -> None:
        started = False
        try:
            with self._state_lock:
                started = self._started_ok
                if not started:
                    return                      # an ambiguously launched actor aborts without decoder access
            self._run()
        except BaseException as exc:
            prior = (BaseException.__cause__.__get__(exc), BaseException.__context__.__get__(exc),
                     BaseException.__suppress_context__.__get__(exc))
            errors = []
            # A worker failure must not strand accepted callers in box.get().
            # Closing excludes concurrent producers before queues are drained.
            with self._state_lock:
                self._closing = True
                self._failure = exc
                if self._request_limit is not None:
                    self._request_limit.close()
            if started and self.serving_profiles is not None:
                if self.serving_profiles.failure is None:
                    try:
                        self.serving_profiles._poison(exc)
                    except BaseException as containment:
                        errors.append(containment)
                self._cleanup_failure = exc  # Do not issue new model work or drop unknown raw owners.
            elif started:
                try:
                    self.decoder.drop()
                except BaseException as cleanup:
                    self._cleanup_failure = cleanup
                    errors.append(cleanup)
            boxes = list(self.boxes.values())
            if self._request_limit is not None:
                with self._state_lock:
                    boxes.extend(self._request_boxes)
            self.boxes.clear()
            if self.held is not None:
                _, box = self.held
                self.held = None
                boxes.append(box)
            while True:
                try:
                    item = self.waiting.get_nowait()
                except queue.Empty:
                    break
                if item is not None and not self.waiting.is_wake(item):
                    _, box = item
                    boxes.append(box)
            self._clear_failure_frames()
            # A put can enqueue before raising; interrupted background handoff
            # may therefore retain two ownership references to the same box.
            for box in dict.fromkeys(boxes):
                try:
                    if self._request_limit is None:
                        box.put(("error", exc))
                    elif self._cleanup_failure is None:
                        self._terminal(box, "error", exc)
                    else:
                        box.put(("error", exc))  # failed native cleanup keeps the admission owner retained
                except BaseException as publication:
                    errors.append(publication)
            BaseException.__cause__.__set__(exc, prior[0])
            BaseException.__context__.__set__(exc, prior[1])
            BaseException.__suppress_context__.__set__(exc, prior[2])
            if errors:
                try:
                    raise_failures(exc, errors)
                except BaseException as reported:
                    self._failure = reported     # every reply was attempted before diagnostic transport
        finally:
            retirement_primary = self._failure if self._request_limit is not None else None
            prior = (None if retirement_primary is None else (
                BaseException.__cause__.__get__(retirement_primary), BaseException.__context__.__get__(retirement_primary),
                BaseException.__suppress_context__.__get__(retirement_primary)))
            try:
                if self._request_limit is not None and self._cleanup_failure is None:
                    with self._state_lock:
                        remaining = tuple(self._request_boxes)
                    publications = []
                    for box in remaining:
                        try:
                            self._terminal(box, "error", RuntimeError("CUDA scheduler stopped before replying"))
                        except BaseException as publication:
                            publications.append(publication)
                    raise_failures(None, publications)
            except BaseException as cleanup:
                self._cleanup_failure = cleanup
                if retirement_primary is None:
                    self._failure = cleanup
                else:
                    BaseException.__cause__.__set__(retirement_primary, prior[0])
                    BaseException.__context__.__set__(retirement_primary, prior[1])
                    BaseException.__suppress_context__.__set__(retirement_primary, prior[2])
                    try:
                        raise_failures(retirement_primary, [cleanup])
                    except BaseException as reported:
                        self._failure = reported   # close publishes this owned asynchronous failure
            else:
                if retirement_primary is not None:
                    BaseException.__cause__.__set__(retirement_primary, prior[0])
                    BaseException.__context__.__set__(retirement_primary, prior[1])
                    BaseException.__suppress_context__.__set__(retirement_primary, prior[2])
            finally:
                self._worker_done.set()

    def _worker(self) -> None:
        with self._worker_lease:
            self._worker_entered.set()
            self._loop()

    def _stream_limit(self) -> int:
        """A lower ceiling blocks new admissions without evicting retained state."""
        controller = getattr(self.decoder, "controller", None)
        return self.max_streams if controller is None else min(self.max_streams, controller.admission_limit())

    def _execution_boundary(self) -> None:
        if self.serving_profiles is not None:
            self.serving_profiles.boundary(self)
            if not self._serving_profile_available:
                return
        controller = getattr(self.decoder, "controller", None)
        if controller is not None:
            controller.boundary()
        if self._pending_manifest is not None:
            with self._state_lock:
                request, self._pending_manifest = self._pending_manifest, None
            if request is None:
                return
            generation, name = request
            error = None
            try:
                self.decoder.install_execution_manifest(name)
            except ValueError as failed:
                # Expected invalid evidence leaves the previous registry and
                # every retained state intact. Protocol/device failures escape.
                error = str(failed)[:1024]
            with self._state_lock:
                status = dict(self._manifest_status)
                if error is None:
                    status["effective"] = name
                if self._manifest_status["request_epoch"] == generation:
                    status.update(pending=False, error=error)
                self._manifest_status = status

    def _run(self) -> None:
        if self.admission_policy == "fifo":
            return self._run_fifo()
        return self._run_legacy()

    def _round_requests(self, done, *, only_runnable=False):
        """One shared execution/error/retirement path for both admission policies."""
        ran, leases, primary = False, [], None
        try:
            done += self._cancelled()
            execute = True
            if only_runnable:
                readiness = getattr(self.decoder, "input_readiness", None)
                failed, media = ([], {}) if readiness is None else readiness()
                done += failed
                leases, production = self._production_plan(media)
                self._suspend_blocked()
                maintenance = getattr(self.decoder, "maintenance_pending", None)
                ready = maintenance is not None and maintenance()
                runnable = getattr(self.decoder, "media_runnable", None)
                active = self.decoder.live() > 0
                execute = self._serving_profile_available and (ready or production and active and (runnable is None or runnable()))
            if execute:
                done += self.decoder.round()
                ran = True
        except Exception as error:
            primary = error
            dropped = (self.decoder.drop(error) if only_runnable and getattr(self.decoder, "retention_catalog", None) is not None
                       else self.decoder.drop())
            if self.serving_profiles is not None:
                self.serving_profiles.error_outcome(error, state_retired=True,
                    states=tuple(stream.st for stream in dropped if stream.st is not None))
            for stream in dropped:
                if getattr(stream, "_media_request", None) is not None:
                    stream.error, stream.done = error, True
                    stream._media_outcome_retired = self.serving_profiles is not None
                    done.append(stream)
                else:
                    if self.serving_profiles is not None:
                        stream.st = stream.snap = stream.copies = stream.vision = None
                    self._reply(stream, "error", error, retired=self.serving_profiles is not None)
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                finish(tuple(lease.close for lease in leases))
            except BaseException as cleanup:
                if primary is not None:
                    raise_failures(primary, (cleanup,))
                raise
        self._finish_ready(done, finish_empty=not only_runnable)
        return ran

    def _run_fifo(self) -> None:
        """Oldest eligible ticket owns admission; deferred work remains in the same queue."""
        while True:
            if self._retiring:
                self._finish_ready([], finish_empty=False)
            with self._state_lock:
                self._stopping = self._stopping or self._stop_published
            # Observe admission closure before the final control boundary: a
            # control accepted before close must run before worker retirement.
            self._execution_boundary()
            revision, entries = self.waiting.requests()
            done = self._admit_fifo(entries)
            ran = self._round_requests(done, only_runnable=True) if self._serving_profile_available else False
            _current, waiting = self.waiting.requests()
            # A between-subruns claim has positively retired its backend and
            # retains only the logical caller's input credit. That caller may
            # itself be closing this scheduler; its credit must outlive the
            # worker, rather than make the worker wait for caller retirement.
            backend_claims = False
            if self._stopping:
                _all_revision, claims = self.waiting.requests(include_reserved=True)
                backend_claims = any(
                    entry[2][0].admission_status.phase != "between_subruns" for entry in claims)
            live = self.decoder.live() if self._serving_profile_available else 0
            boundary = getattr(self.decoder, "retention_boundary", None)
            drained, changed = True, False
            if boundary is not None and self._serving_profile_available:
                progress = boundary(idle=not ran,
                    needs_slot=any(stream.admission_status.blocked_reason == "gpu_capacity"
                                   for _priority, _ticket, (stream, _box) in waiting),
                    shutdown=self._stopping and not waiting and not live and not self._retiring and not backend_claims)
                drained, changed = progress["drained"], progress["changed"]
            if ran or changed:
                continue  # Runnable generation progresses without waiting for queue arrivals.
            if self._stopping and not waiting and not live and not self._retiring and not backend_claims:
                if drained:
                    return
            deadlines = [stream.admission_status.deadline for _priority, _ticket, (stream, _box) in waiting
                         if stream.error is None and not stream.cancel.is_set()
                         and not stream.admission_status.gpu_admitted
                         and stream.admission_status.deadline is not None and stream.admission_status.issued_at is None]
            expiry = getattr(self.decoder, "retention_deadline", None)
            if expiry is not None:
                at = expiry()
                if at is not None:
                    deadlines.append(at)
            control = self.waiting.get_control(since=revision, deadline=min(deadlines, default=None))
            if control is None:
                self._stopping = True

    def _run_legacy(self) -> None:
        while True:
            if self._retiring:
                self._finish_ready([])
            self._execution_boundary()
            if not self._serving_profile_available:
                # Keep the same worker/transport alive while a refused profile
                # awaits an explicit in-process retry. No model method is valid.
                item = self.waiting.get()
                if item is None:
                    return
                if not self.waiting.is_wake(item):
                    stream, box = item
                    from .profile_binding import ProfileUnavailable

                    self._terminal(box, "error", mark_error_state_retired(ProfileUnavailable("the serving profile is unavailable")))
                continue
            self._yield()
            idle = not self.decoder.live() and self.held is None
            if idle and self._stopping and not self._retiring:
                return
            maintenance = getattr(self.decoder, "maintenance_pending", None)
            ready = maintenance is not None and maintenance()
            runnable = getattr(self.decoder, "media_runnable", None)
            media_wait = runnable is not None and not runnable() and (
                self.decoder.live() or self._media_held_wait or self._retiring)
            if media_wait and self._occupied() >= self._stream_limit() and not ready:
                first = self.waiting.get_control()
            else:
                first = self.waiting.get() if (idle or media_wait) and not ready else None
            if idle and not self._retiring and not ready and first is None:
                return                                                                # close()
            if media_wait and not ready and first is None:
                self._stopping = True
            if idle:
                self._execution_boundary()  # a control may arrive while the worker waits on the queue
                if not self._serving_profile_available:
                    if first is not None and not self.waiting.is_wake(first):
                        self.waiting.put(first)
                    continue
            done = self._admit(first)
            self._round_requests(done)

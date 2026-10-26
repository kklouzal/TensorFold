"""Requests submit from any thread; one worker thread runs the rounds, and a slow client only fills its own queue."""
# Background requests go last; one decoding yields its lane to a waiting request and re-queues to replay later.

from __future__ import annotations

import itertools
import queue
import threading
import time
from typing import Any, Callable

from tensorfold.cleanup import finish, raise_failures, rollback
from tensorfold.server.request_limits import RequestLimit, optional_limit

from .memory_gate import NoRoom
from .output_mailbox import OutputMailbox
from .streams import Stream, mark_error_state_retired

TICK_S = 15.0           # a request waiting this long for tokens (queued or prefilling) ticks ``emit([])``: a keepalive


class Waiting(queue.PriorityQueue):
    """(stream, box) pairs in arrival order, background streams after every other."""

    def __init__(self) -> None:
        super().__init__()
        self._order = itertools.count()
        self._wake_marker = object()
        self._wake_pending = False
        self._wake_entry = self._stop_entry = None
        self._stop_published = self._stop_queued = False

    def put(self, item, block: bool = True, timeout: float | None = None) -> None:
        super().put((1 if item[0].background else 0, next(self._order), item), block, timeout)

    def get(self, block: bool = True, timeout: float | None = None):
        return super().get(block, timeout)[2]

    def get_control(self):
        """A full media cohort waits for completion without consuming queued requests."""
        with self.not_empty:
            while not self._wake_pending and not (self._stop_queued and not self.queue):
                self.not_empty.wait()
            if self._wake_pending:
                self._wake_pending = False
                return self._wake_marker
            self._stop_queued = False
            return None

    def _qsize(self):
        # Controls have fixed owner cells rather than heap/task entries. A
        # notification interruption cannot grow the request heap on retry.
        size = super()._qsize()
        return size + int(self._wake_pending) + int(self._stop_queued)

    def _get(self):
        if self._wake_pending and (not self.queue or self._wake_entry < self.queue[0]):
            item = self._wake_entry
            self._wake_pending = False
            return item
        if self.queue:
            return super()._get()
        if self._stop_queued:
            self._stop_published = True
            self._stop_queued = False
            return self._stop_entry
        raise RuntimeError("waiting queue has no owned request or control")

    def qsize(self) -> int:
        with self.mutex:
            return super()._qsize()

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
            return bool(self.queue) and self.queue[0][0] == 0


class Scheduler:
    """Construct, publish the owner, then call ``start`` before submitting.

    A failed start can have launched a native actor. The published scheduler
    retains it and its decoder until ``close`` acquires its retired work scope
    and joins;
    an unresolved start remains owned for retry or process containment.
    """

    def __init__(self, decoder: Any, *, max_streams: int = 4,
                 max_pending_requests: int | None = None) -> None:
        if type(max_streams) is not int or max_streams <= 0:
            raise ValueError("max_streams must be a positive integer")
        self.decoder = decoder
        self.max_streams = max_streams
        self.max_pending_requests = optional_limit(max_pending_requests, "max_pending_requests")
        self._request_limit = RequestLimit(max_pending_requests) if max_pending_requests is not None else None
        self._request_boxes: set[OutputMailbox] = set()
        self._error_retirements = {}
        self._submitted = False
        self.waiting = Waiting()
        self._media_wake = self.waiting.wake
        self.held: tuple | None = None               # a request waiting for memory, admitted before any other
        self._media_held_wait = False
        self._retiring: list[Stream] = []            # completed device work whose CPU consumers still retire
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
                self._request_limit.release(box)
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
               profile_epoch: int | None = None, _error_retired=None) -> dict:
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
        box = OutputMailbox(max(1, count))
        cancel = threading.Event()
        stream = Stream(list(prompt), max(1, count), sampling, draft=draft, stop_eos=stop_eos, vision=vision,
                        constraint=constraint, background=background, probabilities=probabilities, cancel=cancel,
                        penalties=penalties, token_limit=token_limit, profile_epoch=profile_epoch)
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
                    self.waiting.put((stream, box))
                except BaseException as error:
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
                    if not cancel.is_set() and emit(value):
                        self._request_cancel(cancel)
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
        if getattr(self.decoder, "prepare_request", None) is not None:
            self.waiting.wake()  # CPU readiness can leave the coordinator asleep on its existing queue.

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
        return self.decoder.live() + sum(stream.st is not None and stream.sid not in
            getattr(self.decoder, "streams", {}) for stream in self._retiring)

    def _finish_ready(self, done):
        """Retain each request and mailbox until both CPU and rank consumers finish."""
        retire = getattr(self.decoder, "retire_request", None)
        ready, pending = [], []
        for stream in [*self._retiring, *done]:
            if retire is None or retire(stream):
                ready.append(stream)
            else:
                pending.append(stream)
        self._retiring = pending
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
        gone = [s for s in getattr(self.decoder, "filling", ()) if s.cancel is not None and s.cancel.is_set()]
        return drop(gone) if drop is not None and gone else []

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
            try:
                done += self._cancelled()
                done += self.decoder.round()
            except Exception as exc:                 # noqa: BLE001  (the live requests fail)
                dropped = self.decoder.drop()
                if self.serving_profiles is not None:
                    self.serving_profiles.error_outcome(exc, state_retired=True,
                        states=tuple(s.st for s in dropped if s.st is not None))
                for s in dropped:
                    if getattr(s, "_media_request", None) is not None:
                        s.error, s.done = exc, True
                        s._media_outcome_retired = self.serving_profiles is not None
                        done.append(s)
                        continue
                    if self.serving_profiles is not None:
                        s.st = s.snap = s.copies = s.vision = None
                    self._reply(s, "error", exc, retired=self.serving_profiles is not None)
            self._finish_ready(done)

"""Requests submit from any thread; one worker thread runs the rounds, and a slow client only fills its own queue."""
# Background requests go last; one decoding yields its lane to a waiting request and re-queues to replay later.

from __future__ import annotations

import itertools
import queue
import threading
from typing import Any, Callable

from tensorfold.cleanup import finish, raise_failures, rollback
from tensorfold.server.request_limits import RequestLimit, optional_limit

from .memory_gate import NoRoom
from .streams import Stream


class Waiting(queue.PriorityQueue):
    """(stream, box) pairs in arrival order, background streams after every other."""

    def __init__(self) -> None:
        super().__init__()
        self._order = itertools.count()

    def put(self, item, block: bool = True, timeout: float | None = None) -> None:
        super().put((1 if item[0].background else 0, next(self._order), item), block, timeout)

    def get(self, block: bool = True, timeout: float | None = None):
        return super().get(block, timeout)[2]

    def stop(self) -> None:
        """Wake an idle worker to stop: None comes after every waiting request."""

        super().put((2, next(self._order), None))

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
        self._request_boxes: set[queue.Queue] = set()
        self._submitted = False
        self.waiting = Waiting()
        self.held: tuple | None = None               # a request waiting for memory, admitted before any other
        self.boxes: dict[int, queue.Queue] = {}
        self._state_lock = threading.Lock()
        self._closing = True                     # admission opens only after explicit startup commits
        self._stop_published = False
        self._stopping = False                    # worker consumed the marker; accepted work still drains
        self._failure: BaseException | None = None
        self._cleanup_failure: BaseException | None = None
        self._worker_done = threading.Event()
        self._worker_entered = threading.Event()
        self._worker_lease = threading.Lock()
        self._start_attempted = False
        self._started_ok = False
        self._start_error: BaseException | None = None
        self._arrival_applied = False
        self._arrival_hook = self.waiting.foreground
        try:
            self._arrival_original = decoder.arrived
        except AttributeError:
            self._has_arrival = False
        else:
            self._has_arrival = True
        self.yields = 0                              # background streams that gave up their lane
        self.thread = threading.Thread(target=self._worker, daemon=True)

    def configure_requests(self, limit: RequestLimit) -> None:
        """Share the App's logical owner before the endpoint publishes any work."""
        with self._state_lock:
            if self._submitted or self._worker_done.is_set():
                raise RuntimeError("request admission must be configured before the first submission")
            if self._request_limit is not None:
                raise RuntimeError("CUDA request admission was already configured")
            self.max_pending_requests = limit.maximum
            self._request_limit = limit

    @property
    def pending_requests(self) -> int | None:
        """Configured logical ownership count; uncapped mode has no such registry."""
        return self._request_limit.used if self._request_limit is not None else None

    def _terminal(self, box: queue.Queue, kind: str, value: Any) -> None:
        if self._request_limit is not None:
            with self._state_lock:
                self._request_limit.release(box)
            # Keep the reply owner journaled through a fallible publication.
            box.put((kind, value))
            with self._state_lock:
                self._request_boxes.discard(box)
        else:
            box.put((kind, value))

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
               constraint: Any = None, background: bool = False, probabilities: Any = None) -> dict:
        """Decode one request; ``emit`` runs on the calling thread and returns True to stop. Returns its stats."""

        box: queue.Queue = queue.Queue()
        stream = Stream(list(prompt), max(1, count), sampling, draft=draft, stop_eos=stop_eos, vision=vision,
                        constraint=constraint, background=background, probabilities=probabilities)
        cancel = threading.Event()
        stream.emit = lambda new: (box.put(("tokens", new)), cancel.is_set())[1]
        try:
            # Publication can succeed before an interrupt is delivered. Own
            # cancellation before entering the acceptance scope, so accepted
            # work never loses its stop signal when its caller cannot wait.
            with self._state_lock:
                if self._closing:
                    raise RuntimeError("CUDA scheduler is closing; new requests are rejected")
                if self._request_limit is not None:
                    try:
                        self._request_limit.borrow(box)
                        self._request_boxes.add(box)
                    except BaseException as primary:
                        self._request_limit.rollback_registration(
                            self, box, primary, lambda: self._request_boxes.discard(box), self._stop_requests)
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
                kind, value = box.get()
                if kind == "tokens":
                    if not cancel.is_set() and emit(value):
                        cancel.set()                # the client left: the stream ends after its next round
                elif kind == "error":
                    raise value
                else:
                    return value
        except BaseException:
            cancel.set()                            # interrupted callers also stop at the next owned round
            raise

    def _admit(self, first=None) -> list[Stream]:
        done = []
        while self.decoder.live() < self.max_streams:
            if first is not None:
                (stream, box), first = first, None
            elif self.held is not None:
                (stream, box), self.held = self.held, None
            else:
                try:
                    item = self.waiting.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    self._stopping = True
                    break
                stream, box = item
            self.boxes[id(stream)] = box
            try:
                self.decoder.admit(stream)
            except NoRoom as exc:
                self.boxes.pop(id(stream))
                if self.decoder.live():              # waits, first in line, until a live stream finishes
                    self.held = (stream, box)
                    break
                if self._request_limit is None:
                    box.put(("error", exc))
                else:
                    self._terminal(box, "error", exc)
                continue
            except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
                box = self.boxes.pop(id(stream))
                if self._request_limit is None:
                    box.put(("error", exc))
                else:
                    self._terminal(box, "error", exc)
                continue
            if stream.done:
                done.append(stream)
        return done

    def _yield(self) -> None:
        """Lanes full, a foreground request waiting: the newest background stream (no grammar or images) re-queues."""

        if self.decoder.live() < self.max_streams or not self.waiting.foreground():
            return
        live = list(getattr(self.decoder, "streams", {}).values())
        stream = next((s for s in reversed(live) if s.background and not s.done and s.constraint is None
                       and s.vision is None and len(s.out) < s.count), None)
        if stream is None:
            return
        box = self.boxes[id(stream)]
        self.decoder.finish([stream])
        self.waiting.put((stream.continued(), box))
        self.boxes.pop(id(stream))
        self.yields += 1

    def _reply(self, s: Stream, kind: str, value: Any) -> None:
        box = self.boxes.pop(id(s), None)            # None: the stream's request has had its reply
        if box is not None:
            if self._request_limit is None:
                box.put((kind, value))
            else:
                self._terminal(box, kind, value)

    def _clear_failure_frames(self) -> None:
        """Keep failure objects/context without retaining foreign decoder frames.

        Native BaseException descriptors bypass foreign overrides. Request
        callers may add their own traceback when receiving an error; the
        worker's and cleanup's frames are removed before publication and again
        at joined shutdown. Failed cleanup still retains its explicit decoder.
        """
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
            if started:
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
                if item is not None:
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

    def _run(self) -> None:
        while True:
            self._yield()
            idle = not self.decoder.live() and self.held is None
            if idle and self._stopping:
                return
            first = self.waiting.get() if idle else None                              # idle: wait for a request
            if idle and first is None:
                return                                                                # close()
            done = self._admit(first)
            try:
                done += self.decoder.round()
            except Exception as exc:                 # noqa: BLE001  (the live requests fail)
                for s in self.decoder.drop():
                    self._reply(s, "error", exc)
            self.decoder.finish(done)
            for s in done:
                self._reply(s, *(("error", s.error) if s.error is not None else ("done", s.stats())))

"""Requests submit from any thread; one worker thread runs the rounds, and a slow client only fills its own queue."""
# Background requests go last; one decoding yields its lane to a waiting request and re-queues to replay later.

from __future__ import annotations

import itertools
import queue
import threading
from typing import Any, Callable

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

    def __init__(self, decoder: Any, *, max_streams: int = 4) -> None:
        if type(max_streams) is not int or max_streams <= 0:
            raise ValueError("max_streams must be a positive integer")
        self.decoder = decoder
        self.max_streams = max_streams
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
                self.waiting.put((stream, box))
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
                box.put(("error", exc))
                continue
            except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
                self.boxes.pop(id(stream)).put(("error", exc))
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
            box.put((kind, value))

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
            # A worker failure must not strand accepted callers in box.get().
            # Closing excludes concurrent producers before queues are drained.
            with self._state_lock:
                self._closing = True
                self._failure = exc
            if started:
                try:
                    self.decoder.drop()
                except BaseException as cleanup:
                    self._cleanup_failure = cleanup
                    if cleanup is not exc:
                        BaseException.add_note(exc, "scheduler decoder cleanup also failed; retained as _cleanup_failure")
            boxes = list(self.boxes.values())
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
                box.put(("error", exc))
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

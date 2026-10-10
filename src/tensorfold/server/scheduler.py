"""The engine thread: requests queued as jobs, admitted into the engine's lanes, stepped round by round."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import itertools
import math
from pathlib import Path
import queue
import threading
import time
import traceback
from typing import Any, Callable

from tensorfold.cleanup import finish, raise_failures, rollback
from tensorfold.engine.lane_engine import LaneStream
from tensorfold.server.cancellation import Cancellation, RequestCancelled
from tensorfold.server.checkpoints import CheckpointStore
from tensorfold.server.errors import CapacityError, RequestError, RoundError
from tensorfold.server.request_limits import RequestLimit, optional_limit
from tensorfold.server.live import ChunkRate, Meter
from tensorfold.server.prompt_fill import Filling, PromptFill
from tensorfold.server.stream_gate import StreamGate
from tensorfold.thread_work import ThreadWork


@dataclass
class ChatJob:
    job_id: str
    prompt_ids: list[int]
    max_tokens: int
    temperature: float
    history_len: int = 0
    # Snapshot system-and-tools blocks for reuse across sessions; history boundaries already include session-specific text.
    shared_prefix_lens: tuple[int, ...] = ()
    # the request's proposer (suffix copies, a drafter model, tool-call structure); None: the engine's default
    proposer: Any = None
    # False ("draft": false): one token a round and no drafts, the serial reference output is checked against
    drafts: bool = True
    # exact_sampling.Sampling for this request (None = greedy)
    sampling: Any = None
    # thinking budget (0: none) and the tokens that close a think block ("\n</think>\n\n"; LaneStream)
    think_budget: int = 0
    think_close: tuple[int, ...] = ()
    think_end: int = -1
    # Background jobs yield to foreground arrivals, and callers restart preempted jobs from the beginning.
    background: bool = False
    preempted: bool = False
    submitted_at: float = field(default_factory=time.perf_counter)
    started_at: float = 0.0
    prefilled_at: float = 0.0
    finished_at: float = 0.0
    cached_tokens: int = 0
    stream: LaneStream | None = None
    error: BaseException | None = None
    chunks: "queue.Queue[list[int] | None]" = field(default_factory=queue.Queue)
    done: threading.Event = field(default_factory=threading.Event)
    cancellation: Cancellation = field(default_factory=Cancellation)
    ignore_eos: bool = False
    stop_check: Callable[[list[int]], bool] | None = None
    call_gate: Any = None  # tool_choice "required": the answer opens a tool call (LaneStream)
    constraint: Any = None  # response_format's grammar (engine.grammar.Constraint), or None
    vision: Any = None
    # a decision: prefill ends at these labels' last-row logits and joins no round (empty: a chat)
    label_ids: tuple[int, ...] = ()
    scored: tuple[list[float], float] | None = None


class _JobQueue(queue.PriorityQueue):
    """Jobs in arrival order, background jobs after every other job."""

    def __init__(self) -> None:
        super().__init__()
        self._order = itertools.count()

    def put(self, job: ChatJob, block: bool = True, timeout: float | None = None) -> None:
        super().put((1 if job.background else 0, next(self._order), job), block, timeout)

    def get(self, block: bool = True, timeout: float | None = None) -> ChatJob:
        return super().get(block, timeout)[2]

    def foreground_waiting(self) -> bool:
        return self.peek_foreground() is not None

    def peek_foreground(self) -> ChatJob | None:
        with self.mutex:
            return self.queue[0][2] if self.queue and self.queue[0][0] == 0 else None

    def remove(self, cancellation: Cancellation) -> list[ChatJob]:
        with self.mutex:
            removed = [entry[2] for entry in self.queue if entry[2].cancellation is cancellation]
            self.queue[:] = [entry for entry in self.queue if entry[2].cancellation is not cancellation]
            heapq.heapify(self.queue)
            return removed


class Scheduler(PromptFill):
    """Owns the engine on one thread: admits jobs, steps rounds, delivers tokens."""

    def __init__(
        self,
        engine: Any,
        *,
        lanes: int,
        eos_ids: frozenset[int],
        checkpoints: CheckpointStore | None = None,
        proposer_factory: Callable[[], Any] | None = None,
        idle_wait: float = 0.02,
        snapshot_dir: Any = None,
        session_dir: Any = None,
        model_id: str = "",
        admission: Any = None,
        prompt_memory: Any = None,
        decode_share: float = 0.25,
        snapshot_registry: Any = None,
        snapshot_codec: Any = None,
        max_pending_requests: int | None = None,
        max_engine_calls: int | None = None,
        request_limit: RequestLimit | None = None,
    ) -> None:
        self.max_pending_requests = optional_limit(max_pending_requests, "max_pending_requests")
        self.max_engine_calls = optional_limit(max_engine_calls, "max_engine_calls")
        if request_limit is not None and request_limit.maximum != self.max_pending_requests:
            raise ValueError("scheduler and App request limits must match")
        self._request_limit = request_limit or (
            RequestLimit(max_pending_requests) if max_pending_requests is not None else None)
        self._finalizing = False
        if lanes < 1:
            raise ValueError("lanes must be positive")
        if decode_share < 0:
            raise ValueError("decode_share must be 0 or more")
        # ``engine.memory.Admission``: a job starts beside live streams only while the projected memory fits
        self.admission = admission
        self.snapshot_dir = snapshot_dir
        self.session_dir = session_dir
        self.model_id = model_id
        self.snapshot_registry, self.snapshot_codec = snapshot_registry, snapshot_codec
        self.disk_blocks: Any = None
        self.session_blocks: Any = None
        if model_id and snapshot_registry is not None and (snapshot_dir is not None or session_dir is not None):
            from tensorfold.engine.prefix_snapshots import DiskBlocks

            if snapshot_dir is not None:
                self.disk_blocks = DiskBlocks(Path(snapshot_dir), model_id, registry=snapshot_registry)
            if session_dir is not None:
                self.session_blocks = DiskBlocks(Path(session_dir), model_id, registry=snapshot_registry)
        self.engine = engine
        self.lanes = int(lanes)
        self.eos_ids = eos_ids
        self.checkpoints = checkpoints
        self.prompt_memory = prompt_memory
        self.proposer_factory = proposer_factory
        self.idle_wait = float(idle_wait)
        self.slow_round_ms = 1000.0
        self._held: ChatJob | None = None
        self._queue = _JobQueue()
        self._engine_calls: queue.Queue[tuple[Callable[[Any], Any], queue.Queue[Any]]] = queue.Queue()
        # This registry owns jobs across fallible queued/held/filling/active handoffs.
        self._state_lock = threading.RLock()
        self._accepted: dict[int, ChatJob] = {}
        self._engine_waiters: dict[int, queue.Queue[Any]] = {}
        self._failure: BaseException | None = None
        self._cleanup_failure: BaseException | None = None
        self._shutdown_owners: tuple[Any, ...] = ()
        self.preemptions = 0
        self._jobs: dict[str, ChatJob] = {}
        self._stop = threading.Event()
        self._work = ThreadWork(self._run)
        self._thread = threading.Thread(target=self._work.run, name="tensorfold-engine", daemon=True)
        self.rounds = 0
        self.failed_rounds = 0
        self.completed = 0
        self.cancelled = 0
        self.starts = 0
        self._starting: ChatJob | None = None
        self._fills: list[Filling] = []  # open prompts, oldest first, each filling a chunk a step
        self._owned_fills: dict[int, Filling] = {}  # retains fallible open/end/preemption handoffs
        # while prompts fill, rounds get decode_share of each chunk's time, at most fill_rounds between two chunks
        self.decode_share = float(decode_share)
        self.fill_rounds = 32
        self.fill_guard = 8  # a prompt passed over this many chunks takes the next one (no starvation)
        self._credit, self._rounds_left = 0.0, 0
        self._released_at = 0  # ``starts`` when MLX's freed buffers were last handed back
        # Evaluate and save cache arrays on the scheduler thread that owns their streams during shutdown.
        self.on_stop: Callable[[], Any] | None = None
        # streams take memory as they grow: the newest waits, or ends, when the next round's growth can't fit
        self.gate = (
            StreamGate(
                prompt_memory,
                admission.memory.per_token,
                admission.memory.round_bytes,
                admission.budget,
                lanes=self.lanes,
            )
            if admission is not None and prompt_memory is not None and self.lanes > 1
            else None
        )
        self.stall_s = 120.0  # no round, start or finish while requests wait: dump stacks
        self.stall_prefill_s = 900.0  # the same while one prefill runs
        self._watch_work = ThreadWork(self._watch)
        self._watchdog = threading.Thread(target=self._watch_work.run, name="tensorfold-watchdog", daemon=True)
        self.decoded, self.prefilled = Meter(), ChunkRate()  # the live line's decode and prefill tok/s

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        with self._state_lock:
            if self._stop.is_set():
                raise RuntimeError("the scheduler is closed") from self._failure
            if self._work.cancelled_before_entry or self._watch_work.cancelled_before_entry:
                raise RuntimeError("scheduler startup failed; close its retained owner")
            if not self._work.launched:
                self._work.launch(self._thread)
            if not self._watch_work.launched:
                self._watch_work.launch(self._watchdog)

    def _watch(self) -> None:
        """Dump every thread stack once when waiting requests stall, including prefill chunks; SIGUSR1 also dumps stacks on demand."""

        import faulthandler
        import sys

        mark: tuple[int, int, int, int, bool] | None = None
        since = time.perf_counter()
        dumped = False
        while not self._stop.wait(1.0):
            waiting = self._queue.qsize() + (self._held is not None) + len(self._jobs) + (self._starting is not None)
            starting = self._starting is not None
            now_mark = (self.rounds, self.starts, self.completed, getattr(self.engine, "prefill_chunks", 0), starting)
            now = time.perf_counter()
            if now_mark != mark or not waiting:
                mark, since, dumped = now_mark, now, False
                continue
            limit = self.stall_prefill_s if starting else self.stall_s
            if now - since > limit and not dumped:
                dumped = True
                print(
                    f"[tensorfold] stalled {now - since:.0f}s: queued={self._queue.qsize()} "
                    f"held={self._held is not None} jobs={len(self._jobs)} active={self.engine.active_count} "
                    f"starting={starting}; every thread's stack follows",
                    flush=True,
                )
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

    def stop(self, timeout: float = 5.0) -> None:
        """Close acceptance, observe callback retirement, and settle native joins.

        Callback completion is independent of Thread.join/is_alive: an
        interrupted CPython join can report a still-executing target stopped.
        """
        if threading.current_thread() in (self._thread, self._watchdog):
            raise RuntimeError("a scheduler worker cannot join itself")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("scheduler stop timeout must be a finite nonnegative number")
        with self._state_lock:
            self._stop.set()
            if self._work.cancelled_before_entry:
                # Failed startup removed the callback while holding its entry
                # gate. Any delayed original bootstrap owns only an empty
                # controller; finalization gets a separately published owner.
                self._work = ThreadWork(self._run)
                self._thread = threading.Thread(target=self._work.run, name="tensorfold-engine", daemon=True)
            if not self._work.launched:
                self._work.launch(self._thread)
        deadline = time.monotonic() + timeout
        failures = []
        for work, thread, cancel in ((self._work, self._thread, False), (self._watch_work, self._watchdog, True)):
            try:
                work.drain(thread, timeout=max(0.0, deadline - time.monotonic()), cancel_unentered=cancel)
            except BaseException as error:
                failures.append(error)
        if failures:
            if len(failures) > 1:
                raise failures[0] from BaseExceptionGroup("scheduler worker drain failures", failures[1:])
            raise failures[0]
        if self._cleanup_failure is None and self._failure is not None:
            self._clear_failure_frames(self._failure)
        if self._failure is not None:
            raise RuntimeError("the scheduler stopped after an engine worker failure") from self._failure
        if self._cleanup_failure is not None:
            raise RuntimeError("the scheduler could not release its owned state") from self._cleanup_failure

    def submit(self, job: ChatJob) -> None:
        with self._state_lock:
            if self._stop.is_set():
                raise RuntimeError("the scheduler is closed") from self._failure
            if job.done.is_set() or id(job) in self._accepted:
                raise ValueError("a job may be submitted only once")
            if self._request_limit is not None:
                try:
                    self._request_limit.borrow(id(job))
                    self._accepted[id(job)] = job
                except BaseException as primary:
                    self._request_limit.rollback_registration(
                        self, id(job), primary, lambda: self._accepted.pop(id(job), None), self._stop.set)
            else:
                self._accepted[id(job)] = job
            try:
                self._queue.put(job)
            except BaseException as error:
                if self._request_limit is not None:
                    # A queue may enqueue before raising; only worker shutdown
                    # and its drain can safely retire this acceptance journal.
                    self._failure = error
                    rollback(self, error, lambda: finish((self._request_limit.close, self._stop.set)))
                raise

    @property
    def pending_requests(self) -> int | None:
        """Configured logical callers, including canceled but unretired work."""
        return self._request_limit.used if self._request_limit is not None else None

    @property
    def engine_calls(self) -> int:
        """Queued and running RPCs, including callbacks whose callers timed out."""
        with self._state_lock:
            return len(self._engine_waiters)

    def on_engine(self, fn: Callable[[Any], Any], timeout: float = 600.0) -> Any:
        """Run ``fn`` on the engine thread once no stream is live and no prompt is open."""

        done: queue.Queue[Any] = queue.Queue(1)
        with self._state_lock:
            if self._stop.is_set():
                raise RuntimeError("the scheduler is closed") from self._failure
            if self.max_engine_calls is not None and len(self._engine_waiters) >= self.max_engine_calls:
                raise CapacityError(f"engine call capacity reached ({self.max_engine_calls} unfinished calls); retry later")
            try:
                self._engine_waiters[id(done)] = done
            except BaseException:
                self._engine_waiters.pop(id(done), None)
                raise
            try:
                self._engine_calls.put((fn, done))
            except BaseException as error:
                if self.max_engine_calls is not None:
                    self._failure = error
                    operations = (self._stop.set,) if self._request_limit is None else (
                        self._request_limit.close, self._stop.set)
                    rollback(self, error, lambda: finish(operations))
                raise
        try:
            result = done.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("the engine did not score the prompt in time") from exc
        if isinstance(result, BaseException):
            raise result
        return result

    def _run_engine_call(self) -> bool:
        try:
            fn, done = self._engine_calls.get_nowait()
        except queue.Empty:
            return False
        try:
            result = fn(self.engine)
        except Exception as exc:  # noqa: BLE001 - the waiter raises this on its own thread
            self._clear_failure_frames(exc)
            self._complete_call(done, exc)
        else:
            self._complete_call(done, result)
        return True

    def _complete_call(self, done: queue.Queue[Any], result: Any) -> None:
        with self._state_lock:
            if id(done) in self._engine_waiters:
                try:
                    done.put_nowait(result)
                except queue.Full:
                    if not self._stop.is_set():
                        raise
                    # A prior publication enqueued before failing. The owned
                    # one-item box already contains this callback's terminal
                    # reply; shutdown must not overwrite or duplicate it.
                self._engine_waiters.pop(id(done), None)

    def _fail_engine_calls(self, error: BaseException) -> list[BaseException]:
        errors = []
        with self._state_lock:
            for done in tuple(self._engine_waiters.values()):
                try:
                    self._complete_call(done, error)
                except BaseException as publication:
                    errors.append(publication)
        while True:
            try:
                self._engine_calls.get_nowait()
            except queue.Empty:
                return errors

    def cancel(self, cancellation: Cancellation) -> None:
        cancellation.cancel()
        for job in self._queue.remove(cancellation):
            self._finish_cancelled(job)

    def _finish_cancelled(self, job: ChatJob) -> None:
        with self._state_lock:
            if job.done.is_set():
                return
            job.error = RequestCancelled("request cancelled")
            job.proposer = None
            self.cancelled += 1
            self._finish(job)

    def _discard_job(self, job: ChatJob) -> None:
        self._jobs.pop(job.job_id, None)
        if job.stream is not None:
            self.engine.discard_stream(job.stream)
            job.stream.history_checkpoints = []
            job.stream.proposer = None
        self._finish_cancelled(job)

    def _cancel_active(self) -> None:
        if self._held is not None and self._held.cancellation.cancelled:
            self._finish_cancelled(self._held)
            self._held = None
        for filling in [f for f in self._fills if f.job.cancellation.cancelled]:
            self._fill(filling, abort=RequestCancelled("request cancelled"))  # stops between chunks, progress kept
        for job in list(self._jobs.values()):
            if job.cancellation.cancelled:
                self._discard_job(job)

    @property
    def active(self) -> int:
        return len(self._jobs)

    @property
    def waiting(self) -> int:
        """Requests not started yet: queued, or held until they fit."""

        return self._queue.qsize() + (self._held is not None)

    @property
    def filling(self) -> list[Any]:
        """The jobs whose prompts are prefilling."""

        return [f.job for f in self._fills]

    @staticmethod
    def finish_job(job: ChatJob, reason: str = "stop") -> None:
        """Ask the engine to stop a stream at the next round (safe from any thread)."""

        stream = job.stream
        if stream is not None and not stream.finished:
            stream.finish_reason = reason
            stream.finished = True
            stream.finished_at = time.perf_counter()

    # -- the loop -------------------------------------------------------------
    def _run(self) -> None:
        failure = None
        try:
            self._loop()
        except BaseException as exc:
            failure = exc
        finally:
            self._finalize(failure if failure is not None else self._failure)

    @staticmethod
    def _clear_failure_frames(error: BaseException) -> None:
        """Retire worker frames through native descriptors, bypassing foreign hooks."""
        seen = set()
        pending = [error]
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            for name in ("__cause__", "__context__"):
                linked = BaseException.__dict__[name].__get__(current)
                if linked is not None:
                    pending.append(linked)
            BaseException.__dict__["__traceback__"].__set__(current, None)

    @staticmethod
    def _terminal_error(error: BaseException | None) -> BaseException:
        if error is None:
            return RequestCancelled("server stopping")
        args = BaseException.__dict__["args"].__get__(error)
        safe = all(type(arg) in (str, int, float, bool, type(None)) for arg in args)
        message = BaseException.__str__(error) if safe else "engine worker failed"
        return RoundError(type.__getattribute__(type(error), "__name__"), message)

    def _finalize(self, failure: BaseException | None) -> None:
        """Terminate accepted work; release array owners only after an explicit drain."""
        with self._state_lock:
            self._stop.set()
            self._finalizing = True
            if self._request_limit is not None:
                self._request_limit.close()
            self._failure = failure
            jobs = tuple(self._accepted.values())
            fills = tuple(self._owned_fills.values())
            self._shutdown_owners = (jobs, fills)
            for job in jobs:
                if not job.done.is_set():
                    job.cancellation.cancel()
                    job.error = self._terminal_error(failure)
                    if failure is None:
                        self.cancelled += 1
                    self._finish(job)
            rpc_errors = self._fail_engine_calls(self._terminal_error(failure))
        try:
            # CPU callbacks have ended on this worker. GPU work must also end
            # before generators, captured snapshots, and model rows are freed.
            self.engine.drain()
            if failure is None:
                for filling in list(self._fills):
                    self._fill(filling, abort=RequestCancelled("server stopping"))
            for filling in fills:
                if filling.steps is not None:
                    filling.steps.close()
                if id(filling) in self._owned_fills and filling.held is not None:
                    self.prompt_memory.end(filling.held)
            # Prefill generator cancellation may enqueue copies of valid
            # partial-prefix caches. Complete that cleanup work before release.
            self.engine.drain()
            self.engine.reset()
            self.engine.prefill_guard = None
            for job in jobs:
                job.proposer = None
                if job.stream is not None:
                    job.stream.finished = True
                    job.stream.finish_reason = "error" if failure is not None else "cancelled"
                    job.stream.history_checkpoints = []
                    job.stream.proposer = None
            self._jobs.clear()
            self._fills.clear()
            self._owned_fills.clear()
            self._held = self._starting = None
            while not self._queue.empty():
                self._queue.get_nowait()
            if failure is None and self.on_stop is not None:
                self.on_stop()
            raise_failures(None, rpc_errors)
        except BaseException as cleanup:
            self._cleanup_failure = cleanup
            if failure is not None:
                try:
                    BaseException.add_note(failure, "scheduler cleanup failed; owned state remains retained")
                except BaseException as annotation:
                    self._cleanup_failure = BaseExceptionGroup(
                        "scheduler cleanup and annotation failed", [cleanup, annotation]
                    )
        else:
            if self._request_limit is not None:
                for job in jobs:
                    self._request_limit.release(id(job))
            self._shutdown_owners = ()
            if failure is not None:
                self._clear_failure_frames(failure)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._cancel_active()
            self._preempt_background()
            self._admit()
            self._starting = self._fills[0].job if self._fills else None
            self._retire_externally_finished()
            if self._chunk_due():
                self._fill()
                continue
            if self.engine.active_count == 0:
                if self._held is None and not self._fills:
                    if self._run_engine_call():
                        continue
                    self._release_idle()
                    try:
                        self._held = self._queue.get(timeout=self.idle_wait)
                    except queue.Empty:
                        continue
                continue  # _admit starts it
            if not self._fills:
                self._credit, self._rounds_left = 0.0, 0  # no prompt waits: rounds owe nothing
            if self.gate is not None:
                self._gate_round()
            started = self.clock()
            try:
                landed = self.engine.step()
            except Exception as exc:  # noqa: BLE001 - one bad round must not kill the server
                self.failed_rounds += 1
                traceback.print_exception(exc)
                error_type, message = type(exc).__name__, str(exc)
                exc.__traceback__ = exc.__cause__ = exc.__context__ = None
                for job in list(self._jobs.values()):
                    job.error = RoundError(error_type, message)
                    if job.stream is not None:
                        self.engine.discard_stream(job.stream)
                        job.stream.finish_reason = "error"
                        job.stream.history_checkpoints = []
                        job.stream.proposer = None
                    job.proposer = None
                    self._finish(job)
                self._jobs.clear()
                self.engine.reset()
                job = landed = tokens = None
                continue
            self.rounds += 1
            self._spend_round(self.clock() - started)
            self.decoded.add(sum(len(t) for t in landed.values()))
            self._cancel_active()
            if self.engine.round_stats:
                last = self.engine.round_stats[-1]
                if last.total_ms > self.slow_round_ms:
                    print(
                        f"[tensorfold] slow round {last.total_ms:.0f} ms streams={last.streams} "
                        f"width={last.width} rows={last.rows} forward={last.forward_ms:.0f}",
                        flush=True,
                    )
                if len(self.engine.round_stats) > 4096:
                    del self.engine.round_stats[:2048]
            for stream_id, tokens in landed.items():
                job = self._jobs.get(stream_id)
                if job is None:
                    continue
                if tokens:
                    job.chunks.put(list(tokens))
                if job.stream is not None and job.stream.finished:
                    del self._jobs[stream_id]
                    self._retire(job)
            job = landed = tokens = None

    def _release_idle(self) -> None:
        """Once no stream is left and nothing waits, hand MLX's freed buffers back: no round can want them now."""

        if self.prompt_memory is not None and self._released_at != self.starts and self._queue.empty():
            self._released_at = self.starts
            getattr(self.engine, "release_rounds", lambda: None)()
            self.prompt_memory.release_freed()

    def _preempt_background(self) -> None:
        """Release background work until the next foreground request has both a lane and enough memory."""

        waiting = self._held
        if waiting is None or waiting.background:
            waiting = self._queue.peek_foreground()
        if waiting is None or waiting.cancellation.cancelled:
            return
        for filling in [f for f in reversed(self._fills) if f.job.background]:  # the newest first
            if self._room() and self._fits(waiting):
                return
            self._preempt_filling(filling)
        for job in self._jobs.values():
            if self._room() and self._fits(waiting):
                break
            if job.background and not job.preempted and job.stream is not None and not job.stream.finished:
                job.preempted = True
                self.preemptions += 1
                self.engine.discard_stream(job.stream)
                job.stream.finish_reason = "preempted"
                job.stream.history_checkpoints = []

    def _retire_externally_finished(self) -> None:
        for stream_id, job in list(self._jobs.items()):
            if job.stream is not None and job.stream.finished:
                del self._jobs[stream_id]
                self._retire(job)

    def _room(self) -> bool:
        """Whether a lane is free: live streams and open prompts each hold one."""

        return self.engine.active_count + len(self._fills) < self.lanes

    def _admit(self) -> None:
        """Admit fitting jobs while lanes are free; each opens a prompt that fills beside the others, a chunk a step."""

        while self._room():
            job = self._held
            self._held = None
            if job is not None and job.background and self._queue.foreground_waiting():
                self._queue.put(job)  # a request that arrived since goes first
                job = None
            if job is None:
                try:
                    job = self._queue.get_nowait()
                except queue.Empty:
                    return
            if not self._fits(job):
                self._held = job  # waits until a live stream finishes and frees its memory
                return
            if job.cancellation.cancelled:
                self._finish_cancelled(job)
                continue
            if self.decode_share <= 0:
                self._start_job(job)  # decode_share 0: each prompt whole before any round (0.3.6.2)
                continue
            self._open_job(job)
            if self._chunk_due():
                self._fill()  # no round is owed: a chunk now, as before

    def _fits(self, job: ChatJob) -> bool:
        """Alone, prompt admission checks memory; beside streams or open prompts, both projections must fit."""

        if self.engine.active_count == 0 and not self._fills:
            return True
        reply = self._reserved(int(job.max_tokens))
        memory = self.prompt_memory
        if memory is not None and not memory.would_fit(len(job.prompt_ids), reply):
            return False
        if self.admission is None:
            return True
        live = [
            (n, min(len(j.prompt_ids) + int(j.max_tokens), n + self._reserved(int(j.max_tokens))))
            for j in self._jobs.values()
            if j.stream is not None and not j.stream.finished
            for n in [j.stream.context_len]
        ]
        # an open prompt grows from the rows it holds to its prompt and reply horizon; its chunks' workspace counts too
        live += [
            (len(f.job.prompt_ids) - f.left, len(f.job.prompt_ids) + self._reserved(int(f.job.max_tokens)))
            for f in self._fills
        ]
        widest = max([len(job.prompt_ids), *(len(f.job.prompt_ids) for f in self._fills)])
        return self.admission.admits(widest, len(job.prompt_ids) + reply, live)

    def _reserved(self, reply: int) -> int:
        """The reply tokens a stream holds memory for ahead of its length: the gate's horizon when it guards rounds."""

        return reply if self.gate is None else min(reply, self.gate.horizon)

    def _gate_round(self) -> None:
        """Hold back the newest streams whose growth can't fit, or end the newest when even the oldest can't grow."""

        live = sorted(
            (j for j in self._jobs.values() if j.stream is not None and not j.stream.finished),
            key=lambda j: j.started_at,
        )
        plan = self.gate.plan(
            [(j.stream.stream_id, j.stream.context_len, len(j.prompt_ids) + int(j.max_tokens)) for j in live]
        )
        if set(plan.paused) != self.engine.paused:
            print(
                f"[tensorfold] memory: {len(plan.paused)} of {len(live)} streams wait for room (newest first)",
                flush=True,
            )
        self.engine.paused = set(plan.paused)
        for job in (j for j in live if j.stream.stream_id in plan.ended):
            job.error = RequestError(
                f"This server ran out of memory with {len(live)} streams decoding, so the newest (this request, "
                f"after {len(job.stream.emitted)} tokens) was stopped for the older ones to finish. Retry it, "
                "shorten the prompt or max_tokens, or start the server with a smaller --parallel."
            )
            print(f"[tensorfold] memory: ended {job.job_id}, the newest of {len(live)} streams", flush=True)
            self.engine.discard_stream(job.stream)
            job.stream.finish_reason = "error"
            job.stream.history_checkpoints = []
            del self._jobs[job.stream.stream_id]
            self._finish(job)
            if self.prompt_memory is not None:
                self.prompt_memory.release_freed()  # its arrays are free now: the next plan sees the room

    def _read_disk_block(self, prompt: list[int], usable: Any = None) -> None:
        """Put the longest stored prefix ``usable`` accepts (system blocks, saved conversations) in the store."""

        if self.checkpoints is None or (self.disk_blocks is None and self.session_blocks is None):
            return
        try:
            have = self.checkpoints.longest(prompt, usable)
            found: Any = None
            for blocks, pinned in ((self.disk_blocks, True), (self.session_blocks, False)):
                hit = None if blocks is None else blocks.best(prompt, have if found is None else len(found[1]), usable)
                if hit is not None:
                    found = (hit[0], hit[1], blocks, pinned)
            if found is None:
                return
            from tensorfold.engine.prefix_snapshots import load_snapshot

            started = time.perf_counter()
            if self.prompt_memory is not None and not self.prompt_memory.allow_load(found[0].stat().st_size):
                print(
                    f"[tensorfold] left the stored prefix of {len(found[1])} tokens on disk: memory cannot "
                    "hold a copy beside what runs; this prompt re-prefills it",
                    flush=True,
                )
                return
            loaded = load_snapshot(
                found[0],
                self.model_id,
                registry=self.snapshot_registry,
                codec=self.snapshot_codec,
                expected_tokens=found[1],
            )
            if loaded is None:
                return
            tokens, cache = loaded
            self.checkpoints.insert(tokens, cache, last_prompt=tokens, pinned=found[3])
            found[2].touch(tokens)
            print(
                f"[tensorfold] read {'system-block' if found[3] else 'conversation'} snapshot tokens={len(tokens)} "
                f"from disk in {time.perf_counter() - started:.2f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - a bad file costs a prefill, never the request
            from tensorfold.engine.snapshot_file import reject_snapshot_cleanup_failure

            reject_snapshot_cleanup_failure(exc)
            print(f"[tensorfold] snapshot read failed: {type(exc).__name__}: {exc}", flush=True)

    def _keep_checkpoints(self, job: ChatJob, shared_at: set[int]) -> None:
        """Store the prefixes the job's prefill kept (system blocks pinned and saved to disk), once."""

        stream = job.stream
        if stream is None or job.vision is not None:
            return
        kept, stream.history_checkpoints = stream.history_checkpoints, []
        for tokens, snapshot in kept if self.checkpoints is not None else ():
            shared = len(tokens) in shared_at
            self.checkpoints.insert(tokens, snapshot, last_prompt=job.prompt_ids, pinned=shared)
            if self.snapshot_dir is not None and shared:
                self._persist(tokens, snapshot)

    def _persist(self, tokens: list[int], cache: list[Any]) -> None:
        """Write a system-block snapshot to disk once, so a restart does not lose it."""

        from tensorfold.engine.prefix_snapshots import save_snapshot

        if self.snapshot_registry is None:
            return

        try:
            started = time.perf_counter()
            path = save_snapshot(
                self.snapshot_dir,
                self.model_id,
                tokens,
                cache,
                registry=self.snapshot_registry,
                codec=self.snapshot_codec,
            )
            if path is not None:
                print(
                    f"[tensorfold] saved system-block snapshot tokens={len(tokens)} "
                    f"in {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - a full disk must not fail the request
            from tensorfold.engine.snapshot_file import reject_snapshot_cleanup_failure

            reject_snapshot_cleanup_failure(exc)
            print(f"[tensorfold] snapshot save failed: {type(exc).__name__}: {exc}", flush=True)

    def _retire(self, job: ChatJob) -> None:
        if job.cancellation.cancelled:
            self._discard_job(job)
            return
        stream = job.stream
        if stream is not None and getattr(stream, "error", None) is not None and job.error is None:
            job.error = stream.error  # its grammar failed: this request alone answers with the error
        retained = self.engine.finished_caches.pop(job.job_id, None)
        if retained is not None and self.checkpoints is not None and job.vision is None:
            tokens, cache = retained
            if len(tokens) > len(job.prompt_ids):
                self.checkpoints.insert(tokens, cache, last_prompt=job.prompt_ids)
        if stream is not None and stream in self.engine.streams:
            self.engine.streams.remove(stream)
        self.completed += 1
        self._finish(job)

    def _finish(self, job: ChatJob) -> None:
        with self._state_lock:
            if job.done.is_set():
                return
            job.finished_at = time.perf_counter()
            if self._request_limit is not None and not self._finalizing:
                self._request_limit.release(id(job))
            job.chunks.put(None)
            job.done.set()
            self._accepted.pop(id(job), None)

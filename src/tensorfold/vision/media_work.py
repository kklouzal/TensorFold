"""Bounded CPU unit lookahead; the GPU coordinator never waits for decoding.

One model-owned executor is borrowed, never closed here. Its fresh CPU process
and pixel/replay ownership contracts remain authoritative. Public operations
are serialized coordinator calls; workers only invoke ``pixel_unit``. A taken
pixel lease belongs to its caller, including through asynchronous GPU staging.
"""

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import os
import threading
import time

from tensorfold.cleanup import raise_failures, rollback
from tensorfold.thread_work import ThreadWork

from .media_frontend import MediaExecutor, PreparedReplayMedia
from .replay import MediaBusy, MediaCancelled


def _identity(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(eq=False)
class _UnitWork:
    prepared: object
    unit_index: int
    generation: int
    profile_identity: str
    media_identity: str
    plan: object
    probes: object
    replay: object
    probe: object
    pixel_index: int
    cancelled: threading.Event = field(default_factory=threading.Event)
    encoded: object = None
    pixels: object = None
    work: object = None


class MediaWork:
    """At most one queued/running/ready CPU unit per prepared request.

    Worker/lease/paired-borrow limits bound all outstanding jobs and CPU
    orchestrators; the executor additionally bounds fresh processes and bytes.
    Readiness never joins a worker. Cancel returns False while an entered
    callback still owns source data; the caller may continue other GPU work.
    Shutdown waits have an explicit timeout and uncertain retirement keeps all
    affected owners latched. No normalized video or completed-unit archive is
    retained by this coordinator.
    """

    def __init__(self, executor: MediaExecutor):
        self._initialize_owner()
        self._initialize(executor)

    @classmethod
    def create(cls, executor, *, register):
        """Publish an empty close-safe owner before any native thread launch."""
        if not callable(register):
            raise TypeError("CPU media work construction requires its owning journal callback")
        owner = cls.__new__(cls)
        owner._initialize_owner()
        register(owner)
        owner._initialize(executor)
        return owner

    def _initialize_owner(self):
        self.executor = self.envelope = None
        self.pending_limit = self.worker_limit = 0
        self._drain_timeout = 0.
        self._pid = os.getpid()
        self._control = threading.RLock()
        self._active = False
        self._condition = threading.Condition()
        self._pending = deque()
        self._jobs = {}
        self._workers = []
        self._closing = self._closed = False
        self._failure = None
        self._initialized = False
        self._wake = None
        self._wake_due = False
        self._capacity_callback = self._capacity_changed

    def _initialize(self, executor):
        if self._closing or self._closed or self._workers:
            raise RuntimeError("CPU media work initialization is one-shot")
        if type(executor) is not MediaExecutor:
            raise TypeError("CPU media work requires the admitted actual MediaExecutor")
        executor.check_health()
        if executor.envelope.max_replay_borrows < 2:
            raise ValueError("CPU media lookahead requires encoded and pixel replay borrows together")
        limit = executor.envelope.max_worker_wall_seconds + executor.envelope.retirement_grace_seconds
        if not math.isfinite(limit) or limit > threading.TIMEOUT_MAX:
            raise ValueError("CPU media retirement wait budget must be representable within threading.TIMEOUT_MAX")
        with self._condition:
            self.executor = executor
            self.envelope = executor.envelope
            self._drain_timeout = limit
            self.pending_limit = self.worker_limit = min(self.envelope.max_workers, self.envelope.max_replay_leases,
                                                         self.envelope.max_replay_borrows // 2)
        try:
            self.executor.set_capacity_wake(self._capacity_callback)
            for index in range(self.worker_limit):
                work = ThreadWork(self._serve)
                thread = threading.Thread(target=work.run, name=f"media-unit-{index}", daemon=True)
                with self._condition:
                    self._workers.append((work, thread))  # Retain before the fallible native launch.
                work.launch(thread)
            with self._condition:
                self._initialized = True
        except BaseException as primary:
            self._set_failure(primary)
            try:
                self.close()
            except BaseException as cleanup:
                raise_failures(primary, (cleanup,))
            raise

    @property
    def closed(self):
        return self._closed

    def check_health(self):
        """Thread-safe read-only health, never waiting behind coordinator work.

        HTTP preparation may inspect this concurrently with the GPU actor.
        Only short lifecycle metadata is copied under the condition mutex;
        neither decoder work nor worker draining holds that mutex.
        """
        if self._pid != os.getpid():
            raise RuntimeError("CPU media work belongs to another process")
        with self._condition:
            closing, closed, failure, initialized = self._closing, self._closed, self._failure, self._initialized
            executor, workers = self.executor, tuple(self._workers)
        if closing or closed or failure is not None:
            raise RuntimeError("CPU media work is closed or poisoned; retain its owners") from failure
        if not initialized or executor is None:
            raise RuntimeError("CPU media work construction has not completed")
        executor.check_health()
        for work, _thread in workers:
            if work.retired.is_set():
                error = work.failure if work.failure is not None else RuntimeError("CPU media worker stopped before shutdown")
                raise RuntimeError("CPU media worker stopped; retain pending owners") from error

    def _set_failure(self, error):
        with self._condition:
            if self._failure is None:
                self._failure = error

    def record(self):
        """Bounded runtime receipt; counts describe this coordinator's claims."""
        with self._operation():
            return {"workers": self.worker_limit, "pending_limit": self.pending_limit,
                    "owned_units": len(self._jobs), "one_ahead_per_request": True,
                    "closing": self._closing, "closed": self._closed}

    def set_wake(self, callback):
        """Bind one scheduler's stable, thread-safe non-GPU wake callback.

        Owned CPU workers dispatch a coalesced control outside locks. Close
        retires those workers before dropping this callback's lifetime.
        """
        if not callable(callback):
            raise TypeError("CPU media scheduler wake must be callable")
        with self._operation():
            self.check_health()
            with self._condition:
                if self._wake is callback:
                    return  # Stable admission rebind is not a capacity/completion event.
                if self._wake is not None:
                    raise RuntimeError("CPU media work already belongs to another scheduler wake owner")
                self._wake = callback
                self._wake_due = True
                self._condition.notify()

    def _capacity_changed(self):
        with self._condition:
            if not self._closing:
                self._wake_due = True
                self._condition.notify()

    @contextmanager
    def _operation(self):
        if self._pid != os.getpid():
            raise RuntimeError("CPU media work belongs to another process")
        if not self._control.acquire(blocking=False):
            raise RuntimeError("CPU media coordinator operations must be serialized")
        try:
            if self._active:
                raise RuntimeError("CPU media coordinator operations cannot reenter")
            self._active = True
            try:
                yield
            finally:
                self._active = False
        finally:
            self._control.release()

    def _context(self, prepared, unit_index, generation, profile_identity, media_identity):
        if type(prepared) is not PreparedReplayMedia or prepared.executor is not self.executor:
            raise TypeError("CPU media work requires this executor's actual prepared replay")
        if (type(generation) is not int or not 0 <= generation < 1 << 63
                or not _identity(profile_identity) or not _identity(media_identity)):
            raise ValueError("CPU media work needs current generation/profile/media identities")
        plan = prepared.plan
        if (plan.profile_identity != profile_identity or plan.media_identity != media_identity
                or type(unit_index) is not int or not 0 <= unit_index < len(plan.units)):
            raise ValueError("CPU media unit differs from the current admitted plan or identities")
        return generation, profile_identity, media_identity

    def _current(self, job, prepared, context):
        if ((job.generation, job.profile_identity, job.media_identity) != context
                or job.prepared is not prepared or prepared.plan is not job.plan
                or prepared.probes is not job.probes or prepared.replay is not job.replay):
            raise ValueError("CPU media completion belongs to a stale request generation or plan")

    def _serve(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closing or self._pending or self._wake_due and self._wake is not None)
                    if self._closing:
                        return
                    if self._wake_due and self._wake is not None:
                        wake, job = self._wake, None
                        self._wake_due = False
                    else:
                        wake, job = None, self._pending.popleft()
                if wake is not None:
                    wake()  # No media/store/executor lock is held during scheduler code.
                else:
                    job.work.run()
                    self._capacity_changed()  # Completion, including error/cancel, is a control boundary.
        except BaseException as primary:
            if self._failure is None:
                self._set_failure(primary)
            from tensorfold.cuda.cohort import notify_failure

            try:
                notify_failure()  # Existing process-owner channel; never a GPU operation.
            except BaseException as notification:
                raise_failures(primary, (notification,))
            raise

    def _decode(self, job):
        # No tensor, model, GPU context or decoder alternative is created here.
        job.pixels = self.executor.pixel_unit(job.probe, job.pixel_index, cancelled=job.cancelled.is_set)

    def _schedule(self, job):
        job.work = ThreadWork(lambda: self._decode(job))
        with self._condition:
            self._pending.append(job)
            self._condition.notify()

    @staticmethod
    def _clean_outcome(error):
        return error.__cause__ is None and (error.__context__ is None or error.__suppress_context__)

    def _parked_busy(self, job):
        return (not job.cancelled.is_set() and job.work.retired.is_set()
                and type(job.work.failure) is MediaBusy and self._clean_outcome(job.work.failure))

    def _retry_busy(self, job):
        if not self._parked_busy(job):
            return False
        # Only the executor can prove failed work released all process/FD and
        # staging claims. Polling schedules one new attempt, never a busy loop.
        if not self.executor.unit_available(job.probe, replay_borrows=1):
            return False
        previous = job.work
        try:
            self._schedule(job)
        except BaseException as error:
            job.cancelled.set()
            job.work.cancel()
            self._set_failure(error)
            raise
        previous.failure = None
        return True

    def request(self, prepared, unit_index, *, generation, profile_identity, media_identity):
        """Schedule one unit, or return False without waiting when capacity is busy."""
        with self._operation():
            self.check_health()
            context = self._context(prepared, unit_index, generation, profile_identity, media_identity)
            old = self._jobs.get(prepared)
            if old is not None:
                self._current(old, prepared, context)
                if old.unit_index == unit_index:
                    self._retry_busy(old)
                    return not old.cancelled.is_set()
                # RAM resume can need the current unit after its GPU features
                # were evicted, while the one CPU lookahead owns a later unit.
                # Stop that obsolete lookahead without waiting for decoding.
                # Only positive callback/lease retirement permits replacement.
                if not self._cancel_job(old):
                    return False
            if len(self._jobs) >= self.pending_limit:
                return False
            unit = prepared.plan.units[unit_index]
            if (type(unit.probe_index) is not int or not 0 <= unit.probe_index < len(prepared.probes)
                    or type(unit.run.temporal_group) is not int):
                raise ValueError("CPU media unit lost its actual probe/temporal-group binding")
            probe = prepared.probes[unit.probe_index]
            if not self.executor.unit_available(probe, replay_borrows=2):
                return False
            job = _UnitWork(prepared, unit_index, *context, prepared.plan, prepared.probes,
                            prepared.replay, probe, unit.run.temporal_group)
            self._jobs[prepared] = job  # Publish an owner before borrowing or launching work.
            try:
                job.encoded = prepared.borrow_replay()
                if probe.ref not in job.encoded.refs:
                    raise ValueError("CPU media unit is not owned by its encoded replay borrow")
                self._schedule(job)
            except BaseException as primary:
                busy_unentered = (type(primary) is MediaBusy and self._clean_outcome(primary)
                                  and job.work is None and job.encoded is None)
                job.cancelled.set()
                if job.work is not None:
                    job.work.cancel()
                if job.work is None or job.work.retired.is_set():
                    try:
                        self._retire(job, transfer=False)
                    except BaseException as cleanup:
                        self._set_failure(primary)
                        raise_failures(primary, (cleanup,))
                else:
                    self._set_failure(primary)  # An accepted callback still owns the encoded source.
                if busy_unentered:
                    return False
                raise
            return True

    def _find(self, prepared, unit_index, generation, profile_identity, media_identity):
        context = self._context(prepared, unit_index, generation, profile_identity, media_identity)
        job = self._jobs.get(prepared)
        if job is not None:
            self._current(job, prepared, context)
        return job if job is not None and job.unit_index == unit_index else None

    def ready(self, prepared, unit_index, *, generation, profile_identity, media_identity):
        """Nonblocking readiness includes completed errors, which take_ready surfaces."""
        with self._operation():
            return self._completed_job(prepared, unit_index, generation, profile_identity, media_identity) is not None

    def _completed_job(self, prepared, unit_index, generation, profile_identity, media_identity):
        self.check_health()
        job = self._find(prepared, unit_index, generation, profile_identity, media_identity)
        if job is None or not job.work.retired.is_set():
            return None
        if self._parked_busy(job):
            self._retry_busy(job)
            return None
        return job

    def check_ready(self, prepared, unit_index, *, generation, profile_identity, media_identity):
        """Pre-GPU admission gate: retire a completed error before raising it.

        A successful unit remains Work-owned until take_ready transfers it;
        pending/busy work returns False without waiting or changing ownership.
        """
        with self._operation():
            job = self._completed_job(prepared, unit_index, generation, profile_identity, media_identity)
            if job is None:
                return False
            error = self._completion_error(job)
            if error is not None:
                self._raise_completed_error(job, error)
            return True

    def _completion_error(self, job):
        error = job.work.failure
        if job.cancelled.is_set() and (error is None or type(error) is MediaBusy and self._clean_outcome(error)):
            return MediaCancelled("CPU media unit was cancelled before publication")
        return error

    def _raise_completed_error(self, job, error):
        def retire():
            try:
                self._retire(job, transfer=False)
            except BaseException:
                self._set_failure(error)
                raise

        rollback(job, error, retire)  # Existing unretired-owner transport, without a second retirement attempt.

    def _retire(self, job, *, transfer):
        # A returned CPU callback is insufficient if its process/FD owner is
        # poisoned. The executor's existing containment latch is authoritative.
        self.executor.check_health()
        if job.work is not None and not job.work.retired.is_set():
            raise RuntimeError("CPU media callback still owns its replay")
        pixels = job.pixels
        if pixels is not None and not transfer:
            pixels.close()
            job.pixels = None
        if job.encoded is not None:
            job.encoded.close()
            job.encoded = None
        with self._condition:
            try:
                self._pending.remove(job)
            except ValueError:
                pass  # Unpublished or dequeued; ThreadWork retirement proves callback quiescence.
        if self._jobs.get(job.prepared) is job:
            self._jobs.pop(job.prepared)
        if job.work is not None:
            # Callers captured the outcome before retirement. An idle pool
            # thread may retain this empty controller, never its old traceback.
            job.work.failure = None
        job.pixels = None
        job.prepared = job.plan = job.probes = job.replay = job.probe = None
        return pixels

    def take_ready(self, prepared, unit_index, *, generation, profile_identity, media_identity):
        """Transfer only a completed pixel lease; return None while work is pending."""
        with self._operation():
            job = self._completed_job(prepared, unit_index, generation, profile_identity, media_identity)
            if job is None:
                return None
            error = self._completion_error(job)
            if error is not None:
                self._raise_completed_error(job, error)
            try:
                return self._retire(job, transfer=True)
            except BaseException as cleanup:
                self._set_failure(cleanup)
                raise

    @staticmethod
    def _ordinary_cancel(error):
        return type(error) in (MediaBusy, MediaCancelled) and MediaWork._clean_outcome(error)

    def cancel(self, prepared):
        """Request stop; True proves this coordinator no longer owns request work."""
        with self._operation():
            job = self._jobs.get(prepared)
            if job is None:
                return True
            return self._cancel_job(job)

    def _cancel_job(self, job):
        """Serialized owner; no callback wait or prepared-request retirement."""
        job.cancelled.set()
        if job.work is not None:
            job.work.cancel()
        if job.work is not None and not job.work.retired.is_set():
            return False
        error = None if job.work is None else job.work.failure
        try:
            self._retire(job, transfer=False)
        except BaseException as cleanup:
            self._set_failure(cleanup if error is None else error)
            raise_failures(error, (cleanup,))
        if error is not None and not self._ordinary_cancel(error):
            raise error
        return True

    def close(self, *, timeout=None):
        """Stop admission and prove callbacks retired before releasing their claims.

        Timeout bounds waits, using the admitted worker wall/grace by default.
        File/mapping retirement follows the existing executor's failure contract.
        Taken pixel leases and prepared replay owners remain caller-owned.
        """
        with self._operation():
            if self._closed:
                return
            limit = self._drain_timeout if timeout is None else timeout
            if (type(limit) not in (int, float) or not math.isfinite(limit)
                    or limit < 0 or limit > threading.TIMEOUT_MAX):
                raise ValueError("CPU media close timeout must be finite, nonnegative and within threading.TIMEOUT_MAX")
            deadline = time.monotonic() + limit
            with self._condition:
                self._closing = True
            for job in self._jobs.values():
                job.cancelled.set()
                if job.work is not None:
                    job.work.cancel()
            with self._condition:
                self._condition.notify_all()
            errors = []
            for work, thread in self._workers:
                try:
                    work.drain(thread, timeout=max(0., deadline - time.monotonic()), cancel_unentered=False)
                except BaseException as error:
                    errors.append(error)
            if errors:
                if self._failure is None:
                    self._set_failure(errors[0])
                if any(not work.retired.is_set() for work, _thread in self._workers):
                    raise_failures(self._failure, errors)
            for job in tuple(self._jobs.values()):
                error = None if job.work is None else job.work.failure
                try:
                    self._retire(job, transfer=False)
                except BaseException as cleanup:
                    errors.extend(value for value in (error, cleanup) if value is not None)
                else:
                    if error is not None and not self._ordinary_cancel(error):
                        errors.append(error)
            if self._jobs:
                if self._failure is None:
                    self._set_failure(errors[0])
                raise_failures(self._failure, errors)
            if self.executor is not None:
                self.executor.clear_capacity_wake(self._capacity_callback)
            self._wake = self._capacity_callback = None
            with self._condition:
                self._closed = True
            if self._failure is not None:
                raise_failures(self._failure, errors)
            raise_failures(None, errors)

"""Optional admission caps. A request stays owned until caller and engine work retire."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
import inspect
import threading
from typing import Any, Callable, Iterator
import weakref

from tensorfold.cleanup import finish, raise_failures, rollback
from tensorfold.server.errors import CapacityError


def optional_limit(value: int | None, name: str) -> int | None:
    """None preserves uncapped admission; configured counts are strictly positive integers."""
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(eq=False)
class _Request:
    # A native lock's scoped release also survives interruption of Python
    # retirement bookkeeping. Admission can reclaim the journal afterward.
    scope: Any = field(default_factory=threading.Lock)
    workers: set[Any] = field(default_factory=set)


class RequestLimit:
    """Count logical requests, including work retained after caller cancellation.

    App entry creates a fresh owner, including nested calls. Its sequential
    engine jobs borrow that owner, so replay and gated continuations cannot be
    refused after admission. Direct scheduler submissions create independent
    owners. One owner may have only one outstanding engine job. All publication
    and retirement use this owner's lock; no interpreter-lock assumption is
    required. The scheduler retires a borrow only after engine work ends.
    """

    def __init__(self, maximum: int) -> None:
        self.maximum = optional_limit(maximum, "max_pending_requests")
        if self.maximum is None:
            raise ValueError("max_pending_requests must be a positive integer")
        self._lock = threading.RLock()
        self._owners: set[_Request] = set()
        self._callers: dict[int, list[_Request]] = {}
        self._workers: dict[Any, _Request] = {}
        # Independent thread-safe fail-closed signal also works if registration
        # rollback cannot reacquire the admission lock.
        self._closed = threading.Event()

    @property
    def used(self) -> int:
        with self._lock:
            self._prune()
            return len(self._owners)

    def _prune(self) -> None:
        for worker, owner in tuple(self._workers.items()):
            if worker not in owner.workers:
                self._workers.pop(worker, None)
        for owner in tuple(self._owners):
            self._retire(owner)
        for identity, stack in tuple(self._callers.items()):
            live = [owner for owner in stack if owner.scope.locked()]
            if live:
                self._callers[identity] = live
            else:
                del self._callers[identity]

    def _check(self) -> None:
        if self._closed.is_set():
            raise RuntimeError("request admission is closed")
        if len(self._workers) >= self.maximum:
            self._prune()
        if len(self._owners) >= self.maximum:
            self._prune()
            if len(self._owners) >= self.maximum:
                raise CapacityError(f"request capacity reached ({self.maximum} unfinished requests); retry later")

    def close(self) -> None:
        """Reject future callers while preserving every outstanding owner."""
        with self._lock:
            self._closed.set()

    def _retire(self, owner: _Request) -> None:
        if not owner.scope.locked() and not owner.workers:
            self._owners.discard(owner)

    @contextmanager
    def request(self) -> Iterator[None]:
        identity = threading.get_ident()
        owner = _Request()
        primary = None
        prior = None
        try:
            with owner.scope:
                with self._lock:
                    self._check()
                    self._owners.add(owner)
                    self._callers.setdefault(identity, []).append(owner)
                yield
        except BaseException as error:
            primary = error
            prior = (BaseException.__cause__.__get__(error), BaseException.__context__.__get__(error),
                     BaseException.__suppress_context__.__get__(error))
            raise
        finally:
            try:
                with self._lock:
                    stack = self._callers.get(identity)
                    if stack and owner in stack:
                        if stack[-1] is not owner:
                            raise RuntimeError("request admission scopes must retire in nesting order")
                        stack.pop()
                    if not stack:
                        self._callers.pop(identity, None)
                    self._retire(owner)
            except BaseException as cleanup:
                if primary is not None:
                    BaseException.__cause__.__set__(primary, prior[0])
                    BaseException.__context__.__set__(primary, prior[1])
                    BaseException.__suppress_context__.__set__(primary, prior[2])
                    raise_failures(primary, [cleanup])
                raise
            else:
                if primary is not None:
                    BaseException.__cause__.__set__(primary, prior[0])
                    BaseException.__context__.__set__(primary, prior[1])
                    BaseException.__suppress_context__.__set__(primary, prior[2])

    def wrap(self, call: Callable) -> Callable:
        """Selected only for configured Apps; uncapped calls keep their original method."""
        function, owner = call.__func__, weakref.ref(call.__self__)
        signature = inspect.signature(call)

        @wraps(function)
        def limited(*args, **kwargs):
            instance = owner()
            if instance is None:
                raise RuntimeError("request App has retired")
            with self.request():
                # This local strong owner spans every active call. The stored
                # function/metadata hold no bound App, so idle App retirement
                # does not depend on cyclic garbage collection.
                return function(instance, *args, **kwargs)
        limited.__signature__ = signature
        return limited

    def borrow(self, worker: Any) -> None:
        """Journal a queued engine job before publication; caller cancellation does not release it."""
        owner = None
        try:
            with self._lock:
                if self._closed.is_set():
                    raise RuntimeError("request admission is closed")
                if worker in self._workers:
                    raise ValueError("an engine job may borrow request admission only once")
                stack = self._callers.get(threading.get_ident())
                if stack and not stack[-1].scope.locked():
                    self._prune()
                    stack = self._callers.get(threading.get_ident())
                selected = stack[-1] if stack else _Request()
                if selected.workers:
                    raise RuntimeError("one logical request cannot publish concurrent engine jobs")
                if not stack:
                    self._check()
                owner = selected
                owner.workers.add(worker)
                self._workers[worker] = owner
                self._owners.add(owner)
        except BaseException as primary:
            if owner is None:
                raise
            prior = (BaseException.__cause__.__get__(primary), BaseException.__context__.__get__(primary),
                     BaseException.__suppress_context__.__get__(primary))
            try:
                # No queue has been published yet: registration is reversible.
                with self._lock:
                    self._workers.pop(worker, None)
                    owner.workers.discard(worker)
                    self._retire(owner)
            except BaseException as cleanup:
                self._closed.set()
                BaseException.__cause__.__set__(primary, prior[0])
                BaseException.__context__.__set__(primary, prior[1])
                BaseException.__suppress_context__.__set__(primary, prior[2])
                raise_failures(primary, [cleanup])
            BaseException.__cause__.__set__(primary, prior[0])
            BaseException.__context__.__set__(primary, prior[1])
            BaseException.__suppress_context__.__set__(primary, prior[2])
            raise

    def release(self, worker: Any) -> None:
        """Retire a job after all engine uses, before publishing its terminal reply."""
        with self._lock:
            owner = self._workers.get(worker)
            if owner is not None:
                owner.workers.discard(worker)
                self._retire(owner)
                self._workers.pop(worker, None)

    def rollback_registration(self, scheduler: Any, worker: Any, primary: BaseException,
                              unregister: Callable[[], Any], stop: Callable[[], Any]) -> None:
        """Undo a journal before queue publication; failed rollback closes admission.

        The scheduler holds its publication lock. Retire the journal before
        freeing its borrow. If either step fails, preserve that ownership and
        attempt both admission closure and worker wakeup. Native failure roots
        and the scheduler owner remain attached to the original failure.
        """
        def undo():
            try:
                unregister()
                self.release(worker)
            except BaseException as failure:
                rollback(scheduler, failure, lambda: finish((self.close, stop)))
        rollback(scheduler, primary, undo)


__all__ = ["RequestLimit", "optional_limit"]

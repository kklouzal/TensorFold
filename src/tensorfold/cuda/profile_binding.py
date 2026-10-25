"""Model-profile request leases spanning preparation, queues and publication.

Only the opt-in Flash Next structural-profile owner installs these scopes.
Readers admitted before a drain may finish; later readers cannot indefinitely
extend it. Native scope locks preserve retirement evidence across interrupted
Python bookkeeping. This object owns no CUDA allocation or serving worker.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import wraps
import inspect
import threading
from typing import Any
import weakref

from tensorfold.cleanup import finish, raise_failures
from tensorfold.server.errors import CapacityError


class ProfileUnavailable(CapacityError):
    code = "model_profile_unavailable"


def retry_control(operation, failed):
    """Retire an idempotent control despite one allocation/cancellation fault.

    A post-publication retry is safe only for the declared pruning/coalesced
    wake actions. Failure is still reported; a second failure invokes the
    owning terminal policy instead of leaving an idle pending control asleep.
    """
    try:
        return operation()
    except (KeyboardInterrupt, MemoryError) as primary:
        try:
            operation()
        except BaseException as secondary:
            failed(primary, secondary)
        raise
    except BaseException as primary:
        failed(primary)
        raise


@dataclass(eq=False)
class _Reader:
    epoch: int
    scopes: set[Any] = field(default_factory=set)  # weak holder refs; the journal cannot pin abandoned native scopes


def _release_hold(scope_ref, close_lock_ref):
    scope, close_lock = scope_ref(), close_lock_ref()
    # During finalization the holder is unreachable, so a collected close lock
    # has no concurrent explicit caller. The native scope may already be gone.
    with close_lock if close_lock is not None else nullcontext():
        if scope is not None and scope.locked():
            scope.release()


def _release_with_retry(gate, scope_ref, close_lock_ref):
    try:
        _release_hold(scope_ref, close_lock_ref)
    except (KeyboardInterrupt, MemoryError) as primary:
        try:
            _release_hold(scope_ref, close_lock_ref)
        except BaseException as secondary:
            if gate is not None:
                gate._retirement_failed(primary, secondary)
            raise_failures(primary, [secondary])
        raise
    except BaseException as primary:
        if gate is not None:
            gate._retirement_failed(primary)
        raise


def _complete_hold(gate_ref, scope_ref, close_lock_ref):
    gate = gate_ref()
    primary = prior = None
    try:
        _release_with_retry(gate, scope_ref, close_lock_ref)
    except BaseException as error:
        primary = error
        prior = (BaseException.__cause__.__get__(error), BaseException.__context__.__get__(error),
                 BaseException.__suppress_context__.__get__(error))
        raise
    finally:
        if gate is not None:
            try:
                gate._retire_scope()
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


class ProfileLease:
    """A lexical scope or independently closable holder of one model epoch."""

    def __init__(self, gate, reader, scope, *, lexical=False):
        self.gate, self.reader, self.scope = gate, reader, scope
        self._close_lock = None if lexical else threading.Lock()
        self._scope_ref = weakref.ref(scope)
        self._close_lock_ref = None if lexical else weakref.ref(self._close_lock)
        # Neither the registry nor this callback owns the holder/native locks.
        # A lost/interrupted callback therefore cannot keep a locked orphan live.
        self._finalizer = (None if lexical else weakref.finalize(
            self, _complete_hold, weakref.ref(gate), self._scope_ref, self._close_lock_ref))

    @property
    def epoch(self):
        return self.reader.epoch

    def hold(self):
        return self.gate.acquire(self)

    def close(self):
        if self._finalizer is None:
            raise RuntimeError("a lexical model-profile scope retires on context exit; close standalone holds instead")
        _complete_hold(weakref.ref(self.gate), self._scope_ref, self._close_lock_ref)
        # Required release and notification precede consuming the GC fallback.
        self._finalizer.detach()


class ProfileGate:
    def __init__(self, *, wake=None, fatal=None):
        self._lock = threading.RLock()
        self._local = threading.local()
        self._readers = set()
        self.epoch = 0
        self.available = self.accepting = True
        self.wake, self.fatal = wake, fatal
        self._poisoned = threading.Lock()
        self.failure = None

    def _prune(self):
        for reader in tuple(self._readers):
            reader.scopes = {ref for ref in reader.scopes
                             if (holder := ref()) is not None and holder.scope.locked()}
            if not reader.scopes:
                self._readers.discard(reader)

    def current(self):
        stack = getattr(self._local, "stack", ())
        while stack and not stack[-1].scope.locked():
            stack.pop()  # Reclaim interrupted Python bookkeeping after native release.
        return stack[-1] if stack else None

    def _register(self, scope, binding, *, lexical):
        binding = self.current() if binding is None else binding
        with self._lock:
            self._prune()
            if self._poisoned.locked():
                raise ProfileUnavailable("the serving profile has a fatal ownership failure")
            if binding is None:
                if not self.available or not self.accepting:
                    raise ProfileUnavailable("the serving profile is draining or unavailable; retry after it is ready")
                reader = _Reader(self.epoch)
            else:
                if (not isinstance(binding, ProfileLease) or binding.gate is not self
                        or not binding.scope.locked() or binding.epoch != self.epoch or not self.available):
                    raise ProfileUnavailable("the prepared request belongs to a retired or unavailable model profile")
                reader = binding.reader
                if reader not in self._readers:
                    raise RuntimeError("a model-profile binding lost its admitted owner")
            lease = ProfileLease(self, reader, scope, lexical=lexical)
            reader.scopes.add(weakref.ref(lease))
            self._readers.add(reader)
            return lease

    def acquire(self, binding=None):
        if self._poisoned.locked():
            raise ProfileUnavailable("the serving profile has a fatal ownership failure")
        scope = threading.Lock()
        scope.acquire()
        try:
            return self._register(scope, binding, lexical=False)
        except BaseException as primary:
            if scope.locked():
                scope.release()
            try:
                self._retire_scope()
            except BaseException as cleanup:
                raise_failures(primary, [cleanup])
            raise

    def retire(self):
        wake = None
        with self._lock:
            self._prune()
            if not self._readers and not self.accepting:
                wake = self.wake
        if wake is not None:
            wake()  # No gate->Scheduler lock inversion.

    def _retire_scope(self):
        retry_control(self.retire, self._retirement_failed)

    def poison(self, error):
        # One-way native latch precedes fallible diagnostics/publication. Even
        # an interrupted lock acquisition cannot reopen model admission.
        self._poisoned.acquire(blocking=False)
        with self._lock:
            if self.failure is None:
                self.failure = error
            self.accepting = False

    def _retirement_failed(self, primary, secondary=None):
        self.poison(primary)
        if self.fatal is not None:
            try:
                self.fatal(primary)
            except BaseException as containment:
                raise_failures(primary, [error for error in (secondary, containment) if error is not None])
        raise_failures(primary, [] if secondary is None else [secondary])

    @property
    def borrowers(self):
        with self._lock:
            self._prune()
            return len(self._readers)

    def drain(self):
        with self._lock:
            self.accepting = False

    def unavailable(self):
        with self._lock:
            self._prune()
            if self._readers:
                raise RuntimeError("model borrowers must retire before releasing the profile")
            self.available = self.accepting = False

    def publish(self, epoch):
        with self._lock:
            self._prune()
            if self._poisoned.locked() or self._readers or type(epoch) is not int or epoch != self.epoch + 1:
                raise RuntimeError("a replacement profile needs an empty predecessor and the next epoch")
            self.epoch = epoch
            self.available = self.accepting = True

    @contextmanager
    def scope(self, binding=None):
        if self._poisoned.locked():
            raise ProfileUnavailable("the serving profile has a fatal ownership failure")
        lease = stack = primary = prior = None
        scope = threading.Lock()
        try:
            # Native context release survives an interruption of Python stack
            # retirement; the next owner can prune that released journal.
            with scope:
                lease = self._register(scope, binding, lexical=True)
                stack = getattr(self._local, "stack", None)
                if stack is None:
                    stack = self._local.stack = []
                stack.append(lease)
                yield lease
        except BaseException as error:
            primary = error
            prior = (BaseException.__cause__.__get__(error), BaseException.__context__.__get__(error),
                     BaseException.__suppress_context__.__get__(error))
            raise
        finally:
            def unstack():
                if stack and lease in stack:
                    if stack[-1] is not lease:
                        raise RuntimeError("model-profile scopes must retire in nesting order")
                    stack.pop()
            try:
                finish((unstack, self._retire_scope))
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

    def wrap(self, call):
        """Wrap a real bound entrypoint without retaining its App/engine cyclically."""
        instance = getattr(call, "__self__", None)
        function = getattr(call, "__func__", call)
        owner = weakref.ref(instance) if instance is not None else None
        signature = inspect.signature(call)

        @wraps(function)
        def bound(*args, **kwargs):
            with self.scope():
                if owner is None:
                    return function(*args, **kwargs)
                target = owner()
                if target is None:
                    raise RuntimeError("the model-profile entrypoint owner has retired")
                return function(target, *args, **kwargs)
        bound.__signature__ = signature
        return bound

    def record(self):
        with self._lock:
            self._prune()
            poisoned = self._poisoned.locked()
            return {"epoch": self.epoch, "available": self.available and not poisoned,
                    "accepting": self.accepting and not poisoned, "borrowers": len(self._readers)}

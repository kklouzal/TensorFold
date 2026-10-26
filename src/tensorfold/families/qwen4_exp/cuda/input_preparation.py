"""One bounded CPU preparation protocol shared by a bound model's callers.

The coordinator serializes operations. Same-thread nested staging belongs to
the outer phase; no forward collective may run inside a preparation operation.
This initial correctness policy has unqualified per-operation control cost.
"""

import threading
from types import SimpleNamespace
import weakref

from tensorfold.cleanup import raise_failures

from .execution_identity import _ManifestRefusal, _fatal, _phase, _retire_store, _unretired
from .position_coordinates import _digest

KEY = "input_preparation_context"


class InputPreparationContext:
    def __init__(self, weights):
        identity = weights.meta.get("coordinate_profile_identity")
        if weights.positions is None or not _digest(identity):
            raise ValueError("input preparation requires the admitted full coordinate/profile identity")
        comm = weights.comm
        rank, world = weights.meta.get("rank", 0), weights.meta.get("world", 1)
        if (type(rank) is not int or type(world) is not int or not 0 <= rank < world
                or world > 1 and (comm is None or comm.world != world or comm.rank != rank or not hasattr(comm, "store"))
                or world == 1 and (rank != 0 or comm is not None)):
            raise ValueError("input preparation requires the actual rank/world and existing same-host store")
        self.identity, self.ordinal = identity, 0
        self.cohort = SimpleNamespace(rank=rank, tp=world, comm=comm, multi=None,
                                      _execution_manifest_uncertain=False, _abort_comm=False)
        self._decoder = None
        self._lock = threading.RLock()
        self._active = None
        self._pure = False

    def bind_decoder(self, decoder):
        if decoder.w.meta.get(KEY) is not self:
            raise ValueError("input preparation belongs to another resident model")
        self._decoder = weakref.ref(decoder)

    def fail(self, error):
        """A post-preparation invariant failure cannot become a local request refusal."""
        self.cohort.multi = None if self._decoder is None else self._decoder()
        try:
            _fatal(self.cohort, error)
        finally:
            self.cohort.multi = None

    def prepare(self, label, operation, *, may_refuse=False, contract=None):
        """Agree local preparation and an optional top-level publication digest.

        The same phase/channel compares every contract before any caller may
        submit the next collective. Mismatch is fatal, without key retirement.
        """
        if not self._lock.acquire(blocking=False):
            error = RuntimeError("input preparation must stay on its serialized coordinator")
            _fatal(self.cohort, error)
            raise error
        try:
            if self.cohort._execution_manifest_uncertain:
                raise RuntimeError("input preparation cohort is poisoned")
            if (type(label) is not str or not 0 < len(label) <= 64
                    or any(not ("a" <= c <= "z" or "0" <= c <= "9" or c in "-_") for c in label)
                    or type(may_refuse) is not bool or not callable(operation)):
                error = ValueError("input preparation requires a bounded role label and explicit operation policy")
                _fatal(self.cohort, error)
                raise error
            if contract is not None and not _digest(contract):
                error = ValueError("input preparation contract must be None or a lowercase SHA256 digest")
                _fatal(self.cohort, error)
                raise error
            thread = threading.get_ident()
            if self._active is not None:
                if contract is not None:
                    error = RuntimeError("rank publication contracts require top-level input preparation")
                    _fatal(self.cohort, error)
                    raise error
                if self._active != thread or self._pure and not may_refuse:
                    error = RuntimeError("a pure input preflight cannot contain device preparation")
                    _fatal(self.cohort, error)
                    raise error
                return operation()
            if self.ordinal >= (1 << 63) - 1:
                error = RuntimeError("input preparation ordinal exhausted; retire the resident cohort")
                _fatal(self.cohort, error)
                raise error
            self.ordinal += 1  # Before any window, lease, proof or tensor construction.
            channel = f"tensorfold/input-preparation/{self.identity}/{self.ordinal}/{label}"
            self._active, self._pure = thread, may_refuse
            self.cohort.multi = None if self._decoder is None else self._decoder()
            try:
                if self.cohort.comm is None:
                    try:
                        return operation()
                    except BaseException as error:
                        if not may_refuse or not isinstance(error, (ValueError, OSError)) or _unretired(error):
                            _fatal(self.cohort, error)
                        raise
                result = []
                def build():
                    result.append(operation())
                    return contract
                primary = None
                try:
                    contracts = _phase(self.cohort, channel, "prepare", build, may_refuse=may_refuse)
                    if any(value != contract for value in contracts):
                        error = RuntimeError("input preparation ranks disagree on the publication contract")
                        _fatal(self.cohort, error)
                        raise error
                except _ManifestRefusal as error:
                    primary = error
                if not self.cohort._execution_manifest_uncertain:
                    try:
                        _retire_store(self.cohort, channel, stages=("prepare",))
                    except BaseException as retirement:
                        try:
                            _fatal(self.cohort, retirement)
                        except BaseException as notification:
                            raise_failures(primary, (retirement, notification))
                        raise_failures(primary, (retirement,))
                if primary is not None:
                    raise primary
                return result[0]
            finally:
                self.cohort.multi = None  # A long-lived model must not retain its decoder through this borrow.
                self._active, self._pure = None, False
        finally:
            self._lock.release()


def reject_prepared(buffers, error):
    """Contain a post-preparation invariant failure before peer forward work."""
    context = getattr(buffers, "input_preparation_context", None)
    if context is not None:
        try:
            context.fail(error)
        except BaseException as notification:
            raise_failures(error, (notification,))
    raise error


def install_input_preparation(weights):
    """Install once after trusted identity/position binding, before any warmup."""
    existing = weights.meta.get(KEY)
    if existing is not None:
        if type(existing) is not InputPreparationContext or existing.identity != weights.meta.get("coordinate_profile_identity"):
            raise ValueError("input preparation rebinding requires retiring the resident model")
        return existing
    context = InputPreparationContext(weights)
    weights.meta[KEY] = context
    return context

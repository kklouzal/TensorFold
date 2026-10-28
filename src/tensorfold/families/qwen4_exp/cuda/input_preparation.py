"""One bounded CPU preparation protocol shared by a bound model's callers.

The coordinator serializes operations. Same-thread nested staging belongs to
the outer phase; no forward collective may run inside a preparation operation.
This initial correctness policy has unqualified per-operation control cost.
"""

import threading
from datetime import timedelta
import json
from types import MappingProxyType, SimpleNamespace
import weakref

from tensorfold.cleanup import finish, raise_failures
from tensorfold.cuda.store_fields import validate_store_key
from tensorfold.engine.model_identity import canonical_runtime_json

from .execution_identity import (
    _ManifestRefusal, _fatal, _phase, _retire_store, _unretired, input_preparation_value_bytes,
)
from .position_coordinates import _digest

KEY = "input_preparation_context"

# These trusted coordinator roles have fixed ASCII spellings. Other supported
# labels and every non-success envelope keep the general canonical serializer.
_SUCCESS_PREFIXES = MappingProxyType({label: '{"label":"' + label + '","ordinal":' for label in (
    "prefill-window", "decode-window", "mixed-prefill-window", "mtp-refresh-window", "mtp-proposal-window",
)})
_SUCCESS_KEYS = frozenset(("ok", "value"))
_MISSING_RESULT = object()


def _success_envelope(ordinal, label, outcome, max_bytes):
    """Return exact canonical wire text only for the bounded immutable domain.

    This validates the entire known success schema, not just an ok bit. Equality
    against these bytes later proves every envelope field. No mutable protocol
    value can take this path; refusal/unknown/noncanonical data use old parsing.
    """
    if (type(ordinal) is not int or not 1 <= ordinal <= (1 << 63) - 1
            or type(label) is not str or label not in _SUCCESS_PREFIXES
            or type(max_bytes) is not int or max_bytes <= 0
            or type(outcome) is not dict or any(type(key) is not str for key in outcome)
            or outcome.keys() != _SUCCESS_KEYS or outcome["ok"] is not True):
        return None
    value = outcome["value"]
    prefix = _SUCCESS_PREFIXES[label]
    if value is None:
        encoded = f'{prefix}{ordinal},"outcome":{{"ok":true,"value":null}},"version":1}}'
    elif _digest(value):
        encoded = f'{prefix}{ordinal},"outcome":{{"ok":true,"value":"{value}"}},"version":1}}'
    else:
        return None
    # Every fragment/value here is ASCII. The general serializer's complete
    # UTF8 byte bound therefore equals this character count exactly.
    if len(encoded) > max_bytes:
        raise ValueError("runtime identity exceeds byte bound")
    return encoded


class _PreparationMailbox:
    """One model/coordinator's bounded same-host outcome slots.

    A native ADD response orders each sender's prior SET. Count W publishes
    arrival readiness; count2W publishes validated consumer retirement. Count3W
    acknowledges every consumption-latch reader before the leader deletes it. No
    outcome may be overwritten until every consumer has completed its read.
    Only four epoch keys are deleted; W outcome slots live until model drain.
    """

    def __init__(self, context, store, read_many, add):
        self.context, self.store, self.read_many, self.add = context, store, read_many, add
        self.base = f"tensorfold/input-mailbox-v1/{context.identity}"
        if context.cohort.tp > ((1 << 63) - 1) // 3:
            raise OverflowError("input mailbox rank count exceeds its signed counter domain")
        self.slots = tuple(f"{self.base}/rank/{rank}" for rank in range(context.cohort.tp))
        self.active = None
        for key in (*self.slots, f"{self.base}/epoch/{(1 << 63) - 1}/count",
                    f"{self.base}/epoch/{(1 << 63) - 1}/ready", f"{self.base}/epoch/{(1 << 63) - 1}/consumed",
                    f"{self.base}/epoch/{(1 << 63) - 1}/ack"):
            validate_store_key(key)

    def begin(self, ordinal, label, channel):
        if self.active is not None:
            raise RuntimeError("input preparation mailbox still owns an unresolved epoch")
        prefix = f"{self.base}/epoch/{ordinal}"
        self.active = (ordinal, label, channel, prefix + "/count", prefix + "/ready", prefix + "/consumed", prefix + "/ack")

    def _epoch(self, channel):
        epoch = self.active
        if epoch is None or epoch[2] != channel:
            raise RuntimeError("input preparation mailbox differs from its active epoch")
        return epoch

    def exchange(self, channel, stage, outcome, fields):
        ordinal, label, _channel, counter, ready, _consumed, _ack = self._epoch(channel)
        if stage != "prepare":
            raise RuntimeError("input mailbox is only the preparation phase transport")
        context, world = self.context, self.context.cohort.tp
        encoded = _success_envelope(ordinal, label, outcome, context.cohort._phase_value_bytes)
        expected = None
        if encoded is None:
            record = {"version": 1, "ordinal": ordinal, "label": label, "outcome": outcome}
            encoded = canonical_runtime_json(record, max_bytes=context.cohort._phase_value_bytes)
        else:
            expected = encoded.encode("ascii")
        try:
            if fields is not None:
                fields.value(encoded)
            self.store.set(self.slots[context.cohort.rank], encoded)
        finally:
            encoded = None
        arrived = self.add(counter, 1)
        if type(arrived) is not int or not 1 <= arrived <= world:
            raise RuntimeError("input preparation arrival counter left its epoch")
        if arrived == world:
            self.store.set(ready, "1")
        self.store.wait([ready], timedelta(seconds=3600))
        values = self.read_many(self.slots)
        try:
            if len(values) != world:
                raise RuntimeError("input preparation received an incomplete outcome batch")
            outcomes = []
            for value in values:
                if type(value) is not bytes or len(value) > context.cohort._phase_value_bytes:
                    raise ValueError("input preparation outcome exceeds its bounded byte schema")
                if fields is not None:
                    fields.received(value)
                if expected is not None and value == expected:
                    # read_many preserves the ordered per-rank slot keys. Exact
                    # equality proves version/epoch/label/full outcome for this
                    # rank too. _phase only reads these per-phase dictionaries,
                    # then projects their immutable None/SHA values; no mutable
                    # outcome reference escapes that boundary.
                    outcomes.append(outcome)
                    continue
                row = json.loads(value)
                if (type(row) is not dict or row.keys() != {"version", "ordinal", "label", "outcome"}
                        or type(row["version"]) is not int or row["version"] != 1
                        or type(row["ordinal"]) is not int or row["ordinal"] != ordinal or row["label"] != label):
                    raise RuntimeError("input preparation mailbox contains a stale or different epoch")
                outcomes.append(row["outcome"])
            return outcomes
        finally:
            values = None

    def retire(self, channel, stages):
        _ordinal, _label, _channel, counter, ready, consumed, ack = self._epoch(channel)
        if tuple(stages or ()) != ("prepare",):
            raise RuntimeError("input mailbox retirement requires its preparation phase")
        world = self.context.cohort.tp
        retired = self.add(counter, 1)
        if type(retired) is not int or not world < retired <= 2 * world:
            raise RuntimeError("input preparation consumer counter left its epoch")
        if retired == 2 * world:
            self.store.set(consumed, "1")
        self.store.wait([consumed], timedelta(seconds=3600))
        seen = self.add(counter, 1)
        if type(seen) is not int or not 2 * world < seen <= 3 * world:
            raise RuntimeError("input preparation latch acknowledgment left its epoch")
        if seen == 3 * world:
            self.store.set(ack, "1")
        # The final latch has only one reader. Its publication proves every
        # other latch read is complete, including a delayed WAIT submission.
        if self.context.cohort.rank == 0:
            self.store.wait([ack], timedelta(seconds=3600))
            finish(tuple(lambda key=key: self.store.delete_key(key) for key in (counter, ready, consumed, ack)))
        self.active = None

    def close(self):
        if not self.context.cohort._execution_manifest_uncertain:
            if self.active is not None:
                raise RuntimeError("healthy input mailbox cannot close an unretired epoch")
            if self.context.cohort.rank == 0:
                finish(tuple(lambda key=key: self.store.delete_key(key) for key in self.slots))
        # A poisoned epoch is retained in the terminal Store, never reused.
        self.store = self.read_many = self.add = None
        self.context = None


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
        read_many = None if comm is None else getattr(comm.store, "multi_get", None)
        self.cohort._phase_read_many = read_many if callable(read_many) else None
        self.cohort._phase_value_bytes = input_preparation_value_bytes()
        add = None if comm is None else getattr(comm.store, "add", None)
        self.cohort._phase_mailbox = (_PreparationMailbox(self, comm.store, read_many, add)
            if world > 1 and callable(read_many) and callable(add) else None)
        self._decoder = None
        self._lock = threading.RLock()
        self._active = None
        self._pure = False
        self._closed = False
        self._closing = False

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
            if self._closing:
                raise RuntimeError("input preparation model owner has retired")
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
            try:
                self.cohort.multi = None if self._decoder is None else self._decoder()
                self.ordinal += 1  # Before any window, lease, proof or tensor construction.
                channel = f"tensorfold/input-preparation/{self.identity}/{self.ordinal}/{label}"
                mailbox = self.cohort._phase_mailbox
                if mailbox is not None:
                    mailbox.begin(self.ordinal, label, channel)
                self._active, self._pure = thread, may_refuse
            except BaseException as error:
                try:
                    _fatal(self.cohort, error)
                except BaseException as notification:
                    raise_failures(error, (notification,))
                finally:
                    self.cohort.multi = None
                    self._active, self._pure = None, False
                raise
            try:
                if self.cohort.comm is None:
                    try:
                        return operation()
                    except BaseException as error:
                        if not may_refuse or not isinstance(error, (ValueError, OSError)) or _unretired(error):
                            _fatal(self.cohort, error)
                        raise
                result = _MISSING_RESULT
                def build():
                    nonlocal result
                    result = operation()
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
                if result is _MISSING_RESULT:
                    raise RuntimeError("input preparation phase did not invoke its operation")
                return result
            finally:
                self.cohort.multi = None  # A long-lived model must not retain its decoder through this borrow.
                self._active, self._pure = None, False
        finally:
            self._lock.release()

    def close(self):
        """Existing model drain owns this call before retained transport reuse."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("input preparation is still borrowed during model retirement")
        try:
            if self._closed:
                return
            if self._active is not None:
                raise RuntimeError("input preparation cannot retire inside its active operation")
            self._closing = True
            if self.cohort._phase_mailbox is not None:
                self.cohort._phase_mailbox.close()
                self.cohort._phase_mailbox = None
            self.cohort._phase_read_many = None
            self._closed = True
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

"""One certified HTTP App's source-payload credits, including its TP peers.

The pool is an application allocation bound, not an OS reservation or an RSS
certificate. A lease outlives every borrower; closing a socket never retires a
worker's borrow. Growing an allocated owner never waits for another growing
owner: it either obtains the complete next phase or refuses before allocation.
Only an initial, still-empty owner may wait in the bounded FIFO.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
import math
import threading
import time
import weakref

from .cancellation import RequestCancelled
from .errors import CapacityError


_CURRENT = ContextVar("tensorfold_http_host_payload", default=None)


def _bytes(value):
    if type(value) is not int or value < 0:
        raise ValueError("host payload credits require nonnegative integer bytes")
    return value


def json_wire_bound(value, *, maximum, depth_limit=10000, shape=None):
    """Bound default ASCII JSON without allocating escaped strings or a wire.

    This measures source-owned exact JSON trees, not a replacement serializer.
    Each Unicode codepoint needs at most twelve ASCII bytes: two escaped
    UTF16 surrogate units for an astral character. Object header/allocation
    sizes are NOT encoded lengths. Decimal int digits are bounded from bit length
    and the selected 4300-digit policy; binary64 repr has at most24 characters.
    Iterators/active ancestors are bounded by the selected C encoder's depth.
    """
    maximum = _bytes(maximum)
    stack, active = [(iter((value,)), None)], {}
    total = 0
    nodes = chars = 0

    def scalar(item, *, key=False):
        nonlocal nodes, chars
        nodes += 1
        kind = type(item)
        if kind is str:
            chars += len(item)
            return 12 * len(item) + 2
        if kind is bool:
            count = 4 if item else 5
        elif item is None:
            count = 4
        elif kind is int:
            count = min(4301, max(1, (item.bit_length() * 30103 + 99999) // 100000) + (item < 0))
        elif kind is float:
            count = 24
        else:
            raise TypeError("certified HTTP serialization requires owned exact JSON values")
        return count + (2 if key else 0)

    try:
        while stack:
            iterator, identity = stack[-1]
            try:
                item = next(iterator)
            except StopIteration:
                stack.pop()
                if identity is not None:
                    active.pop(identity)
                continue
            kind = type(item)
            if kind in (dict, list, tuple):
                nodes += 1
                identity = id(item)
                if identity in active:
                    raise ValueError("Circular reference detected")
                if len(stack) > depth_limit:
                    raise ValueError("JSON tree exceeds the certified encoder depth")
                total += 2 + max(0, len(item) - 1) * 2
                if kind is dict:
                    for key in item:
                        total += scalar(key, key=True) + 2
                    children = iter(item.values())
                else:
                    children = iter(item)
                active[identity] = None
                stack.append((children, identity))
            else:
                total += scalar(item)
            if total > maximum:
                raise CapacityError("JSON wire cannot fit the host payload allowance")
        if shape is not None:
            shape.update(nodes=nodes, string_chars=chars)
        return total
    finally:
        # A failed measurement must not retain an uncharged traversal journal.
        stack.clear()
        active.clear()


class HostPayloadPool:
    def __init__(self, capacity, *, body_bound, generation_bound, encoding_bound,
                 cache_bound, measurement_bytes, control_bound=None,
                 prepaid_encoder_bytes=0, max_handlers=None, deadline_seconds=None):
        self.capacity = _bytes(capacity)
        self.body_bound, self.generation_bound = body_bound, generation_bound
        self.encoding_bound, self.cache_bound = encoding_bound, cache_bound
        self.measurement_bytes = _bytes(measurement_bytes)
        self.prepaid_encoder_bytes = _bytes(prepaid_encoder_bytes)
        self.control_bound = control_bound
        if max_handlers is not None and (type(max_handlers) is not int or max_handlers <= 0):
            raise ValueError("host payload handlers require the actual positive connection cap")
        self.max_handlers = max_handlers
        self.active_handlers = 0
        if not all(callable(value) for value in
                   (body_bound, generation_bound, encoding_bound, cache_bound)):
            raise TypeError("host payload phase bounds must be owned callables")
        if deadline_seconds is not None and (type(deadline_seconds) not in (int, float)
                or not math.isfinite(deadline_seconds) or deadline_seconds <= 0):
            raise ValueError("host payload deadline must be a finite positive duration")
        self.deadline_seconds = deadline_seconds
        self._condition = threading.Condition()
        self._leases = set()
        self._waiters = deque()
        self.used = self.highwater = 0
        self.failure = None
        self.last_refusal = None
        self.capacity_changed = None

    def lease(self):
        lease = HostPayloadLease(self)
        with self._condition:
            self._healthy()
            if self.max_handlers is not None and self.active_handlers >= self.max_handlers:
                raise CapacityError("accepted HTTP handler capacity reached")
            try:
                self._leases.add(lease)
                self.active_handlers += 1
            except BaseException as primary:
                self.poison(primary)  # Registration may have published before exhaustion.
                raise
            return lease

    def _healthy(self):
        if self.failure is not None:
            raise RuntimeError("HTTP host payload owner is poisoned") from self.failure

    def poison(self, failure):
        with self._condition:
            if self.failure is None:
                self.failure = failure
            self._condition.notify_all()

    def record(self):
        with self._condition:
            return {"schema": "tensorfold.http-host-payload-pool-v1",
                    "capacity_bytes": self.capacity, "reserved_bytes": self.used,
                    "highwater_bytes": self.highwater, "owners": len(self._leases),
                    "active_handlers": self.active_handlers,
                    "waiters": len(self._waiters), "poisoned": self.failure is not None,
                    "last_refusal": None if self.last_refusal is None else dict(self.last_refusal)}

    def fixed_slots_full(self):
        with self._condition:
            return self.max_handlers is not None and self.active_handlers >= self.max_handlers


class HostPayloadLease:
    def __init__(self, pool):
        self.pool = pool
        self.claims = {}
        self.borrowers = 1  # The HTTP handler, including final socket publication.
        self.owner_released = False
        self.fixed_slot_released = False
        self.fixed_slot_transferred = False
        self.closed = False
        self.body_number = 0
        self.body_bytes = 0
        self._generation_output_bound = self._cache_output_bound = self._control_output_bound = 0
        self._ids_output_bound = 0
        self._snapshot_output_bound = self._snapshot_wire = self._snapshot_graph = 0
        self._ids_count = self._ids_entries = self._ids_positions = self._metadata_chars = 0
        self.tokenization_reply = self.lexical_output = False
        self.failure = None
        self.retired_handler_bytes = getattr(pool, "header_payload_bytes", 0) + getattr(pool, "header_other_bytes", 0)
        self.cancelled = None
        self.control_required = False
        self.control_kind = "metadata"
        self.deadline = (None if pool.deadline_seconds is None else
                         time.monotonic() + pool.deadline_seconds)

    def set(self, role, amount, *, allow_wait=True):
        """Reserve a whole phase before allocation; a live owner's growth cannot wait."""
        amount = _bytes(amount)
        pool = self.pool
        with pool._condition:
            pool._healthy()
            if self.closed:
                raise RuntimeError("retired HTTP host payload lease")
            previous = self.claims.get(role, 0)
            delta = amount - previous
            if (delta > 0 and role != "retired-handler" and self.cancelled is not None
                    and self.cancelled()):
                raise RequestCancelled("client left before host payload allocation")
            replacement = dict(self.claims)
            replacement[role] = amount
            if sum(replacement.values()) > pool.capacity:
                raise self._capacity_error(role, amount,
                    "request host payload exceeds the available physical/cgroup allowance")
            ticket = None
            primary = None
            try:
                while delta > pool.capacity - pool.used or (delta > 0 and pool._waiters
                        and (ticket is None or pool._waiters[0] is not ticket)):
                    # Waiting with allocated roots can deadlock all readers or
                    # phase upgrades. Refuse atomically instead; no partial grow.
                    if not allow_wait or any(self.claims.values()) or self.deadline is None:
                        raise self._capacity_error(role, amount,
                            "host payload capacity is busy; retry after consumers retire")
                    if self.cancelled is not None and self.cancelled():
                        raise RequestCancelled("client left while waiting for host payload capacity")
                    remaining = self.deadline - time.monotonic()
                    if remaining <= 0:
                        raise CapacityError("host payload capacity wait expired")
                    if ticket is None:
                        ticket = object()
                        pool._waiters.append(ticket)
                    pool._condition.wait(min(remaining, 0.05))
                    pool._healthy()
                used = pool.used + delta
                highwater = max(pool.highwater, used)
                self.claims = replacement
                pool.used, pool.highwater = used, highwater
                pool._condition.notify_all()
            except BaseException as error:
                primary = error
                raise
            finally:
                if ticket is not None:
                    try:
                        if ticket in pool._waiters:
                            pool._waiters.remove(ticket)
                        pool._condition.notify_all()
                    except BaseException as cleanup:
                        from tensorfold.cleanup import raise_failures
                        pool.poison(primary if primary is not None else cleanup)
                        raise_failures(primary, [cleanup])

    def borrow(self):
        borrowed = HostPayloadBorrow(self)
        with self.pool._condition:
            self.pool._healthy()
            if self.closed:
                raise RuntimeError("cannot borrow retired HTTP host payload")
            self.borrowers += 1
        return borrowed

    def _capacity_error(self, role, amount, message):
        """Bounded operator status; no request/header data or public JSON fields."""
        error = CapacityError(message)
        try:
            record = {"schema": "tensorfold.http-host-payload-refusal-v1", "phase": role,
                "requested_phase_bytes": amount, "capacity_bytes": self.pool.capacity,
                "reserved_bytes": self.pool.used, "owner_reserved_bytes": sum(self.claims.values()),
                "active_handlers": self.pool.active_handlers}
            self.pool.last_refusal = record
            error._tensorfold_host_refusal = record
            BaseException.add_note(error, f"host payload phase={role}; requested={amount}; "
                f"reserved={self.pool.used}; capacity={self.pool.capacity}")
        except BaseException as cleanup:
            from tensorfold.cleanup import raise_failures
            self.poison(cleanup)
            raise_failures(error, [cleanup])
        return error

    def close(self):
        with self.pool._condition:
            if self.owner_released:
                raise RuntimeError("HTTP host payload owner retired twice")
            if self.failure is None:
                if self.borrowers > 1 and not self.fixed_slot_transferred:
                    try:
                        self.set("retired-handler", self.retired_handler_bytes)
                        self.fixed_slot_transferred = True
                    except CapacityError:
                        # Keep the existing fixed H slot until the final borrow
                        # retires. Listener backpressure prevents new unpaid
                        # headers; do not grow/wait or drop the old callback.
                        pass
                self._release()
                if not self.closed and self.fixed_slot_transferred:
                    self._release_fixed_slot()
            # An escaping uncertain failure owns this reference instead of the
            # socket. The poisoned pool retains the journal until containment.
            self.owner_released = True

    def poison(self, error):
        with self.pool._condition:
            if self.failure is None:
                self.failure = error
            self.pool.poison(error)

    def _release(self):
        """Retire one already-published reference under the shared pool lock."""
        with self.pool._condition:
            if self.closed or self.borrowers <= 0:
                raise RuntimeError("HTTP host payload owner retired twice")
            self.borrowers -= 1
            if self.borrowers:
                return
            self.pool.used -= sum(self.claims.values())
            self.pool._leases.remove(self)
            self.closed = True
            self._release_fixed_slot()
            self.pool._condition.notify_all()

    def _release_fixed_slot(self):
        if not self.fixed_slot_released:
            self.pool.active_handlers -= 1
            self.fixed_slot_released = True
            if self.pool.capacity_changed is not None:
                self.pool.capacity_changed()

    def begin_body(self):
        self.body_number += 1
        body = BodyPayload(self, "body/" + str(self.body_number))
        body.receive(0)  # Even an empty frame's {} parse has fixed working owners.
        return body

    def generation(self, *, body_bytes, output_tokens):
        amount, wire = self.pool.generation_bound(body_bytes, output_tokens)
        self.set("generation", max(amount, self.claims.get("generation", 0)))
        self._generation_output_bound = max(self._generation_output_bound, wire)

    def rpc(self, encoded_bytes):
        # Generation already pays one maximum request-child/source allowance.
        # Larger retained peer/error trees are charged before publication.
        amount = max(0, self.pool.rpc_bound(encoded_bytes) - self.pool.rpc_request_base)
        self.set("rpc", max(amount, self.claims.get("rpc", 0)))

    def cache(self, encoded_bytes, *, copying):
        # No cache snapshot is borrowed yet; small facade metadata is prepaid.
        if self.claims.get("control"):
            self.set("control", 0)
            self._control_output_bound = 0
        self.set("cache", max(self.pool.cache_bound(encoded_bytes, copying=copying), self.claims.get("cache", 0)))
        self._cache_output_bound = max(self._cache_output_bound, 12 * encoded_bytes + self.pool.metadata_wire_bytes)

    def control(self, source_bytes):
        if self.pool.control_bound is None:
            raise RuntimeError("HTTP control payload has no owned source bound")
        amount, wire = self.pool.control_bound(source_bytes, self.control_kind)
        self.set("control", max(amount, self.claims.get("control", 0)))
        self._control_output_bound = max(self._control_output_bound, wire)

    @property
    def funded_output_bound(self):
        # Only still-paid phases can underwrite later encoder allocations.
        return max(self._generation_output_bound, self._cache_output_bound,
                   self._control_output_bound, self._ids_output_bound, self._snapshot_output_bound)

    def snapshot(self, value):
        """Pay a new metadata clone before copying its identity-locked source."""
        try:
            wire = json_wire_bound(value, maximum=self.pool.capacity)
            total_wire = self._snapshot_wire + wire
            graph = self._snapshot_graph + self.pool.snapshot_bound(wire)
            future = max(self.pool.metadata_wire_bytes, self.pool.monitor_wire_bytes) + total_wire
            amount = graph + max(0, self.pool.encoding_bound(future) - self.pool.prepaid_encoder_bytes)
            self.set("metadata-snapshots", amount, allow_wait=False)
        except (CapacityError, RequestCancelled) as primary:
            # No clone was made. Do not pin a replaced source generation in
            # this failed observer's traceback after releasing its identity lock.
            del value
            try:
                _retire_measurement_frames(primary)
            except BaseException:
                self.poison(primary)
                raise
            raise
        self._snapshot_wire, self._snapshot_graph = total_wire, graph
        self._snapshot_output_bound = future

    def ids(self, count, *, entries=0, positions=0, metadata_chars=0, metadata_snapshot=False):
        """High-water of simultaneously owned native getter/Python ID extents."""
        values = tuple(_bytes(value) for value in (count, entries, positions, metadata_chars))
        if metadata_snapshot:
            if not self.pool.tokenizer_serialized_chars:
                raise RuntimeError("HTTP tokenizer serialization extent was not captured before admission")
            values = (*values[:3], max(values[3], self.pool.tokenizer_serialized_chars))
        with self.pool._condition:
            values = tuple(max(old, new) for old, new in zip(
                (self._ids_count, self._ids_entries, self._ids_positions, self._metadata_chars), values))
            amount, wire = self.pool.ids_bound(*values,
                reply=self.tokenization_reply, lexical=self.lexical_output)
            base = self.pool.generation_ids_base if self.claims.get("generation") else 0
            self.set("tokenizer-ids", max(0, amount - base), allow_wait=False)
            self._ids_count, self._ids_entries, self._ids_positions, self._metadata_chars = values
            self._ids_output_bound = max(self._ids_output_bound, wire)

    def segment_cache(self, limit, entries, text_chars):
        amount = self.pool.segment_bound(_bytes(limit), _bytes(entries), _bytes(text_chars))
        self.set("segment-cache", max(amount, self.claims.get("segment-cache", 0)), allow_wait=False)

    def render(self, chars, parts):
        amount = self.pool.render_bound(_bytes(chars), _bytes(parts))
        self.set("render", max(amount, self.claims.get("render", 0)), allow_wait=False)

    def render_input(self, value):
        # This is the already-paid parsed request; bounded traversal metadata
        # is fixed-paid per handler. Count actual nodes instead of assuming
        # a one-MiB string contains half a million empty container objects.
        shape = {}
        json_wire_bound(value, maximum=self.pool.capacity, shape=shape)
        amount = self.pool.render_bound(shape["string_chars"], shape["nodes"])
        self.set("render-input", max(amount, self.claims.get("render-input", 0)), allow_wait=False)

    def encode(self, value, *, prefix="", suffix=""):
        # Measurement metadata and one bounded refusal encoder are prepaid per
        # accepted handler at startup, including when the variable pool is full.
        wire = json_wire_bound(value, maximum=self.pool.capacity)
        # Framing is source-owned ASCII in the certified HTTP producer.
        if not prefix.isascii() or not suffix.isascii():
            raise ValueError("certified HTTP framing requires ASCII")
        wire += len(prefix) + len(suffix)
        # A generation/cache lease already funds its future serialization.
        # Reusing it cannot fail after a previously admitted answer exists.
        prepaid = max(self.pool.prepaid_encoder_bytes,
                      0 if not self.funded_output_bound else self.pool.encoding_bound(self.funded_output_bound))
        amount = max(0, self.pool.encoding_bound(wire) - prepaid)
        self.set("encoder", max(amount, self.claims.get("encoder", 0)))
        return wire


def reserve_rpc(source, encoded_bytes):
    borrowed = getattr(source, "_host_payload_borrow", None)
    if borrowed is not None:
        borrowed.lease.rpc(encoded_bytes)


def _retire_measurement_frames(primary):
    import types
    codes = {json_wire_bound.__code__}
    codes.update(code for code in json_wire_bound.__code__.co_consts if isinstance(code, types.CodeType))
    trace = primary.__traceback__
    try:
        while trace is not None:
            frame = trace.tb_frame
            if frame.f_code in codes:
                frame.clear()
                _ = frame.f_locals  # Refresh any diagnostic locals cache after clear.
            trace = trace.tb_next
    except BaseException as cleanup:
        from tensorfold.cleanup import raise_failures
        raise_failures(primary, [cleanup])


def borrow_request(pool, stream, mailbox):
    owner = current()
    if owner is None:
        return  # Direct callers retain their separately checked static path.
    if owner.pool is not pool:
        raise ValueError("backend work belongs to a different HTTP payload owner")
    borrowed = owner.borrow()
    try:
        # The same loan follows BOTH containers, including idle worker-frame
        # snapshots. The callback owns only the lease, never its weak target.
        borrowed._container_retirement = weakref.finalize(borrowed, _retire_container_borrow, owner)
        mailbox._host_payload_borrow = borrowed
        stream._host_payload_borrow = borrowed
    except BaseException as failure:
        owner.poison(failure)
        raise


def retire_mailbox(mailbox):
    """Only a certified unpublished rollback may retire a live container loan."""
    borrowed = getattr(mailbox, "_host_payload_borrow", None)
    if borrowed is not None:
        try:
            borrowed.close()
            mailbox._host_payload_borrow = None
        except BaseException as failure:
            borrowed.lease.poison(failure)
            raise


def _retire_container_borrow(lease):
    try:
        lease._release()
    except BaseException as failure:
        lease.poison(failure)  # Owning status retains asynchronous cleanup failure.


class BodyPayload:
    def __init__(self, lease, role):
        self.lease, self.role, self.encoded_bytes = lease, role, 0

    def receive(self, amount):
        amount = _bytes(amount)
        # Complete raw/JSON-parse/error working phase, before read/json.loads.
        self.lease.set(self.role, self.lease.pool.body_bound(amount))
        if self.lease.control_required:
            self.lease.control(amount)
        self.lease.body_bytes += amount - self.encoded_bytes
        self.encoded_bytes = amount


class HostPayloadBorrow:
    def __init__(self, lease):
        self.lease, self.closed = lease, False

    def close(self):
        with self.lease.pool._condition:
            if self.closed:
                raise RuntimeError("HTTP host payload borrow retired twice")
            retirement = getattr(self, "_container_retirement", None)
            if retirement is not None:
                retirement.detach()
            self.lease._release()
            self.closed = True


def current():
    return _CURRENT.get()


def _retire_error_borrows(borrows):
    # Weak finalization retires payload credit only when the exception and its
    # traceback graph have no consumers. A failed release poisons its owning
    # pool; it is never an unreported destructor failure or a capacity credit.
    for borrowed in borrows:
        try:
            borrowed.close()
        except BaseException as failure:
            borrowed.lease.poison(failure)


class _ErrorPayloadOwner:
    def __init__(self):
        self.borrows = []
        self.finalizer = weakref.finalize(self, _retire_error_borrows, self.borrows)


def keep_error(error, *, lease=None):
    """A handled error can outlive logical/socket completion through its frames."""
    lease = current() if lease is None else lease
    if lease is None:
        return
    with lease.pool._condition:
        try:
            namespace = BaseException.__dict__["__dict__"].__get__(error, type(error))
            detached = (BaseException.__traceback__.__get__(error) is None
                    and BaseException.__cause__.__get__(error) is None
                    and BaseException.__context__.__get__(error) is None)
            owned_refusal = (type(error) is CapacityError
                and namespace.get("_tensorfold_host_refusal") is lease.pool.last_refusal
                and all(key in ("_tensorfold_host_refusal", "__notes__") for key in namespace)
                and all(type(note) is str and len(note) <= 1024 for note in namespace.get("__notes__", ())))
            if (detached and (not namespace or owned_refusal)
                    and all(type(value) is str and len(value) <= 1024 for value in error.args)):
                return  # Bounded detached diagnostic, covered by fixed refusal payload.
            # Tracebacks/cancel callbacks can pin self, headers and measurement
            # frames after its socket/H slot ends. Transfer that fixed slot's
            # obligation to this variable owner before publishing the borrow.
            try:
                lease.set("retired-handler", lease.retired_handler_bytes)
                lease.fixed_slot_transferred = True
            except CapacityError:
                pass  # The H slot itself remains held until this error retires.
            owner = namespace.get("_tensorfold_http_payload_owner")
            if owner is None:
                owner = _ErrorPayloadOwner()
                namespace["_tensorfold_http_payload_owner"] = owner
            if type(owner) is not _ErrorPayloadOwner:
                raise RuntimeError("HTTP error payload owner was replaced")
            if all(borrow.lease is not lease for borrow in owner.borrows):
                borrowed = lease.borrow()
                owner.borrows.append(borrowed)
        except BaseException as failure:
            # Uncertain publication retains all locals in this journal instead
            # of granting the credit back while the exception is still live.
            lease.poison(failure)
            raise


@contextmanager
def handler_scope(app):
    pool = getattr(app, "host_payload_pool", None)
    if pool is None:
        yield None
        return
    lease = pool.lease()
    token = None
    primary = None
    try:
        token = _CURRENT.set(lease)
        yield lease
    except BaseException as error:
        primary = error
        lease.poison(error)
        raise
    finally:
        from tensorfold.cleanup import raise_failures

        failures = []
        if token is not None:
            try:
                _CURRENT.reset(token)
            except BaseException as error:
                failures.append(error)
        try:
            lease.close()
        except BaseException as error:
            failures.append(error)
        if failures:
            pool.poison(primary if primary is not None else failures[0])
            raise_failures(primary, failures)



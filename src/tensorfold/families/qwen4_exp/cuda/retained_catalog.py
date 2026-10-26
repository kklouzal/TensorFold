"""One owner of retained compute claims, prefix entries and RAM generations.

The decoder actor owns admission/capture/claim. A single owned worker performs
bounded host/device transfers using a cloned CPU store connection; it never
calls model collectives. A manifest becomes readable only after every rank has
finished its exact bytes. Queue consumers and logical owners are distinct.
"""

from dataclasses import dataclass, field, replace
from datetime import timedelta
import hashlib
import json
import math
import os
import sys
import threading
import time
import weakref
import uuid
from types import SimpleNamespace

import torch

from tensorfold.cleanup import raise_failures, rollback
from tensorfold.cuda.store_fields import validate_store_key
from tensorfold.thread_work import ThreadWork
from tensorfold.engine.model_identity import canonical_runtime_json
from tensorfold.server.errors import CapacityError

from .execution_identity import _ManifestRefusal, _fatal, _phase, _retire_store, _small, _unretired
from .ram_backing import RamObjects, digest as is_digest
from .method_retention import method_restore_resources, validate_method_restore
from .retained_parts import PartSpec, TensorPart, manifest_bound
from .retention_policy import RetentionPolicy
from .state_retention import (canonical_tokens, commit_state_restore, new_turn_gpu_snapshot, new_turn_metadata,
                              new_turn_uses_part, rebind_state_inventory, request_record, state_inventory, state_restore_plan,
                              state_restore_resources)

LOCAL_PRINCIPAL = "local-unauthenticated"
_LIBUV_VALUE_BYTES = 8 * 1024**2
_LIBUV_KEY_BYTES = 8 * 1024


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


class RetentionError(ValueError):
    """Completed, healthy-owner request refusal; never uncertain device IO."""


class RetentionExpired(RetentionError):
    pass


class RetentionCorrupt(RetentionError):
    pass


class RetentionCapacity(RetentionError, CapacityError):
    """Same owned domain error and existing public 503 admission category."""
    pass


class RetentionMismatch(RetentionError):
    """Pure incompatible derived-prefix candidate; same-request may not replay."""


def _next_generation(current, kind):
    # Both actor-owned counters start at zero and only advance here. Refuse
    # exhaustion before installing a transfer or issuing a settlement phase.
    if current >= (1 << 63) - 1:
        raise RetentionCapacity(f"retained {kind} generation exhausted")
    return current + 1


@dataclass(frozen=True)
class LogicalReference:
    principal: str
    sequence: str
    generation: int

    def __post_init__(self):
        if (type(self.principal) is not str or not self.principal or type(self.sequence) is not str
                or not self.sequence.startswith(("response:", "conversation:"))
                or type(self.generation) is not int or not 1 <= self.generation < (1 << 63)):
            raise ValueError("retained compute requires a trusted standard logical reference/generation")


@dataclass(eq=False)
class RetainedHandle:
    catalog: object
    serial: int
    expires: float
    future_bytes: int
    owners: dict
    phase: str = "reserved"
    state: object = None
    inventory: object = None
    manifest: tuple | None = None
    keys: tuple = ()
    consumers: set = field(default_factory=set)
    request_holds: set = field(default_factory=set)
    transfer_readers: int = 0
    host: object = None
    ready_handoff: object = None
    failure: BaseException | None = None
    transfer: object = None
    clean_state: object = None
    clean_generation: tuple | None = None
    transfer_generation: int = 0
    canonical_ids: object = None
    retiring_keys: tuple = ()
    transient: bool = False
    pending: bool = True
    expired: bool = False
    failed_transfer: object = None
    role: str = "logical"
    internal_owner: bool = False
    principal: str | None = None


class RetainedRequest:
    """A queued/resuming consumer; close never deletes the protected owner."""
    def __init__(self, catalog, handle, *, continuation, target_handle):
        self.catalog, self.handle, self.closed = catalog, handle, True
        self.continuation, self.target_handle, self.claimed = continuation, target_handle, False
        self.restore_handle = None
        self.handoff = None
        self.source_active = True

    def eligibility(self):
        self._check()
        self.catalog.check_health()
        owner = self.restore_handle or self.handle
        if owner.failure is not None:
            raise owner.failure
        phase = owner.phase
        return "restoring" if phase in ("saving", "restoring") else None if phase in ("ram", "gpu") else "tier_unavailable"

    def claim(self, state, *, stream=None, continuation=None, ticket=None):
        self._check()
        selected = self.continuation if continuation is None else continuation
        if selected != self.continuation:
            raise ValueError("retained consumer continuation cannot change after queue admission")
        return self.catalog.claim_resume(self, state=state, stream=stream, continuation=selected, ticket=ticket)

    @property
    def resident_state(self):
        self._check()
        with self.catalog._condition:
            return self.handle.state if self.handle.phase == "gpu" else None

    def _check(self):
        if self.closed or (self not in self.handle.consumers and
                (self.target_handle is None or self not in self.target_handle.request_holds)):
            raise RuntimeError("retained request consumer is retired")

    def close(self):
        if not self.closed:
            self.catalog._release_consumer(self)


@dataclass(eq=False)
class _Transfer:
    handle: RetainedHandle
    direction: str
    generation: int = 0
    plan: object = None
    source_handle: object = None
    retired: threading.Event = field(default_factory=threading.Event)
    failure: BaseException | None = None
    handoff: object = None
    byte_sources: tuple = ()
    specs: tuple = ()
    request: object = None
    event: object = None
    retained_keys: object = None
    owner_payload: object = None
    resources: object = None


@dataclass(eq=False)
class _ActorCall:
    operation: object
    device: object
    issued: threading.Event = field(default_factory=threading.Event)
    completed: threading.Event = field(default_factory=threading.Event)
    result: object = None
    event: object = None
    error: BaseException | None = None


class RetainedCatalog:
    def __init__(self, policy, weights, states, *, prefix_limit, recovery_versions, wake=None):
        self.policy, self.w, self.states = policy, weights, states
        self.comm = weights.comm
        self.rank = weights.meta["rank"] if self.comm is None else self.comm.rank
        self.world = 1 if self.comm is None else self.comm.world
        self.prefix_limit, self.recovery_versions = prefix_limit, recovery_versions
        self.protected_recovery_versions = recovery_versions
        self._pid, self._condition = os.getpid(), threading.Condition()
        self._handles, self._references, self._prefixes = {}, {}, []
        self._logical_events = {}
        self._released_states = {}
        self._serial = 0
        self._wake = wake
        self._failure = None
        self._fatal_reported = False
        self._registration_owner = None
        self._reader_limit = None
        self._device_resources = None
        self.host_resources = None
        self._closing = self._closed = self._terminal = False
        self._pending, self._current, self._actor_call = [], None, None
        self._worker = self._thread = self._store = None
        self._wake_relay = None
        self._ram = self._buffer = None
        self._reserved = 0
        self._cohort = SimpleNamespace(comm=self.comm, rank=self.rank, tp=self.world, multi=None)
        self._actor_cohort = SimpleNamespace(comm=self.comm, rank=self.rank, tp=self.world, multi=None)
        from .retention_resources import INITIAL_COHORT_VALUE_BYTES, StoreFieldGuard

        # Preflight publishes only the bounded scalar constructor schema. The
        # observed backend capability replaces this guard BETWEEN phases.
        self._cohort.store_fields = self._actor_cohort.store_fields = StoreFieldGuard(INITIAL_COHORT_VALUE_BYTES)
        self._settlement_generation = 0
        self._manifest_bytes = 0
        self._manifest_bounds = ()
        self._native_value_limit = self._native_key_limit = None
        self.transfer_chunk_bytes = None
        self.transport_origin = None
        self.native_payload_runtime = None
        self._object_bound = 0
        self.minimum_future_bytes = 0
        self.identity = None
        self.evidence = "source-candidate; V100/full-model/tier-progress qualification pending"
        channel = "tf_retention/construction"
        stages = ("preflight", "agreement", "resources", "launch")

        def preflight():
            if policy is not None and type(policy) is not RetentionPolicy:
                raise ValueError("retention needs the typed administrator policy")
            if (type(prefix_limit) is not int or prefix_limit < 0 or type(recovery_versions) is not int
                    or recovery_versions < 0 or type(self.states) not in (list, tuple) or not self.states):
                raise ValueError("retention needs actual States and bounded prefix/recovery counts")
            self.states = tuple(self.states)
            self.recovery_versions = max(recovery_versions, prefix_limit)
            if policy is not None:
                policy.validate_world(self.world)
                if policy.tier == "gpu" and policy.max_sessions > len(self.states):
                    raise ValueError("GPU-only retained promises exceed the actual full-window slot count")
            if self.rank >= self.world or any(state.history_rank != self.rank for state in self.states):
                raise ValueError("retention catalog rank differs from its actual State owners")
            if any(state.limit != self.states[0].limit or state.kv_identity != self.states[0].kv_identity
                   for state in self.states):
                raise ValueError("retention State pool must share its actual model/context/codec/position binding")
            if any(state.history_placement is not None and (self.comm is None
                   or state.history_placement.world != self.world or state.history_comm is not self.comm)
                   for state in self.states):
                raise ValueError("unique retained histories require their complete actual State/communicator cohort")
            for state in self.states:
                state.retention_method_recovery_versions = self.recovery_versions
                if policy is not None and state.capacity != state.limit:
                    raise ValueError("retention requires actual admitted full-window States before startup publication")
            first = self.states[0]
            if self.comm is not None:
                origin = getattr(self.comm, "store_origin", None)
                if origin is not None:
                    from tensorfold.cuda.comm import validate_store_origin

                    validate_store_origin(origin, self.comm.store, self.rank, self.world, verified=True)
                    self.transport_origin = origin.record()
                    if first.pos_dev.is_cuda:
                        from .retention_resources import require_native_payload_runtime

                        self.native_payload_runtime = require_native_payload_runtime()
                elif first.pos_dev.is_cuda:
                    raise ValueError("unique FIFO progress requires the verified owned synchronous daemon cohort")
                # CPU reference stores exercise primitive/value boundaries;
                # they never issue a CUDA native-memory qualification claim.
                backend = getattr(self.comm.store, "libuvBackend", None)
                if type(backend) is not bool and first.pos_dev.is_cuda:
                    raise ValueError("retained native transfers require the actual TCPStore backend capability")
                if backend is True:
                    # Pinned Torch5ab libuv parser: independent vector/key
                    # field limits, not a whole-wire-frame size restriction.
                    self._native_value_limit = _LIBUV_VALUE_BYTES
                    self._native_key_limit = _LIBUV_KEY_BYTES
            if policy is not None and policy.tier == "ram":
                self.transfer_chunk_bytes = min(policy.chunk_bytes, self._native_value_limit or policy.chunk_bytes)
            identity = _digest({"coordinate": weights.meta.get("coordinate_profile_identity"),
                "kv": first.kv_identity, "context": first.limit,
                "position": first.position_profile_identity, "epoch": first.position_epoch})
            host = (None if self.comm is None else hashlib.sha256(_small("/proc/sys/kernel/random/boot_id")
                    + b"\0" + _small("/etc/machine-id")).hexdigest())
            payload = (0 if policy is None else max(state_inventory(state, recovery_versions=self.recovery_versions).future_bytes
                                                   for state in self.states))
            return {"payload": payload, "policy": None if policy is None else policy.record(),
                    "identity": identity, "host": host, "context": first.limit,
                    "objects": max(self._component_bound(state) for state in self.states),
                    "prefix_limit": prefix_limit, "recovery_versions": self.recovery_versions,
                    "activation": uuid.uuid4().hex if self.rank == 0 else None,
                    "native_value_limit": self._native_value_limit, "native_key_limit": self._native_key_limit,
                    "native_payload_runtime": self.native_payload_runtime}

        acquired = False
        try:
            rows = _phase(self._cohort, channel, "preflight", preflight)
            self._cohort.store_fields = self._actor_cohort.store_fields = StoreFieldGuard(self._native_value_limit)

            def agreement():
                if any(any(row[key] != rows[0][key] for key in (
                        "policy", "identity", "host", "context", "prefix_limit", "recovery_versions",
                        "native_value_limit", "native_key_limit", "native_payload_runtime")) for row in rows):
                    raise ValueError("retention requires one actual same-host policy/profile rank cohort")
                control = self.world * sum(manifest_bound(row["objects"]) for row in rows)
                minimum = 2 * sum(row["payload"] for row in rows) + control + rows[0]["context"] * 4
                if policy is not None and policy.tier == "ram" and minimum > policy.ram_bytes:
                    raise ValueError("retention RAM bound cannot back one complete future/replacement/manifest claim")
                return True

            _phase(self._cohort, channel, "agreement", agreement)
            self.identity = _digest({"profile": rows[0]["identity"], "activation": rows[0]["activation"]})
            self._manifest_bounds = tuple(manifest_bound(row["objects"]) for row in rows)
            self._object_bound = rows[self.rank]["objects"]
            self._manifest_bytes = self.world * sum(self._manifest_bounds)
            if policy is not None:
                # Authoritative and replacement generations coexist until ACK.
                # The control manifest reserve is replicated across ranks.
                self.minimum_future_bytes = (2 * sum(row["payload"] for row in rows)
                                             + self._manifest_bytes + rows[0]["context"] * 4)

            def resources():
                nonlocal acquired
                acquired = True  # Journal precedes the first acquisition.
                if policy is not None and policy.tier == "ram":
                    objects = 2 * rows[self.rank]["objects"] * policy.max_sessions
                    # One transfer worker; dedup and restore each borrow only
                    # one object at a time, including checksum/retirement.
                    self._ram = RamObjects(policy.ram_bytes, max_objects=objects, max_readers=1)
                    self._buffer = torch.empty(self.transfer_chunk_bytes, dtype=torch.uint8, device="cpu",
                                               pin_memory=self.states[0].pos_dev.is_cuda)
                    if self.comm is not None:
                        # Documented independent connection for this worker.
                        self._store = self.comm.store.clone()
                        self._store.set_timeout(timedelta(seconds=policy.lifetime_s))
                        if self.states[0].pos_dev.is_cuda:
                            from .retention_wake import RetentionWakeRelay

                            RetentionWakeRelay.create(self.comm, "tf_retention/" + self.identity,
                                wake=self._wake_local, fatal=self._wake_failed,
                                register=lambda relay: setattr(self, "_wake_relay", relay))
                    self._worker = ThreadWork(self._serve)
                    self._thread = threading.Thread(target=self._worker.run, name="tensorfold-retention", daemon=False)
                return True

            _phase(self._cohort, channel, "resources", resources)
            _phase(self._cohort, channel, "launch", self._launch_worker, may_refuse=False)
        except _ManifestRefusal:
            def cleanup():
                if acquired:
                    if self._ram is not None:
                        self._ram.close()
                    if self._wake_relay is not None:
                        self._wake_relay.close(timeout=policy.lifetime_s)
                        self._wake_relay = None
                    self._ram = self._buffer = self._store = self._worker = self._thread = None
                return True

            _phase(self._cohort, channel, "cleanup", cleanup, may_refuse=False)
            if self.comm is not None:
                _retire_store(self._cohort, channel, stages=(*stages, "cleanup"))
            raise
        if self.comm is not None:
            _retire_store(self._cohort, channel, stages=stages)
        self._cohort.comm = None if self._store is None else SimpleNamespace(store=self._store)

    def _launch_worker(self):
        if self._wake_relay is not None:
            self._wake_relay.start()
        if self._worker is not None:
            self._worker.launch(self._thread)
        return True

    def _component_bound(self, state):
        # Every payload has at least one byte except fixed empty fields. History
        # pages have at most four codec fields, one pool, and bounded tails.
        caches = [*state.kc, *([state.mtp_kc] if hasattr(state, "mtp_kc") else [])]
        width = 128 if state.history_placement is None else state.history_placement.page_tokens
        history = sum(((getattr(cache, "reserved_rows", state.limit) + width - 1) // width)
                      * (2 + int(cache.key_format.quantized) + int(cache.value_format.quantized)) for cache in caches)
        pools = sum((value.shape[0] + max(1, width // state.ratio) - 1) // max(1, width // state.ratio)
                    for value in [*state.pooled, *([state.mtp_pooled] if hasattr(state, "mtp_kc") else [])])
        compact = 0 if state.compact_indexer is None else sum(entry is not None for entry in state.compact_indexer.entries)
        legacy = len(caches) if state.compact_indexer is None else 0
        window = getattr(state, "dflash2", None)
        method = (0 if window is None else 2 * window.kv.shape[0] * window.kv.shape[1]) + 2 * int(state.mtp_hidden_width > 0)
        recovery = 3 + 2 * compact + int(window is not None) + 2 * int(state.mtp_hidden_width > 0)
        return 4 + history + pools + 8 * compact + legacy + method + self.recovery_versions * recovery

    def future_claim(self):
        """Global complete future + authoritative replacement/metadata claim."""
        return self.minimum_future_bytes

    def configure_admission(self, *, max_waiting_input_bytes, max_pending_requests=None):
        """Bind the EXISTING queue's count/byte envelope before prefix borrowing."""
        if (type(max_waiting_input_bytes) is not int or max_waiting_input_bytes <= 0
                or max_pending_requests is not None and (type(max_pending_requests) is not int or max_pending_requests <= 0)):
            raise ValueError("RAM prefix consumers require the actual bounded admission envelope")
        # Waiting charges at least the real empty prompt-list footprint, while
        # at most the admitted State pool can already be on GPU.
        maximum = (max_pending_requests if max_pending_requests is not None else
                   max_waiting_input_bytes // sys.getsizeof([]) + len(self.states))
        with self._condition:
            if sum(handle.transient for handle in self._handles.values()) > maximum:
                raise RetentionCapacity("accepted prefix readers exceed the proposed reduced admission bound")
            self._reader_limit = maximum
        if self._ram is not None:
            # Same object allocator, including temporary live continuations and
            # optional prefix generations. Only payload quota grants bytes.
            with self._ram._lock:
                self._ram.max_objects = 2 * self._object_bound * (self.policy.max_sessions + maximum + self.prefix_limit)

    def configure_host_resources(self, *, baseline_host_bytes, materialized_host_bytes=0,
                                 prepaid_native_fixed_payload_bytes=0, native_state_slots=None,
                                 native_fd_limit=None, prepaid_native_prefix_payload_bytes=0,
                                 native_host_bytes=0):
        """Ordinary GLOBAL steady obligation, excluding this policy's fixed RAM.

        Loading scratch is not a concurrent serving promise. Materialized
        credit requires an actual owned payload proof from the loader/arena;
        mapped virtual extents, RSS and advisory snapshots provide no credit.
        """
        from .retention_resources import HostBaseline

        record = HostBaseline(baseline_host_bytes, materialized_host_bytes, prepaid_native_fixed_payload_bytes,
                              native_state_slots, native_fd_limit, prepaid_native_prefix_payload_bytes,
                              native_host_bytes)
        with self._condition:
            if self._handles or self._registration_owner is not None:
                raise ValueError("ordinary host resources must bind before requests")
            if self.host_resources is not None and self.host_resources != record:
                raise ValueError("ordinary host obligations are immutable for this activation")
            self.host_resources = record

    def configure_device_resources(self, *, recovery_pool_bytes, pending_hc_bytes, snapshot_bytes):
        """THIS rank's source-header categories, bound before requests.

        The recovery pool excludes active per-piece POINTS, already paid by
        family geometry. snapshot_bytes uses that same single-version width,
        including DF when selected. No physical-free/RSS observation substitutes
        for the lifetime reservation.
        """
        values = (recovery_pool_bytes, pending_hc_bytes, snapshot_bytes)
        if any(type(value) is not int or not 0 <= value <= sys.maxsize for value in values):
            raise ValueError("retention device resources need nonnegative per-rank header bytes")
        with self._condition:
            if self._handles or self._registration_owner is not None:
                raise ValueError("retention device reservations must bind before requests")
            record = dict(zip(("recovery_pool_bytes", "pending_hc_bytes", "snapshot_bytes"), values))
            if self._device_resources is not None and self._device_resources != record:
                raise ValueError("retention device resource categories are immutable for this activation")
            self._device_resources = record

    @staticmethod
    def _tensor_storages(value, result):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            result[(value.device, storage.data_ptr())] = storage.nbytes()
        elif type(value) is dict:
            for child in value.values():
                RetainedCatalog._tensor_storages(child, result)
        elif type(value) in (tuple, list):
            for child in value:
                RetainedCatalog._tensor_storages(child, result)
        else:
            from .position_coordinates import IssuedCoordinates

            if type(value) is IssuedCoordinates:
                RetainedCatalog._tensor_storages(value.tensor, result)

    def _owned_device_versions(self):
        recovery, hidden = {}, {}
        pending_recovery = pending_hidden = 0
        with self._condition:
            for _ids, _state, snapshot, tail in self._prefixes:
                self._tensor_storages(snapshot, recovery)
                self._tensor_storages(tail, recovery)
            for handle in self._handles.values():
                handoff = handle.ready_handoff
                if handoff is not None:
                    self._tensor_storages(handoff["recoveries"], recovery)
                    self._tensor_storages(handoff["pending_hc"], hidden)
                transfer = handle.transfer
                if (handle.state is not None and handle.state.mtp_boundary is not None
                        and (transfer is None or transfer.resources is None)):
                    self._tensor_storages(handle.state.mtp_boundary.hidden, hidden)
                    self._tensor_storages(handle.state.mtp_boundary.coordinate_record, hidden)
                if handle.inventory is not None:
                    for part in handle.inventory.parts:
                        if part.name.startswith(("recovery.", "method.recovery.")):
                            self._tensor_storages(part.tensor, recovery)
                if transfer is not None and transfer.resources is not None:
                    # Each plan allocates independent payload, unlike views of
                    # existing caches. Count its proved allocation contract,
                    # including future coordinate copies, until publication.
                    pending_recovery += transfer.resources["recovery_device_bytes"]
                    pending_hidden += transfer.resources["pending_hc_device_bytes"]
        # A deferred HC may also be one prefix's recovery tail. Its actual
        # storage is charged once, with all live source references still owned.
        for key in recovery:
            hidden.pop(key, None)
        return sum(recovery.values()) + pending_recovery, sum(hidden.values()) + pending_hidden

    def _check_device_resources(self, state, resources):
        limits = self._device_resources
        if limits is None:
            if state.pos_dev.is_cuda:
                raise RetentionCapacity("retained transfer has no bound per-rank header resource categories")
            return  # CPU reference execution does not issue a GPU resource grant.
        recovery, hidden = self._owned_device_versions()
        phase = resources["phase_device_bytes"]
        with self._condition:
            for handle in self._handles.values():
                transfer = handle.transfer
                if transfer is not None and transfer.resources is not None:
                    phase = max(phase, transfer.resources["phase_device_bytes"])
        # One worker commits coordinate issuance at a time. Persistent versions
        # across all queued plans still sum; only the phase scratch takes max.
        extra = (max(0, recovery + resources["recovery_device_bytes"] - limits["recovery_pool_bytes"])
                 + max(0, hidden + resources["pending_hc_device_bytes"] - limits["pending_hc_bytes"]) + phase)
        overlap = 0 if self.policy is None or self.policy.tier != "ram" else self.policy.overlap_bytes
        if extra > overlap:
            raise RetentionCapacity("retained restore exceeds admitted recovery/deferred-HC/combined GPU overlap")

    def check_health(self):
        with self._condition:
            if self._pid != os.getpid() or self._closed or self._closing:
                raise RuntimeError("retention catalog is retired or belongs to another process")
            failure = self._failure
            worker = self._worker
            ram = self._ram
        if failure is not None:
            raise RuntimeError("retention catalog has an unretired transfer failure") from failure
        if worker is not None and worker.retired.is_set():
            raise RuntimeError("retention CPU owner retired unexpectedly") from worker.failure
        if ram is not None:
            ram.check_health()  # Independent durable store poison precedes diagnostics/worker transport.

    def _latch_failure(self, error):
        with self._condition:
            if self._failure is None:
                self._failure = error
            call = self._actor_call
            if call is not None and not call.completed.is_set():
                call.error = error
                call.completed.set()  # Failure cancellation, never positive DMA retirement.
            self._condition.notify_all()

    def _fatal_once(self, error):
        with self._condition:
            if self._fatal_reported:
                return
            self._fatal_reported = True
        _fatal(self._actor_cohort, error)

    def _clean_retention_error(self, error):
        try:
            with self._condition:
                if self._failure is not None:
                    return False
            if not isinstance(error, RetentionError) or _unretired(error):
                return False
            if self._ram is not None:
                self._ram.check_health()
            return True
        except BaseException as inspection:
            self._latch_failure(error)
            failures = [inspection]
            try:
                self._fatal_once(error)
            except BaseException as notification:
                failures.append(notification)
            raise_failures(error, failures)

    def _registration_failed(self, owner, error, retire):
        def cleanup():
            try:
                retire()
            except BaseException:
                self._latch_failure(error)
                raise
        rollback(owner, error, cleanup)

    def _install_transfer(self, transfer, *, state, source=None, request=None, worker=True):
        """Publish a job and both ownership claims as one known transaction."""
        target = transfer.handle
        with self._condition:
            previous = (target.transfer, target.phase, target.state, self._pending,
                        None if source is None else source.transfer_readers,
                        None if request is None else (request.claimed, request.restore_handle))
            queue = [*self._pending, transfer] if worker else self._pending
            readers = None if source is None else source.transfer_readers + 1
            self._registration_owner = transfer
            try:
                target.transfer, target.phase, target.state = transfer, ("saving" if transfer.direction == "save" else "restoring"), state
                if source is not None:
                    source.transfer_readers = readers
                if request is not None:
                    request.claimed, request.restore_handle = True, target
                self._pending = queue
                if worker:
                    self._condition.notify()
                self._registration_owner = None
            except BaseException as error:
                def retire():
                    target.transfer, target.phase, target.state, self._pending = previous[:4]
                    if source is not None:
                        source.transfer_readers = previous[4]
                    if request is not None:
                        request.claimed, request.restore_handle = previous[5]
                    self._registration_owner = None
                self._registration_failed(transfer, error, retire)

    def _publication_failed(self, owner, error):
        self._registration_owner = owner
        self._latch_failure(error)
        # The exact partial owner remains in the catalog/error transport;
        # profile/error retirement must prove its fence before clearing it.
        rollback(owner, error, lambda: self.close(timeout=0))

    def set_wake(self, wake):
        if not callable(wake):
            raise TypeError("retention wake must be the scheduler's callable")
        with self._condition:
            if self._wake is not None and self._wake is not wake:
                raise RuntimeError("retention catalog already has its scheduler wake owner")
            self._wake = wake

    def bind_decoder(self, decoder):
        if decoder.w is not self.w or tuple(decoder.states) != self.states:
            raise ValueError("retention catalog needs its actual decoder/State inventory")
        self._cohort.multi = weakref.proxy(decoder)
        self._actor_cohort.multi = weakref.proxy(decoder)

    def _wake_local(self):
        with self._condition:
            wake = self._wake
        if wake is not None:
            try:
                wake()
            except BaseException as error:
                self._latch_failure(error)
                raise

    def _wake_failed(self, error):
        self._latch_failure(error)
        self._fatal_once(error)

    def _notify(self):
        if self._wake_relay is not None and threading.current_thread() is self._thread:
            self._wake_relay.notify()
        else:
            self._wake_local()

    @property
    def prefix_entries(self):
        with self._condition:
            return list(self._prefixes)

    @prefix_entries.setter
    def prefix_entries(self, values):
        values = list(values)
        # Existing prefix remember publishes one candidate before its bounded
        # eviction pass; that transient is part of the existing cache contract.
        if len(values) > self.prefix_limit + 1:
            raise ValueError("prefix catalog exceeds its admitted entry bound")
        with self._condition:
            self._prefixes = values

    def reserve(self, reference, *, future_bytes):
        """Before GPU admission; reserve the complete future, not current rows."""
        self.check_health()
        if self._terminal:
            raise RetentionError("retained service is shutting down; no new promise can be admitted")
        if self.policy is None:
            return None
        if type(reference) is not LogicalReference or type(future_bytes) is not int or future_bytes <= 0:
            raise ValueError("retained admission needs an exact reference and full future byte claim")
        if future_bytes < self.minimum_future_bytes:
            raise ValueError("retained claim is smaller than actual full-window/replacement/manifest geometry")
        self._evict_optional_for(future_bytes)
        with self._condition:
            if reference in self._references:
                raise ValueError("logical retained generation is already owned")
            if self._serial >= (1 << 63) - 1:
                raise RetentionCapacity("retained catalog generation exhausted; restart without identity reuse")
            if sum(handle.role == "logical" and not handle.transient for handle in self._handles.values()) >= self.policy.max_sessions:
                raise RetentionCapacity("retained session promises are fully committed")
            if self.policy.tier == "ram" and self._reserved + future_bytes > self.policy.ram_bytes:
                raise RetentionCapacity("complete future RAM backing is fully committed")
            next_serial, next_reserved = self._serial + 1, self._reserved + future_bytes
            self._serial = next_serial  # Consumed even if later acquisition refuses.
            handle = RetainedHandle(self, next_serial, time.monotonic() + self.policy.lifetime_s,
                                    future_bytes, {reference: False})
            prior_reserved = self._reserved
            self._registration_owner = handle
            try:
                self._handles[handle.serial] = handle
                self._references[reference] = handle
                self._reserved = next_reserved
                self._registration_owner = None
            except BaseException as error:
                def retire():
                    if self._references.get(reference) is handle:
                        self._references.pop(reference)
                    if self._handles.get(handle.serial) is handle:
                        self._handles.pop(handle.serial)
                    self._reserved = prior_reserved
                    self._registration_owner = None
                self._registration_failed(handle, error, retire)
        return handle

    @staticmethod
    def _principals(handle):
        return ({reference.principal for reference in handle.owners}
                | ({handle.principal} if handle.internal_owner else set()))

    @staticmethod
    def _prefix_published(handle, principal):
        return (handle.role == "prefix" and handle.internal_owner and handle.principal == principal
                or any(ready and ref.principal == principal for ref, ready in handle.owners.items()))

    def _evict_optional_for(self, future_bytes, *, need_entry=False, drop_all=False):
        """Required claims reclaim only unborrowed optional RAM prefixes."""
        if self.policy is None or self.policy.tier != "ram":
            return
        with self._condition:
            candidates = tuple(sorted((handle for handle in self._handles.values() if handle.role == "prefix"
                and not handle.consumers and not handle.request_holds and not handle.transfer_readers
                and handle.transfer is None), key=lambda handle: handle.serial))
        for handle in candidates:
            with self._condition:
                enough = self._reserved + future_bytes <= self.policy.ram_bytes
                entries = sum(value.role == "prefix" for value in self._handles.values())
                if not drop_all and enough and (not need_entry or entries < self.prefix_limit):
                    return
            self.release_owner(handle)

    def drop_optional_prefixes(self):
        """Coordinated profile/terminal owner may discard only derived caches.

        Actual source readers, transfers and required request/logical claims
        keep their owners. A profile transition must still test has_retained
        after this call; this is not permission to force a protected drain.
        """
        self.check_health()
        if self.policy is not None and self.policy.tier == "ram":
            with self._condition:
                before = len(self._handles)
            self._evict_optional_for(0, drop_all=True)
            with self._condition:
                changed = len(self._handles) != before
            if changed:
                self._notify()
            return changed
        return False

    def _reserve_internal(self, principal, *, role, register):
        self.check_health()
        if self._terminal or self.policy is None or self.policy.tier != "ram":
            raise RetentionCapacity("request suspension and RAM prefix spill require the admitted RAM tier")
        if type(principal) is not str or not principal or role not in ("temporary", "prefix"):
            raise ValueError("internal checkpoint needs an actual principal and bounded role")
        if self._reader_limit is None:
            raise ValueError("internal checkpoints require the actual request admission envelope")
        self._evict_optional_for(self.minimum_future_bytes, need_entry=role == "prefix")
        with self._condition:
            maximum = self._reader_limit if role == "temporary" else self.prefix_limit
            if sum(handle.role == role for handle in self._handles.values()) >= maximum:
                raise RetentionCapacity("internal checkpoint role capacity is fully committed")
            if self._reserved + self.minimum_future_bytes > self.policy.ram_bytes:
                raise RetentionCapacity("complete future RAM backing is fully committed")
            if self._serial >= (1 << 63) - 1:
                raise RetentionCapacity("retained catalog generation exhausted")
            next_reserved = self._reserved + self.minimum_future_bytes
            self._serial += 1
            handle = RetainedHandle(self, self._serial, time.monotonic() + self.policy.lifetime_s,
                self.minimum_future_bytes, {}, pending=False, role=role, internal_owner=True, principal=principal)
            self._registration_owner = handle
            if register is not None:
                try:
                    register(handle)  # Partial owner precedes either catalog publication.
                except BaseException:
                    self._registration_owner = None  # No acquisition has occurred.
                    raise
            prior_reserved = self._reserved
            try:
                self._handles[handle.serial] = handle
                self._reserved = next_reserved
                self._registration_owner = None
            except BaseException as error:
                def retire():
                    if self._handles.get(handle.serial) is handle:
                        self._handles.pop(handle.serial)
                    self._reserved = prior_reserved
                    self._registration_owner = None
                self._registration_failed(handle, error, retire)
            return handle

    def reserve_temporary(self, principal, *, register=None):
        """Actual blocked live request; no client logical ref or permanent promise."""
        return self._reserve_internal(principal, role="temporary", register=register)

    def release_temporary(self, handle):
        self._owned(handle)
        if handle.role != "temporary":
            raise ValueError("temporary retirement cannot drop a logical or prefix owner")
        self.release_owner(handle)

    def capture_prefix(self, stream, *, principal, pending_hc=None, recoveries=(), register=None,
                       from_handle=None, detach_host=None):
        """Best-effort completed current cut, sharing the same allocator/transfer."""
        if from_handle is not None:
            self._owned(from_handle)
            if from_handle.role != "temporary" or from_handle.principal != principal:
                raise ValueError("optional prefix promotion needs the same principal's actual temporary claim")
            self.capture_idle(stream, handle=from_handle, host=from_handle.host,
                              pending_hc=pending_hc, recoveries=recoveries)
            return self._promote_prefix(from_handle, detach_host=detach_host)
        handle = self._reserve_internal(principal, role="prefix", register=register)
        try:
            return self.capture_idle(stream, handle=handle, pending_hc=pending_hc, recoveries=recoveries)
        except BaseException as error:
            self._registration_failed(handle, error, lambda: self.release_owner(handle))

    def _promote_prefix(self, handle, *, detach_host):
        self._evict_optional_for(0, need_entry=True)
        with self._condition:
            if (handle.role != "temporary" or not handle.internal_owner or handle.transfer is not None
                    or handle.consumers or handle.request_holds or handle.transfer_readers
                    or handle.phase != "gpu" or handle.failure is not None):
                raise RetentionCapacity("temporary checkpoint is not at an exclusively completed GPU boundary")
            if sum(value.role == "prefix" for value in self._handles.values()) >= self.prefix_limit:
                raise RetentionCapacity("optional prefix role capacity is fully committed")
            host = handle.host
            handoff = dict(handle.ready_handoff, host=None)
            expiry = time.monotonic() + self.policy.lifetime_s
        if host is not None:
            if not callable(detach_host):
                raise ValueError("prefix promotion requires the actual CPU host-owner handoff")
            detach_host(host)  # Caller journals it; no retirement or socket close is inferred.
        try:
            with self._condition:
                handle.role, handle.expires = "prefix", expiry
                handle.host, handle.ready_handoff = None, handoff
            return handle
        except BaseException as error:
            self._publication_failed(handle, error)

    def reserve_eligibility(self):
        """Leader's nonblocking quota predicate before a fresh GPU claim."""
        self.check_health()
        if self.policy is None:
            return None
        with self._condition:
            if self._terminal:
                raise RetentionError("retained service is shutting down")
            if sum(handle.role == "logical" and not handle.transient for handle in self._handles.values()) >= self.policy.max_sessions:
                return "tier_unavailable"
            if self.policy.tier == "ram":
                reclaimable = sum(handle.future_bytes for handle in self._handles.values()
                    if handle.role == "prefix" and not handle.consumers and not handle.request_holds
                    and not handle.transfer_readers and handle.transfer is None)
                if self._reserved - reclaimable + self.minimum_future_bytes > self.policy.ram_bytes:
                    return "tier_unavailable"
        return None

    def reserve_pending(self, reference, *, future_bytes):
        """One logical API operation may run several fresh backend subruns."""
        self.check_health()
        with self._condition:
            handle = self._references.get(reference)
            if handle is not None:
                if (not handle.pending or any(handle.owners.values())
                        or handle.future_bytes != future_bytes):
                    raise RetentionError("published compute generation cannot be reused as a pending target")
                return handle  # Preserve the original absolute lifetime and quota.
        return self.reserve(reference, future_bytes=future_bytes)

    def replace_pending(self, handle, *, retire_host=None):
        """Fresh backend job under the SAME uncommitted logical generation.

        This never replaces a published version. The caller owns FIFO order
        and reuses the returned already-owned GPU window, or claims one if
        None. Previous optional GPU prefix entries remain under normal cache
        policy; generation bookkeeping is rebuilt for the new backend job.
        """
        self.check_health()
        with self._condition:
            self._owned(handle)
            if (not handle.pending or any(handle.owners.values()) or handle.consumers or handle.request_holds
                    or handle.transfer_readers or handle.transfer is not None or handle.retiring_keys
                    or handle.phase not in ("reserved", "gpu", "ram") or handle.failure is not None
                    or any(reference in self._logical_events for reference in handle.owners)):
                raise RetentionCapacity("pending target still has publication/readers/backend work")
            host = handle.host
        if host is not None:
            if not callable(retire_host):
                raise ValueError("pending backend replacement requires its actual host-owner retirement")
            retire_host(host)  # Outside catalog locks; failure retains the old owner.
        with self._condition:
            release = (self._release_plan(handle.state, excluding=handle, keep_prefix=True)
                       if self.policy.tier == "ram" and handle.state is not None else None)
            empty_retire = []
            handle.retiring_keys = list(handle.keys)
        try:
            for index, key in enumerate(handle.retiring_keys):
                self._ram.release(key)
                handle.retiring_keys[index] = None
        except BaseException as error:
            self._latch_failure(error)
            raise
        try:
            with self._condition:
                handle.keys, handle.manifest, handle.retiring_keys = (), None, empty_retire
                handle.inventory = handle.ready_handoff = handle.canonical_ids = handle.host = None
                handle.clean_state = handle.clean_generation = None
                handle.phase = "reserved"
                if self.policy.tier == "ram":
                    handle.state = None
                    self._apply_release(release)
                return handle.state
        except BaseException as error:
            self._publication_failed(handle, error)

    def _owned(self, handle):
        if type(handle) is not RetainedHandle or handle.catalog is not self or self._handles.get(handle.serial) is not handle:
            raise ValueError("retained handle is not issued by this current catalog")

    def borrow(self, handle, *, continuation="same-request", target_handle=None, register=None):
        self.check_health()
        if self._terminal:
            raise RetentionError("retained service is shutting down; no new consumer can be queued")
        with self._condition:
            self._owned(handle)
            if (not handle.owners and not handle.internal_owner or self.world == 1 and not handle.pending
                    and handle.role != "temporary" and time.monotonic() >= handle.expires):
                raise RetentionExpired("retained generation is expired or logically unowned")
            if handle.failure is not None:
                raise handle.failure
            if continuation not in ("same-request", "new-turn"):
                raise ValueError("retained consumer needs its immutable continuation mode")
            if continuation == "same-request":
                if target_handle is not None or handle.consumers or handle.request_holds:
                    raise RetentionCapacity("retained sequence already has a consumer; mutation is exclusive")
            else:
                self._owned(target_handle)
                if (target_handle is handle or target_handle.phase != "reserved"
                        or any(request.continuation == "same-request" for request in handle.consumers)
                        or target_handle.request_holds
                        or not target_handle.transient and self._principals(target_handle) != self._principals(handle)):
                    raise RetentionCapacity("new-turn consumer needs its independent same-principal target promise")
            request = RetainedRequest(self, handle, continuation=continuation, target_handle=target_handle)
            if register is not None:
                register(request)  # Caller journal precedes either owned set.
            self._registration_owner = request
            try:
                handle.consumers.add(request)
                if target_handle is not None:
                    target_handle.request_holds.add(request)
                request.closed = False
                self._registration_owner = None
            except BaseException as error:
                def retire():
                    handle.consumers.discard(request)
                    if target_handle is not None:
                        target_handle.request_holds.discard(request)
                    request.closed = True
                    self._registration_owner = None
                self._registration_failed(request, error, retire)
            return request

    def best_prefix(self, prompt, *, principal):
        """Qualified-input lookup needs no logical/conversation reference."""
        self.check_health()
        if self.rank != 0:
            raise RuntimeError("only the admission leader selects a rendered prefix")
        with self._condition:
            candidates = tuple(handle for handle in self._handles.values()
                if not handle.transient and handle.phase in ("gpu", "ram") and handle.failure is None
                and not handle.expired and self._prefix_published(handle, principal)
                and not any(request.continuation == "same-request" for request in handle.consumers))
        return max((handle for handle in candidates if self.prefix_matches(handle, prompt, principal=principal)),
                   key=lambda handle: handle.canonical_ids.numel(), default=None)

    def borrow_prefix(self, handle, *, principal, register=None):
        """Disposable request uses input/GPU claims, never a new RAM promise."""
        self.check_health()
        with self._condition:
            self._owned(handle)
            if not self._prefix_published(handle, principal):
                raise RetentionMismatch("RAM prefix is not published for the current principal")
            if self._reader_limit is None:
                raise ValueError("RAM prefix borrowing must bind the existing admission envelope first")
            if sum(target.transient for target in self._handles.values()) >= self._reader_limit:
                raise RetentionCapacity("RAM prefix reader admission is fully committed")
            if self._serial >= (1 << 63) - 1:
                raise RetentionCapacity("RAM prefix generation exhausted")
            self._serial += 1
            target = RetainedHandle(self, self._serial, handle.expires, 0, {}, transient=True, pending=False)
            self._registration_owner = target
            try:
                self._handles[target.serial] = target
                request = self.borrow(handle, continuation="new-turn", target_handle=target, register=register)
                self._registration_owner = None
            except BaseException as error:
                def retire():
                    if target.request_holds:
                        raise RuntimeError("unretired transient prefix consumer remains owned")
                    if self._handles.get(target.serial) is target:
                        self._handles.pop(target.serial)
                    self._registration_owner = None
                self._registration_failed(target, error, retire)
            return request

    def lookup(self, reference):
        with self._condition:
            handle = self._references.get(reference)
            return handle if handle is not None and handle.owners.get(reference) is True else None

    def by_serial(self, serial):
        with self._condition:
            if type(serial) is not int or serial not in self._handles:
                raise RetentionExpired("retained actor command refers to an unavailable issued generation")
            return self._handles[serial]

    def suspendable_handles(self, *, active_states):
        """Stable enumeration; the caller owns measured victim selection policy."""
        self.check_health()
        if self.rank != 0:
            raise RuntimeError("only the admission leader chooses idle suspension candidates")
        if type(active_states) not in (tuple, list):
            raise ValueError("suspension selection needs the actual active/filling State inventory")
        with self._condition:
            if self.policy is None or self.policy.tier != "ram":
                return ()
            return tuple(sorted((handle for handle in self._handles.values()
                if not handle.transient and handle.phase == "gpu" and handle.state is not None
                and all(handle.state is not active for active in active_states)
                and handle.transfer is None and handle.failed_transfer is None and not handle.transfer_readers
                and not handle.request_holds and not any(request.claimed for request in handle.consumers)),
                key=lambda handle: handle.serial))

    def associate(self, handle, reference, *, committed=False):
        with self._condition:
            self._owned(handle)
            if type(committed) is not bool or type(reference) is not LogicalReference or reference in self._references:
                raise ValueError("retained logical alias is invalid or already owned")
            if any(owner.principal != reference.principal for owner in handle.owners):
                raise ValueError("retained logical aliases cannot cross authenticated principals")
            # Standard Response/conversation aliases share exactly one manifest.
            if len(handle.owners) >= 2:
                raise ValueError("retained generation already has its response/conversation aliases")
            self._registration_owner = handle
            try:
                handle.owners[reference] = committed
                self._references[reference] = handle
                self._registration_owner = None
            except BaseException as error:
                def retire():
                    if self._references.get(reference) is handle:
                        self._references.pop(reference)
                    handle.owners.pop(reference, None)
                    self._registration_owner = None
                self._registration_failed(handle, error, retire)

    def logical_reference_committed(self, principal, logical_sequence_id, *, generation):
        reference = LogicalReference(principal, logical_sequence_id, generation)
        self._logical_event(reference, "committed", None)

    def logical_reference_unavailable(self, principal, logical_sequence_id, *, generation, reason):
        if type(reason) is not str or not reason:
            raise ValueError("retained logical retirement requires its actual reason")
        reference = LogicalReference(principal, logical_sequence_id, generation)
        self._logical_event(reference, "unavailable", reason)

    def _logical_event(self, reference, operation, reason):
        self.check_health()
        if self.rank != 0:
            raise RuntimeError("logical API events belong to the authenticated admission leader")
        with self._condition:
            if reference not in self._references:
                return
            old = self._logical_events.get(reference)
            if old is not None and (old["operation"] == "unavailable" or old["operation"] == operation):
                return
            handle = self._references[reference]
            self._logical_events[reference] = {"principal": reference.principal, "sequence": reference.sequence,
                "generation": reference.generation, "operation": operation, "reason": reason,
                "expired": handle.expired or time.monotonic() >= handle.expires}
        self._notify()

    def logical_events(self):
        if self.rank != 0:
            raise RuntimeError("only the admission leader publishes logical lifetime events")
        with self._condition:
            return tuple(dict(event) for event in self._logical_events.values())

    def apply_logical_events(self, events):
        """Actor applies an existing TP command after all ranks receive it."""
        self.check_health()
        maximum = 0 if self.policy is None else 2 * self.policy.max_sessions
        if type(events) not in (tuple, list) or len(events) > maximum:
            raise ValueError("retained logical event batch exceeds owned alias capacity")
        parsed = []
        for event in events:
            if (type(event) is not dict or event.keys() != {"principal", "sequence", "generation", "operation", "reason", "expired"}
                    or event["operation"] not in ("committed", "unavailable")
                    or type(event["expired"]) is not bool
                    or event["operation"] == "committed" and event["reason"] is not None
                    or event["operation"] == "unavailable" and (type(event["reason"]) is not str or not event["reason"])):
                raise ValueError("retained logical event schema is invalid")
            parsed.append((LogicalReference(event["principal"], event["sequence"], event["generation"]), event))
        reclaimed = []
        with self._condition:
            actions = []
            for reference, event in parsed:
                handle = self._references.get(reference)
                prior = []
                if handle is not None and event["operation"] == "committed" and not event["expired"]:
                    if (handle.phase not in ("gpu", "ram") or handle.phase == "gpu" and handle.inventory is None
                            and handle.ready_handoff.get("continuation") != "same-request"):
                        raise RuntimeError("logical compute publication has no complete current GPU/RAM checkpoint")
                    if reference.sequence.startswith("conversation:"):
                        prior = [old for old in self._references if old.principal == reference.principal
                                 and old.sequence == reference.sequence and old.generation < reference.generation]
                actions.append((reference, event, handle, prior))
            for reference, event, handle, prior in actions:
                if handle is not None:
                    handle.pending = False
                    if event["operation"] == "committed" and not event["expired"]:
                        handle.owners[reference] = True
                        # Conversation versions are internal authority updates;
                        # preserve every independently addressable response alias.
                        if reference.sequence.startswith("conversation:"):
                            for old in prior:
                                previous = self._references.pop(old, None)
                                if previous is None:
                                    continue
                                previous.owners.pop(old)
                                reclaimed.append(previous)
                    else:
                        self._references.pop(reference)
                        handle.owners.pop(reference)
                        reclaimed.append(handle)
                if self._logical_events.get(reference) == event:
                    self._logical_events.pop(reference)
        for handle in reclaimed:
            self._reclaim(handle)

    def _release_consumer(self, request):
        with self._condition:
            request._check()
            if request.source_active:
                request.handle.consumers.remove(request)
                request.source_active = False
            if request.target_handle is not None:
                request.target_handle.request_holds.remove(request)
            request.closed = True
            request.handoff = None
        if self._handles.get(request.handle.serial) is request.handle:
            self._reclaim(request.handle)
        if request.target_handle is not None:
            self._reclaim(request.target_handle)
        self._notify()

    def release_owner(self, handle):
        with self._condition:
            self._owned(handle)
            for ref in tuple(handle.owners):
                self._references.pop(ref)
            handle.owners.clear()
            handle.internal_owner = False
            handle.pending = False
        self._reclaim(handle)

    def _reclaim(self, handle):
        with self._condition:
            if (handle.pending or handle.owners or handle.internal_owner or handle.consumers or handle.request_holds or handle.transfer_readers
                    or handle.transfer is not None or handle.retiring_keys or handle.failed_transfer is not None):
                return
            self._owned(handle)
            handle.retiring_keys = list(handle.keys)
        try:
            for index, key in enumerate(handle.retiring_keys):
                self._ram.release(key)
                handle.retiring_keys[index] = None
            with self._condition:
                release = self._release_plan(handle.state, excluding=handle,
                    keep_prefix=handle.transient) if handle.state is not None else None
                next_reserved = self._reserved - handle.future_bytes
                self._registration_owner = handle
                self._handles.pop(handle.serial)
                self._reserved = next_reserved
                self._apply_release(release)
                handle.keys, handle.retiring_keys = (), []
                handle.state = handle.inventory = handle.host = handle.manifest = handle.ready_handoff = None
                handle.canonical_ids = None
                handle.clean_state = handle.clean_generation = None
                self._registration_owner = None
        except BaseException as error:
            self._publication_failed(handle, error)

    def _release_plan(self, state, *, excluding=None, keep_prefix=False):
        if any(handle is not excluding and handle.state is state for handle in self._handles.values()):
            return None
        if keep_prefix and any(entry[1] is state for entry in self._prefixes):
            return None  # Existing optional cache remains its sole owner.
        prefixes = [entry for entry in self._prefixes if entry[1] is not state]
        released = dict(self._released_states)
        released[id(state)] = state
        return prefixes, released

    def _apply_release(self, plan):
        if plan is not None:
            self._prefixes, self._released_states = plan

    def protected_states(self):
        with self._condition:
            values = [handle.state for handle in self._handles.values() if handle.state is not None]
            owner = self._registration_owner
            if type(owner) is RetainedHandle and owner.state is not None:
                values.append(owner.state)
            elif type(owner) is _Transfer:
                if owner.handle.state is not None:
                    values.append(owner.handle.state)
                if owner.plan is not None:
                    values.append(owner.plan.state)
            return tuple({id(state): state for state in values}.values())

    def migration_blocked(self, state):
        with self._condition:
            return any(handle.state is state and (handle.transfer is not None or handle.transfer_readers or state.mtp_drafted)
                       for handle in self._handles.values())

    def take_released_states(self):
        """Actor removes these from derived caches before returning their slots."""
        with self._condition:
            result = tuple(self._released_states.values())
            self._released_states.clear()
            return result

    def capture_idle(self, stream, *, pending_hc=None, recoveries=(), host=None, handle=None):
        self.check_health()
        if self._terminal:
            return None  # Explicit service closure ends process-lifetime promises.
        handle = getattr(stream, "retention_handle", None) if handle is None else handle
        if handle is None:
            return None
        self._owned(handle)
        if not any(stream.st is state for state in self.states):
            raise ValueError("retained source is outside its admitted State pool")
        if handle.transient:
            return None
        if handle.transfer is not None or handle.phase not in ("reserved", "gpu"):
            raise RuntimeError("retained capture is not at its exclusively owned GPU boundary")
        inventory = state_inventory(stream.st, stream=stream, pending_hc=pending_hc,
                                    recoveries=recoveries, recovery_versions=self.recovery_versions)
        if inventory.future_bytes > handle.future_bytes:
            raise MemoryError("retained State exceeds its complete admitted future promise")
        # The final accepted HC may be a view into the shared execution frame.
        # This bounded row must outlive that frame before another request runs.
        parts = tuple(TensorPart(part.name, part.tensor.clone(), part.category, part.semantic)
                      if part.name == "method.mtp.hc" and stream.st.mtp_boundary is None else part
                      for part in inventory.parts)
        inventory = replace(inventory, parts=parts)
        hidden = next((part.tensor for part in parts if part.name == "method.mtp.hc"), None)
        handle.canonical_ids = next((part.tensor for part in parts if part.name == "state.canonical_ids"), None)
        handle.state, handle.inventory, handle.host, handle.phase = stream.st, inventory, host, "gpu"
        handle.ready_handoff = {"ready": True, "pending_hc": hidden, "recoveries": tuple(recoveries), "host": host}
        return handle

    def prefix_matches(self, handle, prompt, *, principal, continuation="new-turn"):
        """Leader compares actual rendered IDs; geometry/placeholder IDs alone do not authorize media reuse."""
        self._owned(handle)
        if self.rank != 0 or type(prompt) is not list or type(principal) is not str:
            raise ValueError("retained prefix selection needs leader-owned rendered input/principal")
        if principal not in self._principals(handle):
            return False
        ids = handle.canonical_ids
        if (continuation not in ("same-request", "new-turn") or ids is None or len(prompt) < ids.numel()
                or continuation == "new-turn" and len(prompt) == ids.numel()):
            return False
        # The saved profile/media binding is independently checked against the
        # CURRENT admitted State before any restore target allocation/write.
        return all(type(token) is int and token == int(saved) for token, saved in zip(prompt, ids.numpy(), strict=False))

    def suspend(self, handle):
        """Idle GPU only; caller keeps the slot unavailable until saved()."""
        self.check_health()
        self._owned(handle)
        if (handle.transient or self.policy.tier != "ram" or handle.phase != "gpu"
                or any(request.claimed for request in handle.consumers)):
            raise ValueError("only exclusively idle RAM-backed generations can suspend")
        generation = _next_generation(handle.transfer_generation, "transfer")
        # Completed physical migration may have changed page offsets since the
        # idle capture. Rebind actual source views before acquiring DMA leases.
        if handle.inventory is not None:
            pending = (None if handle.state.mtp_boundary is not None else handle.ready_handoff["pending_hc"])
            fresh = rebind_state_inventory(handle.state, handle.inventory, pending_hc=pending,
                recoveries=handle.ready_handoff["recoveries"], recovery_versions=self.recovery_versions)
            handle.inventory = fresh
            handle.canonical_ids = next((part.tensor for part in fresh.parts if part.name == "state.canonical_ids"), None)
        if handle.inventory is None:
            # A restored, never-committed window is still an unchanged copy of
            # its retained RAM authority. Do not copy it out merely to free it.
            metadata = None if handle.manifest is None else handle.manifest[self.rank]["metadata"]
            state = handle.state
            if (metadata is None or handle.ready_handoff.get("continuation") != "same-request"
                    or state.target_tentative_length != state.pos
                    or (state.pos, state.mtp_len, state.mtp_drafted) !=
                       (metadata["pos"], metadata["mtp_len"], metadata["mtp_drafted"])):
                raise RuntimeError("retained GPU window changed without its quiescent capture")
            handle.transfer_generation = generation
            transfer = _Transfer(handle, "save", generation=handle.transfer_generation)
            with self._condition:
                handle.transfer, handle.phase = transfer, "saving"
            transfer.retired.set()
            self._notify()
            return transfer
        handle.transfer_generation = generation
        transfer = _Transfer(handle, "save", generation=handle.transfer_generation,
            byte_sources=tuple(part.tensor.detach().view(torch.uint8).reshape(-1) for part in handle.inventory.parts),
            specs=handle.inventory.specs)
        self._install_transfer(transfer, state=handle.state)
        return transfer

    def saved(self, handle):
        self.check_health()
        self._owned(handle)
        self._settle(handle)
        return handle.phase == "ram"

    def claim_resume(self, request, *, state, stream=None, continuation="new-turn", ticket=None):
        self.check_health()
        if self._terminal:
            raise RetentionError("retained service is shutting down; no restore can be admitted")
        request._check()
        if request.handoff is not None:
            return request.handoff
        if request.restore_handle is not None:
            return self.restored(request)
        handle = request.handle
        self._settle(handle)
        if not any(state is owned for owned in self.states):
            raise ValueError("retained restore destination is outside the admitted State pool")
        allowed = handle if continuation == "same-request" else request.target_handle
        if any(owner is not allowed and owner.state is state for owner in self._handles.values()):
            raise RuntimeError("retained GPU claim overlaps another required sequence window")
        if handle.phase not in ("gpu", "ram"):
            return None
        source = (handle.inventory.metadata if handle.inventory is not None else handle.manifest[self.rank]["metadata"])
        if stream is None:
            raise RetentionError("retained resume requires its actual current request and canonical rendered input")
        if continuation == "new-turn" and len(stream.prompt) <= source["pos"]:
            raise RetentionMismatch("new-turn full-prompt candidate lacks fresh target logits or an earlier common cut")
        _ids, fingerprint = canonical_tokens(stream, source["pos"], new_turn=continuation == "new-turn")
        if fingerprint != source["canonical_sha256"]:
            raise RetentionMismatch("retained canonical prefix differs from the actual current rendered input")
        if continuation == "same-request" and request_record(stream) != source["request"]:
            raise RetentionMismatch("same-request continuation changed its original input/limit/method contract")
        if (continuation == "new-turn" and state.mtp_hidden_width and stream.draft
                and source["mtp_len"] - source["mtp_drafted"] == source["pos"] and source["pos"]):
            position, draws = source["pos"], source["method"]["draws"]
            old_input = (draws["pending_anchor"] if draws is not None and draws["pending_anchor"] is not None else
                         source["next_prompt_input"])
            if old_input is None or stream.prompt[position] != old_input:
                raise RetentionMismatch("retained MTP lookahead differs from the current next input; rebuild from a valid cut")
        if (_json(source["kv_identity"]) != _json(state.kv_identity) or source["limit"] != state.limit
                or source["media_profile"] != state.media_profile_identity or source["media_identity"] != state.media_identity):
            raise RetentionMismatch("retained prefix does not match the current admitted numerical/media context")
        if continuation == "new-turn":
            state.media_mtp_enabled = bool(state.mtp_hidden_width and stream.draft)
            if getattr(state, "dflash2", None) is not None:
                state.dflash2.enabled = stream.draft
        method_source = new_turn_metadata(state, source, stream) if continuation == "new-turn" else source
        source_specs = (handle.inventory.specs if handle.inventory is not None else
            tuple(PartSpec(**dict(spec, shape=tuple(spec["shape"]), semantic=tuple(spec["semantic"])))
                  for spec in handle.manifest[self.rank]["specs"]))
        try:
            validate_method_restore(state, method_source["method"], tuple(spec for spec in source_specs
                if spec.name.startswith("method.") and (continuation != "new-turn" or
                    new_turn_uses_part(state, spec, stream))), stream=stream, continuation=continuation)
        except ValueError as error:
            if continuation == "new-turn":
                raise RetentionMismatch("retained method prefix cannot initialize the current new turn") from error
            raise
        if handle.phase == "gpu":
            if continuation == "same-request":
                if state is not handle.state:
                    raise ValueError("same-request GPU continuation needs its exact resident State")
                request.claimed, request.handoff = True, handle.ready_handoff
                return request.handoff
            if state is handle.state:
                raise ValueError("a new turn must fork its protected GPU source into another full-window slot")
            # Reuse the established all-rank GPU prefix relocation contract.
            # The protected source remains byte-authoritative for its owners.
            source = handle.state
            target = request.target_handle
            generation = _next_generation(target.transfer_generation, "transfer")
            resources = None
            if self._device_resources is not None:
                boundary = source.mtp_boundary if stream.draft else None
                resources = {"recovery_device_bytes": self._device_resources["snapshot_bytes"],
                    "pending_hc_device_bytes": (0 if boundary is None else
                        boundary.hidden.nbytes + boundary.coordinates.nbytes), "phase_device_bytes": 0}
                self._check_device_resources(state, resources)
            elif state.pos_dev.is_cuda:
                raise RetentionCapacity("retained GPU prefix has no bound per-rank header resources")
            target.transfer_generation = generation
            transfer = _Transfer(target, "fork", generation=target.transfer_generation, source_handle=handle,
                                 request=request, resources=resources)
            self._install_transfer(transfer, state=state, source=handle, request=request, worker=False)
            try:
                snapshot = new_turn_gpu_snapshot(source, state, stream)
                transfer.owner_payload = snapshot
                relocated = state.copy_prefix(source, source.pos, snapshot["mtp_len"], copy_dflash=stream.draft,
                    **({"indexer_snapshot": snapshot["compact_indexer"],
                        "poison_control": None if self._cohort.multi is None else self._cohort.multi.poison_control}
                       if source.compact_indexer is not None else {}))
                if relocated is not None:
                    snapshot["compact_indexer"] = relocated
                state.restore(snapshot)
                pending_hc = (None if not stream.draft else state.mtp_boundary.hidden
                              if state.mtp_boundary is not None else handle.ready_handoff["pending_hc"])
                transfer.handoff = {**handle.ready_handoff, "pending_hc": pending_hc, "recoveries": (), "forked": True,
                                    "host": None, "continuation": "new-turn", "canonical_ids": handle.canonical_ids}
                if state.pos_dev.is_cuda:
                    transfer.event = torch.cuda.Event()
                    transfer.event.record(torch.cuda.current_stream(state.pos_dev.device))
                else:
                    transfer.retired.set()
            except BaseException as error:
                self._latch_failure(error)
                self._fatal_once(error)
                raise
            self._notify()
            return self.restored(request)
        if handle.phase != "ram":
            return None
        target = handle if continuation == "same-request" else request.target_handle
        generation = _next_generation(target.transfer_generation, "transfer")
        records = tuple((row["metadata"], tuple(PartSpec(**dict(spec, shape=tuple(spec["shape"]),
            semantic=tuple(spec["semantic"]))) for spec in row["specs"])) for row in handle.manifest)
        extra = method_restore_resources(state, method_source["method"])
        if (extra["device_bytes"] > self.policy.overlap_bytes or
                2 * self.world * self.policy.chunk_bytes + self.world * extra["host_bytes"] > self.policy.progress_bytes):
            raise RetentionError("retained media coordinate restore exceeds admitted transfer progress/overlap")
        resources = state_restore_resources(state, records, stream=stream, continuation=continuation)
        self._check_device_resources(state, resources)
        plan = state_restore_plan(state, records, stream=stream, continuation=continuation,
                                  recovery_versions=self.recovery_versions)
        if ticket is not None and getattr(stream, "admission_ticket", ticket) != ticket:
            raise ValueError("retained resume must preserve its new admission ticket")
        target.transfer_generation = generation
        transfer = _Transfer(target, "restore", generation=target.transfer_generation, plan=plan,
                             source_handle=handle, request=request, resources=resources)
        self._install_transfer(transfer, state=state, source=handle, request=request)
        return None

    def restored(self, request):
        self.check_health()
        request._check()
        handle = request.restore_handle or request.handle
        self._settle(handle)
        transfer = handle.transfer
        if transfer is not None:
            return None
        if handle.phase == "gpu":
            request.handoff = handle.ready_handoff
            return request.handoff
        return None

    def _settle(self, handle, *, coordinated=False):
        if self.world > 1 and not coordinated:
            return
        transfer = handle.transfer
        if transfer is None or not transfer.retired.is_set():
            return
        if transfer.failure is not None:
            if self._clean_retention_error(transfer.failure):
                source = transfer.source_handle
                following = None if source is None else source.transfer_readers - 1
                with self._condition:
                    handle.phase, handle.failure, handle.failed_transfer = "failed", transfer.failure, transfer
                    if source is not None:
                        source.transfer_readers = following
                        if isinstance(transfer.failure, RetentionCorrupt):
                            source.failure, source.phase = transfer.failure, "failed"
                    handle.transfer = None
                if source is not None:
                    self._reclaim(source)
            else:
                self._latch_failure(transfer.failure)
                self._fatal_once(transfer.failure)
            raise transfer.failure
        source = transfer.source_handle
        direction = transfer.direction
        continuation = "new-turn" if direction == "fork" else (None if direction == "save" else transfer.plan.continuation)
        # Prepare fallible metadata before acquiring new refs or publishing a
        # different authority. A preparation failure leaves an ordinary retry.
        if direction == "save":
            with self._condition:
                release = self._release_plan(handle.state, excluding=handle)
            handoff = clean = following = None
        else:
            host = handle.host if continuation == "same-request" else None
            handoff = {**transfer.handoff, "host": host, "continuation": continuation}
            clean = (handle.state.slot_generation, handle.state.history_generation)
            following = source.transfer_readers - 1
            release = None
        if direction == "restore" and source is not handle and not handle.transient:
            transfer.retained_keys = [None] * len(source.keys)
            try:
                for index, key in enumerate(source.keys):
                    self._ram.retain_into(key, transfer.retained_keys, index)
            except BaseException as error:
                def retire():
                    for index, key in enumerate(transfer.retained_keys):
                        if key is not None:
                            self._ram.release(key)
                            transfer.retained_keys[index] = None
                self._registration_failed(transfer, error, retire)
        self._registration_owner = transfer
        try:
            with self._condition:
                if direction == "save":
                    handle.phase = "ram"
                    handle.state = handle.inventory = None
                    self._apply_release(release)
                    handle.ready_handoff = None
                    handle.clean_state = handle.clean_generation = None
                else:
                    if direction == "restore" and source is not handle and not handle.transient:
                        handle.manifest, handle.keys = source.manifest, source.keys
                    handle.phase, handle.ready_handoff = "gpu", handoff
                    handle.canonical_ids = transfer.handoff["canonical_ids"]
                    handle.clean_state, handle.clean_generation = handle.state, clean
                    source.transfer_readers = following
                    request = transfer.request
                    if request is not None and continuation == "new-turn" and request.source_active:
                        source.consumers.discard(request)
                        request.source_active = False
                handle.transfer = None
                self._registration_owner = None
        except BaseException as error:
            self._publication_failed(transfer, error)
        if source is not None:
            self._reclaim(source)
        transfer.byte_sources = ()
        transfer.plan = transfer.handoff = transfer.source_handle = transfer.request = transfer.owner_payload = None
        transfer.retained_keys = transfer.event = None

    def retire_failed_destination(self, request, *, fenced):
        """After actual actor cleanup/fence, retire only a known pre-write refusal."""
        self.check_health()
        request._check()
        if fenced is not True:
            raise ValueError("failed retained destination requires its actual device fence")
        target = request.restore_handle
        if target is None:
            return ()  # Pure preflight mismatch acquired no catalog destination.
        with self._condition:
            transfer = target.failed_transfer
            if (target.phase != "failed" or not isinstance(target.failure, RetentionCorrupt)
                    or transfer is None or not transfer.retired.is_set() or target.transfer is not None):
                raise RuntimeError("retained destination has no completed pre-write error retirement proof")
            state = target.state
            release = None if state is None else self._release_plan(state, excluding=target)
            target.state = None
            self._apply_release(release)
            transfer.plan = transfer.handoff = transfer.source_handle = transfer.request = None
            transfer.byte_sources = ()
            target.failed_transfer = None
            return () if state is None else (state,)

    def retire_backend(self, handle, *, state, fenced, error, detach_host=None):
        """Retire a reset failed UNCOMMITTED target, preserving its RAM claim.

        The existing actor must close its consumer, finish request cleanup and
        prove the device fence first. This hook never resets State, destroys a
        published parent, refunds a pending API promise or fabricates replay.
        """
        self.check_health()
        self._owned(handle)
        if fenced is not True or not isinstance(error, BaseException):
            raise ValueError("backend retirement requires its actual fence and original request error")
        with self._condition:
            if handle.phase == "failed" and handle.state is None and handle.failure is error:
                return ()
            if (not (handle.role == "temporary" or handle.role == "logical" and handle.pending
                    and not any(handle.owners.values())) or handle.consumers or handle.request_holds
                    or handle.transfer_readers or handle.transfer is not None or handle.failed_transfer is not None
                    or handle.retiring_keys):
                raise RuntimeError("backend retirement still has published ownership/readers or transfer state")
            if (not any(state is value for value in self.states)
                    or handle.state is not None and handle.state is not state
                    or any(owner is not handle and owner.state is state for owner in self._handles.values())):
                raise ValueError("backend retirement does not own this exact target State")
            owned = handle.state is state
            host = handle.host
        state.kv_check()
        if (state.pos != 0 or state.target_tentative_length != 0 or state.mtp_len != 0 or state.mtp_drafted != 0
                or state.media_input_span is not None or state._resize_error is not None):
            raise RuntimeError("backend target has no completed healthy reset proof")
        if state.dflash2 is not None:
            state.dflash2.check()  # A live/pending/poisoned feature owner is not retired by a flag.
        with self._condition:
            release = self._release_plan(state, excluding=handle) if owned else None
        if host is not None:
            if not callable(detach_host):
                raise ValueError("failed backend retirement requires the actual CPU host-owner handoff")
            detach_host(host)
        try:
            with self._condition:
                handle.state = handle.inventory = handle.ready_handoff = handle.canonical_ids = None
                handle.host = handle.clean_state = handle.clean_generation = None
                handle.failure, handle.phase = error, "failed"
                self._apply_release(release)
            return (state,) if owned else ()
        except BaseException as failure:
            self._publication_failed(handle, failure)

    def completed_transfers(self):
        with self._condition:
            return tuple((handle.serial, handle.transfer.generation) for handle in self._handles.values()
                         if handle.transfer is not None and handle.transfer.retired.is_set())

    def settle_ready(self):
        """ALL-rank actor boundary; publish only common completed generations."""
        self.check_health()
        if self.world == 1:
            self.progress()
            return
        self._settlement_generation = _next_generation(self._settlement_generation, "settlement")
        channel = f"tf_retention/{self.identity}/settle/{self._settlement_generation}"
        stages = ("ready", "validate", "published")
        rows = _phase(self._actor_cohort, channel, "ready", lambda: {
            "complete": [list(value) for value in self.completed_transfers()]}, may_refuse=False)
        ready = set.intersection(*(set(tuple(item) for item in row["complete"]) for row in rows))

        def validate():
            with self._condition:
                for serial, generation in ready:
                    handle = self._handles.get(serial)
                    if (handle is None or handle.transfer is None or handle.transfer.generation != generation
                            or not handle.transfer.retired.is_set()):
                        raise RuntimeError("retained completion changed before all-rank publication")
            return True

        _phase(self._actor_cohort, channel, "validate", validate, may_refuse=False)
        local_error = None

        def publish():
            nonlocal local_error
            try:
                for serial, _generation in sorted(ready):
                    try:
                        self._settle(self._handles[serial], coordinated=True)
                    except RetentionError as error:
                        if not self._clean_retention_error(error):
                            raise
                        if local_error is None:
                            local_error = error
            except BaseException as error:
                self._latch_failure(error)
                raise
            return {"session_error": local_error is not None}

        _phase(self._actor_cohort, channel, "published", publish, may_refuse=False)
        _retire_store(self._actor_cohort, channel, stages=stages)
        if local_error is not None:
            raise local_error

    def progress(self, *, idle=False):
        """Service device work on the actor; idle may fence one issued chunk.

        The scheduler sets idle only when no GPU sequence is runnable. This
        conservative source candidate has no claimed V100 arrival-latency
        bound. It never waits for Store/disk/worker IO on the actor.
        """
        if type(idle) is not bool:
            raise TypeError("retention progress idle state must be explicit boolean")
        self.check_health()
        if self.world == 1:
            self.apply_logical_events(self.logical_events())
        # This is called only by the existing GPU coordinator between model
        # operations. The CPU worker requests, but never issues, device work.
        with self._condition:
            call = self._actor_call
        if call is not None and not call.issued.is_set():
            try:
                call.result = call.operation()
                if call.device.type == "cuda":
                    call.event = torch.cuda.Event()
                    call.event.record(torch.cuda.current_stream(call.device))
            except BaseException as error:
                call.error = error
            finally:
                call.issued.set()
        if call is not None and call.issued.is_set() and not call.completed.is_set():
            try:
                complete = call.error is not None or call.event is None or call.event.query()
                if not complete and idle:
                    call.event.synchronize()  # Outside every catalog/decoder lock.
                    complete = True
                if complete:
                    call.completed.set()
            except BaseException as error:
                call.error = error
                call.completed.set()
        if call is not None and call.completed.is_set() and call.error is not None:
            self._latch_failure(call.error)
            try:
                self._fatal_once(call.error)
            except BaseException as notification:
                raise_failures(call.error, (notification,))
            raise call.error
        with self._condition:
            handles = tuple(self._handles.values())
        fork_pending = False
        fenced_fork = False
        for handle in handles:
            transfer = handle.transfer
            if transfer is None or transfer.direction != "fork" or transfer.retired.is_set():
                continue
            try:
                complete = transfer.event is None or transfer.event.query()
                if not complete and idle and not fenced_fork:
                    transfer.event.synchronize()
                    complete, fenced_fork = True, True
                if complete:
                    transfer.retired.set()
                else:
                    fork_pending = True
            except BaseException as error:
                self._latch_failure(error)
                self._fatal_once(error)
                raise
        for handle in handles:
            self._settle(handle)
            with self._condition:
                if self._handles.get(handle.serial) is not handle:
                    continue
            if (self.world == 1 and handle.role != "temporary" and not handle.expired
                    and time.monotonic() >= handle.expires):
                self.expire((handle.serial,))
            elif not handle.owners:
                self._reclaim(handle)
        return fork_pending or call is not None and not call.completed.is_set()

    def expired_handles(self):
        """Leader's nonblocking expiry decision, replicated by existing TP control."""
        if self.rank != 0:
            raise RuntimeError("only the admission leader decides absolute expiry")
        with self._condition:
            now = time.monotonic()
            return tuple(handle.serial for handle in self._handles.values()
                         if (handle.owners or handle.role == "prefix" and handle.internal_owner)
                         and not handle.expired and now >= handle.expires)

    def begin_shutdown(self):
        """ALL-rank actor service close; never used by profile transitions.

        Logical records survive. Active consumers and transfers still own their
        actual readers/source payload; the scheduler retires them before final
        close. Idle process-lifetime claims can end at this explicit boundary.
        """
        self.check_health()
        with self._condition:
            self._terminal = True
            handles = tuple(self._handles.values())
        for handle in handles:
            self.release_owner(handle)
        self._notify()

    def shutdown_progress(self, *, idle=True):
        if not self._terminal:
            raise RuntimeError("terminal retention progress requires explicit all-rank service shutdown")
        self.progress(idle=idle)
        with self._condition:
            return not self._handles and not self._pending and self._current is None and self._actor_call is None

    def expire(self, serials):
        if type(serials) not in (tuple, list) or len(serials) > (0 if self.policy is None else self.policy.max_sessions + self.prefix_limit):
            raise ValueError("retained expiry command exceeds its admitted session bound")
        with self._condition:
            if any(type(serial) is not int or serial not in self._handles for serial in serials):
                raise ValueError("retained expiry references an unavailable generation")
            handles = tuple(self._handles[serial] for serial in serials)
        for handle in handles:
            with self._condition:
                handle.expired = True
                pending = handle.pending
            if not pending:
                self.release_owner(handle)

    def next_expiry(self):
        """Monotonic leader deadline for the existing scheduler wait primitive."""
        if self.rank != 0:
            raise RuntimeError("only the admission leader owns expiry timing")
        with self._condition:
            return min((handle.expires for handle in self._handles.values()
                        if (handle.owners or handle.role == "prefix" and handle.internal_owner)
                        and not handle.expired), default=None)

    def _actor(self, operation, device):
        if device.type != "cuda":
            return operation()
        call = _ActorCall(operation, device)
        with self._condition:
            if self._actor_call is not None:
                raise RuntimeError("retention transfer has more than one issued GPU operation")
            self._actor_call = call
        self._notify()
        call.completed.wait()
        try:
            if call.error is not None:
                raise call.error
            return call.result
        finally:
            with self._condition:
                self._actor_call = None

    @property
    def has_retained(self):
        with self._condition:
            return bool(self._handles) or self._registration_owner is not None

    def record(self):
        with self._condition:
            handles = tuple(self._handles.values())
            required = tuple(handle for handle in handles if handle.role == "logical" and not handle.transient)
            counts = {name: sum(handle.phase == name for handle in required)
                      for name in ("reserved", "gpu", "saving", "ram", "restoring", "failed")}
            return {"policy": None if self.policy is None else self.policy.record(), "evidence": self.evidence,
                    "sessions": len(required), "blocking": len(handles), "phases": counts,
                    "transient_requests": sum(handle.transient for handle in handles),
                    "temporary_checkpoints": sum(handle.role == "temporary" for handle in handles),
                    "ram_prefix_generations": sum(handle.role == "prefix" for handle in handles),
                    "effective_transfer_chunk_bytes": self.transfer_chunk_bytes,
                    "native_store_value_limit_bytes": self._native_value_limit,
                    "native_store_key_limit_bytes": self._native_key_limit,
                    "transport_origin": self.transport_origin,
                    "native_payload_runtime": self.native_payload_runtime,
                    "transfers": sum(handle.transfer is not None for handle in handles),
                    "cpu_receiver_workers": (0 if self._wake_relay is None else
                                             self._wake_relay.record()["cpu_receiver_workers"]),
                    "readers": sum(len(handle.consumers) for handle in handles),
                    "future_backing_bytes": self._reserved, "prefix_entries": len(self._prefixes),
                    "recovery_versions": self.recovery_versions,
                    "protected_recovery_versions": self.protected_recovery_versions,
                    "logical_events": len(self._logical_events), "failure": self._failure is not None}

    def _serve(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending or self._closing or self._failure is not None)
                if self._failure is not None:
                    return
                if not self._pending:
                    return
                transfer = self._current = self._pending.pop(0)
            try:
                if transfer.direction == "save":
                    self._save(transfer)
                else:
                    self._restore(transfer)
            except BaseException as error:
                transfer.failure = error
                if not self._clean_retention_error(error):
                    self._latch_failure(error)
                    failures = []
                    if transfer.direction == "restore":
                        try:
                            transfer.plan.poison(error)
                        except BaseException as cleanup:
                            failures.append(cleanup)
                    try:
                        self._fatal_once(error)
                    except BaseException as notification:
                        failures.append(notification)
                    if failures:
                        raise_failures(error, failures)
            finally:
                transfer.retired.set()
                with self._condition:
                    self._current = None
                try:
                    self._notify()
                except BaseException as error:
                    self._latch_failure(error)
                    if transfer.failure is not None:
                        raise_failures(transfer.failure, (error,))
                    raise
            with self._condition:
                if self._failure is not None:
                    return  # Quiesce callback; poisoned payload owners stay latched.

    def _channel(self, transfer):
        return f"tf_retention/{self.identity}/{transfer.handle.serial}/{transfer.direction}/{transfer.generation}"

    def _store_payload(self, key, value=None):
        """Validate actual independent native fields BEFORE publication/get.

        Keys share one project bound even on a daemon without a native cap.
        The independently captured native value cap does not imply a whole
        wire-frame limit. TCPStore adds one '/' byte to the key.
        """
        try:
            validate_store_key(key)
        except ValueError as error:
            raise RetentionError(str(error)) from error
        if value is not None:
            size = len(value.encode("utf-8")) if type(value) is str else len(value)
            if self._native_value_limit is not None and size > self._native_value_limit:
                raise RetentionError("retained transport value exceeds the selected native value field limit")

    def _copy_source(self, raw, builder, buffer):
        for offset in range(0, raw.numel(), self.transfer_chunk_bytes):
            count = min(self.transfer_chunk_bytes, raw.numel() - offset)
            self._actor(lambda: buffer[:count].copy_(raw[offset:offset + count], non_blocking=raw.is_cuda), raw.device)
            builder.write(memoryview(buffer[:count].numpy()))

    def _manifest(self, channel, local):
        """Page descriptor metadata within existing bounded cohort messages."""
        # Fixed state/method metadata retains the existing strict frame bound;
        # only the geometry-bounded component tables may span many frames.
        canonical_runtime_json({"document": _json(local["metadata"])}, max_bytes=16 << 20)
        document = _json(local)
        if len(document) > self._manifest_bounds[self.rank]:
            raise ValueError("retained manifest exceeds its complete admitted schema/geometry bound")
        # Inner JSON is ASCII. Any character needs at most two bytes when
        # quoted in an outer fragment record; reserve the actual envelope.
        width = self._manifest_width()
        header = {"bytes": len(document), "pieces": (len(document) + width - 1) // width,
                  "sha256": hashlib.sha256(document.encode("ascii")).hexdigest()}
        headers = _phase(self._cohort, channel, "manifest-header", lambda: header, may_refuse=False)
        if any(type(row) is not dict or row.keys() != {"bytes", "pieces", "sha256"}
               or type(row["bytes"]) is not int or not 0 < row["bytes"] <= self._manifest_bounds[rank]
               or type(row["pieces"]) is not int or not is_digest(row["sha256"])
               or row["pieces"] != (row["bytes"] + width - 1) // width
               for rank, row in enumerate(headers)):
            raise RuntimeError("retained peer manifest differs from its admitted geometry")
        pieces = [[] for _ in headers]
        checksums = [hashlib.sha256() for _ in headers]
        for index in range(max(row["pieces"] for row in headers)):
            fragment = document[index * width:(index + 1) * width]
            subchannel = channel + "/manifest/" + str(index)
            rows = _phase(self._cohort, subchannel, "fragment", lambda: {"fragment": fragment}, may_refuse=False)
            for rank, row in enumerate(rows):
                value = row["fragment"]
                expected = max(0, min(width, headers[rank]["bytes"] - index * width))
                if type(value) is not str or not value.isascii() or len(value) != expected:
                    raise RuntimeError("retained manifest fragment length/encoding differs from its header")
                pieces[rank].append(value)
                checksums[rank].update(value.encode("ascii"))
            if self._store is not None:
                _retire_store(self._cohort, subchannel, stages=("fragment",))
            # pieces owns the received strings. The source slice and temporary
            # outcome dictionaries must not overlap the next slice/GET wave.
            fragment = rows = None
        result = []
        for rank, values in enumerate(pieces):
            if checksums[rank].hexdigest() != headers[rank]["sha256"]:
                raise ValueError("retained manifest metadata checksum mismatch")
            result.append(json.loads("".join(values)))
            pieces[rank] = None  # Do not retain both fragment and parsed archives.
        return result

    def _manifest_width(self):
        envelope = len(canonical_runtime_json({"ok": True, "value": {"fragment": ""}}))
        value_limit = min(16 << 20, self._native_value_limit or (16 << 20))
        return min(self.transfer_chunk_bytes, (value_limit - envelope) // 2)

    def _save(self, transfer):
        from dataclasses import asdict

        handle, inventory = transfer.handle, transfer.handle.inventory
        buffer = self._buffer
        keys = [None] * len(transfer.specs)  # Journal cells precede owned refs.
        published = False
        channel = self._channel(transfer)
        previous_parts = {}
        if (handle.clean_state is handle.state and handle.manifest is not None
                and handle.clean_generation == (handle.state.slot_generation, handle.state.history_generation)):
            local = handle.manifest[self.rank]
            previous_parts = {old["name"]: (old, key) for old, key in zip(local["specs"], local["keys"], strict=True)}
        try:
            for number, (part, spec) in enumerate(zip(inventory.parts, transfer.specs, strict=True)):
                previous = None
                if part.category == "history" and spec.name in previous_parts:
                    old, key = previous_parts[spec.name]
                    if _json(old) == _json(asdict(spec)):
                        previous = key
                if previous is not None:
                    with self._ram.acquire(previous):
                        self._ram.retain_into(previous, keys, number)
                    continue
                identity = _digest({"runtime": self.identity, "rank": self.rank, "part": asdict(spec)})
                builder = self._ram.begin(identity, spec.nbytes)
                try:
                    self._copy_source(transfer.byte_sources[number], builder, buffer)
                    builder.finish_into(keys, number)
                except BaseException as error:
                    rollback(builder, error, builder.close)
            local = {"metadata": inventory.metadata, "specs": [asdict(spec) for spec in transfer.specs], "keys": keys,
                     "objects": [self._ram.descriptor(key) for key in keys]}
            rows = self._manifest(channel, local)
            _phase(self._cohort, channel, "complete", lambda: True, may_refuse=False)
            if self._store is not None:
                _retire_store(self._cohort, channel, stages=("manifest-header", "complete"))
            old_keys = handle.keys
            handle.retiring_keys = old_keys
            handle.manifest, handle.keys = tuple(rows), tuple(keys)
            published = True
            handle.clean_state = handle.state
            handle.clean_generation = (handle.state.slot_generation, handle.state.history_generation)
            for index, key in enumerate(old_keys):
                self._ram.release(key)
                handle.retiring_keys = old_keys[index + 1:]
        except BaseException as error:
            if published:
                raise  # New manifest and unretired old references stay owned.
            failures = []
            for key in keys:
                if key is None:
                    continue
                try:
                    self._ram.release(key)
                except BaseException as cleanup:
                    failures.append(cleanup)
            raise_failures(error, failures)

    def _restore(self, transfer):
        handle, plan = transfer.source_handle, transfer.plan
        device = plan.state.pos_dev.device
        buffer = self._buffer
        channel = self._channel(transfer)
        # Check every source object before any destination write. A fully
        # retired corrupt RAM object is an isolated session failure; uncertain
        # transfer/write failures remain cohort-fatal below.
        def validate_objects():
            for spec, key in zip(plan.records[self.rank][1], handle.manifest[self.rank]["keys"], strict=True):
                if plan.continuation == "new-turn" and not new_turn_uses_part(plan.state, spec, plan.stream):
                    continue
                with self._ram.acquire(key):
                    pass
            return True

        try:
            _phase(self._cohort, channel, "validated", validate_objects)
        except _ManifestRefusal as error:
            if self._store is not None:
                _retire_store(self._cohort, channel, stages=("validated",))
            raise RetentionCorrupt("retained RAM generation failed its complete pre-write checksum check") from error
        for rank, record in enumerate(handle.manifest):
            for number, (spec, key) in enumerate(zip(plan.records[rank][1], record["keys"], strict=True)):
                if plan.continuation == "new-turn" and not new_turn_uses_part(plan.state, spec, plan.stream):
                    continue
                reader = self._ram.acquire(key, verify=False) if rank == self.rank else None
                checksum = hashlib.sha256()
                raw = plan.byte_targets.get((rank, number))
                try:
                    for offset in range(0, spec.nbytes, self.transfer_chunk_bytes):
                        count = min(self.transfer_chunk_bytes, spec.nbytes - offset)
                        name = f"{channel}/chunk/{rank}/{number}/{offset}"
                        if reader is not None:
                            payload = reader.read(offset, count)
                            self._store_payload(name, payload)
                            if self._store is not None:
                                self._store.set(name, payload)
                        elif self._store is not None:
                            self._store_payload(name)
                            payload = self._store.get(name)
                        else:
                            raise RuntimeError("RAM restore lacks the actual source rank")
                        if len(payload) != count:
                            raise ValueError("RAM transfer chunk differs from its complete descriptor")
                        checksum.update(payload)
                        if raw is not None:
                            buffer[:count].numpy()[:] = memoryview(payload)
                            self._actor(lambda: raw[offset:offset + count].copy_(buffer[:count], non_blocking=raw.is_cuda),
                                        raw.device)
                        if self._store is not None:
                            ack = f"{name}/ack/{self.rank}"
                            self._store.set(ack, "1")
                            if rank == self.rank:
                                ack_keys = [f"{name}/ack/{peer}" for peer in range(self.world)]
                                self._store.wait(ack_keys)
                                for item in (*ack_keys, name):
                                    self._store.delete_key(item)
                        # The next read allocates its owned bytes before
                        # assignment. Retire this chunk after its last CPU/DMA
                        # and peer use, preserving the one-ordinary-chunk bound.
                        payload = None
                    descriptor = record["objects"][number]
                    if descriptor["bytes"] != spec.nbytes or checksum.hexdigest() != descriptor["checksum"]:
                        raise ValueError("RAM transfer checksum differs from its complete immutable object")
                except BaseException as error:
                    if reader is not None:
                        rollback(reader, error, reader.close)
                    raise
                else:
                    if reader is not None:
                        reader.close()
        _phase(self._cohort, channel, "copied", lambda: True, may_refuse=False)
        _phase(self._cohort, channel, "committed",
               lambda: self._commit_on_actor(transfer, device), may_refuse=False)
        if self._store is not None:
            _retire_store(self._cohort, channel, stages=("validated", "copied", "committed"))

    def _commit_on_actor(self, transfer, device):
        transfer.handoff = self._actor(lambda: commit_state_restore(transfer.plan), device)
        return True

    def close(self, *, timeout):
        """Never evict the sole protected state to force a profile transition."""
        if timeout is None and self._worker is None:
            timeout = 0  # No worker or wait exists for default/GPU-only policy.
        if type(timeout) not in (int, float) or not 0 <= timeout <= threading.TIMEOUT_MAX or not math.isfinite(timeout):
            raise ValueError("retention close timeout exceeds the runtime wait domain")
        with self._condition:
            if self._closed:
                return
            if self._handles or self._registration_owner is not None:
                raise RuntimeError("retained sessions/readers/transfers must drain before catalog retirement")
            self._closing = True
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.drain(self._thread, timeout=timeout, cancel_unentered=False)
        if self._wake_relay is not None:
            # All peer senders must stop before the leader retires its queue.
            channel = "tf_retention/" + self.identity + "/close"
            self._wake_relay.fence()  # Same-sender socket processing proof.
            _phase(self._actor_cohort, channel, "workers-retired", lambda: True, may_refuse=False)
            self._wake_relay.close(timeout=timeout)
            _phase(self._actor_cohort, channel, "relay-retired", lambda: True, may_refuse=False)
            _retire_store(self._actor_cohort, channel, stages=("workers-retired", "relay-retired"))
            self._wake_relay = None
        if self._ram is not None:
            self._ram.close()
        with self._condition:
            self._closed = True
            self._store = self._buffer = None
            self.w, self.states, self._prefixes = None, (), []
            self._logical_events.clear()
            self._released_states.clear()
            if self._cohort is not None:
                self._cohort.comm = self._cohort.multi = None
            if self._actor_cohort is not None:
                self._actor_cohort.comm = self._actor_cohort.multi = None


def make_retention_catalog(policy, w, states, *, prefix_limit, recovery_versions, wake=None, register=None):
    catalog = RetainedCatalog.__new__(RetainedCatalog)
    catalog._condition = threading.Condition()
    catalog._pid = os.getpid()
    catalog._handles, catalog._prefixes = {}, []
    catalog._logical_events = {}
    catalog._released_states = {}
    catalog._closed = catalog._closing = False
    catalog._worker = catalog._thread = catalog._store = catalog._buffer = catalog._ram = None
    catalog._wake_relay = None
    catalog._cohort = None
    catalog._actor_cohort = None
    catalog._registration_owner = None
    # A caller's partial-owner journal precedes cloned stores, pinned buffers
    # and native thread launch. Failed initialization remains owned for cleanup.
    if register is not None:
        register(catalog)
    catalog.__init__(policy, w, states, prefix_limit=prefix_limit,
                     recovery_versions=recovery_versions, wake=wake)
    return catalog

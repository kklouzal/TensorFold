"""Leader-owned scheduling over the existing tensor-parallel MultiDecoder.

Only rank zero owns HTTP admission and Scheduler queues. Peers apply the same
admit/cancel/round/finish operations to their own model shard and request state.
Arrival, cancellation and adaptive prompt-piece choices are frozen per operation;
no follower polls a private client queue or chooses a different collective path.
"""

from __future__ import annotations

from datetime import timedelta
import hashlib
import json

from tensorfold.cuda.memory_gate import MemoryGate, NoRoom
from tensorfold.cuda.streams import Stream
from tensorfold.cuda.store_fields import validate_store_key
from tensorfold.cleanup import raise_failures
from .owner_execution import FAMILY_FIELDS
from . import media_requests

MAX_PROTOCOL_SEQUENCE = (1 << 63) - 1
FIFO_CHILD_SCHEMA = ("tensorfold.tp-fifo-retained-children", 1)


def _check_protocol_advance(value, label):
    """NEXT-id counters issue 0..MAX-1; MAX is the terminal exhausted value."""
    if type(value) is not int or value < 0:
        raise ValueError(f"tensor-parallel {label} requires a nonnegative integer")
    if value >= MAX_PROTOCOL_SEQUENCE:
        raise OverflowError(f"tensor-parallel {label} sequence exhausted")


class SharedMemoryGate(MemoryGate):
    """All ranks take the same growth decision; ordinary usage reads stay local."""

    def __init__(self, original, comm, *, rank_local=False):
        self.comm = comm
        # A placed media encoder has a distinct held payload and reserve.
        # Preserve that rank's admitted ledger; fits still votes every local
        # predicate before any rank allocates. Legacy equal-reserve layouts
        # retain their existing minimum-room initialization collective.
        room = original.room if rank_local else self._minimum(original.room)
        super().__init__(room, original.reserve, original.live, allow_lone_growth=False)
        self.held = original.held

    def _minimum(self, value):
        import torch

        source = torch.tensor([int(value)], dtype=torch.int64, device=self.comm.device)
        peers = torch.empty(self.comm.world, dtype=torch.int64, device=self.comm.device)
        self.comm.all_gather(source, peers)
        return int(peers.min().item())

    def fits(self, extra):
        return bool(self._minimum(int(super().fits(extra))))


class FrozenCancellation:
    """Client threads set the original event; a round reads its shared snapshot."""

    def __init__(self, original=None):
        while isinstance(original, FrozenCancellation):
            original = original.original
        self.original = original
        self.frozen = False
        self.active = False

    def is_set(self):
        return self.frozen if self.active or self.original is None else self.original.is_set()

    def set(self):
        if self.original is not None:
            self.original.set()

    def snapshot(self):
        return bool(self.original is not None and self.original.is_set())


class TensorParallelDecoder:
    """Scheduler-compatible control adapter; model execution remains MultiDecoder."""

    def __init__(self, decoder, engine):
        self.decoder, self.engine, self.comm = decoder, engine, decoder.w.comm
        self.sequence = 0
        self.next_wire = 0
        self.requests = {}
        self.wires = {}
        self.arrived = lambda: False
        self._arrival = False
        self._pass_rows = decoder._pass_rows
        self._rows = decoder.prefill_rows
        self._points = decoder.points
        self.failure = None
        self.prefix_operation = None
        self.prefix_sequence = 0
        self._operation_error = None
        self.admission_policy = "legacy"
        decoder.arrived = lambda: self._arrival
        decoder._pass_rows = lambda: self._rows
        decoder.prefix_decisions = self._prefix_decisions
        decoder.poison_control = self._poison_control

    def configure_admission_policy(self, policy):
        if policy not in ("legacy", "fifo"):
            raise ValueError("admission policy must be legacy or fifo")
        if self.requests or self.wires:
            if policy != self.admission_policy:
                raise RuntimeError("admission policy cannot change with live request owners")
        if policy == "fifo" and self.decoder is not None:
            from .input_preparation import InputPreparationContext

            if type(getattr(self.decoder, "input_preparation_context", None)) is not InputPreparationContext:
                raise ValueError("FIFO TP admission requires its actual installed coordinate/profile preparation owner")
        self.admission_policy = policy
        if self.decoder is not None:
            self.decoder.admission_policy = policy

    def validate_request(self, stream):
        validate = getattr(self.decoder, "validate_request", None)
        if validate is not None:
            validate(stream)
        owner = media_requests.runtime(getattr(self.decoder, "vision", None))
        if ((stream.vision is not None and owner is None)
                or stream.probabilities is not None or stream.penalties is not None):
            raise ValueError("tensor-parallel Flash Next requests do not support vision, logprobs or penalties")
        if stream.vision is not None and (tuple(stream.prompt) != stream.vision.token_ids
                or stream.vision.plan.profile_identity != owner.identity):
            raise ValueError("media input differs from its canonical prompt/profile")
        self._retained_scope(stream)

    @staticmethod
    def _retained_scope(stream):
        intent = getattr(stream, "retention_intent", None)
        if intent is not None:
            from .retained_catalog import LOCAL_PRINCIPAL

            principal = stream.principal or LOCAL_PRINCIPAL
            refs = tuple(ref for ref in (intent.target, intent.parent, *intent.aliases) if ref is not None)
            if any(ref.principal != principal for ref in refs):
                raise ValueError("retained logical input differs from its authenticated request principal")

    def prepare_enqueue(self, stream):
        """Copy application identities on the caller, without retaining its DB/operation."""
        self.validate_request(stream)
        stream.retention_intent = None
        catalog = getattr(self.decoder, "retention_catalog", None)
        if catalog is not None and catalog.policy is not None:
            from .retained_host import RetentionIntent

            stream.retention_intent = RetentionIntent.current()
            self._retained_scope(stream)

    def _retention_cpu(self, operation, *, pure=False):
        """Only proved, completed session refusals can cross the healthy actor boundary."""
        if self.failure is not None:
            raise RuntimeError("tensor-parallel scheduler is poisoned; retire the rank cohort") from self.failure
        try:
            self.decoder.retention_catalog.check_health()
            result = operation()
            self.decoder.retention_catalog.check_health()
            return result
        except BaseException as error:
            try:
                from .retained_catalog import RetentionError
                from .execution_identity import _unretired

                ordinary = (self.failure is None and (isinstance(error, (RetentionError, NoRoom))
                            or pure and isinstance(error, ValueError)) and not _unretired(error))
                if ordinary:
                    self.decoder.retention_catalog.check_health()
            except BaseException as inspection:
                try:
                    self._poison_control(error)
                except BaseException as containment:
                    raise_failures(error, (inspection, containment))
                raise_failures(error, (inspection,))
            if not ordinary:
                try:
                    self._poison_control(error)
                except BaseException as containment:
                    raise_failures(error, (containment,))
            raise

    @staticmethod
    def _request_error_types():
        from tensorfold.server.errors import RequestError, CapacityError, ContextLengthError
        from tensorfold.server.cancellation import RequestCancelled
        from tensorfold.vision.images_http import ImageInputError
        from tensorfold.vision.replay import MediaBusy, MediaCancelled
        from .retained_catalog import RetentionError, RetentionCapacity, RetentionCorrupt, RetentionExpired, RetentionMismatch

        return {value.__name__: value for value in (ValueError, NoRoom, RequestError, CapacityError,
            ContextLengthError, RequestCancelled, ImageInputError, MediaBusy, MediaCancelled,
            RetentionError, RetentionCapacity, RetentionCorrupt, RetentionExpired, RetentionMismatch)}

    def _request_error_record(self, error):
        if error is None:
            return None
        try:
            kinds = self._request_error_types()
            name = type(error).__name__
            if kinds.get(name) is not type(error):
                raise RuntimeError("unregistered request failure cannot cross the healthy TP retirement boundary")
            return {"type": name, "message": BaseException.__str__(error)[:1024]}
        except BaseException as inspection:
            try:
                self._poison_control(error)
            except BaseException as containment:
                raise_failures(error, (inspection, containment))
            raise_failures(error, (inspection,))

    def stage_request(self, stream, *, wake, max_waiting_input_bytes, max_pending_requests):
        """Existing actor replicates queued readers before it applies logical expiry."""
        catalog = getattr(self.decoder, "retention_catalog", None)
        if catalog is None or catalog.policy is None or getattr(stream, "_retention_staged", False):
            return None
        return self._retention_cpu(lambda: self._stage_request(stream, catalog, wake,
            max_waiting_input_bytes, max_pending_requests), pure=True)

    def _stage_request(self, stream, catalog, wake, max_waiting_input_bytes, max_pending_requests):
        allocate_wire = not hasattr(stream, "_media_wire")
        if allocate_wire:
            _check_protocol_advance(self.next_wire, "wire")
        catalog.set_wake(wake)
        intent = stream.retention_intent
        if allocate_wire:
            stream._media_wire, self.next_wire = self.next_wire, self.next_wire + 1
        owner = media_requests.runtime(getattr(self.decoder, "vision", None))
        if owner is not None and stream.vision is not None:
            self._media_cpu(lambda: media_requests.bind(stream, owner, stream._media_wire))
        from .retained_catalog import LOCAL_PRINCIPAL

        principal = stream.principal or LOCAL_PRINCIPAL
        parent, prefix_serial = None, None
        from .prefix_policy import reuse_allowed

        reuse = reuse_allowed(self.decoder.w, stream.sampling, draft=stream.draft, vision=stream.vision,
                              position=len(stream.prompt), kv_pair=self.decoder.kv_pair)
        if reuse:
            candidate = None if intent is None or intent.parent is None else catalog.lookup(intent.parent)
            if candidate is None or not catalog.prefix_matches(candidate, stream.prompt, principal=principal):
                candidate = catalog.best_prefix(stream.prompt, principal=principal)
            if candidate is not None:
                if candidate.role == "prefix":
                    prefix_serial = candidate.serial
                else:
                    parent = next(reference for reference, ready in candidate.owners.items()
                                  if ready and reference.principal == principal)
        body = {"kind": "enqueue", "wire": stream._media_wire, "stream": self._pack(stream),
                "parent": None if parent is None else [parent.principal, parent.sequence, parent.generation],
                "prefix_serial": prefix_serial,
                "queue_bytes": max_waiting_input_bytes, "queue_count": max_pending_requests}
        self._invoke(body, source=stream)
        return None

    def request_blocked_reason(self, stream):
        if getattr(self.decoder, "retention_catalog", None) is None:
            return None
        return self._retention_cpu(lambda: self.decoder.retention_eligibility(stream))

    def owns_admission(self, stream):
        return getattr(stream, "retention_window", None) is not None

    def admission_claims(self):
        return sum(self.owns_admission(stream) or (getattr(stream, "retained_host", None) is not None
            and stream.retention_handle.state is not None) for stream in self.requests.values())

    def suspend_requests(self, streams):
        catalog = getattr(self.decoder, "retention_catalog", None)
        if catalog is None or catalog.policy is None or catalog.policy.tier != "ram":
            return []
        suspended = []
        for stream in streams:
            if self._invoke({"kind": "retention_suspend", "wire": self.wires[id(stream)]}):
                suspended.append(stream)
        return suspended

    def retention_boundary(self, *, idle, needs_slot=False, shutdown=False, drop_optional=False):
        catalog = getattr(self.decoder, "retention_catalog", None)
        if catalog is None or catalog.policy is None:
            return {"drained": True, "changed": False}
        return self._retention_cpu(lambda: self._retention_boundary(catalog, idle, needs_slot, shutdown, drop_optional))

    def _retention_boundary(self, catalog, idle, needs_slot, shutdown, drop_optional):
        suspend = None
        if needs_slot and not self.decoder.free and catalog.policy.tier == "ram" and not shutdown:
            active = tuple(stream.st for stream in [*self.decoder.streams.values(), *self.decoder.filling])
            candidates = catalog.suspendable_handles(active_states=active)
            if candidates:
                suspend = candidates[0].serial
        return self._invoke({"kind": "retention_boundary", "idle": idle, "shutdown": shutdown,
                             "events": catalog.logical_events(), "expired": list(catalog.expired_handles()),
                             "suspend": suspend, "drop_optional": drop_optional})

    def drop_optional_retention(self):
        """Existing actor only, after the managed profile's request/gate drain."""
        return self.retention_boundary(idle=True, drop_optional=True)

    def retention_deadline(self):
        catalog = getattr(self.decoder, "retention_catalog", None)
        return None if catalog is None or catalog.policy is None else self._retention_cpu(catalog.next_expiry)

    def suspend_profile(self):
        if self.requests or self.wires or self._operation_error is not None:
            raise RuntimeError("TP request owners must retire before profile replacement")
        self.decoder = None
        self._pass_rows = self._points = None
        self._rows = 0

    def bind_profile(self, decoder):
        if self.decoder is not None or self.requests or self.wires:
            raise RuntimeError("a TP profile can bind only after predecessor retirement")
        self.decoder = decoder
        self._pass_rows, self._points = decoder._pass_rows, decoder.points
        self._rows = decoder.prefill_rows
        decoder.arrived = lambda: self._arrival
        decoder._pass_rows = lambda: self._rows
        decoder.prefix_decisions = self._prefix_decisions
        decoder.poison_control = self._poison_control
        self.configure_admission_policy(self.admission_policy)

    def __getattr__(self, name):
        return getattr(self.decoder, name)

    def _key(self, sequence, peer):
        key = f"tensorfold/flashnext/round/{sequence}/rank/{peer}"
        validate_store_key(key)
        return key

    def _ack_key(self, sequence, peer):
        key = f"tensorfold/flashnext/round/{sequence}/ack/{peer}"
        validate_store_key(key)
        return key

    def _poison_control(self, error):
        first = self.failure is None
        if self.failure is None:
            self.failure = error
        self.engine._abort_comm = True
        if first:
            try:
                from tensorfold.cuda.cohort import notify_failure

                notify_failure()
            except BaseException as notification:
                raise_failures(error, [notification])

    def _media_cpu(self, operation):
        """Expected, retired input failures stay local; uncertain owners contain the cohort."""
        if self.failure is not None:
            raise RuntimeError("tensor-parallel scheduler is poisoned; retire the rank cohort") from self.failure
        try:
            return operation()
        except BaseException as error:
            try:
                from tensorfold.vision.images_http import ImageInputError

                ordinary = isinstance(error, ImageInputError)
                if ordinary:
                    from .execution_identity import _unretired

                    ordinary = not _unretired(error)
                    if ordinary:
                        owner = media_requests.runtime(getattr(self.decoder, "vision", None))
                        owner.check_media_health()
            except BaseException as inspection:
                try:
                    self._poison_control(error)
                except BaseException as containment:
                    raise_failures(error, (inspection, containment))
                raise_failures(error, (inspection,))
            if not ordinary:
                try:
                    self._poison_control(error)
                except BaseException as containment:
                    raise_failures(error, (containment,))
            raise

    def _remember(self, wire, stream):
        self.requests[wire] = stream
        self.wires[id(stream)] = wire
        stream._media_rank_owned = getattr(stream, "_media_request", None) is not None
        cancel = FrozenCancellation(stream.cancel)
        cancel.active = True
        stream.cancel = cancel
        emit = getattr(stream.emit, "_tensorfold_tp_emit", stream.emit)

        def shared_emit(tokens):
            if emit is not None:
                emit(tokens)
            return cancel.frozen

        shared_emit._tensorfold_tp_emit = emit
        stream.emit = shared_emit

    def _forget(self, wire):
        stream = self.requests.pop(wire, None)
        if stream is not None:
            self.wires.pop(id(stream), None)

    def _fifo_child_geometry(self):
        context = getattr(self.decoder, "logical_context", None)
        vocab = getattr(getattr(getattr(self.decoder, "w", None), "cfg", None), "vocab", None)
        if type(context) is not int or context <= 0 or type(vocab) is not int or vocab <= 0:
            raise ValueError("FIFO child schema requires the actual logical context and model vocabulary")
        return context, vocab

    def fifo_retained_child_schema(self):
        """Describe the guards on this actual bound FIFO adapter, not a mode hint."""
        if self.admission_policy != "fifo":
            return None
        context, vocab = self._fifo_child_geometry()
        return {"schema": FIFO_CHILD_SCHEMA[0], "version": FIFO_CHILD_SCHEMA[1],
                "context": context, "vocab": vocab}

    def _validate_fifo_children(self, prompt, owed, sampling, epoch, committed):
        context, vocab = self._fifo_child_geometry()
        for role, values in (("prompt", prompt), ("owed", owed)):
            if type(values) is not list or len(values) > context:
                raise ValueError(f"FIFO {role} must be a token list within the logical context")
            if any(type(token) is not int or not 0 <= token < vocab for token in values):
                raise ValueError(f"FIFO {role} token IDs differ from the selected model vocabulary")
        # These fields otherwise survive directly in Stream/Sampling. Reject
        # opaque wire trees; scalar integer widths remain the parser's domain.
        if (sampling is not None and (type(sampling) is not list or len(sampling) != 5
                or not isinstance(sampling[0], int)
                or any(not isinstance(value, (int, float)) for value in sampling[1:]))):
            raise ValueError("FIFO sampling requires a flat native numeric record")
        if epoch is not None and (type(epoch) is not int or epoch < 0):
            raise ValueError("FIFO profile epoch must be None or a nonnegative native integer")
        if type(committed) is not int or committed < 0:
            raise ValueError("FIFO committed output count must be a nonnegative native integer")

    def _pack(self, stream):
        if self.admission_policy == "fifo" and stream.carry is not None:
            raise ValueError("FIFO request cannot carry legacy continuation statistics")
        from tensorfold.engine.grammar import pack

        sampling = stream.sampling
        owner = media_requests.runtime(getattr(self.decoder, "vision", None))
        if ((stream.vision is not None and owner is None)
                or stream.probabilities is not None or stream.penalties is not None):
            raise ValueError("tensor-parallel Flash Next requests do not support vision, logprobs or penalties")
        wire_sampling = None if sampling is None else [sampling.seed, sampling.temperature, sampling.top_k,
                                                       sampling.top_p, sampling.min_p]
        if self.admission_policy == "fifo":
            self._validate_fifo_children(stream.prompt, stream.owed, wire_sampling,
                                         stream.profile_epoch, stream.committed_output_tokens)
        result = {"prompt": stream.prompt, "count": stream.count, "draft": stream.draft,
                "stop_eos": stream.stop_eos, "background": stream.background,
                "sampling": wire_sampling,
                "grammar": pack(stream.constraint), "owed": stream.owed, "carry": stream.carry,
                "token_limit": stream.token_limit is not None, "profile_epoch": stream.profile_epoch,
                "committed_output_tokens": stream.committed_output_tokens}
        if self.admission_policy == "fifo":
            ticket = getattr(stream, "admission_ticket", None)
            if type(ticket) is not int or not 0 <= ticket < 1 << 63:
                raise ValueError("FIFO request lost its immutable validated enqueue ticket")
            if stream.admission_status.ticket != ticket:
                raise ValueError("FIFO request ticket changed after validated enqueue")
            principal = stream.principal
            if principal is not None and (type(principal) is not str or not principal):
                raise ValueError("FIFO request lost its trusted principal scope")
            if stream.admission_status.principal != principal:
                raise ValueError("FIFO request principal changed after validated enqueue")
            result["admission_ticket"] = ticket
            result["principal"] = principal
            intent = stream.retention_intent
            result["retention"] = None if intent is None else intent.record()
            result["output_capacity"] = getattr(stream, "output_capacity", max(1, stream.count))
        if stream.vision is not None:
            from tensorfold.cuda.rank_protocol import MAX_MESSAGE_BYTES
            from .media_wire import pack_media

            request = getattr(stream, "_media_request", None)
            if request is None:
                raise ValueError("media CPU request must prepare before TP admission")
            result["media"] = pack_media(request.prepared, epoch=request.epoch, max_bytes=MAX_MESSAGE_BYTES,
                                         envelope=owner.profile.cpu)
        return result

    def _unpack(self, body):
        if self.admission_policy == "fifo":
            if body["carry"] is not None:
                raise ValueError("received FIFO request cannot carry legacy continuation statistics")
            self._validate_fifo_children(body["prompt"], body["owed"], body["sampling"],
                                         body["profile_epoch"], body["committed_output_tokens"])
        from tensorfold.engine.exact_sampling import Sampling

        constraint = None
        if body["grammar"]:
            from tensorfold.engine import grammar

            constraint = grammar.compiler(self.engine, self.engine.model_dir, self.engine.eos).follow(body["grammar"])
        stream = Stream(list(body["prompt"]), int(body["count"]),
                      None if body["sampling"] is None else Sampling(*body["sampling"]),
                      draft=bool(body["draft"]), stop_eos=bool(body["stop_eos"]),
                      background=bool(body["background"]), constraint=constraint,
                      owed=list(body["owed"]), carry=body["carry"], profile_epoch=body["profile_epoch"],
                      committed_output_tokens=body["committed_output_tokens"])
        stream._tp_token_limit = bool(body.get("token_limit", False))
        if self.admission_policy == "fifo":
            ticket = body["admission_ticket"]
            if type(ticket) is not int or not 0 <= ticket < 1 << 63:
                raise ValueError("received FIFO request has an invalid enqueue ticket")
            stream.admission_ticket = ticket
            principal = body["principal"]
            if principal is not None and (type(principal) is not str or not principal):
                raise ValueError("received FIFO request has an invalid principal scope")
            stream.principal = principal
            capacity = body["output_capacity"]
            if type(capacity) is not int or not 1 <= capacity <= max(1, stream.count):
                raise ValueError("received FIFO output capacity differs from its validated reply allowance")
            stream.output_capacity = capacity
            if body["retention"] is None:
                stream.retention_intent = None
            else:
                from .retained_host import RetentionIntent

                stream.retention_intent = RetentionIntent.from_record(body["retention"])
        if "media" in body:
            from tensorfold.cuda.rank_protocol import MAX_MESSAGE_BYTES
            from .media_wire import unpack_media

            owner = media_requests.runtime(getattr(self.decoder, "vision", None))
            if owner is None:
                raise ValueError("TP media metadata requires the admitted runtime")
            stream.vision = unpack_media(body["media"], token_ids=stream.prompt, contract=owner.contract,
                                         max_bytes=MAX_MESSAGE_BYTES, envelope=owner.profile.cpu)
            media_requests.bind(stream, owner, stream.vision.epoch)
        return stream

    def _prefix_decisions(self, decisions, error=None):
        """One leader-owned stop decision, before any peer publishes state.

        CPU metadata uses the existing ordered store; no hot GPU allocation or
        extra device-wide fence is introduced. Operation acknowledgments bound
        the lifetime of the peer-owned keys.
        """
        if self.prefix_operation is None:
            raise RuntimeError("stop decisions require an ordered TP scheduler operation")
        _check_protocol_advance(self.prefix_sequence, "prefix")
        step = self.prefix_sequence
        self.prefix_sequence += 1
        def key(peer):
            value = f"tensorfold/flashnext/prefix/{self.prefix_operation}/{step}/rank/{peer}"
            validate_store_key(value)
            return value
        if self.comm.rank == 0:
            payload = json.dumps({"decisions": decisions, "failed": error is not None}, separators=(",", ":"))
            keys = [key(peer) for peer in range(1, self.comm.world)]
            try:
                many = getattr(self.comm.store, "multi_set", None)
                if callable(many):
                    many(keys, [payload] * len(keys))
                else:
                    for item in keys:
                        self.comm.store.set(item, payload)
            except BaseException as failure:
                # A partial publication leaves peers at different control
                # positions. Abort this cohort; normal ACK aggregation cannot
                # finish while an unpublished peer waits for its decision.
                self._poison_control(failure)
                raise
            failed = error is not None
        else:
            own = key(self.comm.rank)
            self.comm.store.wait([own], timedelta(hours=1))
            payload = json.loads(self.comm.store.get(own))
            self.comm.store.delete_key(own)
            decisions, failed = payload["decisions"], payload["failed"]
        if failed:
            raise RuntimeError("tensor-parallel stop-boundary policy failed") from error
        return decisions

    def _controls(self, execution=None):
        forgotten = [wire for wire, stream in self.requests.items()
                     if stream.st is None and stream.cancel.snapshot()
                     and not getattr(stream, "_retention_staged", False)
                     and (getattr(stream, "_media_request", None) is None or stream._media_request.retired)]
        result = {"forget": forgotten,
                  "cancel": [[wire, stream.cancel.snapshot()] for wire, stream in self.requests.items()
                             if wire not in forgotten],
                  "arrived": bool(self.arrived())}
        result["admission_policy"] = self.admission_policy
        if self.admission_policy == "fifo":
            result["flow"] = [[wire, getattr(stream, "blocked_input", None),
                               getattr(stream, "production_budget", None)] for wire, stream in self.requests.items()
                              if stream.st is not None]
        controller = getattr(self.decoder, "controller", None)
        quota = None
        if controller is not None:
            settings = None if execution is None else {name: execution[name] for name in controller.maximum}
            if execution is None:
                boundary = controller.freeze_boundary(settings)
            else:
                boundary = controller.freeze_boundary(settings, family={name: execution[name] for name in FAMILY_FIELDS},
                                                      profile_id=execution["profile_id"])
            result["execution_boundary"] = boundary
            quota = getattr(self.decoder, "pass_rows_for", None)
        result["pass_rows"] = (0 if self._pass_rows is None else self._pass_rows()) if quota is None else quota(boundary["settings"])
        return result

    def _apply_controls(self, body):
        if body.get("admission_policy", "legacy") != self.admission_policy:
            try:
                self.configure_admission_policy(body.get("admission_policy", "legacy"))
            except BaseException as failure:
                self._poison_control(failure)
                raise
        if self.admission_policy == "fifo":
            try:
                for wire, blocked, budget in body["flow"]:
                    if (wire not in self.requests or blocked not in (None, "media", "client_input", "output", "tier_unavailable", "restoring", "sequence_busy")
                            or budget is not None and (type(budget) is not int or budget < 0)):
                        raise ValueError("FIFO flow control differs from its owned requests and legal production credit")
                    self.requests[wire].blocked_input = blocked
                    self.requests[wire].production_budget = budget
            except BaseException as failure:
                self._poison_control(failure)
                raise
        controller = getattr(self.decoder, "controller", None)
        boundary = body.get("execution_boundary")
        if controller is not None:
            controller.apply_boundary(boundary)
            apply_family = getattr(self.decoder, "apply_execution_family", None)
            if apply_family is not None:
                apply_family()
        elif boundary is not None:
            raise ValueError("this tensor-parallel decoder has no execution controller")
        for wire in body["forget"]:
            self._forget(wire)
        for wire, cancelled in body["cancel"]:
            self.requests[wire].cancel.frozen = bool(cancelled)
            self.requests[wire].cancel.active = True
        self._arrival, self._rows = bool(body["arrived"]), int(body["pass_rows"])

    def _state_digest(self):
        state = [(wire, stream.sid, stream.count, stream.cached, len(stream.out), stream.out[-8:],
                  stream.drafts, stream.owed, bool(stream.done), bool(stream.waiting),
                  None if stream.st is None else [stream.st.pos, getattr(stream.st, "mtp_len", None),
                                                  getattr(stream.st, "mtp_drafted", None),
                                                  getattr(stream.st, "cur", None),
                                                  getattr(stream.st, "slot_generation", None),
                                                  getattr(stream.st, "history_generation", None)],
                  None if stream.error is None else type(stream.error).__name__)
                 for wire, stream in self.requests.items()]
        controller = getattr(self.decoder, "controller", None)
        execution = None if controller is None else [controller.epoch, controller.effective,
                                                     controller.last_served, controller.effective_family]
        profiles = getattr(self.engine, "serving_profiles", None)
        serving = None if profiles is None else [profiles.gate.epoch, profiles.gate.available]
        record = [state, execution, serving]
        if self.admission_policy == "fifo":
            record.append([self.admission_policy, [(wire, stream.admission_ticket, stream.principal)
                                                  for wire, stream in self.requests.items()]])
        digest = hashlib.sha256()
        for chunk in json.JSONEncoder(sort_keys=True, separators=(",", ":")).iterencode(record):
            digest.update(chunk.encode("utf-8"))
        return digest.hexdigest()

    def _dispatch(self, body, source=None):
        self.prefix_operation = body["sequence"]
        self.prefix_sequence = 0
        self._apply_controls(body["control"])
        kind = body["kind"]
        if kind == "retire_admission_error":
            if self._operation_error is None:
                raise RuntimeError("no admission error ownership to retire")
            self.engine.serving_profiles.error_outcome(self._operation_error)
            self._operation_error = None
            return None
        if kind in ("serving_prepare", "serving_replace"):
            try:
                owner = self.engine.serving_profiles
                return (owner.preflight(body["profile"], body["request_epoch"]) if kind == "serving_prepare"
                        else owner.replace())
            except BaseException as failure:
                self._poison_control(failure)
                raise
        if kind in ("enqueue", "admit"):
            wire = body["wire"]
            def prepare_stream():
                stream = self.requests.get(wire)
                if stream is None:
                    stream = source if source is not None else self._unpack(body["stream"])
                request = getattr(stream, "_media_request", None)
                if request is not None and (request.epoch != wire
                        or request.plan.media_identity != body["stream"]["media"]["media_identity"]):
                    raise ValueError("TP media metadata differs from its stable request wire/identity")
                if self.admission_policy == "fifo":
                    if stream.admission_ticket != body["stream"]["admission_ticket"]:
                        raise ValueError("FIFO request ticket changed between leader and peers")
                    if stream.principal != body["stream"]["principal"]:
                        raise ValueError("FIFO request principal changed between leader and peers")
                    self.validate_request(stream)
                return stream

            if "media" in body["stream"] or self.admission_policy == "fifo":
                context = getattr(self.decoder, "input_preparation_context", None)
                if context is None:
                    failure = RuntimeError("TP media admission requires its installed cohort preparation owner")
                    self._poison_control(failure)
                    raise failure
                # All followers validate metadata before any admitted slot can
                # enter a GPU allocation/growth collective. No model collective
                # is permitted inside this pure, existing host protocol phase.
                try:
                    if self.admission_policy == "fifo":
                        from tensorfold.cuda.rank_protocol import bounded_json

                        record = body["stream"] if kind == "admit" else {
                            "stream": body["stream"], "parent": body["parent"],
                            "prefix_serial": body["prefix_serial"],
                            "queue_bytes": body["queue_bytes"], "queue_count": body["queue_count"]}
                        contract = hashlib.sha256(bounded_json(record, separators=(",", ":")).encode("ascii")).hexdigest()
                        label = "fifo-request-admission"
                    else:
                        contract = hashlib.sha256(f"media-request:{wire}".encode()).hexdigest()
                        label = "media-request-admission"
                except BaseException as failure:
                    self._poison_control(failure)
                    raise
                stream = context.prepare(label, prepare_stream, may_refuse=True,
                                         contract=contract)
            else:
                stream = prepare_stream()
            if wire not in self.requests:
                self._remember(wire, stream)
            if kind == "enqueue":
                from .retained_catalog import LogicalReference

                catalog = getattr(self.decoder, "retention_catalog", None)
                catalog.configure_admission(max_waiting_input_bytes=body["queue_bytes"],
                                            max_pending_requests=body["queue_count"])
                parent = None if body["parent"] is None else LogicalReference(*body["parent"])
                stream._retention_staged = True
                self._retention_cpu(lambda: self.decoder.enqueue_retained(stream, parent=parent,
                                                                          prefix_serial=body["prefix_serial"]))
                return None
            original = self.decoder.points
            self.decoder.points = lambda prompt: body["points"]
            try:
                if getattr(stream, "retention_request", None) is not None:
                    return self._retention_cpu(lambda: self.decoder.admit(stream))
                return self.decoder.admit(stream)
            except BaseException as error:
                if (not getattr(stream, "_retention_staged", False)
                        and (not isinstance(error, NoRoom) or not self.decoder.live())):
                    self._forget(wire)
                raise
            finally:
                self.decoder.points = original
        if kind == "retention_retire":
            stream = self.requests[body["wire"]]
            record = body["error"]
            if record is not None:
                error_type = self._request_error_types()[record["type"]]
                if source is not None and type(source) is not error_type:
                    raise RuntimeError("request retirement lost the original leader error identity")
                stream.error = source if source is not None else error_type(record["message"])
            if stream.st is None:
                stream.done = True
            result = self._retention_cpu(lambda: self.decoder.retire_retained_request(stream))
            if result and stream.st is None:
                host = getattr(stream, "retirement_host", None)
                if host is not None and host.delivery is None:
                    host.retire_backend()
                    stream.retirement_host = None
                self._forget(body["wire"])
            return result
        if kind == "retention_suspend":
            return self._retention_cpu(lambda: self.decoder.suspend_request(self.requests[body["wire"]]))
        if kind == "retention_boundary":
            from .retained_catalog import RetentionError, RetentionExpired

            catalog = getattr(self.decoder, "retention_catalog", None)
            def progress():
                record = catalog.record()
                before = (record["phases"], record["transfers"], record["blocking"])
                catalog.apply_logical_events(body["events"])
                catalog.expire(body["expired"])
                if body.get("drop_optional", False):
                    catalog.drop_optional_prefixes()
                if body["shutdown"]:
                    catalog.begin_shutdown()
                if body["suspend"] is not None:
                    try:
                        victim = catalog.by_serial(body["suspend"])
                    except RetentionExpired:
                        if not body["events"] and not body["expired"]:
                            raise
                        # The same ordered lifetime command may have released
                        # this formerly idle victim and its slot already.
                    else:
                        catalog.suspend(victim)
                try:
                    catalog.progress(idle=body["idle"])
                    catalog.settle_ready()
                except RetentionError:
                    # Common settlement records the completed refusal on its
                    # consumer; readiness projects it to that request only.
                    catalog.check_health()
                self.decoder._release_retained_states()
                drained = catalog.shutdown_progress(idle=body["idle"]) if body["shutdown"] else False
                record = catalog.record()
                after = (record["phases"], record["transfers"], record["blocking"])
                return {"drained": drained, "changed": before != after}
            return self._retention_cpu(progress)
        if kind == "round":
            if "media_limits" in body:
                self.decoder._media_limits = {self.requests[wire].sid: tuple(limit)
                                              for wire, limit in body["media_limits"]}
            if "execution" in body:
                self.decoder.accept_execution(body["execution"])
            return self.decoder.round()
        if kind == "media_failure":
            from tensorfold.vision.images_http import ImageInputError
            from tensorfold.vision.replay import MediaBusy, MediaCancelled

            stream = self.requests[body["wire"]]
            error_type = {"input": ImageInputError, "busy": MediaBusy, "cancelled": MediaCancelled}[body["error_type"]]
            if source is not None and type(source) is not error_type:
                raise RuntimeError("media failure publication lost its original leader input outcome")
            stream.error = source if source is not None else error_type(body["message"])
            return self.decoder.cancel([stream])
        if kind == "media_forget":
            self._forget(body["wire"])
            return None
        if kind == "profile_manifest":
            return self.decoder.install_execution_manifest(body["name"])
        if kind in ("cancel", "finish"):
            streams = [self.requests[wire] for wire in body["wires"]]
            result = getattr(self.decoder, kind)(streams)
            if kind == "finish":
                profiles = getattr(self.engine, "serving_profiles", None)
                if profiles is not None:
                    for stream in streams:
                        if stream.error is not None:
                            profiles.error_outcome(stream.error, state_retired=True, states=() if stream.st is None else (stream.st,))
                        stream.st = stream.snap = stream.copies = stream.vision = None
                for wire in body["wires"]:
                    self._forget(wire)
            return result
        if kind == "drop":
            record = body.get("error")
            owning = self._operation_error
            if record is not None and owning is None:
                owning = source if source is not None else self._request_error_types()[record["type"]](record["message"])
            result = self.decoder.drop(owning) if record is not None else self.decoder.drop()
            profiles = getattr(self.engine, "serving_profiles", None)
            if profiles is not None and self._operation_error is not None:
                profiles.error_outcome(self._operation_error, state_retired=True,
                    states=tuple(stream.st for stream in result if stream.st is not None))
                self._operation_error = None
            # MultiDecoder.drop retires only unfinished streams. Scheduler
            # still calls finish(done) for streams cancelled before the failed
            # round; retain their wire owners until that normal retirement.
            for stream in result:
                wire = self.wires.get(id(stream))
                if wire is not None:
                    self._forget(wire)
            return result
        raise ValueError(f"unknown tensor-parallel scheduler operation {kind!r}")

    def _execute(self, body, source=None):
        error, result = None, None
        try:
            result = self._dispatch(body, source)
        except BaseException as caught:
            error = caught
            if getattr(self.engine, "serving_profiles", None) is not None and body["kind"] in ("admit", "round"):
                if self._operation_error is not None:
                    self.engine.serving_profiles._poison(caught)
                self._operation_error = caught
        finally:
            for stream in self.requests.values():
                stream.cancel.active = False
        status = {"error": None if error is None else type(error).__name__, "state": self._state_digest()}
        if body["kind"] in ("retention_retire", "retention_boundary", "retention_suspend"):
            status["complete"] = result
        return result, error, status

    def _invoke(self, body, source=None):
        if self.failure is not None:
            raise RuntimeError("tensor-parallel scheduler is poisoned; retire the rank cohort") from self.failure
        if self.comm.rank != 0:
            raise RuntimeError("only rank zero submits tensor-parallel scheduler operations")
        _check_protocol_advance(self.sequence, "operation")
        body["control"] = self._controls(body.get("execution"))
        sequence = self.sequence
        body["sequence"] = sequence
        from tensorfold.cuda.rank_protocol import MAX_MESSAGE_BYTES, bounded_json

        text = bounded_json(body, separators=(",", ":"))
        self.sequence += 1
        try:
            messages = [self._key(sequence, peer) for peer in range(1, self.comm.world)]
            set_many = getattr(self.comm.store, "multi_set", None)
            if callable(set_many):
                set_many(messages, [text] * len(messages))
            else:
                for key in messages:
                    self.comm.store.set(key, text)
            text = None
            result, error, status = self._execute(body, source)
            if self.failure is not None:
                raise RuntimeError("tensor-parallel control publication failed; retire all rank owners") from self.failure
            keys = [self._ack_key(sequence, peer) for peer in range(1, self.comm.world)]
            self.comm.store.wait(keys, timedelta(hours=1))
            get_many = getattr(self.comm.store, "multi_get", None)
            payloads = get_many(keys) if callable(get_many) else [self.comm.store.get(key) for key in keys]
            if len(payloads) != len(keys):
                raise RuntimeError("tensor-parallel scheduler received an incomplete acknowledgment batch")
            for peer, (key, payload) in enumerate(zip(keys, payloads), 1):
                if type(payload) is not bytes or len(payload) > MAX_MESSAGE_BYTES:
                    raise ValueError("tensor-parallel acknowledgment requires at most 96 MiB of bytes")
                remote = json.loads(payload)
                self.comm.store.delete_key(key)
                if remote != status:
                    failure = RuntimeError(f"tensor-parallel scheduler diverged on rank {peer}: {remote} != {status}")
                    if isinstance(source, BaseException):
                        self._poison_control(source)
                        raise_failures(source, (failure,))
                    self._poison_control(failure)
            if self.admission_policy == "fifo" and self.failure is None:
                # The publisher owns FIFO payload reclamation after every
                # consumer's ACK. Reclaim on this same sending connection;
                # peers never delete a live batch publication concurrently.
                for key in messages:
                    self.comm.store.delete_key(key)
        except BaseException as failure:
            # A set/get may have succeeded before interruption. Never publish
            # a later operation after uncertain peer execution or retirement.
            if isinstance(source, BaseException) and failure is not source:
                try:
                    self._poison_control(source)
                except BaseException as containment:
                    raise_failures(source, (failure, containment))
                raise_failures(source, (failure,))
            self._poison_control(failure)
            raise
        if error is not None and not self._recoverable_operation(error, body["kind"]):
            self._poison_control(error)
        if self.failure is not None:
            self.engine._abort_comm = True
            raise RuntimeError("tensor-parallel scheduler operation failed; retire all rank owners") from self.failure
        if error is not None:
            raise error
        return result

    def admit(self, stream):
        wire = self.wires.get(id(stream))
        if wire is None:
            wire = getattr(stream, "_media_wire", None)
            if wire is None:
                _check_protocol_advance(self.next_wire, "wire")
                wire, self.next_wire = self.next_wire, self.next_wire + 1
        body = {"kind": "admit", "wire": wire, "stream": self._pack(stream),
                "points": [] if self._points is None else list(self._points(stream.prompt))}
        return self._invoke(body, source=stream)

    def round(self):
        body = {"kind": "round"}
        ended = []
        prepare = getattr(self.decoder, "prepare_media_round", None)
        if prepare is not None:
            limits, failures = self._media_cpu(prepare)
            for stream, error in failures:
                from tensorfold.vision.replay import MediaBusy, MediaCancelled

                error_type = "busy" if type(error) is MediaBusy else "cancelled" if type(error) is MediaCancelled else "input"
                ended += self._invoke({"kind": "media_failure", "wire": self.wires[id(stream)],
                                       "message": str(error)[:1024], "error_type": error_type}, source=error)
            if media_requests.runtime(getattr(self.decoder, "vision", None)) is not None:
                body["media_limits"] = [[self.wires[id(stream)], limits[stream.sid]]
                                         for stream in self.decoder.filling if stream.sid in limits]
        freeze = getattr(self.decoder, "freeze_execution", None)
        if freeze is not None:
            body["execution"] = freeze()
        return ended + self._invoke(body)

    def input_readiness(self):
        """Project completed media errors before a production reservation or GPU step."""
        prepare = getattr(self.decoder, "prepare_media_round", None)
        if prepare is None:
            return [], {}
        limits, failures = self._media_cpu(prepare)
        ended = []
        for stream, error in failures:
            from tensorfold.vision.replay import MediaBusy, MediaCancelled

            error_type = "busy" if type(error) is MediaBusy else "cancelled" if type(error) is MediaCancelled else "input"
            ended += self._invoke({"kind": "media_failure", "wire": self.wires[id(stream)],
                "message": str(error)[:1024], "error_type": error_type}, source=error)
        return ended, {stream.sid: "media" for stream in self.decoder.filling
                       if stream.sid in limits and limits[stream.sid][1] == 0}

    def prepare_request(self, stream, *, wake):
        owner = media_requests.runtime(getattr(self.decoder, "vision", None))
        if owner is None or stream.vision is None:
            return True
        if not hasattr(stream, "_media_wire"):
            _check_protocol_advance(self.next_wire, "wire")
            stream._media_wire, self.next_wire = self.next_wire, self.next_wire + 1
        host = getattr(stream, "retained_host", None)
        if host is not None:
            if host.fill is None:
                return True  # Decoding consumes text anchors; completed prompt media needs no replay decode.
            def ready():
                owner.check_media_health()
                owner.work.set_wake(wake)
                request = stream._media_request
                start = host.fill.cursor
                request.prime(start)
                return request.limit(start, min(owner.profile.execution.input_rows, len(stream.prompt) - start)) > 0
            return self._media_cpu(ready)
        return self._media_cpu(lambda: media_requests.prepare(stream, owner, stream._media_wire, wake))

    def media_runnable(self):
        ready = getattr(self.decoder, "media_runnable", None)
        return True if ready is None else self._media_cpu(ready)

    def retire_request(self, stream):
        if not self._media_cpu(lambda: media_requests.retire(stream)):
            return False
        wire = self.wires.get(id(stream))
        if getattr(stream, "_retention_staged", False) and wire is not None:
            if not self._invoke({"kind": "retention_retire", "wire": wire,
                                 "error": self._request_error_record(stream.error)}, source=stream.error):
                return False
        if stream.st is None and getattr(stream, "_media_rank_owned", False):
            wire = self.wires.get(id(stream))
            if wire is not None:
                self._invoke({"kind": "media_forget", "wire": wire})
            stream._media_rank_owned = False
            self._media_cpu(lambda: media_requests.release(stream))
        return True

    def prepare_serving_profile(self, profile, request_epoch):
        return self._invoke({"kind": "serving_prepare", "profile": profile, "request_epoch": request_epoch})

    def replace_serving_profile(self):
        return self._invoke({"kind": "serving_replace"})

    def retire_admission_error(self):
        return self._invoke({"kind": "retire_admission_error"})

    def _recoverable_operation(self, error, kind):
        if isinstance(error, (NoRoom, ValueError)):
            return True
        if getattr(self.engine, "serving_profiles", None) is None or kind not in ("admit", "round"):
            return False
        from tensorfold.engine.grammar import GrammarError
        from tensorfold.server.cancellation import RequestCancelled

        namespace = BaseException.__dict__["__dict__"].__get__(error)
        return (isinstance(error, (GrammarError, RequestCancelled))
                or dict.get(namespace, "_tensorfold_request_state_retired") is True)

    def install_execution_manifest(self, relative_name):
        return self._invoke({"kind": "profile_manifest", "name": relative_name})

    def cancel(self, streams):
        return self._invoke({"kind": "cancel", "wires": [self.wires[id(stream)] for stream in streams]})

    def finish(self, streams):
        owned = []
        for stream in streams:
            wire = self.wires.get(id(stream))
            if wire is None and getattr(stream, "_media_request", None) is None:
                raise KeyError(id(stream))
            if wire is not None:
                owned.append(wire)
        # A successful drop already retired/acknowledged every rank's State and
        # wire. Its CPU owner can finish later without replaying a missing wire.
        result = self._invoke({"kind": "finish", "wires": owned}) if owned or not streams else None
        for stream in streams:
            self._media_cpu(lambda: media_requests.release(stream))  # All ACKs precede the leader's replay close.
        return result

    def drop(self, error=None):
        return self._invoke({"kind": "drop", "error": self._request_error_record(error)}, source=error)

    def shutdown(self):
        if self.comm.rank != 0:
            raise RuntimeError("only rank zero sends scheduler shutdown")
        _check_protocol_advance(self.sequence, "stop")
        for peer in range(1, self.comm.world):
            self.comm.store.set(self._key(self.sequence, peer), json.dumps(
                {"kind": "stop", "abort": self.failure is not None or self.engine._abort_comm}))
        self.sequence += 1

    def receive(self):
        from torch.distributed import DistNetworkError
        from tensorfold.cuda.store_wait import timed_out

        _check_protocol_advance(self.sequence, "receive")
        key = self._key(self.sequence, self.comm.rank)
        while True:
            try:
                self.comm.store.wait([key], timedelta(hours=1))
                break
            except DistNetworkError:
                self.engine._abort_comm = True
                return None
            except Exception as error:
                if not timed_out(error, [key], timedelta(hours=1)):
                    raise
                with self.engine._lifecycle:
                    if self.engine._closing:
                        return None
        body = json.loads(self.comm.store.get(key))
        if body["kind"] == "stop" or body.get("control", {}).get("admission_policy", "legacy") != "fifo":
            self.comm.store.delete_key(key)
        body["sequence"] = self.sequence
        self.sequence += 1
        if body["kind"] == "stop":
            self.engine._abort_comm = self.engine._abort_comm or bool(body.get("abort"))
            return None
        return body

    def apply(self, body):
        from tensorfold.cuda.rank_protocol import bounded_json

        try:
            _result, error, status = self._execute(body)
            self.comm.store.set(self._ack_key(body["sequence"], self.comm.rank),
                                bounded_json(status, separators=(",", ":")))
        except BaseException as failure:
            self._poison_control(failure)
            raise
        if error is not None and not self._recoverable_operation(error, body["kind"]):
            self._poison_control(error)
            raise RuntimeError("tensor-parallel scheduler peer failed; retire all rank owners") from error

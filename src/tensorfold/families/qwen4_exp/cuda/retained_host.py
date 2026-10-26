"""Actual CPU continuation ownership, separate from retained tensor inventory.

Same-request suspension keeps the original matcher, sampler/output ledgers and
calling-thread delivery path. It never serializes native objects or network
callbacks, and never retains a slot engine or an old State after detachment.
"""

from dataclasses import dataclass
import weakref

from tensorfold.cuda.scheduler import DeliveryOwner
from tensorfold.cuda.streams import Stream
from tensorfold.engine.grammar import Constraint
from tensorfold.engine.exact_sampling import Sampling

from .retained_catalog import LogicalReference, RetentionMismatch


@dataclass(frozen=True)
class RetentionIntent:
    """Copied standard API identities; holds no operation, DB connection or live record view."""
    target: LogicalReference | None
    parent: LogicalReference | None
    aliases: tuple[LogicalReference, ...]
    operation: str

    def __post_init__(self):
        if (self.target is not None and type(self.target) is not LogicalReference
                or self.parent is not None and type(self.parent) is not LogicalReference
                or type(self.aliases) is not tuple or len(self.aliases) > 1
                or any(type(value) is not LogicalReference for value in self.aliases)
                or type(self.operation) is not str or not self.operation
                or self.aliases and self.target is None):
            raise ValueError("retained identity must copy actual typed standard logical references")
        values = tuple(value for value in (self.target, self.parent, *self.aliases) if value is not None)
        if values and any(value.principal != values[0].principal for value in values):
            raise ValueError("retained identity cannot cross authenticated logical domains")
        if (self.target is not None and self.target.sequence.startswith("response:")
                and self.target.sequence != "response:" + self.operation
                or self.aliases and self.aliases[0].sequence != "response:" + self.operation):
            raise ValueError("retained target aliases differ from the actual new response identity")

    @classmethod
    def current(cls):
        from tensorfold.server.api_records import current_operation

        operation = current_operation()
        if operation is None or operation.read_only:
            return None
        target = (LogicalReference(operation.principal, operation.logical_sequence_id, operation.target_generation)
                  if operation.logical_retained else None)
        parent = None
        if operation.conversation_id is not None and operation.base_generation:
            parent = LogicalReference(operation.principal, "conversation:" + operation.conversation_id,
                                      operation.base_generation)
        elif operation.parent_response_id is not None:
            parent = LogicalReference(operation.principal, "response:" + operation.parent_response_id,
                                      operation.parent_generation)
        aliases = ()
        if target is not None and operation.stored and operation.conversation_id is not None:
            aliases = (LogicalReference(operation.principal, "response:" + operation.response_id, operation.target_generation),)
        return cls(target, parent, aliases, operation.response_id)

    def record(self):
        def reference(value):
            return None if value is None else [value.principal, value.sequence, value.generation]
        return {"target": reference(self.target), "parent": reference(self.parent),
                "aliases": [reference(value) for value in self.aliases], "operation": self.operation}

    @classmethod
    def from_record(cls, body):
        if (type(body) is not dict or body.keys() != {"target", "parent", "aliases", "operation"}
                or type(body["aliases"]) is not list or len(body["aliases"]) > 1
                or type(body["operation"]) is not str or not body["operation"]):
            raise ValueError("retained request identity record differs from its typed standard API schema")
        def reference(value):
            if value is None:
                return None
            if type(value) is not list or len(value) != 3:
                raise ValueError("retained logical reference has an invalid wire shape")
            return LogicalReference(*value)
        return cls(reference(body["target"]), reference(body["parent"]),
                   tuple(reference(value) for value in body["aliases"]), body["operation"])


@dataclass(frozen=True)
class FillContinuation:
    mtp: bool
    cursor: int
    stops: tuple[int, ...]


class RetainedHost:
    @classmethod
    def capture(cls, stream, *, leader, fill=None, completed=False, frozen=False):
        if type(completed) is not bool:
            raise TypeError("backend completion must be an explicit boolean")
        if (type(stream) is not Stream or stream.st is None or stream.error is not None
                or stream.done != completed):
            raise RetentionMismatch("same-request CPU retention needs its actual unresolved Stream")
        if type(leader) is not bool or fill is not None and type(fill) is not FillContinuation:
            raise ValueError("retained host ownership needs an explicit rank role and typed fill frontier")
        if fill is not None and (type(fill.mtp) is not bool or type(fill.cursor) is not int
                or fill.cursor != stream.st.pos or type(fill.stops) is not tuple
                or any(type(value) is not int for value in fill.stops)):
            raise ValueError("retained fill frontier differs from actual committed prompt progress")
        delivery = getattr(stream, "delivery_owner", None) if leader else None
        if leader:
            if type(delivery) is not DeliveryOwner:
                raise RetentionMismatch("same-request retention has no actual original delivery owner")
            if completed or frozen:
                # TP already froze this round's cancellation decision. A
                # concurrent HTTP disconnect cannot change rank participation
                # while we capture its completed backend for logical cleanup.
                if not delivery.limit.worker_active(delivery.mailbox):
                    raise RuntimeError("completed backend lost its actual worker owner before rank retirement")
            else:
                delivery.require_live()
        if stream.cancel is not None and stream.cancel.is_set():
            raise RetentionMismatch("cancelled requests cannot resume an old CPU continuation")
        if stream.snap is not None or stream.penalties is not None or stream.probabilities is not None:
            raise RetentionMismatch("this TP continuation has an unsupported opaque drafter/logprob/penalty owner")
        if stream.history is not None or stream.sampling is not None and type(stream.sampling) is not Sampling:
            raise RetentionMismatch("retained CPU request has an unsupported history or sampler owner")
        if stream.copies is not None:
            from tensorfold.families.qwen3_5.cuda.decode import CopyIndex

            if type(stream.copies) is not CopyIndex:
                raise RetentionMismatch("retained copy proposals require the actual bounded CPU suffix index")
        if stream.constraint is not None:
            if type(stream.constraint) is not Constraint:
                raise RetentionMismatch("retained grammar must be its actual owned native matcher")
            stream.constraint.retire_device_staging()
        result = cls.__new__(cls)
        result.stream, result.delivery, result.fill = stream, delivery, fill
        result._source = weakref.ref(stream.st)
        result._source_generation = (stream.st.slot_generation, stream.st.history_generation)
        result._detached = False
        result.profile = stream.st.coordinate_profile_identity
        result.position, result.mtp_len = stream.st.pos, stream.st.mtp_len
        result.mtp_drafted = stream.st.mtp_drafted
        result.closed = False
        return result

    def detach(self, state):
        """After catalog capture takes the GPU owner; readers remain catalog-owned."""
        if (self.closed or self._detached or self._source() is not state or self.stream.st is not state
                or (state.slot_generation, state.history_generation) != self._source_generation):
            raise RuntimeError("retained host cannot detach a different or already reused State")
        self.stream.st = None
        self._detached = True

    def attach(self, state, *, handoff, frozen=False):
        """After actual tensor restore; no draw, replay, matcher reset or output re-emission."""
        if self.closed:
            raise RuntimeError("retained CPU continuation is retired")
        if type(handoff) is not dict or handoff.get("host") is not self:
            raise RetentionMismatch("CPU continuation needs its actual completed catalog handoff")
        if not self._detached and (self.stream.st is not state or self._source() is not state
                or (state.slot_generation, state.history_generation) != self._source_generation):
            raise RuntimeError("resident CPU continuation belongs to a different or reused State")
        if self._detached and self.stream.st is not None:
            raise RuntimeError("retained CPU continuation is already attached to another State")
        if self.delivery is not None:
            if frozen:
                if not self.delivery.limit.worker_active(self.delivery.mailbox):
                    raise RuntimeError("live continuation lost its original backend worker before rank retirement")
            else:
                self.delivery.require_live()
        if self.stream.cancel is not None and self.stream.cancel.is_set():
            raise RetentionMismatch("the original request was cancelled while suspended")
        if (state.coordinate_profile_identity != self.profile or state.pos != self.position
                or state.mtp_len != self.mtp_len or state.mtp_drafted != self.mtp_drafted):
            raise RetentionMismatch("restored CPU continuation differs from the actual target/draft frontiers")
        self.stream.st = state
        return self.stream, self.fill

    def close(self):
        """Drop only this CPU attachment; caller/request and tensor owners retire independently."""
        if not self.closed:
            self.closed = True
            self.stream = self.delivery = self.fill = self._source = None

    def retire_backend(self):
        """Positive old-job retirement for a fresh subrun under one pending API target."""
        if self.closed:
            return
        if self.delivery is not None:
            if self.delivery.limit.worker_active(self.delivery.mailbox):
                raise RuntimeError("the previous backend job still owns its CPU/GPU continuation")
        elif not self.stream.done and self.stream.error is None:
            raise RuntimeError("the peer's previous backend continuation has not retired")
        if self.stream.constraint is not None:
            self.stream.constraint.retire_device_staging()
        self.stream.st = None  # The catalog still owns an optional resident window.
        self.close()

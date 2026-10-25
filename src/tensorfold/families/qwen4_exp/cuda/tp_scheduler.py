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
from tensorfold.cleanup import raise_failures
from .owner_execution import FAMILY_FIELDS


class SharedMemoryGate(MemoryGate):
    """All ranks take the same growth decision; ordinary usage reads stay local."""

    def __init__(self, original, comm):
        self.comm = comm
        room = self._minimum(original.room)
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
        decoder.arrived = lambda: self._arrival
        decoder._pass_rows = lambda: self._rows
        decoder.prefix_decisions = self._prefix_decisions
        decoder.poison_control = self._poison_control

    def __getattr__(self, name):
        return getattr(self.decoder, name)

    def _key(self, sequence, peer):
        return f"tensorfold/flashnext/round/{sequence}/rank/{peer}"

    def _ack_key(self, sequence, peer):
        return f"tensorfold/flashnext/round/{sequence}/ack/{peer}"

    def _poison_control(self, error):
        first = self.failure is None
        if self.failure is None:
            self.failure = error
        self.engine._abort_comm = True
        if first:
            from tensorfold.cuda.cohort import notify_failure

            try:
                notify_failure()
            except BaseException as notification:
                raise_failures(error, [notification])

    def _remember(self, wire, stream):
        self.requests[wire] = stream
        self.wires[id(stream)] = wire
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

    @staticmethod
    def _pack(stream):
        from tensorfold.engine.grammar import pack

        sampling = stream.sampling
        if stream.vision is not None or stream.probabilities is not None or stream.penalties is not None:
            raise ValueError("tensor-parallel Flash Next requests do not support vision, logprobs or penalties")
        return {"prompt": stream.prompt, "count": stream.count, "draft": stream.draft,
                "stop_eos": stream.stop_eos, "background": stream.background,
                "sampling": None if sampling is None else [sampling.seed, sampling.temperature, sampling.top_k,
                                                           sampling.top_p, sampling.min_p],
                "grammar": pack(stream.constraint), "owed": stream.owed, "carry": stream.carry,
                "token_limit": stream.token_limit is not None}

    def _unpack(self, body):
        from tensorfold.engine.exact_sampling import Sampling

        constraint = None
        if body["grammar"]:
            from tensorfold.engine import grammar

            constraint = grammar.compiler(self.engine, self.engine.model_dir, self.engine.eos).follow(body["grammar"])
        stream = Stream(list(body["prompt"]), int(body["count"]),
                      None if body["sampling"] is None else Sampling(*body["sampling"]),
                      draft=bool(body["draft"]), stop_eos=bool(body["stop_eos"]),
                      background=bool(body["background"]), constraint=constraint,
                      owed=list(body["owed"]), carry=body["carry"])
        stream._tp_token_limit = bool(body.get("token_limit", False))
        return stream

    def _prefix_decisions(self, decisions, error=None):
        """One leader-owned stop decision, before any peer publishes state.

        CPU metadata uses the existing ordered store; no hot GPU allocation or
        extra device-wide fence is introduced. Operation acknowledgments bound
        the lifetime of the peer-owned keys.
        """
        if self.prefix_operation is None:
            raise RuntimeError("stop decisions require an ordered TP scheduler operation")
        step = self.prefix_sequence
        self.prefix_sequence += 1
        def key(peer):
            return f"tensorfold/flashnext/prefix/{self.prefix_operation}/{step}/rank/{peer}"
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
                     if stream.st is None and stream.cancel.snapshot()]
        result = {"forget": forgotten,
                  "cancel": [[wire, stream.cancel.snapshot()] for wire, stream in self.requests.items()
                             if wire not in forgotten],
                  "arrived": bool(self.arrived())}
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
        result["pass_rows"] = self._pass_rows() if quota is None else quota(boundary["settings"])
        return result

    def _apply_controls(self, body):
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
        return hashlib.sha256(json.dumps([state, execution], sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _dispatch(self, body, source=None):
        self.prefix_operation = body["sequence"]
        self.prefix_sequence = 0
        self._apply_controls(body["control"])
        kind = body["kind"]
        if kind == "admit":
            wire = body["wire"]
            if wire not in self.requests:
                self._remember(wire, source if source is not None else self._unpack(body["stream"]))
            stream = self.requests[wire]
            original = self.decoder.points
            self.decoder.points = lambda prompt: body["points"]
            try:
                return self.decoder.admit(stream)
            except BaseException as error:
                if not isinstance(error, NoRoom) or not self.decoder.live():
                    self._forget(wire)
                raise
            finally:
                self.decoder.points = original
        if kind == "round":
            if "execution" in body:
                self.decoder.accept_execution(body["execution"])
            return self.decoder.round()
        if kind == "profile_manifest":
            return self.decoder.install_execution_manifest(body["name"])
        if kind in ("cancel", "finish"):
            streams = [self.requests[wire] for wire in body["wires"]]
            result = getattr(self.decoder, kind)(streams)
            if kind == "finish":
                for wire in body["wires"]:
                    self._forget(wire)
            return result
        if kind == "drop":
            result = self.decoder.drop()
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
        finally:
            for stream in self.requests.values():
                stream.cancel.active = False
        status = {"error": None if error is None else type(error).__name__, "state": self._state_digest()}
        return result, error, status

    def _invoke(self, body, source=None):
        if self.failure is not None:
            raise RuntimeError("tensor-parallel scheduler is poisoned; retire the rank cohort") from self.failure
        if self.comm.rank != 0:
            raise RuntimeError("only rank zero submits tensor-parallel scheduler operations")
        body["control"] = self._controls(body.get("execution"))
        sequence = self.sequence
        self.sequence += 1
        body["sequence"] = sequence
        text = json.dumps(body, separators=(",", ":"))
        try:
            messages = [self._key(sequence, peer) for peer in range(1, self.comm.world)]
            set_many = getattr(self.comm.store, "multi_set", None)
            if callable(set_many):
                set_many(messages, [text] * len(messages))
            else:
                for key in messages:
                    self.comm.store.set(key, text)
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
                remote = json.loads(payload)
                self.comm.store.delete_key(key)
                if remote != status:
                    self._poison_control(RuntimeError(
                        f"tensor-parallel scheduler diverged on rank {peer}: {remote} != {status}"))
        except BaseException as failure:
            # A set/get may have succeeded before interruption. Never publish
            # a later operation after uncertain peer execution or retirement.
            self._poison_control(failure)
            raise
        if error is not None and not isinstance(error, (NoRoom, ValueError)):
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
            wire, self.next_wire = self.next_wire, self.next_wire + 1
        body = {"kind": "admit", "wire": wire, "stream": self._pack(stream),
                "points": [] if self._points is None else list(self._points(stream.prompt))}
        return self._invoke(body, source=stream)

    def round(self):
        body = {"kind": "round"}
        freeze = getattr(self.decoder, "freeze_execution", None)
        if freeze is not None:
            body["execution"] = freeze()
        return self._invoke(body)

    def install_execution_manifest(self, relative_name):
        return self._invoke({"kind": "profile_manifest", "name": relative_name})

    def cancel(self, streams):
        return self._invoke({"kind": "cancel", "wires": [self.wires[id(stream)] for stream in streams]})

    def finish(self, streams):
        return self._invoke({"kind": "finish", "wires": [self.wires[id(stream)] for stream in streams]})

    def drop(self):
        return self._invoke({"kind": "drop"})

    def shutdown(self):
        if self.comm.rank != 0:
            raise RuntimeError("only rank zero sends scheduler shutdown")
        for peer in range(1, self.comm.world):
            self.comm.store.set(self._key(self.sequence, peer), json.dumps(
                {"kind": "stop", "abort": self.failure is not None or self.engine._abort_comm}))
        self.sequence += 1

    def receive(self):
        from torch.distributed import DistNetworkError
        from tensorfold.cuda.store_wait import timed_out

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
        self.comm.store.delete_key(key)
        body["sequence"] = self.sequence
        self.sequence += 1
        if body["kind"] == "stop":
            self.engine._abort_comm = self.engine._abort_comm or bool(body.get("abort"))
            return None
        return body

    def apply(self, body):
        try:
            _result, error, status = self._execute(body)
            self.comm.store.set(self._ack_key(body["sequence"], self.comm.rank), json.dumps(status))
        except BaseException as failure:
            self._poison_control(failure)
            raise
        if error is not None and not isinstance(error, (NoRoom, ValueError)):
            self._poison_control(error)
            raise RuntimeError("tensor-parallel scheduler peer failed; retire all rank owners") from error

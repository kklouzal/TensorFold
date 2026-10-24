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
        decoder.arrived = lambda: self._arrival
        decoder._pass_rows = lambda: self._rows

    def __getattr__(self, name):
        return getattr(self.decoder, name)

    def _key(self, sequence, peer):
        return f"tensorfold/flashnext/round/{sequence}/rank/{peer}"

    def _ack_key(self, sequence, peer):
        return f"tensorfold/flashnext/round/{sequence}/ack/{peer}"

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
                "grammar": pack(stream.constraint), "owed": stream.owed, "carry": stream.carry}

    def _unpack(self, body):
        from tensorfold.engine.exact_sampling import Sampling

        constraint = None
        if body["grammar"]:
            from tensorfold.engine import grammar

            constraint = grammar.compiler(self.engine, self.engine.model_dir, self.engine.eos).follow(body["grammar"])
        return Stream(list(body["prompt"]), int(body["count"]),
                      None if body["sampling"] is None else Sampling(*body["sampling"]),
                      draft=bool(body["draft"]), stop_eos=bool(body["stop_eos"]),
                      background=bool(body["background"]), constraint=constraint,
                      owed=list(body["owed"]), carry=body["carry"])

    def _controls(self):
        forgotten = [wire for wire, stream in self.requests.items()
                     if stream.st is None and stream.cancel.snapshot()]
        return {"forget": forgotten,
                "cancel": [[wire, stream.cancel.snapshot()] for wire, stream in self.requests.items()
                           if wire not in forgotten],
                "arrived": bool(self.arrived()), "pass_rows": self._pass_rows()}

    def _apply_controls(self, body):
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
                                                  getattr(stream.st, "cur", None)],
                  None if stream.error is None else type(stream.error).__name__)
                 for wire, stream in self.requests.items()]
        return hashlib.sha256(json.dumps(state, separators=(",", ":")).encode()).hexdigest()

    def _dispatch(self, body, source=None):
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
            return self.decoder.round()
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
        body["control"] = self._controls()
        sequence = self.sequence
        self.sequence += 1
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
                    self.failure = RuntimeError(f"tensor-parallel scheduler diverged on rank {peer}: {remote} != {status}")
        except BaseException as failure:
            # A set/get may have succeeded before interruption. Never publish
            # a later operation after uncertain peer execution or retirement.
            self.failure = failure
            self.engine._abort_comm = True
            raise
        if error is not None and not isinstance(error, (NoRoom, ValueError)):
            self.failure = error
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
        return self._invoke({"kind": "round"})

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
        _result, error, status = self._execute(body)
        self.comm.store.set(self._ack_key(body["sequence"], self.comm.rank), json.dumps(status))
        if error is not None and not isinstance(error, (NoRoom, ValueError)):
            self.failure = error
            self.engine._abort_comm = True
            raise RuntimeError("tensor-parallel scheduler peer failed; retire all rank owners") from error

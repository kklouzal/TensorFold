"""Bounded draft windows and accepted-target-feature transactions for DFlash2.

Target features are never archived beyond one bounded execution frame. The
committed FP16 ring is updated only after the target chooses its accepted prefix.
Proposal KV is ephemeral and cannot overwrite entries needed after rejection.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import uuid
import weakref

import torch

from .dflash2_features import FeatureFrame
from .dflash2_provider import DFlash2Provider, bind_target
from .dflash2_weights import load_shard


class DraftWindow:
    def __init__(self, config, *, tp, context, device, generation=0, profile_identity=None):
        if type(context) is not int or not 1 <= context <= config.context:
            raise ValueError("DFlash2 context must fit its declared absolute positions")
        if profile_identity is not None and (not isinstance(profile_identity, str) or not 1 <= len(profile_identity) <= 256):
            raise ValueError("DFlash2 window profile identity must be a bounded string")
        local = config.partition(*tp)
        self.config = config
        self.context = context
        self.profile_identity = profile_identity if profile_identity is not None else uuid.uuid4().hex
        self.capacity = min(config.window, context)
        self.kv = torch.empty((2, config.layers, self.capacity, local["kv_heads"], config.head_dim),
                              dtype=torch.float16, device=device)
        self.end = self.available = 0
        self.generation, self.pending, self.poisoned, self.closed = generation, None, False, False
        self.enabled = True

    @property
    def nbytes(self):
        return self.kv.numel() * self.kv.element_size()

    def check(self, *, require_resolved=True):
        if self.closed or self.poisoned:
            raise ValueError("DFlash2 draft state is poisoned and must be exclusively retired")
        if require_resolved and self.pending is not None:
            raise ValueError("DFlash2 target features await an accepted-prefix decision")

    def append(self, values, start):
        if self.closed or self.poisoned or start != self.end or values.shape[:2] != self.kv.shape[:2] or values.shape[3:] != self.kv.shape[3:]:
            raise ValueError("DFlash2 ring append lost the committed absolute frontier or tensor geometry")
        rows = values.shape[2]
        if start + rows > self.context:
            raise ValueError("DFlash2 conditioning append exceeds its absolute position limit")
        keep = min(rows, self.capacity)
        if keep:
            positions = torch.arange(start + rows - keep, start + rows, device=self.kv.device, dtype=torch.int64)
            try:
                self.kv.index_copy_(2, positions.remainder(self.capacity), values[:, :, -keep:])
            except BaseException:
                self.poisoned = True
                raise
        self.end += rows
        self.available = min(self.capacity, self.available + rows)

    def layer(self, index):
        if self.closed or self.poisoned:
            raise ValueError("DFlash2 cannot read a closed or poisoned window")
        positions = torch.arange(self.end - self.available, self.end, dtype=torch.int64, device=self.kv.device)
        return self.kv[:, index].index_select(1, positions.remainder(self.capacity)), positions

    def snapshot_payload(self, values, position):
        return {"schema": "dflash2-f16-window-v1", "config": self.config.identity,
                "profile": self.profile_identity,
                "end": position, "available": min(self.capacity, position), "generation": self.generation,
                "kv": values}

    def snapshot(self, position=None):
        if self.closed or self.poisoned:
            raise ValueError("DFlash2 cannot snapshot a closed or poisoned window")
        position = self.end if position is None else position
        if type(position) is not int or not 0 <= position <= self.end:
            raise ValueError("DFlash2 committed snapshot position is outside the current frontier")
        start = max(0, position - self.capacity)
        if start < self.end - self.available:
            raise ValueError("DFlash2 complete earlier window is no longer available; use its saved snapshot")
        positions = torch.arange(start, position, device=self.kv.device, dtype=torch.int64)
        return self.snapshot_payload(self.kv.index_select(2, positions.remainder(self.capacity)), position)

    def validate_snapshot(self, snapshot, position=None):
        """Host metadata/type validation; no mutation, device allocation or sync."""
        if self.closed or self.poisoned or not isinstance(snapshot, dict):
            raise ValueError("DFlash2 snapshot needs a live unpoisoned destination")
        end, available, values = snapshot.get("end"), snapshot.get("available"), snapshot.get("kv")
        if (snapshot.get("schema") != "dflash2-f16-window-v1" or snapshot.get("config") != self.config.identity
                or snapshot.get("profile") != self.profile_identity or type(end) is not int
                or not 0 <= end <= self.context or type(available) is not int or available != min(self.capacity, end)
                or not isinstance(values, torch.Tensor) or values.dtype != torch.float16
                or values.shape != (*self.kv.shape[:2], available, *self.kv.shape[3:])
                or type(snapshot.get("generation")) is not int or not 0 <= snapshot["generation"] < (1 << 63)
                or position is not None and (type(position) is not int or end != position)):
            raise ValueError("DFlash2 snapshot does not match its allocated window/profile/target position")

    def restore(self, snapshot):
        self.check()
        self.validate_snapshot(snapshot)
        positions = torch.arange(snapshot["end"] - snapshot["available"], snapshot["end"], device=self.kv.device, dtype=torch.int64)
        try:
            self.kv.index_copy_(2, positions.remainder(self.capacity), snapshot["kv"])
        except BaseException:
            self.poisoned = True
            raise
        self.end, self.available = snapshot["end"], snapshot["available"]
        # Target slot identity is the destination's, never the stale snapshot's.

    def reset(self, generation):
        if self.closed:
            raise ValueError("DFlash2 reset cannot revive a retired window")
        if self.pending is not None:
            raise ValueError("DFlash2 pending readers must retire before reset")
        self.end = self.available = 0
        self.generation = generation
        self.poisoned = False


@dataclass
class PendingFeatures:
    frame: FeatureFrame
    generation: int
    slot_generation: int
    start: int
    values: torch.Tensor


class DFlash2Runtime:
    target_capturable = False  # Capture needs explicit per-frame transactions; eager is canonical initially.
    capturable = False

    def __init__(self, target=None, plan=None, *, context=None, embedding=None, head=None, profile_identity=None, pairing=None):
        self.target = self.plan = self.context = self.weights = self.provider = None
        self._loading_tensors = {}
        self.closed = False
        self._states, self._frames = [], []
        self._profile_identity = profile_identity
        self._context_identity = uuid.uuid4().hex
        if target is not None:
            self.load(target, plan, context=context, embedding=embedding, head=head,
                      profile_identity=profile_identity, pairing=pairing)

    def load(self, target, plan, *, context, embedding=None, head=None, profile_identity=None, pairing=None):
        if self.closed or self.target is not None or getattr(target, "dflash2", None) not in (None, self):
            raise ValueError("DFlash2 runtime loading is one-shot with one target owner")
        if type(context) is not int or not 1 <= context <= plan.config.context:
            raise ValueError("DFlash2 selected context exceeds the draft's supported absolute positions")
        if getattr(target, "dspark", None) is not None or getattr(target, "mtp", None) is not None:
            raise ValueError("DFlash2 requires retirement of the previous speculative method")
        if torch.device(target.device).type == "cuda" and pairing is None:
            raise ValueError("DFlash2 CUDA allocation requires pre-load pairing/content authority")
        if pairing is not None:
            from .dflash2_manifest import PairingBinding
            if not isinstance(pairing, PairingBinding) or pairing.plan != plan:
                raise ValueError("DFlash2 pairing binding differs from the actual source plan")
            pairing.verify_unchanged()
            profile_identity = pairing.identity if profile_identity is None else profile_identity
        binding = bind_target(plan.config, target, plan.tp)
        self.target, self.plan, self.context = target, plan, context
        if profile_identity is not None:
            if not isinstance(profile_identity, str) or not 1 <= len(profile_identity) <= 256:
                raise ValueError("DFlash2 numerical profile identity must be a bounded nonempty string")
            self._profile_identity = profile_identity
        target.dflash2 = self  # published owner before any fallible device allocation
        self.weights = load_shard(plan, device=target.device, destination=self._loading_tensors,
                                  source_authority=None if pairing is None else pairing.authority)
        self.provider = DFlash2Provider(self.weights, target, embedding=embedding, head=head, _binding=binding)
        return self

    @property
    def config(self):
        return self.plan.config

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self._loading_tensors.values())

    @property
    def identity(self):
        return None if self.weights is None else self.weights.identity

    def device_tensors(self):
        return tuple(self._loading_tensors.values())

    def prepare_buffers(self, buffers):
        if getattr(buffers, "dflash2_features", None) is not None:
            raise ValueError("DFlash2 feature buffers are prepared once")
        buffers.dflash2_features = FeatureFrame(self.config, buffers.rows, self.target.device,
                                                target_dtype=getattr(getattr(buffers, "h", None), "dtype", torch.float16))
        self._frames.append(weakref.ref(buffers.dflash2_features))
        return buffers.dflash2_features

    def init_state(self, target_state):
        if getattr(target_state, "dflash2", None) is not None:
            raise ValueError("DFlash2 target slot already owns draft state")
        state = DraftWindow(self.config, tp=self.plan.tp, context=self.context, device=self.target.device,
                            generation=getattr(target_state, "slot_generation", 0),
                            profile_identity=hashlib.sha256((self._context_identity + "|" + str(self._profile_identity)
                                + "|" + str(self.plan.tp) + "|" + self.identity).encode()).hexdigest())
        target_state.dflash2 = state
        self._states = [ref for ref in self._states if ref() is not None]
        self._states.append(weakref.ref(state))
        return state

    def begin(self, segs, buffers):
        if self.closed:
            raise ValueError("DFlash2 runtime is closed")
        active = [(target, lo, hi) for target, lo, hi in segs if target.dflash2.enabled]
        if not active:
            return None
        for target, _, _ in active:
            state = target.dflash2
            state.check()
            if state.end != target.pos or state.generation != getattr(target, "slot_generation", 0):
                raise ValueError("DFlash2 target/draft frontier or slot generation differs")
        frame = getattr(buffers, "dflash2_features", None)
        if frame is None:
            raise ValueError("DFlash2 bounded capture buffers were not prepared before execution")
        return frame.begin(active, buffers)

    def finish(self, capture, segs):
        if capture is None:
            return
        segs = [(target, lo, hi) for target, lo, hi in segs if target.dflash2.enabled]
        features = capture.finish()
        if capture.owner.readers or any(target.dflash2.pending is not None for target, _, _ in segs):
            raise ValueError("DFlash2 target frame was already published")
        if len(segs) != len(capture.segs) or any(
                target is not original or (lo, hi) != (a, z)
                for (target, lo, hi), (original, a, z) in zip(segs, capture.segs)):
            raise ValueError("DFlash2 target segment identities changed during capture")
        for target, lo, hi in segs:
            state = target.dflash2
            state.pending = PendingFeatures(capture.owner, capture.generation, state.generation, state.end, features[lo:hi])
            capture.owner.readers += 1

    def abort(self, _capture, segs):
        for target, _, _ in segs:
            self.discard(target)

    def _pending(self, target, rows):
        state = target.dflash2
        if state.closed or state.poisoned:
            raise ValueError("DFlash2 accepted-prefix work cannot use a closed or poisoned window")
        pending = state.pending
        if pending is None or type(rows) is not int or not 0 <= rows <= len(pending.values):
            raise ValueError("DFlash2 commit requires a staged permitted accepted prefix")
        if (pending.generation != pending.frame.generation or pending.slot_generation != state.generation
                or state.generation != getattr(target, "slot_generation", 0) or pending.start != state.end):
            raise ValueError("DFlash2 accepted-prefix feature transaction is stale")
        return state, pending

    def validate_commit(self, target, rows):
        if target.dflash2.enabled:
            self._pending(target, rows)

    def _checked_materialize(self, features, positions, states):
        try:
            values = self.provider.materialize(features, positions)
            good = torch.isfinite(values).all().to(torch.int32).reshape(1)
            if self.target.comm is not None:
                peers = torch.empty(self.target.comm.world, dtype=torch.int32, device=self.target.device)
                self.target.comm.all_gather(good, peers)
                good = peers.min()
            if not bool(good):
                raise ValueError("DFlash2 materialized KV exceeds its finite FP16 contract")
            return values
        except BaseException:
            for state in states:
                state.poisoned = True
            raise

    def commit_many(self, accepted):
        if len({id(target) for target, _ in accepted}) != len(accepted):
            raise ValueError("DFlash2 accepted batch repeats a target sequence")
        records = [(target, rows, *self._pending(target, rows)) for target, rows in accepted]
        # Group by first logical occurrence, never process/address/set order:
        # numeric-status collectives must be identical on every TP rank.
        frames = tuple(dict.fromkeys(id(pending.frame) for _, _, _, pending in records))
        for frame in frames:
            live = [(target, rows, state, pending) for target, rows, state, pending in records
                    if rows and id(pending.frame) == frame]
            if not live:
                continue
            features = torch.cat([pending.values[:rows] for _, rows, _, pending in live])
            positions = torch.tensor([position for _, rows, state, _ in live for position in range(state.end, state.end + rows)],
                                     dtype=torch.int64, device=self.target.device)
            values = self._checked_materialize(features, positions, [state for _, _, state, _ in live])
            start = 0
            try:
                for _target, rows, state, _pending in live:
                    state.append(values[:, :, start:start + rows], state.end)
                    start += rows
            except BaseException:
                for _, _, state, _ in records:
                    state.poisoned = True
                raise
        for target, _rows, _state, _pending in records:
            self.discard(target)

    def commit(self, target, accepted_rows):
        if not target.dflash2.enabled:
            return
        return self.commit_many([(target, accepted_rows)])

    def discard(self, target):
        state = getattr(target, "dflash2", None)
        if state is not None and state.pending is not None:
            state.pending.frame.readers -= 1
            if state.pending.frame.readers < 0:
                raise RuntimeError("DFlash2 feature frame reader accounting underflow")
            state.pending = None

    def reset(self, target):
        self.discard(target)
        target.dflash2.reset(getattr(target, "slot_generation", 0))
        target.dflash2.enabled = True

    def set_enabled(self, target, enabled):
        """Bind one admitted request's method participation after reset/restore.

        Enabling cannot reconstruct a skipped history. The caller must have
        reset to zero or restored a valid active-method snapshot first.
        """
        state = target.dflash2
        state.check()
        if type(enabled) is not bool or enabled and state.end != target.pos:
            raise ValueError("DFlash2 activation requires valid conditioning at the actual target frontier")
        state.enabled = enabled

    def snapshot(self, target, position=None):
        state = target.dflash2
        if state.closed or state.poisoned:
            raise ValueError("DFlash2 cannot snapshot a closed or poisoned window")
        position = target.pos if position is None else position
        if type(position) is not int or not 0 <= position <= self.context:
            raise ValueError("DFlash2 snapshot position exceeds its context commitment")
        if not state.enabled:
            return {"schema": "dflash2-inactive-v1", "config": self.config.identity,
                    "profile": state.profile_identity, "generation": state.generation, "target_end": position}
        if position <= state.end:
            return state.snapshot(position)
        rows = position - state.end
        _state, pending = self._pending(target, rows)
        start = max(0, position - state.capacity)
        pieces = []
        old_end = min(state.end, position)
        if start < old_end:
            if start < state.end - state.available:
                raise ValueError("DFlash2 cut needs retired draft history")
            indices = torch.arange(start, old_end, device=self.target.device, dtype=torch.int64)
            pieces.append(state.kv.index_select(2, indices.remainder(state.capacity)))
        fresh_start = max(start, state.end)
        if fresh_start < position:
            lo, hi = fresh_start - state.end, position - state.end
            positions = torch.arange(fresh_start, position, device=self.target.device, dtype=torch.int64)
            pieces.append(self._checked_materialize(pending.values[lo:hi], positions, [state]))
        values = pieces[0] if len(pieces) == 1 else torch.cat(pieces, 2)
        return state.snapshot_payload(values, position)

    def validate_snapshot(self, target, snapshot, position=None):
        state = getattr(target, "dflash2", None)
        if state is None:
            raise ValueError("DFlash2 snapshot has no allocated destination window")
        if isinstance(snapshot, dict) and snapshot.get("schema") == "dflash2-inactive-v1":
            if (state.enabled or state.closed or state.poisoned or set(snapshot) != {
                    "schema", "config", "profile", "generation", "target_end"}
                    or snapshot["config"] != self.config.identity or snapshot["profile"] != state.profile_identity
                    or type(snapshot["generation"]) is not int or not 0 <= snapshot["generation"] < (1 << 63)
                    or type(snapshot["target_end"]) is not int or not 0 <= snapshot["target_end"] <= self.context
                    or position is not None and snapshot["target_end"] != position):
                raise ValueError("inactive DFlash2 snapshot cannot supply enabled method conditioning")
            return
        if not state.enabled:
            raise ValueError("inactive DFlash2 state requires an explicit valid conditioning reconstruction")
        state.validate_snapshot(snapshot, position)

    def validate_copy(self, destination, source, position):
        if (getattr(destination, "dflash2_runtime", None) is not self
                or getattr(source, "dflash2_runtime", None) is not self
                or destination.dflash2.profile_identity != source.dflash2.profile_identity
                or type(position) is not int or not 0 <= position <= min(destination.limit, source.pos, self.context)):
            raise ValueError("DFlash2 prefix copy requires the same live numerical/runtime profile")
        destination.dflash2.check()
        source.dflash2.check()
        if not source.dflash2.enabled or not destination.dflash2.enabled:
            raise ValueError("DFlash2 prefix borrowing requires valid enabled method state")

    def restore(self, target, snapshot):
        self.validate_snapshot(target, snapshot, target.pos)
        self.discard(target)
        if not target.dflash2.enabled:
            target.dflash2.reset(getattr(target, "slot_generation", 0))
            return
        target.dflash2.restore(snapshot)
        target.dflash2.generation = getattr(target, "slot_generation", 0)

    def clone_state(self, source, destination):
        if getattr(destination, "dflash2", None) is source.dflash2:
            raise ValueError("DFlash2 clone destination cannot alias its source window")
        if getattr(destination, "dflash2", None) is None:
            self.init_state(destination)
        destination.dflash2.enabled = source.dflash2.enabled
        self.restore(destination, self.snapshot(source))

    def propose_many(self, targets, anchors, proposal_counts, samplings, first_positions):
        from tensorfold.engine.rejection import ChainPacket

        if not (len(targets) == len(anchors) == len(proposal_counts) == len(samplings) == len(first_positions)):
            raise ValueError("DFlash2 runtime proposal batch lengths differ")
        if len({id(target) for target in targets}) != len(targets):
            raise ValueError("DFlash2 proposal batch repeats one logical sequence")
        result = [ChainPacket((), (), position) for position in first_positions]
        live = []
        for index, (target, requested) in enumerate(zip(targets, proposal_counts)):
            state = target.dflash2
            state.check()
            if not state.enabled:
                raise ValueError("an inactive DFlash2 request cannot construct proposals")
            if state.end != target.pos or state.generation != getattr(target, "slot_generation", 0):
                raise ValueError("DFlash2 warm proposal conditioning is stale")
            if first_positions[index] != state.end + 1:
                raise ValueError("DFlash2 proposal RNG/output position differs from the pending anchor")
            if type(requested) is not int or not 0 <= requested < self.config.block:
                raise ValueError("DFlash2 requested proposals must be zero through three")
            if requested and state.end + self.config.block <= self.config.context:
                live.append(index)
        if live:
            packets = self.provider.propose_many([targets[i].dflash2 for i in live], [anchors[i] for i in live],
                [proposal_counts[i] for i in live], [samplings[i] for i in live], [first_positions[i] for i in live])
            for index, packet in zip(live, packets):
                result[index] = packet
        return result

    def propose(self, target, anchor, proposals, sampling, first_position):
        return self.propose_many([target], [anchor], [proposals], [sampling], [first_position])[0]

    def record(self, target):
        state = target.dflash2
        return {"method": "dflash2", "enabled": state.enabled, "draft_identity": self.identity,
                "snapshot_scope": "current-runtime-numerical-profile-rank" if self._profile_identity is not None else "current-runtime-only",
                "draft_valid_length": state.end if state.enabled else None,
                "retained_window_rows": state.available if state.enabled else 0,
                "physical_window_rows": state.capacity, "owned_window_bytes": state.nbytes,
                "poisoned": state.poisoned,
                "slot_generation": state.generation,
                "pending_feature_rows": 0 if state.pending is None else len(state.pending.values)}

    def close_state(self, target):
        self.discard(target)
        if getattr(target, "dflash2", None) is not None:
            target.dflash2.closed = True
            target.dflash2.kv = None
        target.dflash2 = None

    def _feature_reader_owners(self):
        """Validate the complete live catalogue without changing any claim."""
        states = [state for ref in self._states if (state := ref()) is not None]
        frames = [frame for ref in self._frames if (frame := ref()) is not None]
        for frame in frames:
            readers = sum(state.pending is not None and state.pending.frame is frame for state in states)
            if type(frame.readers) is not int or frame.readers != readers:
                raise RuntimeError("DFlash2 retirement retains an unowned feature reader")
        for state in states:
            pending = state.pending
            if pending is not None:
                if all(pending.frame is not frame for frame in frames):
                    raise RuntimeError("DFlash2 retirement lost its feature frame owner")
        return states, frames

    def retire_request_after_fence(self, states, *, fenced):
        """Settle only affected request claims after actual cleanup and its fence.

        The actor retains primary error trees/tracebacks until this operation
        succeeds. An empty tuple covers a refusal before slot acquisition.
        Known unrelated readers/windows remain intact; uncertain ownership or
        failed slot cleanup refuses retirement and requires fatal containment.
        """
        if fenced is not True or type(states) is not tuple or self.closed:
            raise ValueError("request retirement requires a live runtime, bounded State tuple and completed fence")
        affected = []
        try:
            owned = [state for ref in self._states if (state := ref()) is not None]
            if len(states) > len(owned):
                raise ValueError("request retirement exceeds its owned window catalogue")
            for target in states:
                window = getattr(target, "dflash2", None)
                if (getattr(target, "dflash2_runtime", None) is not self
                        or all(window is not candidate for candidate in owned)
                        or any(window is candidate for candidate in affected)):
                    raise ValueError("request retirement lost its unique runtime/window owner")
                affected.append(window)
                if (window.closed or window.poisoned or window.kv is None
                        or window.generation != target.slot_generation
                        or any(getattr(target, name, None) is not None for name in (
                            "_resize_error", "_resize_frames_pending", "_history_error", "_kv_error"))):
                    raise ValueError("request retirement retains an invalid or unsuccessfully reset State")
                pending = window.pending
                if (window.enabled and window.end != target.pos
                        or not window.enabled and pending is not None):
                    raise ValueError("request retirement has not resolved its target/draft frontier")
                if pending is not None and (
                        pending.generation != pending.frame.generation
                        or pending.slot_generation != window.generation or pending.start != window.end):
                    raise ValueError("request retirement retains a stale feature claim")
            self._feature_reader_owners()
            for target in states:
                self.discard(target)
            self._feature_reader_owners()
        except BaseException:
            for window in affected:
                window.poisoned = True
            raise

    def retire_after_fence(self, *, fenced):
        """Retire constructor/failed-frame owners after coordinated quiescence.

        The engine/structural construction owner retains every traceback and
        partial Buffer/State until its device fence AND this retirement finish.
        Unknown feature readers retain the catalogue for fatal containment;
        callers may not clear constructor frames on that path.
        """
        if fenced is not True:
            raise ValueError("DFlash2 retirement requires the owning coordinated device fence")
        try:
            states, _frames = self._feature_reader_owners()
            for state in states:
                if state.pending is not None:
                    state.pending.frame.readers -= 1
                    state.pending = None
                state.closed = True
                state.kv = None
        except BaseException:
            for ref in self._states:
                state = ref()
                if state is not None:
                    state.poisoned = True
            raise

    def close(self):
        states = [state for ref in self._states if (state := ref()) is not None]
        frames = [frame for ref in self._frames if (frame := ref()) is not None]
        if any(not state.closed for state in states):
            raise RuntimeError("DFlash2 live windows must retire before their weight owner")
        if any(frame.readers for frame in frames):
            raise RuntimeError("DFlash2 feature readers must retire before their weight owner")
        self.closed = True
        if self.target is not None and getattr(self.target, "dflash2", None) is self:
            self.target.dflash2 = None
        self.target = None
        self.provider = None
        self.weights = None
        self._loading_tensors.clear()

# SPDX-License-Identifier: Apache-2.0
# Block/Markov ordering follows SGLang v0.5.21 (Apache-2.0),
# speculative/dspark_components/{dspark_draft,dspark_planner}.py.
"""Unregistered DSpark provider contract for the existing Flash Next engine.

CPU, unsharded FP16 reference only. Target callbacks keep shared embedding/head
ownership. The caller supplies its existing target State, commits verified rows
first, then publishes exactly those tap-derived context rows here. Proposal
sampling receives a callback; target rejection/acceptance remains in the engine.
Each state is owned by one existing decoder worker; simultaneous mutation of the
same state is unsupported. Target callbacks own and budget their existing
workspace separately. No CLI path, target forward/state class or serving default
is changed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol

import torch

from . import dspark_reference as ref
from .dspark_weights import tensor_shapes


@dataclass(frozen=True)
class DraftSample:
    ids: torch.Tensor  # global int64 [1]
    probabilities: torch.Tensor  # actual post-filter proposal q, FP32 [1,V]


class DraftSampler(Protocol):
    def __call__(self, logits: torch.Tensor, step: int) -> DraftSample: ...


def greedy_sample(logits, step):
    ids = logits.argmax(-1)
    return DraftSample(ids, torch.zeros_like(logits, dtype=torch.float32).scatter_(1, ids[:, None], 1.0))


@dataclass(frozen=True)
class DraftProposal:
    ids: torch.Tensor  # [gamma], excludes already-known bonus
    verify_ids: torch.Tensor  # [bonus, proposals...], target verify inputs
    corrected_logits: torch.Tensor  # [gamma,V], FP16 Markov-corrected values
    probabilities: torch.Tensor  # [gamma,V], actual q for existing verifier
    confidence: torch.Tensor | None  # learned per-position acceptance estimate
    start: int
    state_version: int
    identity: str


@dataclass
class ContextUpdate:
    start: int
    keys: torch.Tensor
    values: torch.Tensor
    version: int
    owner: object

    @property
    def rows(self):
        return self.keys.shape[1]


@dataclass
class DraftState:
    target: object
    identity: str
    keys: torch.Tensor
    values: torch.Tensor
    length: int = 0
    version: int = 0
    pending: ContextUpdate | None = None

    @property
    def capacity(self):
        return self.keys.shape[1]

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.keys, self.values))


class TapCollector:
    """One bounded target piece: captures outputs by the next layer's input.

    At before_layer2 we capture configured output layer1. FP32 HC mean, then
    FP16 storage, matches the pinned Qwen4 add-on's explicit dtype boundary.
    ``finish`` transfers the owned feature buffer; it is never a borrowed view
    of target scratch that later layers can overwrite.
    """

    def __init__(self, config, rows, *, limit):
        if type(rows) is not int or not 0 < rows <= limit:
            raise ValueError("DSpark capture piece exceeds admitted rows")
        self.config = config
        self.seen = set()
        self.closed = False
        self.features = torch.empty((rows, len(config.taps), config.hidden), dtype=torch.float16)

    def before_layer(self, index, streams):
        if self.closed:
            raise ValueError("DSpark capture piece is already finished")
        if index not in self.config.capture_before:
            return
        if index in self.seen:
            raise ValueError("DSpark duplicate layer capture")
        rows = self.features.shape[0]
        d = self.config.hidden
        if (
            streams.device.type != "cpu"
            or streams.dtype != torch.float16
            or streams.shape not in [(rows, 4 * d), (rows, 4, d)]
        ):
            raise ValueError("DSpark capture requires CPU FP16 four-branch HC rows")
        slot = self.config.capture_before.index(index)
        self.features[:, slot].copy_(streams.reshape(rows, 4, d).float().mean(1).half())
        self.seen.add(index)

    def finish(self):
        if self.closed or self.seen != set(self.config.capture_before):
            raise ValueError("DSpark capture piece is incomplete or finished")
        self.closed = True
        return self.features.flatten(1)


class DraftProvider(Protocol):
    """Minimal worker-owned draft extension; the existing engine verifies q."""

    identity: str
    capturable: bool
    distributed: bool

    def taps(self, rows: int) -> TapCollector: ...
    def make_state(self, target_state, *, capacity: int) -> DraftState: ...
    def stage_context(self, state: DraftState, features: torch.Tensor) -> ContextUpdate: ...
    def commit(self, state: DraftState, update: ContextUpdate, accepted: int) -> None: ...
    def reset(self, state: DraftState) -> None: ...
    def check_proposal(self, state: DraftState, proposal: DraftProposal) -> None: ...
    def propose(self, state: DraftState, bonus: int, *, block=None, sampler=greedy_sample) -> DraftProposal: ...


class DSparkProvider:
    execution_identity = "torch-f16-f32acc-m32-dspark-reference-v1"
    capturable = False
    distributed = False

    def __init__(self, weights, *, target, embed, head, target_identity, prefill_rows=256):
        c = weights.config
        c.check_target(target)
        if type(prefill_rows) is not int or prefill_rows <= 0:
            raise ValueError("DSpark prefill_rows must be positive")
        if not isinstance(target_identity, str) or not target_identity:
            raise ValueError("DSpark target identity must be explicit")
        shapes = tensor_shapes(c)
        if weights.tensors.keys() != shapes.keys():
            raise ValueError("DSpark reference weight namespace differs")
        for name, shape in shapes.items():
            value = weights.tensors[name]
            dtype = torch.float32 if "norm" in name or name.startswith("confidence_head.") else torch.float16
            if value.device.type != "cpu" or value.shape != shape or value.dtype != dtype or not value.is_contiguous():
                raise ValueError(f"DSpark CPU unsharded reference weight differs: {name}")
        self.weights = weights
        self.config = c
        self.embed = embed
        self.head = head
        self.prefill_rows = prefill_rows
        self.identity = hashlib.sha256(
            (weights.identity + "|" + target_identity + "|" + self.execution_identity).encode()
        ).hexdigest()

    def taps(self, rows):
        return TapCollector(self.config, rows, limit=self.prefill_rows)

    def make_state(self, target_state, *, capacity):
        if type(capacity) is not int or not 0 < capacity <= self.config.context:
            raise ValueError("DSpark cache capacity exceeds its configured context")
        if getattr(target_state, "pos", None) != 0 or getattr(target_state, "capacity", capacity) < capacity:
            raise ValueError("DSpark state starts beside an empty target State of sufficient capacity")
        c = self.config
        shape = (c.layers, capacity, c.kv_heads, c.head_dim)
        return DraftState(
            target_state,
            self.identity,
            torch.empty(shape, dtype=torch.float16),
            torch.empty(shape, dtype=torch.float16),
        )

    def _check(self, state, *, aligned=True):
        if state.identity != self.identity or state.keys.device.type != "cpu":
            raise ValueError("DSpark provider/state identity differs")
        c = self.config
        if (
            state.keys.dtype != torch.float16
            or state.values.dtype != torch.float16
            or state.values.device.type != "cpu"
            or state.keys.shape != state.values.shape
            or state.keys.ndim != 4
            or state.keys.shape[0] != c.layers
            or tuple(state.keys.shape[2:]) != (c.kv_heads, c.head_dim)
            or type(state.length) is not int
            or not 0 <= state.length <= state.capacity
            or state.capacity > c.context
        ):
            raise ValueError("DSpark state storage differs from its admitted geometry")
        if aligned and getattr(state.target, "pos", None) != state.length:
            raise ValueError("DSpark target/context cursor differs")

    @torch.no_grad()
    def stage_context(self, state, features):
        self._check(state, aligned=False)
        if state.pending is not None:
            raise ValueError("DSpark allows one pending context piece per state")
        if (
            features.device.type != "cpu"
            or features.dtype != torch.float16
            or features.ndim != 2
            or features.shape[1] != len(self.config.taps) * self.config.hidden
            or not 0 < features.shape[0] <= self.prefill_rows
            or state.length + features.shape[0] > state.capacity
        ):
            raise ValueError("DSpark context piece differs from its admitted geometry")
        if not torch.isfinite(features).all():
            raise ValueError("DSpark nonfinite target features")
        keys, values = ref.context_kv(self.weights, features, state.length)
        if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
            raise ValueError("DSpark nonfinite projected context")
        update = ContextUpdate(state.length, keys, values, state.version, state)
        state.pending = update
        return update

    @torch.no_grad()
    def commit(self, state, update, accepted):
        self._check(state, aligned=False)
        if (
            update is not state.pending
            or update.owner is not state
            or update.version != state.version
            or update.start != state.length
            or type(accepted) is not int
            or not 0 <= accepted <= update.rows
        ):
            raise ValueError("DSpark stale/foreign context update or invalid accepted prefix")
        end = state.length + accepted
        if getattr(state.target, "pos", None) != end:
            raise ValueError("DSpark context commit requires the matching verified target prefix")
        state.keys[:, state.length : end].copy_(update.keys[:, :accepted])
        state.values[:, state.length : end].copy_(update.values[:, :accepted])
        state.length = end
        state.version += 1
        state.pending = None

    def discard(self, state):
        self._check(state, aligned=False)
        state.pending = None
        state.version += 1

    def rewind(self, state, position):
        self._check(state, aligned=False)
        if (
            type(position) is not int
            or not 0 <= position <= state.length
            or getattr(state.target, "pos", None) != position
        ):
            raise ValueError("DSpark rewind requires the same retained target prefix")
        state.length = position
        state.pending = None
        state.version += 1

    def reset(self, state):
        self.rewind(state, 0)

    def copy_prefix(self, source, target_state, *, capacity):
        self._check(source)
        if type(capacity) is not int:
            raise ValueError("DSpark copied capacity must be an integer")
        if (
            source.pending is not None
            or getattr(target_state, "pos", None) != source.length
            or capacity < source.length
        ):
            raise ValueError("DSpark prefix copy requires matching committed target state")
        if (
            type(capacity) is not int
            or not 0 < capacity <= self.config.context
            or getattr(target_state, "capacity", capacity) < capacity
        ):
            raise ValueError("DSpark copied cache capacity exceeds its admission")
        c = self.config
        shape = (c.layers, capacity, c.kv_heads, c.head_dim)
        result = DraftState(
            target_state,
            self.identity,
            torch.empty(shape, dtype=torch.float16),
            torch.empty(shape, dtype=torch.float16),
            source.length,
        )
        result.keys[:, : source.length].copy_(source.keys[:, : source.length])
        result.values[:, : source.length].copy_(source.values[:, : source.length])
        return result

    def check_proposal(self, state, proposal):
        """Call before the existing target verifier consumes a proposal."""
        self._check(state)
        if (
            not isinstance(proposal, DraftProposal)
            or proposal.identity != self.identity
            or proposal.start != state.length
            or proposal.state_version != state.version
            or state.pending is not None
        ):
            raise ValueError("DSpark proposal belongs to a stale/foreign target prefix")

    @torch.no_grad()
    def propose(self, state, bonus, *, block=None, sampler=greedy_sample):
        self._check(state)
        c, t = self.config, self.weights.tensors
        block = c.block if block is None else block
        if (
            state.pending is not None
            or type(block) is not int
            or not 1 <= block <= c.block
            or type(bonus) is not int
            or not 0 <= bonus < c.vocab
            or state.length + block + 1 > state.capacity
        ):
            raise ValueError("DSpark proposal exceeds its admitted block/context or has pending state")
        inputs = torch.full((block,), c.mask, dtype=torch.int64)
        inputs[0] = bonus
        embeds = self.embed(inputs)
        if embeds.device.type != "cpu" or embeds.dtype != torch.float16 or embeds.shape != (block, c.hidden):
            raise ValueError("DSpark target embedding callback must return full CPU FP16 rows")
        hidden = ref.block_hidden(self.weights, embeds, state.keys, state.values, state.length)
        base = self.head(hidden)
        if (
            base.device.type != "cpu"
            or base.shape != (block, c.vocab)
            or base.dtype not in (torch.float16, torch.float32)
        ):
            raise ValueError("DSpark target head callback must return full CPU vocabulary logits")
        base = base.half()
        corrected = torch.empty_like(base)
        probabilities = torch.empty(base.shape, dtype=torch.float32)
        chosen = torch.empty(block, dtype=torch.int64)
        previous = bonus
        prev_embeds = torch.empty((block, c.rank), dtype=torch.float16)
        for step in range(block):
            prev = t["markov_head.markov_w1.weight"][previous : previous + 1]
            prev_embeds[step].copy_(prev[0])
            bias = ref.linear(prev, t["markov_head.markov_w2.weight"])
            logits = (base[step : step + 1].float() + bias.float()).half()
            if not torch.isfinite(logits).all():
                raise ValueError("DSpark nonfinite corrected proposal logits")
            corrected[step].copy_(logits[0])
            sample = sampler(logits, step)
            if (
                not isinstance(sample, DraftSample)
                or sample.ids.device.type != "cpu"
                or sample.ids.dtype != torch.int64
                or sample.ids.shape != (1,)
                or sample.probabilities.device.type != "cpu"
                or sample.probabilities.dtype != torch.float32
                or sample.probabilities.shape != (1, c.vocab)
            ):
                raise ValueError("DSpark sampler must return global IDs and its actual full proposal distribution")
            token = int(sample.ids[0])
            q = sample.probabilities[0]
            if (
                not 0 <= token < c.vocab
                or not torch.isfinite(q).all()
                or (q < 0).any()
                or abs(float(q.sum()) - 1) > 2e-5
                or q[token] <= 0
            ):
                raise ValueError("DSpark sampler returned invalid proposal probabilities/ID")
            chosen[step] = token
            probabilities[step].copy_(q)
            previous = token
        confidence = None
        if c.confidence:
            joined = torch.cat((hidden, prev_embeds), -1).float()
            confidence = (
                (joined @ t["confidence_head.proj.weight"].T + t["confidence_head.proj.bias"]).flatten().sigmoid()
            )
        return DraftProposal(
            chosen,
            torch.cat((torch.tensor([bonus]), chosen)),
            corrected,
            probabilities,
            confidence,
            state.length,
            state.version,
            self.identity,
        )

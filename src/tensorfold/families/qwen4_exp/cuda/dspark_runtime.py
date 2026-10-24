"""Explicit DSpark target integration: taps, committed context and keyed choices.

No automatic provider selection occurs here. The existing family owns target
forward, cache publication, sampling and verification; this owner supplies only
bounded auxiliary state and proposal execution.
"""

from __future__ import annotations

import weakref

import torch

from .dspark_cuda import DSparkCUDA, PositionKeyedChoice, _Stamp, _borrow, _record, load_shard
from .dspark_taps import TapWorkspace


class DSparkRuntime:
    capturable = False
    target_capturable = True

    def __init__(self):
        # Publish this empty owner to w.dspark before any fallible CUDA load.
        self.provider = self.plan = self.weights = None
        self.owned = {}
        self._states = []
        self._taps = []
        self._proposal_graphs = None
        self._contexts = []
        self.closed = False

    @property
    def config(self):
        return self.plan.config

    @property
    def identity(self):
        return self.plan.identity

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.owned.values())

    def device_tensors(self):
        return tuple(self.owned.values())

    def _state(self, st, *, aligned=True):
        if self.closed or self.provider is None:
            raise ValueError("DSpark runtime is closed or incompletely loaded")
        aux = getattr(st, "dspark", None)
        self.provider._check(aux, aligned=aligned)
        return aux

    def init_state(self, st):
        aux = self.provider.make_state(st, capacity=st.capacity)
        st.dspark = aux
        self._states = [ref for ref in self._states if ref() is not None]
        self._states.append(weakref.ref(aux))
        return aux

    def reset(self, st):
        aux = getattr(st, "dspark", None)
        if aux is not None:
            self.provider.reset(aux)

    def resize(self, st):
        aux = getattr(st, "dspark", None)
        if aux is not None:
            # Target admission/growth has already reserved both caches. Any
            # failure poisons the target through its existing resize protocol.
            self.provider.resize(aux, st.capacity)

    def close_state(self, st):
        aux = getattr(st, "dspark", None)
        if aux is not None and not aux.closed:
            self.provider.close_state(aux)
        st.dspark = None

    def new_taps(self, rows, device):
        owner = TapWorkspace(self.config, rows, device)
        self._taps = [ref for ref in self._taps if ref() is not None]
        self._taps.append(weakref.ref(owner))
        return owner

    def clone_state(self, source, destination):
        aux = self.provider.copy_prefix(source.dspark, destination, capacity=destination.capacity)
        destination.dspark = aux
        self._states.append(weakref.ref(aux))

    def begin(self, segs, buffers):
        for st, _, _ in segs:
            aux = self._state(st)
            if aux.pending is not None:
                raise ValueError("DSpark previous target frame was not committed or discarded")
        return buffers.dspark_taps.begin(segs[-1][2], segs)

    def finish(self, capture, segs):
        batch = capture.finish()
        return self.provider.stage_taps([st.dspark for st, _, _ in segs], batch)

    def begin_graph(self, buffers, rows):
        """Pure target feature recording, with no auxiliary State mutation."""
        if self.closed or self.provider is None or not getattr(buffers, "dspark_graph", False):
            raise ValueError("DSpark graph recording requires an attached bound verify buffer")
        return buffers.dspark_taps.begin_graph(rows)

    def finish_graph_capture(self, capture, segs=None, buffers=None):
        capture.finish_graph_sink()
        context = getattr(buffers, "dspark_context", None)
        if context is not None:
            context.compute(capture.owner.features[:capture.rows].flatten(1), segs)

    def prepare_graph_context(self, buffers, bound):
        from .dspark_context_graphs import ContextWorkspace, _ENABLED
        if hasattr(buffers, "dspark_context"):
            raise ValueError("DSpark bound context preparation is one-shot")
        if not bound._agree(_ENABLED):
            buffers.dspark_context = None
            return False
        if not getattr(buffers, "dspark_graph", False):
            raise ValueError("DSpark captured projection requires its bound verify buffer")
        owner = ContextWorkspace(self.provider, buffers.rows, self.provider.device)
        buffers.dspark_context = owner  # journal before fallible CUDA allocations
        self._contexts = [ref for ref in self._contexts if ref() is not None]
        self._contexts.append(weakref.ref(owner))
        owner.allocate()
        return True

    def check_graph(self, segs, buffers=None):
        """Validate original slots before replay, never their shallow graph views."""
        for st, _, _ in segs:
            aux = self._state(st)
            if aux.pending is not None:
                raise ValueError("DSpark previous target frame was not committed or discarded")
        if buffers is not None:
            if not getattr(buffers, "dspark_graph", False):
                raise ValueError("DSpark replay requires its bound verify tap buffer")
            buffers.dspark_taps.prepare_replay()
            context = getattr(buffers, "dspark_context", None)
            if context is not None:
                context.prepare_replay()

    def stage_graph(self, source, bound, segs, index):
        """Stage only real exported rows; publication still requires target commit."""
        if not getattr(bound, "dspark_graph", False):
            raise ValueError("DSpark graph export requires a bound verify buffer")
        for st, _, _ in segs:
            aux = self._state(st, aligned=False)
            if st.pos != aux.length or aux.pending is not None:
                raise ValueError("DSpark graph export lost its original committed cursor")
        context = getattr(bound, "dspark_context", None)
        if context is not None:
            return context.stage([st.dspark for st, _, _ in segs], segs, index)
        batch = bound.dspark_taps.export_graph(source.dspark_taps, segs, index)
        return self.provider.stage_taps([st.dspark for st, _, _ in segs], batch)

    def discard_graph(self, segs):
        """Synthetic graph warming must leave no unpublished target frame."""
        for st, _, _ in segs:
            aux = getattr(st, "dspark", None)
            if aux is not None and aux.pending is not None:
                self.provider.discard(aux)

    def retire_graph(self, buffers):
        """The bound owner calls this after resetting its captured graph pool."""
        owner = getattr(buffers, "dspark_taps", None)
        if owner is not None:
            owner.retire_graph()
            owner.close()
        context = getattr(buffers, "dspark_context", None)
        if context is not None:
            context.retire_graph()
            context.close()

    def prepare_proposal_graphs(self, bound):
        from .dspark_proposal_graphs import DSparkProposalGraphs, _ENABLED
        # Every rank participates, even when a local optional flag declines.
        if not bound._agree(_ENABLED):
            return False
        if self._proposal_graphs is not None:
            if self._proposal_graphs.bound is not bound:
                raise ValueError("DSpark proposals already belong to another target graph owner")
            return True
        self._proposal_graphs = DSparkProposalGraphs(self, bound)
        return True

    def warm_proposals(self, states):
        if self._proposal_graphs is not None:
            self._proposal_graphs.warm(states)

    def invalidate_proposals(self, state):
        if self._proposal_graphs is not None:
            self._proposal_graphs.invalidate(state)

    def retire_proposals(self, bound):
        if self._proposal_graphs is not None:
            if self._proposal_graphs.bound is not bound:
                raise ValueError("DSpark proposal retirement belongs to another graph owner")
            self._proposal_graphs.close()
            self._proposal_graphs = None

    def abort(self, capture, segs):
        if capture is not None and not capture.closed:
            capture.abort()
        for st, _, _ in segs:
            aux = getattr(st, "dspark", None)
            if aux is not None and aux.pending is not None:
                self.provider.discard(aux)

    def commit(self, st, keep):
        aux = getattr(st, "dspark", None)
        if aux is None or aux.pending is None:
            raise ValueError("DSpark target commit has no matching staged context")
        self.provider.commit(aux, aux.pending, keep)

    def snapshot(self, st, position=None):
        aux = self._state(st, aligned=False)
        pos = st.pos if position is None else position
        maximum = aux.length + (aux.pending.rows if aux.pending is not None else 0)
        if type(pos) is not int or not 0 <= pos <= maximum:
            raise ValueError("DSpark snapshot marker lies outside its retained/staged prefix")
        return {"dspark_identity": self.identity, "dspark_pos": pos}

    def validate_snapshot(self, st, snapshot):
        if snapshot.get("dspark_identity") != self.identity or snapshot.get("dspark_pos") != snapshot.get("pos"):
            raise ValueError("DSpark snapshot identity/cursor differs from its target prefix")
        pos = snapshot["dspark_pos"]
        aux = getattr(st, "dspark", None)
        if aux is None or type(pos) is not int or not 0 <= pos <= aux.valid_length <= aux.capacity:
            raise ValueError("DSpark snapshot exceeds its auxiliary cache admission")

    def restore(self, st, snapshot):
        self.validate_snapshot(st, snapshot)
        aux = st.dspark
        self.provider._check(aux, aligned=False, stamp=False)
        pos = snapshot["dspark_pos"]
        if pos > aux.valid_length or st.pos != pos or getattr(st, "_kv_pending", False):
            raise ValueError("DSpark restore requires the same retained target/cache prefix")
        self.provider._retire_pending(aux)
        aux.length, aux.version, aux.stamp = pos, aux.version + 1, _Stamp.read(st)
        _record(aux.ready, self.provider.device)

    @torch.no_grad()
    def validate_copy(self, destination, source, pos):
        src = self._state(source)
        dst = self._state(destination, aligned=False)
        if (
            type(pos) is not int
            or not 0 <= pos <= min(src.length, dst.capacity)
            or src.pending is not None
            or dst.pending is not None
            or src.stamp.codec != dst.stamp.codec
        ):
            raise ValueError("DSpark prefix copy requires a matching committed source/cache format")
        return src, dst

    @torch.no_grad()
    def copy_prefix(self, destination, source, pos):
        src, dst = self.validate_copy(destination, source, pos)
        _borrow((src.keys, src.values))
        try:
            dst.keys[:, :pos].copy_(src.keys[:, :pos])
            dst.values[:, :pos].copy_(src.values[:, :pos])
        finally:
            _record(src.ready, self.provider.device)
        dst.length, dst.valid_length, dst.version = pos, pos, dst.version + 1
        dst.stamp = _Stamp.read(destination)
        _record(dst.ready, self.provider.device)

    @torch.no_grad()
    def draft(self, e, bonus, count, sampling, confidence):
        if count <= 0:
            return []
        aux = self._state(e.st)
        count = min(int(count), self.config.block, aux.capacity - aux.length - 1)
        if count <= 0:
            return []

        if self._proposal_graphs is not None:
            result = self._proposal_graphs.draft(e.st, bonus, count, sampling, confidence)
            if result is not None:
                return result

        def choose(logits, *, step, position, shard):
            ids = e.sample(logits, [position], sampling, draft=False, gathered=False)
            return PositionKeyedChoice(torch.tensor(ids, dtype=torch.int64, device=logits.device))

        proposal = self.provider.propose(aux, int(bonus), block=count, sampler=choose).wait()
        self.provider.check_proposal(aux, proposal)
        ids = proposal.ids.cpu().tolist()
        if confidence > 0 and proposal.confidence is not None:
            values = proposal.confidence.cpu().tolist()
            for index, value in enumerate(values):
                if value < confidence:
                    ids = ids[: max(1, index)]
                    break
        return ids

    def close(self):
        if self.closed:
            return
        if self._proposal_graphs is not None and not self._proposal_graphs.closed:
            raise ValueError("DSpark proposal graphs must retire before their runtime/cache owners")
        failures = []
        for ref in self._taps:
            owner = ref()
            if owner is not None and not owner._closed:
                try:
                    owner.close()
                except BaseException as error:
                    failures.append(error)
        for ref in self._contexts:
            owner = ref()
            if owner is not None and not owner.closed:
                try:
                    owner.close()
                except BaseException as error:
                    failures.append(error)
        if failures:
            raise BaseExceptionGroup("DSpark tap retirement failed; complete owner retained for retry", failures)
        for ref in self._states:
            aux = ref()
            if aux is not None and not aux.closed:
                try:
                    self.provider.close_state(aux)
                except BaseException as error:
                    failures.append(error)
        if failures:
            raise BaseExceptionGroup("DSpark auxiliary retirement failed; owner retained for retry", failures)
        self.provider = self.weights = None
        self.owned.clear()
        self._states.clear()
        self._taps.clear()
        self._contexts.clear()
        self.closed = True


def prepare(w, plan, *, prefill_rows):
    if getattr(w, "dspark", None) is not None:
        raise ValueError("DSpark runtime attachment is one-shot")
    owner = DSparkRuntime()
    owner.plan = plan
    w.dspark = owner
    plan.check_source()
    weights = load_shard(
        plan.directory,
        target=w,
        checkpoint_identity=plan.source_identity,
        tp=(plan.rank, plan.world),
        device=w.device,
        chunk_bytes=plan.chunk_bytes,
        _journal=owner.owned,
    )
    owner.weights = weights
    plan.check_source()
    if int(w.meta.get("vocab_offset", 0)) != (
        plan.rank * (plan.config.vocab // plan.world) + min(plan.rank, plan.config.vocab % plan.world)
    ):
        raise ValueError("DSpark full target head shard offset differs")

    def embed(ids):
        from .forward import _embed

        out = torch.empty((ids.numel(), plan.config.hidden), dtype=torch.float16, device=w.device)
        return _embed(w, ids, 1, out)

    def head(hidden, shard):
        if w.head.n != shard.width:
            raise ValueError("DSpark requires original full target vocabulary head")
        return w.head.matmul(hidden, f32=True)

    target_identity = "|".join(
        str(w.meta.get(key, "")) for key in ("gguf_entrypoint", "execution_provider", "execution_dtype")
    )
    owner.provider = DSparkCUDA(
        weights,
        target=w,
        embed=embed,
        head=head,
        target_identity=target_identity,
        comm=w.comm,
        prefill_rows=prefill_rows,
    )
    return owner

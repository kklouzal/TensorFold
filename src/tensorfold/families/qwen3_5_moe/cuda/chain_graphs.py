"""One fixed verification state whose chain rounds replay CUDA graphs per (rows, context bucket).

The runner's attention buffers are reserved at ``capacity`` rows once and never move, so captured graphs keep their
addresses. They are also the one stream's working attention cache: ``begin`` hands the engine a state on them that
the request's prefill extends in place, and the prefix cache's kept prompt states share them (clones share one KV
list), so the stream holds one full-capacity attention cache, not a second copy for the graphs. ``draft_decode(
runner=...)`` then copies only the request's recurrent state in (``load``) and verifies every chain round (DFlash v1
and copy proposals are chains) by replaying the captured ``tree_forward`` of its width; any other round (a
grammar-pruned tree) runs the eager forward on the same fixed state. Replay executes exactly the eager kernels on the
same buffers, so tokens equal eager decoding.
"""

from __future__ import annotations

import gc
from typing import Sequence

import torch

from tensorfold.families.qwen3_5.cuda.forward import State, reserve, stage, tree_forward

from . import CHAIN_BUCKET


def copy_kv(dst, src, rows: int) -> None:
    """The first ``rows`` rows of every attention layer's (K, V) in the KV list ``src``, copied into ``dst``'s."""

    for kv, other in zip(dst, src):
        if kv is not None:
            kv[0][:rows].copy_(other[0][:rows])
            kv[1][:rows].copy_(other[1][:rows])


def copy_state(dst: State, st: State) -> None:
    """``st``'s committed recurrent/conv state and first ``st.pos`` KV rows (already there when the two share one KV
    list), copied into ``dst``'s buffers in place."""

    for a, b in zip(dst.rec + dst.conv, st.rec + st.conv):
        if a is not None:
            a.copy_(b)
    if dst.kv is not st.kv:
        copy_kv(dst.kv, st.kv, st.pos)
    dst.pos = st.pos


def capture(pool, run):
    """``run()`` captured as a new CUDA graph in the private ``pool``: (graph, run's outputs, static across replays)."""

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    enabled = gc.isenabled()
    gc.disable()                             # collecting an old graph mid-capture invalidates it
    try:
        # thread_local: an unrelated CUDA call from another server thread cannot invalidate the capture.
        with torch.cuda.graph(graph, pool=pool, capture_error_mode="thread_local"):
            out = run()
    finally:
        if enabled:
            gc.enable()
    return graph, out


class ChainGraphs:
    """Graphs and the fixed state they read; one request at a time (the engine serializes generate)."""

    def __init__(self, w, capacity: int, *, bucket: int = CHAIN_BUCKET) -> None:
        if type(capacity) is not int or capacity < 1 or type(bucket) is not int or bucket < 1:
            raise ValueError("chain graphs need a positive row capacity and context bucket")
        self.w, self.capacity, self.bucket = w, capacity, bucket
        self.st = State(w)
        reserve(self.st, capacity)
        self._buffers = self._addresses()
        self.pool = None                         # one private pool for every graph, made at the first capture
        self.graphs: dict[tuple[int, int, bool], tuple] = {}

    def _span(self, end: int) -> int:
        return min(self.capacity, max(self.bucket, 1 << (end - 1).bit_length()))

    def begin(self, cache, room, base: State | None, prompt: Sequence[int]) -> State:
        """The state ``prompt``'s prefill extends in place on the fixed attention buffers.

        ``base``: the prefix-cache state the request resumes (None: a fresh prompt); ``cache`` (streams.PrefixCache,
        whose extensions of ``base`` the engine already dropped) and ``room`` (streams.KVRoom or None) are the
        stream's. Kept states on the fixed buffers that the request would overwrite (every one, unless it resumes
        from them) first move to private buffers sized to their rows (while ``room`` admits them; else they are
        dropped); a resumed state on private buffers is copied in and its kept relatives return to the fixed ones.
        Every kept state on the fixed buffers must be a prefix of one prompt (at most ``cache.keep`` of them): the
        rows they share hold only that prompt. Anything else is a broken cache invariant and raises, since a kept
        state would read another prompt's rows.
        """

        fixed = self.st.kv
        if base is not None and base.kv is fixed:
            self._check_chain(cache, prompt)     # prefill writes past base.pos: only prefixes of the prompt stay
            return base
        self._move_out(cache, room)
        if base is None:
            st = State(self.w)
            st.kv, st.limit = fixed, self.capacity
            return st
        private = base.kv
        copy_kv(fixed, private, base.pos)
        for _, kept, _ in cache.entries:         # base's chain (its extensions are gone): rows < base.pos
            if kept.kv is private:
                kept.kv = fixed
        base.kv, base.limit = fixed, self.capacity
        return base

    def _check_chain(self, cache, prompt: Sequence[int]) -> list:
        """The kept entries on the fixed buffers, each verified a prefix of ``prompt`` (at most ``cache.keep``)."""

        held = [entry for entry in cache.entries if entry[1].kv is self.st.kv]
        if len(held) > cache.keep or any(len(ids) > len(prompt) or list(prompt[:len(ids)]) != list(ids)
                                         or kept.pos != len(ids) for ids, kept, _ in held):
            raise RuntimeError("chain-graph buffers hold kept prompt states that are not one prompt's prefixes")
        return held

    def _move_out(self, cache, room) -> None:
        fixed = self.st.kv
        held = [entry for entry in cache.entries if entry[1].kv is fixed]
        if not held:
            return
        self._check_chain(cache, max((ids for ids, _, _ in held), key=len))
        rows = max(kept.pos for _, kept, _ in held)
        size = sum(rows * (t.stride(0) * t.element_size()) for kv in fixed if kv is not None for t in kv)
        if room is not None:
            room(self.st, size)                  # evicts kept entries on other private buffers, oldest first
        if rows == 0 or room is not None and room.held(self.st) + size > room.budget:
            cache.entries = [entry for entry in cache.entries if entry[1].kv is not fixed]
            cache.hit &= {tuple(entry[0]) for entry in cache.entries}
            return
        private = [None if kv is None else (kv[0][:rows].clone(), kv[1][:rows].clone()) for kv in fixed]
        for _, kept, _ in held:                  # one prompt's chain of prefixes: the longest one's rows hold them all
            kept.kv = private

    def load(self, st: State, need: int) -> State:
        """``st``'s committed state in the fixed state: its recurrent state (and attention rows, unless ``st`` is
        on the fixed buffers already), room for ``need`` rows."""

        if st.rope_delta:
            raise ValueError("chain graphs verify text positions; image prompts decode eagerly")
        if max(need, st.pos + 1) > self.capacity:
            raise ValueError("chain graphs hold at most their capacity's rows")
        copy_state(self.st, st)
        return self.st

    def _addresses(self) -> tuple[int, ...]:
        return tuple(t.data_ptr() for kv in self.st.kv if kv is not None for t in kv)

    @torch.no_grad()
    def prepare(self, widths: Sequence[int], *, taps: bool) -> int:
        """Capture the smallest context span's chain graphs now (engine startup) instead of inside the first rounds.

        An empty state is loaded and each width's warm-up forward only reads it, so nothing is committed. ``taps``:
        the drafter's tap form (drafted rounds); the one-row serial form is captured too. Larger spans still capture
        on first use. Returns the number of graphs held.
        """
        empty = State(self.w)
        empty.kv = self.st.kv
        self.load(empty, 1)
        keys = [(width, taps) for width in widths] + ([(1, False)] if taps else [])
        for width, tap in keys:
            if width <= self.capacity:
                self.verify([0] * width, list(range(-1, width - 1)), capture_taps=tap)
        torch.cuda.synchronize()
        return len(self.graphs)

    @torch.no_grad()
    def verify(self, tokens: Sequence[int], parents: Sequence[int], *, capture_taps: bool):
        """``tree_forward(..., capture_taps)``'s (logits, record[, taps]) on the fixed state: replayed for a chain.

        A replayed round returns the graph's static outputs, valid until the next verify of the same (width, span,
        taps) key. Callers finish the round's host read (sampling) before the next verify: that is what lets
        ``staged.refresh`` rewrite the pinned mirror whose previous copy is then complete (draft_decode synchronizes).
        """

        st, width = self.st, len(tokens)
        if list(parents) != list(range(-1, width - 1)) or st.pos + width > self.capacity:
            ids = torch.tensor(tokens, dtype=torch.int32, device=self.w.norm.device)
            return tree_forward(self.w, ids, parents, st, capture_taps=capture_taps)
        if self._addresses() != self._buffers:
            # A commit past the reserved rows moved a cache: every captured address is stale.
            raise RuntimeError("chain graph state buffers moved after capture")
        key = (width, self._span(st.pos + width), bool(capture_taps))
        entry = self.graphs.get(key)
        if entry is None:
            staged = stage(self.w, st, width, key[1])
            staged.refresh(tokens, st.pos)
            staged_parents = list(range(-1, width - 1))

            def run():
                return tree_forward(self.w, staged.ids, staged_parents, st, capture_taps=capture_taps, staged=staged)

            run()                                # compiles every kernel and retains lazily built buffers
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            graph, out = capture(self.pool, run)
            entry = self.graphs[key] = (graph, staged, out)
        graph, staged, out = entry
        # The pinned mirror is rewritten only after the previous round's host sample, so its copy has completed.
        staged.refresh(tokens, st.pos)
        graph.replay()
        return out

"""One fixed verification state whose chain rounds replay CUDA graphs per (rows, context bucket).

``draft_decode(runner=...)`` copies a request's committed state in once (``load``), then verifies every chain round
(DFlash v1 and copy proposals are chains) by replaying the captured ``tree_forward`` of its width. Any other round
(a grammar-pruned tree) runs the eager forward on the same fixed state. Replay executes exactly the eager kernels on
the same buffers, so tokens equal eager decoding. The state's KV buffers never move between loads (``reserve``); they
grow by powers of two up to ``capacity``, dropping every graph, only when a request needs more rows.
"""

from __future__ import annotations

import gc
from typing import Sequence

import torch

from tensorfold.families.qwen3_5.cuda.forward import State, reserve, stage, tree_forward

from . import CHAIN_BUCKET


class ChainGraphs:
    """Graphs and the fixed state they read; one request at a time (the engine serializes generate)."""

    def __init__(self, w, capacity: int, *, bucket: int = CHAIN_BUCKET) -> None:
        if type(capacity) is not int or capacity < 1 or type(bucket) is not int or bucket < 1:
            raise ValueError("chain graphs need a positive row capacity and context bucket")
        self.w, self.capacity, self.bucket = w, capacity, bucket
        self.st: State | None = None
        self.rows = 0
        self.pool = None                         # a new private pool each time the state grows
        self.graphs: dict[tuple[int, int, bool], tuple] = {}
        self._buffers: tuple[int, ...] = ()

    def _span(self, end: int) -> int:
        return min(self.rows, max(self.bucket, 1 << (end - 1).bit_length()))

    def load(self, st: State, need: int) -> State:
        """``st``'s committed state copied into the fixed buffers (grown, graphs dropped, past ``need`` rows)."""

        if st.rope_delta:
            raise ValueError("chain graphs verify text positions; image prompts decode eagerly")
        need = min(max(need, st.pos + 1), self.capacity)
        if need > self.rows:
            self.graphs.clear()
            self.st = None
            gc.collect()
            self.rows = min(self.capacity, max(self.bucket, 1 << (need - 1).bit_length()))
            # torch refuses a capture into a pool left registered without a graph: capture into a new one
            self.pool = torch.cuda.graph_pool_handle()
            self.st = State(self.w)
            reserve(self.st, self.rows)
            self._buffers = self._addresses()
        dst = self.st
        for a, b in zip(dst.rec + dst.conv, st.rec + st.conv):
            if a is not None:
                a.copy_(b)
        for kv, src in zip(dst.kv, st.kv):
            if kv is not None:
                kv[0][:st.pos].copy_(src[0][:st.pos])
                kv[1][:st.pos].copy_(src[1][:st.pos])
        dst.pos = st.pos
        return dst

    def _addresses(self) -> tuple[int, ...]:
        return tuple(t.data_ptr() for kv in self.st.kv if kv is not None for t in kv)

    def _capture(self, run):
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        enabled = gc.isenabled()
        gc.disable()                             # collecting an old graph mid-capture invalidates it
        try:
            with torch.cuda.graph(graph, pool=self.pool):
                out = run()
        finally:
            if enabled:
                gc.enable()
        return graph, out

    @torch.no_grad()
    def verify(self, tokens: Sequence[int], parents: Sequence[int], *, capture_taps: bool):
        """``tree_forward(..., capture_taps)``'s (logits, record[, taps]) on the fixed state: replayed for a chain."""

        st, width = self.st, len(tokens)
        if st is None:
            raise RuntimeError("chain graphs verify only after load")
        if list(parents) != list(range(-1, width - 1)) or st.pos + width > self.rows:
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

            def run():
                return tree_forward(self.w, staged.ids, staged_parents, st, capture_taps=capture_taps, staged=staged)

            staged_parents = list(range(-1, width - 1))
            run()                                # compiles every kernel and retains lazily built buffers
            graph, out = self._capture(run)
            entry = self.graphs[key] = (graph, staged, out)
        graph, staged, out = entry
        # The pinned mirror is rewritten only after the previous round's host sample, so its copy has completed.
        staged.refresh(tokens, st.pos)
        graph.replay()
        return out

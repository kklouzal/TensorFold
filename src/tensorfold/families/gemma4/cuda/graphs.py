"""Gemma4 text forwards (serial steps and DFlash verify chains) replayed as CUDA graphs.

The engine's single persistent State owns every cache at a fixed address and reads the forward's first position from
its device ``pos``; ``Gemma4Target.forward_staged`` is the capturable forward over a fixed device token buffer and a
fixed full-attention grid. One graph per (rows, attention span, taps) replays exactly the staged kernels on the same
buffers, so a replay equals the eager forward's bits. Visual-block forwards stay eager (the engine's other path).
"""

from __future__ import annotations

import gc
from typing import Sequence

import torch

from .capacity import GRAPH_SPAN as SPAN
from .forward import ForwardResult
from .state import GLOBAL_CHUNK


class Graphs:
    """Graphs over one target and its engine-lifetime state; callers serialize forwards on one stream."""

    def __init__(self, target, state, *, max_rows: int, span: int = SPAN) -> None:
        if type(max_rows) is not int or not 1 <= max_rows <= state.rows or type(span) is not int or span < 1:
            raise ValueError("Gemma4 graphs need 1..state.rows rows and a positive attention span")
        self.target, self.state, self.max_rows, self.span = target, state, max_rows, span
        self.limit = -(-state.context // GLOBAL_CHUNK)
        self.ids = torch.zeros((max_rows,), dtype=torch.int32, device=target.device)
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: dict[tuple[int, int, bool], tuple] = {}

    def chunks(self, end: int) -> int:
        """The covering grid for positions below ``end``: SPAN chunks doubled, at most the admitted context's."""
        need = -(-end // GLOBAL_CHUNK)
        return min(self.limit, max(self.span, 1 << (need - 1).bit_length()))

    def _capture(self, run):
        # torch.cuda.graph synchronizes the device and captures on its own side stream; replays then run on the
        # caller's stream in its order.
        graph = torch.cuda.CUDAGraph()
        enabled = gc.isenabled()
        gc.disable()                                    # collecting an old graph mid-capture invalidates it
        try:
            # thread_local: an unrelated CUDA call from another server thread cannot invalidate the capture.
            with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
                out = run()
        finally:
            if enabled:
                gc.enable()
        return graph, out

    @torch.inference_mode()
    def prepare(self, taps: Sequence[bool] = (False, True), *, max_rows: int | None = None) -> int:
        """Capture smallest-grid graphs, leaving the state where it was.

        Captures write KV rows past the state's position, which the rollback releases; the state holds no request
        yet. By default every admitted row count is prepared. ``max_rows`` limits
        startup work only; all supported row/tap/grid combinations remain lazy
        in forward. Returns the number of graphs held.
        """
        if max_rows is not None and (type(max_rows) is not int or not 1 <= max_rows <= self.max_rows):
            raise ValueError("Gemma4 graph preparation rows must be within 1..max_rows")
        state = self.state
        start = state.position
        smallest = min(self.limit, self.span)
        limit = self.max_rows if max_rows is None else max_rows
        for rows in range(1, min(limit, state.context - start) + 1):
            for tap in taps:
                if self.chunks(start + rows) == smallest:
                    self.forward([0] * rows, state, output_taps=tap)
                    state.rollback(start)
        return len(self.graphs)

    def prepare_generation(self, *, drafts: bool) -> int:
        """Prepare the engine's reachable startup pairs in their original order.

        Without active drafts, generate carries one pending token and requests
        no taps. Active drafts can carry every admitted row count with taps;
        grammar windows only remove rows. The general preparation/forward APIs
        keep other pairs available to callers on first use.
        """
        if type(drafts) is not bool:
            raise ValueError("Gemma4 graph generation preparation needs a boolean draft role")
        count = self.prepare(taps=(False,), max_rows=1)
        return self.prepare(taps=(True,)) if drafts else count

    @torch.inference_mode()
    def forward(self, tokens: Sequence[int], state, *, output_taps: bool) -> ForwardResult:
        """``Gemma4Target.forward(tokens, state, output_taps=...)`` for trusted text tokens, replayed.

        Outputs are the graph's static buffers: valid until the next replay of the same graph.
        """
        if state is not self.state:
            raise ValueError("Gemma4 graphs replay only over their engine's state")
        rows = len(tokens)
        if not 1 <= rows <= self.max_rows:
            raise ValueError("Gemma4 graph forwards carry 1..max_rows rows")
        state.check(rows)
        start = state.position
        chunks = self.chunks(start + rows)
        key = (rows, chunks, bool(output_taps))
        ids = self.ids[:rows]
        # A pageable source: the copy has consumed it when copy_ returns, so the next round may rewrite it.
        ids.copy_(torch.tensor(tokens, dtype=torch.int32))
        state.pos.fill_(start)

        def run():
            return self.target.forward_staged(ids, state, chunks=chunks, output_logits=True,
                                              output_taps=bool(output_taps))

        entry = self.graphs.get(key)
        try:
            if entry is None:
                run()                                   # compiles kernels, retains lazily built buffers
                graph, out = self._capture(run)
                entry = self.graphs[key] = (graph, out)
            graph, (hidden, logits, taps) = entry
            graph.replay()
        except BaseException:
            state.poisoned = True
            raise
        state.advance(rows)
        return ForwardResult(hidden, logits, taps, start)

"""Batched decode shapes for Gemma 4 SM70: serial steps and DFlash rounds over B streams, eager or CUDA graphs.

A shape (mode, B) owns a Layout of B segments (one per stream, padded with the pad slot), its staged token
inputs and its outputs. Serial: one row per stream (its last token); serial-taps also appends the row's taps
to the drafter (drafting requests that take a serial round).
Round: the drafter proposes block - 1 rows per stream (greedy, all-gathered argmax), the target verifies
[anchor, proposals] and the drafter appends every verified row's taps (positions a later block rewrites before
reading them). Both leave no pending state, so a stream may switch modes between rounds. Outputs: per verify
row the all-gathered shard maxima (value, global index, NaN/+inf flag) [world, rows, 3]; the proposals; and
the local logit shard for requests that need complete rows (gathered after the replay).

Graphs are captured identically on every rank at startup (the bodies contain collectives) in one shared pool,
after an eager warm-up has grown every scratch buffer to its largest shape. Slots and positions arrive through
the layout's device tables, so one graph per shape serves every stream and position.
"""

from __future__ import annotations

import numpy as np
import torch

from .target import Layout

MODE_SERIAL, MODE_ROUND, MODE_SERIAL_TAPS = 1, 2, 3     # TAPS: serial steps that keep the drafter context
BUCKETS = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 32)


def buckets(streams: int) -> tuple[int, ...]:
    """Batch sizes with a shape: the bucket ladder up to ``streams``, and ``streams`` itself."""
    return tuple(sorted({b for b in BUCKETS if b < streams} | {streams}))


class Shape:
    def __init__(self, engine, mode: int, count: int):
        t = engine.target
        self.e, self.mode, self.count = engine, mode, count
        self.n = engine.block if mode == MODE_ROUND else 1
        heads = (2, engine.head.heads) if engine.head is not None else (2,)
        self.lay = Layout(t.device, self.n, count, heads)
        rows = self.lay.rows
        self.ids_host = np.zeros(rows, dtype=np.int32)
        self.ids = torch.zeros(rows, dtype=torch.int64, device=t.device)
        self.host_proposals = None
        if mode == MODE_ROUND:
            self.ids_host.fill(engine.head.mask_id)
            self.verify = torch.zeros(rows, dtype=torch.int64, device=t.device)
            index = torch.arange(rows, device=t.device)
            self.proposal_rows = index[index % self.n != 0].contiguous()
            m = count * (self.n - 1)
            self.prop_arange = torch.arange(m, device=t.device)
            if t.rank == 0:
                self.host_proposals = torch.empty(m, dtype=torch.int64, pin_memory=True)
        self.row_arange = torch.arange(rows, device=t.device)
        # Plain serial shapes of one row tile (the sampled/grammar/penalty path) also gather every row's complete
        # vocabulary inside the graph, as the C1 engine did: no eager collective after the replay.
        self.gather_all = mode == MODE_SERIAL and rows <= 16
        # Only rank 0 consumes decisions. A mode runs at most once per scheduler
        # round, and its landing buffers are read before any next round starts.
        self.host_compact = (torch.empty((t.world, rows, 3), dtype=torch.float32, pin_memory=True)
                             if t.rank == 0 and not self.gather_all else None)
        self.full_rows = None
        self.graph = None
        self.out = None
        self.local = None

    def stage(self, slots, positions, anchors):
        self.ids_host[0::self.n] = anchors
        self.lay.stage(slots, positions, self.ids_host)
        self.lay.upload()

    def _compact(self, local):
        t = self.e.target
        values, index = local.max(dim=-1)
        bad = (torch.isnan(local) | torch.isposinf(local)).any(dim=-1)
        packed = torch.stack([values, (index + t.rank * t.shard).float(), bad.float()], -1).contiguous()
        recv = torch.empty((t.world,) + tuple(packed.shape), dtype=torch.float32, device=t.device)
        t.comm.all_gather(packed, recv)
        return recv

    def body(self):
        t, head = self.e.target, self.e.head
        self.ids.copy_(self.lay.ids32)          # staged int32 ids -> the embedding's int64 (captured)
        if self.mode != MODE_ROUND:
            append = self.mode == MODE_SERIAL_TAPS
            local, taps = t.forward(self.ids, self.lay, prompt=False, logits="all", taps=append)
            if append:
                head.append(taps, self.lay)
            self.local = local
            if self.gather_all:                  # complete rows replace the compact maxima
                recv = torch.empty((t.world,) + tuple(local.shape), dtype=torch.float32, device=t.device)
                t.comm.all_gather(local, recv)
                self.full_rows = recv.permute(1, 0, 2).reshape(local.shape[0], -1)
                return None, None
            return self._compact(local), None
        drafted = head.logits(self.ids, self.lay)
        m = drafted.shape[0]
        values, index = drafted.max(dim=-1)
        packed = torch.stack([values, (index + t.rank * t.shard).float()], -1).contiguous()
        recv = torch.empty((t.world, m, 2), dtype=torch.float32, device=t.device)
        t.comm.all_gather(packed, recv)
        best = recv[:, :, 0].argmax(dim=0)          # first (lowest) rank among equal maxima
        proposals = recv[best, self.prop_arange, 1].long()
        self.verify.copy_(self.ids)
        self.verify.index_copy_(0, self.proposal_rows, proposals)
        local, taps = t.forward(self.verify, self.lay, prompt=False, logits="all", taps=True)
        head.append(taps, self.lay)
        self.local = local
        return self._compact(local), proposals

    def run(self):
        """Execute the staged shape: (compact [world, rows, 3], proposals or None) on device."""
        if self.graph is not None:
            self.graph.replay()
            return self.out
        return self.body()

    def land(self, compact, proposals):
        """Enqueue rank zero's copies on the replay stream, before the round's done event."""
        if self.host_compact is not None:
            self.host_compact.copy_(compact, non_blocking=True)
        if self.host_proposals is not None:
            self.host_proposals.copy_(proposals, non_blocking=True)
        return self.host_compact, self.host_proposals

    def capture(self, pool):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):          # warm allocations/kernels outside capture
                self.body()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            self.out = self.body()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = graph

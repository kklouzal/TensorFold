"""CUDA graphs for Gemma 4 SM70 decode and DFlash rounds, captured identically on every rank at startup.

One graph per shape: serial decode (one row) and, with DFlash, one round graph per previously committed
row count c in 0..block (append c accepted taps, propose block - 1 rows greedily, verify the block). Inputs
arrive through one pinned staging vector copied to the device before each replay: [start, anchor,
previous start]. Graphs read positions from device memory, and full-attention grids cover the allocated
context, so a graph serves every position. Outputs (gathered logits, proposals) are graph-owned static
buffers, valid until the next replay.
"""

from __future__ import annotations

import torch


class Graphs:
    def __init__(self, engine):
        self.e = engine
        t = engine.target
        dev = t.device
        self.stage_host = torch.zeros(4, dtype=torch.int64, pin_memory=True)
        self.stage = torch.zeros(4, dtype=torch.int64, device=dev)
        self.pos = torch.zeros(1, dtype=torch.int32, device=dev)          # target start
        self.prev_pos = torch.zeros(1, dtype=torch.int32, device=dev)     # previous verify start (append)
        self.block_pos = torch.zeros(1, dtype=torch.int32, device=dev)    # drafter committed length
        self.ids = torch.zeros(16, dtype=torch.int64, device=dev)
        self.pool = torch.cuda.graph_pool_handle()
        self.serial = None
        self.rounds = {}
        self.serial_logits = None
        self.round_out = {}

    def _stage(self, start, anchor, previous):
        h = self.stage_host
        h[0], h[1], h[2] = start, anchor, previous
        self.stage.copy_(h, non_blocking=True)
        self.pos.copy_(self.stage[0:1])
        self.prev_pos.copy_(self.stage[2:3])
        self.block_pos.copy_(self.stage[0:1])

    def _serial_body(self):
        e, t = self.e, self.e.target
        st = e._state
        st.pos = self.pos
        local, _ = t.forward(self.ids[:1], st, logits="all", taps=False, staged=True)
        return e._gathered(local)

    def _round_body(self, committed):
        e, t, head = self.e, self.e.target, self.e.head
        st = e._state
        st.pos = self.pos
        if committed:
            head.append(e._taps[:committed], 0, pos=self.prev_pos)
        local = head.logits(self.stage[1:2], head.block - 1, pos=self.block_pos)
        n = local.shape[0]
        values, index = local.max(dim=-1)
        packed = torch.stack([values, (index + t.rank * t.shard).float()], -1).contiguous()
        recv = torch.empty((t.world, n, 2), dtype=torch.float32, device=t.device)
        t.comm.all_gather(packed, recv)
        best = recv[:, :, 0].argmax(dim=0)
        proposals = recv[best, torch.arange(n, device=t.device), 1].long()
        self.ids[0:1].copy_(self.stage[1:2])
        self.ids[1:n + 1].copy_(proposals)
        logits, taps = t.forward(self.ids[:n + 1], st, logits="all", taps=True, staged=True)
        e._taps = taps
        return e._gathered(logits), proposals

    def _capture(self, body):
        st = self.e._state
        length = st.length
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):          # warm allocations/kernels outside capture
                st.length = length
                out = body()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        st.length = length
        with torch.cuda.graph(graph, pool=self.pool, stream=stream):
            out = body()
        torch.cuda.current_stream().wait_stream(stream)
        st.length = length
        return graph, out

    @torch.inference_mode()
    def capture(self):
        """Every rank captures in the same order (the bodies contain collectives)."""
        e = self.e
        st = e._state
        saved_pos = st.pos
        try:
            st.length = 0
            self._stage(0, 0, 0)
            self.serial, self.serial_logits = self._capture(self._serial_body)
            if e.head is not None:
                e.head.length = 1
                for c in range(e.head.block + 1):
                    graph, out = self._capture(lambda c=c: self._round_body(c))
                    self.rounds[c] = graph
                    self.round_out[c] = out
        finally:
            st.pos = saved_pos
            st.reset()
            if e.head is not None:
                e.head.reset()
        torch.cuda.synchronize()

    def run_serial(self, start, token):
        st = self.e._state
        self._stage(start, token, 0)
        self.ids[0:1].copy_(self.stage[1:2])
        self.serial.replay()
        st.length = start + 1
        return self.serial_logits

    def run_round(self, start, anchor, previous, committed):
        st, head = self.e._state, self.e.head
        self._stage(start, anchor, previous)
        self.rounds[committed].replay()
        st.length = start + head.block
        head.length = start
        logits, proposals = self.round_out[committed]
        return logits, proposals

"""Startup CUDA graphs for DeepSeek-V4.1 SM70 decode forwards (one row per stream, B streams).

A ``Shape`` (bucket B) owns static device inputs: one staged int64 [3, B] upload per round (token ids, positions,
slots), constant token_req / qsl / valid, and the batch's block table gathered inside the graph from ``SlotPages``
(every slot's KV pages at the fixed width of the admitted context, -1 past its pages). The graph body is the eager
forward (``Target.begin/layers/end``) plus the logits all-gather, so a replay issues exactly the eager kernels and
collectives. Captured identically on every rank at startup (bodies contain NCCL calls), largest bucket first, in
one shared memory pool; never lazily.

Padding rows (B real streams in a larger bucket) are position-0 tokens of the scratch slot (the last slot) whose
block-table row is all -1: the hash/C2 commits and the ring insert skip the scratch slot (slot < S), the packed KV
inserts and index/attention reads skip page -1, so padding writes no state and real rows never read padding rows
(every cross-row operation is per token row or per request). The window ring and C2/hash state of the scratch slot
stay at their reset values.

Buckets: the friend's vLLM FULL_DECODE_ONLY capture sizes (1..16, 24, 32) up to --parallel, plus --parallel.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .target import Step

BUCKETS = tuple(range(1, 17)) + (24, 32)


def buckets(streams: int) -> tuple[int, ...]:
    return tuple(sorted({b for b in BUCKETS if b < streams} | {streams}))


class SlotPages:
    """Device page table [slots, width] int32 of every slot's KV pages (-1 past them); the scratch slot stays -1.
    Updated on every rank by attach/detach (stream order: before any later forward reads it)."""

    def __init__(self, slots: int, width: int, device):
        self.width = width
        self.table = torch.full((slots, width), -1, dtype=torch.int32, device=device)
        self._row = np.full(width, -1, dtype=np.int32)

    def set(self, slot: int, pages) -> None:
        if len(pages) > self.width:
            raise ValueError(f"slot {slot}: {len(pages)} KV pages exceed the {self.width}-page admitted context")
        row = self._row
        row.fill(-1)
        row[:len(pages)] = pages
        # pageable source: staged at once by the copy, so the host row may be rewritten immediately
        self.table[slot].copy_(torch.from_numpy(row), non_blocking=True)


class Shape:
    def __init__(self, engine, count: int):
        t = engine.target
        dev = t.device
        self.e, self.count = engine, count
        self.scratch = engine.slots - 1
        self.pad_id = engine.geometry.bos
        self.host = np.zeros((3, count), dtype=np.int64)
        self.staged = torch.zeros((3, count), dtype=torch.int64, device=dev)
        req_slot = torch.full((count + 8,), self.scratch, dtype=torch.int32, device=dev)
        self.req_slot = req_slot
        self.rows = torch.arange(count, dtype=torch.long, device=dev)
        self.step = Step(ids=self.staged[0], positions=self.staged[1],
                         valid=torch.ones(count, dtype=torch.bool, device=dev),
                         token_req=torch.arange(count, dtype=torch.int32, device=dev),
                         qsl=torch.arange(count + 1, dtype=torch.int32, device=dev), qsl_cpu=list(range(count + 1)),
                         num_reqs=count, req_slot=req_slot,
                         block_table=torch.full((count, engine.slot_pages.width), -1, dtype=torch.int32, device=dev),
                         prefill=[], decode_index=None)
        self.graph = None
        self.out = None

    def stage(self, entries, ids) -> None:
        """entries [(slot, position, 1)] in row order, ids their tokens; the rest of the bucket pads."""
        n = len(entries)
        h = self.host
        h[0, :n] = ids
        h[1, :n] = [entry[1] for entry in entries]
        h[2, :n] = [entry[0] for entry in entries]
        h[0, n:], h[1, n:], h[2, n:] = self.pad_id, 0, self.scratch
        self.staged.copy_(torch.from_numpy(h), non_blocking=True)

    def body(self):
        e, t = self.e, self.e.target
        step = self.step
        self.req_slot[:self.count].copy_(self.staged[2])
        torch.index_select(e.slot_pages.table, 0, self.req_slot[:self.count], out=step.block_table)
        state = t.begin(step)
        t.layers(state, t.units)
        return e.coll.gather(t.end(state, self.rows))

    def run(self):
        """The staged batch's full-vocabulary FP32 logits [count, vocab] (graph-owned: valid until the next run)."""
        if self.graph is not None:
            self.graph.replay()
            return self.out
        return self.body()

    def capture(self, pool, stream) -> None:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):          # compile/grow every kernel and scratch outside capture
                self.body()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            self.out = self.body()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = graph


class DecodeGraphs:
    """Every bucket's Shape, captured at startup on every rank in the same order."""

    def __init__(self, engine, streams: int):
        self.e = engine
        self.sizes = buckets(streams)
        self.shapes = {b: Shape(engine, b) for b in self.sizes}
        torch.cuda.synchronize()
        before = torch.cuda.mem_get_info()[0]
        self.reserved_before = torch.cuda.memory_reserved()
        t0 = time.perf_counter()
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()            # one capture stream (one cuBLAS workspace) for every bucket
        self.trace = []                         # (bucket, device bytes in use after its capture, torch reserved)
        # Inference mode as every forward (the warm-up's persistent scratch is inference tensors); the static
        # inputs above stay normal tensors so staging works from any caller.
        with torch.inference_mode():
            for b in sorted(self.sizes, reverse=True):
                shape = self.shapes[b]
                shape.stage([], [])
                shape.capture(pool, stream)
                torch.cuda.synchronize()
                self.trace.append((b, before - torch.cuda.mem_get_info()[0], torch.cuda.memory_reserved()))
        torch.cuda.synchronize()
        self.seconds = time.perf_counter() - t0
        torch.cuda.empty_cache()
        self.bytes = before - torch.cuda.mem_get_info()[0]

    def bucket(self, n: int) -> int | None:
        return next((b for b in self.sizes if b >= n), None)

    def run(self, entries, ids):
        """Replay the smallest bucket holding ``entries``; rank 0 gets its own copy of the real rows' logits."""
        shape = self.shapes[self.bucket(len(entries))]
        shape.stage(entries, ids)
        out = shape.run()
        return out[:len(entries)].clone() if self.e.rank == 0 else None

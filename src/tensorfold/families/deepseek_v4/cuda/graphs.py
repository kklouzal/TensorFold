"""Startup CUDA graphs for DeepSeek-V4.1 SM70 decode forwards (one row per stream, B streams).

A ``Shape`` (bucket B) owns static device inputs: one staged int64 [3, B] upload per round (token ids, positions,
slots), constant token_req / qsl / valid, and the batch's block table gathered inside the graph from ``SlotPages``
(every slot's KV pages at the fixed width of the admitted context, -1 past its pages). The graph body is the eager
forward (``Target.begin/layers/end``) plus the logits all-gather, so a replay issues exactly the eager kernels and
collectives. The graph's output is the compact greedy packet (per rank and row: shard maximum, its global column, a
NaN/+inf flag; all-gathered, 8 x [B, 3] FP32: ``round_sampling``); the full-vocabulary rows are gathered eagerly after
the replay only when a caller needs them (every rank alike: the OP_STEP flags). Every rank also derives each row's
greedy token from the gathered packet on the device (``Shape.next_ids``: the first rank holding the row maximum,
its column; the same rule as ``round_sampling.greedy_tokens`` on the host), so a run-ahead round (``run(ahead=True)``)
takes its token ids from the previous replay without a host round trip (``engine.Decoder``). Captured identically on every rank at startup (bodies contain NCCL calls), largest bucket first, in
one shared memory pool; never lazily.

Padding rows (B real streams in a larger bucket) are position-0 tokens of the scratch slot (the last slot) whose
block-table row is all -1: the hash/C2 commits and the ring insert skip the scratch slot (slot < S), the packed KV
inserts and index/attention reads skip page -1, so padding writes no state and real rows never read padding rows
(every cross-row operation is per token row or per request). The window ring and C2/hash state of the scratch slot
stay at their reset values.

Buckets: the friend's vLLM FULL_DECODE_ONLY capture sizes (1..16, 24, 32) up to --parallel, plus --parallel.

Index classes: buckets of at least ``CLASS_MIN`` streams are also captured with the decode indexer sized for
``INDEX_CLASSES`` positions below the admitted context (``Target.index_context``): a replay takes the smallest class
holding every row's position + 1, so the indexer's per-rank FP32 scores, their 8-rank sum and the top-512 read
cover the live lengths, not the context (every row's top-512 reads only its own (position + 1) // ratio scores
either way). The cross-rank sum's collective follows the narrower size (``Collectives.path``).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .round_sampling import compact_packet
from .target import Step

BUCKETS = tuple(range(1, 17)) + (24, 32)
INDEX_CLASSES = (8192, 32768)
CLASS_MIN = 4


def buckets(streams: int) -> tuple[int, ...]:
    return tuple(sorted({b for b in BUCKETS if b < streams} | {streams}))


def shape_keys(streams: int, context: int) -> list[tuple[int, int]]:
    """Every captured (bucket, index context) pair."""
    classes = [c for c in INDEX_CLASSES if c < context]
    return [(b, c) for b in buckets(streams) for c in ([*classes, context] if b >= CLASS_MIN else [context])]


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
    def __init__(self, engine, count: int, index_context: int):
        t = engine.target
        dev = t.device
        self.e, self.count, self.index_context = engine, count, index_context
        self.scratch = engine.slots - 1
        self.pad_id = engine.geometry.bos
        self.host = np.zeros((3, count), dtype=np.int64)
        self.staged = torch.zeros((3, count), dtype=torch.int64, device=dev)
        req_slot = torch.full((count + 8,), self.scratch, dtype=torch.int32, device=dev)
        self.req_slot = req_slot
        self.rows = torch.arange(count, dtype=torch.long, device=dev)
        # This graph also accepts a one-token prompt. It has the same
        # single-row slot ownership and no transactional backup.
        pack_window_ring = t.rotor_cache is not None
        self.step = Step(ids=self.staged[0], positions=self.staged[1],
                         valid=torch.ones(count, dtype=torch.bool, device=dev),
                         token_req=torch.arange(count, dtype=torch.int32, device=dev),
                         qsl=torch.arange(count + 1, dtype=torch.int32, device=dev), qsl_cpu=list(range(count + 1)),
                         num_reqs=count, req_slot=req_slot,
                         block_table=torch.full((count, engine.slot_pages.width), -1, dtype=torch.int32, device=dev),
                         prefill=[], decode_index=None, single_token_slots=True, pack_window_ring=pack_window_ring)
        self.graph = None
        self.out = None
        self.local = None               # this rank's FP32 logits shard [count, vocab / 8] (graph-owned)
        self.next_ids = None            # int64 [count]: each row's greedy token from the packet (graph-owned)
        # rank 0: the compact packet's landing buffers, alternating (two decode batches in flight: run-ahead)
        self.host_compact = [torch.empty((engine.world, count, 3), dtype=torch.float32, pin_memory=True)
                             for _ in range(2)] if engine.rank == 0 else None
        self.flip = 0

    def stage(self, entries, ids) -> None:
        """entries [(slot, position, 1)] in row order, ids their tokens; the rest of the bucket pads."""
        n = len(entries)
        slots = [entry[0] for entry in entries]
        if self.step.pack_window_ring and (len(set(slots)) != n or any(not 0 <= slot < self.scratch for slot in slots)):
            raise ValueError("DeepSeek decode graph needs distinct live slots")
        h = self.host
        h[0, :n] = ids
        h[1, :n] = [entry[1] for entry in entries]
        h[2, :n] = slots
        h[0, n:], h[1, n:], h[2, n:] = self.pad_id, 0, self.scratch
        self.staged.copy_(torch.from_numpy(h), non_blocking=True)

    def body(self):
        e, t = self.e, self.e.target
        step = self.step
        self.req_slot[:self.count].copy_(self.staged[2])
        torch.index_select(e.slot_pages.table, 0, self.req_slot[:self.count], out=step.block_table)
        state = t.begin(step)
        t.index_context = self.index_context
        try:
            t.layers(state, t.units)
        finally:
            t.index_context = t.context
        local = t.end(state, self.rows)
        self.local = local
        packed = compact_packet(local, e.rank)
        recv = torch.empty((e.world,) + tuple(packed.shape), dtype=torch.float32, device=local.device)
        e.comm.all_gather(packed, recv)
        # greedy_tokens' rule on the device: torch.argmax takes the first (lowest) rank among equal maxima
        choice = recv[:, :, 0].argmax(0)
        self.next_ids = recv[:, :, 1].gather(0, choice[None])[0].long()
        return recv

    def run(self):
        """The staged batch's compact packet [world, count, 3] (graph-owned: valid until the next run)."""
        if self.graph is not None:
            self.graph.replay()
            return self.out
        return self.body()

    def capture(self, pool, stream) -> None:
        t0 = time.perf_counter()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.body()                 # compile/grow every kernel and scratch outside capture (one run does it)
        torch.cuda.current_stream().wait_stream(stream)
        stream.synchronize()            # (startup timeline: the warm-up runs' share of the capture time)
        self.warm_seconds = time.perf_counter() - t0
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
        context = engine.target.context
        self.keys = shape_keys(streams, context)
        self.classes = {b: sorted(c for k, c in self.keys if k == b) for b in self.sizes}
        self.last = None                        # (Shape, rows) of the last replay (run-ahead token source)
        self.shapes = {key: Shape(engine, *key) for key in self.keys}
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
            for key in sorted(self.keys, reverse=True):
                shape = self.shapes[key]
                shape.stage([], [])
                shape.capture(pool, stream)
                torch.cuda.synchronize()
                self.trace.append((key, before - torch.cuda.mem_get_info()[0], torch.cuda.memory_reserved()))
        torch.cuda.synchronize()
        self.seconds = time.perf_counter() - t0
        self.warm_seconds = sum(shape.warm_seconds for shape in self.shapes.values())
        torch.cuda.empty_cache()
        self.bytes = before - torch.cuda.mem_get_info()[0]

    def bucket(self, n: int) -> int | None:
        return next((b for b in self.sizes if b >= n), None)

    def index_class(self, bucket: int, entries) -> int:
        """The smallest captured index context of ``bucket`` holding every row's position + 1."""
        need = max(entry[1] for entry in entries) + 1
        return next(c for c in self.classes[bucket] if c >= need)

    def run(self, entries, ids, compact=False, full=True, ahead=False):
        """Replay the smallest bucket holding ``entries``. ``full``: the real rows' full-vocabulary FP32 logits are
        gathered after the replay (every rank). Rank 0 gets those rows (``compact`` False), or (the compact packet's
        pinned host buffer [world, bucket, 3], copied behind the replay, landed once the caller's later event
        completes; the rows or None); other ranks get None. ``ahead``: the rows' token ids are the previous replay's
        device greedy tokens (``ids`` ignored): the same rows in the same order, one position on, stream-ordered
        behind that replay. A packet buffer stays valid until the second later run."""
        n = len(entries)
        bucket = self.bucket(n)
        shape = self.shapes[(bucket, self.index_class(bucket, entries))]
        if ahead:
            last = self.last
            if last is None or last[1] != n:
                raise RuntimeError("a run-ahead decode round needs the previous replay of the same rows")
            shape.stage(entries, [shape.pad_id] * n)
            shape.staged[0, :n].copy_(last[0].next_ids[:n])
        else:
            shape.stage(entries, ids)
        packet = shape.run()
        self.last = (shape, n)
        rows = self.e.coll.gather(shape.local[:n]) if full else None
        if self.e.rank != 0:
            return None
        if not compact:
            return rows
        host = shape.host_compact[shape.flip]
        shape.flip ^= 1
        host.copy_(packet, non_blocking=True)
        return host, rows

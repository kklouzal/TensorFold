"""Gemma 4 26B-A4B on eight V100s: TP8/ETP4 ranks driven by rank 0, up to ``--parallel`` streams together.

Rank 0 owns the HTTP App, the shared CUDA Scheduler (tensorfold.cuda.scheduler: admission queue, cancellation,
per-request delivery on the caller's thread) and this provider's ``Decoder``: KV slots, prompt chunks and
batched decode rounds. Ranks 1-7 run ``follow``. Every model operation is announced to the followers through
the same-host control pipes as one small frame before rank 0 issues its own identical kernels and collectives,
so all ranks submit the same NCCL sequence. Followers never sample.

Memory: every slot has sliding-window rings from startup; a stream's full-attention caches (target and drafter)
are attached at admission with the capacity its prompt and reply may reach, and detached when it finishes.
Admission takes a slot and its bytes from a ledger whose budget is the smallest free memory over the ranks after
warm-up, graph capture and the measured prompt transient; a request that does not fit waits (``NoRoom``) until
a stream finishes.
"""

from __future__ import annotations

import array
import math
import os
import struct
import time
from pathlib import Path
from typing import Any, Callable

from tensorfold.engine.draft_acceptance import accepted_prefix as _accepted

from .ops import (OP_ATTACH, OP_BATCH, OP_COPY, OP_DETACH, OP_DROP, OP_FORWARD, OP_GROW, OP_HRESTORE, OP_HSPILL,
                  OP_LAYERS, OP_RESTORE, OP_SNAP, OP_STOP)
from .prefix import PrefixOwner
F_LOGITS_LAST, F_APPEND, F_SCORE = 2, 4, 16
STOP_SHIFT = 8                  # OP_FORWARD flags >> STOP_SHIFT: the layer a split prompt forward pauses at (0: all)
SCORE_TOP = 20
MAX_STREAMS = 32
SAFETY = 1 << 30                # bytes left free per rank beyond the measured prompt transient
RING_SHARE = (1, 16)            # default keep_slots: all slots' target rings in this share of the free memory
_HEAD = struct.Struct("<qqqqq")  # op, a, b, c, d
PIPE_TIMEOUT = 600.0
TRACE = os.environ.get("TF_GEMMA_ROUND_TRACE") == "1"     # diagnostic: per-shape round timing lines on rank 0


HOST_DEFAULT_BYTES = 64 << 30   # host prefix tier across the eight ranks when --prefix-host-bytes is unset


def _mem_available() -> int:
    """The host's MemAvailable (bytes), or 0 when /proc/meminfo has none (the tier then stays off)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


class Gemma4SM70Engine:
    tp = 8
    concurrent = True
    supports_canonical_prefix_archive = False
    supports_teacher_forcing = True
    penalties_supported = True

    def __init__(self, model_dir: Path, *, rank: int, tp: int, master: str, master_port: int,
                 context: int | None = None, context_explicit: bool | None = None, drafter: str = "",
                 drafter_bits: int = 8, depth: int | None = None, no_drafts: bool = False, streams: int = 1,
                 eos_token_ids=None, chat_template_file=None, chat_template_name=None, graphs: bool = True,
                 keep_slots: int | None = None, prefix_host_bytes: int | None = None,
                 queue_timeout_s: float | None = None, kv_format: str = "fp16") -> None:
        import torch
        from tensorfold.checkpoint_metadata import read_config
        from tensorfold.cuda.comm import open_comm
        from tensorfold.token_metadata import resolve_eos
        from . import weights as W
        from .target import Target, CHUNK, DECODE_ROWS, prompt_rows
        from .graphs import Shape, MODE_SERIAL, MODE_ROUND, MODE_SERIAL_TAPS, buckets

        if tp != W.WORLD or not 0 <= rank < tp:
            raise ValueError("Gemma 4 SM70 serving requires --tp 8 (TP8 attention/dense, ETP4 experts)")
        if type(streams) is not int or not 1 <= streams <= MAX_STREAMS:
            raise ValueError(f"Gemma 4 SM70 serves --parallel 1 to {MAX_STREAMS}")
        raw = read_config(model_dir)
        text = raw.get("text_config") or raw
        native = int(text["max_position_embeddings"])
        self.context_window = native if not context else min(int(context), native)
        self.rank, self.world = rank, tp
        self.streams = streams
        self.prompt_rows = PROMPT_ROWS = prompt_rows()     # --prefill-rows (identical environment on every rank)
        torch.cuda.set_device(0)
        # The NVLink small all-gather for the per-layer FP32 decode partials is this provider's measured default
        # (6.2 -> 5.0 ms per serial step); every rank sets the same value before the communicator opens.
        os.environ.setdefault("TF_SMALL_GATHER_BYTES", "262144")
        # row_reduce/fp32_rows: the bulk mailbox of the decode rows' FP32 all-reduce (Target._reduce)
        self.comm = open_comm(rank, tp, master or "127.0.0.1", master_port, row_reduce=True, fp32_rows=True)
        self.w = W.load(model_dir, rank)
        self.snaps = {}                           # prefix snapshot id -> (target rings, drafter rings) on this rank
        # Host prefix tier (prefix_host.py, on by default; --prefix-host-bytes 0 disables): one pinned slab per rank,
        # allocated before the KV budget and the HTTP host pool are measured; host entry id -> its tensors' slab
        # regions on this rank. Unset: HOST_DEFAULT_BYTES across the ranks, but at most a quarter of the host's
        # MemAvailable, agreed as the smallest rank's choice (every rank must lay out identical blocks).
        if prefix_host_bytes is not None and (type(prefix_host_bytes) is not int or prefix_host_bytes < 0):
            raise ValueError("prefix_host_bytes must be a nonnegative byte count")
        # Per rank, a power of two (pinned_slab_bytes: the pinned allocator rounds up to one, so this is the bytes
        # actually pinned) of at least one 2 MiB block.
        from tensorfold.families.qwen4_exp.cuda.prefix_host import pinned_slab_bytes
        explicit = prefix_host_bytes is not None
        if not explicit:
            prefix_host_bytes = min(HOST_DEFAULT_BYTES, _mem_available() // 4)
        mine = torch.tensor([pinned_slab_bytes(prefix_host_bytes // tp)], dtype=torch.int64, device="cuda")
        agreed = torch.empty(tp, dtype=torch.int64, device="cuda")
        self.comm.all_gather(mine, agreed)
        self.host_per_rank = int(agreed.min())
        self.host_slab, failure = None, None
        if self.host_per_rank:                    # rank-agreed, so every rank takes this branch and its all-gather
            try:
                self.host_slab = torch.empty(self.host_per_rank, dtype=torch.uint8, pin_memory=True)
            except RuntimeError as error:         # cudaHostAlloc refused (host memory/locking limits)
                failure = error
            pinned = torch.tensor([failure is None], dtype=torch.int64, device="cuda")
            every = torch.empty(tp, dtype=torch.int64, device="cuda")
            self.comm.all_gather(pinned, every)
            refused = [r for r, ok in enumerate(every.tolist()) if not ok]
            if refused:
                self.host_slab = None
                message = (f"pinning the {self.host_per_rank} B/rank host prefix tier failed on rank(s) "
                           f"{refused}" + ("" if failure is None else f": {failure}"))
                if explicit:
                    raise RuntimeError(f"--prefix-host-bytes {prefix_host_bytes}: {message}") from failure
                if rank == 0:
                    print(f"[tensorfold] Gemma 4 SM70: {message}; serving without the host prefix tier "
                          "(--prefix-host-bytes sets it explicitly)", flush=True)
                self.host_per_rank = 0
        self.host_layout, self.host_extents = {}, {}    # prefix_host entries' snapshots, row extents
        # Slots: the live streams', ``keep_slots`` more for kept prefix chains (idle conversations, resumed in place
        # from VRAM), and the last one pads graph batches. Each slot's rings exist from startup, while a kept chain's
        # full caches come from the KV budget, so the slot count, not the budget, caps how many conversations stay
        # hot. Unset: as many slots as fit their target rings into RING_SHARE of the device memory free after the
        # weights (the smallest rank's count; at least max(8, streams) kept; rings stay FP16 for every --kv-dtype).
        if keep_slots is None:
            from .target import ring_slot_bytes
            free = torch.tensor([torch.cuda.mem_get_info()[0]], dtype=torch.int64, device="cuda")
            every = torch.empty(tp, dtype=torch.int64, device="cuda")
            self.comm.all_gather(free, every)
            fit = int(every.min()) * RING_SHARE[0] // RING_SHARE[1] // ring_slot_bytes(self.w, PROMPT_ROWS)
            keep_slots = max(8, streams, fit - streams - 1)
        if type(keep_slots) is not int or keep_slots < 0:
            raise ValueError("keep_slots must be a nonnegative integer")
        self.keep_slots = keep_slots
        slots = streams + keep_slots + 1
        self.pad = slots - 1
        self.target = Target(self.w, self.comm, context=self.context_window, slots=slots, kv_format=kv_format,
                             prompt_rows=PROMPT_ROWS)
        self.vocab = self.w.config.vocab
        self.head = None
        self.block = 1
        if drafter and not no_drafts:
            from .dflash import DFlash
            block = 3 if depth is None else depth + 1
            self.head = DFlash(drafter, self.target, context=self.context_window, block=block, bits=drafter_bits,
                               slots=slots)
            self.block = block
        self.layer_count = len(self.w.layers)
        self.part = None                # a paused prompt forward (_forward/_layers), identical on every rank
        self.layer_marks = None         # rank 0: per-layer CUDA events of prompt forwards (Decoder part sizing)
        self.max_rows = self.block
        self.max_draft = self.block - 1
        self._gather = torch.empty(tp * 16 * self.target.shard, dtype=torch.float32, device="cuda")
        from tensorfold.cuda.control_pipes import ControlPipes
        self.pipes = ControlPipes(self.comm.store, rank, tp, timeout_s=PIPE_TIMEOUT)
        if rank == 0:
            from tensorfold.cuda.chat_template import prepare_chat_template
            self.chat_template = prepare_chat_template(model_dir, template_file=chat_template_file,
                                                       template_name=chat_template_name)
            self.eos = resolve_eos(model_dir, text.get("eos_token_id"), vocab_size=self.vocab,
                                   override=eos_token_ids)
            if sorted(self.pipes.piped()) != list(range(1, tp)):
                raise RuntimeError("Gemma 4 SM70 requires all eight ranks on one host (control pipes)")
        # Startup runs identically on every rank (collectives), without control frames.
        self._attach(self.pad, PROMPT_ROWS + CHUNK)
        self.shapes = {}
        modes = (MODE_SERIAL, MODE_ROUND, MODE_SERIAL_TAPS) if self.head is not None else (MODE_SERIAL,)
        for mode in modes:
            for count in buckets(streams):
                self.shapes[(mode, count)] = Shape(self, mode, count)
        # Grow every prompt scratch to its admitted maximum (the scoring op sizes the all-row logits scratch),
        # measuring the transient peak a prompt chunk adds, then every decode scratch at the widest shape.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        warm = PROMPT_ROWS
        self.layer_marks = [] if rank == 0 else None
        self._forward(self.pad, 0, warm, F_SCORE | (F_APPEND if self.head is not None else 0), [0] * (warm + 1))
        torch.cuda.synchronize()
        transient = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()
        if rank == 0:     # the Decoder's first per-layer prompt cost estimate (position 0; serving refines it)
            (_, events), = self.layer_marks
            self.warm_layer_ms = [events[j - 1].elapsed_time(events[j]) / warm for j in range(1, len(events))]
            self.layer_marks = None
        # Prompt tails of at most DECODE_ROWS rows (a prompt's last chunk, or a resumed prompt's suffix) take the
        # decode arithmetic eagerly: size their scratch too (graph shapes may be narrower than DECODE_ROWS rows).
        self._forward(self.pad, 0, DECODE_ROWS, F_SCORE | (F_APPEND if self.head is not None else 0),
                      [0] * (DECODE_ROWS + 1))
        for mode in modes:
            shape = self.shapes[(mode, streams)]
            shape.stage([self.pad] * streams, [0] * streams, [0] * streams)
            shape.run()
        self.graphs = graphs
        if graphs:
            pool = torch.cuda.graph_pool_handle()
            for key in sorted(self.shapes, key=lambda k: (-k[1], k[0])):
                shape = self.shapes[key]
                shape.stage([self.pad] * shape.count, [0] * shape.count, [0] * shape.count)
                shape.capture(pool)
            self.target.frozen = True
        # Calibrated round costs (ms) per shape: the leader's prior for choosing DFlash or serial rounds.
        self.costs = {}
        for key, shape in self.shapes.items():
            shape.stage([self.pad] * shape.count, [0] * shape.count, [0] * shape.count)
            shape.run()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                shape.run()
            torch.cuda.synchronize()
            self.costs[key] = (time.perf_counter() - t0) * 1000 / 3
        torch.cuda.synchronize()
        torch.cuda.empty_cache()            # graph pools keep their blocks; free cached ones count as device free
        free, total = torch.cuda.mem_get_info()
        own = free - transient - SAFETY
        mine = torch.tensor([own], dtype=torch.int64, device="cuda")
        every = torch.empty(tp, dtype=torch.int64, device="cuda")
        self.comm.all_gather(mine, every)
        self.budget = max(0, int(every.min()))
        self.token_bytes = self.target.pool.token_bytes() + (self.head.token_bytes() if self.head else 0)
        self._closed = False
        self.vision = None
        self.decoder = self.scheduler = None
        if rank == 0:
            fits = self.budget // self.slot_bytes(self.target.alloc)
            print(f"[tensorfold] Gemma 4 SM70: --parallel {streams}, KV budget {self.budget / 2**30:.2f} GiB/rank "
                  f"({self.token_bytes} B/token, full-attention KV {kv_format}, + FP16 rings): {fits} streams of "
                  f"{self.context_window} tokens fit at once; longer admissions wait for memory; {self.keep_slots} "
                  "slots keep idle conversations", flush=True)
            curve = ", ".join(f"{'?SRT'[m]}{b}: {ms:.1f}" for (m, b), ms in sorted(self.costs.items()))
            print(f"[tensorfold] round ms by shape (calibration, position 0): {curve}", flush=True)
            from tensorfold.cuda.markers import resume_points
            from tensorfold.cuda.scheduler import Scheduler
            self.points = resume_points(model_dir, turn=True, template=self.chat_template)
            self.decoder = Decoder(self)
            # Requests beyond --parallel wait in the scheduler's queue; --queue-timeout-s bounds that wait (503 after).
            self.scheduler = Scheduler(self.decoder, max_streams=streams, queue_timeout_s=queue_timeout_s)
            try:
                self.scheduler.start()
            except BaseException as primary:
                try:
                    self.scheduler.close()
                except BaseException as cleanup:
                    raise primary from cleanup
                raise
        torch.cuda.synchronize()

    # ---------------------------------------------------------------------------------------------- memory
    def slot_bytes(self, capacity: int) -> int:
        """Ledger bytes of one admitted slot: its full-attention caches, each allocation rounded to 2 MiB."""
        from .target import CHUNK
        capacity = -(-capacity // CHUNK) * CHUNK
        granule = 2 << 20
        rows = [self.target.pool.token_bytes()] + ([self.head.token_bytes()] if self.head is not None else [])
        return sum(-(-capacity * b // granule) * granule for b in rows)

    def _attach(self, slot, capacity):
        """A new sequence in ``slot``: full caches of ``capacity`` rows (kept when already that size), empty rings."""
        pool = self.target.pool
        if pool.capacity[slot] != capacity:
            self._detach(slot)
            pool.attach(slot, capacity)
            if self.head is not None:
                self.head.attach(slot, capacity)
        else:
            pool.ring_p[:, slot].fill_(-1)
            if self.head is not None:
                self.head.ring_p[:, slot].fill_(-1)

    def _detach(self, slot):
        self.target.pool.detach(slot)
        if self.head is not None:
            self.head.detach(slot)

    def _prefix_op(self, op, a, b, c):
        """Prefix-cache operations (identical on every rank; rank 0 decides, prefix.py)."""
        pool, head = self.target.pool, self.head
        if op == OP_SNAP:                      # slot a, position b -> snapshot id c
            self.snaps[c] = (pool.snapshot(a, b), head.snapshot(a, b) if head is not None else None)
        elif op == OP_RESTORE:                 # snapshot b into slot a's rings
            target, drafter = self.snaps[b]
            pool.restore(a, target)
            if head is not None:
                head.restore(a, drafter)
        elif op == OP_DROP:
            del self.snaps[a]
        elif op == OP_COPY:                    # full-cache rows [0, c) of slot a into slot b
            pool.copy_rows(a, b, c)
            if head is not None:
                head.copy_rows(a, b, c)
        elif op == OP_GROW:                    # slot a to capacity b keeping rows [0, c)
            pool.grow(a, b, c)
            if head is not None:
                head.grow(a, b, c)

    def _host_op(self, op, a, b, c, d, ids):
        """Host prefix tier copies (prefix_host.py), ordered on the device stream; slab offsets follow from the
        copied tensors' shapes, identical on every rank. A row extent stores rows [first, end) row-major (row, layer,
        k/v, width), so any row range of it is one contiguous host span."""
        import torch
        pool, head, slab = self.target.pool, self.head, self.host_slab
        caches = (pool,) if head is None else (pool, head)
        if op == OP_HSPILL:                    # entry a <- slot b: extent rows [first, c) from its block, snapshots
            n = ids[0]                         # from block d; ids: n, sids, extent id (-1: none), first, block, gone
            sids, (xid, first, xblock), gone = ids[1:1 + n], ids[1 + n:4 + n], ids[4 + n:]
            for hid in gone:
                self.host_layout.pop(hid, None)
                self.host_extents.pop(hid, None)
            at = 0

            def put(t):
                nonlocal at
                size = t.numel() * t.element_size()
                slab[at:at + size].view(t.dtype).view(t.shape).copy_(t, non_blocking=True)
                region = (at, t.dtype, tuple(t.shape))
                at += size
                return region

            if xid >= 0:
                at = xblock * (2 << 20)
                self.host_extents[xid] = (first, [put(cache.full[b][:, :, first:c].permute(2, 0, 1, 3))
                                                  for cache in caches])
            at = d * (2 << 20)
            regions, snaps = [], []
            for sid in sids:
                target, drafter = self.snaps[sid]
                snaps.append((target[:2], None if drafter is None else drafter[:2]))
                for t in (target[2], target[3]) + (() if drafter is None else (drafter[2], drafter[3])):
                    regions.append(put(t))
            self.host_layout[a] = (regions, snaps)
        else:                                  # OP_HRESTORE: entry a -> slot b; ids: d (entry snapshot index, new
            #                                    snapshot id) pairs, then c (extent id, first row, end row) copies
            regions, snaps = self.host_layout[a]

            def get(region):
                at, dtype, shape = region
                size = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
                return slab[at:at + size].view(dtype).view(shape)

            def rows_of(region, lo, hi):        # rows [lo, hi) of an extent region (row-major: contiguous)
                at, dtype, shape = region
                row = torch.Size(shape[1:]).numel() * torch.empty((), dtype=dtype).element_size()
                return slab[at + lo * row:at + hi * row].view(dtype).view((hi - lo, *shape[1:]))

            for xid, lo, hi in zip(ids[2 * d::3], ids[2 * d + 1::3], ids[2 * d + 2::3]):
                first, rows = self.host_extents[xid]
                for cache, region in zip(caches, rows):
                    cache.full[b][:, :, lo:hi].permute(2, 0, 1, 3).copy_(rows_of(region, lo - first, hi - first),
                                                                          non_blocking=True)
            device = pool.full[b].device
            per = 2 if head is None else 4
            dev = lambda i: get(regions[i]).to(device, non_blocking=True)
            for k, sid in zip(ids[0:2 * d:2], ids[1:2 * d:2]):
                base = per * k
                target, drafter = snaps[k]
                self.snaps[sid] = ((*target, dev(base), dev(base + 1)),
                                   None if drafter is None else (*drafter, dev(base + 2), dev(base + 3)))

    # ---------------------------------------------------------------------------------------------- shared ops
    def _forward(self, slot, start, rows, flags, ids):
        """One stream's prompt chunk at ``start``; returns its gathered last-row logits, scores or None. A prompt
        chunk may pause after layer ``flags >> STOP_SHIFT`` (``_layers`` continues it; decode rounds run between):
        it then returns None. The pause changes no kernel or row, so the chunk's arithmetic is the same."""
        from .target import Layout, DECODE_ROWS
        if self.part is not None:
            raise RuntimeError("a Gemma 4 prompt forward began while another one is paused")
        score = None
        if flags & F_SCORE:                 # payload carries one extra id: the token after the last row
            ids, score = ids[:rows], ids[1:rows + 1]
        prompt = rows > DECODE_ROWS          # <= 16-row tails keep the decode arithmetic
        heads = (2, self.head.heads) if self.head is not None else (2,)
        lay = Layout(self.target.device, rows, 1, heads)
        lay.stage([slot], [start], ids)
        lay.upload()
        tokens = lay.ids32.long()
        mode = "all" if score is not None else "last" if flags & F_LOGITS_LAST else "none"
        append = bool(flags & F_APPEND) and self.head is not None
        state = self.target.begin(tokens, lay, prompt=prompt, logits=mode, taps=append)
        self.part = (state, lay, flags, score, append)
        stop = flags >> STOP_SHIFT
        return self._layers(stop if prompt and 0 < stop < self.layer_count else self.layer_count)

    def _layers(self, stop):
        """Advance the paused prompt forward to layer ``stop``; at the last layer finish it (``_forward``'s result)."""
        if stop < 0:                         # the request left: drop the paused forward
            self.part = None
            return None
        state, lay, flags, score, append = self.part
        self.target.layers(state, stop, self.layer_marks)
        if stop < self.layer_count:
            return None
        self.part = None
        local, taps = self.target.end(state)
        if append:
            self.head.append(taps, lay, prompt=state["prompt"])
        if score is not None:
            scores = self._score_rows(local, score)
            # F_LOGITS_LAST too: a scored prompt's last chunk also returns its last row (same forward, so the
            # scorer's prompt arithmetic is the serving path's).
            return (scores, self._gathered(local[-1:])) if flags & F_LOGITS_LAST else scores
        return self._gathered(local)

    def _score_rows(self, local, targets):
        """Per row: (log-sum-exp, target logit, top values, top ids) merged across vocabulary shards."""
        import torch
        rows, shard = local.shape
        base = self.rank * shard
        lse = torch.logsumexp(local, -1)
        t = torch.tensor(targets, dtype=torch.int64, device="cuda") - base
        inside = (t >= 0) & (t < shard)
        tl = torch.where(inside, local.gather(1, t.clamp(0, shard - 1)[:, None])[:, 0],
                         torch.full_like(lse, float("-inf")))
        tv, ti = torch.topk(local, SCORE_TOP, -1)
        packet = torch.cat([lse[:, None], tl[:, None], tv, (ti + base).float()], 1).contiguous()
        recv = torch.empty((self.world,) + tuple(packet.shape), dtype=torch.float32, device="cuda")
        self.comm.all_gather(packet, recv)
        glse = torch.logsumexp(recv[:, :, 0], 0)
        target = recv[:, :, 1].max(0).values - glse
        values = recv[:, :, 2:2 + SCORE_TOP].permute(1, 0, 2).reshape(rows, -1)
        index = recv[:, :, 2 + SCORE_TOP:].permute(1, 0, 2).reshape(rows, -1)
        top, order = torch.topk(values, SCORE_TOP, -1)
        return target.tolist(), (top - glse[:, None]).tolist(), index.gather(1, order).long().tolist()

    def _gathered(self, local):
        """Complete vocabulary rows [n, vocab], owned by the caller (never a view of the shared receive buffer:
        a pipelined round reads its rows after the next prompt chunk has gathered)."""
        import torch
        if local is None:
            return None
        n = local.shape[0]
        recv = (self._gather[:self.world * n * local.shape[1]].view(self.world, n, local.shape[1]) if n <= 16
                else torch.empty((self.world, n, local.shape[1]), device="cuda"))
        self.comm.all_gather(local.contiguous(), recv)
        out = recv.permute(1, 0, 2).reshape(n, -1)
        return out.clone() if out.data_ptr() == recv.data_ptr() else out

    def _batch(self, mode, count, entries, full):
        """One decode round over ``entries`` [(slot, start, anchor)] padded to the ``count`` shape. Returns the
        device compact maxima, proposals, the gathered complete rows of the ``full`` segments (or None) and
        whether those rows are indexed by segment (every segment gathered) rather than by position in ``full``."""
        import torch
        shape = self.shapes[(mode, count)]
        pad = count - len(entries)
        slots = [e[0] for e in entries] + [self.pad] * pad
        starts = [e[1] for e in entries] + [0] * pad
        anchors = [e[2] for e in entries] + [0] * pad
        if TRACE:
            self.marks = [time.perf_counter()]
        shape.stage(slots, starts, anchors)
        if TRACE:
            self.marks.append(time.perf_counter())
        compact, proposals = shape.run()
        if TRACE:
            self.marks.append(time.perf_counter())
        rows = None
        if shape.gather_all:                           # every row gathered inside the graph (graph-owned)
            return compact, proposals, shape.full_rows, True
        if full:
            n = shape.n
            if full == list(range(len(full))):          # a prefix of the segments: a view, no index upload
                local = shape.local[:len(full) * n]
            else:                                       # pageable non-blocking upload: never waits for the GPU
                index = torch.tensor([g * n + r for g in full for r in range(n)], dtype=torch.int64)
                local = shape.local.index_select(0, index.to("cuda", non_blocking=True))
            rows = self._gathered(local)
        return compact, proposals, rows, False

    def _run(self, op, a, b, c, d, ids):
        if op == OP_ATTACH:
            return self._attach(a, b)
        if OP_SNAP <= op <= OP_GROW:
            return self._prefix_op(op, a, b, c)
        if op in (OP_HSPILL, OP_HRESTORE):
            return self._host_op(op, a, b, c, d, ids)
        if op == OP_DETACH:
            return self._detach(a)
        if op == OP_FORWARD:
            return self._forward(a, b, c, d, ids)
        if op == OP_LAYERS:
            return self._layers(a)
        if op == OP_BATCH:
            n = 3 * c
            entries = [tuple(ids[i:i + 3]) for i in range(0, n, 3)]
            return self._batch(a, b, entries, list(ids[n:]))
        raise RuntimeError(f"unknown Gemma 4 SM70 control operation {op}")

    def announce(self, op, a=0, b=0, c=0, d=0, ids=()):
        self.pipes.send(_HEAD.pack(op, a, b, c, d) + array.array("i", ids).tobytes())
        return self._run(op, a, b, c, d, ids)

    def follow(self) -> None:
        import torch
        with torch.inference_mode():
            while True:
                try:
                    frame = self.pipes.receive(PIPE_TIMEOUT)
                except TimeoutError:
                    continue
                if frame is None:
                    return
                op, a, b, c, d = _HEAD.unpack_from(frame)
                if op == OP_STOP:
                    return
                ids = array.array("i")
                ids.frombytes(frame[_HEAD.size:])
                self._run(op, a, b, c, d, list(ids))

    # ---------------------------------------------------------------------------------------------- leader
    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens: Callable[[list[int]], Any],
                 draft: bool = True, stop_eos: bool = True, *, vision=None, constraint=None,
                 background: bool = False, progress=None, penalties=None, cache_prompt: bool = True,
                 probabilities=None, **_ignored) -> dict[str, Any]:
        if self.rank != 0:
            raise RuntimeError("only rank 0 generates")
        if vision is not None:
            raise ValueError("Gemma 4 SM70 does not serve image/video input yet")
        if not isinstance(prompt, (list, tuple)) or not prompt or any(
                type(t) is not int or not 0 <= t < self.vocab for t in prompt):
            raise ValueError("Gemma 4 needs a nonempty list of in-vocabulary prompt token IDs")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if len(prompt) + max_tokens > self.context_window:
            raise ValueError(f"prompt and reply exceed the {self.context_window}-token admitted context")
        if self._closed or self.scheduler is None:
            raise RuntimeError("Gemma 4 SM70 engine is closed")
        options = {"background": True} if background else {}
        if penalties is not None and penalties.active:
            options["penalties"] = penalties
        return self.scheduler.submit(list(prompt), max_tokens, sampling, draft, on_tokens, stop_eos=stop_eos,
                                     constraint=constraint, probabilities=probabilities, progress=progress, **options)

    def drain(self) -> None:
        """Requests retire inside the scheduler; nothing to fence for a caller."""

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.rank == 0:
            try:
                if self.scheduler is not None:
                    self.scheduler.close()
                    self.scheduler = None
            finally:
                self.pipes.send(_HEAD.pack(OP_STOP, 0, 0, 0, 0))


class Decoder(PrefixOwner):
    """The shared Scheduler's decoder on rank 0: slots, prompt chunks and batched decode rounds.

    A round runs at most one prompt chunk (the oldest foreground filling request's next prompt_rows rows: fixed
    boundaries, so a prompt's numerics never depend on load), or while streams decode one layer group of it, and
    one batched round over every decoding stream (decode interleaves by scheduling: one round between chunks or
    layer groups). Greedy requests
    without grammar/penalties draft with DFlash (``block`` verify rows a stream) when the measured round costs
    and acceptance favour it over serial steps; other requests take one row a round. All decoding streams of a
    mode share one batched forward (graph shape padded to a bucket)."""

    PROBE = 32          # serial rounds of drafting streams before one DFlash round refreshes the acceptance rate
    # Target decode gap while a prompt fills (ms): one layer group of the prompt chunk plus one decode round. The
    # user's budget is a live-stream gap p99 <= 200 ms; the margin covers cost-estimate error and the p99 spread.
    GAP_MS = 175.0

    def __init__(self, engine: Gemma4SM70Engine):
        self.e = engine
        self.streams: dict = {}
        self.filling: list = []
        self.next_id = 0
        self.broken: BaseException | None = None
        self.eos = tuple(engine.eos)
        self.tokens_per_round = float(engine.block)     # DFlash tokens a stream gains per round (EMA)
        self.serial_rounds = 0
        self.costs = dict(engine.costs)                 # measured ms per (mode, bucket), EMA
        self.trace = {}
        engine.marks = [0.0, 0.0, 0.0]
        self.inflight = []                              # launched groups whose tokens are not decided yet
        self.split = None                               # (stream, chunk, remaining layer stops) of a paused chunk
        self.round_ms = max(engine.costs.values())      # GPU time of a recent decode round (EMA; CUDA events)
        self.layer_ms = engine.warm_layer_ms            # per prompt row, per layer: the last measured chunk
        self.measuring = None                           # (rows, marks) of the chunk being timed
        import torch
        # The round's own GPU time, start to done: launch-to-done would include the prompt group queued ahead of it
        # (the whole gap), which would shrink the next groups.
        self.started, self.done = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self._init_prefix(engine, engine.points)
        if engine.host_per_rank:
            from .prefix_host import HostTier
            self.host_tier = HostTier(self, engine.host_per_rank)

    @property
    def broken(self) -> BaseException | None:
        return self._broken

    @broken.setter
    def broken(self, error: BaseException | None) -> None:
        """Fail closed: once an operation fails on the leader the ranks' states may differ, so the first failure
        retires the whole cohort through the TP supervisor (``cuda.cohort.notify_failure``: every rank ends and the
        service restarts) instead of leaving a server that refuses every request."""
        if error is None or getattr(self, "_broken", None) is not None:
            self._broken = getattr(self, "_broken", None) if error is None else self._broken
            return
        self._broken = error
        print(f"[tensorfold] Gemma 4 SM70 cohort failed: {type(error).__name__}: {error}; retiring every rank",
              flush=True)
        from tensorfold.cuda.cohort import notify_failure
        try:
            notify_failure()
        except Exception as failure:          # noqa: BLE001 - the original failure stays the served error
            print(f"[tensorfold] TP supervisor notification failed: {failure}", flush=True)

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    def usage(self) -> dict:
        """/stats' KV-memory and prefix-cache readings (ints and lists this worker replaces whole)."""
        budget = self.e.budget
        active = sum(self._bytes(s.st) for s in (*self.streams.values(), *self.filling) if s.st is not None)
        per_token = self.e.token_bytes
        return {"kv": {"budget_bytes": budget, "held_bytes": self.held, "active_bytes": active,
                       "usage_ratio": round(active / budget, 6) if budget > 0 else None, "bytes_per_token": per_token,
                       "free_tokens_estimate": max(0, budget - active) // per_token if per_token else None},
                "prefix_cache": self.prefix_usage()}

    def _check(self):
        if self.broken is not None:
            raise RuntimeError("Gemma 4 SM70 ranks may be out of step after an error; restart the server") \
                from self.broken

    def _ends(self, s):
        return self.eos if s.stop_eos else ()

    # ------------------------------------------------------------------------------------------ admission
    def admit(self, s) -> None:
        from tensorfold.cuda.memory_gate import NoRoom
        from tensorfold.engine.penalties import History
        self._check()
        e = self.e
        if s.vision is not None:
            raise ValueError("Gemma 4 SM70 does not serve image/video input yet")
        if len(s.prompt) + s.count > e.context_window:
            raise ValueError(f"prompt and reply exceed the {e.context_window}-token admitted context")
        capacity = self._cap(len(s.prompt) + s.count + e.block)
        need = e.slot_bytes(capacity)
        if need > e.budget:
            raise ValueError(f"a {len(s.prompt)}-token prompt with {s.count} reply tokens needs {need >> 20} MiB of "
                             f"KV per GPU; {e.budget >> 20} MiB is available")
        # Prompt reuse: drafting-eligible requests without media; a teacher-forcing sink only when it asks to borrow
        # (it scores the rows it computes). Sampling does not change prompt state (positive temperature reuses).
        reuse = bool(s.draft and s.vision is None)
        if hasattr(s.probabilities, "add_prompt") and not getattr(s.probabilities, "reuse", False):
            reuse = False
        try:
            self.place(s, capacity, reuse)
        except NoRoom:
            raise
        except BaseException as error:
            self.broken = error
            raise
        s.sid, self.next_id = self.next_id, self.next_id + 1
        s.history = History(s.penalties, s.prompt) if s.penalties is not None and s.penalties.active else None
        s.forced = getattr(s.probabilities, "forced", None)
        s.greedy = s.sampling is None or s.sampling.temperature <= 0 or s.forced is not None
        # Sampled requests draft too: every verify row is sampled from the target with the request's position-keyed
        # sampler and a proposal survives only if it equals that token (accepted_prefix), so the reply is the serial
        # sampler's reply.
        s.drafts = bool(s.draft and e.head is not None and s.constraint is None and s.history is None)
        s.full = not (s.greedy and s.constraint is None and s.history is None and s.probabilities is None)
        self.filling.append(s)

    def cancel(self, gone) -> list:
        """Filling requests whose clients left end now (their slots retire in ``finish``)."""
        out = []
        for s in gone:
            if self.split is not None and self.split[0] is s:
                # its paused chunk is abandoned on every rank: rows at or past the chunk start are rewritten before
                # any read (kept points lie at or before it and carry their own ring snapshots)
                self.split = None
                self.e.announce(OP_LAYERS, -1)
            if s in self.filling:
                self.filling.remove(s)
                s.done, s.finished = True, time.perf_counter()
                out.append(s)
        return out

    def finish(self, done) -> None:
        for s in done:
            self.streams.pop(getattr(s, "sid", None), None)
            if s in self.filling:
                self.filling.remove(s)
            if getattr(s, "st", None) is None:
                continue
            s.slot = None
            if self.broken is None:
                try:
                    self.retire_slot(s)
                except BaseException as error:
                    self.broken = error
                    raise

    def drop(self, error=None) -> list:
        """After an error in a round: forget every live stream; the ranks can no longer be trusted to agree."""
        live = [s for s in self.streams.values() if not s.done] + self.filling
        self.streams, self.filling, self.inflight, self.split = {}, [], [], None
        if self.broken is None:
            self.broken = error or RuntimeError("a Gemma 4 SM70 round failed")
        return live

    # ------------------------------------------------------------------------------------------ rounds
    def round(self) -> list:
        """One pipelined step: wait for the batch in flight and decide its tokens, launch the next batch at once,
        then (while it runs) deliver the decided tokens and run the next prompt chunk. A stream that ends or is
        cancelled during delivery may have one more launched row block; its results are ignored, and its slot
        detaches after that work in stream order."""
        import torch
        self._check()
        done = []
        with torch.inference_mode():
            try:
                decided = self._collect() if self.inflight else []
                ending = {id(s) for s, chosen in decided if chosen is None or self._ends_after(s, chosen)}
                live = [s for s in self.streams.values() if not s.done and id(s) not in ending]
                if live:
                    self._launch(live, {id(s): chosen for s, chosen in decided if chosen is not None})
                for s, chosen in decided:
                    if chosen is not None:
                        s.take(chosen, self._ends(s))
                    if s.done:
                        done.append(s)
                if self.filling:
                    done += self._fill()
            except BaseException as error:
                self.broken = error
                raise
        return done

    def _ends_after(self, s, chosen):
        return len(s.out) + len(chosen) >= s.count or (s.stop_eos and chosen[-1] in self.eos)

    def _fill(self) -> list:
        """The next fixed 1,024-row chunk of the next filling request, round-robin: concurrent prompts finish
        together, so their decode batches stay full. While streams decode, a chunk runs in layer groups, one per
        round (``_plan``), so each decode round waits for one group instead of the whole chunk."""
        e = self.e
        if self.split is not None:                  # the paused chunk continues first (one paused forward at most)
            s, chunk, plan = self.split
            stop = plan.pop(0) if self.streams and plan else e.layer_count
            t0 = time.perf_counter()
            out = e.announce(OP_LAYERS, stop)
            s.prefill_s += time.perf_counter() - t0
            if stop < e.layer_count:
                return []
            self.split = None
            return self._filled(s, chunk, out)
        foreground = [x for x in self.filling if not x.background]
        s = (foreground or self.filling)[0]
        prompt, n, start = s.prompt, len(s.prompt), s.pos
        stop = min(n, start + e.prompt_rows)        # load-invariant chunk boundaries: a prompt's arithmetic never
        #                                             depends on what else the server is doing
        last = stop == n
        # Every prompt row feeds the drafter (a kept point serves later drafting requests too).
        flags = (F_LOGITS_LAST if last else 0) | (F_APPEND if e.head is not None else 0)
        t0 = time.perf_counter()
        sink = s.probabilities if hasattr(s.probabilities, "add_prompt") else None
        if s.head is not None:                     # an exact repeat: the kept prompt end's own last-row logits
            logits, s.head = s.head, None
            return self._filled(s, (start, n, True, None, False), logits)
        self.keep_before(s, start)
        plan = self._plan(stop - start)
        flags |= plan[0] << STOP_SHIFT if plan[0] < e.layer_count else 0
        if sink is not None:
            # One forward per chunk, as unscored prompts: every row whose next token is a prompt token is
            # scored; a last chunk's final row (scored against the first forced token, unused) also returns
            # its logits.
            after = prompt[stop] if not last else (s.forced[0] if s.forced is not None else 0)
            out = e.announce(OP_FORWARD, s.slot, start, stop - start, flags | F_SCORE, prompt[start:stop] + [after])
        else:
            out = e.announce(OP_FORWARD, s.slot, start, stop - start, flags, prompt[start:stop])
        s.prefill_s += time.perf_counter() - t0
        if plan[0] < e.layer_count:
            self.split = (s, (start, stop, last, sink, True), plan[1:])
            return []
        return self._filled(s, (start, stop, last, sink, True), out)

    def _filled(self, s, chunk, out) -> list:
        """A prompt chunk ``chunk`` (start, stop, last, scoring sink, computed: False for an exact repeat's kept
        logits) of ``s`` is done (``out``: its result)."""
        start, stop, last, sink, computed = chunk
        prompt, n = s.prompt, len(s.prompt)
        logits = out
        if sink is not None:                   # scored chunk: (scores) or, for a last chunk, (scores, last row)
            (tl, tv, ti), logits = out if last else (out, None)
            for r in range(stop - start - (1 if last else 0)):
                pos = start + r + 1
                sink.add_prompt([pos], [prompt[pos]], [tl[r]], [ti[r][:sink.top]], [tv[r][:sink.top]])
        if last and computed and s.keep_end:
            self.keep_end(s, logits)
        s.pos = stop
        self.filling.remove(s)
        if s.on_fill is not None:
            s.on_fill(stop)
        if not last:
            self.filling.append(s)                     # round-robin: back of the line
            return []
        s.context = list(prompt)
        s.started = time.perf_counter()
        self.streams[s.sid] = s
        try:
            if s.forced is not None:
                first = s.forced[0]
            else:
                from tensorfold.cuda.sampling import sample_rows
                first = sample_rows(self._shape(s, logits), [n], s.sampling)[0]
            if s.probabilities is not None:
                self._score(s, logits, n - 1, first)
            if s.constraint is not None:
                s.constraint.advance([first])
            if s.history is not None:
                s.history.extend([first])
        except Exception as error:          # noqa: BLE001  (this request fails alone)
            s.error, s.done, s.finished = error, True, time.perf_counter()
            return [s]
        s.take([first], self._ends(s))
        return [s] if s.done else []

    def _plan(self, rows) -> list:
        """Layer stops of the next prompt chunk (the last is every layer). With no stream decoding: one forward.
        Otherwise groups of about equal measured cost, so that a group plus a decode round stays near GAP_MS:
        a decoding stream then waits one group, not the whole chunk, for its next round. The grouping changes
        no kernel (bitwise the same chunk), so it may follow the load."""
        e = self.e
        L = e.layer_count
        from .target import DECODE_ROWS
        self._absorb()
        if rows > DECODE_ROWS:                  # time prompt-arithmetic chunks only (short tails decode eagerly)
            self.measuring = (rows, [])
            e.layer_marks = self.measuring[1]
        if not self.streams:
            return [L]
        cost = [ms * rows for ms in self.layer_ms]
        total = sum(cost)
        parts = min(L, math.ceil(total / max(self.GAP_MS - self.round_ms, self.GAP_MS / 4)))
        if parts <= 1:
            return [L]
        stops, done = [], 0.0
        for i, ms in enumerate(cost):
            if done + ms / 2 >= total * (len(stops) + 1) / parts:     # cut at the nearest layer boundary
                stops.append(i)
            done += ms
        return [x for x in stops if 0 < x < L] + [L]

    def _absorb(self) -> None:
        """Per-layer per-row costs from the timed chunk once its events completed (never waits)."""
        if self.measuring is None:
            return
        rows, marks = self.measuring
        self.measuring, self.e.layer_marks = None, None
        ms = [None] * self.e.layer_count
        for first, events in marks:
            if not events[-1].query():
                return
            for j in range(1, len(events)):
                ms[first + j - 1] = events[j - 1].elapsed_time(events[j]) / rows
        if None not in ms:
            self.layer_ms = ms

    def _mode(self, count):
        """DFlash for latency-bound batches (padded rows within one tile: the C1 policy); wider batches draft when
        the measured tokens per ms beat serial steps at this batch size (periodic probe)."""
        from .graphs import MODE_ROUND, MODE_SERIAL_TAPS, buckets
        from .target import DECODE_ROWS
        bucket = next(b for b in buckets(self.e.streams) if b >= count)
        if bucket * self.e.block <= DECODE_ROWS:
            return MODE_ROUND
        round_ms, serial_ms = self.costs[(MODE_ROUND, bucket)], self.costs[(MODE_SERIAL_TAPS, bucket)]
        if self.tokens_per_round * serial_ms >= round_ms or self.serial_rounds >= self.PROBE:
            self.serial_rounds = 0
            return MODE_ROUND
        self.serial_rounds += 1
        return MODE_SERIAL_TAPS

    def _launch(self, live, pending):
        """Launch the next batch: one forward per mode group; ``pending`` holds decided, undelivered tokens."""
        from .graphs import MODE_ROUND, MODE_SERIAL, MODE_SERIAL_TAPS, buckets
        e = self.e
        drafting = [s for s in live if s.drafts]
        serial = [s for s in live if not s.drafts]
        groups = []
        if drafting and self._mode(len(drafting)) == MODE_ROUND:
            groups.append((MODE_ROUND, drafting))
        elif drafting:
            serial = drafting + serial              # one forward; their taps keep the drafter context
        if serial:
            groups.append((MODE_SERIAL_TAPS if drafting and not groups else MODE_SERIAL, serial))
        self.inflight = []
        self.started.record()
        for mode, group in groups:
            bucket = next(b for b in buckets(e.streams) if b >= len(group))
            entries, full = [], []
            for g, s in enumerate(group):
                extra = pending.get(id(s), ())
                last = extra[-1] if extra else s.out[-1]
                entries.append((s.slot, len(s.prompt) + len(s.out) + len(extra) - 1, last))
                if s.full:
                    full.append(g)
            t0 = time.perf_counter()
            out = e.announce(OP_BATCH, mode, bucket, len(entries), 0, [x for entry in entries for x in entry] + full)
            self.inflight.append((mode, bucket, group, entries, full, out, t0))
        self.done.record()

    def _collect(self):
        """Wait (GIL released) for the batch in flight; decide each live stream's tokens: [(stream, chosen | None)]
        (None: the request failed alone, its error set)."""
        import numpy as np
        from tensorfold.cuda.sampling import sample_rows
        from .graphs import MODE_ROUND
        e = self.e
        waited = time.perf_counter()
        self.done.synchronize()
        now = time.perf_counter()
        if self.inflight:
            self.round_ms = 0.8 * self.round_ms + 0.2 * self.started.elapsed_time(self.done)
        decided = []
        for mode, bucket, group, entries, full, (compact, proposals, rows, aligned), t0 in self.inflight:
            n = e.block if mode == MODE_ROUND else 1
            key = (mode, bucket)
            self.costs[key] = 0.9 * self.costs[key] + 0.1 * (now - t0) * 1000
            proposed = proposals.view(bucket, n - 1).tolist() if proposals is not None else None
            if compact is not None:
                best = compact.cpu().numpy()                   # [world, rows, 3] (already complete)
                choice = best[:, :, 0].argmax(axis=0)          # first (lowest) rank among equal maxima
                tokens = best[choice, np.arange(best.shape[1]), 1].astype(np.int64).tolist()
                maxima = best[choice, np.arange(best.shape[1]), 0]
                bad = best[:, :, 2].any(axis=0) | ~np.isfinite(maxima)
                fullrow = {g: g if aligned else i for i, g in enumerate(full)}
            else:                                              # complete rows for every segment
                fullrow = {g: g for g in range(len(group))}
            gained = drafted = 0
            for g, s in enumerate(group):
                if s.done or self.streams.get(s.sid) is not s:
                    continue                                    # ended/cancelled/yielded after this launch
                start, anchor = entries[g][1], entries[g][2]
                inputs = [anchor] + (proposed[g] if proposed is not None else [])
                emitted = len(s.out)                    # the launch-time pending tokens were delivered since
                room = s.count - emitted
                try:
                    logits = rows[fullrow[g] * n:(fullrow[g] + 1) * n] if g in fullrow else None
                    if s.forced is not None:
                        sampled = s.forced[emitted:emitted + n]
                    elif logits is None:
                        if bad[g * n:(g + 1) * n].any():
                            raise ValueError("sampling scores contain NaN or positive infinity, or no finite score")
                        sampled = tokens[g * n:(g + 1) * n]
                    elif mode == MODE_ROUND:
                        sampled = sample_rows(logits, list(range(start + 1, start + n + 1)), s.sampling)
                    else:
                        sampled = sample_rows(self._shape(s, logits), [start + 1], s.sampling)
                    chosen = _accepted(sampled, inputs, room, self._ends(s)) if mode == MODE_ROUND else sampled[:1]
                    if s.probabilities is not None:
                        for row in range(len(chosen)):
                            self._score(s, logits[row:row + 1], start + row, chosen[row], origin=(row, n))
                    if s.constraint is not None:
                        s.constraint.advance(chosen)
                    if s.history is not None:
                        s.history.extend(chosen)
                except Exception as error:          # noqa: BLE001  (this request fails alone)
                    s.error, s.done, s.finished = error, True, time.perf_counter()
                    decided.append((s, None))
                    continue
                s.counted(n)
                if s.drafts and mode == MODE_ROUND:
                    gained += len(chosen)
                    drafted += 1
                decided.append((s, chosen))
            if drafted:
                self.tokens_per_round = 0.8 * self.tokens_per_round + 0.2 * (gained / drafted)
            if TRACE:
                k = self.trace.setdefault(key, [0] + [0.0] * 7)
                m = e.marks
                k[0] += 1
                for i, dt in enumerate((now - t0, time.perf_counter() - now, m[0] - t0, m[1] - m[0], m[2] - m[1],
                                        waited - m[2], now - waited)):
                    k[1 + i] += dt * 1000
                if k[0] % 200 == 0:
                    f = [round(v / k[0], 3) for v in k[1:]]
                    print(f"[trace] {key}: {k[0]} rounds, launch-to-done {f[0]} decide {f[1]} | send {f[2]} stage "
                          f"{f[3]} replay-call {f[4]} until-wait {f[5]} wait {f[6]} ms, DFlash tokens/round "
                          f"{self.tokens_per_round:.2f}", flush=True)
        self.inflight = []
        return decided

    @staticmethod
    def _shape(s, logits):
        if s.constraint is not None:
            logits = logits.clone()
            s.constraint.mask(logits)
        if s.history is not None:
            logits = s.history.apply(logits, copy=s.constraint is None)
        return logits

    @staticmethod
    def _score(s, logits, position, token, origin="prompt-end"):
        """Probabilities sink: log probability of ``token`` after ``position`` (+ top)."""
        import torch
        sink = s.probabilities
        lp = torch.log_softmax(logits[-1:].float(), -1)[0]
        if sink.top:
            values, ids = torch.topk(lp, sink.top)
            tops, topv = ids.tolist(), values.tolist()
        else:
            tops, topv = [], []
        nxt = position + 1
        if nxt < sink.start:
            return
        sink.add([nxt], [token], [float(lp[token])], [tops], [topv])
        if hasattr(sink, "origins"):
            sink.origins([nxt], [origin])

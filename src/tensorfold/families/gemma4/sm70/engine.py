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
import os
import struct
import time
from pathlib import Path
from typing import Any, Callable

from tensorfold.engine.draft_acceptance import accepted_prefix as _accepted

from .ops import (OP_ATTACH, OP_BATCH, OP_COPY, OP_DETACH, OP_DROP, OP_FORWARD, OP_GROW, OP_HRESTORE, OP_HSPILL,
                  OP_RESTORE, OP_SNAP, OP_STOP)
from .prefix import PrefixOwner
F_LOGITS_LAST, F_APPEND, F_SCORE = 2, 4, 16
SCORE_TOP = 20
MAX_STREAMS = 32
SAFETY = 1 << 30                # bytes left free per rank beyond the measured prompt transient
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
                 queue_timeout_s: float | None = None) -> None:
        import torch
        from tensorfold.checkpoint_metadata import read_config
        from tensorfold.cuda.comm import open_comm
        from tensorfold.token_metadata import resolve_eos
        from . import weights as W
        from .target import Target, PROMPT_ROWS, CHUNK, DECODE_ROWS
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
        self.prompt_rows = PROMPT_ROWS
        torch.cuda.set_device(0)
        # The NVLink small all-gather for the per-layer FP32 decode partials is this provider's measured default
        # (6.2 -> 5.0 ms per serial step); every rank sets the same value before the communicator opens.
        os.environ.setdefault("TF_SMALL_GATHER_BYTES", "262144")
        self.comm = open_comm(rank, tp, master or "127.0.0.1", master_port)
        self.w = W.load(model_dir, rank)
        # Slots: the live streams', ``keep_slots`` more for kept prefix chains (idle conversations), and the last
        # one pads graph batches.
        self.keep_slots = keep_slots if keep_slots is not None else max(8, streams)
        if type(self.keep_slots) is not int or self.keep_slots < 0:
            raise ValueError("keep_slots must be a nonnegative integer")
        slots = streams + self.keep_slots + 1
        self.pad = slots - 1
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
        self.host_layout = {}
        self.target = Target(self.w, self.comm, context=self.context_window, slots=slots)
        self.vocab = self.w.config.vocab
        self.head = None
        self.block = 1
        if drafter and not no_drafts:
            from .dflash import DFlash
            block = 3 if depth is None else depth + 1
            self.head = DFlash(drafter, self.target, context=self.context_window, block=block, bits=drafter_bits,
                               slots=slots)
            self.block = block
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
        self._forward(self.pad, 0, warm, F_SCORE | (F_APPEND if self.head is not None else 0), [0] * (warm + 1))
        torch.cuda.synchronize()
        transient = torch.cuda.max_memory_allocated() - torch.cuda.memory_allocated()
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
                  f"({self.token_bytes} B/token + rings): {fits} streams of {self.context_window} tokens fit at "
                  "once; longer admissions wait for memory", flush=True)
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
        copied tensors' shapes, identical on every rank."""
        import torch
        pool, head, slab = self.target.pool, self.head, self.host_slab
        if op == OP_HSPILL:                    # entry a <- slot b rows [0, c) + snapshots, from block d
            n = ids[0]
            sids, gone = ids[1:1 + n], ids[1 + n:]
            for hid in gone:
                self.host_layout.pop(hid, None)
            at = d * (2 << 20)
            regions = []

            def put(t):
                nonlocal at
                size = t.numel() * t.element_size()
                slab[at:at + size].view(t.dtype).view(t.shape).copy_(t, non_blocking=True)
                regions.append((at, t.dtype, tuple(t.shape)))
                at += size

            put(pool.full[b][:, :, :c])
            if head is not None:
                put(head.full[b][:, :, :c])
            snaps = []
            for sid in sids:
                target, drafter = self.snaps[sid]
                snaps.append((target[:2], None if drafter is None else drafter[:2]))
                for t in (target[2], target[3]) + (() if drafter is None else (drafter[2], drafter[3])):
                    put(t)
            self.host_layout[a] = (regions, snaps)
        else:                                  # OP_HRESTORE: entry a -> slot b rows [0, c), snapshot d -> id ids[0]
            regions, snaps = self.host_layout[a]

            def get(i):
                at, dtype, shape = regions[i]
                size = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
                return slab[at:at + size].view(dtype).view(shape)

            pool.full[b][:, :, :c].copy_(get(0), non_blocking=True)
            first = 1
            if head is not None:
                head.full[b][:, :, :c].copy_(get(1), non_blocking=True)
                first = 2
            per = 2 if head is None else 4
            base = first + per * d
            dev = lambda i: get(i).to("cuda", non_blocking=True)
            target, drafter = snaps[d]
            self.snaps[ids[0]] = ((*target, dev(base), dev(base + 1)),
                                  None if drafter is None else (*drafter, dev(base + 2), dev(base + 3)))

    # ---------------------------------------------------------------------------------------------- shared ops
    def _forward(self, slot, start, rows, flags, ids):
        """One stream's prompt chunk at ``start``; returns its gathered last-row logits, scores or None."""
        from .target import Layout, DECODE_ROWS
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
        local, taps = self.target.forward(tokens, lay, prompt=prompt, logits=mode, taps=append)
        if append:
            self.head.append(taps, lay, prompt=prompt)
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

    A round runs at most one prompt chunk (the oldest foreground filling request's next PROMPT_ROWS rows: fixed
    boundaries, so a prompt's numerics never depend on load) and then one batched round over every decoding
    stream (decode interleaves by scheduling, one round between chunks). Greedy requests
    without grammar/penalties draft with DFlash (``block`` verify rows a stream) when the measured round costs
    and acceptance favour it over serial steps; other requests take one row a round. All decoding streams of a
    mode share one batched forward (graph shape padded to a bucket)."""

    PROBE = 32          # serial rounds of drafting streams before one DFlash round refreshes the acceptance rate

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
        import torch
        self.done = torch.cuda.Event()
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
        self.streams, self.filling, self.inflight = {}, [], []
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
        together, so their decode batches stay full."""
        e = self.e
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
        logits = None
        if s.head is not None:                     # an exact repeat: the kept prompt end's own last-row logits
            logits, s.head, stop, last = s.head, None, n, True
        else:
            self.keep_before(s, start)
            if sink is not None:
                # One forward per chunk, as unscored prompts: every row whose next token is a prompt token is
                # scored; a last chunk's final row (scored against the first forced token, unused) also returns
                # its logits.
                after = prompt[stop] if not last else (s.forced[0] if s.forced is not None else 0)
                out = e.announce(OP_FORWARD, s.slot, start, stop - start, flags | F_SCORE,
                                 prompt[start:stop] + [after])
                (tl, tv, ti), logits = out if last else (out, None)
                for r in range(stop - start - (1 if last else 0)):
                    pos = start + r + 1
                    sink.add_prompt([pos], [prompt[pos]], [tl[r]], [ti[r][:sink.top]], [tv[r][:sink.top]])
            else:
                logits = e.announce(OP_FORWARD, s.slot, start, stop - start, flags, prompt[start:stop])
            if last and s.keep_end:
                self.keep_end(s, logits)
        s.pos = stop
        s.prefill_s += time.perf_counter() - t0
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

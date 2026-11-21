"""DeepSeek-V4.1-Flash on eight V100s: TP8 attention/dense, ETP4 x EP2 routed experts.

Rank 0 drives; ranks 1-7 run ``follow``. Every model operation is announced to the followers through the
same-host control pipes as one small frame before rank 0 issues its own identical kernels and collectives
(the Gemma 4 SM70 provider's protocol), so all ranks submit the same NCCL sequence. Only rank 0 samples.

Per-stream state lives in slots (patch 85's SlotTable semantics): KV pages, the window rings, the Engram n-gram
history and the ratio-2 compressor partials. A slot is reset when a stream takes it; the last slot is scratch.

Serving (``serve=True``, the family's ``cuda_engine``): rank 0 owns the shared CUDA Scheduler and this provider's
``Decoder`` (the Gemma 4 SM70 Decoder's round structure): one batched decode forward over every decoding stream a
round, and between rounds the next prompt chunk of a filling stream; while streams decode, a chunk runs in layer
groups sized from measured per-layer costs so a decode round waits about ``DECODE_GAP_MS`` (175 ms), not a chunk.
"""

from __future__ import annotations

import array
import os
import struct
import time
from pathlib import Path
from typing import Any, Callable

from .prefix import OPS as PREFIX_OPS, PrefixOwner

# The Gemma 4 SM70 operation codes (the prefix owner announces them; prefix.SlotState runs the prefix ones); OP_STEP
# (this provider's batched forward) takes Gemma's OP_FORWARD code.
from tensorfold.families.gemma4.sm70.ops import OP_ATTACH, OP_DETACH, OP_FORWARD as OP_STEP, OP_GROW, OP_LAYERS, OP_STOP
_HEAD = struct.Struct("<qqqqq")
PIPE_TIMEOUT = 600.0
F_SCORE, F_LAST = 1 << 20, 1 << 21  # OP_STEP d flags (low bits: the stop layer): score every row; and the last row
SCORE_TOP = 20
MAX_TOKENS = 8192          # tokens per forward (the friend's --max-num-batched-tokens); also the prompt chunk
MAX_STREAMS = 32
RESERVE = 3 << 30          # bytes left free per rank for prompt transients, after the KV pool
GRAPH_ROW_BYTES = 32 << 20  # decode graph pool bytes per row of the largest captured bucket (sizing bound)


class DS41Engine:
    tp = 8
    concurrent = True                # serving: the shared Scheduler runs concurrent streams
    supports_canonical_prefix_archive = False
    supports_teacher_forcing = True      # /v1/tensorfold/score (tools/quality_eval: the lane's perplexity gate)
    penalties_supported = True

    def __init__(self, model_dir: Path, *, rank: int, tp: int, master: str = "", master_port: int = 29551,
                 context: int | None = None, slots: int = 4, max_tokens: int = MAX_TOKENS, parallel: int | None = None,
                 serve: bool = False, queue_timeout_s: float | None = None, graphs: bool = True,
                 keep_slots: int | None = None, prefix_host_bytes: int | None = None, **_options) -> None:
        import torch
        from tensorfold.checkpoint_metadata import read_config
        from tensorfold.cuda.comm import open_comm
        from tensorfold.cuda.control_pipes import ControlPipes
        from . import weights as W
        from .engram import HostEngram
        from .collectives import Collectives
        from .target import Target

        if tp != W.WORLD or not 0 <= rank < tp:
            raise ValueError("DeepSeek-V4.1 SM70 serving requires --tp 8 (TP8 attention/dense, ETP4 x EP2 experts)")
        model_dir = Path(model_dir)
        if parallel is not None:
            if type(parallel) is not int or not 1 <= parallel <= MAX_STREAMS:
                raise ValueError(f"DeepSeek-V4.1 SM70 serves --parallel 1 to {MAX_STREAMS}")
            slots = parallel
        self.streams = slots
        raw = read_config(model_dir)
        self.geometry = g = W.Geometry.from_config(raw)
        self.context = self.context_window = min(int(context or g.max_positions), g.max_positions)
        self.rank, self.world = rank, tp
        self.vocab = g.vocab
        self.max_tokens = max_tokens
        if rank == 0:              # metadata errors fail before the weights load
            from tensorfold.token_metadata import resolve_eos
            self.eos = resolve_eos(model_dir, g.eos, vocab_size=self.vocab)
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        device = torch.device("cuda", 0)
        started = time.perf_counter()
        # The Flash Next / Gemma 4 NVLink small all-gather mailbox (collectives.py: small gathers and the LL row
        # all-reduce); every rank sets the same value before the communicator opens (its open checks they agree).
        from tensorfold.cuda import small_gather
        os.environ.setdefault(small_gather.ENV, str(small_gather.MAX_CAPACITY))
        self.comm = open_comm(rank, tp, master or "127.0.0.1", master_port)
        self.coll = Collectives(self.comm)
        # Prompt-prefix reuse (prefix.py): ``keep_slots`` more slots keep idle conversations' chains.
        from .prefix import default_keep_slots, host_slab
        if keep_slots is None:
            keep_slots = default_keep_slots(slots)
        if type(keep_slots) is not int or keep_slots < 0:
            raise ValueError("keep_slots must be a nonnegative integer")
        self.keep_slots = keep_slots
        entry = raw.get("gguf_file") or raw.get("gguf_entrypoint")
        engram_file = raw.get("engram_file")
        if not entry or not engram_file:
            raise ValueError("the DeepSeek-V4.1 model directory's config.json must name gguf_file (the backbone's "
                             "first shard) and engram_file (the llama.cpp Engram GGUF)")
        resolve = lambda p: Path(p) if Path(p).is_absolute() else model_dir / p
        shards = W.Shards(resolve(entry), resolve(engram_file))
        # Development smoke (TENSORFOLD_DS41_SMOKE_LAYERS=N): only the first N decoder layers load, so plumbing and
        # integration changes run in seconds; outputs are not the model's. Below the first Engram layer the table is
        # never read, so it is not mapped either. Numerics gates and benchmarks never set it.
        smoke = int(os.environ.get("TENSORFOLD_DS41_SMOKE_LAYERS") or 0)
        layers = range(min(smoke, g.layers)) if smoke > 0 else None
        self.engram = None
        if layers is None or len(layers) > min(g.engram_layers):
            rows = W.engram_split(shards, g)
            self.engram = HostEngram(shards.engram.path, shards.engram.offset, rows, rank, self.comm.store)
            print(f"[tensorfold] rank {rank}: Engram table mapped and registered in {self.engram.seconds:.0f} s", flush=True)
        if layers is not None:
            print(f"[tensorfold] rank {rank}: DEVELOPMENT SMOKE: {len(layers)} of {g.layers} layers loaded", flush=True)
        self._barrier()
        self.w = W.load(shards, g, rank, device, layers=layers)
        shards.close()
        torch.cuda.synchronize()
        free = torch.cuda.mem_get_info()[0]
        mine = torch.tensor([free], dtype=torch.int64, device=device)
        every = torch.empty(tp, dtype=torch.int64, device=device)
        self.comm.all_gather(mine, every)
        from .target import Pool, BLOCK
        from .kernels.mu_packed import PackedLayout
        page_bytes = PackedLayout({l: g.compress_ratios[l] for l in g.kv_sources}, BLOCK).page_bytes
        self.slots = slots + keep_slots + 1          # streams, kept chains, the scratch slot
        ring_bytes = self.slots * g.layers * 128 * 512 * 2
        # The decode graphs' private memory pool is captured after the KV pool exists, so it comes out of the free
        # memory too (measured ~28.5 MiB per row of the largest bucket, int2-gate-h1: 910 MiB at 32).
        from .graphs import buckets
        capture = graphs and not os.environ.get("TENSORFOLD_DS41_ENGRAM_TRACE")
        graph_bytes = GRAPH_ROW_BYTES * max(buckets(slots)) if capture else 0
        pages = max(1, (int(every.min()) - RESERVE - graph_bytes - ring_bytes) // page_bytes)
        self.target = Target(self.w, self.coll, slots=self.slots, pages=pages, max_tokens=max_tokens,
                             context=self.context, engram=self.engram, device=device)
        self.layer_count = len(self.w.layers)
        from .prefix import SlotState
        self.prefix = SlotState(self.target)
        self.layout = self.target.pool.layout
        self.token_bytes = self.layout.bytes_per_token
        self.snap_bytes = self.prefix.snap_bytes
        self.budget = pages * page_bytes            # the prefix owner's ledger: slot pages and snapshot pages
        # Prompt chunks: fixed multiples of prompt_rows from position 0 (load-invariant); kept prefix points lie on
        # them, so a resumed prompt computes exactly the cold chunks (prefix.py).
        self.prompt_rows = max_tokens
        self.part = None                    # a paused prompt forward (state, logits rows), identical on every rank
        self.layer_marks = None             # rank 0: per-layer CUDA events of the prompt forward being timed
        from .graphs import DecodeGraphs, SlotPages
        self.slot_pages = SlotPages(self.slots, -(-self.target.hash_state.max_context // BLOCK), device)
        self.graphs = None
        self._warm()
        # Decode forwards replay startup CUDA graphs (every rank, same order, no control frames; never lazily).
        # The Engram lookup trace (rank 0) reads ids on the host every forward, so tracing keeps the eager decode;
        # every rank decides from the same environment (all ranks capture, or none).
        if capture:
            self.graphs = DecodeGraphs(self, slots)
            spare = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            if self.graphs.bytes > graph_bytes or spare < RESERVE:
                raise RuntimeError(f"DeepSeek-V4.1 SM70 rank {rank}: decode graphs took {self.graphs.bytes >> 20} MiB "
                                   f"(sized {graph_bytes >> 20} MiB); {spare >> 20} MiB left for prompt transients, "
                                   f"{RESERVE >> 20} MiB required")
        elif rank == 0:
            print("[tensorfold] DeepSeek-V4.1 SM70: eager decode (" +
                  ("graphs disabled)" if not graphs else "TENSORFOLD_DS41_ENGRAM_TRACE set)"), flush=True)
        # The host tier's pinned slab (Gemma 4 SM70 sizing, on by default), pinned after the Engram registration and
        # the weights: pinning while the Engram table registers slowed m-pf1's registration 4x.
        pinned = time.perf_counter()
        self.host_per_rank, self.host_slab = host_slab(self.comm, rank, tp, prefix_host_bytes)
        self.prefix.slab = self.host_slab
        if rank == 0:
            print(f"[tensorfold] DeepSeek-V4.1 SM70: host prefix tier {self.host_per_rank >> 20} MiB/rank pinned in "
                  f"{time.perf_counter() - pinned:.1f} s", flush=True)
        self.pipes = ControlPipes(self.comm.store, rank, tp, timeout_s=PIPE_TIMEOUT)
        if rank == 0 and sorted(self.pipes.piped()) != list(range(1, tp)):
            raise RuntimeError("DeepSeek-V4.1 SM70 requires all eight ranks on one host (control pipes)")
        self.load_seconds = time.perf_counter() - started
        self.pages = pages
        self.page_bytes = page_bytes
        self.decoder = self.scheduler = None
        self._closed = False
        if rank == 0:
            if self.graphs is not None:
                print(f"[tensorfold] DeepSeek-V4.1 SM70: decode graphs for batches {list(self.graphs.sizes)} captured "
                      f"in {self.graphs.seconds:.1f} s, {self.graphs.bytes / 2**20:.0f} MiB", flush=True)
            engram = ("not mapped (development smoke)" if self.engram is None else
                      f"{self.engram.size / 2**30:.1f} GiB in {self.engram.seconds:.0f} s, host RAM only: VRAM residency "
                      f"{self.engram.device_bytes} B")
            print(f"[tensorfold] DeepSeek-V4.1 SM70: loaded in {self.load_seconds:.0f} s (Engram table {engram} + per-step working buffer "
                  f"{self.target.engram_rows.numel() * 2 >> 20} MiB); KV pool {pages} pages of "
                  f"{page_bytes} B ({pages * 64} tokens), {slots} streams", flush=True)
        if rank == 0 and serve:
            from tensorfold.cuda.scheduler import Scheduler
            self.points = None
            try:                                    # message starts of a rendered prompt (kept-point candidates)
                from tensorfold.cuda.markers import resume_points
                from .. import cuda_chat_template
                self.points = resume_points(model_dir, turn=True, template=cuda_chat_template(model_dir))
            except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                print(f"[tensorfold] DeepSeek-V4.1 SM70: no message-start prefix points ({error}); prompt ends only",
                      flush=True)
            self.decoder = Decoder(self)
            # Requests beyond --parallel wait in the scheduler's queue; --queue-timeout-s bounds that wait (503 after).
            self.scheduler = Scheduler(self.decoder, max_streams=slots, queue_timeout_s=queue_timeout_s)
            try:
                self.scheduler.start()
            except BaseException as primary:
                try:
                    self.scheduler.close()
                except BaseException as cleanup:
                    raise primary from cleanup
                raise

    def _warm(self):
        """Startup, every rank, no control frames: build the native extensions and compile the forward's Triton
        kernels on slot 0 (a prompt chunk past the grouped-experts and prefill-indexer thresholds, then one decode
        row), so the first request builds nothing. Slot 0 is reset by its next attach."""
        import torch
        from tensorfold.cuda.sm70 import ds41_prefill_moe, onecat_ds41
        from .target import DECODE_MAX
        ds41_prefill_moe.prepare()
        onecat_ds41.prepare()
        rows = max(ds41_prefill_moe.MIN_TOKENS, DECODE_MAX + 1) * 2
        with torch.inference_mode():
            self._attach(0, rows + 1)
            self.layer_marks = [] if self.rank == 0 else None
            self._step([(0, 0, rows)], [self.geometry.bos] * rows, [rows - 1])
            if self.rank == 0:          # the Decoder's first per-layer, per-row prompt cost estimate (ms)
                torch.cuda.synchronize()
                (_, events), = self.layer_marks
                self.warm_layer_ms = [events[j - 1].elapsed_time(events[j]) / rows for j in range(1, len(events))]
            self.layer_marks = None
            self._step([(0, rows, 1)], [self.geometry.bos], [0])
            self._detach(0)
        torch.cuda.synchronize()

    def _barrier(self):
        import torch
        x = torch.zeros(1, dtype=torch.int64, device="cuda")
        y = torch.empty(self.world, dtype=torch.int64, device="cuda")
        self.comm.all_gather(x, y)
        torch.cuda.synchronize()

    # ---------------------------------------------------------------------------------------- operations
    def _attach(self, slot, tokens):
        self.target.pool.detach(slot)
        self.slot_pages.set(slot, self.target.pool.attach(slot, tokens))
        self.target.reset_slot(slot)

    def _detach(self, slot):
        self.target.pool.detach(slot)
        self.slot_pages.set(slot, ())

    def slot_bytes(self, capacity: int) -> int:
        """Ledger bytes of a slot holding ``capacity`` positions: its pages."""
        return -(-capacity // 64) * self.page_bytes

    def _step(self, entries, ids, logits_rows, stop=0):
        flags, stop = stop & ~0xFFFF, stop & 0xFFFF
        """entries: [(slot, start, count)] in row order; ids: the tokens; logits_rows: token rows to score.
        Returns rank 0's full-vocabulary FP32 logits [len(logits_rows), vocab] (None elsewhere). With
        0 < ``stop`` < layers the forward pauses after layer ``stop`` (``_layers`` continues it) and returns None;
        the pause changes no kernel or row. One row per entry, every row scored and ``stop`` 0 (a decode batch,
        or a one-token prompt chunk) replays the batch's startup graph; ``stop`` = layers runs the same
        forward eagerly."""
        import torch
        from .target import Step, DECODE_MAX
        device = self.target.device
        pool = self.target.pool
        positions, token_req, qsl = [], [], [0]
        for slot, start, count in entries:
            if start + count > len(pool.owned.get(slot, ())) * 64:
                raise RuntimeError(f"slot {slot}: positions up to {start + count} exceed its KV pages")
        if (stop == 0 and self.graphs is not None and len(entries) <= self.graphs.sizes[-1]
                and all(count == 1 for _, _, count in entries) and list(logits_rows) == list(range(len(entries)))):
            return self.graphs.run(entries, ids)
        for r, (slot, start, count) in enumerate(entries):
            positions.extend(range(start, start + count))
            token_req.extend([r] * count)
            qsl.append(qsl[-1] + count)
        width = max(len(pool.owned[slot]) for slot, _, _ in entries)
        table = [pool.owned[slot] + [0] * (width - len(pool.owned[slot])) for slot, _, _ in entries]
        rows = len(entries) + 8
        req_slot = [slot for slot, _, _ in entries] + [self.slots - 1] * 8
        t = lambda v, dt: torch.tensor(v, dtype=dt).to(device, non_blocking=True)
        prefill = [(r, qsl[r], qsl[r + 1]) for r in range(len(entries)) if qsl[r + 1] - qsl[r] > DECODE_MAX]
        decode_index = None
        if prefill:
            inside = set()
            for _, s0, s1 in prefill:
                inside.update(range(s0, s1))
            rest = [i for i in range(qsl[-1]) if i not in inside]
            decode_index = t(rest, torch.long) if rest else torch.empty(0, dtype=torch.long, device=device)
        step = Step(ids=t(ids, torch.long), positions=t(positions, torch.long),
                    valid=torch.ones(len(ids), dtype=torch.bool, device=device), token_req=t(token_req, torch.int32),
                    qsl=t(qsl, torch.int32), qsl_cpu=qsl, num_reqs=len(entries), req_slot=t(req_slot, torch.int32),
                    block_table=t(table, torch.int32), prefill=prefill, decode_index=decode_index)
        del rows
        state = self.target.begin(step)
        if flags & F_SCORE:                # logits_rows carries each row's next token: score all rows
            rows = (logits_rows, bool(flags & F_LAST))
        else:
            rows = t(logits_rows, torch.long) if logits_rows else None
        marks = self.layer_marks if qsl[-1] > len(entries) else None     # prompt chunks only, never decode rows
        if 0 < stop < self.layer_count:
            if self.part is not None:
                raise RuntimeError("a DeepSeek-V4.1 prompt forward began while another one is paused")
            self.part = (state, rows)
            self.target.layers(state, stop, marks)
            return None
        self.target.layers(state, self.layer_count, marks)
        return self._end(state, rows)

    def _layers(self, stop):
        """Advance the paused prompt forward to layer ``stop`` (< 0: drop it); at the last layer finish it. Decode
        forwards may run between its layers (other slots; ``Target.begin``)."""
        if stop < 0:
            self.part = None
            return None
        state, rows = self.part
        self.target.layers(state, stop, self.layer_marks)
        if stop < self.layer_count:
            return None
        self.part = None
        return self._end(state, rows)

    def _end(self, state, rows):
        import torch
        if isinstance(rows, tuple):                 # F_SCORE: (targets, last)
            targets, last = rows
            local = self.target.end(state, torch.arange(len(targets), device=self.target.device))
            scores = self._score_rows(local, targets)
            if not last:
                return scores
            # the last row's logits as an unscored chunk computes them (a one-row head GEMM), so a scored request
            # and a prefix-cached repeat of an unscored one start their replies from the same bits
            last_row = torch.tensor([len(targets) - 1], dtype=torch.long, device=self.target.device)
            full = self.coll.gather(self.target.end(state, last_row))
            return (scores, full) if self.rank == 0 else None
        local = self.target.end(state, rows)
        if local is None:
            return None
        full = self.coll.gather(local)
        return full if self.rank == 0 else None

    def _score_rows(self, local, targets):
        """Per row: (target log probability, top log probabilities, top ids) merged across the vocabulary shards
        (each rank sends its log-sum-exp, target logit and top entries; the Gemma 4 SM70 provider's packet)."""
        import torch
        rows, shard = local.shape
        base = self.rank * shard
        lse = torch.logsumexp(local, -1)
        t = torch.tensor(targets, dtype=torch.int64, device=local.device) - base
        inside = (t >= 0) & (t < shard)
        tl = torch.where(inside, local.gather(1, t.clamp(0, shard - 1)[:, None])[:, 0],
                         torch.full_like(lse, float("-inf")))
        tv, ti = torch.topk(local, SCORE_TOP, -1)
        packet = torch.cat([lse[:, None], tl[:, None], tv, (ti + base).float()], 1).contiguous()
        recv = torch.empty((self.world,) + tuple(packet.shape), dtype=torch.float32, device=local.device)
        self.comm.all_gather(packet, recv)
        glse = torch.logsumexp(recv[:, :, 0], 0)
        target = recv[:, :, 1].max(0).values - glse
        values = recv[:, :, 2:2 + SCORE_TOP].permute(1, 0, 2).reshape(rows, -1)
        index = recv[:, :, 2 + SCORE_TOP:].permute(1, 0, 2).reshape(rows, -1)
        top, order = torch.topk(values, SCORE_TOP, -1)
        return target.tolist(), (top - glse[:, None]).tolist(), index.gather(1, order).long().tolist()

    def _run(self, op, a, b, c, d, ids):
        if op == OP_ATTACH:
            return self._attach(a, b)
        if op == OP_DETACH:
            return self._detach(a)
        if op == OP_STEP:
            n = 3 * a
            entries = [tuple(ids[i:i + 3]) for i in range(0, n, 3)]
            tokens = ids[n:n + b]
            rows = ids[n + b:n + b + c]
            return self._step(entries, tokens, rows, d)
        if op == OP_LAYERS:
            return self._layers(a)
        if op in PREFIX_OPS:
            out = self.prefix.run(op, a, b, c, d, ids)
            if op == OP_GROW:                   # the slot gained pages: the graphs' device page table follows
                self.slot_pages.set(a, self.target.pool.owned[a])
            return out
        raise RuntimeError(f"unknown DeepSeek-V4.1 control operation {op}")

    def announce(self, op, a=0, b=0, c=0, d=0, ids=()):
        self.pipes.send(_HEAD.pack(op, a, b, c, d) + array.array("i", ids).tobytes())
        return self._run(op, a, b, c, d, list(ids))

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
                payload = array.array("i")
                payload.frombytes(frame[_HEAD.size:])
                self._run(op, a, b, c, d, list(payload))

    # ---------------------------------------------------------------------------------------- leader
    def step(self, entries, ids, logits_rows, stop=0):
        payload = [v for e in entries for v in e] + list(ids) + list(logits_rows)
        return self.announce(OP_STEP, len(entries), len(ids), len(logits_rows), stop, payload)

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens: Callable[[list[int]], Any],
                 draft: bool = True, stop_eos: bool = True, *, vision=None, constraint=None,
                 background: bool = False, progress=None, penalties=None, cache_prompt: bool = True,
                 probabilities=None, **_ignored) -> dict[str, Any]:
        if self.rank != 0:
            raise RuntimeError("only rank 0 generates")
        if vision is not None:
            raise ValueError("DeepSeek-V4.1 SM70 does not serve image input yet")
        if not isinstance(prompt, (list, tuple)) or not prompt or any(
                type(t) is not int or not 0 <= t < self.vocab for t in prompt):
            raise ValueError("DeepSeek-V4.1 needs a nonempty list of in-vocabulary prompt token IDs")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if len(prompt) + max_tokens > self.context:
            raise ValueError(f"prompt and reply exceed the {self.context}-token admitted context")
        if self._closed or self.scheduler is None:
            raise RuntimeError("DeepSeek-V4.1 SM70 engine is not serving")
        options = {"background": True} if background else {}
        if penalties is not None and penalties.active:
            options["penalties"] = penalties
        # ``draft`` carries prompt-prefix eligibility (no drafter yet), as the Gemma 4 SM70 Decoder reads it
        return self.scheduler.submit(list(prompt), max_tokens, sampling, bool(draft and cache_prompt), on_tokens,
                                     stop_eos=stop_eos,
                                     constraint=constraint, probabilities=probabilities, progress=progress, **options)

    def drain(self) -> None:
        """Requests retire inside the scheduler; nothing to fence for a caller."""

    def greedy(self, prompt: list[int], max_tokens: int, slot: int = 0, on_token: Callable | None = None,
               stop_eos: bool = True, top_logprobs: int = 0, eager: bool = False) -> dict[str, Any]:
        """Single-stream greedy generation (the M1 correctness path): prompt chunks, then one token a step.
        With ``top_logprobs`` > 0 the result also carries each step's chosen-token logprob and its top entries.
        ``eager``: decode steps run the eager forward instead of the startup graphs (A/B checks)."""
        import torch
        if self.rank != 0:
            raise RuntimeError("only rank 0 generates")
        with torch.inference_mode():
            self.announce(OP_ATTACH, slot, len(prompt) + max_tokens)
            t0 = time.perf_counter()
            at = 0
            logits = None
            while at < len(prompt):
                n = min(self.max_tokens, len(prompt) - at)
                last = at + n == len(prompt)
                logits = self.step([(slot, at, n)], prompt[at:at + n], [n - 1] if last else [])
                at += n
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            out, logprobs = [], []

            def choose(row):
                if top_logprobs:
                    lp = torch.log_softmax(row.float(), -1)
                    values, ids = lp.topk(top_logprobs)
                    chosen = int(ids[0])
                    logprobs.append({"logprob": float(values[0]),
                                     "top": [[int(i), float(v)] for i, v in zip(ids.tolist(), values.tolist())]})
                    return chosen
                return int(row.argmax())

            token = choose(logits[-1])
            while True:
                out.append(token)
                if on_token is not None:
                    on_token(token)
                if len(out) >= max_tokens or (stop_eos and token in self.eos):
                    break
                logits = self.step([(slot, len(prompt) + len(out) - 1, 1)], [token], [0],
                                   self.layer_count if eager else 0)
                token = choose(logits[-1])
            t2 = time.perf_counter()
            self.announce(OP_DETACH, slot)
        return {"tokens": out, "logprobs": logprobs, "prefill_s": t1 - t0, "decode_s": t2 - t1,
                "decode_tok_s": (len(out) - 1) / (t2 - t1) if len(out) > 1 else 0.0}

    def score(self, prompt: list[int], continuation: list[int], slot: int = 0) -> list[float]:
        """Teacher-forced log probabilities of ``continuation`` after ``prompt`` (rank 0; prompt chunks as greedy)."""
        import torch
        if self.rank != 0:
            raise RuntimeError("only rank 0 scores")
        tokens = list(prompt) + list(continuation)
        first = len(prompt) - 1                      # row whose logits predict continuation[0]
        out = []
        with torch.inference_mode():
            self.announce(OP_ATTACH, slot, len(tokens))
            at = 0
            while at < len(tokens) - 1:
                n = min(self.max_tokens, len(tokens) - 1 - at)
                rows = [r for r in range(n) if at + r >= first]
                logits = self.step([(slot, at, n)], tokens[at:at + n], rows)
                if rows:
                    targets = torch.tensor([tokens[at + r + 1] for r in rows], device=logits.device)
                    lp = torch.log_softmax(logits.float(), -1).gather(1, targets[:, None])[:, 0]
                    out += lp.tolist()
                at += n
            self.announce(OP_DETACH, slot)
        return out

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.rank == 0:
                try:
                    if self.scheduler is not None:
                        self.scheduler.close()
                        self.scheduler = None
                finally:
                    self.pipes.send(_HEAD.pack(OP_STOP, 0, 0, 0, 0))
        finally:
            self.graphs = None              # graphs read the mapped Engram table: release them first
            if self.engram is not None:
                self.engram.close()


class Decoder(PrefixOwner):
    """The shared Scheduler's decoder on rank 0: slots, prompt chunks and batched decode rounds.

    A round decides the batch in flight, launches the next batch over every decoding stream (one row each), then
    delivers the decided tokens and runs the next prompt chunk of the next filling request (round-robin, fixed
    ``max_tokens`` boundaries: a prompt's arithmetic never depends on load). While streams decode, the chunk runs
    in layer groups (one per round) of about equal measured cost, so a decoding stream waits one group plus its
    round (about ``GAP_MS``), not the whole chunk (the Gemma 4 SM70 Decoder's schedule).

    Slots and their pages come from the prompt-prefix owner (``prefix.PrefixOwner``, the Gemma 4 SM70 / Flash Next
    policy): admission places a request in a kept chain (resume in place, fork, host tier) or cold; prompt chunks
    keep points at their boundaries and the exact prompt end; a finished stream's slot stays idle while it keeps
    points."""

    def __init__(self, engine: DS41Engine):
        import torch
        from tensorfold.cuda.geometry import DECODE_GAP_MS
        self.e = engine
        self.GAP_MS = DECODE_GAP_MS
        self.streams: dict = {}
        self.filling: list = []
        self.next_id = 0
        self._broken = None
        self.eos = tuple(engine.eos)
        self.inflight = None                         # (group, entries, logits, t0) of the launched decode batch
        self.split = None                            # (stream, chunk, remaining layer stops) of a paused chunk
        self.round_ms = 50.0                         # GPU time of a recent decode round (EMA; CUDA events)
        self.layer_ms = list(engine.warm_layer_ms)   # per prompt row, per layer: the last measured chunk
        self.measuring = None
        self.started, self.done = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self._init_prefix(engine, engine.points)

    @property
    def broken(self):
        return self._broken

    @broken.setter
    def broken(self, error):
        """Fail closed: after a failed operation the ranks may disagree; the TP supervisor retires the cohort."""
        if error is None or self._broken is not None:
            return
        self._broken = error
        print(f"[tensorfold] DeepSeek-V4.1 SM70 cohort failed: {type(error).__name__}: {error}; retiring every rank",
              flush=True)
        from tensorfold.cuda.cohort import notify_failure
        try:
            notify_failure()
        except Exception as failure:          # noqa: BLE001 - the original failure stays the served error
            print(f"[tensorfold] TP supervisor notification failed: {failure}", flush=True)

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    def usage(self) -> dict:
        """/stats' KV-memory and prefix-cache readings."""
        budget = self.e.budget
        active = sum(self._bytes(s.st) for s in (*self.streams.values(), *self.filling) if s.st is not None)
        per_token = self.e.token_bytes
        return {"kv": {"budget_bytes": budget, "held_bytes": self.held, "active_bytes": active,
                       "usage_ratio": round(active / budget, 6) if budget else None, "bytes_per_token": per_token,
                       "free_tokens_estimate": max(0, budget - active) // per_token},
                "prefix_cache": self.prefix_usage()}

    def _check(self):
        if self._broken is not None:
            raise RuntimeError("DeepSeek-V4.1 SM70 ranks may be out of step after an error; restart the server") \
                from self._broken

    def _ends(self, s):
        return self.eos if s.stop_eos else ()

    # ------------------------------------------------------------------------------------------ admission
    def admit(self, s) -> None:
        from tensorfold.cuda.memory_gate import NoRoom
        from tensorfold.engine.penalties import History
        self._check()
        e = self.e
        if s.vision is not None:
            raise ValueError("DeepSeek-V4.1 SM70 does not serve image input yet")
        if len(s.prompt) + s.count > e.context:
            raise ValueError(f"prompt and reply exceed the {e.context}-token admitted context")
        capacity = self._cap(len(s.prompt) + s.count)
        if e.slot_bytes(capacity) > e.budget:
            raise ValueError(f"a {len(s.prompt)}-token prompt with {s.count} reply tokens needs "
                             f"{-(-capacity // 64)} KV pages per GPU; {e.pages} exist")
        # Prompt reuse: eligible requests without media; a teacher-forcing sink only when it asks to borrow (it
        # scores the rows it computes). Sampling does not change prompt state.
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
        self.filling.append(s)

    def cancel(self, gone) -> list:
        """Filling requests whose clients left end now (their slots retire in ``finish``)."""
        out = []
        for s in gone:
            if self.split is not None and self.split[0] is s:
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
            if self._broken is None:
                try:
                    self.retire_slot(s)
                except BaseException as error:
                    self.broken = error
                    raise

    def drop(self, error=None) -> list:
        """After an error in a round: forget every live stream; the ranks can no longer be trusted to agree."""
        live = [s for s in self.streams.values() if not s.done] + self.filling
        self.streams, self.filling, self.inflight, self.split = {}, [], None, None
        if self._broken is None:
            self.broken = error or RuntimeError("a DeepSeek-V4.1 SM70 round failed")
        return live

    # ------------------------------------------------------------------------------------------ rounds
    def round(self) -> list:
        """Wait for the batch in flight and decide its tokens, launch the next batch at once, then (while it runs)
        deliver the decided tokens and run the next prompt chunk or layer group."""
        import torch
        self._check()
        done = []
        with torch.inference_mode():
            try:
                decided = self._collect() if self.inflight is not None else []
                ending = {id(s) for s, chosen in decided if chosen is None or self._ends_after(s, chosen)}
                live = [s for s in self.streams.values() if not s.done and id(s) not in ending]
                if live:
                    self._launch(live, {id(s): chosen for s, chosen in decided if chosen is not None})
                for s, chosen in decided:
                    if chosen is not None:
                        s.take(chosen, self._ends(s))
                    if s.done:
                        done.append(s)
                if self.filling or self.split is not None:
                    done += self._fill()
            except BaseException as error:
                self.broken = error
                raise
        return done

    def _ends_after(self, s, chosen):
        return len(s.out) + len(chosen) >= s.count or (s.stop_eos and chosen[-1] in self.eos)

    def _fill(self) -> list:
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
        n, start = len(s.prompt), s.pos
        if s.head is not None:                     # an exact repeat: the kept prompt end's own last-row logits
            logits, s.head = s.head, None
            return self._filled(s, (start, n, True, None, False), logits)
        rows = e.prompt_rows                       # chunk boundaries at multiples of rows from position 0
        stop = min(n, (start // rows + 1) * rows)
        last = stop == n
        self.keep_before(s, start)
        plan = self._plan(stop - start)
        sink = s.probabilities if hasattr(s.probabilities, "add_prompt") else None
        layer = plan[0] if plan[0] < e.layer_count else 0
        t0 = time.perf_counter()
        if sink is not None:
            # every row scored for its next prompt token (a last chunk's final row against the first forced token,
            # unused) in the serving chunk's own forward; a last chunk also returns its last row
            after = s.prompt[stop] if not last else (s.forced[0] if s.forced is not None else 0)
            out = e.step([(s.slot, start, stop - start)], s.prompt[start:stop], s.prompt[start + 1:stop] + [after],
                         layer | F_SCORE | (F_LAST if last else 0))
        else:
            out = e.step([(s.slot, start, stop - start)], s.prompt[start:stop], [stop - start - 1] if last else [],
                         layer)
        s.prefill_s += time.perf_counter() - t0
        if plan[0] < e.layer_count:
            self.split = (s, (start, stop, last, sink, True), plan[1:])
            return []
        return self._filled(s, (start, stop, last, sink, True), out)

    def _filled(self, s, chunk, out) -> list:
        """Prompt chunk ``chunk`` (start, stop, last, scoring sink, computed: False for an exact repeat's kept
        logits) of ``s`` is done (``out``: its result)."""
        start, stop, last, sink, computed = chunk
        n = len(s.prompt)
        logits = out
        if sink is not None:                   # scored chunk: scores, or (scores, last row) for a last chunk
            (tl, tv, ti), logits = out if last else (out, None)
            for r in range(stop - start - (1 if last else 0)):
                pos = start + r + 1
                sink.add_prompt([pos], [s.prompt[pos]], [tl[r]], [ti[r][:sink.top]], [tv[r][:sink.top]])
        if last and computed and s.keep_end:
            self.keep_end(s, logits)
        s.pos = stop
        self.filling.remove(s)
        if s.on_fill is not None:
            s.on_fill(stop)
        if not last:
            self.filling.append(s)                     # round-robin: back of the line
            return []
        s.context = list(s.prompt)
        s.started = time.perf_counter()
        self.streams[s.sid] = s
        try:
            first = self._choose(s, logits[-1:], n)
        except Exception as error:          # noqa: BLE001  (this request fails alone)
            s.error, s.done, s.finished = error, True, time.perf_counter()
            return [s]
        s.take([first], self._ends(s))
        return [s] if s.done else []

    def _plan(self, rows) -> list:
        """Layer stops of the next prompt chunk (the last is every layer): one forward with no stream decoding,
        otherwise groups of about equal measured cost so a group plus a decode round stays near GAP_MS. The
        grouping changes no kernel, so it may follow the load."""
        from tensorfold.cuda.geometry import group_target, layer_groups
        e = self.e
        L = e.layer_count
        self._absorb()
        if rows > 64:
            self.measuring = (rows, [])
            e.layer_marks = self.measuring[1]
        if not self.streams:
            return [L]
        return layer_groups([ms * rows for ms in self.layer_ms], group_target(self.GAP_MS, self.round_ms))

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

    def _launch(self, live, pending):
        """Launch one decode forward over ``live`` (row order = stream order); ``pending``: decided, undelivered
        tokens."""
        e = self.e
        entries, ids = [], []
        for s in live:
            extra = pending.get(id(s), ())
            token = extra[-1] if extra else s.out[-1]
            entries.append((s.slot, len(s.prompt) + len(s.out) + len(extra) - 1, 1))
            ids.append(token)
        self.started.record()
        t0 = time.perf_counter()
        logits = e.step(entries, ids, list(range(len(entries))))
        self.done.record()
        self.inflight = (live, entries, logits, t0)

    def _collect(self):
        """Wait for the batch in flight; decide each stream's token: [(stream, [token] | None)]."""
        live, entries, logits, t0 = self.inflight
        self.inflight = None
        self.done.synchronize()
        self.round_ms = 0.8 * self.round_ms + 0.2 * self.started.elapsed_time(self.done)
        decided = []
        for g, s in enumerate(live):
            if s.done or self.streams.get(s.sid) is not s:
                continue                                    # ended/cancelled/yielded after this launch
            try:
                chosen = [self._choose(s, logits[g:g + 1], entries[g][1] + 1, emitted=len(s.out))]
                s.counted(1)
            except Exception as error:          # noqa: BLE001  (this request fails alone)
                s.error, s.done, s.finished = error, True, time.perf_counter()
                decided.append((s, None))
                continue
            decided.append((s, chosen))
        return decided

    def _choose(self, s, row, position, emitted=0) -> int:
        """Sample (or force) the token at ``position`` from one full-vocabulary row; advance grammar/penalties."""
        from tensorfold.cuda.sampling import sample_rows
        if s.forced is not None:
            token = s.forced[emitted]
        else:
            if s.constraint is not None:
                row = row.clone()
                s.constraint.mask(row)
            if s.history is not None:
                row = s.history.apply(row, copy=s.constraint is None)
            token = sample_rows(row, [position], s.sampling)[0]
        if s.probabilities is not None:
            self._score(s, row, position - 1, token)
        if s.constraint is not None:
            s.constraint.advance([token])
        if s.history is not None:
            s.history.extend([token])
        return token

    @staticmethod
    def _score(s, logits, position, token):
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
            sink.origins([nxt], ["prompt-end" if nxt == len(s.prompt) else (0, 1)])

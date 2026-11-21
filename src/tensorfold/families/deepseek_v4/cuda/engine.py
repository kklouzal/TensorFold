"""DeepSeek-V4.1-Flash on eight V100s: TP8 attention/dense, ETP4 x EP2 routed experts.

Rank 0 drives; ranks 1-7 run ``follow``. Every model operation is announced to the followers through the
same-host control pipes as one small frame before rank 0 issues its own identical kernels and collectives
(the Gemma 4 SM70 provider's protocol), so all ranks submit the same NCCL sequence. Only rank 0 samples.

Per-stream state lives in slots (patch 85's SlotTable semantics): KV pages, the window rings, the Engram n-gram
history and the ratio-2 compressor partials. A slot is reset when a stream takes it; the last slot is scratch.
"""

from __future__ import annotations

import array
import os
import struct
import time
from pathlib import Path
from typing import Any, Callable

OP_ATTACH, OP_DETACH, OP_STEP, OP_STOP = 1, 2, 3, 9
_HEAD = struct.Struct("<qqqqq")
PIPE_TIMEOUT = 600.0
MAX_TOKENS = 8192          # tokens per forward (the friend's --max-num-batched-tokens)
RESERVE = 3 << 30          # bytes left free per rank for prompt transients, after the KV pool


class DS41Engine:
    tp = 8
    concurrent = False

    def __init__(self, model_dir: Path, *, rank: int, tp: int, master: str = "", master_port: int = 29551,
                 context: int | None = None, slots: int = 4, max_tokens: int = MAX_TOKENS, **_options) -> None:
        import torch
        from tensorfold.checkpoint_metadata import read_config
        from tensorfold.cuda.comm import open_comm
        from tensorfold.cuda.control_pipes import ControlPipes
        from . import weights as W
        from .engram import HostEngram
        from .target import Collectives, Target

        if tp != W.WORLD or not 0 <= rank < tp:
            raise ValueError("DeepSeek-V4.1 SM70 serving requires --tp 8 (TP8 attention/dense, ETP4 x EP2 experts)")
        model_dir = Path(model_dir)
        raw = read_config(model_dir)
        self.geometry = g = W.Geometry.from_config(raw)
        self.context = min(int(context or g.max_positions), g.max_positions)
        self.rank, self.world = rank, tp
        self.vocab = g.vocab
        self.max_tokens = max_tokens
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        device = torch.device("cuda", 0)
        started = time.perf_counter()
        self.comm = open_comm(rank, tp, master or "127.0.0.1", master_port)
        self.coll = Collectives(self.comm)
        entry = raw.get("gguf_file") or raw.get("gguf_entrypoint")
        engram_file = raw.get("engram_file")
        if not entry or not engram_file:
            raise ValueError("the DeepSeek-V4.1 model directory's config.json must name gguf_file (the backbone's "
                             "first shard) and engram_file (the llama.cpp Engram GGUF)")
        resolve = lambda p: Path(p) if Path(p).is_absolute() else model_dir / p
        shards = W.Shards(resolve(entry), resolve(engram_file))
        rows = W.engram_split(shards, g)
        self.engram = HostEngram(shards.engram.path, shards.engram.offset, rows, rank, self.comm.store)
        self._barrier()
        self.w = W.load(shards, g, rank, device)
        shards.close()
        torch.cuda.synchronize()
        free = torch.cuda.mem_get_info()[0]
        mine = torch.tensor([free], dtype=torch.int64, device=device)
        every = torch.empty(tp, dtype=torch.int64, device=device)
        self.comm.all_gather(mine, every)
        from .target import Pool, BLOCK
        from .kernels.mu_packed import PackedLayout
        page_bytes = PackedLayout({l: g.compress_ratios[l] for l in g.kv_sources}, BLOCK).page_bytes
        ring_bytes = slots * g.layers * 128 * 512 * 2
        pages = max(1, (int(every.min()) - RESERVE - ring_bytes) // page_bytes)
        self.slots = slots + 1
        self.target = Target(self.w, self.coll, slots=self.slots, pages=pages, max_tokens=max_tokens,
                             context=self.context, engram=self.engram, device=device)
        self.pipes = ControlPipes(self.comm.store, rank, tp, timeout_s=PIPE_TIMEOUT)
        self.load_seconds = time.perf_counter() - started
        if rank == 0:
            from tensorfold.token_metadata import resolve_eos
            self.eos = resolve_eos(model_dir, g.eos, vocab_size=self.vocab)
            print(f"[tensorfold] DeepSeek-V4.1 SM70: loaded in {self.load_seconds:.0f} s (Engram table "
                  f"{self.engram.size / 2**30:.1f} GiB in {self.engram.seconds:.0f} s); KV pool {pages} pages of "
                  f"{page_bytes} B ({pages * 64} tokens), {slots} streams", flush=True)

    def _barrier(self):
        import torch
        x = torch.zeros(1, dtype=torch.int64, device="cuda")
        y = torch.empty(self.world, dtype=torch.int64, device="cuda")
        self.comm.all_gather(x, y)
        torch.cuda.synchronize()

    # ---------------------------------------------------------------------------------------- operations
    def _attach(self, slot, tokens):
        self.target.pool.detach(slot)
        self.target.pool.attach(slot, tokens)
        self.target.reset_slot(slot)

    def _detach(self, slot):
        self.target.pool.detach(slot)

    def _step(self, entries, ids, logits_rows):
        """entries: [(slot, start, count)] in row order; ids: the tokens; logits_rows: token rows to score.
        Returns rank 0's full-vocabulary FP32 logits [len(logits_rows), vocab] (None elsewhere)."""
        import torch
        from .target import Step, DECODE_MAX
        device = self.target.device
        pool = self.target.pool
        positions, token_req, qsl = [], [], [0]
        for r, (slot, start, count) in enumerate(entries):
            if start + count > len(pool.owned.get(slot, ())) * 64:
                raise RuntimeError(f"slot {slot}: positions up to {start + count} exceed its KV pages")
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
        local = self.target.forward(step, t(logits_rows, torch.long) if logits_rows else None)
        if local is None:
            return None
        full = self.coll.gather(local)
        return full if self.rank == 0 else None

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
            return self._step(entries, tokens, rows)
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
    def step(self, entries, ids, logits_rows):
        payload = [v for e in entries for v in e] + list(ids) + list(logits_rows)
        return self.announce(OP_STEP, len(entries), len(ids), len(logits_rows), 0, payload)

    def greedy(self, prompt: list[int], max_tokens: int, slot: int = 0, on_token: Callable | None = None,
               stop_eos: bool = True) -> dict[str, Any]:
        """Single-stream greedy generation (the M1 correctness path): prompt chunks, then one token a step."""
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
            out = []
            token = int(logits[-1].argmax())
            while True:
                out.append(token)
                if on_token is not None:
                    on_token(token)
                if len(out) >= max_tokens or (stop_eos and token in self.eos):
                    break
                logits = self.step([(slot, len(prompt) + len(out) - 1, 1)], [token], [0])
                token = int(logits[-1].argmax())
            t2 = time.perf_counter()
            self.announce(OP_DETACH, slot)
        return {"tokens": out, "prefill_s": t1 - t0, "decode_s": t2 - t1,
                "decode_tok_s": (len(out) - 1) / (t2 - t1) if len(out) > 1 else 0.0}

    def close(self) -> None:
        if self.rank == 0:
            self.pipes.send(_HEAD.pack(OP_STOP, 0, 0, 0, 0))
        self.engram.close()

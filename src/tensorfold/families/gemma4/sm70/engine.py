"""Gemma 4 26B-A4B on eight V100s: one request lane, TP8/ETP4 ranks driven by rank 0.

Rank 0 owns the HTTP App, sampling and the request lane; ranks 1-7 run ``follow``. Every model operation is
announced to the followers through the same-host control pipes as one small frame before rank 0 issues its
own identical kernels and collectives, so all ranks submit the same NCCL sequence. Followers never sample.
"""

from __future__ import annotations

import struct
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

OP_FORWARD, OP_STOP, OP_ROUND, OP_GSERIAL, OP_GROUND = 1, 2, 3, 4, 5
F_LOGITS_ALL, F_LOGITS_LAST, F_APPEND, F_RESET, F_SCORE = 1, 2, 4, 8, 16
SCORE_TOP = 20
_HEAD = struct.Struct("<qqqq")      # op, start, rows, flags
_ROUND = struct.Struct("<qqqqq")    # start, anchor, count, previous verify start, previous committed rows
PIPE_TIMEOUT = 600.0


def _accepted(sampled, inputs, room: int, eos) -> list[int]:
    chosen = []
    for row, token in enumerate(sampled):
        chosen.append(token)
        if len(chosen) >= room or token in eos or row + 1 >= len(inputs) or token != inputs[row + 1]:
            break
    return chosen


class Gemma4SM70Engine:
    tp = 8
    concurrent = False
    supports_canonical_prefix_archive = False
    supports_teacher_forcing = True
    penalties_supported = True

    def __init__(self, model_dir: Path, *, rank: int, tp: int, master: str, master_port: int,
                 context: int | None = None, context_explicit: bool | None = None, drafter: str = "",
                 drafter_bits: int = 16, depth: int | None = None, no_drafts: bool = False, streams: int = 1,
                 eos_token_ids=None, chat_template_file=None, chat_template_name=None, graphs: bool = True) -> None:
        import torch
        from tensorfold.checkpoint_metadata import read_config
        from tensorfold.cuda.comm import open_comm
        from tensorfold.token_metadata import resolve_eos
        from . import weights as W
        from .target import Target, PROMPT_ROWS, DECODE_ROWS

        if tp != W.WORLD or not 0 <= rank < tp:
            raise ValueError("Gemma 4 SM70 serving requires --tp 8 (TP8 attention/dense, ETP4 experts)")
        if type(streams) is not int or streams != 1:
            raise ValueError("Gemma 4 SM70 currently serves one request lane (--parallel 1)")
        raw = read_config(model_dir)
        text = raw.get("text_config") or raw
        native = int(text["max_position_embeddings"])
        self.context_window = native if not context else min(int(context), native)
        self.rank, self.world = rank, tp
        self.max_rows = 1
        self.max_draft = 0
        self.prompt_rows = PROMPT_ROWS
        torch.cuda.set_device(0)
        self.comm = open_comm(rank, tp, master or "127.0.0.1", master_port)
        self.w = W.load(model_dir, rank)
        self.target = Target(self.w, self.comm, context=self.context_window)
        self.vocab = self.w.config.vocab
        self._state = self.target.make_state(self.context_window)
        self.head = None
        if drafter and not no_drafts:
            from .dflash import DFlash
            block = 5 if depth is None else depth + 1
            self.head = DFlash(drafter, self.target, context=self.context_window, block=block, bits=drafter_bits)
            self.max_rows = block
            self.max_draft = block - 1
        self._taps = None
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
        self.graphs = None
        if graphs:
            from .graphs import Graphs
            # Grow every prompt scratch to its admitted maximum before capture freezes the scratch set.
            warm = min(PROMPT_ROWS, self.context_window - 16)
            self._run(OP_FORWARD, 0, warm, F_LOGITS_ALL | F_RESET | (F_APPEND if self.head is not None else 0),
                      [0] * warm)
            # ... and every decode-arithmetic scratch to its widest (DECODE_ROWS) eager forward.
            self._run(OP_FORWARD, 0, DECODE_ROWS, F_LOGITS_ALL | F_RESET | (F_APPEND if self.head is not None else 0),
                      [0] * DECODE_ROWS)
            self.graphs = Graphs(self)
            self.graphs.capture()
            self.target.frozen = True
        self._lock = threading.Lock()
        self._closed = False
        self.vision = None
        torch.cuda.synchronize()

    # ---------------------------------------------------------------------------------- shared op
    def _run(self, op, start, rows, flags, ids):
        """Execute one announced operation on this rank. Returns gathered logits on every rank (or None)."""
        import torch
        st = self._state
        if op == OP_ROUND:
            return self._round(*ids)
        if op == OP_GSERIAL:
            return self.graphs.run_serial(ids[0], ids[1])
        if op == OP_GROUND:
            return self.graphs.run_round(*ids[:4])
        if flags & F_RESET:
            st.reset()
            if self.head is not None:
                self.head.reset()
        st.length = start
        score = None
        if flags & F_SCORE:                 # payload carries one extra id: the token after the last row
            ids, score = ids[:rows], ids[1:rows + 1]
            flags |= F_LOGITS_ALL
        tokens = torch.tensor(ids, dtype=torch.int64, device="cuda")
        mode = "all" if flags & F_LOGITS_ALL else "last" if flags & F_LOGITS_LAST else "none"
        append = bool(flags & F_APPEND) and self.head is not None
        local, taps = self.target.forward(tokens, st, logits=mode, taps=append)
        if append:
            self.head.append(taps, start)
        if score is not None:
            return self._score_rows(local, score)
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

    def _round(self, start, anchor, count, previous, committed):
        """Append the previous round's accepted taps, propose greedily, verify [anchor, proposals...]."""
        import torch
        if committed:
            self.head.append(self._taps[:committed], previous)
        local = self.head.logits(anchor, count)
        proposals = [] if local is None else self._argmax(local)
        tokens = torch.tensor([anchor, *proposals], dtype=torch.int64, device="cuda")
        self._state.length = start
        logits, taps = self.target.forward(tokens, self._state, logits="all", taps=True)
        self._taps = taps
        return self._gathered(logits), [anchor, *proposals]

    def _argmax(self, local):
        """Greedy proposals from vocabulary shards: global max, lowest token ID on ties (as a full argmax)."""
        import torch
        n = local.shape[0]
        values, index = local.max(dim=-1)
        packed = torch.stack([values, (index + self.rank * self.target.shard).float()], -1)
        recv = torch.empty((self.world, n, 2), dtype=torch.float32, device="cuda")
        self.comm.all_gather(packed.contiguous(), recv)
        best = recv[:, :, 0].argmax(dim=0)       # first (lowest) rank among equal maxima
        return [int(v) for v in recv[best, torch.arange(n, device="cuda"), 1].tolist()]

    def _gathered(self, local):
        import torch
        if local is None:
            return None
        n = local.shape[0]
        recv = (self._gather[:self.world * n * local.shape[1]].view(self.world, n, local.shape[1]) if n <= 16
                else torch.empty((self.world, n, local.shape[1]), device="cuda"))
        self.comm.all_gather(local.contiguous(), recv)
        return recv.permute(1, 0, 2).reshape(n, -1)

    def _announce(self, op, start, rows, flags, ids):
        import array
        if op in (OP_ROUND, OP_GSERIAL, OP_GROUND):
            body = _ROUND.pack(*ids, *([0] * (5 - len(ids))))
        else:
            body = array.array("i", ids).tobytes()
        payload = _HEAD.pack(op, start, rows, flags) + body
        self.pipes.send(payload)
        return self._run(op, start, rows, flags, ids)

    def follow(self) -> None:
        import array
        while True:
            try:
                frame = self.pipes.receive(PIPE_TIMEOUT)
            except TimeoutError:
                continue
            if frame is None:
                return
            op, start, rows, flags = _HEAD.unpack_from(frame)
            if op == OP_STOP:
                return
            if op in (OP_ROUND, OP_GSERIAL, OP_GROUND):
                self._run(op, start, rows, flags, list(_ROUND.unpack_from(frame, _HEAD.size)))
                continue
            ids = array.array("i")
            ids.frombytes(frame[_HEAD.size:])
            if len(ids) != rows + (1 if flags & F_SCORE else 0):
                raise RuntimeError("Gemma 4 SM70 control frame row count differs from its payload")
            self._run(op, start, rows, flags, list(ids))

    # ---------------------------------------------------------------------------------- leader
    def _prefill(self, prompt, progress, started, drafts=False, sink=None):
        """Committed prompt rows; returns the last row's logits [1, vocab]. A teacher-forcing sink scores every
        prompt row (prompt-row arithmetic) except the last, whose logits stay the generation's own."""
        logits = None
        start = 0
        while start < len(prompt):
            stop = min(len(prompt), start + self.prompt_rows)
            last = stop == len(prompt)
            flags = (F_LOGITS_LAST if last else 0) | (F_RESET if start == 0 else 0) | (F_APPEND if drafts else 0)
            if sink is not None:
                end = stop - 1 if last else stop          # rows whose next token is a prompt token
                if end > start:
                    nexts = prompt[start + 1:end + 1]
                    tl, tv, ti = self._announce(OP_FORWARD, start, end - start, flags & ~F_LOGITS_LAST | F_SCORE,
                                                prompt[start:end] + [nexts[-1]])
                    for r in range(end - start):
                        pos = start + r + 1
                        sink.add_prompt([pos], [prompt[pos]], [tl[r]], [ti[r][:sink.top]], [tv[r][:sink.top]])
                    flags &= ~F_RESET
                    start = end
                    if not last:
                        continue
                logits = self._announce(OP_FORWARD, start, stop - start, flags, prompt[start:stop])
                start = stop
                continue
            logits = self._announce(OP_FORWARD, start, stop - start, flags, prompt[start:stop])
            start = stop
            if progress is not None:
                progress(len(prompt), 0, start, (time.perf_counter() - started) * 1000)
        return logits

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens: Callable[[list[int]], Any],
                 draft: bool = True, stop_eos: bool = True, *, vision=None, constraint=None,
                 background: bool = False, progress=None, penalties=None, cache_prompt: bool = True,
                 probabilities=None, **_ignored) -> dict[str, Any]:
        import torch
        from tensorfold.cuda.sampling import sample_rows
        from tensorfold.engine.penalties import History

        if self.rank != 0:
            raise RuntimeError("only rank 0 generates")
        if vision is not None:
            raise ValueError("Gemma 4 SM70 does not serve image/video input yet")
        if not isinstance(prompt, (list, tuple)) or not prompt or any(
                type(t) is not int or not 0 <= t < self.vocab for t in prompt):
            raise ValueError("Gemma 4 needs a nonempty list of in-vocabulary prompt token IDs")
        prompt = list(prompt)
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if len(prompt) + max_tokens > self.context_window:
            raise ValueError(f"prompt and reply exceed the {self.context_window}-token admitted context")
        forced = getattr(probabilities, "forced", None)
        history = History(penalties, prompt) if penalties is not None and penalties.active else None
        queued = time.perf_counter()
        with self._lock, torch.inference_mode():
            if self._closed:
                raise RuntimeError("Gemma 4 SM70 engine is closed")
            started = time.perf_counter()
            greedy = sampling is None or sampling.temperature <= 0 or forced is not None
            drafts = bool(draft and self.head is not None and greedy and constraint is None and history is None)
            logits = self._prefill(prompt, progress, started, drafts, probabilities)
            if probabilities is not None:
                self._score(probabilities, prompt, logits, len(prompt) - 1)
            logits = self._shape(logits, constraint, history)
            first = forced[0] if forced is not None else sample_rows(logits, [len(prompt)], sampling)[0]
            if constraint is not None:
                constraint.advance([first])
            if history is not None:
                history.extend([first])
            prefill_s = time.perf_counter() - started
            decoded = time.perf_counter()
            out = [first]
            rounds = drafted = accepted = 0
            eos = tuple(self.eos) if stop_eos else ()
            stopped = bool(on_tokens([first]))
            previous = committed = 0
            while drafts and not stopped and len(out) < max_tokens and out[-1] not in eos:
                room = max_tokens - len(out)
                start = len(prompt) + len(out) - 1
                count = min(self.max_draft, room - 1)
                if self.graphs is not None and count == self.max_draft:
                    logits, proposals = self._announce(OP_GROUND, 0, 0, 0, [start, out[-1], previous, committed])
                    inputs = [out[-1], *proposals.tolist()]
                else:
                    logits, inputs = self._announce(OP_ROUND, start, 0, 0,
                                                    [start, out[-1], count, previous, committed])
                if forced is not None:
                    sampled = forced[len(out):len(out) + len(inputs)]
                else:
                    sampled = sample_rows(logits, list(range(start + 1, start + len(inputs) + 1)), sampling)
                chosen = _accepted(sampled, inputs, room, eos)
                if probabilities is not None:
                    for row in range(len(chosen)):
                        self._score(probabilities, None, logits[row:row + 1], start + row,
                                    origin=(row, len(inputs)))
                previous, committed = start, len(chosen)
                rounds += 1
                drafted += len(inputs) - 1
                accepted += len(chosen) - 1
                out.extend(chosen)
                stopped = bool(on_tokens(chosen))
            while (not drafts and not stopped and len(out) < max_tokens and out[-1] not in eos
                   and not (constraint is not None and constraint.finished)):
                position = len(prompt) + len(out) - 1
                if self.graphs is not None:
                    logits = self._announce(OP_GSERIAL, 0, 0, 0, [position, out[-1]])
                else:
                    logits = self._announce(OP_FORWARD, position, 1, F_LOGITS_ALL, [out[-1]])
                if probabilities is not None:
                    self._score(probabilities, None, logits, position, origin=(0, 1))
                logits = self._shape(logits, constraint, history)
                token = (forced[len(out)] if forced is not None else
                         sample_rows(logits, [position + 1], sampling)[0])
                if constraint is not None:
                    constraint.advance([token])
                if history is not None:
                    history.extend([token])
                out.append(token)
                rounds += 1
                stopped = bool(on_tokens([token]))
            decoded_s = time.perf_counter() - decoded
        return {"prefill_s": round(prefill_s, 4), "decode_s": round(decoded_s, 4),
                "queue_s": round(started - queued, 4), "cached": 0, "drafts": drafts, "rounds": rounds,
                "drafted": drafted, "accepted": accepted, "min_rows": 1 if rounds else 0}

    @staticmethod
    def _shape(logits, constraint, history):
        if constraint is not None:
            logits = logits.clone()
            constraint.mask(logits)
        if history is not None:
            logits = history.apply(logits, copy=constraint is None)
        return logits

    def _score(self, sink, prompt, logits, position, origin="prompt-end"):
        """Teacher-forcing sink: log probabilities of the forced next token at ``position`` (+ top)."""
        import torch
        lp = torch.log_softmax(logits[-1:].float(), -1)[0]
        if sink.top:
            values, ids = torch.topk(lp, sink.top)
            tops, topv = ids.tolist(), values.tolist()
        else:
            tops, topv = [], []
        nxt = position + 1
        if nxt < sink.start:
            return
        token = sink.forced[nxt - sink.start]
        sink.add([nxt], [token], [float(lp[token])], [tops], [topv])
        sink.origins([nxt], [origin])

    def drain(self) -> None:
        import torch
        with self._lock:
            torch.cuda.synchronize()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self.rank == 0:
                self.pipes.send(_HEAD.pack(OP_STOP, 0, 0, 0))

"""Verify MTP chains against the same keyed samples as serial decoding, committing only rows before the first mismatched draft so drafts never change emitted tokens."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from tensorfold.cuda.logprobs import capture

from tensorfold.cuda.sampling import (
    ShardSampler, TokenMap, _tp_choose, comm_gather, sample_rows, sample_shards,
    valid_scores, validate_policy, validate_rows,
)
from tensorfold.engine.exact_sampling import MARGIN, Sampling

from . import CONFIDENCE, DEPTH
from .forward import Cut, commit, cut_snapshot, forward, read_ahead
from . import image_rows
from .state import CAND, Buffers, State
from .mtp import mtp_forward
from .weights import Weights


def sample_mapped(logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None,
                  id_map: torch.Tensor) -> list[int]:
    """Apply the shared exact candidate rule to the draft vocabulary's real token IDs."""

    return sample_rows(logits, positions, sampling, id_map=id_map)


class SamplingShard(ShardSampler):
    """Bind the shared shard owner to this model's communicator at startup."""

    def __init__(self, w: Weights, width: int, *, offset: int = 0, id_map=None):
        if w.comm is None or int(w.meta["world"]) != w.comm.world:
            raise ValueError("TP sampler needs the model's matching communicator")
        super().__init__(width, w.comm.device, gather=comm_gather(w.comm), world=w.comm.world,
                         offset=offset, id_map=id_map, identity=w.comm)

    def check_model(self, w: Weights, logits: torch.Tensor) -> None:
        self.check(logits, identity=w.comm)


def tp_sample_rows(w: Weights, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None,
                   offset: int = 0, id_map: torch.Tensor | None = None, with_prob: bool = False, *,
                   plan: SamplingShard | None = None):
    """Use the shared exact shard rule and this family's original temperature-1 probability contract."""

    if plan is None:
        validate_rows(logits, positions, sampling)
        plan = SamplingShard(w, logits.shape[1], offset=offset, id_map=id_map)
    plan.check_model(w, logits)
    return sample_shards(logits, positions, sampling, plan=plan, with_prob=with_prob)


def choose_gathered(w: Weights, cand_all: torch.Tensor, R: int, positions: Sequence[int], sampling: Sampling | None,
                    with_prob: bool = False, *, logits: torch.Tensor | None = None,
                    plan: SamplingShard | None = None):
    """Use original FP32-score/FP32-lse packets; borrow original logits to repair tied truncation.

    ``cand_all`` must be the completed packet produced with these exact source
    rows by ``forward.candidates``. Its FP32 log-sum-exp covers the complete
    score domain, including omitted scores. Every rank sees the same packet and
    takes the same fallback collectives. A standalone ambiguous packet cannot
    prove omitted token IDs and therefore requires its original source.
    """

    world, width = int(w.meta["world"]), 2 * CAND + 1
    if (type(R) is not int or R < 0 or len(positions) != R
            or not isinstance(cand_all, torch.Tensor) or cand_all.dtype != torch.float32
            or not cand_all.is_cuda or not cand_all.is_contiguous() or cand_all.numel() < world * R * width):
        raise ValueError("complete CUDA candidate packet and one absolute position per row required")
    validate_policy(positions, sampling)
    if logits is not None:
        validate_rows(logits, positions, sampling)
        if logits.shape[0] != R:
            raise ValueError("gathered packet and original source must have the same rows")
        if plan is not None:
            plan.check_model(w, logits)
    if R == 0:
        return ([], []) if with_prob else []
    # Graph token words are int32. A startup owner proves this range across
    # every rank; larger signed64 identities must use the original source
    # before the lossy graph token words are interpreted.
    need_source = not _gathered_fits(sampling) or (plan is not None and not plan.graph_ids_safe)
    if need_source:
        if logits is None:
            raise ValueError("gathered policy or token-ID range requires the complete original logits")
        return tp_sample_rows(w, logits, positions, sampling, with_prob=with_prob, plan=plan,
                              offset=int(w.meta.get("vocab_offset", 0)))
    g = cand_all[:world * R * width].view(world, R, width).cpu().numpy()
    values = g[:, :, :CAND].astype(np.float32)
    tokens = np.ascontiguousarray(g[:, :, CAND:2 * CAND]).view(np.int32).astype(np.int64)
    if (tokens < 0).any():
        raise ValueError("graph-gathered token IDs must be nonnegative int32")
    valid_scores(values, require_finite=False)
    norms = g[:, :, 2 * CAND]
    if np.isnan(norms).any() or np.isposinf(norms).any():
        raise ValueError("a rank's complete sampling scores contain NaN or positive infinity")
    if not np.isfinite(norms).any(axis=0).all():
        raise ValueError("a complete vocabulary row needs at least one finite score")
    greedy = sampling is None or sampling.temperature <= 0
    need_source = False
    k = 1 if greedy else int(sampling.top_k)
    if not need_source:
        for row in range(R):
            v = values[:, row].reshape(-1)
            cutoff = np.partition(v, len(v) - k)[len(v) - k]
            for rank in range(world):
                shard_complete = plan is not None and int(plan.widths[rank]) <= CAND
                edge = values[rank, row].min()
                if not shard_complete and np.isfinite(edge) and edge >= cutoff:
                    need_source = True
                    break
            if need_source:
                break
    if need_source:
        if logits is None:
            raise ValueError("ambiguous gathered cutoff requires the complete original logits")
        return tp_sample_rows(w, logits, positions, sampling, with_prob=with_prob, plan=plan,
                              offset=int(w.meta.get("vocab_offset", 0)))
    chosen = _tp_choose(values, tokens, positions, sampling)
    if not with_prob:
        return chosen
    lse = norms.astype(np.float64)
    top = lse.max(axis=0)
    total = top + np.log(np.exp(lse - top).sum(axis=0))
    probs = []
    for row, token in enumerate(chosen):
        hit = np.nonzero(tokens[:, row].reshape(-1) == token)[0]
        if not len(hit):
            raise ValueError("selected gathered token is absent from its exact score packet")
        probs.append(float(np.exp(float(values[:, row].reshape(-1)[hit[0]]) - total[row])))
    return chosen, probs


def _gathered_fits(sampling: Sampling | None) -> bool:
    """Whether a step's gathered candidates (CAND a rank) cover the sampler's top-k plus its margin."""

    return sampling is None or sampling.temperature <= 0 or (bool(sampling.top_k) and sampling.top_k + MARGIN <= CAND)


PREFILL_ROWS = 2048      # rows of a prompt chunk


def entry_end(prompt: Sequence[int]) -> int:
    """Where a prompt's kept state ends: one token early, since a next turn sent back without its reasoning renders ``<think>`` and two newlines there."""

    return max(1, len(prompt) - 1)


class Engine:
    """Weights, one sequence's state, buffers for decode windows (main model and MTP head) and for prompt chunks."""

    def __init__(self, w: Weights, *, capacity: int = 4096, max_rows: int = 8, prefill_rows: int = PREFILL_ROWS,
                 graphs: bool = False, kv_dtype: str = "bf16", kv_pair=None,
                 kv_key_dtype: str | None = None, kv_value_dtype: str | None = None) -> None:
        self.w = w
        self.capacity = capacity
        self.rows, self.prefill_rows = max_rows, prefill_rows
        self.buf = Buffers(w, max_rows, capacity, moe_prefill=True)       # the experts' arithmetic MultiDecoder's use
        self.mbuf = Buffers(w, max_rows, capacity) if w.mtp is not None else None
        self.pbuf = Buffers(w, prefill_rows, capacity, prefill=True)
        from .hc_plans import enroll

        enroll(w, self.buf)
        enroll(w, self.mbuf, mtp=True)
        self.st = State(w, capacity, max_rows, kv_dtype, kv_pair=kv_pair,
                        kv_key_dtype=kv_key_dtype, kv_value_dtype=kv_value_dtype)
        self.kv_pair = self.st.kv_pair
        self._sample_plan = self._draft_plan = None
        self._draft_map = None
        if w.comm is not None:
            self._sample_plan = SamplingShard(w, int(w.head.n), offset=int(w.meta["vocab_offset"]))
            if self.mbuf is not None:
                self._draft_plan = SamplingShard(w, int(w.draft_head.n if w.draft_head is not None else w.head.n),
                                                offset=int(w.meta["vocab_offset"]), id_map=w.draft_ids)
                self._draft_map = self._draft_plan.token_map
        elif w.draft_ids is not None:
            self._draft_map = TokenMap(w.draft_ids, int(w.draft_ids.numel()))
        self.graphs = None
        if graphs:
            from .graphs import Graphs

            # experts that read their plan on the host (NVFP4) can't be captured: decline graphs before a capture fails
            if any(getattr(layer.moe.experts, "capturable", True) is False for layer in w.layers):
                print("[tensorfold] CUDA graphs off: this MoE reads its plan's item list on the host, which a "
                      "capture rejects; decode runs eagerly (correct, slower)")
            else:
                self.graphs = Graphs(self, max_rows=max_rows)

    @property
    def kv_dtype(self) -> str:
        """Legacy symmetric cache name; mixed callers use kv_pair."""

        if not self.kv_pair.symmetric:
            raise ValueError("mixed KV engine has no single dtype; use kv_pair")
        return self.kv_pair.key_dtype

    def reset(self) -> None:
        self.st.reset(self.w)

    def twin(self) -> "Engine":
        """Share weights and scratch with an independent serial state, without MTP or graphs; requests run sequentially so scratch reuse leaves this engine's state and prefix cache intact."""

        other = object.__new__(Engine)
        other.w, other.capacity, other.rows, other.prefill_rows = self.w, self.capacity, self.rows, self.prefill_rows
        other.buf, other.mbuf, other.pbuf, other.graphs = self.buf, None, self.pbuf, None
        other.kv_pair = self.kv_pair
        other._sample_plan, other._draft_plan, other._draft_map = self._sample_plan, self._draft_plan, self._draft_map
        other.st = State(self.w, self.capacity, self.rows, kv_pair=self.st.kv_pair)
        return other

    def forward(self, tokens: Sequence[int]) -> torch.Tensor:
        """A decode step's forward (a CUDA graph when enabled): logits [R, V]."""

        if self.graphs is not None:
            return self.graphs.forward(tokens)
        return forward(self.w, self.st, self.buf, tokens)

    def sample(self, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None, *,
               draft: bool = False, gathered: bool = True) -> list[int]:
        """Rows of logits at their positions -> tokens (``draft``: the MTP head's; ``gathered``: own candidates)."""

        self.st.kv_check()
        mapped = draft and self.w.draft_ids is not None
        if self.w.comm is not None:
            b = self.mbuf if draft else self.buf
            if gathered and logits.data_ptr() == b.logits.data_ptr() and _gathered_fits(sampling):
                return choose_gathered(self.w, b.cand_all, logits.shape[0], positions, sampling, logits=logits,
                                       plan=self._draft_plan if draft else self._sample_plan)
            return tp_sample_rows(self.w, logits, positions, sampling, offset=self.w.meta["vocab_offset"],
                                  id_map=self.w.draft_ids if mapped else None,
                                  plan=self._draft_plan if draft else self._sample_plan)
        if mapped:
            return sample_rows(logits, positions, sampling, id_map=self._draft_map)
        return sample_rows(logits, positions, sampling)

    def sample_draft(self, logits: torch.Tensor, position: int, sampling: Sampling | None) -> tuple[int, float]:
        """Return the position's keyed draft and its temperature-1 probability; this confidence ends draft chains early without changing output."""

        self.st.kv_check()
        w = self.w
        mapped = w.draft_ids is not None
        if w.comm is not None:
            if logits.data_ptr() == self.mbuf.logits.data_ptr() and _gathered_fits(sampling):
                toks, probs = choose_gathered(w, self.mbuf.cand_all, 1, [position], sampling, with_prob=True,
                                             logits=logits[:1], plan=self._draft_plan)
            else:
                toks, probs = tp_sample_rows(w, logits[:1], [position], sampling, offset=w.meta["vocab_offset"],
                                             id_map=w.draft_ids if mapped else None, with_prob=True,
                                             plan=self._draft_plan)
            return toks[0], probs[0]
        toks, probs = sample_rows(logits[:1], [position], sampling, id_map=self._draft_map if mapped else None,
                                  with_prob=True)
        return toks[0], probs[0]

    def mtp_forward(self, next_tokens: Sequence[int], streams: torch.Tensor) -> torch.Tensor:
        if self.graphs is not None:
            return self.graphs.mtp_forward(next_tokens, streams)
        return mtp_forward(self.w, self.st, self.mbuf, next_tokens, streams)


def absorb(e: Engine, streams: torch.Tensor, next_tokens: Sequence[int]) -> torch.Tensor:
    """The MTP cache takes positions with main-model streams [n, S*D] and next tokens; logits of the last."""

    st = e.st
    if st.mtp_drafted:
        st.set_mtp_len(st.mtp_len - st.mtp_drafted)
        st.mtp_drafted = 0
    logits = e.mtp_forward(next_tokens, streams)
    st.set_mtp_len(st.mtp_len + len(next_tokens))
    return logits


def draft(e: Engine, streams: torch.Tensor, next_tokens: Sequence[int], position: int, count: int,
          sampling: Sampling | None, confidence: float = 0.0) -> list[int]:
    """Absorb kept rows and chain drafts, always retaining the first even below ``confidence``, then stopping before later drafts below it or after a low-confidence first draft."""

    st = e.st
    logits = absorb(e, streams, next_tokens)
    drafts: list[int] = []
    for j in range(count):
        low = False
        if confidence > 0:
            d, p = e.sample_draft(logits, position + j, sampling)
            low = p < confidence
            if low and j > 0:
                break
        else:
            d = e.sample(logits[:1], [position + j], sampling, draft=True)[0]
        drafts.append(d)
        if low:
            break
        if j + 1 < count:
            prev = e.mbuf.streams[len(next_tokens) - 1:len(next_tokens)] if j == 0 else e.mbuf.streams[:1]
            logits = e.mtp_forward([d], prev)
            st.set_mtp_len(st.mtp_len + 1)
            st.mtp_drafted += 1
            next_tokens = [d]
    return drafts


def _absorbs(e: Engine, mtp: bool) -> bool:
    return mtp and e.w.mtp is not None and e.mbuf is not None


@torch.no_grad()
def prefill_begin(e: Engine, prompt: Sequence[int], *, mtp: bool = True, resume: dict | None = None) -> int:
    """Empty the state, or restore a kept prompt end and absorb its tail; returns the first prompt row to commit."""

    if not prompt:
        raise ValueError("prefill requires at least one token")
    if resume is None:
        e.reset()
        return 0
    st = e.st
    st.restore(resume["state"])
    if not 0 < st.pos < len(prompt):
        raise ValueError("a resumed prompt must extend the cached tokens")
    if _absorbs(e, mtp) and resume.get("tail") is not None:
        mtp_forward(e.w, st, e.pbuf, [prompt[st.pos]], resume["tail"], logits=False)
        st.set_mtp_len(st.mtp_len + 1)
    return st.pos


@torch.no_grad()
def prefill_chunk(e: Engine, prompt: Sequence[int], start: int, *, mtp: bool = True,
                  keep_at: int | None = None, end: int | None = None) -> torch.Tensor | None:
    """Commit up to ``e.prefill_rows`` rows from ``start`` (the last chunk returns its logits); a chunk holding ``keep_at`` sets ``e.kept``."""

    w, st, pb = e.w, e.st, e.pbuf
    end = min(start + e.prefill_rows, len(prompt) if end is None else end)
    chunk = list(prompt[start:end])
    R = len(chunk)
    final = end == len(prompt)
    if not final and st.ple_history is not None:
        hist = np.concatenate([np.asarray(st.ple_history, dtype=np.int64),
                               np.asarray(chunk, dtype=np.int64)])[-(w.cfg.ngram_size - 1):]
        read_ahead(w, hist, prompt[end:end + e.prefill_rows])
    point = keep_at - start if keep_at is not None and start < keep_at <= end else 0     # the kept point's row
    cut = Cut(point) if 0 < point < R else None           # inside the chunk, not at its end
    # only the prompt's last row is sampled: the head runs on the final chunk alone
    logits = forward(w, st, pb, chunk, logits=final, cut=cut)
    last = logits.clone() if final else None
    e.last_streams = pb.streams[R - 1:R].clone()
    use_mtp = _absorbs(e, mtp)
    if point:                    # before the MTP head writes the streams: the point's tail, its state inside the chunk
        mtp_len = st.mtp_len + point - 1 if use_mtp else st.mtp_len       # every row but the point's last
        tail = pb.streams[point - 1:point].clone() if use_mtp else None
        snap = cut_snapshot(w, st, pb, cut, mtp_len) if cut is not None else None
    nxt = list(prompt[start + 1:end + 1])
    if use_mtp and nxt:
        mtp_forward(w, st, pb, nxt, pb.streams[:len(nxt)], logits=False)
        st.set_mtp_len(st.mtp_len + len(nxt))
    commit(w, st, pb, R, R)
    if point:                                            # as a fresh prefill of prompt[:keep_at] leaves it
        e.kept = {"state": snap if snap is not None else {**st.snapshot(), "mtp_len": mtp_len}, "tail": tail}
    return last


@torch.no_grad()
def prefill(e: Engine, prompt: Sequence[int], sampling: Sampling | None, *, mtp: bool = True,
            resume: dict | None = None, constraint=None, probabilities=None, keep_at: int | None = None,
            stops: Sequence[int] = (), keep=None, vision=None) -> int:
    """Commit the prompt in chunks and sample the first token (``resume`` equals a fresh run); ``e.kept`` resumes prompt[:keep_at]."""

    if vision is not None and (resume is not None or keep_at is not None or stops or keep is not None):
        raise ValueError("an image prompt prefills from its start and keeps no token-only snapshot")
    start, last = prefill_begin(e, prompt, mtp=mtp, resume=resume), None
    if vision is not None:
        image_rows.attach(e.st, vision, len(prompt))
    if keep_at is not None and not start <= keep_at <= len(prompt):
        raise ValueError(f"keep_at {keep_at} is outside the prefilled range [{start}, {len(prompt)}]")
    saved = e.kept = resume if keep_at == start else None
    stops = sorted({p for p in stops if start < p < len(prompt)})
    while start < len(prompt):
        end = min(start + e.prefill_rows, next((p for p in stops if p > start), len(prompt)))
        if keep_at is not None and start < keep_at < end and end in stops:
            end = keep_at
        point = keep_at if keep_at is not None and start < keep_at <= end else end if end in stops else None
        last = prefill_chunk(e, prompt, start, mtp=mtp, keep_at=point, end=end)
        if point is not None:
            if point == keep_at:
                saved = e.kept
            if keep is not None:
                keep(point, e.kept["state"], e.kept["tail"])
        start = end
    if keep_at is not None:
        e.kept = saved
    image_rows.finish(e.st)
    if constraint is not None:                           # a reply's grammar: this rank's vocabulary columns
        last = constraint.mask(last, None, e.w.meta.get("vocab_offset", 0))
    first = e.sample(last, [len(prompt)], sampling)[0]
    if probabilities is not None:
        capture(last, [first], [len(prompt)], probabilities)
    if constraint is not None:
        constraint.advance([first])
    e.first = first
    return first


WARM_TAIL = 18      # a partial chunk after a full one: neither its rows nor the MTP head's 17 divide by 16


@torch.no_grad()
def warm(e: Engine) -> None:
    """Prefill a synthetic prompt (a full chunk, then a partial one cut at the kept point a row before its end) and empty the state, so no request compiles or loads a prompt kernel."""

    prompt = [0] * min(e.prefill_rows + WARM_TAIL + 1, e.capacity)
    prefill(e, prompt, None, keep_at=entry_end(prompt))
    e.kept = None
    e.reset()


@dataclass
class DecodeResult:
    tokens: list[int]
    seconds: float
    rounds: int
    drafted: int = 0
    accepted: int = 0
    keeps: list[int] = field(default_factory=list)      # tokens each round kept
    committed: list[int] = field(default_factory=list)  # the tokens now in the caches (all but the pending one)
    widths: list[int] = field(default_factory=list)     # rows each round verified

    @property
    def tokens_per_second(self) -> float:
        return (len(self.tokens) - 1) / self.seconds if self.seconds else 0.0


@torch.no_grad()
def serial_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *, stop_eos: bool = False,
                  on_tokens=None, constraint=None, probabilities=None) -> DecodeResult:
    """One token a step through the same kernels and sampler; ``pending`` is the first sampled token. ``on_tokens(new)`` hears each step's token; it returns True to stop early."""

    w, st, b = e.w, e.st, e.buf
    out = [pending]
    torch.cuda.synchronize()
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        logits = e.forward([out[-1]])
        if constraint is not None:
            constraint.mask(logits[:1], None, w.meta.get("vocab_offset", 0))
        tok = e.sample(logits[:1], [st.pos + 1], sampling, gathered=constraint is None)[0]
        if probabilities is not None:
            capture(logits[:1], [tok], [st.pos + 1], probabilities)
        commit(w, st, b, 1, 1)
        out.append(tok)
        if constraint is not None:
            constraint.advance([tok])
        if on_tokens is not None and on_tokens([tok]):
            break
    torch.cuda.synchronize()
    return DecodeResult(out, time.perf_counter() - start, len(out) - 1, committed=out[:-1], widths=[1] * (len(out) - 1))


@torch.no_grad()
def mtp_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *, depth: int = DEPTH,
               confidence: float = CONFIDENCE, stop_eos: bool = False, on_tokens=None, constraint=None,
               probabilities=None) -> DecodeResult:
    """Verify pending and drafted tokens from the prefill state, commit rows before the first mismatched draft, and call ``on_tokens(new)`` with kept tokens after pending, stopping on True."""

    w, st, b = e.w, e.st, e.buf
    out = [pending]
    rounds = drafted = accepted = 0
    keeps: list[int] = []
    widths: list[int] = []
    pos0 = st.pos
    unabsorbed = None                                  # the last round's kept rows, not yet in the MTP cache
    torch.cuda.synchronize()
    start = time.perf_counter()
    drafts = draft(e, e.last_streams, [pending], st.pos + 1, min(depth, count - len(out)), sampling, confidence)
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        tokens = [out[-1]] + drafts
        window = constraint.window(tokens, list(range(-1, len(tokens) - 1))) if constraint is not None else None
        if window is not None:                           # the drafts no accepted path can hold are cut first
            tokens, drafts = window.tokens, window.tokens[1:]
        R = len(tokens)
        logits = e.forward(tokens)
        if window is not None:
            constraint.mask(logits[:R], window, w.meta.get("vocab_offset", 0))
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling, gathered=window is None)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        if probabilities is not None:
            n = min(keep, count - len(out))
            capture(logits[:n], sampled[:n], list(range(st.pos + 1, st.pos + 1 + n)), probabilities)
        commit(w, st, b, R, keep)
        unabsorbed = (keep, sampled[:keep])
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        keeps.append(keep)
        widths.append(R)
        new = sampled[:keep][:max(0, count - len(out))]
        if constraint is not None:
            constraint.advance(sampled[:keep])
        out.extend(sampled[:keep])
        if on_tokens is not None and new and on_tokens(new):
            break
        if len(out) >= count or (stop_eos and out[-1] in w.cfg.eos):
            break
        n = min(depth, count - len(out))
        drafts = []
        if n > 0:
            drafts = draft(e, b.streams[:keep], sampled[:keep], st.pos + 1, n, sampling, confidence)
            unabsorbed = None
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    if unabsorbed is not None:              # the MTP cache takes the last kept rows: it then covers the sequence
        absorb(e, b.streams[:unabsorbed[0]], unabsorbed[1])
    committed = out[:st.pos - pos0]
    return DecodeResult(out[:count], seconds, rounds, drafted, accepted, keeps, committed, widths)

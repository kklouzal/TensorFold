"""Keyed sampler for CUDA graphs: a token depends only on (seed, position, logits), so serial and drafted rows agree."""

from __future__ import annotations

import math
import operator

import torch
import triton
import triton.language as tl
from triton.language.core import TRITON_MAX_TENSOR_NUMEL

from tensorfold.engine.exact_sampling import MARGIN, Sampling

C1 = 0x9E3779B97F4A7C15
C2 = 0xD1B54A32D192ED03
M1 = 0xBF58476D1CE4E5B9
M2 = 0x94D049BB133111EB


def token_list(values, vocab: int, multiple: int) -> list[int]:
    """Validate a model-owned map before device allocation or head indexing."""

    if type(vocab) is not int or vocab <= 0 or type(multiple) is not int or multiple <= 0:
        raise ValueError("positive vocabulary and token-map tile multiple required")
    result = []
    seen = set()
    for value in values:
        if len(result) >= vocab:
            raise ValueError("draft token-map cardinality exceeds the real vocabulary")
        if isinstance(value, bool):
            raise ValueError("draft token IDs must be integers")
        try:
            token = operator.index(value)
        except TypeError as error:
            raise ValueError("draft token IDs must be integers") from error
        if not 0 <= token < vocab or token > (1 << 31) - 1:
            raise ValueError("draft token IDs must fit the model vocabulary and signed32 output")
        if token in seen:
            raise ValueError("draft token IDs must be distinct")
        seen.add(token)
        result.append(token)
    if not result or len(result) % multiple:
        raise ValueError(f"draft token IDs must be distinct and hold a positive multiple of {multiple}")
    return result


def _outputs(logits, out):
    floats = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or not 0 < logits.shape[1] <= 1 << 31
            or logits.dtype not in floats):
        raise ValueError("floating logits [rows, nonempty signed32 vocabulary] required")
    rows, width = logits.shape
    if (not isinstance(out, torch.Tensor) or out.shape != (rows,) or out.dtype not in (torch.int32, torch.int64)
            or out.device != logits.device or (rows > 1 and out.stride(0) <= 0)):
        raise ValueError("one distinct integer output per row on the score device required")
    return rows, width


def _buffers(logits, meta, params, out, offset, prob, id_map=None, *, row_ids=False, cuda=False):
    """Check metadata before indexing/launch; device data stays on its producer stream.

    Scores must be finite or -inf masks, with a finite score in each nonempty
    row. Maps contain distinct nonnegative signed32 token IDs. Model map
    owners prove these value invariants at startup with ``token_list``; raw
    device maps retain that proof and remain immutable through the operation.
    """

    floats = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    integers = (torch.int32, torch.int64)
    rows, width = _outputs(logits, out)
    if rows == 0:
        return rows, width
    if cuda and not logits.is_cuda:
        raise ValueError("native keyed sampling requires CUDA scores")
    if (type(offset) is not int or not -(1 << 63) <= offset < (1 << 63)
            or not isinstance(meta, torch.Tensor) or meta.ndim != 1 or meta.numel() < 1
            or meta.dtype not in integers or meta.device != logits.device or not meta.is_contiguous()
            or params.seed.shape != (1,) or params.seed.dtype != torch.int64
            or params.seed.device != logits.device or not params.seed.is_contiguous()
            or params.fp.shape != (3,) or params.fp.dtype != torch.float64
            or params.fp.device != logits.device or not params.fp.is_contiguous()):
        raise ValueError("integer position/offset and matching device sampling buffers required")
    if (prob is not None and (not isinstance(prob, torch.Tensor) or prob.shape != (rows,)
                             or prob.dtype not in floats or prob.device != logits.device
                             or (rows > 1 and prob.stride(0) <= 0))):
        raise ValueError("one distinct floating confidence per row on the score device required")
    if id_map is not None:
        shapes = ((width,), (rows, width)) if row_ids else ((width,),)
        if (not isinstance(id_map, torch.Tensor) or id_map.dtype not in integers
                or id_map.device != logits.device or id_map.shape not in shapes):
            raise ValueError("matching integer token map geometry/device required")
    return rows, width


def _order(values, ids=None):
    """Complete lexicographic (value descending, global ID ascending) order."""

    if ids is None:
        return torch.argsort(values, dim=-1, descending=True, stable=True)
    first = torch.argsort(ids, dim=-1, stable=True)
    by_id = values[:, first] if ids.ndim == 1 else values.gather(1, first)
    ranked = torch.argsort(by_id, dim=-1, descending=True, stable=True)
    return first[ranked] if ids.ndim == 1 else first.gather(1, ranked)


def candidates(logits, count: int, id_map=None, *, sorted: bool = True):
    """Exact complete-row candidates, retaining the original topk value slots.

    Only IDs at the retained boundary need replacement. Keeping every FP32
    value in its original slot preserves the greedy confidence reduction tree.
    The full order also covers ties wider than the readback margin. No host
    readback or data-dependent graph dispatch is used.
    """

    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] <= 0
            or logits.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)):
        raise ValueError("floating candidate source rows required")
    if (id_map is not None and (not isinstance(id_map, torch.Tensor) or id_map.shape != (logits.shape[1],)
                               or id_map.dtype not in (torch.int32, torch.int64) or id_map.device != logits.device)):
        raise ValueError("one matching integer token ID per source column required")
    if type(count) is not int or not 1 <= count <= logits.shape[1]:
        raise ValueError("positive candidate count within the vocabulary required")
    scores = logits.float()
    vals, cols = torch.topk(scores, count, dim=-1, sorted=sorted)
    if count == logits.shape[1]:
        return vals, cols if id_map is None else id_map[cols]
    edge = vals.min(dim=-1, keepdim=True).values
    exact = _order(scores, id_map)[:, :count]
    above = (vals > edge).sum(dim=-1, keepdim=True)
    slot = above + (vals == edge).to(torch.int64).cumsum(dim=-1) - 1
    repaired = exact.gather(1, slot.clamp(0, count - 1))
    cols = torch.where(vals == edge, repaired, cols)
    return vals, cols if id_map is None else id_map[cols]


def candidate_count(width: int, sampling, *, minimum: int = 0) -> int:
    if type(width) is not int or width <= 0 or type(minimum) is not int or minimum < 0:
        raise ValueError("positive shard width and nonnegative retained minimum required")
    top_k = 20 if sampling is None or sampling.temperature <= 0 else int(sampling.top_k)
    count = width if not top_k else min(width, max(minimum, top_k + MARGIN))
    if top_k and count > 256:
        raise ValueError("the GPU sampler takes at most 256 retained candidates per shard")
    return count


def gather_candidates(gather, vals, ids, *, world: int):
    """Exchange FP32 values and signed64 IDs as exact signed32 words, rank first."""

    if (type(world) is not int or world <= 0 or not isinstance(vals, torch.Tensor)
            or vals.ndim != 2 or vals.shape[1] <= 0 or vals.dtype != torch.float32):
        raise ValueError("positive world and FP32 candidate rows required")
    rows, count = vals.shape
    if (not isinstance(ids, torch.Tensor) or ids.shape != vals.shape
            or ids.dtype not in (torch.int32, torch.int64) or ids.device != vals.device):
        raise ValueError("candidate IDs must match their value table")
    payload = torch.cat([vals.contiguous().view(torch.int32), ids.to(torch.int64).contiguous().view(torch.int32)], dim=1)
    got = gather(payload)
    if (not isinstance(got, torch.Tensor) or got.dtype != torch.int32 or got.device != vals.device
            or got.shape != (world * rows, 3 * count)):
        raise ValueError("gather must preserve exact candidate words, device and rank-row geometry")
    both = got.contiguous().view(world, rows, 3 * count)
    v = both[:, :, :count].contiguous().view(torch.float32)
    i = both[:, :, count:].contiguous().view(torch.int64)
    return torch.cat(list(v.unbind(0)), dim=1), torch.cat(list(i.unbind(0)), dim=1)


def _launch(vals, ids, meta, params, out, offset, prob, *, k, cut, greedy_mode, minp):
    """The native candidate ABI owns contiguous inputs and output staging."""

    rows, count = vals.shape
    native_out = out if out.is_contiguous() else torch.empty_like(out, memory_format=torch.contiguous_format)
    native_prob = prob
    if prob is not None and not prob.is_contiguous():
        native_prob = torch.empty_like(prob, memory_format=torch.contiguous_format)
    _keyed[(rows,)](vals.contiguous(), ids.to(torch.int64).contiguous(), meta, native_out, params.seed, params.fp,
                    native_prob if native_prob is not None else native_out, C1, C2, M1, M2,
                    offset, C=count, CP=triton.next_power_of_2(count), K=k, CUT=cut, GREEDY=greedy_mode,
                    WRITE_PROB=prob is not None, MINP=minp, num_warps=1)
    if native_out is not out:
        out.copy_(native_out)
    if native_prob is not prob:
        prob.copy_(native_prob)


@triton.jit
def _mix(x, m1, m2):
    x = x ^ (x >> 30)
    x = x * m1
    x = x ^ (x >> 27)
    x = x * m2
    return x ^ (x >> 31)


@triton.jit
def _keyed(VALS, IDS, META, OUT, SEED, FP, PROB, c1, c2, m1, m2, offset,
           C: tl.constexpr, CP: tl.constexpr, K: tl.constexpr, CUT: tl.constexpr, GREEDY: tl.constexpr = False,
           WRITE_PROB: tl.constexpr = False, MINP: tl.constexpr = False):
    r = tl.program_id(0)
    seed = tl.load(SEED)
    temp = tl.load(FP)
    top_p = tl.load(FP + 1)
    i = tl.arange(0, CP)
    ok = i < C
    address = r.to(tl.int64) * C + i.to(tl.int64)
    v = tl.load(VALS + address, mask=ok, other=float("-inf")).to(tl.float64)
    ids = tl.load(IDS + address, mask=ok, other=2**40)
    # rank by (value desc, id asc), as the host's lexsort
    better = (v[None, :] > v[:, None]) | ((v[None, :] == v[:, None]) & (ids[None, :] < ids[:, None]))
    rank = tl.sum(tl.where(better & ok[None, :], 1, 0), axis=1)
    kept = (rank < K) & ok
    if GREEDY:
        tl.store(OUT + r, tl.sum(tl.where(rank == 0, ids, 0), axis=0).to(tl.int32))
        if WRITE_PROB:                     # the argmax's share of the top-K mass (temperature 1)
            top0 = tl.max(tl.where(kept, v, float("-inf")), axis=0)
            e0 = tl.where(kept, tl.exp(v - top0), 0.0)
            tl.store(PROB + r, (1.0 / tl.sum(e0, axis=0)).to(tl.float32))
        return
    scaled = v / temp
    top = tl.max(tl.where(kept, scaled, float("-inf")), axis=0)
    p = tl.where(kept, tl.exp(scaled - top), 0.0)
    total = tl.sum(tl.where(rank == 0, p, 0.0), axis=0)
    for k in tl.static_range(1, K):
        total += tl.sum(tl.where(rank == k, p, 0.0), axis=0)
    limit = K
    if CUT:
        run = tl.sum(tl.where(rank == 0, p, 0.0), axis=0) / total
        below = tl.where(run < top_p, 1, 0)
        for k in tl.static_range(1, K):
            run += tl.sum(tl.where(rank == k, p, 0.0), axis=0) / total
            below += tl.where(run < top_p, 1, 0)
        limit = below + 1
    if MINP:                               # the tokens within ln(min_p) of the top: a prefix of the rank order
        floor = top + tl.load(FP + 2)
        limit = tl.minimum(limit, tl.sum(tl.where(kept & (scaled >= floor), 1, 0), axis=0))
    pos = (tl.load(META) + r + 1 + offset).to(tl.uint64)
    x = _mix(seed.to(tl.uint64) + c1, m1, m2)
    x = _mix(x ^ (pos * c2), m1, m2)
    x = _mix(x ^ ids.to(tl.uint64), m1, m2)
    u = (x >> 11).to(tl.float64) * (2.0 ** -53) + 2.0 ** -54
    score = scaled - tl.log(-tl.log(u))
    score = tl.where(kept & (rank < limit), score, float("-inf"))
    best = tl.max(score, axis=0)
    first = tl.min(tl.where(score == best, rank, CP), axis=0)
    tok = tl.sum(tl.where(rank == first, ids, 0), axis=0)
    tl.store(OUT + r, tok.to(tl.int32))
    if WRITE_PROB:                         # the drawn token's share of the top-k mass
        tl.store(PROB + r, (tl.sum(tl.where(rank == first, p, 0.0), axis=0) / total).to(tl.float32))


class Params:
    """Sampling parameters live on the device so captured graphs serve any request; top_k is compiled in."""

    def __init__(self, device):
        self.seed = torch.zeros(1, dtype=torch.int64, device=device)
        self.fp = torch.zeros(3, dtype=torch.float64, device=device)
        self.sampling: Sampling | None = None

    def set(self, sampling: Sampling | None) -> None:
        if sampling is not None and (not isinstance(sampling, Sampling) or type(sampling.seed) is not int
                                     or type(sampling.top_k) is not int or sampling.top_k < 0
                                     or not math.isfinite(float(sampling.temperature))
                                     or not math.isfinite(float(sampling.top_p)) or not 0 <= sampling.top_p <= 1
                                     or not math.isfinite(float(sampling.min_p)) or not 0 <= sampling.min_p <= 1):
            raise ValueError("finite sampling policy and integer keyed seed required")
        self.sampling = sampling
        if sampling is not None:
            self.seed.fill_(int(sampling.seed) & ((1 << 63) - 1))
            self.fp.copy_(torch.tensor([max(float(sampling.temperature), 1e-6), float(sampling.top_p),
                                        sampling.min_log], dtype=torch.float64))


def keyed(logits: torch.Tensor, meta: torch.Tensor, params: Params, out: torch.Tensor, *, offset: int = 0,
          prob: torch.Tensor | None = None, id_map: torch.Tensor | None = None):
    """Row r samples position meta[0] + r + 1 + offset; ``prob`` is the existing top-k confidence.

    Mapped greedy ties select the lowest global token ID over the complete row.
    Greedy confidence retains its original approximate top-20 FP32-candidate mass.
    """

    s = params.sampling
    greedy_mode = s is None or s.temperature <= 0
    top_k = 20 if greedy_mode else int(s.top_k)
    rows, vocab = _buffers(logits, meta, params, out, offset, prob, id_map, cuda=bool(top_k))
    if rows == 0:
        return out
    if greedy_mode and prob is None:
        out.copy_(_greedy_ids(logits, id_map).to(out.dtype))
        return out
    if not top_k:                                  # top_k off: the nucleus over the whole vocabulary
        return nucleus(logits, meta, params, out, offset=offset, prob=prob, id_map=id_map)
    count = min(vocab, top_k + MARGIN) if top_k else vocab
    if count > 256:
        raise ValueError("the GPU sampler takes top_k + margin <= 256 candidates")
    vals, ids = candidates(logits, count, id_map, sorted=False)
    # Resolve original-dtype greedy identity before any aliased output writes.
    chosen = _greedy_ids(logits, id_map) if greedy_mode else None
    k = max(1, min(top_k if top_k else count, count))
    cut = (not greedy_mode) and 0.0 < float(s.top_p) < 1.0
    _launch(vals, ids, meta, params, out, offset, prob, k=k, cut=cut, greedy_mode=greedy_mode,
            minp=(not greedy_mode) and float(s.min_p) > 0.0)
    if chosen is not None:
        out.copy_(chosen.to(out.dtype))
    return out


def _signed(c: int) -> int:
    return c - (1 << 64) if c >= 1 << 63 else c


def _shr(x: torch.Tensor, k: int) -> torch.Tensor:
    return (x >> k) & ((1 << (64 - k)) - 1)       # a logical shift on int64 (uint64 bits)


def _mix_t(x: torch.Tensor) -> torch.Tensor:
    x = x ^ _shr(x, 30)
    x = x * _signed(M1)
    x = x ^ _shr(x, 27)
    x = x * _signed(M2)
    return x ^ _shr(x, 31)


def nucleus(logits: torch.Tensor, meta: torch.Tensor, params: Params, out: torch.Tensor, *, offset: int = 0,
            prob: torch.Tensor | None = None, id_map: torch.Tensor | None = None):
    """``_keyed``'s rule with top_k off: rank the whole vocabulary by (value desc, id asc), cut at top_p, draw."""

    rows, vocab = _buffers(logits, meta, params, out, offset, prob, id_map, row_ids=True)
    if rows == 0:
        return out
    if params.sampling is None or params.sampling.temperature <= 0:
        raise ValueError("nucleus requires a positive-temperature sampling policy")
    v = logits.double()
    ids = torch.arange(vocab, dtype=torch.int64, device=logits.device)
    ids = id_map.to(torch.int64) if id_map is not None else ids
    temp, top_p = params.fp[0], params.fp[1]
    scaled = v / temp
    order = _order(v, ids if id_map is not None else None)
    ranked_ids = ids[order] if ids.ndim == 1 else ids.gather(1, order)
    ranked = torch.gather(scaled, 1, order)
    p = torch.exp(ranked - ranked[:, :1])
    run = torch.cumsum(p, dim=-1) / p.sum(dim=-1, keepdim=True)
    cut = params.sampling is not None and 0.0 < float(params.sampling.top_p) < 1.0
    limit = (run < top_p).sum(dim=-1, keepdim=True) + 1 if cut else torch.full_like(run[:, :1], vocab, dtype=torch.int64)
    if params.sampling is not None and params.sampling.min_p > 0.0:        # the kernel's min-p prefix
        limit = torch.minimum(limit, (ranked >= ranked[:, :1] + params.fp[2]).sum(dim=-1, keepdim=True))
    pos = (meta[0].to(torch.int64) + torch.arange(rows, device=logits.device) + 1 + offset)[:, None]
    x = _mix_t(params.seed.to(torch.int64) + _signed(C1))
    x = _mix_t(x ^ (pos * _signed(C2)))
    x = _mix_t(x ^ ranked_ids)
    u = _shr(x, 11).double() * 2.0 ** -53 + 2.0 ** -54
    score = ranked - torch.log(-torch.log(u))
    rank = torch.arange(vocab, device=logits.device)[None, :]
    score = torch.where(rank < limit, score, torch.full_like(score, float("-inf")))
    first = torch.argmax(score, dim=-1)                                        # the lowest rank among equal scores
    out.copy_(ranked_ids.gather(1, first[:, None])[:, 0].to(out.dtype))
    if prob is not None:
        prob.copy_((p.gather(1, first[:, None])[:, 0] / p.sum(dim=-1)).to(prob.dtype))
    return out


def sample_candidates(vals: torch.Tensor, ids: torch.Tensor, meta: torch.Tensor, params: Params, out: torch.Tensor,
                      *, offset: int = 0, prob: torch.Tensor | None = None):
    """As ``sample`` over candidate lists (R, C) holding each row's top top_k + MARGIN, such as the ranks' union."""

    s = params.sampling
    greedy_mode = s is None or s.temperature <= 0
    if not isinstance(ids, torch.Tensor) or ids.shape != vals.shape:
        raise ValueError("one candidate ID per candidate value required")
    rows, count = _buffers(vals, meta, params, out, offset, prob, ids, row_ids=True,
                          cuda=greedy_mode or bool(s.top_k))
    if rows == 0:
        return out
    if not greedy_mode and not s.top_k:
        return nucleus(vals, meta, params, out, offset=offset, prob=prob, id_map=ids)
    k = (20 if prob is not None else 1) if greedy_mode else max(1, min(int(s.top_k) if s.top_k else count, count))
    k = min(k, count)
    cut = (not greedy_mode) and 0.0 < float(s.top_p) < 1.0
    if triton.next_power_of_2(count) ** 2 > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError("candidate rank matrix exceeds the declared Triton tensor capacity")
    _launch(vals, ids, meta, params, out, offset, prob, k=k, cut=cut, greedy_mode=greedy_mode,
            minp=(not greedy_mode) and float(s.min_p) > 0.0)
    return out


def _greedy_ids(logits, id_map=None):
    if id_map is None:
        return torch.argmax(logits, dim=-1)
    maximum = logits.max(dim=-1, keepdim=True).values
    return torch.where(logits == maximum, id_map.to(torch.int64), (1 << 63) - 1).min(dim=-1).values


def greedy(logits: torch.Tensor, out: torch.Tensor):
    _outputs(logits, out)
    out.copy_(torch.argmax(logits, dim=-1).to(torch.int32))
    return out


def sample(logits: torch.Tensor, meta: torch.Tensor, params: Params, out: torch.Tensor, *, offset: int = 0,
           prob: torch.Tensor | None = None, id_map: torch.Tensor | None = None):
    """Greedy uses the complete original row; mapped ties choose its lowest global ID.

    Confidence is the original approximate top-20 share, independent of the
    full-row greedy identity. Positive-temperature draws use exact ID ties.
    """

    s = params.sampling
    if (s is None or s.temperature <= 0) and prob is None and id_map is None:
        return greedy(logits, out)
    return keyed(logits, meta, params, out, offset=offset, prob=prob, id_map=id_map)

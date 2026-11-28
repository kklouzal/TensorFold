"""Keyed top-k-off nucleus/min-p draws of decode rows on the device, under the host sampler's rule.

``round_sampling._Nucleus`` + ``tensorfold.cuda.sampling._support``/``_draw_support`` choose a positive-temperature
token from the full gathered row: FP64 scores centered on the row maximum and multiplied by the row's FP64 inverse
temperature, fixed-point masses ``floor(exp(scaled) * 2**40)``, the nucleus as the shortest (score desc, id asc)
prefix whose exact integer mass reaches ``ceil(top_p * total)``, the min-p prefix ``scaled >= ln(min_p)``, and the
keyed Gumbel argmax ``scaled - log(-log(uniform(seed, position, id)))`` over that support.

Here every rank keeps its contiguous vocabulary shard. The boundary of the nucleus is found by an exact radix
select over the monotone FP32 score key (four 8-bit passes, each an all-gathered integer mass histogram summed in
rank order, so every rank takes identical decisions), then the boundary's equal-score ties are taken in global ID
order. Triton kernels recompute each element's scaled score, mass and key in registers (no row-sized storage);
integer atomics keep every histogram deterministic. Integer masses and the exact limb product for ``top_p * total`` keep the support identical to the host's.
The draw is the same keyed Gumbel per (seed, position, global ID); each rank sends its best (score, key, id) and the
first lexicographic maximum wins: the host's first maximum in support order. Scores, masses and support are
bit-identical to the host's; the Gumbel log runs on the device's FP64 log instead of the host libm, a last-ulp
difference that can only reorder two noisy scores equal to within one ulp.

No host synchronization: the result stays on the device (a run-ahead decode round reads it). Rows whose inverse
temperature word is zero are not sampled (their token comes from the greedy packet).
"""

from __future__ import annotations

import math
import struct

import torch

from tensorfold.cuda.sampling import MASS

from .kernels.triton_utils import jit, tl, tldevice, triton

if MASS != 1099511627776.0:          # _element's fixed-point scale is the host sampler's
    raise ImportError("device sampling assumes the host sampler's 2**40 fixed-point mass")

WORDS = 9                   # int32 words per row: seed (2), inverse temperature bits (2), top_p numerator (2),
                            # top_p denominator log2 (1), min_log bits (2)
MAX_SHIFT = 104             # top_p denominators up to 2**104 (every top_p >= 2**-51 that is a binary64 value)


def _signed(c: int) -> int:
    c &= (1 << 64) - 1
    return c - (1 << 64) if c >= 1 << 63 else c


def _split(value: int) -> tuple[int, int]:
    """A signed64 as (low, high) signed32 words."""
    value = _signed(value)
    low = value & 0xFFFFFFFF
    return (low - (1 << 32) if low >= 1 << 31 else low), value >> 32


def _bits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0]


def eligible(sampling) -> bool:
    """A policy this sampler draws: positive finite temperature, top_k off, a top_p it can represent exactly."""
    if (sampling is None or not math.isfinite(sampling.temperature) or sampling.temperature <= 0
            or sampling.top_k or not math.isfinite(sampling.top_p) or not math.isfinite(sampling.min_p)):
        return False
    if 0.0 < sampling.top_p < 1.0:
        _, den = float(sampling.top_p).as_integer_ratio()
        return den.bit_length() - 1 <= MAX_SHIFT
    return True


def row_words(sampling) -> list[int]:
    """The row's WORDS int32 control words; zeros (greedy) for None."""
    if sampling is None:
        return [0] * WORDS
    if not eligible(sampling):
        raise ValueError("device sampling needs positive temperature, top_k off and an exact top_p")
    inverse = 1.0 / max(float(sampling.temperature), 1e-6)      # round_sampling._Nucleus's reciprocal
    if 0.0 < sampling.top_p < 1.0:
        num, den = float(sampling.top_p).as_integer_ratio()
        shift = den.bit_length() - 1
    else:
        num, shift = 0, 0                                       # no nucleus cut
    return [*_split(int(sampling.seed)), *_split(_bits(inverse)), *_split(num), shift,
            *_split(_bits(sampling.min_log))]


def lane_noise(words) -> tuple[int, int]:
    """A row's (keyed seed, inverse temperature FP64 bits) as signed64, from its words (draft proposals)."""
    pair = lambda i: _signed((words[i] & 0xFFFFFFFF) | (words[i + 1] << 32))
    return pair(0), pair(2)


BLOCK = 1024


@jit
def _mix64(x):
    x = x ^ (x >> 30)
    x = x * 0xBF58476D1CE4E5B9
    x = x ^ (x >> 27)
    x = x * 0x94D049BB133111EB
    return x ^ (x >> 31)


@jit
def _element(local, top, inverse, b, v, valid, V: tl.constexpr):
    """The row's FP32 score -> (FP64 scaled score, fixed-point mass, monotone uint32 key as int64)."""
    raw = tl.load(local + b * V + v, valid, float("-inf"))
    inv = tl.load(inverse + b)
    inv = tl.where(inv > 0, inv, 1.0)          # greedy rows: temperature one (finite; their result is unused)
    scaled = (raw.to(tl.float64) - tl.load(top + b).to(tl.float64)) * inv
    mass = tl.where(valid, tldevice.floor(tldevice.exp(scaled) * 1099511627776.0).to(tl.int64), 0)
    bits = (raw + 0.0).to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF   # +0.0: -0.0 and 0.0 are one key
    key = tl.where(bits >= 2147483648, (~bits) & 0xFFFFFFFF, bits | 2147483648)
    return scaled, mass, key


@jit
def _histogram(local, top, inverse, state, hist, SHIFT: tl.constexpr, V: tl.constexpr, BLOCK: tl.constexpr):
    pid, b = tl.program_id(0), tl.program_id(1)
    v = pid * BLOCK + tl.arange(0, BLOCK)
    valid = v < V
    _, mass, key = _element(local, top, inverse, b, v, valid, V)
    if SHIFT < 24:
        valid = valid & ((key >> (SHIFT + 8)) == tl.load(state + b * 4 + 2))
    tl.atomic_add(hist + b * 256 + ((key >> SHIFT) & 255), mass, mask=valid & (mass > 0), sem="relaxed")


@jit
def _need(total, num, shift):
    """ceil(num * total / 2**shift) exactly in int64 limbs (total < 2**58, num < 2**53, 1 <= shift <= 104)."""
    t1, t0 = total >> 26, total & 67108863
    n1, n0 = num >> 26, num & 67108863
    low = t0 * n0
    mid = t1 * n0 + t0 * n1 + (low >> 26)
    high = t1 * n1 + (mid >> 26)
    rest = ((mid & 67108863) << 26) | (low & 67108863)          # product = high * 2**52 + rest, rest < 2**52
    small = shift <= 52
    s1 = tl.where(small, shift, 0)
    s2 = tl.where(small, 0, shift - 52)
    q = tl.where(small, (high << (52 - s1)) + (rest >> s1), high >> s2)
    one = tl.full(shift.shape, 1, tl.int64)
    r = tl.where(small, rest & ((one << s1) - 1), (high & ((one << s2) - 1)) | rest)
    return q + (r != 0).to(tl.int64)


@jit
def _select(every, policy, state, WORLD: tl.constexpr, FIRST: tl.constexpr, LAST: tl.constexpr):
    """One radix pass's decision for row b from the ranks' histograms [WORLD, rows, 256] (summed in rank order).
    state int64 [rows, 4]: need, mass above the prefix, prefix, the boundary key's mass (last pass)."""
    b = tl.program_id(0)
    rows = tl.num_programs(0)
    bins = tl.arange(0, 256)
    hist = tl.zeros((256,), tl.int64)
    for r in tl.static_range(WORLD):
        hist += tl.load(every + (r * rows + b) * 256 + bins)
    if FIRST:
        total = tl.sum(hist, 0)
        num = tl.load(policy + b * 5 + 2)
        shift = tl.load(policy + b * 5 + 3)
        cut = (num > 0) & (tl.load(policy + b * 5 + 1).to(tl.float64, bitcast=True) > 0)
        need = tl.where(cut, _need(total, num, tl.where(cut, shift, 1)), total)
        above = total * 0
        prefix = total * 0
    else:
        need = tl.load(state + b * 4)
        above = tl.load(state + b * 4 + 1)
        prefix = tl.load(state + b * 4 + 2)
    reach = above + (tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist)    # inclusive mass from the top bin down
    chosen = tl.maximum(tl.max(tl.where(reach >= need, bins, -1), 0), 0)
    above += tl.sum(tl.where(bins > chosen, hist, 0), 0)
    tl.store(state + b * 4, need)
    tl.store(state + b * 4 + 1, above)
    tl.store(state + b * 4 + 2, (prefix << 8) | chosen)
    if LAST:
        tl.store(state + b * 4 + 3, tl.sum(tl.where(bins == chosen, hist, 0), 0))


@jit
def _ties(local, state, counts, V: tl.constexpr, BLOCK: tl.constexpr, BLOCKS: tl.constexpr):
    pid, b = tl.program_id(0), tl.program_id(1)
    v = pid * BLOCK + tl.arange(0, BLOCK)
    valid = v < V
    raw = tl.load(local + b * V + v, valid, float("-inf"))
    bits = (raw + 0.0).to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    key = tl.where(bits >= 2147483648, (~bits) & 0xFFFFFFFF, bits | 2147483648)
    tie = valid & (key == tl.load(state + b * 4 + 2))
    tl.store(counts + b * BLOCKS + pid, tl.sum(tie.to(tl.int64), 0))


@jit
def _draw(local, top, inverse, policy, positions, state, every_counts, best, RANK: tl.constexpr, WORLD: tl.constexpr,
          FIRST_ID: tl.constexpr, V: tl.constexpr, BLOCK: tl.constexpr, BLOCKS: tl.constexpr,
          PB: tl.constexpr):
    """Each block's lexicographic best (keyed score, key, lowest id) over the row's support."""
    pid, b = tl.program_id(0), tl.program_id(1)
    rows = tl.num_programs(1)
    v = pid * BLOCK + tl.arange(0, BLOCK)
    valid = v < V
    scaled, _, key = _element(local, top, inverse, b, v, valid, V)
    boundary = tl.load(state + b * 4 + 2)
    need = tl.load(state + b * 4)
    above = tl.load(state + b * 4 + 1)
    tie_mass = tl.load(state + b * 4 + 3)
    i = tl.arange(0, PB)
    tie_count = 0
    before = 0
    for r in tl.static_range(WORLD):
        c = tl.sum(tl.load(every_counts + (r * rows + b) * BLOCKS + i, i < BLOCKS, 0), 0)
        tie_count += c
        if r < RANK:
            before += c
    before += tl.sum(tl.load(every_counts + (RANK * rows + b) * BLOCKS + i, i < pid, 0), 0)
    tie_count = tl.maximum(tie_count, 1)
    each = tie_mass // tie_count                                     # equal scores: equal masses
    quota = tl.where(each > 0, (need - above + tl.maximum(each, 1) - 1) // tl.maximum(each, 1), tie_count)
    tie = valid & (key == boundary)
    rank = before + tl.cumsum(tie.to(tl.int64), 0)
    nucleus = (key > boundary) | (tie & (rank <= quota))
    num = tl.load(policy + b * 5 + 2)
    min_log = tl.load(policy + b * 5 + 4).to(tl.float64, bitcast=True)
    support = valid & ((num == 0) | nucleus) & (scaled >= min_log)
    seed = tl.load(policy + b * 5).to(tl.uint64, bitcast=True)
    pos = tl.load(positions + b).to(tl.uint64, bitcast=True)
    x = _mix64(seed + 0x9E3779B97F4A7C15)
    x = _mix64(x ^ (pos * 0xD1B54A32D192ED03))
    x = _mix64(x ^ (v + FIRST_ID).to(tl.uint64))
    u = (x >> 12).to(tl.float64) * (2.0 ** -52) + (2.0 ** -53)
    score = tl.where(support, scaled - tldevice.log(-tldevice.log(u)), float("-inf"))
    m = tl.max(score, 0)
    hit = score == m
    k = tl.max(tl.where(hit, key, -1), 0)
    idx = tl.min(tl.where(hit & (key == k), v, 2147483647), 0)
    out = best + (b * BLOCKS + pid) * 3
    tl.store(out, m.to(tl.int64, bitcast=True))
    tl.store(out + 1, k)
    tl.store(out + 2, idx.to(tl.int64) + FIRST_ID)


@jit
def _row_best(best, packet, BLOCKS: tl.constexpr, PB: tl.constexpr):
    b = tl.program_id(0)
    i = tl.arange(0, PB)
    valid = i < BLOCKS
    score = tl.load(best + (b * BLOCKS + i) * 3, valid, 0).to(tl.float64, bitcast=True)
    score = tl.where(valid, score, float("-inf"))
    key = tl.load(best + (b * BLOCKS + i) * 3 + 1, valid, -1)
    ids = tl.load(best + (b * BLOCKS + i) * 3 + 2, valid, 9223372036854775807)
    m = tl.max(score, 0)
    hit = valid & (score == m)
    k = tl.max(tl.where(hit, key, -1), 0)
    idx = tl.min(tl.where(hit & (key == k), ids, 9223372036854775807), 0)
    tl.store(packet + b * 3, m.to(tl.int64, bitcast=True))
    tl.store(packet + b * 3 + 1, k)
    tl.store(packet + b * 3 + 2, idx)


@jit
def _merge(every, tokens, WORLD: tl.constexpr, PW: tl.constexpr):
    """The ranks' packets [WORLD, rows, 3] -> the first lexicographic maximum (ranks hold ascending ids)."""
    b = tl.program_id(0)
    rows = tl.num_programs(0)
    r = tl.arange(0, PW)
    valid = r < WORLD
    score = tl.where(valid, tl.load(every + (r * rows + b) * 3, valid, 0).to(tl.float64, bitcast=True),
                     float("-inf"))
    key = tl.load(every + (r * rows + b) * 3 + 1, valid, -1)
    ids = tl.load(every + (r * rows + b) * 3 + 2, valid, 9223372036854775807)
    m = tl.max(score, 0)
    hit = valid & (score == m)
    k = tl.max(tl.where(hit, key, -1), 0)
    tl.store(tokens + b, tl.min(tl.where(hit & (key == k), ids, 9223372036854775807), 0))


class DeviceSampler:
    """One rank's part of the keyed draw over its contiguous vocabulary shard (every rank, the same calls)."""

    def __init__(self, comm, *, rank: int, world: int, shard: int, device):
        self.comm, self.rank, self.world, self.shard = comm, rank, world, shard
        self.device = torch.device(device)
        self.blocks = triton.cdiv(shard, BLOCK)

    def _gather(self, t):
        recv = torch.empty((self.world,) + tuple(t.shape), dtype=t.dtype, device=t.device)
        self.comm.all_gather(t, recv)
        return recv

    def upload(self, values):
        """A host int64 list on the device without a host wait: a pinned staging copy (the caching host
        allocator keeps the block until the copy completes), queued behind the stream's work."""
        return torch.tensor(values, dtype=torch.int64).pin_memory().to(self.device, non_blocking=True)

    def words(self, words, rows: int):
        """Host words [rows * WORDS] -> the device policy int64 [rows, 5]: seed, inverse temperature FP64 bits,
        top_p numerator, its denominator's log2, min_log FP64 bits."""
        w = torch.tensor(words, dtype=torch.int64).view(rows, WORDS)
        pair = lambda i: (w[:, i] & 0xFFFFFFFF) | (w[:, i + 1] << 32)
        packed = torch.stack([pair(0), pair(2), pair(4), w[:, 6], pair(7)], 1).contiguous()
        return packed.pin_memory().to(self.device, non_blocking=True)

    def sample(self, local, top, positions, policy):
        """local FP32 [rows, shard] (contiguous); top FP32 [rows] (global row maxima); positions int64 [rows] (the
        drawn token's position); policy int64 [rows, 5] (``words``). -> int64 [rows] tokens, identical on every
        rank (rows with zero inverse temperature: arbitrary)."""
        rows, width = local.shape
        if width != self.shard or not local.is_contiguous() or local.dtype != torch.float32:
            raise ValueError("device sampling needs this rank's contiguous FP32 vocabulary shard rows")
        dev = local.device
        top = top.contiguous()
        positions = positions.contiguous()
        inverse = policy[:, 1].contiguous().view(torch.float64)
        state = torch.empty((rows, 4), dtype=torch.int64, device=dev)
        grid = (self.blocks, rows)
        for n, shift in enumerate((24, 16, 8, 0)):
            hist = torch.zeros((rows, 256), dtype=torch.int64, device=dev)
            _histogram[grid](local, top, inverse, state, hist, SHIFT=shift, V=width, BLOCK=BLOCK,
                             num_warps=8)
            _select[(rows,)](self._gather(hist), policy, state, WORLD=self.world, FIRST=n == 0, LAST=n == 3,
                             num_warps=4)
        counts = torch.empty((rows, self.blocks), dtype=torch.int64, device=dev)
        _ties[grid](local, state, counts, V=width, BLOCK=BLOCK, BLOCKS=self.blocks, num_warps=8)
        best = torch.empty((rows, self.blocks, 3), dtype=torch.int64, device=dev)
        pb = triton.next_power_of_2(self.blocks)
        _draw[grid](local, top, inverse, policy, positions, state, self._gather(counts), best, RANK=self.rank,
                    WORLD=self.world, FIRST_ID=self.rank * width, V=width, BLOCK=BLOCK, BLOCKS=self.blocks, PB=pb,
                    num_warps=8)
        packet = torch.empty((rows, 3), dtype=torch.int64, device=dev)
        _row_best[(rows,)](best, packet, BLOCKS=self.blocks, PB=pb, num_warps=1)
        tokens = torch.empty(rows, dtype=torch.int64, device=dev)
        _merge[(rows,)](self._gather(packet), tokens, WORLD=self.world, PW=triton.next_power_of_2(self.world),
                        num_warps=1)
        return tokens

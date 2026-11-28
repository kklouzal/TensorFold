"""Keyed top-k-off nucleus/min-p draws of decode rows on the device, under the host sampler's rule.

``round_sampling._Nucleus`` + ``tensorfold.cuda.sampling._support``/``_draw_support`` choose a positive-temperature
token from the full gathered row: FP64 scores centered on the row maximum and multiplied by the row's FP64 inverse
temperature, fixed-point masses ``floor(exp(scaled) * 2**40)``, the nucleus as the shortest (score desc, id asc)
prefix whose exact integer mass reaches ``ceil(top_p * total)``, the min-p prefix ``scaled >= ln(min_p)``, and the
keyed Gumbel argmax ``scaled - log(-log(uniform(seed, position, id)))`` over that support.

Here every rank keeps its contiguous vocabulary shard. The boundary of the nucleus is found by an exact radix
select over the monotone FP32 score key (four 8-bit passes, each an all-gathered integer mass histogram summed in
rank order, so every rank takes identical decisions), then the boundary's equal-score ties are taken in global ID
order. Integer masses and the exact limb product for ``top_p * total`` keep the support identical to the host's.
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

WORDS = 9                   # int32 words per row: seed (2), inverse temperature bits (2), top_p numerator (2),
                            # top_p denominator log2 (1), min_log bits (2)
_C1, _C2 = 0x9E3779B97F4A7C15, 0xD1B54A32D192ED03
_M1, _M2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
_MASS = int(MASS)
_LIMB = 26
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


def _shr(x, k):
    return (x >> k) & ((1 << (64 - k)) - 1)                     # logical shift of uint64 bits held in int64


def _mix(x):
    x = x ^ _shr(x, 30)
    x = x * _signed(_M1)
    x = x ^ _shr(x, 27)
    x = x * _signed(_M2)
    return x ^ _shr(x, 31)


def _need(total, num, shift):
    """ceil(num * total / 2**shift) exactly in int64 limbs (total < 2**58, num < 2**53, 1 <= shift <= 104)."""
    mask = (1 << _LIMB) - 1
    t1, t0 = total >> _LIMB, total & mask
    n1, n0 = num >> _LIMB, num & mask
    low = t0 * n0
    mid = t1 * n0 + t0 * n1 + (low >> _LIMB)
    high = t1 * n1 + (mid >> _LIMB)
    rest = ((mid & mask) << _LIMB) | (low & mask)               # product = high * 2**52 + rest, rest < 2**52
    small = shift <= 2 * _LIMB
    s1 = torch.where(small, shift, 0)
    s2 = torch.where(small, 0, shift - 2 * _LIMB)
    q_small = (high << (2 * _LIMB - s1)) + (rest >> s1)
    r_small = rest & (torch.bitwise_left_shift(torch.ones_like(s1), s1) - 1)
    q_large = high >> s2
    r_large = (high & (torch.bitwise_left_shift(torch.ones_like(s2), s2) - 1)) | rest
    q = torch.where(small, q_small, q_large)
    r = torch.where(small, r_small, r_large)
    return q + (r != 0).to(q.dtype)


class DeviceSampler:
    """One rank's part of the keyed draw over its contiguous vocabulary shard (every rank, the same calls)."""

    def __init__(self, comm, *, rank: int, world: int, shard: int, device):
        self.comm, self.rank, self.world, self.shard = comm, rank, world, shard
        self.ids = torch.arange(rank * shard, (rank + 1) * shard, dtype=torch.int64, device=device)
        self.device = torch.device(device)

    def _gather(self, t):
        recv = torch.empty((self.world,) + tuple(t.shape), dtype=t.dtype, device=t.device)
        self.comm.all_gather(t.contiguous(), recv)
        return recv

    def upload(self, values):
        """A host int64 list on the device without a host wait: a pinned staging copy (the caching host
        allocator keeps the block until the copy completes), queued behind the stream's work."""
        return self._staged(torch.tensor(values, dtype=torch.int64))

    def _staged(self, host):
        if self.device.type == "cuda":
            host = host.pin_memory()
        return host.to(self.device, non_blocking=True)

    def words(self, words, rows: int):
        """Host words [rows * WORDS] -> the device policy int64 [rows, 5]."""
        w = torch.tensor(words, dtype=torch.int64).view(rows, WORDS)
        pair = lambda i: (w[:, i] & 0xFFFFFFFF) | (w[:, i + 1] << 32)
        packed = torch.stack([pair(0), pair(2), pair(4), w[:, 6], pair(7)], 1)
        return self._staged(packed)

    def sample(self, local, top, positions, policy):
        """local FP32 [rows, shard]; top FP32 [rows] (global row maxima); positions int64 [rows] (the drawn
        token's position); policy int64 [rows, 5] (``words``). -> int64 [rows] tokens (rows with zero inverse
        temperature: arbitrary)."""
        rows = local.shape[0]
        seed, num, shift = policy[:, 0], policy[:, 2], policy[:, 3]
        inverse = policy[:, 1].view(torch.float64)
        min_log = policy[:, 4].view(torch.float64)
        sampled = inverse > 0
        scores = local.double()
        # greedy rows run the same arithmetic at temperature one (finite throughout; their result is unused)
        scaled = (scores - top.double()[:, None]) * torch.where(sampled, inverse, 1.0)[:, None]
        mass = torch.floor(torch.exp(scaled) * MASS).to(torch.int64)
        total = self._gather(mass.sum(-1)).sum(0)                       # exact: integers
        # monotone key of the FP32 score (+0.0 for -0.0: equal scores are one key)
        bits = (local + 0.0).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        key = torch.where(bits >= 1 << 31, (~bits) & 0xFFFFFFFF, bits | (1 << 31))
        cut = (num > 0) & sampled
        need = torch.where(cut, _need(total, num, torch.where(cut, shift, 1)), total)
        prefix = torch.zeros(rows, dtype=torch.int64, device=local.device)
        above = torch.zeros_like(prefix)
        bins = torch.arange(256, device=local.device)
        hist = None
        for shift_bits in (24, 16, 8, 0):
            active = (key >> (shift_bits + 8)) == prefix[:, None] if shift_bits < 24 else None
            bucket = (key >> shift_bits) & 255
            part = torch.where(active, mass, 0) if active is not None else mass
            hist = torch.zeros((rows, 256), dtype=torch.int64, device=local.device)
            hist.scatter_add_(1, bucket, part)
            hist = self._gather(hist).sum(0)
            # inclusive mass from the top bin down: above + sum_{b' >= b}
            reach = above[:, None] + hist.flip(1).cumsum(1).flip(1)
            chosen = torch.where(reach >= need[:, None], bins, -1).max(1).values.clamp(min=0)
            above = above + torch.where(bins > chosen[:, None], hist, 0).sum(1)
            prefix = (prefix << 8) | chosen
        boundary = prefix
        ties = key == boundary[:, None]
        tie_mass = hist.gather(1, (boundary & 255)[:, None])[:, 0]      # every boundary tie's mass, summed
        counts = self._gather(ties.sum(-1))                              # [world, rows]
        tie_count = counts.sum(0).clamp(min=1)
        each = tie_mass // tie_count                                     # equal scores: equal masses
        quota = torch.where(each > 0, -(-(need - above) // each.clamp(min=1)), tie_count)
        before = counts[:self.rank].sum(0) if self.rank else torch.zeros_like(quota)
        local_quota = (quota - before).clamp(min=0)
        tie_rank = ties.to(torch.int64).cumsum(-1)
        nucleus = (key > boundary[:, None]) | (ties & (tie_rank <= local_quota[:, None]))
        support = torch.where(cut[:, None], nucleus, torch.ones_like(nucleus)) & (scaled >= min_log[:, None])
        # keyed Gumbel: exact_sampling.uniform(seed, position, id)
        x = _mix(seed + _signed(_C1))
        x = _mix(x ^ (positions * _signed(_C2)))
        x = _mix(x[:, None] ^ self.ids[None, :])
        u = _shr(x, 12).double() * 2.0 ** -52 + 2.0 ** -53
        score = torch.where(support, scaled - torch.log(-torch.log(u)), float("-inf"))
        best = score.max(-1).values
        tied = score == best[:, None]
        best_key = torch.where(tied, key, -1).max(-1).values
        first = (tied & (key == best_key[:, None])).to(torch.int8).argmax(-1)
        packet = torch.stack([best.view(torch.int64), best_key, self.ids[first]], 1)
        got = self._gather(packet)                                       # [world, rows, 3]
        g_best = got[:, :, 0].view(torch.float64)
        top_score = g_best.max(0).values
        hit = g_best == top_score[None]
        top_key = torch.where(hit, got[:, :, 1], -1).max(0).values
        hit &= got[:, :, 1] == top_key[None]
        choice = hit.to(torch.int8).argmax(0)                            # first rank: lowest global ID
        return got[:, :, 2].gather(0, choice[None])[0]

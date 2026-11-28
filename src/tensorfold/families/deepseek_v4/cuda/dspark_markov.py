"""Batched DSpark Markov greedy steps over a vocabulary shard (TP8: the target head's rows of this rank).

One step for B requests: ``value[b, v] = logits[b, v] + W_bf16[v] . embed(prev_b)`` over this rank's rows v, the
local maximum with its lowest index (block maxima, then one reduce), then the ranks' (maximum, global index) pairs
are all-gathered and the first rank holding the global maximum gives the token: exactly ``argmax`` of the
concatenated row (shards are contiguous in rank order; lowest index on ties), as upstream ``MarkovGreedy``
(1Cat-vLLM 4d57473d ``vllm/models/deepseek_v41/markov_fused.py``), whose per-row bias arithmetic (BF16 bits shifted
to FP32, ``tl.sum`` over the 256 rank columns) is kept verbatim. Graph-capturable (no host reads).
"""

from __future__ import annotations

import torch

from .kernels.triton_utils import jit, tl, triton

ROWS = 128


@jit
def _markov_rows(weight, embed_rows, logits, stride, best_val, best_idx, V: tl.constexpr, K: tl.constexpr,
                 ROWS: tl.constexpr, BLOCKS: tl.constexpr):
    pid, b = tl.program_id(0), tl.program_id(1)
    v = pid * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, K)
    valid = v < V
    raw = tl.load(weight + v[:, None] * K + k[None, :], valid[:, None], 0)
    w = (raw.to(tl.int32) << 16).to(tl.float32, bitcast=True)
    e = (tl.load(embed_rows + b * K + k).to(tl.int32) << 16).to(tl.float32, bitcast=True)   # BF16 bits (SM70)
    bias = tl.sum(w * e[None, :], 1)
    value = tl.where(valid, tl.load(logits + b * stride + v, valid, float("-inf")) + bias, float("-inf"))
    m = tl.max(value, 0)
    idx = tl.min(tl.where(value == m, v, 2147483647), 0)
    tl.store(best_val + b * BLOCKS + pid, m)
    tl.store(best_idx + b * BLOCKS + pid, idx)


@jit
def _mix64(x):
    x = x ^ (x >> 30)
    x = x * 0xBF58476D1CE4E5B9
    x = x ^ (x >> 27)
    x = x * 0x94D049BB133111EB
    return x ^ (x >> 31)


@jit
def _markov_rows_keyed(weight, embed_rows, logits, stride, best_val, best_idx, seeds, inverse, positions, first,
                       V: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr, BLOCKS: tl.constexpr):
    """``_markov_rows`` scoring ``value * inverse + Gumbel(uniform(seed, position, global id))`` for rows with a
    positive inverse temperature (the target sampler's keyed noise at this draft row's position); others keep
    ``value``, so greedy rows choose exactly as ``_markov_rows``."""
    pid, b = tl.program_id(0), tl.program_id(1)
    v = pid * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, K)
    valid = v < V
    raw = tl.load(weight + v[:, None] * K + k[None, :], valid[:, None], 0)
    w = (raw.to(tl.int32) << 16).to(tl.float32, bitcast=True)
    e = (tl.load(embed_rows + b * K + k).to(tl.int32) << 16).to(tl.float32, bitcast=True)   # BF16 bits (SM70)
    bias = tl.sum(w * e[None, :], 1)
    value = tl.where(valid, tl.load(logits + b * stride + v, valid, float("-inf")) + bias, float("-inf"))
    inv = tl.load(inverse + b)
    seed = tl.load(seeds + b).to(tl.uint64, bitcast=True)
    pos = tl.load(positions + b).to(tl.uint64, bitcast=True)
    x = _mix64(seed + 0x9E3779B97F4A7C15)
    x = _mix64(x ^ (pos * 0xD1B54A32D192ED03))
    x = _mix64(x ^ (v + first).to(tl.uint64))
    u = (x >> 12).to(tl.float64) * (2.0 ** -52) + (2.0 ** -53)
    noisy = (value.to(tl.float64) * inv - tl.log(-tl.log(u))).to(tl.float32)
    value = tl.where(inv > 0, tl.where(valid, noisy, float("-inf")), value)
    m = tl.max(value, 0)
    idx = tl.min(tl.where(value == m, v, 2147483647), 0)
    tl.store(best_val + b * BLOCKS + pid, m)
    tl.store(best_idx + b * BLOCKS + pid, idx)


@jit
def _markov_local(best_val, best_idx, first, packet, BLOCKS: tl.constexpr, PB: tl.constexpr):
    b = tl.program_id(0)
    i = tl.arange(0, PB)
    valid = i < BLOCKS
    val = tl.load(best_val + b * BLOCKS + i, valid, float("-inf"))
    idx = tl.load(best_idx + b * BLOCKS + i, valid, 2147483647)
    m = tl.max(val, 0)
    token = tl.min(tl.where(val == m, idx, 2147483647), 0)
    tl.store(packet + b * 2, m)
    tl.store(packet + b * 2 + 1, (token + first).to(tl.float32))     # global ids < 2^24: exact in FP32


@jit
def _markov_tp8(every, tokens, B: tl.constexpr):
    b = tl.program_id(0)
    rank = tl.arange(0, 8)
    value = tl.load(every + 2 * (rank * B + b))
    nan = value != value
    maximum = tl.max(tl.where(nan, float("-inf"), value), 0)
    first = tl.min(tl.where(value == maximum, rank, 8), 0)
    # Torch max propagates NaN, so the old equality mask is all false and
    # its argmax selects rank zero, even when the NaN belongs to another rank.
    owner = tl.where(tl.sum(nan.to(tl.int32), 0) != 0, 0, first)
    token = tl.load(every + 2 * (owner * B + b) + 1)
    tl.store(tokens + b, token.to(tl.int64))


def merge_rank_major(every):
    """Select one int64 ID per request from contiguous FP32 [8, B, 2] packets."""
    b = every.shape[1]
    tokens = torch.empty(b, dtype=torch.int64, device=every.device)
    _markov_tp8[(b,)](every, tokens, B=b, num_warps=1)
    return tokens


class ShardedMarkov:
    """Greedy Markov steps per shard. For TP8, gather [B, 2] packets into contiguous [8, B, 2]."""

    def __init__(self, weight_bf16, embed_bf16, first: int, gather_rank_major, world: int):
        self.weight = weight_bf16.contiguous().view(torch.int16)
        # BF16 bits as int16: SM70 PTX has no BF16 conversion, so kernels widen by a 16-bit shift (exact)
        self.embed = embed_bf16.contiguous().view(torch.int16)
        self.vocab, self.rank = self.weight.shape
        self.first, self.gather_rank_major, self.world = first, gather_rank_major, world
        self.blocks = triton.cdiv(self.vocab, ROWS)

    def local_packet(self, logits, prev, noise=None):
        """This rank's FP32 [B, 2] (local maximum, its lowest global index) of logits FP32 [B, V_local] (row stride
        ``logits.stride(0)``) plus the Markov bias of prev int64 [B]. ``noise``: (seeds int64 [B], inverse
        temperatures FP64 [B], positions int64 [B]): rows with a positive inverse take the keyed Gumbel maximum."""
        b = prev.shape[0]
        dev = logits.device
        rows = torch.index_select(self.embed, 0, prev).contiguous()
        best_val = torch.empty(b, self.blocks, dtype=torch.float32, device=dev)
        best_idx = torch.empty(b, self.blocks, dtype=torch.int32, device=dev)
        if noise is None:
            _markov_rows[(self.blocks, b)](self.weight, rows, logits, logits.stride(0), best_val, best_idx,
                                           V=self.vocab, K=self.rank, ROWS=ROWS, BLOCKS=self.blocks, num_warps=4)
        else:
            seeds, inverse, positions = noise
            _markov_rows_keyed[(self.blocks, b)](self.weight, rows, logits, logits.stride(0), best_val, best_idx,
                                                 seeds, inverse, positions, self.first, V=self.vocab, K=self.rank,
                                                 ROWS=ROWS, BLOCKS=self.blocks, num_warps=4)
        packet = torch.empty(b, 2, dtype=torch.float32, device=dev)
        _markov_local[(b,)](best_val, best_idx, self.first, packet, BLOCKS=self.blocks,
                            PB=triton.next_power_of_2(self.blocks), num_warps=4)
        return packet

    def step(self, logits, prev, noise=None):
        """logits FP32 [B, V_local], prev int64 [B] -> int64 [B] tokens (identical on every rank)."""
        packet = self.local_packet(logits, prev, noise)
        if self.world == 1:
            return packet[:, 1].long()
        return merge_rank_major(self.gather_rank_major(packet))

    def bias(self, prev):
        """FP32 [B, V_local] Markov bias rows of prev int64 [B] (the sampling path)."""
        rows = torch.index_select(self.embed, 0, prev).contiguous()
        out = torch.empty(rows.shape[0], self.vocab, dtype=torch.float32, device=rows.device)
        if rows.shape[0]:
            _markov_bias_bits[(self.blocks, rows.shape[0])](self.weight, rows, out, V=self.vocab, K=self.rank,
                                                            ROWS=ROWS, num_warps=4)
        return out


@jit
def _markov_bias_bits(weight, embed_rows, out, V: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr):
    """upstream ``_markov_bias`` with BF16-bit embedding rows."""
    pid, n = tl.program_id(0), tl.program_id(1)
    v = pid * ROWS + tl.arange(0, ROWS)
    k = tl.arange(0, K)
    valid = v < V
    raw = tl.load(weight + v[:, None] * K + k[None, :], valid[:, None], 0)
    w = (raw.to(tl.int32) << 16).to(tl.float32, bitcast=True)
    e = (tl.load(embed_rows + n * K + k).to(tl.int32) << 16).to(tl.float32, bitcast=True)
    tl.store(out + n * V + v, tl.sum(w * e[None, :], 1), valid)

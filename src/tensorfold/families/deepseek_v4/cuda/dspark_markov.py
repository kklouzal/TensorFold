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


class ShardedMarkov:
    """Greedy Markov steps over B requests on one vocabulary shard; ``gather`` concatenates the ranks' [B, 2]
    packets along the last dimension (``Collectives.gather``; identity for world 1)."""

    def __init__(self, weight_bf16, embed_bf16, first: int, gather, world: int):
        self.weight = weight_bf16.contiguous().view(torch.int16)
        # BF16 bits as int16: SM70 PTX has no BF16 conversion, so kernels widen by a 16-bit shift (exact)
        self.embed = embed_bf16.contiguous().view(torch.int16)
        self.vocab, self.rank = self.weight.shape
        self.first, self.gather, self.world = first, gather, world
        self.blocks = triton.cdiv(self.vocab, ROWS)

    def local_packet(self, logits, prev):
        """This rank's FP32 [B, 2] (local maximum, its lowest global index) of logits FP32 [B, V_local] (row stride
        ``logits.stride(0)``) plus the Markov bias of prev int64 [B]."""
        b = prev.shape[0]
        dev = logits.device
        rows = torch.index_select(self.embed, 0, prev).contiguous()
        best_val = torch.empty(b, self.blocks, dtype=torch.float32, device=dev)
        best_idx = torch.empty(b, self.blocks, dtype=torch.int32, device=dev)
        _markov_rows[(self.blocks, b)](self.weight, rows, logits, logits.stride(0), best_val, best_idx,
                                       V=self.vocab, K=self.rank, ROWS=ROWS, BLOCKS=self.blocks, num_warps=4)
        packet = torch.empty(b, 2, dtype=torch.float32, device=dev)
        _markov_local[(b,)](best_val, best_idx, self.first, packet, BLOCKS=self.blocks,
                            PB=triton.next_power_of_2(self.blocks), num_warps=4)
        return packet

    @staticmethod
    def merge(every, world):
        """[B, world * 2] rank-ordered packets -> int64 [B]: the first rank holding the global maximum."""
        every = every.view(every.shape[0], world, 2)
        values = every[:, :, 0]
        best = values.max(dim=1, keepdim=True).values
        owner = (values == best).to(torch.int32).argmax(dim=1, keepdim=True)
        return every[:, :, 1].gather(1, owner)[:, 0].long()

    def step(self, logits, prev):
        """logits FP32 [B, V_local], prev int64 [B] -> int64 [B] tokens (identical on every rank)."""
        packet = self.local_packet(logits, prev)
        if self.world == 1:
            return packet[:, 1].long()
        return self.merge(self.gather(packet), self.world)

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

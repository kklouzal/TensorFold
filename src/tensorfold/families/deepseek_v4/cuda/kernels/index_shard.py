# SPDX-License-Identifier: Apache-2.0
"""Position-sharded decode indexer for TP8 (TensorFold-owned; ledger candidate K4): every rank scores all 32
indexer heads over every 8th key block instead of its 4 heads over every key.

The TP8 layout gives each rank 4 of the 32 indexer heads (``idx_wq_b`` / ``idx_proj`` column shards). The
head-sharded decode scores (``mu_packed.index_scores``: 4 heads padded to a 16-head tile) unpack every packed
FP4 key on every rank and sum the per-rank partials with an FP32 all-reduce of [rows, width]. Here the ranks first
all-gather the rows' queries and head weights (FP16, 8 x [rows, 4 x 128 + 4]); rank r then scores the key blocks
b (BN = 64 compressed positions) with b % 8 == r for all 32 heads (``_shard_scores``: ``_pk_index_scores``'
arithmetic per key block and head: unscaled E2M1 keys, one FP32 dot per 32-wide group, power-of-two scaling, FP16
round, ReLU, x weight, FP16 round), sums the 32 heads in FP32 and rounds once to FP16 (the head-sharded path sums
4 heads per rank, then the 8 rank partials, in FP32, and rounds once: the same values, summed in another order).
An FP16 all-gather of the [rows, width / 8] shards (a quarter of the FP32 all-reduce's bytes per rank) gives
every rank the whole row in rank-major block order; one copy restores position order and the head-sharded path's
own selection (``mu_topk.topk``: index_topk / index_topk_parallel) picks the top k. Each rank unpacks 1/8 of the keys and runs 4x fewer head dots than the padded head-sharded kernel.
"""

import torch

from .attention_ops import _e2m1_bits
from . import mu_topk
from .triton_utils import fma_dot, jit, tl, triton

WORLD = 8
BN = 64                 # compressed positions per key block (the shard unit)
HEADS = 32


@jit(do_not_specialize=["N_OUT", "TABLE_STRIDE"])
def _shard_scores(q, w, cache, block_table, positions, valid_tokens, token_req, out, N_OUT, TABLE_STRIDE, RANK,
                  PAGE: tl.constexpr, BLOCK_SIZE: tl.constexpr, RATIO: tl.constexpr, OFFSET: tl.constexpr,
                  H: tl.constexpr, BN_: tl.constexpr, BLOCKS: tl.constexpr, WORLD_: tl.constexpr):
    """Scores of token t over its request's key blocks RANK, RANK + WORLD, ... into out[t, j * BN + i] (block j of
    this rank); _pk_index_scores' per-block arithmetic over H heads."""
    t, part = tl.program_id(0), tl.program_id(1)
    r = tl.load(token_req + t).to(tl.int64)
    position = tl.load(positions + t).to(tl.int64)
    live = (tl.load(valid_tokens + t) != 0) & (r >= 0)
    end = tl.where(live, (position + 1) // RATIO, 0)
    ROWS: tl.constexpr = BLOCK_SIZE // RATIO
    first_j = part * BLOCKS
    for _ in range(((first_j * WORLD_ + RANK) * BN_ < end).to(tl.int32)):
        h = tl.arange(0, H)
        d = tl.arange(0, 32)
        row = (t * H + h)[:, None] * 128
        qv0 = tl.load(q + row + d[None, :])
        qv1 = tl.load(q + row + 32 + d[None, :])
        qv2 = tl.load(q + row + 64 + d[None, :])
        qv3 = tl.load(q + row + 96 + d[None, :])
        wv = tl.load(w + t * H + h).to(tl.float32)
        for j in range(first_j, first_j + BLOCKS):
            n0 = (j * WORLD_ + RANK) * BN_
            for _live in range((n0 < end).to(tl.int32)):      # 0/1-trip loop around the dots (sm70 codegen, as upstream)
                n = n0 + tl.arange(0, BN_)
                inside = n < end
                page = tl.load(block_table + r * TABLE_STRIDE + n // ROWS, inside, -1).to(tl.int64)
                inside &= page >= 0
                base = cache + page * PAGE + OFFSET + (n % ROWS).to(tl.int64) * 68
                b16 = tl.arange(0, 16)
                acc = tl.zeros((H, BN_), tl.float32)
                for g in tl.static_range(4):
                    qv = qv0 if g == 0 else (qv1 if g == 1 else (qv2 if g == 2 else qv3))
                    byte = tl.load(base[:, None] + g * 16 + b16[None, :], inside[:, None], 0).to(tl.int32)
                    lo = _e2m1_bits(byte & 7) | (((byte >> 3) & 1).to(tl.uint32) << 31)
                    hi = _e2m1_bits((byte >> 4) & 7) | (((byte >> 7) & 1).to(tl.uint32) << 31)
                    pair = tl.join(lo.to(tl.float32, bitcast=True), hi.to(tl.float32, bitcast=True))
                    kv = tl.reshape(pair, (BN_, 32)).to(tl.float16)  # unscaled E2M1, exact
                    piece = fma_dot(qv, tl.trans(kv))
                    exponent = tl.load(base + 64 + g, inside, 127).to(tl.int32)
                    acc += piece * (exponent << 23).to(tl.float32, bitcast=True)[None, :]
                score = tl.maximum(acc.to(tl.float16).to(tl.float32), 0.0)
                score = (score * wv[:, None]).to(tl.float16).to(tl.float32)
                total = tl.sum(score, 0)
                tl.store(out + t.to(tl.int64) * N_OUT + j * BN_ + tl.arange(0, BN_), total.to(tl.float16), n < end)


def eligible(heads: int, world: int, width: int) -> bool:
    return heads * world == HEADS and world == WORLD and width % (WORLD * BN) == 0


def select_decode(query, weights, positions, valid, token_req, cache, block_table, *, layout, source, ratio, width,
                  rank, comm, out=None, k=512, hierarchy=None):
    """Top-k compressed positions [T, k] int32 (ascending, -1 padded) of decode rows from this rank's indexer
    heads query [T, 4, 128] FP16 (after RoPE + FP4 QDQ) and weights [T, 4] FP16; ``comm.all_gather`` is the
    8-rank gather (every rank calls this with the same shapes)."""
    tokens, heads = query.shape[0], query.shape[1]
    device = query.device
    # 1. queries and head weights of all 32 heads (global head 4 r + i), one FP16 gather
    packed = torch.cat([query.reshape(tokens, heads * 128), weights.reshape(tokens, heads)], 1).contiguous()
    gathered = torch.empty((WORLD,) + tuple(packed.shape), dtype=torch.float16, device=device)
    comm.all_gather(packed, gathered)
    q_all = gathered[:, :, :heads * 128].reshape(WORLD, tokens, heads, 128).permute(1, 0, 2, 3).reshape(
        tokens, HEADS, 128).contiguous()
    w_all = gathered[:, :, heads * 128:].permute(1, 0, 2).reshape(tokens, HEADS).contiguous()
    # 2. this rank's key blocks, all heads
    part = width // WORLD
    shard = torch.empty(tokens, part, dtype=torch.float16, device=device)
    ratio_, _rows, _c, i_offset = layout.regions[source]
    blocks = part // BN
    per = 4
    _shard_scores[(tokens, triton.cdiv(blocks, per))](
        q_all, w_all, cache, block_table, positions, valid, token_req, shard, shard.stride(0),
        block_table.stride(0), rank, PAGE=layout.page_bytes, BLOCK_SIZE=layout.block_size, RATIO=ratio_,
        OFFSET=i_offset, H=HEADS, BN_=BN, BLOCKS=per, WORLD_=WORLD, num_warps=8)
    # 3. every rank: the whole rows (rank-major blocks -> position order), then the head-sharded path's selection
    rows = torch.empty((WORLD, tokens, part), dtype=torch.float16, device=device)
    comm.all_gather(shard, rows)
    return select_gathered(rows, positions, valid, ratio, out=out, k=k, hierarchy=hierarchy)


def select_gathered(rows, positions, valid, ratio, *, out=None, k=512, hierarchy=None):
    """Top-k [T, k] of the all-gathered shards rows [WORLD, T, width / WORLD] (rank-major key blocks)."""
    _, tokens, part = rows.shape
    scores = rows.view(WORLD, tokens, part // BN, BN).permute(1, 2, 0, 3).reshape(tokens, WORLD * part)
    result = out if out is not None else torch.empty(tokens, k, dtype=torch.int32, device=rows.device)
    if hierarchy is not None:
        from .mu_reindex import select_ranked

        candidates, block, publish = hierarchy
        return select_ranked(result, scores, positions, valid, ratio, mu_topk.topk,
                             candidates=candidates, block=block, publish=publish)
    return mu_topk.topk(result, scores, positions, valid, ratio, k)

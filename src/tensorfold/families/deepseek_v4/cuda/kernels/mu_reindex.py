# SPDX-License-Identifier: Apache-2.0
# Provenance: friend overlay patch 170 (SPDX Apache-2.0), snapshot lp-snap-c1es8-numa-admit2-imgtext-share2-fix5-ram4-stepbud-ram3-memfix-ctx1m-ppovl-opt2-pfx-1-eng-pfxfix4,
#   patches/ds41mu_files/local_ds41_mu_reindex.py (sha256 4a7e181eab7f29cd711a2bc5d1b20c209cd08ca71da29fc4f2325360314a0995).
# TensorFold changes (kernel arithmetic, candidate rules and prefill selector verbatim):
# - step.prefill entries may carry extra host fields after (r, s0, s1) (Target.Step's last position): ignored.
# - tl.dot -> triton_utils.fma_dot(_acc): the FP32 FMA dot his Triton 3.6 runs on SM70 (Triton 3.2's MMAv1
#   results cannot be reduced on SM70; see triton_utils).
# - @triton.jit -> triton_utils.jit: the same JITFunction; launches add the SM70 Triton 3.2 profile options
#   (ptx_version=83, needed with CUDA 12.9 ptxas). Kernel arithmetic unchanged.
# - vLLM imports -> the vendored package (attention_ops, mu_packed, index_topk, mu_topk); triton via triton_utils.
# - Flags at production values: MU_TOPK_SEG=1 (mu_topk.topk), CTX_1M=1 (the >512K-key tile rule applies
#   unconditionally), PF_SRC_TOPK unset (its patch-190 hook dropped), MU_INDEX=paged (its check dropped).
# - Not vendored: enabled() (flag policy); the MU_IDX8 (patch 172) CUDA-extension hook of candidate_scores;
#   in select(): DEC_FUSE (263), DEC_SPLIT (195), PF_RX/PF_SYNC (190) and PF_INDEX (180) hooks. Production
#   enables DEC_SPLIT and PF_INDEX; this module is the path those hooks fall back to.
# - select() is vLLM-free: the caller computes the indexer query/weights, passes the block table, the TP
#   all-reduce callable and the source's candidates explicitly, and receives (positions, candidates).
# - TP8 (TensorFold, 2026-11-20): candidate_scores accepts an FP32 `out` (unrounded head sums, as
#   mu_packed.index_scores); FP16 `out` keeps the overlay's store bit for bit; other dtypes are refused.
# - The overlay's private copy of attention_ops._e2m1_bits (identical body) is imported instead.
# - Lint-only source edits: range(0, n) -> range(n) for the 0/1-trip loops (identical Triton loop).
# - Triton 3.2: no rewrite was needed.
"""LOCAL-PATCH 170 (ds41_mu_reindex) hierarchical sparse indexer (Full / Reindex / Reuse).

Upstream reference: deepseek-ai/DeepSeek-V4.1-Flash-1017 inference/model.py
(rev 1bae0e10) `Indexer.forward` + `select_candidate_blocks`, config
candidate_source_layer 20, candidate_topk_blocks 2048, candidate_block_size 8;
paper 2609.19969 section 2.3.2 / 4.2.1 (decoder: 1 Full, 4 Reindex, 15 Reuse).

  Full     index sources 2, 8, 14 (ratio 2) and 20 (ratio 1, the candidate
           source): score every causally visible index row, top-512.
  source   layer 20 additionally keeps, per query, the 2048 blocks of 8
           positions with the highest block score (max over the block's
           scores; the block holding the newest position is pinned in).
  Reindex  index sources 24, 28, 32, 36 (source < layer): their own query and
           weights, but scored ONLY inside the source's candidate blocks
           (<= 16384 positions, constant in context), top-512 of those.
  Reuse    every other compressed layer reuses the last top-k (unchanged).

The fork (attention_ops.select_compressed_positions: "only candidate-block
prefilter is bypassed") and patches 85/87 score every visible row in all 8
index layers. Below 16384 visible positions per source (<= 2048 blocks) every
block is a candidate, so this path is bit-identical to patch 87 there; above
it, it follows upstream (deliberate change of the selected positions).

Decode rows (patch 85's batched rows): Full layers keep 87's scores and top-k
kernels; the source computes block maxima of its all-reduced FP16 scores and
an exact top-2048 (patch 87's index_topk kernel, K=2048: lowest block index
on ties, ascending); Reindex layers score [T, 2048 x 8] candidate slots with
87's per-key arithmetic (_rx_scores: the _pk_index_scores block with
candidate addressing), all-reduce 16384 instead of max_model_len scores, and
take index_topk over the live prefix (candidates are ascending and the pinned
newest block is the last one, so live slots form a prefix). Graph-safe: fixed
shapes, lengths read on the device.

Prefill rows (> DECODE_MAX tokens): the upstream single-request selector
(select_compressed_positions' arithmetic, tile by tile) with the candidate
mask applied exactly like upstream (masked_fill of non-candidates to -inf).

Candidates live on the shared attention state for one forward (the source
and its users are on the same PP stage: layers 20..39); a user without a
source result in this forward raises.

Flag: VLLM_LOCAL_DS41_MU_REINDEX=1 (needs VLLM_LOCAL_DS41_MU_PACKED=1 and
VLLM_LOCAL_DS41_MU_INDEX=paged). No vLLM import.
"""

import torch

from . import attention_ops as ops
from . import mu_packed as pk
from .attention_ops import _e2m1_bits
from .index_topk import index_topk
from .mu_topk import topk as _seg_topk
from .triton_utils import fma_dot, jit, tl, triton

LOG = "LOCAL-PATCH 170"
TOP_K = 512


def hierarchy(config):
    """(candidate source layer, top blocks, block size) or None (no hierarchy)."""
    source = getattr(config, "candidate_source_layer_id", None)
    if source is None:
        source = getattr(config, "candidate_source_layer", -1)
    blocks = int(getattr(config, "candidate_topk_blocks", 0) or 0)
    size = int(getattr(config, "candidate_block_size", 0) or 0)
    if source is None or int(source) < 0 or blocks <= 0 or size <= 0:
        return None
    if blocks & (blocks - 1) or (blocks - 1) * size < TOP_K:
        # index_topk needs a power-of-two K; a pool that cannot hold top-512
        # live positions would let upstream's top-k pick masked (-inf) rows.
        raise ValueError(f"{LOG}: candidate pool {blocks} x {size} not supported")
    return int(source), blocks, size


def role(layer, source):
    """'source' | 'reindex' | 'full' for an index-source layer (upstream rule)."""
    if layer == source:
        return "source"
    if 0 <= source < layer:
        return "reindex"
    return "full"


# ------------------------------------------------------------------ kernels
@jit(do_not_specialize=["N_OUT", "TABLE_STRIDE", "CAND_STRIDE"])
def _rx_scores(
    q,
    w,
    cache,
    block_table,
    positions,
    valid_tokens,
    token_req,
    cand,
    out,
    N_OUT,
    TABLE_STRIDE,
    CAND_STRIDE,
    PAGE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RATIO: tl.constexpr,
    OFFSET: tl.constexpr,
    H: tl.constexpr,
    HB: tl.constexpr,
    BN: tl.constexpr,
    BLOCKS: tl.constexpr,
    CB: tl.constexpr,
):
    """Reindex scores of token t at its candidate slots j: position cand[t, j // CB]
    * CB + j % CB. Per key block: local_mu_packed._pk_index_scores' arithmetic
    verbatim (unscaled E2M1 FP16 keys, one dot per 32-group, FP32 power-of-two
    scaling, FP16 round, ReLU, x weight, FP16 round, head sum, FP16 store)."""
    t, part = tl.program_id(0), tl.program_id(1)
    r = tl.load(token_req + t).to(tl.int64)
    position = tl.load(positions + t).to(tl.int64)
    live = (tl.load(valid_tokens + t) != 0) & (r >= 0)
    end = tl.where(live, (position + 1) // RATIO, 0)
    first = part * BLOCKS * BN
    ROWS: tl.constexpr = BLOCK_SIZE // RATIO
    # 0/1-trip loop instead of a branch around the dots (sm70 codegen).
    for _ in range(((end > 0) & (first < N_OUT)).to(tl.int32)):
        h = tl.arange(0, HB)
        d = tl.arange(0, 32)
        head = (h < H)[:, None]
        row = (t * H + h)[:, None] * 128
        qv0 = tl.load(q + row + d[None, :], head, 0.0)
        qv1 = tl.load(q + row + 32 + d[None, :], head, 0.0)
        qv2 = tl.load(q + row + 64 + d[None, :], head, 0.0)
        qv3 = tl.load(q + row + 96 + d[None, :], head, 0.0)
        wv = tl.load(w + t * H + h, h < H, 0.0).to(tl.float32)
        stop = tl.minimum(first + BLOCKS * BN, N_OUT)
        for j0 in range(first, stop, BN):
            j = j0 + tl.arange(0, BN)
            blk = tl.load(cand + t.to(tl.int64) * CAND_STRIDE + j // CB, j < N_OUT, -1).to(tl.int64)
            n = blk * CB + j % CB
            inside = (blk >= 0) & (n < end)
            page = tl.load(
                block_table + r * TABLE_STRIDE + n // ROWS, inside, -1
            ).to(tl.int64)
            inside &= page >= 0
            base = cache + page * PAGE + OFFSET + (n % ROWS).to(tl.int64) * 68
            b16 = tl.arange(0, 16)
            acc = tl.zeros((HB, BN), tl.float32)
            for g in tl.static_range(4):
                qv = qv0 if g == 0 else (qv1 if g == 1 else (qv2 if g == 2 else qv3))
                byte = tl.load(
                    base[:, None] + g * 16 + b16[None, :], inside[:, None], 0
                ).to(tl.int32)
                lo = _e2m1_bits(byte & 7) | (((byte >> 3) & 1).to(tl.uint32) << 31)
                hi = _e2m1_bits((byte >> 4) & 7) | (((byte >> 7) & 1).to(tl.uint32) << 31)
                pair = tl.join(
                    lo.to(tl.float32, bitcast=True), hi.to(tl.float32, bitcast=True)
                )
                kv = tl.reshape(pair, (BN, 32)).to(tl.float16)  # unscaled E2M1, exact
                piece = fma_dot(qv, tl.trans(kv))
                exponent = tl.load(base + 64 + g, inside, 127).to(tl.int32)
                acc += piece * (exponent << 23).to(tl.float32, bitcast=True)[None, :]
            score = tl.maximum(acc.to(tl.float16).to(tl.float32), 0.0)
            score = (score * wv[:, None]).to(tl.float16).to(tl.float32)
            total = tl.sum(tl.where(head, score, 0.0), 0)
            # FP16 out: the overlay's FP16 store; FP32 out (TP8): the unrounded head sum (see mu_packed).
            tl.store(out + t.to(tl.int64) * N_OUT + j, total.to(out.dtype.element_ty), inside)


def candidate_scores(q, weights, cache, block_table, positions, valid_tokens, token_req, cand, *, layout, layer, block, out):
    """Rank-local reindex scores [T, cand.shape[1] * block] FP16, or FP32 unrounded head sums (dead slots unset)."""
    pk._check(cache, block_table, layout.page_bytes)
    ratio, _rows, _c, i_offset = layout.regions[layer]
    tokens, heads = q.shape[0], q.shape[1]
    width = cand.shape[1] * block
    if (
        q.dtype != torch.float16
        or q.shape[2] != 128
        or weights.shape != (tokens, heads)
        or out.shape[0] < tokens
        or out.shape[1] != width
        or out.dtype not in (torch.float16, torch.float32)
        or out.stride(1) != 1
        or cand.dtype != torch.int32
        or cand.stride(1) != 1
        or heads > 64
    ):
        raise ValueError(f"{LOG}: reindex score shapes")
    bn = 64
    if tokens:
        _rx_scores[(tokens, triton.cdiv(width, bn * pk.DECODE_BLOCKS))](
            q.contiguous(),
            weights.contiguous(),
            cache,
            block_table,
            positions,
            valid_tokens,
            token_req,
            cand,
            out,
            out.stride(0),
            block_table.stride(0),
            cand.stride(0),
            PAGE=layout.page_bytes,
            BLOCK_SIZE=layout.block_size,
            RATIO=ratio,
            OFFSET=i_offset,
            H=heads,
            HB=max(16, triton.next_power_of_2(heads)),
            BN=bn,
            BLOCKS=pk.DECODE_BLOCKS,
            CB=block,
            num_warps=8,  # as 87's indexer (sm_70: 128 registers)
        )
    return out


# ------------------------------------------------------------- decode helpers
def visible(positions, valid, token_req, ratio):
    """Visible index rows per token (0 for padding / invalid)."""
    n = torch.div(positions.to(torch.int64) + 1, ratio, rounding_mode="floor")
    return torch.where((valid != 0) & (token_req >= 0), n, torch.zeros_like(n))


def decode_candidates(scores, n, *, blocks, block, topk_fn):
    """Upstream select_candidate_blocks on all-reduced decode scores.

    scores [T, W] FP16 (columns >= n[t] unset); n [T] visible rows. Returns
    [T, blocks] int32 ascending block ids, -1 padded. Block score = max of the
    block's scores (exact in FP16); the block of the newest row is pinned to
    +inf; unreachable blocks are past the length handed to the top-k, so the
    unset columns are never read (except inside the pinned block)."""
    tokens, width = scores.shape
    if width % block:
        scores = torch.nn.functional.pad(scores, (0, block - width % block), value=float("-inf"))
    bscores = scores.view(tokens, -1, block).amax(-1)
    last = torch.div(n - 1, block, rounding_mode="floor").clamp_min(0)
    pin = torch.full((tokens, 1), float("inf"), dtype=bscores.dtype, device=bscores.device)
    bscores.scatter_(1, last[:, None], pin)
    reachable = torch.div(n + block - 1, block, rounding_mode="floor")
    out = torch.empty(tokens, blocks, dtype=torch.int32, device=scores.device)
    # topk_fn reads (positions + 1) // 1 entries: positions = reachable - 1.
    topk_fn(out, bscores.contiguous(), reachable - 1, n > 0, 1, blocks)
    return out


def candidate_lengths(cand, n, block):
    """Live prefix of the compact candidate slots per token (pinned block last)."""
    count = (cand >= 0).sum(1).to(torch.int64)
    last = torch.div(n - 1, block, rounding_mode="floor")
    tail = n - last * block  # rows of the newest (pinned) block: 1..block
    return torch.where(count > 0, (count - 1) * block + tail, torch.zeros_like(count))


def compact_to_positions(picked, cand, block):
    """Compact slot indices (ascending, -1 padded) -> positions (ascending, -1)."""
    j = picked.to(torch.int64)
    safe = j.clamp_min(0)
    blk = cand.to(torch.int64).gather(1, torch.div(safe, block, rounding_mode="floor"))
    position = blk * block + safe % block
    return torch.where(j >= 0, position, torch.full_like(position, -1)).to(torch.int32)


# ------------------------------------------------------------ prefill selector
def prefill_select(
    q, weights, keys, positions, valid_tokens, ratio, all_reduce, *, kind, blocks,
    block, cand_in=None, cand_out=None, top_k=TOP_K, fused_topk=None, tile_bytes=None,
):
    """attention_ops.select_compressed_positions + upstream's candidate step.

    kind 'full' (unchanged), 'source' (also writes cand_out [T, blocks]: the
    upstream select_candidate_blocks per query, ascending ids, -1 padded) or
    'reindex' (scores outside cand_in's blocks are -inf before the top-k)."""
    if fused_topk is None:
        fused_topk = ops._FUSED_TOPK
    if tile_bytes is None:
        tile_bytes = ops._INDEX_TILE_BYTES
    if (
        q.dtype != torch.float16
        or q.shape[2] != 128
        or weights.shape != q.shape[:2]
        or weights.dtype != torch.float16
        or keys.shape[1] != 128
    ):
        raise ValueError(f"{LOG}: prefill indexer requires FP16 queries, keys and weights")
    width = min(top_k, keys.shape[0])
    result = torch.empty(q.shape[0], width, device=q.device, dtype=torch.int32)
    column = torch.arange(keys.shape[0], device=q.device)[None]
    budget = tile_bytes // max(1, q.shape[1] * keys.shape[0] * 2)
    tile = max(32, min(1024, budget // 32 * 32))
    if keys.shape[0] > 524288:  # VLLM_LOCAL_DS41_CTX_1M=1 in production
        # LOCAL-PATCH 267 (ds41_ctx_1m tile): above 512K keys the 32-row floor makes the
        # [tile, heads, N] FP16 scores 512 MiB at 1M keys; keep the tile at the 512K size
        # (32 rows x 524288 keys): 16 rows at 1M (rows are independent: same selections).
        tile = max(8, min(tile, 32 * 524288 // keys.shape[0] // 8 * 8))
    nblocks = (keys.shape[0] + block - 1) // block
    for first in range(0, q.shape[0], tile):
        last = min(first + tile, q.shape[0])
        heads = q.shape[1]
        scores = torch.mm(q[first:last].reshape(-1, 128), keys.T).view(last - first, heads, -1)
        scores.relu_().mul_(weights[first:last, :, None])
        scores = all_reduce(scores.sum(1))
        length = ((positions[first:last] + 1) // ratio)[:, None]
        if kind == "source":
            masked = scores.masked_fill(column >= length, float("-inf"))
            masked = torch.nn.functional.pad(masked, (0, nblocks * block - masked.shape[1]), value=float("-inf"))
            bscores = masked.view(last - first, nblocks, block).amax(-1)
            newest = (length - 1) // block
            bscores = bscores.masked_fill(torch.arange(nblocks, device=q.device)[None] == newest, float("inf"))
            take = min(blocks, nblocks)
            # Upstream topk over block scores; ties -> lowest block (stable sort),
            # the rule of the decode path's index_topk.
            top = bscores.sort(dim=-1, descending=True, stable=True)
            values, indices = top.values[:, :take], top.indices[:, :take]
            ids = torch.where(values > float("-inf"), indices, nblocks + 1)
            ids = ids.sort(dim=-1).values
            ids = torch.where(ids <= nblocks, ids, -1).to(torch.int32)
            cand_out[first:last].fill_(-1)
            cand_out[first:last, :take] = ids
        elif kind == "reindex":
            ids = cand_in[first:last].to(torch.int64)
            keep = torch.zeros(last - first, nblocks + 1, dtype=torch.bool, device=q.device)
            keep.scatter_(1, torch.where(ids >= 0, ids, nblocks), True)
            keep = keep[:, :nblocks].repeat_interleave(block, dim=1)[:, : keys.shape[0]]
            scores = scores.masked_fill(~keep, float("-inf"))
        if fused_topk and width == top_k and scores.dtype == torch.float16:
            index_topk(
                result[first:last], scores.contiguous(), positions[first:last],
                valid_tokens[first:last], ratio, top_k,
            )
            continue
        scores.masked_fill_((column >= length) | ~valid_tokens[first:last, None], float("-inf"))
        indices = scores.topk(width, dim=-1, sorted=False).indices.sort(dim=-1).values
        indices = torch.where(indices < length, indices, -1)
        indices = torch.where(valid_tokens[first:last, None], indices, -1)
        result[first:last].copy_(indices)
    return result


# ------------------------------------------------------------------ selector
def select(
    kind, query, weights, positions, step, cache, block_table, *, layout, source, ratio, width, blocks,
    block, all_reduce, candidates=None, score_dtype=torch.float16,
):
    """The overlay's select() without its vLLM glue: ([T, 512] int32 positions, -1 padded; candidates).

    kind = role(layer, candidate source). query [T, H, 128] FP16 is the indexer query after RoPE and the FP4
    (group 32) QDQ; weights [T, H] FP16 is weights_proj(x) * (1 / 64); width = max_model_len // ratio;
    all_reduce is the TP sum. 'source' returns its new candidates [T, blocks] int32 (ascending block ids,
    -1 padded); a 'reindex' layer must be given the same forward's source candidates ([T, blocks]), which it
    returns unchanged; 'full' returns None. The caller owns the hand-off (the overlay kept it on the shared
    attention state with a step-identity check). score_dtype=torch.float32 (TP8): decode scores stay
    unrounded through all_reduce and are rounded to FP16 once (the FP16 default is the overlay's arithmetic)."""
    tokens = query.shape[0]
    result = torch.full((tokens, TOP_K), -1, dtype=torch.int32, device=query.device)
    if kind == "source":
        cand = torch.full((tokens, blocks), -1, dtype=torch.int32, device=query.device)
    elif kind == "reindex":
        cand = candidates
        if cand is None or cand.shape[0] != tokens:
            raise RuntimeError(
                f"{LOG}: reindex layer has no candidates from its source in this forward "
                "(PP split between them?)"
            )
    else:
        cand = None
    topk_fn = _seg_topk
    decode_rows = tokens - sum(s1 - s0 for _, s0, s1, *_ in step.prefill)
    if decode_rows > 0:
        rows = step.decode_index
        pick = (lambda t: t) if rows is None else (lambda t: t.index_select(0, rows))
        q_d, w_d, pos_d = pick(query), pick(weights), pick(positions)
        valid_d, req_d = pick(step.valid), pick(step.token_req)
        n_d = visible(pos_d, valid_d, req_d, ratio)
        if kind == "reindex":
            cand_d = pick(cand).contiguous()
            scores = torch.empty(q_d.shape[0], blocks * block, dtype=score_dtype, device=query.device)
            candidate_scores(
                q_d, w_d, cache, block_table, pos_d, valid_d, req_d, cand_d,
                layout=layout, layer=source, block=block, out=scores,
            )
            scores = all_reduce(scores).to(torch.float16)
            lengths = candidate_lengths(cand_d, n_d, block)
            picked = torch.empty(q_d.shape[0], TOP_K, dtype=torch.int32, device=query.device)
            index_topk(picked, scores, lengths - 1, valid_d, 1, TOP_K)
            picked = compact_to_positions(picked, cand_d, block)
        else:
            scores = torch.empty(q_d.shape[0], width, dtype=score_dtype, device=query.device)
            pk.index_scores(
                q_d, w_d, cache, block_table, pos_d, valid_d, req_d,
                layout=layout, layer=source, out=scores,
            )
            scores = all_reduce(scores).to(torch.float16)
            picked = torch.empty(q_d.shape[0], TOP_K, dtype=torch.int32, device=query.device)
            topk_fn(picked, scores, pos_d, valid_d, ratio, TOP_K)
            if kind == "source":
                found = decode_candidates(scores, n_d, blocks=blocks, block=block, topk_fn=topk_fn)
                if rows is None:
                    cand.copy_(found)
                else:
                    cand.index_copy_(0, rows, found)
        if rows is None:
            result.copy_(picked)
        else:
            result.index_copy_(0, rows, picked)
    for r, s0, s1, *_ in step.prefill:
        end = int(positions[s1 - 1].item() + 1) // ratio
        keys = pk.unpack_request_index(
            cache, block_table[r], end, max(end, TOP_K), layout=layout, layer=source
        )
        result[s0:s1] = prefill_select(
            query[s0:s1], weights[s0:s1], keys, positions[s0:s1], step.valid[s0:s1],
            ratio, all_reduce, kind=kind, blocks=blocks, block=block,
            cand_in=None if cand is None else cand[s0:s1],
            cand_out=None if cand is None else cand[s0:s1],
        )
    return result, cand

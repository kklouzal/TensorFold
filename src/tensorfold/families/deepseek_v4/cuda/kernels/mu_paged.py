# SPDX-License-Identifier: Apache-2.0
# Provenance: friend overlay patch 85 (SPDX Apache-2.0), snapshot lp-snap-c1es8-numa-admit2-imgtext-share2-fix5-ram4-stepbud-ram3-memfix-ctx1m-ppovl-opt2-pfx-1-eng-pfxfix4,
#   patches/ds41mu_files/local_mu_paged.py (sha256 3ac959e66df2226bde2736e987e91daab6ceffff9a27f6ada12d5c951a742712).
# TensorFold changes: imports through the package's triton_utils; kernels and wrappers verbatim.
# - tl.dot -> triton_utils.fma_dot(_acc): the FP32 FMA dot his Triton 3.6 runs on SM70 (Triton 3.2's MMAv1
#   results cannot be reduced on SM70; see triton_utils).
# - @triton.jit -> triton_utils.jit: the same JITFunction; launches add the SM70 Triton 3.2 profile options
#   (ptx_version=83, needed with CUDA 12.9 ptxas). Kernel arithmetic unchanged.
# - Lint-only source edits: range(0, n) -> range(n) for the 0/1-trip loops (identical Triton loop).
# - Triton 3.2: no rewrite was needed.
"""LOCAL-PATCH 85 (ds41_mu) paged module.

Multi-request versions of the V4.1 FP16 paged-latent kernels in
vllm/models/deepseek_v41/attention_ops.py (4d57473d). Upstream validates
`block_table.shape[0] == 1` and every kernel reads row 0 of the block table:
one sequence. Here each token t reads the block-table row of its own request
(token_req[t], -1 = padding, never read or written), and keys are bounded by
the token's own position, so a token can only see its own request's pages.

Kernels (each is the upstream kernel plus the per-token row, nothing else):
  * mu_insert_latents   <- _insert            (window / compressed / index rows)
  * mu_gather_attention <- _gather_attention  (window + top-k compressed keys)
  * mu_index_scores     <- the decode indexer score of select_compressed_positions,
    over paged FP16 index keys, per token (bounded by (pos + 1) // ratio):
    scores[t, n] = fp16(sum_h fp16(fp16(relu(fp16(q[t,h] . k[n]))) * w[t,h]))
    (rank-local; the caller all-reduces, then index_topk, exactly like
    upstream). The dot accumulates in FP32 like cuBLAS; the summation order
    differs from cuBLAS, so this kernel is not bit-identical to the
    single-request reference, but it is row-wise: a token's scores never depend
    on the other rows of the batch (batch invariance), which the equivalence
    harness relies on.

Pure Triton; no vLLM import.
"""

import torch

from .triton_utils import fma_dot_acc, jit, tl, triton


def _check_table(block_table, cache):
    if (
        block_table.ndim != 2
        or block_table.dtype != torch.int32
        or block_table.stride(1) != 1
        or cache.ndim != 3
        or cache.stride(2) != 1
        or cache.stride(1) != cache.shape[2]
        or cache.dtype != torch.float16
        or block_table.device != cache.device
    ):
        raise ValueError("LOCAL-PATCH 85: paged V4.1 caches need int32 tables, FP16 rows")


@jit
def _mu_insert(
    values,
    positions,
    valid_tokens,
    token_req,
    block_table,
    cache,
    TABLE_STRIDE,
    STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RATIO: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    r = tl.load(token_req + token).to(tl.int64)
    position = tl.load(positions + token).to(tl.int64)
    valid = (
        (tl.load(valid_tokens + token) != 0)
        & (r >= 0)
        & (position >= 0)
        & ((position + 1) % RATIO == 0)
    )
    page = tl.load(
        block_table + r * TABLE_STRIDE + position // BLOCK_SIZE, valid, -1
    ).to(tl.int64)
    valid &= page >= 0
    col = tl.arange(0, BLOCK)
    data = tl.load(values + token * D + col, valid & (col < D), 0)
    row = (position % BLOCK_SIZE) // RATIO
    tl.store(cache + page * STRIDE + row * D + col, data, valid & (col < D))


def mu_insert_latents(
    cache, values, positions, valid_tokens, token_req, block_table, *, block_size, ratio=1
):
    _check_table(block_table, cache)
    if (
        values.shape != (positions.numel(), cache.shape[-1])
        or values.dtype != torch.float16
        or cache.shape[1] != block_size // ratio
        or token_req.shape[0] < positions.numel()
    ):
        raise ValueError("LOCAL-PATCH 85: paged latent insertion shapes")
    if values.shape[0]:
        _mu_insert[(values.shape[0],)](
            values.contiguous(),
            positions,
            valid_tokens,
            token_req,
            block_table,
            cache,
            block_table.stride(0),
            STRIDE=cache.stride(0),
            BLOCK_SIZE=block_size,
            RATIO=ratio,
            D=values.shape[1],
            BLOCK=triton.next_power_of_2(values.shape[1]),
        )


@jit
def _mu_gather_attention(
    window_cache,
    window_table,
    compressed_cache,
    compressed_table,
    positions,
    valid_tokens,
    token_req,
    topk,
    keys,
    unused,
    WINDOW_TABLE_STRIDE,
    COMPRESSED_TABLE_STRIDE,
    WINDOW_STRIDE: tl.constexpr,
    COMPRESSED_STRIDE: tl.constexpr,
    WINDOW_BLOCK: tl.constexpr,
    COMPRESSED_BLOCK: tl.constexpr,
    RATIO: tl.constexpr,
    WINDOW: tl.constexpr,
    TOPK: tl.constexpr,
):
    token, key = tl.program_id(0), tl.program_id(1)
    r = tl.load(token_req + token).to(tl.int64)
    position = tl.load(positions + token).to(tl.int64)
    valid = (tl.load(valid_tokens + token) != 0) & (r >= 0)
    col = tl.arange(0, 512)
    if key < WINDOW:
        at = position - WINDOW + 1 + key
        valid &= at >= 0
        page = tl.load(
            window_table + r * WINDOW_TABLE_STRIDE + at // WINDOW_BLOCK, valid, -1
        ).to(tl.int64)
        valid &= page >= 0
        row = at % WINDOW_BLOCK
        value = tl.load(window_cache + page * WINDOW_STRIDE + row * 512 + col, valid, 0)
    else:
        index = tl.load(topk + token * TOPK + key - WINDOW).to(tl.int64)
        valid &= (index >= 0) & (index < (position + 1) // RATIO)
        page = tl.load(
            compressed_table
            + r * COMPRESSED_TABLE_STRIDE
            + index // (COMPRESSED_BLOCK // RATIO),
            valid,
            -1,
        ).to(tl.int64)
        valid &= page >= 0
        row = index % (COMPRESSED_BLOCK // RATIO)
        value = tl.load(
            compressed_cache + page * COMPRESSED_STRIDE + row * 512 + col, valid, 0
        )
    tl.store(keys + (token * (WINDOW + TOPK) + key) * 512 + col, value)
    tl.store(unused + token * (WINDOW + TOPK) + key, ~valid)


def mu_gather_attention(
    keys,
    unused,
    positions,
    valid_tokens,
    token_req,
    window_cache,
    window_table,
    *,
    window_block,
    compressed_cache=None,
    compressed_table=None,
    compressed_block=16,
    ratio=1,
    topk=None,
    window=128,
):
    """keys [n, window + k, 512] FP16 and unused [n, window + k] for n tokens."""
    _check_table(window_table, window_cache)
    compressed = compressed_cache is not None
    if compressed:
        _check_table(compressed_table, compressed_cache)
    count = topk.shape[1] if compressed else 0
    n = positions.numel()
    if keys.shape != (n, window + count, 512) or unused.shape != (n, window + count):
        raise ValueError("LOCAL-PATCH 85: gather output shapes")
    if n:
        _mu_gather_attention[(n, window + count)](
            window_cache,
            window_table,
            compressed_cache if compressed else window_cache,
            compressed_table if compressed else window_table,
            positions,
            valid_tokens,
            token_req,
            topk if compressed else positions,
            keys,
            unused,
            window_table.stride(0),
            (compressed_table if compressed else window_table).stride(0),
            WINDOW_STRIDE=window_cache.stride(0),
            COMPRESSED_STRIDE=compressed_cache.stride(0) if compressed else 0,
            WINDOW_BLOCK=window_block,
            COMPRESSED_BLOCK=compressed_block,
            RATIO=ratio,
            WINDOW=window,
            TOPK=count,
        )


@jit(do_not_specialize=["N_OUT"])
def _mu_index_scores(
    q,
    w,
    cache,
    block_table,
    positions,
    valid_tokens,
    token_req,
    out,
    N_OUT,
    TABLE_STRIDE,
    STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RATIO: tl.constexpr,
    H: tl.constexpr,
    HB: tl.constexpr,
    BN: tl.constexpr,
):
    t, nb = tl.program_id(0), tl.program_id(1)
    r = tl.load(token_req + t).to(tl.int64)
    position = tl.load(positions + t).to(tl.int64)
    live = (tl.load(valid_tokens + t) != 0) & (r >= 0)
    end = tl.where(live, (position + 1) // RATIO, 0)
    n0 = nb * BN
    # 0/1-trip loop instead of a branch around the dot (upstream's sm70
    # codegen rule in attention_ops._index_scores_decode).
    for _ in range((n0 < end).to(tl.int32)):
        h = tl.arange(0, HB)
        d = tl.arange(0, 32)
        n = n0 + tl.arange(0, BN)
        inside = n < end
        rows_per_block: tl.constexpr = BLOCK_SIZE // RATIO
        page = tl.load(
            block_table + r * TABLE_STRIDE + n // rows_per_block, inside, -1
        ).to(tl.int64)
        inside &= page >= 0
        base = page[:, None] * STRIDE + (n % rows_per_block)[:, None] * 128
        # FP32 dot over K = 128 as four chained K = 32 dots (same sequential
        # FMA order on SM70; a single K = 128 FMA dot spills ~3.4 KB of stack).
        acc = tl.zeros((HB, BN), tl.float32)
        for k in tl.static_range(4):
            qv = tl.load(
                q + (t * H + h)[:, None] * 128 + k * 32 + d[None, :], (h < H)[:, None], 0.0
            )
            kv = tl.load(cache + base + k * 32 + d[None, :], inside[:, None], 0.0)
            acc = fma_dot_acc(qv, tl.trans(kv), acc)
        score = acc.to(tl.float16).to(tl.float32)
        score = tl.maximum(score, 0.0)
        wv = tl.load(w + t * H + h, h < H, 0.0).to(tl.float32)
        score = (score * wv[:, None]).to(tl.float16).to(tl.float32)
        score = tl.where((h < H)[:, None], score, 0.0)
        total = tl.sum(score, 0)
        tl.store(out + t.to(tl.int64) * N_OUT + n, total.to(tl.float16), n < end)


def mu_index_scores(q, weights, cache, block_table, positions, valid_tokens, token_req, *, block_size, ratio, out):
    """Rank-local decode indexer scores [T, N_OUT] FP16 (rows past each length unset)."""
    _check_table(block_table, cache)
    tokens, heads = q.shape[0], q.shape[1]
    if (
        q.dtype != torch.float16
        or q.shape[2] != 128
        or weights.shape != (tokens, heads)
        or weights.dtype != torch.float16
        or cache.shape[2] != 128
        or out.dtype != torch.float16
        or out.shape[0] < tokens
        or out.stride(1) != 1
    ):
        raise ValueError("LOCAL-PATCH 85: paged indexer shapes")
    bn = 64
    if tokens:
        _mu_index_scores[(tokens, triton.cdiv(out.shape[1], bn))](
            q.contiguous(),
            weights.contiguous(),
            cache,
            block_table,
            positions,
            valid_tokens,
            token_req,
            out,
            out.stride(0),
            block_table.stride(0),
            STRIDE=cache.stride(0),
            BLOCK_SIZE=block_size,
            RATIO=ratio,
            H=heads,
            HB=max(16, triton.next_power_of_2(heads)),
            BN=bn,
            num_warps=8,  # sm_70: 144 registers, no spill (4 warps spill)
        )
    return out


def index_scores_reference(q, weights, cache, block_table, positions, valid_tokens, token_req, *, block_size, ratio, width):
    """torch: per token, its own request's keys, upstream's rounding boundaries."""
    out = torch.zeros(q.shape[0], width, dtype=torch.float16)
    per_block = block_size // ratio
    for t in range(q.shape[0]):
        r = int(token_req[t])
        if r < 0 or not bool(valid_tokens[t]):
            continue
        end = (int(positions[t]) + 1) // ratio
        if not end:
            continue
        rows = torch.arange(end)
        pages = block_table[r, rows // per_block].long()
        keys = cache[pages, rows % per_block].float()
        s = (q[t].float() @ keys.T).half().float().relu()
        s = (s * weights[t].float()[:, None]).half().float()
        out[t, :end] = s.sum(0).half()
    return out

# SPDX-License-Identifier: Apache-2.0
# Provenance: friend overlay patch 87 (SPDX Apache-2.0), snapshot lp-snap-c1es8-numa-admit2-imgtext-share2-fix5-ram4-stepbud-ram3-memfix-ctx1m-ppovl-opt2-pfx-1-eng-pfxfix4,
#   patches/ds41mu_files/local_mu_packed.py (sha256 73d7e820dbf31321f26a5c4676b2d69353a3d5ec9fa744234526db81d638efe0).
# TensorFold changes (kernel arithmetic and wrapper validation verbatim):
# - tl.dot -> triton_utils.fma_dot(_acc): the FP32 FMA dot his Triton 3.6 runs on SM70 (Triton 3.2's MMAv1
#   results cannot be reduced on SM70; see triton_utils).
# - @triton.jit -> triton_utils.jit: the same JITFunction; launches add the SM70 Triton 3.2 profile options
#   (ptx_version=83, needed with CUDA 12.9 ptxas). Kernel arithmetic unchanged.
# - vllm.models.deepseek_v41.{attention_ops,kv_pack} -> the vendored package modules; triton via triton_utils.
# - Flags at production values (runtime.ds41.production.json): MU_PACKED_BLOCKS unset -> DECODE_BLOCKS = 4;
#   MU_INDEX=paged (loop A/B branch not vendored); MU_TOPK_SEG=1 (mu_topk.topk); ATTN_FUSED=1.
# - Not vendored: admission_errors / contract_context / _logger (vLLM config policy), _vllm_cache_classes,
#   MuWindow, MuPacked (vLLM-bound owners; TensorFold writes its own); the MU_IDX (patch 111) and MU_IDX8
#   (patch 172) CUDA-extension hooks in index_scores (production enables both; the Triton kernel here is the
#   path they fall back to); LONG_PREFILL_MEM's keys_buffer allocator (plain torch.empty, same values);
#   DEC_FUSE (263) / PF_INDEX (180) / PF_ATTN (181) / MU_ATTN (138) hooks of _select/_attend/attention_forward.
# - _select and _attend are restated vLLM-free as select_positions / attend (caller passes the indexer query
#   and weights, block table, req_slot and all-reduce); attention_forward (vLLM layer glue) is not vendored.
# - TP8 (TensorFold, 2026-11-20): index_scores also accepts an FP32 `out` and then stores the head-sum total
#   without the FP16 rounding, so the cross-rank sum is FP32 and the caller rounds once; an FP16 `out` keeps
#   the overlay's store bit for bit (store dtype = out element type). Index heads 4 (TP8) already fit: the
#   head rows are padded to HB = max(16, next_pow2(H)) masked rows for tl.dot (overlay code, unchanged).
# - The overlay's private copy of attention_ops._e2m1_bits (identical body) is imported instead.
# - Lint-only source edits: range(0, n) -> range(n) for the 0/1-trip loops (identical Triton loop).
# - Triton 3.2: no rewrite was needed.
# - select_positions (ds41-short, K36): a prefill request's live length comes from the host (step.prefill's 4th
#   field, its last position) instead of positions[s1 - 1].item(): no host sync per indexer layer; same value.
"""LOCAL-PATCH 87 (ds41_mu_packed) paged FP4 latent rows.

Multi-request DeepSeek-V4.1 (patch 85) with the compressed latents and indexer
keys stored as the packed rows of kv_pack.py (288 B per E2M1 compressed
row or 272 B in the RotorQuant mode; 68 B per index row: 64 B nibbles +
4 power-of-two exponents), paged through vLLM's KV
manager, on the GPU. Patch 85 keeps the fork's FP16 paged latents (1 KiB per
row after the page alignment) and caps contexts at 16K; this lifts the cap to
the contract's 524288 and stores 534 B per token on PP stage 0 (sources 2, 8,
14, ratio 2) and 356 B on stage 1 (source 20, ratio 1).

Layout (one KV layer per stage, MULTIUSER_DESIGN.md section 15):
  * ONE vLLM KV layer per process / PP stage, "<first local layer>.attn.
    local_mu_packed": MLAAttentionSpec(block_size=B, head_size=bytes per
    token, dtype=uint8). A page (B tokens) holds, for every local KV source in
    layer order, B/r compressed rows then B/r index rows. With no sliding
    window layer left, vLLM builds exactly one KV group (no SWA groups, one
    block table, no page padding).
  * Window latents (128 keys per token) live in per-slot rings
    [S + 1, RING, 512] FP16 or [S + 1, RING, 272] uint8 per layer, like the fork's single-sequence
    window rings: keys of this forward come from its own latents, older
    ones from the ring, and the last RING positions of each request are
    written back after attention. A slot is owned by one request (patch 85's
    SlotTable), and a fresh request starts at position 0, so stale ring rows
    are never read (at < 0 is masked).

Numerics (bit-exactness, tests/ds41mu_interpret_check.py section "packed"):
  * E2M1 inserts call kv_pack.pack_c_group / pack_i_group on the pre-QDQ
    values, exactly like the single-sequence packed store. RotorQuant latents
    arrive encoded and are copied by byte to their paged and window stores;
  * attention keys are kv_pack.unpack_c_row of those bytes (the single-
    sequence packed gather), equal to the FP16 QDQ cache values bit for bit;
  * prefill requests (> DECODE_MAX tokens) unpack their live index keys once
    (attention_ops._unpack_i_block) and run the upstream
    select_compressed_positions: the single-sequence packed prefill path;
  * decode rows: one batched launch over each token's own pages with the
    arithmetic of attention_ops._decode_block (unscaled E2M1 dots per
    32-group, FP32 power-of-two scaling, FP16 boundaries), 16 head rows per
    token (row-wise, batch invariant; the upstream launch pads tokens x heads
    to 64 rows, so the head-sum association may differ in the last bit);
    VLLM_LOCAL_DS41_MU_INDEX=loop instead gathers each request's rows and
    calls the upstream select_packed_positions (bit-identical A/B).

Imports the vendored kv_pack and attention_ops (TensorFold package) at module load; no vLLM import.
"""

import torch

from .attention_ops import (ROW_SPLIT_MIN, _attend_fused, _e2m1_bits, _unpack_i_block, select_compressed_positions,
                            select_rows_split)
from .kv_pack import (
    C_DIM,
    C_GROUP,
    C_ROW,
    I_DIM,
    I_GROUP,
    I_ROW,
    ROTOR_ROW,
    pack_c_group,
    pack_i_group,
    unpack_c_row,
    unpack_rotor_row,
)
from .mu_topk import topk as _topk
from .triton_utils import fma_dot, jit, tl, triton

LOG = "LOCAL-PATCH 87"
# Window ring rows per slot and layer: a token at p reads p-127..p, of which
# p-127..start-1 come from the ring (<= 127 rows), the rest from this forward.
RING = 128
WINDOW = 128
# VLLM_LOCAL_DS41_MU_PACKED_BLOCKS at its production value (unset: "4").
DECODE_BLOCKS = 4


class PackedLayout:
    """Byte layout of one KV page (block_size tokens) for a stage's sources.

    sources: {layer: ratio}. Regions in ascending layer order:
    [B/r compressed rows][B/r index rows of 68 B] per source."""

    def __init__(self, sources, block_size, *, rotorquant=False):
        if not sources:
            raise ValueError(f"{LOG}: a packed V4.1 page needs at least one KV source")
        self.block_size = block_size
        self.rotorquant = rotorquant
        self.compressed_row_bytes = ROTOR_ROW if rotorquant else C_ROW
        self.regions = {}
        offset = 0
        for layer in sorted(sources):
            ratio = sources[layer]
            if block_size % ratio:
                raise ValueError(f"{LOG}: block {block_size} splits a ratio-{ratio} group")
            rows = block_size // ratio
            self.regions[layer] = (ratio, rows, offset, offset + rows * self.compressed_row_bytes)
            offset += rows * (self.compressed_row_bytes + I_ROW)
        if offset % block_size:
            raise ValueError(f"{LOG}: page bytes {offset} not a whole bytes-per-token")
        self.page_bytes = offset
        self.bytes_per_token = offset // block_size


# ------------------------------------------------------------------ kernels
@jit
def _pk_insert(
    values,
    positions,
    valid_tokens,
    token_req,
    block_table,
    cache,
    TABLE_STRIDE,
    PAGE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RATIO: tl.constexpr,
    OFFSET: tl.constexpr,
    DIM: tl.constexpr,
    ROW: tl.constexpr,
    COMPRESSED: tl.constexpr,
    ROTORQUANT: tl.constexpr,
):
    """attention_ops._insert_packed with the token's own block-table row."""
    token, group = tl.program_id(0), tl.program_id(1)
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
    if valid & (page >= 0):
        row = cache + page * PAGE + OFFSET + ((position % BLOCK_SIZE) // RATIO) * ROW
        if ROTORQUANT:
            col = tl.arange(0, 512)
            data = tl.load(values + token.to(tl.int64) * ROW + col, col < ROW, 0)
            tl.store(row + col, data, col < ROW)
        elif COMPRESSED:
            pack_c_group(values + token.to(tl.int64) * DIM, row, group)
        else:
            pack_i_group(values + token.to(tl.int64) * DIM, row, group)


def _check(cache, block_table, page_bytes, ndim=2):
    if (
        cache.dtype != torch.uint8
        or cache.ndim < 2
        or not cache.is_contiguous()
        or cache.shape[1:].numel() != page_bytes
        or block_table.ndim != ndim
        or block_table.dtype != torch.int32
        or block_table.stride(-1) != 1
        or block_table.device != cache.device
    ):
        raise ValueError(f"{LOG}: packed V4.1 pages need uint8 pages and int32 tables")


def insert_rows(
    cache, values, positions, valid_tokens, token_req, block_table, *,
    layout, layer, compressed,
):
    """RoPE-applied FP16 [T, 512] (E2M1) / [T, 128] (index), or packed uint8 RotorQuant rows."""
    _check(cache, block_table, layout.page_bytes)
    ratio, _rows, c_offset, i_offset = layout.regions[layer]
    rotorquant = compressed and layout.rotorquant
    size = layout.compressed_row_bytes if compressed else I_ROW
    if rotorquant:
        dim, groups, dtype = size, 1, torch.uint8
    elif compressed:
        dim, groups, dtype = C_DIM, C_DIM // C_GROUP, torch.float16
    else:
        dim, groups, dtype = I_DIM, I_DIM // I_GROUP, torch.float16
    if (
        values.dtype != dtype
        or values.shape != (positions.numel(), dim)
        or token_req.numel() < positions.numel()
    ):
        raise ValueError(f"{LOG}: packed insertion shapes")
    if values.shape[0]:
        _pk_insert[(values.shape[0], groups)](
            values.contiguous(),
            positions,
            valid_tokens,
            token_req,
            block_table,
            cache,
            block_table.stride(0),
            PAGE=layout.page_bytes,
            BLOCK_SIZE=layout.block_size,
            RATIO=ratio,
            OFFSET=c_offset if compressed else i_offset,
            DIM=dim,
            ROW=size,
            COMPRESSED=compressed,
            ROTORQUANT=rotorquant,
        )


@jit(do_not_specialize=["N_OUT", "TABLE_STRIDE"])
def _pk_index_scores(
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
    PAGE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RATIO: tl.constexpr,
    OFFSET: tl.constexpr,
    H: tl.constexpr,
    HB: tl.constexpr,
    BN: tl.constexpr,
    BLOCKS: tl.constexpr,
):
    """Decode indexer scores of token t over its own request's packed keys.

    Per key block: attention_ops._decode_block's arithmetic (unscaled E2M1
    FP16 keys, one dot per 32-wide group, FP32 power-of-two scaling, FP16
    round, ReLU, x weight, FP16 round, head sum in FP32, FP16 store)."""
    t, part = tl.program_id(0), tl.program_id(1)
    r = tl.load(token_req + t).to(tl.int64)
    position = tl.load(positions + t).to(tl.int64)
    live = (tl.load(valid_tokens + t) != 0) & (r >= 0)
    end = tl.where(live, (position + 1) // RATIO, 0)
    first = part * BLOCKS * BN
    ROWS: tl.constexpr = BLOCK_SIZE // RATIO
    # 0/1-trip loops instead of branches around the dots (sm70 codegen).
    for _ in range((first < end).to(tl.int32)):
        h = tl.arange(0, HB)
        d = tl.arange(0, 32)
        head = (h < H)[:, None]
        row = (t * H + h)[:, None] * 128
        qv0 = tl.load(q + row + d[None, :], head, 0.0)
        qv1 = tl.load(q + row + 32 + d[None, :], head, 0.0)
        qv2 = tl.load(q + row + 64 + d[None, :], head, 0.0)
        qv3 = tl.load(q + row + 96 + d[None, :], head, 0.0)
        wv = tl.load(w + t * H + h, h < H, 0.0).to(tl.float32)
        stop = tl.minimum(first + BLOCKS * BN, end)
        for n0 in range(first, stop, BN):
            n = n0 + tl.arange(0, BN)
            inside = n < end
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
            # FP16 out: the overlay's FP16 store. FP32 out (TP8): the unrounded head sum, rounded once by
            # the caller after the cross-rank FP32 sum.
            tl.store(out + t.to(tl.int64) * N_OUT + n, total.to(out.dtype.element_ty), n < end)


def index_scores(
    q, weights, cache, block_table, positions, valid_tokens, token_req, *,
    layout, layer, out,
):
    """Rank-local decode indexer scores [T, N_OUT] FP16, or FP32 unrounded head sums (unset past each length)."""
    _check(cache, block_table, layout.page_bytes)
    ratio, _rows, _c, i_offset = layout.regions[layer]
    tokens, heads = q.shape[0], q.shape[1]
    if (
        q.dtype != torch.float16
        or q.shape[2] != 128
        or weights.shape != (tokens, heads)
        or weights.dtype != torch.float16
        or out.dtype not in (torch.float16, torch.float32)
        or out.shape[0] < tokens
        or out.stride(1) != 1
        or heads > 64
    ):
        raise ValueError(f"{LOG}: packed indexer shapes")
    bn = 64
    if tokens:
        _pk_index_scores[(tokens, triton.cdiv(out.shape[1], bn * DECODE_BLOCKS))](
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
            PAGE=layout.page_bytes,
            BLOCK_SIZE=layout.block_size,
            RATIO=ratio,
            OFFSET=i_offset,
            H=heads,
            HB=max(16, triton.next_power_of_2(heads)),
            BN=bn,
            BLOCKS=DECODE_BLOCKS,
            num_warps=8,  # sm_70: 128 registers (4 warps: 255)
        )
    return out


@jit(do_not_specialize=["ROWS_OUT", "LENGTH"])
def _pk_unpack_index(
    cache,
    table_row,
    out,
    ROWS_OUT,
    LENGTH,
    PAGE: tl.constexpr,
    ROWS: tl.constexpr,
    OFFSET: tl.constexpr,
    BN: tl.constexpr,
):
    """FP16 index keys [ROWS_OUT, 128] of one request (rows >= LENGTH zero),
    attention_ops._unpack_index_keys over pages."""
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    live = n < LENGTH
    page = tl.load(table_row + n // ROWS, live, -1).to(tl.int64)
    live &= page >= 0
    d = tl.arange(0, 128)
    base = cache + page * PAGE + OFFSET + (n % ROWS).to(tl.int64) * 68
    value = _unpack_i_block(base[:, None], live, d)
    tl.store(
        out + n.to(tl.int64)[:, None] * 128 + d[None, :], value, (n < ROWS_OUT)[:, None]
    )


def unpack_request_index(cache, block_table_row, length, rows_out, *, layout, layer):
    _check(cache, block_table_row, layout.page_bytes, ndim=1)
    _ratio, rows, _c, i_offset = layout.regions[layer]
    out = torch.empty(rows_out, 128, dtype=torch.float16, device=cache.device)
    _pk_unpack_index[(triton.cdiv(rows_out, 64),)](
        cache,
        block_table_row,
        out,
        rows_out,
        length,
        PAGE=layout.page_bytes,
        ROWS=rows,
        OFFSET=i_offset,
        BN=64,
    )
    return out


@jit(do_not_specialize=["LENGTH"])
def _pk_gather_rows(
    cache, table_row, out, LENGTH, PAGE: tl.constexpr, ROWS: tl.constexpr,
    OFFSET: tl.constexpr, ROW: tl.constexpr, BLOCK: tl.constexpr,
):
    """Position-addressed packed rows [LENGTH, ROW] of one request (byte copy)."""
    n = tl.program_id(0)
    page = tl.load(table_row + n // ROWS).to(tl.int64)
    b = tl.arange(0, BLOCK)
    inside = (b < ROW) & (page >= 0) & (n < LENGTH)
    data = tl.load(cache + page * PAGE + OFFSET + (n % ROWS) * ROW + b, inside, 0)
    tl.store(out + n.to(tl.int64) * ROW + b, data, b < ROW)


def gather_request_rows(cache, block_table_row, length, *, layout, layer, compressed):
    """The single-sequence store's position-addressed rows of one request."""
    _check(cache, block_table_row, layout.page_bytes, ndim=1)
    _ratio, rows, c_offset, i_offset = layout.regions[layer]
    size = layout.compressed_row_bytes if compressed else I_ROW
    out = torch.zeros(max(length, 1), size, dtype=torch.uint8, device=cache.device)
    if length:
        _pk_gather_rows[(length,)](
            cache,
            block_table_row,
            out,
            length,
            PAGE=layout.page_bytes,
            ROWS=rows,
            OFFSET=c_offset if compressed else i_offset,
            ROW=size,
            BLOCK=triton.next_power_of_2(size),
        )
    return out


@jit(do_not_specialize=["FIRST", "TABLE_STRIDE"])
def _pk_gather_attention(
    ring,
    values,
    qsl,
    req_slot,
    cache,
    block_table,
    positions,
    valid_tokens,
    token_req,
    topk,
    centroids,
    keys,
    unused,
    FIRST,
    TABLE_STRIDE,
    PAGE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    RATIO: tl.constexpr,
    OFFSET: tl.constexpr,
    S: tl.constexpr,
    RING_ROWS: tl.constexpr,
    WINDOW_KEYS: tl.constexpr,
    TOPK: tl.constexpr,
    WINDOW_ONLY: tl.constexpr,
    ROTORQUANT: tl.constexpr,
    LATENT_ROW: tl.constexpr,
):
    """Window keys from this forward / the slot ring (attention_ops.
    _gather_attention_ring per request), compressed keys unpacked from the
    token's own pages (attention_ops._gather_attention_packed)."""
    local, key = tl.program_id(0), tl.program_id(1)
    token = local + FIRST
    r = tl.load(token_req + token).to(tl.int64)
    position = tl.load(positions + token).to(tl.int64)
    valid = (tl.load(valid_tokens + token) != 0) & (r >= 0)
    col = tl.arange(0, 512)
    ONLY_WINDOW: tl.constexpr = WINDOW_ONLY or TOPK == 0
    # The window specialization has no centroid table. Let Triton discard the
    # compressed branch at the frontend before it lowers pointer arithmetic.
    if (True if ONLY_WINDOW else key < WINDOW_KEYS):
        rr = tl.where(r >= 0, r, 0)
        s0 = tl.load(qsl + rr, valid, 0).to(tl.int64)
        start = tl.load(positions + s0, valid, 0).to(tl.int64)
        slot = tl.load(req_slot + rr, valid, S).to(tl.int64)
        slot = tl.where((slot >= 0) & (slot <= S), slot, S)
        at = position - WINDOW_KEYS + 1 + key
        valid &= at >= 0
        inside = at >= start
        if ROTORQUANT:
            new = values + (s0 + at - start) * LATENT_ROW
            old = ring + (slot * RING_ROWS + at % RING_ROWS) * LATENT_ROW
            value = unpack_rotor_row(tl.where(inside, new, old), valid, centroids, not ONLY_WINDOW)
        else:
            new = tl.load(values + (s0 + at - start) * 512 + col, valid & inside, 0)
            old = tl.load(
                ring + (slot * RING_ROWS + at % RING_ROWS) * 512 + col, valid & ~inside, 0
            )
            value = tl.where(inside, new, old)
    else:
        index = tl.load(topk + token * TOPK + key - WINDOW_KEYS).to(tl.int64)
        valid &= (index >= 0) & (index < (position + 1) // RATIO)
        ROWS: tl.constexpr = BLOCK_SIZE // RATIO
        page = tl.load(
            block_table + r * TABLE_STRIDE + index // ROWS, valid, -1
        ).to(tl.int64)
        valid &= page >= 0
        if ROTORQUANT:
            row = cache + page * PAGE + OFFSET + (index % ROWS) * LATENT_ROW
            value = unpack_rotor_row(row, valid, centroids, True)
        else:
            # unpack_c_row loads unconditionally: keep the address in bounds.
            page = tl.where(valid, page, 0)
            safe = tl.where(valid, index, 0)
            row = cache + page * PAGE + OFFSET + (safe % ROWS) * LATENT_ROW
            value = tl.where(valid, unpack_c_row(row), 0.0).to(tl.float16)
    tl.store(keys + (local * (WINDOW_KEYS + TOPK) + key) * 512 + col, value)
    tl.store(unused + local * (WINDOW_KEYS + TOPK) + key, ~valid)


@jit
def _pk_ring_insert(
    values, positions, valid_tokens, token_req, qsl, req_slot, ring,
    S: tl.constexpr, RING_ROWS: tl.constexpr, ROW: tl.constexpr,
):
    """After attention: each request's last RING_ROWS positions into its slot."""
    t = tl.program_id(0)
    r = tl.load(token_req + t).to(tl.int64)
    rr = tl.where(r >= 0, r, 0)
    live = (tl.load(valid_tokens + t) != 0) & (r >= 0)
    s1 = tl.load(qsl + rr + 1, live, 0).to(tl.int64)
    last = tl.load(positions + s1 - 1, live, 0).to(tl.int64)
    position = tl.load(positions + t).to(tl.int64)
    slot = tl.load(req_slot + rr, live, S).to(tl.int64)
    ok = live & (position >= 0) & (slot >= 0) & (slot < S) & (position > last - RING_ROWS)
    col = tl.arange(0, 512)
    valid = ok & (col < ROW)
    data = tl.load(values + t.to(tl.int64) * ROW + col, valid, 0)
    tl.store(ring + (slot * RING_ROWS + position % RING_ROWS) * ROW + col, data, valid)


def _window_row_width(ring, values, tokens):
    width, dtype = (ROTOR_ROW, torch.uint8) if ring.dtype == torch.uint8 else (C_DIM, torch.float16)
    if (ring.ndim != 3 or ring.shape[0] < 1 or ring.shape[1:] != (RING, width)
            or not ring.is_contiguous() or ring.dtype != dtype or values.dtype != dtype
            or values.shape != (tokens, width) or values.device != ring.device):
        raise ValueError(f"{LOG}: window ring shapes or dtype")
    return width


def insert_ring(ring, values, positions, valid_tokens, token_req, qsl, req_slot):
    width = _window_row_width(ring, values, positions.numel())
    if values.shape[0]:
        _pk_ring_insert[(values.shape[0],)](
            values.contiguous(), positions, valid_tokens, token_req, qsl, req_slot,
            ring, S=ring.shape[0] - 1, RING_ROWS=RING, ROW=width,
        )


def gather_attention(
    keys, unused, first, last, positions, valid_tokens, token_req, qsl, req_slot,
    ring, values, *, cache=None, block_table=None, layout=None, layer=None,
    topk=None, rotor_centroids=None, window_only=False,
):
    """keys [n, 128 + k, 512] FP16 / unused [n, 128 + k] for tokens [first, last).

    topk, positions, valid_tokens, token_req and values are the whole batch's.
    RotorQuant keys are decoded to FP16 in their rotated basis. Gathering a new compressed selection
    uses the cache's centroid table. With window_only, the caller owns a valid compressed suffix
    for this exact selection. Both output strides still include the suffix."""
    compressed = cache is not None
    count = topk.shape[1] if compressed else 0
    n = last - first
    rotorquant = _window_row_width(ring, values, positions.numel()) == ROTOR_ROW
    use_table = rotorquant and count > 0 and not window_only
    if use_table and (not isinstance(rotor_centroids, torch.Tensor) or rotor_centroids.shape != (16,)
                      or rotor_centroids.dtype != torch.float32 or rotor_centroids.device != ring.device
                      or not rotor_centroids.is_contiguous()):
        raise ValueError(f"{LOG}: H64 compressed attention requires the device centroid table")
    if keys.shape != (n, WINDOW + count, 512) or unused.shape != (n, WINDOW + count):
        raise ValueError(f"{LOG}: gather output shapes")
    if compressed:
        _check(cache, block_table, layout.page_bytes)
        if layout.rotorquant != rotorquant:
            raise ValueError(f"{LOG}: compressed pages and window require a common attention basis")
        ratio, _rows, c_offset, _i = layout.regions[layer]
        latent_row = layout.compressed_row_bytes
    else:
        ratio, c_offset = 1, 0
        latent_row = ROTOR_ROW if rotorquant else C_ROW
    if n:
        _pk_gather_attention[(n, WINDOW if window_only else WINDOW + count)](
            ring,
            values.contiguous(),
            qsl,
            req_slot,
            cache if compressed else ring,
            block_table if compressed else qsl.view(1, -1),
            positions,
            valid_tokens,
            token_req,
            topk if compressed else positions,
            rotor_centroids if use_table else None,
            keys,
            unused,
            first,
            block_table.stride(0) if compressed else 0,
            PAGE=layout.page_bytes if compressed else 0,
            BLOCK_SIZE=layout.block_size if compressed else 1,
            RATIO=ratio,
            OFFSET=c_offset,
            S=ring.shape[0] - 1,
            RING_ROWS=RING,
            WINDOW_KEYS=WINDOW,
            TOPK=count,
            WINDOW_ONLY=window_only,
            ROTORQUANT=rotorquant,
            LATENT_ROW=latent_row,
        )


# ---------------------------------------------------------------- attention
def select_positions(
    query, weights, positions, step, cache, block_table, *, layout, source, ratio, width,
    all_reduce, score_dtype=torch.float16, row_split=None,
):
    """Patch 87's _select without its vLLM glue: top-512 compressed positions [T, 512] int32 (-1 padded).

    query [T, H, 128] FP16 is the indexer query after RoPE and the FP4 (group 32) QDQ; weights [T, H] FP16 is
    weights_proj(x) * (1 / 64); width = max_model_len // ratio; all_reduce is the TP sum (identity at TP1).
    Decode rows (step.decode_index, or every row when None) share one index_scores launch, then the
    segmented top-k (VLLM_LOCAL_DS41_MU_TOPK_SEG=1); each step.prefill request (r, s0, s1, end) unpacks its live
    index keys once and runs select_compressed_positions. VLLM_LOCAL_DS41_MU_INDEX=paged (the production
    mode), so the overlay's "loop" A/B branch is not vendored. score_dtype=torch.float32 (TP8) keeps the
    rank-local decode scores unrounded through all_reduce and rounds the sum to FP16 once; the FP16 default
    is the overlay's arithmetic."""
    tokens = query.shape[0]
    result = torch.full((tokens, 512), -1, dtype=torch.int32, device=query.device)
    decode_rows = tokens - sum(s1 - s0 for _, s0, s1, _ in step.prefill)
    if decode_rows > 0:
        rows = step.decode_index
        pick = (lambda t: t) if rows is None else (lambda t: t.index_select(0, rows))
        q_d, w_d, pos_d = pick(query), pick(weights), pick(positions)
        valid_d, req_d = pick(step.valid), pick(step.token_req)
        scores = torch.empty(q_d.shape[0], width, dtype=score_dtype, device=query.device)
        index_scores(
            q_d, w_d, cache, block_table, pos_d, valid_d, req_d,
            layout=layout, layer=source, out=scores,
        )
        scores = all_reduce(scores).to(torch.float16)
        if rows is None:
            _topk(result, scores, positions, step.valid, ratio, 512)
        else:
            picked = torch.empty(q_d.shape[0], 512, dtype=torch.int32, device=query.device)
            _topk(picked, scores, pos_d, valid_d, ratio, 512)
            result.index_copy_(0, rows, picked)
    for r, s0, s1, last in step.prefill:
        # Single-sequence packed prefill: unpack the live keys once, then the
        # reference cuBLAS path (select_packed_positions' prefill branch). ``last`` is the request's last position
        # (positions[s1 - 1]) from the host (TensorFold: no device read, no host sync per indexer layer).
        end = (last + 1) // ratio
        keys = unpack_request_index(
            cache, block_table[r], end, max(end, 512), layout=layout, layer=source
        )
        if row_split is not None and s1 - s0 >= ROW_SPLIT_MIN:
            rank, world, gather = row_split        # patch 180: rows split over the ranks, all heads per row
            result[s0:s1] = select_rows_split(
                query[s0:s1], weights[s0:s1], keys, positions[s0:s1], step.valid[s0:s1], ratio,
                first_position=last - (s1 - s0 - 1), rank=rank, world=world, all_gather=gather,
            )
            continue
        result[s0:s1] = select_compressed_positions(
            query[s0:s1], weights[s0:s1], keys, positions[s0:s1], step.valid[s0:s1],
            ratio, all_reduce,
        )
    return result


def gather_buffers(tokens, width, device):
    return (torch.empty(tokens, width, 512, dtype=torch.float16, device=device),
            torch.empty(tokens, width, dtype=torch.bool, device=device))


def attend(
    query, positions, step, req_slot, window_ring, window, sink, *, cache=None, block_table=None,
    layout=None, source=None, topk=None, tokens_per_pass=64, rotor_centroids=None, key_buffers=None,
    reuse_compressed=False,
):
    """Patch 87's _attend without its vLLM glue: sparse attention output [T, heads, 512] FP16.

    Per pass of tokens_per_pass tokens: gather_attention (window keys from `window` / the slot ring,
    compressed keys unpacked from the token's own pages), then attention_ops._attend_fused (scale 512**-0.5,
    attn_sink; VLLM_V41_ATTN_FUSED=1, the production default)."""
    result = torch.empty_like(query)
    count = topk.shape[1] if cache is not None else 0
    width = WINDOW + count
    if key_buffers is not None and (query.shape[0] > tokens_per_pass or not count or not layout.rotorquant):
        raise ValueError(f"{LOG}: reusable gather buffers require one RotorQuant compressed pass")
    if reuse_compressed and key_buffers is None:
        raise ValueError(f"{LOG}: compressed reuse requires initialized gather buffers")
    for first in range(0, query.shape[0], tokens_per_pass):
        last = min(first + tokens_per_pass, query.shape[0])
        n, heads = last - first, query.shape[1]
        keys, unused = gather_buffers(n, width, query.device) if key_buffers is None else key_buffers
        gather_attention(
            keys, unused, first, last, positions, step.valid, step.token_req,
            step.qsl, req_slot, window_ring, window, cache=cache,
            block_table=block_table, layout=layout, layer=source, topk=topk, rotor_centroids=rotor_centroids,
            window_only=reuse_compressed,
        )
        scores = torch.empty(n, heads, width, dtype=torch.float32, device=query.device)
        _attend_fused(query[first:last], keys, unused, 512**-0.5, sink, result[first:last], scores)
    return result

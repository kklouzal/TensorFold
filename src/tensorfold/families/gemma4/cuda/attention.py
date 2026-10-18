"""Gemma 4 attention over persistent caches, row-count invariant by construction.

Arithmetic contract. Every query row r at absolute position p attends keys
partitioned into absolute chunks of ``CHUNK`` keys. Inside a chunk, keys are
visited in absolute ``tile``-key tiles with an online softmax that starts from
(-inf, 0, 0); a tile without a visible key for that row is an exact no-op.
Chunk partials (m, l, o) fold in ascending absolute chunk order through
``_fold_step``; an empty partial is an exact no-op. The number of rows in a
forward, the rows sharing a program and the launch geometry (parallel chunk
programs plus a merge, or one program folding sequentially) never change any
row's operations, so serial decode, DFlash verification and every prefill
chunking produce identical bits. That scope is one GPU architecture, Triton
build and these launch options; no cross-architecture bit identity is implied.

Caches. Sliding layers keep a ring of ``ring`` slots (slot = position % ring)
holding RoPE'd normalized K and normalized V; the ring retains the window plus
the widest forward, so a forward writes its own rows before attending.
Full-attention layers share K and V (attention_k_eq_v): only the raw
projection and its FP32 reciprocal RMS are stored, and K = RoPE(bf16(x*inv*w))
and V = bf16(x*inv) are rebuilt per tile with the same per-element operation
order and roundings as the unfused norm and RoPE kernels. The prep kernels sum
the RMS reduction in two halves, so inv may differ from the unfused kernel's
in its last FP32 bit; every row and forward shares this one arithmetic.

Positions are read from device memory, so a captured graph replays any step.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .state import GLOBAL_CHUNK, QUERY_TILE, SEQUENTIAL_ROWS, SLIDING_CHUNK

SLIDING_TILE = 64
GLOBAL_TILE = 32
MERGE_COLUMNS = 64
GLOBAL_DIM = 512         # full-attention head width this layout implements
ROTATED = 64             # rotated column pairs of a full-attention head (partial_rotary_factor 0.25)


@triton.jit
def _scaled(normalized, weight, ROUND: tl.constexpr):
    """Learned RMS scale with one BF16 result; MLX rounds the normalized value first."""
    if ROUND:
        normalized = normalized.to(tl.bfloat16).to(tl.float32)
    return (normalized * weight.to(tl.float32)).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _prep(QR, KR, VR, QW, KW, FREQ, POS, QO, KC, VC, RING,
          H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr, ROUND: tl.constexpr):
    """Sliding layers: head norms, RoPE and ring publication for one (row, head).

    Programs j < H rotate query heads into QO, the next HK rotate keys and the
    last HK normalize values (no learned scale) into slot position % RING.
    """
    row = tl.program_id(0)
    j = tl.program_id(1)
    half: tl.constexpr = D // 2
    d = tl.arange(0, half)
    pos = tl.load(POS) + row
    if j < H:
        src = QR + (row * H + j) * D
        weight = QW
    elif j < H + HK:
        src = KR + (row * HK + j - H) * D
        weight = KW
    else:
        src = VR + (row * HK + j - H - HK) * D
        weight = KW
    x1 = tl.load(src + d).to(tl.float32)
    x2 = tl.load(src + half + d).to(tl.float32)
    inv = tl.rsqrt((tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)) / D + EPS)
    if j >= H + HK:
        dst = VC + ((pos % RING) * HK + j - H - HK) * D
        tl.store(dst + d, (x1 * inv).to(tl.bfloat16))
        tl.store(dst + half + d, (x2 * inv).to(tl.bfloat16))
    else:
        # Learned norm: FP32 normalization (BF16-rounded under MLX numerics)
        # then weight, one BF16 rounding, then RoPE on the rounded values.
        y1 = _scaled(x1 * inv, tl.load(weight + d), ROUND)
        y2 = _scaled(x2 * inv, tl.load(weight + half + d), ROUND)
        angle = pos.to(tl.float32) * tl.load(FREQ + d)
        c, s = tl.cos(angle), tl.sin(angle)
        a = (y1 * c - y2 * s).to(tl.bfloat16)
        b = (y1 * s + y2 * c).to(tl.bfloat16)
        if j < H:
            dst = QO + (row * H + j) * D
        else:
            dst = KC + ((pos % RING) * HK + j - H) * D
        tl.store(dst + d, a)
        tl.store(dst + half + d, b)


@triton.jit
def _prep_shared(QR, KR, QW, FREQ, POS, QO, KC, IC, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                 EPS: tl.constexpr, ROUND: tl.constexpr):
    """Full attention: rotate query heads; store raw key rows and FP32 reciprocal RMS."""
    row = tl.program_id(0)
    j = tl.program_id(1)
    half: tl.constexpr = D // 2
    d = tl.arange(0, half)
    pos = tl.load(POS) + row
    if j < H:
        src = QR + (row * H + j) * D
    else:
        src = KR + (row * HK + j - H) * D
    x1 = tl.load(src + d).to(tl.float32)
    x2 = tl.load(src + half + d).to(tl.float32)
    inv = tl.rsqrt((tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0)) / D + EPS)
    if j < H:
        y1 = _scaled(x1 * inv, tl.load(QW + d), ROUND)
        y2 = _scaled(x2 * inv, tl.load(QW + half + d), ROUND)
        angle = pos.to(tl.float32) * tl.load(FREQ + d)
        c, s = tl.cos(angle), tl.sin(angle)
        dst = QO + (row * H + j) * D
        tl.store(dst + d, (y1 * c - y2 * s).to(tl.bfloat16))
        tl.store(dst + half + d, (y1 * s + y2 * c).to(tl.bfloat16))
    else:
        dst = KC + (pos * HK + j - H) * D
        tl.store(dst + d, x1.to(tl.bfloat16))
        tl.store(dst + half + d, x2.to(tl.bfloat16))
        tl.store(IC + pos * HK + j - H, inv)


@triton.jit
def _online(scores, valid, m, l, o, v):
    """One key tile of the online softmax; an all-masked row keeps its bits."""
    scores = tl.where(valid, scores, -float("inf"))
    tile_max = tl.max(scores, 1)
    active = tile_max != -float("inf")
    next_m = tl.where(active, tl.maximum(m, tile_max), m)
    alpha = tl.where(active, tl.where(m == -float("inf"), 0.0, tl.exp(m - next_m)), 1.0)
    prob = tl.where(valid & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(prob.to(tl.bfloat16), v)
    l = l * alpha + tl.sum(prob, 1)
    return next_m, l, o


@triton.jit
def _fold_step(m, l, o, cm, cl, co):
    """Fold one chunk partial; an empty partial (cl == 0) leaves every bit unchanged."""
    active = cl > 0.0
    next_m = tl.where(active, tl.maximum(m, cm), m)
    a = tl.where(m == -float("inf"), 0.0, tl.exp(m - next_m))
    b = tl.exp(cm - next_m)
    o = tl.where(active[:, None], o * a[:, None] + co * b[:, None], o)
    l = tl.where(active, l * a + cl * b, l)
    return next_m, l, o


@triton.jit
def _queries(t, hk, R, POS, BS, BE, G: tl.constexpr, WINDOW: tl.constexpr, BLOCKS: tl.constexpr):
    """A program's 16 (row, head) pairs, their visibility bounds and the union key range.

    WINDOW 0 is unbounded (causal from position 0). BLOCKS loads per-row
    bidirectional block bounds [BS, BE); -1 marks a row outside any block.
    """
    rpt: tl.constexpr = 16 // G
    qi = tl.arange(0, 16)
    r = t * rpt + qi // G
    ok = r < R
    head = hk * G + qi % G
    pos = tl.load(POS) + r
    if BLOCKS:
        bs = tl.load(BS + r, mask=ok, other=-1)
        be = tl.load(BE + r, mask=ok, other=-1)
        last = tl.maximum(pos + 1, be)
    else:
        bs = tl.full((16,), -1, tl.int32)
        be = tl.full((16,), -1, tl.int32)
        last = pos + 1
    if WINDOW:
        earliest = tl.maximum(pos - WINDOW + 1, 0)
    else:
        earliest = tl.zeros((16,), tl.int32)
    lo = tl.min(tl.where(ok, earliest, 2147483647), 0)
    hi = tl.max(tl.where(ok, last, 0), 0)
    return r, ok, head, pos, bs, be, earliest, last, lo, hi


@triton.jit
def _sliding_chunk(q, KC, VC, RING, chunk, hk, pos, ok, bs, be, earliest, last, lo, hi,
                   HK: tl.constexpr, D: tl.constexpr, CH: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr):
    dd = tl.arange(0, D)
    m = tl.full((16,), -float("inf"), tl.float32)
    l = tl.zeros((16,), tl.float32)
    o = tl.zeros((16, D), tl.float32)
    begin = chunk * CH
    for at in range(begin, tl.minimum(begin + CH, hi), BK):
        key = at + tl.arange(0, BK)
        held = (key >= lo) & (key < hi)
        slot = key % RING
        k = tl.load(KC + (slot[:, None] * HK + hk) * D + dd[None, :], mask=held[:, None], other=0.0)
        v = tl.load(VC + (slot[:, None] * HK + hk) * D + dd[None, :], mask=held[:, None], other=0.0)
        visible = (key[None, :] <= pos[:, None]) | ((key[None, :] >= bs[:, None]) & (key[None, :] < be[:, None]))
        valid = visible & (key[None, :] >= earliest[:, None]) & (key[None, :] < last[:, None]) & ok[:, None]
        scores = tl.dot(q, tl.trans(k)).to(tl.float32)
        if SCALE != 1.0:
            scores = scores * SCALE
        m, l, o = _online(scores, valid, m, l, o, v)
    return m, l, o


@triton.jit
def _sliding(Q, KC, VC, POS, BS, BE, PO, PM, PL, R, RING, SPLITS,
             H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, WINDOW: tl.constexpr,
             BLOCKS: tl.constexpr, CH: tl.constexpr, BK: tl.constexpr, SCALE: tl.constexpr):
    """Program (query tile, KV head, chunk slot): slot s is absolute chunk lo // CH + s."""
    t, hk, s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    r, ok, head, pos, bs, be, earliest, last, lo, hi = _queries(t, hk, R, POS, BS, BE, G, WINDOW, BLOCKS)
    dd = tl.arange(0, D)
    q = tl.load(Q + (r[:, None] * H + head[:, None]) * D + dd[None, :], mask=ok[:, None], other=0.0)
    m, l, o = _sliding_chunk(q, KC, VC, RING, lo // CH + s, hk, pos, ok, bs, be, earliest, last, lo, hi,
                             HK, D, CH, BK, SCALE)
    base = (r * H + head) * SPLITS + s
    tl.store(PO + base[:, None] * D + dd[None, :], o, mask=ok[:, None])
    tl.store(PM + base, m, mask=ok)
    tl.store(PL + base, l, mask=ok)


@triton.jit
def _global_columns(D: tl.constexpr, ROT: tl.constexpr, UW: tl.constexpr):
    """Rotated column pairs [0, ROT) and [D/2, D/2 + ROT), then the unrotated
    columns [ROT, D/2) and [D/2 + ROT, D) in order, as three parts of UW."""
    half: tl.constexpr = D // 2
    ra = tl.arange(0, ROT)
    uu = tl.arange(0, UW)
    c1 = ROT + UW + uu
    return ra, half + ra, ROT + uu, tl.where(c1 < half, c1, c1 + ROT), 2 * ROT + 2 * UW + uu


@triton.jit
def _normalized(row, cols, inv, held):
    """One column part of raw key rows times their reciprocal RMS (FP32)."""
    return tl.load(row + cols[None, :], mask=held[:, None], other=0.0).to(tl.float32) * inv


@triton.jit
def _key(row, cols, inv, held, KW, ROUND: tl.constexpr):
    """One column part of K before RoPE: bf16(x*inv*w), held in FP32."""
    return _scaled(_normalized(row, cols, inv, held), tl.load(KW + cols)[None, :], ROUND)


@triton.jit
def _global_chunk(qa, qb, q0, q1, q2, RAW, INV, KW, FREQ, chunk, hk, pos, ok, hi,
                  HK: tl.constexpr, D: tl.constexpr, ROT: tl.constexpr, UW: tl.constexpr,
                  CH: tl.constexpr, BK: tl.constexpr, ROUND: tl.constexpr):
    """One chunk's online softmax over raw K=V tiles: K = RoPE(bf16(x*inv*w)) and V = bf16(x*inv).

    Each column part is rebuilt where it is consumed (scores first, values after the softmax update, the
    second read served by the cache), so only one part's tile is live: holding all five parts spills.
    """
    ca, cb, c0, c1, c2 = _global_columns(D, ROT, UW)
    m = tl.full((16,), -float("inf"), tl.float32)
    l = tl.zeros((16,), tl.float32)
    oa = tl.zeros((16, ROT), tl.float32)
    ob = tl.zeros((16, ROT), tl.float32)
    o0 = tl.zeros((16, UW), tl.float32)
    o1 = tl.zeros((16, UW), tl.float32)
    o2 = tl.zeros((16, UW), tl.float32)
    begin = chunk * CH
    for at in range(begin, tl.minimum(begin + CH, hi), BK):
        key = at + tl.arange(0, BK)
        held = key < hi
        row = RAW + (key[:, None] * HK + hk) * D
        inv = tl.load(INV + key * HK + hk, mask=held, other=0.0)[:, None]
        ka = _key(row, ca, inv, held, KW, ROUND)
        kb = _key(row, cb, inv, held, KW, ROUND)
        angle = key.to(tl.float32)[:, None] * tl.load(FREQ + ca)[None, :]
        c, s = tl.cos(angle), tl.sin(angle)
        scores = tl.dot(qa, tl.trans((ka * c - kb * s).to(tl.bfloat16)))
        scores = tl.dot(qb, tl.trans((ka * s + kb * c).to(tl.bfloat16)), scores)
        scores = tl.dot(q0, tl.trans(_key(row, c0, inv, held, KW, ROUND).to(tl.bfloat16)), scores)
        scores = tl.dot(q1, tl.trans(_key(row, c1, inv, held, KW, ROUND).to(tl.bfloat16)), scores)
        scores = tl.dot(q2, tl.trans(_key(row, c2, inv, held, KW, ROUND).to(tl.bfloat16)), scores)
        valid = (key[None, :] <= pos[:, None]) & ok[:, None]
        scores = tl.where(valid, scores, -float("inf"))
        tile_max = tl.max(scores, 1)
        active = tile_max != -float("inf")
        next_m = tl.where(active, tl.maximum(m, tile_max), m)
        alpha = tl.where(active, tl.where(m == -float("inf"), 0.0, tl.exp(m - next_m)), 1.0)
        prob = tl.where(valid & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
        p16 = prob.to(tl.bfloat16)
        oa = oa * alpha[:, None] + tl.dot(p16, _normalized(row, ca, inv, held).to(tl.bfloat16))
        ob = ob * alpha[:, None] + tl.dot(p16, _normalized(row, cb, inv, held).to(tl.bfloat16))
        o0 = o0 * alpha[:, None] + tl.dot(p16, _normalized(row, c0, inv, held).to(tl.bfloat16))
        o1 = o1 * alpha[:, None] + tl.dot(p16, _normalized(row, c1, inv, held).to(tl.bfloat16))
        o2 = o2 * alpha[:, None] + tl.dot(p16, _normalized(row, c2, inv, held).to(tl.bfloat16))
        l = l * alpha + tl.sum(prob, 1)
        m = next_m
    return m, l, oa, ob, o0, o1, o2


@triton.jit
def _global(Q, RAW, INV, KW, FREQ, POS, PO, PM, PL, OUT, R, SPLITS,
            H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, ROT: tl.constexpr,
            UW: tl.constexpr, CH: tl.constexpr, BK: tl.constexpr, SEQUENTIAL: tl.constexpr,
            ROUND: tl.constexpr):
    """Parallel: program (query tile, KV head, chunk) writes one partial. Sequential:
    program (query tile, KV head) folds every chunk in order and writes the output."""
    t, hk = tl.program_id(0), tl.program_id(1)
    r, ok, head, pos, bs, be, earliest, last, lo, hi = _queries(t, hk, R, POS, POS, POS, G, 0, False)
    ca, cb, c0, c1, c2 = _global_columns(D, ROT, UW)
    qrow = Q + (r[:, None] * H + head[:, None]) * D
    qa = tl.load(qrow + ca[None, :], mask=ok[:, None], other=0.0)
    qb = tl.load(qrow + cb[None, :], mask=ok[:, None], other=0.0)
    q0 = tl.load(qrow + c0[None, :], mask=ok[:, None], other=0.0)
    q1 = tl.load(qrow + c1[None, :], mask=ok[:, None], other=0.0)
    q2 = tl.load(qrow + c2[None, :], mask=ok[:, None], other=0.0)
    if SEQUENTIAL:
        m = tl.full((16,), -float("inf"), tl.float32)
        l = tl.zeros((16,), tl.float32)
        oa = tl.zeros((16, ROT), tl.float32)
        ob = tl.zeros((16, ROT), tl.float32)
        o0 = tl.zeros((16, UW), tl.float32)
        o1 = tl.zeros((16, UW), tl.float32)
        o2 = tl.zeros((16, UW), tl.float32)
        for chunk in range(0, (hi + CH - 1) // CH):
            cm, cl, pa, pb, p0, p1, p2 = _global_chunk(qa, qb, q0, q1, q2, RAW, INV, KW, FREQ, chunk, hk, pos, ok,
                                                      hi, HK, D, ROT, UW, CH, BK, ROUND)
            m2, l2, oa = _fold_step(m, l, oa, cm, cl, pa)
            ob = _fold_step(m, l, ob, cm, cl, pb)[2]
            o0 = _fold_step(m, l, o0, cm, cl, p0)[2]
            o1 = _fold_step(m, l, o1, cm, cl, p1)[2]
            o2 = _fold_step(m, l, o2, cm, cl, p2)[2]
            m, l = m2, l2
        orow = OUT + (r[:, None] * H + head[:, None]) * D
        tl.store(orow + ca[None, :], (oa / l[:, None]).to(tl.bfloat16), mask=ok[:, None])
        tl.store(orow + cb[None, :], (ob / l[:, None]).to(tl.bfloat16), mask=ok[:, None])
        tl.store(orow + c0[None, :], (o0 / l[:, None]).to(tl.bfloat16), mask=ok[:, None])
        tl.store(orow + c1[None, :], (o1 / l[:, None]).to(tl.bfloat16), mask=ok[:, None])
        tl.store(orow + c2[None, :], (o2 / l[:, None]).to(tl.bfloat16), mask=ok[:, None])
    else:
        chunk = tl.program_id(2)
        if chunk * CH < hi:
            m, l, oa, ob, o0, o1, o2 = _global_chunk(qa, qb, q0, q1, q2, RAW, INV, KW, FREQ, chunk, hk, pos, ok,
                                                     hi, HK, D, ROT, UW, CH, BK, ROUND)
            base = (r * H + head) * SPLITS + chunk
            prow = PO + base[:, None] * D
            tl.store(prow + ca[None, :], oa, mask=ok[:, None])
            tl.store(prow + cb[None, :], ob, mask=ok[:, None])
            tl.store(prow + c0[None, :], o0, mask=ok[:, None])
            tl.store(prow + c1[None, :], o1, mask=ok[:, None])
            tl.store(prow + c2[None, :], o2, mask=ok[:, None])
            tl.store(PM + base, m, mask=ok)
            tl.store(PL + base, l, mask=ok)


@triton.jit
def _merge(PO, PM, PL, OUT, POS, SPLITS, H: tl.constexpr, D: tl.constexpr, DS: tl.constexpr,
           CH: tl.constexpr, GLOBAL: tl.constexpr):
    """Program (row, head, DS columns) folds its row's chunk partials in ascending order."""
    r, head = tl.program_id(0), tl.program_id(1)
    d = tl.program_id(2) * DS + tl.arange(0, DS)
    if GLOBAL:
        count = (tl.load(POS) + r) // CH + 1
    else:
        count = SPLITS
    m = tl.full((1,), -float("inf"), tl.float32)
    l = tl.zeros((1,), tl.float32)
    o = tl.zeros((1, DS), tl.float32)
    base = (r * H + head) * SPLITS
    for s in tl.range(0, count, num_stages=3):
        cm = tl.load(PM + base + s + tl.zeros((1,), tl.int32))
        cl = tl.load(PL + base + s + tl.zeros((1,), tl.int32))
        co = tl.load(PO + (base + s) * D + d)[None, :]
        m, l, o = _fold_step(m, l, o, cm, cl, co)
    tl.store(OUT + (r * H + head) * D + d[None, :], (o / l[:, None]).to(tl.bfloat16))


def check_geometry(config) -> None:
    """Admit the head geometry these kernels implement before any cache allocation."""
    for name, heads, dim in (("sliding", config.kv_heads, config.head_dim),
                             ("full", config.global_kv_heads, config.global_head_dim)):
        if config.heads % heads or QUERY_TILE % (config.heads // heads):
            raise ValueError(f"Gemma4 CUDA {name} attention needs a power-of-two query group of at most 16")
        if dim not in (64, 128, 256, 512):
            raise ValueError(f"Gemma4 CUDA {name} attention head width must be 64/128/256/512")
    if config.head_dim > 256:
        raise ValueError("Gemma4 CUDA sliding attention head width must be at most 256")
    if config.global_head_dim != GLOBAL_DIM or round(config.global_head_dim * config.global_rotary_fraction) != 2 * ROTATED:
        raise ValueError("Gemma4 CUDA shared K=V attention implements 512-wide heads with 128 rotated columns")
    # Kernel offsets are int32: stored key rows and parallel chunk partials must index below 2**31.
    chunks = -(-config.max_position // GLOBAL_CHUNK)
    if max(config.max_position * config.global_kv_heads, (SEQUENTIAL_ROWS - 1) * config.heads * chunks) * GLOBAL_DIM \
            >= 2**31:
        raise ValueError("Gemma4 CUDA attention offsets exceed int32 at the checkpoint's position limit")


# Four warps for both head widths: on sm_120/121 eight warps for the 512-wide full-attention tile were
# slower (dev-01 32k: 425 vs 320 us per layer) despite fewer register spills.
LAUNCH = dict(num_warps=4, num_stages=1, enable_fp_fusion=False)


def prep_sliding(q_raw, k_raw, v_raw, q_norm, k_norm, inverse, pos, keys, values, *, eps: float,
                 round_normalized: bool):
    """[R, H*D] query and [R, Hk*D] key/value projections -> rotated queries [R, H, D].

    Publishes the forward's keys and values into ring slots position % ring
    before attention reads them; the ring must retain window + R - 1 keys.
    """
    rows = q_raw.shape[0]
    ring, hk, d = keys.shape
    h = q_raw.shape[1] // d
    out = torch.empty((rows, h, d), dtype=torch.bfloat16, device=q_raw.device)
    _prep[(rows, h + 2 * hk)](q_raw, k_raw, v_raw, q_norm, k_norm, inverse, pos, out, keys, values, ring,
                              H=h, HK=hk, D=d, EPS=eps, ROUND=round_normalized, num_warps=4,
                              enable_fp_fusion=False)
    return out


def prep_full(q_raw, k_raw, q_norm, inverse, pos, raw, inv, *, eps: float, round_normalized: bool):
    """Rotated queries; raw K=V rows and their reciprocal RMS published at their positions."""
    rows = q_raw.shape[0]
    _, hk, d = raw.shape
    h = q_raw.shape[1] // d
    out = torch.empty((rows, h, d), dtype=torch.bfloat16, device=q_raw.device)
    _prep_shared[(rows, h + hk)](q_raw, k_raw, q_norm, inverse, pos, out, raw, inv, H=h, HK=hk, D=d, EPS=eps,
                                 ROUND=round_normalized, num_warps=4, enable_fp_fusion=False)
    return out


def sliding(q, keys, values, pos, block_starts, block_ends, *, window: int, slots: int, blocks: bool = True,
            scale: float = 1.0, chunk: int = SLIDING_CHUNK):
    """Ring attention: causal, windowed (window 0 = unbounded), optionally with
    bidirectional per-row blocks, over keys already published in the ring.

    ``slots`` must cover every absolute chunk a query tile's union key range
    spans (state.sliding_slots); an uncovered chunk would be silently omitted.
    """
    rows, h, d = q.shape
    ring, hk, _ = keys.shape
    g = h // hk
    tiles = -(-rows * g // QUERY_TILE)
    po = torch.empty((rows, h, slots, d), dtype=torch.float32, device=q.device)
    pm = torch.empty((rows, h, slots), dtype=torch.float32, device=q.device)
    pl = torch.empty_like(pm)
    _sliding[(tiles, hk, slots)](q, keys, values, pos, block_starts, block_ends, po, pm, pl, rows, ring, slots,
                                 H=h, HK=hk, D=d, G=g, WINDOW=window, BLOCKS=blocks, CH=chunk,
                                 BK=SLIDING_TILE, SCALE=scale, **LAUNCH)
    out = torch.empty_like(q)
    _merge[(rows, h, d // MERGE_COLUMNS)](po, pm, pl, out, pos, slots, H=h, D=d, DS=MERGE_COLUMNS,
                                          CH=chunk, GLOBAL=False, num_warps=1, enable_fp_fusion=False)
    return out


def full(q, raw, inv, k_norm, inverse, pos, *, chunks: int, round_normalized: bool, sequential: bool | None = None):
    """Causal full attention over shared raw K=V rows.

    ``chunks`` bounds the absolute chunks of the forward's last row (a graph
    passes the context's). Parallel and sequential launches compute the same bits.
    """
    rows, h, d = q.shape
    _, hk, _ = raw.shape
    g = h // hk
    tiles = -(-rows * g // QUERY_TILE)
    if sequential is None:
        sequential = rows >= SEQUENTIAL_ROWS
    out = torch.empty_like(q)
    common = dict(H=h, HK=hk, D=d, G=g, ROT=ROTATED, UW=(d - 2 * ROTATED) // 3, CH=GLOBAL_CHUNK, BK=GLOBAL_TILE,
                  ROUND=round_normalized, **LAUNCH)
    if sequential:
        _global[(tiles, hk, 1)](q, raw, inv, k_norm, inverse, pos, out, out, out, out, rows, 1, SEQUENTIAL=True,
                                **common)
        return out
    po = torch.empty((rows, h, chunks, d), dtype=torch.float32, device=q.device)
    pm = torch.empty((rows, h, chunks), dtype=torch.float32, device=q.device)
    pl = torch.empty_like(pm)
    _global[(tiles, hk, chunks)](q, raw, inv, k_norm, inverse, pos, po, pm, pl, out, rows, chunks,
                                 SEQUENTIAL=False, **common)
    _merge[(rows, h, d // MERGE_COLUMNS)](po, pm, pl, out, pos, chunks, H=h, D=d, DS=MERGE_COLUMNS,
                                          CH=GLOBAL_CHUNK, GLOBAL=True, num_warps=1, enable_fp_fusion=False)
    return out

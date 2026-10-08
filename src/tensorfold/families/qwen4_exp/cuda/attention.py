"""Attend each row over fixed absolute-position chunks, selecting QSA blocks by score with ties to lower block ids so window size never changes its bits."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .image_rows import rope_axis
from .kv_pair import load_side, resolve_pair, validate_storage

from .kvquant import dequant_group_4, dequant_group_8, h32
from .rotorquant_kernel import dequant_group_3 as rotor_dequant_3, dequant_group_4 as rotor_dequant_4
from .rotorquant_kernel import dequant_group_6 as rotor_dequant_6
from .rotorquant_kernel import dequant_group_7 as rotor_dequant_7, dequant_group_8 as rotor_dequant_8
from .rotorquant_kernel import rotate as rotor_rotate

CHUNK = 512
TILE = 64
SELECT_REGS = 32768  # _select holds a row's block scores in registers up to this many; past it they spill


@triton.jit
def _tile(q, k, v, m, l, o, valid, scale: tl.constexpr):
    scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
    scores = tl.where(valid[None, :], scores, float("-inf"))
    tile_m = tl.max(scores, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    l = l * alpha + tl.sum(p, 1)
    return next_m, l, o


@triton.jit
def _chunks(Q, KC, VC, KSC, VSC, POS0, PO, PM, PL, IDS, NKR, SPR,
            H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr,
            NCH: tl.constexpr, SCALE: tl.constexpr, IDW: tl.constexpr, QSA: tl.constexpr, BITS: tl.constexpr,
            CODEC: tl.constexpr = 0, K_BITS: tl.constexpr = -1, K_CODEC: tl.constexpr = 0,
            V_BITS: tl.constexpr = -1, V_CODEC: tl.constexpr = 0):
    r = tl.program_id(0)
    hk = tl.program_id(1)
    c = tl.program_id(2)
    n = tl.load(POS0) + r + 1
    sparse = False
    if QSA:
        sparse = tl.load(SPR + r) != 0
        n = tl.where(sparse, tl.load(NKR + r), n)
    if K_BITS >= 0:
        _chunk_pair(Q, KC, VC, KSC, VSC, n, sparse, r, hk, c, PO, PM, PL, IDS,
                    H, HK, D, G, CH, NCH, SCALE, IDW, QSA, K_BITS, K_CODEC, V_BITS, V_CODEC)
    else:
        _chunk(Q, KC, VC, KSC, VSC, n, sparse, r, hk, c, PO, PM, PL, IDS,
               H, HK, D, G, CH, NCH, SCALE, IDW, QSA, BITS, CODEC=CODEC)


@triton.jit
def _chunk(Q, KC, VC, KSC, VSC, n, sparse, r, hk, c, PO, PM, PL, IDS,
           H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr,
           NCH: tl.constexpr, SCALE: tl.constexpr, IDW: tl.constexpr, QSA: tl.constexpr, BITS: tl.constexpr,
           CODEC: tl.constexpr = 0):
    """Row r's keys in chunk c of its ``n`` (a sparse row's through IDS): the chunk's partial o, m and l."""

    start = c * CH
    if start < n:                       # chunks past a row's keys write nothing: the merge never reads them
        gg = tl.arange(0, 16)
        d = tl.arange(0, D)
        q = tl.load(Q + (r * H + hk * G + gg[:, None]) * D + d[None, :], mask=gg[:, None] < G, other=0.0)
        m = tl.full((16,), float("-inf"), tl.float32)
        l = tl.zeros((16,), tl.float32)
        o = tl.zeros((16, D), tl.float32)
        tiles = tl.minimum(n - start, CH)
        for t in range(0, tl.cdiv(tiles, 64)):
            ki = start + t * 64 + tl.arange(0, 64)
            valid = ki < n
            if QSA:
                if sparse:
                    ki = tl.load(IDS + r * IDW + ki, mask=valid, other=0)
            off = (ki[:, None].to(tl.int64) * HK + hk) * D + d[None, :]
            if CODEC:
                gs = tl.arange(0, D // 128)
                sgc = (ki[:, None].to(tl.int64) * HK + hk) * (D // 128) + gs[None, :]
                sc = tl.load(KSC + sgc, mask=valid[:, None], other=0.0)
                sv = tl.load(VSC + sgc, mask=valid[:, None], other=0.0)
                if BITS == 3:
                    lo = tl.arange(0, D // 4)
                    hi = tl.arange(0, D // 8)
                    base = (ki[:, None].to(tl.int64) * HK + hk) * (D * 3 // 8)
                    ol = base + (lo[None, :] // 32) * 48 + lo[None, :] % 32
                    oh = base + (hi[None, :] // 16) * 48 + 32 + hi[None, :] % 16
                    kl = tl.load(KC + ol, mask=valid[:, None], other=0)
                    kh = tl.load(KC + oh, mask=valid[:, None], other=0)
                    vl = tl.load(VC + ol, mask=valid[:, None], other=0)
                    vh = tl.load(VC + oh, mask=valid[:, None], other=0)
                    kk = rotor_dequant_3(kl, kh, sc, M=64, W=D)
                    vv = rotor_dequant_3(vl, vh, sv, M=64, W=D)
                elif BITS == 6:
                    lo = tl.arange(0, D // 2)
                    hi = tl.arange(0, D // 4)
                    base = (ki[:, None].to(tl.int64) * HK + hk) * (D * 3 // 4)
                    ol = base + (lo[None, :] // 64) * 96 + lo[None, :] % 64
                    oh = base + (hi[None, :] // 32) * 96 + 64 + hi[None, :] % 32
                    kl = tl.load(KC + ol, mask=valid[:, None], other=0)
                    kh = tl.load(KC + oh, mask=valid[:, None], other=0)
                    vl = tl.load(VC + ol, mask=valid[:, None], other=0)
                    vh = tl.load(VC + oh, mask=valid[:, None], other=0)
                    kk = rotor_dequant_6(kl, kh, sc, M=64, W=D)
                    vv = rotor_dequant_6(vl, vh, sv, M=64, W=D)
                elif BITS == 7:
                    lo = tl.arange(0, D // 2)
                    mid = tl.arange(0, D // 4)
                    hi = tl.arange(0, D // 8)
                    base = (ki[:, None].to(tl.int64) * HK + hk) * (D * 7 // 8)
                    ol = base + (lo[None, :] // 64) * 112 + lo[None, :] % 64
                    om = base + (mid[None, :] // 32) * 112 + 64 + mid[None, :] % 32
                    oh = base + (hi[None, :] // 16) * 112 + 96 + hi[None, :] % 16
                    kl = tl.load(KC + ol, mask=valid[:, None], other=0)
                    km = tl.load(KC + om, mask=valid[:, None], other=0)
                    kh = tl.load(KC + oh, mask=valid[:, None], other=0)
                    vl = tl.load(VC + ol, mask=valid[:, None], other=0)
                    vm = tl.load(VC + om, mask=valid[:, None], other=0)
                    vh = tl.load(VC + oh, mask=valid[:, None], other=0)
                    kk = rotor_dequant_7(kl, km, kh, sc, M=64, W=D)
                    vv = rotor_dequant_7(vl, vm, vh, sv, M=64, W=D)
                elif BITS == 8:
                    offb = (ki[:, None].to(tl.int64) * HK + hk) * D + d[None, :]
                    kc = tl.load(KC + offb, mask=valid[:, None], other=0)
                    vc = tl.load(VC + offb, mask=valid[:, None], other=0)
                    kk = rotor_dequant_8(kc, sc, M=64, W=D)
                    vv = rotor_dequant_8(vc, sv, M=64, W=D)
                else:
                    tl.static_assert(BITS == 4)
                    db = tl.arange(0, D // 2)
                    offb = (ki[:, None].to(tl.int64) * HK + hk) * (D // 2) + db[None, :]
                    kc = tl.load(KC + offb, mask=valid[:, None], other=0)
                    vc = tl.load(VC + offb, mask=valid[:, None], other=0)
                    kk = rotor_dequant_4(kc, sc, M=64, W=D)
                    vv = rotor_dequant_4(vc, sv, M=64, W=D)
            elif BITS:
                # codes with an fp16 scale per 32 values, dequantized before the dot; q carries the keys' rotation
                gs = tl.arange(0, D // 32)
                sgc = (ki[:, None].to(tl.int64) * HK + hk) * (D // 32) + gs[None, :]
                sc = tl.load(KSC + sgc, mask=valid[:, None], other=0.0)
                sv = tl.load(VSC + sgc, mask=valid[:, None], other=0.0)
                if BITS == 4:
                    db = tl.arange(0, D // 2)
                    offb = (ki[:, None].to(tl.int64) * HK + hk) * (D // 2) + db[None, :]
                    kc = tl.load(KC + offb, mask=valid[:, None], other=0)
                    vc = tl.load(VC + offb, mask=valid[:, None], other=0)
                else:
                    kc = tl.load(KC + off, mask=valid[:, None], other=0)
                    vc = tl.load(VC + off, mask=valid[:, None], other=0)
                if BITS == 4:
                    kk = dequant_group_4(kc, sc, M=64, W=D)
                    vv = dequant_group_4(vc, sv, M=64, W=D)
                else:
                    kk = dequant_group_8(kc, sc, M=64, W=D)
                    vv = dequant_group_8(vc, sv, M=64, W=D)
            else:
                kk = tl.load(KC + off, mask=valid[:, None], other=0.0)
                vv = tl.load(VC + off, mask=valid[:, None], other=0.0)
            m, l, o = _tile(q, kk, vv, m, l, o, valid, SCALE)
        base = (r * NCH + c) * H + hk * G + gg
        tl.store(PO + base[:, None] * D + d[None, :], o, mask=gg[:, None] < G)
        tl.store(PM + base, m, mask=gg < G)
        tl.store(PL + base, l, mask=gg < G)


@triton.jit
def _chunk_pair(Q, KC, VC, KSC, VSC, n, sparse, r, hk, c, PO, PM, PL, IDS,
                H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr,
                NCH: tl.constexpr, SCALE: tl.constexpr, IDW: tl.constexpr, QSA: tl.constexpr,
                K_BITS: tl.constexpr, K_CODEC: tl.constexpr, V_BITS: tl.constexpr, V_CODEC: tl.constexpr):
    """Canonical 64-key arithmetic with each side decoded in its own stored basis."""

    start = c * CH
    if start < n:
        gg = tl.arange(0, 16)
        d = tl.arange(0, D)
        q = tl.load(Q + (r * H + hk * G + gg[:, None]) * D + d[None, :], mask=gg[:, None] < G, other=0.0)
        m = tl.full((16,), float("-inf"), tl.float32)
        denominator = tl.zeros((16,), tl.float32)
        o = tl.zeros((16, D), tl.float32)
        tiles = tl.minimum(n - start, CH)
        for t in range(0, tl.cdiv(tiles, 64)):
            ki = start + t * 64 + tl.arange(0, 64)
            valid = ki < n
            if QSA:
                if sparse:
                    ki = tl.load(IDS + r * IDW + ki, mask=valid, other=0)
            k = load_side(KC, KSC, ki, valid, hk, HK, D, 64, K_BITS, K_CODEC)
            v = load_side(VC, VSC, ki, valid, hk, HK, D, 64, V_BITS, V_CODEC)
            m, denominator, o = _tile(q, k, v, m, denominator, o, valid, SCALE)
        base = (r * NCH + c) * H + hk * G + gg
        tl.store(PO + base[:, None] * D + d[None, :], o, mask=gg[:, None] < G)
        tl.store(PM + base, m, mask=gg < G)
        tl.store(PL + base, denominator, mask=gg < G)


@triton.jit
def _merge(PO, PM, PL, POS0, OUT, NKR, SPR, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
           CH: tl.constexpr, NCH: tl.constexpr, QSA: tl.constexpr, BITS: tl.constexpr, CODEC: tl.constexpr = 0,
           V_BITS: tl.constexpr = -1, V_CODEC: tl.constexpr = 0):
    r = tl.program_id(0)
    hk = tl.program_id(1)
    n = tl.load(POS0) + r + 1
    if QSA:
        n = tl.where(tl.load(SPR + r) != 0, tl.load(NKR + r), n)
    if V_BITS >= 0:
        _merge_row(PO, PM, PL, OUT, n, r, hk, H, HK, D, G, CH, NCH, V_BITS, CODEC=V_CODEC)
    else:
        _merge_row(PO, PM, PL, OUT, n, r, hk, H, HK, D, G, CH, NCH, BITS, CODEC=CODEC)


@triton.jit
def _merge_row(PO, PM, PL, OUT, n, r, hk, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
               CH: tl.constexpr, NCH: tl.constexpr, BITS: tl.constexpr, CODEC: tl.constexpr = 0):
    """Row r's chunk partials merged in chunk order into its output heads (rotated back when the cache is)."""

    gg = tl.arange(0, 16)
    d = tl.arange(0, D)
    head = hk * G + gg
    m = tl.full((16,), float("-inf"), tl.float32)
    l = tl.zeros((16,), tl.float32)
    o = tl.zeros((16, D), tl.float32)
    for c in range(0, tl.cdiv(n, CH)):
        base = (r * NCH + c) * H + head
        cm = tl.load(PM + base, mask=gg < G, other=float("-inf"))
        cl = tl.load(PL + base, mask=gg < G, other=0.0)
        co = tl.load(PO + base[:, None] * D + d[None, :], mask=gg[:, None] < G, other=0.0)
        active = cl > 0.0
        next_m = tl.where(active, tl.maximum(m, cm), m)
        a = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
        b = tl.where(active, tl.exp(cm - next_m), 0.0)
        o = o * a[:, None] + co * b[:, None]
        l = l * a + cl * b
        m = next_m
    result = o / l[:, None]
    if CODEC:
        result = rotor_rotate(result, M=16, W=D, CODEC=CODEC, INVERSE=True)
    elif BITS:
        # values are stored rotated: p . (H v) = H (p . v), so one H32 a row restores the merged output
        result = tl.reshape(h32(tl.reshape(result, (16 * D // 32, 32)), M=16 * D // 32), (16, D))
    tl.store(OUT + (r * H + head[:, None]) * D + d[None, :], result.to(tl.bfloat16), mask=gg[:, None] < G)


class AttnScratch:
    def __init__(self, rows: int, heads: int, head_dim: int, capacity: int, device, *, budget: int = 2048,
                 ratio: int = 4) -> None:
        # a row reads at most budget + ratio - 1 keys (dense below the budget, its selected blocks and tail past it)
        self.nch = -(-min(capacity, budget + ratio - 1) // CHUNK)
        self.po = torch.zeros((rows, self.nch, heads, head_dim), dtype=torch.float32, device=device)
        self.pm = torch.zeros((rows, self.nch, heads), dtype=torch.float32, device=device)
        self.pl = torch.zeros((rows, self.nch, heads), dtype=torch.float32, device=device)
        self.out = torch.empty((rows, heads, head_dim), dtype=torch.bfloat16, device=device)
        # sparse attention (QSA): each row's key list, its length, whether it is sparse, block scores
        self.budget, self.ratio = budget, ratio
        self.idw = budget + ratio
        self.qsa = capacity > budget
        self.nb = -(-capacity // ratio)
        self.ids = torch.zeros((rows, self.idw), dtype=torch.int32, device=device)
        self.nk = torch.zeros((rows,), dtype=torch.int32, device=device)
        self.sparse = torch.zeros((rows,), dtype=torch.int32, device=device)
        self.scores = torch.zeros((rows, self.nb), dtype=torch.float32, device=device) if self.qsa else None


def attention(q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, pos0: torch.Tensor, scratch: AttnScratch,
              rows: int, scale: float, out: torch.Tensor | None = None, *,
              context: int | None = None, ks: torch.Tensor | None = None, vs: torch.Tensor | None = None,
              bits: int = 0, codec: int = 0, k_bits: int | None = None, k_codec: int | None = None,
              v_bits: int | None = None, v_codec: int | None = None) -> torch.Tensor:
    """BF16 queries against independent stored-basis K/V sides; sparse rows read scratch IDs.

    The query already carries the key basis. Only the value basis is inverted
    after the canonical chunk merge. Symmetric native formats keep their
    original shader dispatch; optional side traits are an all-or-none plan.
    """

    _, h, d = q.shape
    kb, kf, vb, vf = resolve_pair(bits, codec, k_bits, k_codec, v_bits, v_codec, d)
    native = kf == vf == 0 and kb == vb
    if (kb and ks is None) or (vb and vs is None):
        raise ValueError("each quantized KV side needs its own scale tensor")
    ks = kc if ks is None else ks
    vs = vc if vs is None else vs
    pair = {} if native else dict(K_BITS=kb, K_CODEC=kf, V_BITS=vb, V_CODEC=vf)
    merge_pair = {} if native else dict(V_BITS=vb, V_CODEC=vf)
    hk = kc.shape[1]
    if not native:
        if q.dtype != torch.bfloat16 or not q.is_contiguous() or q.shape[0] < rows:
            raise ValueError("pair attention requires compatible contiguous BF16 query rows")
        validate_storage(kc, ks, kc.shape[0], hk, d, kb, kf, q.device)
        validate_storage(vc, vs, kc.shape[0], hk, d, vb, vf, q.device)
    g = h // hk
    if g > 16:
        raise ValueError(
            f"this attention kernel tiles 16 query heads per KV head (G={g} does not fit); "
            "a G under 16 is masked off after the rotation, not rotated differently")
    nch = scratch.nch
    keys = nch * CHUNK if context is None else context
    if scratch.qsa:
        keys = min(keys, (scratch.budget // scratch.ratio + 1) * scratch.ratio - 1)
    chunks = min(nch, triton.cdiv(keys, CHUNK))
    out = scratch.out if out is None else out
    _chunks[(rows, hk, chunks)](q, kc, vc, ks, vs, pos0, scratch.po, scratch.pm, scratch.pl, scratch.ids, scratch.nk,
                             scratch.sparse, H=h, HK=hk, D=d, G=g, CH=CHUNK, NCH=nch, SCALE=scale,
                             IDW=scratch.idw, QSA=scratch.qsa, BITS=kb, CODEC=0, **pair, num_warps=4, num_stages=1)
    _merge[(rows, hk)](scratch.po, scratch.pm, scratch.pl, pos0, out, scratch.nk, scratch.sparse, H=h,
                       HK=hk, D=d, G=g, CH=CHUNK, NCH=nch, QSA=scratch.qsa, BITS=vb, CODEC=0, **merge_pair, num_warps=4)
    return out


@triton.jit
def _pool(IKC, POOLED, POS0, W, INV, eps, R, DI: tl.constexpr, HALF: tl.constexpr, RATIO: tl.constexpr,
          ROPE=None, DELTA=None, length=0, MODE: tl.constexpr = 0, S1: tl.constexpr = 11, S2: tl.constexpr = 10,
          ROPE_SCALE: tl.constexpr = 1.0):
    """Pool each complete RATIO-key block in fp32 order, then bf16 RMSNorm and rotate-half RoPE at its first position; recomputing a block preserves its bits."""

    _pool_block(IKC, POOLED, tl.load(POS0), tl.program_id(0), W, INV, eps, R, DI, HALF, RATIO,
                ROPE, DELTA, length, MODE, S1, S2, ROPE_SCALE)


@triton.jit
def _pool_block(IKC, POOLED, p0, i, W, INV, eps, R, DI: tl.constexpr, HALF: tl.constexpr, RATIO: tl.constexpr,
                ROPE=None, DELTA=None, length=0, MODE: tl.constexpr = 0, S1: tl.constexpr = 11, S2: tl.constexpr = 10,
                ROPE_SCALE: tl.constexpr = 1.0):
    """Block i past p0 // RATIO, if rows [p0, p0 + R) complete it."""

    b = p0 // RATIO + i
    if RATIO * b + RATIO <= p0 + R:
        d = tl.arange(0, DI)
        acc = tl.load(IKC + (RATIO * b).to(tl.int64) * DI + d).to(tl.float32)
        for k in tl.static_range(1, RATIO):
            acc = acc + tl.load(IKC + (RATIO * b + k).to(tl.int64) * DI + d).to(tl.float32)
        x = (acc / RATIO).to(tl.bfloat16).to(tl.float32)
        rinv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / DI + eps)
        xn = (x * rinv * tl.load(W + d)).to(tl.bfloat16).to(tl.float32)
        partner = tl.where(d < HALF, d + HALF, tl.where(d < 2 * HALF, d - HALF, d))
        xp = tl.load(IKC + (RATIO * b).to(tl.int64) * DI + partner).to(tl.float32)
        for k in tl.static_range(1, RATIO):
            xp = xp + tl.load(IKC + (RATIO * b + k).to(tl.int64) * DI + partner).to(tl.float32)
        xp = (xp / RATIO).to(tl.bfloat16).to(tl.float32)
        xpn = (xp * rinv * tl.load(W + partner)).to(tl.bfloat16).to(tl.float32)
        j = tl.where(d < HALF, d, tl.where(d < 2 * HALF, d - HALF, 0))
        axis = rope_axis(RATIO * b, ROPE, DELTA, length, j, MODE, S1, S2)
        ang = axis.to(tl.float32) * tl.load(INV + j)
        cos, sin = tl.cos(ang), tl.sin(ang)
        if ROPE_SCALE != 1.0:
            cos, sin = cos * ROPE_SCALE, sin * ROPE_SCALE
        rot = tl.where(d < HALF, xn * cos - xpn * sin, tl.where(d < 2 * HALF, xpn * sin + xn * cos, xn))
        tl.store(POOLED + b.to(tl.int64) * DI + d, rot.to(tl.bfloat16))


@triton.jit
def _scores(IQ, POOLED, POS0, SC, NB, HI: tl.constexpr, DI: tl.constexpr, RATIO: tl.constexpr, TOP: tl.constexpr,
            BB: tl.constexpr):
    """Score each sparse row's complete blocks with the fp32 indexer-head sum, in order, of relu(q . k) / sqrt(DI)."""

    r = tl.program_id(0)
    j = tl.program_id(1)
    complete = (tl.load(POS0) + r + 1) // RATIO
    if complete > TOP and j * BB < complete:
        b = j * BB + tl.arange(0, BB)
        ok = b < complete
        d = tl.arange(0, DI)
        k = tl.load(POOLED + b[:, None].to(tl.int64) * DI + d[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
        total = tl.zeros((BB,), dtype=tl.float32)
        for h in tl.static_range(HI):
            q = tl.load(IQ + (r * HI + h) * DI + d).to(tl.float32)
            total = total + tl.maximum(tl.sum(k * q[None, :], axis=1), 0.0)
        tl.store(SC + r * NB + b, total / tl.sqrt(DI * 1.0), mask=ok)


@triton.jit
def _select(SC, POS0, IDS, NKR, SPR, NB, RATIO: tl.constexpr, TOP: tl.constexpr, IDW: tl.constexpr,
            BLOCK: tl.constexpr):
    """List each sparse row's TOP blocks in block order, breaking score ties by lower block id, followed by the unfinished tail; dense rows retain only their length."""

    r = tl.program_id(0)
    end = tl.load(POS0) + r + 1
    complete = end // RATIO
    if complete <= TOP:
        tl.store(NKR + r, end)
        tl.store(SPR + r, 0)
    else:
        b = tl.arange(0, BLOCK)
        ok = b < complete
        v = tl.load(SC + r * NB + b, mask=ok, other=0.0)
        bits = v.to(tl.uint32, bitcast=True)
        key = tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)
        key = tl.where(ok, key, 0)
        # the largest t with at least TOP keys >= t: 32 halvings over [0, 2^32)
        lo = tl.zeros((), dtype=tl.uint64)
        hi = tl.full((), 0xFFFFFFFF, dtype=tl.uint64)
        k64 = key.to(tl.uint64)
        for _ in range(33):
            mid = (lo + hi + 1) // 2
            count = tl.sum(tl.where(k64 >= mid, 1, 0), axis=0)
            take = count >= TOP
            lo = tl.where(take, mid, lo)
            hi = tl.where(take, hi, mid - 1)
        cut = lo
        above = ok & (k64 > cut)
        equal = ok & (k64 == cut)
        need = TOP - tl.sum(above.to(tl.int32), axis=0)
        rank = tl.cumsum(equal.to(tl.int32), axis=0)
        chosen = above | (equal & (rank <= need))
        place = tl.cumsum(chosen.to(tl.int32), axis=0) - 1
        for k in tl.static_range(RATIO):
            tl.store(IDS + r * IDW + place * RATIO + k, b * RATIO + k, mask=chosen)
        t = tl.arange(0, RATIO)
        tail = RATIO * complete + t
        tl.store(IDS + r * IDW + TOP * RATIO + t, tail, mask=tail < end)
        tl.store(NKR + r, TOP * RATIO + end - RATIO * complete)
        tl.store(SPR + r, 1)


@triton.jit
def _block_keys(ROW, b, ok):
    """``_select``'s order-preserving keys of a row's block scores, 0 past the row."""

    bits = tl.load(ROW + b, mask=ok, other=0.0).to(tl.uint32, bitcast=True)
    key = tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)
    return tl.where(ok, key, 0).to(tl.uint64)


@triton.jit
def _select_tiles(SC, POS0, IDS, NKR, SPR, NB, RATIO: tl.constexpr, TOP: tl.constexpr, IDW: tl.constexpr,
                  TB: tl.constexpr):
    """``_select``'s lists for rows with more blocks than its registers hold, reading the row in TB-block tiles: the same cut (the TOP-th largest key) by radix select, one byte a pass from the top, then the same blocks placed tile by tile."""

    r = tl.program_id(0)
    end = tl.load(POS0) + r + 1
    complete = end // RATIO
    if complete <= TOP:
        tl.store(NKR + r, end)
        tl.store(SPR + r, 0)
    else:
        row = SC + r.to(tl.int64) * NB
        digits = tl.arange(0, 256)
        cut = tl.zeros((), dtype=tl.uint64)
        need = tl.full((), TOP, dtype=tl.int32)       # the cut's rank among the keys sharing its leading bytes
        for p in tl.static_range(4):
            shift = 24 - 8 * p
            hist = tl.zeros((256,), dtype=tl.int32)
            for t0 in range(0, complete, TB):
                b = t0 + tl.arange(0, TB)
                ok = b < complete
                k64 = _block_keys(row, b, ok)
                if p > 0:
                    ok = ok & ((k64 >> (shift + 8)) == (cut >> (shift + 8)))
                hist += tl.histogram(((k64 >> shift) & 0xFF).to(tl.int32), 256, mask=ok)
            atleast = tl.sum(hist, axis=0) - tl.cumsum(hist, axis=0) + hist    # keys with this byte or a larger one
            d = tl.max(tl.where(atleast >= need, digits, -1), axis=0)
            need = need - tl.sum(tl.where(digits > d, hist, 0), axis=0)
            cut = cut | (d.to(tl.uint64) << shift)
        # every key above the cut, then the first ``need`` keys equal to it (lower block ids first), in block order
        seen = tl.zeros((), dtype=tl.int32)
        placed = tl.zeros((), dtype=tl.int32)
        t0 = tl.zeros((), dtype=tl.int32)
        stop = complete
        while t0 < stop:
            b = t0 + tl.arange(0, TB)
            ok = b < complete
            k64 = _block_keys(row, b, ok)
            above = ok & (k64 > cut)
            equal = ok & (k64 == cut)
            rank = seen + tl.cumsum(equal.to(tl.int32), axis=0)
            chosen = above | (equal & (rank <= need))
            place = placed + tl.cumsum(chosen.to(tl.int32), axis=0) - 1
            for k in tl.static_range(RATIO):
                tl.store(IDS + r * IDW + place * RATIO + k, b * RATIO + k, mask=chosen)
            seen += tl.sum(equal.to(tl.int32), axis=0)
            placed += tl.sum(chosen.to(tl.int32), axis=0)
            t0 += TB
            stop = tl.where(placed >= TOP, 0, stop)        # all TOP placed: no later tile holds a chosen block
        t = tl.arange(0, RATIO)
        tail = RATIO * complete + t
        tl.store(IDS + r * IDW + TOP * RATIO + t, tail, mask=tail < end)
        tl.store(NKR + r, TOP * RATIO + end - RATIO * complete)
        tl.store(SPR + r, 1)


def _launch_select(scratch: AttnScratch, pos0: torch.Tensor, rows: int, blocks: int) -> None:
    """List each row's blocks from ``scratch.scores``: ``_select`` while ``blocks`` fit its registers, ``_select_tiles`` (the same lists) past them."""

    top = scratch.budget // scratch.ratio
    width = triton.next_power_of_2(blocks)
    if width <= SELECT_REGS:
        _select[(rows,)](scratch.scores, pos0, scratch.ids, scratch.nk, scratch.sparse, scratch.nb,
                         RATIO=scratch.ratio, TOP=top, IDW=scratch.idw, BLOCK=width, num_warps=16)
        return
    tb, warps = (4096, 8) if rows >= 64 else (8192, 16)      # a prompt's row blocks, a decode window
    _select_tiles[(rows,)](scratch.scores, pos0, scratch.ids, scratch.nk, scratch.sparse, scratch.nb,
                           RATIO=scratch.ratio, TOP=top, IDW=scratch.idw, TB=tb, num_warps=warps)


def qsa_select(iq: torch.Tensor, ikc: torch.Tensor, pooled: torch.Tensor, pos0: torch.Tensor, ik_scale: torch.Tensor,
               inv_freq: torch.Tensor, eps: float, scratch: AttnScratch, rows: int,
               *, context: int | None = None, rope=None, delta=None, length=0, sections=(11, 11, 10),
               rope_scale: float = 1.0) -> None:
    """Pool the blocks the window completes, score and select each sparse row's blocks (scratch.ids/nk/sparse)."""

    qsa_pool(ikc, pooled, pos0, ik_scale, inv_freq, eps, scratch, rows,
             rope=rope, delta=delta, length=length, sections=sections, rope_scale=rope_scale)
    qsa_rows(iq, pooled, pos0, scratch, rows, context=context)


def qsa_pool(ikc: torch.Tensor, pooled: torch.Tensor, pos0: torch.Tensor, ik_scale: torch.Tensor,
             inv_freq: torch.Tensor, eps: float, scratch: AttnScratch, rows: int, *, rope=None, delta=None, length=0,
             sections=(11, 11, 10), rope_scale: float = 1.0) -> None:
    """Pool completed blocks with full prompt rotary positions, including a block spanning prompt pieces."""

    mode = 2 if rope is not None else 1 if delta is not None else 0
    _pool[(rows // scratch.ratio + 2,)](ikc, pooled, pos0, ik_scale, inv_freq, eps, rows,
                                        DI=ikc.shape[1], HALF=inv_freq.numel(), RATIO=scratch.ratio,
                                        ROPE=rope, DELTA=delta, length=length,
                                        MODE=mode, S1=sections[1], S2=sections[2], ROPE_SCALE=rope_scale, num_warps=1)


def qsa_rows(iq: torch.Tensor, pooled: torch.Tensor, pos0: torch.Tensor, scratch: AttnScratch, rows: int, *,
             context: int | None = None) -> None:
    """Score and select the blocks of rows P0 .. P0 + rows - 1; ``context`` bounds the launches to keys in use."""

    di = pooled.shape[1]
    ratio, top = scratch.ratio, scratch.budget // scratch.ratio
    bb = 64
    blocks = scratch.nb if context is None else min(scratch.nb, max(1, triton.cdiv(context, ratio)))
    _scores[(rows, triton.cdiv(blocks, bb))](iq, pooled, pos0, scratch.scores, scratch.nb, HI=iq.shape[1],
                                                 DI=di, RATIO=ratio, TOP=top, BB=bb, num_warps=4)
    _launch_select(scratch, pos0, rows, blocks)

"""Attend each row over fixed absolute-position chunks, selecting QSA blocks by score with ties to lower block ids so window size never changes its bits."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorfold.cuda import pdl
from tensorfold.cuda.sm70 import family_ops as _half_ops
from tensorfold.cuda.pdl import wait as pdl_wait
from tensorfold.cuda.geometry import QSA_SELECT_ROWS, QSA_SELECT_SLICES, QSA_SELECT_TILE

from .image_rows import rope_axis
from .kv_pair import load_quarter, load_side, resolve_pair, validate_storage

from .kvquant import dequant_group_4, dequant_group_8, h32
from .rotorquant_kernel import LIFTING_8, rotate as rotor_rotate
from .rotorquant_ref import CENTROIDS_6, CENTROIDS_8

CHUNK = 512
TILE = 64
SELECT_REGS = 32768  # _select holds a row's block scores in registers up to this many; past it they spill
# T (``_select_multi``) splits each row's selection over slices for windows of fewer than MULTI_ROWS rows (a prompt's
# row blocks already fill the SMs a row a CTA) past MULTI_MIN_BLOCKS blocks (below, ``_select``'s one launch wins).
# Its scratch's shape (rows, tile, slices) comes from the memory plan's constants (geometry.QSA_SELECT_*).
MULTI_ROWS = QSA_SELECT_ROWS + 1
MULTI_MIN_BLOCKS = 16384
SELECT_TB = QSA_SELECT_TILE         # T's tile: the keys a CTA holds at once, 16 a thread
SELECT_FILL = 4                     # T's CTAs an SM a pass
SELECT_SLICES = QSA_SELECT_SLICES   # T's slices a row at most (its placement counts' scratch)
SELECT_SPAN = 5      # T's tiles a slice at most, while the slices fit: longer slices serialize placement


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
def _chunks(Q, KC, VC, KSC, VSC, POS0, PO, PM, PL, IDS, NKR, SPR, CENTROIDS,
            H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr,
            NCH: tl.constexpr, SCALE: tl.constexpr, IDW: tl.constexpr, QSA: tl.constexpr, BITS: tl.constexpr,
            K_BITS: tl.constexpr = -1, K_CODEC: tl.constexpr = 0,
            V_BITS: tl.constexpr = -1, V_CODEC: tl.constexpr = 0, ROTOR_LOOKUP: tl.constexpr = False,
            GATHER6: tl.constexpr = False, PDL: tl.constexpr = False):
    pdl_wait(PDL)
    r = tl.program_id(0)
    hk = tl.program_id(1)
    c = tl.program_id(2)
    n = tl.load(POS0) + r + 1
    sparse = False
    if QSA:
        sparse = tl.load(SPR + r) != 0
        n = tl.where(sparse, tl.load(NKR + r), n)
    if K_BITS >= 0:
        _chunk_pair(Q, KC, VC, KSC, VSC, n, sparse, r, hk, c, PO, PM, PL, IDS, CENTROIDS,
                    H, HK, D, G, CH, NCH, SCALE, IDW, QSA, K_BITS, K_CODEC, V_BITS, V_CODEC, ROTOR_LOOKUP=ROTOR_LOOKUP,
                    GATHER6=GATHER6)
    else:
        _chunk(Q, KC, VC, KSC, VSC, n, sparse, r, hk, c, PO, PM, PL, IDS,
               H, HK, D, G, CH, NCH, SCALE, IDW, QSA, BITS)


@triton.jit
def _chunk(Q, KC, VC, KSC, VSC, n, sparse, r, hk, c, PO, PM, PL, IDS,
           H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr,
           NCH: tl.constexpr, SCALE: tl.constexpr, IDW: tl.constexpr, QSA: tl.constexpr, BITS: tl.constexpr):
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
            if BITS:
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
def _chunk_pair(Q, KC, VC, KSC, VSC, n, sparse, r, hk, c, PO, PM, PL, IDS, CENTROIDS,
                H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr,
                NCH: tl.constexpr, SCALE: tl.constexpr, IDW: tl.constexpr, QSA: tl.constexpr,
                K_BITS: tl.constexpr, K_CODEC: tl.constexpr, V_BITS: tl.constexpr, V_CODEC: tl.constexpr,
                ROTOR_LOOKUP: tl.constexpr = False, GATHER6: tl.constexpr = False):
    """Canonical 64-key arithmetic with each side decoded in its own stored basis.

    RotorQuant K and V of six or eight bits at head width 256 decode through ``_chunk_quarters`` (``CENTROIDS``:
    the scratch's centroid tables), the same bits, except six-bit sides in prompt passes (``GATHER6``)."""

    K_Q: tl.constexpr = K_CODEC != 0 and (K_BITS == 6 or K_BITS == 8)
    V_Q: tl.constexpr = V_CODEC != 0 and (V_BITS == 6 or V_BITS == 8)
    if K_Q and V_Q and D == 256 and not GATHER6:
        tl.static_assert(not ROTOR_LOOKUP)      # the lookup is seven-bit only, as ``load_side`` asserts
        _chunk_quarters(Q, KC, VC, KSC, VSC, n, sparse, r, hk, c, PO, PM, PL, IDS, CENTROIDS, H, HK, D, G, CH, NCH,
                        SCALE, IDW, QSA, K_BITS, V_BITS)
    else:
        _chunk_tiles(Q, KC, VC, KSC, VSC, n, sparse, r, hk, c, PO, PM, PL, IDS, H, HK, D, G, CH, NCH, SCALE, IDW,
                     QSA, K_BITS, K_CODEC, V_BITS, V_CODEC, ROTOR_LOOKUP, GATHER6)


@triton.jit
def _chunk_tiles(Q, KC, VC, KSC, VSC, n, sparse, r, hk, c, PO, PM, PL, IDS,
                 H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr,
                 NCH: tl.constexpr, SCALE: tl.constexpr, IDW: tl.constexpr, QSA: tl.constexpr,
                 K_BITS: tl.constexpr, K_CODEC: tl.constexpr, V_BITS: tl.constexpr, V_CODEC: tl.constexpr,
                 ROTOR_LOOKUP: tl.constexpr, GATHER6: tl.constexpr):
    """``_chunk_pair`` on whole [64, D] tiles of each side."""

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
            k = load_side(KC, KSC, ki, valid, hk, HK, D, 64, K_BITS, K_CODEC, ROTOR_LOOKUP=ROTOR_LOOKUP,
                          GATHER6=GATHER6)
            v = load_side(VC, VSC, ki, valid, hk, HK, D, 64, V_BITS, V_CODEC, ROTOR_LOOKUP=ROTOR_LOOKUP,
                          GATHER6=GATHER6)
            m, denominator, o = _tile(q, k, v, m, denominator, o, valid, SCALE)
        base = (r * NCH + c) * H + hk * G + gg
        tl.store(PO + base[:, None] * D + d[None, :], o, mask=gg[:, None] < G)
        tl.store(PM + base, m, mask=gg < G)
        tl.store(PL + base, denominator, mask=gg < G)


@triton.jit
def _quarter_pv(o, p, VC, row, valid, scale, J: tl.constexpr, D: tl.constexpr, BITS: tl.constexpr, CENTROIDS,
                alpha):
    """``_tile``'s value update for output columns [64 J, 64 J + 64): the same per-column arithmetic."""
    return o * alpha[:, None] + tl.dot(p, load_quarter(VC, row, valid, scale, J, D, BITS, CENTROIDS))


@triton.jit
def _chunk_quarters(Q, KC, VC, KSC, VSC, n, sparse, r, hk, c, PO, PM, PL, IDS, CENTROIDS,
                    H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, CH: tl.constexpr,
                    NCH: tl.constexpr, SCALE: tl.constexpr, IDW: tl.constexpr, QSA: tl.constexpr,
                    K_BITS: tl.constexpr, V_BITS: tl.constexpr):
    """``_chunk_tiles`` for six- or eight-bit RotorQuant K and V at D 256, decoded in 64-column quarters with
    table centroids (``CENTROIDS``: six-bit table at 0, eight-bit at 64).

    Bits: the scores accumulate the four quarters' MMAs into one accumulator in column order, the same MMA
    sequence (and FP32 accumulation order) as the whole-tile dot from zero; the value update is per output column,
    so each quarter's dot is the whole tile's columns; decoding is elementwise (``load_quarter``). The softmax runs
    on the same [16, 64] score tile and layout. Quarters keep each tile's decoded operands in registers (whole
    [64, 256] tiles spill) and the tables replace the six-bit 63-select chain and the eight-bit register gather.
    """

    tl.static_assert(D == 256)
    KT = CENTROIDS + (64 if K_BITS == 8 else 0)
    VT = CENTROIDS + (64 if V_BITS == 8 else 0)
    start = c * CH
    if start < n:
        gg = tl.arange(0, 16)
        d = tl.arange(0, 64)
        qm = gg[:, None] < G
        qp = Q + (r * H + hk * G + gg[:, None]) * D + d[None, :]
        q0 = tl.load(qp, mask=qm, other=0.0)
        q1 = tl.load(qp + 64, mask=qm, other=0.0)
        q2 = tl.load(qp + 128, mask=qm, other=0.0)
        q3 = tl.load(qp + 192, mask=qm, other=0.0)
        m = tl.full((16,), float("-inf"), tl.float32)
        denominator = tl.zeros((16,), tl.float32)
        o0 = tl.zeros((16, 64), tl.float32)
        o1 = tl.zeros((16, 64), tl.float32)
        o2 = tl.zeros((16, 64), tl.float32)
        o3 = tl.zeros((16, 64), tl.float32)
        tiles = tl.minimum(n - start, CH)
        for t in range(0, tl.cdiv(tiles, 64)):
            ki = start + t * 64 + tl.arange(0, 64)
            valid = ki < n
            if QSA:
                if sparse:
                    ki = tl.load(IDS + r * IDW + ki, mask=valid, other=0)
            row = ki.to(tl.int64) * HK + hk
            ks0 = tl.load(KSC + row * (D // 128), mask=valid, other=0.0)
            ks1 = tl.load(KSC + row * (D // 128) + 1, mask=valid, other=0.0)
            acc = tl.zeros((16, 64), tl.float32)
            acc = tl.dot(q0, tl.trans(load_quarter(KC, row, valid, ks0, 0, D, K_BITS, KT)), acc)
            acc = tl.dot(q1, tl.trans(load_quarter(KC, row, valid, ks0, 1, D, K_BITS, KT)), acc)
            acc = tl.dot(q2, tl.trans(load_quarter(KC, row, valid, ks1, 2, D, K_BITS, KT)), acc)
            acc = tl.dot(q3, tl.trans(load_quarter(KC, row, valid, ks1, 3, D, K_BITS, KT)), acc)
            scores = tl.where(valid[None, :], acc * SCALE, float("-inf"))
            tile_m = tl.max(scores, 1)
            active = tile_m != float("-inf")
            next_m = tl.where(active, tl.maximum(m, tile_m), m)
            alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
            p = tl.where(valid[None, :] & active[:, None], tl.exp(scores - next_m[:, None]), 0.0)
            pb = p.to(tl.bfloat16)
            vs0 = tl.load(VSC + row * (D // 128), mask=valid, other=0.0)
            vs1 = tl.load(VSC + row * (D // 128) + 1, mask=valid, other=0.0)
            o0 = _quarter_pv(o0, pb, VC, row, valid, vs0, 0, D, V_BITS, VT, alpha)
            o1 = _quarter_pv(o1, pb, VC, row, valid, vs0, 1, D, V_BITS, VT, alpha)
            o2 = _quarter_pv(o2, pb, VC, row, valid, vs1, 2, D, V_BITS, VT, alpha)
            o3 = _quarter_pv(o3, pb, VC, row, valid, vs1, 3, D, V_BITS, VT, alpha)
            denominator = denominator * alpha + tl.sum(p, 1)
            m = next_m
        base = (r * NCH + c) * H + hk * G + gg
        op = PO + base[:, None] * D + d[None, :]
        tl.store(op, o0, mask=qm)
        tl.store(op + 64, o1, mask=qm)
        tl.store(op + 128, o2, mask=qm)
        tl.store(op + 192, o3, mask=qm)
        tl.store(PM + base, m, mask=gg < G)
        tl.store(PL + base, denominator, mask=gg < G)


@triton.jit
def _merge(PO, PM, PL, POS0, OUT, NKR, SPR, H: tl.constexpr, D: tl.constexpr, CH: tl.constexpr, NCH: tl.constexpr,
           QSA: tl.constexpr, BITS: tl.constexpr,
           V_BITS: tl.constexpr = -1, V_CODEC: tl.constexpr = 0, PDL: tl.constexpr = False):
    pdl_wait(PDL)
    r = tl.program_id(0)
    n = tl.load(POS0) + r + 1
    if QSA:
        n = tl.where(tl.load(SPR + r) != 0, tl.load(NKR + r), n)
    if V_BITS >= 0:
        _merge_row(PO, PM, PL, OUT, n, r, tl.program_id(1), H, D, CH, NCH, V_BITS, CODEC=V_CODEC)
    else:
        _merge_row(PO, PM, PL, OUT, n, r, tl.program_id(1), H, D, CH, NCH, BITS)


@triton.jit
def _merge_row(PO, PM, PL, OUT, n, r, h, H: tl.constexpr, D: tl.constexpr, CH: tl.constexpr, NCH: tl.constexpr,
               BITS: tl.constexpr, CODEC: tl.constexpr = 0):
    """Row r's chunk partials of query head h merged in chunk order into its output (rotated back when the cache is).
    The merge and the inverse rotation are per head and per element, so a program a head keeps the bits of any
    grouping of heads; one warp a head is the shortest path."""

    head = h + tl.arange(0, 1)
    d = tl.arange(0, D)
    m = tl.full((1,), float("-inf"), tl.float32)
    l = tl.zeros((1,), tl.float32)
    o = tl.zeros((1, D), tl.float32)
    for c in range(0, tl.cdiv(n, CH)):
        base = (r * NCH + c) * H + head
        cm = tl.load(PM + base)
        cl = tl.load(PL + base)
        co = tl.load(PO + base[:, None] * D + d[None, :])
        active = cl > 0.0
        next_m = tl.where(active, tl.maximum(m, cm), m)
        a = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
        b = tl.where(active, tl.exp(cm - next_m), 0.0)
        o = o * a[:, None] + co * b[:, None]
        l = l * a + cl * b
        m = next_m
    result = o / l[:, None]
    if CODEC:
        result = rotor_rotate(result, M=1, W=D, CODEC=CODEC, INVERSE=True)
    elif BITS:
        # values are stored rotated: p . (H v) = H (p . v), so one H32 a row restores the merged output
        result = tl.reshape(h32(tl.reshape(result, (D // 32, 32)), M=D // 32), (1, D))
    tl.store(OUT + (r * H + head[:, None]) * D + d[None, :], result.to(tl.bfloat16))


class AttnScratch:
    def __init__(self, rows: int, heads: int, head_dim: int, capacity: int, device, *, budget: int = 2048,
                 ratio: int = 4, dtype=torch.bfloat16) -> None:
        # a row reads at most budget + ratio - 1 keys (dense below the budget, its selected blocks and tail past it)
        self.capacity = capacity
        self.nch = -(-min(capacity, budget + ratio - 1) // CHUNK)
        self.po = torch.zeros((rows, self.nch, heads, head_dim), dtype=torch.float32, device=device)
        self.pm = torch.zeros((rows, self.nch, heads), dtype=torch.float32, device=device)
        self.pl = torch.zeros((rows, self.nch, heads), dtype=torch.float32, device=device)
        self.out = torch.empty((rows, heads, head_dim), dtype=dtype, device=device)
        # sparse attention (QSA): each row's key list, its length, whether it is sparse, block scores
        self.budget, self.ratio = budget, ratio
        self.idw = budget + ratio
        self.qsa = capacity > budget
        self.nb = -(-capacity // ratio)
        self.ids = torch.zeros((rows, self.idw), dtype=torch.int32, device=device)
        self.nk = torch.zeros((rows,), dtype=torch.int32, device=device)
        self.sparse = torch.zeros((rows,), dtype=torch.int32, device=device)
        self.scores = torch.zeros((rows, self.nb), dtype=torch.float32, device=device) if self.qsa else None
        # T's rows (fewer than MULTI_ROWS): radix histograms (zeroed by each call) and their slices' placement counts
        # (written by the last pass before placement reads them), at most ``cw`` slices a row
        self.cw = min(SELECT_SLICES, triton.next_power_of_2(-(-self.nb // SELECT_TB)))
        multi = min(rows, MULTI_ROWS - 1)
        self.hist = torch.zeros((multi, 4, 256), dtype=torch.int32, device=device) if self.qsa else None
        self.suf = torch.zeros((multi, self.cw, 257), dtype=torch.int32, device=device) if self.qsa else None
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count if self.qsa else 0
        self.centroids, self.thresholds = codebook(device)


def codebook(device) -> tuple[torch.Tensor, torch.Tensor]:
    """The RotorQuant constants kernels read from device memory, as float32 (the constants' exact values): the
    centroids, CENTROIDS_6 at 0 and CENTROIDS_8 at 64 (``_chunk_quarters``, the eight-bit encoders of
    ``glue._prep_row``), and the eight-bit lifting thresholds ``LIFTING_8`` (the eight-bit encoders)."""

    return (torch.tensor(CENTROIDS_6 + CENTROIDS_8, dtype=torch.float32, device=device),
            torch.tensor(LIFTING_8, dtype=torch.float32, device=device))


def attention(q: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor, pos0: torch.Tensor, scratch: AttnScratch,
              rows: int, scale: float, out: torch.Tensor | None = None, *,
              context: int | None = None, ks: torch.Tensor | None = None, vs: torch.Tensor | None = None,
              bits: int = 0, codec: int = 0, k_bits: int | None = None, k_codec: int | None = None,
              v_bits: int | None = None, v_codec: int | None = None, rotor_lookup: bool = False,
              prompt: bool = False) -> torch.Tensor:
    """BF16 queries against independent stored-basis K/V sides; sparse rows read scratch IDs.

    The query already carries the key basis. Only the value basis is inverted
    after the canonical chunk merge. Symmetric native formats keep their
    original shader dispatch; optional side traits are an all-or-none plan.
    ``prompt`` (a prompt pass's row blocks) decodes six-bit sides' centroids by register gather, the same bits.
    """
    if q.dtype == torch.float16:
        return _half_ops.attention(q, kc, vc, pos0, scratch, rows, scale, out, context=context, ks=ks, vs=vs, bits=bits, codec=codec, k_bits=k_bits, k_codec=k_codec, v_bits=v_bits, v_codec=v_codec, rotor_lookup=rotor_lookup, prompt=prompt)


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
            f"the attention chunk kernels tile 16 query heads per KV head (G={g} does not fit); "
            "a G under 16 leaves the tile's spare rows masked off")
    nch = scratch.nch
    keys = nch * CHUNK if context is None else context
    if scratch.qsa:
        keys = min(keys, (scratch.budget // scratch.ratio + 1) * scratch.ratio - 1)
    chunks = min(nch, triton.cdiv(keys, CHUNK))
    out = scratch.out if out is None else out
    _chunks[(rows, hk, chunks)](q, kc, vc, ks, vs, pos0, scratch.po, scratch.pm, scratch.pl, scratch.ids, scratch.nk,
                             scratch.sparse, scratch.centroids, H=h, HK=hk, D=d, G=g, CH=CHUNK, NCH=nch, SCALE=scale,
                             IDW=scratch.idw, QSA=scratch.qsa, BITS=kb, ROTOR_LOOKUP=rotor_lookup,
                             GATHER6=prompt and 6 in (kb, vb), **pair, num_warps=4, num_stages=1, PDL=pdl.ON,
                             launch_pdl=pdl.ON)
    _merge[(rows, h)](scratch.po, scratch.pm, scratch.pl, pos0, out, scratch.nk, scratch.sparse, H=h, D=d,
                      CH=CHUNK, NCH=nch, QSA=scratch.qsa, BITS=vb, **merge_pair, num_warps=1,
                      PDL=pdl.ON, launch_pdl=pdl.ON)
    return out


@triton.jit
def _pool(IKC, POOLED, POS0, W, INV, eps, R, DI: tl.constexpr, HALF: tl.constexpr, RATIO: tl.constexpr,
          ROPE=None, DELTA=None, length=0, MODE: tl.constexpr = 0, S1: tl.constexpr = 11, S2: tl.constexpr = 10,
          ROPE_SCALE: tl.constexpr = 1.0, PDL: tl.constexpr = False):
    """Pool each complete RATIO-key block in fp32 order, then bf16 RMSNorm and rotate-half RoPE at its first position; recomputing a block preserves its bits."""

    pdl_wait(PDL)
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
        ang = tl.where(d < 2 * HALF, axis.to(tl.float32) * tl.load(INV + j), 0.0)   # as ``glue._prep_row``'s
        cos, sin = tl.cos(ang), tl.sin(ang)
        if ROPE_SCALE != 1.0:
            cos, sin = cos * ROPE_SCALE, sin * ROPE_SCALE
        rot = tl.where(d < HALF, xn * cos - xpn * sin, tl.where(d < 2 * HALF, xpn * sin + xn * cos, xn))
        tl.store(POOLED + b.to(tl.int64) * DI + d, rot.to(tl.bfloat16))


@triton.jit
def _scores(IQ, POOLED, POS0, SC, NB, HI: tl.constexpr, DI: tl.constexpr, RATIO: tl.constexpr, TOP: tl.constexpr,
            BB: tl.constexpr, PDL: tl.constexpr = False):
    """Score each sparse row's complete blocks with the fp32 indexer-head sum, in order, of relu(q . k) / sqrt(DI)."""

    pdl_wait(PDL)
    r = tl.program_id(0)
    _score_rows(IQ, POOLED, SC, NB, r, tl.program_id(1), (tl.load(POS0) + r + 1) // RATIO, HI, DI, TOP, BB)


@triton.jit
def _score_rows(IQ, POOLED, SC, NB, r, j, complete, HI: tl.constexpr, DI: tl.constexpr, TOP: tl.constexpr,
                BB: tl.constexpr):
    """Row r's scores of blocks [j * BB, j * BB + BB) below ``complete``; every scoring launch shares these bits."""

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
def _row_end(POS, r, PER_ROW: tl.constexpr):
    """Row r's key count: its own position's (``POS[r]``, a concurrent step's rows) or the window's ``POS[0] + r``."""

    if PER_ROW:
        return tl.load(POS + r) + 1
    return tl.load(POS) + r + 1


@triton.jit
def _select(SC, POS0, IDS, NKR, SPR, NB, RATIO: tl.constexpr, TOP: tl.constexpr, IDW: tl.constexpr,
            BLOCK: tl.constexpr, PER_ROW: tl.constexpr = False, PDL: tl.constexpr = False):
    """List each sparse row's TOP blocks in block order, breaking score ties by lower block id, followed by the unfinished tail; dense rows retain only their length."""

    pdl_wait(PDL)
    r = tl.program_id(0)
    end = _row_end(POS0, r, PER_ROW)
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
                  TB: tl.constexpr, PER_ROW: tl.constexpr = False, PDL: tl.constexpr = False):
    """``_select``'s lists for rows with more blocks than its registers hold, reading the row in TB-block tiles: the same cut (the TOP-th largest key) by radix select, one byte a pass from the top, then the same blocks placed tile by tile."""

    pdl_wait(PDL)
    r = tl.program_id(0)
    end = _row_end(POS0, r, PER_ROW)
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


@triton.jit
def _keys32(ROW, b, ok):
    """``_block_keys``' order in 32 bits; slots off ``ok`` hold no key, and every use masks them."""

    bits = tl.load(ROW + b, mask=ok, other=0.0).to(tl.uint32, bitcast=True)
    return tl.where((bits & 0x80000000) != 0, ~bits, bits | 0x80000000)


@triton.jit
def _resolve(H, PASSES: tl.constexpr, TOP: tl.constexpr):
    """The cut's leading PASSES bytes and ``need`` (its rank among the keys sharing them) from a row's whole-row
    histograms H [4, 256], derived as ``_select_tiles`` derives them; every CTA derives the same values."""

    digits = tl.arange(0, 256)
    cut = tl.zeros((), dtype=tl.uint32)
    need = tl.full((), TOP, dtype=tl.int32)
    for p in tl.static_range(PASSES):
        hist = tl.load(H + p * 256 + digits)
        atleast = tl.sum(hist, axis=0) - tl.cumsum(hist, axis=0) + hist
        d = tl.max(tl.where(atleast >= need, digits, -1), axis=0)     # >= 0: the prefix's keys number >= need
        need = need - tl.sum(tl.where(digits > d, hist, 0), axis=0)
        cut = cut | (d.to(tl.uint32) << (24 - 8 * p))
    return cut, need


@triton.jit
def _select_pass(SC, POS, HIST, SUF, NB, SL, PASS: tl.constexpr, RATIO: tl.constexpr, TOP: tl.constexpr,
                 CW: tl.constexpr, TB: tl.constexpr, PER_ROW: tl.constexpr, PDL: tl.constexpr = False):
    """Radix pass PASS over slice s (program 0) of row r (program 1): the histogram of byte 3 - PASS of the slice's
    keys that share the cut's leading bytes, added into the row's (integer atomics: any order, the same sums). The
    last pass also keeps the slice's counts for placement: SUF[r, s, x] = its keys above the cut's leading three
    bytes plus those sharing them with a last byte >= x (x = 256: the former alone)."""

    pdl_wait(PDL)
    s = tl.program_id(0)
    r = tl.program_id(1)
    complete = _row_end(POS, r, PER_ROW) // RATIO
    lo = s * SL
    if complete > TOP and lo < complete:
        hi = tl.minimum(lo + SL, complete)
        row = SC + r.to(tl.int64) * NB
        H = HIST + r * 1024
        cut, _ = _resolve(H, PASS, TOP)
        SHIFT: tl.constexpr = 24 - 8 * PASS
        hist = tl.zeros((256,), dtype=tl.int32)
        above = tl.zeros((TB,), dtype=tl.int32)
        for t0 in range(lo, hi, TB):
            b = t0 + tl.arange(0, TB)
            ok = b < hi
            k = _keys32(row, b, ok)
            if PASS == 3:
                above += (ok & ((k >> 8) > (cut >> 8))).to(tl.int32)
            if PASS > 0:
                ok = ok & ((k >> (SHIFT + 8)) == (cut >> (SHIFT + 8)))
            hist += tl.histogram(((k >> SHIFT) & 0xFF).to(tl.int32), 256, mask=ok)
        d = tl.arange(0, 256)
        tl.atomic_add(H + PASS * 256 + d, hist, mask=hist > 0, sem="relaxed")
        if PASS == 3:
            mine = SUF + (r * CW + s) * 257
            over = tl.sum(above, axis=0)
            tl.store(mine + d, over + tl.sum(hist, axis=0) - tl.cumsum(hist, axis=0) + hist)
            tl.store(mine + 256, over)


@triton.jit
def _select_emit(SC, POS, HIST, SUF, IDS, NKR, SPR, NB, SL, RATIO: tl.constexpr, TOP: tl.constexpr,
                 IDW: tl.constexpr, CW: tl.constexpr, TB: tl.constexpr, PER_ROW: tl.constexpr,
                 PDL: tl.constexpr = False):
    """``_select_tiles``' placement of slice s of row r, its running counts seeded by the slices before it, so each
    chosen block lands where the one-CTA scan puts it; slice 0 writes the tail, length and flag (a dense row's
    length alone)."""

    pdl_wait(PDL)
    tl.static_assert(TB < 32768)                            # a tile's two counts (<= TB each) share one int32 sum
    s = tl.program_id(0)
    r = tl.program_id(1)
    end = _row_end(POS, r, PER_ROW)
    complete = end // RATIO
    lo = s * SL
    if complete <= TOP:
        if s == 0:
            tl.store(NKR + r, end)
            tl.store(SPR + r, 0)
    elif lo < complete:
        cut, need = _resolve(HIST + r * 1024, 4, TOP)
        d = (cut & 0xFF).to(tl.int32)
        i = tl.arange(0, CW)
        counts = SUF + (r * CW + i) * 257
        before = i < s
        over = tl.load(counts + d + 1, mask=before, other=0)                     # keys above the cut
        seen = tl.sum(tl.load(counts + d, mask=before, other=0) - over, axis=0)  # keys equal to it, earlier slices
        placed = tl.sum(over, axis=0) + tl.minimum(seen, need)
        mine = SUF + (r * CW + s) * 257
        over_here = tl.load(mine + d + 1)
        last = placed + over_here + tl.minimum(tl.maximum(need - seen, 0), tl.load(mine + d) - over_here)
        hi = tl.minimum(lo + SL, complete)
        row = SC + r.to(tl.int64) * NB
        for t0 in range(lo, hi, TB):
            if placed < last:                               # tiles past the slice's last chosen block: nothing
                b = t0 + tl.arange(0, TB)
                ok = b < hi
                k = _keys32(row, b, ok)
                above = ok & (k > cut)
                equal = ok & (k == cut)
                both = tl.sum(equal.to(tl.int32) * 65536 + above.to(tl.int32), axis=0)     # ties << 16 | above
                ties = both >> 16
                if (seen < need) & (seen + ties > need):    # the tile holding the need-th equal key ranks them
                    rank = seen + tl.cumsum(equal.to(tl.int32), axis=0)
                    chosen = above | (equal & (rank <= need))
                else:                                       # all of its equal keys are chosen, or none
                    chosen = above | (equal & (seen < need))
                place = placed + tl.cumsum(chosen.to(tl.int32), axis=0) - 1
                for q in tl.static_range(RATIO):
                    tl.store(IDS + r * IDW + place * RATIO + q, b * RATIO + q, mask=chosen)
                placed += (both & 65535) + tl.minimum(tl.maximum(need - seen, 0), ties)
                seen += ties
        if s == 0:
            t = tl.arange(0, RATIO)
            tail = RATIO * complete + t
            tl.store(IDS + r * IDW + TOP * RATIO + t, tail, mask=tail < end)
            tl.store(NKR + r, TOP * RATIO + end - RATIO * complete)
            tl.store(SPR + r, 1)


def _slices(scratch: AttnScratch, rows: int, blocks: int) -> tuple[int, int]:
    """T's slice length (whole SELECT_TB tiles) and count: about SELECT_FILL CTAs an SM a pass, at most SELECT_SPAN
    tiles a slice (a slice's placement scans its tiles in turn), and at most ``scratch.cw`` slices (the slices its
    placement counts hold), which wins: ``tiles / cw`` tiles a slice number at most ``cw``."""

    want = min(scratch.cw, max(1, -(-SELECT_FILL * scratch.sms // rows)))
    tiles = -(-blocks // SELECT_TB)
    sl = SELECT_TB * max(-(-tiles // scratch.cw), min(SELECT_SPAN, -(-tiles // want)))
    return sl, -(-blocks // sl)


def _select_multi(scratch: AttnScratch, pos: torch.Tensor, rows: int, blocks: int, per_row: bool) -> None:
    """T: ``_select_tiles``' lists from a (slices, rows) grid: four histogram passes, then placement."""

    if rows > scratch.hist.shape[0]:
        raise ValueError(f"select scratch holds {scratch.hist.shape[0]} rows, not {rows}")
    sl, n = _slices(scratch, rows, blocks)
    common = dict(RATIO=scratch.ratio, TOP=scratch.budget // scratch.ratio, CW=scratch.cw, TB=SELECT_TB,
                  PER_ROW=per_row, num_warps=SELECT_TB // 512)
    scratch.hist.zero_()                     # every call: a dirty histogram would move the cut silently
    for p in range(4):
        _select_pass[(n, rows)](scratch.scores, pos, scratch.hist, scratch.suf, scratch.nb, sl, PASS=p, **common,
                                PDL=pdl.ON, launch_pdl=pdl.ON)
    _select_emit[(n, rows)](scratch.scores, pos, scratch.hist, scratch.suf, scratch.ids, scratch.nk, scratch.sparse,
                            scratch.nb, sl, IDW=scratch.idw, **common, PDL=pdl.ON, launch_pdl=pdl.ON)


def _launch_select(scratch: AttnScratch, pos: torch.Tensor, rows: int, blocks: int, *,
                   per_row: bool = False) -> None:
    """List each row's blocks from ``scratch.scores``: T for windows of long rows, else ``_select`` while ``blocks``
    fit its registers and ``_select_tiles`` (the same lists) past them. ``per_row``: row r ends at ``pos[r] + 1``."""

    if blocks > scratch.nb:
        raise ValueError(f"{blocks} blocks exceed the scratch's {scratch.nb}")
    if rows < MULTI_ROWS and blocks > MULTI_MIN_BLOCKS:
        _select_multi(scratch, pos, rows, blocks, per_row)
        return
    top = scratch.budget // scratch.ratio
    width = triton.next_power_of_2(blocks)
    if width <= SELECT_REGS:
        _select[(rows,)](scratch.scores, pos, scratch.ids, scratch.nk, scratch.sparse, scratch.nb,
                         RATIO=scratch.ratio, TOP=top, IDW=scratch.idw, BLOCK=width, PER_ROW=per_row, num_warps=16,
                         PDL=pdl.ON, launch_pdl=pdl.ON)
        return
    tb, warps = (4096, 8) if rows >= 64 else (8192, 16)      # a prompt's row blocks, a decode window
    _select_tiles[(rows,)](scratch.scores, pos, scratch.ids, scratch.nk, scratch.sparse, scratch.nb,
                           RATIO=scratch.ratio, TOP=top, IDW=scratch.idw, TB=tb, PER_ROW=per_row, num_warps=warps,
                           PDL=pdl.ON, launch_pdl=pdl.ON)


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
    if ikc.dtype == torch.float16:
        return _half_ops.qsa_pool(ikc, pooled, pos0, ik_scale, inv_freq, eps, scratch, rows, rope=rope, delta=delta, length=length, sections=sections, rope_scale=rope_scale)


    mode = 2 if rope is not None else 1 if delta is not None else 0
    _pool[(rows // scratch.ratio + 2,)](ikc, pooled, pos0, ik_scale, inv_freq, eps, rows,
                                        DI=ikc.shape[1], HALF=inv_freq.numel(), RATIO=scratch.ratio,
                                        ROPE=rope, DELTA=delta, length=length,
                                        MODE=mode, S1=sections[1], S2=sections[2], ROPE_SCALE=rope_scale, num_warps=1,
                                        PDL=pdl.ON, launch_pdl=pdl.ON)


def qsa_rows(iq: torch.Tensor, pooled: torch.Tensor, pos0: torch.Tensor, scratch: AttnScratch, rows: int, *,
             context: int | None = None) -> None:
    """Score and select the blocks of rows P0 .. P0 + rows - 1; ``context`` bounds the launches to keys in use."""
    if iq.dtype == torch.float16:
        return _half_ops.qsa_rows(iq, pooled, pos0, scratch, rows, context=context)


    di = pooled.shape[1]
    ratio, top = scratch.ratio, scratch.budget // scratch.ratio
    bb = 64
    blocks = scratch.nb if context is None else min(scratch.nb, max(1, triton.cdiv(context, ratio)))
    _scores[(rows, triton.cdiv(blocks, bb))](iq, pooled, pos0, scratch.scores, scratch.nb, HI=iq.shape[1],
                                                 DI=di, RATIO=ratio, TOP=top, BB=bb, num_warps=4,
                                                 PDL=pdl.ON, launch_pdl=pdl.ON)
    _launch_select(scratch, pos0, rows, blocks)

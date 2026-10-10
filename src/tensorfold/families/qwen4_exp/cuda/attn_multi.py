"""A concurrent step's attention in one launch a kernel: rows find their caches by table and run the one-stream code."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from tensorfold.cuda.kernels import gdn as shared

from . import attention as attn_mod, glue
from .attention import CHUNK, _chunk, _chunk_pair, _merge_row, _pool_block

PTRS = 6                 # a stream's pointers a layer: keys, values, key scales, value scales, index keys, pooled


@triton.jit
def _ptr(TABLE, s, T: tl.constexpr):
    return tl.multiple_of(tl.load(TABLE + s).to(tl.pointer_type(T)), 16)


@triton.jit
def _prep_multi(P, POSR, SID, CP, VP, QW, KW, IW, INV, Q, IQ, eps, N, PW: tl.constexpr, NQ: tl.constexpr,
                NKV: tl.constexpr, HD: tl.constexpr, NI: tl.constexpr, IHD: tl.constexpr, HALF: tl.constexpr,
                BITS: tl.constexpr, KT: tl.constexpr, VISION: tl.constexpr,
                S1: tl.constexpr = 11, S2: tl.constexpr = 10, ROPE_SCALE: tl.constexpr = 1.0,
                CODEC: tl.constexpr = 0, STATUS=None, LAYER: tl.constexpr = 0,
                K_BITS: tl.constexpr = -1, K_CODEC: tl.constexpr = 0,
                V_BITS: tl.constexpr = -1, V_CODEC: tl.constexpr = 0, VT: tl.constexpr = None):
    r = tl.program_id(0)
    s = tl.load(SID + r)
    rope, delta, length = CP, CP, 0
    if VISION:
        rope, delta = _ptr(VP, s, tl.int32), _ptr(VP + N, s, tl.int32)
        length = tl.load(VP + 2 * N + s).to(tl.int32)
    PAIR: tl.constexpr = K_BITS >= 0
    KS_TYPE: tl.constexpr = tl.float32 if (K_CODEC if PAIR else CODEC) else tl.float16
    VS_TYPE: tl.constexpr = tl.float32 if (V_CODEC if PAIR else CODEC) else tl.float16
    VALUE_TYPE: tl.constexpr = VT if PAIR else KT
    status = _ptr(STATUS, s, tl.int32) if PAIR or CODEC else None
    glue._prep_row(P, tl.load(POSR + r), r, tl.program_id(1), QW, KW, IW, INV, Q, _ptr(CP, s, KT),
                   _ptr(CP + N, s, VALUE_TYPE), _ptr(CP + 2 * N, s, KS_TYPE), _ptr(CP + 3 * N, s, VS_TYPE), IQ,
                   _ptr(CP + 4 * N, s, tl.bfloat16), eps, PW, NQ, NKV, HD, NI, IHD, HALF, BITS,
                   ROPE=rope, DELTA=delta, length=length, MODE=2 if VISION else 0, S1=S1, S2=S2,
                   ROPE_SCALE=ROPE_SCALE, CODEC=CODEC, STATUS=status, VALIDATE=PAIR or CODEC != 0, LAYER=LAYER,
                   K_BITS=K_BITS, K_CODEC=K_CODEC, V_BITS=V_BITS, V_CODEC=V_CODEC)


@triton.jit
def _pool_multi(CP, VP, P0, RS, W, INV, eps, N, DI: tl.constexpr, HALF: tl.constexpr, RATIO: tl.constexpr,
                VISION: tl.constexpr, S1: tl.constexpr = 11, S2: tl.constexpr = 10,
                ROPE_SCALE: tl.constexpr = 1.0):
    s = tl.program_id(0)
    rope, delta, length = CP, CP, 0
    if VISION:
        rope, delta = _ptr(VP, s, tl.int32), _ptr(VP + N, s, tl.int32)
        length = tl.load(VP + 2 * N + s).to(tl.int32)
    _pool_block(_ptr(CP + 4 * N, s, tl.bfloat16), _ptr(CP + 5 * N, s, tl.bfloat16), tl.load(P0 + s),
                tl.program_id(1), W, INV, eps, tl.load(RS + s), DI, HALF, RATIO,
                ROPE=rope, DELTA=delta, length=length, MODE=2 if VISION else 0, S1=S1, S2=S2,
                ROPE_SCALE=ROPE_SCALE)


@triton.jit
def _chunks_multi(Q, CP, POSR, SID, PO, PM, PL, IDS, NKR, N, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr,
                  G: tl.constexpr, CH: tl.constexpr, NCH: tl.constexpr, SCALE: tl.constexpr, IDW: tl.constexpr,
                  QSA: tl.constexpr, BITS: tl.constexpr, KT: tl.constexpr, RATIO: tl.constexpr, TOP: tl.constexpr,
                  CODEC: tl.constexpr = 0, K_BITS: tl.constexpr = -1, K_CODEC: tl.constexpr = 0,
                  V_BITS: tl.constexpr = -1, V_CODEC: tl.constexpr = 0, VT: tl.constexpr = None,
                  ROTOR_LOOKUP: tl.constexpr = False):
    r = tl.program_id(0)
    s = tl.load(SID + r)
    n = tl.load(POSR + r) + 1
    sparse = False
    if QSA:                                          # a row is sparse when its select would mark it (end past TOP)
        sparse = n // RATIO > TOP
        n = tl.where(sparse, tl.load(NKR + r), n)
    if K_BITS >= 0:
        KS_TYPE: tl.constexpr = tl.float32 if K_CODEC else tl.float16
        VS_TYPE: tl.constexpr = tl.float32 if V_CODEC else tl.float16
        _chunk_pair(Q, _ptr(CP, s, KT), _ptr(CP + N, s, VT), _ptr(CP + 2 * N, s, KS_TYPE),
                    _ptr(CP + 3 * N, s, VS_TYPE), n, sparse, r, tl.program_id(1), tl.program_id(2),
                    PO, PM, PL, IDS, H, HK, D, G, CH, NCH, SCALE, IDW, QSA, K_BITS, K_CODEC, V_BITS, V_CODEC, ROTOR_LOOKUP=ROTOR_LOOKUP)
    else:
        ST: tl.constexpr = tl.float32 if CODEC else tl.float16
        _chunk(Q, _ptr(CP, s, KT), _ptr(CP + N, s, KT), _ptr(CP + 2 * N, s, ST), _ptr(CP + 3 * N, s, ST),
               n, sparse, r, tl.program_id(1), tl.program_id(2), PO, PM, PL, IDS, H, HK, D, G, CH, NCH, SCALE, IDW,
               QSA, BITS, CODEC=CODEC)


@triton.jit
def _merge_multi(PO, PM, PL, POSR, OUT, NKR, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
                 CH: tl.constexpr, NCH: tl.constexpr, QSA: tl.constexpr, BITS: tl.constexpr, RATIO: tl.constexpr,
                 TOP: tl.constexpr, CODEC: tl.constexpr = 0,
                 V_BITS: tl.constexpr = -1, V_CODEC: tl.constexpr = 0):
    r = tl.program_id(0)
    n = tl.load(POSR + r) + 1
    if QSA:
        n = tl.where(n // RATIO > TOP, tl.load(NKR + r), n)
    if V_BITS >= 0:
        _merge_row(PO, PM, PL, OUT, n, r, tl.program_id(1), H, HK, D, G, CH, NCH, V_BITS, CODEC=V_CODEC)
    else:
        _merge_row(PO, PM, PL, OUT, n, r, tl.program_id(1), H, HK, D, G, CH, NCH, BITS, CODEC=CODEC)


class Step:
    """A step's row, stream and cache-pointer tables (the MTP head's with ``mtp``), read now: caches may move."""

    def __init__(self, w, segs: Sequence, mtp: bool, *, rotor_lookup: bool = False) -> None:
        self.rotor_lookup = rotor_lookup
        n, rows = len(segs), segs[-1][2]
        self.identity = segs[0][0].kv_identity
        self.pair = segs[0][0].kv_pair
        for st, _, _ in segs:
            if (st.kv_identity != self.identity or st.kv_pair != self.pair
                    or st.kv_working_policy != "stored-basis-native64-v1"
                    or st.kv_identity != (st.kv_pair.identity, st.kv_working_policy)):
                raise ValueError("a concurrent attention step needs matching ordered formats and supported working policy")
        layers = [l for l in w.layers if not l.linear] if not mtp else [w.mtp.layer]
        self.index = {layer.index: i for i, layer in enumerate(layers)}
        first = [st.mtp_len if mtp else st.pos for st, _, _ in segs]
        posr, sid = np.empty((rows,), np.int32), np.empty((rows,), np.int32)
        for s, ((_, a0, a1), p0) in enumerate(zip(segs, first)):
            posr[a0:a1] = p0 + np.arange(a1 - a0)
            sid[a0:a1] = s
        counts = [a1 - a0 for _, a0, a1 in segs]
        ptrs = np.empty((len(layers), PTRS, n), np.int64)
        for s, (st, _, _) in enumerate(segs):
            for i, layer in enumerate(layers):
                if mtp:
                    kc, ikc, pooled = st.mtp_kc, st.mtp_ikc, st.mtp_pooled
                else:
                    a = st.att_index[layer.index]
                    kc, ikc, pooled = st.kc[a], st.ikc[a], st.pooled[a]
                if kc.pair != self.pair:
                    raise ValueError("a concurrent attention step needs matching ordered cache format pairs")
                ptrs[i, :, s] = [kc.k.data_ptr(), kc.v.data_ptr(), kc.ks.data_ptr(), kc.vs.data_ptr(),
                                 ikc.data_ptr(), pooled.data_ptr()]
        dev = w.device
        ints = shared.to_device(np.concatenate([posr, sid, first, counts]).tolist(), torch.int32, dev)
        self.posr, self.sid = ints[:rows], ints[rows:2 * rows]
        self.first, self.counts = ints[2 * rows:2 * rows + n], ints[2 * rows + n:]
        self.ptrs = shared.to_device(ptrs.ravel().tolist(), torch.int64, dev).view(len(layers), PTRS * n)
        key, value = self.pair.key, self.pair.value
        self.k_bits, self.v_bits = (0 if key.bits == 16 else key.bits), (0 if value.bits == 16 else value.bits)
        self.native = self.pair.symmetric and key.codec == 0
        self.kt = tl.bfloat16 if key.bits == 16 else tl.uint8 if key.codec or key.bits == 4 else tl.int8
        self.vt = tl.bfloat16 if value.bits == 16 else tl.uint8 if value.codec or value.bits == 4 else tl.int8
        self.shader_args = ({} if self.native else dict(K_BITS=self.k_bits, K_CODEC=key.codec,
                                                        V_BITS=self.v_bits, V_CODEC=value.codec, VT=self.vt))
        self.merge_shader_args = ({} if self.native else dict(V_BITS=self.v_bits, V_CODEC=value.codec))
        if not self.native and any(st.kv_status is None for st, _, _ in segs):
            raise ValueError("mixed or RotorQuant Multi steps require each owning State status")
        self.status_ptrs = (shared.to_device([st.kv_status.data_ptr() for st, _, _ in segs], torch.int64, dev)
                            if not self.native else None)
        self.n, self.rows, self.segs = n, rows, list(segs)
        self.ends = [p0 + c for p0, c in zip(first, counts)]
        self.most = max(counts)
        self.vision = any(st.image_positions is not None for st, _, _ in segs)
        self.vision_ptrs = self.ptrs[0]
        if self.vision:
            vp = [[st.image_positions.data_ptr() if st.image_positions is not None else st.pos_dev.data_ptr(),
                   st.rope_delta_dev.data_ptr(), 0 if st.image_positions is None else st.image_positions.shape[0]]
                  for st, _, _ in segs]
            self.vision_ptrs = shared.to_device(np.asarray(vp, dtype=np.int64).T.ravel().tolist(), torch.int64, dev)


def layer(layer, w, b, step: Step, mtp: bool, scale: float) -> torch.Tensor:
    """Every stream's ``layer`` attention: prep, pool, sparse streams' own selects, chunks, merge -> b.attn_o[:R]."""

    c, a, sc = w.cfg, layer.attn, b.attn
    n, rows, cp = step.n, step.rows, step.ptrs[step.index[layer.index]]
    kb, vb, kt = step.k_bits, step.v_bits, step.kt
    pair, merge_pair = step.shader_args, step.merge_shader_args
    heads = c.heads + c.kv_heads + c.index_heads + 1
    sections = w.cfg.mrope_section
    _prep_multi[(rows, heads)](b.pa[:rows], step.posr, step.sid, cp, step.vision_ptrs,
                               a.q_scale, a.k_scale, a.iq_scale, w.inv_freq,
                               b.q, b.iq, c.eps, n, PW=b.pa.shape[1], NQ=c.heads, NKV=c.kv_heads, HD=c.head_dim,
                               NI=c.index_heads, IHD=c.index_dim, HALF=w.inv_freq.numel(), BITS=kb, KT=kt,
                               VISION=step.vision, S1=sections[1], S2=sections[2],
                               ROPE_SCALE=c.rope_attention_factor, CODEC=0, STATUS=step.status_ptrs,
                               LAYER=layer.index, **pair,
                               num_warps=2)
    top = sc.budget // sc.ratio
    if sc.qsa:
        _pool_multi[(n, step.most // sc.ratio + 2)](cp, step.vision_ptrs, step.first, step.counts,
                                                    a.ik_scale, w.inv_freq, c.eps, n,
                                                    DI=c.index_dim, HALF=w.inv_freq.numel(), RATIO=sc.ratio,
                                                    VISION=step.vision, S1=sections[1], S2=sections[2],
                                                    ROPE_SCALE=c.rope_attention_factor,
                                                    num_warps=1)
        for (st, a0, a1), end in zip(step.segs, step.ends):
            if end // sc.ratio > top:                # this stream has sparse rows: its own select
                _, _, pooled, pos, _ = _caches(layer, st, mtp)
                attn_mod.qsa_rows(b.iq[a0:a1], pooled, pos, _rows_from(sc, a0), a1 - a0, context=end)
    keys = max(step.ends)
    if sc.qsa:
        keys = min(keys, (top + 1) * sc.ratio - 1)
    chunks = min(sc.nch, triton.cdiv(keys, CHUNK))
    hk = c.kv_heads
    g = c.heads // hk
    _chunks_multi[(rows, hk, chunks)](b.q, cp, step.posr, step.sid, sc.po, sc.pm, sc.pl, sc.ids, sc.nk, n, H=c.heads,
                                      HK=hk, D=c.head_dim, G=g, CH=CHUNK, NCH=sc.nch, SCALE=scale, IDW=sc.idw,
                                      QSA=sc.qsa, BITS=kb, KT=kt, RATIO=sc.ratio, TOP=top, CODEC=0, ROTOR_LOOKUP=step.rotor_lookup, **pair, num_warps=4,
                                      num_stages=1)
    _merge_multi[(rows, hk)](sc.po, sc.pm, sc.pl, step.posr, b.attn_o, sc.nk, H=c.heads, HK=hk, D=c.head_dim, G=g,
                             CH=CHUNK, NCH=sc.nch, QSA=sc.qsa, BITS=vb, RATIO=sc.ratio, TOP=top, CODEC=0, **merge_pair,
                             num_warps=4)
    return b.attn_o[:rows]


def _caches(layer, st, mtp: bool) -> tuple:
    if mtp:
        return st.mtp_kc, st.mtp_ikc, st.mtp_pooled, st.mtp_pos, st.mtp_len
    ai = st.att_index[layer.index]
    return st.kc[ai], st.ikc[ai], st.pooled[ai], st.pos_dev, st.pos


def _rows_from(sc, a0: int) -> SimpleNamespace:
    """The scratch as seen by a launch whose row 0 is window row ``a0``."""

    return SimpleNamespace(ratio=sc.ratio, budget=sc.budget, nb=sc.nb, idw=sc.idw, scores=sc.scores[a0:],
                           ids=sc.ids[a0:], nk=sc.nk[a0:], sparse=sc.sparse[a0:])

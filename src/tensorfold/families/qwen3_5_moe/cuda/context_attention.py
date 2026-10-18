"""DFlash v1 block attention over preallocated drafter contexts, split over fixed key chunks.

Each stream's committed context lives in a store [layers, kv_heads, capacity, head_dim] (``dflash._Store``); a
block's queries attend every committed key and every row of their own block (v1 blocks are noncausal and the
drafter context is full: ``dflash.DFlash`` keeps every block inside its declared window, so no key is masked).

Context keys are split at committed rows ``c * CHUNK``. One kernel reduces each complete chunk except a stream's
last to (row max, row sum, unnormalised output) partials; a second folds those partials in chunk order, then that
last chunk's keys and the block's own keys, and normalises. The partition depends only on each key's committed row,
so a stream's result never depends on how its context was appended, on the other streams, or on its capacity.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..dflash_metadata import CONTEXT_CHUNK as CHUNK, QUERY_TILE, padded_block

KEY_TILE = 64       # context keys an online-softmax step reads
# Integer arguments that vary between launches of one block length (layer, rows, chunk count) are not value-
# specialised: each kernel compiles once per block length instead of once per layer 0 / layer 1 / other, and their
# address multiples come from the constexpr D and BM factors either way (same PTX loads, stores and mma).
_RUNTIME_INTS = ("layer", "R", "PARTS")


@triton.jit
def _query_tile(Q, j, g, tile, R, G: tl.constexpr, L: tl.constexpr, LP: tl.constexpr, D: tl.constexpr,
                BM: tl.constexpr):
    rows = tile * BM + tl.arange(0, BM)
    head = g * G + rows // LP
    r = rows % LP
    live = (r < L) & (rows < G * LP)
    d = tl.arange(0, D)
    qrow = (j * L + r).to(tl.int64)
    q = tl.load(Q + head[:, None].to(tl.int64) * R * D + qrow[:, None] * D + d[None, :], mask=live[:, None],
                other=0.0)
    return q, head, live, qrow


@triton.jit
def _context(META, j, g, layer, HKV: tl.constexpr, D: tl.constexpr):
    """Stream j's layer-``layer`` key/value rows of KV head g, and its committed length."""
    capacity = tl.load(META + 4 * j + 2)
    s = tl.load(META + 4 * j + 3)
    offset = (layer * HKV + g).to(tl.int64) * capacity * D
    kc = tl.multiple_of(tl.load(META + 4 * j).to(tl.pointer_type(tl.bfloat16)), 16) + offset
    vc = tl.multiple_of(tl.load(META + 4 * j + 1).to(tl.pointer_type(tl.bfloat16)), 16) + offset
    return kc, vc, s


@triton.jit
def _keys(q, kc, vc, start, stop, s, m_i, l_i, acc, scale, D: tl.constexpr, BN: tl.constexpr, MASKED: tl.constexpr):
    """Online softmax over context keys [start, stop) (``MASKED``: only keys below s exist)."""
    d = tl.arange(0, D)
    for n0 in range(start, stop, BN):
        kidx = n0 + tl.arange(0, BN)
        if MASKED:
            inside = kidx < s
            k = tl.load(kc + kidx[None, :].to(tl.int64) * D + d[:, None], mask=inside[None, :], other=0.0)
        else:
            k = tl.load(kc + kidx[None, :].to(tl.int64) * D + d[:, None])
        qk = tl.dot(q, k) * scale
        if MASKED:
            qk = tl.where(inside[None, :], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        if MASKED:
            v = tl.load(vc + kidx[:, None].to(tl.int64) * D + d[None, :], mask=inside[:, None], other=0.0)
        else:
            v = tl.load(vc + kidx[:, None].to(tl.int64) * D + d[None, :])
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    return m_i, l_i, acc


@triton.jit(do_not_specialize=_RUNTIME_INTS)
def _partials(Q, META, PM, PL, PO, scale, layer, R, PARTS,
              G: tl.constexpr, HKV: tl.constexpr, L: tl.constexpr, LP: tl.constexpr, D: tl.constexpr,
              BN: tl.constexpr, CH: tl.constexpr, BM: tl.constexpr, TILES: tl.constexpr):
    """Program (stream x chunk, kv head, query tile): one complete chunk that is not the stream's last."""
    j = tl.program_id(0) // PARTS
    c = tl.program_id(0) % PARTS
    g = tl.program_id(1)
    tile = tl.program_id(2)
    kc, vc, s = _context(META, j, g, layer, HKV, D)
    if c >= (s + CH - 1) // CH - 1:
        return
    q = _query_tile(Q, j, g, tile, R, G, L, LP, D, BM)[0]
    m_i = tl.full((BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    m_i, l_i, acc = _keys(q, kc, vc, c * CH, c * CH + CH, s, m_i, l_i, acc, scale, D, BN, False)
    slot = ((j * HKV + g) * PARTS + c).to(tl.int64) * (TILES * BM) + tile * BM + tl.arange(0, BM)
    tl.store(PM + slot, m_i)
    tl.store(PL + slot, l_i)
    tl.store(PO + slot[:, None] * D + tl.arange(0, D)[None, :], acc)


@triton.jit(do_not_specialize=_RUNTIME_INTS)
def _finish(Q, KB, VB, META, PM, PL, PO, O, scale, layer, R, PARTS,
            G: tl.constexpr, HKV: tl.constexpr, L: tl.constexpr, LP: tl.constexpr, D: tl.constexpr,
            BN: tl.constexpr, CH: tl.constexpr, BM: tl.constexpr, TILES: tl.constexpr):
    """Program (stream, kv head, query tile): partials in chunk order, the last chunk, the block; normalise."""
    j = tl.program_id(0)
    g = tl.program_id(1)
    tile = tl.program_id(2)
    kc, vc, s = _context(META, j, g, layer, HKV, D)
    q, head, live, qrow = _query_tile(Q, j, g, tile, R, G, L, LP, D, BM)
    d = tl.arange(0, D)
    m_i = tl.full((BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    chunks = (s + CH - 1) // CH
    base = ((j * HKV + g) * PARTS).to(tl.int64) * (TILES * BM) + tile * BM + tl.arange(0, BM)
    for c in range(0, chunks - 1):
        slot = base + c * (TILES * BM)
        m_c = tl.load(PM + slot)
        m_new = tl.maximum(m_i, m_c)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_c - m_new)
        l_i = l_i * alpha + tl.load(PL + slot) * beta
        acc = acc * alpha[:, None] + tl.load(PO + slot[:, None] * D + d[None, :]) * beta[:, None]
        m_i = m_new
    if chunks > 0:
        m_i, l_i, acc = _keys(q, kc, vc, (chunks - 1) * CH, s, s, m_i, l_i, acc, scale, D, BN, True)
    cols = tl.arange(0, LP)
    krow = (j * L + cols).to(tl.int64)
    in_block = cols < L
    kb = tl.load(KB + g.to(tl.int64) * R * D + krow[None, :] * D + d[:, None], mask=in_block[None, :], other=0.0)
    qk = tl.dot(q, kb) * scale
    qk = tl.where(in_block[None, :], qk, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(qk, 1))
    p = tl.exp(qk - m_new[:, None])
    alpha = tl.exp(m_i - m_new)
    l_i = l_i * alpha + tl.sum(p, 1)
    vb = tl.load(VB + g.to(tl.int64) * R * D + krow[:, None] * D + d[None, :], mask=in_block[:, None], other=0.0)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vb)
    out = (acc / l_i[:, None]).to(tl.bfloat16)
    tl.store(O + qrow[:, None] * (G * HKV * D) + head[:, None] * D + d[None, :], out, mask=live[:, None])


class Round:
    """One proposal round's launch state, shared by every drafter layer: the streams' context table (key store,
    value store, capacity, committed length) as one int64 device copy, and the partial scratch for the most chunks
    any stream has (reused layer after layer; each layer's finish has read it before the next writes)."""

    def __init__(self, stores, lengths, *, heads, kv_heads, dim, length, device):
        if not stores or len(stores) != len(lengths) or heads % kv_heads or length < 1:
            raise ValueError("context attention needs one store and committed length per stream")
        flat = []
        for store, n in zip(stores, lengths):
            flat += [store.k.data_ptr(), store.v.data_ptr(), store.k.shape[2], n]
        self.table = torch.tensor(flat, dtype=torch.int64).pin_memory().to(device, non_blocking=True)
        self.streams, self.heads, self.kv_heads, self.dim, self.length = len(stores), heads, kv_heads, dim, length
        self.group, self.padded = heads // kv_heads, padded_block(length)
        self.tiles = triton.cdiv(self.group * self.padded, QUERY_TILE)
        self.parts = max(0, max(-(-n // CHUNK) for n in lengths) - 1)
        if self.parts:
            slots = self.streams * kv_heads * self.parts * self.tiles * QUERY_TILE
            self.pm = torch.empty(slots, dtype=torch.float32, device=device)
            self.pl = torch.empty_like(self.pm)
            self.po = torch.empty((slots, dim), dtype=torch.float32, device=device)
        else:       # no partials: never read, but FP32 like the scratch so _finish keeps one compiled signature
            self.pm = self.pl = self.po = torch.empty(0, dtype=torch.float32, device=device)


def block_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, launch: Round, layer: int,
                    scale: float) -> torch.Tensor:
    """Contiguous CUDA bf16 q [H, S*L, D] and block k/v [Hkv, S*L, D] of ``launch``'s streams -> [S*L, H*D] bf16."""
    heads, rows, dim = q.shape
    kv_heads, length = launch.kv_heads, launch.length
    if ((heads, rows, dim) != (launch.heads, launch.streams * length, launch.dim) or k.shape != (kv_heads, rows, dim)
            or v.shape != k.shape or q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16
            or v.dtype != torch.bfloat16 or not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous())):
        raise ValueError("context attention needs contiguous bf16 q [H, S*L, D], k/v [Hkv, S*L, D] of its round")
    out = torch.empty((rows, heads * dim), dtype=torch.bfloat16, device=q.device)
    parts, tiles = launch.parts, launch.tiles
    meta = dict(G=launch.group, HKV=kv_heads, L=length, LP=launch.padded, D=dim, BN=KEY_TILE, CH=CHUNK, BM=QUERY_TILE,
                TILES=tiles, num_warps=8, num_stages=2)
    if parts:
        _partials[(launch.streams * parts, kv_heads, tiles)](q, launch.table, launch.pm, launch.pl, launch.po, scale,
                                                             layer, rows, parts, **meta)
    _finish[(launch.streams, kv_heads, tiles)](q, k, v, launch.table, launch.pm, launch.pl, launch.po, out, scale,
                                               layer, rows, max(parts, 1), **meta)
    return out

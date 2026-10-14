"""Flash Next's group-32 4-bit matmuls; ordered groups and fixed K slices summed in order keep a row's bits its own."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from tensorfold.cuda import pdl
from tensorfold.cuda.pdl import wait as pdl_wait

from tensorfold.cuda.kernels import qmm as shared

BN = 64                   # columns per stored tile
GS = 32                   # inputs per quantization group


@triton.jit
def _deq(words, shifts, ROWS: tl.constexpr):
    """[ROWS, 4] int32 words -> [ROWS, 32] bf16 operand, q in 0..15 (exact in bf16)."""

    q = (words[:, :, None] >> shifts[None, None, :]) & 0xF
    return tl.reshape(q, (ROWS, 32)).to(tl.bfloat16)


@dataclass
class Q4:
    """4-bit group-32 [n, k]: "frag" for the shared lane matmul, "tiled" ([N/BN, K/32, BN, 4] words) for HC kernels."""

    weight: torch.Tensor
    scales: torch.Tensor
    biases: torch.Tensor
    n: int
    k: int
    layout: str = "frag"
    gs: int = GS

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.weight, self.scales, self.biases))


def tile_words(words: torch.Tensor) -> torch.Tensor:
    """MLX (N, K/8) words -> [N/BN][K/32][BN][4] (N padded to BN with zeros). Works on stacked experts."""

    *lead, n, k8 = words.shape
    npad = -(-n // BN) * BN
    if npad != n:
        pad = words.new_zeros((*lead, npad - n, k8))
        words = torch.cat([words, pad], dim=-2)
    out = words.reshape(*lead, npad // BN, BN, k8 // 4, 4)
    nd = len(lead)
    perm = list(range(nd)) + [nd, nd + 2, nd + 1, nd + 3]
    return out.permute(*perm).contiguous()


def untile_words(tiled: torch.Tensor, n: int) -> torch.Tensor:
    *lead, t, kg, bn, four = tiled.shape
    nd = len(lead)
    perm = list(range(nd)) + [nd, nd + 2, nd + 1, nd + 3]
    return tiled.permute(*perm).reshape(*lead, t * bn, kg * four)[..., :n, :].contiguous()


def make_q4(weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, layout: str = "frag") -> Q4:
    """From the checkpoint's arrays: weight (N, K/8) uint32 or int32, scales/biases (N, K/32) bf16."""

    w = weight.view(torch.int32) if weight.dtype != torch.int32 else weight
    n, k8 = w.shape
    if layout == "frag":
        p = shared.pack(w, scales, biases, GS)
        return Q4(p.weight, p.scales, p.biases, n, k8 * 8)
    return Q4(tile_words(w), scales.t().contiguous(), biases.t().contiguous(), n, k8 * 8, "tiled")


def stack_q4(parts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], layout: str = "frag") -> Q4:
    """Rows of several (weight, scales, biases) of the same K stacked in order, then packed."""

    w = torch.cat([p[0].view(torch.int32) if p[0].dtype != torch.int32 else p[0] for p in parts])
    s = torch.cat([p[1] for p in parts])
    b = torch.cat([p[2] for p in parts])
    return make_q4(w, s, b, layout)


def to_mlx(q: Q4) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The stored MLX layout again: (N, K/8) words, (N, K/32) scales and biases."""

    if q.layout == "frag":
        return shared.unpack(q)
    return untile_words(q.weight, q.n), q.scales.t().contiguous(), q.biases.t().contiguous()


def dequantize(words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """Reference: MLX (N, K/8) words -> (N, K) fp32 values s * q + b (group 32)."""

    k8 = words.shape[-1]
    w = words.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, device=words.device, dtype=torch.int64) * 4
    q = ((w[..., None] >> shifts) & 0xF).reshape(*words.shape[:-1], k8 * 8).to(torch.float32)
    s = scales.to(torch.float32).repeat_interleave(GS, dim=-1)
    b = biases.to(torch.float32).repeat_interleave(GS, dim=-1)
    return q * s + b


def dequantize_q4(q: Q4) -> torch.Tensor:
    return dequantize(*to_mlx(q))


def split_k(n: int, k: int, target: int = 160) -> int:
    """K slices for an (n, k) weight: a function of the shape only (never of the row count)."""

    tiles = -(-n // BN)
    groups = k // GS
    sk = 1
    while sk < 32 and tiles * sk < target and groups % (sk * 2) == 0 and groups // (sk * 2) >= 8:
        sk *= 2
    return sk


def bucket(m: int) -> int:
    """Rows a program takes: 16 to 128, then tiles of 128 (a row's bits never depend on its tile)."""

    for b in (16, 32, 64, 128):
        if m <= b:
            return b
    return 128


@triton.jit
def _group_sums(X, XS, x_stride, K: tl.constexpr, GB: tl.constexpr, PDL: tl.constexpr = False):
    pdl_wait(PDL)
    m = tl.program_id(0)
    gb = tl.program_id(1)
    KG: tl.constexpr = K // 32
    g = gb * GB + tl.arange(0, GB)
    k = tl.arange(0, 32)
    ok = g < KG
    x = tl.load(X + m * x_stride + g[:, None] * 32 + k[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
    tl.store(XS + m * KG + g, tl.sum(x, axis=1), mask=ok)


def group_sums(x: torch.Tensor) -> torch.Tensor:
    """(M, K) bf16 (rows may be strided) -> (M, K/32) fp32 sums of each 32-input group."""

    m, k = x.shape
    kg = k // GS
    xs = torch.empty((m, kg), dtype=torch.float32, device=x.device)
    gb = 32
    _group_sums[(m, triton.cdiv(kg, gb))](x, xs, x.stride(0), K=k, GB=gb, num_warps=2, PDL=pdl.ON, launch_pdl=pdl.ON)
    return xs


@triton.jit
def _sbx(S, B, XS, g, rn, n_ok, rm, m_ok, N: tl.constexpr, KG: tl.constexpr):
    """Group g's scales and biases (stored dtype) for the program's columns and its rows' group sums."""

    s = tl.load(S + g * N + rn, mask=n_ok, other=0.0)
    b = tl.load(B + g * N + rn, mask=n_ok, other=0.0)
    xs = tl.load(XS + rm * KG + g, mask=m_ok, other=0.0)
    return s, b, xs


@triton.jit
def _qmm(X, XS, W, S, B, OUT, PART, M, x_stride,
         N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
         BLOCK_N: tl.constexpr, PD: tl.constexpr, F32: tl.constexpr, SBN: tl.constexpr, PDL: tl.constexpr = False):
    """A K slice's groups one a loop step, summed in order. Triton pipelines the words and inputs (async copies;
    Triton 3.9 leaves the PD 2 loop unpipelined) but would load the scales, biases and group sums synchronously
    behind the dot's shared-memory barriers, one memory latency a step: with PD 1 or 2 they are loaded PD groups
    ahead into loop-carried registers (the same values; a slice's last steps reload its last group, in bounds and
    unused). The stored tile's rows past N (``tile_words``' zero padding) are masked to the same zero words."""

    pdl_wait(PDL)
    KG: tl.constexpr = K // 32
    PER: tl.constexpr = KG // SK
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, 32)
    rw = tl.arange(0, 4)
    shifts = tl.arange(0, 8) * 4
    m_ok = rm < M
    n_ok = rn < N
    SUB: tl.constexpr = SBN // BLOCK_N                  # program tiles per stored tile (BLOCK_N divides SBN)
    tile = W + (pid_n // SUB) * (KG * SBN * 4)
    local = (pid_n % SUB) * BLOCK_N + tl.arange(0, BLOCK_N)     # stored rows: local + (pid_n // SUB) * SBN == rn
    g0 = pid_s * PER
    last = g0 + PER - 1                                 # PER >= 1: SK divides the K/32 groups (matmul checks)
    if PD >= 1:
        s_n, b_n, xs_n = _sbx(S, B, XS, g0, rn, n_ok, rm, m_ok, N, KG)
    if PD == 2:
        s_n2, b_n2, xs_n2 = _sbx(S, B, XS, tl.minimum(g0 + 1, last), rn, n_ok, rm, m_ok, N, KG)
    acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for i in range(PER):
        g = g0 + i
        if PD == 1:
            s, b, xs = s_n, b_n, xs_n
            s_n, b_n, xs_n = _sbx(S, B, XS, tl.minimum(g + 1, last), rn, n_ok, rm, m_ok, N, KG)
        elif PD == 2:
            s, b, xs = s_n, b_n, xs_n
            s_n, b_n, xs_n = s_n2, b_n2, xs_n2
            s_n2, b_n2, xs_n2 = _sbx(S, B, XS, tl.minimum(g + 2, last), rn, n_ok, rm, m_ok, N, KG)
        if N % SBN != 0:
            words = tl.load(tile + g * (SBN * 4) + local[:, None] * 4 + rw[None, :], mask=n_ok[:, None], other=0)
        else:
            words = tl.load(tile + g * (SBN * 4) + local[:, None] * 4 + rw[None, :])
        x = tl.load(X + rm[:, None] * x_stride + (g * 32 + rk)[None, :], mask=m_ok[:, None], other=0.0)
        q = _deq(words, shifts, BLOCK_N)
        p = tl.dot(x, tl.trans(q))
        if PD == 0:
            s, b, xs = _sbx(S, B, XS, g, rn, n_ok, rm, m_ok, N, KG)
        acc = acc + p * s.to(tl.float32)[None, :] + xs[:, None] * b.to(tl.float32)[None, :]
    out_mask = m_ok[:, None] & n_ok[None, :]
    if SK == 1:
        if F32:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc, mask=out_mask)
        else:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)


@triton.jit
def _reduce(PART, OUT, total, SK: tl.constexpr, BLOCK: tl.constexpr, F32: tl.constexpr, PDL: tl.constexpr = False):
    pdl_wait(PDL)
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = offs < total
    acc = tl.load(PART + offs, mask=ok, other=0.0)
    for s in tl.static_range(1, SK):
        acc = acc + tl.load(PART + s * total + offs, mask=ok, other=0.0)
    if F32:
        tl.store(OUT + offs, acc, mask=ok)
    else:
        tl.store(OUT + offs, acc.to(tl.bfloat16), mask=ok)


# The tiled kernel's launch: (groups whose scales/biases/group sums are loaded ahead, warps, stages, columns per
# program), measured on the hyper-connection shapes on dev-01 (sm_120); every choice gives the same bits. By row
# bucket, and for those shapes by window rows: the smallest windows run narrow 2-warp programs without pipelining.
CONFIG = {16: (2, 4, 3, 64), 32: (1, 4, 3, 64), 64: (1, 4, 2, 64), 128: (0, 8, 2, 64)}
TUNED = {                   # (n, k): ((up to rows, launch), ...): the first that covers the window
    (324, 10240): ((4, (1, 2, 1, 32)), (16, (2, 2, 3, 64)), (32, (0, 2, 1, 64))),
    (320, 10240): ((4, (1, 2, 1, 32)), (16, (1, 4, 1, 64))),
    (10240, 320): ((4, (1, 2, 1, 32)), (16, (1, 4, 1, 64)), (32, (2, 2, 3, 64))),
}


def launch_for(n: int, k: int, m: int) -> tuple[int, int, int, int]:
    """The tiled kernel's (groups loaded ahead, warps, stages, columns per program) for m rows of an (n, k) matrix."""

    for rows, got in TUNED.get((n, k), ()):
        if m <= rows:
            return got
    return CONFIG[bucket(m)]


# K slices are fixed per shape to preserve summation order.
SPLITS = {
    (324, 10240): 32,         # hyper-connection down + inject
    (320, 10240): 32,         # a mixer's down
    (10240, 320): 1,          # hyper-connection up
    (16480, 2560): 1,         # DeltaNet q/k/v, z, b, a
    (2560, 6144): 8,          # DeltaNet / attention output
    (13952, 2560): 1,         # attention q|gate, k, v, indexer
    (248320, 2560): 1,        # head
}


def split_for(n: int, k: int) -> int:
    """The K slices of an (n, k) matrix: the shape constant when there is one, else ``split_k``."""

    return SPLITS.get((n, k)) or split_k(n, k)


# The frag lane matmul's launch by chip (major, minor, SMs) and shape, where measured: per (n, k), (up to rows, block
# of TENSORFOLD_QMM_LANE_TILES, clusters allowed), the first that covers the window; else the row bucket's block with
# clusters. Blocks and the cluster/reduce-kernel slice sum never change a row's bits (fixed K slices, groups in order).
# dev-01 (RTX PRO 2000, sm_120, 34 SMs; data/qmm micro, DRAM-cold weights, vs the bucket blocks): 128 columns on 4
# warps, 3 stages to 16 rows (0.90-0.99x the 64-column blocks' time; out projections 0.88x), 32 x 128 on 4 warps to
# 32 rows (0.80-0.89x at 17-20 rows; to 32, the same grids); the draft head (fp32 out) 2 stages (0.93x at 1-4 rows);
# the 2560 x 2560 K split staged instead of clusters (at most 17 clusters of 4 fit at a time: 0.53-0.71x to 16 rows,
# with 32-row blocks 0.45-0.67x at 20-80 rows; to 96, the grids of 80); the head (fp32 out) 32 x 128 at 17-32 rows
# (0.83x at 20 rows), to 16 rows the bucket's block (the best others 0.986-0.999x at 1, 4 and 5 rows, one run).
LANE = {
    (12, 0, 34): {
        (16480, 2560): ((16, 4, True), (32, 5, True)),          # DeltaNet q/k/v, z, b, a
        (13952, 2560): ((16, 4, True), (32, 5, True)),          # attention q|gate, k, v, indexer
        (10240, 2560): ((16, 4, True), (32, 5, True)),          # PLE key
        (2560, 6144): ((16, 4, True), (32, 5, True)),           # DeltaNet / attention output (unreduced slices)
        (2560, 2560): ((16, 1, False), (96, 5, False)),         # PLE value, MTP fc_e / fc_h
        (79591, 2560): ((16, 6, True),),                        # draft head
        (248320, 2560): ((16, 1, True), (32, 5, True)),         # head: the bucket's block to 16 rows
    },
}


def lane_launch(n: int, k: int, m: int, chip: tuple[int, int, int]) -> tuple[int, bool]:
    """(block, clusters allowed) for m rows of an (n, k) frag matrix on ``chip``: (0, True) is the row bucket's."""

    for rows, tile, clusters in LANE.get(chip, {}).get((n, k), ()):
        if m <= rows:
            return tile, clusters
    return 0, True


def matmul(x: torch.Tensor, q: Q4, xs: torch.Tensor | None = None, *, out: torch.Tensor | None = None,
           f32: bool = False, sk: int | None = None, part: torch.Tensor | None = None,
           pd: int | None = None, num_warps: int | None = None, num_stages: int | None = None,
           block_n: int | None = None, reduce: bool = True) -> torch.Tensor:
    """x (M, K) bf16 (rows may be strided) @ q.T -> (M, N) bf16 (or fp32 sums with ``f32``). ``reduce=False`` with a
    split K returns the unreduced fp32 slices [SK, M, N] (the caller sums them in slice order). ``sk`` (whole groups a
    slice) replaces the shape's K split, and with it the order of the sums. ``pd``, ``num_warps``, ``num_stages`` and
    ``block_n`` (16, 32 or 64: dividing ``BN``) override the tiled kernel's launch, never its bits; the "frag" layout
    ignores them. The tiled kernel's arguments are checked before any launch (ValueError)."""

    if q.layout == "frag":
        if x.shape[1] != q.k or x.stride(1) != 1:
            raise ValueError(f"matmul: x {tuple(x.shape)} does not match K={q.k}")
        tile, clusters = lane_launch(q.n, q.k, x.shape[0], shared.chip(x.device.index))
        return shared.matmul(x, q, group_sums(x) if xs is None else xs, sk=int(sk) if sk else split_for(q.n, q.k),
                             f32=f32, out=out, part=part, reduce=reduce, tile=tile, clusters=clusters)
    m, k = x.shape
    if k != q.k or x.stride(1) != 1:
        raise ValueError(f"matmul: x {tuple(x.shape)} does not match K={q.k}")
    bm = bucket(m)
    c_pd, c_warps, c_stages, c_bn = launch_for(q.n, q.k, m)
    pd = c_pd if pd is None else int(pd)
    bn = block_n or c_bn
    sk = int(sk) if sk else split_for(q.n, q.k)
    kg = k // GS
    if pd not in (0, 1, 2):
        raise ValueError(f"matmul: pd {pd} must be 0, 1 or 2")
    if bn < 16 or BN % bn:              # program tiles inside one stored tile, each at least a dot's 16 columns
        raise ValueError(f"matmul: block_n {bn} must divide BN={BN} and be at least 16")
    if not 1 <= sk <= kg or kg % sk:    # no group dropped or past K; the loads ahead stay inside a slice
        raise ValueError(f"matmul: {sk} K slices do not split {kg} groups into whole groups")
    if out is None:
        out = torch.empty((m, q.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    elif out.shape != (m, q.n) or not out.is_contiguous():
        raise ValueError(f"matmul: out {tuple(out.shape)} must be a contiguous ({m}, {q.n})")
    if sk > 1 and part is None:
        part = torch.empty((sk, m, q.n), dtype=torch.float32, device=x.device)
    if sk > 1 and part.numel() < sk * m * q.n:
        raise ValueError("matmul: split-K scratch too small")
    if xs is None:
        xs = group_sums(x)
    grid = (triton.cdiv(m, bm), triton.cdiv(q.n, bn), sk)
    _qmm[grid](x, xs, q.weight, q.scales, q.biases, out, part if sk > 1 else out, m, x.stride(0),
               N=q.n, K=k, SK=sk, BM=bm, BLOCK_N=bn, PD=pd, F32=f32, SBN=BN,
               num_warps=num_warps or c_warps, num_stages=num_stages or c_stages, PDL=pdl.ON, launch_pdl=pdl.ON)
    if sk > 1 and reduce:
        total = m * q.n
        _reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, F32=f32, num_warps=4,
                                             PDL=pdl.ON, launch_pdl=pdl.ON)
    return out if (sk == 1 or reduce) else part[:sk * m * q.n].view(sk, m, q.n)


def prefill_matmul(x: torch.Tensor, q: Q4, xs: torch.Tensor | None = None, *, out: torch.Tensor | None = None,
                   f32: bool = False, part: torch.Tensor | None = None, reduce: bool = True) -> torch.Tensor:
    """A prompt chunk's matmul ("frag": shared prefill matmul, "tiled": ``matmul``); row-invariant at any chunk."""

    if q.layout == "frag":
        return shared.prefill_matmul(x, q, f32=f32, out=out)
    return matmul(x, q, xs, out=out, f32=f32, part=part, reduce=reduce)

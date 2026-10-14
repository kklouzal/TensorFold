"""Decode hyper-connection read-out bits against the kernels it replaced (oracle, below: verbatim copies of 18a7c448's
tiled matmul ``_qmm`` with its launch tables, and of the fused decode read-out ``_qmm_hcdown`` / ``_qmm_upmix`` that
windows of up to 16 rows took): every output bit-identical, rows 1..20 (decode windows) and each row bucket, both
down shapes (324 with the inject rows, 320) and the up shape, on normal inputs and on adversarial ones (values over
many binades, signed zeros) where any change of operation order, rounding point or reduction tree shows."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from tensorfold.cuda import pdl  # noqa: E402
from tensorfold.cuda.pdl import wait as pdl_wait  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import glue, qmm  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.qmm import BN, GS, Q4  # noqa: E402

DEV = "cuda"
S, D, LOW = 4, 2560, 320
SENTINEL = float.fromhex("0x1.5555p+100")        # outputs start here: a stray or a missing write shows


# ---- oracle: 18a7c448 (qwen4_exp/cuda/qmm.py), verbatim but for the wrappers' names, the tiled-only matmul and the
# K split read from SHAPES16 (split_for's value at these shapes); qmm.bucket, qmm.group_sums and qmm._reduce are
# production's, the same as at 18a7c448 ----

@triton.jit
def _deq(words, shifts, ROWS: tl.constexpr):
    """[ROWS, 4] int32 words -> [ROWS, 32] bf16 operand, q in 0..15 (exact in bf16)."""

    q = (words[:, :, None] >> shifts[None, None, :]) & 0xF
    return tl.reshape(q, (ROWS, 32)).to(tl.bfloat16)


def gpi_for(per: int, want: int) -> int:
    for g in (want, 8, 4, 2, 1):
        if g <= want and per % g == 0:
            return g
    return 1


@triton.jit
def _qmm(X, XS, W, S, B, OUT, PART, M, x_stride,
         N: tl.constexpr, K: tl.constexpr, SK: tl.constexpr, BM: tl.constexpr,
         BLOCK_N: tl.constexpr, GPI: tl.constexpr, F32: tl.constexpr, SBN: tl.constexpr, PDL: tl.constexpr = False):
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
    SUB: tl.constexpr = SBN // BLOCK_N                  # program tiles per stored tile
    tile = W + (pid_n // SUB) * (KG * SBN * 4)
    local = (pid_n % SUB) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for i in range(PER // GPI):
        for j in tl.static_range(GPI):
            g = pid_s * PER + i * GPI + j
            words = tl.load(tile + g * (SBN * 4) + local[:, None] * 4 + rw[None, :])
            x = tl.load(X + rm[:, None] * x_stride + (g * 32 + rk)[None, :], mask=m_ok[:, None], other=0.0)
            q = _deq(words, shifts, BLOCK_N)
            p = tl.dot(x, tl.trans(q))
            s = tl.load(S + g * N + rn, mask=n_ok, other=0.0).to(tl.float32)
            b = tl.load(B + g * N + rn, mask=n_ok, other=0.0).to(tl.float32)
            xs = tl.load(XS + rm * KG + g, mask=m_ok, other=0.0)
            acc = acc + p * s[None, :] + xs[:, None] * b[None, :]
    out_mask = m_ok[:, None] & n_ok[None, :]
    if SK == 1:
        if F32:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc, mask=out_mask)
        else:
            tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)


# (groups per unrolled step, warps, stages) by row bucket: every choice gives the same bits
CONFIG = {16: (4, 4, 3), 32: (2, 4, 3), 64: (2, 4, 2), 128: (1, 8, 2)}

# K slices are fixed per shape to preserve summation order; other launch settings do not change bits.
SHAPES16 = {
    (324, 10240): (32, 2, 4, 3, 64),         # hyper-connection down + inject
    (320, 10240): (32, 2, 4, 3, 64),         # a mixer's down
    (10240, 320): (1, 2, 4, 3, 64),          # hyper-connection up
    (16480, 2560): (1, 1, 4, 3, 64),         # DeltaNet q/k/v, z, b, a
    (2560, 6144): (8, 2, 4, 2, 64),          # DeltaNet / attention output
    (13952, 2560): (1, 2, 4, 3, 64),         # attention q|gate, k, v, indexer
    (248320, 2560): (1, 4, 4, 2, 64),        # head
}


def oracle_matmul(x: torch.Tensor, q: Q4, xs: torch.Tensor | None = None, *, out: torch.Tensor | None = None,
           f32: bool = False, sk: int | None = None, part: torch.Tensor | None = None,
           gpi: int | None = None, num_warps: int | None = None, num_stages: int | None = None,
           block_n: int | None = None, reduce: bool = True) -> torch.Tensor:
    """x (M, K) bf16 (rows may be strided) @ q.T -> (M, N) bf16 (or fp32 sums with ``f32``). ``reduce=False`` with a split K returns the unreduced fp32 slices [SK, M, N] (the caller sums them in slice order)."""

    m, k = x.shape
    if k != q.k or x.stride(1) != 1:
        raise ValueError(f"matmul: x {tuple(x.shape)} does not match K={q.k}")
    bm = qmm.bucket(m)
    c_gpi, c_warps, c_stages = CONFIG[bm]
    c_bn = BN
    tuned = SHAPES16.get((q.n, q.k)) if bm == 16 else None
    if tuned is not None:
        _, c_gpi, c_warps, c_stages, c_bn = tuned
    if xs is None:
        xs = qmm.group_sums(x)
    sk = int(sk) if sk else SHAPES16[(q.n, q.k)][0]
    per = (k // GS) // sk
    g = gpi_for(per, gpi or c_gpi)
    if out is None:
        out = torch.empty((m, q.n), dtype=torch.float32 if f32 else torch.bfloat16, device=x.device)
    elif out.shape != (m, q.n) or not out.is_contiguous():
        raise ValueError(f"matmul: out {tuple(out.shape)} must be a contiguous ({m}, {q.n})")
    if sk > 1 and part is None:
        part = torch.empty((sk, m, q.n), dtype=torch.float32, device=x.device)
    if sk > 1 and part.numel() < sk * m * q.n:
        raise ValueError("matmul: split-K scratch too small")
    bn = block_n or c_bn
    grid = (triton.cdiv(m, bm), triton.cdiv(q.n, bn), sk)
    _qmm[grid](x, xs, q.weight, q.scales, q.biases, out, part if sk > 1 else out, m, x.stride(0),
               N=q.n, K=k, SK=sk, BM=bm, BLOCK_N=bn, GPI=g, F32=f32, SBN=BN,
               num_warps=num_warps or c_warps, num_stages=num_stages or c_stages, PDL=pdl.ON, launch_pdl=pdl.ON)
    if sk > 1 and reduce:
        total = m * q.n
        qmm._reduce[(triton.cdiv(total, 1024),)](part, out, total, SK=sk, BLOCK=1024, F32=f32, num_warps=4,
                                             PDL=pdl.ON, launch_pdl=pdl.ON)
    return out if (sk == 1 or reduce) else part[:sk * m * q.n].view(sk, m, q.n)


@triton.jit
def _bsig(x):
    return (1.0 / (1.0 + tl.exp(-x))).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _qmm_hcdown(H, PSS, SCALE, NORMED, W, S, B, OUT, PART, M, eps,
                N: tl.constexpr, K: tl.constexpr, D: tl.constexpr, NC: tl.constexpr, SS: tl.constexpr,
                SK: tl.constexpr, BM: tl.constexpr, BLOCK_N: tl.constexpr, GPI: tl.constexpr, SBN: tl.constexpr,
                PDL: tl.constexpr = False):
    """Compute the down projection with glue.hc_normed's bf16 rounding and ordered partial sums, keeping each K slice within one stream and writing normed values once for the up mix."""

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
    st = (pid_s * PER * 32) // D
    total = tl.zeros((BM,), dtype=tl.float32)
    for c in range(NC):
        total += tl.load(PSS + (rm * NC + c) * SS + st, mask=m_ok, other=0.0)
    rinv = 1.0 / tl.sqrt(total / D + eps)
    SUB: tl.constexpr = SBN // BLOCK_N
    tile = W + (pid_n // SUB) * (KG * SBN * 4)
    local = (pid_n % SUB) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)
    for i in range(PER // GPI):
        for j in tl.static_range(GPI):
            g = pid_s * PER + i * GPI + j
            hv = tl.load(H + rm[:, None] * K + (g * 32 + rk)[None, :], mask=m_ok[:, None], other=0.0).to(tl.float32)
            sc = tl.load(SCALE + g * 32 + rk).to(tl.float32)
            x = (hv * rinv[:, None] * sc[None, :]).to(tl.bfloat16)
            tl.store(NORMED + rm[:, None] * K + (g * 32 + rk)[None, :], x, mask=m_ok[:, None] & (pid_n == 0))
            xs = tl.sum(x.to(tl.float32), axis=1)
            words = tl.load(tile + g * (SBN * 4) + local[:, None] * 4 + rw[None, :])
            q = _deq(words, shifts, BLOCK_N)
            p = tl.dot(x, tl.trans(q))
            s = tl.load(S + g * N + rn, mask=n_ok, other=0.0).to(tl.float32)
            b = tl.load(B + g * N + rn, mask=n_ok, other=0.0).to(tl.float32)
            acc = acc + p * s[None, :] + xs[:, None] * b[None, :]
    out_mask = m_ok[:, None] & n_ok[None, :]
    if SK == 1:
        tl.store(OUT + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=out_mask)
    else:
        tl.store(PART + (pid_s * M + rm[:, None]) * N + rn[None, :], acc, mask=out_mask)


def oracle_hc_down(h: torch.Tensor, pss: torch.Tensor, scale: torch.Tensor, normed: torch.Tensor, q: Q4, eps: float,
            streams: int, *, out: torch.Tensor, part: torch.Tensor) -> torch.Tensor:
    """normed = bf16(h * rinv * scale) written to ``normed``; returns the down projection's unreduced K slices [SK, R, N] (or [R, N] bf16 when SK is 1). The K split is the shape's constant (``split_for``)."""

    if q.layout != "tiled":
        raise ValueError("hc_down takes tiled weights")

    m, k = h.shape
    d = k // streams
    sk = SHAPES16[(q.n, q.k)][0]
    _, gpi, warps, stages, bn = SHAPES16.get((q.n, q.k), (sk, 2, 4, 3, BN))
    per = (k // GS) // sk
    if (per * GS) > d or d % (per * GS):
        raise ValueError("hc_down: a K slice must lie inside one stream")
    grid = (triton.cdiv(m, 16), triton.cdiv(q.n, bn), sk)
    _qmm_hcdown[grid](h, pss, scale, normed, q.weight, q.scales, q.biases, out, part, m, eps, N=q.n, K=k, D=d,
                      NC=pss.shape[1], SS=streams, SK=sk, BM=16, BLOCK_N=bn, GPI=gpi_for(per, gpi), SBN=BN,
                      num_warps=warps, num_stages=stages, PDL=pdl.ON, launch_pdl=pdl.ON)
    return out if sk == 1 else part[:sk * m * q.n].view(sk, m, q.n)


@triton.jit
def _qmm_upmix(X, XS, W, S, B, NORMED, MIXED, XSM, M,
               N: tl.constexpr, K: tl.constexpr, D: tl.constexpr, SS: tl.constexpr, BM: tl.constexpr,
               DB: tl.constexpr, GPI: tl.constexpr, SBN: tl.constexpr, PDL: tl.constexpr = False):
    """The up projection for dims [DB j, DB (j + 1)) of every stream, then the mix: mixed = bf16(sum over streams in order of bf16(bf16(sigmoid(bf16(up_s))) * normed_s) / S) (glue.hc_mix's bits) and its group sum."""

    pdl_wait(PDL)
    KG: tl.constexpr = K // 32
    j = tl.program_id(1)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    m_ok = rm < M
    rk = tl.arange(0, 32)
    rw = tl.arange(0, 4)
    shifts = tl.arange(0, 8) * 4
    dd = tl.arange(0, DB)
    total = tl.zeros((BM, DB), dtype=tl.float32)
    for st in tl.static_range(SS):
        row0 = st * D + j * DB
        tile = W + (row0 // SBN) * (KG * SBN * 4)
        local = row0 % SBN + dd
        rn = row0 + dd
        acc = tl.zeros((BM, DB), dtype=tl.float32)
        for i in range(KG // GPI):
            for jj in tl.static_range(GPI):
                g = i * GPI + jj
                words = tl.load(tile + g * (SBN * 4) + local[:, None] * 4 + rw[None, :])
                x = tl.load(X + rm[:, None] * K + (g * 32 + rk)[None, :], mask=m_ok[:, None], other=0.0)
                q = _deq(words, shifts, DB)
                p = tl.dot(x, tl.trans(q))
                s = tl.load(S + g * N + rn).to(tl.float32)
                b = tl.load(B + g * N + rn).to(tl.float32)
                xs = tl.load(XS + rm * KG + g, mask=m_ok, other=0.0)
                acc = acc + p * s[None, :] + xs[:, None] * b[None, :]
        up = acc.to(tl.bfloat16).to(tl.float32)
        nv = tl.load(NORMED + rm[:, None] * N + (st * D + j * DB + dd)[None, :], mask=m_ok[:, None],
                     other=0.0).to(tl.float32)
        total += (_bsig(up) * nv).to(tl.bfloat16).to(tl.float32)
    mixed = (total / SS).to(tl.bfloat16)
    tl.store(MIXED + rm[:, None] * D + (j * DB + dd)[None, :], mixed, mask=m_ok[:, None])
    GB: tl.constexpr = DB // 32
    sums = tl.sum(tl.reshape(mixed.to(tl.float32), (BM, GB, 32)), axis=2)         # each 32-dim group's sum
    gi = j * GB + tl.arange(0, GB)
    tl.store(XSM + rm[:, None] * (D // 32) + gi[None, :], sums, mask=m_ok[:, None])


def oracle_hc_upmix(act: torch.Tensor, xs_act: torch.Tensor, q: Q4, normed: torch.Tensor, mixed: torch.Tensor,
             xs_mixed: torch.Tensor, streams: int) -> None:
    """The up projection and the stream mix in one kernel, 32 dims of every stream a program: the bits of ``matmul`` then ``glue.hc_mix``."""

    if q.layout != "tiled":
        raise ValueError("hc_upmix takes tiled weights")

    m, k = act.shape
    d = q.n // streams
    db = 32
    grid = (triton.cdiv(m, 16), d // db)
    _qmm_upmix[grid](act, xs_act, q.weight, q.scales, q.biases, normed, mixed, xs_mixed, m, N=q.n, K=k, D=d,
                     SS=streams, BM=16, DB=db, GPI=gpi_for(k // GS, 2), SBN=BN, num_warps=4, num_stages=3,
                     PDL=pdl.ON, launch_pdl=pdl.ON)


# ---- inputs ----

def _wide(shape, gen, lo, hi, zeros=0.03):
    """bf16 values with exponents in [lo, hi), random signs and some +-0."""

    e = torch.randint(lo, hi, shape, generator=gen, device=DEV).float()
    v = (1 + torch.rand(shape, generator=gen, device=DEV)) * torch.exp2(e)
    v = torch.where(torch.rand(shape, generator=gen, device=DEV) < 0.5, -v, v)
    sign = torch.where(torch.rand(shape, generator=gen, device=DEV) < 0.5, -1.0, 1.0)
    return torch.where(torch.rand(shape, generator=gen, device=DEV) < zeros, 0.0 * sign, v).to(torch.bfloat16)


def _weights(n, k, gen, adversarial):
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), generator=gen, device=DEV,
                          dtype=torch.int64).to(torch.int32)
    if adversarial:
        return words, _wide((n, k // 32), gen, -12, 2, 0.0).abs(), _wide((n, k // 32), gen, -14, 1)
    scales = (torch.rand((n, k // 32), generator=gen, device=DEV) * 0.02 + 0.001).to(torch.bfloat16)
    biases = (torch.randn((n, k // 32), generator=gen, device=DEV) * 0.02).to(torch.bfloat16)
    return words, scales, biases


def _rows(shape, gen, adversarial, lo=-24, hi=4, scale=1.0):
    if adversarial:
        return _wide(shape, gen, lo, hi)
    return (torch.randn(shape, generator=gen, device=DEV) * scale).to(torch.bfloat16)


def _bits(t):
    return t.view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _same(a, b):
    return torch.equal(_bits(a), _bits(b))


SHAPES = [(324, 10240), (320, 10240), (10240, 320)]
BUCKET_ROWS = [24, 32, 33, 48, 64, 100, 128, 130]


@pytest.mark.parametrize("adversarial", [False, True])
@pytest.mark.parametrize("n,k", SHAPES)
def test_tiled_matmul_matches_the_oracle(n, k, adversarial):
    """The tiled matmul (scales, biases and group sums loaded ahead, padding rows masked) gives the oracle's bits:
    the bf16 product, or the fp32 K slices the read-out sums in order."""

    gen = torch.Generator(device=DEV).manual_seed(n + k + adversarial)
    q = qmm.make_q4(*_weights(n, k, gen, adversarial), "tiled")
    sk = qmm.split_for(n, k)
    for m in [*range(1, 21), *BUCKET_ROWS]:
        x = _rows((m, k), gen, adversarial)
        xs = qmm.group_sums(x)
        got, want = [], []
        for fn, res in ((qmm.matmul, got), (oracle_matmul, want)):
            out = torch.full((m, n), SENTINEL, device=DEV).to(torch.bfloat16)
            part = torch.full((sk * m * n + 64,), SENTINEL, device=DEV)
            fn(x, q, xs, out=out, part=part, reduce=False)
            res += [out, part]
        assert all(_same(a, b) for a, b in zip(got, want)), (n, k, m)
    x = _rows((5, k), gen, adversarial)
    assert _same(qmm.matmul(x, q), oracle_matmul(x, q))          # with the slice sum (reduce) as well


@pytest.mark.parametrize("n,k", SHAPES)
def test_tiled_matmul_launch_settings_keep_the_bits(n, k):
    """Groups loaded ahead, warps, stages and the program's column width change the schedule, never a sum."""

    gen = torch.Generator(device=DEV).manual_seed(3 * n + k)
    q = qmm.make_q4(*_weights(n, k, gen, True), "tiled")
    for m in (5, 20):
        x = _rows((m, k), gen, True)
        want = oracle_matmul(x, q, reduce=False)
        for pd in (0, 1, 2):
            for warps, stages, block_n in ((4, 3, 64), (8, 2, 64), (4, 2, 32)):
                got = qmm.matmul(x, q, reduce=False, pd=pd, num_warps=warps, num_stages=stages, block_n=block_n)
                assert _same(got, want), (n, k, m, pd, warps, stages, block_n)


@pytest.mark.parametrize("adversarial", [False, True])
@pytest.mark.parametrize("inject", [True, False])
def test_decode_readout_matches_the_former_fused_kernels(inject, adversarial):
    """Every decode window takes the separate read-out kernels: the normed streams, the activation and its group
    sums, the inject gates, the mix and its group sums have the bits of the fused decode read-out that windows of
    up to 16 rows used (and of the separate kernels' 18a7c448 forms beyond)."""

    from types import SimpleNamespace

    from tensorfold.families.qwen4_exp.cuda import forward as fwd
    from tensorfold.families.qwen4_exp.cuda.weights import HC

    gen = torch.Generator(device=DEV).manual_seed(41 + 2 * inject + adversarial)
    down = qmm.stack_q4([_weights(LOW, S * D, gen, adversarial)] + ([_weights(S, S * D, gen, adversarial)]
                                                                     if inject else []), "tiled")
    up = qmm.make_q4(*_weights(S * D, LOW, gen, adversarial), "tiled")
    scale = (_wide((S * D,), gen, -6, 3, 0.0).float() if adversarial else
             1 + 0.05 * torch.randn((S * D,), generator=gen, device=DEV))
    hc = HC(down, up, scale, inject)
    f32, bf = torch.float32, torch.bfloat16
    for rows in [*range(1, 17), 17, 20]:
        h = _rows((rows, S * D), gen, adversarial, -20, 6, 3.0)
        pss = torch.empty((rows, D // 256, S), dtype=f32, device=DEV)
        glue.hc_writeback(h, h, pss, S, 0)

        def buffers():
            def full(shape, dtype):
                return torch.full(shape, SENTINEL, device=DEV).to(dtype)
            return SimpleNamespace(pss=pss, normed=full((rows, S * D), bf), xs_normed=full((rows, S * D // 32), f32),
                                   dn=full((rows, LOW + S), bf), dn_mix=full((rows, LOW), bf),
                                   act=full((rows, LOW), bf), xs_act=full((rows, LOW // 32), f32),
                                   up=full((rows, S * D), bf), mixed=full((rows, D), bf),
                                   xs_mixed=full((rows, D // 32), f32), part=torch.empty((32 * rows * (LOW + S),),
                                                                                       dtype=f32, device=DEV),
                                   inj=full((rows, S), bf), prefill=False, hc_plans=None)

        new, old = buffers(), buffers()
        fwd._readout(hc, new, h, rows, 1e-6, S, LOW, new.inj if inject else None)
        out = old.dn if inject else old.dn_mix
        got = oracle_hc_down(h, pss, scale, old.normed, down, 1e-6, S, out=out, part=old.part)
        glue.hc_reduce_act(got, old.act, old.xs_act, old.inj if inject else None, S, LOW)
        oracle_hc_upmix(old.act, old.xs_act, up, old.normed, old.mixed, old.xs_mixed, S)
        for name in ("normed", "act", "xs_act", "mixed", "xs_mixed") + (("inj",) if inject else ()):
            assert _same(getattr(new, name), getattr(old, name)), (rows, name)

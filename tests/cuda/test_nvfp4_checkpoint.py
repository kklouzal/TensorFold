"""Checkpoint-math kernels: quantizers byte-exact to an fp32 reference, matmuls the fp64 product, rows independent."""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("the block-scaled FP4 mma needs an SM 12.x GPU", allow_module_level=True)

from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.cuda.nvfp4 import format as fmt
from tensorfold.cuda.nvfp4.linear import Fp4Linear, Fp8Linear
from kernel_inputs import fp4_weight

E2M1 = torch.tensor(fmt.E2M1[:8].tolist(), dtype=torch.float32)


def _e2m1_codes(v: torch.Tensor) -> torch.Tensor:
    """Nearest e2m1 code (ties to the even code), saturating: the quantizer's rule in torch."""

    a = v.abs()
    edges = [(0.25, True), (0.75, False), (1.25, True), (1.75, False), (2.5, True), (3.5, False), (5.0, True)]
    code = torch.zeros_like(a, dtype=torch.int32)
    for c, (edge, inclusive) in enumerate(edges):
        code = torch.where(a > edge if inclusive else a >= edge, torch.full_like(code, c + 1), code)
    return torch.where((v < 0) & (code > 0), code | 8, code)


def _ref_quant4(x: torch.Tensor, act: float):
    g = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(act, dtype=torch.float32)
    g = g.to(x.device)
    m, k = x.shape
    blocks = x.float().view(m, k // 16, 16)
    amax = blocks.abs().amax(-1)
    sf8 = (g * (amax * torch.tensor(1.0 / 6.0, dtype=torch.float32))).clamp(max=448.0).to(torch.float8_e4m3fn)
    sf = sf8.float()
    mul = torch.where(sf != 0, g / sf, torch.zeros_like(sf))
    codes = _e2m1_codes(blocks * mul[..., None]).view(m, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return packed, sf8.view(torch.uint8), sf


def _ref_quant8(x: torch.Tensor, act: float) -> torch.Tensor:
    inv = (torch.tensor(1.0, dtype=torch.float32) / torch.tensor(act, dtype=torch.float32)).to(x.device)
    q = (x.float() * inv).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    m, k = x.shape
    m16 = torch.arange(16, device=x.device)
    src = (m16 // 4) * 2 + (m16 % 4 % 2) + (m16 % 4 // 2) * 8          # byte 4q + j holds input 2q + j%2 + 8(j/2)
    return q.view(m, k // 16, 16)[:, :, src].reshape(m, k)


def _rows(m, k, seed, spread=1.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((m, k), generator=g) * spread
    x[:, :: 97] *= 12.0                                                           # outliers past the calibrated range
    return x.to(torch.bfloat16).cuda()


@pytest.mark.parametrize("m,k", [(1, 64), (5, 1024), (130, 5120)])
def test_nvfp4_rows_are_the_checkpoint_format_byte_for_byte(m, k):
    x, act = _rows(m, k, m), 0.0123
    codes, scales = checkpoint.quant4(x, act)
    want_codes, want_sf8, _ = _ref_quant4(x, act)
    assert torch.equal(codes, want_codes)
    got_sf = scales[:, :m].permute(1, 0, 2).reshape(m, k // 16)
    assert torch.equal(got_sf, want_sf8)


def test_original_ct_multiplier_retains_sensitive_e4m3_rounding():
    # Independent exact-rational/RNE counterexample: raw g*(.5*F32(1/6))
    # lies on the E4M3 midpoint and rounds to even0x0a. F32(1/F32(1/g))
    # shifts below that midpoint, producing0x09 instead.
    raw = float(torch.tensor(0x3E63FFFF, dtype=torch.int32).view(torch.float32))
    act = float(torch.tensor(1., dtype=torch.float32) / torch.tensor(raw))
    x = torch.full((1, 64), .5, dtype=torch.bfloat16, device="cuda")
    direct, legacy = checkpoint.quant4(x, act, inverse=raw), checkpoint.quant4(x, act)
    assert torch.equal(direct.scales[:, 0], torch.full_like(direct.scales[:, 0], 0x0A))
    assert torch.equal(legacy.scales[:, 0], torch.full_like(legacy.scales[:, 0], 0x09))
    assert torch.equal(direct.codes, torch.full_like(direct.codes, 0x77))


@pytest.mark.parametrize("m,k", [(1, 64), (7, 2048)])
def test_fp8_rows_are_e4m3_under_the_static_scale_in_fragment_order(m, k):
    x, act = _rows(m, k, 50 + m), 0.0071
    assert torch.equal(checkpoint.quant8(x, act), _ref_quant8(x, act))


def _tiled(rows: torch.Tensor, step: int, swz) -> torch.Tensor:
    """Row-major bytes (M, K') -> the prompt GEMM's tiles [mpad / TB][K/64][TB][step], 16-byte chunks swizzled."""

    tb, (m, kb) = checkpoint.TB, rows.shape
    mpad = -(-m // tb) * tb
    full = torch.zeros((mpad, kb), dtype=torch.uint8, device=rows.device)
    full[:m] = rows
    t = full.view(mpad // tb, tb, kb // step, step // 16, 16)
    out = torch.empty_like(t)
    for r in range(tb):
        for c in range(step // 16):
            out[:, r, :, swz(r, c)] = t[:, r, :, c]
    return out.permute(0, 2, 1, 3, 4).reshape(-1)


@pytest.mark.parametrize("m,k", [(1, 64), (130, 5120)])
def test_tiled_rows_are_the_row_quantizers_bytes_in_the_prompt_gemms_tiles(m, k):
    x = _rows(m, k, 70 + m)
    codes, scales = checkpoint.quant4(x, 0.0123)
    tc, ts = checkpoint.quant4(x, 0.0123, checkpoint.TB)
    assert torch.equal(tc.view(-1), _tiled(codes, 32, lambda r, c: c ^ ((r >> 2) & 1)))
    ts = ts.permute(1, 0, 2, 3).reshape(k // 64, -1, 4)                       # [K/64, mpad, 4], as the rows'
    assert torch.equal(ts[:, :m], scales[:, :m]) and not ts[:, m:].any()
    t8 = checkpoint.quant8(x, 0.0071, checkpoint.TB)
    assert torch.equal(t8.view(-1), _tiled(checkpoint.quant8(x, 0.0071), 64, lambda r, c: c ^ ((r >> 1) & 3)))


def _fp4_ref(x, act, packed, scale, g):
    """fp64 product of the quantized rows and the stored weights, scaled as the kernel scales."""

    codes, _, sf = _ref_quant4(x, act)
    m, k = x.shape
    lo, hi = (codes & 0xF).long(), (codes >> 4).long()
    vals = torch.stack([fmt_e2m1(lo), fmt_e2m1(hi)], -1).view(m, k).double()
    xq = vals * sf.double().repeat_interleave(16, 1)
    w = torch.from_numpy(fmt.dequant("nvfp4", packed, scale, 1.0)).double().cuda()
    return (xq @ w.t()) * (np.float32(act) * np.float32(g))


def fmt_e2m1(code: torch.Tensor) -> torch.Tensor:
    return torch.tensor(fmt.E2M1.tolist(), dtype=torch.float64, device=code.device)[code]


ROWS = (1, 2, 3, 5, 12, 16, 17, 31, 33, 64, 65, 100, 128)


@pytest.mark.parametrize("n,k", [(128, 256), (1000, 5120), (320, 17408)])
def test_fp4_lane_is_the_quantized_product_and_rows_are_independent(n, k):
    packed, scale, g = fp4_weight(n, k, n)
    act = 0.0123
    lin = Fp4Linear.from_checkpoint(torch.from_numpy(packed).cuda(), torch.from_numpy(scale).cuda(), g, act=act)
    x = _rows(128, k, 3, 0.7)
    full = lin(x)
    for rows in ROWS:
        assert torch.equal(lin(x[:rows].contiguous()), full[:rows]), rows
    one = torch.cat([lin(x[r:r + 1].contiguous()) for r in range(0, 128, 9)])
    assert torch.equal(one, full[0:128:9])
    ref = _fp4_ref(x, act, packed, scale, g)
    exact = checkpoint.matmul(checkpoint.A4, x, lin, f32=True).double()
    err = ((exact - ref).abs() / (ref.abs() + ref.abs().mean())).max().item()
    assert err < 1e-5, err                                                      # exact products, fp32 sums
    err = ((full.double() - ref).abs() / (ref.abs() + ref.abs().mean())).max().item()
    assert err < 1e-2, err                                                      # and bf16 out


@pytest.mark.parametrize("n,k", [(128, 256), (1024, 5120)])
def test_fp8_lane_is_the_quantized_product_and_rows_are_independent(n, k):
    rng = np.random.default_rng(n)
    w = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    w[(w & 0x7F) >= 0x70] = 0x30
    s, act = 0.0371, 0.0071
    lin = Fp8Linear.from_checkpoint(torch.from_numpy(w).cuda().view(torch.float8_e4m3fn), s, act=act)
    x = _rows(128, k, 4, 0.5)
    full = lin(x)
    for rows in ROWS:
        assert torch.equal(lin(x[:rows].contiguous()), full[:rows]), rows
    inv = (torch.tensor(1.0, dtype=torch.float32) / torch.tensor(act, dtype=torch.float32)).cuda()
    xq = (x.float() * inv).clamp(-448, 448).to(torch.float8_e4m3fn).double()
    wq = torch.from_numpy(w).cuda().view(torch.float8_e4m3fn).double()
    ref = (xq @ wq.t()) * (np.float32(act) * np.float32(s))
    exact = checkpoint.matmul(checkpoint.A8, x, lin, f32=True).double()
    err = ((exact - ref).abs() / (ref.abs() + ref.abs().mean())).max().item()
    assert err < 1e-5, err


@pytest.mark.parametrize("mode", [checkpoint.A4, checkpoint.A8])
def test_slices_meet_in_part_with_the_clusters_bits(mode):
    """SM 8.9 has no clusters: K slices add through ``part`` and the reduce, in the clusters' order: the same bits."""

    from tensorfold.cuda.kernels import qmm

    n, k, m = 5120, 17408, 37                                                   # 4 slices (a down projection)
    if mode == checkpoint.A4:
        packed, scale, g = fp4_weight(n, k, 7)
        weight, scales = torch.from_numpy(packed).cuda(), torch.from_numpy(scale).cuda()
        lin = Fp4Linear.from_checkpoint(weight, scales, g, act=0.0123)
    else:
        w = np.random.default_rng(7).integers(0, 256, size=(n, k), dtype=np.uint8)
        w[(w & 0x7F) >= 0x70] = 0x30
        lin = Fp8Linear.from_checkpoint(torch.from_numpy(w).cuda().view(torch.float8_e4m3fn), 0.0371, act=0.0071)
    sk = qmm.split_k(lin.n, lin.k)
    assert 1 < sk <= 8
    x = _rows(m, k, 5, 0.6)
    rows = checkpoint._rowsq(mode, x, lin.act)
    clusters = checkpoint.matmul(mode, x, lin)
    y = torch.empty_like(clusters)
    part = torch.empty((sk, m, lin.n), dtype=torch.float32, device="cuda")
    w, ws = (lin.words, lin.bs) if mode == checkpoint.A4 else (lin.w8, None)
    checkpoint._ext().lane(mode, rows[0], rows[1], w, ws, checkpoint.alpha(lin.act, lin.scale), None, None, y, part,
                           lin.n, lin.k, sk, lin.npad, qmm.bucket(m), False)
    assert torch.equal(y, clusters)


def test_head_tiles_keep_the_checkpoint_math():
    """The drafter's head slices (64-column tiles as views) keep the input scale and the FP4 mma's word order."""

    n, k = 640, 512
    packed, scale, g = fp4_weight(n, k, 11)
    lin = Fp4Linear.from_checkpoint(torch.from_numpy(packed).cuda(), torch.from_numpy(scale).cuda(), g, act=0.02)
    x = _rows(9, k, 12)
    part = lin.tiles(2, 7)
    assert part.act == lin.act and torch.equal(part(x), lin(x)[:, 128:448])


CHUNKS = (1, 37, 128, 135)
TILES = (1, 2, 3, 11, 12, 13)               # the GEMM's tiles, then the warp-specialized GEMM's


def _chunked(fn, x):
    out, a = [], 0
    while a < x.shape[0]:
        for size in CHUNKS:
            out.append(fn(x[a:a + size].contiguous()))
            a += size
            if a >= x.shape[0]:
                break
    return torch.cat(out)[:x.shape[0]]


@pytest.mark.parametrize("n,k", [(128, 256), (1000, 5120), (12288, 5120)])
def test_fp4_prompt_rows_are_chunk_invariant_and_the_quantized_product(n, k):
    """The prompt GEMM: one K chain a row, so any chunking and every tile give a row the same bits."""

    from tensorfold.cuda.kernels import qmm

    packed, scale, g = fp4_weight(n, k, 7 * n)
    act = 0.0123
    lin = Fp4Linear.from_checkpoint(torch.from_numpy(packed).cuda(), torch.from_numpy(scale).cuda(), g, act=act)
    x = _rows(300, k, 5, 0.7)
    full = checkpoint.prompt(checkpoint.A4, x, lin)
    for tile in (0, 11):
        assert torch.equal(_chunked(lambda c: checkpoint.prompt(checkpoint.A4, c, lin, tile=tile), x), full), tile
    for tile in TILES:
        assert torch.equal(checkpoint.prompt(checkpoint.A4, x, lin, tile=tile), full), tile
    ref = _fp4_ref(x, act, packed, scale, g)
    exact = checkpoint.prompt(checkpoint.A4, x, lin, f32=True).double()
    err = ((exact - ref).abs() / (ref.abs() + ref.abs().mean())).max().item()
    assert err < 1e-5, err
    if qmm.split_k(n, k) == 1:                                                  # one K slice: the lane's chain too
        assert torch.equal(full[:128], lin(x[:128].contiguous()))


@pytest.mark.parametrize("n,k", [(128, 256), (1024, 5120)])
def test_fp8_prompt_rows_are_chunk_invariant_and_the_quantized_product(n, k):
    rng = np.random.default_rng(n + 1)
    w = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    w[(w & 0x7F) >= 0x70] = 0x30
    s, act = 0.0371, 0.0071
    lin = Fp8Linear.from_checkpoint(torch.from_numpy(w).cuda().view(torch.float8_e4m3fn), s, act=act)
    x = _rows(300, k, 6, 0.5)
    full = checkpoint.prompt(checkpoint.A8, x, lin)
    for tile in (0, 11):
        assert torch.equal(_chunked(lambda c: checkpoint.prompt(checkpoint.A8, c, lin, tile=tile), x), full), tile
    for tile in TILES:
        assert torch.equal(checkpoint.prompt(checkpoint.A8, x, lin, tile=tile), full), tile
    inv = (torch.tensor(1.0, dtype=torch.float32) / torch.tensor(act, dtype=torch.float32)).cuda()
    xq = (x.float() * inv).clamp(-448, 448).to(torch.float8_e4m3fn).double()
    wq = torch.from_numpy(w).cuda().view(torch.float8_e4m3fn).double()
    ref = (xq @ wq.t()) * (np.float32(act) * np.float32(s))
    exact = checkpoint.prompt(checkpoint.A8, x, lin, f32=True).double()
    err = ((exact - ref).abs() / (ref.abs() + ref.abs().mean())).max().item()
    assert err < 1e-5, err


@pytest.mark.parametrize("fp32", [True, False])
def test_fused_mlp_is_swiglu_then_down_and_chunk_invariant(monkeypatch, fp32):
    """gate|up -> SiLU(gate) * up -> down, quantized in the GEMM's epilogue: the fp64 product, chunk-invariant bits."""

    monkeypatch.setattr(checkpoint, "SWIGLU_FP32", fp32)
    hidden, inter, act, act_d = 256, 384, 0.0123, 0.0071
    lins, raw = [], []
    for n, k, a in ((inter, hidden, act), (inter, hidden, act), (hidden, inter, act_d)):
        packed, scale, g = fp4_weight(n, k, n + k + len(lins))
        raw.append((packed, scale, g))
        lins.append(Fp4Linear.from_checkpoint(torch.from_numpy(packed).cuda(), torch.from_numpy(scale).cuda(), g,
                                              act=a))
    gate, up, down = lins
    x = _rows(300, hidden, 13, 0.7)
    y = checkpoint.mlp_prompt(x, gate, up, down)
    assert y is not None and y.shape == (300, hidden)
    assert torch.equal(_chunked(lambda c: checkpoint.mlp_prompt(c, gate, up, down), x), y)
    gq = checkpoint.prompt(checkpoint.A4, x, gate, f32=True)
    uq = checkpoint.prompt(checkpoint.A4, x, up, f32=True)
    if not fp32:
        gq, uq = gq.bfloat16().float(), uq.bfloat16().float()
    a = gq * torch.sigmoid(gq) * uq
    if not fp32:
        a = a.bfloat16().float()
    ref = _fp4_ref(a, act_d, *raw[2])                           # SwiGLU rows quantized as the epilogue does
    rel = (y.double() - ref).abs() / (ref.abs() + ref.abs().mean())
    assert (rel < 1e-2).double().mean().item() > 0.999, rel.max().item()   # exp's last ulp can flip a code
    assert ((y.double() - ref).norm() / ref.norm()).item() < 1e-2


# Per-channel FP8 weights under dynamic per-token FP8 rows (ModelOpt FP8_PER_CHANNEL_PER_TOKEN): each row's scale
# s = max(amax / 448, 1 / (448 * 512)), codes e4m3(x / s), out = (sum * the column's scale) * s, all against fp64.
TOKEN_FLOOR = torch.tensor(1.0 / (448.0 * 512.0), dtype=torch.float32)


def _ref_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent per-token quantizer in torch: (fragment-order e4m3 bytes, fp32 row scales)."""

    xf = x.float()
    amax = xf.abs().amax(-1)
    # tensor / tensor divides correctly rounded on CUDA (torch turns a CPU-scalar divisor into a reciprocal multiply)
    s = torch.maximum(amax / torch.full_like(amax, 448.0), TOKEN_FLOOR.to(x.device))
    q = (xf / s[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    m, k = x.shape
    m16 = torch.arange(16, device=x.device)
    src = (m16 // 4) * 2 + (m16 % 4 % 2) + (m16 % 4 // 2) * 8
    return q.view(m, k // 16, 16)[:, :, src].reshape(m, k), s


def _token_rows(m, k, seed):
    x = _rows(m, k, seed, 0.8)
    if m > 3:
        x[1] = 0                                                                # all-zero row: the floor scale
        x[2] *= 1e-4                                                            # amax below 1/512: the floor too
        x[3, ::5] *= 300.0                                                      # e4m3's top binade
    return x


@pytest.mark.parametrize("m,k", [(1, 64), (7, 5120), (130, 6144), (5, 17408)])
def test_token_rows_are_e4m3_under_each_rows_own_scale(m, k):
    x = _token_rows(m, k, 80 + m)
    codes, scales = checkpoint.quant8_token(x)
    want, s = _ref_token(x)
    assert torch.equal(scales, s)
    assert torch.equal(codes, want)
    t8, ts = checkpoint.quant8_token(x, checkpoint.TB)
    assert torch.equal(ts, s)
    assert torch.equal(t8.view(-1), _tiled(codes, 64, lambda r, c: c ^ ((r >> 1) & 3)))
    one = [checkpoint.quant8_token(x[r:r + 1].contiguous()) for r in range(0, m, max(1, m // 5))]
    for j, r in enumerate(range(0, m, max(1, m // 5))):
        assert torch.equal(one[j][0][0], codes[r]) and torch.equal(one[j][1][0], scales[r])


def _channel_lin(n, k, seed, token=True):
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    w[(w & 0x7F) >= 0x70] = 0x30
    cols = torch.from_numpy((np.exp(rng.normal(-6.0, 1.5, size=n))).astype(np.float32))   # AEON-like 1e-7..3e-3
    lin = Fp8Linear.from_checkpoint(torch.from_numpy(w).cuda().view(torch.float8_e4m3fn), cols.cuda(), token=token)
    return lin, torch.from_numpy(w).cuda().view(torch.float8_e4m3fn).double(), cols.cuda().double()


def _token_ref(x, wq, cols):
    """fp64 of the declared math: (e4m3 rows x their scales) @ (e4m3 weights x their channel scales)^T, and the sums of
    the products' magnitudes (fp32 accumulation's error scale)."""

    codes, s = _ref_token(x)
    m, k = x.shape
    m16 = torch.arange(16, device=x.device)
    src = (m16 // 4) * 2 + (m16 % 4 % 2) + (m16 % 4 // 2) * 8
    back = torch.empty_like(codes).view(m, k // 16, 16)
    back[:, :, src] = codes.view(m, k // 16, 16)
    xq = back.view(m, k).view(torch.float8_e4m3fn).double() * s.double()[:, None]
    wd = wq * cols[:, None]
    return xq @ wd.t(), xq.abs() @ wd.abs().t()


def _err(got, ref, absum, rounding=0.0):
    """The worst error over its bound: ``rounding`` of the entry (bf16 output) plus 1e-5 of the sum of its products'
    magnitudes. Products of e4m3 (or bf16 by e4m3) values are exact in fp32, so the second term bounds only the fp32
    accumulation, whatever a row's cancellation or a channel's scale (an fp32 sum of K terms stays near
    sqrt(K) * 2^-24 of it; a narrower accumulator would not). Pass while below 1."""

    bound = rounding * ref.abs() + 1e-5 * absum
    return ((got.double() - ref).abs() / bound.clamp_min(torch.finfo(torch.float64).tiny)).max().item()


@pytest.mark.parametrize("n,k", [(128, 256), (1024, 5120), (640, 17408)])
def test_token_lane_is_the_declared_product_and_rows_are_independent(n, k):
    lin, wq, cols = _channel_lin(n, k, n + k)
    assert lin.token and lin.act is None and lin.cols.shape == (lin.npad,)
    x = _token_rows(128, k, 9)
    full = lin(x)
    for rows in ROWS:
        assert torch.equal(lin(x[:rows].contiguous()), full[:rows]), rows
    one = torch.cat([lin(x[r:r + 1].contiguous()) for r in range(0, 128, 9)])
    assert torch.equal(one, full[0:128:9])
    ref, absum = _token_ref(x, wq, cols)
    assert _err(checkpoint.matmul(checkpoint.A8, x, lin, f32=True), ref, absum) < 1     # exact products, fp32 sums
    assert _err(full, ref, absum, 2 ** -8) < 1                                          # and bf16 out
    part = lin.tiles(1, 2 if n <= 128 else 5)
    assert torch.equal(part(x), full[:, 64:64 * (2 if n <= 128 else 5)])             # head-tile views keep the math


def test_fp8_mma_accumulates_in_full_fp32():
    """Partial sums near 2^22 then a +1: exact in an fp32 accumulator, lost in a reduced-precision one."""

    k = 17408
    x = torch.full((1, k), 448.0, dtype=torch.float32)
    x[0, k // 2:] = -448.0
    x[0, -1] = 1.0                                                     # 448 (k/2) - 448 (k/2 - 1) + 1 = 449
    w = torch.full((64, k), 1.0, dtype=torch.float32).to(torch.float8_e4m3fn)
    lin = Fp8Linear.from_checkpoint(w.cuda(), torch.ones(64), token=True)
    got = checkpoint.matmul(checkpoint.A8, x.to(torch.bfloat16).cuda(), lin, f32=True)
    assert torch.equal(got, torch.full_like(got, 449.0)), got[0, :4]
    y = checkpoint.prompt(checkpoint.A8, x.to(torch.bfloat16).cuda().expand(130, k).contiguous(), lin, f32=True)
    assert torch.equal(y, torch.full_like(y, 449.0)), y[0, :4]


def test_token_slices_meet_in_part_with_the_clusters_bits():
    from tensorfold.cuda.kernels import qmm

    lin, _, _ = _channel_lin(5120, 17408, 7)
    sk = qmm.split_k(lin.n, lin.k)
    assert 1 < sk <= 8
    x = _token_rows(37, lin.k, 5)
    clusters = checkpoint.matmul(checkpoint.A8, x, lin)
    codes, scales = checkpoint.quant8_token(x)
    y = torch.empty_like(clusters)
    part = torch.empty((sk, 37, lin.n), dtype=torch.float32, device="cuda")
    checkpoint._ext().lane(checkpoint.A8, codes, None, lin.w8, None, 1.0, scales, lin.cols, y, part, lin.n, lin.k,
                           sk, lin.npad, qmm.bucket(37), False)
    assert torch.equal(y, clusters)


@pytest.mark.parametrize("n,k", [(128, 256), (1024, 5120), (10240, 5120)])
def test_token_prompt_rows_are_chunk_invariant_and_the_declared_product(n, k):
    lin, wq, cols = _channel_lin(n, k, 3 * n + k)
    x = _token_rows(300, k, 6)
    full = checkpoint.prompt(checkpoint.A8, x, lin)
    for tile in (0, 11):
        assert torch.equal(_chunked(lambda c: checkpoint.prompt(checkpoint.A8, c, lin, tile=tile), x), full), tile
    for tile in TILES:
        assert torch.equal(checkpoint.prompt(checkpoint.A8, x, lin, tile=tile), full), tile
    ref, absum = _token_ref(x, wq, cols)
    assert _err(checkpoint.prompt(checkpoint.A8, x, lin, f32=True), ref, absum) < 1
    assert _err(full, ref, absum, 2 ** -8) < 1


def test_projections_of_one_input_share_one_quantization():
    a, _, _ = _channel_lin(640, 5120, 1)
    b, _, _ = _channel_lin(384, 5120, 2)
    x = _token_rows(40, 5120, 3)
    for prompt_rows in (False, True):
        outs = checkpoint.matmul_group(x, [a, b], prompt_rows=prompt_rows)
        run = checkpoint.prompt if prompt_rows else checkpoint.matmul
        assert outs is not None and all(torch.equal(o, run(checkpoint.A8, x, lin)) for o, lin in zip(outs, (a, b)))
    static = Fp8Linear.from_checkpoint(a.w8.new_zeros((128, 5120)).view(torch.float8_e4m3fn), 0.5, act=0.01)
    assert checkpoint.matmul_group(x, [a, static]) is None                    # rows quantized differently


@pytest.mark.parametrize("n,k", [(128, 256), (1024, 5120), (640, 17408)])
def test_full_precision_channel_fp8_is_w8a16_exact(n, k):
    """--precision full: bf16 rows times the stored e4m3 bytes, each channel's scale in the epilogue (decode and
    prompt kernels), against fp64."""

    lin, wq, cols = _channel_lin(n, k, 5 * n + k, token=False)
    assert not lin.token and lin.scale == 1.0
    x = _rows(77, k, 8, 0.6)
    wd = wq * cols[:, None]
    ref, absum = x.double() @ wd.t(), x.double().abs() @ wd.abs().t()
    y32 = torch.empty((77, n), dtype=torch.float32, device="cuda")
    lin(x, y32)
    assert _err(y32, ref, absum) < 1
    full = lin(x)
    assert _err(full, ref, absum, 2 ** -8) < 1
    for rows in ROWS[:8]:
        assert torch.equal(lin(x[:rows].contiguous()), full[:rows]), rows
    prompt = lin.prefill(x)
    assert _err(prompt, ref, absum, 2 ** -8) < 1
    assert torch.equal(_chunked(lin.prefill, x), prompt)
    with pytest.raises(ValueError, match="--prefill-fp8"):
        lin.prefill8(None)

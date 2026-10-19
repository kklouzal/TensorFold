"""Fused elementwise stages of the Qwen DFlash v1 drafter (one launch each where the eager drafter issued several).

Each keeps the BF16 rounding points of the Qwen3 expressions it replaces (normalized value before the weight
multiply; rotation products and their sum; SiLU before the product). The FP32 work between them (a row's mean of
squares in one program, Triton's rsqrt and exp) may round differently from torch's in low-order bits, so the fused
drafter is not bit-identical to an eager one, but every row's result depends only on that row (no batch-size-dependent
reduction). Proposals only: the target verifies every committed token. Every tensor argument is contiguous (the
kernels index rows by their declared widths): the drafter's projections, embedding and norms produce such tensors.

- ``rmsnorm``: optional BF16 residual add, then Qwen3 RMSNorm: fp32 mean of squares, the normalized value rounded to
  BF16 before the BF16 weight multiply (rounded again).
- ``qkv_heads``: one [rows, (H + 2 Hkv) D] projection -> q [H, rows, D], k [Hkv, rows, D] (each head RMS-normed and
  rotated: products and sum rounded to BF16 as BF16 tensor ops do) and v [Hkv, rows, D].
- ``kv_append``: the context rows' [rows, 2 Hkv D] projection -> normed, rotated K and raw V written straight into
  each row's store slot (per-row destination table), replacing per-layer temporaries and copies.
- ``silu_mul``: [rows, 2 F] gate|up -> BF16(BF16(silu(gate)) * up).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(X, R, W, H, Y, eps, N: tl.constexpr, BLOCK: tl.constexpr, HAS_R: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    live = cols < N
    x = tl.load(X + row * N + cols, mask=live, other=0.0)
    if HAS_R:
        x = (x.to(tl.float32) + tl.load(R + row * N + cols, mask=live, other=0.0).to(tl.float32)).to(tl.bfloat16)
        tl.store(H + row * N + cols, x, mask=live)
    v = x.to(tl.float32)
    inv = tl.rsqrt(tl.sum(v * v, 0) / N + eps)
    n = (v * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, mask=live, other=0.0).to(tl.float32)
    tl.store(Y + row * N + cols, (w * n).to(tl.bfloat16), mask=live)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, residual: torch.Tensor | None = None):
    """BF16 rows x [M, N] (+ residual) -> (x + residual or x, RMSNorm(that) * weight)."""
    rows, n = x.shape
    y = torch.empty_like(x)
    h = torch.empty_like(x) if residual is not None else x
    _rmsnorm[(rows,)](x, residual if residual is not None else x, weight, h, y, eps, N=n,
                      BLOCK=triton.next_power_of_2(n), HAS_R=residual is not None, num_warps=8)
    return h, y


@triton.jit
def _rope_head(src, wn, cos_row, sin_row, eps, D: tl.constexpr):
    """The head at ``src`` RMS-normed (``wn`` weight) and rotated by its row's ``cos_row``/``sin_row``, as BF16."""
    d = tl.arange(0, D)
    half: tl.constexpr = D // 2
    partner = tl.where(d < half, d + half, d - half)
    x = tl.load(src + d).to(tl.float32)
    xp = tl.load(src + partner).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
    cos = tl.load(cos_row + d).to(tl.float32)
    sin = tl.load(sin_row + d).to(tl.float32)
    n = ((x * inv).to(tl.bfloat16).to(tl.float32) * tl.load(wn + d).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    m = ((xp * inv).to(tl.bfloat16).to(tl.float32) * tl.load(wn + partner).to(tl.float32))
    m = m.to(tl.bfloat16).to(tl.float32)
    rot = tl.where(d < half, -m, m)
    a = (n * cos).to(tl.bfloat16).to(tl.float32)
    b = (rot * sin).to(tl.bfloat16).to(tl.float32)
    return (a + b).to(tl.bfloat16)


# ``rows`` (here) and ``layer`` (_kv_append) are not value-specialised: one compiled variant serves every block length
# and layer instead of one each for 1 / multiples of 16 / other values; addresses keep the constexpr D factor either
# way (same PTX loads and stores).
@triton.jit(do_not_specialize=["rows"])
def _qkv_heads(QKV, QN, KN, COS, SIN, QO, KO, VO, rows, eps, H: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    src = QKV + r * ((H + 2 * HKV) * D) + h * D
    # q and k each inline _rope_head: on sm_120 one shared inline (branch-selected weight and destination) measured
    # 0.7% slower at 240 rows, and a norm prologue shared before the branch 7-11% slower at 16-240 rows.
    if h < H:
        tl.store(QO + (h * rows + r) * D + d, _rope_head(src, QN, COS + r * D, SIN + r * D, eps, D))
    elif h < H + HKV:
        tl.store(KO + ((h - H) * rows + r) * D + d, _rope_head(src, KN, COS + r * D, SIN + r * D, eps, D))
    else:
        tl.store(VO + ((h - H - HKV) * rows + r) * D + d, tl.load(src + d))


def qkv_heads(qkv: torch.Tensor, q_norm: torch.Tensor, k_norm: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
              eps: float, *, heads: int, kv_heads: int, head_dim: int):
    """[rows, (H + 2 Hkv) D] BF16 and per-row BF16 cos/sin [rows, D] -> q [H, rows, D], k, v [Hkv, rows, D]."""
    rows = qkv.shape[0]
    q = torch.empty((heads, rows, head_dim), dtype=torch.bfloat16, device=qkv.device)
    k = torch.empty((kv_heads, rows, head_dim), dtype=torch.bfloat16, device=qkv.device)
    v = torch.empty_like(k)
    _qkv_heads[(rows, heads + 2 * kv_heads)](qkv, q_norm, k_norm, cos, sin, q, k, v, rows, eps, H=heads, HKV=kv_heads,
                                             D=head_dim, num_warps=1)
    return q, k, v


@triton.jit(do_not_specialize=["layer"])
def _kv_append(KV, KN, COS, SIN, DEST, layer, eps, HKV: tl.constexpr, D: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    kbase = tl.multiple_of(tl.load(DEST + 4 * r).to(tl.pointer_type(tl.bfloat16)), 16)
    vbase = tl.multiple_of(tl.load(DEST + 4 * r + 1).to(tl.pointer_type(tl.bfloat16)), 16)
    capacity = tl.load(DEST + 4 * r + 2)
    slot = tl.load(DEST + 4 * r + 3)
    kh = h % HKV
    offset = ((layer * HKV + kh).to(tl.int64) * capacity + slot) * D
    src = KV + r * (2 * HKV * D) + h * D
    if h < HKV:
        tl.store(kbase + offset + d, _rope_head(src, KN, COS + r * D, SIN + r * D, eps, D))
    else:
        tl.store(vbase + offset + d, tl.load(src + d))


def destinations(stores, starts, sizes, device) -> torch.Tensor:
    """Per appended row (key store, value store, capacity, slot) as one int64 device table. Each store's k and v are
    contiguous BF16 [layers, kv_heads, capacity, head_dim] allocations (16-byte-aligned bases: kv_append's
    ``multiple_of`` hint) that outlive the launches. The pinned staging tensor is freed to PyTorch's caching host
    allocator, which reuses it only after the non-blocking copy's stream event."""
    flat = []
    for store, start, n in zip(stores, starts, sizes):
        if n:
            k, v, capacity = store.k.data_ptr(), store.v.data_ptr(), store.k.shape[2]
            for slot in range(start, start + n):
                flat += [k, v, capacity, slot]
    return torch.tensor(flat, dtype=torch.int64).pin_memory().to(device, non_blocking=True)


def kv_append(kv: torch.Tensor, k_norm: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, dest: torch.Tensor,
              layer: int, eps: float, *, kv_heads: int, head_dim: int) -> None:
    """[rows, 2 Hkv D] context projection -> its normed, rotated K and raw V at each row's ``dest`` slot."""
    rows = kv.shape[0]
    _kv_append[(rows, 2 * kv_heads)](kv, k_norm, cos, sin, dest, layer, eps, HKV=kv_heads, D=head_dim, num_warps=1)


@triton.jit
def _silu_mul(GU, OUT, F: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    live = cols < F
    g = tl.load(GU + row * (2 * F) + cols, mask=live, other=0.0).to(tl.float32)
    u = tl.load(GU + row * (2 * F) + F + cols, mask=live, other=0.0).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + row * F + cols, (s * u).to(tl.bfloat16), mask=live)


def silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
    """[rows, 2 F] BF16 gate|up -> [rows, F] BF16(BF16(silu(gate)) * up)."""
    rows, width = gate_up.shape
    f = width // 2
    out = torch.empty((rows, f), dtype=torch.bfloat16, device=gate_up.device)
    block = 1024
    _silu_mul[(rows, triton.cdiv(f, block))](gate_up, out, F=f, BLOCK=block, num_warps=4)
    return out

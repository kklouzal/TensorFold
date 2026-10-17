"""Prefill attention in 64-key tiles by absolute position, so chunking never changes bits."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

BM = 64
BN = 64


@triton.jit
def _head_offsets(rows, head, heads: tl.constexpr, dim: tl.constexpr):
    """Contiguous head-row offsets: widen before multiplying, including caches beyond 4 GiB."""

    return (rows.to(tl.int64) * heads + head) * dim


@triton.jit
def _tile(q, k, v, m, l, o, valid, SCALE: tl.constexpr):
    s = tl.dot(q, tl.trans(k)).to(tl.float32) * SCALE
    s = tl.where(valid, s, float("-inf"))
    tile_m = tl.max(s, 1)
    active = tile_m != float("-inf")
    next_m = tl.where(active, tl.maximum(m, tile_m), m)
    alpha = tl.where(active, tl.where(m == float("-inf"), 0.0, tl.exp(m - next_m)), 1.0)
    p = tl.where(valid & active[:, None], tl.exp(s - next_m[:, None]), 0.0)
    o = o * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    # Match the CUDA online-normalizer recurrence with one FP32 rounding;
    # relying on compiler contraction changes the bits across Triton releases.
    l = tl.fma(l, alpha, tl.sum(p, 1))
    return next_m, l, o


@triton.jit
def _attend(Q, K, V, OUT, p0, W, H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, BM: tl.constexpr,
            BN: tl.constexpr, SCALE: tl.constexpr):
    block = tl.program_id(0)
    head = tl.program_id(1)
    hk = head // (H // HK)
    rows = block * BM + tl.arange(0, BM)
    ok = rows < W
    pos = p0 + rows
    d = tl.arange(0, D)
    offsets = _head_offsets(rows, head, H, D)
    q = tl.load(Q + offsets[:, None] + d[None, :], mask=ok[:, None], other=0.0)
    m = tl.full((BM,), float("-inf"), tl.float32)
    l = tl.zeros((BM,), tl.float32)
    o = tl.zeros((BM, D), tl.float32)
    first_pos = p0 + block * BM
    last_pos = p0 + tl.minimum(block * BM + BM, W) - 1
    full = (first_pos + 1) // BN                  # tiles every row of the block sees whole
    for t in range(full):
        keys = t * BN + tl.arange(0, BN)
        cache_offsets = _head_offsets(keys, hk, HK, D)
        k = tl.load(K + cache_offsets[:, None] + d[None, :])
        v = tl.load(V + cache_offsets[:, None] + d[None, :])
        m, l, o = _tile(q, k, v, m, l, o, keys[None, :] <= pos[:, None], SCALE)
    for t in range(full, last_pos // BN + 1):
        keys = t * BN + tl.arange(0, BN)
        seen = keys <= last_pos
        cache_offsets = _head_offsets(keys, hk, HK, D)
        k = tl.load(K + cache_offsets[:, None] + d[None, :], mask=seen[:, None], other=0.0)
        v = tl.load(V + cache_offsets[:, None] + d[None, :], mask=seen[:, None], other=0.0)
        m, l, o = _tile(q, k, v, m, l, o, keys[None, :] <= pos[:, None], SCALE)
    out = o / l[:, None]
    tl.store(OUT + offsets[:, None] + d[None, :], out.to(tl.bfloat16), mask=ok[:, None])


def attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, p0: int, *, scale: float) -> torch.Tensor:
    """q (W, H, D) bf16 at positions [p0, p0 + W); caches hold every key through p0 + W - 1.

    Native D128/D256 queries require 4/16-byte alignment; their caches require 16-byte alignment.
    """

    hk = _check(q, k_cache, v_cache, p0)
    _, h, d = q.shape
    out = torch.empty_like(q)
    if d == 64:
        return triton_attention(q, k_cache, v_cache, p0, scale=scale, out=out)
    _ext().prefill_attention(q, k_cache, v_cache, out, p0, scale, heads_a_block(h // hk))
    return out


def triton_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, p0: int, *, scale: float,
                     out: torch.Tensor | None = None) -> torch.Tensor:
    """The same through ``_attend``: one program a 64-row block and query head.

    Explicit output must be disjoint from q and each cache's filled [0, p0 + W) prefix.
    """

    w, h, d = q.shape
    hk = _check(q, k_cache, v_cache, p0)
    if out is None:
        out = torch.empty_like(q)
    else:
        if (out.shape != q.shape or out.device != q.device or out.dtype != q.dtype or not out.is_contiguous()):
            raise ValueError("prefill attention: out must be contiguous and like q on q's CUDA device")
        begin, end = out.data_ptr(), out.data_ptr() + out.numel() * out.element_size()
        cache_reads = (p0 + w) * hk * d
        for read, count in ((q, q.numel()), (k_cache, cache_reads), (v_cache, cache_reads)):
            read_begin = read.data_ptr()
            if max(begin, read_begin) < min(end, read_begin + count * read.element_size()):
                raise ValueError("prompt attention output must not overlap q or filled cache inputs")
    if w == 0:
        return out
    _attend[(triton.cdiv(w, BM), h)](q, k_cache, v_cache, out, p0, w, H=h, HK=hk, D=d, BM=BM, BN=BN, SCALE=scale,
                                     num_warps=8, num_stages=1 if d > 128 else 2)
    return out


def heads_a_block(group: int) -> int:
    """Query heads of one KV head in a CUDA block of eight warps: the largest of 8, 4, 2, 1 dividing the group."""

    return next(n for n in (8, 4, 2, 1) if group % n == 0)


def _check(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, p0: int) -> int:
    if (q.ndim != 3 or not q.is_cuda or q.dtype != torch.bfloat16 or q.shape[1] < 1
            or q.shape[2] not in (64, 128, 256)):
        raise ValueError("prefill attention: q must be CUDA (W, positive H, 64, 128 or 256) bf16")
    w, h, d = q.shape
    if (k_cache.ndim != 3 or v_cache.shape != k_cache.shape or k_cache.shape[1] < 1
            or k_cache.shape[2] != d or k_cache.device != q.device or v_cache.device != q.device
            or k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16):
        raise ValueError("prefill attention: caches must be bf16 (keys, positive HK, D) on q's CUDA device")
    hk = k_cache.shape[1]
    if (not isinstance(p0, int) or p0 < 0 or w > 2**31 - 128 or h > 2**31 - 1
            or p0 > 2**31 - 128 - w):
        raise ValueError("prefill attention position/dimensions exceed int32")
    if k_cache.shape[0] < p0 + w or h % hk:
        raise ValueError("prefill attention: caches must hold the chunk's keys; heads a multiple of kv heads")
    if not (q.is_contiguous() and k_cache.is_contiguous() and v_cache.is_contiguous()):
        raise ValueError("prefill attention takes contiguous tensors")
    return hk


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_prefill_attention_v1", sources=[str(here / "prefill_attention.cpp"),
                                                                 str(here / "prefill_attention.cu")],
                extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)

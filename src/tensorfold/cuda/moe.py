"""Softmax top-k MoE with a sigmoid-gated shared expert, on fp32 logits (ties to the lower id); each row routes and runs on its own."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

from . import pdl
from .pdl import wait as pdl_wait

from . import experts as grouped


def _tile(m: int) -> int:
    """Rows a router program takes: 16 to 128, then tiles of 128 (a row's bits never depend on its tile)."""

    for b in (16, 32, 64, 128):
        if m <= b:
            return b
    return 128


@triton.jit
def _router(X, W, OUT, M, x_stride, D: tl.constexpr, NE: tl.constexpr, BM: tl.constexpr,
            BLOCK_E: tl.constexpr, BK: tl.constexpr, PDL: tl.constexpr = False):
    """OUT[m, e] = fp32 x[m] . w[e] (bf16 inputs, tensor cores, K in BK steps in order)."""

    pdl_wait(PDL)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    re = tl.program_id(1) * BLOCK_E + tl.arange(0, BLOCK_E)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    e_ok = re < NE
    acc = tl.zeros((BM, BLOCK_E), dtype=tl.float32)
    for k0 in range(0, D, BK):
        x = tl.load(X + rm[:, None] * x_stride + (k0 + rk)[None, :], mask=m_ok[:, None], other=0.0)
        w = tl.load(W + re[:, None] * D + (k0 + rk)[None, :], mask=e_ok[:, None], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    tl.store(OUT + rm[:, None] * NE + re[None, :], acc, mask=m_ok[:, None] & e_ok[None, :])


@lru_cache(maxsize=1)
def _route_ext():
    from .build import load

    return load(name="tensorfold_route_v1", sources=[str(Path(__file__).parent / "route.cu")],
                extra_cuda_cflags=["-O3", "--ftz=false", "--prec-div=true"], verbose=False)


def route_logits(x: torch.Tensor, rows: torch.Tensor, out: torch.Tensor, *,
                 prenorm: tuple[float, float, torch.Tensor] | None = None) -> torch.Tensor:
    """FP32 router logits of BF16 rows x [R, K] and router rows [E, K] (K a multiple of 32) into ``out`` [R, E].

    A row-invariant split-K tensor-core GEMV: eight K slices fixed by K, each one sequential m16n8k16 chain, added in
    slice order, so a row's bits never depend on R. ``prenorm`` = (eps, root, scale [K] bf16) first forms the compressed-tensors
    Gemma router input bf16(bf16(bf16(x * rsqrt(mean(x^2) + eps)) * root) * scale) in FP32 with BF16 roundings.
    """

    if x.dtype == torch.float16:
        value = x.float()
        if prenorm is not None:
            eps, root, scale = prenorm
            value = ((value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)).half().float()
                     * root).half().float() * scale.float()
            value = value.half().float()
        out.copy_(value @ rows.float().T)
        return out
    eps, root, scale = prenorm if prenorm is not None else (0.0, 0.0, None)
    _route_ext().logits(x, rows, out, float(eps), float(root), scale)
    return out


def route_select(logits: torch.Tensor, expert_scale: torch.Tensor, picks: torch.Tensor,
                 weights: torch.Tensor) -> None:
    """Gemma routing per row: FP32 softmax over all experts, the top ``picks.shape[1]`` (stable descending order,
    lower id among equal logits), weights renormalized over the selected probabilities then times the BF16 per-expert
    scale; INT32 picks and FP32 weights."""

    _route_ext().select(logits, expert_scale, picks, weights)


def router(x: torch.Tensor, rows: torch.Tensor, out: torch.Tensor | None = None, *,
           independent_rows: bool = False) -> torch.Tensor:
    """Rows/router weights to FP32 logits; FP16 decode can preserve serial GEMV arithmetic."""

    m, d = x.shape
    ne = rows.shape[0]
    if out is None:
        out = torch.empty((m, ne), dtype=torch.float32, device=x.device)
    if x.dtype == torch.float16:
        if rows.device != x.device or rows.ndim != 2 or rows.shape[1] != d:
            raise ValueError("FP16 router rows must share the input device and width")
        if independent_rows:
            from .sm70 import routing_native

            if routing_native.supports_logits(x, rows, out):
                return routing_native.logits(x, rows, out)
        value, weight = x.float(), rows.float()
        if independent_rows:
            # Actual M changes the FP32 GEMM reduction, occasionally moving the
            # selected softmax weight across an FP16 rounding boundary. Decode
            # rows must retain the serial target's arithmetic during verification.
            for row in range(m):
                out[row:row + 1].copy_(value[row:row + 1] @ weight.T)
        else:
            out.copy_(value @ weight.T)
        return out
    bm = _tile(m)
    if bm == 16:
        be, bk, stages = 32, 256, 4
    else:
        be, bk, stages = 64, 64, 3               # Smaller K tiles keep prefill within shared-memory limits.
    grid = (triton.cdiv(m, bm), triton.cdiv(ne, be))
    _router[grid](x, rows, out, m, x.stride(0), D=d, NE=ne, BM=bm, BLOCK_E=be, BK=bk, num_warps=4, num_stages=stages,
                  PDL=pdl.ON, launch_pdl=pdl.ON)
    return out


@triton.jit
def _topk_rows(L, PICK, WTS, NE: tl.constexpr, NL: tl.constexpr, TOPK: tl.constexpr, SLOTS: tl.constexpr,
               BLOCK: tl.constexpr, SLOTP: tl.constexpr, PDL: tl.constexpr = False):
    """Program r: row r's TOPK experts by fp32 logit (largest first, the lower id among equal logits), weights exp(l_k - l_0) / sum (fp32, rounded to bf16), then the shared expert (id NE) as slot TOPK with weight bf16(sigmoid(bf16(shared gate logit)))."""

    pdl_wait(PDL)
    r = tl.program_id(0)
    ar = tl.arange(0, BLOCK)
    ak = tl.arange(0, SLOTP)
    v = tl.load(L + r * NL + ar, mask=ar < NE, other=float("-inf"))
    top = tl.max(v, axis=0)
    total = 0.0
    picks = tl.zeros((SLOTP,), dtype=tl.int32)
    exs = tl.zeros((SLOTP,), dtype=tl.float32)
    alive = ar < NE                    # experts not yet picked: a row with fewer finite logits than TOPK still picks
    for k in tl.static_range(TOPK):    # distinct ids (its -inf experts at weight 0), never a repeat or BLOCK
        m = tl.max(tl.where(alive, v, float("-inf")), axis=0)
        idx = tl.min(tl.where(alive & (v == m), ar, BLOCK), axis=0)
        idx = tl.where(idx == BLOCK, tl.min(tl.where(alive, ar, BLOCK), axis=0), idx)   # no equal (NaN): a fresh id
        ex = tl.exp(m - top)
        picks = tl.where(ak == k, idx, picks)
        exs = tl.where(ak == k, ex, exs)
        total += ex
        alive = alive & (ar != idx)
    w = (exs / total).to(tl.bfloat16).to(tl.float32)
    sg = tl.load(L + r * NL + NE).to(tl.bfloat16).to(tl.float32)
    sgw = (1.0 / (1.0 + tl.exp(-sg))).to(tl.bfloat16).to(tl.float32)
    picks = tl.where(ak == TOPK, NE, picks)
    w = tl.where(ak == TOPK, sgw, w)
    tl.store(PICK + r * SLOTS + ak, picks, mask=ak < SLOTS)
    tl.store(WTS + r * SLOTS + ak, w, mask=ak < SLOTS)


class MoEBuffers:
    """Static scratch for up to ``rows`` rows (CUDA-graph safe); ``prefill`` picks the experts' prefill arithmetic,
    ``tile`` its plan's pairs an item (``select``'s default; the kernels' row tile, never a pair's bits)."""

    def __init__(self, rows: int, cfg, device: torch.device | str, *, prefill: bool = False,
                 tile: int = grouped.PREFILL_TILE) -> None:
        slots = cfg.num_experts_per_tok + 1
        self.rows, self.slots, self.tile = rows, slots, tile
        self.execution_dtype = getattr(cfg, "execution_dtype", "bfloat16")
        dtype = torch.float16 if self.execution_dtype == "float16" else torch.bfloat16
        self.logits = torch.empty((rows, cfg.num_experts + 1), dtype=torch.float32, device=device)
        self.pick = torch.empty((rows, slots), dtype=torch.int32, device=device)
        self.wts = torch.empty((rows, slots), dtype=torch.float32, device=device)
        self.plan = grouped.Plan(rows, slots, cfg.num_experts + 1, device, prefill=prefill)
        self.act = torch.empty((rows, slots, cfg.moe_intermediate_size), dtype=dtype, device=device)
        self.y = torch.empty((rows, slots, cfg.hidden_size), dtype=dtype if prefill else torch.float32,
                             device=device)


def select_rows(logits: torch.Tensor, buf: MoEBuffers, top_k: int, experts: int) -> None:
    """Each row's experts and weights, rows in parallel (buf.pick, buf.wts); EXL3 experts group themselves in their kernel."""

    rows = logits.shape[0]
    if buf.act.dtype == torch.float16:
        if logits.shape[1] != experts + 1 or not 1 <= top_k <= experts:
            raise ValueError("FP16 expert selection geometry is invalid")
        from tensorfold.cuda.sm70 import family_ops

        if family_ops._NATIVE_FAMILY and logits.is_cuda and experts == 512 and top_k <= 32:
            from tensorfold.cuda.sm70.routing_native import indices as stable_indices, route as stable_route

            picks, rounded = buf.pick[:rows], buf.wts[:rows]
            if (picks.shape[1] == top_k + 1 and rounded.shape == picks.shape
                    and picks.dtype == torch.int32 and rounded.dtype == torch.float32
                    and picks.is_contiguous() and rounded.is_contiguous()):
                stable_route(logits, picks, rounded)
                return
            indices = stable_indices(logits, top_k)
        else:
            indices = logits[:, :experts].argsort(dim=-1, descending=True, stable=True)[:, :top_k]
        selected = logits[:, :experts].gather(1, indices)
        weights = selected.softmax(-1).half().float()
        buf.pick[:rows, :top_k].copy_(indices)
        buf.pick[:rows, top_k].fill_(experts)
        buf.wts[:rows, :top_k].copy_(weights)
        buf.wts[:rows, top_k].copy_(logits[:, experts].half().float().sigmoid().half().float())
        return
    _topk_rows[(rows,)](logits, buf.pick, buf.wts, NE=experts, NL=logits.shape[1], TOPK=top_k, SLOTS=top_k + 1,
                        BLOCK=triton.next_power_of_2(experts + 1), SLOTP=triton.next_power_of_2(top_k + 1), num_warps=4,
                        PDL=pdl.ON, launch_pdl=pdl.ON)


def select(logits: torch.Tensor, buf: MoEBuffers, top_k: int, experts: int, tile: int | None = None) -> None:
    """Each row's experts and weights (rows in parallel), then the pairs grouped by expert, ``tile`` (default the
    buffer's) a prefill plan's item."""

    select_rows(logits, buf, top_k, experts)
    grouped.route(buf.pick[:logits.shape[0]], buf.plan, buf.tile if tile is None else tile)


def moe(x: torch.Tensor, router_rows: torch.Tensor, ex: grouped.Experts, buf: MoEBuffers, top_k: int,
        experts: int) -> MoEBuffers:
    """Route x [R, D] and run its experts into buf.y [R, k + 1, D] (bf16 in prefill); slot k is the shared expert."""

    rows = x.shape[0]
    router(x, router_rows, buf.logits[:rows])
    if getattr(ex, "kernel", None) in ("gguf", "sm70_dense"):
        select_rows(buf.logits[:rows], buf, top_k, experts)
        ex.forward(x, buf.pick[:rows], out=buf.y[:rows])
        return buf
    select(buf.logits[:rows], buf, top_k, experts)
    grouped.gate_up(x, ex, buf.plan, buf.act.view(-1, ex.width), rows)
    grouped.down(buf.act.view(-1, ex.width), ex, buf.plan, buf.y.view(-1, ex.dims), rows)
    return buf


@triton.jit
def _combine(Y, W, OUT, S: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    """Program (row, column block): the row's slots summed in slot order, fp32 w * y, rounded once to bf16."""

    r = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    ok = d < D
    acc = tl.zeros((BLOCK,), tl.float32)
    for k in tl.static_range(S):
        w = tl.load(W + r * S + k)
        acc += w * tl.load(Y + (r * S + k) * D + d, mask=ok, other=0.0).to(tl.float32)
    tl.store(OUT + r * D + d, acc.to(tl.bfloat16), mask=ok)


def combine(y: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
    """y [R, S, D] (fp32, or bf16 in prefill) and wts [R, S] fp32 -> [R, D] bf16; a row's bits never depend on R."""

    rows, slots, dims = y.shape
    out = torch.empty((rows, dims), dtype=torch.bfloat16, device=y.device)
    _combine[(rows, triton.cdiv(dims, 512))](y, wts, out, S=slots, D=dims, BLOCK=512, num_warps=4)
    return out


def swiglu_bf16(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """BF16 SiLU followed by a BF16 product, matching the Qwen MoE checkpoint reference.

    The intermediate rounding is part of the operator: a fused FP32 SiLU/product
    is a different activation and must not replace this path without comparison.
    """

    return torch.nn.functional.silu(gate) * up


swiglu_bf16.activation = 0              # the fused native-A4 epilogue's SiLU (experts_a4.cu activate<0>)


@dataclass
class Routed:
    """A layer's router rows [E + 1, D] bf16 (the shared expert's gate row last) and its E + 1 experts."""

    router: torch.Tensor
    experts: grouped.Experts
    top_k: int
    run: Callable = moe                         # selected by the loader; affine remains the default

    @property
    def count(self) -> int:
        return self.router.shape[0] - 1


class _Shape:
    def __init__(self, m: Routed) -> None:
        self.num_experts_per_tok, self.num_experts = m.top_k, m.count
        self.moe_intermediate_size, self.hidden_size = m.experts.width, m.experts.dims


_scratch: dict[tuple, MoEBuffers] = {}


def run(x: torch.Tensor, m: Routed, *, prefill: bool = False) -> torch.Tensor:
    """x [R, D] bf16 -> [R, D] bf16: its top-k experts by renormalized softmax weight plus the shared expert."""

    rows = x.shape[0]
    size = 1 << max(4, (rows - 1).bit_length())                 # scratch per power of two, reused by every layer
    key = (size, m.count, m.top_k, m.experts.width, m.experts.dims, prefill, x.device)
    buf = _scratch.get(key)
    if buf is None:
        buf = _scratch[key] = MoEBuffers(size, _Shape(m), x.device, prefill=prefill)
    m.run(x.contiguous(), m.router, m.experts, buf, m.top_k, m.count)
    return combine(buf.y[:rows], buf.wts[:rows])

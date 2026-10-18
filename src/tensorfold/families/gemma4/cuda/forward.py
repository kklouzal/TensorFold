"""Layer-major Gemma 4 verification with fixed per-row arithmetic.

The quantized projections, RMS reductions and attention tile order do not depend
on the number of verification rows. A token-at-a-time scan remains an independent
execution oracle. Taps are raw outputs after zero-based target layer IDs; final
hidden rows have the target's final norm applied exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import triton
import triton.language as tl

from .state import State
from .weights import Gemma4Weights, TAP_LAYERS


@triton.jit
def _rms(X, W, Y, D: tl.constexpr, EPS: tl.constexpr, SCALE: tl.constexpr, B: tl.constexpr,
         ROUND_NORMALIZED: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, B)
    x = tl.load(X + row * D + d, d < D, 0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    y = x * inv
    if ROUND_NORMALIZED:
        # Actual MLX RMSNorm rounds before scale; the CT contract keeps FP32.
        y = y.to(tl.bfloat16).to(tl.float32)
    if SCALE:
        weight = tl.load(W + d, d < D, 0).to(tl.float32)
        y = y * weight
    tl.store(Y + row * D + d, y.to(tl.bfloat16), d < D)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float, *,
             round_normalized: bool = True) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    d = x.shape[-1]
    _rms[(x.numel() // d,)](x, weight if weight is not None else x, out, D=d, EPS=eps,
                           SCALE=weight is not None, B=triton.next_power_of_2(d), ROUND_NORMALIZED=round_normalized,
                           num_warps=4 if d <= 1024 else 8, enable_fp_fusion=False)
    return out


def ct_rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    """CT learned RMS: FP32 normalization and weight product, then one BF16 cast."""
    return rms_norm(x, weight, eps, round_normalized=False)


def _geglu_mlx(gate, up):
    return (torch.nn.functional.gelu(gate.float(), approximate="tanh") * up.float()).to(gate.dtype)


def _geglu_ct(gate, up):
    return torch.nn.functional.gelu(gate, approximate="tanh") * up


@triton.jit
def _rope(X, INV, Y, BASE, H: tl.constexpr, D: tl.constexpr, B: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, B)
    half = D // 2
    x1 = tl.load(X + (row * H + head) * D + d, d < half, 0).to(tl.float32)
    x2 = tl.load(X + (row * H + head) * D + d + half, d < half, 0).to(tl.float32)
    freq = tl.load(INV + d, d < half, 0)
    angle = (BASE + row).to(tl.float32) * freq
    c, s = tl.cos(angle), tl.sin(angle)
    a, b = x1 * c - x2 * s, x1 * s + x2 * c
    tl.store(Y + (row * H + head) * D + d, a.to(tl.bfloat16), d < half)
    tl.store(Y + (row * H + head) * D + d + half, b.to(tl.bfloat16), d < half)


def apply_rope(x: torch.Tensor, inverse: torch.Tensor, position: int) -> torch.Tensor:
    out = torch.empty_like(x)
    rows, heads, d = x.shape
    _rope[(rows, heads)](x, inverse, out, position, H=heads, D=d, B=triton.next_power_of_2(d // 2),
                         num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _attention(Q, K, V, BLOCK_START, BLOCK_END, O, BASE, FIRST, KEYS,
               H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, WINDOW: tl.constexpr):
    row, hk = tl.program_id(0), tl.program_id(1)
    gg = tl.arange(0, 16)
    dd = tl.arange(0, D)
    pos = BASE + row
    block_start = tl.load(BLOCK_START + row)
    block_end = tl.load(BLOCK_END + row)
    earliest = tl.maximum(FIRST, pos - WINDOW + 1) if WINDOW else FIRST
    last = tl.maximum(pos + 1, block_end) if WINDOW else pos + 1
    q = tl.load(Q + (row * H + hk * G + gg[:, None]) * D + dd[None, :], gg[:, None] < G, 0)
    m = tl.full((16,), -float("inf"), tl.float32)
    total = tl.zeros((16,), tl.float32)
    out = tl.zeros((16, D), tl.float32)
    # Absolute 32-key tiles preserve row partition invariance. Single-stage
    # MMA operands must fit GB10 shared memory even for 512-wide global heads.
    for at in range(earliest // 32 * 32, (last + 31) // 32 * 32, 32):
        logical = at + tl.arange(0, 32)
        exists = (logical >= FIRST) & (logical < FIRST + KEYS)
        visible = (logical <= pos) | ((logical >= block_start) & (logical < block_end))
        valid = exists & visible & (logical >= earliest) & (logical < last)
        key = tl.load(K + ((logical[:, None] - FIRST) * HK + hk) * D + dd[None, :], exists[:, None], 0)
        value = tl.load(V + ((logical[:, None] - FIRST) * HK + hk) * D + dd[None, :], exists[:, None], 0)
        scores = tl.dot(q, tl.trans(key)).to(tl.float32)
        scores = tl.where(valid[None, :], scores, -float("inf"))
        tile_max = tl.max(scores, 1)
        active = tile_max != -float("inf")
        next_max = tl.where(active, tl.maximum(m, tile_max), m)
        alpha = tl.where(active, tl.where(m == -float("inf"), 0.0, tl.exp(m - next_max)), 1.0)
        prob = tl.where(valid[None, :] & active[:, None], tl.exp(scores - next_max[:, None]), 0.0)
        out = out * alpha[:, None] + tl.dot(prob.to(tl.bfloat16), value)
        total = total * alpha + tl.sum(prob, 1)
        m = next_max
    result = out / total[:, None]
    tl.store(O + (row * H + hk * G + gg[:, None]) * D + dd[None, :], result.to(tl.bfloat16), gg[:, None] < G)


def attend(q, k, v, position: int, first: int, window: int, block_starts, block_ends):
    """Q[T,H,D], chronological K/V[N,Hk,D], scale1, causal or visual-block SWA."""
    if q.shape[-1] not in (64, 128, 256, 512):
        raise ValueError("Gemma4 CUDA attention head dimensions must be64/128/256/512")
    h, hk, d = q.shape[1], k.shape[1], q.shape[2]
    if h % hk or not 1 <= h // hk <= 16:
        raise ValueError("Gemma4 CUDA attention requires1..16 query heads per KV head")
    out = torch.empty_like(q)
    _attention[(q.shape[0], hk)](q, k, v, block_starts, block_ends, out, position, first, k.shape[0],
                                H=h, HK=hk, D=d, G=h // hk, WINDOW=window,
                                num_warps=4 if d <= 128 else 8, num_stages=1, enable_fp_fusion=False)
    return out


def attention_reference(q, k, v, position: int, first: int, window: int = 0,
                        vision_blocks: Sequence[tuple[int, int]] = ()):
    """Independent unfused FP32 softmax/matmul oracle, not production arithmetic."""
    output = []
    logical = torch.arange(first, first + k.shape[0], device=q.device)
    groups = q.shape[1] // k.shape[1]
    keys = k.repeat_interleave(groups, dim=1).float().transpose(0, 1)
    values = v.repeat_interleave(groups, dim=1).float().transpose(0, 1)
    for row in range(q.shape[0]):
        pos = position + row
        mask = logical <= pos
        if window:
            for begin, end in vision_blocks:
                if begin <= pos < end:
                    mask |= (logical >= begin) & (logical < end)
            mask &= logical >= pos - window + 1
        score = torch.einsum("hd,hkd->hk", q[row].float(), keys)
        probs = torch.softmax(score.masked_fill(~mask, -float("inf")), -1)
        output.append(torch.einsum("hk,hkd->hd", probs, values).to(q.dtype))
    return torch.stack(output)


@dataclass
class ForwardResult:
    hidden: torch.Tensor
    logits: torch.Tensor | None
    taps: dict[int, torch.Tensor]
    start_position: int


class Gemma4Target:
    def __init__(self, weights: Gemma4Weights):
        self.weights, self.config = weights, weights.config
        self.device = weights.norm.device
        policy = weights.numerics
        if policy == "mlx":
            self._norm, self._geglu, self._route = rms_norm, _geglu_mlx, self._route_mlx
            self._embedding = weights.embed.embedding
        elif policy == "ct_nvfp4":
            self._norm, self._geglu, self._route = ct_rms_norm, _geglu_ct, self._route_ct
            self._embedding = lambda ids: torch.nn.functional.embedding(ids, weights.embed.weight)
        else:
            raise ValueError(f"unknown Gemma4 numerical policy: {policy}")
        self._router_root = torch.tensor(self.config.hidden**-0.5, dtype=torch.bfloat16, device=self.device)
        # The BF16 root's exact value, for the fused CT router (read once at construction).
        self._router_root_value = float(torch.tensor(self.config.hidden**-0.5, dtype=torch.bfloat16))
        self.tap_layers = tuple(i for i in TAP_LAYERS if i < self.config.layers)
        self._inverse = []
        c = self.config
        for i in range(c.layers):
            _, d = c.attention_geometry(i)
            rotated = d if c.sliding(i) else int(d * c.global_rotary_fraction)
            base = c.rope_theta if c.sliding(i) else c.global_rope_theta
            # Proportional RoPE divides by full head dimension, not rotary width.
            exponent = torch.arange(0, d, 2, dtype=torch.float32, device=self.device) / d
            inv = torch.pow(base, -exponent)
            inv[rotated // 2:] = 0
            self._inverse.append(inv)

    def make_state(self, context: int, max_rows: int = 16) -> State:
        return State(self.config, self.device, context, min(max_rows, context))

    def raw_embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """Unscaled tied embedding rows for the official Torch DFlash contract."""
        return self._embedding(ids)

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        out = self.raw_embed_tokens(ids)
        # MLX and HF both downcast the scalar to the embedding dtype first.
        scale = torch.tensor(self.config.hidden**0.5, dtype=out.dtype, device=out.device)
        return out * scale

    def project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.weights.embed(hidden)
        return torch.tanh(logits / self.config.softcap) * self.config.softcap

    def _route_mlx(self, h, layer):
        c = self.config
        router_scale = layer.router_scale * (c.hidden**-0.5)
        scores = layer.router(rms_norm(h, router_scale, c.eps))
        picks = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :c.top_k].contiguous()
        selected = torch.gather(scores, 1, picks)
        probabilities = torch.softmax(selected.float(), -1).to(h.dtype)
        return picks, (probabilities * layer.expert_scale[picks]).to(h.dtype)

    def _route_ct(self, h, layer):
        # Publisher GateLinear preprocessing is the unweighted CT RMS norm, root_size then learned scale, each in
        # BF16; its original BF16 matrix emits FP32 logits. Softmax over every expert, the top-k (stable descending,
        # IDs in 0..E-1 as INT32), renormalized over the selected probabilities, times the per-expert scale. Two fused
        # row-invariant kernels; no intermediate leaves the GPU.
        from tensorfold.cuda import moe

        c = self.config
        rows = h.shape[0]
        logits = torch.empty((rows, c.experts), dtype=torch.float32, device=h.device)
        moe.route_logits(h.contiguous(), layer.router.weight, logits,
                         prenorm=(c.eps, self._router_root_value, layer.router_scale))
        picks = torch.empty((rows, c.top_k), dtype=torch.int32, device=h.device)
        routing = torch.empty((rows, c.top_k), dtype=torch.float32, device=h.device)
        moe.route_select(logits, layer.expert_scale, picks, routing)
        return picks, routing

    def _blocks(self, rows: int, position: int, vision_blocks):
        starts = [-1] * rows
        ends = [-1] * rows
        previous = -1
        for block in vision_blocks:
            if (not isinstance(block, (tuple, list)) or len(block) != 2
                    or any(type(x) is not int for x in block)):
                raise ValueError("Gemma4 visual blocks require integer half-open absolute spans")
            begin, end = block
            if not 0 <= begin < end or begin < previous:
                raise ValueError("Gemma4 visual blocks must be nonoverlapping ordered spans")
            previous = end
            if end <= position or begin >= position + rows:
                continue
            if begin < position or end > position + rows:
                raise ValueError("Gemma4 forward must contain each complete uncommitted visual block")
            for row in range(begin - position, end - position):
                starts[row], ends[row] = begin, end
        return (torch.tensor(starts, dtype=torch.int32, device=self.device),
                torch.tensor(ends, dtype=torch.int32, device=self.device))

    @torch.inference_mode()
    def forward(self, tokens, state: State, *, embeddings: torch.Tensor | None = None,
                vision_blocks: Sequence[tuple[int, int]] = (), output_logits: bool = True,
                output_taps: bool = True, trusted_tokens: bool = False) -> ForwardResult:
        """Verify/commit one causal token block; caller rolls back rejected suffix.

        Embeddings, when supplied, already include text sqrt(hidden) scaling and
        visual replacements. A visual block must fit wholly in this call so its
        bidirectional sliding attention sees every row before the layer commits.
        trusted_tokens is reserved for the engine after its prompt/sampler boundary
        proves IDs in range for this unchanged tensor. Invalid API input leaves
        state untouched; a failed GPU forward poisons it.
        """
        if state.config != self.config or torch.device(state.device) != self.device:
            raise ValueError("Gemma4 state belongs to another target")
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.int64, device=self.device)
        if tokens.dtype not in (torch.int32, torch.int64) or tokens.device != self.device or tokens.ndim not in (1, 2):
            raise ValueError("Gemma4 tokens require device-local integer rows")
        if tokens.ndim == 2 and tokens.shape[0] != 1:
            raise ValueError("Gemma4 target forward owns one sequence at a time")
        tokens = tokens.reshape(-1)
        rows = tokens.shape[0]
        state.check(rows)
        if not trusted_tokens and bool(torch.any((tokens < 0) | (tokens >= self.config.vocab)).item()):
            raise ValueError("Gemma4 token ID is outside the target vocabulary")
        if embeddings is not None and (embeddings.shape != (rows, self.config.hidden)
                                       or embeddings.dtype != torch.bfloat16 or embeddings.device != self.device):
            raise ValueError("Gemma4 embedding overrides require BF16[T,hidden] on the target device")
        if vision_blocks and not self.config.vision_bidirectional:
            raise ValueError("Gemma4 checkpoint does not permit visual-block attention")
        start = state.position
        block_starts, block_ends = self._blocks(rows, start, vision_blocks)
        taps = {}
        try:
            h = self.embed_tokens(tokens) if embeddings is None else embeddings
            c = self.config
            for i, layer in enumerate(self.weights.layers):
                hk, d = c.attention_geometry(i)
                normed = self._norm(h, layer.input_norm, c.eps)
                raw_k = layer.k(normed).reshape(rows, hk, d)
                q = self._norm(layer.q(normed).reshape(rows, c.heads, d), layer.q_norm, c.eps)
                k = self._norm(raw_k, layer.k_norm, c.eps)
                v = self._norm(raw_k if layer.v is None else layer.v(normed).reshape(rows, hk, d), None, c.eps)
                q, k = apply_rope(q, self._inverse[i], start), apply_rope(k, self._inverse[i], start)
                cache = state.kv[i]
                if c.sliding(i):
                    prev_k, prev_v, first = cache.before(start, c.window)
                    keys, values = torch.cat((prev_k, k)), torch.cat((prev_v, v))
                else:
                    cache.write(k, v, start, state.context)
                    first = 0
                    keys, values = cache.keys[:start + rows], cache.values[:start + rows]
                out = attend(q, keys, values, start, first, c.window if c.sliding(i) else 0,
                             block_starts, block_ends)
                if c.sliding(i):
                    cache.write(k, v, start, state.context)
                h = h + self._norm(layer.o(out.reshape(rows, -1)), layer.attention_norm, c.eps)
                residual = h
                dense_in = self._norm(h, layer.mlp_norm, c.eps)
                gate, up = layer.gate(dense_in), layer.up(dense_in)
                active = self._geglu(gate, up)
                dense = self._norm(layer.down(active), layer.dense_output_norm, c.eps)
                picks, routing = self._route(h, layer)
                expert_in = self._norm(h, layer.expert_norm, c.eps)
                routed = layer.experts.forward(expert_in, picks, routing)
                routed = self._norm(routed, layer.expert_output_norm, c.eps)
                h = (residual + self._norm(dense + routed, layer.output_norm, c.eps)) * layer.scalar
                if output_taps and i in self.tap_layers:
                    taps[i] = h
            hidden = self._norm(h, self.weights.norm, c.eps)
            logits = self.project_logits(hidden) if output_logits else None
            state.position = start + rows
            return ForwardResult(hidden, logits, taps, start)
        except BaseException:
            state.poisoned = True
            raise

    def forward_serial(self, tokens, state: State, *, output_logits: bool = True) -> ForwardResult:
        """Token-at-a-time oracle for row-count/rollback checks, with no visual blocks."""
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.int64, device=self.device)
        tokens = tokens.reshape(-1)
        state.check(tokens.numel())
        parts = [self.forward(tokens[i:i + 1], state, output_logits=output_logits) for i in range(tokens.numel())]
        return ForwardResult(torch.cat([p.hidden for p in parts]),
                             torch.cat([p.logits for p in parts]) if output_logits else None,
                             {i: torch.cat([p.taps[i] for p in parts]) for i in self.tap_layers},
                             parts[0].start_position)

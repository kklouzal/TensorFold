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

from . import attention
from .state import State, sliding_slots
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


@dataclass
class ForwardResult:
    hidden: torch.Tensor
    logits: torch.Tensor | None
    taps: dict[int, torch.Tensor]
    start_position: int


class Gemma4Target:
    def __init__(self, weights: Gemma4Weights):
        self.weights, self.config = weights, weights.config
        attention.check_geometry(self.config)
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
        # Device-resident once: a per-forward scalar upload would be a host copy inside captured graphs.
        self._embed_scale = torch.tensor(self.config.hidden**0.5, dtype=torch.bfloat16, device=self.device)
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

    def make_state(self, context: int, rows: int = 16) -> State:
        """Persistent caches for one sequence whose forwards carry at most ``rows`` rows."""
        return State(self.config, self.device, context, min(rows, context))

    def raw_embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """Unscaled tied embedding rows for the official Torch DFlash contract."""
        return self._embedding(ids)

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        # MLX and HF both downcast the scalar to the embedding dtype first.
        return self.raw_embed_tokens(ids) * self._embed_scale

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
        # Publisher GateLinear preprocessing is root_size then learned scale,
        # each in BF16; its original BF16 matrix emits FP32 logits.
        normalized = ct_rms_norm(h, None, self.config.eps)
        router_input = (normalized * self._router_root) * layer.router_scale
        scores = layer.router(router_input)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        picks = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :self.config.top_k].contiguous()
        selected = torch.gather(probabilities, 1, picks)
        routing = selected / selected.sum(-1, keepdim=True)
        routing = routing * layer.expert_scale[picks].float()
        # argsort constructs IDs in0..E-1; the native expert ABI uses int32.
        return picks.to(torch.int32), routing.contiguous()

    def _blocks(self, rows: int, position: int, vision_blocks):
        """Per-row visual block bounds on the device and the longest block in this forward (None, None, 0 without)."""
        starts = [-1] * rows
        ends = [-1] * rows
        previous = -1
        longest = 0
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
            longest = max(longest, end - begin)
            for row in range(begin - position, end - position):
                starts[row], ends[row] = begin, end
        if not longest:
            return None, None, 0
        return (torch.tensor(starts, dtype=torch.int32, device=self.device),
                torch.tensor(ends, dtype=torch.int32, device=self.device), longest)

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
        block_starts, block_ends, longest = self._blocks(rows, start, vision_blocks)
        try:
            h = self.embed_tokens(tokens) if embeddings is None else embeddings
            state.pos.fill_(start)
            chunks = (start + rows - 1) // attention.GLOBAL_CHUNK + 1
            hidden, logits, taps = self._layers(h, state, rows, chunks, block_starts, block_ends, longest,
                                                output_logits, output_taps)
            state.advance(rows)
            return ForwardResult(hidden, logits, taps, start)
        except BaseException:
            state.poisoned = True
            raise

    def forward_staged(self, tokens: torch.Tensor, state: State, *, chunks: int, output_logits: bool = True,
                       output_taps: bool = True):
        """Text rows from device token IDs at the device position ``state.pos``, without host syncs.

        For CUDA graph capture/replay: ``tokens`` is a fixed device int32/int64
        buffer of trusted IDs, the caller writes the first position into
        ``state.pos`` before each launch and commits rows with
        ``state.advance``/``state.rollback`` afterwards. ``chunks`` is the fixed
        full-attention grid (absolute 512-key chunks of the longest admitted
        position, e.g. a context bucket); programs past a row's keys exit early,
        so any bucket covering the last row computes identical bits.
        This call admits the host position it sees; before every replay the
        caller re-checks ``state.check(rows)`` and that ``chunks`` covers
        ``state.position + rows`` (a short grid would fold unwritten partials).
        A failed launch poisons the state. Returns (hidden, logits or None, taps).
        """
        if (state.config != self.config or torch.device(state.device) != self.device
                or not isinstance(tokens, torch.Tensor) or tokens.ndim != 1
                or tokens.dtype not in (torch.int32, torch.int64) or tokens.device != self.device):
            raise ValueError("Gemma4 staged forward needs this target's state and a device-local 1-D token buffer")
        rows = tokens.shape[0]
        state.check(rows)
        if (type(chunks) is not int or chunks <= (state.position + rows - 1) // attention.GLOBAL_CHUNK
                or chunks > -(-state.context // attention.GLOBAL_CHUNK)):
            raise ValueError("Gemma4 staged full-attention grid must cover the last row within the admitted context")
        try:
            return self._layers(self.embed_tokens(tokens), state, rows, chunks, None, None, 0, output_logits,
                                output_taps)
        except BaseException:
            state.poisoned = True
            raise

    def _layers(self, h, state: State, rows: int, chunks: int, block_starts, block_ends, longest: int,
                output_logits: bool, output_taps: bool):
        c = self.config
        mlx = self.weights.numerics == "mlx"
        slots = sliding_slots(c.window, rows, c.heads // c.kv_heads, longest)
        if not longest:
            # Block bounds are read only when blocks are present; any int32 device pointer stands in.
            block_starts = block_ends = state.pos
        taps = {}
        for i, layer in enumerate(self.weights.layers):
            sliding, index = state.slot[i]
            normed = self._norm(h, layer.input_norm, c.eps)
            q_raw, k_raw = layer.q(normed), layer.k(normed)
            if sliding:
                keys, values = state.keys[index], state.values[index]
                q = attention.prep_sliding(q_raw, k_raw, layer.v(normed), layer.q_norm, layer.k_norm,
                                           self._inverse[i], state.pos, keys, values, eps=c.eps,
                                           round_normalized=mlx)
                out = attention.sliding(q, keys, values, state.pos, block_starts, block_ends,
                                        window=c.window, slots=slots, blocks=bool(longest))
            else:
                raw, inv = state.raw[index], state.inv[index]
                q = attention.prep_full(q_raw, k_raw, layer.q_norm, self._inverse[i], state.pos, raw, inv,
                                        eps=c.eps, round_normalized=mlx)
                out = attention.full(q, raw, inv, layer.k_norm, self._inverse[i], state.pos, chunks=chunks,
                                     round_normalized=mlx)
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
        return hidden, logits, taps

    def forward_serial(self, tokens, state: State, *, output_logits: bool = True) -> ForwardResult:
        """Token-at-a-time oracle for row-count/rollback checks, with no visual blocks."""
        if not isinstance(tokens, torch.Tensor):
            tokens = torch.tensor(tokens, dtype=torch.int64, device=self.device)
        tokens = tokens.reshape(-1)
        # Whole-scan admission before the first row; each single-row forward rechecks its own state.
        if state.poisoned or not 0 < tokens.numel() <= state.context - state.position:
            raise ValueError("Gemma4 serial scan exceeds the admitted context or uses a poisoned state")
        parts = [self.forward(tokens[i:i + 1], state, output_logits=output_logits) for i in range(tokens.numel())]
        return ForwardResult(torch.cat([p.hidden for p in parts]),
                             torch.cat([p.logits for p in parts]) if output_logits else None,
                             {i: torch.cat([p.taps[i] for p in parts]) for i in self.tap_layers},
                             parts[0].start_position)

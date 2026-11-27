"""Pure-torch reference of the 1Cat-vLLM V4.1 DSpark head math (no Triton kernels), for unit gates only.

Written independently from upstream's equations (``vllm/models/deepseek_v41/{dspark,model,quant_rope,
attention_ops,markov_fused}.py`` and the mHC/router kernels @4d57473d) with the same dtype boundaries: FP32 HC
streams, FP16 projection outputs (FP32 accumulation), FP32 norms rounded to FP16, FP8 QDQ of window KV and of the
attention output (group 128, power-of-two scale), FP16 attention probabilities, FP32 routing, MXFP4 experts with
exact weights (``mxfp4_ds41.reference``), FP32 partial sums. It is a mathematical oracle, not a bit-parity claim:
reduction order differs from the kernels.
"""

from __future__ import annotations

import torch

from tensorfold.cuda.sm70 import mxfp4_ds41 as mx

from . import dspark_weights as dw

EPS = 1e-20


def rms(x, weight):
    v = x.float()
    return (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + EPS) * weight).to(x.dtype)


def linear(x, weight):
    return (x.float() @ weight.float().t()).half()


def rope(x, positions, inverse=False):
    """Interleaved rotation of the last 64 dims (FP32 angles position * 10000^(-2i/64))."""
    freq = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=x.device) / 64))
    angle = positions.float()[:, None] * freq[None]
    c, s = torch.cos(angle), torch.sin(angle)
    if inverse:
        s = -s
    shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (32,)
    c, s = c.view(shape), s.view(shape)
    out = x.clone()
    a, b = x[..., -64::2].float(), x[..., -63::2].float()
    out[..., -64::2] = (a * c - b * s).half()
    out[..., -63::2] = (a * s + b * c).half()
    return out


def qdq_fp8(x, group=128):
    v = x.float().view(*x.shape[:-1], -1, group)
    amax = v.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    m, e = torch.frexp(amax / 448.0)                 # ceil to a power of two
    scale = torch.ldexp(torch.ones_like(m), torch.where(m == 0.5, e - 1, e))
    q = (v / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (q * scale).view(x.shape).half()


def hc_mixes(residual, fn, scale, base, iters=20, eps=1e-6):
    flat = residual.flatten(1).float()
    mixes = (flat @ fn.t()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + EPS)
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + eps
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])
    comb = torch.softmax((mixes[:, 8:] * scale[2] + base[8:]).view(-1, 4, 4), -1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def collapse(residual, pre, weight=None):
    value = (residual.float() * pre[..., None]).sum(1)
    if weight is None:
        return value
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + EPS) * weight).half()


def expand(x, residual, post, comb):
    return torch.einsum("tij,tih->tjh", comb, residual.float()) + post[..., None] * x[:, None].float()


def route(x, gate, bias, top=3, scale=1.5):
    logits = x.float() @ gate.t()
    score = torch.sqrt(torch.nn.functional.softplus(logits))
    ids = (score + bias).topk(top, -1).indices
    p = score.gather(1, ids)
    return p / (p.sum(-1, keepdim=True) + 1e-20) * scale, ids


def shared_swiglu(g, u):
    g, u = g.float().clamp(max=10), u.float().clamp(-10, 10)
    return (g / (1 + torch.exp(-g)) * u).half()


def attention(s, x, positions, window, valid):
    """x FP16 [5, H]; window FP16 [128, 512] with valid [128] (one request)."""
    n = x.shape[0]
    heads = s.sink.shape[0]
    q = rope(linear(rms(linear(x, s.wq_a), s.q_norm), s.wq_b).view(n, heads, 512), positions)
    kv = qdq_fp8(rope(rms(linear(x, s.wkv), s.kv_norm), positions))
    keys = torch.cat((torch.where(valid[:, None], window, torch.zeros_like(window)), kv))
    live = torch.cat((valid, torch.ones(n, dtype=torch.bool, device=x.device)))
    scores = (q.float() @ keys.float().t()) * 512 ** -0.5                       # [n, heads, 133]
    scores = scores.masked_fill(~live, float("-inf"))
    z = s.sink.view(1, heads, 1)
    m = torch.maximum(scores.amax(-1, keepdim=True), z)
    e = torch.exp(scores - m)
    p = (e / (e.sum(-1, keepdim=True) + torch.exp(z - m))).half()
    out = (p.float() @ keys.float()).half()
    out = rope(out, positions, inverse=True)
    out = qdq_fp8(out.reshape(n, s.wo_a.shape[0], -1))
    projected = torch.einsum("ngk,gok->ngo", out.float(), s.wo_a.float()).half().reshape(n, -1)
    return linear(projected, s.wo_b).float()


def moe(s, x, first, local_count, q8=False):
    probs, ids = route(x, s.gate, s.gate_bias)
    local = ids - first
    local = torch.where((local >= 0) & (local < local_count), local, torch.full_like(local, -1)).to(torch.int32)
    routed = mx.reference(x, local, probs, s.exp_gate, s.exp_gate_scale, s.exp_up, s.exp_up_scale, s.exp_down,
                          s.exp_down_scale, q8=q8)
    shared = linear(shared_swiglu(linear(x, s.sh_gate), linear(x, s.sh_up)), s.sh_down)
    return routed.float() + shared.float()


def context_rows(w, taps, positions):
    states = rms(linear(taps, w.main_proj), w.main_norm)
    return [qdq_fp8(rope(rms(linear(states, s.wkv), s.kv_norm), positions)) for s in w.stages]


def forward_one(w, embeds, anchor_pos, windows, valid, q8=False):
    """One request: embeds FP16 [5, H] of [anchor, noise x4]; windows per stage FP16 [128, 512]."""
    positions = anchor_pos + torch.arange(dw.BLOCK, device=embeds.device)
    hidden = embeds.float()[:, None].expand(-1, 4, -1).clone()
    pre = torch.zeros(dw.BLOCK, 4, device=embeds.device)
    pre[:, 0] = 1
    first, count = w.first_expert, len(w.layout.experts)
    for s, window in zip(w.stages, windows):
        hc = s.hc
        a_pre, post, comb = hc_mixes(hidden, hc["attn_fn"], hc["attn_scale"], hc["attn_base"])
        hidden = expand(attention(s, collapse(hidden, pre, s.attn_norm), positions, window, valid), hidden, post,
                        comb)
        pre, post, comb = hc_mixes(hidden, hc["ffn_fn"], hc["ffn_scale"], hc["ffn_base"])
        hidden = expand(moe(s, collapse(hidden, a_pre, s.ffn_norm), first, count, q8), hidden, post, comb)
    return collapse(hidden, pre)


def markov_greedy(w, logits, anchor):
    """logits FP32 [5, V]; argmax(logits_j + W_head @ embed(prev_j)), lowest index on ties."""
    head = w.markov_head.float()
    prev, out = int(anchor), []
    for j in range(dw.BLOCK):
        bias = head @ w.markov_embed[prev].float()
        value = logits[j].float() + bias
        prev = int(torch.nonzero(value == value.max())[0])
        out.append(prev)
    return out

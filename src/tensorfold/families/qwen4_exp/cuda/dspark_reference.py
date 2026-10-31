# SPDX-License-Identifier: Apache-2.0
# DSpark/DFlash equations follow SGLang v0.5.21, sgl-project/sglang
# (Apache-2.0), models/{dspark,dflash}.py and speculative/dspark_components.
# HC tap reduction follows shadowpa0327/Qwen3.8-Flash-Next-DSpark revision
# 3f74bdfa4a1b9deaaa56cc7c1377440ec5095fad, adapted from SGLang/SpecForge.
"""Pure Torch FP16 operands with explicit FP32 arithmetic, CPU reference only.

This is a mathematical reference, not a claim of BF16/Hopper kernel bit parity.
The proposal block is bidirectional. Only target-derived context K/V persists.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def linear(x, weight):
    # Match the fork's fixed-row projection policy for committed prefix rows:
    # stage/commit piece boundaries must not choose a different GEMM geometry.
    result = torch.empty((x.shape[0], weight.shape[0]), dtype=torch.float16, device=x.device)
    full_weight = weight.float()
    for start in range(0, x.shape[0], 32):
        count = min(32, x.shape[0] - start)
        padded = torch.zeros((32, x.shape[1]), dtype=torch.float32, device=x.device)
        padded[:count].copy_(x[start : start + count])
        value = padded @ full_weight.T
        result[start : start + count].copy_(value[:count])
    return result


def rms(x, scale, eps):
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * scale.float()).half()


def add_norm(hidden, residual, scale, eps):
    # SGLang RMSNorm.forward_native normalizes the unrounded FP32 sum, while
    # retaining its FP16-rounded residual for the next fused add.
    value = hidden.float() if residual is None else hidden.float() + residual.float()
    saved = hidden if residual is None else value.half()
    return rms(value, scale, eps), saved


def rope(x, positions, theta):
    dim = x.shape[-1]
    inv = theta ** (-torch.arange(0, dim, 2, dtype=torch.float32, device=x.device) / dim)
    angle = positions.float()[:, None] * inv[None]
    cosine, sine = angle.cos()[:, None], angle.sin()[:, None]
    a, b = x.float().chunk(2, dim=-1)
    return torch.cat((a * cosine - b * sine, a * sine + b * cosine), dim=-1).half()


def context_kv(weights, features, start):
    c, t = weights.config, weights.tensors
    projected = rms(linear(features, t["fc.weight"]), t["hidden_norm.weight"], c.eps)
    positions = torch.arange(start, start + features.shape[0], device=features.device)
    keys = []
    values = []
    for layer in range(c.layers):
        base = f"layers.{layer}.self_attn."
        k = linear(projected, t[base + "k_proj.weight"]).view(-1, c.kv_heads, c.head_dim)
        keys.append(rope(rms(k, t[base + "k_norm.weight"], c.eps), positions, c.theta))
        values.append(linear(projected, t[base + "v_proj.weight"]).view(-1, c.kv_heads, c.head_dim))
    return torch.stack(keys), torch.stack(values)


def attend(q, k, v):
    """Unmasked block+prefix GQA, FP32 scores/softmax/accumulation."""
    rows, heads, dim = q.shape
    kv = k.shape[1]
    grouped = q.float().reshape(rows, kv, heads // kv, dim)
    scores = torch.einsum("rhgd,chd->hgrc", grouped, k.float()) * (dim**-0.5)
    probs = scores.softmax(-1)
    return torch.einsum("hgrc,chd->rhgd", probs, v.float()).reshape(rows, heads * dim).half()


def block_hidden(weights, input_embeds, keys, values, length):
    c, t = weights.config, weights.tensors
    positions = torch.arange(length, length + input_embeds.shape[0], device=input_embeds.device)
    hidden, residual = input_embeds, None
    for layer in range(c.layers):
        base = f"layers.{layer}."
        x, residual = add_norm(hidden, residual, t[base + "input_layernorm.weight"], c.eps)
        ab = base + "self_attn."
        q = linear(x, t[ab + "q_proj.weight"]).view(-1, c.heads, c.head_dim)
        k = linear(x, t[ab + "k_proj.weight"]).view(-1, c.kv_heads, c.head_dim)
        v = linear(x, t[ab + "v_proj.weight"]).view(-1, c.kv_heads, c.head_dim)
        q = rope(rms(q, t[ab + "q_norm.weight"], c.eps), positions, c.theta)
        k = rope(rms(k, t[ab + "k_norm.weight"], c.eps), positions, c.theta)
        all_k = torch.cat((keys[layer, :length], k))
        all_v = torch.cat((values[layer, :length], v))
        hidden = linear(attend(q, all_k, all_v), t[ab + "o_proj.weight"])
        x, residual = add_norm(hidden, residual, t[base + "post_attention_layernorm.weight"], c.eps)
        gate = linear(x, t[base + "mlp.gate_proj.weight"])
        up = linear(x, t[base + "mlp.up_proj.weight"])
        activation = (F.silu(gate.float()) * up.float()).half()
        hidden = linear(activation, t[base + "mlp.down_proj.weight"])
    return add_norm(hidden, residual, t["norm.weight"], c.eps)[0]

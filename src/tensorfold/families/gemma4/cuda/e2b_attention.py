"""Row-invariant attention over native HF chronological independent-K/V caches.

Absolute256-key partitions and fixed64/32-key inner tiles give each causal query
the same BF16 dots/FP32 online-softmax reductions regardless of verifier rows.
Transformers continues to own projections, norms, RoPE and cache publication.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention, apply_rotary_pos_emb

from .attention import _online, _fold_step

CHUNK = 256


@triton.jit(do_not_specialize=["R", "LENGTH", "START", "ORIGIN", "BASE", "SPLITS", "MQ", "MK", "MASKED", "KH", "VH"],
            do_not_specialize_on_alignment=["MASK"])
def _parts(Q, K, V, MASK, PO, PM, PL, H: tl.constexpr, D: tl.constexpr, R,
           LENGTH, START, ORIGIN, BASE, SPLITS, WINDOW: tl.constexpr, QS: tl.constexpr, QH: tl.constexpr,
           KS: tl.constexpr, VS: tl.constexpr, MQ, MK,
           MASKED, BK: tl.constexpr, CH: tl.constexpr, PACK: tl.constexpr,
           HK: tl.constexpr, KH, VH):
    split = tl.program_id(1)
    # PACK1 is a qualification candidate. The selected incumbent packs two
    # queries; runtime mask dispatch must preserve its single arithmetic graph.
    lane = tl.arange(0, 16)
    if PACK == 1:
        row = tl.full((16,), tl.program_id(0), tl.int32)
        head = lane
        live = head < H
    else:
        row = tl.program_id(0) * 2 + lane // H
        head = lane % H
        live = row < R
    if HK != 1:
        group: tl.constexpr = H // HK
        row = tl.program_id(0) * PACK + lane // group
        head = tl.program_id(2) * group + lane % group
        live = (row < R) & (lane < PACK * group)
    absolute = START + row
    dd = tl.arange(0, D)
    q = tl.load(Q + row[:, None] * QS + head[:, None] * QH + dd[None, :],
                mask=live[:, None], other=0.0)
    m = tl.full((16,), -float("inf"), tl.float32)
    mass = tl.zeros((16,), tl.float32)
    out = tl.zeros((16, D), tl.float32)
    for offset in range(0, CH, BK):
        key = (BASE + split) * CH + offset + tl.arange(0, BK)
        index = key - ORIGIN
        held = (index >= 0) & (index < LENGTH)
        if HK == 1:
            k = tl.load(K + index[:, None] * KS + dd[None, :], mask=held[:, None], other=0.0)
            v = tl.load(V + index[:, None] * VS + dd[None, :], mask=held[:, None], other=0.0)
        else:
            k = tl.load(K + tl.program_id(2) * KH + index[:, None] * KS + dd[None, :],
                        mask=held[:, None], other=0.0)
            v = tl.load(V + tl.program_id(2) * VH + index[:, None] * VS + dd[None, :],
                        mask=held[:, None], other=0.0)
        valid = live[:, None] & held[None, :] & (key[None, :] <= absolute[:, None])
        if WINDOW:
            valid &= key[None, :] >= absolute[:, None] - WINDOW + 1
        score = tl.dot(q, tl.trans(k)).to(tl.float32)
        if MASKED:
            mask = tl.load(MASK + row[:, None] * MQ + index[None, :] * MK,
                           mask=live[:, None] & held[None, :], other=0)
            valid &= mask != 0
        m, mass, out = _online(score, valid, m, mass, out, v)
    slot = (row * H + head) * SPLITS + split
    tl.store(PO + slot[:, None] * D + dd[None, :], out, mask=live[:, None])
    tl.store(PM + slot, m, mask=live)
    tl.store(PL + slot, mass, mask=live)


@triton.jit(do_not_specialize=["SPLITS"])
def _finish(PO, PM, PL, OUT, H: tl.constexpr, D: tl.constexpr, SPLITS):
    row, head = tl.program_id(0), tl.program_id(1)
    dd = tl.program_id(2) * 64 + tl.arange(0, 64)
    m = tl.full((1,), -float("inf"), tl.float32)
    mass = tl.zeros((1,), tl.float32)
    out = tl.zeros((1, 64), tl.float32)
    for split in range(SPLITS):
        slot = (row * H + head) * SPLITS + split
        cm = tl.load(PM + slot + tl.zeros((1,), tl.int32))
        cl = tl.load(PL + slot + tl.zeros((1,), tl.int32))
        co = tl.load(PO + slot * D + dd)[None, :]
        m, mass, out = _fold_step(m, mass, out, cm, cl, co)
    value = tl.where(mass[:, None] > 0, out / mass[:, None], 0.0)
    tl.store(OUT + (row * H + head) * D + dd[None, :], value.to(tl.bfloat16))


def chronological_attention(query, key, value, mask, *, start: int, window: int, query_rows=2):
    """Single sequence,8 query heads/1 or2 KV heads, BF16D256/512, scale1/dropout0."""
    _, heads, rows, dim = query.shape
    kv_heads = key.shape[1]
    length = key.shape[-2]
    if (query.dtype != torch.bfloat16 or key.dtype != query.dtype or value.dtype != query.dtype
            or query.shape[0] != 1 or heads != 8 or dim not in (256, 512)
            or kv_heads not in (1, 2) or tuple(key.shape) != (1, kv_heads, length, dim) or value.shape != key.shape
            or query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1
            or window not in (0, 512) or type(start) is not int or start < 0
            or type(query_rows) is not int or query_rows not in (1, 2)):
        raise ValueError("Gemma E2B canonical attention geometry differs")
    origin = start + rows - length
    base = max(0, start - window + 1) // CHUNK if window else 0
    splits = (start + rows + CHUNK - 1) // CHUNK - base
    mq = mk = 0
    if mask is not None:
        if (mask.dtype != torch.bool or mask.ndim != 4 or mask.shape[:2] != (1, 1)
                or mask.shape[-1] != length or mask.shape[-2] not in (1, rows)):
            raise ValueError("Gemma E2B canonical attention requires its native SDPA boolean broadcast mask")
        mq, mk = (0 if mask.shape[-2] == 1 else mask.stride(-2)), mask.stride(-1)
    partial = torch.empty((rows, heads, splits, dim), device=query.device, dtype=torch.float32)
    maxima = torch.empty((rows, heads, splits), device=query.device, dtype=torch.float32)
    masses = torch.empty_like(maxima)
    out = torch.empty((1, rows, heads, dim), device=query.device, dtype=torch.bfloat16)
    # U8 views preserve one kernel pointer signature. With no external mask,
    # the borrowed query-byte view is never read by the uniform runtime branch.
    mask_bytes = (mask if mask is not None else query).view(torch.uint8)
    _parts[(triton.cdiv(rows, query_rows), splits, kv_heads)](
        query, key, value, mask_bytes, partial, maxima, masses,
        heads, dim, rows, length, start, origin, base, splits, window,
        query.stride(2), query.stride(1), key.stride(2), value.stride(2), mq, mk,
        mask is not None, 32 if dim == 512 else 64, CHUNK, query_rows,
        kv_heads, key.stride(1), value.stride(1),
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    _finish[(rows, heads, dim // 64)](partial, maxima, masses, out, heads, dim, splits,
                                    num_warps=1, enable_fp_fusion=False)
    return out


class E2BTextAttention(Gemma4TextAttention):
    """Owned dispatch gap adapter; native HF math surrounds the selected core."""

    def __init__(self, native, policy):
        torch.nn.Module.__init__(self)
        # Retain loaded children/parameters and state_dict paths, not copies.
        for name, module in native.named_children():
            self.add_module(name, module)
        for name in ("config", "layer_idx", "layer_type", "is_sliding", "sliding_window", "head_dim",
                     "use_alternative_attention", "num_key_value_groups", "scaling", "attention_dropout",
                     "is_causal", "is_kv_shared_layer", "store_full_length_kv"):
            setattr(self, name, getattr(native, name))
        if not native.is_kv_shared_layer and native.v_proj is None:
            self.v_proj = None
        self.policy = policy
        self.train(native.training)

    def forward(self, hidden_states, position_embeddings, attention_mask, shared_kv_states,
                past_key_values=None, **kwargs):
        if self.policy.prompt:
            return super().forward(hidden_states, position_embeddings, attention_mask, shared_kv_states,
                                   past_key_values=past_key_values, **kwargs)
        shape = hidden_states.shape[:-1]
        projection_shape = (*shape, -1, self.head_dim)
        cos, sin = position_embeddings
        query = self.q_norm(self.q_proj(hidden_states).view(projection_shape))
        query = apply_rotary_pos_emb(query, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        if self.is_kv_shared_layer:
            key, value = shared_kv_states[self.layer_type]
        else:
            key = self.k_norm(self.k_proj(hidden_states).view(projection_shape))
            key = apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2).transpose(1, 2)
            value = self.v_norm(self.v_proj(hidden_states).view(projection_shape)).transpose(1, 2)
            key, value = past_key_values.update(key, value, self.layer_idx)
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = key, value
        result = chronological_attention(query, key, value, attention_mask,
                                         start=self.policy.position, window=self.sliding_window or 0)
        return self.o_proj(result.reshape(*shape, -1).contiguous()), None


def install_attention(language_model, policy):
    for layer in language_model.layers:
        layer.self_attn = E2BTextAttention(layer.self_attn, policy)

"""FP16 execution adapters for TensorFold's Flash Next family contracts.

These operations retain the family's explicit stored-tensor roundings, FP32
reductions and read-only committed-state rules. They use ordinary CUDA ATen
operations, so Volta never reaches a BF16/Triton tensor-core implementation.
The existing family owns all weights, request state, cache formats and buffers.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tensorfold.families.qwen4_exp.cuda.indexer_history import HistoryLayer


def group_sums(value, out):
    out.copy_(value.float().reshape(*value.shape[:-1], value.shape[-1] // 32, 32).sum(-1))
    return out


def _half(value):
    return value.to(torch.float16)


def _sigmoid(value):
    return torch.sigmoid(value.float()).half()


def _silu(value):
    return F.silu(value.float()).half()


def _matmul32(left, right):
    if left.is_cuda and left.dtype == right.dtype == torch.float16:
        return torch.mm(left, right, out_dtype=torch.float32)
    return left.float() @ right.float()


def h32(value):
    width = value.shape[-1]
    if width % 32:
        raise ValueError("native integer KV needs complete 32-value groups")
    data = value.float().reshape(*value.shape[:-1], width // 32, 32)
    for low in (1, 2, 4, 8, 16):
        shape = (*data.shape[:-1], 32 // (2 * low), 2, low)
        pair = data.reshape(shape)
        first, second = pair[..., 0, :], pair[..., 1, :]
        data = torch.stack((first + second, first - second), dim=-2).reshape(data.shape)
    return (data * (1.0 / math.sqrt(32))).reshape(value.shape)


def _kv_plan(bits, codec, k_bits, k_codec, v_bits, v_codec):
    fields = (k_bits, k_codec, v_bits, v_codec)
    if all(field is None for field in fields):
        fields = (bits, codec, bits, codec)
    from tensorfold.families.qwen4_exp.cuda.sm70_kv import check_kernel_fields

    return check_kernel_fields(*fields)


def _kv_payload_dtype(bits):
    return torch.float16 if not bits else torch.int8 if bits == 8 else torch.uint8


def _rotor_variant(codec):
    if codec == 4:
        return "signed-iso128"
    if codec == 5:
        return "signed-iso64-norm"
    raise ValueError("SM70 Rotor Q4 requires an admitted signed basis")


def _rotor_rotate(value, out, *, codec=4, inverse=False, status=None, error=1, layer=0, valid_rows=None):
    """The selected cache basis has a single staged FP32 rotation authority.

    CUDA uses the capture-safe native primitive even when the surrounding
    family operation uses Torch. The independent CPU reference is also useful
    for callers without a CUDA execution context; it is not a GPU readback.
    """
    if value.is_cuda:
        from .rotorquant_q4_native import rotate

        return rotate(value, out, codec=codec, inverse=inverse, status=status, error=error,
                      layer=layer, valid_rows=valid_rows)
    from tensorfold.families.qwen4_exp.cuda.rotorquant_ref import rotate_ref

    result = rotate_ref(value, _rotor_variant(codec), inverse=inverse)
    stored = result.to(out.dtype)
    stored = _guard(stored, status, error, layer, valid_rows)
    out.copy_(stored)
    return out


def _rotor_encode(value, codes, scales, position, *, codec=4, status=None, error=2, layer=0, valid_rows=None):
    """Write only live cache rows; their payload and metadata keep one owner."""
    if value.is_cuda:
        from .rotorquant_q4_native import encode

        return encode(value, codes, scales, codec=codec, position=position, status=status, error=error,
                      layer=layer, valid_rows=valid_rows)
    from tensorfold.families.qwen4_exp.cuda.rotorquant_ref import quantize_ref, dequant_ref

    live = value.shape[0] if valid_rows is None else int(valid_rows.reshape(-1)[0])
    base = int(position.reshape(-1)[0])
    if not 0 <= live <= value.shape[0] or live and not 0 <= base <= codes.shape[0] - live:
        if status is None:
            raise ValueError("Rotor Q4 live rows and position must fit the owned cache")
        status[0].bitwise_or_(16)
        if status[1] == -2:
            status[1] = layer
        return codes, scales
    if not live:
        return codes, scales
    guarded = _guard(value[:live], status, error, layer)
    variant = _rotor_variant(codec)
    payload, metadata = quantize_ref(guarded, 4, variant)
    # Storage validity alone does not prove the FP16 working operand finite.
    working = dequant_ref(payload, metadata, 4, variant).half()
    bad = ~torch.isfinite(working)
    if bool(bad.any()):
        if status is None:
            raise ValueError("Rotor Q4 reconstruction exceeds finite FP16 working operands")
        status[0].bitwise_or_(8)
        if status[1] == -2:
            status[1] = layer
        payload.zero_()
        metadata.zero_()
    codes[base:base + live].copy_(payload)
    scales[base:base + live].copy_(metadata)
    return codes, scales


def _copy_rotor_frame_rows(target, values, positions, valid_rows, status, layer):
    """A Torch-prepared Q4 frame still preserves inactive raw/index cache rows.

    The native row copy avoids a CUDA nonzero/readback for dynamic live counts.
    Invalid positions poison the existing status before an invalid access.
    """
    rows = values.shape[0]
    if not rows:
        return
    if values.is_cuda:
        from .rotorquant_q4_native import copy_rows

        copy_rows(values, target, positions[:1], status=status, valid_rows=valid_rows, layer=layer)
        return
    live = rows if valid_rows is None else int(valid_rows.reshape(-1)[0])
    if not 0 <= live <= rows:
        if status is None:
            raise ValueError("Rotor Q4 live row count must fit its frame")
        status[0].bitwise_or_(16)
        if status[1] == -2:
            status[1] = layer
        return
    for row in range(live):
        position = int(positions[row])
        if not 0 <= position < target.shape[0]:
            if status is None:
                raise ValueError("Rotor Q4 row must fit its owned cache")
            status[0].bitwise_or_(16)
            if status[1] == -2:
                status[1] = layer
        else:
            target[position].copy_(values[row])


def _rotor_decode(codes, scales, width, codec):
    """Reconstruct only an FP16 working view, leaving packed ownership intact."""
    out = torch.empty((*codes.shape[:-1], width), dtype=torch.float16, device=codes.device)
    if codes.is_cuda:
        from .rotorquant_q4_native import decode

        return decode(codes, scales, out, codec=codec)
    from tensorfold.families.qwen4_exp.cuda.rotorquant_ref import dequant_ref

    out.copy_(dequant_ref(codes, scales, 4, _rotor_variant(codec)))
    return out


def _live_mask(value, valid_rows):
    if valid_rows is None:
        return None
    return (torch.arange(value.shape[0], device=value.device) < valid_rows.reshape(-1)[0]).reshape(
        value.shape[0], *([1] * (value.ndim - 1)))


def _guard(value, status, error, layer, valid_rows=None):
    if status is None:
        return value
    valid = torch.isfinite(value) & (value.abs() <= 2**30)
    live = _live_mask(value, valid_rows)
    bad = ((~valid) if live is None else (~valid & live)).any()
    status[0].bitwise_or_(bad.int() * error)
    status[1].copy_(torch.where(bad & (status[1] == -2), layer, status[1]))
    return torch.where(valid, value, 0)


def quantize_int8(value, *, status=None, layer=0, valid_rows=None):
    rotated = h32(value).reshape(*value.shape[:-1], value.shape[-1] // 32, 32)
    scale = rotated.abs().amax(-1) + 1e-10
    stored = scale.half()
    inverse = 1.0 / scale
    codes = (torch.floor(rotated * inverse[..., None] * 128.0) + 128).clamp(0, 255)
    if status is not None:
        invalid = ~torch.isfinite(stored) | (stored < 0)
        live = _live_mask(stored, valid_rows)
        bad = (invalid if live is None else invalid & live).any()
        status[0].bitwise_or_(bad.int() * 8)
        status[1].copy_(torch.where(bad & (status[1] == -2), layer, status[1]))
        stored = torch.where(torch.isfinite(stored) & (stored >= 0), stored, 0)
    return (codes - 128).to(torch.int8).reshape(value.shape), stored


def dequantize_int8(value, scale):
    grouped = value.float().reshape(*value.shape[:-1], value.shape[-1] // 32, 32)
    return ((grouped + 0.5) * scale.float()[..., None] * (1.0 / 128)).reshape(value.shape).half()


def rms(value, weight, eps, *, group=None):
    width = value.shape[-1]
    group = width if group is None else group
    if group <= 0 or width % group or not math.isfinite(eps) or eps <= 0:
        raise ValueError("RMSNorm needs positive complete groups and finite positive epsilon")
    x = value.float().reshape(*value.shape[:-1], width // group, group)
    y = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)
    return (y.reshape(value.shape) * weight.float()).half()


def _torch_rmsnorm(x, w, eps, group=None, out=None, xs=None):
    value = rms(x, w, eps, group=group)
    out = torch.empty_like(value) if out is None else out
    out.copy_(value)
    if xs is None:
        xs = torch.empty((*x.shape[:-1], x.shape[-1] // 32), device=x.device, dtype=torch.float32)
    group_sums(out, xs)
    return out, xs


def embed(ids, weight, scales, biases, dims, copies=1, out=None):
    selected = weight.index_select(0, ids.long()).to(torch.int64)
    shifts = torch.arange(8, device=ids.device) * 4
    codes = ((selected[..., None] >> shifts) & 15).reshape(ids.numel(), dims).float()
    s = scales.index_select(0, ids.long()).float().repeat_interleave(32, -1)
    b = biases.index_select(0, ids.long()).float().repeat_interleave(32, -1)
    value = (codes * s + b).half().repeat(1, copies)
    if out is None:
        return value
    out.copy_(value)
    return out


def _torch_hc_writeback(h, hout, pss, streams, mode, branch=None, inject=None, y=None, wts=None, block=256):
    rows, wide = h.shape
    dims = wide // streams
    if mode == 0:
        value = h
    else:
        if inject is None:
            raise ValueError("a hyper-connection branch requires injection weights")
        if mode == 1:
            update = branch.half()
        elif mode == 2:
            total = torch.zeros((rows, dims), dtype=torch.float32, device=h.device)
            for slot in range(wts.shape[1]):
                total = total + y[:, slot].float() * wts[:, slot, None].float()
            update = total.half()
        elif mode in (3, 4):
            total = branch[0].float()
            for rank in range(1, branch.shape[0]):
                total = total + branch[rank].float()
            update = total.half()
        else:
            raise ValueError("unknown hyper-connection writeback mode")
        add = (update[:, None].float() * inject[:, :, None].float()).half()
        value = (h.reshape(rows, streams, dims).float() + add.float()).half().reshape(rows, wide)
    hout.copy_(value)
    squared = hout.reshape(rows, streams, dims).float().square().reshape(rows, streams, -1, block).sum(-1)
    pss.copy_(squared.transpose(1, 2))


def _torch_hc_normed(h, pss, scale, normed, xs, streams, eps, block=512):
    rows, wide = h.shape
    dims = wide // streams
    total = pss[:, 0].float()
    for chunk in range(1, pss.shape[1]):
        total = total + pss[:, chunk].float()
    value = h.reshape(rows, streams, dims).float() * torch.rsqrt(total[:, :, None] / dims + eps)
    normed.copy_((value.reshape(rows, wide) * scale.float()).half())
    group_sums(normed, xs)


def _torch_hc_act(dn, act, xs, inj, streams, low):
    rounded = (dn.float() / streams).half()
    act.copy_(_silu(rounded[:, :low]))
    group_sums(act, xs)
    if inj is not None:
        inj.copy_((2.0 * _sigmoid(rounded[:, low:low + streams]).float()).half())


def _torch_hc_reduce_act(part, act, xs, inj, streams, low):
    total = part[0].float()
    for split in range(1, part.shape[0]):
        total = total + part[split].float()
    hc_act(total.half(), act, xs, inj, streams, low)


def _torch_hc_mix(up, normed, mixed, xs, streams, block=256):
    rows, wide = up.shape
    dims = wide // streams
    gates = _sigmoid(up).reshape(rows, streams, dims)
    values = (gates.float() * normed.reshape(rows, streams, dims).float()).half()
    total = values[:, 0].float()
    for stream in range(1, streams):
        total = total + values[:, stream].float()
    mixed.copy_((total / streams).half())
    group_sums(mixed, xs)


def _torch_moe_partial(y, wts, out, rows):
    total = torch.zeros_like(out[:rows], dtype=torch.float32)
    for slot in range(wts.shape[1]):
        total = total + y[:rows, slot].float() * wts[:rows, slot, None].float()
    out[:rows].copy_(total)


def rotary(value, inv_freq, positions, *, rope=None, delta=None, length=0, sections=(11, 11, 10), scale=1.0):
    """The family's split-half rotary prefix, including interleaved M-RoPE axes."""
    half = inv_freq.numel()
    if 2 * half > value.shape[-1]:
        raise ValueError("rotary prefix exceeds the head width")
    indices = torch.arange(half, device=value.device)
    positions = positions.long().reshape(-1)
    axes = positions[:, None].expand(-1, half)
    if delta is not None:
        axes = axes + delta.reshape(-1)[0]
    if rope is not None:
        safe = positions.clamp(0, max(0, rope.shape[0] - 1))
        per_row = rope.index_select(0, safe)
        axis_id = torch.where((indices % 3 == 1) & (indices < 3 * sections[1]), 1,
                              torch.where((indices % 3 == 2) & (indices < 3 * sections[2]), 2, 0))
        image_axes = per_row[:, axis_id]
        axes = torch.where((positions < length)[:, None], image_axes, axes)
    angles = axes.float() * inv_freq.float()[None]
    while angles.ndim < value.ndim:
        angles = angles.unsqueeze(1)
    sine, cosine = torch.sin(angles) * scale, torch.cos(angles) * scale
    a, b = value[..., :half].float(), value[..., half:2 * half].float()
    result = value.clone()
    result[..., :half] = (a * cosine - b * sine).half()
    result[..., half:2 * half] = (b * cosine + a * sine).half()
    return result


def _history_stage(history, kind, capacity, rope):
    """Select a captured stage using caller-owned phase and logical capacity."""
    if rope is not None:
        raise ValueError("RAM indexer history supports the SM70 text rotary path only")
    if kind not in ("prefill", "decode"):
        raise ValueError("RAM indexer history requires an explicit prefill/decode phase")
    if type(capacity) is not int or not 0 < capacity <= history.owner.limit:
        raise ValueError("RAM indexer history requires the current positive cache capacity")
    stage = history.stage(kind)
    from .family_native import extension

    return extension(), stage


def attn_prep(p, pos0, q_scale, k_scale, i_scale, inv_freq, q, kc, vc, iq, ikc, eps, *,
              q_heads, kv_heads, head_dim, index_heads, index_dim, ks=None, vs=None, bits=0,
              rope=None, delta=None, length=0, sections=(11, 11, 10), rope_scale=1.0, codec=0,
              status=None, layer=0, k_bits=None, k_codec=None, v_bits=None, v_codec=None,
              centroids=None, thresholds=None, valid_rows=None, independent_rows=False,
              history_kind=None, history_capacity=None):
    kb, vb = _kv_plan(bits, codec, k_bits, k_codec, v_bits, v_codec)
    kf = codec if k_codec is None else k_codec
    vf = codec if v_codec is None else v_codec
    if isinstance(ikc, HistoryLayer):
        if history_capacity != kc.shape[0]:
            raise ValueError("RAM indexer preparation requires the current K/V cache capacity")
        native, stage = _history_stage(ikc, history_kind, history_capacity, rope)
        return native.prep_staged(p, pos0, q_scale, k_scale, i_scale, inv_freq, q, kc, vc, iq, stage,
                                  ks if kb else None, vs if vb else None, status, delta, valid_rows,
                                  q_heads, kv_heads, head_dim, index_heads, index_dim, kb, vb, layer,
                                  eps, rope_scale, independent_rows, kf, vf, ikc.origin)
    native = _fused(p) if rope is None and head_dim <= 256 and index_dim <= 256 else None
    if native is not None:
        if kf or vf:
            return native.prep_q4(p, pos0, q_scale, k_scale, i_scale, inv_freq, q, kc, vc, iq, ikc,
                                  ks if kb else None, vs if vb else None, status, delta, valid_rows,
                                  q_heads, kv_heads, head_dim, index_heads, index_dim, kb, vb, layer,
                                  eps, rope_scale, independent_rows, kf, vf)
        return native.prep(p, pos0, q_scale, k_scale, i_scale, inv_freq, q, kc, vc, iq, ikc,
                           ks if kb else None, vs if vb else None, status, delta, valid_rows,
                           q_heads, kv_heads, head_dim, index_heads, index_dim, kb, vb, layer, eps, rope_scale,
                           independent_rows)
    if kc.dtype != _kv_payload_dtype(kb) or vc.dtype != _kv_payload_dtype(vb):
        raise ValueError("SM70 execution requires its declared FP16/int8/Rotor Q4 KV payload identity")
    if (kb and ks is None) or (vb and vs is None):
        raise ValueError("quantized cache sides require their own declared scales")
    if kf and ks.dtype != torch.float32 or vf and vs.dtype != torch.float32:
        raise ValueError("Rotor Q4 cache metadata must use FP32 scales")
    rows = p.shape[0]

    def normalize(value, weight):
        if independent_rows and rows:
            return torch.cat([rms(value[row:row + 1], weight, eps) for row in range(rows)], dim=0)
        return rms(value, weight, eps)

    at = q_heads * 2 * head_dim
    queries = _guard(p[:, :at].reshape(rows, q_heads, 2, head_dim)[:, :, 0], status, 1, layer, valid_rows)
    keys = _guard(p[:, at:at + kv_heads * head_dim].reshape(rows, kv_heads, head_dim), status, 2, layer, valid_rows)
    at += kv_heads * head_dim
    values = _guard(p[:, at:at + kv_heads * head_dim].reshape(rows, kv_heads, head_dim), status, 4, layer, valid_rows)
    at += kv_heads * head_dim
    index_queries = _guard(p[:, at:at + index_heads * index_dim].reshape(rows, index_heads, index_dim), status, 16, layer, valid_rows)
    at += index_heads * index_dim
    index_keys = _guard(p[:, at:at + index_dim], status, 16, layer, valid_rows)
    positions = pos0.reshape(-1)[0].long() + torch.arange(rows, device=p.device)

    def store(target, value):
        if kf or vf:
            _copy_rotor_frame_rows(target, value, positions, valid_rows, status, layer)
        else:
            target.index_copy_(0, positions, value)

    options = dict(rope=rope, delta=delta, length=length, sections=sections, scale=rope_scale)
    prepared_q = _guard(rotary(normalize(queries, q_scale), inv_freq, positions, **options), status, 1, layer, valid_rows)
    if kf:
        _rotor_rotate(prepared_q, q[:rows], codec=kf, status=status, error=1, layer=layer, valid_rows=valid_rows)
    else:
        q[:rows].copy_(h32(prepared_q).half() if kb else prepared_q)
    normalized = _guard(rotary(normalize(keys, k_scale), inv_freq, positions, **options), status, 2, layer, valid_rows)
    if kf:
        _rotor_encode(normalized, kc, ks, pos0, codec=kf, status=status, error=2,
                      layer=layer, valid_rows=valid_rows)
    elif kb:
        encoded, scales = quantize_int8(normalized, status=status, layer=layer, valid_rows=valid_rows)
        store(kc, encoded)
        store(ks, scales)
    else:
        store(kc, normalized)
    if vf:
        _rotor_encode(values, vc, vs, pos0, codec=vf, status=status, error=4,
                      layer=layer, valid_rows=valid_rows)
    elif vb:
        encoded, scales = quantize_int8(values, status=status, layer=layer, valid_rows=valid_rows)
        store(vc, encoded)
        store(vs, scales)
    else:
        store(vc, values.half())
    iq[:rows].copy_(_guard(rotary(normalize(index_queries, i_scale), inv_freq, positions, **options), status, 16, layer, valid_rows))
    store(ikc, index_keys.half())


def _torch_attn_gate(o, p, out, xs, *, q_heads, head_dim):
    rows = o.shape[0]
    gates = p[:, :q_heads * 2 * head_dim].reshape(rows, q_heads, 2, head_dim)[:, :, 1]
    out.copy_((o.float() * torch.sigmoid(gates.float())).half().reshape(rows, -1))
    group_sums(out, xs)


def ple_embed(rows, weight, scales, biases, heads, dh, out, xs, *, scale=1.0):
    shifts = torch.arange(8, device=out.device) * 4
    codes = ((weight[:rows * heads].long()[..., None] >> shifts) & 15).reshape(rows * heads, dh).float()
    value = (codes * scales[:rows * heads].float().repeat_interleave(32, -1)
             + biases[:rows * heads].float().repeat_interleave(32, -1)).half()
    ple_embed_bf16(rows, value, heads, dh, out, xs, scale=scale)


def _torch_ple_embed_bf16(rows, values, heads, dh, out, xs, *, scale=1.0):
    value = values[:rows * heads].reshape(rows, heads * dh)
    out.copy_((value.float() * scale).half())
    group_sums(out, xs)


def _torch_ple_gate(keys, vals, h, norm_key, norm_query, gated, pss, eps, streams):
    rows, wide = h.shape
    dims = wide // streams
    k = rms(keys.reshape(rows, streams, dims), norm_key.reshape(streams, dims), eps)
    query = rms(h.reshape(rows, streams, dims), norm_query.reshape(streams, dims), eps)
    dot = (k.float() * query.float()).half().float().sum(-1)
    score = (dot / math.sqrt(dims)).half().float()
    signed = (torch.sign(score) * torch.sqrt(score.abs().clamp_min(1e-6))).half()
    gate = _sigmoid(signed)
    value = (gate[:, :, None].float() * vals[:, None].float()).half()
    gated.copy_(value.reshape(rows, wide))
    pss.copy_(value.float().square().sum(-1))


def _torch_ple_conv(gated, pss, norm_conv, tail, conv_w, h, hout, nrow, eps, streams, dilation):
    rows, wide = h.shape
    dims, taps = wide // streams, conv_w.shape[1]
    norms = torch.rsqrt(pss.float() / dims + eps).repeat_interleave(dims, -1)
    own = (gated.float() * norms * norm_conv.float()).half()
    nrow.copy_(own)
    history = torch.cat((tail, own), dim=0)
    acc = torch.zeros((rows, wide), dtype=torch.float32, device=h.device)
    for tap in range(taps):
        acc = acc + history[tap * dilation:tap * dilation + rows].float() * conv_w[:, tap].float()
    act = _silu(acc.half())
    branch = (gated.float() + act.float()).half()
    hout.copy_((h.float() + branch.float()).half())


def _torch_add_streams(e, hs, out, streams):
    out.copy_((e[:, None].float() + hs.reshape(e.shape[0], streams, e.shape[1]).float()).half().reshape(out.shape))


def qsa_pool(ikc, pooled, pos0, ik_scale, inv_freq, eps, scratch, rows, *, rope=None, delta=None,
             length=0, sections=(11, 11, 10), rope_scale=1.0,
             history_kind=None, history_capacity=None, status=None, valid_rows=None, layer=0):
    if isinstance(ikc, HistoryLayer):
        native, stage = _history_stage(ikc, history_kind, history_capacity, rope)
        ring, origin = ikc.ring, ikc.origin
        # Pooling must consume old carry before publishing the new ring tail.
        # The full current capacity preserves the original RMS reduction mode.
        native.pool_staged(stage, ring, pooled, pos0, ik_scale, inv_freq, delta, scratch.ratio,
                           rows, history_capacity, eps, rope_scale, status, valid_rows, layer, origin)
        return native.ring_publish(stage, ring, pos0, rows, history_capacity, status, valid_rows, layer, origin)
    ratio, dims = scratch.ratio, ikc.shape[1]
    native = _fused(ikc) if rope is None and dims <= 256 else None
    if native is not None:
        return native.pool(ikc, pooled, pos0, ik_scale, inv_freq, delta, ratio, rows, eps, rope_scale)
    count = ikc.shape[0] // ratio
    if not count:
        return
    blocks = torch.arange(count, device=ikc.device)
    raw = ikc[:count * ratio].reshape(count, ratio, dims).float()
    total = raw[:, 0]
    for index in range(1, ratio):
        total = total + raw[:, index]
    value = (total / ratio).half()
    value = rotary(rms(value, ik_scale, eps), inv_freq, blocks * ratio, rope=rope, delta=delta,
                   length=length, sections=sections, scale=rope_scale)
    start = pos0.reshape(-1)[0]
    valid = (blocks >= torch.div(start, ratio, rounding_mode="floor")) & ((blocks + 1) * ratio <= start + rows)
    pooled[:count].copy_(torch.where(valid[:, None], value, pooled[:count]))


def qsa_rows(iq, pooled, pos0, scratch, rows, *, context=None):
    ratio, top = scratch.ratio, scratch.budget // scratch.ratio
    count = pooled.shape[0] if context is None else min(pooled.shape[0], (context + ratio - 1) // ratio)
    native = _fused(iq) if count <= 4096 else None
    if native is not None:
        if count <= top:
            return native.qsa_dense(pos0, scratch.sparse, scratch.nk, rows)
        return native.qsa_rows(iq, pooled, pos0, scratch.sparse, scratch.nk, scratch.ids, scratch.scores,
                               rows, ratio, scratch.budget, count)
    ends = pos0.reshape(-1)[0] + torch.arange(rows, device=iq.device) + 1
    complete = torch.div(ends, ratio, rounding_mode="floor")
    sparse = complete > top
    scratch.sparse[:rows].copy_(sparse.int())
    scratch.nk[:rows].copy_(torch.where(sparse, top * ratio + ends % ratio, ends).int())
    if count <= top:
        return
    scores = torch.zeros((rows, count), dtype=torch.float32, device=iq.device)
    for head in range(iq.shape[1]):
        scores = scores + _matmul32(iq[:rows, head], pooled[:count].T).clamp_min(0)
    scores = scores / math.sqrt(pooled.shape[-1])
    blocks = torch.arange(count, device=iq.device)
    scores.masked_fill_(blocks[None] >= complete[:, None], -torch.inf)
    scratch.scores[:rows, :count].copy_(scores)
    chosen = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :top]
    chosen = torch.sort(chosen, dim=-1).values
    offsets = torch.arange(ratio, device=iq.device)
    selected = (chosen[:, :, None] * ratio + offsets).reshape(rows, top * ratio)
    scratch.ids[:rows, :top * ratio].copy_(selected.int())
    tail = complete[:, None] * ratio + offsets[None]
    scratch.ids[:rows, top * ratio:top * ratio + ratio].copy_(tail.int())


def attention(q, kc, vc, pos0, scratch, rows, scale, out=None, *, context=None, ks=None, vs=None,
              bits=0, codec=0, k_bits=None, k_codec=None, v_bits=None, v_codec=None,
              rotor_lookup=False, prompt=False, _native_allowed=True):
    kb, vb = _kv_plan(bits, codec, k_bits, k_codec, v_bits, v_codec)
    kf = codec if k_codec is None else k_codec
    vf = codec if v_codec is None else v_codec
    if kc.dtype != _kv_payload_dtype(kb) or vc.dtype != _kv_payload_dtype(vb):
        raise ValueError("SM70 attention requires matching FP16/int8/Rotor Q4 K/V payloads")
    keys = kc.shape[0] if context is None else min(context, kc.shape[0])
    out = scratch.out if out is None else out
    if (_native_allowed and _NATIVE_FAMILY and _NATIVE_ATTENTION and q.is_cuda and q.dtype == torch.float16):
        from .attention_qualification import runtime_identity, supports

        if supports(rows=rows, heads=q.shape[1], kv_heads=kc.shape[1], head_dim=q.shape[2],
                    context=keys, first_end=keys-rows+1, identity=runtime_identity(q.device),
                    capability=torch.cuda.get_device_capability(q.device), prompt=prompt):
            from .attention_native import attention as fixed_attention

            if kf or vf:
                return fixed_attention(q, kc, vc, pos0, scratch, rows, scale, out, keys, ks, vs, kb, vb,
                                       kcodec=kf, vcodec=vf)
            return fixed_attention(q, kc, vc, pos0, scratch, rows, scale, out, keys, ks, vs, kb, vb)
    heads, kh = q.shape[1], kc.shape[1]
    if heads % kh:
        raise ValueError("query heads must be divisible by local KV heads")
    group = heads // kh
    positions = torch.arange(keys, device=q.device)
    key_data = (_rotor_decode(kc[:keys], ks[:keys], q.shape[2], kf) if kf else
                dequantize_int8(kc[:keys], ks[:keys]) if kb else kc[:keys])
    value_data = (_rotor_decode(vc[:keys], vs[:keys], q.shape[2], vf) if vf else
                  dequantize_int8(vc[:keys], vs[:keys]) if vb else vc[:keys])
    for row in range(rows):
        end = pos0.reshape(-1)[0] + row + 1
        visible = positions < end
        if scratch.qsa:
            ids = scratch.ids[row].long()
            valid = (torch.arange(ids.numel(), device=q.device) < scratch.nk[row]) & (ids < end)
            selected = torch.zeros(keys, dtype=torch.int32, device=q.device)
            selected.scatter_add_(0, ids.clamp(0, max(0, keys - 1)), valid.int())
            visible = torch.where(scratch.sparse[row].bool(), visible & (selected != 0), visible)
        # A graph's context bucket includes uncommitted or discarded cache
        # rows. Zero probabilities alone cannot exclude NaN/Inf values from
        # the value product (0 * NaN is NaN), so mask those operands as well.
        visible_values = torch.where(visible[:, None, None], value_data, 0)
        for head in range(kh):
            query = q[row, head * group:(head + 1) * group]
            scores = _matmul32(query, key_data[:, head].T) * scale
            scores.masked_fill_(~visible[None], -torch.inf)
            probabilities = scores.softmax(-1).half()
            result = _matmul32(probabilities, visible_values[:, head])
            if vf:
                rotated = torch.empty_like(result)
                _rotor_rotate(result[None], rotated[None], codec=vf, inverse=True)
                result = rotated
            elif vb:
                result = h32(result)
            out[row, head * group:(head + 1) * group].copy_(result.half())
    return out


def gdn_chain(p, conv_state, conv_w, state_in, a_log, dt_bias, norm_w, eps, rows, scratch,
              state_out, out, xs):
    nv, dv, dk = state_in.shape
    nk = scratch.k.shape[1]
    conv = 2 * nk * dk + nv * dv
    taps = conv_w.shape[1]
    if p.shape[1] != conv + nv * dv + 2 * nv or nv % nk:
        raise ValueError("GDN projection does not match the local head geometry")
    history = torch.cat((conv_state, p[:rows, :conv]), dim=0)
    current = state_in.clone()
    live = getattr(scratch, "keep_rows", None)
    if live is not None:
        state_out.copy_(state_in)
    mapped = torch.arange(nv, device=p.device) // (nv // nk)
    for row in range(rows):
        acc = torch.zeros(conv, dtype=torch.float32, device=p.device)
        for tap in range(taps):
            acc = acc + history[row + tap].float() * conv_w[:, tap].float()
        values = _silu(acc).float()
        query = values[:nk * dk].reshape(nk, dk)
        key = values[nk * dk:2 * nk * dk].reshape(nk, dk)
        value = values[2 * nk * dk:].reshape(nv, dv).half()
        query = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6) / math.sqrt(dk)
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        beta = torch.sigmoid(p[row, conv + nv * dv:conv + nv * dv + nv].float()).half().float()
        decay = torch.exp(-torch.exp(a_log.float()) * F.softplus(
            p[row, conv + nv * dv + nv:].float() + dt_bias.float()))
        scratch.k[row].copy_(key)
        scratch.v[row].copy_(value)
        scratch.g[row].copy_(decay)
        scratch.b[row].copy_(beta)
        current = current * decay[:, None, None]
        prediction = (current * key[mapped, None, :]).sum(-1)
        difference = (value.float() - prediction) * beta[:, None]
        current = current + difference[:, :, None] * key[mapped, None, :]
        raw = (current * query[mapped, None, :]).sum(-1).half()
        normalized = (raw.float() * torch.rsqrt(raw.float().square().mean(-1, keepdim=True) + eps)).half()
        normalized = (normalized.float() * norm_w.float()).half()
        gate = p[row, conv:conv + nv * dv].reshape(nv, dv).float()
        out[row].copy_((normalized.float() * torch.sigmoid(gate)).half().reshape(-1))
        group_sums(out[row:row + 1], xs[row:row + 1])
        if live is not None:
            state_out.copy_(torch.where(live.reshape(-1)[0] == row + 1, current, state_out))
    if live is None:
        state_out.copy_(current)


def gdn_replay(state_in, scratch, rows, state_out):
    nv = state_in.shape[0]
    nk = scratch.k.shape[1]
    mapped = torch.arange(nv, device=state_in.device) // (nv // nk)
    current = state_in.clone()
    for row in range(rows):
        key = scratch.k[row, mapped].float()
        current = current * scratch.g[row, :, None, None]
        prediction = (current * key[:, None]).sum(-1)
        delta = (scratch.v[row].float() - prediction) * scratch.b[row, :, None]
        current = current + delta[:, :, None] * key[:, None]
    state_out.copy_(current)


def _torch_shift_windows(old, new, keep, channels):
    taps = old.shape[1]
    if not taps:
        return
    old.copy_(torch.cat((old, new[..., :channels]), dim=1)[:, keep:keep + taps])


# Keep the hardware-generic implementations above as numerical references.
# CPU tensors always use them. The private selector is for numerical oracle
# tests; promotion follows the native component and full-forward receipts.
_NATIVE_FAMILY = False
_NATIVE_ATTENTION = False
_reference = {name: globals()["_torch_" + name] for name in (
    "hc_writeback", "hc_normed", "hc_act", "hc_reduce_act", "hc_mix", "rmsnorm", "attn_gate",
    "ple_embed_bf16", "ple_gate", "ple_conv", "add_streams", "moe_partial", "shift_windows")}


def _fused(tensor, *, half=True):
    if _NATIVE_FAMILY and tensor.is_cuda and (not half or tensor.dtype == torch.float16):
        from .family_native import extension

        return extension()
    return None


def hc_writeback(h, hout, pss, streams, mode, branch=None, inject=None, y=None, wts=None, block=256):
    native = _fused(h) if block == 256 else None
    if native is not None:
        return native.writeback(h, hout, pss, streams, mode, branch, inject, y, wts)
    return _reference["hc_writeback"](h, hout, pss, streams, mode, branch, inject, y, wts, block)


def hc_normed(h, pss, scale, normed, xs, streams, eps, block=512):
    native = _fused(h)
    if native is not None:
        return native.normed(h, pss, scale, normed, xs, streams, eps)
    return _reference["hc_normed"](h, pss, scale, normed, xs, streams, eps, block)


def hc_act(dn, act, xs, inj, streams, low):
    native = _fused(act)
    if native is not None:
        return native.act(dn, act, xs, inj, streams, low)
    return _reference["hc_act"](dn, act, xs, inj, streams, low)


def hc_reduce_act(part, act, xs, inj, streams, low):
    native = _fused(act)
    if native is not None:
        return native.act(part, act, xs, inj, streams, low)
    return _reference["hc_reduce_act"](part, act, xs, inj, streams, low)


def hc_mix(up, normed, mixed, xs, streams, block=256):
    native = _fused(mixed)
    if native is not None:
        return native.mix(up, normed, mixed, xs, streams)
    return _reference["hc_mix"](up, normed, mixed, xs, streams, block)


def rmsnorm(x, w, eps, group=None, out=None, xs=None):
    native = _fused(x)
    if native is not None:
        out = torch.empty_like(x) if out is None else out
        if xs is None:
            xs = torch.empty((*x.shape[:-1], x.shape[-1] // 32), device=x.device, dtype=torch.float32)
        native.rms(x, w, out, xs, x.shape[-1] if group is None else group, eps)
        return out, xs
    return _reference["rmsnorm"](x, w, eps, group, out, xs)


def attn_gate(o, p, out, xs, *, q_heads, head_dim):
    native = _fused(out)
    if native is not None:
        return native.gate(o, p, out, xs, q_heads, head_dim)
    return _reference["attn_gate"](o, p, out, xs, q_heads=q_heads, head_dim=head_dim)


def ple_embed_bf16(rows, values, heads, dh, out, xs, *, scale=1.0):
    native = _fused(out)
    if native is not None:
        return native.ple_embed(values, out, xs, rows, heads, dh, scale)
    return _reference["ple_embed_bf16"](rows, values, heads, dh, out, xs, scale=scale)


def ple_gate(keys, vals, h, norm_key, norm_query, gated, pss, eps, streams, valid_rows=None, independent_rows=False):
    native = _fused(h)
    if native is not None:
        return native.ple_gate(keys, vals, h, norm_key, norm_query, gated, pss, streams, eps, valid_rows, independent_rows)
    if independent_rows:
        for row in range(h.shape[0]):
            section = slice(row, row + 1)
            _reference["ple_gate"](keys[section], vals[section], h[section], norm_key, norm_query,
                                   gated[section], pss[section], eps, streams)
        return None
    return _reference["ple_gate"](keys, vals, h, norm_key, norm_query, gated, pss, eps, streams)


def ple_conv(gated, pss, norm_conv, tail, conv_w, h, hout, nrow, eps, streams, dilation):
    native = _fused(h)
    if native is not None:
        return native.ple_conv(gated, pss, norm_conv, tail, conv_w, h, hout, nrow, streams, dilation, eps)
    return _reference["ple_conv"](gated, pss, norm_conv, tail, conv_w, h, hout, nrow, eps, streams, dilation)


def add_streams(e, hs, out, streams):
    native = _fused(out)
    if native is not None:
        return native.add(e, hs, out, streams)
    return _reference["add_streams"](e, hs, out, streams)


def moe_partial(y, wts, out, rows):
    native = _fused(out, half=False)
    if native is not None:
        return native.partial(y, wts, out, rows)
    return _reference["moe_partial"](y, wts, out, rows)


def shift_windows(old, new, keep, channels):
    native = _fused(old) if old.shape[1] <= 16 else None
    if native is not None:
        return native.shift(old, new, keep, channels)
    return _reference["shift_windows"](old, new, keep, channels)

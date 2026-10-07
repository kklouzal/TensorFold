"""Bit-exact pageable host storage in the existing grouped CUDA expert format.

Only expert residency changes. Packing rearranges integer bits and never
dequantizes weights, converts BF16 scales, prunes experts, or pins the cold store.
The cache owns its separately bounded pinned transfer staging.
"""

from __future__ import annotations

import numpy as np
import torch

from tensorfold.cuda import experts as grouped


def pack_cpu(
    words: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, gs: int, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Native ``experts_pack.cu`` bytes as contiguous CPU int32 blocks.

    Inputs are contiguous CPU int32/uint32 words [E,N,K/8] and BF16 scale/bias
    arrays [E,N,K/gs]. The optional output is caller-owned contiguous CPU int32
    [E,N/32,K/gs,128*(gs/32)+32], disjoint from all inputs. Input bits, including
    signed words and every BF16 bit pattern, are retained exactly. Temporaries
    are bounded by the input chunk; callers chunk stacked experts before reading.
    """

    if type(gs) is not int or gs not in (32, 64):
        raise ValueError("experts: group size must be 32 or 64")
    tensors = (words, scales, biases)
    if any(not isinstance(t, torch.Tensor) or t.device.type != "cpu" or not t.is_contiguous() for t in tensors):
        raise ValueError("experts: packing inputs must be contiguous CPU tensors")
    if words.dtype not in (torch.int32, torch.uint32) or words.ndim != 3:
        raise ValueError("experts: words must be int32/uint32 [E,N,K/8]")
    e, n, k8 = words.shape
    k = k8 * 8
    if e <= 0 or n <= 0 or k <= 0 or n % 32 or k % gs:
        raise ValueError("experts: positive E, N divisible by 32, and K divisible by the group size required")
    if any(t.dtype != torch.bfloat16 or tuple(t.shape) != (e, n, k // gs) for t in (scales, biases)):
        raise ValueError("experts: scales and biases must be BF16 [E,N,K/gs]")
    h, nb, kg = gs // 32, n // 32, k // gs
    weight_words = 128 * h
    shape = (e, nb, kg, weight_words + 32)
    if out is None:
        out = torch.empty(shape, dtype=torch.int32, device="cpu")
    if (
        not isinstance(out, torch.Tensor)
        or out.device.type != "cpu"
        or out.dtype != torch.int32
        or tuple(out.shape) != shape
        or not out.is_contiguous()
    ):
        raise ValueError("experts: output must be contiguous CPU int32 native blocks")
    out_begin = out.data_ptr()
    out_end = out_begin + out.numel() * out.element_size()
    if any(out_begin < t.data_ptr() + t.numel() * t.element_size() and t.data_ptr() < out_end for t in tensors):
        raise ValueError("experts: output must not overlap an input")

    bits = words.detach().numpy().view(np.uint32)
    even = bits & np.uint32(0x0F0F0F0F)
    odd = (bits >> np.uint32(4)) & np.uint32(0x0F0F0F0F)
    even = (even | (even >> np.uint32(4))) & np.uint32(0x00FF00FF)
    odd = (odd | (odd >> np.uint32(4))) & np.uint32(0x00FF00FF)
    even = (even | (even >> np.uint32(8))) & np.uint32(0x0000FFFF)
    odd = (odd | (odd >> np.uint32(8))) & np.uint32(0x0000FFFF)
    shuffled = even | (odd << np.uint32(16))

    # Derive the lane destination from the kernel's row/word coordinates; no
    # floating-point operations or lossy signed casts enter the permutation.
    row = np.arange(32, dtype=np.int64)[:, None]
    kk = np.arange(4 * h, dtype=np.int64)[None, :]
    f = (((row // 8) * h + kk % h) * 8 + row % 8) * 4 + kk // h
    destination = ((f // 128) * 128 + (f % 32) * 4 + (f // 32) % 4).reshape(-1)
    output = out.numpy().view(np.uint32)
    values = shuffled.reshape(e, nb, 32, kg, 4 * h).transpose(0, 1, 3, 2, 4)
    output[..., destination] = values.reshape(e, nb, kg, weight_words)

    tail = output[..., weight_words:].reshape(e, nb, kg, 4, 2, 4)
    for kind, tensor in enumerate((scales, biases)):
        raw = tensor.detach().view(torch.int16).numpy().view(np.uint16)
        pairs = raw.reshape(e, nb, 4, 4, 2, kg).transpose(0, 1, 5, 3, 2, 4)
        tail[..., kind, :] = pairs[..., 0].astype(np.uint32) | (pairs[..., 1].astype(np.uint32) << np.uint32(16))
    return out


def load_host_experts(reader, name: str, *, prefix: str = "", chunk_experts: int = 2) -> grouped.Experts:
    """Load affine group-32 MoE directly into pageable CPU grouped.Experts.

    ``name`` is the MoE base (for example ``model.layers.0.mlp``); ``prefix`` is
    the checkpoint's optional language-model prefix. The shared expert retains
    logical ID E after all routed experts. The final packed buffers are allocated
    once. At most ``chunk_experts`` raw experts of one projection and that
    projection's packing temporaries exist at a time; no whole stacked input,
    pinned cold copy, CUDA upload, or read-ahead future is consumed here.
    """

    if type(chunk_experts) is not int or chunk_experts <= 0:
        raise ValueError("chunk_experts must be a positive integer")
    base = prefix + name
    projections = ("gate_proj", "up_proj", "down_proj")
    components = ("weight", "scales", "biases")
    metadata = {}
    for part in ("switch_mlp", "shared_expert"):
        for projection in projections:
            triple = [reader.info(f"{base}.{part}.{projection}.{component}") for component in components]
            rank = 3 if part == "switch_mlp" else 2
            if any(any(type(size) is not int or size <= 0 for size in t["shape"]) for t in triple):
                raise ValueError(f"{base}.{part}.{projection}: dimensions must be positive integers")
            if (
                triple[0]["dtype"] not in ("U32", "I32")
                or len(triple[0]["shape"]) != rank
                or any(t["dtype"] != "BF16" or len(t["shape"]) != rank for t in triple[1:])
            ):
                raise ValueError(f"{base}.{part}.{projection}: expected affine4 words and BF16 scales/biases")
            shape = triple[0]["shape"]
            if (
                any(t["shape"][:-1] != shape[:-1] or t["shape"][-1] * 4 != shape[-1] for t in triple[1:])
                or shape[-2] % 32
            ):
                raise ValueError(f"{base}.{part}.{projection}: affine group-32 shapes do not match")
            metadata[part, projection] = shape
    e, width, d8 = metadata["switch_mlp", "gate_proj"]
    dims = d8 * 8
    expected = {
        "gate_proj": (e, width, dims // 8),
        "up_proj": (e, width, dims // 8),
        "down_proj": (e, dims, width // 8),
    }
    if (
        dims % 32
        or width % 32
        or any(
            metadata["switch_mlp", p] != expected[p] or metadata["shared_expert", p] != expected[p][1:]
            for p in projections
        )
    ):
        raise ValueError(f"{base}: routed/shared projections must have equal compatible widths")
    block = 160
    up = torch.empty((e + 1, width // 32, dims // 32, 2, block), dtype=torch.int32, device="cpu")
    down = torch.empty((e + 1, dims // 32, width // 32, 1, block), dtype=torch.int32, device="cpu")

    for projection_index, projection in enumerate(projections):
        for lo in range(0, e, chunk_experts):
            hi = min(e, lo + chunk_experts)
            source = [
                reader.get_rows(f"{base}.switch_mlp.{projection}.{component}", lo, hi) for component in components
            ]
            packed = pack_cpu(*source, 32)
            if projection_index < 2:
                up[lo:hi, :, :, projection_index, :].copy_(packed)
            else:
                down[lo:hi, :, :, 0, :].copy_(packed)
            del source, packed
        n = expected[projection][1]
        source = [
            reader.get_rows(f"{base}.shared_expert.{projection}.{component}", 0, n).unsqueeze(0)
            for component in components
        ]
        packed = pack_cpu(*source, 32)
        if projection_index < 2:
            up[e : e + 1, :, :, projection_index, :].copy_(packed)
        else:
            down[e : e + 1, :, :, 0, :].copy_(packed)
        del source, packed
    return grouped.Experts(up, down, 32, width, dims)

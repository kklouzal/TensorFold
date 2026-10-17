"""Bit-exact owned host storage in the existing grouped CUDA expert format.

Packing rearranges integer bits without numerical conversion. Optional model-
owned authority allocation and verified disk artifacts preserve these bytes.
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


def load_host_experts(reader, name: str, *, prefix: str = "", chunk_experts: int = 2,
                      owned_ids: range | list[int] | None = None, include_shared: bool = True,
                      authority=None, packed_cache=None) -> grouped.Experts:
    """Load exact affine group-32 bytes into final owned CPU authority.

    Returned compact local order is ``owned_ids`` followed by shared expert E
    when requested; omitted IDs allocate/read no source rows. ``None`` preserves
    all routed IDs and shared E. Metadata is validated before source reads.
    ``authority.allocate_authority`` is the model-owned pin policy; absent it,
    storage is pageable. Packing temporaries remain bounded by chunk_experts.
    A configured PackedExpertCache hashes exact contributing bytes at each
    startup and verifies cached bytes while filling the same final authority.
    """

    if type(chunk_experts) is not int or chunk_experts <= 0:
        raise ValueError("chunk_experts must be a positive integer")
    base = prefix + name
    projections = ("gate_proj", "up_proj", "down_proj")
    components = ("weight", "scales", "biases")
    if type(include_shared) is not bool:
        raise ValueError("include_shared must be a boolean")
    metadata, infos = {}, {}
    parts = ("switch_mlp", "shared_expert") if include_shared else ("switch_mlp",)
    for part in parts:
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
            for component, info in zip(components, triple):
                infos[f"{base}.{part}.{projection}.{component}"] = info
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
            metadata["switch_mlp", p] != expected[p]
            or (include_shared and metadata["shared_expert", p] != expected[p][1:])
            for p in projections
        )
    ):
        raise ValueError(f"{base}: routed/shared projections must have equal compatible widths")
    if owned_ids is not None and not isinstance(owned_ids, (range, list)):
        raise ValueError("owned_ids must be a range or list of unique routed expert IDs")
    ids = tuple(range(e) if owned_ids is None else owned_ids)
    if any(type(i) is not int or not 0 <= i < e for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("owned_ids must contain unique in-range routed expert IDs")
    count = len(ids) + int(include_shared)
    if not count:
        raise ValueError("a host expert bank must own at least one expert")
    # Consecutive source runs preserve caller order, never reading gaps owned by
    # another rank. Cache identity is independent of packing chunk size.
    runs = []
    for i in ids:
        if runs and i == runs[-1][1]:
            runs[-1] = (runs[-1][0], i + 1)
        else:
            runs.append((i, i + 1))
    block = 160
    shapes = ((count, width // 32, dims // 32, 2, block),
              (count, dims // 32, width // 32, 1, block))
    def allocate(shape, tag):
        if authority is None:
            return torch.empty(shape, dtype=torch.int32, device="cpu")
        return authority.allocate_authority(shape, dtype=torch.int32, tag=tag)
    up, down = (allocate(shape, base + suffix) for shape, suffix in zip(shapes, (".up", ".down")))
    for tensor, shape in zip((up, down), shapes):
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.int32
                or tuple(tensor.shape) != shape or not tensor.is_contiguous()):
            raise ValueError("expert authority allocator must return contiguous CPU int32 arrays with the requested shape")
    if up.data_ptr() < down.data_ptr() + down.numel() * 4 and down.data_ptr() < up.data_ptr() + up.numel() * 4:
        raise ValueError("expert authority allocations must be disjoint")
    def pack():
        for projection_index, projection in enumerate(projections):
            local = 0
            for begin, end in runs:
                for lo in range(begin, end, chunk_experts):
                    hi = min(end, lo + chunk_experts)
                    source = [reader.get_rows(f"{base}.switch_mlp.{projection}.{c}", lo, hi) for c in components]
                    packed = pack_cpu(*source, 32)
                    target = up[..., projection_index, :] if projection_index < 2 else down[..., 0, :]
                    target[local:local + hi - lo].copy_(packed)
                    local += hi - lo
                    del source, packed
            if include_shared:
                n = expected[projection][1]
                source = [reader.get_rows(f"{base}.shared_expert.{projection}.{c}", 0, n).unsqueeze(0)
                          for c in components]
                packed = pack_cpu(*source, 32)
                target = up[..., projection_index, :] if projection_index < 2 else down[..., 0, :]
                target[len(ids):].copy_(packed)
                del source, packed
    if packed_cache is None:
        pack()
    else:
        from .packed_experts import SourceIdentity

        ranges = []
        for field, info in infos.items():
            for lo, hi in (runs if ".switch_mlp." in field else [(0, info["shape"][0])]):
                ranges.append((field, info, lo, hi))
        geometry = {"name": base, "ids": list(ids), "shared": include_shared,
                    "width": width, "dims": dims, "group": 32}
        with SourceIdentity(ranges, geometry) as identity:
            status = packed_cache.read(identity, (up, down))
            if status != "hit":
                print(f"Packed expert cache {base}: {status}", flush=True)
                pack()
                identity.verify()
                status += "; " + packed_cache.write(identity, (up, down))
            print(f"Packed expert cache {base}: {status}", flush=True)
    result = grouped.Experts(up, down, 32, width, dims)
    result.logical_ids = ids + ((e,) if include_shared else ())
    return result

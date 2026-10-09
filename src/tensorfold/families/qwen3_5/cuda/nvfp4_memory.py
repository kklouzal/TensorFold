"""Header-only loading bound for the Qwen NVFP4 loader with BF16 prompts.

Direct reads allocate one final tensor. Projection conversions retain only that
projection's raw parts and bounded pack temporaries, rather than three copies
of the largest dense head. The byte formulas cover full and checkpoint math;
they do not change the loaded precision, scales, tensor order, or kernels.
"""
from __future__ import annotations

import math

_DIRECT_WEIGHT_SUFFIXES = (".norm", ".input_layernorm", ".post_attention_layernorm", ".q_norm", ".k_norm",
                           ".conv1d")


def projection_loading_bytes(entries: dict) -> dict:
    """Transient GPU allocation bound above final weights, for validated headers.

    The caller must disable FP8 prompt copies. Metadata must come from the
    capacity header boundary. The 4096-column pack loop remains unmodified.
    Raw parts and final projection outputs are deliberately counted again so
    the estimate also covers expression and prior-iteration lifetimes.
    """
    from tensorfold.cuda.capacity import SIZES as sizes

    groups = {}
    needs = {}
    for name, info in entries.items():
        if (name.startswith(("model.visual.", "visual.", "vision_tower", "mtp."))
                or ".visual." in name or ".mtp." in name):
            continue
        stem, suffix = name.rsplit(".", 1)
        if info["dtype"] not in sizes:
            raise ValueError(f"{name}: unsupported loading-bound dtype {info['dtype']}")
        groups.setdefault(stem, {})[suffix] = info
    for stem, parts in groups.items():
        weight = parts.get("weight", parts.get("weight_packed"))
        # A_log/dt_bias and scale-only groups need at most a pair of FP32
        # conversions; centered norms need float, add, and output together.
        if weight is None:
            need = sum(math.prod(p["shape"]) * (sizes[p["dtype"]] + 8) for p in parts.values())
            needs[stem] = need
            continue
        dtype, shape = weight["dtype"], weight["shape"]
        raw = sum(math.prod(p["shape"]) * sizes[p["dtype"]] for p in parts.values())
        if stem.endswith(".embed_tokens"):
            # Embeddings bypass fmt.scheme: get(...).to(BF16) accepts every
            # declared scalar dtype. BF16 retains its direct-read storage.
            if len(shape) != 2 or min(shape) <= 0:
                raise ValueError(f"{stem}: embedding must be a positive matrix")
            need = 0 if dtype == "BF16" else raw + 2 * math.prod(shape)
        elif stem.endswith(_DIRECT_WEIGHT_SUFFIXES):
            # Norms/conv tensors may retain their input dtype. Count the raw
            # input, both FP32 norm expression planes and the final dtype plane
            # together, independent of expression-temporary retirement timing.
            need = raw + (8 + max(2, sizes[dtype])) * math.prod(shape)
        elif len(shape) != 2:
            raise ValueError(f"{stem}: projection must be a matrix")
        elif dtype == "U8":
            if len(shape) != 2 or "weight_scale" not in parts or parts["weight_scale"]["dtype"] != "F8_E4M3":
                raise ValueError(f"{stem}: loading bound needs a supported NVFP4 projection")
            n, k = shape[0], 2 * shape[1]
            if n <= 0 or k <= 0 or k % 64:
                raise ValueError(f"{stem}: loading bound needs positive NVFP4 dimensions and K divisible by 64")
            if parts["weight_scale"]["shape"] != [n, k // 16]:
                raise ValueError(f"{stem}: NVFP4 block scales must have shape [{n}, {k // 16}]")
            npad = -(-n // 128) * 128
            c = min(npad, 4096)
            # Full path: output words, both block-scale orders, dummy + both
            # returned affine scales, then <=32 bytes per bounded pack cell.
            # Checkpoint pack4 uses fewer temporaries than the same bound.
            need = raw + npad * k // 2 + 4 * npad * k // 16 + 6 * npad * k // 64 + 32 * c * k
            need += 4 << 20  # fixed index/shift tensors and small expression inputs
        elif dtype == "F8_E4M3":
            if len(shape) != 2 or min(shape) <= 0 or shape[1] % 64:
                raise ValueError(f"{stem}: loading bound needs positive FP8 dimensions and K divisible by 64")
            n, k = shape
            npad = -(-n // 128) * 128
            # Raw e4m3, padded and indexed order; coordinate arrays expand as
            # views and their allocated sources are bounded by these planes.
            need = raw + 4 * npad * k + (4 << 20)
        elif dtype in ("BF16", "F16", "F32"):
            if dtype == "BF16" and stem.endswith(("lm_head", "embed_tokens")):
                # SafeTensors returns contiguous BF16; .to(BF16) and Plain
                # retain that very storage, so a dense final read has no copy.
                need = 0
            else:
                # A plain projection converts once to contiguous BF16. The
                # direct reader already returns contiguous input storage.
                need = raw + 2 * math.prod(shape)
        else:
            raise ValueError(f"{stem}: unsupported loading-bound dtype {dtype}")
        needs[stem] = need
    return needs


def loading_bytes(entries: dict) -> int:
    """Conservative per-projection envelope, independently of load schedule."""
    return max(projection_loading_bytes(entries).values(), default=0)


def scheduled_loading_bytes(entries: dict, transform) -> int:
    """Extra above final resident using the loader's exact acquisition order.

    All layer projections complete before constructing Weights. That expression
    acquires embed, then final norm, then head. Thus the two late modules' final
    bytes are absent throughout every earlier projection conversion. Only the
    head's bytes remain absent while embedding or final norm converts. Raw
    parts and all current outputs are still double-counted in the envelope.
    """
    from collections import defaultdict

    resident = defaultdict(int)
    for name, info in entries.items():
        stem = name.rsplit(".", 1)[0]
        resident[stem] += transform(name, info)[0]
    needs = projection_loading_bytes(entries)
    embeds = [n for n in needs if n.endswith(".embed_tokens")]
    embed = embeds[0] if len(embeds) == 1 else None
    head = "lm_head" if "lm_head" in needs else None
    if embed is None or head is None:
        raise ValueError("loading schedule requires a separate embedding and lm_head")
    late = resident[embed] + resident[head]
    peak = 0
    for stem, need in needs.items():
        pending = 0 if stem == head else resident[head] if stem == embed or stem.endswith(".norm") else late
        peak = max(peak, need - pending)
    # FP32 CUDA inv_freq at the end and fixed setup tensors. This reserve is
    # additional to, and does not modify, the configured allocator reserve.
    return peak + (4 << 20)


def weight_estimate(model_dir, transform):
    """Return the capacity Weights and separate conservative host read budget."""
    from tensorfold.cuda.capacity import Weights, _estimate_weights_from_headers, headers, itemsize

    entries = headers(model_dir)
    existing = _estimate_weights_from_headers(entries, transform)
    staging = scheduled_loading_bytes(entries, transform)
    # weight_bytes pads packed weight rows, but stored block scales still have
    # their original row count. Full math retains a separately padded bs plane;
    # checkpoint's 64-row padding is covered by this 128-row bound as well.
    padding = 0
    for name, info in entries.items():
        retained = transform(name, info)[0]
        if not retained:
            continue
        if name.endswith(".embed_tokens.weight"):
            # A one-byte embedding becomes retained two-byte BF16 storage.
            # Other input dtypes' generic resident bounds already overcount it.
            padding += max(0, 2 * math.prod(info["shape"]) - retained)
            continue
        stem = name.rsplit(".", 1)[0]
        if (stem.endswith(_DIRECT_WEIGHT_SUFFIXES) or info["dtype"] != "U8" or len(info["shape"]) != 2
                or not name.endswith((".weight", ".weight_packed"))):
            continue
        n, half = info["shape"]
        padding += (-(-n // 128) * 128 - n) * (2 * half // 16)
    # Buffered direct-read refusal can materialize the largest raw tensor on
    # the host. Two pinned 64-MiB rings are counted as well, including alignment.
    largest = max((math.prod(i["shape"]) * itemsize(i, n) for n, i in entries.items()
                   if transform(n, i)[0]), default=0)
    # Keep the original host conversion/staging allowance. This specialization
    # reduces only the proven device overlap; CPU admission is not relaxed.
    host = max(existing.staging, largest + 2 * ((64 << 20) + 3 * 4096))
    return Weights(existing.resident + padding, staging, existing.mapped), host

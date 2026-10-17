"""PRIVATE generated single-blob loading variant; regenerate, do not hand-edit."""

import torch

from .host_experts import _ids, load_host_experts_a4


def load_blob_host_experts_a4(ctx, name, full, device="cuda", *, cache=None, owned_ids=None, layer_id=None,
                         swiglu):
    """Load CT routed native-A4 experts in owned local order, without shared reads.

    Header/declaration validation precedes authority allocation; each projection
    reads its two scalar calibrations before code/scale payload. Packing one
    projection at a time bounds CPU temporary storage by three projection blobs.
    Resident loading releases CPU code/scale authority after synchronous device
    transfer, retaining36 bytes/expert of original and prepared calibration metadata.
    The model owns cache rollback/close if later model construction fails.
    """
    from . import format as fmt
    from .blob_cache_a4 import BlobCachedExpertsA4, BlobExpertCache
    from .experts_a4 import pack_projection_a4
    from .host_layout import authority_fields_a4

    if cache is None:
        return load_host_experts_a4(ctx, name, full, device, owned_ids=owned_ids, swiglu=swiglu)
    if not isinstance(cache, BlobExpertCache):
        raise TypeError('private one-blob loader requires BlobExpertCache')
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("native A4 experts require a CUDA device")
    device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
    if cache is not None and device != cache.device:
        raise ValueError("native A4 bank and cache must use the same CUDA device")
    if not ctx.checkpoint_math or not ctx.own.get("nvfp4"):
        raise ValueError("native A4 routed experts require checkpoint precision and native FP4 MMA")
    if not ctx.metadata.reciprocal or not callable(swiglu):
        raise ValueError("native A4 experts require CT reciprocal calibrations and a prepared BF16 SwiGLU")
    ids = _ids(full.experts, owned_ids)
    width, dims = full.moe_width, full.hidden
    authority_fields_a4(len(ids), width, dims)
    from .blob_layout_a4 import blob_geometry_a4

    if cache.blob_geometry != blob_geometry_a4(width, dims):
        raise ValueError('private blob cache differs from this expert authority')
    projections = (("gate_proj", width, dims), ("up_proj", width, dims), ("down_proj", dims, width))
    for expert in ids:
        for projection, n, k in projections:
            base = f"{name}.experts.{expert}.{projection}"
            parts = {suffix: ctx.info.get(f"{base}.{suffix}") for suffix in
                     ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")}
            expected = {"weight_packed": ("U8", [n, k // 2]), "weight_scale": ("F8_E4M3", [n, k // 16]),
                        "weight_global_scale": ("F32", [1]), "input_global_scale": ("F32", [1])}
            if parts != expected:
                raise ValueError(f"{base}: incompatible native A4 routed projection storage")
            kind, activation = ctx.metadata.validate(base, parts, checkpoint=True)
            if (kind, activation) != ("nvfp4", 4):
                raise ValueError(f"{base}: native A4 routed experts require declared four-bit activations")

    blob = cache.allocate_authority((len(ids), cache.entry_bytes), dtype=torch.uint8, tag=name)
    fields = cache.source_views(blob)
    cache.zero_padding(blob)
    for local, expert in enumerate(ids):
        for p, (projection, _, _) in enumerate(projections):
            base = f"{name}.experts.{expert}.{projection}"
            weight_global = ctx.read(base + ".weight_global_scale", device="cpu")
            input_global = ctx.read(base + ".input_global_scale", device="cpu")
            scale = fmt.calibration(float(weight_global.reshape(-1)[0]), name=base + ".weight_global_scale",
                                    reciprocal=True)
            inverse = fmt.calibration(float(input_global.reshape(-1)[0]), name=base + ".input_global_scale")
            act = fmt.calibration(inverse, name=base + ".input_global_scale", reciprocal=True)
            fmt.native_calibration(act, scale, name=base, act_inverse=inverse)
            if p < 2:
                fields[2][local, p], fields[4][local, p] = scale, inverse
            else:
                fields[3][local, 0], fields[5][local, 0] = scale, inverse
            codes = ctx.read(base + ".weight_packed", device="cpu")
            blocks = ctx.read(base + ".weight_scale", device="cpu")
            fmt.validate_native_a4_scales(blocks, name=base + ".weight_scale")
            packed = pack_projection_a4(codes, blocks)
            (fields[0][local, p] if p < 2 else fields[1][local]).copy_(packed)
            del codes, blocks, packed
    key = len(cache._layers) if layer_id is None else layer_id
    bank = BlobCachedExpertsA4(fields, width, dims, blob=blob, cache=cache, layer_id=key, swiglu=swiglu)
    ctx.used["nvfp4"] = True
    return bank

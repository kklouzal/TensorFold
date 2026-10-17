"""Original NVFP4 grouped bytes in host authority, with bounded CUDA leases.

Only routed experts spill. The separately represented shared expert stays on
the leader, and MTP retains its existing draft-only BF16/FP8 conversion.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from . import experts as nvx


def _ids(count, owned_ids):
    ids = list(range(count)) if owned_ids is None else list(owned_ids)
    if any(type(i) is not int or not 0 <= i < count for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("NVFP4 owned expert IDs must be distinct in-range integers")
    return ids


def _read(reader, name, *, device="cpu"):
    info = reader.info(name)
    if info["shape"]:
        return reader.get_rows(name, 0, info["shape"][0], device=device)
    # Scalars are tiny, but Reader.get follows the reader's selected device.
    return reader.get(name).to(device)


def _draft_projection(reader, base, device):
    codes = _read(reader, base + ".weight", device=device)
    if codes.dtype != torch.float8_e4m3fn:
        raise ValueError(f"{base}: expected FP8 e4m3 MTP expert weights")
    if reader.has(base + ".weight_scale_inv"):
        from .linear import Fp8BlockLinear

        scales = Fp8BlockLinear.column_scales(_read(reader, base + ".weight_scale_inv", device=device), *codes.shape)
        return (codes.float().view(codes.shape[0], -1, 64) * scales[..., None]).view(codes.shape).to(torch.bfloat16)
    scale = _read(reader, base + ".weight_scale", device=device)
    if scale.dtype not in (torch.bfloat16, torch.float16, torch.float32) or scale.numel() not in (1, codes.shape[0]):
        raise ValueError(f"{base}: FP8 MTP weight_scale must hold one float per tensor or output row")
    return (codes.float() * scale.float().reshape(-1, 1)).to(torch.bfloat16)


def load_nvfp4_host_experts(reader, name, full, device="cuda", *, cache=None, owned_ids=None,
                           include_shared=True):
    """Read only owned routed experts, in compact local order, one expert at a time.

    Shared payload is deliberately outside this authority. ``include_shared``
    is accepted with the complete-bank API; its resident owner loads it there.
    CPU packing only permutes original codes and block-scale bits. Draft-only
    conversion runs on the same device and operations as the resident loader.
    """
    if type(include_shared) is not bool:
        raise ValueError("include_shared must be a bool")
    ids = _ids(full.experts, owned_ids)
    width, dims, count = full.moe_width, full.hidden, len(ids)
    if any(type(n) is not int or n <= 0 or n % 32 for n in (width, dims)):
        raise ValueError("NVFP4 expert dimensions must be positive multiples of 32")
    if not count:
        return None
    stacked = reader.has(name + ".experts.gate_up_proj")
    if stacked:
        if not name.startswith("mtp."):
            raise ValueError("only draft-only MTP experts may use stacked BF16 weights")
        for field, shape in (("gate_up_proj", (full.experts, 2 * width, dims)),
                             ("down_proj", (full.experts, dims, width))):
            info = reader.info(name + ".experts." + field)
            if tuple(info["shape"]) != shape or info["dtype"] not in ("BF16", "F16", "F32"):
                raise ValueError(f"{name}: incompatible stacked MTP expert {field}")
    else:
        for expert in ids:
            for projection, n, k in (("gate_proj", width, dims), ("up_proj", width, dims), ("down_proj", dims, width)):
                base = f"{name}.experts.{expert}.{projection}"
                info = reader.info(base + ".weight")
                draft8 = name.startswith("mtp.") and info["dtype"] == "F8_E4M3"
                if tuple(info["shape"]) != (n, k if draft8 else k // 2) or info["dtype"] != (
                        "F8_E4M3" if draft8 else "U8"):
                    raise ValueError(f"{base}: incompatible NVFP4 routed projection")
                if not draft8:
                    scale = reader.info(base + ".weight_scale")
                    global_scale = reader.info(base + ".weight_scale_2")
                    if (scale["dtype"] != "F8_E4M3" or tuple(scale["shape"]) != (n, k // 16)
                            or global_scale["dtype"] not in ("BF16", "F16", "F32")
                            or tuple(global_scale["shape"]) not in ((), (1,))):
                        raise ValueError(f"{base}: incompatible NVFP4 block/global scales")

    def empty(shape, dtype):
        if cache is not None:
            return cache.allocate_authority(shape, dtype=dtype, tag=name)
        return torch.empty(shape, dtype=dtype, device="cpu")

    up = empty((count, width // 32, dims // 32, 2, nvx.WORDS), torch.int32)
    down = empty((count, dims // 32, width // 32, 1, nvx.WORDS), torch.int32)
    us, ds = empty((count, 2), torch.float32), empty((count, 1), torch.float32)
    for local, expert in enumerate(ids):
        if stacked:
            gu = reader.get_rows(name + ".experts.gate_up_proj", expert, expert + 1, device=device).to(torch.bfloat16)
            dn = reader.get_rows(name + ".experts.down_proj", expert, expert + 1, device=device).to(torch.bfloat16)
            triples = (nvx.quantize(gu[:, :width]), nvx.quantize(gu[:, width:]), nvx.quantize(dn))
        else:
            triples = []
            for projection in ("gate_proj", "up_proj", "down_proj"):
                base = f"{name}.experts.{expert}.{projection}"
                if reader.info(base + ".weight")["dtype"] == "F8_E4M3":
                    if not name.startswith("mtp."):
                        raise ValueError(f"{base}: FP8 routed experts are supported only in the MTP drafter")
                    triples.append(nvx.quantize(_draft_projection(reader, base, device).unsqueeze(0)))
                else:
                    w = _read(reader, base + ".weight")
                    s = _read(reader, base + ".weight_scale")
                    g = _read(reader, base + ".weight_scale_2")
                    if w.dtype != torch.uint8 or s.dtype != torch.float8_e4m3fn or g.numel() != 1:
                        raise ValueError(f"{base}: expected NVFP4 codes, e4m3 block scales and one global scale")
                    triples.append((w.unsqueeze(0), s.unsqueeze(0), g.float().reshape(1)))
        packed = nvx.make(*triples)
        for target, source in ((up, packed.up), (down, packed.down), (us, packed.up_scale), (ds, packed.down_scale)):
            target[local].copy_(source[0].to("cpu"))
        del triples, packed
    return nvx.Experts4(up, down, us, ds, width, dims)


class CachedExperts4:
    """Four synchronized per-expert fields; original logical IDs map to hot slots."""

    capturable = False

    def __init__(self, authority, cache, layer_id):
        self.cache, self.layer_id = cache, layer_id
        self.width, self.dims, self.limit = authority.width, authority.dims, authority.limit
        self.count = authority.count
        cache.register(layer_id, (authority.up, authority.down, authority.up_scale, authority.down_scale))

    @property
    def capacity(self):
        return self.cache.capacity

    def device_tensors(self):
        return (self.cache._pool,)

    @contextmanager
    def lease(self, ids):
        with self.cache.lease(self.layer_id, ids) as (payloads, mapping):
            yield nvx.Experts4(*payloads, self.width, self.dims, self.limit), mapping


def load_host_experts_a4(ctx, name, full, device="cuda", *, cache=None, owned_ids=None, layer_id=None,
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
    from .experts_a4 import CachedExpertsA4, ExpertsA4, pack_projection_a4
    from .host_layout import authority_fields_a4

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
    geometry = authority_fields_a4(len(ids), width, dims)
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

    fields = tuple(cache.allocate_authority(shape, dtype=torch.uint8 if dtype == "U8" else torch.float32, tag=name)
                   if cache is not None else torch.empty(shape, dtype=torch.uint8 if dtype == "U8" else torch.float32)
                   for shape, dtype in geometry)
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
    if cache is not None:
        key = len(cache._layers) if layer_id is None else layer_id
        bank = CachedExpertsA4(fields, width, dims, cache=cache, layer_id=key, swiglu=swiglu)
    else:
        bank = ExpertsA4(tuple(t.to(device) for t in fields), fields[2:], width, dims, swiglu=swiglu)
    ctx.used["nvfp4"] = True
    return bank

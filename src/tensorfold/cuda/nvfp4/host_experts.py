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


def _original_projection(reader, base, device):
    """Read original payload on the pack device; scalar conversion stays on CPU."""
    w = _read(reader, base + ".weight", device=device)
    s = _read(reader, base + ".weight_scale", device=device)
    g = _read(reader, base + ".weight_scale_2")
    if w.dtype != torch.uint8 or s.dtype != torch.float8_e4m3fn or g.numel() != 1:
        raise ValueError(f"{base}: expected NVFP4 codes, e4m3 block scales and one global scale")
    return w.unsqueeze(0), s.unsqueeze(0), g.float().reshape(1)


def load_nvfp4_host_experts(reader, name, full, device="cuda", *, cache=None, owned_ids=None,
                           include_shared=True):
    """Read only owned routed experts, in compact local order, one expert at a time.

    Shared payload is deliberately outside this authority. ``include_shared``
    is accepted with the complete-bank API; its resident owner loads it there.
    Packing permutes original codes and block-scale bits on the selected device;
    synchronous CPU transfers finish before publishing each expert. Global scale
    conversion stays on CPU. Draft-only conversion retains the resident math.
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
                    triples.append(_original_projection(reader, base, device))
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


def _preflight_a4_calibrations(calibrations, owners, projections, fields):
    """Prepare private fields once, or explicitly select ordered input errors.

    Only shared validators' represented invalid calibration results return
    False. No exception, helper defect or resource failure becomes eligibility.
    A later invalid global must not supersede an earlier invalid block scale;
    the canonical ordered path reports that failure using the original APIs.
    """
    from . import format as fmt

    for local, owner in enumerate(owners):
        for p, (projection, _, _) in enumerate(projections):
            base = f"{owner}.{projection}"
            offset = 6 * local + 2 * p
            scale, error = fmt._calibration_result(float(calibrations[offset]), reciprocal=True)
            if error is not None:
                return False
            inverse, error = fmt._calibration_result(float(calibrations[offset + 1]))
            if error is not None:
                return False
            act, error = fmt._calibration_result(inverse, reciprocal=True)
            if error is not None:
                return False
            _, error = fmt._native_factors_result(act, scale, name=base, act_inverse=inverse)
            if error is not None:
                return False
            if p < 2:
                fields[2][local, p], fields[4][local, p] = scale, inverse
            else:
                fields[3][local, 0], fields[5][local, 0] = scale, inverse
    return True


def load_host_experts_a4(ctx, name, full, device="cuda", *, cache=None, owned_ids=None, layer_id=None,
                         swiglu, shared=False):
    """Load CT routed native-A4 experts in owned local order.

    ``shared`` (resident banks only) appends the layer's ``shared_expert`` as the
    last cell, so its ID ``len(ids)`` runs in the same grouped device plan, when
    its three projections store the routed geometry's native A4 fields with
    declared four-bit activations (headers only); any other shared expert stays
    out of the bank (``count == len(ids)``) and its owner loads it separately.
    Without a shared cell no shared field is read.

    Header/declaration validation precedes authority allocation. One bounded
    CPU scalar batch contains exactly the validated owners' F32 calibrations.
    Canonical Reader runs can prepare all globals without changing arithmetic,
    own one SF run and bounded code chunks, and write exact tiled bytes into
    final authority. Invalid globals/unsupported topology retain ordered reads;
    invalid SF retains original projection error order using private SF views.
    All raw owners/alignment stay inside the admitted reader window, with
    planning metadata charged separately for resident and offloaded loading.
    Resident loading releases CPU code/scale authority after synchronous device
    transfer. Serving retains no CPU calibration metadata for either residency.
    The model owns cache rollback/close if later model construction fails.
    """
    from . import format as fmt
    from .experts_a4 import CachedExpertsA4, ExpertsA4, pack_projection_a4, uniform_input_bits
    from .host_layout import authority_fields_a4, bulk_layout_a4, entry_layout_a4

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
    if shared and cache is not None:
        raise ValueError("a shared-expert cell requires a resident native A4 bank")
    ids = _ids(full.experts, owned_ids)
    width, dims = full.moe_width, full.hidden
    projections = (("gate_proj", width, dims), ("up_proj", width, dims), ("down_proj", dims, width))

    def mismatch(base, n, k):
        """Why one projection's headers/declaration differ from the routed native-A4 form, else None."""
        parts = {suffix: ctx.info.get(f"{base}.{suffix}") for suffix in
                 ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")}
        if parts != {"weight_packed": ("U8", [n, k // 2]), "weight_scale": ("F8_E4M3", [n, k // 16]),
                     "weight_global_scale": ("F32", [1]), "input_global_scale": ("F32", [1])}:
            return "incompatible native A4 routed projection storage"
        if ctx.metadata.validate(base, parts, checkpoint=True) != ("nvfp4", 4):
            return "native A4 routed experts require declared four-bit activations"
        return None

    # A shared expert of another width or representation keeps its separate projections (the owner's path).
    shared = shared and not any(mismatch(f"{name}.shared_expert.{projection}", n, k)
                                for projection, n, k in projections)
    owners = [f"{name}.experts.{expert}" for expert in ids] + ([f"{name}.shared_expert"] if shared else [])
    geometry = authority_fields_a4(len(owners), width, dims)
    for owner in owners[:len(ids)]:          # an admitted shared cell matched above
        for projection, n, k in projections:
            if error := mismatch(f"{owner}.{projection}", n, k):
                raise ValueError(f"{owner}.{projection}: {error}")

    if cache is None:
        fields = tuple(torch.empty(shape, dtype=torch.uint8 if dtype == "U8" else torch.float32)
                       for shape, dtype in geometry)
    else:
        if cache.field_layout != entry_layout_a4(width, dims, count=len(ids)):
            raise ValueError("native A4 cache differs from the admitted expert field geometry")
        source = cache.allocate_authority((len(ids), cache.entry_bytes), dtype=torch.uint8, tag=name)
        fields = cache.source_views(source)
        cache.zero_padding(source)
    scalar_names = tuple(f"{owner}.{projection}.{suffix}" for owner in owners
                         for projection, _, _ in projections
                         for suffix in ("weight_global_scale", "input_global_scale"))
    # Every selected header was proved F32[1] above. Derive the exact cap,
    # including a possible resident shared cell, before any batched I/O.
    calibrations = ctx.read_f32_scalars(scalar_names, max_bytes=len(scalar_names) * 4)
    bulk = bulk_layout_a4(len(owners), width, dims)
    code_run = scale_run = physical = physical_at = all_scales = None
    prepared = packed = False
    if bulk is not None:
        physical = sorted(range(len(owners)), key=lambda local: owners[local])
        code_names = tuple(f"{owners[local]}.{projection}.weight_packed" for local in physical
                           for projection in ("down_proj", "gate_proj", "up_proj"))
        scale_names = tuple(f"{owners[local]}.{projection}.weight_scale" for local in physical
                            for projection in ("down_proj", "gate_proj", "up_proj"))
        code_run = ctx.prepare_cpu_run(code_names, dtype="U8", field_bytes=bulk.code_field_bytes,
                                       max_read_bytes=bulk.chunk_experts * 3 * bulk.code_field_bytes)
        scale_run = ctx.prepare_cpu_run(scale_names, dtype="F8_E4M3", field_bytes=bulk.scale_field_bytes,
                                        max_read_bytes=bulk.scale_bytes)
        del code_names, scale_names
        if code_run is not None and scale_run is not None:
            prepared = _preflight_a4_calibrations(calibrations, owners, projections, fields)
            if prepared:
                all_scales = scale_run(0, 3 * len(owners))
                # Canonical owned plain CPU U8, no floating reduction or bridge.
                # Invalid runs retain projection-ordered names/errors below.
                if int(all_scales.amax().item()) <= 0x7e:
                    code_bytes, scale_bytes = bulk.code_field_bytes, bulk.scale_field_bytes
                    for first in range(0, len(owners), bulk.chunk_experts):
                        stop = min(first + bulk.chunk_experts, len(owners))
                        codes = code_run(3 * first, 3 * stop)
                        for at in range(first, stop):
                            local = physical[at]
                            code_cell = (at - first) * 3 * code_bytes
                            scale_cell = at * 3 * scale_bytes
                            for p, (_, n, k) in enumerate(projections):
                                # The file is down/gate/up; destination rows
                                # retain the caller's gate/up/down local order.
                                wire = (p + 1) % 3
                                code_offset = code_cell + wire * code_bytes
                                scale_offset = scale_cell + wire * scale_bytes
                                weight = codes[code_offset:code_offset + code_bytes].view(n, k // 2)
                                blocks = (all_scales[scale_offset:scale_offset + scale_bytes]
                                          .view(torch.float8_e4m3fn).view(n, k // 16))
                                target = fields[0][local, p] if p < 2 else fields[1][local]
                                pack_projection_a4(weight, blocks, out=target)
                                del weight, blocks, target
                        del codes
                    packed = True
                else:
                    physical_at = [0] * len(owners)
                    for at, local in enumerate(physical):
                        physical_at[local] = at
    if not packed:
        for local, owner in enumerate(owners):
            for p, (projection, n, k) in enumerate(projections):
                base = f"{owner}.{projection}"
                if not prepared:
                    # One ordered owner; retain the original scalar conversion,
                    # validation/error order and CT multiplier without a roundtrip.
                    offset = 6 * local + 2 * p
                    scale = fmt.calibration(float(calibrations[offset]), name=base + ".weight_global_scale",
                                            reciprocal=True)
                    inverse = fmt.calibration(float(calibrations[offset + 1]), name=base + ".input_global_scale")
                    act = fmt.calibration(inverse, name=base + ".input_global_scale", reciprocal=True)
                    fmt.native_calibration(act, scale, name=base, act_inverse=inverse)
                    if p < 2:
                        fields[2][local, p], fields[4][local, p] = scale, inverse
                    else:
                        fields[3][local, 0], fields[5][local, 0] = scale, inverse
                codes = ctx.read(base + ".weight_packed", device="cpu")
                if all_scales is None:
                    blocks = ctx.read(base + ".weight_scale", device="cpu")
                else:
                    at = (physical_at[local] * 3 + (p + 1) % 3) * bulk.scale_field_bytes
                    blocks = all_scales[at:at + bulk.scale_field_bytes].view(torch.float8_e4m3fn).view(n, k // 16)
                fmt.validate_native_a4_scales(blocks, name=base + ".weight_scale")
                target = fields[0][local, p] if p < 2 else fields[1][local]
                pack_projection_a4(codes, blocks, out=target)
                del codes, blocks
    del calibrations, scalar_names, code_run, scale_run, physical, physical_at, all_scales
    if cache is not None:
        key = len(cache._layers) if layer_id is None else layer_id
        bank = CachedExpertsA4(len(ids), width, dims, cache=cache, layer_id=key, swiglu=swiglu)
        cache.register(key, (source,))
    else:
        # The complete CPU gate/up multiplier bit proof precedes exact transfer.
        bank = ExpertsA4(tuple(t.to(device) for t in fields), width, dims, swiglu=swiglu,
                         shared_input=uniform_input_bits(fields[4]))
    ctx.used["nvfp4"] = True
    return bank

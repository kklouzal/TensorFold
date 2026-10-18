"""ModelOpt static-A4 experts: original direct scales and device-resident routes.

The six fields share native A4 code/scale packing with CT, but their input and
weight globals are direct dequantization scales. That distinction is prepared
once; no reciprocal roundtrip may change the stored ModelOpt alpha.
"""

import torch

from tensorfold.cuda.moe import swiglu_bf16

from . import format as fmt
from .experts_a4 import CachedExpertsA4, ExpertsA4
from .grouped_a4 import _ext
from .host_layout import authority_fields_a4


def uniform_input_bits(calibrations):
    """Prove exact prepared gate/up FP32 equality over the complete CPU bank."""
    if (calibrations.device.type != "cpu" or calibrations.dtype != torch.float32 or calibrations.ndim != 2
            or calibrations.shape[0] <= 0 or calibrations.shape[1] != 2 or not calibrations.is_contiguous()):
        raise ValueError("common input proof requires complete contiguous CPU FP32[experts,2] calibrations")
    bits = calibrations.view(torch.int32)
    return bool((bits == bits[0, 0]).all())


class ModelOptExpertsA4(ExpertsA4):
    """Resident bank with live GPU routing plans; graph replay never reads picks on CPU."""

    direct_input = True
    capturable = True

    def __init__(self, fields, width, dims, *, shared_input=False):
        # True is supplied only by the loader's complete CPU bit proof, before
        # exact CPU->CUDA transfer; immutable bank fields preserve that proof.
        self.shared_input = shared_input
        super().__init__(fields, width, dims, swiglu=swiglu_bf16)

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.fields)

    def run_plan(self, x, buf, *, prefill):
        from tensorfold.cuda.kernels import qmm

        rows, slots = x.shape[0], buf.slots
        plan = buf.plan
        if plan.tile != 16:
            raise ValueError("native ModelOpt A4 requires device-plan items of at most16 pairs")
        self._check(x, buf.pick[:rows], buf.y[:rows], prefill=prefill)
        # The buffer, not every expert bank, owns reusable shape-specific
        # storage. Allocate all retained tensors before publishing the tuple.
        scratch = getattr(buf, "modelopt_a4", None)
        if scratch is None:
            pairs = buf.rows * slots
            with torch.inference_mode(False):
                scratch = (torch.empty((2, pairs, self.dims // 2), dtype=torch.uint8, device=x.device),
                           torch.empty((2, pairs, self.dims // 16), dtype=torch.uint8, device=x.device),
                           torch.empty((2, pairs, self.width), dtype=torch.bfloat16, device=x.device),
                           torch.empty((1, pairs, self.width // 2), dtype=torch.uint8, device=x.device),
                           torch.empty((1, pairs, self.width // 16), dtype=torch.uint8, device=x.device))
            buf.modelopt_a4 = scratch
        pairs = rows * slots
        dc, ds = (t[:, :pairs] for t in scratch[3:])
        # Projection-major narrowing leaves non-contiguous outer strides when
        # the live row count is smaller than the admitted buffer. Use a flat
        # prefix reshaped to the actual plan; all backing remains buffer-owned.
        codes = scratch[0].view(-1)[:2 * pairs * (self.dims // 2)].view(2, pairs, self.dims // 2)
        scales = scratch[1].view(-1)[:2 * pairs * (self.dims // 16)].view(2, pairs, self.dims // 16)
        gu = scratch[2].view(-1)[:2 * pairs * self.width].view(2, pairs, self.width)
        # Shared/unused plan positions have no routed GEMM. Initialize before
        # the eager SiLU consumes the complete tensor; routed rows overwrite it.
        gu.zero_()
        ext = _ext()
        fields = self.fields
        items = plan.items[:min(pairs, self.count + 1) + pairs // 16]
        members = plan.members[:pairs]
        common = self.shared_input and rows <= 65535
        if common:
            row_plan = getattr(buf, "modelopt_a4_row_plan", None)
            if row_plan is None:
                with torch.inference_mode(False):
                    indices = torch.arange(buf.rows, dtype=torch.int32, device=x.device)
                    zero = torch.zeros_like(indices)
                    row_plan = torch.stack((indices, zero, zero), dim=1)
                buf.modelopt_a4_row_plan = row_plan
                del indices, zero
            # Reuse the existing input backing; the shared representation is a
            # smaller prefix. Mixed common/nonuniform layers remain bounded by
            # the same one pair of admitted code/scale allocations.
            codes = scratch[0].view(-1)[:rows * (self.dims // 2)].view(1, rows, self.dims // 2)
            scales = scratch[1].view(-1)[:rows * (self.dims // 16)].view(1, rows, self.dims // 16)
            ext.quantize(x, row_plan[:rows], fields[4][:1, :1], codes, scales, False, True)
        else:
            ext.quantize_plan(x, items, plan.counts, members, fields[4], codes, scales, slots, False)
        ext.gemm_plan(codes, scales, fields[0], fields[2], fields[4], items, plan.counts, members,
                      gu, self.width, self.dims, 1 if prefill else qmm.split_k(self.width, self.dims), slots, False,
                      common)
        hidden = swiglu_bf16(gu[0], gu[1])
        ext.quantize_plan(hidden, items, plan.counts, members, fields[5], dc, ds, slots, True)
        ext.gemm_plan(dc, ds, self.native_down, fields[3], fields[5], items, plan.counts, members,
                      buf.y[:rows], self.dims, self.width,
                      1 if prefill else qmm.split_k(self.dims, self.width), slots, True)


class CachedModelOptExpertsA4(CachedExpertsA4):
    """Original direct calibrations under the shared native-A4 lease contract."""

    direct_input = True

    def __init__(self, fields, cache, layer_id, width, dims):
        super().__init__(fields[0].shape[0], width, dims, cache=cache, layer_id=layer_id, swiglu=swiglu_bf16)
        cache.register(layer_id, tuple(fields))


def pack_projection_a4_device(weight, scale):
    """GPU-assisted original-byte permutation into the native A4 layout.

    This owned pack4 kernel copies original code bits; E4M3 bytes only undergo
    a tiled permutation. It never dequantizes or requantizes either payload.
    """
    from .checkpoint import pack4

    if (weight.device.type != "cuda" or scale.device != weight.device or weight.dtype != torch.uint8
            or weight.ndim != 2 or scale.dtype != torch.float8_e4m3fn):
        raise ValueError("device A4 packing requires CUDA U8 codes and E4M3 scales on the same device")
    n, k = weight.shape[0], weight.shape[1] * 2
    if n <= 0 or k <= 0 or n % 64 or k % 64 or tuple(scale.shape) != (n, k // 16):
        raise ValueError("device A4 projection dimensions/scales must form complete64-wide tiles")
    words = pack4(weight, n)
    bs = scale.contiguous().view(torch.uint8).view(n // 64, 64, k // 64, 4).permute(0, 2, 1, 3).contiguous()
    return torch.cat((words.view(torch.uint8).reshape(-1), bs.reshape(-1)))


def load_modelopt_a4(reader, name, full, *, device, cache=None, owned_ids=None, layer_id=None):
    """Pack original bytes on the selected GPU, one projection at a time.

    Resident payloads go straight into final device fields. Only a host-cache
    bank copies those packed bytes back to admitted CPU authority. Scalar
    validation stays on CPU; all original block-scale bytes are validated by
    one bank-wide GPU reduction before any native bank is published.
    """
    from .host_experts import _ids, _read

    ids = _ids(full.experts, owned_ids)
    if not ids:
        return None
    geometry = authority_fields_a4(len(ids), full.moe_width, full.hidden)
    fields = tuple((cache.allocate_authority(shape, dtype=torch.uint8 if dtype == "U8" else torch.float32)
                    if cache is not None else torch.empty(shape,
                        dtype=torch.uint8 if dtype == "U8" else torch.float32,
                        device=device if index < 2 else "cpu"))
                   for index, (shape, dtype) in enumerate(geometry))
    maximum_scale = None
    for local, expert in enumerate(ids):
        for projection, n, k, payload, scalar, column in (
                ("gate_proj", full.moe_width, full.hidden, 0, 2, 0),
                ("up_proj", full.moe_width, full.hidden, 0, 2, 1),
                ("down_proj", full.hidden, full.moe_width, 1, 3, 0)):
            base = f"{name}.experts.{expert}.{projection}"
            weight_scale = fmt.calibration(_read(reader, base + ".weight_scale_2"), name=base + ".weight_scale_2")
            act = fmt.calibration(_read(reader, base + ".input_scale"), name=base + ".input_scale")
            fmt.native_calibration(act, weight_scale, name=base)
            words = _read(reader, base + ".weight", device=device)
            scales = _read(reader, base + ".weight_scale", device=device)
            local_maximum = scales.view(torch.uint8).amax()
            maximum_scale = local_maximum if maximum_scale is None else torch.maximum(maximum_scale, local_maximum)
            packed = pack_projection_a4_device(words, scales)
            target = fields[payload][local, column] if payload == 0 else fields[payload][local]
            target.copy_(packed)
            fields[scalar][local, column] = weight_scale
            fields[scalar + 2][local, column] = act
            # Retire raw/permutation/join temporaries before the next read or
            # pack. The admitted transform peak is one projection, not two.
            del words, scales, packed, target, local_maximum
    if int(maximum_scale) > 0x7e:
        raise ValueError(f"{name}: native A4 block scales must be finite unsigned E4M3 bytes in 0x00..0x7e")
    if cache is not None:
        key = len(cache._layers) if layer_id is None else layer_id
        return CachedModelOptExpertsA4(fields, cache, key, full.moe_width, full.hidden)
    # Compare the complete prepared FP32 bit inventory, not sampled experts
    # or floating equality. Cache banks retain their general leased quantizer.
    shared_input = uniform_input_bits(fields[4])
    return ModelOptExpertsA4(tuple(t if index < 2 else t.to(device=device)
                                   for index, t in enumerate(fields)), full.moe_width, full.hidden,
                            shared_input=shared_input)

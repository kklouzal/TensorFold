"""ModelOpt static-A4 experts: original direct scales and device-resident routes.

The six fields share native A4 code/scale packing with CT, but their input and
weight globals are direct dequantization scales. That distinction is prepared
once; no reciprocal roundtrip may change the stored ModelOpt alpha.
"""

import torch

from tensorfold.cuda.moe import swiglu_bf16

from . import format as fmt
from .experts_a4 import CachedExpertsA4, ExpertsA4, pack_projection_a4
from .grouped_a4 import _ext
from .host_layout import authority_fields_a4


class ModelOptExpertsA4(ExpertsA4):
    """Resident bank with live GPU routing plans; graph replay never reads picks on CPU."""

    direct_input = True
    capturable = True

    def __init__(self, fields, width, dims):
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
        # Plan backing has conservative spare rows, but every live item owns
        # at least one pair. Never expose more item rows than actual pairs.
        items = plan.items[:min(pairs, self.count + 1 + pairs // 16)]
        members = plan.members[:pairs]
        ext.quantize_plan(x, items, plan.counts, members, fields[4], codes, scales, slots, False)
        ext.gemm_plan(codes, scales, fields[0], fields[2], fields[4], items, plan.counts, members,
                      gu, self.width, self.dims, 1 if prefill else qmm.split_k(self.width, self.dims), slots, False)
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


def load_modelopt_a4(reader, name, full, *, device, cache=None, owned_ids=None, layer_id=None):
    """Pack admitted original ModelOpt fields on CPU, one projection at a time.

    The caller validates the bank's complete metadata/header before any payload
    reads. Cache authority owns all six fields; no raw checkpoint views survive.
    """
    from .host_experts import _ids, _read

    ids = _ids(full.experts, owned_ids)
    if not ids:
        return None
    geometry = authority_fields_a4(len(ids), full.moe_width, full.hidden)
    fields = tuple((cache.allocate_authority(shape, dtype=torch.uint8 if dtype == "U8" else torch.float32)
                    if cache is not None else
                    torch.empty(shape, dtype=torch.uint8 if dtype == "U8" else torch.float32, device="cpu"))
                   for shape, dtype in geometry)
    for local, expert in enumerate(ids):
        for projection, n, k, payload, scalar, column in (
                ("gate_proj", full.moe_width, full.hidden, 0, 2, 0),
                ("up_proj", full.moe_width, full.hidden, 0, 2, 1),
                ("down_proj", full.hidden, full.moe_width, 1, 3, 0)):
            base = f"{name}.experts.{expert}.{projection}"
            weight_scale = fmt.calibration(_read(reader, base + ".weight_scale_2"), name=base + ".weight_scale_2")
            act = fmt.calibration(_read(reader, base + ".input_scale"), name=base + ".input_scale")
            fmt.native_calibration(act, weight_scale, name=base)
            words, scales = _read(reader, base + ".weight"), _read(reader, base + ".weight_scale")
            fmt.validate_native_a4_scales(scales, name=base)
            packed = pack_projection_a4(words, scales)
            target = fields[payload][local, column] if payload == 0 else fields[payload][local]
            target.copy_(packed)
            fields[scalar][local, column] = weight_scale
            fields[scalar + 2][local, column] = act
            # Retire raw/permutation/join temporaries before the next read or
            # pack. The admitted transform peak is one projection, not two.
            del words, scales, packed, target
    if cache is not None:
        key = len(cache._layers) if layer_id is None else layer_id
        return CachedModelOptExpertsA4(fields, cache, key, full.moe_width, full.hidden)
    return ModelOptExpertsA4(tuple(t.to(device=device) for t in fields), full.moe_width, full.hidden)

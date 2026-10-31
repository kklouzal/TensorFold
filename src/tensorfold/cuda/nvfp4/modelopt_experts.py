"""ModelOpt static-A4 experts: original direct scales and device-resident routes.

The six fields share native A4 code/scale packing with CT, but their input and
weight globals are direct dequantization scales. That distinction is prepared
once; no reciprocal roundtrip may change the stored ModelOpt alpha.
"""

import torch

from tensorfold.cuda.moe import swiglu_bf16

from . import format as fmt
from .experts_a4 import CachedExpertsA4, ExpertsA4, uniform_input_bits
from .host_layout import authority_fields_a4


class ModelOptExpertsA4(ExpertsA4):
    """Resident bank with original direct calibrations; device plans via ``ExpertsA4.run_plan``."""

    direct_input = True

    def __init__(self, fields, width, dims, *, shared_input=False):
        super().__init__(fields, width, dims, swiglu=swiglu_bf16, shared_input=shared_input)


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

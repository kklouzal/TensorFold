"""Original ModelOpt static-A4 fields and separate individual-linear oracle."""

import struct
from types import SimpleNamespace

import torch

PREFIX = "model.language_model.layers.0.mlp"


class Reader:
    def __init__(self, *, count=3, width=64, dims=128):
        self.cfg = SimpleNamespace(experts=count, moe_width=width, hidden=dims, top_k=min(2, count))
        self.payload, self.reads = {}, []
        for expert in range(count):
            for p, (projection, n, k) in enumerate((("gate_proj", width, dims), ("up_proj", width, dims),
                                                   ("down_proj", dims, width))):
                base = f"{PREFIX}.experts.{expert}.{projection}"
                values = {
                    "weight": ((torch.arange(n * k // 2, dtype=torch.int64) * 17 + 31 * expert + p)
                               % 256).to(torch.uint8).view(n, k // 2),
                    "weight_scale": ((torch.arange(n * k // 16, dtype=torch.int64) + expert + p) % 16 + 40)
                                      .to(torch.uint8).view(n, k // 16).view(torch.float8_e4m3fn),
                    "weight_scale_2": torch.tensor(.00020781018247362226 * (1 + expert + p), dtype=torch.float32),
                    "input_scale": torch.tensor(.001964750699698925 * (1 + expert + p), dtype=torch.float32)}
                if expert == p == 0:
                    # A direct-scale alpha must preserve this bit pattern;
                    # FP32(1 / FP32(1 / scale)) differs from the stored value.
                    values["input_scale"] = torch.tensor(struct.unpack("f", struct.pack("I", 0x3b350913))[0])
                self.payload.update({base + "." + key: value for key, value in values.items()})
        self.where = dict.fromkeys(self.payload, "fixture.safetensors")

    def info(self, name):
        value = self.payload[name]
        dtype = {torch.uint8: "U8", torch.float32: "F32", torch.float8_e4m3fn: "F8_E4M3", torch.bfloat16: "BF16"}[value.dtype]
        return {"dtype": dtype, "shape": tuple(value.shape)}

    def has(self, name):
        return name in self.payload

    def _take(self, name, device):
        self.reads.append(name)
        return self.payload[name].to(device)

    def get(self, name):
        return self._take(name, "cpu")

    def get_rows(self, name, lo, hi, device="cpu"):
        return self._take(name, device)[lo:hi]

    def headers(self):
        return {name: self.info(name) for name in self.payload}


def quant_reference(x, direct_scale):
    """Independent FP32/direct-scale CPU activation encoder.

    Separate from the production reciprocal/quantizer helpers and native pack:
    E4M3 uses CPU RNE encoding; E2M1 selects the nearest representable value,
    breaking exact midpoint ties by the even code. Zero has canonical +0.
    """
    blocks = x.cpu().float().reshape(x.shape[0], -1, 16)
    inverse = torch.tensor(1., dtype=torch.float32) / torch.tensor(direct_scale, dtype=torch.float32)
    # CUDA fmax starts from0 and ignores a NaN operand. The quantizer's
    # saturating E2M1 conversion maps a later NaN product to positive MAX_NORM.
    amax = torch.nan_to_num(blocks.abs(), nan=0., posinf=float("inf")).amax(-1)
    scale = (inverse * (amax * torch.tensor(1. / 6., dtype=torch.float32))).clamp(max=448.)
    encoded = scale.to(torch.float8_e4m3fn)
    multiplier = torch.where(encoded.float() != 0, inverse / encoded.float(), torch.zeros_like(scale))
    values = blocks * multiplier[..., None]
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    magnitude = torch.nan_to_num(values.abs(), nan=6., posinf=6.).clamp(max=6.)
    distances = (magnitude[..., None] - levels).abs()
    # Even indices are considered first, making tie policy explicit rather
    # than inheriting argmin's lower-index rule (incorrect at .75/1.75/3.5).
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
    codes = order[distances[..., order].argmin(-1)].to(torch.uint8)
    codes |= ((values < 0) & (codes != 0)).to(torch.uint8) * 8
    codes = codes.reshape(x.shape)
    return codes[:, ::2] | (codes[:, 1::2] << 4), encoded.view(torch.uint8)


def individual_reference(reader, x, picks, *, prefill):
    """Separate linear pack/matmul for each original expert and routed pair."""
    from tensorfold.cuda.nvfp4 import checkpoint
    from tensorfold.cuda.nvfp4.linear import Fp4Linear

    linears = []
    for expert in range(reader.cfg.experts):
        projections = []
        for projection in ("gate_proj", "up_proj", "down_proj"):
            base = f"{PREFIX}.experts.{expert}.{projection}"
            data = reader.payload
            projections.append(Fp4Linear.from_checkpoint(data[base + ".weight"].cuda(),
                data[base + ".weight_scale"].cuda(), float(data[base + ".weight_scale_2"]),
                act=float(data[base + ".input_scale"])))
        linears.append(projections)
    output = torch.zeros((*picks.shape, reader.cfg.hidden), dtype=torch.bfloat16 if prefill else torch.float32,
                         device=x.device)
    native = checkpoint.prompt if prefill else checkpoint.matmul
    for row, values in enumerate(picks.cpu().tolist()):
        for slot, expert in enumerate(values):
            if not 0 <= expert < reader.cfg.experts:
                continue
            gate, up, down = linears[expert]
            g = native(checkpoint.A4, x[row:row + 1], gate)
            u = native(checkpoint.A4, x[row:row + 1], up)
            hidden = torch.nn.functional.silu(g) * u
            output[row, slot] = native(checkpoint.A4, hidden, down, f32=not prefill)[0]
    return output

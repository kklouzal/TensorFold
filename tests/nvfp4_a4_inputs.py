"""Tiny CT experts with deliberately unequal per-projection calibrations."""

import struct
from types import SimpleNamespace

import torch

from tensorfold.cuda.nvfp4.format import LinearMetadata

PREFIX = "model.language_model.layers.0.mlp"


def swiglu(gate, up):
    return torch.nn.functional.silu(gate) * up


swiglu.activation = 0       # the declared BF16 SiLU/product that native A4 device plans fuse


class Context:
    def __init__(self, *, count=3, width=64, dims=128):
        self.cfg = SimpleNamespace(experts=count, moe_width=width, hidden=dims, top_k=2)
        self.info, self.payload, self.reads, self.used = {}, {}, [], {}
        self.checkpoint_math, self.own = True, {"nvfp4": True}
        self.metadata = LinearMetadata({"quant_method": "compressed-tensors", "config_groups": {"nv": {
            "targets": ["Linear"],
            "weights": {"num_bits": 4, "type": "float", "group_size": 16, "strategy": "tensor_group"},
            "input_activations": {"num_bits": 4, "type": "float", "group_size": 16,
                                  "strategy": "tensor_group", "dynamic": "local"}}}})
        for expert in range(count):
            for p, (projection, n, k) in enumerate((("gate_proj", width, dims), ("up_proj", width, dims),
                                                   ("down_proj", dims, width))):
                base = f"{PREFIX}.experts.{expert}.{projection}"
                values = {
                    "weight_packed": ((torch.arange(n * k // 2, dtype=torch.int64) * 17 + 31 * expert + p)
                                      % 256).to(torch.uint8).view(n, k // 2),
                    "weight_scale": ((torch.arange(n * k // 16, dtype=torch.int64) + expert + p) % 16 + 40)
                                      .to(torch.uint8).view(n, k // 16).view(torch.float8_e4m3fn),
                    "weight_global_scale": torch.tensor([37 + 3 * expert + 7 * p], dtype=torch.float32),
                    "input_global_scale": torch.tensor([19 + 5 * expert + 11 * p], dtype=torch.float32)}
                if expert == 0 and p == 0:
                    # BF16amax0.5 produces an E4M3 rounding boundary for this
                    # original multiplier; reciprocal roundtrip changes0x0a to0x09.
                    values["input_global_scale"][0] = struct.unpack("f", struct.pack("I", 0x3e63ffff))[0]
                for suffix, value in values.items():
                    name = base + "." + suffix
                    self.payload[name] = value
                    dtype = {torch.uint8: "U8", torch.float32: "F32", torch.float8_e4m3fn: "F8_E4M3"}[value.dtype]
                    self.info[name] = (dtype, list(value.shape))

    def read(self, name, device="cpu"):
        if name in self.reads:
            raise AssertionError("duplicate checkpoint consumption: " + name)
        self.reads.append(name)
        return self.payload[name].to(device)


def ct_quant_reference(x, raw_multiplier):
    """Independent CPU FP32/per16 reference consuming original CT globals.

    This deliberately does not call checkpoint/calibration/reciprocal helpers.
    E4M3 conversion uses Torch's CPU encoding; E2M1 ties use explicit decision
    intervals rather than a native quantizer or table implementation.
    """
    if x.device.type != "cpu":
        raise ValueError("CT quantizer reference requires CPU input")
    m, k = x.shape
    blocks = x.float().view(m, k // 16, 16)
    raw = torch.tensor(raw_multiplier, dtype=torch.float32)
    local = (raw * (blocks.abs().amax(-1) * torch.tensor(1.0 / 6.0, dtype=torch.float32))).clamp(max=448.)
    encoded = local.to(torch.float8_e4m3fn)
    local = encoded.float()
    multiplier = torch.where(local != 0, raw / local, torch.zeros_like(local))
    v = blocks * multiplier[..., None]
    absolute = v.abs()
    code = torch.zeros(v.shape, dtype=torch.uint8)
    for next_code, (edge, exclusive) in enumerate(((.25, True), (.75, False), (1.25, True), (1.75, False),
                                                  (2.5, True), (3.5, False), (5., True)), start=1):
        code = torch.where(absolute > edge if exclusive else absolute >= edge, next_code, code)
    code = torch.where((v < 0) & (code != 0), code | 8, code).view(m, k)
    return (code[:, ::2] | (code[:, 1::2] << 4)), encoded.view(torch.uint8)


def linears(ctx, expert):
    from tensorfold.cuda.nvfp4.linear import Fp4Linear

    result = []
    for projection in ("gate_proj", "up_proj", "down_proj"):
        base = f"{PREFIX}.experts.{expert}.{projection}"
        fields = ctx.payload
        result.append(Fp4Linear.from_checkpoint(fields[base + ".weight_packed"].cuda(),
            fields[base + ".weight_scale"].cuda(), 1. / float(fields[base + ".weight_global_scale"][0]),
            act=1. / float(fields[base + ".input_global_scale"][0]),
            act_inverse=float(fields[base + ".input_global_scale"][0])))
    return result


def individual_reference(ctx, x, picks, *, prefill):
    """Original CPU checkpoint fields packed by the separate CUDA implementation.

    Each row/slot executes alone, with its original expert's three independent
    globals and two separate BF16 operations before down quantization.
    """
    from tensorfold.cuda.nvfp4 import checkpoint

    outputs = torch.zeros((*picks.shape, ctx.cfg.hidden), dtype=torch.bfloat16 if prefill else torch.float32,
                          device=x.device)
    native = checkpoint.prompt if prefill else checkpoint.matmul
    lins = [linears(ctx, e) for e in range(ctx.cfg.experts)]
    for row, slots in enumerate(picks.cpu().tolist()):
        for slot, expert in enumerate(slots):
            if not 0 <= expert < ctx.cfg.experts:
                continue
            gate, up, down = lins[expert]
            g = native(checkpoint.A4, x[row:row + 1], gate)
            u = native(checkpoint.A4, x[row:row + 1], up)
            silu = torch.nn.functional.silu(g)
            hidden = silu * u
            outputs[row, slot] = native(checkpoint.A4, hidden, down, f32=not prefill)[0]
    return outputs

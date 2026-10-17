"""Tiny CT experts with deliberately unequal per-projection calibrations."""

from types import SimpleNamespace

import torch

from tensorfold.cuda.nvfp4.format import LinearMetadata

PREFIX = "model.language_model.layers.0.mlp"


def swiglu(gate, up):
    return torch.nn.functional.silu(gate) * up


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

"""Fixed-row BF16 projections for the E2B verifier's canonical arithmetic.

The existing raw-BF16 router projection is a general matrix product with fixed
K partitions and ordered FP32 slice sums. Borrowing it retains checkpoint
operands, eliminates cuBLAS row-count dispatch, and rounds each result once to
the original BF16 linear output. Prompt arithmetic remains native HF, matching
TensorFold's established separate prompt/decode numerical contract.
"""
from __future__ import annotations

import torch
from torch import nn


class E2BRowPolicy:
    """One target lane exclusively sets this mode for the whole model forward."""

    def __init__(self, projection="native"):
        if projection not in ("native", "padded", "route"):
            raise ValueError("Gemma E2B canonical projection must be native, padded or fixed-route")
        self.projection = projection
        self.prompt = True
        self.position = 0


class E2BRowLinear(nn.Module):
    def __init__(self, native: nn.Linear, policy: E2BRowPolicy):
        super().__init__()
        if (native.weight.dtype != torch.bfloat16 or native.weight.ndim != 2
                or not native.weight.is_contiguous() or native.in_features % 32
                or native.bias is not None):
            raise ValueError("Gemma E2B fixed-row projection requires contiguous BF16 bias-free weights and K/32")
        self.weight = native.weight
        self.in_features, self.out_features = native.in_features, native.out_features
        self.policy = policy

    def forward(self, value):
        if self.policy.prompt:
            return torch.nn.functional.linear(value, self.weight)
        from tensorfold.cuda.moe import route_logits

        shape = value.shape
        rows = value.reshape(-1, self.in_features)
        if rows.shape[0] > 8:
            raise ValueError("Gemma E2B decode projection permits at most eight verifier rows")
        if self.policy.projection == "padded":
            padded = value.new_zeros((8, self.in_features))
            padded[:rows.shape[0]].copy_(rows)
            out = torch.nn.functional.linear(padded, self.weight)[:rows.shape[0]]
            return out.reshape(*shape[:-1], self.out_features)
        out = torch.empty((rows.shape[0], self.out_features), dtype=torch.float32, device=value.device)
        route_logits(rows, self.weight, out)
        return out.to(torch.bfloat16).reshape(*shape[:-1], self.out_features)


def install_row_linears(module: nn.Module, policy: E2BRowPolicy) -> None:
    """Replace only loaded decoder/head projections, retaining parameter owners."""
    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, E2BRowLinear(child, policy))
        else:
            install_row_linears(child, policy)

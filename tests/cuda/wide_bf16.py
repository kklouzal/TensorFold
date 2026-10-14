"""Adversarial bf16 inputs shared by the CUDA kernel tests."""

import torch


def wide(shape, lo: int, hi: int, seed: int) -> torch.Tensor:
    """bf16 values over a wide exponent range, both signs, with zeros and negative zeros."""

    g = torch.Generator(device="cuda").manual_seed(seed)
    mant = torch.rand(shape, generator=g, device="cuda") + 1.0
    exp = torch.randint(lo, hi + 1, shape, generator=g, device="cuda").float()
    sign = torch.where(torch.rand(shape, generator=g, device="cuda") < 0.5, -1.0, 1.0)
    v = torch.where(torch.rand(shape, generator=g, device="cuda") < 0.03, sign * 0.0, sign * mant * torch.exp2(exp))
    return v.to(torch.bfloat16)

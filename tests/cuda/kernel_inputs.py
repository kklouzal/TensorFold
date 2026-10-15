"""Random and adversarial inputs shared by the CUDA kernel tests."""

import numpy as np
import torch


def wide(shape, lo: int, hi: int, seed: int) -> torch.Tensor:
    """bf16 values over a wide exponent range, both signs, with zeros and negative zeros."""

    g = torch.Generator(device="cuda").manual_seed(seed)
    mant = torch.rand(shape, generator=g, device="cuda") + 1.0
    exp = torch.randint(lo, hi + 1, shape, generator=g, device="cuda").float()
    sign = torch.where(torch.rand(shape, generator=g, device="cuda") < 0.5, -1.0, 1.0)
    v = torch.where(torch.rand(shape, generator=g, device="cuda") < 0.03, sign * 0.0, sign * mant * torch.exp2(exp))
    return v.to(torch.bfloat16)


def gdn_inputs(rows: int, seed: int):
    """Flash Next GatedDeltaNet step inputs: projections, conv state and weights, state, A_log, dt, norm weight."""

    from tensorfold.families.qwen4_exp.cuda import gdn

    g = torch.Generator(device="cuda").manual_seed(seed)
    p = (torch.randn((rows, gdn.PW), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    cs = (torch.randn((3, gdn.CONV), generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    cw = (torch.randn((gdn.CONV, 4), generator=g, device="cuda") * 0.3).to(torch.bfloat16)
    state = torch.randn((gdn.NV, gdn.DV, gdn.DK), generator=g, device="cuda") * 0.05
    a_log = torch.randn((gdn.NV,), generator=g, device="cuda") * 0.5
    dt = torch.randn((gdn.NV,), generator=g, device="cuda") * 0.5
    nw = (1 + 0.1 * torch.randn((gdn.DV,), generator=g, device="cuda")).to(torch.bfloat16)
    return p, cs, cw, state, a_log, dt, nw


def fp4_weight(n: int, k: int, seed: int):
    """A random NVFP4 checkpoint weight [n, k]: packed E2M1 codes, e4m3 block scales and a tensor scale."""

    rng = np.random.default_rng(seed)
    packed = rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)
    scale = rng.integers(0x20, 0x50, size=(n, k // 16), dtype=np.uint8)           # e4m3 0.03-4
    return packed, scale, 0.0123

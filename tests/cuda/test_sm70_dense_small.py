"""SM70 small-M dense projection (``dense_small``) numerics vs FP64 on the served TP8 shapes.

Bound per element (any summation order, split-K included): FP16 x FP16
products are exact, and V100 tensor cores add four products to the FP32
accumulator with truncated alignment and normalization (Fasi et al., "Numerical
behavior of NVIDIA tensor cores", PeerJ CS 2021): at most five truncations of
2^-23 relative to the step's largest magnitude, i.e. under 1.25 * K * 2^-23 *
sum|x||w| over K terms. The test uses gamma_K with u = 2^-22 (2 * K * 2^-23),
which also covers the IEEE FP32 split-K reduction. The final rounding to the
output dtype adds its unit roundoff times |result| plus half its smallest
subnormal spacing. A violation means a lost or duplicated partial, a wrong
row/column or a nondeterministic reduction, not a tolerance miss.
"""

import pytest
import torch

from tensorfold.cuda.sm70 import dense_small

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="needs an SM70 GPU")

# (name, N, K, FP32 output): Qwen4-exp TP8 projections served through PaddedLinears.
SHAPES = [("hc_down", 324, 10240, True), ("hc_up", 10240, 320, False), ("attention_in", 2688, 2560, False),
          ("gdn_in", 2060, 2560, False), ("attention_out", 2560, 768, True), ("ple", 10240, 2560, False),
          ("head", 31040, 2560, True)]
ROWS = [1, 2, 5, 16, 17, 32]
SENTINEL = 7.0


@pytest.fixture(scope="module")
def workspace():
    return dense_small.Workspace(torch.device("cuda"))


@pytest.mark.parametrize("name,n,k,fp32", SHAPES, ids=[shape[0] for shape in SHAPES])
def test_logical_rows_are_within_the_fp32_accumulation_bound_and_deterministic(workspace, name, n, k, fp32):
    generator = torch.Generator(device="cuda").manual_seed(n * 131 + k)
    weight = (torch.randn(n, k, device="cuda", generator=generator) * k ** -0.5).half()
    dtype = torch.float32 if fp32 else torch.float16
    u_out, tiny = (2.0 ** -24, 2.0 ** -150) if fp32 else (2.0 ** -11, 2.0 ** -25)
    gamma = k * 2.0 ** -22 / (1 - k * 2.0 ** -22)
    exact, absolute = weight.double().T, weight.double().abs().T
    panel = torch.zeros(32, k, dtype=torch.float16, device="cuda")
    for rows in ROWS:
        panel.zero_()
        panel[:rows].normal_(generator=generator)
        x = panel[:rows]
        out = torch.full((32, n), SENTINEL, dtype=dtype, device="cuda")
        dense_small.matmul(x, weight, out[:rows], workspace)
        first = out.clone()
        dense_small.matmul(x, weight, out[:rows], workspace)
        torch.cuda.synchronize()
        assert torch.equal(first, out), f"{name} rows={rows}: nondeterministic"
        assert torch.all(out[rows:] == SENTINEL), f"{name} rows={rows}: padding rows were written"
        reference = x.double() @ exact
        magnitude = x.double().abs() @ absolute
        bound = gamma * magnitude * (1 + u_out) + u_out * reference.abs() + tiny
        error = (out[:rows].double() - reference).abs()
        assert torch.isfinite(out[:rows]).all() and torch.all(error <= bound), (
            f"{name} rows={rows}: max excess {(error - bound).max().item():.3e}")

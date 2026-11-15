"""SM70 decode shared-expert FFN (cuda/sm70/shared_ffn) against FP64 and the padded-M32 cuBLAS program.

Same operation boundaries as the reference (FP32 sums, FP32 SiLU and product, one FP16 rounding of the
activation, FP32 output); only the FP32 summation order differs. Checked: the error versus an FP64 evaluation
stays within the reference program's own error (plus one FP32 summation-order allowance), and a row's output
does not depend on the call's row count.
"""

import pytest
import torch

from tensorfold.cuda.sm70 import shared_ffn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="needs an SM70 GPU")


def padded(x, weight):
    """The reference program: zero-padded M32 panels, FP32 output."""
    outs = []
    for first in range(0, x.shape[0], 32):
        count = min(32, x.shape[0] - first)
        panel = torch.zeros((32, weight.shape[1]), dtype=torch.float16, device=x.device)
        panel[:count].copy_(x[first:first + count])
        outs.append(torch.mm(panel, weight.T, out_dtype=torch.float32)[:count])
    return torch.cat(outs)


@pytest.mark.parametrize("width", [80, 128, 256])
@pytest.mark.parametrize("rows", [1, 2, 5, 16, 32, 40, 64])
@pytest.mark.parametrize("case", ["normal", "large", "small", "outlier"])
def test_error_matches_the_reference_program_and_rows_are_independent(width, rows, case):
    generator = torch.Generator(device="cuda").manual_seed(rows * 7 + width + len(case))
    scale = {"normal": 1.0, "large": 8.0, "small": 0.01, "outlier": 1.0}[case]
    x = torch.randn(rows, 2560, device="cuda", generator=generator) * scale
    if case == "outlier":
        x[:, ::97] *= 50
    x = x.half()
    gate = (torch.randn(width, 2560, device="cuda", generator=generator) * 0.02).half()
    up = (torch.randn(width, 2560, device="cuda", generator=generator) * 0.02).half()
    down = (torch.randn(2560, width, device="cuda", generator=generator) * 0.03).half()
    assert shared_ffn.selected(x, gate, up, down)
    out = shared_ffn.run(x, gate, up, down)
    act = (torch.nn.functional.silu(padded(x, gate)) * padded(x, up)).half()
    reference = padded(act, down)
    g, u = x.double() @ gate.double().T, x.double() @ up.double().T
    exact = (g / (1 + torch.exp(-g)) * u) @ down.double().T
    magnitude = exact.abs().max()
    new_error = (out.double() - exact).abs().max() / magnitude
    reference_error = (reference.double() - exact).abs().max() / magnitude
    assert torch.isfinite(out).all()
    assert new_error <= reference_error * 1.05 + 2.0 ** -20, (new_error.item(), reference_error.item())
    last = shared_ffn.run(x[-1:].contiguous(), gate, up, down)
    assert torch.equal(last.view(torch.int32), out[-1:].view(torch.int32)), "row result depends on the row count"

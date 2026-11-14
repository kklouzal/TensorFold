"""SM70 multi-row serial router logits: every row equals the per-row FP32 GEMV reference bit for bit.

The native kernel serves up to eight rows per CTA with one register copy of the FP32 router weights; each row
keeps the serial operation sequence, so results must equal the separate per-row ``x.float() @ w.T`` calls the
decode path otherwise issues, and must not depend on how many rows share the call.
"""

import pytest
import torch

from tensorfold.cuda.sm70 import routing_native

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="needs an SM70 GPU")


def reference(x, weight):
    value = x.float()
    out = torch.empty(x.shape[0], 513, device=x.device)
    for row in range(x.shape[0]):
        out[row:row + 1].copy_(value[row:row + 1] @ weight.T)
    return out


@pytest.mark.parametrize("rows", [1, 2, 3, 5, 7, 8, 9, 16, 17, 32])
@pytest.mark.parametrize("case", ["normal", "large", "small", "zeros_outlier"])
def test_prepared_rows_match_the_serial_reference_bitwise(rows, case):
    generator = torch.Generator(device="cuda").manual_seed(rows * 31 + len(case))
    weight = torch.randn(513, 2560, device="cuda", generator=generator) * 0.02
    scale = {"normal": 1.0, "large": 30.0, "small": 1e-3, "zeros_outlier": 1.0}[case]
    x = (torch.randn(rows, 2560, device="cuda", generator=generator) * scale).half()
    if case == "zeros_outlier":
        x[:, ::7] = 0
        x[0, 5] = 60000.0
    out = torch.empty(rows, 513, device="cuda")
    routing_native.extension().prepared_logits(x, weight, out)
    assert torch.equal(out.view(torch.int32), reference(x, weight).view(torch.int32))
    last = torch.empty(1, 513, device="cuda")
    routing_native.extension().prepared_logits(x[-1:].contiguous(), weight, last)
    assert torch.equal(last.view(torch.int32), out[-1:].view(torch.int32)), "row result depends on the row count"

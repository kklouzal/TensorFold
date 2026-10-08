"""Every BF16 encoding survives the native oracle's actual operand layouts."""
from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tests.kv_pair_native_legacy_oracle import operand_layout_bit_probe  # noqa: E402


@pytest.mark.parametrize("role,rows,columns", [(0, 16, 256), (1, 64, 256), (2, 64, 256), (3, 16, 64)],
                         ids=["query", "key-transpose", "value", "probability"])
def test_native_operand_layout_preserves_all_65536_bf16_encodings(role, rows, columns):
    source = torch.arange(1 << 16, device="cuda", dtype=torch.int32).to(torch.int16).view(torch.uint16)
    assert source.view(torch.int16).unique().numel() == 1 << 16
    output = torch.empty_like(source)
    operand_layout_bit_probe[((source.numel() + rows * columns - 1) // (rows * columns),)](
        source, output, source.numel(), role, num_warps=4, num_stages=1, enable_fp_fusion=False
    )
    torch.cuda.synchronize()
    assert torch.equal(output.view(torch.int16), source.view(torch.int16))

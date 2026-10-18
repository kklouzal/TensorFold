"""Invalid native item spans report bounded host metadata without launching."""

import pytest
import torch
from modelopt_a4_inputs import PREFIX, Reader

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("native activation FP4 requires SM12", allow_module_level=True)

from tensorfold.cuda.nvfp4.grouped_a4 import _ext
from tensorfold.cuda.nvfp4.modelopt_experts import load_modelopt_a4


def test_native_item_refusal_keeps_shape_context_and_does_not_write_output():
    reader = Reader()
    bank = load_modelopt_a4(reader, PREFIX, reader.cfg, device="cuda:0")
    codes = torch.zeros((2, 1, 64), dtype=torch.uint8, device="cuda")
    scales = torch.zeros((2, 1, 8), dtype=torch.uint8, device="cuda")
    items = torch.zeros((2, 3), dtype=torch.int32, device="cuda")
    pairs = torch.zeros((1, 3), dtype=torch.int32, device="cuda")
    output = torch.full((2, 1, 64), 37, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="bounded plan/projection geometry: pairs=1 items=2 cells=3 projections=2 down=0"):
        _ext().gemm(codes, scales, bank.fields[0], bank.fields[2], bank.fields[4], items, pairs,
                    output, 64, 128, 1, False, True)
    assert torch.equal(output, torch.full_like(output, 37))
    torch.cuda.synchronize()

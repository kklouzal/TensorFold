"""The prepared BF16 vocabulary head retains singleton bits at its actual shape."""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
    pytest.skip("prepared Qwen vocabulary head is measured on SM120", allow_module_level=True)

# CUDA availability must be checked before importing the native backend.
from tensorfold.families.qwen3_5.cuda.b16 import matmul
from tensorfold.families.qwen3_5.cuda.weights import Plain
from tensorfold.families.qwen3_5_moe.cuda.head import prepare_head


def test_actual_vocabulary_shape_and_proposal_tails_keep_singleton_bits():
    generator = torch.Generator(device="cuda").manual_seed(17032)
    weight = torch.empty((248320, 2048), dtype=torch.bfloat16, device="cuda").uniform_(
        -.02, .02, generator=generator)
    x = torch.empty((17, 2048), dtype=weight.dtype, device=weight.device).uniform_(
        -.5, .5, generator=generator)
    head = prepare_head(Plain(weight))
    assert type(head) is not Plain and head.weight is weight
    reference = torch.cat([matmul(row[None], weight) for row in x])
    for rows in (1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 17):
        assert torch.equal(head(x[:rows]).view(torch.int16), reference[:rows].view(torch.int16)), rows
    for rows in (3, 7, 15):
        order = torch.arange(rows - 1, -1, -1, device="cuda")
        assert torch.equal(head(x[order]).view(torch.int16), reference[order].view(torch.int16)), rows

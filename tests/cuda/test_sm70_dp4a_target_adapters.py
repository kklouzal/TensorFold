"""New-profile containment/selected-shared adapters; no model/native24 proof."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70.dp4a_experts import extension
from tensorfold.families.qwen4_exp.cuda.dp4a_target import Role
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFExperts
from test_sm70_dp4a_profile_cuda import bank, exact

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0), reason="actualSM70targetadapter"
)


def math_contract():
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction is True
    assert torch.backends.cuda.matmul.allow_fp16_accumulation is False


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows", [1, 8, 32])
@pytest.mark.parametrize("kinds,offset", [((18, 18), 0), ((22, 22), 16), ((18, 22), 0), ((22, 18), 16)])
def test_selected_shared_only_is_bitwise_for_overlapping_output_layouts(monkeypatch, dtype, rows, kinds, offset):
    math_contract()
    monkeypatch.setattr(GGUFExperts, "implementation", "packed_blocks2w")
    original, _, _ = bank(*kinds, offset)
    torch.manual_seed(815 + rows)
    x = torch.randn((rows, 2560), dtype=torch.float16, device="cuda") * .03
    # Shared in multiple positions makes the temporary firstR output slots
    # overlap future destinations across rows, exercising the staged copy.
    picks = torch.tensor([[512, 7, 0, 511, 512, 18, 88, 4, 91, 1, 512]] * rows,
                         dtype=torch.int32, device="cuda")
    reference = torch.empty((rows, 11, 2560), dtype=dtype, device="cuda")
    original.forward(x, picks, reference)
    out = torch.full_like(reference, float("nan"))
    role = Role(NS(weights=NS(device="cuda"), account_metadata=lambda: None, check_alive=lambda: None), "main")
    role.bind_output(out)
    module = extension()
    role._shared_only(original, x, picks, out, module)
    exact(out[picks == 512], reference[picks == 512])
    # The enrolled down entry must not replace any selected shared bits.
    role.qh.zero_()
    module.project_routed(role.qh[:rows * 11], picks, original.down.data, 20, True, out, role.status)
    exact(out[picks == 512], reference[picks == 512])
    exact(out[picks != 512], torch.zeros_like(out[picks != 512]))
    assert role.status.cpu().tolist() == [0]
    alias = out.view(torch.int32).reshape(-1)[:rows * 11].view(rows, 11)
    with pytest.raises(RuntimeError, match="alias picks"):
        module.place_shared(out, alias)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fatal_containment_keeps_finite_bits_and_latches_without_reset(dtype):
    math_contract()
    module = extension()
    x = torch.zeros((1, 2560), dtype=torch.float16, device="cuda")
    x[0, :5] = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0, 1.25], device="cuda")
    status = torch.zeros(1, dtype=torch.int32, device="cuda")
    module.contain(x, status, 1)
    exact(x[0, :5], torch.tensor([0., 0., 0., -0.0, 1.25], device="cuda", dtype=torch.float16))
    output = torch.zeros((1, 11, 2560), dtype=dtype, device="cuda")
    output[0, 0, 0] = float("inf")
    output[0, 0, 1] = -0.0
    module.contain(output, status, 16)
    assert status.cpu().tolist() == [17] and torch.isfinite(output).all()
    exact(output[0, 0, :2], torch.tensor([0., -0.0], dtype=dtype, device="cuda"))
    module.contain(x, status, 1)
    assert status.cpu().tolist() == [17]
    terminal = torch.zeros((1, 10240), dtype=torch.float16, device="cuda")
    terminal[0, -1] = float("nan")
    module.contain(terminal, status, 32)
    assert status.cpu().tolist() == [49] and torch.isfinite(terminal).all()
    with pytest.raises(RuntimeError, match="inference-only"):
        module.contain(x.requires_grad_(True), status, 1)

"""Actual SM70 operator gates; exact dyadic oracles, no inferred numerical tolerance."""

import numpy as np
import pytest
import torch

from tensorfold.cuda.sm70.nvfp4_layout import EXPERT_DEVICE, LINEAR_REFERENCE, LINEAR_WMMA
from tensorfold.families.qwen4_exp.cuda.sm70_nvfp4 import ExpertBank, FloatLinear, PackedLinear, Workspace

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="requires actual SM70 V100; compilation/CPU controls are separate")


@pytest.mark.parametrize("family", [LINEAR_REFERENCE, LINEAR_WMMA])
@pytest.mark.parametrize("rows,n,k", [(1, 80, 80), (31, 64, 256), (32, 65, 512), (33, 80, 512)])
def test_dyadic_weight_and_row_products_match_independent_full_oracle(family, rows, n, k):
    words = np.arange(n * k // 2, dtype=np.uint8).reshape(n, k // 2)
    levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
    codes = np.empty((n, k), dtype=np.uint8)
    codes[:, ::2], codes[:, 1::2] = words & 15, words >> 4
    weight = (levels[codes] * np.float32(.125)).astype(np.float16)
    values = np.resize(np.array([-.125, .0625, .125, -.0625], dtype=np.float16), (rows, k))
    # All products/sums are exact dyadics within FP32 mantissa range, so this
    # independent FP64 oracle requires exact FP32 outputs without a tolerance.
    expected = (values.astype(np.float64) @ weight.astype(np.float64).T).astype(np.float32)
    workspace = Workspace("cuda", kernel_family=family)
    linear = PackedLinear(torch.from_numpy(words.copy()).cuda(), torch.full((n, k // 16), 0x38,
                          dtype=torch.uint8, device="cuda"), torch.tensor([.125], device="cuda"), workspace)
    actual = linear.matmul(torch.from_numpy(values).cuda(), f32=True).cpu().numpy()
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("rows", [1, 7, 33])
def test_device_expert_plan_is_stable_and_preserves_original_slots_with_mixed_weights(rows):
    import math

    dims, width, slots, experts = 32, 16, 3, 3
    workspace = Workspace("cuda", width=width, dims=dims, selection_rows=rows, slots=slots, experts=experts,
                          kernel_family=LINEAR_WMMA, expert_family=EXPERT_DEVICE)
    def packed(n, k, factor):
        return PackedLinear(torch.full((n, k // 2), 0x22, dtype=torch.uint8, device="cuda"),
            torch.full((n, k // 16), 0x38, dtype=torch.uint8, device="cuda"),
            torch.tensor([factor], dtype=torch.float32, device="cuda"), workspace)
    groups = [(packed(width, dims, f), packed(width, dims, f), packed(dims, width, f)) for f in (.125, .25)]
    groups.append((FloatLinear(torch.full((width, dims), .0625, dtype=torch.float16, device="cuda")),
                   FloatLinear(torch.full((width, dims), .125, dtype=torch.float16, device="cuda")),
                   FloatLinear(torch.full((dims, width), .5, dtype=torch.float16, device="cuda"))))
    bank = ExpertBank(groups, workspace, width=width, dims=dims)
    selected = np.resize(np.array([2, 0, 1, 0, 2, 0, 1], dtype=np.int32), (rows, slots))
    values = np.resize(np.array([.25, .5], dtype=np.float16), rows)
    actual = bank.forward(torch.from_numpy(np.repeat(values[:, None], dims, axis=1)).cuda(),
                          torch.from_numpy(selected).cuda()).cpu().numpy()
    expected = np.empty((rows, slots, dims), dtype=np.float32)
    for row in range(rows):
        for slot in range(slots):
            expert = int(selected[row, slot])
            gweight, uweight, down = ((.125, .125, .125), (.25, .25, .25), (.0625, .125, .5))[expert]
            gate = float(values[row]) * dims * gweight
            upward = float(values[row]) * dims * uweight
            # These chosen scalar results sit well inside their FP16 rounding
            # cells; the final down sums are exact dyadics, so no tolerance is
            # inferred from the candidate or from a different GPU schedule.
            activation = np.float16((gate / (1 + math.exp(-gate))) * upward)
            expected[row, slot].fill(np.float32(float(activation) * width * down))
    assert np.array_equal(actual, expected)
    counts, starts = workspace.counts_device.cpu().numpy(), workspace.starts_device.cpu().numpy()
    members = workspace.members_device.cpu().numpy()
    flat = selected.reshape(-1)
    for expert in range(experts):
        truth = np.flatnonzero(flat == expert)
        assert counts[expert] == len(truth)
        assert np.array_equal(members[starts[expert]:starts[expert + 1]], truth)

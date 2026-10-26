"""The selected SM70 QSA order against independent scalar rank decisions."""

import math
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.sm70 import family_ops
from tensorfold.families.qwen4_exp.qsa_selection import POLICY


@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("pattern", ["zero", "cutoff"])
def test_actual_target_mtp_selector_is_stable_across_batches_restore_and_logical_reassembly(rows, pattern):
    blocks, ratio, top, dim = 600, 4, 512, 128
    query = torch.zeros((rows, 4, dim), dtype=torch.float16)
    query[:, :, 0] = 1
    pooled = torch.zeros((blocks, dim), dtype=torch.float16)
    pooled[::2, 0] = -0.0
    if pattern == "cutoff":
        pooled[:, 0] = torch.arange(blocks).remainder(5) - 2

    def run(iq, history, start):
        count = iq.shape[0]
        scratch = SimpleNamespace(ratio=ratio, budget=ratio * top,
                                  sparse=torch.empty(count, dtype=torch.int32),
                                  nk=torch.empty(count, dtype=torch.int32),
                                  ids=torch.full((count, ratio * (top + 1)), -1, dtype=torch.int32),
                                  scores=torch.empty((count, blocks), dtype=torch.float32))
        family_ops.qsa_rows(iq, history, torch.tensor([start], dtype=torch.int32), scratch, count,
                            context=blocks * ratio)
        return scratch

    start = blocks * ratio - rows - 2
    batched = run(query, pooled, start)
    # A restored cache and a physically permuted owner copy reconstitute the
    # same logical pooled table before the canonical selection boundary.
    physical_order = torch.randperm(blocks, generator=torch.Generator().manual_seed(32))
    reassembled = pooled.index_select(0, physical_order).index_select(0, physical_order.argsort())
    for history in (pooled.clone(), reassembled):
        restored = run(query, history, start)
        assert torch.equal(restored.ids, batched.ids)
    for row in range(rows):
        serial = run(query[row:row + 1], pooled, start + row)
        end = start + row + 1
        complete = end // ratio
        scalar_scores = [max(float(pooled[group, 0]), 0.0) * 4 / math.sqrt(dim) for group in range(complete)]
        chosen = sorted(sorted(range(complete), key=lambda group: (-scalar_scores[group], group))[:top])
        expected = [ratio * group + member for group in chosen for member in range(ratio)]
        expected += list(range(complete * ratio, end))
        count = int(batched.nk[row])
        assert batched.scores.dtype == torch.float32 and batched.sparse[row] == 1
        assert batched.ids[row, :count].tolist() == expected
        assert torch.equal(serial.ids[0, :count], batched.ids[row, :count])


def test_selection_policy_is_part_of_prefix_compatibility():
    from tensorfold.families.qwen4_exp.cuda.prefix_policy import identity

    assert POLICY in identity("optimized")
    assert POLICY in identity("reference")
    assert identity(None) is None

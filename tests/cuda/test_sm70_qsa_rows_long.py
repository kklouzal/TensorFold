"""Eager qsa_rows past 4096 pooled groups. Decode-shaped rows (qsa_rows_long): the captured round selector's score
and radix bodies; scores are bitwise the single-block kernel's per-group scores, the selection is the exact
lexicographic top 512 of those scores, and within the single-block envelope the long route equals native
qsa_rows bitwise. Prompt windows: the Torch reference scores with qsa_select512 instead of the stable sort, bitwise
the reference's ids, counts, flags and score slab."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tensorfold.cuda.sm70 import family_ops as ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='SM70 native selector')


@pytest.fixture
def native(monkeypatch):
    from tensorfold.cuda.sm70.family_native import extension

    return extension()


def scratch(rows, groups, budget=2048, ratio=4):
    return NS(ratio=ratio, budget=budget, ids=torch.full((rows, budget + ratio), -5, dtype=torch.int32, device='cuda'),
              nk=torch.full((rows,), -5, dtype=torch.int32, device='cuda'),
              sparse=torch.full((rows,), -5, dtype=torch.int32, device='cuda'),
              scores=torch.full((rows, groups), 3., dtype=torch.float32, device='cuda'))


def inputs(rows, groups, ties, seed):
    torch.manual_seed(seed)
    iq = torch.randn((rows, 4, 128), device='cuda').half()
    pooled = torch.randn((groups, 128), device='cuda').half()
    if ties:  # coarse values make many equal scores: ties resolve to the lower group ID
        iq = (iq * 2).round().half()
        pooled = (pooled * 1.5).round().clamp(-2, 2).half()
        pooled[1000:3000] = pooled[0]
    return iq, pooled


def window_scores(native, iq, pooled, complete):
    """Per-group scores from the single-block kernel, 4096 groups at a time (each group's score is independent)."""
    rows, out = iq.shape[0], []
    for first in range(0, complete, 4096):
        part = pooled[first:first + 4096]
        target = scratch(rows, 4096)
        pos = torch.tensor([4 * 4096 * 4], dtype=torch.int64, device='cuda')  # every window group complete
        native.qsa_rows(iq, part, pos, target.sparse, target.nk, target.ids, target.scores, rows, 4, 2048,
                        part.shape[0])
        out.append(target.scores[:, :part.shape[0]])
    return torch.cat(out, 1)


@pytest.mark.parametrize('rows,start,ties', [(256, 16_384, False), (37, 40_001, True), (256, 61_000, True),
                                             (1, 262_000, False), (200, 100_003, False)])
@torch.no_grad()
def test_long_route_scores_and_exact_selection(native, rows, start, ties):
    groups = 262_144 // 4
    iq, pooled = inputs(rows, groups, ties, rows + start)
    pos = torch.tensor([start], dtype=torch.int64, device='cuda')
    context = start + rows
    count = -(-context // 4)
    target = scratch(rows, groups)
    status = torch.tensor([0, -2], dtype=torch.int32, device='cuda')
    ops.qsa_rows(iq, pooled, pos, target, rows, context=context, status=status, layer=7)
    assert status.tolist() == [0, -2]
    for row in range(rows):
        end = start + row + 1
        complete = end // 4
        assert int(target.sparse[row]) == 1 and int(target.nk[row]) == 2048 + end % 4
    reference = window_scores(native, iq, pooled, count)
    scores = target.scores.cpu().numpy()
    expected_scores = reference.cpu().numpy()
    ids = target.ids.cpu().numpy()
    for row in range(rows):
        complete = (start + row + 1) // 4
        assert np.array_equal(scores[row, :complete].view(np.int32), expected_scores[row, :complete].view(np.int32))
        chosen = np.sort(np.lexsort((np.arange(complete), -scores[row, :complete].astype(np.float64)))[:512])
        assert np.array_equal(ids[row, :2048], (chosen[:, None] * 4 + np.arange(4)).reshape(-1))
        assert np.array_equal(ids[row, 2048:], complete * 4 + np.arange(4))


@pytest.mark.parametrize('rows,start', [(256, 4_000), (64, 15_000)])
@torch.no_grad()
def test_long_route_equals_single_block_kernel_inside_its_envelope(native, rows, start):
    iq, pooled = inputs(rows, 4096, True, start)
    pos = torch.tensor([start], dtype=torch.int64, device='cuda')
    count = -(-(start + rows) // 4)
    short, long = scratch(rows, 4096), scratch(rows, 4096)
    status = torch.tensor([0, -2], dtype=torch.int32, device='cuda')
    native.qsa_rows(iq, pooled, pos, short.sparse, short.nk, short.ids, short.scores, rows, 4, 2048, count)
    native.qsa_rows_long(iq, pooled, pos, long.sparse, long.nk, long.ids, long.scores, status, rows, 4, 2048, count, 7)
    assert status.tolist() == [0, -2]
    for name in ('ids', 'nk', 'sparse'):
        assert torch.equal(getattr(long, name), getattr(short, name)), name
    for row in range(rows):
        complete = (start + row + 1) // 4
        assert torch.equal(long.scores[row, :complete].view(torch.int32), short.scores[row, :complete].view(torch.int32))


@pytest.mark.parametrize('prompt,start', [(False, 20_000), (True, 20_000), (True, 3_000)])
@torch.no_grad()
def test_a_state_without_a_status_word_selects_with_the_torch_reference(native, monkeypatch, prompt, start):
    # Symmetric FP16 KV without unique-history placement has no kv_status (and no round graphs): the native routes
    # need it to latch a failure, so such a State keeps the Torch reference rather than refusing the frame.
    iq, pooled = inputs(4, 8192, False, 3)
    pos = torch.tensor([start], dtype=torch.int64, device='cuda')
    candidate, reference = scratch(4, 8192), scratch(4, 8192)
    ops.qsa_rows(iq, pooled, pos, candidate, 4, context=start + 4, prompt=prompt)
    monkeypatch.setattr(ops, '_selector', lambda tensor: None)
    ops.qsa_rows(iq, pooled, pos, reference, 4, context=start + 4)
    for name in ('ids', 'nk', 'sparse', 'scores'):
        assert torch.equal(getattr(candidate, name), getattr(reference, name)), name


@pytest.mark.parametrize('fault', ['nonfinite', 'slab'])
@torch.no_grad()
def test_failure_latches_structural_status_with_in_bounds_ids(native, fault):
    rows, start = 8, 20_000
    iq, pooled = inputs(rows, 32_768 // 4, False, 5)
    count = -(-(start + rows) // 4)
    if fault == 'nonfinite':
        pooled[17] = float('inf')  # an invariant breach upstream of the guarded inputs
        blocks = count
    else:
        blocks = count - 8  # a context past the score columns the caller funded
    target = scratch(rows, blocks)
    status = torch.tensor([0, -2], dtype=torch.int32, device='cuda')
    pos = torch.tensor([start], dtype=torch.int64, device='cuda')
    native.qsa_rows_long(iq, pooled, pos, target.sparse, target.nk, target.ids, target.scores, status, rows, 4, 2048,
                         blocks, 7)
    assert status.tolist() == [32, 7]
    assert int(target.ids.min()) >= 0 and int(target.ids.max()) < start + rows


@pytest.mark.parametrize('rows,start,ties', [(256, 16_384, True), (256, 61_000, False), (37, 200_001, True),
                                             (256, 1_900, True), (256, 3_000, False), (100, 9_000, True)])
@torch.no_grad()
def test_prompt_route_equals_the_torch_sort_reference(native, monkeypatch, rows, start, ties):
    groups = 262_144 // 4
    iq, pooled = inputs(rows, groups, ties, rows + start + 1)
    pos = torch.tensor([start], dtype=torch.int64, device='cuda')
    reference, candidate = scratch(rows, groups), scratch(rows, groups)
    status = torch.tensor([0, -2], dtype=torch.int32, device='cuda')
    ops.qsa_rows(iq, pooled, pos, candidate, rows, context=start + rows, status=status, layer=7, prompt=True)
    monkeypatch.setattr(ops, '_selector', lambda tensor: None)  # the Torch stable-sort reference
    ops.qsa_rows(iq, pooled, pos, reference, rows, context=start + rows)
    assert status.tolist() == [0, -2]
    for name in ('ids', 'nk', 'sparse', 'scores'):
        assert torch.equal(getattr(candidate, name), getattr(reference, name)), name


@torch.no_grad()
def test_prompt_route_latches_a_nonfinite_score(native):
    rows, start = 8, 20_000
    iq, pooled = inputs(rows, 32_768 // 4, False, 9)
    pooled[17] = float('inf')
    target = scratch(rows, 32_768 // 4)
    status = torch.tensor([0, -2], dtype=torch.int32, device='cuda')
    pos = torch.tensor([start], dtype=torch.int64, device='cuda')
    ops.qsa_rows(iq, pooled, pos, target, rows, context=start + rows, status=status, layer=7, prompt=True)
    assert status.tolist() == [32, 7]
    # Selected ids stay in bounds; the ratio tail ids past a row's end are masked by attention as always.
    assert int(target.ids.min()) >= 0 and int(target.ids[:, :2048].max()) < start + rows

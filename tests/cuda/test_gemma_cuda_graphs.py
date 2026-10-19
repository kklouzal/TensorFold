"""Gemma4 text forwards replayed as CUDA graphs equal the eager forward's bits across rows, grids and rollbacks."""

from __future__ import annotations

import pytest
import torch
from test_gemma_cuda_target import target

from tensorfold.families.gemma4.cuda.graphs import Graphs


def test_graph_rounds_equal_eager_rounds_with_rollback_and_growing_grids():
    model = target()
    tokens = [(5 * i + 1) % 128 for i in range(500)]
    eager, staged = model.make_state(1600, rows=128), model.make_state(1600, rows=128)
    for state in (eager, staged):
        for i in range(0, 500, 125):
            model.forward(tokens[i:i + 125], state, output_logits=False, output_taps=False)
    graphs = Graphs(model, staged, max_rows=5, span=1)
    generator = torch.Generator().manual_seed(4)
    for round_ in range(60):
        rows = 1 + round_ % 5
        step = torch.randint(0, 128, (rows,), generator=generator).tolist()
        taps = round_ % 3 != 0
        expected = model.forward(step, eager, output_taps=taps, trusted_tokens=True)
        got = graphs.forward(step, staged, output_taps=taps)
        assert got.start_position == expected.start_position and staged.position == eager.position
        assert torch.equal(got.hidden, expected.hidden) and torch.equal(got.logits, expected.logits)
        assert set(got.taps) == set(expected.taps)
        assert all(torch.equal(got.taps[layer], expected.taps[layer]) for layer in expected.taps)
        keep = expected.start_position + 1 + round_ % rows          # accept a prefix, as verification does
        eager.rollback(keep)
        staged.rollback(keep)
    # The positions crossed GLOBAL_CHUNK boundaries: several grids (and both tap settings) were captured.
    assert len({key[1] for key in graphs.graphs}) > 1 and {key[2] for key in graphs.graphs} == {True, False}
    assert len({key[0] for key in graphs.graphs}) == 5


def test_graphs_refuse_foreign_states_and_oversized_rounds():
    model = target()
    state, other = model.make_state(256, rows=8), model.make_state(256, rows=8)
    model.forward([1, 2, 3], state)
    graphs = Graphs(model, state, max_rows=4)
    with pytest.raises(ValueError, match="engine's state"):
        graphs.forward([4], other, output_taps=False)
    with pytest.raises(ValueError, match="max_rows"):
        graphs.forward([4] * 5, state, output_taps=False)
    assert state.position == 3 and not state.poisoned


@pytest.mark.parametrize("span,smallest,after", [(1, 1, 7), (16, 4, 6)])
def test_startup_prepares_the_context_clipped_smallest_grid_and_larger_grids_capture_on_first_use(
        span, smallest, after):
    model = target()
    eager, staged = model.make_state(1600, rows=128), model.make_state(1600, rows=128)
    graphs = Graphs(model, staged, max_rows=3, span=span)
    assert graphs.prepare() == 6
    assert staged.position == 0 and {key[1] for key in graphs.graphs} == {smallest}
    tokens = [(5 * i + 1) % 128 for i in range(600)]
    for state in (eager, staged):
        for start in range(0, len(tokens), 100):
            model.forward(tokens[start:start + 100], state, output_logits=False, output_taps=False)
    expected = model.forward([5, 6], eager, output_taps=True, trusted_tokens=True)
    result = graphs.forward([5, 6], staged, output_taps=True)
    assert len(graphs.graphs) == after and (2, max(2, smallest), True) in graphs.graphs
    assert torch.equal(result.logits, expected.logits) and torch.equal(result.hidden, expected.hidden)
    assert all(torch.equal(result.taps[layer], expected.taps[layer]) for layer in expected.taps)

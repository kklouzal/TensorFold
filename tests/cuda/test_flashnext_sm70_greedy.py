"""Single-V100 exact original packet, binding and dynamic graph oracles."""

import pytest
import torch

from greedy_inputs import inputs

from tensorfold.cuda.sm70 import greedy_native




def reference(values, ids, norm, status):
    maximum = values.max(-1).values
    selected = torch.where(values[0] == maximum, ids, torch.iinfo(torch.int64).max).min()
    return torch.cat((maximum.view(torch.int32).long(), selected.view(1), norm.view(torch.int32).long(), status.long()))


@pytest.mark.parametrize("width", [1, 31, 32, 33, 9948, 9949, 31040])
@pytest.mark.parametrize("case", ["random", "ties", "zeros", "subnormal", "adjacent", "minus_inf", "plus_inf", "nan"])
def test_local_matches_original_cuda_words(width, case):
    args = [value.cuda() for value in inputs(width, case)]
    output = torch.empty(5, dtype=torch.int64, device="cuda")
    greedy_native.extension().local(*args, output)
    torch.testing.assert_close(output, reference(*args), rtol=0, atol=0)


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("case", ["ties", "zeros", "subnormal", "minus_inf", "plus_inf", "nan"])
def test_global_matches_original_cuda_and_safe_staging(world, case):
    values, ids, norm, status = [value.cuda() for value in inputs(world, case)]
    frame = torch.stack([reference(values[:, rank:rank+1], ids[rank:rank+1], norm, status) for rank in range(world)])
    token = torch.empty(1, dtype=torch.int64, device="cuda")
    staged = torch.empty(1, dtype=torch.int32, device="cuda")
    greedy_native.extension().global_winner(frame, token, staged, 1000)
    scores = frame[:, 0].int().view(torch.float32)
    expected = torch.where(scores == scores.max(), frame[:, 1], torch.iinfo(torch.int64).max).min()
    torch.testing.assert_close(token[0], expected, rtol=0, atol=0)
    safe = torch.where((expected >= 0) & (expected < 1000), expected, 0).int()
    torch.testing.assert_close(staged[0], safe, rtol=0, atol=0)


def test_captured_dynamic_packet_replays_ids_scores_status_and_global_history():
    extension = greedy_native.extension()
    values, ids, norm, status = [value.cuda() for value in inputs(9949, "ties")]
    frame = torch.empty((2, 5), dtype=torch.int64, device="cuda")
    token = torch.empty(1, dtype=torch.int64, device="cuda")
    staged = torch.empty(1, dtype=torch.int32, device="cuda")

    def run():
        extension.local(values, ids, norm, status, frame[0])
        frame[1].copy_(frame[0])
        frame[1, 1].add_(10)
        extension.global_winner(frame, token, staged, 20000)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for case in ("random", "subnormal", "zeros", "nan", "plus_inf"):
        replacement = inputs(9949, case)
        for actual, new in zip((values, ids, norm, status), replacement):
            actual.copy_(new)
        ids.copy_(ids.flip(0))
        status[0].fill_(8)
        graph.replay()
        torch.testing.assert_close(frame[0], reference(values, ids, norm, status), rtol=0, atol=0)
        expected = frame[0, 1]
        torch.testing.assert_close(token[0], expected, rtol=0, atol=0)


def test_binding_refusals_precede_aliased_or_malformed_writes():
    extension = greedy_native.extension()
    values, ids, norm, status = [value.cuda() for value in inputs(32, "random")]
    output = torch.empty(5, dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError, match="overlaps"):
        extension.local(values, ids, norm, status, values.view(torch.int64).view(-1)[:5])
    with pytest.raises(RuntimeError, match="geometry"):
        extension.local(values, ids[:-1], norm, status, output)
    with pytest.raises(RuntimeError, match="declared storage"):
        extension.local(values.half(), ids, norm, status, output)
    frame = torch.zeros((2, 5), dtype=torch.int64, device="cuda")
    staged = torch.empty(1, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="overlaps"):
        extension.global_winner(frame, frame.view(-1)[:1], staged, 1000)

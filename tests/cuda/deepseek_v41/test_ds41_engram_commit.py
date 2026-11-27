"""Exact hash and history comparisons for the decode graph's single-slot commit."""
import pytest
import torch

from tensorfold.families.deepseek_v4.cuda.kernels.mu_state import (
    ERR_CONTEXT, ERR_CONTINUE, ERR_SLOT, ERR_TOKEN, MuEngramHashState,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("live,rows,trap", [(1, 1, True), (3, 4, False), (30, 32, True)])
def test_engram_single_commit_matches_split_replay(live, rows, trap):
    device = "cuda"
    rng = torch.Generator(device=device).manual_seed(832 + rows)
    vocab, slots, context = 129, rows + 1, 65536
    mapping = torch.randint(0, 38, (vocab,), dtype=torch.int32, generator=rng, device=device)
    prime = (257 + 2 * torch.arange(48, device=device)).view(2, 24)
    offset = (2000 * torch.arange(48, device=device)).view(2, 24)
    multiplier = torch.tensor([[3, 7, 2**60 + 9, -(2**60 + 3)], [17, -3, 2**59 - 1, 23]], device=device)
    reference, fused = (MuEngramHashState(mapping, prime, offset, multiplier, slots=slots,
                                            max_tokens=rows, max_context=context, trap=trap)
                        for _ in range(2))
    slots_by_row = torch.full((rows + 8,), slots, dtype=torch.int32, device=device)
    slots_by_row[:live] = torch.arange(live - 1, -1, -1, device=device, dtype=torch.int32)
    token_req = torch.arange(rows, dtype=torch.int32, device=device)
    qsl = torch.arange(rows + 1, dtype=torch.int32, device=device)
    ids = torch.full((rows,), 2, dtype=torch.long, device=device)
    positions = torch.zeros(rows, dtype=torch.long, device=device)
    start = 4090 + torch.arange(live, device=device) % 17
    start[0] = 0
    valid = torch.ones(rows, dtype=torch.bool, device=device)
    initial = torch.randint(-1, 38, fused.history.shape, generator=rng, device=device)

    def reset(state):
        state.history.copy_(initial)
        state.history[slots_by_row[:live].long(), 3] = start
        state.err.zero_()

    def run():
        return fused(ids, positions, valid, token_req, qsl, slots_by_row, commit_single=True)

    for state in (reference, fused):
        reset(state)
    positions[:live].copy_(start)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
        reset(fused)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = run()

    for step in range(7):
        positions[:live] = start + step
        ids[:live] = torch.randint(0, vocab, (live,), generator=rng, device=device)
        valid[1:live:3] = step % 2 == 0
        bad = step == 6 and not trap
        if bad:
            ids[0] = -5
            slots_by_row[1] = slots + 1
            positions[2] = context
            for state in (reference, fused):
                state.history[slots_by_row[0].long(), 3] -= 1
                state.history[slots_by_row[2].long(), 3] = context
        expected = reference(ids, positions, valid, token_req, qsl, slots_by_row)
        reference.commit_all(ids, positions, valid, qsl, slots_by_row, rows, token_req)
        graph.replay()
        assert torch.equal(actual, expected)
        assert torch.equal(fused.history, reference.history)
        assert torch.equal(fused.history[slots], initial[slots])
        errors = ERR_TOKEN | ERR_CONTINUE | ERR_CONTEXT | ERR_SLOT if bad else 0
        assert int(fused.err) == int(reference.err) == errors

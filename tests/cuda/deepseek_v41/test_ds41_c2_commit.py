"""Exact slot and output checks across replay of the one-row decode commit."""
import pytest
import torch

from tensorfold.families.deepseek_v4.cuda.kernels.mu_state import ERR_CONTINUE, MuC2State


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("live,rows,trap", [(1, 1, True), (3, 4, False), (30, 32, True)])
def test_single_slot_pool_and_commit_matches_sequential_replay(live, rows, trap):
    rng = torch.Generator(device="cuda").manual_seed(319 + rows)
    slots = rows + 3
    norm = torch.rand(512, generator=rng, device="cuda") + 0.5
    reference, fused = (MuC2State(norm, slots=slots, trap=trap) for _ in range(2))
    initial_partial = torch.randn(fused.partial.shape, generator=rng, device="cuda")
    req_slot = torch.full((rows + 8,), slots, dtype=torch.int32, device="cuda")
    live_slots = torch.arange(live, device="cuda") + 2
    req_slot[:live] = live_slots.flip(0).to(torch.int32)
    token_req = torch.arange(rows, dtype=torch.int32, device="cuda")
    qsl = torch.arange(rows + 1, dtype=torch.int32, device="cuda")
    positions = torch.zeros(rows, dtype=torch.int64, device="cuda")
    positions[:live] = 8191 - torch.arange(live, device="cuda") % 4
    start = positions.clone()
    kv = torch.randn(rows, 512, generator=rng, device="cuda")
    scores = torch.randn(rows, 512, generator=rng, device="cuda")

    def reset(state):
        state.partial.copy_(initial_partial)
        state.committed.zero_()
        state.committed[req_slot[:live].long()] = start[:live]
        state.err.zero_()

    def run_fused():
        output, complete = fused(kv, scores, positions, token_req, qsl, req_slot, commit_single=True)
        assert complete is None
        return output

    reset(reference)
    reset(fused)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run_fused()
        reset(fused)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = run_fused()

    for step in range(5):
        positions[:live] = start[:live] + step
        kv.neg_()
        scores.mul_(0.5)
        stale = not trap and step == 4
        if stale:
            # With trapping disabled an odd row with a stale position still
            # records ERR_CONTINUE, as in split pooling followed by commit.
            slot = req_slot[0].long()
            for state in (reference, fused):
                state.committed[slot] -= 1
        expected, _ = reference(kv, scores, positions, token_req, qsl, req_slot)
        reference.commit_all(kv, scores, positions, qsl, req_slot, rows, token_req)
        graph.replay()
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
        assert torch.equal(fused.partial.view(torch.int32), reference.partial.view(torch.int32))
        assert torch.equal(fused.committed, reference.committed)
        assert int(fused.err) == int(reference.err) == (ERR_CONTINUE if stale else 0)
        assert torch.equal(fused.partial[slots], initial_partial[slots])

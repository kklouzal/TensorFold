"""Real allocator accounting at an explicit Flash shared-cache growth cap."""

import pytest
import torch

from tensorfold.cuda.capacity import LIMIT_ENV, cuda_limit_bytes
from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda.multi import FIRST, STEP, MultiDecoder, _tensors
from test_flashnext_forward import _model


def _state_bytes(state):
    return [tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
            for tensor in _tensors(state)]


@pytest.mark.parametrize("keep_prefix", [False, True])
def test_absolute_growth_cap_refuses_before_allocation_and_preserves_following_request(monkeypatch, keep_prefix):
    monkeypatch.delenv(LIMIT_ENV, raising=False)
    decoder = MultiDecoder(_model(), slots=1, capacity=STEP, depth=2, confidence=0.0,
                           prefill_rows=64, stop_eos=False)
    prompt = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]

    def run():
        stream = Stream(prompt, 2, draft=keep_prefix, stop_eos=False)
        decoder.admit(stream)
        while not stream.done:
            decoder.finish(decoder.round())
        assert stream.error is None and decoder.live() == 0
        return stream

    try:
        # Warm the same request/shape before measuring live allocator bytes.
        control = run()
        state = control.st
        assert state.capacity == FIRST
        assert bool(decoder.kept) is keep_prefix
        owners = [(tuple(ids), id(st), id(snap), id(tail)) for ids, st, snap, tail in decoder.kept]
        free = [id(st) for st in decoder.free]
        saved, held, next_id = _state_bytes(state), decoder.memory_gate.held, decoder.next_id
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        # FIRST+1 required rows rounds up to the existing STEP policy. The
        # cap is exactly one byte below the declared copy peak, derived from
        # this allocator and cache format rather than an arbitrary GiB budget.
        growth = state.cache_bytes(STEP) - state.cache_bytes()
        cap = allocated + growth + state.layer_bytes() - 1
        monkeypatch.setenv(LIMIT_ENV, repr(cap / 1024**3))
        assert cuda_limit_bytes() == cap
        long_prompt = [*prompt, *([7] * (FIRST - decoder.depth - 1 - len(prompt)))]
        assert len(long_prompt) + decoder.depth + 2 == FIRST + 1
        refused = Stream(long_prompt, 2, draft=keep_prefix, stop_eos=False)
        torch.cuda.reset_peak_memory_stats()
        with pytest.raises(NoRoom, match=LIMIT_ENV):
            decoder.admit(refused)
        assert torch.cuda.memory_allocated() == allocated
        assert torch.cuda.max_memory_allocated() == allocated
        assert refused.st is None and decoder.live() == 0
        assert decoder.fills == {} and decoder.filling == [] and decoder.held == {}
        assert decoder.next_id == next_id and decoder.memory_gate.held == held
        assert [id(st) for st in decoder.free] == free
        assert [(tuple(ids), id(st), id(snap), id(tail)) for ids, st, snap, tail in decoder.kept] == owners
        assert _state_bytes(state) == saved

        following = run()
        assert following.st is state and following.out == control.out
        expected_cached = len(prompt) - 1 if keep_prefix else 0
        assert following.cached == expected_cached
        assert decoder.next_id == next_id + 1
        assert torch.cuda.max_memory_allocated() <= cap

        # The bound itself admits growth: refusal must not be an unconditional
        # explicit-cap policy. Recompute from the following request's live
        # ownership, then exercise the real layer-at-a-time allocator copy.
        copy_peak = torch.cuda.memory_allocated() + growth + state.layer_bytes()
        monkeypatch.setenv(LIMIT_ENV, repr(copy_peak / 1024**3))
        assert cuda_limit_bytes() == copy_peak
        torch.cuda.reset_peak_memory_stats()
        assert decoder._grow(state, FIRST + 1, alone=True)
        assert state.capacity == STEP
        assert torch.cuda.max_memory_allocated() <= copy_peak
    finally:
        decoder.drop()
        torch.cuda.synchronize()
        for buffer in (decoder.buf, decoder.mbuf):
            if buffer is not None and buffer.hc_plans is not None:
                buffer.hc_plans.close()

"""Real CUDA shared-window pressure has explicit outcomes and preserves successful tokens."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_tp import _checkpoint

from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen4_exp.cuda import multi
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine


@pytest.mark.parametrize("formats,sampling", [(("bf16", "bf16"), None),
                                             (("rotorquant8-norm", "rotorquant6-norm"),
                                              Sampling(seed=23, top_k=20, top_p=0.95))])
def test_one_shared_window_bounds_three_streams_and_preserves_retry(tmp_path, monkeypatch, formats, sampling):
    # Smaller steps exercise the same byte gate and resize paths without a
    # multi-thousand-token numerical fixture. No memory/failure check is mocked.
    monkeypatch.setattr(multi, "FIRST", 8)
    monkeypatch.setattr(multi, "STEP", 16)
    monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", "64")
    _checkpoint(tmp_path)
    engine = FlashNextEngine(tmp_path, streams=3, max_len=64, context_explicit=True,
                             depth=3, confidence=0.0, prefetch=False, graphs=False,
                             vram_experts="auto", kv_full_windows=1, draft_vocab=None,
                             kv_key_dtype=formats[0], kv_value_dtype=formats[1])
    try:
        # Join the idle scheduler before taking deterministic test ownership of
        # its decoder. The engine and expert authority remain alive until close.
        engine.scheduler.close()
        decoder = engine.multi
        assert not decoder.live() and not decoder.memory_gate.allow_lone_growth
        receipt = engine.capacity_plan["vram_experts_auto"]
        assert receipt["kv_full_windows_reserved"] == 1 and receipt["kv_stream_slots"] == 3
        assert receipt["reserved_context_growth_bytes"] * 3 == receipt["full_context_growth_bytes"]
        prompts = [[1 + (i * step) % 3000 for i in range(24)] for step in (7, 11, 13)]
        streams = [Stream(prompt, 40, sampling, stop_eos=False) for prompt in prompts]
        for stream in streams:
            decoder.admit(stream)
        assert len(decoder.filling) == 3
        rounds = 0
        while decoder.live():
            done = decoder.round()
            assert decoder.memory_gate.held <= decoder.memory_gate.room
            decoder.finish(done)
            rounds += 1
            assert rounds <= sum(s.count for s in streams) + len(streams)
        assert streams[0].error is None and len(streams[0].out) == streams[0].count
        assert any(s.error is not None for s in streams[1:])
        assert decoder.memory_gate.ends and decoder.memory_gate.waits
        oracle = Engine(engine.w, capacity=decoder.capacity, max_rows=8, graphs=False,
                        prefill_rows=64, kv_pair=decoder.free[0].kv_pair)
        for stream in streams:
            first = prefill(oracle, stream.prompt, stream.sampling)
            expected = serial_decode(oracle, first, stream.count, stream.sampling, stop_eos=False).tokens
            assert stream.out == expected[:len(stream.out)]
            if stream.error is None:
                assert stream.out == expected
            else:
                assert "stopped" in str(stream.error) and len(stream.out) < stream.count
        # A following request can reclaim the retained prefix and finish alone.
        retry = Stream(prompts[-1], streams[-1].count, sampling, stop_eos=False)
        decoder.admit(retry)
        while decoder.live():
            done = decoder.round()
            assert decoder.memory_gate.held <= decoder.memory_gate.room
            decoder.finish(done)
        expected = serial_decode(oracle, prefill(oracle, retry.prompt, sampling), retry.count,
                                 sampling, stop_eos=False).tokens
        assert retry.error is None and retry.out == expected
        assert len(decoder.free) + len({id(k[1]) for k in decoder.kept}) == 3
        # A partial native resize is retired before admission can reuse it.
        # Fail the MTP layer after the main attention layer was replaced.
        from tensorfold.families.qwen4_exp.cuda import state

        free = [id(st) for st in decoder.free]
        held = decoder.memory_gate.held
        rows, calls = state._rows, 0
        failure = torch.OutOfMemoryError("controlled partial CUDA cache growth failure")

        def failed_rows(array, count, keep):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise failure
            return rows(array, count, keep)

        with monkeypatch.context() as patch:
            patch.setattr(state, "_rows", failed_rows)
            with pytest.raises(torch.OutOfMemoryError) as caught:
                decoder.admit(Stream([3] * 9, 8, sampling, draft=False, stop_eos=False))
        assert caught.value is failure and not decoder.failed_slots
        assert [id(st) for st in decoder.free] == free and decoder.memory_gate.held == held
        following = Stream([3] * 9, 8, sampling, draft=False, stop_eos=False)
        decoder.admit(following)
        while decoder.live():
            decoder.finish(decoder.round())
        expected = serial_decode(oracle, prefill(oracle, following.prompt, sampling), following.count,
                                 sampling, stop_eos=False).tokens
        assert following.error is None and following.out == expected
    finally:
        engine.close()

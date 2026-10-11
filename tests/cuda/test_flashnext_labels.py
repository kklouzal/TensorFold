"""Native unary label scoring preserves retained prompts and request ownership."""

import numpy as np
import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill_begin, prefill_chunk
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
from tensorfold.families.qwen4_exp.cuda.multi import _tensors
from test_flashnext_tp import PROMPT, V, _checkpoint


def _saved(value):
    if isinstance(value, torch.Tensor):
        return (value.dtype, tuple(value.shape),
                value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    if isinstance(value, np.ndarray):
        return (str(value.dtype), value.shape, value.tobytes())
    if isinstance(value, dict):
        return {key: _saved(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_saved(item) for item in value)
    return value


def _retained(engine):
    if engine.multi is None:
        entries = [(tuple(ids), id(snap), _saved(snap)) for ids, snap in engine.cache]
        states = [engine.e.st]
    else:
        entries = [(tuple(ids), id(st), id(snap), _saved(snap), _saved(tail))
                   for ids, st, snap, tail in engine.multi.kept]
        states = [entry[1] for entry in engine.multi.kept]
    assert entries, "the control request must retain a real prompt state"
    # Include cache payloads as well as recurrent snapshots. Byte comparisons
    # preserve signed zero and do not interpret unused scratch storage.
    return entries, [(id(st), _saved(st.snapshot()), [_saved(t) for t in _tensors(st)]) for st in states]


@pytest.mark.parametrize("streams", [1, 2])
def test_public_score_labels_matches_target_head_and_preserves_prefixes(tmp_path, monkeypatch, streams):
    monkeypatch.setenv("TENSORFOLD_PREFILL_ROWS", "256")
    _checkpoint(tmp_path)
    engine = FlashNextEngine(tmp_path, depth=2, confidence=0.0, draft_vocab=None,
                             max_len=128, prefetch=False, graphs=False, streams=streams)
    reference = None
    try:
        def ask(prompt, *, draft):
            tokens = []
            stats = engine.generate(prompt, 4, None, lambda new: tokens.extend(new), draft=draft,
                                    stop_eos=False)
            return tokens, stats

        ask(PROMPT, draft=True)
        before = _retained(engine)
        prompt, labels = [*PROMPT, 9, 15], [V - 1, 5, 5, 0]
        # Independently run the complete target head without capture/scoring.
        # Its fresh state and ordinary prefill arithmetic match the public
        # scorer's declared one-token, temperature-1 target distribution.
        reference = Engine(engine.w, capacity=128, max_rows=3, prefill_rows=256)
        prefill_begin(reference, prompt, mtp=False)
        target = prefill_chunk(reference, prompt, 0, mtp=False)[0].float().cpu()
        label_logits, lse = engine.score_labels(prompt, labels)
        assert label_logits == [float(target[label]) for label in labels]
        # Reuse the existing test_logprobs FP64-oracle absolute error budget;
        # no new fitted tolerance or relative slack is introduced for labels.
        normalized = target.double().log_softmax(0)
        for label, logit in zip(labels, label_logits, strict=True):
            assert logit - lse == pytest.approx(float(normalized[label]), abs=3e-6, rel=0)
        assert _retained(engine) == before
        assert engine._calls == {}
        if engine.multi is not None:
            assert engine.multi.live() == 0

        # A refused public request reaches neither generation nor ownership.
        with monkeypatch.context() as patch:
            def forbidden(*args, **kwargs):
                raise AssertionError("invalid IDs reached generation")
            patch.setattr(engine, "generate", forbidden)
            for bad in ([-1], [V], [True], [1.5], []):
                with pytest.raises(ValueError):
                    engine.score_labels(prompt, bad)
        assert _retained(engine) == before
        assert engine._calls == {}

        resumed, stats = ask(prompt, draft=True)
        fresh, _ = ask(prompt, draft=False)
        assert resumed == fresh
        assert stats["cached"] == len(PROMPT) - 1
        assert engine._calls == {}
        engine.close()
        assert engine._closed
        with pytest.raises(RuntimeError, match="closing or closed"):
            engine.score_labels(prompt, labels)
    finally:
        if reference is not None:
            for buffer in (reference.buf, reference.mbuf):
                if buffer is not None and buffer.hc_plans is not None:
                    buffer.hc_plans.close()
        engine.close()

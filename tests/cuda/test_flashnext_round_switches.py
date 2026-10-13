"""Flash Next concurrent-round paths behind the benchmark switches (A/B baseline: the "old" values are removed after
selection): every value emits each stream's solo tokens, and a failed frame's codec status still wins."""

import json

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _model  # noqa: E402

from tensorfold.cuda import ab_switch  # noqa: E402
from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import KVNumericError, State  # noqa: E402

PROMPTS = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13], [8, 8, 9, 2000, 31]]
SAMPLINGS = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, top_k=20, top_p=0.95), None]
RQ = "rotorquant6-norm"                                   # the production cache format: a status word a stream


def _switch(monkeypatch, tmp_path, **values) -> None:
    """The benchmark switches a test's rounds read (``ab_switch.refresh`` once a round)."""

    control = tmp_path / "ab.json"
    control.write_text(json.dumps(values))
    monkeypatch.setattr(ab_switch, "_path", str(control))
    monkeypatch.setattr(ab_switch, "_values", {})
    monkeypatch.setattr(ab_switch, "_stamp", None)


def _solo(w, prompts, samplings, count, kv_dtype, capacity=1024):
    refs = []
    for prompt, sampling in zip(prompts, samplings):
        e = Engine(w, capacity=capacity, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        refs.append(serial_decode(e, prefill(e, prompt, sampling), count, sampling).tokens)
    return refs


def _decode(w, prompts, samplings, count, kv_dtype, *, depth=3, confidence=0.3, capacity=1024, emitted=None):
    dec = MultiDecoder(w, slots=len(prompts), capacity=capacity, depth=depth, confidence=confidence,
                       kv_dtype=kv_dtype)
    streams = []
    for i, (prompt, sampling) in enumerate(zip(prompts, samplings)):
        s = Stream(list(prompt), count, sampling,
                   emit=(lambda new, i=i: emitted[i].extend(new)) if emitted is not None else None)
        dec.admit(s)
        streams.append(s)
    while dec.live():
        dec.finish(dec.round())
    return dec, streams


def _status_reads(monkeypatch) -> list:
    """Counts each blocking readback of a codec status word (``State.kv_check``'s own ``cpu()``)."""

    reads = []
    real = torch.Tensor.cpu

    def cpu(t, *a, **k):
        if t.dtype == torch.int32 and t.shape == (2,) and t.is_cuda:
            reads.append(t.data_ptr())
        return real(t, *a, **k)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    return reads


@pytest.mark.parametrize("kv_dtype", ["bf16", RQ])
@pytest.mark.parametrize("b1", ["new", "old"])
def test_b1_rounds_emit_each_streams_solo_tokens(monkeypatch, tmp_path, kv_dtype, b1):
    """B1: a step's codec status words are read back in its sampling packet's wait; tokens equal the solo runs in
    both values; the new value never reads a status word on its own once the streams decode."""

    w = _model()
    refs = _solo(w, PROMPTS, SAMPLINGS, 20, kv_dtype)
    _switch(monkeypatch, tmp_path, b1=b1)
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype)
    streams = [Stream(list(p), 20, smp, draft=i != 3) for i, (p, smp) in enumerate(zip(PROMPTS, SAMPLINGS))]
    for s in streams:
        dec.admit(s)
    dec.finish(dec.round())                              # the prompts' passes (their own checks), then a round
    reads = _status_reads(monkeypatch)
    rounds = 0
    while dec.live():
        dec.finish(dec.round())
        rounds += 1
    assert [s.out for s in streams] == refs
    assert rounds > 3
    if kv_dtype == RQ:
        assert (len(reads) == 0) == (b1 == "new"), len(reads)


@pytest.mark.parametrize("b1", ["new", "old"])
@pytest.mark.parametrize("where", ["verify", "mtp"])
def test_b1_failed_status_wins_over_nonfinite_scores_and_publishes_nothing(monkeypatch, tmp_path, b1, where):
    """A NaN query scale fails the step's status and makes its scores NaN: the round raises the codec failure (not the
    sampler's or the draft confidence's), nothing of the frame is emitted, the stream's state stays poisoned until
    reset, and its slot serves again."""

    w = _model()
    _switch(monkeypatch, tmp_path, b1=b1)
    emitted = [[], []]
    dec = MultiDecoder(w, slots=2, capacity=256, depth=3, confidence=0.0, kv_dtype=RQ)
    streams = [Stream(list(PROMPTS[0]), 12, SAMPLINGS[1], emit=lambda new: emitted[0].extend(new)),
               Stream(list(PROMPTS[1]), 12, None, emit=lambda new: emitted[1].extend(new))]
    for s in streams:
        dec.admit(s)
    dec.finish(dec.round())
    before = [list(e) for e in emitted]
    states = [s.st for s in streams]
    positions = [st.pos for st in states]
    scale = w.mtp.layer.attn.q_scale if where == "mtp" else w.layers[1].attn.q_scale
    saved = scale.clone()
    scale.fill_(float("nan"))
    try:
        with pytest.raises(KVNumericError, match="layer=MTP" if where == "mtp" else "layer=1"):
            dec.round()
    finally:
        scale.copy_(saved)
    assert emitted == before                             # the failed round published nothing
    assert all(st._kv_error is not None for st in states)
    if where == "verify":                                # nothing of the verified frame was committed
        assert [st.pos for st in states] == positions
    with pytest.raises(KVNumericError):
        states[0].kv_check()
    failed = dec.drop()
    assert {id(s) for s in failed} == {id(s) for s in streams}
    assert all(st._kv_error is None and st.kv_status.cpu().tolist() == [0, -2] for st in dec.free)
    again = Stream(list(PROMPTS[0]), 6, SAMPLINGS[1])
    dec.admit(again)
    while dec.live():
        dec.finish(dec.round())
    assert again.error is None and again.out == _solo(w, PROMPTS[:1], SAMPLINGS[1:2], 6, RQ, capacity=256)[0]


def test_b1_words_latch_every_state_in_order_and_run_once():
    """``KVWords``: the first failure in stream order is raised after every state latched; the follow-up runs only
    for a valid frame; a second check or settle does nothing."""

    from tensorfold.families.qwen4_exp.cuda.forward import KVWords

    w = _model()
    states = [State(w, 64, 4, RQ) for _ in range(3)]
    host = torch.zeros((4, 2), dtype=torch.int32, pin_memory=True)
    after = []
    words = KVWords([(st, i, i + 1) for i, st in enumerate(states)], host, lambda: after.append(1))
    for st, value in zip(states, ([0, -2], [2, 1], [4, -1])):
        st.kv_begin()
        st.kv_status.copy_(torch.tensor(value, dtype=torch.int32, device="cuda"))
    words.queue()
    torch.cuda.synchronize()
    with pytest.raises(KVNumericError, match="layer=1"):
        words.check()
    assert states[0]._kv_error is None and states[1]._kv_error is not None and states[2]._kv_error is not None
    assert not any(st._kv_pending for st in states) and not after
    words.settle()                                       # checked once: nothing more
    good = [State(w, 64, 4, RQ) for _ in range(2)]
    for st in good:
        st.kv_begin()
    words = KVWords([(st, i, i + 1) for i, st in enumerate(good)], host, lambda: after.append(1))
    words.settle()
    words.settle()
    assert after == [1] and not any(st._kv_pending for st in good)

"""Flash Next concurrent-round paths behind the benchmark switches (A/B baseline: the "old" values are removed after
selection): every value emits each stream's solo tokens, and a failed frame's codec status still wins."""

import json

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import _model  # noqa: E402

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


class _FailedFrame:
    """A frame's status words (``KVWords``' protocol) whose settle reports the frame's codec failure."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def queue(self) -> None:
        self.calls.append("queue")

    def check(self) -> None:
        self.calls.append("check")

    def settle(self) -> None:
        self.calls.append("settle")
        raise KVNumericError("the frame failed")


def _raise(error):
    def fail(*args, **kwargs):
        raise error
    return fail


def test_b1_a_failure_before_the_status_check_reports_the_failed_frame_first(monkeypatch):
    """KVWords' precedence past validation: a failure of the draft picks' device work, of their readback's wait, or
    of the sampler's readback wait settles the frame's words first, so its codec failure is the error raised (the
    other failure is its context), as the check right after the forward did before B1."""

    from tensorfold.cuda.sampling import collect_streams, launch_streams

    w = _model()
    dec = MultiDecoder(w, slots=2, capacity=256, depth=3, kv_dtype=RQ)
    logits = torch.randn((2, 64), device="cuda")
    for target, name, error, calls in ((torch, "logsumexp", MemoryError("device work"), ["settle"]),
                                       (torch.cuda, "current_stream", RuntimeError("wait"), ["queue", "settle"])):
        words = _FailedFrame()
        with monkeypatch.context() as patched:
            patched.setattr(target, name, _raise(error))
            with pytest.raises(KVNumericError) as caught:
                dec._picks(logits, [5, 6], [None, SAMPLINGS[1]], False, words, None)
        assert words.calls == calls and caught.value.__context__ is error
    packets = launch_streams(logits, [0, 1, 2], [[5], [6]], [None, SAMPLINGS[1]])
    words, error = _FailedFrame(), RuntimeError("wait")
    with monkeypatch.context() as patched:
        patched.setattr(torch.cuda, "current_stream", _raise(error))
        with pytest.raises(KVNumericError) as caught:
            collect_streams(packets, words)
    assert words.calls == ["queue", "settle"] and caught.value.__context__ is error


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


def _ssd_ngram_model(tmp_path, monkeypatch):
    """The synthetic model with its n-gram branch on a 4-bit table read from disk by the production reader
    (``--ple-on-ssd`` with TENSORFOLD_SSD_NATIVE_THREADS: the native reader behind the read-ahead wrapper)."""

    import struct

    import numpy as np

    from test_flashnext_forward import _Rand, _cfg, _ple
    from tensorfold.families.qwen4_exp import host_table

    c = _cfg(ple=True)
    rows, dims = c.ngram(0).rows, c.ngram(0).dims
    rng = np.random.default_rng(29)
    words = rng.integers(0, 2**32, (rows, dims // 8), dtype=np.uint32)
    scales = torch.from_numpy(rng.uniform(1e-3, 4e-3, (rows, dims // 32)).astype(np.float32)).to(torch.bfloat16)
    biases = (-scales.float() * 7.5).to(torch.bfloat16)
    parts = [words.view(np.uint8), scales.view(torch.uint16).numpy().view(np.uint8),
             biases.view(torch.uint16).numpy().view(np.uint8)]
    entries, at = [], 0
    for part, shape, dtype in zip(parts, ((rows, dims // 8), (rows, dims // 32), (rows, dims // 32)),
                                  ("U32", "BF16", "BF16")):
        entries.append({"dtype": dtype, "shape": list(shape), "data_offsets": [at, at + part.nbytes]})
        at += part.nbytes
    header = json.dumps(dict(zip(("w", "s", "b"), entries))).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    path = tmp_path / "ngram.safetensors"
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for part in parts:
            f.write(part.tobytes())
    monkeypatch.setenv("TENSORFOLD_SSD_NATIVE_THREADS", "16")
    table = host_table._ssd_table([(path, *entries)])
    return _model(ple=_ple(c, table, _Rand(3))), table


@pytest.mark.parametrize("b2,profiled", [("new", False), ("new", True), ("old", False)])
def test_b2_ngram_rows_read_behind_the_first_layers_give_each_streams_solo_tokens(monkeypatch, tmp_path, b2,
                                                                                    profiled):
    """B2: a round's n-gram rows (read through the production SSD reader) are read and copied when the forward
    reaches the n-gram layer, after the layers before it are queued; tokens equal the solo runs either way, also
    with the round profile timing the late reads (ple_ms)."""

    from tensorfold.cuda import round_profile
    from tensorfold.families.qwen4_exp.cuda import forward as fwd

    w, table = _ssd_ngram_model(tmp_path, monkeypatch)
    path = tmp_path / "rounds.jsonl"
    if profiled:
        monkeypatch.setenv(round_profile.ENV, str(path))
    else:
        monkeypatch.delenv(round_profile.ENV, raising=False)
    try:
        prompts = PROMPTS[:3]
        samplings = SAMPLINGS[:3]
        refs = _solo(w, prompts, samplings, 16, RQ)
        _switch(monkeypatch, tmp_path, b2=b2)
        order, real_late, real_embed = [], fwd.stage_late, fwd._embed
        monkeypatch.setattr(fwd, "stage_late", lambda *a: order.append("late") or real_late(*a))
        monkeypatch.setattr(fwd, "_embed", lambda *a: order.append("embed") or real_embed(*a))
        dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, confidence=0.3, kv_dtype=RQ)
        streams = [Stream(list(p), 16, smp) for p, smp in zip(prompts, samplings)]
        for s in streams:
            dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
        assert [s.out for s in streams] == refs
        lates = order.count("late")
        assert (lates > 3) == (b2 == "new") and (lates == 0) == (b2 == "old")
        if b2 == "new":                                  # each late read follows its own forward's embedding
            assert all(order[i - 1] == "embed" for i, x in enumerate(order) if x == "late")
        if profiled:
            dec.close_telemetry()
            records = [json.loads(line) for line in path.read_text().splitlines()]
            assert records and all(r["ple_ms"] >= 0 for r in records) and any(r["ple_ms"] > 0 for r in records)
    finally:
        table.close()

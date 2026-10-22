"""Retained streams, caches and complete replies equal the former always-readout forward."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from prefix_checks import same_bits
from test_flashnext_forward import _model
from test_flashnext_prompt_cache import _assert_same_state

from tensorfold.cuda import precision
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen4_exp.cuda import forward as fwd
from tensorfold.families.qwen4_exp.cuda import mtp
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode
from tensorfold.families.qwen4_exp.cuda.weights import load


def _released_finish(w, mixer, b, rows, pending, logits=True, ends=()):
    """Prior control oracle: retain its unused mixer work while reusing unchanged numerical kernels."""

    c = w.cfg
    b.streams[:rows].copy_(b.h[:rows])
    fwd._writeback(b.streams, b, rows, c, pending)
    if b.prefill:
        if len(ends) > 1:
            at = torch.tensor(list(ends), dtype=torch.long, device=b.pss.device)
            b.pss[:len(ends)].copy_(b.pss.index_select(0, at))
            streams = b.streams.index_select(0, at)
        else:
            last = ends[0] if ends else rows - 1
            b.pss[0].copy_(b.pss[last])
            streams = b.streams[last:last + 1]
        rows = max(1, len(ends))
    else:
        streams = b.streams
    fwd._readout(mixer, b, streams, rows, c.eps, c.streams, c.low, None)
    if not logits:
        return None
    out = fwd._mm(b.mixed[:rows], w.head, b.xs_mixed[:rows], b.logits[:rows], b)
    if w.comm is not None:
        fwd.candidates(w, b, out, rows, offset=int(w.meta["vocab_offset"]))
    return out


@pytest.fixture(scope="module", params=["affine", "modelopt-a4"])
def weights(request, tmp_path_factory):
    if request.param == "affine":
        yield _model()
        return
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("native ModelOpt A4 requires SM12")
    from test_flashnext_modelopt_a4_model import _calibrated_model

    model = _calibrated_model(tmp_path_factory.mktemp("final-readout-modelopt"))
    with precision.using(precision.CHECKPOINT, asked=True):
        w = load(model, draft_vocab=128, ple_on_ssd=True)
        # Pytest preserves a test failure separately if table retirement fails.
        request.addfinalizer(lambda: w.meta["ple_tables"].close(torch.cuda.synchronize))
        yield w


@pytest.mark.parametrize("graphs", [False, True], ids=["eager", "graphs"])
@pytest.mark.parametrize("draft", [False, True], ids=["serial", "mtp"])
def test_omitted_readout_preserves_resumed_states_and_complete_replies(weights, monkeypatch, graphs, draft):
    w = weights
    sampling = Sampling(seed=7, top_k=20, top_p=.95)
    prompt = [5, 17, 99, 250, 7, 64, 30, 11, 12, 13, 9, 6, 8]
    long_boundary = w.cfg.quant == "mlx" and not graphs and not draft
    if long_boundary:
        prompt = [(37 * index + 11) % w.cfg.vocab for index in range(2051)]
    extended = [*prompt, 2, 14, 3]

    def run(released):
        calls = {"main": 0, "mtp": 0}
        readout = fwd._readout

        def measured(hc, *args, **kwargs):
            if hc is w.mixer:
                calls["main"] += 1
            elif hc is w.mtp.mixer:
                calls["mtp"] += 1
            return readout(hc, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(fwd, "_readout", measured)
            if released:
                patch.setattr(fwd, "finish", _released_finish)
                patch.setattr(mtp, "finish", _released_finish)
                # The old finish already mixed MTP rows; suppress only the new explicit stage.
                patch.setattr(mtp, "_final_readout", lambda *_args: None)
            e = Engine(w, capacity=4096 if long_boundary else 128, max_rows=8,
                       prefill_rows=2048 if long_boundary else 5, graphs=graphs)
            first = prefill(e, prompt, sampling, mtp=draft, keep_at=7)
            cold_state, cold_tail = e.st.clone(), e.last_streams.clone()
            resumed = prefill(e, extended, sampling, mtp=draft, resume=e.kept)
            resumed_state, resumed_tail = e.st.clone(), e.last_streams.clone()
            prefill_calls = dict(calls)
            result = (mtp_decode(e, resumed, 8, sampling, depth=2, confidence=0., stop_eos=False) if draft else
                      serial_decode(e, resumed, 8, sampling, stop_eos=False))
            torch.cuda.synchronize()
            return e, first, cold_state, cold_tail, resumed_state, resumed_tail, prefill_calls, result

    reference, first_ref, cold_ref, tail_ref, resumed_ref, resumed_tail_ref, calls_ref, result_ref = run(True)
    candidate, first, cold, tail, resumed, resumed_tail, calls, result = run(False)
    assert first == first_ref
    _assert_same_state(cold, cold_ref)
    _assert_same_state(resumed, resumed_ref)
    _assert_same_state(candidate.st, reference.st)
    assert same_bits(tail, tail_ref) and same_bits(resumed_tail, resumed_tail_ref)
    assert len(result.tokens) == 8 and result.tokens == result_ref.tokens
    assert (result.rounds, result.drafted, result.accepted, result.keeps, result.committed, result.widths) == (
        result_ref.rounds, result_ref.drafted, result_ref.accepted, result_ref.keeps, result_ref.committed, result_ref.widths)
    assert calls["main"] < calls_ref["main"]
    if draft:
        assert calls["mtp"] == 0 < calls_ref["mtp"]
    else:
        assert calls["mtp"] == calls_ref["mtp"] == 0

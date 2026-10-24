"""Six depth3 windows use packed native rows and preserve common State/graphs."""

import numpy as np
import pytest
import torch

from test_flashnext_sm70_family import _small_weights
from test_sm70_gguf_prepared import dense_fixture
from test_sm70_gguf_decode_rows import bank_fixture
from tensorfold.cuda.sm70 import family_ops
from tensorfold.families.qwen4_exp.cuda import padded_linears
from tensorfold.families.qwen4_exp.cuda.decode import prefill
from tensorfold.families.qwen4_exp.cuda.forward import commit, compute, stage
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder, _slot
from tensorfold.families.qwen4_exp.cuda.multi_graphs import RoundGraphs
from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 packed six lanes")


def exact(a, b):
    torch.testing.assert_close(a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0)


def committed_equal(a, b):
    assert a.pos == b.pos
    for index, parity in enumerate(a.cur):
        exact(a.rec[parity, index], b.rec[b.cur[index], index])
    for name in ("conv", "ple_tail"):
        exact(getattr(a, name), getattr(b, name))
    for left, right in zip(a.kc, b.kc):
        for name in ("k", "v", "ks", "vs"):
            exact(getattr(left, name)[: a.pos], getattr(right, name)[: b.pos])
    for left, right in zip(a.ikc, b.ikc):
        exact(left[: a.pos], right[: b.pos])
    for left, right in zip(a.pooled, b.pooled):
        exact(left[: a.pos // a.ratio], right[: b.pos // b.ratio])
    np.testing.assert_array_equal(a.ple_history, b.ple_history)


@pytest.mark.parametrize("cache", ["fp16", "int8"])
def test_six_four_row_native_graphs_match_independent_windows_partial_commit_and_prefix(cache, monkeypatch):
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", False)
    monkeypatch.setattr(padded_linears, "_ENABLED", True)
    w = dense_fixture(_small_weights(include_ple=True))
    w.cfg.experts, w.cfg.top_k = 10, 10
    torch.manual_seed(5751)
    for layer in w.layers:
        layer.moe.router = (torch.randn(11, w.cfg.hidden, device="cuda") * 0.03).half().float()
        layer.moe.experts = bank_fixture(12 if layer.linear else 13, count=10, up_kind=13, down_kind=7)
        layer.moe.experts.implementation = "packed_blocks2w"
    decoder = MultiDecoder(w, slots=6, capacity=64, depth=3, prefill_rows=12, kv_dtype=cache, graphs=False, keep=0)
    graphs = RoundGraphs(decoder, 512 << 20)
    reference = Buffers(w, 4, 64, moe_prefill=True)
    try:
        graphs.capture()
        decoder.graphs = graphs
        for index, state in enumerate(decoder.free):
            prefill(
                _slot(w, state, decoder.buf, None, decoder.pbuf, 64, 12),
                [4, 17, 3, 99, 33, 2, 7, 5, 6],
                None,
                mtp=False,
            )
            if index % 2:
                state.rec[1 - state.cur[0]].copy_(state.rec[state.cur[0]])
                state.cur = [1 - value for value in state.cur]
        states = list(reversed(decoder.free))
        for iteration in range(2):
            windows = [[(13 * lane + row + iteration * 23) % w.cfg.vocab for row in range(4)] for lane in range(6)]
            refs = [state.clone() for state in states]
            logits, streams = [], []
            keeps = [1, 2, 4, 3, 2, 1] if iteration == 0 else [4, 3, 2, 1, 3, 4]
            for copied, tokens, kept in zip(refs, windows, keeps):
                seg = stage(w, reference, [(copied, tokens)])
                logits.append(compute(w, seg, reference, context=256).clone())
                streams.append(reference.streams[:4].clone())
                commit(w, copied, reference, 4, kept)
            segs = stage(w, decoder.buf, list(zip(states, windows)), late=True)
            assert segs[-1][2] == 24
            actual = graphs.verify_forward(segs)
            exact(actual, torch.cat(logits))
            exact(decoder.buf.streams[:24], torch.cat(streams))
            for (state, start, _), copied, kept in zip(segs, refs, keeps):
                commit(w, state, decoder.buf, 4, kept, at=start)
                committed_equal(state, copied)
        assert graphs.bound.hits > 0
        # Prefix copies resume through the same native R1 path, independently
        # owning committed cache storage and PLE history.
        source = states[0]
        restored = State(w, 64, 4, cache)
        restored.copy_prefix(source, source.pos, 0)
        restored.restore(source.snapshot())
        committed_equal(restored, source)
        pair = [source, restored]
        outputs = []
        for state in pair:
            seg = stage(w, reference, [(state, [31])])
            outputs.append(compute(w, seg, reference, context=256).clone())
            commit(w, state, reference, 1, 1)
        exact(outputs[0], outputs[1])
        committed_equal(restored, source)
    finally:
        graphs.close()
        decoder.graphs = None

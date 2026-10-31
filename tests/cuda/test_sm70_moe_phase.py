"""Cached prompt suffixes preserve the prompt expert policy through full forward."""
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tensorfold.cuda.sm70 import family_ops as ops
from test_flashnext_sm70_family import _small_weights

pytestmark = pytest.mark.skipif(torch.cuda.get_device_capability() != (7, 0), reason="SM70 prompt phase")


def test_short_cached_prompt_suffix_uses_prompt_experts_and_preserves_complete_state():
    from tensorfold.families.qwen4_exp.cuda.forward import commit, forward
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    w = _small_weights(include_ple=True)
    calls = []

    class PhaseExperts:
        kernel = "gguf"

        def __init__(self, base):
            self.base = base

        def forward(self, x, picks, out=None):
            calls.append(("decode", x.shape[0]))
            return self.base.forward(x, picks, out)

        def forward_prefill(self, x, picks, out=None):
            calls.append(("prefill", x.shape[0]))
            return self.base.forward(x, picks, out)

    for layer in w.layers:
        layer.moe.experts = PhaseExperts(layer.moe.experts)
    cold, source, cached = [State(w, 64, 32, "fp16") for _ in range(3)]
    prompt = Buffers(w, 32, 64, prefill=True)
    decode = Buffers(w, 32, 64)
    tokens = list(range(25))

    def committed_equal(actual, expected):
        assert actual.pos == expected.pos
        # A restored prefix owns the active recurrent bank; its inactive bank
        # is scratch and need not contain the same previous transaction.
        for index, parity in enumerate(actual.cur):
            torch.testing.assert_close(actual.rec[parity, index], expected.rec[expected.cur[index], index], atol=0, rtol=0)
        for name in ("conv", "ple_tail"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=0, rtol=0)
        for a, b in zip(actual.kc, expected.kc):
            for name in ("k", "v", "ks", "vs"):
                torch.testing.assert_close(getattr(a, name)[:actual.pos], getattr(b, name)[:expected.pos], atol=0, rtol=0)
        for name in ("ikc", "pooled"):
            end = actual.pos if name == "ikc" else actual.pos // actual.ratio
            for a, b in zip(getattr(actual, name), getattr(expected, name)):
                torch.testing.assert_close(a[:end], b[:end], atol=0, rtol=0)
        np.testing.assert_array_equal(actual.ple_history, expected.ple_history)
    # Retained Torch family adapters isolate this phase change from optional
    # attention kernels. The native recurrent/cache lifecycle remains selected.
    with patch.object(ops, "_NATIVE_FAMILY", False), patch.object(ops, "_NATIVE_ATTENTION", False):
        expected = forward(w, cold, prompt, tokens).clone()
        commit(w, cold, prompt, 25, 25)
        forward(w, source, prompt, tokens[:24])
        commit(w, source, prompt, 24, 24)
        cached.copy_prefix(source, 24, 0)
        cached.restore(source.snapshot())
        actual = forward(w, cached, prompt, tokens[24:]).clone()
        commit(w, cached, prompt, 1, 1)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        committed_equal(cached, cold)
        assert {rows for phase, rows in calls} == {1, 24, 25}
        assert all(phase == "prefill" for phase, _ in calls)
        calls.clear()
        for state in (cold, cached):
            forward(w, state, decode, [31])
            commit(w, state, decode, 1, 1)
        assert calls and all(phase == "decode" and rows == 1 for phase, rows in calls)
        committed_equal(cached, cold)

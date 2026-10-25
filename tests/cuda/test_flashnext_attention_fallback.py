"""The real eager decode fallback borrows positions without a prompt-only owner."""

import pytest
import torch

from test_flashnext_sm70_family import _small_weights
from tensorfold.cuda.sm70 import family_ops
from tensorfold.families.qwen4_exp.cuda.forward import commit, forward
from tensorfold.families.qwen4_exp.cuda.state import Buffers, State


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("cache", ["fp16", "int8", "rotorquant-signed128-4", "rotorquant-signed64-norm4"])
def test_unqualified_decode_uses_existing_position_and_exact_serial_attention(monkeypatch, rows, cache):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 fallback gate")
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", False)
    w = _small_weights(True, hidden=2560, lowrank=320, head_dim=256)
    base = State(w, 128, 3, cache)
    pbuf = Buffers(w, 20, 128, prefill=True)
    prefix = [4, 17, 3, 99, 33, 7, 31, 81, 2, 5, 9, 6, 15, 19, 21, 24, 27, 32, 41, 53]
    forward(w, base, pbuf, prefix)
    commit(w, base, pbuf, len(prefix), len(prefix))
    candidate, oracle = base.clone(), base.clone()
    buf, ref = Buffers(w, 3, 128), Buffers(w, 3, 128)
    assert not hasattr(buf, "pos_blk"), "decode must not require a prompt-only position owner"
    tokens = [71, 33, 18][:rows]
    expected = []
    for token in tokens:
        expected.append(forward(w, oracle, ref, [token]).clone())
        commit(w, oracle, ref, 1, 1)
    monkeypatch.setattr(family_ops, "_NATIVE_ATTENTION", True)
    original = family_ops.attention
    borrowed = []

    def observe(*args, **kwargs):
        assert kwargs.get("_native_allowed") is False
        borrowed.append(args[3])
        return original(*args, **kwargs)

    monkeypatch.setattr(family_ops, "attention", observe)
    position = candidate.pos_dev.clone()
    actual = forward(w, candidate, buf, tokens).clone()
    assert candidate.pos == len(prefix) and torch.equal(candidate.pos_dev, position)
    assert [value.cpu().tolist() for value in borrowed] == [[len(prefix)+i] for i in range(rows)]
    assert borrowed[0].data_ptr() == candidate.pos_dev.data_ptr()
    assert torch.equal(actual, torch.cat(expected))
    assert not hasattr(buf.attn, "native_workspace")
    commit(w, candidate, buf, rows, rows)
    assert candidate.pos == oracle.pos
    assert torch.equal(candidate.rec[candidate.cur[0]], oracle.rec[oracle.cur[0]])
    assert torch.equal(candidate.conv, oracle.conv)
    assert torch.equal(candidate.ple_tail, oracle.ple_tail)
    for left, right in zip(candidate.kc, oracle.kc):
        for field in ("k", "v", "ks", "vs"):
            a, b = getattr(left, field), getattr(right, field)
            assert (a is b is None) or torch.equal(a[:candidate.pos], b[:oracle.pos])

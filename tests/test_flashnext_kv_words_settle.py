"""A completed collective validation needs no second GPU wait."""
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.forward import KVWords
from tensorfold.families.qwen4_exp.cuda.state import KVNumericError, State

pytestmark = pytest.mark.torch


def state(*, pending, error=None):
    result = State.__new__(State)
    result.kv_status = torch.tensor([0, -2], dtype=torch.int32)
    result._kv_pending, result._kv_error, result._resize_error = pending, error, None
    result.kv_key_dtype = result.kv_value_dtype = "int8"
    result.pos, result.mtp_len = 17, 16
    return result


@pytest.mark.parametrize("pending", [False, True])
def test_settle_waits_only_for_words_that_are_still_pending(monkeypatch, pending):
    events = []
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: SimpleNamespace(
        synchronize=lambda: events.append("wait")))
    st = state(pending=pending)
    # Stale invalid host words must be ignored after the collective checked st.
    host = torch.tensor([[7, -1]], dtype=torch.int32)
    words = KVWords([(st, 0, 1)], host, lambda: events.append("after"))
    words.settle()
    words.settle()
    assert events == (["wait", "after"] if pending else ["after"])
    assert not st._kv_pending and st._kv_error is None


def test_latched_failure_precedes_callback_without_an_empty_wait(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: SimpleNamespace(
        synchronize=lambda: events.append("wait")))
    error = KVNumericError("already latched by the rank cohort")
    st = state(pending=False, error=error)
    words = KVWords([(st, 0, 1)], torch.zeros((1, 2), dtype=torch.int32), lambda: events.append("after"))
    with pytest.raises(KVNumericError) as caught:
        words.settle()
    assert caught.value is error
    assert events == []

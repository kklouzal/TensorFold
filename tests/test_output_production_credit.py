"""Actual bounded output ownership and pre-step production reservations."""

import queue

import pytest

from tensorfold.cuda.output_mailbox import OutputMailbox


def test_pending_and_actual_callback_own_credit_until_completion():
    box, wakes = OutputMailbox(6, pending_tokens=2), []
    def wake():
        wakes.append(True)
    box.set_wake(wake)
    first = box.reserve(2)
    assert first is not None
    chunk = [3, 7]
    box.put(("tokens", chunk))
    first.close()
    assert box.reserve(1) is None
    assert box.get_nowait() == ("tokens", chunk)
    assert box.record()["callback_inflight_tokens"] == 2 and box.reserve(1) is None
    box.complete_delivery(chunk)
    assert wakes == [True]
    next_step = box.reserve(2)
    assert next_step is not None
    box.put(("tokens", [11]))
    next_step.close()  # An accepted shorter prefix releases unused pre-step credit.
    assert box.record()["reserved_production_tokens"] == 0
    assert box.record()["queued_token_ids"] == 1


def test_full_reply_default_retains_original_transport_without_callback_ack():
    box = OutputMailbox(4)
    box.put(("tokens", [1, 2]))
    box.put(("tokens", [3, 4]))
    box.put(("done", {}))
    assert box.get_nowait() == ("tokens", [1, 2])
    assert box.get_nowait() == ("tokens", [3, 4])
    assert box.get_nowait() == ("done", {})
    with pytest.raises(queue.Empty):
        box.get_nowait()


def test_wrong_callback_or_overwide_emission_never_releases_or_overdraws_credit():
    box = OutputMailbox(4, pending_tokens=1)
    lease = box.reserve(1)
    with pytest.raises(RuntimeError, match="production capacity"):
        box.put(("tokens", [1, 2]))
    assert box.record()["published_token_ids"] == 0
    chunk = [3]
    box.put(("tokens", chunk))
    lease.close()
    box.get_nowait()
    with pytest.raises(RuntimeError, match="actual owned"):
        box.complete_delivery([3])
    assert box.reserve(1) is None
    box.complete_delivery(chunk)
    next_step = box.reserve(1)
    assert next_step is not None
    next_step.close()

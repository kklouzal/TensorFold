# ruff: noqa: E402 - maintained CUDA fixture imports require optional Torch.
"""Whole logical input credit across real FIFO backend jobs and caller retirement."""

import pytest
pytest.importorskip("torch")

from tensorfold.server.errors import CapacityError
from test_cuda_fifo_admission import Decoder, client, fifo, joined


class Growing(Decoder):
    def input_reservation_bytes(self, stream):
        # Explicit controlled full-four-token future, not a service default.
        return 512


def test_larger_accepted_subrun_reuses_original_logical_ticket_and_future_credit(monkeypatch):
    import tensorfold.cuda.scheduler as scheduling

    clock = [1.]
    monkeypatch.setattr(scheduling.time, "monotonic", lambda: clock[0])
    decoder = Growing(blocked=(2,))
    scheduler = fifo(decoder, streams=1, byte_limit=1024, timeout=10)
    output = []
    newer = None
    try:
        with scheduler._request_limit.request():
            scheduler.submit([1], 1, None, False, lambda values: output.extend(values))
            record = scheduler.queue_record()
            assert record["waiting"] == [] and record["charged_live_input_bytes"] == 512
            assert record["reserved_live_requests"] == 1
            clock[0] = 100.  # Initial queue age cannot truncate an admitted logical operation.
            newer = client(scheduler, 2)
            with scheduler.waiting.not_empty:
                assert scheduler.waiting.not_empty.wait_for(lambda: scheduler.waiting.revision >= 3, 3)
            scheduler.submit([1, 1, 1, 1], 1, None, False, lambda values: output.extend(values))
            assert output == [2, 2]
            assert [row[1] for row in decoder.admissions] == [0, 0]
            assert scheduler.queue_record()["charged_live_input_bytes"] == 1024
        decoder.allow(2)
        joined(newer)
        assert newer.errors == [] and scheduler.queue_record()["charged_live_input_bytes"] == 0
    finally:
        decoder.allow(2) if decoder._wake is not None else None
        if newer is not None:
            joined(newer)
        scheduler.close()


def test_other_principal_cannot_consume_a_live_logical_readmission_credit():
    decoder = Growing()
    scheduler = fifo(decoder, byte_limit=1024, principal_bytes=512)
    try:
        with scheduler._request_limit.principal("alice"), scheduler._request_limit.request():
            scheduler.submit([1], 1, None, False, lambda _values: None)
            refused = client(scheduler, 3, principal="alice")
            joined(refused)
            assert len(refused.errors) == 1 and isinstance(refused.errors[0], CapacityError)
            other = client(scheduler, 3, principal="bob")
            joined(other)
            assert other.errors == [] and other.output == [4]
            scheduler.submit([1, 1, 1, 1], 1, None, False, lambda _values: None)
        later = client(scheduler, 3, principal="alice")
        joined(later)
        assert later.errors == [] and scheduler.queue_record()["charged_live_input_bytes"] == 0
    finally:
        scheduler.close()


def test_interrupted_worker_map_removal_never_retires_a_later_subrun_credit():
    import time
    from tensorfold.cuda.scheduler import AdmissionStatus, Waiting
    from tensorfold.cuda.streams import Stream
    from tensorfold.server.request_limits import RequestLimit

    class Registry(dict):
        failed = False
        def pop(self, *args):
            if not self.failed:
                self.failed = True
                raise KeyboardInterrupt("controlled retirement journal interruption")
            return super().pop(*args)
    waiting, limit = Waiting(policy="fifo", max_input_bytes=512), RequestLimit(1)
    first, second = object(), object()
    def publish(worker, prompt):
        stream = Stream(prompt, 1)
        stream.principal = None
        stream.admission_status = AdmissionStatus(-1, time.monotonic(), 72 + len(prompt) * 28, None)
        return stream, limit.publish_inputs(worker, waiting, lambda claim, register:
            waiting.publish((stream, worker), claim=claim, register=register, reserved_bytes=512))
    with limit.request():
        limit.borrow(first)
        old, claim = publish(first, [1])
        waiting.activate(claim)
        limit._workers = Registry(limit._workers)
        with pytest.raises(KeyboardInterrupt):
            limit.release(first)
        assert not limit.worker_active(first)
        assert old.admission_status.phase == "between_subruns"
        limit.borrow(second)
        new, same = publish(second, [1, 1])
        assert same is claim and new.admission_ticket == old.admission_ticket
        limit.release(first)  # Retried old tombstone cannot touch the new backend/credit.
        assert limit.worker_active(second) and new.admission_status.phase == "waiting"
        assert waiting.requests()[1][0] is claim
        limit.release(second)
    assert limit.used == 0 and not waiting.requests(include_reserved=True)[1]

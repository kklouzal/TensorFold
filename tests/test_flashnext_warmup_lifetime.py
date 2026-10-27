"""The actual warm caller must retire every bounded prefill/decode owner."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder, WARM_TAIL


class WarmModel:
    warm = MultiDecoder.warm
    _fill = MultiDecoder._fill
    _order = MultiDecoder._order
    _runnable = MultiDecoder._runnable
    _warm_grouped_experts = MultiDecoder._warm_grouped_experts

    def __init__(self, *, adaptive, rows, eos=False, failure=None, returned_error=False):
        self.w = SimpleNamespace(meta={"sm70_profile": "reference"})
        self.prefill_rows, self.capacity, self.depth = rows, 262147, 2
        self.admission_policy = "legacy"
        self.streams, self.fills, self.passed, self.held = {}, {}, {}, {}
        self.filling, self.free, self.kept = [], [], []
        self.graph_record, self.graphs = {"reason": None}, None
        self.events, self.pass_rows, self.rounds = [], [], 0
        self.eos, self.failure, self.returned_error = eos, failure, returned_error
        self.controller = SimpleNamespace(reset_observations=lambda: self.events.append("reset")) if adaptive else None
        self.state = object()

    def admit(self, stream):
        assert not hasattr(stream, "admission_ticket")
        self.synthetic = stream
        stream.sid, stream.st = 7, self.state
        self.filling.append(stream)
        self.fills[stream.sid] = [None, True, 0]
        self.passed[stream.sid] = 1

    def _pass(self):
        stream = self.filling[0]
        start = self.fills[stream.sid][2]
        count = min(self.prefill_rows, len(stream.prompt) - start)
        self.pass_rows.append(count)
        self.fills[stream.sid][2] += count
        if start + count == len(stream.prompt):
            self.filling.remove(stream)
            self.fills.pop(stream.sid)
            self.streams[stream.sid] = stream
            stream.take([0 if self.eos else 1], (0,) if stream.stop_eos else ())
            if self.returned_error:
                stream.done = True
                stream.error = self.failure if self.returned_error else None
        return []

    def round(self):
        self.rounds += 1
        if self.failure is not None and not self.returned_error:
            raise self.failure
        self._fill()
        stream = self.synthetic
        if not self.filling and not stream.done:
            stream.take([2], ())
        return [stream] if stream.done else []

    def arrived(self):
        return False

    def _drop_kept(self, state):
        assert state is self.state and not self.filling and not self.fills
        self.events.append("drop-kept")

    def _shrink(self, state):
        assert state is self.state and not self.filling and not self.fills
        self.events.append("shrink")

    def capture_graphs(self):
        assert not self.streams and not self.filling and not self.fills and not self.passed
        assert self.free == [self.state]
        self.events.append("capture")


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("rows", [128, 256])
@pytest.mark.parametrize("eos", [False, True])
def test_warm_finishes_full_and_partial_pass_before_fifo_policy(adaptive, rows, eos):
    model = WarmModel(adaptive=adaptive, rows=rows, eos=eos)
    model.warm()
    assert sum(model.pass_rows) == rows + WARM_TAIL + 1 == len(model.synthetic.prompt)
    assert model.pass_rows[0] == rows and model.pass_rows[-1] == WARM_TAIL + 1
    assert model.rounds == (2 if adaptive else 1)
    assert len(model.synthetic.out) == 2 and model.synthetic.out[0] == (0 if eos else 1)
    assert model.synthetic.stop_eos is False
    assert model.synthetic.done and not hasattr(model.synthetic, "admission_ticket")
    assert not model.streams and not model.filling and not model.fills and not model.passed and not model.held
    assert model.free == [model.state]
    assert model.events == ["drop-kept", "shrink", "capture"] + (["reset"] if adaptive else [])
    model.admission_policy = "fifo"
    assert model._order() == []


class Primary(BaseException):
    def __str__(self):
        raise AssertionError("warmup primary must not be formatted")


@pytest.mark.parametrize("returned", [False, True])
def test_warm_failure_is_not_published_as_ready_or_recycled(returned):
    primary = Primary()
    model = WarmModel(adaptive=True, rows=256, failure=primary, returned_error=returned)
    with pytest.raises(Primary) as caught:
        model.warm()
    assert caught.value is primary
    assert model.events == [] and not model.free


def test_done_flag_cannot_hide_an_unfinished_synthetic_prompt_owner():
    model = WarmModel(adaptive=True, rows=256)

    def premature():
        model.synthetic.done = True

    model.round = premature
    with pytest.raises(RuntimeError, match="prompt owner retired"):
        model.warm()
    assert model.filling == [model.synthetic] and model.synthetic.sid in model.fills
    assert model.events == [] and not model.free

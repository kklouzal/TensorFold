from types import SimpleNamespace

import pytest


def decoder_at_boundary(monkeypatch):
    torch = pytest.importorskip("torch")
    from tensorfold.families.deepseek_v4.cuda.engine import Decoder

    events = []
    device = object()
    decoder = object.__new__(Decoder)
    decoder.e = SimpleNamespace(part=None, layer_marks=None, image_features=None, tower_states={}, tower_spans={},
                                target=SimpleNamespace(device=device))
    decoder._broken = None
    decoder.streams, decoder.filling = {}, []
    decoder.inflight = decoder.ahead = decoder.split = decoder.finishing = decoder.timing = None
    decoder.timed = []

    def stream(bound_device):
        assert bound_device is device
        return SimpleNamespace(synchronize=lambda: events.append("fence"))

    monkeypatch.setattr(torch.cuda, "current_stream", stream)
    return decoder, events


def test_rebalance_drains_orphaned_run_ahead_before_retiring_timing(monkeypatch):
    decoder, events = decoder_at_boundary(monkeypatch)
    decoder.inflight = object()
    decoder.finishing = (None, None, None)
    decoder.timed = [object()]

    def collect():
        events.append("collect")
        decoder.inflight = None
        return []

    def absorb():
        assert decoder.finishing is None
        events.append("absorb")
        decoder.timed.clear()

    decoder._collect, decoder._absorb = collect, absorb
    decoder.drain_placement_work()
    assert events == ["collect", "fence", "absorb"]
    assert decoder.inflight is None and not decoder.timed


@pytest.mark.parametrize("owner", ("streams", "filling", "split", "ahead", "finishing", "part", "tower_states"))
def test_rebalance_refuses_active_work_before_fencing(monkeypatch, owner):
    decoder, events = decoder_at_boundary(monkeypatch)
    if owner in ("streams", "tower_states"):
        getattr(decoder if owner == "streams" else decoder.e, owner)[0] = object()
    elif owner == "filling":
        decoder.filling.append(object())
    elif owner == "part":
        decoder.e.part = object()
    else:
        setattr(decoder, owner, object())

    with pytest.raises(RuntimeError, match="requires drained"):
        decoder.drain_placement_work()
    assert events == []

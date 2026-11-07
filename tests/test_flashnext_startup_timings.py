"""Startup phase ownership controls, not device-completion or load-performance tests."""
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")


@pytest.fixture
def loader(monkeypatch):
    import torch
    from tensorfold.families.qwen4_exp.cuda import gguf_weights, sm70_mtp_weights

    clock = SimpleNamespace(now=10.0)
    events, seen = [], []
    primary = RuntimeError("injected phase failure")
    state = SimpleNamespace(failure=None, clock=clock, events=events, seen=seen, primary=primary)
    rope = SimpleNamespace(inverse_frequencies=lambda _torch: torch.ones(2), metadata=lambda: {"kind": "fixture"})
    prototype = SimpleNamespace(cfg=SimpleNamespace(rope=rope), meta={"rank": 0, "world": 1},
                                embed=None, layers=["layer"], mixer="mixer", head="head")

    def plan(*_args, **_kwargs):
        events.append("target-plan")
        clock.now += 2
        return prototype, None

    class Reader:
        def __init__(self):
            events.append("reader")

        def close(self):
            events.append("reader-close")

    def materialize(value):
        events.append(value)
        clock.now += 1
        if state.failure == "target" and value == "head":
            raise primary
        return value

    def attach(target, _path, *, tp):
        assert tp == (0, 1)
        assert target.meta["load_seconds"] == 5
        assert "stock_mtp_attach_seconds" not in target.meta
        seen.append(target)
        events.append("attach")
        clock.now += 7
        if state.failure == "attach":
            raise primary

    def vocabulary(target, _prototype, _reader, device):
        assert device == "cpu" and target.meta["stock_mtp_attach_seconds"] == 7
        assert "mtp_vocabulary_bind_seconds" not in target.meta
        events.append("vocabulary")
        clock.now += 3
        if state.failure == "vocabulary":
            raise primary

    monkeypatch.setattr(gguf_weights, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(gguf_weights, "model_plan", plan)
    monkeypatch.setattr(gguf_weights, "Reader", Reader)
    monkeypatch.setattr(gguf_weights, "materializer", lambda *_a, **_k: materialize)
    monkeypatch.setattr(sm70_mtp_weights, "attach", attach)
    monkeypatch.setattr(gguf_weights, "_bind_mtp_vocabulary", vocabulary)
    # Existing failure retirement fences remain visible. Timing adds no fence.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append(("fence", device)))
    state.load = lambda mtp: gguf_weights.load_gguf(Path("timing-boundary.gguf"), "cpu", mtp=mtp,
                                                   mtp_model_dir=Path("companion.gguf") if mtp else None)
    return state


@pytest.mark.parametrize("mtp", [False, True])
def test_successful_loader_phases_keep_original_target_interval(loader, mtp):
    loaded = loader.load(mtp)
    assert loaded.meta["load_seconds"] == 5  # target planning2 + three materializations; excludes all MTP work
    if mtp:
        assert loaded.meta["stock_mtp_attach_seconds"] == 7
        assert loaded.meta["mtp_vocabulary_bind_seconds"] == 3
        assert loader.events[-3:] == ["attach", "vocabulary", "reader-close"]
    else:
        assert "stock_mtp_attach_seconds" not in loaded.meta
        assert "mtp_vocabulary_bind_seconds" not in loaded.meta
    assert loader.events.count("reader-close") == 1
    assert not any(isinstance(event, tuple) and event[0] == "fence" for event in loader.events)
    loaded.meta["ple_tables"].close(lambda: None)


@pytest.mark.parametrize("phase", ["target", "attach", "vocabulary"])
def test_failure_never_publishes_success_for_the_unfinished_phase(loader, phase):
    loader.failure = phase
    with pytest.raises(RuntimeError) as caught:
        loader.load(True)
    assert caught.value is loader.primary and caught.value.__cause__ is None
    assert loader.events.count("reader-close") == 1
    assert loader.events.count(("fence", "cpu")) == 1  # unchanged existing cleanup boundary
    if phase == "target":
        assert not loader.seen and "attach" not in loader.events
    else:
        meta = loader.seen[0].meta
        assert meta["load_seconds"] == 5
        assert "mtp_vocabulary_bind_seconds" not in meta
        if phase == "attach":
            assert "stock_mtp_attach_seconds" not in meta and "vocabulary" not in loader.events
        else:
            assert meta["stock_mtp_attach_seconds"] == 7


def test_constructor_diagnostics_do_not_enter_prepared_profile_options(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    from tensorfold.families.qwen4_exp.cuda import serving_runtime

    captured = []
    counter = iter((3.0, 9000.0))
    def initialize(self, *_args, **_kwargs):
        self.startup_timings = {"weights_call_seconds": next(counter)}
    def profiles(engine, root, options):
        assert engine.startup_timings and root == tmp_path
        captured.append(dict(options))
        return SimpleNamespace()
    monkeypatch.setattr(FlashNextEngine, "_initialize", initialize)
    monkeypatch.setattr(serving_runtime, "ServingProfiles", profiles)
    first = FlashNextEngine(tmp_path, serving_profile_root=tmp_path, adaptive_sessions=True)
    second = FlashNextEngine(tmp_path, serving_profile_root=tmp_path, adaptive_sessions=True)
    assert first.startup_timings != second.startup_timings
    assert captured[0] == captured[1]
    assert not any("timing" in key or "started" in key or "seconds" in key for key in captured[0])

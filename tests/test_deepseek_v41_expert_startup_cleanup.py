"""Opt-in DeepSeek startup failures retire the resources actually acquired."""

import os
import sys
from threading import Event
from types import SimpleNamespace

import pytest


def test_optional_constructor_retires_partial_cpu_owners(monkeypatch):
    from tensorfold.cuda import cohort
    from tensorfold.families.deepseek_v4.cuda.engine import DS41Engine

    order = []
    failure = ValueError("bad expert history acknowledgment")

    def partial(engine, model, *, startup, **options):
        assert options["expert_balance_state"] == "history.json"
        startup.shards = SimpleNamespace(close=lambda: order.append("shards"))
        startup.history = SimpleNamespace(close=lambda: order.append("history"))
        raise failure

    monkeypatch.setattr(DS41Engine, "_initialize", partial)
    monkeypatch.setattr(cohort, "notify_failure", lambda: order.append("notify"))
    with pytest.raises(ValueError) as caught:
        DS41Engine("model", rank=0, tp=8, serve=True, expert_balance_state="history.json")
    assert caught.value is failure
    assert order == ["notify", "shards", "history"]


def test_default_constructor_keeps_original_initialization(monkeypatch):
    from tensorfold.families.deepseek_v4.cuda.engine import DS41Engine

    def primary(engine, model, *, startup, **options):
        assert model == "model" and startup is None
        assert options["rank"] == 1 and options["tp"] == 8

    monkeypatch.setattr(DS41Engine, "_initialize", primary)
    DS41Engine("model", rank=1, tp=8)


def test_mapper_can_duplicate_after_original_shards_close(monkeypatch, tmp_path):
    from tensorfold.cuda import cohort
    from tensorfold.families.deepseek_v4.cuda.startup import ExpertStartup

    model = tmp_path / "model.bin"
    model.write_bytes(b"same inode")
    startup = ExpertStartup()
    original = os.open(model, os.O_RDONLY)
    lease = os.dup(original)
    begin, retired = Event(), Event()
    order = []

    def close_shards():
        os.close(original)
        order.append("shards")
        retired.set()

    def construct(open_fd):
        begin.set()
        assert retired.wait(5)
        with os.fdopen(open_fd(model), "rb") as stream:
            assert stream.read() == b"same inode"
        order.append("mapper")

    startup.shards = SimpleNamespace(close=close_shards)
    monkeypatch.setattr(cohort, "notify_failure", lambda: order.append("notify"))
    j = startup.map_engram(lease, construct)
    assert begin.wait(5)
    startup.abort(SimpleNamespace())
    j.result()
    assert order == ["notify", "shards", "mapper"]
    assert startup.mapper is startup.engram_fd is None
    with pytest.raises(OSError):
        os.fstat(lease)


def test_failed_device_fence_retains_native_owner(monkeypatch):
    from tensorfold.cuda import cohort
    from tensorfold.families.deepseek_v4.cuda.startup import ExpertStartup

    startup = ExpertStartup()
    events = []
    engine = SimpleNamespace(graphs=object())
    original_graphs = engine.graphs
    startup.observer = SimpleNamespace(abort_startup=lambda: events.append("observer"))
    startup.engram = SimpleNamespace(close=lambda: events.append("engram"))

    def fence():
        events.append("fence")
        raise RuntimeError("device lost")

    monkeypatch.setattr(cohort, "notify_failure", lambda: events.append("notify"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=fence)))
    with pytest.raises(RuntimeError, match="device lost"):
        startup.abort(engine)
    assert events == ["notify", "observer", "fence"]
    assert startup.engine is engine and engine.graphs is original_graphs


@pytest.mark.parametrize("foreground_fails", [True, False])
def test_constructor_keeps_concurrent_mapper_failure_once(monkeypatch, tmp_path, foreground_fails):
    from tensorfold.cuda import cohort
    from tensorfold.families.deepseek_v4.cuda.engine import DS41Engine

    target = tmp_path / "engram.bin"
    target.write_bytes(b"table")
    fd = os.open(target, os.O_RDONLY)
    foreground = ValueError("weights failed")
    background = RuntimeError("mapper failed after acquisition")
    ready, release = Event(), Event()
    events = []

    def partial(engine, model, *, startup, **options):
        def construct(open_fd):
            with os.fdopen(open_fd(target), "rb") as stream:
                assert stream.read() == b"table"
            startup.engram = SimpleNamespace(close=lambda: events.append("engram"))
            ready.set()
            assert release.wait(5)
            raise background

        future = startup.map_engram(fd, construct)
        assert ready.wait(5)
        if foreground_fails:
            # Let the already accepted mapper finish during the rollback.
            raise foreground
        release.set()
        future.result()

    def notify():
        release.set()

    monkeypatch.setattr(DS41Engine, "_initialize", partial)
    monkeypatch.setattr(cohort, "notify_failure", notify)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda: events.append("fence"))))
    with pytest.raises((ValueError, RuntimeError)) as caught:
        DS41Engine("model", rank=0, tp=8, expert_placement="placement.json")
    if foreground_fails:
        assert caught.value is foreground and caught.value.__cause__ is background
    else:
        assert caught.value is background and caught.value.__cause__ is None
    assert events == ["fence", "engram"]
    with pytest.raises(OSError):
        os.fstat(fd)


def test_failed_mapper_join_retains_job_lease_and_native_state(monkeypatch, tmp_path):
    from concurrent.futures import Future
    from tensorfold.cuda import cohort
    from tensorfold.families.deepseek_v4.cuda.startup import ExpertStartup

    startup = ExpertStartup()
    job = startup.engram_job = Future()
    fd = startup.engram_fd = os.open(tmp_path, os.O_RDONLY)
    events = []
    startup.engram = SimpleNamespace(close=lambda: events.append("engram"))

    def failed_join(**options):
        assert options == {"wait": True, "cancel_futures": True}
        raise RuntimeError("mapper join failed")

    startup.mapper = SimpleNamespace(shutdown=failed_join)
    monkeypatch.setattr(cohort, "notify_failure", lambda: events.append("notify"))
    try:
        with pytest.raises(RuntimeError, match="mapper join failed"):
            startup.abort(SimpleNamespace())
        assert events == ["notify"]
        assert startup.mapper is not None and startup.engram_job is job and startup.engram_fd == fd
        os.fstat(fd)
    finally:
        os.close(fd)

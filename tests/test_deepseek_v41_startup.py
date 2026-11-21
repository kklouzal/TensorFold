"""DeepSeek-V4.1 SM70 startup helpers on CPU: the phase timeline and the parallel extension prebuild (every module's
``extension()`` runs once, concurrently; a build failure re-raises at ``wait``)."""

import sys
import threading
import time
import types

import pytest

from tensorfold.families.deepseek_v4.cuda.startup import Prebuild, Timeline


def _fake(name, seconds, error=None, calls=None):
    module = types.ModuleType(name)

    def extension():
        calls.append((name, threading.get_ident()))
        time.sleep(seconds)
        if error:
            raise error
    module.extension = extension
    sys.modules[name] = module
    return name


def test_timeline_phases_and_notes():
    clock = Timeline(3)
    with clock.phase("engram") as note:
        note[0] = "fill 1.0"
    clock.add("graphs", time.perf_counter(), 2.0)
    line = clock.line()
    assert line.startswith("[tensorfold] rank 3 startup (")
    assert "engram (fill 1.0) @" in line and "graphs @" in line and "2.0s" in line


def test_prebuild_runs_every_extension_once_in_parallel():
    calls = []
    names = [_fake(f"_ds41_fake_ext{i}", 0.2, calls=calls) for i in range(4)]
    clock = Timeline(0)
    started = time.perf_counter()
    prebuild = Prebuild(clock, names)
    prebuild.wait(names[0])
    prebuild.wait()
    assert time.perf_counter() - started < 0.6          # four 0.2 s builds at once, not one after another
    assert sorted(n for n, _ in calls) == sorted(names) and len({t for _, t in calls}) == 4
    assert "extensions (slowest _ds41_fake_ext" in clock.line()
    prebuild.wait()                                     # idempotent: no second timeline row
    assert clock.line().count("extensions (") == 1


def test_prebuild_reraises_a_build_failure():
    calls = []
    good, bad = _fake("_ds41_fake_good", 0.0, calls=calls), _fake("_ds41_fake_bad", 0.0, RuntimeError("nvcc"), calls)
    prebuild = Prebuild(Timeline(0), [good, bad])
    prebuild.wait(good)
    with pytest.raises(RuntimeError, match="nvcc"):
        prebuild.wait()


@pytest.mark.parametrize("rank", [0, 1])
def test_engram_unmaps_after_device_retirement(monkeypatch, rank):
    torch = pytest.importorskip("torch")
    from tensorfold.families.deepseek_v4.cuda.engine import DS41Engine

    e = object.__new__(DS41Engine)
    e._closed, e.rank, e.graphs = False, rank, object()
    actions = []

    def unmap():
        assert e.graphs is None
        actions.append("unmap")

    e.scheduler = types.SimpleNamespace(close=lambda: actions.append("drain")) if rank == 0 else None
    e.pipes = types.SimpleNamespace(send=lambda _: actions.append("stop peers"))
    e.engram = types.SimpleNamespace(close=unmap)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: actions.append("device fence"))
    e.close()
    e.close()
    assert actions == (["drain", "stop peers"] if rank == 0 else []) + ["device fence", "unmap"]


def test_engram_stays_mapped_if_device_retirement_fails(monkeypatch):
    torch = pytest.importorskip("torch")
    from tensorfold.families.deepseek_v4.cuda.engine import DS41Engine

    e = object.__new__(DS41Engine)
    graph = object()
    e._closed, e.rank, e.graphs = False, 1, graph
    e.engram = types.SimpleNamespace(close=lambda: pytest.fail("released unretired mapping"))

    def fail():
        raise RuntimeError("device progress failed")

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    with pytest.raises(RuntimeError, match="device progress failed"):
        e.close()
    assert e.graphs is graph

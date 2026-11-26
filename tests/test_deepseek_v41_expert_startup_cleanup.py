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


@pytest.mark.parametrize(("variable", "flag"), [
    ("TENSORFOLD_DS41_EXPERIMENT_W576_FP32", "w576_fp32"),
    ("TENSORFOLD_DS41_EXPERIMENT_SP_FP32", "sp_fp32"),
])
def test_experimental_constructor_owns_startup_without_balance_state(monkeypatch, variable, flag):
    from tensorfold.families.deepseek_v4.cuda.engine import DS41Engine
    from tensorfold.families.deepseek_v4.cuda.startup import ExpertStartup

    monkeypatch.setenv(variable, "1")

    def initialize(engine, model, *, startup, **arguments):
        assert isinstance(startup, ExpertStartup)
        assert arguments[flag] and arguments["expert_balance_state"] is None
        assert not arguments["expert_balance_live"]

    monkeypatch.setattr(DS41Engine, "_initialize", initialize)
    DS41Engine("model", rank=0, tp=8, serve=(flag == "w576_fp32"), no_expert_balance=True)


def test_wrong_agreed_host_tier_refused_before_pin_can_start(monkeypatch):
    import concurrent.futures
    from tensorfold.families.deepseek_v4.cuda import prefix
    from tensorfold.families.deepseek_v4.cuda.startup import ExpertStartup

    tensor = lambda values, **options: values
    empty = lambda length, **options: SimpleNamespace(min=lambda: 1 << 30)
    monkeypatch.setattr(prefix, "torch", SimpleNamespace(tensor=tensor, empty=empty, int64=object()))
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", lambda *args, **kwargs: pytest.fail("pin started"))
    comm = SimpleNamespace(all_gather=lambda mine, agreed: None)
    startup = ExpertStartup()
    with pytest.raises(ValueError, match="host prefix bytes per rank"):
        prefix.HostSlab(comm, 0, 8, 16 << 30, required_per_rank=2 << 30, startup=startup)
    assert startup.slab is None


def test_abort_joins_experimental_pin_before_releasing_native_owners(monkeypatch):
    from tensorfold.cuda import cohort
    from tensorfold.families.deepseek_v4.cuda.startup import ExpertStartup

    startup, events = ExpertStartup(), []
    failure = RuntimeError("asynchronous pin failed")

    def outcome():
        events.append("pin result")
        raise failure

    def join(*, cancel):
        assert cancel
        events.append("pin join")

    startup.slab = SimpleNamespace(join_pin=join, report_pin_error=outcome)
    startup.engram = SimpleNamespace(close=lambda: events.append("engram"))
    monkeypatch.setattr(cohort, "notify_failure", lambda: events.append("notify"))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        synchronize=lambda: events.append("fence"))))
    with pytest.raises(RuntimeError) as caught:
        startup.abort(SimpleNamespace())
    assert caught.value is failure and startup.slab is None
    assert events == ["notify", "pin join", "pin result", "fence", "engram"]


@pytest.mark.parametrize("interrupted", [False, True])
def test_pin_wait_keeps_any_distinct_worker_failure_for_abort(monkeypatch, interrupted):
    from concurrent.futures import Future
    from tensorfold.families.deepseek_v4.cuda.prefix import HostSlab
    from tensorfold.families.qwen4_exp.cuda import prefix_host

    worker_failure = RuntimeError("pin worker failed")
    interrupt = KeyboardInterrupt("pin wait interrupted")
    future = Future()
    future.set_exception(worker_failure)
    if interrupted:
        original = future.result

        def interrupt_once():
            future.result = original
            raise interrupt

        future.result = interrupt_once
    slab = object.__new__(HostSlab)
    slab._pin, slab._pin_worker, slab._pin_observed = future, object(), False
    slab.tp, slab.rank, slab.total, slab.explicit, slab.per_rank = 8, 0, 16 << 30, True, 2 << 30
    monkeypatch.setattr(prefix_host, "pin_agreed", lambda allocate, *args, **kwargs: allocate())
    with pytest.raises((RuntimeError, KeyboardInterrupt)) as caught:
        slab.finish()
    assert caught.value is (interrupt if interrupted else worker_failure)
    assert slab._pin_observed is not interrupted
    if interrupted:
        with pytest.raises(RuntimeError) as pending:
            slab.report_pin_error()
        assert pending.value is worker_failure
    else:
        slab.report_pin_error()
    assert slab._pin is None


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

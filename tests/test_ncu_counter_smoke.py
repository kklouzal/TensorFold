"""Counter-smoke commands and independent finite outcome controls; no GPU."""
from pathlib import Path
import pytest
from tools import ncu_counter_smoke as smoke


def test_plan_uses_one_stateless_kernel_and_preserves_host_controls():
    argv = smoke.command(Path("/immutable/probe.py"), Path("/evidence/fresh"))
    def option(name):
        return argv[argv.index(name) + 1]
    assert option("--clock-control") == option("--cache-control") == "none"
    assert option("--launch-count") == "1" and option("--launch-skip") == "0"
    assert option("--profile-from-start") == "off"
    assert option("--target-processes") == "application-only"
    assert option("--replay-mode") == "kernel"
    assert option("--config-file") == "off" and option("--rename-kernels") == "off"
    assert option("--metrics") == ",".join(smoke.METRICS)
    assert "--kill" not in argv and "--run-smoke" in argv


def test_primary_and_profiler_stop_failures_are_both_preserved():
    pytest.importorskip("torch")
    primary, secondary = ValueError("original kernel enqueue failed"), RuntimeError("profiler stop failed")
    calls = []
    def operation():
        raise primary
    def stop():
        calls.append("stop")
        raise secondary
    with pytest.raises(ValueError) as caught:
        smoke.profiled_call(operation, lambda: calls.append("start"), stop)
    assert caught.value is primary and calls == ["start", "stop"]
    assert secondary is primary.__context__ or secondary is primary.__cause__ or primary.__notes__


def test_failed_start_is_not_followed_by_unsupported_stop():
    pytest.importorskip("torch")
    error = RuntimeError("start refused")
    def start():
        raise error
    with pytest.raises(RuntimeError) as caught:
        smoke.profiled_call(lambda: pytest.fail("operation after failed start"), start,
                            lambda: pytest.fail("stop after failed start"))
    assert caught.value is error


def test_csv_requires_actual_counter_rows_and_keeps_legitimate_zero_dram():
    raw = '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n' +           '"vectorized_elementwise_kernel","sm__cycles_elapsed.avg","cycle","1,234"\n' +           '"vectorized_elementwise_kernel","gpu__time_duration.sum","ns","123.5"\n' +           '"vectorized_elementwise_kernel","dram__bytes_read.sum","byte" ,"0"\n'
    assert smoke.parse_counters(raw) == {"sm__cycles_elapsed.avg": 1234.,
        "gpu__time_duration.sum": 123.5, "dram__bytes_read.sum": 0.}
    assert smoke.parse_counters("==WARNING== No kernels profiled") == {}


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_nonfinite_or_negative_hardware_counter_is_not_a_pass(value):
    text = '"Metric Name","Metric Value"\n"sm__cycles_elapsed.avg","' + value + '"\n'
    with pytest.raises(ValueError):
        smoke.parse_counters(text)

def test_profiler_outcome_storage_exists_before_start_and_is_not_repacked(monkeypatch):
    import sys
    cleanup = pytest.importorskip("tensorfold.cleanup")
    original = cleanup.raise_failures
    observed = []
    primary, secondary = ValueError("operation failed"), MemoryError("stop failed")
    def start():
        slots = sys._getframe(1).f_locals["secondary"]
        assert slots == [None]
        observed.append(slots)
    def operation():
        raise primary
    def stop():
        raise secondary
    def transport(error, errors):
        assert errors is observed[0] and errors == [secondary]
        original(error, errors)
    monkeypatch.setattr(cleanup, "raise_failures", transport)
    with pytest.raises(ValueError) as caught:
        smoke.profiled_call(operation, start, stop)
    assert caught.value is primary

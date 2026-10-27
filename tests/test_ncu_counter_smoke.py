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

def test_successful_profiled_call_returns_actual_result_in_start_operation_stop_order():
    pytest.importorskip("torch")
    result, calls = object(), []
    def operation():
        calls.append("operation")
        return result
    got = smoke.profiled_call(operation, lambda: calls.append("start"), lambda: calls.append("stop"))
    assert got is result and calls == ["start", "operation", "stop"]


def test_stop_only_error_uses_actual_profiler_failure_without_none_primary_transport():
    pytest.importorskip("torch")
    error = RuntimeError("profiler stop alone failed")
    def stop():
        raise error
    with pytest.raises(RuntimeError) as caught:
        smoke.profiled_call(lambda: 123, lambda: None, stop)
    assert caught.value is error

def test_actual_gv100_exported_raw_report_wide_csv_and_units():
    path = Path(__file__).parent / "fixtures/ncu_smoke_gv100_2025_2.csv"
    assert smoke.sha(path) == "2ee619a7d93f69a260bfdc57a05332e1f34e6dcf94b2e55a664dfd7b050a206f"
    counters, units = smoke.parse_counter_table(path.read_text())
    assert counters == {"gpu__time_duration.sum": 2.88, "sm__cycles_elapsed.avg": 3723.825,
                        "dram__bytes_read.sum": 640.}
    assert units == {"gpu__time_duration.sum": "us", "sm__cycles_elapsed.avg": "cycle",
                     "dram__bytes_read.sum": "byte"}
    assert smoke.parse_counters(path.read_text()) == counters


def test_saved_report_import_is_cpu_read_only_and_explicit_raw_csv():
    argv = smoke.import_command(Path("/evidence/smoke.ncu-rep"), Path("/evidence/counters.csv"))
    assert argv == [smoke.NCU, "--config-file", "off", "--import", "/evidence/smoke.ncu-rep",
                    "--csv", "--page", "raw", "--log-file", "/evidence/counters.csv"]
    assert "--run-smoke" not in argv and "--launch-count" not in argv


def test_more_than_one_wide_kernel_is_not_accepted_as_one_launch_smoke():
    text = '"ID","gpu__time_duration.sum","sm__cycles_elapsed.avg","dram__bytes_read.sum"\n' +            '"","us","cycle","byte"\n"0","2.88","3723.825","640"\n"1","1","2","3"\n'
    with pytest.raises(ValueError, match="exactly one"):
        smoke.parse_counter_table(text)

"""Actual closed parser/timeline integration and own failure contracts, CPU only."""

import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from tools import rescan_cpu_parent as scan


@pytest.fixture
def helpers():
    return scan.trusted_helpers(Path(os.environ["TENSORFOLD_TEST_ANALYZER"]))


def event(name, ts, dur, *, cat="cpu_op", tid=1, correlation=None):
    return dict(
        ph="X",
        cat=cat,
        name=name,
        ts=ts,
        dur=dur,
        tid=tid,
        args={} if correlation is None else {"correlation": correlation},
    )


def run_actual(tmp_path, helpers, events):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"baseTimeNanoseconds": 17, "traceEvents": events}))
    result, stream = scan.scan(path, helpers, 105, 190)
    assert stream.file.closed
    return result


def test_nested_project_thread_clip_and_out_of_order_events_attributed_once(tmp_path, helpers):
    events = [
        event("aten::outer", 110, 55),
        event("aten::inner", 125, 4),
        event("aten::cross_start", 95, 15),
        event("aten::cross_end", 185, 20),
        event("aten::other", 130, 5, tid=2),
        event("aten::unlabelled", 130, 2, tid=3),
        event("ignoredGPU", 120, 5, cat="kernel"),
        event("tensorfold/body/compute", 100, 100, cat="user_annotation"),
        event("tensorfold/cpu/descriptor", 120, 30, cat="user_annotation"),
        event("tensorfold/body/other", 110, 60, cat="user_annotation", tid=2),
    ]
    result = run_actual(tmp_path, helpers, events)
    by_name = {row["name"]: row for row in result["groups"]}
    assert sum(row["count"] for row in result["groups"]) == 6 == result["selected_events"]
    assert by_name["aten::outer"]["parent"] == "tensorfold/body/compute"
    assert by_name["aten::inner"]["parent"] == "tensorfold/cpu/descriptor"
    assert by_name["aten::cross_start"]["inclusive_us"] == 5
    assert by_name["aten::cross_end"]["inclusive_us"] == 5
    assert by_name["aten::other"]["parent"] == "tensorfold/body/other"
    assert by_name["aten::unlabelled"]["parent"] == "outside_project_labels"
    assert by_name["aten::outer"]["inclusive_us"] == 55  # Inclusive; nested sums are not CPUbusy.


def test_runtime_correlation_not_double_assigned(tmp_path, helpers):
    events = [
        event("tensorfold/body/compute", 100, 100, cat="user_annotation"),
        event("cudaLaunchKernel", 120, 5, cat="cuda_runtime", correlation=4),
    ]
    result = run_actual(tmp_path, helpers, events)
    assert result["groups"] == [
        {
            "parent": "tensorfold/body/compute",
            "category": "cuda_runtime",
            "name": "cudaLaunchKernel",
            "count": 1,
            "inclusive_us": 5.0,
            "maximum_us": 5.0,
        }
    ]
    events.append(event("cudaLaunchKernel", 130, 5, cat="cuda_runtime", correlation=4))
    with pytest.raises(ValueError, match="duplicate runtime correlation"):
        run_actual(tmp_path, helpers, events)


def test_event_cap_refuses_without_truncation_and_retires_file(tmp_path, helpers, monkeypatch):
    monkeypatch.setattr(scan, "MAX_ROWS", 1)
    path = tmp_path / "trace.json"
    path.write_text(json.dumps({"traceEvents": [event("a", 110, 1), event("b", 112, 1)]}))
    owners = []
    original = helpers.Stream.__init__

    def observed(owner, source):
        original(owner, source)
        owners.append(owner)

    monkeypatch.setattr(helpers.Stream, "__init__", observed)
    with pytest.raises(ValueError, match="events exceed"):
        scan.scan(path, helpers, 105, 190)
    assert len(owners) == 1 and owners[0].file.closed


def test_actual_scan_preserves_primary_and_native_roots_on_close_failure(helpers):
    from tests.test_v100_backend_probe import retained_failure

    primary, secondary = ValueError("parse"), OSError("close")
    cause, context = RuntimeError("nativecause"), LookupError("nativecontext")
    primary.__cause__, primary.__context__ = cause, context

    class Stream:
        def __init__(self, _path):
            self.file = NS(close=lambda: (_ for _ in ()).throw(secondary))

        def events(self, _header):
            raise primary

    with pytest.raises(ValueError) as caught:
        scan.scan(Path("unused"), NS(Stream=Stream, timelines=helpers.timelines, phase_at=helpers.phase_at), 105, 190)
    assert caught.value is primary
    assert all(retained_failure(primary, e) for e in (secondary, cause, context))


def test_actual_write_failure_keeps_primary_and_close_failure(monkeypatch):
    from tests.test_v100_backend_probe import retained_failure

    primary, secondary = OSError("write"), OSError("close")
    native = ValueError("priorcause")
    primary.__cause__ = native

    class Output:
        def write(self, _text):
            raise primary

        def close(self):
            raise secondary

    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: Output())
    with pytest.raises(OSError) as caught:
        scan.write_result(Path("unused"), "value")
    assert caught.value is primary
    assert retained_failure(primary, secondary) and retained_failure(primary, native)


def test_positive_actual_write_preserves_fresh_output_contract(tmp_path):
    path = tmp_path / "result.json"
    scan.write_result(path, "{}\n")
    assert path.read_text() == "{}\n"
    with pytest.raises(FileExistsError):
        scan.write_result(path, "overwrite")


def test_untrusted_helper_source_refuses(tmp_path):
    path = tmp_path / "helper.py"
    path.write_text('raise RuntimeError("must not import")')
    with pytest.raises(ValueError, match="helper source differs"):
        scan.trusted_helpers(path)


def test_actual_current_capture_metadata_and_closed_core_before_rescan(tmp_path):
    metadata = Path(os.environ["TENSORFOLD_TEST_CAPTURE"])
    reports = Path(os.environ["TENSORFOLD_TEST_REPORTS"])
    (tmp_path / "results").mkdir()
    (tmp_path / "results/run").symlink_to(metadata / "capture/run", target_is_directory=True)
    for name in ("candidate.json", "controller-progress.json"):
        (tmp_path / name).symlink_to(metadata / "controller" / name)
    folder = tmp_path / "analysis-concurrent-v1"
    folder.mkdir()
    for name in ("all8-summary.json", "window-definition.json"):
        (folder / name).write_bytes((reports / name).read_bytes())
    path, profile, definition, start, end = scan.capture_context(tmp_path)
    assert path.name == "rank-0-trace.json" and profile["rank"] == 0
    assert 0 < end - start <= 10_000_000 and definition["offered"] == 24
    changed = json.loads((folder / "window-definition.json").read_text())
    changed["replies"][0]["first_commit_upper_us"] += 1
    (folder / "window-definition.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="core differs from closed"):
        scan.capture_context(tmp_path)


def test_actual_scan_setup_memoryerror_happens_before_any_file_open(tmp_path, helpers, monkeypatch):
    primary = MemoryError("setup")
    opened = []
    original = helpers.Stream.__init__

    def observe(owner, path):
        opened.append(path)
        original(owner, path)

    def fail_setup(*_args):
        raise primary

    monkeypatch.setattr(helpers.Stream, "__init__", observe)
    monkeypatch.setattr(scan, "defaultdict", fail_setup)
    with pytest.raises(MemoryError) as caught:
        scan.scan(tmp_path / "not-opened", helpers, 105, 190)
    assert caught.value is primary and opened == []


def test_actual_stream_constructor_fault_after_open_retires_file(tmp_path, helpers, monkeypatch):
    path = tmp_path / "trace.json"
    path.write_text('{"traceEvents":[]}')
    primary = MemoryError("after open")
    owners = []
    original = helpers.Stream.__init__

    def fail(owner, source):
        original(owner, source)
        owners.append(owner)
        raise primary

    monkeypatch.setattr(helpers.Stream, "__init__", fail)
    with pytest.raises(MemoryError) as caught:
        scan.scan(path, helpers, 105, 190)
    assert caught.value is primary and len(owners) == 1 and owners[0].file.closed

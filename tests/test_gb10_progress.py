"""Owned heartbeat/hot-refusal integration; no device or upstream algorithm tests."""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import gb10_progress as progress


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    monkeypatch.setattr(progress.Progress, "path", None)
    monkeypatch.setattr(progress.Progress, "state", {})
    monkeypatch.setattr(progress.Progress, "written", 0.0)
    monkeypatch.setattr(progress.HotGate, "path", None)
    monkeypatch.setattr(progress.HotGate, "max_s", 600.0)
    monkeypatch.setattr(progress.HotGate, "waited_s", 0.0)


def leaves(error):
    # The qualified transport keeps the original primary and attaches cleanup
    # through native causes; groups may contain a reference back to that root.
    pending, seen, result = [error], set(), []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        result.append(current)
        assert len(seen) <= 8, "this fixture owns only its bounded raw failure graph"
        cause = BaseException.__cause__.__get__(current)
        if cause is not None:
            pending.append(cause)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
    return result


def test_progress_retains_format_cadence_carried_fields_and_atomic_replace(tmp_path, monkeypatch):
    ticks = iter((10.0, 10.5, 11.5, 11.6))
    monkeypatch.setattr(progress, "time", SimpleNamespace(time=lambda: next(ticks)))
    monkeypatch.setattr(progress, "os", SimpleNamespace(getpid=lambda: 123, replace=os.replace))
    output = tmp_path / "progress.json"
    progress.Progress.path = str(output)
    progress.Progress.update("prefill", request=7, prompt_tokens=48, max_tokens=64)
    first = output.read_bytes()
    assert first == (b'{"phase": "prefill", "phase_since": 10.0, "request": 7, '
                     b'"prompt_tokens": 48, "max_tokens": 64, "last_event": 10.0, "pid": 123}')
    progress.Progress.update("prefill", chunks=1)
    assert output.read_bytes() == first
    progress.Progress.update("prefill", chunks=2)
    assert json.loads(output.read_bytes())["chunks"] == 2
    progress.Progress.update("decode")
    assert json.loads(output.read_bytes()) == {"phase": "decode", "phase_since": 11.6,
        "request": 7, "prompt_tokens": 48, "max_tokens": 64, "last_event": 11.6, "pid": 123}
    assert not (tmp_path / "progress.json.123.tmp").exists()


def test_disabled_progress_does_not_read_clock_or_open_stream(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("disabled heartbeat performed work")
    monkeypatch.setattr(progress, "time", SimpleNamespace(time=forbidden))
    monkeypatch.setattr(progress, "opened_file", forbidden)
    progress.Progress.configure(None)
    progress.Progress.update("decode", request=1)


@pytest.mark.parametrize("fail_body", (False, True))
def test_progress_preserves_write_and_close_roots_without_publication(tmp_path, monkeypatch, fail_body):
    write_error, close_error = OSError("write failed"), OSError("close failed")
    replaces = []
    stream = SimpleNamespace(close_calls=0)

    def write(_text):
        if fail_body:
            raise write_error

    def close():
        stream.close_calls += 1
        raise close_error

    stream.write, stream.close = write, close
    original_open = Path.open
    temporary = tmp_path / "progress.json.123.tmp"

    def selected_open(path, *args, **kwargs):
        if path == temporary:
            assert args == ("w",) and kwargs == {"encoding": "utf-8"}
            return stream
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(progress, "time", SimpleNamespace(time=lambda: 10.0))
    monkeypatch.setattr(progress, "os", SimpleNamespace(getpid=lambda: 123,
        replace=lambda *args: replaces.append(args)))
    progress.Progress.path = str(tmp_path / "progress.json")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", selected_open)
        with pytest.raises(BaseException) as caught:
            progress.Progress.update("decode")
    assert stream.close_calls == 1 and replaces == [] and progress.Progress.written == 0.0
    assert close_error in leaves(caught.value)
    if fail_body:
        assert write_error in leaves(caught.value)
    else:
        assert caught.value is close_error


def test_progress_retains_existing_temporary_truncation_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "time", SimpleNamespace(time=lambda: 10.0))
    monkeypatch.setattr(progress, "os", SimpleNamespace(getpid=lambda: 123, replace=os.replace))
    temporary = tmp_path / "progress.json.123.tmp"
    temporary.write_text("old bytes must not remain" * 10)
    temporary.chmod(0o640)
    output = tmp_path / "progress.json"
    progress.Progress.path = str(output)
    progress.Progress.update("idle")
    assert json.loads(output.read_bytes())["phase"] == "idle"
    assert output.stat().st_mode & 0o777 == 0o640


def timeout_gate(tmp_path, monkeypatch):
    path = tmp_path / "hot"
    path.write_bytes(b"over temperature\r\n")
    progress.HotGate.path = path
    progress.HotGate.max_s = 600.0
    ticks = iter((1.0, 601.0))
    monkeypatch.setattr(progress, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    return path


def test_hot_timeout_retains_reason_and_default_text_read(tmp_path, monkeypatch):
    timeout_gate(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match=r"host stayed hot for 600 s \(over temperature\): no new prompt"):
        progress.HotGate.wait()


@pytest.mark.parametrize("kind", ("missing", "unreadable"))
@pytest.mark.parametrize("stage", ("open", "read"))
def test_hot_timeout_retains_single_read_failure_policy(tmp_path, monkeypatch, kind, stage):
    path = timeout_gate(tmp_path, monkeypatch)
    original_open = Path.open
    closes = []

    def read_failure():
        if kind == "missing":
            raise FileNotFoundError("cooled during read")
        raise OSError("reason unavailable")

    def selected_open(value, *args, **kwargs):
        if value == path:
            if stage == "open":
                return read_failure()
            return SimpleNamespace(read=read_failure, close=lambda: closes.append(True))
        return original_open(value, *args, **kwargs)

    # A cooled-between-stat-and-open case reaches the original elapsed clock.
    ticks = iter((1.0, 601.0, 602.0))
    monkeypatch.setattr(progress, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", selected_open)
        if kind == "missing":
            assert progress.HotGate.wait() == 601.0
        else:
            with pytest.raises(RuntimeError, match="hot-state reason unreadable: OSError"):
                progress.HotGate.wait()
    assert closes == ([True] if stage == "read" else [])


@pytest.mark.parametrize("failure_kind", ("read_oserror", "read_missing", "close_missing"))
def test_hot_timeout_preserves_read_and_close_roots_and_never_retries(tmp_path, monkeypatch, failure_kind):
    path = timeout_gate(tmp_path, monkeypatch)
    read_error = FileNotFoundError("read failed") if failure_kind == "read_missing" else OSError("read failed")
    close_error = FileNotFoundError("close failed") if failure_kind == "close_missing" else OSError("close failed")
    stream = SimpleNamespace(read_calls=0, close_calls=0)

    def read():
        stream.read_calls += 1
        if failure_kind == "close_missing":
            return "over temperature"
        raise read_error

    def close():
        stream.close_calls += 1
        raise close_error

    stream.read, stream.close = read, close
    original_open = Path.open

    def selected_open(value, *args, **kwargs):
        return stream if value == path else original_open(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", selected_open)
        with pytest.raises(BaseException) as caught:
            progress.HotGate.wait()
    assert stream.read_calls == stream.close_calls == 1
    assert close_error in leaves(caught.value)
    if failure_kind != "close_missing":
        assert read_error in leaves(caught.value)

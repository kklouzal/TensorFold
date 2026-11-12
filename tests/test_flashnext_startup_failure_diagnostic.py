"""The constructor's original context precedes fatal peer notification."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.cuda import cohort
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
from tensorfold.families.qwen4_exp.cuda.execution_identity import _fatal


def capture_writes(monkeypatch):
    records = []
    original = os.write
    def write(fd, data):
        records.append((fd, data))
        return len(data) if fd == 2 else original(fd, data)
    monkeypatch.setattr(cohort.os, 'write', write)
    return records


def test_constructed_fatal_error_reports_before_poison_notification_and_only_once(monkeypatch):
    records = capture_writes(monkeypatch)
    read, write = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
    monkeypatch.setenv(cohort.FAILURE_FD_ENV, str(write))
    owner = SimpleNamespace(_execution_manifest_uncertain=False, _abort_comm=False,
                            multi=SimpleNamespace(poison_control=lambda error: cohort.notify_failure()))
    primary = RuntimeError('capture prohibits this CPU operation')
    try:
        with cohort.startup_failure_context(3):
            _fatal(owner, primary)  # Not inside except: the explicit original is required.
            _fatal(owner, primary)
        assert [fd for fd, _ in records] == [2, write]
        assert os.read(read, 8) == b'F'
        context = json.loads(records[0][1])
        assert context['rank'] == 3 and context['error_type'] == 'RuntimeError'
        assert context['message'] == 'capture prohibits this CPU operation'
        assert owner._execution_manifest_uncertain and owner._abort_comm
    finally:
        os.close(write)
        os.close(read)


def test_direct_fatal_notification_captures_active_constructor_exception_first(monkeypatch):
    records = capture_writes(monkeypatch)
    read, write = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
    monkeypatch.setenv(cohort.FAILURE_FD_ENV, str(write))
    try:
        with cohort.startup_failure_context(6):
            try:
                raise ValueError('actual CUDA capture callback rejected')
            except ValueError:
                assert cohort.notify_failure()
        assert [fd for fd, _ in records] == [2, write]
        assert os.read(read, 8) == b'F'
        record = json.loads(records[0][1])
        expected = 'test_direct_fatal_notification_captures_active_constructor_exception_first'
        assert record['frames'][-1]['function'] == expected[:48]
    finally:
        os.close(write)
        os.close(read)


def test_report_payload_is_bounded_sanitized_and_never_formats_custom_exception_or_locals(monkeypatch):
    records = capture_writes(monkeypatch)
    class OpaqueError(RuntimeError):
        def __str__(self):
            pytest.fail('diagnostics must not call custom exception formatting')
    local_secret = 'controlled-test-local-must-not-be-rendered'
    with cohort.startup_failure_context(0):
        try:
            raise OpaqueError(('\\"\x00\n☃' * 10000))
        except OpaqueError as error:
            assert cohort.report_startup_failure(error)
            assert not cohort.report_startup_failure(error)
    assert len(records) == 1 and len(records[0][1]) <= 4096
    record = json.loads(records[0][1])
    assert len(record['message']) <= 512 and '\x00' not in record['message'] and '\n' not in record['message']
    assert local_secret.encode() not in records[0][1]


def test_diagnostic_failure_still_notifies_and_preserves_original_identity(monkeypatch):
    read, write = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
    monkeypatch.setenv(cohort.FAILURE_FD_ENV, str(write))
    original = os.write
    calls = []
    def fail_stderr(fd, data):
        calls.append(fd)
        if fd == 2:
            raise OSError('controlled diagnostic sink failure')
        return original(fd, data)
    monkeypatch.setattr(cohort.os, 'write', fail_stderr)
    primary = RuntimeError('actual original constructor error')
    owner = SimpleNamespace(_execution_manifest_uncertain=False, _abort_comm=False, multi=None)
    try:
        with cohort.startup_failure_context(0):
            with pytest.raises(RuntimeError) as found:
                _fatal(owner, primary)
        assert found.value is primary and os.read(read, 8) == b'F'
        assert calls == [2, write] and primary.__cause__ is not None
    finally:
        os.close(write)
        os.close(read)


def test_constructor_reports_before_rollback_and_resets_the_scope(monkeypatch):
    records = capture_writes(monkeypatch)
    primary = RuntimeError('controlled initialization failure')
    events = []
    def fail(engine, *args, **kwargs):
        raise primary
    def retire(engine):
        assert records and json.loads(records[0][1])['message'] == str(primary)
        events.append('retired')
    monkeypatch.setattr(FlashNextEngine, '_initialize', fail)
    monkeypatch.setattr(FlashNextEngine, '_rollback_startup', retire)
    with pytest.raises(RuntimeError) as found:
        FlashNextEngine(Path('/unused'), rank=2)
    assert found.value is primary and events == ['retired']
    assert not cohort.report_startup_failure(primary)


def test_constructor_rolls_back_even_if_diagnostic_write_fails(monkeypatch):
    primary = RuntimeError('controlled initialization failure')
    events = []
    monkeypatch.setattr(FlashNextEngine, '_initialize', lambda *args, **kwargs: (_ for _ in ()).throw(primary))
    monkeypatch.setattr(FlashNextEngine, '_rollback_startup', lambda engine: events.append('retired'))
    monkeypatch.setattr(cohort.os, 'write', lambda *args: (_ for _ in ()).throw(OSError('sink failed')))
    with pytest.raises(RuntimeError) as found:
        FlashNextEngine(Path('/unused'))
    assert found.value is primary and events == ['retired'] and primary.__cause__ is not None
    assert not cohort.report_startup_failure(primary)


def test_success_and_nested_constructor_scopes_do_not_leave_a_serving_reporter(monkeypatch):
    records = capture_writes(monkeypatch)
    error = ValueError('first')
    with cohort.startup_failure_context(1):
        assert cohort.report_startup_failure(error)
        with cohort.startup_failure_context(2):
            assert cohort.report_startup_failure(error)
        assert not cohort.report_startup_failure(error)
    monkeypatch.setattr(FlashNextEngine, '_initialize', lambda *args, **kwargs: None)
    FlashNextEngine(Path('/unused'))
    assert not cohort.report_startup_failure(error)
    assert [json.loads(raw)['rank'] for fd, raw in records] == [1, 2]

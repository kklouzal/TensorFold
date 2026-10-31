"""CPU-only failure paths for the Linux CUDA-build test's owned child sessions."""

import json
import os
import select
import subprocess
import sys
from contextlib import contextmanager
from itertools import count

import pytest

from tests.cuda import test_build as build

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="native CUDA build cleanup uses Linux sessions")


def test_owned_build_session_retires_descendants_in_separate_groups_after_failure(tmp_path):
    code = """
import json, os, signal, subprocess, sys
child = subprocess.Popen([sys.executable, '-c', 'import signal; signal.pause()'],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         process_group=0)
print(json.dumps({'pid': child.pid, 'group': os.getpgid(child.pid), 'session': os.getsid(child.pid)}), flush=True)
signal.pause()
"""
    error = ValueError("primary build failure")
    stderr = tmp_path / "child.stderr"
    descendant_fd = None
    try:
        with stderr.open("w") as errors, pytest.raises(ValueError) as caught:
            with build._child(code, dict(os.environ), stderr=errors) as owner:
                descendant = json.loads(build._line(owner, stderr, timeout=10))
                assert descendant["session"] == owner.pid
                assert descendant["group"] == descendant["pid"] != owner.pid
                descendant_fd = os.pidfd_open(descendant["pid"])
                raise error
        assert caught.value is error and owner.returncode < 0 and owner._session_retired
        assert select.select([descendant_fd], [], [], 0)[0] == [descendant_fd]
    finally:
        if descendant_fd is not None:
            os.close(descendant_fd)
    build._stop(owner)                          # repeated outer cleanup cannot signal a retired session again


@pytest.mark.parametrize("primary", [False, True])
def test_owned_build_session_reports_cleanup_failure_without_losing_primary(monkeypatch, primary):
    real_stop = build._stop
    cleanup = RuntimeError("controlled cleanup failure")

    def stop_then_fail(process):
        real_stop(process)
        raise cleanup

    monkeypatch.setattr(build, "_stop", stop_then_fail)
    error = ValueError("primary build failure")
    with pytest.raises(ValueError if primary else RuntimeError) as caught:
        with build._child("import signal; signal.pause()", dict(os.environ)) as owner:
            if primary:
                raise error
    assert owner.returncode < 0 and owner._session_retired
    if primary:
        assert caught.value is error
        assert isinstance(error.__cause__, BaseExceptionGroup)
        assert error.__cause__.exceptions == (cleanup,)
    else:
        assert caught.value is cleanup


def test_success_drains_both_pipes_without_reaping_until_descendants_retire(monkeypatch):
    code = """
import json, os, subprocess, sys
child = subprocess.Popen([sys.executable, '-c', 'import signal; signal.pause()'],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         process_group=0)
print(json.dumps({'pid': child.pid, 'group': os.getpgid(child.pid)}), flush=True)
for _ in range(20):
    os.write(1, b'x' * 65536)
    os.write(2, b'y' * 65536)
os._exit(7)
"""
    descendant_fd = None
    waits = []
    real_members = build._session_members
    try:
        with build._child(code, dict(os.environ)) as owner:
            real_wait = owner.wait

            def reap(*, timeout):
                assert owner.returncode is None
                with real_members(owner.pid) as members:
                    assert not members, "the direct child must remain unreaped until all live members retire"
                waits.append(True)
                return real_wait(timeout=timeout)

            @contextmanager
            def members_before_reap(session):
                assert owner.returncode is None, "retired numeric session IDs must never be rescanned"
                with real_members(session) as members:
                    yield members

            monkeypatch.setattr(owner, "wait", reap)
            monkeypatch.setattr(build, "_session_members", members_before_reap)
            stdout, stderr = build._communicate(owner, timeout=10)
            header, _, stdout = stdout.partition("\n")
            descendant = json.loads(header)
            assert descendant["group"] == descendant["pid"] != owner.pid
            descendant_fd = os.pidfd_open(descendant["pid"])
            assert stdout == "x" * (20 * 65536) and stderr == "y" * (20 * 65536)
            assert build._exited(owner) == 7 and owner.returncode is None
            status = os.waitid(os.P_PID, owner.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            assert status is not None and status.si_status == 7
            assert not select.select([descendant_fd], [], [], 0)[0]
        assert owner.returncode == 7 and owner._session_retired and waits == [True]
        assert select.select([descendant_fd], [], [], 0)[0] == [descendant_fd]
        build._stop(owner)                      # must return before scanning the retired numeric identity
    finally:
        if descendant_fd is not None:
            os.close(descendant_fd)


def test_output_timeout_preserves_partial_capture_and_retires_child():
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        with build._child("import os, signal; os.write(1, b'partial'); signal.pause()", dict(os.environ)) as owner:
            build._communicate(owner, timeout=0.2)
    assert caught.value.output == b"partial"
    assert owner.returncode < 0 and owner._session_retired


@pytest.mark.parametrize("reader", ["line", "communicate"])
def test_deadline_stops_reading_even_while_pipe_stays_ready(monkeypatch, tmp_path, reader):
    """A real bounded producer stays readable across a controlled deadline; it must not be drained afterward."""

    stderr = tmp_path / "producer.stderr"
    with stderr.open("w") as errors, build._child(
        "import os, signal; os.write(1, b'ready\\n' + b'x' * 100); signal.pause()", dict(os.environ), stderr=errors
    ) as owner:
        assert build._line(owner, stderr, timeout=10) == "ready"
        real_read = os.read
        with monkeypatch.context() as patch:
            ticks = count()
            patch.setattr(build.time, "monotonic", lambda: float(next(ticks)))
            patch.setattr(build.os, "read", lambda fd, size: real_read(fd, min(size, 1)))
            with pytest.raises(AssertionError if reader == "line" else subprocess.TimeoutExpired):
                if reader == "line":
                    build._line(owner, stderr, timeout=3)
                else:
                    build._communicate(owner, timeout=3)
        assert select.select([owner.stdout], [], [], 0)[0], "reader consumed data beyond its deadline"
        assert os.read(owner.stdout.fileno(), 100) == b"x" * 98
    assert owner.returncode < 0 and owner._session_retired

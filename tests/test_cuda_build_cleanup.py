"""CPU-only failure paths for the Linux CUDA-build test's owned child sessions."""

import json
import os
from pathlib import Path
import sys

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
    with stderr.open("w") as errors, pytest.raises(ValueError) as caught:
        with build._child(code, dict(os.environ), stderr=errors) as owner:
            descendant = json.loads(build._line(owner, stderr, timeout=10))
            assert descendant["session"] == owner.pid
            assert descendant["group"] == descendant["pid"] != owner.pid
            raise error
    assert caught.value is error and owner.returncode < 0
    assert not build._session_groups(owner.pid)
    try:
        state = (Path('/proc') / str(descendant["pid"]) / 'stat').read_bytes().rsplit(b') ', 1)[1].split()[0]
    except FileNotFoundError:
        pass
    else:
        assert state in (b'Z', b'X')             # terminal orphan awaiting its container init's reap
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
    assert owner.returncode < 0 and not build._session_groups(owner.pid)
    if primary:
        assert caught.value is error
        assert isinstance(error.__cause__, BaseExceptionGroup)
        assert error.__cause__.exceptions == (cleanup,)
    else:
        assert caught.value is cleanup

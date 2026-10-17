"""Real nvcc extension builds: cache reuse and recovery after a builder is killed while holding Torch's lock."""

import os
from contextlib import contextmanager
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import pytest

NAME = "tensorfold_qwen4_exp_gdn_io_v2"
START = "from tensorfold.families.qwen4_exp.cuda import gdn_io; gdn_io._ext()"
# Pause the actual JIT after it writes its ninja plan, while its own lock is held.
INTERRUPTED = f"""
from pathlib import Path
from torch.utils import cpp_extension
import signal

def interrupted(build_directory, *args, **kwargs):
    print(Path(build_directory) / 'lock', flush=True)
    signal.pause()

cpp_extension._run_ninja_build = interrupted
{START}
"""


def _start(env: dict) -> subprocess.CompletedProcess:
    with _child(START, env) as process:
        stdout, stderr = process.communicate(timeout=1800)
        return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


@contextmanager
def _child(code, env, *, stderr=subprocess.PIPE):
    """Own the build's Linux session, including compiler groups, through success or failed-test cleanup."""

    process = subprocess.Popen([sys.executable, "-u", "-c", code], env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=stderr, text=True, start_new_session=True)
    primary = None
    try:
        yield process
    except BaseException as error:
        primary = error
        raise
    finally:
        failures = []
        actions = [lambda: _stop(process)]
        actions += [stream.close for stream in (process.stdout, process.stderr) if stream is not None]
        for action in actions:
            try:
                action()
            except BaseException as error:
                failures.append(error)
        if failures:
            if primary is not None:
                BaseException.add_note(primary, f"build session {process.pid} cleanup failed; inspect the chained failures")
                raise primary from BaseExceptionGroup("build-session cleanup failed", failures)
            if len(failures) == 1:
                raise failures[0]
            raise BaseExceptionGroup("build-session cleanup failed", failures)


def _line(process, stderr, timeout=600):
    deadline = time.monotonic() + timeout
    line = bytearray()
    while not line.endswith(b"\n"):
        ready, _, _ = select.select([process.stdout], [], [], max(0, deadline - time.monotonic()))
        assert ready, f"no line within {timeout} s; child exit={process.poll()}; stderr={stderr.read_text()}"
        byte = os.read(process.stdout.fileno(), 1)
        assert byte, f"child stdout closed; exit={process.poll()}; stderr={stderr.read_text()}"
        line.extend(byte)
    return line.decode().strip()


def _session_groups(session):
    """Live groups in this owned Linux session; an exited zombie cannot perform more compiler work."""

    groups = set()
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            # comm may contain spaces/parentheses/non-UTF8 bytes; fields after its final ')' are ASCII metadata.
            fields = (entry / 'stat').read_bytes().rsplit(b') ', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue                              # the process exited while the snapshot was read
        if int(fields[3]) == session and fields[0] not in (b'Z', b'X'):
            groups.add(int(fields[2]))
    return groups


def _stop(process):
    if getattr(process, '_session_retired', False):
        return
    deadline = time.monotonic() + 30
    while True:
        groups = _session_groups(process.pid)
        for group in groups:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass                              # the entire group already exited
        remaining = deadline - time.monotonic()
        if process.returncode is None:
            try:
                process.wait(timeout=max(0, min(remaining, 0.05)))
            except subprocess.TimeoutExpired:
                pass
        # Re-snapshot after the leader exits: children forked during the first scan must also be retired.
        remaining_groups = _session_groups(process.pid)
        if process.returncode is not None and not remaining_groups:
            process._session_retired = True
            return
        if remaining <= 0:
            raise RuntimeError(f"build session {process.pid} did not retire within 30 s; live groups {sorted(remaining_groups)}")
        time.sleep(min(0.01, remaining))


def test_the_first_start_says_it_builds_and_the_next_is_quiet(tmp_path):
    env = dict(os.environ, TORCH_EXTENSIONS_DIR=str(tmp_path))
    first = _start(env)
    assert first.returncode == 0, first.stderr
    assert f"[tensorfold] building CUDA extension {NAME}" in first.stdout
    again = _start(env)
    assert again.returncode == 0, again.stderr
    assert NAME not in again.stdout


def test_a_killed_build_releases_or_reports_its_actual_lock(tmp_path):
    from torch.utils import cpp_extension

    names = cpp_extension._jit_compile.__code__.co_names
    advisory = "FileLock" in names and "FileBaton" not in names
    baton = "FileBaton" in names and "FileLock" not in names
    assert advisory or baton, f"unaudited Torch JIT lock protocol: {names}"
    env = dict(os.environ, TORCH_EXTENSIONS_DIR=str(tmp_path))
    lock = tmp_path / NAME / "lock"
    owner_stderr = tmp_path / "owner.stderr"
    with owner_stderr.open("w") as errors:
        with _child(INTERRUPTED, env, stderr=errors) as owner:
            assert _line(owner, owner_stderr).startswith(f"[tensorfold] building CUDA extension {NAME}")
            assert _line(owner, owner_stderr) == str(lock)
            assert (lock.parent / "build.ninja").is_file()
            assert not (lock.parent / (NAME + cpp_extension.LIB_EXT)).exists()
            assert owner.poll() is None
            if advisory:
                _advisory_recovery(owner, env, lock, tmp_path, cpp_extension)
            else:
                _baton_recovery(owner, env, lock, tmp_path)


def _advisory_recovery(owner, env, lock, tmp_path, cpp_extension):
    from filelock import Timeout

    # The real JIT owns this OS lock, rather than merely leaving a marker file.
    with pytest.raises(Timeout), cpp_extension.FileLock(str(lock), timeout=0):
        pass
    observed = f"""
from torch.utils import cpp_extension

class ObservedLock(cpp_extension.FileLock):
    def __enter__(self):
        print('acquiring ' + self.lock_file, flush=True)
        return super().__enter__()

cpp_extension.FileLock = ObservedLock
{START}
"""
    waiter_stderr = tmp_path / "waiter.stderr"
    with waiter_stderr.open("w") as errors:
        with _child(observed, env, stderr=errors) as waiter:
            assert _line(waiter, waiter_stderr).startswith(f"[tensorfold] building CUDA extension {NAME}")
            assert _line(waiter, waiter_stderr) == "acquiring " + str(lock)
            with pytest.raises(subprocess.TimeoutExpired):
                waiter.wait(timeout=0.2)
            _stop(owner)
            assert owner.returncode < 0
            # No unlink: killing the owner must release the kernel lock, and the
            # waiting JIT must finish the interrupted extension's native build.
            output, _ = waiter.communicate(timeout=1800)
            assert waiter.returncode == 0, waiter_stderr.read_text()
            assert "killed build" not in output and "delete the lock" not in output
            assert lock.is_file()
    again = _start(env)
    assert again.returncode == 0, again.stderr
    assert NAME not in again.stdout


def _baton_recovery(owner, env, lock, tmp_path):
    _stop(owner)
    assert owner.returncode < 0 and lock.is_file()
    waiter_stderr = tmp_path / "waiter.stderr"
    with waiter_stderr.open("w") as errors:
        with _child(START, env, stderr=errors) as waiter:
            line = _line(waiter, waiter_stderr)
            assert str(lock) in line and "stop this start, delete the lock and start again" in line
            assert waiter.poll() is None
    lock.unlink()
    again = _start(env)
    assert again.returncode == 0, again.stderr
    assert f"[tensorfold] building CUDA extension {NAME}" in again.stdout

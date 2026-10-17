"""Real nvcc extension builds: cache reuse and recovery after a builder is killed while holding Torch's lock."""

import os
from contextlib import contextmanager, ExitStack
from pathlib import Path
import select
import selectors
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
        stdout, stderr = _communicate(process, timeout=1800)
        return subprocess.CompletedProcess(process.args, _exited(process), stdout, stderr)


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
        remaining = deadline - time.monotonic()
        ready = select.select([process.stdout], [], [], remaining)[0] if remaining > 0 else []
        assert ready, f"no line within {timeout} s; child exit={_exited(process)}; stderr={stderr.read_text()}"
        byte = os.read(process.stdout.fileno(), 1)
        assert byte, f"child stdout closed; exit={_exited(process)}; stderr={stderr.read_text()}"
        line.extend(byte)
    return line.decode().strip()


def _exited(process):
    """Observe exit without reaping: the direct child's PID reserves our session identity until cleanup."""

    status = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if status is None:
        return None
    return status.si_status if status.si_code == os.CLD_EXITED else -status.si_status


def _wait_exit(process, timeout):
    deadline = time.monotonic() + timeout
    while (status := _exited(process)) is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        time.sleep(min(0.01, remaining))
    return status


def _communicate(process, timeout):
    """Drain both captured pipes under one deadline, retaining the unreaped session leader."""

    deadline = time.monotonic() + timeout
    streams = (process.stdout, process.stderr)
    buffers = [bytearray(), bytearray()]
    with selectors.DefaultSelector() as selector:
        for index, stream in enumerate(streams):
            if stream is not None:
                selector.register(stream, selectors.EVENT_READ, index)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            events = selector.select(remaining) if remaining > 0 else []
            if not events:
                raise subprocess.TimeoutExpired(process.args, timeout, bytes(buffers[0]), bytes(buffers[1]))
            for key, _ in events:
                data = os.read(key.fd, 65536)
                if data:
                    buffers[key.data].extend(data)
                else:
                    selector.unregister(key.fileobj)
    _wait_exit(process, max(0, deadline - time.monotonic()))
    return tuple(bytes(data).decode(stream.encoding, stream.errors).replace('\r\n', '\n').replace('\r', '\n')
                 if stream is not None else None for stream, data in zip(streams, buffers))


@contextmanager
def _session_members(session):
    """Pin each live session member before inspection; neither proc FDs nor pidfds retarget reused PIDs."""

    members = []
    with ExitStack() as handles:
        for entry in Path('/proc').iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                # Open the proc directory BEFORE the pidfd, then inspect that same directory. If the PID was
                # reused between opens, old proc FDs fail rather than describe the replacement process.
                directory = os.open(entry, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
                try:
                    pidfd = os.pidfd_open(int(entry.name))
                    try:
                        with os.fdopen(os.open('stat', os.O_RDONLY, dir_fd=directory), 'rb') as stat:
                            fields = stat.read().rsplit(b') ', 1)[1].split()
                        # comm may contain spaces/parentheses/non-UTF8 bytes; the remaining metadata is ASCII.
                        if int(fields[3]) == session and fields[0] not in (b'Z', b'X'):
                            members.append((pidfd, int(fields[2])))
                            handles.callback(os.close, pidfd)
                            pidfd = None
                    finally:
                        if pidfd is not None:
                            os.close(pidfd)
                finally:
                    os.close(directory)
            except (FileNotFoundError, ProcessLookupError):
                continue                          # process retired during the owned-handle snapshot
        yield members


def _stop(process):
    if getattr(process, '_session_retired', False):
        return
    if process.returncode is not None:
        raise RuntimeError("build leader was reaped before session cleanup; session identity is no longer reserved")
    deadline = time.monotonic() + 30
    while True:
        with _session_members(process.pid) as members:
            for pidfd, _ in members:
                try:
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                except ProcessLookupError:
                    pass                          # this exact process already exited; never signal a reused PID/PGID
        # The leader remains an unreaped zombie while rescanning children forked during the first snapshot.
        with _session_members(process.pid) as members:
            remaining_groups = {group for _, group in members}
        if _exited(process) is not None and not remaining_groups:
            process.wait(timeout=0)                # only reap after quiescence; never scan/signal this SID again
            process._session_retired = True
            return
        remaining = deadline - time.monotonic()
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
    import torch

    target = "sm_" + "".join(str(n) for n in torch.cuda.get_device_capability())
    built_name = NAME + "_" + target
    names = cpp_extension._jit_compile.__code__.co_names
    advisory = "FileLock" in names and "FileBaton" not in names
    baton = "FileBaton" in names and "FileLock" not in names
    assert advisory or baton, f"unaudited Torch JIT lock protocol: {names}"
    env = dict(os.environ, TORCH_EXTENSIONS_DIR=str(tmp_path))
    lock = tmp_path / built_name / "lock"
    owner_stderr = tmp_path / "owner.stderr"
    with owner_stderr.open("w") as errors:
        with _child(INTERRUPTED, env, stderr=errors) as owner:
            assert _line(owner, owner_stderr).startswith(f"[tensorfold] building CUDA extension {NAME}")
            assert _line(owner, owner_stderr) == str(lock)
            assert (lock.parent / "build.ninja").is_file()
            assert not (lock.parent / (built_name + cpp_extension.LIB_EXT)).exists()
            assert _exited(owner) is None
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
                _wait_exit(waiter, timeout=0.2)
            _stop(owner)
            assert owner.returncode < 0
            # No unlink: killing the owner must release the kernel lock, and the
            # waiting JIT must finish the interrupted extension's native build.
            output, _ = _communicate(waiter, timeout=1800)
            assert _exited(waiter) == 0, waiter_stderr.read_text()
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
            assert _exited(waiter) is None
    lock.unlink()
    again = _start(env)
    assert again.returncode == 0, again.stderr
    assert f"[tensorfold] building CUDA extension {NAME}" in again.stdout

"""Fatal TP work wakes its process owner without waiting for stuck collectives."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import signal
import select
import sys

import pytest

from tensorfold.cuda.cohort import FAILURE_FD_ENV, notify_failure
from tensorfold.cuda.expert_launch import supervise


def test_unmanaged_launch_does_not_claim_process_containment(monkeypatch):
    monkeypatch.delenv(FAILURE_FD_ENV, raising=False)
    assert notify_failure() is False


def test_notification_uses_only_a_valid_borrowed_pipe(monkeypatch, tmp_path):
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        monkeypatch.setenv(FAILURE_FD_ENV, str(write_fd))
        assert notify_failure() is True
        assert os.read(read_fd, 8) == b"F"
        os.fstat(write_fd)  # notifier borrows; process owner still owns release
        with (tmp_path / "file").open("wb") as output:
            monkeypatch.setenv(FAILURE_FD_ENV, str(output.fileno()))
            with pytest.raises(RuntimeError, match="not a pipe"):
                notify_failure()
        monkeypatch.setenv(FAILURE_FD_ENV, "-1")
        with pytest.raises(RuntimeError, match="invalid"):
            notify_failure()
    finally:
        os.close(write_fd)
        os.close(read_fd)


@pytest.mark.skipif(sys.platform != "linux", reason="V100 supervisor uses Linux session ownership")
def test_fatal_channel_retires_live_peers_without_waiting_for_their_exit(monkeypatch):
    import tensorfold.cuda.expert_launch as launcher

    read_fd, write_fd = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
    children = []
    original = launcher.subprocess.Popen

    def spawn(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    def failed():
        try:
            return bool(os.read(read_fd, 1))
        except BlockingIOError:
            return False

    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    reporter = f"import os,time;os.write({write_fd},b'F');time.sleep(60)"
    blocked = "import time;time.sleep(60)"
    try:
        assert supervise([[sys.executable, "-c", reporter], [sys.executable, "-c", blocked]],
                         [dict(os.environ)] * 2, grace=.05, failure_probe=failed,
                         pass_fds=(write_fd,)) == 1
        assert len(children) == 2
        assert all(child.returncode is not None for child in children)
        for child in children:
            with pytest.raises(ProcessLookupError):
                os.kill(child.pid, 0)
    finally:
        os.close(write_fd)
        os.close(read_fd)


def test_tp_launcher_wires_rank_channel_and_restores_process_policy(monkeypatch):
    path = Path(__file__).parents[1] / "tools/launch_cuda_tp.py"
    spec = importlib.util.spec_from_file_location("tp_launcher_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = [signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)]
    captured = {}

    def owner(commands, environments, **options):
        captured.update(commands=commands, environments=environments, options=options)
        fd, = options["pass_fds"]
        assert all(env[FAILURE_FD_ENV] == str(fd) for env in environments)
        assert options["failure_probe"]() is False
        os.write(fd, b"F")
        assert options["failure_probe"]() is True
        return 19

    monkeypatch.setattr(module, "supervise", owner)
    assert module.main(["--gpus", " 0,1", "--", "model", "--context", "8192"]) == 19
    assert [env["CUDA_VISIBLE_DEVICES"] for env in captured["environments"]] == ["1", "0"]
    assert captured["options"]["shutdown_rank"] == 1
    assert captured["options"]["shutdown_probe"]() is False
    assert [signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)] == before
    with pytest.raises(OSError):
        os.fstat(captured["options"]["pass_fds"][0])


def test_invalid_supervisor_callbacks_and_capabilities_fail_before_spawn(monkeypatch):
    import tensorfold.cuda.expert_launch as launcher

    def forbidden(*args, **kwargs):
        pytest.fail("invalid control spawned a rank")

    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden)
    for options in ({"failure_probe": True}, {"pass_fds": (-1,)}, {"stop_on_departure": 1},
                    {"shutdown_probe": True}, {"shutdown_rank": True}, {"shutdown_rank": 1}):
        with pytest.raises(ValueError):
            supervise([["unused"]], [dict(os.environ)], **options)


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_tp_signal_stop_preserves_supervisor_failure_and_cleanup_failure(monkeypatch, cleanup_failure):
    path = Path(__file__).parents[1] / "tools/launch_cuda_tp.py"
    spec = importlib.util.spec_from_file_location("tp_signal_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = signal.getsignal(signal.SIGTERM)
    primary = OSError("controlled cohort retirement failure")

    def owner(*args, **kwargs):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert kwargs["shutdown_probe"]()
        assert not kwargs["failure_probe"]()
        if cleanup_failure:
            raise primary
        return 1

    monkeypatch.setattr(module, "supervise", owner)
    if cleanup_failure:
        with pytest.raises(OSError) as caught:
            module.main(["--gpus", "0,1", "--", "model"])
        assert caught.value is primary
    else:
        assert module.main(["--gpus", "0,1", "--", "model"]) == 1
    assert signal.getsignal(signal.SIGTERM) is previous


@pytest.mark.skipif(sys.platform != "linux", reason="Linux rank session ownership")
@pytest.mark.parametrize("ranks,stop_signal,mode", [
    (2, signal.SIGTERM, "clean"), (8, signal.SIGINT, "clean"),
    (2, signal.SIGTERM, "timeout"), (2, signal.SIGTERM, "fatal"),
    (2, signal.SIGINT, "nonzero"),
])
def test_real_tp_operator_stop_coordinates_followers(tmp_path, ranks, stop_signal, mode):
    """Followers have default TERM: only protocol STOP can run their finalizers."""
    import subprocess
    import time

    worker = tmp_path / "rank.py"
    worker.write_text('''import os, pathlib, signal, sys, time
root = pathlib.Path(sys.argv[1])
rank = int(sys.argv[sys.argv.index('--rank') + 1])
mode = sys.argv[2]
def stop(sig, frame):
    raise KeyboardInterrupt
if rank == 0:
    signal.signal(signal.SIGTERM, stop)
(root / f'ready-{rank}').write_text(str(os.getpid()))
try:
    if rank == 0:
        while True: time.sleep(.01)
    else:
        while not (root / 'STOP').exists(): time.sleep(.01)
        if mode == 'timeout':
            while True: time.sleep(.01)
        time.sleep(.15)
        if mode == 'fatal':
            os.write(int(os.environ['TENSORFOLD_COHORT_FAILURE_FD']), b'F')
            while True: time.sleep(.01)
        if mode == 'nonzero': raise SystemExit(23)
except KeyboardInterrupt:
    (root / 'STOP').write_text('stop')
finally:
    (root / f'summary-{rank}').write_text('closed')
'''.replace("TENSORFOLD_COHORT_FAILURE_FD", FAILURE_FD_ENV))
    wrapper = tmp_path / "parent.py"
    launcher = Path(__file__).parents[1] / "tools/launch_cuda_tp.py"
    wrapper.write_text(f'''import importlib.util, sys
spec = importlib.util.spec_from_file_location('tp', {str(launcher)!r})
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
original = m.supervise
m.supervise = lambda commands, envs, **kw: original(commands, envs, **{{**kw, 'grace': .6}})
raise SystemExit(m.main(['--gpus', {','.join(map(str, range(ranks)))!r}, '--', {str(tmp_path)!r}, {mode!r}], _command_prefix=[sys.executable, {str(worker)!r}]))
''')
    environment = dict(os.environ, PYTHONPATH=str(launcher.parents[1] / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    parent = subprocess.Popen([sys.executable, str(wrapper)], env=environment,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pidfds = {}
    try:
        deadline = time.monotonic() + 8
        while len(pidfds) < ranks:
            # Pin each live ready worker before stop, including partial startup.
            for ready in tmp_path.glob("ready-*"):
                if ready.name not in pidfds:
                    pidfds[ready.name] = os.pidfd_open(int(ready.read_text()))
            assert parent.poll() is None, parent.communicate()
            assert time.monotonic() < deadline, "rank readiness timed out"
            time.sleep(.01)
        parent.send_signal(stop_signal)
        # A second operator request must not kill peers or reset grace.
        time.sleep(.05)
        parent.send_signal(stop_signal)
        stdout, stderr = parent.communicate(timeout=5)
        assert parent.returncode == (0 if mode == "clean" else 23 if mode == "nonzero" else 1), (stdout, stderr)
        assert (tmp_path / 'summary-0').read_text() == 'closed'
        if mode == "clean":
            assert len(list(tmp_path.glob("summary-*"))) == ranks
        if mode in ("timeout", "fatal"):
            assert not (tmp_path / 'summary-1').exists()
        assert len(select.select(list(pidfds.values()), [], [], 0)[0]) == ranks
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.communicate(timeout=3)
        for pidfd in pidfds.values():
            try:
                signal.pidfd_send_signal(pidfd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                os.close(pidfd)


def test_shutdown_during_spawn_retires_acquired_rank_without_spawning_peers(monkeypatch):
    import tensorfold.cuda.expert_launch as launcher

    children, pidfds = [], []
    spawn = launcher.subprocess.Popen

    def acquire(*args, **kwargs):
        child = spawn(*args, **kwargs)
        children.append(child)
        pidfds.append(os.pidfd_open(child.pid))
        return child

    monkeypatch.setattr(launcher.subprocess, "Popen", acquire)
    try:
        with pytest.raises(RuntimeError, match="startup cancelled"):
            supervise([[sys.executable, "-c", "import time; time.sleep(60)"]] * 8,
                      [dict(os.environ)] * 8, grace=.05, shutdown_probe=lambda: bool(children))
        assert len(children) == 1 and children[0].returncode is not None
        assert select.select(pidfds, [], [], 0)[0] == pidfds
    finally:
        for fd in pidfds:
            os.close(fd)


def test_spawn_keyboard_interrupt_preserves_cancellation_and_retires_rank(monkeypatch):
    import tensorfold.cuda.expert_launch as launcher

    children, pidfds = [], []
    spawn = launcher.subprocess.Popen
    cancellation = KeyboardInterrupt("cancel rank acquisition")

    def acquire(*args, **kwargs):
        if children:
            raise cancellation
        child = spawn(*args, **kwargs)
        children.append(child)
        pidfds.append(os.pidfd_open(child.pid))
        return child

    monkeypatch.setattr(launcher.subprocess, "Popen", acquire)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            supervise([[sys.executable, "-c", "import time; time.sleep(60)"]] * 2,
                      [dict(os.environ)] * 2, grace=.05)
        assert caught.value is cancellation
        assert len(children) == 1 and children[0].returncode is not None
        assert select.select(pidfds, [], [], 0)[0] == pidfds
    finally:
        for fd in pidfds:
            os.close(fd)

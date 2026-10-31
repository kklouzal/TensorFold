"""Fatal TP work wakes its process owner without waiting for stuck collectives."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import signal
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
    assert captured["options"]["stop_on_departure"] is True
    assert [signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)] == before
    with pytest.raises(OSError):
        os.fstat(captured["options"]["pass_fds"][0])


def test_invalid_supervisor_callbacks_and_capabilities_fail_before_spawn(monkeypatch):
    import tensorfold.cuda.expert_launch as launcher

    def forbidden(*args, **kwargs):
        pytest.fail("invalid control spawned a rank")

    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden)
    for options in ({"failure_probe": True}, {"pass_fds": (-1,)}, {"stop_on_departure": 1}):
        with pytest.raises(ValueError):
            supervise([["unused"]], [dict(os.environ)], **options)


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_tp_signal_stop_preserves_clean_zero_and_reports_cleanup_failure(monkeypatch, cleanup_failure):
    path = Path(__file__).parents[1] / "tools/launch_cuda_tp.py"
    spec = importlib.util.spec_from_file_location("tp_signal_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = signal.getsignal(signal.SIGTERM)
    primary = OSError("controlled cohort retirement failure")

    def owner(*args, **kwargs):
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert kwargs["failure_probe"]()
        if cleanup_failure:
            raise primary
        return 1

    monkeypatch.setattr(module, "supervise", owner)
    if cleanup_failure:
        with pytest.raises(OSError) as caught:
            module.main(["--gpus", "0,1", "--", "model"])
        assert caught.value is primary
    else:
        assert module.main(["--gpus", "0,1", "--", "model"]) == 0
    assert signal.getsignal(signal.SIGTERM) is previous

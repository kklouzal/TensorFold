"""Actual selected-image software issuance, forged/stale/fork controls; no GPU claims."""

import os
import multiprocessing
from pathlib import Path
import sys
import time

import pytest

from tensorfold.cuda.sm70.software_binding import (
    SoftwareBindingError,
    issue_source_binding,
    verify_source_binding,
)

pytestmark = pytest.mark.skipif(not Path("/opt/tensorfold-pins/v100-source-foundation.json").is_file(),
                                reason="requires actual selected source image")


def test_selected_image_issues_only_current_immutable_process_software_authority():
    authority = issue_source_binding(required=True)
    record = verify_source_binding(authority)
    assert record["status"] == "verified-source-software"
    assert record["software_only"] is True and record["release_qualified"] is False
    assert record["recorded_components"]["pytorch"] == "2.14.1"
    assert "image-video-processor-stack" in record["pending_gates"]
    with pytest.raises(SoftwareBindingError, match="forged"):
        verify_source_binding(record)

    root = Path("/opt/tensorfold-pins")
    mode = root.stat().st_mode & 0o7777
    try:
        root.chmod(0o777)
        with pytest.raises(SoftwareBindingError, match="directory"):
            verify_source_binding(authority)
    finally:
        root.chmod(mode)

    # Fault injection changes only metadata in this disposable test container;
    # exact package/provider bytes stay untouched and the old authority fails.
    pins = Path("/opt/tensorfold-pins/source-pins.json")
    before = pins.stat()
    try:
        os.utime(pins, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
        with pytest.raises(SoftwareBindingError, match="inputs changed"):
            verify_source_binding(authority)
    finally:
        os.utime(pins, ns=(before.st_atime_ns, before.st_mtime_ns))


def _forked_binding_child(authority):
    try:
        verify_source_binding(authority)
    except SoftwareBindingError:
        os._exit(0)
    except BaseException:
        os._exit(2)
    os._exit(1)


def _retire_fork_control(process):
    if process.pid is None:
        raise RuntimeError("partial fork start remains uncertain; retain owner for outer session containment")
    if process.is_alive():
        process.kill()
    process.join(15)
    if process.is_alive():
        raise RuntimeError("fork child did not retire; retain owner for outer session containment")
    process.close()


def _exercise_fork_control(process):
    from tensorfold.cleanup import rollback

    try:
        process.start()
        process.join(15)
        if process.is_alive():
            raise TimeoutError("forked software-binding control exceeded its child deadline")
        if process.exitcode != 0:
            raise RuntimeError("forked software-binding authority was not refused")
    except BaseException as primary:
        rollback(process, primary, lambda: _retire_fork_control(process))
    else:
        _retire_fork_control(process)


def _cold_single_threaded_fork_control():
    authority = issue_source_binding(required=True)
    process = multiprocessing.get_context("fork").Process(target=_forked_binding_child, args=(authority,))
    if len(list(Path("/proc/self/task").iterdir())) != 1:
        raise RuntimeError("real-fork control requires exactly one native thread before fork")
    _exercise_fork_control(process)
    print("native_threads=1; inherited authority refused; child joined and closed")


def test_real_fork_rejection_in_a_fresh_proven_single_threaded_process():
    from tensorfold.cuda.expert_launch import supervise

    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       BLIS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    deadline = time.monotonic() + 60
    # The existing session owner also contains an uncertain partial fork or
    # retirement; it never reuses a reaped leader's numeric process identity.
    status = supervise([[sys.executable, str(Path(__file__).resolve())]], [environment], grace=0,
                       failure_probe=lambda: time.monotonic() >= deadline)
    assert status == 0


@pytest.mark.parametrize("failure", ["unpublished-start", "published-start", "retirement"])
def test_fork_control_preserves_start_failure_and_retains_uncertain_owner(failure):
    primary = OSError("injected fork start failure")

    class Process:
        pid, exitcode, closed = None, None, False

        def __init__(self):
            self.alive = False
            self.joins = []

        def start(self):
            if failure != "unpublished-start":
                self.pid, self.alive = 19, True
            if failure != "retirement":
                raise primary

        def join(self, timeout):
            self.joins.append(timeout)

        def is_alive(self):
            return self.alive

        def kill(self):
            if failure != "retirement":
                self.alive, self.exitcode = False, -9

        def close(self):
            self.closed = True

    process = Process()
    if failure == "retirement":
        with pytest.raises(TimeoutError) as result:
            _exercise_fork_control(process)
    else:
        with pytest.raises(OSError) as result:
            _exercise_fork_control(process)
        assert result.value is primary
    if failure == "published-start":
        assert process.closed and process.joins == [15]
    else:
        assert not process.closed and process in result.value._tensorfold_retained_owners
        assert all(timeout == 15 for timeout in process.joins)


if __name__ == "__main__":
    _cold_single_threaded_fork_control()

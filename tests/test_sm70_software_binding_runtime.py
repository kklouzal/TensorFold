"""Actual selected-image software issuance, forged/stale/fork controls; no GPU claims."""

import os
from pathlib import Path

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

    # This child refuses before touching inherited CUDA/library ownership.
    child = os.fork()
    if child == 0:
        try:
            verify_source_binding(authority)
        except SoftwareBindingError:
            os._exit(0)
        os._exit(1)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0

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

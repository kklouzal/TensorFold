"""Failure notification to the owner of a supervised one-host TP process group.

The descriptor is a borrowed process-lifetime capability installed only by the
launcher. Reporting a fatal operation does not destroy a communicator while
other threads or ranks still borrow it. The supervisor retires every process.
Unmanaged launches retain their existing external containment responsibility.
"""

from __future__ import annotations

import os
import stat

FAILURE_FD_ENV = "TENSORFOLD_TP_FAILURE_FD"


def notify_failure() -> bool:
    value = os.environ.get(FAILURE_FD_ENV)
    if value is None:
        return False
    if not value.isdecimal():
        raise RuntimeError("invalid TP supervisor failure descriptor")
    descriptor = int(value)
    if not stat.S_ISFIFO(os.fstat(descriptor).st_mode):
        raise RuntimeError("TP supervisor failure descriptor is not a pipe")
    # One writer per rank emits once after poisoning. Eight bytes fit within a
    # pipe's atomic-write bound; a nonblocking failure is surfaced to the owner.
    if os.write(descriptor, b"F") != 1:
        raise RuntimeError("TP supervisor did not accept the fatal-cohort notification")
    return True

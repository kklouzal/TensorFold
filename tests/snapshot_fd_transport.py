"""Labeled Python FD transport substitute, never native atomicity evidence."""

import os
from unittest.mock import patch


class TransportOwner:
    def __init__(self):
        self._fd = None
        self._opened = False

    def open(self, path, flags, mode=0o600, *, dir_fd=None):
        if self._opened:
            raise ValueError("one-shot transport owner already acquired")
        self._fd = os.open(path, flags, mode) if dir_fd is None else os.open(path, flags, mode, dir_fd=dir_fd)
        self._opened = True

    @property
    def closed(self):
        return self._fd is None

    def fileno(self):
        if self.closed:
            raise ValueError("closed transport owner")
        return self._fd

    def close(self):
        if not self.closed:
            descriptor, self._fd = self._fd, None
            os.close(descriptor)


def substitute_owners(testcase):
    for name in ("tensorfold.file_io", "tensorfold.engine.snapshot_payload"):
        patcher = patch(name + "._owned_slot", TransportOwner)
        patcher.start()
        testcase.addCleanup(patcher.stop)

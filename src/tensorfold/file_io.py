"""Bounded regular-file streams backed by the native POSIX descriptor owner.

FileIO and buffers borrow their descriptor (closefd=False). Calls and retirement
are synchronous, serialized owner operations. All Python IO views retire before
native close; pending buffers cannot later flush against a reused descriptor.
A pre-entry close interruption permits one exact-owner retry while closedfalse;
consumed closes never retry their number. Callers inspect every returned cleanup
status and retain/drain the scope until retired. This core contains no cache,
directory, exception annotation, or recovery policy. Declared POSIX targets only.
"""

from __future__ import annotations

import io
import os
import stat


def _owned_slot():
    from tensorfold._fd_owner import OwnedFD

    return OwnedFD()


def _buffered(raw, mode):
    if mode == "rb":
        return io.BufferedReader(raw)
    if mode == "wb":
        return io.BufferedWriter(raw)
    if mode == "r+b":
        return io.BufferedRandom(raw)
    raise ValueError("unsupported owned snapshot stream mode")


class _Record:
    __slots__ = ("owner", "raw", "stream", "done")

    def __init__(self):
        self.owner = self.raw = self.stream = None
        self.done = False


class FileStreams:
    """Operation-owned bounded journal; failure always remains an operation error."""

    def __init__(self, *, max_files):
        if type(max_files) is not int or max_files <= 0:
            raise ValueError("an explicit positive descriptor budget is required")
        self.maximum = max_files
        self._records = []

    def open_descriptor(self, path, flags, *, min_bytes=0, max_bytes=None):
        """Acquire a regular owner for incremental os.read without IO wrappers.

        The journal precedes acquisition. Failed post-acquisition admission
        leaves its owner in this scope until caller drain. No FileIO or buffer
        is allocated; callers finish all borrowed descriptor IO before close.
        """
        if type(min_bytes) is not int or min_bytes < 0:
            raise ValueError("a nonnegative minimum file size is required")
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < min_bytes):
            raise ValueError("file byte budget must include its declared minimum")
        if len(self._records) == self.maximum:
            raise RuntimeError("descriptor scope exhausted")
        record = _Record()
        self._records.append(record)  # journal storage precedes native acquisition
        # The valid closed slot is published before its native acquisition.
        # A post-open interruption cannot lose a descriptor or close status
        # in an unobserved constructor return/deallocation.
        record.owner = _owned_slot()
        record.owner.open(path, flags, 0o600)
        before = os.fstat(record.owner.fileno())
        if not stat.S_ISREG(before.st_mode) or (
            before.st_size < min_bytes or (max_bytes is not None and before.st_size > max_bytes)
        ):
            raise ValueError("input must be a regular file within its declared byte budget")
        return record, before

    def open(self, path, flags, mode, *, min_bytes=0, max_bytes=None):
        if mode not in ("rb", "wb", "r+b"):
            raise ValueError("unsupported owned regular-file stream mode")
        record, before = FileStreams.open_descriptor(self, path, flags, min_bytes=min_bytes, max_bytes=max_bytes)
        record.raw = io.FileIO(record.owner.fileno(), mode, closefd=False)
        record.stream = _buffered(record.raw, mode)
        return record, before

    @staticmethod
    def _drain(record):
        errors = []
        if record.done:
            return errors
        if record.stream is not None:
            try:
                record.stream.close()
            except BaseException as error:
                errors.append(error)
        # An interrupted/failed BufferedIO.close may leave pending bytes. Retire
        # the exact borrowed raw view before any descriptor ownership retirement.
        if record.raw is not None and not record.raw.closed:
            for _ in range(2):
                try:
                    record.raw.close()
                except BaseException as error:
                    errors.append(error)
                if record.raw.closed:
                    break
        if record.raw is not None and not record.raw.closed:
            if not errors:
                errors.append(RuntimeError("regular-file raw view did not retire"))
            return errors
        if record.owner is not None and not record.owner.closed:
            for _ in range(2):
                try:
                    record.owner.close()
                except BaseException as error:
                    errors.append(error)
                if record.owner.closed:
                    break
        if record.owner is not None and not record.owner.closed:
            if not errors:
                errors.append(RuntimeError("regular-file descriptor did not retire"))
            return errors
        record.done = True
        return errors

    def close(self, record):
        """Return every cleanup failure; an unretired record stays owned."""
        if not any(record is owned for owned in self._records):
            raise ValueError("stream record does not belong to this scope")
        try:
            return self._drain(record)
        except BaseException as error:
            return [error]

    def drain(self):
        errors = []
        for record in reversed(self._records):
            errors.extend(FileStreams.close(self, record))
        return errors

    @property
    def retired(self):
        return all(record.done for record in self._records)

    @property
    def live_owners(self):
        return tuple(record.owner for record in self._records if record.owner is not None and not record.done)

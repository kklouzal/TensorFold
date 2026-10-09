"""Shared table file view: native POSIX ownership and borrowed unbuffered IO.

Construct a valid closed one-shot view, publish it in the operation journal, then
call open(). Admission follows the opened descriptor, preserving authorized HF
snapshot symlinks. Calls/reads/close are serialized; finish every borrow before
close. closed means its complete owner scope retired; observe close completion
and every raised status. NumPy mappings retain their own legitimate references.
"""
from __future__ import annotations

import io
import os
import stat

from tensorfold.file_io import FileStreams


class TableFile:
    """A pathname-bearing file view; POSIX FileIO borrows a journaled owner.

    Construct and publish this closed slot before open. Its bounded core scope
    publishes a native slot before acquisition, retires the borrowed unbuffered
    view before native close, and reports every close failure. Calls are owned
    by one mapping constructor or one prefetch worker; readers finish before
    close. The existing Windows ordinary-file ownership remains unchanged.
    """

    def __init__(self, path, *, min_bytes=0, max_bytes=None):
        if type(min_bytes) is not int or min_bytes < 0 or (max_bytes is not None and (type(max_bytes) is not int or max_bytes < min_bytes)):
            raise ValueError("table file byte budget must include its nonnegative minimum")
        self.name = os.fspath(path)
        self.minimum, self.maximum = min_bytes, max_bytes
        self._scope = FileStreams(max_files=1) if os.name == "posix" else None
        self._stream = None

    def open(self):
        if self._scope is None:
            if self._stream is not None:
                raise ValueError("table file slot already used")
            self._stream = open(self.name, "rb", buffering=0,
                                opener=lambda name, flags: os.open(name, flags))
            info = os.fstat(self._stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size < self.minimum or (self.maximum is not None and info.st_size > self.maximum):
                raise ValueError("table input must be a regular file within its declared byte budget")
        else:
            record, _ = self._scope.open_descriptor(self.name, os.O_RDONLY | os.O_NONBLOCK,
                                                    min_bytes=self.minimum, max_bytes=self.maximum)
            record.raw = io.FileIO(record.owner.fileno(), "rb", closefd=False)
            self._stream = record.raw

    @property
    def closed(self):
        return self._scope.retired if self._scope is not None else self._stream is None or self._stream.closed

    def fileno(self):
        return self._stream.fileno()

    def read(self, count=-1):
        return self._stream.read(count)

    def readinto(self, buffer):
        return self._stream.readinto(buffer)

    def seek(self, offset, whence=0):
        return self._stream.seek(offset, whence)

    def tell(self):
        return self._stream.tell()

    def flush(self):
        return self._stream.flush()

    def close(self):
        if self._scope is None:
            if self._stream is not None:
                self._stream.close()
            return
        errors = self._scope.drain()
        if not self._scope.retired and not errors:
            errors = [RuntimeError("host file owner scope did not retire")]
        if errors:
            primary = errors[0]
            try:
                prior = BaseException.__cause__.__get__(primary)
                others = []
                for error in errors[1:]:
                    if error is not None and error is not primary and all(error is not item for item in others):
                        others.append(error)
                if others and prior is not None and prior is not primary and all(prior is not item for item in others):
                    others.insert(0, prior)
                dictionary = BaseException.__dict__["__dict__"].__get__(primary)
                if not self._scope.retired:
                    dict.__setitem__(dictionary, "_tensorfold_host_file_scope", self._scope)
                if others:
                    dict.__setitem__(dictionary, "_tensorfold_host_file_failures", others)
                    cause = BaseExceptionGroup("host file close failures", others)
                    dict.pop(dictionary, "_tensorfold_host_file_failures", None)
                    raise primary from cause
                if prior is primary:
                    raise primary from None
                raise primary
            except BaseException as transport:
                if transport is primary:
                    raise
                raise primary from transport


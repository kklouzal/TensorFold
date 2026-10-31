"""Owned stdlib preparation IO shared by source capture and its controller.

Hash inputs are immutable regular files for the whole operation. Ordinary
stream options retain each caller's read/write contract. C624's explicit
retirement and TarFile abort protocol are retained; preallocated ownership
survives even diagnostic exhaustion and context-manager traceback removal.
"""
from __future__ import annotations

import hashlib
import os
import stat
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tensorfold.cleanup import raise_failures

READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


def stamp(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


class RetirementFailure(RuntimeError):
    """Preallocated cause owning failed retirement independently of tracebacks.

    The original operation exception remains the raised object. This cause owns
    its native prior links, failed resource, cleanup error and transport outcome.
    It is allocated before acquiring that resource; failure-path assignments use
    existing slots, not newly allocated dictionaries or diagnostic containers.
    """
    __slots__ = ("cleanup", "original_cause", "original_context", "original_suppress_context", "resource", "transport")

    def __init__(self):
        super().__init__("owned resource retirement failed; inspect cleanup, transport and resource")
        self.resource = None
        self.original_cause = None
        self.original_context = None
        self.original_suppress_context = False
        self.cleanup = None
        self.transport = None


def _raise_cleanup(failure, cleanup):
    # Aggregate on the preallocated carrier, never on the original primary.
    # Allocation or entry failure is allowed: the owning caller has already
    # linked the carrier and records the transport outcome in its fixed slot.
    raise_failures(failure, [cleanup])


@contextmanager
def owned_descriptor(path, flags, mode=0o666, *, dir_fd=None):
    """Own exactly one close attempt; an unresolved close remains a failure."""
    failure = RetirementFailure()
    fd = os.open(path, flags, mode, dir_fd=dir_fd)
    failure.resource = fd
    try:
        yield fd
    except BaseException as primary:
        failure.original_cause = BaseException.__cause__.__get__(primary)
        failure.original_context = BaseException.__context__.__get__(primary)
        failure.original_suppress_context = BaseException.__suppress_context__.__get__(primary)
        try:
            os.close(fd)
        except BaseException as cleanup:  # noqa: BLE001 - preserve operation and descriptor retirement roots.
            failure.cleanup = cleanup
            BaseException.__cause__.__set__(failure, failure.original_cause)
            BaseException.__context__.__set__(failure, failure.original_context)
            BaseException.__suppress_context__.__set__(failure, failure.original_suppress_context)
            BaseException.__cause__.__set__(primary, failure)
            try:
                _raise_cleanup(failure, cleanup)
            except BaseException as transport:  # noqa: BLE001 - recorded on the carrier before rethrowing primary.
                failure.transport = (BaseException.__cause__.__get__(failure)
                                     if transport is failure else transport)
            raise primary from failure
        raise
    else:
        try:
            os.close(fd)
        except BaseException as cleanup:
            failure.original_cause = BaseException.__cause__.__get__(cleanup)
            failure.original_context = BaseException.__context__.__get__(cleanup)
            failure.original_suppress_context = BaseException.__suppress_context__.__get__(cleanup)
            failure.cleanup = cleanup
            BaseException.__cause__.__set__(failure, failure.original_cause)
            BaseException.__context__.__set__(failure, failure.original_context)
            raise cleanup from failure


@contextmanager
def opened_file(path, mode, **options):
    """Retire Path.open's ordinary stream without losing body/close failures."""
    failure = RetirementFailure()
    stream = Path(path).open(mode, **options)  # noqa: SIM115 - this contextmanager owns primary-preserving close.
    failure.resource = stream
    try:
        yield stream
    except BaseException as primary:
        failure.original_cause = BaseException.__cause__.__get__(primary)
        failure.original_context = BaseException.__context__.__get__(primary)
        failure.original_suppress_context = BaseException.__suppress_context__.__get__(primary)
        try:
            stream.close()
        except BaseException as cleanup:  # noqa: BLE001 - preserve operation and output retirement roots.
            failure.cleanup = cleanup
            BaseException.__cause__.__set__(failure, failure.original_cause)
            BaseException.__context__.__set__(failure, failure.original_context)
            BaseException.__suppress_context__.__set__(failure, failure.original_suppress_context)
            BaseException.__cause__.__set__(primary, failure)
            try:
                _raise_cleanup(failure, cleanup)
            except BaseException as transport:  # noqa: BLE001 - recorded on the carrier before rethrowing primary.
                failure.transport = (BaseException.__cause__.__get__(failure)
                                     if transport is failure else transport)
            raise primary from failure
        raise
    else:
        try:
            stream.close()
        except BaseException as cleanup:
            failure.original_cause = BaseException.__cause__.__get__(cleanup)
            failure.original_context = BaseException.__context__.__get__(cleanup)
            failure.original_suppress_context = BaseException.__suppress_context__.__get__(cleanup)
            failure.cleanup = cleanup
            BaseException.__cause__.__set__(failure, failure.original_cause)
            BaseException.__context__.__set__(failure, failure.original_context)
            raise cleanup from failure


def new_file(path):
    """Own a new raw output, retaining partial bytes on write/close failure."""
    return opened_file(path, "xb", buffering=0)


@contextmanager
def borrowed_stream(fd):
    """Own only a binary buffered adapter; its enclosing FD stays borrowed."""
    failure = RetirementFailure()
    stream = os.fdopen(fd, "rb", closefd=False)
    failure.resource = stream
    try:
        yield stream
    except BaseException as primary:
        failure.original_cause = BaseException.__cause__.__get__(primary)
        failure.original_context = BaseException.__context__.__get__(primary)
        failure.original_suppress_context = BaseException.__suppress_context__.__get__(primary)
        try:
            stream.close()
        except BaseException as cleanup:  # noqa: BLE001 - preserve both roots before the descriptor owner runs.
            failure.cleanup = cleanup
            BaseException.__cause__.__set__(failure, failure.original_cause)
            BaseException.__context__.__set__(failure, failure.original_context)
            BaseException.__suppress_context__.__set__(failure, failure.original_suppress_context)
            BaseException.__cause__.__set__(primary, failure)
            try:
                _raise_cleanup(failure, cleanup)
            except BaseException as transport:  # noqa: BLE001 - recorded on the carrier before rethrowing primary.
                failure.transport = (BaseException.__cause__.__get__(failure)
                                     if transport is failure else transport)
            raise primary from failure
        raise
    else:
        try:
            stream.close()
        except BaseException as cleanup:
            failure.original_cause = BaseException.__cause__.__get__(cleanup)
            failure.original_context = BaseException.__context__.__get__(cleanup)
            failure.original_suppress_context = BaseException.__suppress_context__.__get__(cleanup)
            failure.cleanup = cleanup
            BaseException.__cause__.__set__(failure, failure.original_cause)
            BaseException.__context__.__set__(failure, failure.original_context)
            raise cleanup from failure


@contextmanager
def archive(path, mode, **options):
    """Retain TarFile's abort protocol; failed writes never get success EOF."""
    failure = RetirementFailure()
    stream = tarfile.open(path, mode, **options)  # noqa: SIM115 - explicit retirement preserves both failure roots.
    failure.resource = stream
    try:
        yield stream
    except BaseException as primary:
        failure.original_cause = BaseException.__cause__.__get__(primary)
        failure.original_context = BaseException.__context__.__get__(primary)
        failure.original_suppress_context = BaseException.__suppress_context__.__get__(primary)
        try:
            stream.__exit__(type(primary), primary, None)
        except BaseException as cleanup:  # noqa: BLE001 - retain add/read primary if archive abort fails.
            failure.cleanup = cleanup
            BaseException.__cause__.__set__(failure, failure.original_cause)
            BaseException.__context__.__set__(failure, failure.original_context)
            BaseException.__suppress_context__.__set__(failure, failure.original_suppress_context)
            BaseException.__cause__.__set__(primary, failure)
            try:
                _raise_cleanup(failure, cleanup)
            except BaseException as transport:  # noqa: BLE001 - recorded on the carrier before rethrowing primary.
                failure.transport = (BaseException.__cause__.__get__(failure)
                                     if transport is failure else transport)
            raise primary from failure
        raise
    else:
        try:
            stream.close()
        except BaseException as cleanup:
            failure.original_cause = BaseException.__cause__.__get__(cleanup)
            failure.original_context = BaseException.__context__.__get__(cleanup)
            failure.original_suppress_context = BaseException.__suppress_context__.__get__(cleanup)
            failure.cleanup = cleanup
            BaseException.__cause__.__set__(failure, failure.original_cause)
            BaseException.__context__.__set__(failure, failure.original_context)
            raise cleanup from failure


def write_all(stream, block):
    view, done = memoryview(block), 0
    while done < len(view):
        count = stream.write(view[done:])
        if type(count) is not int or not 0 < count <= len(view) - done:
            raise OSError("capture output made invalid write progress")
        done += count


def read_text(path, *, encoding=None):
    """Preserve Path.read_text's codec/newline rules with owned retirement."""
    with opened_file(path, "r", encoding=encoding) as stream:
        return stream.read()


def sha256(path):
    with owned_descriptor(path, READ_FLAGS) as fd:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("capture input must remain a regular file")
        checksum = hashlib.sha256()
        while block := os.read(fd, 1 << 20):
            checksum.update(block)
        if stamp(os.fstat(fd)) != stamp(before) or stamp(os.stat(path, follow_symlinks=False)) != stamp(before):
            raise ValueError("capture input changed during verification")
        return checksum.hexdigest()

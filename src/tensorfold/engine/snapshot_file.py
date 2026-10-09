"""Operation-owned immutable copy of a bounded persisted cache file.

The caller supplies its current tensor/header byte budget before any read. An
opened descriptor must be a regular file; nonblocking/no-follow flags refuse
FIFO and symlink inputs. Source stat changes are detected, not a guarantee
against privileged concurrent mutation. The loader consumes only the private
copy while this scope lives and must complete lazy reads before returning.
The maintained native descriptor provider is required. A failed close which
has not retired its IO views/owner retains its bounded scope in the primary
exception's native ``_tensorfold_snapshot_fd_scope`` journal; the caller must
drain that scope before releasing its operation resources.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile

from tensorfold.cuda.tensor_file import read_header_stream
from tensorfold.engine.snapshot_integrity import prepare_digest
from tensorfold.engine.snapshot_stream import SnapshotStreams


class _SnapshotCleanupInspectionFailure(RuntimeError):
    """Metadata inspection failed by raising the primary exception itself."""


def reject_snapshot_cleanup_failure(error: BaseException) -> None:
    """Allow an ordinary cache miss; propagate failed operation retirement.

    Persistence callers may recover from missing/unreadable data or a full
    disk. An explicit secondary cause or nonempty/malformed native notes
    instead marks an affected operation failure, including retained IO owners.
    Foreign exception getters and dictionary hooks cannot hide that status.
    Native note names are strings or string subclasses; other dictionary keys
    remain extra metadata and gain no note authority through foreign equality.
    """
    cause = BaseException.__dict__["__cause__"].__get__(error)
    if cause is not None:
        raise error
    previous = (
        cause,
        BaseException.__context__.__get__(error),
        BaseException.__suppress_context__.__get__(error),
    )
    try:
        fields = BaseException.__dict__["__dict__"].__get__(error)
        failed_notes = False
        # A builtin dict lookup can invoke a stored key's foreign equality.
        # Read string attribute names directly, including string subclasses,
        # using their builtin string value without hashing or overridden ==.
        for key, value in dict.items(fields):
            if issubclass(type(key), str) and str.__eq__(key, "__notes__") is True:
                # Distinct legal keys can share this string value while using
                # different hashes. An empty alias cannot hide another field.
                failed_notes |= type(value) is not list or len(value) != 0
    except BaseException as inspection:
        metadata_failure = inspection  # retains the formed status in fallback traceback frames
        same_primary = metadata_failure is error
        try:
            if same_primary:
                # A callback can mutate and then raise this same object. Its
                # identity still represents a new failed inspection event.
                BaseException.__cause__.__set__(error, previous[0])
                BaseException.__context__.__set__(error, previous[1])
                BaseException.__suppress_context__.__set__(error, previous[2])
                metadata_failure = _SnapshotCleanupInspectionFailure(
                    "snapshot cleanup metadata inspection raised the primary exception"
                )
            SnapshotStreams()._raise(error, [metadata_failure], annotate=False, previous=previous)
        except BaseException as transport:
            if transport is error:
                if same_primary:
                    # Retain the captured presentation/context while the new
                    # explicit cause keeps nested caller policies fail-fast.
                    BaseException.__context__.__set__(error, previous[1])
                    BaseException.__suppress_context__.__set__(error, previous[2])
                raise
            # Native transport may itself fail to allocate. The same primary
            # and the named prior/status references remain frame-owned; this
            # does not promise every allocation succeeds or a group forms.
            raise error from transport
    if failed_notes:
        raise error


@dataclass(frozen=True)
class SnapshotCopy:
    path: Path
    sha256: str | None
    size: int
    source_identity: tuple[int, ...]
    integrity_sha256: str | None = None


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def snapshot_header(path: Path, *, max_bytes: int, sizes: dict[str, int]):
    """Read only a bounded owned snapshot header, never following a symlink.

    Full payload ranges are checked against this opened regular descriptor's
    size. Stat identity detects ordinary mutation of the opened source;
    it is an index hint, not immutable byte authority. Tensor restoration must
    still take ``private_snapshot`` and bind the selected prefix tokens.
    """
    if type(max_bytes) is not int or max_bytes < 8:
        raise ValueError("explicit snapshot file-byte budget of at least eight required")
    owners = SnapshotStreams()
    primary = None
    try:
        record, before = owners.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, "rb", max_bytes=max_bytes)
        stream = record.stream
        _, header = read_header_stream(stream, sizes, label=path)
        if _identity(os.fstat(stream.fileno())) != _identity(before):
            raise ValueError("snapshot changed during header indexing")
        return header, _identity(before)
    except BaseException as error:
        primary = error
        raise
    finally:
        owners.finish(primary)


@contextmanager
def private_snapshot(path: Path, *, max_bytes: int, integrity_sizes=None, hash_copy=True):
    """Yield owned content only after complete bounded copy and source checks.

    There is no tensor/native import or allocation here. Every owned stream and
    private directory closes on success, failure and caller interruption. A
    cleanup error is an operation failure; it never masks the caller's primary.
    The copy's optional raw SHA binds exact bytes, not model/code authority.
    Integrity verification masks only the declared digest value and runs
    during the same copy. A caller opting out of a redundant raw SHA receives
    sha256=None and the separately named verified integrity_sha256 instead.
    """
    if type(max_bytes) is not int or max_bytes < 8:
        raise ValueError("explicit snapshot file-byte budget of at least eight required")
    if type(hash_copy) is not bool or (not hash_copy and integrity_sizes is None):
        raise ValueError("raw copy hash can be disabled only with explicit integrity verification")
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
    owners = SnapshotStreams()
    primary = None
    try:
        source_record, before = owners.open(path, flags, "rb", max_bytes=max_bytes)
        source = source_record.stream
        integrity = None if integrity_sizes is None else prepare_digest(source, integrity_sizes, label=path)
        owners.directory = tempfile.TemporaryDirectory(prefix="tensorfold-snapshot-")
        copied = Path(owners.directory.name) / "snapshot.safetensors"
        target_record, _ = owners.open(copied, os.O_WRONLY | os.O_CREAT | os.O_EXCL, "wb")
        target = target_record.stream
        digest = hashlib.sha256() if hash_copy else None
        protected = hashlib.sha256() if integrity is not None else None
        remaining = before.st_size
        while remaining:
            chunk = source.read(min(1 << 20, remaining))
            if not chunk:
                raise ValueError("snapshot source shortened during copy")
            written = target.write(chunk)
            if written != len(chunk):
                raise OSError("snapshot private copy did not write the complete chunk")
            if digest is not None:
                digest.update(chunk)
            if integrity is not None:
                integrity.update(protected, chunk, before.st_size - remaining)
            remaining -= len(chunk)
        if source.read(1) or _identity(os.fstat(source.fileno())) != _identity(before):
            raise ValueError("snapshot source changed during copy")
        if integrity is not None:
            integrity.validate(protected)
        owners.close(source_record)
        target.flush()
        owners.close(target_record)
        if copied.stat().st_size != before.st_size:
            raise ValueError("snapshot private copy size differs from source")
        yield SnapshotCopy(
            copied,
            None if digest is None else digest.hexdigest(),
            before.st_size,
            _identity(before),
            None if protected is None else protected.hexdigest(),
        )
    except BaseException as error:
        primary = error
        raise
    finally:
        owners.finish(primary)

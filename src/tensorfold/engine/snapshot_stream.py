"""Snapshot-specific bounded stream, private-directory and error policy.

The core regular stream owns native descriptors and borrowed IO retirement.
Unretired IO retains its private directory and exact-primary error journal.
Caller cleanup must drain the scope before releasing borrowed resource state.
"""

from __future__ import annotations

from tensorfold.file_io import FileStreams


class SnapshotStreams(FileStreams):
    def __init__(self):
        super().__init__(max_files=3)
        self.directory = None
        self._cleanup_failed = False

    def open(self, path, flags, mode, *, max_bytes=None):
        return super().open(path, flags, mode, min_bytes=8 if max_bytes is not None else 0, max_bytes=max_bytes)

    def close(self, record):
        errors = super().close(record)
        if errors:
            self._cleanup_failed = True
            self._raise(None, errors, annotate=False)

    def drain(self):
        errors = super().drain()
        if self.directory is not None and self.retired:
            try:
                self.directory.cleanup()
            except BaseException as error:
                errors.append(error)
            else:
                self.directory = None
        if errors:
            self._cleanup_failed = True
        return errors

    def finish(self, primary=None, errors=()):
        previous = (
            (
                BaseException.__cause__.__get__(primary),
                BaseException.__context__.__get__(primary),
                BaseException.__suppress_context__.__get__(primary),
            )
            if primary is not None
            else None
        )
        errors = list(errors)
        try:
            errors.extend(self.drain())
        except BaseException as cleanup:
            errors.append(cleanup)
            self._cleanup_failed = True
        self._raise(primary, errors, previous=previous)

    def _raise(self, primary, errors, *, annotate=True, previous=None):
        cleanup_errors = errors
        if primary is None:
            if not errors:
                return
            primary, errors = errors[0], errors[1:]
        other = []
        if previous is None:
            previous = (
                BaseException.__cause__.__get__(primary),
                BaseException.__context__.__get__(primary),
                BaseException.__suppress_context__.__get__(primary),
            )
        for error in (*previous[:2], *errors):
            if error is not None and error is not primary and all(error is not item for item in other):
                other.append(error)
        live = self.live_owners
        if live or self.directory is not None:
            namespace = BaseException.__dict__["__dict__"].__get__(primary, type(primary))
            dict.__setitem__(namespace, "_tensorfold_snapshot_fd_owners", live)
            dict.__setitem__(namespace, "_tensorfold_snapshot_fd_scope", self)
        else:
            namespace = BaseException.__dict__["__dict__"].__get__(primary, type(primary))
            if dict.get(namespace, "_tensorfold_snapshot_fd_scope") is self:
                dict.pop(namespace, "_tensorfold_snapshot_fd_scope", None)
                dict.pop(namespace, "_tensorfold_snapshot_fd_owners", None)
        if not cleanup_errors and not self._cleanup_failed:
            # Successful cleanup must preserve ordinary suppressed validation
            # context. Turning a parser's ``raise ... from None`` into an
            # explicit cleanup cause changes the caller's cache-miss policy.
            BaseException.__cause__.__set__(primary, previous[0])
            BaseException.__context__.__set__(primary, previous[1])
            BaseException.__suppress_context__.__set__(primary, previous[2])
            raise primary
        if annotate and (cleanup_errors or self._cleanup_failed):
            try:
                names = ", ".join(type.__dict__["__name__"].__get__(type(error), type)[:80] for error in cleanup_errors)
                note = "snapshot owned stream cleanup failed"
                if names:
                    note += " (" + names + ")"
                BaseException.add_note(primary, note)
            except BaseException as annotation:
                if annotation is not primary and all(annotation is not item for item in other):
                    other.append(annotation)
        if other:
            cause = other[0] if len(other) == 1 else BaseExceptionGroup("snapshot secondary failures", other)
            raise primary from cause
        raise primary

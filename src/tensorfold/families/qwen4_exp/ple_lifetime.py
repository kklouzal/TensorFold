"""Loader/engine ownership of PLE tables and their startup prefetch Futures.

The loader serially acquires tables before validating/packing their consumers.
The engine inherits the same journal. Policy owners must reject/drain requests
before closing it. A failed Future drain or device fence retains all tables for
explicit retry; no mapped/pinned authority can retire before quiescence.
"""
from __future__ import annotations


def _note(primary, message):
    """Diagnostics cannot replace an already established operation failure."""
    try:
        BaseException.add_note(primary, message)
    except BaseException as annotation:
        if annotation is primary:
            raise primary
        raise primary from annotation


def _retain_failure(primary, name, value):
    """Foreign exception attributes cannot intercept resource-journal retention."""
    try:
        dictionary = BaseException.__dict__["__dict__"].__get__(primary)
        dict.__setitem__(dictionary, name, value)
    except BaseException as publication:
        if publication is primary:
            raise primary
        raise primary from publication


class PLETables:
    def __init__(self):
        self.tables = []
        self.reads = []
        self.unpublished = None
        self.closed = False

    def acquire(self, table):
        """Publish acquisition before any subsequent shape/consumer operation."""
        self.unpublished = table
        try:
            if self.closed:
                raise ValueError("the PLE table owner is closed")
            self.tables.append(table)
        except BaseException as primary:
            try:
                table.close()
                if self.tables and self.tables[-1] is table:
                    self.tables.pop()
                self.unpublished = None
            except BaseException as cleanup:
                _note(primary, "unpublished PLE table cleanup failed; owner retained")
                if cleanup is primary:
                    raise primary
                raise primary from cleanup
            raise
        self.unpublished = None
        return table

    def close(self, fence):
        """Drain startup reads, fence consumers, then consume each table owner.

        A completed read error still permits resource cleanup and is surfaced
        after it. An interrupted/incomplete read or failed fence retains the
        journal and every table. Successful individual closes are consumed;
        failed table closes remain retryable, with additional errors noted.
        """
        if self.closed:
            return
        errors, remaining = [], []
        for future in self.reads:
            try:
                if not future.cancel():
                    future.result()
            except BaseException as error:
                errors.append(error)
                try:
                    if not future.done():
                        remaining.append(future)
                    elif not future.cancelled():
                        actual = future.exception(timeout=0)
                        if actual is not None and actual is not error:
                            errors.append(actual)
                except BaseException as status:
                    remaining.append(future)
                    errors.append(status)
        self.reads[:] = remaining
        if not remaining:
            try:
                fence()
            except BaseException as error:
                errors.append(error)
            else:
                kept = []
                for table in self.tables:
                    try:
                        table.close()
                    except BaseException as error:
                        kept.append(table)
                        errors.append(error)
                self.tables[:] = kept
                if self.unpublished is not None:
                    try:
                        self.unpublished.close()
                        self.unpublished = None
                    except BaseException as error:
                        errors.append(error)
                if not self.tables and self.unpublished is None:
                    self.closed = True
        if errors:
            primary = errors[0]
            for error in errors[1:]:
                _note(primary, "additional PLE resource cleanup also failed")
            raise primary

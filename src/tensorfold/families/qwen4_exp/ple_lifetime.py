"""Loader/engine ownership of PLE tables and their startup prefetch Futures.

The loader serially acquires tables before validating/packing their consumers.
The engine inherits the same journal. Policy owners must reject/drain requests
before closing it. A failed Future drain or device fence retains all tables for
explicit retry; no mapped/pinned authority can retire before quiescence.
"""
from __future__ import annotations

from tensorfold.cleanup import raise_failures


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
            prior = (BaseException.__cause__.__get__(primary), BaseException.__context__.__get__(primary),
                     BaseException.__suppress_context__.__get__(primary))
            try:
                table.close()
                if self.tables and self.tables[-1] is table:
                    self.tables.pop()
                self.unpublished = None
            except BaseException as cleanup:
                BaseException.__cause__.__set__(primary, prior[0])
                BaseException.__context__.__set__(primary, prior[1])
                BaseException.__suppress_context__.__set__(primary, prior[2])
                raise_failures(primary, [cleanup])
            BaseException.__cause__.__set__(primary, prior[0])
            BaseException.__context__.__set__(primary, prior[1])
            BaseException.__suppress_context__.__set__(primary, prior[2])
            raise
        self.unpublished = None
        return table

    def close(self, fence):
        """Drain startup reads, fence consumers, then consume each table owner.

        A completed read error still permits resource cleanup and is surfaced
        after it. An interrupted/incomplete read or failed fence retains the
        journal and every table. Successful individual closes are consumed;
        failed table closes remain retryable. Every distinct failure survives;
        the first failure's native roots are captured before further callbacks.
        """
        if self.closed:
            return
        errors, remaining, prior = [], [], None

        def record(error):
            nonlocal prior
            if not errors:
                prior = (BaseException.__cause__.__get__(error), BaseException.__context__.__get__(error),
                         BaseException.__suppress_context__.__get__(error))
            errors.append(error)

        for future in self.reads:
            try:
                if not future.cancel():
                    future.result()
            except BaseException as error:
                record(error)
                try:
                    if not future.done():
                        remaining.append(future)
                    elif not future.cancelled():
                        actual = future.exception(timeout=0)
                        if actual is not None and actual is not error:
                            record(actual)
                except BaseException as status:
                    remaining.append(future)
                    record(status)
        self.reads[:] = remaining
        if not remaining:
            try:
                fence()
            except BaseException as error:
                record(error)
            else:
                kept = []
                for table in self.tables:
                    try:
                        table.close()
                    except BaseException as error:
                        kept.append(table)
                        record(error)
                self.tables[:] = kept
                if self.unpublished is not None:
                    try:
                        self.unpublished.close()
                        self.unpublished = None
                    except BaseException as error:
                        record(error)
                if not self.tables and self.unpublished is None:
                    self.closed = True
        if errors:
            primary = errors[0]
            BaseException.__cause__.__set__(primary, prior[0])
            BaseException.__context__.__set__(primary, prior[1])
            BaseException.__suppress_context__.__set__(primary, prior[2])
            raise_failures(primary, errors[1:])

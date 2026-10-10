"""Operation and descriptor ownership shared by the two SSD row readers.

Table methods borrow the pool and ordinary FileIO owners through preparation,
all accepted reads, and result assembly. Close rejects new borrows and drains
existing ones before releasing either resource. A failed drain retains owners
for explicit retry; a consumed FileIO descriptor is never closed by raw number.
"""
from __future__ import annotations

import threading

from tensorfold.cleanup import raise_failures


def _failure_roots(error):
    return (BaseException.__cause__.__get__(error), BaseException.__context__.__get__(error),
            BaseException.__suppress_context__.__get__(error))


def _raise_failures(primary, errors, prior):
    """Retire journals before reporting all concrete statuses and saved roots."""
    BaseException.__cause__.__set__(primary, prior[0])
    BaseException.__context__.__set__(primary, prior[1])
    BaseException.__suppress_context__.__set__(primary, prior[2])
    raise_failures(primary, errors)


def _note(primary, message):
    """Keep the first failed annotation visible without replacing work failure."""
    cause = BaseException.__cause__.__get__(primary, type(primary))
    if type(cause) in (BaseExceptionGroup, ExceptionGroup) and cause.message == "table failure annotation errors":
        return
    try:
        BaseException.add_note(primary, message)
    except BaseException as error:
        errors = [error] if cause is None else [cause, error]
        BaseException.__cause__.__set__(primary, BaseExceptionGroup("table failure annotation errors", errors))


def _cause(primary, cleanup):
    cause = BaseException.__cause__.__get__(primary, type(primary))
    if cause is None or cause is cleanup:
        return cleanup
    pending = [cause]
    while pending:
        prior = pending.pop()
        if prior is cleanup:
            return cause
        if type(prior) in (BaseExceptionGroup, ExceptionGroup):
            pending.extend(prior.exceptions)
    return BaseExceptionGroup("table cleanup and prior failure statuses", [cleanup, cause])


def release_files(fds, files):
    """Consume FileIO owners, retaining every handle whose close did not finish."""
    while files:
        file = files[-1]
        try:
            file.close()
        finally:
            if file.closed:
                files.pop()
    fds.clear()


class PythonPool:
    """The same explicit close contract as the native Reader."""
    def __init__(self, executor):
        self.executor = executor

    def close(self):
        self.executor.shutdown(wait=True)


class TableLifetime:
    def __init__(self, pool, fds, release_fds):
        self.pool, self.fds, self.release_fds = pool, fds, release_fds
        self.condition = threading.Condition()
        self.active = {}
        self.workers = {}
        self.reads = {}
        self.closing = False
        self.closed = False
        self.broken = None

    def call(self, work, *args):
        token, primary = object(), None
        try:
            with self.condition:
                if self.closing or self.closed or self.broken is not None:
                    raise ValueError("the n-gram table is closed or unusable") from self.broken
                self.active[token] = threading.get_ident()
            return work(*args)
        except BaseException as error:
            primary = error
            raise
        finally:
            interrupted = None
            while True:
                try:
                    with self.condition:
                        self.active.pop(token, None)
                        self.condition.notify_all()
                    break
                except KeyboardInterrupt as error:
                    if interrupted is None:
                        interrupted = error
            if interrupted is not None:
                if primary is not None:
                    _note(primary, "table borrower retirement was interrupted")
                    raise primary from _cause(primary, interrupted)
                raise interrupted

    def parallel(self, executor, work, jobs):
        """Drain all accepted reads, including submit accepting before raising.

        Only at most WORKERS already bounded batches reach this method. Each
        invocation writes its own terminal journal; failed submission or caller
        interruption shuts down the pool and observes even a lost Future.
        Failed read ownership makes the table unusable until close completes.
        """
        terminal = [None] * len(jobs)
        futures, primary = [], None
        journal = object()
        with self.condition:
            self.reads[journal] = (terminal, None)

        def invoke(index, job):
            token = object()
            primary = None
            try:
                with self.condition:
                    # Failed dispatch owns cancellation before a late callback
                    # can acquire file authority. An interrupted Thread.join
                    # alone cannot prove that its Python callback has stopped.
                    if terminal[index] is not None:
                        return
                    self.workers[token] = threading.get_ident()
                    terminal[index] = "running"
                work(job)
                with self.condition:
                    terminal[index] = True
            except BaseException as error:
                primary = error
                with self.condition:
                    terminal[index] = error
                raise
            finally:
                interrupted = None
                while True:
                    try:
                        with self.condition:
                            # Publish every retirement error before consumers
                            # can observe this worker as quiescent. The lock
                            # covers status repair and its final propagation.
                            if interrupted is not None:
                                if primary is None:
                                    terminal[index] = interrupted
                                else:
                                    _note(primary, "SSD worker journal retirement was interrupted")
                            self.workers.pop(token, None)
                            self.condition.notify_all()
                            if interrupted is not None:
                                if primary is not None:
                                    raise primary from _cause(primary, interrupted)
                                raise interrupted
                        break
                    except KeyboardInterrupt as cleanup:
                        if cleanup is interrupted or cleanup is primary:
                            raise
                        if interrupted is None:
                            interrupted = cleanup

        try:
            for index, job in enumerate(jobs):
                futures.append(executor.submit(invoke, index, job))
            for future in futures:
                future.result()
        except BaseException as error:
            primary = error
            prior, errors = _failure_roots(primary), []
            with self.condition:
                self.broken = error
                self.reads[journal] = (terminal, error)
                for index, value in enumerate(terminal):
                    if value is None:
                        terminal[index] = False  # never entered: explicitly cancelled
            # Join every accepted callback, including unknown publication.
            while True:
                try:
                    executor.shutdown(wait=True)
                    break
                except KeyboardInterrupt as interruption:
                    if interruption is not primary:
                        errors.append(interruption)
                except BaseException as cleanup:
                    errors.append(cleanup)
                    _raise_failures(primary, errors, prior)
            # Thread.join may be interrupted after consuming its state lock.
            # Only the actual callback journal proves that file users retired.
            while True:
                try:
                    with self.condition:
                        self.condition.wait_for(lambda: not self.workers)
                    break
                except KeyboardInterrupt as interruption:
                    errors.append(interruption)
        if primary is not None:
            for error in terminal:
                if isinstance(error, BaseException) and error is not primary:
                    errors.append(error)
            with self.condition:
                self.reads.pop(journal)
            _raise_failures(primary, errors, prior)
        if any(value is not True for value in terminal):
            raise RuntimeError("SSD read pool completed without every terminal journal")
        with self.condition:
            self.reads.pop(journal)

    def close(self):
        lease = threading.Lock()
        with lease:
            primary = None
            acquired = False
            retained, errors = [], []
            try:
                with self.condition:
                    current = threading.get_ident()
                    if current in self.active.values() or current in self.workers.values():
                        raise RuntimeError("a table borrower cannot close its own read owner")
                    while self.closing and not self.closed:
                        if not self.closing.locked():
                            self.closing = False
                            break
                        self.condition.wait(.1)
                    if self.closed:
                        return
                    self.closing = lease
                    acquired = True
                    self.condition.wait_for(lambda: not self.active)
                # Snapshot failed dispatch roots before a pool callback can
                # rewrite them. Live terminal journals remain authoritative
                # until the pool and every actual worker have retired.
                retained = [(original, _failure_roots(original))
                            for _, original in self.reads.values() if original is not None]
                interrupted = None
                while True:
                    try:
                        self.pool.close()
                        break
                    except KeyboardInterrupt as error:
                        if interrupted is None:
                            interrupted = error
                    except BaseException as cleanup:
                        if interrupted is not None:
                            _note(interrupted, "SSD pool close also failed; owner retained")
                            raise interrupted from _cause(interrupted, cleanup)
                        raise
                with self.condition:
                    self.condition.wait_for(lambda: not self.workers)
                for terminal, original in self.reads.values():
                    if original is not None:
                        errors.append(original)
                    for error in terminal:
                        if isinstance(error, BaseException) and error is not original:
                            if original is None:
                                # This journal never reported its failure to a
                                # dispatch owner. Preserve fail-fast release
                                # ordering and keep the journal for retry.
                                _raise_failures(error, [item for item in terminal
                                                       if isinstance(item, BaseException)], _failure_roots(error))
                            errors.append(error)
                for original, prior in retained:
                    BaseException.__cause__.__set__(original, prior[0])
                    BaseException.__context__.__set__(original, prior[1])
                    BaseException.__suppress_context__.__set__(original, prior[2])
                self.reads.clear()
                while True:
                    try:
                        self.release_fds(self.fds)
                        break
                    except KeyboardInterrupt as error:
                        if interrupted is None:
                            interrupted = error
                with self.condition:
                    self.closed = True
                    self.broken = None
                if interrupted is not None:
                    if errors:
                        errors.append(interrupted)
                        _raise_failures(errors[0], errors[1:], retained[0][1])
                    raise interrupted
                if errors:
                    _raise_failures(errors[0], errors[1:], retained[0][1])
            except BaseException as error:
                primary = error
                if acquired and not self.closed:
                    self.broken = error
                if self.reads:
                    # A failed drain retains journals for retry. Restore their
                    # captured roots now, before the next close can snapshot
                    # foreign cleanup's mutations as authoritative history.
                    for original, prior in retained:
                        BaseException.__cause__.__set__(original, prior[0])
                        BaseException.__context__.__set__(original, prior[1])
                        BaseException.__suppress_context__.__set__(original, prior[2])
                if errors and error is not errors[0]:
                    # Descriptor release may fail after journal retirement.
                    # Keep the previously reported dispatch statuses together
                    # with this new failure rather than losing cleared reads.
                    errors.append(error)
                    primary = errors[0]
                    if acquired and not self.closed:
                        self.broken = primary
                    _raise_failures(primary, errors[1:], retained[0][1])
                raise
            finally:
                interrupted = None
                while True:
                    try:
                        with self.condition:
                            if self.closing is lease:
                                self.closing = False
                            self.condition.notify_all()
                        break
                    except KeyboardInterrupt as error:
                        if interrupted is None:
                            interrupted = error
                if interrupted is not None:
                    if primary is not None:
                        _note(primary, "SSD close journal retirement was interrupted")
                        raise primary from _cause(primary, interrupted)
                    raise interrupted

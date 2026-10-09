"""Operation-owned thread work completion, independent of Thread.join state.

A task publishes completion after its callback has returned or failed. Drain
waits for that journal before joining, so an interrupted CPython join cannot
permit resource retirement while callback work remains. Thread.start is gated:
if it raises after spawning, cancellation is published before the callback can
enter. Such a cancelled startup may finish native thread bookkeeping later, but
cannot access the retired callback or its resources. This is work quiescence,
not a claim that an interrupted native thread join reaped the OS thread.
"""
from __future__ import annotations

import threading


class Task:
    def __init__(self, target, *, name=None, daemon=False):
        self._gate = threading.Lock()
        self._target = target
        self.done = threading.Event()
        self.error = None
        self.started = False
        self.entered = False
        self.start_cancelled = False
        self.thread = threading.Thread(target=self._run,name=name,daemon=daemon)

    def start(self):
        with self._gate:
            if self.started or self.start_cancelled:
                raise RuntimeError('owned task can only start once')
            try:
                self.thread.start()
                self.started = True
            except BaseException:
                self.started = False
                self.start_cancelled = True
                self._target = None
                self.done.set()
                raise

    def cancel_unstarted(self):
        with self._gate:
            if not self.started and not self.entered:
                self.start_cancelled = True
                self._target = None
                self.done.set()

    def _run(self):
        target = None
        try:
            with self._gate:
                if self.start_cancelled:
                    return
                self.entered = True
                target = self._target
            target()
        except BaseException as error:
            self.error = error
        finally:
            self._target = target = None
            self.done.set()


def _raise_failures(primary, errors):
    if primary is None and errors:
        primary, errors = errors[0], errors[1:]
    if primary is None:
        return
    previous = (BaseException.__cause__.__get__(primary), BaseException.__context__.__get__(primary))
    others = []
    for error in errors:
        if error is not primary and all(error is not previous for previous in others):
            others.append(error)
    annotations = []
    annotation_failed = False
    messages = ['owned task cleanup also failed (' +
                str.__getitem__(type.__dict__['__name__'].__get__(type(error)), slice(None, 64)) + ')'
                for error in others[:8]]
    if len(others) > 8:
        messages.append('additional owned failure count: ' + str(len(others) - 8))
    for message in messages:
        try:
            BaseException.add_note(primary, message)
        except BaseException as annotation:
            annotation_failed = True
            if annotation is not primary and all(annotation is not error for error in [*others, *annotations]):
                annotations.append(annotation)
            break
    causes = []
    if others or annotations:
        for error in previous:
            if error is not None and error is not primary and all(error is not item for item in causes):
                causes.append(error)
    for error in [*others, *annotations]:
        if all(error is not previous for previous in causes):
            causes.append(error)
    if len(causes) == 1 and not annotation_failed:
        raise primary from causes[0]
    if causes:
        raise primary from BaseExceptionGroup('owned task cleanup and retained native failures', causes)
    raise primary


def raise_failures(primary, errors):
    """Transport retired statuses without replacing the primary on exhaustion.

    The original status sequence and captured native fields remain owned by
    this traceback frame if cold annotation or grouping cannot allocate. Such
    a transport failure is the explicit cause; no successful grouping or
    continued retention after a caller clears traceback frames is promised.
    """
    if primary is None:
        if not errors:
            return
        primary = errors[0]
    native_cause = BaseException.__cause__.__get__(primary)
    native_context = BaseException.__context__.__get__(primary)
    if native_cause is primary:
        native_cause = None
    if native_context is primary:
        native_context = None
    try:
        _raise_failures(primary, errors)
    except BaseException as failure:
        if failure is primary:
            raise
        # These references deliberately remain in the failure traceback.
        # Required task retirement has already happened in drain().
        raise primary from failure


def drain(tasks, primary=None):
    """Observe every accepted callback and preserve caller interruption.

    Callers stop their producers first. Completion waits remain required even
    when public Thread.is_alive()/join incorrectly report a stopped worker.
    A new interruption during drain is retained while the remaining work is
    still awaited. No task resource may be freed before its done journal.
    """
    errors = []
    index = 0
    while index < len(tasks):
        try:
            task = tasks[index]
            task.cancel_unstarted()
            while not task.done.is_set():
                task.done.wait()
            if task.started:
                try:
                    task.thread.join()
                except BaseException as error:
                    errors.append(error)
            if task.error is not None and all(task.error is not error for error in errors):
                errors.append(task.error)
            index += 1
        except BaseException as error:
            errors.append(error)
    raise_failures(primary,errors)

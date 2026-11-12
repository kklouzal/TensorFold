"""Failure notification to the owner of a supervised one-host TP process group.

The descriptor is a borrowed process-lifetime capability installed only by the
launcher. Reporting a fatal operation does not destroy a communicator while
other threads or ranks still borrow it. The supervisor retires every process.
Unmanaged launches retain their existing external containment responsibility.
"""

from __future__ import annotations

import os
import stat
import sys
import json
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar

from tensorfold.cleanup import raise_failures

FAILURE_FD_ENV = "TENSORFOLD_TP_FAILURE_FD"

# Explicit constructor-thread scope. Tokens restore nested owners and no
# callback, exception or model reference remains installed after construction.
# The successful serving path never reads this context; only fatal reporting does.
_STARTUP_DIAGNOSTIC = ContextVar('tensorfold_startup_diagnostic', default=None)


def _diagnostic_text(value, limit):
    if type(value) is not str:
        return None
    text = value[:limit].encode('ascii', 'replace').decode('ascii')
    return ''.join(character if ' ' <= character <= '~' else ' ' for character in text)


def _exception_label(error):
    name = type.__dict__['__name__'].__get__(type(error), type)
    args = BaseException.args.__get__(error, type(error))
    # Do not invoke user __str__/__repr__, tensor formatting, locals or source
    # lines. Native exception string arguments contain the actionable error.
    message = next((_diagnostic_text(value, 512) for value in args[:3] if type(value) is str), None)
    return _diagnostic_text(name, 96), message


class _StartupDiagnostic:
    def __init__(self, rank):
        self.rank = rank if type(rank) is int and 0 <= rank <= (1 << 31) - 1 else None
        self.attempted = False

    def report(self, error):
        if self.attempted:
            return False
        self.attempted = True  # A failed diagnostic write is retained, never retried to hide uncertainty.
        kind, message = _exception_label(error)
        frames = deque(maxlen=6)
        trace = BaseException.__traceback__.__get__(error, type(error))
        for _ in range(128):
            if trace is None:
                break
            code = trace.tb_frame.f_code
            frames.append({'file': _diagnostic_text(os.path.basename(code.co_filename), 80),
                           'function': _diagnostic_text(code.co_name, 48), 'line': trace.tb_lineno})
            trace = trace.tb_next
        record = {'schema': 'tensorfold.constructor-failure-v1', 'rank': self.rank,
                  'phase': 'FlashNextEngine.constructor', 'error_type': kind, 'message': message,
                  'frames': list(frames), 'frames_truncated': trace is not None}
        cause = BaseException.__cause__.__get__(error, type(error))
        if cause is not None and cause is not error:
            cause_type, cause_message = _exception_label(cause)
            record['cause_type'] = cause_type
            record['cause_message'] = None if cause_message is None else cause_message[:192]
        raw = json.dumps(record, ensure_ascii=True, separators=(',', ':')).encode() + b'\n'
        while len(raw) > 4096 and record['frames']:
            record['frames'].pop(0)
            record['frames_truncated'] = True
            raw = json.dumps(record, ensure_ascii=True, separators=(',', ':')).encode() + b'\n'
        if len(raw) > 4096:
            raise ValueError('constructor failure diagnostic exceeded its bounded record')
        if os.write(2, raw) != len(raw):
            raise OSError('constructor failure diagnostic write was incomplete')
        return True


@contextmanager
def startup_failure_context(rank):
    """Own one bounded first-failure report until this constructor retires."""
    owner = _StartupDiagnostic(rank)
    token = _STARTUP_DIAGNOSTIC.set(owner)
    try:
        yield owner
    finally:
        _STARTUP_DIAGNOSTIC.reset(token)


def report_startup_failure(error):
    owner = _STARTUP_DIAGNOSTIC.get()
    return False if owner is None else owner.report(error)


def notify_failure() -> bool:
    value = os.environ.get(FAILURE_FD_ENV)
    if value is None:
        return False
    if not value.isdecimal():
        raise RuntimeError("invalid TP supervisor failure descriptor")
    descriptor = int(value)
    if not stat.S_ISFIFO(os.fstat(descriptor).st_mode):
        raise RuntimeError("TP supervisor failure descriptor is not a pipe")
    diagnostic = None
    error = sys.exception()
    if error is not None:
        try:
            report_startup_failure(error)
        except BaseException as failure:
            diagnostic = failure
    # One writer per rank emits once after poisoning. Eight bytes fit within a
    # pipe's atomic-write bound; a nonblocking failure is surfaced to the owner.
    try:
        if os.write(descriptor, b"F") != 1:
            raise RuntimeError("TP supervisor did not accept the fatal-cohort notification")
    except BaseException as notification:
        if diagnostic is not None:
            raise_failures(error, [diagnostic, notification])
        raise
    if diagnostic is not None:
        raise_failures(error, [diagnostic])
    return True

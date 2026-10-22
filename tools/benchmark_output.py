"""Publish one complete benchmark JSON document with an owned temporary file.

Successful replacement is atomic: readers see the previous or complete next
report. Concurrent publishers retain the existing last-replacement-wins
contract. A post-replacement directory fsync failure reports committed but
unconfirmed durability; it never overwrites a newer writer during rollback.
An unproved stream close retains that stream and its raw descriptor in the
owning failure traceback until explicit recovery or process retirement. Never
clear those frames and continue as though their resource retirement succeeded.
"""
import json
import os
import stat
import tempfile
from pathlib import Path

if __package__:
    from .worker_lifetime import raise_failures
else:
    from worker_lifetime import raise_failures


def write_json(path, value, *, indent=1):
    target = Path(path)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError('benchmark output must be an ordinary file path')
    mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else None
    descriptor = temporary = stream = directory = None
    stream_close_attempted = False
    replaced = False
    primary = None
    prior_cause = prior_context = None
    prior_suppress = False
    # Four resource retirements, failed-close verification and an annotation.
    # Prepare status storage before creating any owned file or descriptor.
    errors, error_count = [None] * 6, 0
    try:
        descriptor, temporary = tempfile.mkstemp(prefix='.' + target.name + '.', suffix='.tmp', dir=target.parent)
        if mode is not None:
            os.fchmod(descriptor, mode)
        # The raw descriptor remains owned by this operation. closefd=False
        # avoids a duplicate owner during an interrupted stream construction.
        stream = os.fdopen(descriptor, 'w', encoding='utf-8', closefd=False)
        json.dump(value, stream, indent=indent, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
        stream_close_attempted = True
        stream.close()
        stream = None
        try:
            os.close(descriptor)
        finally:
            # A failed close must not be retried against a reused FD number.
            descriptor = None
        os.replace(temporary, target)
        replaced = True
        temporary = None
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        os.fsync(directory)
    except BaseException as error:  # noqa: BLE001 - every owning write/interruption failure reaches independent retirement.
        primary = error
        prior_cause = BaseException.__cause__.__get__(error)
        prior_context = BaseException.__context__.__get__(error)
        prior_suppress = BaseException.__suppress_context__.__get__(error)
    if stream is not None and not stream_close_attempted:
        try:
            stream_close_attempted = True
            stream.close()
        except BaseException as error:  # noqa: BLE001 - every owned resource retirement remains independent.
            errors[error_count] = error
            error_count += 1
        else:
            stream = None
    if stream is not None:
        try:
            if stream.closed is True:
                stream = None
        except BaseException as error:  # noqa: BLE001 - an unproved stream remains paired with its live raw descriptor.
            errors[error_count] = error
            error_count += 1
    if descriptor is not None and stream is None:
        try:
            os.close(descriptor)
        except BaseException as error:  # noqa: BLE001 - a failed close is retained, never retried against a reused FD.
            errors[error_count] = error
            error_count += 1
    if directory is not None:
        try:
            os.close(directory)
        except BaseException as error:  # noqa: BLE001 - directory failure cannot suppress owned temporary retirement.
            errors[error_count] = error
            error_count += 1
    if temporary is not None:
        try:
            os.unlink(temporary)
        except BaseException as error:  # noqa: BLE001 - preserve the exact unretired owned pathname in this frame.
            errors[error_count] = error
            error_count += 1
    if stream is not None or (replaced and (primary is not None or error_count)):
        try:
            BaseException.add_note(primary if primary is not None else errors[0],
                                   'benchmark stream close is unproved; raw descriptor remains owned'
                                   if stream is not None else
                                   'benchmark output was replaced; directory durability was not confirmed')
        except BaseException as error:  # noqa: BLE001 - annotation runs only after every resource retirement attempt.
            errors[error_count] = error
            error_count += 1
    while len(errors) > error_count:
        errors.pop()
    if primary is not None:
        BaseException.__cause__.__set__(primary, prior_cause)
        BaseException.__context__.__set__(primary, prior_context)
        BaseException.__suppress_context__.__set__(primary, prior_suppress)
    raise_failures(primary, errors)

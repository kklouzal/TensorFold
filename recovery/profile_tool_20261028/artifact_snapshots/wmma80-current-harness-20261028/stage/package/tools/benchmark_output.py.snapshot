"""Publish one complete benchmark JSON document with an owned temporary file.

Successful replacement is atomic: readers see the previous or complete next
report. Concurrent publishers retain the existing last-replacement-wins
contract. A post-replacement directory fsync failure reports committed but
unconfirmed durability; it never overwrites a newer writer during rollback.
"""
import json
import os
from pathlib import Path
import stat
import tempfile


def _close(operation, primary):
    try:
        operation()
    except BaseException as cleanup:
        if primary is not None:
            BaseException.add_note(primary, 'benchmark resource close failed (' + type(cleanup).__name__ + ')')
            raise primary from cleanup
        raise


def write_json(path, value, *, indent=1):
    target = Path(path)
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError('benchmark output must be an ordinary file path')
    mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else None
    descriptor = temporary = None
    replaced = False
    primary = None
    try:
        descriptor, name = tempfile.mkstemp(prefix='.' + target.name + '.', suffix='.tmp', dir=target.parent)
        temporary = Path(name)
        if mode is not None:
            os.fchmod(descriptor, mode)
        # The raw descriptor remains owned by this operation. closefd=False
        # avoids a duplicate owner during an interrupted stream construction.
        stream = os.fdopen(descriptor, 'w', encoding='utf-8', closefd=False)
        file_error = None
        try:
            json.dump(value, stream, indent=indent, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException as error:
            file_error = error
            raise
        finally:
            _close(stream.close, file_error)
        try:
            os.close(descriptor)
        finally:
            # A failed close must not be retried against a reused FD number.
            descriptor = None
        os.replace(temporary, target)
        replaced = True
        temporary = None
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        directory_error = None
        try:
            os.fsync(directory)
        except BaseException as error:
            directory_error = error
            raise
        finally:
            _close(lambda: os.close(directory), directory_error)
    except BaseException as error:
        primary = error
        if replaced:
            BaseException.add_note(error, 'benchmark output was replaced; directory durability was not confirmed')
        raise
    finally:
        errors = []
        if descriptor is not None:
            try:
                os.close(descriptor)
            except BaseException as error:
                errors.append(error)
        if temporary is not None:
            try:
                temporary.unlink()
            except BaseException as error:
                errors.append(error)
        if errors:
            if primary is not None:
                for error in errors:
                    BaseException.add_note(primary, 'benchmark temporary cleanup failed (' + type(error).__name__ + ')')
                raise primary from errors[0]
            raise errors[0]

"""Shared cold CUDA stream-context outcome ownership for graph preparation."""
from pathlib import Path

from tensorfold.cleanup import raise_failures


def on_stream(cuda, stream, operation):
    secondary = [None]  # Reserve before stream entry or owned operation.
    primary = None
    try:
        with cuda.stream(stream):
            try:
                operation()
            except BaseException as error:
                primary = error
    except BaseException as error:
        secondary[0] = error
    if secondary[0] is not None:
        raise_failures(primary, secondary)
    if primary is not None:
        raise primary


def host_memory():
    rows = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:')):
            key, number, unit = line.split()
            if unit != 'kB':
                raise ValueError('unexpected host memory units')
            rows[key[:-1]] = int(number) * 1024
    if rows.keys() != {'VmRSS', 'VmHWM'}:
        raise RuntimeError('private graph preparation requires Linux RSS/HWM observations')
    return rows


def finish_forward_frontiers(primary, first, second, third, journal):
    """Attempt every prebound frontier using only preallocated outcome slots.

    The caller allocates (raw3, one1, pair2, triple3) before forward submission.
    A frontier may re-raise the original primary after attaching its own cause.
    """
    raw, one, pair, triple = journal
    try:
        if first is not None:
            first(primary)
    except BaseException as error:
        if error is not primary:
            raw[0] = error
    try:
        if second is not None:
            second(primary)
    except BaseException as error:
        if error is not primary:
            raw[1] = error
    try:
        if third is not None:
            third(primary)
    except BaseException as error:
        if error is not primary:
            raw[2] = error
    count = 0
    if raw[0] is not None:
        triple[count] = raw[0]
        count += 1
    if raw[1] is not None:
        triple[count] = raw[1]
        count += 1
    if raw[2] is not None:
        triple[count] = raw[2]
        count += 1
    if count == 1:
        one[0] = triple[0]
        raise_failures(primary, one)
    if count == 2:
        pair[0], pair[1] = triple[0], triple[1]
        raise_failures(primary, pair)
    if count == 3:
        raise_failures(primary, triple)

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

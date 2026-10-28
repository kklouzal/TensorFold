"""Shared cold CUDA stream-context outcome ownership for graph preparation."""
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

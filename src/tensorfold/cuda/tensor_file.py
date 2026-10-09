"""Safetensors file boundary checks shared by CUDA byte readers.

The public format allows scalars and empty dimensions. Tensor payloads must
cover the data buffer exactly, without overlaps, gaps or trailing data. Header
size follows safetensors' 100,000,000-byte protocol limit. This module never
allocates tensors, interprets tensor values or imports an accelerator runtime.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
import stat
import struct

MAX_HEADER_BYTES = 100_000_000
MAX_DIMENSION = 2**63 - 1  # PyTorch's IntArrayRef / TensorImpl dimension ABI.
MAX_COUNT = 2**64 - 1  # safetensors' checked usize multiplication on the 64-bit TARGET.
_OPEN_NONBLOCK = os.O_NONBLOCK if os.name == "posix" else 0


@contextmanager
def _regular_stream(path):
    """Own a checkpoint file; a FIFO cannot block before type validation.

    Linux/macOS nonblocking open leaves ordinary file reads unchanged and
    permits standard HF snapshot symlinks. The opened descriptor, rather than
    a separately inspected pathname, establishes the regular-file contract.
    Experimental native Windows preserves its ordinary-file open semantics;
    POSIX FIFO nonblocking behavior is not a Windows named-pipe guarantee.
    """
    if os.name == "posix":
        from tensorfold.file_io import FileStreams

        scope = FileStreams(max_files=1)
        primary = None
        try:
            record, _ = scope.open(path, os.O_RDONLY | _OPEN_NONBLOCK, "rb")
            yield record.stream
        except BaseException as error:
            primary = error
        finally:
            # Close may invoke native/profile callbacks. Capture the original
            # transport before draining so successful cleanup cannot change it.
            prior_cause = None if primary is None else BaseException.__cause__.__get__(primary, type(primary))
            prior_context = None if primary is None else BaseException.__context__.__get__(primary, type(primary))
            prior_suppress = None if primary is None else BaseException.__suppress_context__.__get__(primary, type(primary))
            try:
                cleanup = scope.drain()
            except BaseException as error:
                cleanup = [error]
            if not scope.retired and not cleanup:
                cleanup = [RuntimeError("checkpoint file scope did not retire")]
            cleanup_errors = cleanup
            if primary is None and cleanup:
                primary, cleanup = cleanup[0], cleanup[1:]
                prior_cause = BaseException.__cause__.__get__(primary, type(primary))
                prior_context = BaseException.__context__.__get__(primary, type(primary))
            if primary is not None:
                try:
                    dictionary = BaseException.__dict__['__dict__'].__get__(primary, type(primary))
                    if not scope.retired:
                        dict.__setitem__(dictionary, '_tensorfold_checkpoint_file_scope', scope)
                    elif dict.get(dictionary, '_tensorfold_checkpoint_file_scope') is scope:
                        dict.pop(dictionary, '_tensorfold_checkpoint_file_scope', None)
                    if not cleanup_errors:
                        BaseException.__cause__.__set__(primary, prior_cause)
                        BaseException.__context__.__set__(primary, prior_context)
                        BaseException.__suppress_context__.__set__(primary, prior_suppress)
                        raise primary
                    previous = BaseException.__cause__.__get__(primary, type(primary))
                    current_context = BaseException.__context__.__get__(primary, type(primary))
                    others = []
                    for error in (prior_cause, prior_context, previous, current_context, *cleanup):
                        if error is not None and error is not primary and all(error is not item for item in others):
                            others.append(error)
                    try:
                        BaseException.add_note(primary, "checkpoint file cleanup also failed")
                    except BaseException as annotation:
                        if annotation is not primary and all(annotation is not item for item in others):
                            others.append(annotation)
                    if others:
                        # A group allocation failure must not hide the primary
                        # or release its already-formed statuses/live scope.
                        dict.__setitem__(dictionary, '_tensorfold_checkpoint_file_failures', others)
                        cause = BaseExceptionGroup("checkpoint file secondary failures", others)
                        dict.pop(dictionary, '_tensorfold_checkpoint_file_failures', None)
                        raise primary from cause
                    if previous is primary:
                        raise primary from None
                    raise primary
                except BaseException as transport:
                    if transport is primary:
                        raise
                    # Locals in this traceback retain every prior/status and
                    # the scope even if native journal publication cannot allocate.
                    raise primary from transport
        return
    # The existing experimental native Windows ordinary-file path is distinct
    # from POSIX nonblocking admission and needs no raw-descriptor opener.
    stream = open(path, "rb")
    primary = None
    try:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError(f"{path}: checkpoint input must be a regular file")
        yield stream
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            stream.close()
        except BaseException as cleanup:
            if primary is not None:
                if cleanup is primary:
                    raise primary
                try:
                    BaseException.add_note(primary, "checkpoint file close also failed")
                except BaseException as annotation:
                    if annotation is primary:
                        raise primary from cleanup
                    raise primary from BaseExceptionGroup(
                        "checkpoint close and failure annotation errors", [cleanup, annotation]
                    )
                raise primary from cleanup
            raise


def tensor_shape(shape) -> int:
    """Validate native dimension and checked-product ranges, including empty shapes."""
    if not isinstance(shape, list) or any(type(n) is not int or not 0 <= n <= MAX_DIMENSION for n in shape):
        raise ValueError("safetensors shape must contain nonnegative signed-int64 dimensions")
    count = 1
    for dimension in shape:
        if dimension and count > MAX_COUNT // dimension:
            raise ValueError("safetensors shape multiplication exceeds native usize")
        count *= dimension
    if count > MAX_DIMENSION:
        raise ValueError("safetensors element count exceeds the tensor runtime's signed-int64 range")
    return count


def byte_range(offset: int, count: int, size: int, path) -> None:
    """Validate a nonnegative exact-integer file span before tensor allocation."""
    if type(offset) is not int or type(count) is not int or offset < 0 or count < 0:
        raise ValueError("checkpoint byte offsets and lengths must be nonnegative integers")
    if offset > size or count > size - offset:
        raise IOError(f"short read of {path}: bytes {offset}-{offset + count} past its end ({size})")


def _object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate safetensors JSON key")
        # serde_json's UTF-8 strings cannot contain unpaired surrogate escapes.
        name.encode("utf-8")
        result[name] = value
    return result


def _constant(value):
    raise ValueError("non-JSON numeric constant in safetensors header")


def read_metadata_json(path: str | Path) -> dict:
    """Read a bounded UTF-8 configuration/index object without duplicate keys.

    Checkpoint metadata shares the 100-MB file limit with tensor headers.
    Nonfinite numbers and unpaired surrogate strings cannot describe the
    runtime's configuration or authorized file names.
    """
    with _regular_stream(path) as stream:
        size = os.fstat(stream.fileno()).st_size
        if not 0 < size <= MAX_HEADER_BYTES:
            raise ValueError(f"{path}: checkpoint metadata size must lie in [1, {MAX_HEADER_BYTES}]")
        raw = stream.read(size + 1)
    if len(raw) != size:
        raise ValueError(f"{path}: checkpoint metadata changed during its read")
    return _metadata_object(raw, path)


def _metadata_object(raw: bytes, path) -> dict:
    """One strict UTF-8/object JSON contract for headers, indices and config."""
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object, parse_constant=_constant)
    except RecursionError as error:
        raise ValueError(f"{path}: checkpoint metadata nesting exceeds the JSON runtime limit") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: checkpoint metadata must be an object")
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, str):
            item.encode("utf-8")
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{path}: checkpoint metadata numbers must be finite")
    return value


def checkpoint_path(model_dir: str | Path, shard: str) -> Path:
    """Authorize an index-relative file in the model or its own HF blob store.

    Parent traversal and absolute names are refused before following symlinks.
    A standard ``models--*/snapshots/<revision>`` root may borrow its sibling
    ``blobs`` directory. Other resolved targets must stay inside the model.
    """
    if not isinstance(shard, str) or not shard:
        raise ValueError("checkpoint file names must be nonempty relative strings")
    relative = Path(shard)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{shard}: checkpoint file must not be absolute or traverse parent directories")
    root = Path(model_dir).resolve()
    allowed = [root]
    if root.parent.name == "snapshots" and root.parent.parent.name.startswith("models--"):
        store = root.parent.parent
        blobs = (store / "blobs").resolve()
        if not blobs.is_relative_to(store):
            raise ValueError("the model's HF blob root resolves outside its own store")
        allowed.append(blobs)
    path = (root / relative).resolve()
    if not any(path.is_relative_to(directory) for directory in allowed):
        raise ValueError(f"{shard}: resolved checkpoint file is outside the model's authorized roots")
    return path


def read_header(path: str | Path, sizes: dict[str, int]) -> tuple[int, dict]:
    """Own a regular checkpoint stream and return its validated raw header.

    Standard HF shard symlinks remain admitted. Independently owned streams
    with a stricter no-follow policy use ``read_header_stream`` directly.
    """
    with _regular_stream(path) as stream:
        return read_header_stream(stream, sizes, label=path)


def read_header_stream(stream, sizes: dict[str, int], *, label) -> tuple[int, dict]:
    """Borrow an opened regular binary file at offset zero, without reopening.

    The caller owns descriptor lifetime and input immutability throughout this
    read; this function advances its position and never closes it. Complete
    payload geometry/span validation precedes tensor allocation. Runtime byte
    readers still check descriptor bounds when they subsequently read payloads.
    """
    path = label
    info = os.fstat(stream.fileno())
    if not stat.S_ISREG(info.st_mode) or stream.tell() != 0:
        raise ValueError("safetensors header needs a regular stream at offset zero")
    size = info.st_size
    prefix = stream.read(8)
    if len(prefix) != 8:
        raise ValueError(f"{path}: truncated safetensors header length")
    count = struct.unpack("<Q", prefix)[0]
    if not 0 < count <= min(MAX_HEADER_BYTES, size - 8):
        raise ValueError(f"{path}: invalid safetensors header length")
    raw = stream.read(count)
    if len(raw) != count or not raw.startswith(b"{"):
        raise ValueError(f"{path}: truncated or invalid safetensors JSON header")
    header = _metadata_object(raw, path)
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header must be an object")
    spans = []
    for name, entry in header.items():
        if name == "__metadata__":
            if entry is not None and (
                not isinstance(entry, dict) or any(not isinstance(v, str) for v in entry.values())
            ):
                raise ValueError(f"{path}: safetensors metadata must map strings to strings")
            if entry is not None:
                for value in entry.values():
                    value.encode("utf-8")
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("dtype"), str):
            raise ValueError(f"{name}: invalid safetensors tensor metadata")
        dtype, shape, offsets = entry["dtype"], entry.get("shape"), entry.get("data_offsets")
        item = sizes.get(dtype)
        if item is None:
            raise ValueError(f"{name}: unsupported safetensors dtype {dtype!r}")
        elements = tensor_shape(shape)
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int or n < 0 for n in offsets):
            raise ValueError(f"{name}: invalid safetensors shape or data offsets")
        begin, end = offsets
        if begin > end or elements * item != end - begin:
            raise ValueError(f"{name}: tensor byte range differs from its shape")
        byte_range(8 + count + begin, end - begin, size, path)
        spans.append((begin, end))
    cursor = 0
    for begin, end in sorted(spans):
        if begin != cursor:
            raise ValueError(f"{path}: safetensors data ranges overlap or contain gaps")
        cursor = end
    if cursor != size - 8 - count:
        raise ValueError(f"{path}: safetensors data buffer is not fully indexed")
    return 8 + count, header

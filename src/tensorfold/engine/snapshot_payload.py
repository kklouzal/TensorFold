"""Capture validated cache state and atomically publish an owned snapshot.

The caller owns a trusted current-model registry and keeps tensor storage
immutable until the native writer finishes. Native conversion follows schema
validation. The snapshot directory is caller-owned; concurrent publishers may
replace the same key only with state validated for that key.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Any, Callable


@dataclass(frozen=True)
class SnapshotPayload:
    """Owned containers; arrays borrow immutable storage through native write."""

    arrays: dict[str, Any]
    metadata: dict[str, str]
    numpy_keys: frozenset[str]


_MISSING = object()


def _owned_slot():
    # Acquisition follows publication of a valid closed owner. Constructor
    # return/deallocation never owns an unobserved acquisition or close error.
    from tensorfold._fd_owner import OwnedFD

    return OwnedFD()


def _plain(value: Any) -> bool:
    return type(value) in (type(None), bool, int, float, str)


def _same_plain(left: Any, right: Any) -> bool:
    return (
        _plain(left)
        and type(left) is type(right)
        and (left.hex() == right.hex() if type(left) is float else left == right)
    )


def capture_snapshot(registry, model_id, tokens, cache, *, describe_tensor, tensor_kind) -> SnapshotPayload:
    """Capture every declared field, then validate before foreign conversion.

    ``tensor_kind`` returns 'array', 'numpy', or None. ``describe_tensor``
    returns safetensors dtype code and shape. Both callbacks inspect trusted
    tensor references; they do not mutate the cache or registry. The caller
    materializes family-owned lazy state before entry.
    """
    if type(model_id) is not str or not model_id:
        raise ValueError("a current model identity is required")
    model_id.encode("utf-8")
    if (
        type(tokens) not in (list, tuple)
        or len(tokens) > registry.token_limit
        or any(type(t) is not int or not 0 <= t < registry.token_id_limit for t in tokens)
    ):
        raise ValueError("snapshot tokens exceed current-model authority")
    if type(cache) is not list:
        raise ValueError("snapshot cache must be an owned layer list")
    stored = []
    for item in cache:
        marker = inspect.getattr_static(item, "stored", True)
        if type(marker) is not bool:
            raise ValueError("stored cache flag must be a declared boolean")
        if marker:
            stored.append(item)
            if len(stored) > len(registry.layers):
                raise ValueError("snapshot has too many stored layers")
    if len(stored) != len(registry.layers):
        raise ValueError("snapshot layer count differs from current-model authority")
    # Snapshot all fields/lists before inspecting native tensors. A callback
    # cannot change which source references the payload will validate/write.
    captured = []
    for item, schema in zip(stored, registry.layers):
        if type(item) is not type(schema.prototype):
            raise ValueError("snapshot layer class differs from current-model authority")
        source = dict(vars(item))
        fields = schema.fields()
        transient = inspect.getattr_static(type(item), "transient", ())
        if type(transient) not in (tuple, list, set, frozenset) or any(type(n) is not str for n in transient):
            raise ValueError("cache transient fields must be explicitly declared")
        if set(source) - set(transient) - set(fields):
            raise ValueError("snapshot contains an undeclared cache field")
        values = {}
        for name, expected in fields.items():
            value = source.get(name, _MISSING)
            if value is _MISSING:
                declared = inspect.getattr_static(type(item), name, _MISSING)
                if not _same_plain(declared, expected):
                    raise ValueError("snapshot omitted an initialized cache field")
                value = expected
            if type(value) is list:
                length = schema.list_lengths.get(name, len(expected) if type(expected) is list else None)
                if length is None or len(value) != length:
                    raise ValueError("snapshot list exceeds trusted constructor capacity")
                values[name] = tuple(value)
            else:
                values[name] = value
        captured.append((schema, values, {name for name in fields if type(source.get(name)) is list}))
    arrays, entries, numpy_keys = {}, [], set()
    for index, (schema, values, list_fields) in enumerate(captured):
        entry = {"class": schema.class_id, "plain": {}, "arrays": [], "numpy": [], "lists": {}}
        for name, value in values.items():
            if name in list_fields:
                slots = []
                for slot, element in enumerate(value):
                    if element is None:
                        continue
                    if tensor_kind(element) != "array":
                        raise ValueError("cache lists may contain only native arrays or None")
                    arrays[f"{index}.{name}.{slot}"] = element
                    slots.append(slot)
                entry["lists"][name] = {"length": len(value), "slots": slots}
            elif _plain(value):
                if type(value) is float and not math.isfinite(value):
                    raise ValueError("cache scalar must be finite")
                entry["plain"][name] = value
            else:
                kind = tensor_kind(value)
                if kind not in ("array", "numpy"):
                    raise ValueError("cache field has no declared storage representation")
                key = f"{index}.{name}"
                arrays[key] = value
                entry["arrays" if kind == "array" else "numpy"].append(name)
                if kind == "numpy":
                    numpy_keys.add(key)
        entries.append(entry)
    metadata = {
        "format": "2",
        "model": model_id,
        "tokens": json.dumps(list(tokens), allow_nan=False),
        "layers": json.dumps(entries, allow_nan=False),
        "saved": str(time.time()),
    }
    header = {"__metadata__": dict(metadata)}
    for key, value in arrays.items():
        description = describe_tensor(value)
        if (
            type(description) is not dict
            or set(description) != {"dtype", "shape"}
            or type(description["dtype"]) is not str
            or type(description["shape"]) is not list
        ):
            raise ValueError("tensor descriptor must contain its dtype and shape")
        header[key] = {"dtype": description["dtype"], "shape": list(description["shape"])}
    registry.validate(metadata, header, model_id=model_id)
    return SnapshotPayload(dict(arrays), dict(metadata), frozenset(numpy_keys))


def _finish(primary, errors, notes=(), *, previous=None) -> None:
    """Transport failures only after all owned cleanup attempts completed.

    Native descriptors avoid foreign attribute hooks. Exact primary identity
    survives malformed notes; all distinct cleanup/annotation failures remain
    reachable through its cause, without self-cause or foreign formatting.
    """
    if primary is None:
        if not errors:
            return
        primary, errors = errors[0], errors[1:]
    others = []
    if previous is None:
        previous = (
            BaseException.__cause__.__get__(primary),
            BaseException.__context__.__get__(primary),
            BaseException.__suppress_context__.__get__(primary),
        )
    for error in errors:
        if error is not None and error is not primary and all(error is not item for item in others):
            others.append(error)
    annotation_failed = False
    for text in notes:
        try:
            BaseException.add_note(primary, text)
        except BaseException as annotation:
            annotation_failed = True
            if annotation is not primary and all(annotation is not item for item in others):
                others.append(annotation)
    if not others and not annotation_failed:
        BaseException.__cause__.__set__(primary, previous[0])
        BaseException.__context__.__set__(primary, previous[1])
        BaseException.__suppress_context__.__set__(primary, previous[2])
        raise primary
    for error in reversed(previous[:2]):
        if error is not None and error is not primary and all(error is not item for item in others):
            others.insert(0, error)
    if others:
        raise primary from BaseExceptionGroup("owned snapshot secondary failures", others)
    raise primary


def _close_owned(owner) -> None:
    """Retire an interrupted-before-entry close using its native state.

    A close which consumed the descriptor is never retried. One interrupted
    Python dispatch may be settled by a second native invocation; another
    failure retains the live owner in the operation's exception journal.
    """
    try:
        owner.close()
    except BaseException as primary:
        previous = (
            BaseException.__cause__.__get__(primary),
            BaseException.__context__.__get__(primary),
            BaseException.__suppress_context__.__get__(primary),
        )
        errors = []
        if not owner.closed:
            try:
                owner.close()
            except BaseException as secondary:
                errors.append(secondary)
        _finish(primary, errors, previous=previous)


def _valid_existing(target: Path, validate, reuse_observed=None) -> bool:
    if validate is None:
        return False
    try:
        before = target.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(before.st_mode):
        return False
    valid = validate(target)
    if type(valid) is not bool:
        raise TypeError("existing snapshot validator must return a boolean")
    if not valid:
        return False
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
    owner = _owned_slot()
    primary, cleanup, result, observed, previous = None, [], False, None, None
    try:
        try:
            owner.open(target, flags, 0o600)
        except FileNotFoundError:
            return False
        fd = owner.fileno()
        after = os.fstat(fd)

        def identity(s):
            return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

        if not stat.S_ISREG(after.st_mode) or identity(before) != identity(after):
            result = False
        else:
            os.utime(fd, None)
            observed = identity(before), identity(os.fstat(fd))
            result = True
    except BaseException as error:
        primary = error
        previous = (
            BaseException.__cause__.__get__(primary),
            BaseException.__context__.__get__(primary),
            BaseException.__suppress_context__.__get__(primary),
        )
    finally:
        try:
            _close_owned(owner)
        except BaseException as error:
            cleanup.append(error)
        notes = ("existing snapshot descriptor close also failed",) if primary is not None and cleanup else ()
        if not owner.closed:
            recipient = primary if primary is not None else cleanup[0]
            dictionary = BaseException.__dict__["__dict__"].__get__(recipient, type(recipient))
            dict.__setitem__(dictionary, "_tensorfold_snapshot_fd_owners", (owner,))
        _finish(primary, cleanup, notes, previous=previous)
    if result and reuse_observed is not None:
        reuse_observed(target, *observed)
    return result


def publish_snapshot(
    directory: Path, key: str, write: Callable[[Path], None], *, valid_existing=None, reuse_observed=None
) -> Path | None:
    """Publish only after writer completion, file fsync, and directory fsync.

    An existing file suppresses a write only after current-schema validation.
    Each write owns a private directory/file on the same filesystem. Concurrent
    valid publication is checked again before replacement. Commit followed by
    durability failure raises with an explicit committed-target note.

    ``reuse_observed`` receives (target, before, after) only after successful
    validation, same-FD mtime touch/fstat and close. Each identity is
    (device, inode, bytes, mtime_ns, ctime_ns). The observer may refresh a bounded
    immutable-file receipt; it still compares the next current identity.
    """
    if type(key) is not str or re.fullmatch("[0-9a-f]{32}", key) is None:
        raise ValueError("snapshot key must be 32 lowercase hexadecimal digits")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{key}.safetensors"
    if _valid_existing(target, valid_existing, reuse_observed):
        return None
    temporary = tempfile.TemporaryDirectory(prefix=".tensorfold-snapshot-", dir=directory)
    partial = Path(temporary.name) / "payload.safetensors"
    primary, previous, committed = None, None, False
    reservation_owner = file_owner = directory_owner = None
    try:
        # Reserve a private mode-0600 path before the native writer reopens it.
        reservation_owner = _owned_slot()
        reservation_owner.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        _close_owned(reservation_owner)
        write(partial)
        file_owner = _owned_slot()
        file_owner.open(partial, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, 0o600)
        fd = file_owner.fileno()
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size < 8:
            raise ValueError("native snapshot writer did not complete a regular payload")
        os.fchmod(fd, 0o600)
        os.fsync(fd)
        _close_owned(file_owner)
        if _valid_existing(target, valid_existing, reuse_observed):
            return None
        directory_owner = _owned_slot()
        directory_owner.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, 0o600)
        os.replace(partial, target)
        committed = True
        os.fsync(directory_owner.fileno())
        return target
    except BaseException as error:
        primary = error
        previous = (
            BaseException.__cause__.__get__(primary),
            BaseException.__context__.__get__(primary),
            BaseException.__suppress_context__.__get__(primary),
        )
    finally:
        cleanup = []
        for owned in (reservation_owner, file_owner, directory_owner):
            if owned is not None and not owned.closed:
                try:
                    _close_owned(owned)
                except BaseException as error:
                    cleanup.append(error)
        try:
            temporary.cleanup()
        except BaseException as error:
            cleanup.append(error)
        notes = []
        if committed and (primary is not None or cleanup):
            notes.append(f"snapshot target committed before failure: {target}")
        if primary is not None and cleanup:
            notes.append("owned snapshot cleanup also failed; inspect the retained private path")
        if len(cleanup) > 1:
            notes.append("another owned snapshot cleanup operation also failed")
        live = tuple(
            owner
            for owner in (reservation_owner, file_owner, directory_owner)
            if owner is not None and not owner.closed
        )
        if live:
            recipient = primary if primary is not None else cleanup[0]
            dictionary = BaseException.__dict__["__dict__"].__get__(recipient, type(recipient))
            dict.__setitem__(dictionary, "_tensorfold_snapshot_fd_owners", live)
            notes.append("snapshot descriptor owners retained after interrupted cleanup")
        _finish(primary, cleanup, notes, previous=previous)

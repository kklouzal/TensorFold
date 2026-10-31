"""Startup-owned exact-byte model identity; no persisted hash authority.

The caller selects the complete execution input closure, authorizes resolved
paths, and keeps those inputs immutable through model loading. Filesystem
identity checks detect changes; they do not replace that ownership contract.
Only a receipt created in this process may reuse hashes, with an explicit
caller guarantee of immutability since its capture. Disk JSON is diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any
import weakref

from tensorfold.file_io import FileStreams


_ISSUED = object()
_READ_BYTES = 1 << 20


def canonical_runtime_json(value: Any, *, max_bytes=1 << 20, max_depth=64, max_nodes=65536) -> str:
    """Bound and validate exact JSON types before canonical serialization."""
    if (type(value) is not dict
            or any(type(n) is not int or n <= 0 for n in (max_bytes, max_depth, max_nodes))):
        raise ValueError("bounded JSON object runtime identity required")
    pending = [(value, 0, False)]
    active = set()
    nodes = size = 0
    scheduled = 1
    while pending:
        item, depth, leaving = pending.pop()
        if leaving:
            active.remove(id(item))
            continue
        nodes += 1
        if nodes > max_nodes or depth > max_depth:
            raise ValueError("runtime identity exceeds node/depth bounds")
        kind = type(item)
        if kind in (dict, list):
            if id(item) in active:
                raise ValueError("runtime identity contains a cycle")
            children = len(item) * (2 if kind is dict else 1)
            if children > max_nodes - scheduled:
                raise ValueError("runtime identity exceeds node bound")
            scheduled += children
            size += 2 + max(0, len(item) - 1) + (len(item) if kind is dict else 0)
            active.add(id(item))
            pending.append((item, depth, True))
            if kind is dict:
                for key, child in item.items():
                    if type(key) is not str:
                        raise ValueError("runtime identity object keys must be strings")
                    pending.append((key, depth + 1, False))
                    pending.append((child, depth + 1, False))
            else:
                pending.extend((child, depth + 1, False) for child in item)
        elif kind is str:
            if len(item) > max_bytes:
                raise ValueError("runtime identity string exceeds byte bound")
            size += 2
            for char in item:
                code = ord(char)
                if 0xD800 <= code <= 0xDFFF:
                    raise ValueError("runtime identity contains an invalid Unicode scalar")
                if char in ('"', '\\', '\b', '\f', '\n', '\r', '\t'):
                    size += 2
                elif code < 0x20:
                    size += 6
                else:
                    size += 1 if code < 0x80 else 2 if code < 0x800 else 3 if code < 0x10000 else 4
                if size > max_bytes:
                    raise ValueError("runtime identity exceeds byte bound")
        elif kind is int:
            if item.bit_length() > max_bytes * 4:
                raise ValueError("runtime identity integer exceeds byte bound")
            size += len(str(item))
        elif kind is float:
            if not math.isfinite(item):
                raise ValueError("runtime identity numbers must be finite")
            size += len(json.dumps(item, allow_nan=False))
        elif kind in (bool, type(None)):
            size += 4 if item is True or item is None else 5
        else:
            raise ValueError("runtime identity contains a non-JSON value")
        if size > max_bytes:
            raise ValueError("runtime identity exceeds byte bound")
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    if len(encoded.encode('utf-8')) != size:
        raise RuntimeError("runtime identity byte-accounting invariant failed")
    return encoded


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _close(scope, primary):
    """Drain the bounded owner journal; retain every failure and live owner.

    Native close consumes its owner even when reporting an OS error. The core
    only retries the exact unconsumed owner after a pre-entry interruption.
    Unfinished owners remain attached to the primary exception for containment
    and explicit owner-scope recovery; their descriptor numbers never retry.
    """
    # Capture authoritative roots before cleanup can invoke native/profile
    # callbacks. Named references also survive cold transport allocation failure.
    prior_cause = None if primary is None else BaseException.__cause__.__get__(primary, type(primary))
    prior_context = None if primary is None else BaseException.__context__.__get__(primary, type(primary))
    prior_suppress = None if primary is None else BaseException.__suppress_context__.__get__(primary, type(primary))
    try:
        errors = scope.drain()
    except BaseException as cleanup:
        errors = [cleanup]
    cleanup_errors = errors
    if primary is None:
        if not errors:
            return
        primary, errors = errors[0], errors[1:]
        prior_cause = BaseException.__cause__.__get__(primary, type(primary))
        prior_context = BaseException.__context__.__get__(primary, type(primary))
    try:
        namespace = BaseException.__dict__['__dict__'].__get__(primary, type(primary))
        live = scope.live_owners
        if live:
            dict.__setitem__(namespace, '_tensorfold_model_fd_owners', live)
            dict.__setitem__(namespace, '_tensorfold_model_fd_scope', scope)
        elif dict.get(namespace, '_tensorfold_model_fd_scope') is scope:
            dict.pop(namespace, '_tensorfold_model_fd_scope', None)
            dict.pop(namespace, '_tensorfold_model_fd_owners', None)
        if not cleanup_errors:
            # Successful cleanup leaves the original exception transport
            # intact even when profile/foreign callbacks mutate native fields.
            BaseException.__cause__.__set__(primary, prior_cause)
            BaseException.__context__.__set__(primary, prior_context)
            BaseException.__suppress_context__.__set__(primary, prior_suppress)
            return
        previous = BaseException.__cause__.__get__(primary, type(primary))
        current_context = BaseException.__context__.__get__(primary, type(primary))
        other = []
        for error in (prior_cause, prior_context, previous, current_context, *errors):
            if error is not None and error is not primary and all(error is not item for item in other):
                other.append(error)
        try:
            names = ', '.join(type.__dict__['__name__'].__get__(type(error), type)[:80] for error in cleanup_errors)
            BaseException.add_note(primary, 'model input owner cleanup also failed (' + names + ')')
        except BaseException as annotation:
            if annotation is not primary and all(annotation is not item for item in other):
                other.append(annotation)
        if other:
            if len(other) == 1:
                cause = other[0]
            else:
                # Keep the already-formed statuses in the native dictionary if
                # grouping cannot allocate; recovery may inspect them directly.
                dict.__setitem__(namespace, '_tensorfold_model_fd_failures', other)
                cause = BaseExceptionGroup('model input secondary failures', other)
                dict.pop(namespace, '_tensorfold_model_fd_failures', None)
            raise primary from cause
        if previous is primary:
            raise primary from None
        raise primary
    except BaseException as transport:
        if transport is primary:
            raise
        # scope/prior roots/all formed statuses remain owned by this traceback
        # even when native retention or annotation preparation cannot allocate.
        raise primary from transport


def _open_regular(path: Path, limit: int, scope):
    record, info = scope.open_descriptor(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, max_bytes=limit)
    return record.owner.fileno(), info


@dataclass(frozen=True)
class FileContent:
    logical_name: str
    path: str
    sha256: str
    size: int
    stat_identity: tuple[int, ...]


@dataclass(frozen=True)
class ModelIdentity:
    """Immutable content authority issued by capture, never decoded from disk."""

    model_prefix: str
    runtime_json: str
    files: tuple[FileContent, ...]
    total_bytes: int
    bytes_read: int
    hashes_reused: int
    elapsed_seconds: float
    _issued: object = field(repr=False, compare=False)
    _pid: int = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self._issued is not _ISSUED:
            raise ValueError("model identity must be issued by capture")
        object.__setattr__(self, '_pid', os.getpid())
        # A copied/replaced dataclass cannot inherit the original receipt's
        # authority. No process-global mutable receipt registry is needed.
        object.__setattr__(self, '_issued', weakref.ref(self))

    def _check_issued(self):
        if (self._pid != os.getpid() or type(self._issued) is not weakref.ReferenceType
                or self._issued() is not self):
            raise ValueError("model identity receipt was not issued in this process")

    def verify_unchanged(self):
        """Check after model loading, before persisted state is reused."""
        self._check_issued()
        for content in self.files:
            scope = FileStreams(max_files=1)
            primary = None
            try:
                fd, info = _open_regular(Path(content.path), content.size, scope)
                if _identity(info) != content.stat_identity:
                    raise ValueError("model execution input changed after identity capture")
            except BaseException as error:
                primary = error
                raise
            finally:
                _close(scope, primary)


def _hash_descriptor(fd, size):
    hasher, read = hashlib.sha256(), 0
    while read < size:
        block = os.read(fd, min(_READ_BYTES, size - read))
        if not block:
            raise ValueError("model execution input was truncated during hashing")
        hasher.update(block)
        read += len(block)
    if os.read(fd, 1):
        raise ValueError("model execution input grew during hashing")
    return hasher.hexdigest(), read


def _issue_identity(runtime, contents, *, bytes_read, hashes_reused, started):
    envelope = {'format': 'model-content-v1', 'runtime': runtime,
                'files': [{'name': c.logical_name, 'size': c.size, 'sha256': c.sha256} for c in contents]}
    digest = hashlib.sha256(json.dumps(envelope, sort_keys=True, ensure_ascii=False,
                                       separators=(',', ':')).encode('utf-8')).hexdigest()
    return ModelIdentity('model-content-v1:' + digest, runtime, tuple(contents), sum(c.size for c in contents),
                         bytes_read, hashes_reused, time.perf_counter() - started, _ISSUED)


def capture_model_identity(files: dict[str, Path], *, runtime_identity, authorize,
                           max_file_bytes: int, max_total_bytes: int,
                           reuse: ModelIdentity | None = None, reuse_is_immutable: bool = False,
                           max_files: int = 65536, max_name_bytes: int = 4096) -> ModelIdentity:
    """Hash a complete caller-selected, authorized execution input closure.

    Byte budgets are explicit owner policy, not hardware/model-size defaults.
    Names and content affect identity; resolved host paths only bind lifetime
    checks. Reuse requires identical runtime/input/stat identities and a caller
    guarantee that inputs have stayed immutable since the prior capture.
    """
    started = time.perf_counter()
    if (type(files) is not dict or not files
            or any(type(n) is not int or n < 0 for n in (max_file_bytes, max_total_bytes))
            or any(type(n) is not int or n <= 0 for n in (max_files, max_name_bytes))
            or len(files) > max_files
            or type(reuse_is_immutable) is not bool):
        raise ValueError("explicit model input closure and byte budgets required")
    for name, path in files.items():
        if (type(name) is not str or not name or len(name) > max_name_bytes
                or '\0' in name or not isinstance(path, Path)):
            raise ValueError("model input requires a logical name and filesystem path")
        if len(name.encode('utf-8')) > max_name_bytes:
            raise ValueError("model input logical name exceeds byte bound")
    if reuse is not None:
        if type(reuse) is not ModelIdentity or not reuse_is_immutable:
            raise ValueError("hash reuse needs current-process immutable input authority")
        reuse._check_issued()
    runtime = canonical_runtime_json(runtime_identity)
    previous = {}
    if reuse is not None:
        if reuse.runtime_json == runtime and {f.logical_name for f in reuse.files} == set(files):
            previous = {f.logical_name: f for f in reuse.files}
    contents = []
    total = read_bytes = reused = 0
    for name in sorted(files):
        path = authorize(name, files[name])
        if not isinstance(path, Path) or not path.is_absolute():
            raise ValueError("input authorization must return an absolute resolved path")
        scope = FileStreams(max_files=1)
        primary = None
        try:
            fd, info = _open_regular(path, max_file_bytes, scope)
            total += info.st_size
            if total > max_total_bytes:
                raise ValueError("model execution inputs exceed total byte budget")
            before = _identity(info)
            old = previous.get(name)
            if old is not None and old.path == str(path) and old.stat_identity == before:
                digest = old.sha256
                reused += 1
            else:
                digest, read = _hash_descriptor(fd, info.st_size)
                read_bytes += read
            if _identity(os.fstat(fd)) != before or _identity(path.stat(follow_symlinks=False)) != before:
                raise ValueError("model execution input changed during identity capture")
            contents.append(FileContent(name, str(path), digest, info.st_size, before))
        except BaseException as error:
            primary = error
            raise
        finally:
            _close(scope, primary)
    receipt = _issue_identity(runtime, contents, bytes_read=read_bytes, hashes_reused=reused, started=started)
    receipt.verify_unchanged()
    return receipt


def capture_shared_model_identity(files: dict[str, Path], *, runtime_identity, authorize,
                                  max_file_bytes: int, max_total_bytes: int, rank: int, world: int,
                                  exchange, reuse: ModelIdentity | None = None,
                                  reuse_is_immutable: bool = False, max_files: int = 65536,
                                  max_name_bytes: int = 4096) -> ModelIdentity:
    """Issue local authority after an owned same-host cohort content pass.

    ``exchange(stage, operation)`` is the startup owner's matched, fatal-aware
    phase exchange; it executes this rank's operation and returns rank-ordered
    results. It is trusted transport, never persisted manifest/peer JSON used
    as standalone authority. Every rank authorizes and opens the entire closure
    and keeps its descriptor owners through the leader's hash and verification.
    The service owns immutability from these opens through loading/serving.
    Followers read zero payload bytes and explicitly report shared hash reuse.
    """
    started = time.perf_counter()
    if (type(files) is not dict or not files or type(rank) is not int or type(world) is not int
            or not 1 <= world <= 8 or not 0 <= rank < world or not callable(exchange)
            or any(type(n) is not int or n < 0 for n in (max_file_bytes, max_total_bytes))
            or any(type(n) is not int or n <= 0 for n in (max_files, max_name_bytes))
            or len(files) > max_files or type(reuse_is_immutable) is not bool):
        raise ValueError("bounded same-host model content cohort required")
    runtime = None
    previous = {}
    scope, opened = None, {}
    primary = None
    try:
        def inventory():
            nonlocal previous, runtime, scope
            if reuse is not None:
                if type(reuse) is not ModelIdentity or not reuse_is_immutable:
                    raise ValueError("shared hash reuse needs current-process immutable authority")
                reuse._check_issued()
            # A foreign-process receipt refuses within the matched phase,
            # before runtime serialization or opening any execution input.
            runtime = canonical_runtime_json(runtime_identity)
            scope = FileStreams(max_files=max_files)
            if reuse is not None:
                if reuse.runtime_json == runtime and {f.logical_name for f in reuse.files} == set(files):
                    previous = {f.logical_name: f for f in reuse.files}
            total = 0
            for name in sorted(files):
                path = files[name]
                if (type(name) is not str or not name or '\0' in name or len(name) > max_name_bytes
                        or len(name.encode('utf-8')) > max_name_bytes or not isinstance(path, Path)):
                    raise ValueError("model input requires a bounded logical name and filesystem path")
                path = authorize(name, path)
                if not isinstance(path, Path) or not path.is_absolute():
                    raise ValueError("input authorization must return an absolute resolved path")
                fd, info = _open_regular(path, max_file_bytes, scope)
                total += info.st_size
                if total > max_total_bytes:
                    raise ValueError("model execution inputs exceed total byte budget")
                opened[name] = (path, fd, _identity(info))
            # boot_id names the actual running kernel, unlike a hostname alone.
            # Read a bounded regular local file through the same native owner.
            boot_scope = FileStreams(max_files=1)
            boot_error = None
            try:
                fd, _info = _open_regular(Path('/proc/sys/kernel/random/boot_id'), 4096, boot_scope)
                boot = os.read(fd, 4097).decode('ascii').strip()
                if not boot or len(boot) > 128:
                    raise ValueError("actual same-host boot identity is required")
            except BaseException as error:
                boot_error = error
                raise
            finally:
                _close(boot_scope, boot_error)
            return {"rank": rank, "world": world, "host": [os.uname().nodename, boot], "runtime": runtime,
                    "files": {name: {"path": str(path), "stat": list(identity)}
                              for name, (path, _fd, identity) in opened.items()}}

        ranks = exchange("inventory", inventory)

        def agree():
            if (not isinstance(ranks, list) or len(ranks) != world
                    or any(type(row) is not dict or set(row) != {"rank", "world", "host", "runtime", "files"}
                           or row["rank"] != index or row["world"] != world
                           for index, row in enumerate(ranks))):
                raise ValueError("shared content inventory lost rank ownership")
            reference = {key: value for key, value in ranks[rank].items() if key != "rank"}
            if any({key: value for key, value in row.items() if key != "rank"} != reference for row in ranks):
                raise ValueError("shared content requires identical host, runtime and opened file identities")

        exchange("agreement", agree)

        def content():
            if rank:
                return None
            contents, read_bytes, reused = {}, 0, 0
            for name, (path, fd, before) in opened.items():
                old = previous.get(name)
                if old is not None and old.path == str(path) and old.stat_identity == before:
                    sha = old.sha256
                    reused += 1
                else:
                    sha, read = _hash_descriptor(fd, before[2])
                    read_bytes += read
                contents[name] = {"size": before[2], "sha256": sha}
            return {"files": contents, "bytes_read": read_bytes, "hashes_reused": reused}

        outcome = exchange("content", content)[0]

        def verification():
            if (type(outcome) is not dict or set(outcome) != {"files", "bytes_read", "hashes_reused"}
                    or type(outcome["files"]) is not dict or set(outcome["files"]) != set(opened)
                    or type(outcome["bytes_read"]) is not int or not 0 <= outcome["bytes_read"] <= sum(
                        before[2] for _path, _fd, before in opened.values())
                    or type(outcome["hashes_reused"]) is not int or not 0 <= outcome["hashes_reused"] <= len(opened)):
                raise ValueError("shared content pass did not cover the actual input closure")
            for name, (path, fd, before) in opened.items():
                item = outcome["files"][name]
                if (type(item) is not dict or set(item) != {"size", "sha256"}
                        or type(item["size"]) is not int or item["size"] != before[2]
                        or type(item["sha256"]) is not str or len(item["sha256"]) != 64
                        or any(char not in '0123456789abcdef' for char in item["sha256"])):
                    raise ValueError("shared content pass returned invalid file content")
                if _identity(os.fstat(fd)) != before or _identity(path.stat(follow_symlinks=False)) != before:
                    raise ValueError("shared input changed across the leader content pass")

        exchange("verification", verification)
        contents = tuple(FileContent(name, str(path), outcome["files"][name]["sha256"], before[2], before)
                         for name, (path, _fd, before) in opened.items())
        receipt = _issue_identity(runtime, contents, bytes_read=outcome["bytes_read"] if rank == 0 else 0,
                                  hashes_reused=outcome["hashes_reused"] if rank == 0 else len(contents), started=started)
        # Keep the final check in an agreed phase: a local failure must not let
        # a peer begin CUDA acquisition while this rank is still refusing.
        exchange("issuance", receipt.verify_unchanged)
        return receipt
    except BaseException as error:
        primary = error
        raise
    finally:
        if scope is not None:
            _close(scope, primary)

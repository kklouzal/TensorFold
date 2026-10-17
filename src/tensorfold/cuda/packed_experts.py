"""Owned-copy persistence for the affine grouped-expert bit permutation.

A private service-owned directory is the trust boundary: hashes detect damaged
files, not an adversary who can rewrite both bytes and their digest. Checkpoint
inputs remain immutable throughout loading, as for the ordinary reader. Exact
source bytes are hashed again at every start; persisted stat/hash receipts are
never treated as authority. Cache failures due to access, resources or I/O are
startup failures. Missing/stale/damaged artifacts explicitly permit repacking.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import sys
import uuid

import torch

from tensorfold.cleanup import raise_failures

# Version 1 is native experts_pack.cu's group-32, 160-word block: shuffled
# nibble words followed by unconverted BF16 scale/bias pairs. Arrays are LE I32;
# gate then up are interleaved in up's penultimate axis; down has a singleton.
# Bump LAYOUT for any packed-byte semantic change; source refactors preserving
# bytes keep this identity. Native packing and independent scalar oracles apply.
LAYOUT = "grouped-affine4-g32-le-i32-block160-gu-interleaved-v1"
_MAGIC = b"TFEXPERTCACHE\x00\x01\x00"
_LIMIT = 4096  # fixed two-array schema (five <=64-bit dimensions each) fits below 4 KiB
_CHUNK = 8 << 20
_NAME = re.compile(r"[0-9a-f]{64}\.tfc\Z")
_TEMP = re.compile(r"\.tmp-[0-9a-f]{32}\Z")
_SIZES = {"U32": 4, "I32": 4, "BF16": 2}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _stamp(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _trusted(value, *, directory=False):
    if (not (stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode))
            or value.st_uid != os.geteuid() or value.st_mode & 0o022):
        raise PermissionError("packed expert cache must be owned by the service UID and not writable by others")


class SourceIdentity:
    """Hold source descriptors and verify identity through cache fill/packing.

    ``ranges`` are validated reader metadata plus exact selected first-axis row
    intervals. Only these byte ranges are consumed, including shared/MTP banks
    only when requested. The caller continues to own checkpoint immutability.
    """
    def __init__(self, ranges, geometry):
        self.ranges = ranges
        self.geometry = geometry
        self.files = {}
        self.key = None

    def __enter__(self):
        try:
            digest = hashlib.sha256(_json({"layout": LAYOUT, "geometry": self.geometry}))
            for name, info, lo, hi in self.ranges:
                path = Path(info["path"])
                if path not in self.files:
                    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
                    self.files[path] = (fd, None)
                    self.files[path] = (fd, os.fstat(fd))
                fd, original = self.files[path]
                row = info["shape"][1:]
                row_bytes = _SIZES[info["dtype"]]
                for n in row:
                    row_bytes *= n
                begin, size = info["byte_offset"] + lo * row_bytes, (hi - lo) * row_bytes
                if (not stat.S_ISREG(original.st_mode) or begin < 0 or size <= 0
                        or begin + size > original.st_size):
                    raise ValueError(f"{name}: source range exceeds the checkpoint file")
                digest.update(_json({"name": name, "dtype": info["dtype"], "shape": info["shape"],
                                     "rows": [lo, hi], "size": size}))
                offset = 0
                while offset < size:
                    raw = os.pread(fd, min(_CHUNK, size - offset), begin + offset)
                    if not raw:
                        raise EOFError(f"{name}: truncated source while computing packed expert identity")
                    digest.update(raw)
                    offset += len(raw)
            self.verify()
            self.key = digest.hexdigest()
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def verify(self):
        for path, (fd, before) in self.files.items():
            if _stamp(os.fstat(fd)) != _stamp(before) or _stamp(path.stat()) != _stamp(before):
                raise RuntimeError(f"checkpoint changed during packed expert loading: {path}")

    def __exit__(self, kind, error, traceback):
        failures = []
        for fd, _ in self.files.values():
            try:
                os.close(fd)
            except OSError as failure:
                failures.append(failure)
        self.files.clear()
        if failures:
            raise_failures(error, failures)


class DamagedCache(ValueError):
    """An artifact was read successfully but failed its format/integrity contract."""


def _read_exact(fd, size):
    result = bytearray()
    while len(result) < size:
        raw = os.read(fd, size - len(result))
        if not raw:
            raise DamagedCache("truncated artifact")
        result.extend(raw)
    return result


def _views(tensors):
    for tensor in tensors:
        if (tensor.device.type != "cpu" or tensor.dtype != torch.int32 or not tensor.is_contiguous()):
            raise ValueError("packed cache requires owned contiguous CPU int32 authority")
        view = memoryview(tensor.numpy()).cast("B")
        for offset in range(0, len(view), _CHUNK):
            yield view[offset:offset + _CHUNK]


def _write_all(fd, view):
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError("packed expert cache write made no progress")
        view = view[count:]


class PackedExpertCache:
    """Bounded per-layer artifacts; serialized access and atomic publication.

    The root must be private to the service. The bound counts complete artifacts
    plus the publisher's temporary artifact, not filesystem metadata/block
    overhead. At most 4096 artifacts exist; one publication may temporarily
    replace its own old entry. No disk JSON is accepted as source hash authority.
    """
    def __init__(self, root, max_bytes):
        if os.name != "posix":
            raise ValueError("packed expert cache requires POSIX file locking and directory publication; disable it on this host")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("packed expert cache byte limit must be a positive integer")
        if sys.byteorder != "little":
            raise ValueError("packed expert cache layout requires a little-endian host")
        self.root, self.max_bytes = Path(root), max_bytes
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)

    @classmethod
    def from_environment(cls):
        root = os.environ.get("TENSORFOLD_PACKED_EXPERT_CACHE", "").strip()
        limit = os.environ.get("TENSORFOLD_PACKED_EXPERT_CACHE_BYTES", "").strip()
        if not root:
            if "TENSORFOLD_PACKED_EXPERT_CACHE_BYTES" in os.environ:
                raise ValueError("TENSORFOLD_PACKED_EXPERT_CACHE_BYTES requires TENSORFOLD_PACKED_EXPERT_CACHE")
            return None
        if not limit.isascii() or not limit.isdecimal() or int(limit) <= 0:
            raise ValueError("TENSORFOLD_PACKED_EXPERT_CACHE_BYTES must be set to a positive byte count when caching is enabled")
        return cls(root, int(limit))

    @contextmanager
    def _locked(self):
        import fcntl

        root = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
        lock = None
        primary = None
        try:
            _trusted(os.fstat(root), directory=True)
            lock = os.open(".lock", os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600, dir_fd=root)
            _trusted(os.fstat(lock))
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield root
        except BaseException as failure:
            primary = failure
            raise
        finally:
            failures = []
            for fd in (lock, root):
                if fd is not None:
                    try:
                        os.close(fd)  # closing the owned lock descriptor releases its flock
                    except OSError as failure:
                        failures.append(failure)
            if failures:
                raise_failures(primary, failures)

    def _prune(self, root, needed=0, slots=0):
        # All scans have a bounded retained inventory. Only our versioned files
        # count as cache ownership; unrelated files are never deleted. The same
        # root must use one service-owned budget across concurrent processes.
        entries = []
        with os.scandir(root) as directory:
            for entry in directory:
                if _TEMP.fullmatch(entry.name):
                    _trusted(entry.stat(follow_symlinks=False))
                    os.unlink(entry.name, dir_fd=root)
                elif _NAME.fullmatch(entry.name):
                    if len(entries) == 4096:
                        raise OSError("packed expert cache exceeds the 4096-artifact inventory limit")
                    value = entry.stat(follow_symlinks=False)
                    _trusted(value)
                    entries.append((value.st_mtime_ns, entry.name, value.st_size))
        total = sum(size for _, _, size in entries)
        remaining = len(entries)
        for _, victim, size in sorted(entries):
            if total + needed <= self.max_bytes and remaining + slots <= 4096:
                break
            os.unlink(victim, dir_fd=root)
            total -= size
            remaining -= 1
        if total + needed > self.max_bytes:
            raise OSError("packed expert cache cannot satisfy its configured byte budget")

    @staticmethod
    def _header(identity, tensors):
        return {"layout": LAYOUT, "source": identity.key,
                "arrays": [{"shape": list(t.shape), "dtype": "I32", "bytes": t.numel() * 4} for t in tensors]}

    def read(self, identity, tensors):
        """Return an explicit hit/miss/damage status; resources/access propagate."""
        expected = self._header(identity, tensors)
        name = identity.key + ".tfc"
        with self._locked() as root:
            self._prune(root)
            try:
                fd = os.open(name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=root)
            except FileNotFoundError:
                return "miss"
            primary = None
            try:
                before = os.fstat(fd)
                _trusted(before)
                try:
                    if _read_exact(fd, len(_MAGIC)) != _MAGIC:
                        raise DamagedCache("wrong magic/version")
                    size = struct.unpack("<Q", _read_exact(fd, 8))[0]
                    if size > _LIMIT:
                        raise DamagedCache("header exceeds bound")
                    try:
                        encoded = _read_exact(fd, size)
                        header = json.loads(encoded.decode("ascii"))
                    except (ValueError, UnicodeError, RecursionError) as error:
                        raise DamagedCache("invalid header") from error
                    if type(header) is not dict or _json(header) != encoded:
                        raise DamagedCache("header must be a canonical object without duplicate keys")
                    checksum = header.pop("sha256", None)
                    if (_json(header) != _json(expected) or type(checksum) is not str
                            or re.fullmatch("[0-9a-f]{64}", checksum) is None):
                        raise DamagedCache("stale layout/source/geometry or missing checksum")
                    if before.st_size != len(_MAGIC) + 8 + size + sum(t.numel() * 4 for t in tensors):
                        raise DamagedCache("payload size differs from model geometry")
                    digest = hashlib.sha256()
                    for view in _views(tensors):
                        pending = view
                        while pending:
                            count = os.readv(fd, [pending])
                            if count <= 0:
                                raise DamagedCache("truncated payload")
                            pending = pending[count:]
                        digest.update(view)
                    if digest.hexdigest() != checksum or _stamp(os.fstat(fd)) != _stamp(before):
                        raise DamagedCache("payload checksum or inode changed")
                    identity.verify()
                    return "hit"
                except DamagedCache as error:
                    return "rebuild: " + str(error)
            except BaseException as failure:
                primary = failure
                raise
            finally:
                try:
                    os.close(fd)
                except OSError as failure:
                    raise_failures(primary, [failure])

    def write(self, identity, tensors):
        """Publish completed bytes durably, evicting old owned entries first."""
        identity.verify()
        header = self._header(identity, tensors)
        digest = hashlib.sha256()
        for view in _views(tensors):
            digest.update(view)
        header["sha256"] = digest.hexdigest()
        raw = _json(header)
        if len(raw) > _LIMIT:
            raise ValueError("packed expert cache header exceeds bound")
        needed = len(_MAGIC) + 8 + len(raw) + sum(t.numel() * 4 for t in tensors)
        if needed > self.max_bytes:
            return "not stored: layer exceeds configured cache byte limit"
        name, temporary = identity.key + ".tfc", ".tmp-" + uuid.uuid4().hex
        with self._locked() as root:
            self._prune(root, needed, 1)
            fd = None
            try:
                fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                             0o600, dir_fd=root)
                _write_all(fd, memoryview(_MAGIC + struct.pack("<Q", len(raw)) + raw))
                for view in _views(tensors):
                    _write_all(fd, view)
                identity.verify()
                os.fsync(fd)
                finished, fd = fd, None
                os.close(finished)  # never retry a consumed numeric descriptor after close failure
                os.replace(temporary, name, src_dir_fd=root, dst_dir_fd=root)
                os.fsync(root)
            except BaseException as primary:
                failures = []
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError as cleanup:
                        failures.append(cleanup)
                try:
                    os.unlink(temporary, dir_fd=root)
                except FileNotFoundError:
                    pass
                except OSError as cleanup:
                    failures.append(cleanup)
                if failures:
                    raise_failures(primary, failures)
                raise
        return "stored"

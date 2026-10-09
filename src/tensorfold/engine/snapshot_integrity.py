"""Bounded FORMAT2 byte-corruption checks, never attacker authentication.

The literal ASCII metadata key/value reserves one 64-byte digest. SHA256 covers
all file bytes with only that value replaced by ASCII zeroes. A parsed, fully
validated safetensors header binds its unique metadata location; arbitrary text
searches never choose the mask. Native reads require verification while copying.
Sealing modifies only an operation-owned, uncommitted writer output.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path

from tensorfold.cuda.tensor_file import read_header_stream
from tensorfold.engine.snapshot_stream import SnapshotStreams

FIELD = "tensorfold_sha256"
ZERO = "0" * 64
_ZERO_BYTES = ZERO.encode("ascii")
_SPACE = b" \t\r\n"


def expected_digest(metadata):
    digest = metadata.get(FIELD) if type(metadata) is dict else None
    if type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("snapshot requires its FORMAT2 content digest; rebuild by prefill")
    return digest


def _space(raw, position):
    while position < len(raw) and raw[position] in _SPACE:
        position += 1
    return position


def _string_end(raw, position):
    if position >= len(raw) or raw[position] != 34:
        raise ValueError("snapshot digest header requires a JSON string")
    position += 1
    while True:
        end = raw.find(b'"', position)
        if end < 0:
            raise ValueError("snapshot digest header string is incomplete")
        escaped, back = 0, end - 1
        while back >= position and raw[back] == 92:
            escaped += 1
            back -= 1
        if escaped % 2 == 0:
            return end + 1
        position = end + 1


def _value_end(raw, position):
    if raw[position] == 34:
        return _string_end(raw, position)
    if raw[position] not in (123, 91):
        end = position
        while end < len(raw) and raw[end] not in b",}] \t\r\n":
            end += 1
        return end
    depth, end = 0, position
    while end < len(raw):
        byte = raw[end]
        if byte == 34:
            end = _string_end(raw, end)
            continue
        if byte in (123, 91):
            depth += 1
        elif byte in (125, 93):
            depth -= 1
            if depth == 0:
                return end + 1
        end += 1
    raise ValueError("snapshot digest header container is incomplete")


def _members(raw, position):
    if raw[position] != 123:
        raise ValueError("snapshot digest requires an object metadata envelope")
    position = _space(raw, position + 1)
    while position < len(raw) and raw[position] != 125:
        key_start, key_end = position, _string_end(raw, position)
        key = json.loads(raw[key_start:key_end])
        position = _space(raw, key_end)
        if raw[position] != 58:
            raise ValueError("snapshot digest header key has no value")
        start = _space(raw, position + 1)
        end = _value_end(raw, start)
        yield key, key_start, key_end, start, end
        position = _space(raw, end)
        if raw[position] == 125:
            return
        if raw[position] != 44:
            raise ValueError("snapshot digest header member is incomplete")
        position = _space(raw, position + 1)


def digest_location(raw, header):
    """Return expected digest and value-byte offset in a validated raw header."""
    expected = expected_digest(header.get("__metadata__"))
    locations = []
    for name, _, _, start, _ in _members(raw, _space(raw, 0)):
        if name == "__metadata__":
            for field, key_start, key_end, value_start, value_end in _members(raw, start):
                if field == FIELD:
                    if (
                        raw[key_start:key_end] != b'"tensorfold_sha256"'
                        or raw[value_start:value_end] != b'"' + expected.encode("ascii") + b'"'
                    ):
                        raise ValueError("snapshot digest key and value must use the declared literal ASCII encoding")
                    locations.append(value_start + 1)
    if len(locations) != 1:
        raise ValueError("snapshot digest must occur exactly once in its metadata object")
    return expected, locations[0]


@dataclass(frozen=True)
class DigestPlan:
    expected: str
    offset: int

    def update(self, digest, chunk, position):
        start, end = max(0, self.offset - position), min(len(chunk), self.offset + 64 - position)
        if start < end:
            view = memoryview(chunk)
            digest.update(view[:start])
            digest.update(_ZERO_BYTES[: end - start])
            digest.update(view[end:])
        else:
            digest.update(chunk)

    def validate(self, digest):
        if not hmac.compare_digest(self.expected, digest.hexdigest()):
            raise ValueError("snapshot content digest differs; rebuild by prefill")


def prepare_digest(stream, sizes, *, label):
    """Read only the bounded schema/header, then rewind this borrowed stream."""
    offset, header = read_header_stream(stream, sizes, label=label)
    stream.seek(8)
    raw = stream.read(offset - 8)
    if len(raw) != offset - 8:
        raise ValueError("snapshot header shortened during integrity preparation")
    expected, location = digest_location(raw, header)
    stream.seek(0)
    return DigestPlan(expected, 8 + location)


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _owned_digest(path, *, sizes, max_bytes, seal):
    if type(max_bytes) is not int or max_bytes < 8:
        raise ValueError("explicit snapshot integrity byte budget required")
    owners = SnapshotStreams()
    primary = None
    try:
        record, before = owners.open(
            path, (os.O_RDWR if seal else os.O_RDONLY) | os.O_NONBLOCK | os.O_NOFOLLOW,
            "r+b" if seal else "rb", max_bytes=max_bytes,
        )
        stream = record.stream
        plan = prepare_digest(stream, sizes, label=path)
        if seal and plan.expected != ZERO:
            raise ValueError("only newly written zero-placeholder snapshots may be sealed")
        digest, position = hashlib.sha256(), 0
        while position < before.st_size:
            chunk = stream.read(min(1 << 20, before.st_size - position))
            if not chunk:
                raise ValueError("snapshot shortened during integrity read")
            plan.update(digest, chunk, position)
            position += len(chunk)
        if stream.read(1) or _identity(os.fstat(stream.fileno())) != _identity(before):
            raise ValueError("snapshot changed during integrity read")
        if seal:
            stream.seek(plan.offset)
            if stream.write(digest.hexdigest().encode("ascii")) != 64:
                raise OSError("snapshot digest publication was incomplete")
            stream.flush()
        else:
            plan.validate(digest)
        after = os.fstat(stream.fileno())
        if _identity(Path(path).stat(follow_symlinks=False)) != _identity(after):
            raise ValueError("snapshot integrity path changed during operation")
        return _identity(after), digest.hexdigest()
    except BaseException as error:
        primary = error
        raise
    finally:
        owners.finish(primary)


def seal_snapshot(path, *, sizes, max_bytes):
    """Seal only trusted synchronous writer output before atomic publication."""
    return _owned_digest(path, sizes=sizes, max_bytes=max_bytes, seal=True)


def verify_snapshot(path, *, sizes, max_bytes):
    """Verify a complete file; header-only index hints cannot substitute for it."""
    return _owned_digest(path, sizes=sizes, max_bytes=max_bytes, seal=False)


class VerificationMemo:
    """Eight immutable-writer receipts, never persisted or native-read authority.

    This only avoids repeating full verification before an unchanged file can
    suppress another write. Atomic replacement and timestamp changes invalidate
    exact identity matches. Native restore always rehashes during its copy.
    The directory/file owner must keep published bytes immutable except through
    atomic replacement; privileged in-place mutation is outside this promise.
    """

    def __init__(self):
        from collections import OrderedDict
        import threading

        self._entries = OrderedDict()
        self._lock = threading.Lock()

    def matches(self, path, identity, digest):
        key = Path(path).absolute()
        with self._lock:
            entry = self._entries.get(key)
            if entry != (identity, digest):
                self._entries.pop(key, None)
                return False
            self._entries.move_to_end(key)
            return True

    def remember(self, path, identity, digest):
        key = Path(path).absolute()
        with self._lock:
            self._entries[key] = (identity, digest)
            self._entries.move_to_end(key)
            while len(self._entries) > 8:
                self._entries.popitem(last=False)

    def reused(self, path, before, after):
        key = Path(path).absolute()
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry[0] == before:
                self._entries[key] = (after, entry[1])
                self._entries.move_to_end(key)

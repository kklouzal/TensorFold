"""Persist validated prefix caches under explicit current-model authority.

FORMAT2 stores tensor data and fixed JSON fields. A serialized class name only
matches an initialized registered class; it never imports or constructs code.
Legacy/unregistered/malformed files are cache misses and are rebuilt by normal
prefill. The caller owns immutable cache tensors through synchronous writes.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
from typing import Any, Sequence

import mlx.core as mx
import numpy as np

from tensorfold.cuda.tensor_file import MAX_HEADER_BYTES
from tensorfold.engine.snapshot_codec import Codec, SIZES
from tensorfold.engine.snapshot_file import reject_snapshot_cleanup_failure as _miss, snapshot_header
from tensorfold.engine.snapshot_integrity import FIELD, ZERO, expected_digest, seal_snapshot, verify_snapshot
from tensorfold.engine.snapshot_payload import capture_snapshot, publish_snapshot
from tensorfold.engine.snapshot_registry import Registry
from tensorfold.engine.snapshot_restore import restore_snapshot

FORMAT = 2
DEFAULT_DIR = Path.home() / ".cache" / "tensorfold" / "prefix-snapshots"
_NAME = re.compile(r"[0-9a-f]{32}\.safetensors\Z")


def snapshot_key(model_id: str, tokens: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(model_id.encode())
    digest.update(b"\0")
    digest.update(",".join(str(int(t)) for t in tokens).encode())
    return digest.hexdigest()[:32]


def model_group(model_id: str) -> str:
    """Eviction owner, distinct from exact cross-restart reuse identity.

    Trusted custom code gets a per-process namespace for reuse. Its prior-run
    files still belong to the same selected checkpoint for keep/byte pruning;
    otherwise a new UUID on each startup would evade the existing disk budget.
    Only the project's explicitly versioned namespace has this interpretation.
    """
    first = model_id.split("|", 1)[0]
    parts = first.split(":")
    if (
        len(parts) == 3
        and parts[0] == "custom-code-v1"
        and len(parts[1]) == 64
        and len(parts[2]) == 32
        and all(c in "0123456789abcdef" for c in parts[1] + parts[2])
    ):
        return ":".join(parts[:2])
    return first


def _require_registry(registry):
    if type(registry) is not Registry:
        raise TypeError("an explicit initialized current-model cache Registry is required")
    return registry


def _paths(directory):
    if not Path(directory).is_dir():
        return []
    return [path for path in Path(directory).glob("*.safetensors") if _NAME.fullmatch(path.name)]


def _identity(path):
    info = path.lstat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _header(path, registry):
    return snapshot_header(
        Path(path), max_bytes=registry.tensor_byte_limit + MAX_HEADER_BYTES + 8, sizes=registry.sizes
    )


def _validated(path, model_id, registry, *, tokens=None):
    header, identity = _header(path, registry)
    metadata = header.get("__metadata__") or {}
    expected_digest(metadata)  # index hint only; the full payload is verified before native use
    found, _ = registry.validate(metadata, header, model_id=model_id)
    if tokens is not None and found != list(tokens):
        raise ValueError("stored tokens differ from the selected prefix")
    return found, identity


def save_snapshot(
    directory: Path,
    model_id: str,
    tokens: Sequence[int],
    cache: list[Any],
    *,
    registry: Registry,
    keep: int = 8,
    codec: Codec | None = None,
) -> Path | None:
    """Validate then publish atomically; keep this model's newest ``keep`` files."""
    _require_registry(registry)
    if type(keep) is not int or keep < 0:
        raise ValueError("snapshot keep count must be a nonnegative integer")
    codec = Codec(mx, np) if codec is None else codec
    for item in cache:
        materialize = getattr(item, "materialize", None)
        if materialize is not None:
            materialize()
    payload = capture_snapshot(
        registry, model_id, tokens, cache, describe_tensor=codec.describe, tensor_kind=codec.kind
    )
    payload.metadata[FIELD] = ZERO
    key = snapshot_key(model_id, tokens)
    sealed = []

    def write(path):
        codec.write(path, payload)
        sealed.append(
            seal_snapshot(path, sizes=registry.sizes, max_bytes=registry.tensor_byte_limit + MAX_HEADER_BYTES + 8)
        )

    def valid_existing(path):
        try:
            _, identity = _validated(path, model_id, registry, tokens=tokens)
            metadata = read_metadata(path)
            expected = expected_digest(metadata)
            if not registry.integrity_memo.matches(path, identity, expected):
                current, digest = verify_snapshot(
                    path, sizes=registry.sizes, max_bytes=registry.tensor_byte_limit + MAX_HEADER_BYTES + 8
                )
                if current != identity:
                    raise ValueError("snapshot changed between schema and content verification")
                registry.integrity_memo.remember(path, current, digest)
        except (OSError, ValueError) as error:
            _miss(error)
            return False
        return True

    written = publish_snapshot(
        Path(directory), key, write, valid_existing=valid_existing, reuse_observed=registry.integrity_memo.reused
    )
    if written is not None and sealed:
        identity, digest = sealed[-1]
        current = _identity(written)
        # chmod/rename alter ctime, but the private writer's inode/bytes/mtime
        # remain unchanged. A concurrent replacement never gains this receipt.
        if current[:4] == identity[:4]:
            registry.integrity_memo.remember(written, current, digest)
    ours = []
    for path in _paths(directory):
        try:
            metadata = read_metadata(path)
            if model_group(metadata.get("model", "")) == model_group(model_id):
                ours.append((path, _identity(path)))
        except (OSError, ValueError) as error:
            _miss(error)
    ours.sort(key=lambda item: item[1][3], reverse=True)
    for path, identity in ours[keep:]:
        if _identity(path) == identity:
            path.unlink(missing_ok=True)
    return written


def load_snapshot(
    path: Path, model_id: str, *, registry: Registry, expected_tokens=None, codec: Codec | None = None
) -> tuple[list[int], list[Any]] | None:
    """Restore initialized registered state; invalid data costs a normal prefill."""
    _require_registry(registry)
    codec = Codec(mx, np) if codec is None else codec
    try:
        return restore_snapshot(
            path,
            model_id,
            registry,
            load_tensors=codec.load,
            describe_tensor=codec.describe,
            convert_numpy=codec.host_array,
            expected_tokens=expected_tokens,
        )
    except (OSError, ValueError) as error:
        _miss(error)
        return None


def load_snapshots(
    directory: Path,
    model_id: str,
    *,
    registry: Registry,
    limit: int | None = None,
    allow: Any = None,
    codec: Codec | None = None,
):
    """Yield current validated snapshots one at a time, newest first."""
    _require_registry(registry)
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("snapshot load limit must be a nonnegative integer or None")
    candidates = []
    for path in _paths(directory):
        try:
            candidates.append((path, _identity(path)))
        except OSError:
            continue
    candidates.sort(key=lambda item: item[1][3], reverse=True)
    count = 0
    for path, _ in candidates:
        if limit is not None and count >= limit:
            return
        try:
            tokens, _ = _validated(path, model_id, registry)
        except (OSError, ValueError) as error:
            _miss(error)
            continue
        if allow is not None and not allow(path):
            continue
        loaded = load_snapshot(path, model_id, registry=registry, expected_tokens=tokens, codec=codec)
        if loaded is not None:
            count += 1
            yield loaded


class DiskBlocks:
    """Current-schema token index; exact stat identities are freshness hints.

    Selected tokens bind the subsequent private-copy restore, including when a
    pathname is replaced after indexing. Indexed metadata never authorizes code.
    """

    def __init__(self, directory: Path, model_id: str, *, registry: Registry):
        self.directory, self.model_id = Path(directory), model_id
        self.registry = _require_registry(registry)
        self._known: dict[Path, tuple[tuple[int, ...], list[int] | None]] = {}

    def blocks(self):
        known = {}
        for path in _paths(self.directory):
            try:
                identity = _identity(path)
                entry = self._known.get(path)
                if entry is None or entry[0] != identity:
                    try:
                        tokens, observed = _validated(path, self.model_id, self.registry)
                        entry = (observed, tokens)
                    except (OSError, ValueError) as error:
                        _miss(error)
                        entry = (identity, None)
                known[path] = entry
            except OSError as error:
                _miss(error)
                continue
        self._known = known
        return [(path, tokens) for path, (_, tokens) in known.items() if tokens]

    def best(self, prompt: Sequence[int], longer_than: int, usable: Any = None):
        best = None
        for path, tokens in self.blocks():
            if usable is not None and not usable(len(tokens)):
                continue
            if longer_than < len(tokens) < len(prompt) and list(prompt[: len(tokens)]) == tokens:
                if best is None or len(tokens) > len(best[1]):
                    best = (path, tokens)
        return best

    def touch(self, tokens: Sequence[int]):
        wanted = list(tokens)
        for path, (identity, known) in list(self._known.items()):
            if known == wanted:
                try:
                    if _identity(path) != identity:
                        continue
                    os.utime(path)
                    self._known[path] = (_identity(path), known)
                except OSError:
                    continue


def read_metadata(path: Path) -> dict[str, str]:
    """Strict header-only regular-file metadata; no tensor or code loading."""
    # Eviction reads use the native format's signed64 payload bound; actual
    # restoration/indexing instead enforce the current model's memory budget.
    header, _ = snapshot_header(path, max_bytes=2**63 - 1, sizes=SIZES)
    return dict(header.get("__metadata__") or {})


def blocks_to_warm(directory: Path, model_id: str, *, registry: Registry):
    """Validated token prefixes from other runtime modes of identical content."""
    _require_registry(registry)
    have, other = [], []
    for path in _paths(directory):
        try:
            header, identity = _header(path, registry)
            metadata = header.get("__metadata__") or {}
            stored_id = metadata.get("model", "")
            if stored_id.split("|", 1)[0] != model_id.split("|", 1)[0]:
                continue
            tokens, _ = registry.validate(metadata, header, model_id=stored_id)
        except (OSError, ValueError) as error:
            _miss(error)
            continue
        if stored_id == model_id:
            have.append(tokens)
        else:
            other.append((identity[3], tokens))
    other.sort(key=lambda item: item[0], reverse=True)
    wanted = []
    for _, tokens in other:
        if any(tokens == h for h in have) or any(tokens == w or w[: len(tokens)] == tokens for w in wanted):
            continue
        wanted = [w for w in wanted if tokens[: len(w)] != w]
        wanted.append(tokens)
    return wanted


__all__ = [
    "DEFAULT_DIR",
    "DiskBlocks",
    "blocks_to_warm",
    "load_snapshot",
    "load_snapshots",
    "read_metadata",
    "save_snapshot",
    "snapshot_key",
    "model_group",
]

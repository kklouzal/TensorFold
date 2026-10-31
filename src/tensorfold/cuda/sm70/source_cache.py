"""Immutable declared SM70 compiler inputs at stable content-addressed paths.

Only owned cache files are published. Installed sources and toolkit headers are
never changed, and callers retain all architecture/toolchain/ABI validation and
module naming before invoking the ordinary TensorFold build path.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

from tensorfold.cleanup import raise_failures


def _directory(path):
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ValueError(f"SM70 source cache requires an owned directory: {path}")


def _verify(directory, manifest, blobs):
    _directory(directory)
    files = {"manifest.json", *blobs}
    directories = {"."}
    for name in blobs:
        directories.update(str(parent) for parent in Path(name).parents)
    for parent, subdirs, names in os.walk(directory, followlinks=False):
        for name in subdirs:
            path = Path(parent) / name
            relative = str(path.relative_to(directory))
            if relative not in directories or not stat.S_ISDIR(path.lstat().st_mode):
                raise ValueError(f"SM70 source cache contains an undeclared directory: {path}")
        for name in names:
            path = Path(parent) / name
            relative = str(path.relative_to(directory))
            if relative not in files or not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError(f"SM70 source cache contains an undeclared input: {path}")
    expected = {"manifest.json": manifest, **blobs}
    for name, data in expected.items():
        path = directory / name
        if not stat.S_ISREG(path.lstat().st_mode) or path.read_bytes() != data:
            raise ValueError(f"SM70 immutable compiler input differs from its declared bytes: {path}")


def stage_sources(sources, *, headers=(), cache_root=None, source_root=None):
    """Return sources in their original order, with the declared relative header closure.

    A complete nonempty directory publishes atomically. Concurrent publishers
    verify the winner byte-for-byte before discarding their own private staging.
    Existing namespaces are immutable: missing, extra, symlinked or changed
    entries fail rather than being repaired underneath a compiler borrower.
    """

    root = Path(source_root or Path(__file__).parent.parent).resolve(strict=True)
    original = [Path(source) for source in sources]
    if not original:
        raise ValueError("SM70 source staging requires declared compiler sources")
    blobs, names = {}, []
    for index, path in enumerate([*original, *map(Path, headers)]):
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or resolved == root or not stat.S_ISREG(resolved.stat().st_mode):
            raise ValueError(f"SM70 compiler input is outside its declared source root: {path}")
        relative = str(resolved.relative_to(root))
        if relative == "manifest.json" or resolved.suffix not in (".cu", ".cpp", ".cuh", ".h"):
            raise ValueError(f"SM70 compiler input has an unsupported source path: {path}")
        blobs[relative] = resolved.read_bytes()
        if index < len(original):
            names.append(relative)
    manifest = json.dumps({"version": 1, "files": [
        {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in sorted(blobs.items())]}, sort_keys=True, separators=(",", ":")).encode()
    identity = hashlib.sha256(manifest).hexdigest()
    if cache_root is None:
        from torch.utils.cpp_extension import get_default_build_root

        cache_root = os.environ.get("TORCH_EXTENSIONS_DIR") or get_default_build_root()
    namespace = Path(cache_root).expanduser().resolve() / "_sm70_sources"
    namespace.mkdir(parents=True, exist_ok=True)
    _directory(namespace)
    destination = namespace / identity
    if os.path.lexists(destination):
        _verify(destination, manifest, blobs)
        return [str(destination / name) for name in names]
    temporary = Path(tempfile.mkdtemp(prefix=f".{identity}.", dir=namespace))
    primary, cleanup = None, []
    try:
        for name, data in {**blobs, "manifest.json": manifest}.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        _verify(temporary, manifest, blobs)
        try:
            os.rename(temporary, destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
        _verify(destination, manifest, blobs)
    except BaseException as error:
        primary = error
    finally:
        if temporary.exists():
            try:
                shutil.rmtree(temporary)
            except BaseException as error:
                cleanup.append(error)
    raise_failures(primary, cleanup)
    return [str(destination / name) for name in names]

"""Immutable selected-source header authority for owned SM70 JIT builds.

The protected installation receipt names every installed CCCL/CUTLASS header.
This narrow consumer verifies those bytes; it does not replace the foundation's
software audit or qualify any device operation. A live process cannot replace
its selected headers: startup/adoption/rebind checks reject changed inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import threading

from tensorfold.cleanup import raise_failures

MANIFEST_ENV = "TENSORFOLD_SM70_FOUNDATION_MANIFEST"
DEFAULT_MANIFEST = "/opt/tensorfold-pins/v100-source-foundation.json"
_MAX_MANIFEST = 64 << 20
_MAX_HEADERS = 65536
_MAX_BYTES = 1 << 30
_FLAGS_ENV = ("CC", "CXX", "CUDA_HOME", "PYTORCH_NVCC", "NVCC_CCBIN", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
              "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH")
_lock = threading.RLock()
_UNSELECTED = object()
_binding = _UNSELECTED


def _compiler_environment():
    # NVCC consumes both argument channels. GCC's C++ preprocessing consumes
    # CPATH/CPLUS_INCLUDE_PATH; even an empty path element adds the current
    # directory. Torch's JIT does not consume CPPFLAGS/CXXFLAGS, and its CUDA
    # inputs are C++, so C_INCLUDE_PATH is not refused here.
    for key in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "CPATH", "CPLUS_INCLUDE_PATH"):
        value = os.environ.get(key)
        if value is not None and (value.strip() or key in ("CPATH", "CPLUS_INCLUDE_PATH")):
            raise ValueError("selected foundation rejects unbound compiler input: " + key)


def _compiler_options(kwargs, roots):
    # Only the selected roots are admitted through JIT include-path arguments.
    # Owned quoted headers remain at their staged source locations; framework
    # and toolkit paths are supplied by the pinned Torch/toolchain build.
    if kwargs.get("extra_include_paths", []) != list(map(str, roots)):
        raise ValueError("selected foundation rejects unbound JIT include paths")
    forbidden = ("-I", "-isystem", "-iquote", "-idirafter", "-iprefix", "-iwithprefix", "-include", "-imacros",
                 "--include-path", "--system-include", "--pre-include", "--options-file", "-optf", "@",
                 "-Xcompiler", "--compiler-options", "-Xpreprocessor", "-Wp,", "--forward-unknown-to-host-compiler")
    for key in ("extra_cflags", "extra_cuda_cflags"):
        for flag in kwargs.get(key, []):
            if not isinstance(flag, str) or any(token.startswith(forbidden) for token in shlex.split(flag)):
                raise ValueError("selected foundation rejects unbound header/preprocessor options in " + key)


def _stat(path):
    value = path.stat()
    if path.is_symlink() or not path.is_file():
        raise ValueError("selected foundation inputs must be ordinary canonical files")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _read(path, consume):
    """Attempt close once and preserve both a failed read and failed retirement."""
    primary = prior = result = None
    cleanup = [None]
    stream = path.open("rb")
    try:
        result = consume(stream)
    except BaseException as error:
        primary = error
        prior = (BaseException.__cause__.__get__(error), BaseException.__context__.__get__(error),
                 BaseException.__suppress_context__.__get__(error))
    try:
        stream.close()
    except BaseException as error:
        cleanup[0] = error
    if primary is not None:
        BaseException.__cause__.__set__(primary, prior[0])
        BaseException.__context__.__set__(primary, prior[1])
        BaseException.__suppress_context__.__set__(primary, prior[2])
    raise_failures(primary, cleanup if cleanup[0] is not None else ())
    return result


def _sha256(path, expected_bytes):
    before = _stat(path)
    if before[2] != expected_bytes:
        raise ValueError("selected foundation header size differs from the protected receipt")
    digest = hashlib.sha256()
    def consume(stream):
        left = expected_bytes
        while left:
            block = stream.read(min(left, 1 << 20))
            if not block:
                raise ValueError("selected foundation header was truncated")
            digest.update(block)
            left -= len(block)
        if stream.read(1):
            raise ValueError("selected foundation header grew while reading")
    _read(path, consume)
    if _stat(path) != before:
        raise ValueError("selected foundation header changed while reading")
    return digest.hexdigest(), before


def _inventory(roots):
    files = {}
    directories = 0
    for root in roots:
        if root.is_symlink() or root.resolve(strict=True) != root or not root.is_dir():
            raise ValueError("selected foundation include roots must be canonical directories")
        for directory, subdirs, names in os.walk(root, followlinks=False):
            directories += 1
            if directories > _MAX_HEADERS:
                raise ValueError("selected foundation include directory inventory exceeds its bound")
            if any((Path(directory) / name).is_symlink() for name in subdirs):
                raise ValueError("selected foundation include tree contains a directory symlink")
            for name in names:
                path = Path(directory) / name
                files[str(path)] = _stat(path)
                if len(files) > _MAX_HEADERS:
                    raise ValueError("selected foundation header inventory exceeds its bound")
    return files


class SelectedHeaders:
    def __init__(self, path):
        _compiler_environment()
        self.path = path
        before = _stat(path)
        if before[2] > _MAX_MANIFEST:
            raise ValueError("selected foundation manifest exceeds its bound")
        raw = _read(path, lambda stream: stream.read(_MAX_MANIFEST + 1))
        if len(raw) > _MAX_MANIFEST:
            raise ValueError("selected foundation manifest exceeds its bound")
        if _stat(path) != before:
            raise ValueError("selected foundation manifest changed while reading")
        value = json.loads(raw)
        if (not isinstance(value, dict) or value.get("schema") != "tensorfold.v100-source-foundation-install"
                or type(value.get("version")) is not int or value["version"] != 2
                or value.get("runtime_policy") != "torch2141-build-host-v1"
                or not isinstance(value.get("include_roots"), dict)
                or set(value["include_roots"]) != {"cccl", "cutlass"}
                or not isinstance(value.get("artifacts"), list) or len(value["artifacts"]) > _MAX_HEADERS):
            raise ValueError("selected foundation needs the protected v2 installation/include policy")
        if any(not isinstance(root, str) or not root for root in value["include_roots"].values()):
            raise ValueError("selected foundation include roots must be nonempty paths")
        self.roots = tuple(Path(value["include_roots"][role]) for role in ("cccl", "cutlass"))
        if any(not root.is_absolute() for root in self.roots) or self.roots[0] == self.roots[1]:
            raise ValueError("selected foundation include roots must be distinct absolute directories")
        files = _inventory(self.roots)
        declared = {}
        for artifact in value["artifacts"]:
            if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
                raise ValueError("selected foundation artifact record is malformed")
            target = Path(artifact["path"])
            if any(target.is_relative_to(root) for root in self.roots):
                size, sha = artifact.get("bytes"), artifact.get("sha256")
                if (str(target) in declared or type(size) is not int or size < 0 or not isinstance(sha, str)
                        or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)):
                    raise ValueError("selected foundation header receipt is malformed or duplicated")
                declared[str(target)] = (size, sha)
        if not files or set(declared) != set(files) or sum(size for size, _ in declared.values()) > _MAX_BYTES:
            raise ValueError("selected foundation receipt must cover the complete bounded include inventory")
        records, self.stats = [], {str(path): before}
        for filename, (size, expected) in sorted(declared.items()):
            actual, stat = _sha256(Path(filename), size)
            if actual != expected:
                raise ValueError("selected foundation header digest differs from the protected receipt")
            self.stats[filename] = stat
            records.append({"path": filename, "bytes": size, "sha256": actual})
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.headers_sha256 = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.environment = {key: os.environ.get(key) for key in _FLAGS_ENV}
        self.verify()

    def verify(self):
        _compiler_environment()
        observed = _inventory(self.roots)
        observed[str(self.path)] = _stat(self.path)
        if (observed != self.stats
                or self.environment != {key: os.environ.get(key) for key in _FLAGS_ENV}):
            raise ValueError("selected foundation headers/flags changed; retire this process before replacing its build binding")

    def record(self):
        return {"manifest_sha256": self.manifest_sha256, "headers_sha256": self.headers_sha256,
                "include_roots": list(map(str, self.roots)), "environment": dict(self.environment)}

    def build_identity(self, kwargs):
        _compiler_options(kwargs, self.roots)
        # These are the actual final ordered caller/architecture/toolkit flags.
        options = {key: kwargs.get(key, []) for key in
                   ("extra_cflags", "extra_cuda_cflags", "extra_ldflags", "extra_include_paths")}
        return hashlib.sha256(json.dumps({**self.record(), "options": options}, sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()


def selected_headers():
    """Build/preparation boundary only; no file work on native launch/cache hits."""
    global _binding

    configured = os.environ.get(MANIFEST_ENV)
    path = Path(configured or DEFAULT_MANIFEST)
    with _lock:
        selection = None if configured is None and not path.exists() else str(path)
        if _binding is not _UNSELECTED:
            previous = None if _binding is None else str(_binding.path)
            if previous != selection:
                raise ValueError("selected foundation manifest changed; retire this process before rebinding")
            if _binding is not None:
                _binding.verify()
            return _binding
        if selection is None:
            _binding = None
            return None
        if not path.is_absolute() or path.resolve(strict=True) != path:
            raise ValueError("selected foundation manifest must have a canonical absolute path")
        _binding = SelectedHeaders(path)
        return _binding


def selected_identity():
    owner = selected_headers()
    return None if owner is None else owner.record()

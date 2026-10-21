"""Prepared native modules from the managed image's immutable, sealed bundle.

The existing foundation launch boundary owns artifact/input stat validation and
publishes the compact activation identity. This module binds that identity once
per process, before importing an extension. Its explicit preparation lock owns
the bounded module cache; modules and their code remain live through process
exit. Other targets and separate source/build workspaces retain their JIT path.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import sysconfig
import threading
from functools import lru_cache
from pathlib import Path

ACTIVATION = "TF_NATIVE_ARTIFACTS"
MAX_MANIFEST = 2**20
_PREPARATION = threading.RLock()


def canonical(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def abi() -> dict:
    """Actual consumer ABI; compiler/provider identities belong to the foundation."""
    import torch

    return {"host_machine": platform.machine(), "python_soabi": sysconfig.get_config_var("SOABI"),
            "torch_version": torch.__version__, "torch_git_version": torch.version.git_version,
            "torch_cxx11_abi": torch.compiled_with_cxx11_abi(), "cuda_runtime": torch.version.cuda}


def request(package: Path, sources: str | list[str], options: dict) -> dict:
    """One exact semantic build request, shared by preparation and consumption."""
    paths = [sources] if isinstance(sources, str) else sources
    names = [str(Path(path).resolve(strict=True).relative_to(package)) for path in paths]
    allowed = {"extra_cflags", "extra_cuda_cflags", "extra_ldflags", "extra_include_paths", "verbose"}
    if set(options) - allowed:
        raise ValueError("prepared native request has unsupported build options")
    flags = {}
    for name in ("extra_cflags", "extra_cuda_cflags", "extra_ldflags", "extra_include_paths"):
        values = options.get(name, [])
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError("prepared native flags must be lists of strings")
        flags[name] = ([str(Path(value).resolve(strict=True).relative_to(package)) for value in values]
                       if name == "extra_include_paths" else values)
    return {"sources": names, "options": flags}


def _relative(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("native artifact path must be relative")
    path = (root / value).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("native artifact path escapes its bundle")
    return path


@lru_cache(maxsize=1)
def _bundle(activation: str) -> dict:
    # Called under _PREPARATION. The launch-owned mapping is immutable for this
    # process; each source/binary has already passed the foundation stat seal.
    expected = json.loads(activation)
    if not isinstance(expected, dict) or set(expected) != {"manifest", "sha256", "identity"}:
        raise ValueError("invalid prepared native activation")
    path = Path(expected["manifest"]).resolve(strict=True)
    with path.open("rb") as stream:
        raw = stream.read(MAX_MANIFEST + 1)
    if len(raw) > MAX_MANIFEST or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError("prepared native manifest differs from its seal")
    value = json.loads(raw)
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1 or value.get("kind") != "tensorfold-native-artifacts"
            or value.get("identity") != expected["identity"] or value.get("abi") != abi()
            or value.get("target_capability") != [12, 1]
            or not isinstance(value.get("entries"), dict) or len(value["entries"]) != 3):
        raise ValueError("incompatible prepared native manifest")
    package = Path(value["package_root"]).resolve(strict=True)
    if not package.is_dir():
        raise ValueError("prepared native package root is not a directory")
    for name, entry in value["entries"].items():
        if not isinstance(name, str) or not isinstance(entry, dict) or entry.get("module_name") != name:
            raise ValueError("invalid prepared native entry")
        binary = _relative(path.parent, entry["binary"])
        if not binary.is_file() or not isinstance(entry.get("request"), dict):
            raise ValueError("invalid prepared native binary/request")
        entry["path"] = binary
    return {"package": package, "entries": value["entries"], "modules": {}}


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot prepare native module {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_prepared(name: str, sources: str | list[str], options: dict):
    """Bind a matching managed artifact, or retain a distinct valid JIT region.

    The caller must first enforce its actual GPU/minimum-capability contract and
    supply its actual-target qualified name. A declared installed-source entry
    cannot hide corrupt or changed flags behind a JIT retry. Explicit custom
    build directories and separate source workspaces do not consume this bundle.
    """
    activation = os.environ.get(ACTIVATION)
    if not activation or "build_directory" in options:
        return None
    with _PREPARATION:
        bundle = _bundle(activation)
        entry = bundle["entries"].get(name)
        if entry is None:
            return None
        paths = [sources] if isinstance(sources, str) else sources
        if any(not Path(path).resolve(strict=True).is_relative_to(bundle["package"]) for path in paths):
            return None
        actual = request(bundle["package"], sources, options)
        if actual != entry["request"]:
            raise ValueError(f"prepared native recipe changed: {name}")
        if name not in bundle["modules"]:
            bundle["modules"][name] = _import(name, entry["path"])
        return bundle["modules"][name]

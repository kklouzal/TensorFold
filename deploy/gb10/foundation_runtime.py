"""Capture sealed CUDA providers, preload their SONAMEs, and verify mapped objects.

ELF inspection and Vision RECORD hashing run only during capture. Startup uses
sealed stat identities. The bootstrap calls activation once before importing
Torch and retains its returned CDLL tuple for the process lifetime. Drivers are
owned by the separate driver proof; no driver enters this provider contract.
"""
from __future__ import annotations

import base64
import csv
import ctypes
import hashlib
import io
import os
import re
import stat
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from foundation_policy import canonical_name, wheel_url

CUDA_NAME = re.compile(
    r"^lib(?:cudart|cublas(?:Lt)?|cudnn[^/]*|cufft[^/]*|curand|cusolver[^/]*|"
    r"cusparse(?:Lt)?|nvrtc(?:-builtins)?|nvJitLink|nvToolsExt|cupti|cufile[^/]*|"
    r"nvshmem[^/]*|nccl|nvfatbin|nvvm|nvptxcompiler|nvjpeg|nvperf_(?:host|target)|npp[^/]*|nvblas|cuobjclient|cudss)"
    r"(?:[.-][A-Za-z0-9_]+)?\.so(?:\.|$)"
)
SYSTEM_NAMES = frozenset({"libstdc++.so.6", "libgcc_s.so.1"})
DRIVER_NAME = re.compile(r"^lib(?:cuda|nvidia-[^/]+)\.so(?:\.|$)")
DYNAMIC_TAG = re.compile(r"\((SONAME|NEEDED)\).*?\[([^\]]+)\]")
VISION_MEMBER = re.compile(r"torchvision\.libs/lib(cudart|nvjpeg)\.[0-9a-f]{8}\.so\.13\Z")
STAT_FIELDS = ("realpath", "size", "mtime_ns", "mode")


def provider_kind(name: str) -> str | None:
    if any(name == base or name.startswith(base + ".") for base in SYSTEM_NAMES):
        return "cxx_runtime"
    if DRIVER_NAME.match(name):
        return "driver"
    return "cuda" if CUDA_NAME.match(name) else None


def elf(path: Path) -> dict:
    result = subprocess.run(["readelf", "--dynamic", "--wide", str(path)], check=True,
                            capture_output=True, text=True, env={**os.environ, "LC_ALL": "C"})
    sonames, needed = [], []
    for kind, name in DYNAMIC_TAG.findall(result.stdout):
        (sonames if kind == "SONAME" else needed).append(name)
    if len(sonames) > 1:
        raise RuntimeError(f"multiple ELF SONAMEs: {path}")
    return {"soname": sonames[0] if sonames else None, "needed": needed}


def _stat(path: Path) -> tuple[dict, os.stat_result]:
    real = path.resolve(strict=True)
    info = real.stat()
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"runtime provider is not a regular file: {path}")
    return {"realpath": str(real), "size": info.st_size, "mtime_ns": info.st_mtime_ns,
            "mode": stat.S_IMODE(info.st_mode)}, info


def _check_stat(entry: dict) -> os.stat_result:
    actual, info = _stat(Path(entry["path"]))
    for key in STAT_FIELDS:
        if actual[key] != entry[key]:
            raise RuntimeError(f"sealed runtime provider changed: {entry['path']}: {key}")
    return info


def _sealed(path: Path, sealed_files: dict) -> dict:
    actual, _ = _stat(path)
    expected = sealed_files.get(str(path))
    if expected is None:
        matches = [row for row in sealed_files.values() if row["realpath"] == actual["realpath"]]
        if not matches:
            raise RuntimeError(f"runtime provider is outside sealed files: {path}")
        expected = matches[0]
    if any(actual[key] != expected[key] for key in STAT_FIELDS):
        raise RuntimeError(f"runtime provider differs from its seal: {path}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected.get("sha256", "")):
        raise RuntimeError(f"runtime provider has no sealed content identity: {path}")
    return {"path": str(path), **actual, "sha256": expected["sha256"]}


def _excluded(path: Path) -> bool:
    return any(part in {"stubs", "compat"} for part in path.parts)


def _native_index(native_files: set[Path], compiler_files: set[Path], sealed_files: dict, nccl_library: Path,
                  cupti_library: Path | None) -> dict:
    compiler_providers = {path.resolve(strict=True) for path in compiler_files}
    selected_nccl = nccl_library.resolve(strict=True)
    selected_cupti = None if cupti_library is None else cupti_library.resolve(strict=True)
    cupti_dir = None if selected_cupti is None else selected_cupti.parent
    index, processed = {}, set()
    for path in sorted(native_files | compiler_files | {nccl_library}):
        kind = provider_kind(path.name)
        if kind in {None, "driver"} or _excluded(path):
            continue
        if kind == "cxx_runtime" and path.resolve(strict=True) not in compiler_providers:
            continue
        entry = _sealed(path, sealed_files)
        real = Path(entry["realpath"])
        if _excluded(real):
            continue
        if path.name.startswith(("libnvperf_host.so", "libnvperf_target.so")) and real.parent != cupti_dir:
            continue
        if selected_cupti is not None and path.name.startswith("libcupti.so") and real != selected_cupti:
            continue
        if path.name.startswith("libnccl.so") and real != selected_nccl:
            continue
        if real in processed:
            continue
        processed.add(real)
        entry.update(elf(real))
        _check_stat(entry)
        name = entry["soname"]
        kind = provider_kind(name or "")
        if kind not in {"cuda", "cxx_runtime"}:
            raise RuntimeError(f"unexpected native provider SONAME: {real}: {name}")
        if kind == "cxx_runtime" and real not in compiler_providers:
            continue
        if name.startswith("libnccl.so") and real != selected_nccl:
            continue
        if name.startswith(("libnvperf_host.so", "libnvperf_target.so")) and real.parent != cupti_dir:
            continue
        if selected_cupti is not None and name.startswith("libcupti.so") and real != selected_cupti:
            continue
        entry.update(kind=kind, provider_role="native_foundation")
        previous = index.get(name)
        if previous is not None and not Path(previous["realpath"]).samefile(real):
            raise RuntimeError(f"ambiguous native providers for {name}")
        index.setdefault(name, entry)
    if ("libnccl.so.2" not in index or Path(index["libnccl.so.2"]["realpath"]) != selected_nccl):
        raise RuntimeError("selected NCCL library is not the sealed libnccl.so.2 provider")
    for name in SYSTEM_NAMES:
        if name not in index:
            raise RuntimeError(f"missing base C++ runtime provider: {name}")
    return index


def _dependency_order(required: set[str], index: dict) -> list[str]:
    ordered, active, done = [], set(), set()

    def visit(name: str) -> None:
        if name in done:
            return
        if name in active:
            raise RuntimeError(f"cycle in CUDA provider dependencies: {name}")
        if name not in index:
            raise RuntimeError(f"no sealed provider for required {name}")
        active.add(name)
        dependencies = [dep for dep in index[name]["needed"] if provider_kind(dep) in {"cuda", "cxx_runtime"}]
        if name.startswith("libnvrtc.so"):
            builtins = [dep for dep in index if dep.startswith("libnvrtc-builtins.so")]
            if len(builtins) != 1:
                raise RuntimeError("NVRTC requires one explicit sealed builtins provider")
            dependencies = builtins + dependencies
        for dependency in dependencies:
            visit(dependency)
        active.remove(name)
        done.add(name)
        ordered.append(name)

    for name in sorted(required):
        visit(name)
    return ordered


def _torch_requirements(torch_dir: Path, index: dict, sealed_files: dict, cupti_library: Path | None) -> dict:
    paths = {Path(path) for path in sealed_files if
             (Path(path).parent == torch_dir and Path(path).name.startswith("_C") and Path(path).suffix == ".so")
             or (Path(path).parent == torch_dir / "lib" and ".so" in Path(path).name)}
    if not paths or not any(path.parent == torch_dir for path in paths):
        raise RuntimeError("sealed Torch native extension/libraries are missing")
    required, objects = set(SYSTEM_NAMES), []
    for path in sorted(paths):
        entry = _sealed(path, sealed_files)
        entry.update(elf(Path(entry["realpath"])))
        _check_stat(entry)
        objects.append(entry)
        required.update(name for name in entry["needed"] if provider_kind(name) in {"cuda", "cxx_runtime"})
    if cupti_library is not None:
        real = str(cupti_library.resolve(strict=True))
        chosen = [name for name, row in index.items() if row["realpath"] == real and name.startswith("libcupti.so")]
        if len(chosen) != 1:
            raise RuntimeError("selected CUPTI library is outside the native provider index")
        required.update(chosen)
    return {"objects": objects, "required": sorted(required), "order": _dependency_order(required, index)}


def _vision_bundle(site: Path, manifest: dict, sealed_files: dict) -> dict:
    pin = manifest["roles"]["foundation"]["torchvision"]
    official = {key: pin[key] for key in ("version", "url", "sha256")}
    wheel_url("torchvision", official)
    owners = [dist for dist in metadata.distributions(path=[str(site)])
              if canonical_name(dist.metadata.get("Name", "")) == "torchvision"]
    if (len(owners) != 1 or owners[0].version != pin["version"]
            or Path(owners[0].locate_file("torchvision")).resolve(strict=True) != site / "torchvision"):
        raise RuntimeError("Vision bundle metadata owner/version differs from official manifest")
    dist = owners[0]
    records = [str(path) for path in dist.files or [] if str(path).endswith(".dist-info/RECORD")]
    if len(records) != 1:
        raise RuntimeError("Vision bundle requires one installed RECORD owner")
    record_path = Path(dist.locate_file(records[0]))
    record = _sealed(record_path, sealed_files)
    if record["size"] > 2 * 2**20:
        raise RuntimeError("Vision RECORD exceeds its 2 MiB audit limit")
    with record_path.open("rb") as stream:
        record_bytes = stream.read(2 * 2**20 + 1)
    if len(record_bytes) > 2 * 2**20 or hashlib.sha256(record_bytes).hexdigest() != record["sha256"]:
        raise RuntimeError("Vision RECORD differs from its sealed content identity")
    rows = list(csv.reader(io.StringIO(record_bytes.decode("utf-8"))))
    providers = []
    for row in rows:
        if not row or not row[0].startswith("torchvision.libs/"):
            continue
        member = row[0]
        kind = provider_kind(Path(member).name)
        if kind is None:
            continue
        match = VISION_MEMBER.fullmatch(member)
        if match is None or len(row) != 3:
            raise RuntimeError(f"unapproved Vision bundled provider: {member}")
        path = Path(dist.locate_file(member))
        if path.is_symlink() or path.resolve(strict=True) != site / member:
            raise RuntimeError(f"Vision bundle member is not its exact regular file: {member}")
        entry = _sealed(path, sealed_files)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").digest()
        encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        if row[1:] != ["sha256=" + encoded, str(entry["size"])] or digest.hex() != entry["sha256"]:
            raise RuntimeError(f"Vision bundle member differs from its sealed RECORD: {member}")
        entry.update(elf(path))
        _check_stat(entry)
        if entry["soname"] != f"lib{match[1]}.so.13":
            raise RuntimeError(f"Vision bundle member has unexpected ELF SONAME: {member}")
        entry.update(member=member, record=row, kind="cuda", provider_role="official_vision_bundle")
        providers.append(entry)
    if sorted(entry["soname"] for entry in providers) != ["libcudart.so.13", "libnvjpeg.so.13"]:
        raise RuntimeError("Vision exception requires exactly its cudart and nvjpeg RECORD members")
    return {**official, "module_directory": str(site / "torchvision"), "record": record, "providers": providers}


def capture_runtime(*, foundation: Path, manifest: dict, native_files: set[Path], compiler_files: set[Path], sealed_files: dict,
                    nccl_library: Path | str, cupti_library: Path | str | None = None) -> dict:
    """Select C++ runtime files from compiler packages, CUDA from the native SDK.

    SDK tools' private C++ bundles remain sealed files but cannot become the
    application/compiler runtime merely by sharing a SONAME.
    """
    foundation = foundation.resolve(strict=True)
    site = foundation / "lib" / ("python" + manifest["python"]) / "site-packages"
    cupti = None if cupti_library is None else Path(cupti_library)
    index = _native_index(native_files, compiler_files, sealed_files, Path(nccl_library), cupti)
    torch = _torch_requirements(site / "torch", index, sealed_files, cupti)
    vision = _vision_bundle(site, manifest, sealed_files)
    return {"schema_version": 1, "foundation": str(foundation), "index": index, "torch": torch, "vision": vision,
            "cupti_dir": None if cupti is None else str(cupti.resolve(strict=True).parent)}


def activate_runtime(contract: dict) -> tuple:
    """Preload before Torch import; the caller strongly owns returned handles."""
    if type(contract.get("schema_version")) is not int or contract["schema_version"] != 1:
        raise RuntimeError("unsupported sealed runtime contract")
    if "torch" in sys.modules:
        raise RuntimeError("Torch already imported before owned runtime activation")
    for entry in [*contract["index"].values(), *contract["torch"]["objects"], contract["vision"]["record"],
                  *contract["vision"]["providers"]]:
        _check_stat(entry)
    owners = [None] * len(contract["torch"]["order"])
    for slot, name in enumerate(contract["torch"]["order"]):
        owners[slot] = ctypes.CDLL(contract["index"][name]["realpath"], mode=os.RTLD_GLOBAL | os.RTLD_NOW)
    return tuple(owners)


def mapped_runtime(contract: dict, text: str | None = None, *, allow_vision: bool = False) -> list[dict]:
    """Compare live map segment device/inode with stat; never persist those IDs."""
    if type(contract.get("schema_version")) is not int or contract["schema_version"] != 1:
        raise RuntimeError("unsupported sealed runtime contract")
    records = list(contract["index"].values())
    if allow_vision:
        records += contract["vision"]["providers"]
    by_path = {entry["realpath"]: entry for entry in records}
    paths: dict[Path, set[tuple[int, int, int]]] = {}
    for row in (Path("/proc/self/maps").read_text() if text is None else text).splitlines():
        fields = row.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        path = Path(fields[5].removesuffix(" (deleted)"))
        if provider_kind(path.name) in {None, "driver"}:
            continue
        if fields[5].endswith(" (deleted)"):
            raise RuntimeError(f"deleted runtime provider remains mapped: {fields[5]}")
        major, minor = fields[3].split(":")
        paths.setdefault(path, set()).add((int(major, 16), int(minor, 16), int(fields[4])))
    seen, groups, result = set(), {}, []
    for path, segments in sorted(paths.items()):
        if _excluded(path):
            raise RuntimeError(f"stub or compatibility CUDA provider mapped: {path}")
        real = path.resolve(strict=True)
        entry = by_path.get(str(real))
        if entry is None:
            matches = [row for row in records if Path(row["realpath"]).samefile(real)]
            if not matches:
                raise RuntimeError(f"non-foundation runtime provider mapped: {path}")
            entry = matches[0]
        info = _check_stat(entry)
        current = real.stat()
        physical = (os.major(current.st_dev), os.minor(current.st_dev), current.st_ino)
        if segments != {physical} or (info.st_dev, info.st_ino) != (current.st_dev, current.st_ino):
            raise RuntimeError(f"mapped runtime object differs from current file: {path}")
        key = (entry["soname"], current.st_dev, current.st_ino)
        if key in seen:
            continue
        group = groups.setdefault(entry["soname"], [])
        roles = {row["provider_role"] for row in group} | {entry["provider_role"]}
        if group and (len(group) != 1 or roles != {"native_foundation", "official_vision_bundle"}
                      or entry["soname"] not in {"libcudart.so.13", "libnvjpeg.so.13"}):
            raise RuntimeError(f"multiple physical runtime providers mapped: {entry['soname']}")
        seen.add(key)
        value = {**entry, "path": entry["realpath"], "mapped_path": str(path), "mapped_object_verified": True}
        group.append(value)
        result.append(value)
    return result

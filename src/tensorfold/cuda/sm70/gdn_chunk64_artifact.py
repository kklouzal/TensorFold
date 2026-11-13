"""Cold, explicit identity admission for one owned TileLang artifact.

An externally selected manifest digest is an administrator's authority, not a
signature or numerical qualification. Installed packages and the retained wheel
must stay immutable for the dedicated component process. No TileLang/FFI import,
installation, environment mutation or compiler operation belongs in this module.
"""
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tensorfold.cleanup import raise_failures

SCHEMA = "tensorfold.g3-owned-tilelang-attestation-v1"
MAX_METADATA_BYTES = 16 << 20
# The inspected public wheel actually aborts with this protected FFI runtime.
# A selected manifest cannot turn that known contradictory artifact into proof.
UNPATCHED_TILELANG_WHEEL_SHA256 = "85180d1a96defeecdf52d5d075a31c3fc551d8485981e6b636762a9cd7eb02fe"
SOURCE_NAMES = frozenset(("tilelang", "tvm", "tvm_ffi", "dlpack", "libbacktrace",
                          "cutlass", "composable_kernel"))
FFI_WHEEL_SHA256 = "30e63c6fee0028fb0458253752ffe31a7b27f123589edaabdc9132e8b6cb6285"
FFI_WHEEL_BYTES = 3532950
FFI_MEMBERS_SHA256 = "f5ea15da8f845035353e2429a1adca04d4888065994bec7365199a77c6b0393a"
FFI_CORE = {"bytes": 2871944, "sha256": "7f2f46814e4ba5e231d3cd0916a6c3e13cb9816ddcfe89b6e571b4fef5917fb4"}
COMPILER_LIBRARIES = ("libtilelang.so", "libtvm_runtime.so", "libtvm_compiler.so")


def _fields(value, names):
    if type(value) is not dict or set(value) != set(names):
        raise ValueError("G3 owned attestation fields differ")


def _sha(value, length=64):
    if type(value) is not str or re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is None:
        raise ValueError("G3 owned attestation digest is invalid")
    return value


def _extent(value):
    _fields(value, ("bytes", "sha256"))
    if type(value["bytes"]) is not int or not 0 <= value["bytes"] < 2**63:
        raise ValueError("G3 owned artifact extent is invalid")
    _sha(value["sha256"])


def _members(value, package, distribution):
    if type(value) is not dict or not value:
        raise ValueError("G3 owned artifact needs its complete installed member inventory")
    metadata = []
    for name, record in value.items():
        if type(name) is not str or "\\" in name:
            raise ValueError("G3 owned member path is invalid")
        path = PurePosixPath(name)
        if len(path.parts) < 2 or path.is_absolute() or ".." in path.parts or str(path) != name:
            raise ValueError("G3 owned member path escapes its distribution")
        root = path.parts[0]
        if (not (root == package or root.startswith(distribution + "-") and root.endswith(".dist-info"))
                or "__pycache__" in path.parts or path.suffix == ".pyc" or path.name == "RECORD"):
            raise ValueError("G3 owned member has unexpected ownership")
        if package == "tilelang" and path.name in ("libtvm_ffi.so", "libtvm_ffi_testing.so"):
            raise ValueError("G3 owned TileLang must not bundle a second FFI runtime")
        if path.name == "METADATA" and len(path.parts) == 2:
            metadata.append(root)
        _extent(record)
    if len(metadata) != 1 or any(
            not name.startswith(package + "/") and PurePosixPath(name).parts[0] != metadata[0]
            for name in value):
        raise ValueError("G3 owned artifact requires one distribution metadata owner")
    if package + "/__init__.py" not in value:
        raise ValueError("G3 owned artifact omits its Python entrypoint")


def _compiler_members(members):
    """The pinned bootstrap imports bundled TVM and these three native owners."""
    libraries = {}
    for library in COMPILER_LIBRARIES:
        names = [name for name in members if PurePosixPath(name).name == library]
        if len(names) != 1:
            raise ValueError("G3 owned wheel must identify each compiler library exactly once")
        libraries[library] = names[0]
    initializers = [name for name in members if name.endswith("/python/tvm/__init__.py")]
    if len(initializers) != 1:
        raise ValueError("G3 owned wheel must identify its bundled TVM Python tree")
    return libraries, initializers[0]


def validate(value):
    """Validate the one agreed schema; no installed artifact is admitted here."""
    _fields(value, ("schema", "source_pins", "build", "tilelang", "protected_ffi",
                    "compiler_environment", "evidence", "qualification"))
    if value["schema"] != SCHEMA or value["qualification"] != "artifact-identity-only":
        raise ValueError("G3 owned artifact schema/qualification differs")
    _fields(value["source_pins"], SOURCE_NAMES)
    for pin in value["source_pins"].values():
        _sha(pin, 40)
    _fields(value["build"], ("proposal_sha256", "source_archive_sha256", "image_id"))
    for key in ("proposal_sha256", "source_archive_sha256"):
        _sha(value["build"][key])
    image = value["build"]["image_id"]
    if type(image) is not str or not image.startswith("sha256:"):
        raise ValueError("G3 owned build requires its actual immutable image identity")
    _sha(image[7:])
    for key, package, distribution in (("tilelang", "tilelang", "tilelang"),
                                       ("protected_ffi", "tvm_ffi", "apache_tvm_ffi")):
        item = value[key]
        _fields(item, ("version", "wheel", "installed_members"))
        if type(item["version"]) is not str or not item["version"] or len(item["version"]) > 512:
            raise ValueError("G3 owned artifact requires its exact actual metadata version")
        _extent(item["wheel"])
        _members(item["installed_members"], package, distribution)
    _compiler_members(value["tilelang"]["installed_members"])
    ffi = value["protected_ffi"]
    if value["tilelang"]["wheel"]["sha256"] == UNPATCHED_TILELANG_WHEEL_SHA256:
        raise ValueError("G3 known unpatched TileLang wheel is not admitted")
    if (ffi["version"] != "0.1.16" or ffi["wheel"]["sha256"] != FFI_WHEEL_SHA256
            or ffi["wheel"]["bytes"] != FFI_WHEEL_BYTES
            or hashlib.sha256(json.dumps(ffi["installed_members"], sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest() != FFI_MEMBERS_SHA256
            or ffi["installed_members"].get("tvm_ffi/lib/libtvm_ffi.so") != FFI_CORE):
        raise ValueError("G3 owned artifact must retain the exact protected FFI0.1.16")
    env = value["compiler_environment"]
    _fields(env, ("cuda_home", "nvcc_sha256", "headers"))
    if type(env["cuda_home"]) is not str or not Path(env["cuda_home"]).is_absolute():
        raise ValueError("G3 owned compiler home must be absolute")
    _sha(env["nvcc_sha256"])
    _fields(env["headers"], ("manifest_sha256", "headers_sha256", "include_roots", "environment"))
    for name in ("manifest_sha256", "headers_sha256"):
        _sha(env["headers"][name])
    if (type(env["headers"]["include_roots"]) is not list or not env["headers"]["include_roots"]
            or any(type(p) is not str or not Path(p).is_absolute() for p in env["headers"]["include_roots"])
            or type(env["headers"]["environment"]) is not dict
            or any(type(k) is not str or v is not None and type(v) is not str
                   for k, v in env["headers"]["environment"].items())):
        raise ValueError("G3 owned header descriptor is invalid")
    _fields(value["evidence"], ("producer", "audit", "tl_first", "ffi_torch_first"))
    _fields(value["evidence"]["producer"], ("sha256",))
    _sha(value["evidence"]["producer"]["sha256"])
    for name in ("audit", "tl_first", "ffi_torch_first"):
        _extent(value["evidence"][name])
    return value


def canonical(value):
    return json.dumps(validate(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def cache_name(attestation_sha256, compiler_environment):
    _sha(attestation_sha256)
    raw = json.dumps({"attestation_sha256": attestation_sha256, "compiler_environment": compiler_environment},
                     sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "g3-owned-" + hashlib.sha256(raw).hexdigest()


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read(path, expected, *, capture=False, immutable=False):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("G3 owned inputs require absolute ordinary paths without symlinks")
    before = path.stat()
    if (not stat.S_ISREG(before.st_mode) or before.st_size != expected["bytes"]
            or immutable and before.st_mode & 0o222):
        raise ValueError("G3 owned input extent or immutable metadata mode differs")
    digest, blocks = hashlib.sha256(), []
    stream, primary, errors = None, None, []
    try:
        stream = io.FileIO(path, "rb", opener=lambda p, flags: os.open(p, flags | os.O_NOFOLLOW))
        if _stamp(os.fstat(stream.fileno())) != _stamp(before):
            raise ValueError("G3 owned input changed before reading")
        remaining = before.st_size
        while remaining:
            block = stream.read(min(remaining, 1 << 20))
            if not block:
                raise ValueError("G3 owned input was truncated")
            digest.update(block)
            if capture:
                blocks.append(block)
            remaining -= len(block)
        if stream.read(1) or _stamp(os.fstat(stream.fileno())) != _stamp(before):
            raise ValueError("G3 owned input changed while reading")
    except BaseException as error:  # noqa: BLE001 - interruptions must still retire the file
        primary = error
    if stream is not None:
        try:
            stream.close()
        except BaseException as error:  # noqa: BLE001 - retain close failure beside the read failure
            errors.append(error)
    raise_failures(primary, errors)
    if _stamp(path.stat()) != _stamp(before) or digest.hexdigest() != expected["sha256"]:
        raise ValueError("G3 owned input identity differs")
    return b"".join(blocks) if capture else None


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("G3 owned attestation has duplicate JSON keys")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("G3 owned attestation has a nonfinite JSON constant")


def _module_origins(roots):
    for name, module in tuple(sys.modules.items()):
        package = name.partition(".")[0]
        if package in roots:
            origin = getattr(module, "__file__", None)
            if (origin is None or not Path(origin).is_file() or Path(origin).suffix == ".pyc"
                    or not Path(origin).resolve().is_relative_to(Path(roots[package]))):
                raise RuntimeError("G3 preloaded compiler/FFI module has an unverified origin: " + name)


def _tvm_environment(python_directory, library_directories):
    for name, expected in (("TVM_IMPORT_PYTHON_PATH", [python_directory]),
                           ("TVM_LIBRARY_PATH", library_directories)):
        configured = os.environ.get(name)
        if configured is not None:
            values = configured.split(os.pathsep)
            if (not all(v and Path(v).is_absolute() and Path(v) == Path(v).resolve() for v in values)
                    or values != expected):
                raise RuntimeError("G3 compiler path override differs from its owned artifact: " + name)


def read_attestation(path, metadata_bytes, sha256):
    if type(metadata_bytes) is not int or not 0 < metadata_bytes <= MAX_METADATA_BYTES:
        raise ValueError("G3 owned metadata lacks bounded explicit byte credit")
    _sha(sha256)
    raw = _read(path, {"bytes": metadata_bytes, "sha256": sha256}, capture=True, immutable=True)
    return validate(json.loads(raw, object_pairs_hook=_object, parse_constant=_nonfinite))


def hash_bytes(value):
    """Requested bytes read per cold admission pass, excluding metadata/SDK."""
    return value["tilelang"]["wheel"]["bytes"] + sum(
        record["bytes"] for key in ("tilelang", "protected_ffi")
        for record in value[key]["installed_members"].values())


def publish(path, value):
    """Normalize reviewed producer DATA into a new immutable selection file.

    This validates structure only. Successful build/audit/import/retirement
    receipts and external selection remain the producer/operator's authority.
    """
    raw = canonical(value) + b"\n"
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("G3 normalized attestation exceeds metadata credit")
    stream, primary, errors = None, None, []
    try:
        stream = Path(path).open("xb")  # noqa: SIM115 - preserve primary identity if close also fails
        if stream.write(raw) != len(raw):
            raise OSError("G3 attestation publication was incomplete")
    except BaseException as error:  # noqa: BLE001 - report publication and interruption failures after close
        primary = error
    if stream is not None:
        try:
            stream.close()
        except BaseException as error:  # noqa: BLE001 - retain failed close beside publication failure
            errors.append(error)
    raise_failures(primary, errors)
    Path(path).chmod(0o444)
    digest = hashlib.sha256(raw).hexdigest()
    return {"path": str(path), "bytes": len(raw), "sha256": digest,
            "cache_namespace": cache_name(digest, value["compiler_environment"]),
            "required_hash_bytes": hash_bytes(value), "qualification": "structural-normalization-only"}


@dataclass(frozen=True)
class OwnedArtifact:
    attestation: str
    attestation_bytes: int
    attestation_sha256: str
    wheel: str
    hash_byte_credit: int

    def inspect(self, environment, packages):
        """Hash the selected finite closure before any TileLang import."""
        value = read_attestation(self.attestation, self.attestation_bytes, self.attestation_sha256)
        if value["compiler_environment"] != environment:
            raise RuntimeError("G3 owned attestation compiler/header environment differs")
        if (value["tilelang"]["version"] != packages["tilelang"]
                or value["protected_ffi"]["version"] != packages["apache-tvm-ffi"]):
            raise RuntimeError("G3 owned artifact actual package metadata differs")
        required = hash_bytes(value)
        if (type(self.hash_byte_credit) is not int or not 0 <= self.hash_byte_credit < 2**63
                or required > self.hash_byte_credit):
            raise ValueError("G3 owned artifact hashing lacks explicit aggregate byte credit")
        _read(self.wheel, value["tilelang"]["wheel"])
        roots, distributions = {}, {}
        for key, distribution, package in (("tilelang", "tilelang", "tilelang"),
                                            ("protected_ffi", "apache-tvm-ffi", "tvm_ffi")):
            dist = importlib.metadata.distribution(distribution)
            name = dist.metadata.get("Name")
            if type(name) is not str:
                raise RuntimeError("G3 owned installed distribution name is absent")
            name = name.lower().replace("_", "-")
            if dist.version != value[key]["version"] or name != distribution:
                raise RuntimeError("G3 owned installed distribution version differs")
            members = value[key]["installed_members"]
            for name, expected in members.items():
                _read(Path(dist.locate_file(name)).absolute(), expected)
            root = Path(dist.locate_file(package)).absolute()
            expected_files = {name for name in members if name.startswith(package + "/")}
            actual = set()
            for path in root.rglob("*"):
                if path.is_symlink():
                    raise ValueError("G3 owned installed package contains a symlink")
                if not path.is_file() and not path.is_dir():
                    raise ValueError("G3 owned installed package contains a special file")
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    member = package + "/" + str(path.relative_to(root))
                    if member not in expected_files:
                        raise RuntimeError("G3 owned installed package has an extra member")
                    actual.add(member)
            if actual != expected_files:
                raise RuntimeError("G3 owned installed package has extra or missing members")
            roots[package] = root
            distributions[package] = dist
        for package in ("tilelang", "tvm_ffi"):
            spec = importlib.util.find_spec(package)
            if spec is None or spec.origin is None or Path(spec.origin).absolute() != roots[package] / "__init__.py":
                raise RuntimeError("G3 import origin differs from its verified distribution")
        libraries, tvm_init = _compiler_members(value["tilelang"]["installed_members"])
        tl_dist = distributions["tilelang"]
        roots["tvm"] = Path(tl_dist.locate_file(tvm_init)).absolute().parent
        tvm_spec = importlib.util.find_spec("tvm")
        if tvm_spec is not None and (tvm_spec.origin is None
                                    or Path(tvm_spec.origin).resolve() != roots["tvm"] / "__init__.py"):
            raise RuntimeError("G3 discoverable TVM differs from the attested bundled tree")
        _module_origins(roots)
        native = {name: {"path": str(Path(tl_dist.locate_file(member)).absolute()),
                         **value["tilelang"]["installed_members"][member]}
                  for name, member in libraries.items()}
        native["libtvm_ffi.so"] = {"path": str(roots["tvm_ffi"] / "lib/libtvm_ffi.so"), **FFI_CORE}
        optional = {"libtvm_ffi_testing.so": {
            "path": str(roots["tvm_ffi"] / "lib/libtvm_ffi_testing.so"),
            **value["protected_ffi"]["installed_members"]["tvm_ffi/lib/libtvm_ffi_testing.so"]}}
        for item in (*native.values(), *optional.values()):
            item["stat"] = list(_stamp(Path(item["path"]).stat()))
        library_directories = sorted({str(Path(native[name]["path"]).parent) for name in COMPILER_LIBRARIES})
        python_directory = str(roots["tvm"].parent)
        _tvm_environment(python_directory, library_directories)
        namespace = cache_name(self.attestation_sha256, environment)
        configured = os.environ.get("TILELANG_CACHE_DIR")
        cache = Path(configured) if configured else None
        if (cache is None or not cache.is_absolute() or cache.name != namespace
                or cache != cache.resolve()
                or cache.exists() and not cache.is_dir()
                or any(p.is_symlink() for p in (cache, *cache.parents))):
            raise RuntimeError("G3 owned TileLang cache must preselect its exact identity namespace")
        temporary = os.environ.get("TILELANG_TMP_DIR")
        if temporary and (not Path(temporary).is_absolute() or Path(temporary) != Path(temporary).resolve()
                          or not Path(temporary).is_relative_to(cache)
                          or any(p.is_symlink() for p in (Path(temporary), *Path(temporary).parents))):
            raise RuntimeError("G3 owned TileLang temporary cache must remain inside its namespace")
        report = {"schema": SCHEMA, "attestation_sha256": self.attestation_sha256,
                "source_pins": value["source_pins"], "build": value["build"],
                "tilelang_version": value["tilelang"]["version"], "wheel": value["tilelang"]["wheel"],
                "protected_ffi_wheel": value["protected_ffi"]["wheel"],
                "installed_member_count": sum(len(value[k]["installed_members"]) for k in ("tilelang", "protected_ffi")),
                "hashed_bytes": required, "hash_byte_credit": self.hash_byte_credit,
                "cache_directory": str(cache), "cache_namespace": namespace,
                "temporary_directory": str(Path(temporary) if temporary else cache / "tmp"),
                "module_roots": {name: str(path) for name, path in roots.items()},
                "native_libraries": native, "optional_native_libraries": optional,
                "tvm_python_directory": python_directory, "tvm_library_directories": library_directories,
                "qualification": "artifact-identity-only"}
        report["preimport_libraries"] = _library_bindings(report, complete=False)
        return report


def _mapped_libraries():
    paths = set()
    stream, primary, errors = None, None, []
    try:
        stream = Path("/proc/self/maps").open()  # noqa: SIM115 - preserve primary identity across failed close
        for line in stream:
            fields = line.rstrip().split(maxsplit=5)
            if len(fields) == 6 and Path(fields[5].removesuffix(" (deleted)")).name.startswith(("libtvm", "libtilelang")):
                paths.add(fields[5])
    except BaseException as error:  # noqa: BLE001 - interruptions retain the proc-file owner until close
        primary = error
    if stream is not None:
        try:
            stream.close()
        except BaseException as error:  # noqa: BLE001 - report retirement failure alongside primary
            errors.append(error)
    raise_failures(primary, errors)
    return paths


def _library_bindings(record, *, complete):
    paths = _mapped_libraries()
    required = {item["path"] for item in record["native_libraries"].values()}
    allowed = {item["path"]: item for item in (*record["native_libraries"].values(),
                                               *record["optional_native_libraries"].values())}
    if complete and not required <= paths or not paths <= set(allowed):
        raise RuntimeError("G3 must load exactly its verified TileLang/TVM/protected FFI libraries")
    # Installed code is immutable through this dedicated process. Retain the
    # pre-import hash proof and reject any changed file before using its map.
    for path in paths:
        if list(_stamp(Path(path).stat())) != allowed[path]["stat"]:
            raise RuntimeError("G3 compiler library changed after artifact admission")
    return [{key: allowed[path][key] for key in ("path", "bytes", "sha256")} for path in sorted(paths)]


def loaded_compiler(record):
    """Bind actual compiler Python/native owners after imports, before compile."""
    _module_origins(record["module_roots"])
    _tvm_environment(record["tvm_python_directory"], record["tvm_library_directories"])
    return _library_bindings(record, complete=True)

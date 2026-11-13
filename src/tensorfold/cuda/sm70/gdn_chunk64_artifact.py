"""Cold, explicit identity admission for one owned TileLang artifact.

An externally selected manifest digest is an administrator's authority, not a
signature or numerical qualification. Installed packages and the retained wheel
must stay immutable for the dedicated component process. No TileLang/FFI import,
installation, environment mutation or compiler operation belongs in this module.
"""
import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import stat
import sys
import zipfile
from dataclasses import dataclass
from email import policy as email_policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from tensorfold.cleanup import raise_failures

SCHEMA = "tensorfold.g3-owned-tilelang-attestation-v1"
DERIVED_SCHEMA = "tensorfold.g3-owned-tilelang-attestation-v2"
DERIVATION_SCHEMA = "tensorfold.g3-python-derivation-v1"
PROVENANCE_SCHEMA = "tensorfold.g3-python-derivation-provenance-v1"
PROVENANCE_MEMBER = "tilelang/owned-python-derivation.json"
ATTRS_MEMBER = "tilelang/3rdparty/tvm/python/tvm/ir/attrs.py"
ATTRS_BEFORE = (4300, "d5b1e80bb592b87f9870f35612d9400707b62d7f3435538ed23d5a8a856a5e5b")
ATTRS_AFTER = (4617, "d04673f928d70669b809e352b303e6b4229ba2b5080683f226a38dcc33dacdd2")
PYTHON_PINS = (("tilelang", "07827c6ade1125677ce30be043bb9f9ac30cbb2d"),
               ("tvm", "772f65b487db8dd61c590433c16270e50f4a935b"))
NATIVE_PINS = (
    ("tilelang", "708bc2d93ed7e9e4f572a5ba89e85e19fcf32296"),
    ("tvm", "a8d1a6142941ee2bdac67539435274232edca7da"),
    ("tvm_ffi", "3c35034fd1026011736e19a4e0e1ed0f22058c42"),
    ("dlpack", "84d107bf416c6bab9ae68ad285876600d230490d"),
    ("libbacktrace", "793921876c981ce49759114d7bb89bb89b2d3a2d"),
    ("cutlass", "b2dd65dc864e09688245b316ac46c4a6cd07e15c"),
    ("composable_kernel", "b38bb492a1a55b5abb0c345962143c0f9c482cfb"),
)
NATIVE_ARCHIVE = "6bdaac08c8c2d0d7b5f94f50d71224ff9ed475058a43652ac63cbeb1c6327a54"
PACKAGER_GENERATOR = "wheel 0.48.0 (owned Python-only derivation)"
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


def _metadata_root(members):
    return next(PurePosixPath(name).parts[0] for name in members
                if len(PurePosixPath(name).parts) == 2 and name.endswith(".dist-info/METADATA"))


def _normalized_members(members):
    root = _metadata_root(members)
    return {(".dist-info/" + name[len(root) + 1:] if name.startswith(root + "/") else name): record
            for name, record in members.items()}


def _validate_derivation(value):
    item = value["derivation"]
    _fields(item, ("schema", "base_tilelang", "python_source_pins", "changes", "packaging", "embedded_provenance"))
    if (item["schema"] != DERIVATION_SCHEMA or value["source_pins"] != dict(NATIVE_PINS)
            or value["build"]["source_archive_sha256"] != NATIVE_ARCHIVE
            or item["python_source_pins"] != dict(PYTHON_PINS)):
        raise ValueError("G3 Python derivation requires its reviewed native and Python source origins")
    base = item["base_tilelang"]
    _fields(base, ("version", "wheel", "installed_members"))
    if type(base["version"]) is not str or not base["version"] or len(base["version"]) > 512:
        raise ValueError("G3 Python derivation requires the actual native wheel version")
    _extent(base["wheel"])
    _members(base["installed_members"], "tilelang", "tilelang")
    _compiler_members(base["installed_members"])
    before, after = ({"bytes": size, "sha256": sha} for size, sha in (ATTRS_BEFORE, ATTRS_AFTER))
    expected = [{"member": ATTRS_MEMBER, "before": before, "after": after}]
    if type(item["changes"]) is not list or item["changes"] != expected:
        raise ValueError("G3 Python derivation permits only the reviewed DictAttrs change")
    packaging = item["packaging"]
    _fields(packaging, ("proposal_sha256", "producer_sha256", "recipe_identity_sha256", "tool", "image_id"))
    for name in ("proposal_sha256", "producer_sha256", "recipe_identity_sha256"):
        _sha(packaging[name])
    if type(packaging["image_id"]) is not str or not packaging["image_id"].startswith("sha256:"):
        raise ValueError("G3 Python derivation requires its actual derived image")
    _sha(packaging["image_id"][7:])
    _fields(packaging["tool"], ("name", "version", "wheel_sha256"))
    if packaging["tool"]["name"] != "wheel" or packaging["tool"]["version"] != "0.48.0":
        raise ValueError("G3 Python derivation requires the selected public wheel packager")
    _sha(packaging["tool"]["wheel_sha256"])
    provenance = item["embedded_provenance"]
    _fields(provenance, ("member", "bytes", "sha256"))
    if provenance["member"] != PROVENANCE_MEMBER:
        raise ValueError("G3 Python derivation provenance path differs")
    _extent({key: provenance[key] for key in ("bytes", "sha256")})
    old, new = (_normalized_members(package["installed_members"]) for package in (base, value["tilelang"]))
    if (PROVENANCE_MEMBER in old or set(new) != set(old) | {PROVENANCE_MEMBER}
            or old.get(ATTRS_MEMBER) != before or new.get(ATTRS_MEMBER) != after
            or new[PROVENANCE_MEMBER] != {key: provenance[key] for key in ("bytes", "sha256")}
            or base["version"] == value["tilelang"]["version"]):
        raise ValueError("G3 Python derivation inventory does not describe the exact base-to-derived change")
    for name, record in old.items():
        if name not in (ATTRS_MEMBER, ".dist-info/METADATA", ".dist-info/WHEEL") and new[name] != record:
            raise ValueError("G3 Python derivation changed an unaffected or native member: " + name)


def validate(value):
    """Validate either closed identity schema; v1 retains its single-build meaning."""
    names = ("schema", "source_pins", "build", "tilelang", "protected_ffi",
             "compiler_environment", "evidence", "qualification")
    if type(value) is not dict or value.get("schema") not in (SCHEMA, DERIVED_SCHEMA):
        raise ValueError("G3 owned artifact schema differs")
    _fields(value, (*names, "derivation") if value["schema"] == DERIVED_SCHEMA else names)
    if value["qualification"] != "artifact-identity-only":
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
    if value["schema"] == DERIVED_SCHEMA:
        _validate_derivation(value)
    elif PROVENANCE_MEMBER in value["tilelang"]["installed_members"]:
        raise ValueError("G3 Python-derived wheel requires its explicit v2 provenance chain")
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
    """SHA256 input credit per cold pass; metadata/CRC, ZIP IO and SDK are separate."""
    size = value["tilelang"]["wheel"]["bytes"] + sum(
        record["bytes"] for key in ("tilelang", "protected_ffi")
        for record in value[key]["installed_members"].values())
    if value["schema"] == DERIVED_SCHEMA:
        base = value["derivation"]["base_tilelang"]
        size += base["wheel"]["bytes"] + sum(record["bytes"] for package in (base, value["tilelang"])
                                           for record in package["installed_members"].values())
    return size


def _zip_read(archive, name, size, *, capture=False, hashed=True):
    if capture and size > MAX_METADATA_BYTES:
        raise ValueError("G3 wheel metadata exceeds its bounded read envelope")
    stream, primary, errors, blocks = None, None, [], []
    digest = hashlib.sha256()
    try:
        stream = archive.open(name)
        remaining = size
        while remaining:
            block = stream.read(min(remaining, 1 << 20))
            if not block:
                raise ValueError("G3 retained wheel member was truncated")
            if hashed:
                digest.update(block)
            if capture:
                blocks.append(block)
            remaining -= len(block)
        if stream.read(1):
            raise ValueError("G3 retained wheel member grew beyond its declared extent")
    except BaseException as error:  # noqa: BLE001 - close every member before transporting its failure
        primary = error
    if stream is not None:
        try:
            stream.close()
        except BaseException as error:  # noqa: BLE001 - preserve failed CRC/read and retirement
            errors.append(error)
    raise_failures(primary, errors)
    return digest.hexdigest(), b"".join(blocks) if capture else None


def _wheel_record(raw, name, members):
    seen = set()
    for row in csv.reader(io.StringIO(raw.decode("utf-8"), newline="")):
        if len(row) != 3 or row[0] in seen:
            raise ValueError("G3 retained wheel RECORD is malformed or duplicated")
        member, sha, size = row
        seen.add(member)
        if member == name:
            if sha or size:
                raise ValueError("G3 wheel RECORD must leave its own hash/size empty")
        elif member not in members:
            raise ValueError("G3 wheel RECORD names an unowned member")
        else:
            expected = members[member]
            encoded = base64.urlsafe_b64encode(bytes.fromhex(expected["sha256"])).rstrip(b"=").decode()
            if sha != "sha256=" + encoded or size != str(expected["bytes"]):
                raise ValueError("G3 retained wheel RECORD differs from actual member identities")
    if seen != set(members) | {name}:
        raise ValueError("G3 retained wheel RECORD omits a member")


def _wheel_contents(path, members):
    root = _metadata_root(members)
    record = root + "/RECORD"
    captured = {root + "/METADATA", root + "/WHEEL", PROVENANCE_MEMBER}
    result, archive, primary, errors = {}, None, None, []
    try:
        archive = zipfile.ZipFile(path)
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("G3 retained wheel contains duplicate entries")
        directories = {str(parent) + "/" for name in (*members, record)
                       for parent in PurePosixPath(name).parents if str(parent) != "."}
        if any(info.is_dir() and info.filename not in directories for info in archive.infolist()):
            raise ValueError("G3 retained wheel contains an unowned directory")
        ordinary = {info.filename: info for info in archive.infolist() if not info.is_dir()}
        if set(ordinary) != set(members) | {record}:
            raise ValueError("G3 retained wheel inventory differs from the selected installed-member closure")
        for name, expected in members.items():
            if ordinary[name].file_size != expected["bytes"]:
                raise ValueError("G3 retained wheel member size differs")
            sha, raw = _zip_read(archive, name, expected["bytes"], capture=name in captured)
            if sha != expected["sha256"]:
                raise ValueError("G3 retained wheel member digest differs: " + name)
            if raw is not None:
                result[name] = raw
        # RECORD is metadata, bounded separately rather than falsely charged as
        # an installed member. ZipFile owns its CRC algorithm.
        _, raw = _zip_read(archive, record, ordinary[record].file_size, capture=True, hashed=False)
        _wheel_record(raw, record, members)
    except BaseException as error:  # noqa: BLE001 - retain archive validation failure through close
        primary = error
    if archive is not None:
        try:
            archive.close()
        except BaseException as error:  # noqa: BLE001 - report archive retirement as well
            errors.append(error)
    raise_failures(primary, errors)
    return {name[len(root):] if name.startswith(root + "/") else name: raw for name, raw in result.items()}


def _header_change(before, after, name, old_value, new_value):
    parsed = [BytesParser(policy=email_policy.default).parsebytes(raw) for raw in (before, after)]
    for message, expected in zip(parsed, (old_value, new_value)):
        if message.is_multipart() or message.get_all(name) != [expected]:
            raise ValueError("G3 derived wheel metadata header differs: " + name)
    remaining = [[(key.lower(), str(value)) for key, value in message.items() if key.lower() != name.lower()]
                 for message in parsed]
    if remaining[0] != remaining[1] or parsed[0].get_payload() != parsed[1].get_payload():
        raise ValueError("G3 Python derivation changed unrelated wheel metadata")


def _derived_wheels(value, native_wheel, derived_wheel):
    item = value["derivation"]
    base = item["base_tilelang"]
    old = _wheel_contents(native_wheel, base["installed_members"])
    new = _wheel_contents(derived_wheel, value["tilelang"]["installed_members"])
    _header_change(old["/METADATA"], new["/METADATA"], "Version", base["version"], value["tilelang"]["version"])
    old_wheel = BytesParser(policy=email_policy.default).parsebytes(old["/WHEEL"])
    generators = old_wheel.get_all("Generator")
    if generators is None or len(generators) != 1:
        raise ValueError("G3 native wheel requires one Generator header")
    _header_change(old["/WHEEL"], new["/WHEEL"], "Generator", str(generators[0]), PACKAGER_GENERATOR)
    packaging = item["packaging"]
    expected = {
        "schema": PROVENANCE_SCHEMA,
        "native": {"source_pins": value["source_pins"], "build": value["build"],
                   "base_tilelang": {key: base[key] for key in ("version", "wheel")}},
        "python_source_pins": item["python_source_pins"], "changes": item["changes"],
        "packager": {**packaging["tool"], "producer_sha256": packaging["producer_sha256"]},
        "recipe_identity_sha256": packaging["recipe_identity_sha256"],
    }
    actual = json.loads(new[PROVENANCE_MEMBER], object_pairs_hook=_object, parse_constant=_nonfinite)
    if actual != expected:
        raise ValueError("G3 embedded derivation provenance differs from the selected chain")


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
    report = {"path": str(path), "bytes": len(raw), "sha256": digest,
              "cache_namespace": cache_name(digest, value["compiler_environment"]),
              "required_hash_bytes": hash_bytes(value), "qualification": "structural-normalization-only"}
    if value["schema"] == DERIVED_SCHEMA:
        report["archive_record_metadata_limit_bytes"] = 2 * MAX_METADATA_BYTES
    return report


@dataclass(frozen=True)
class OwnedArtifact:
    attestation: str
    attestation_bytes: int
    attestation_sha256: str
    wheel: str
    hash_byte_credit: int
    native_base_wheel: str | None = None

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
        derived = value["schema"] == DERIVED_SCHEMA
        if derived != (self.native_base_wheel is not None):
            raise ValueError("G3 retained native base wheel is required only for the closed derivation schema")
        _read(self.wheel, value["tilelang"]["wheel"])
        if derived:
            _read(self.native_base_wheel, value["derivation"]["base_tilelang"]["wheel"])
            _derived_wheels(value, self.native_base_wheel, self.wheel)
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
        report = {"schema": value["schema"], "attestation_sha256": self.attestation_sha256,
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
        if derived:
            report["derivation"] = {key: value["derivation"][key] for key in (
                "schema", "python_source_pins", "changes", "packaging", "embedded_provenance")}
            report["derivation"]["base_tilelang"] = {
                key: value["derivation"]["base_tilelang"][key] for key in ("version", "wheel")}
            report["archive_record_metadata_limit_bytes"] = 2 * MAX_METADATA_BYTES
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

"""Controlled identity fixtures only; no real owned wheel, imports or GPU proof."""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda.sm70 import gdn_chunk64 as g3
from tensorfold.cuda.sm70 import gdn_chunk64_artifact as artifact


def identity(data):
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def installed(tmp_path, monkeypatch):
    site = tmp_path / "site"
    files = {
        "tilelang/__init__.py": b"# controlled owned package\n",
        "tilelang/compiler.py": b"# controlled compiler input\n",
        "tilelang/lib/libtilelang.so": b"controlled-tilelang-not-an-ELF",
        "tilelang/lib/libtvm_runtime.so": b"controlled-tvm-runtime-not-an-ELF",
        "tilelang/lib/libtvm_compiler.so": b"controlled-tvm-compiler-not-an-ELF",
        "tilelang/3rdparty/tvm/python/tvm/__init__.py": b"# controlled bundled TVM\n",
        "tilelang/3rdparty/tvm/python/tvm/base.py": b"# controlled bundled TVM loader\n",
        "tilelang-0.1.10+fixture.dist-info/METADATA": b"Name: tilelang\nVersion: 0.1.10+fixture\n",
        "tvm_ffi/__init__.py": b"# controlled protected ffi\n",
        "tvm_ffi/lib/libtvm_ffi.so": b"controlled-not-an-ELF",
        "tvm_ffi/lib/libtvm_ffi_testing.so": b"controlled-testing-not-an-ELF",
        "apache_tvm_ffi-0.1.16.dist-info/METADATA": b"Name: apache-tvm-ffi\nVersion: 0.1.16\n",
    }
    for name, data in files.items():
        target = site / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    tl = {n: identity(v) for n, v in files.items() if n.startswith("tilelang")}
    ffi = {n: identity(v) for n, v in files.items() if not n.startswith("tilelang")}
    # These in-process fixtures cannot stand in for the actual protected wheel.
    monkeypatch.setattr(artifact, "FFI_CORE", ffi["tvm_ffi/lib/libtvm_ffi.so"])
    monkeypatch.setattr(artifact, "FFI_MEMBERS_SHA256",
                        hashlib.sha256(json.dumps(ffi, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    env = {"cuda_home": "/protected/cuda", "nvcc_sha256": "a" * 64,
           "headers": {"manifest_sha256": "b" * 64, "headers_sha256": "c" * 64,
                       "include_roots": ["/protected/cutlass", "/protected/cccl"], "environment": {}}}
    value = {"schema": artifact.SCHEMA, "source_pins": {n: "1" * 40 for n in artifact.SOURCE_NAMES},
             "build": {"proposal_sha256": "2" * 64, "source_archive_sha256": "3" * 64,
                       "image_id": "sha256:" + "4" * 64},
             "tilelang": {"version": "0.1.10+fixture", "wheel": identity(b"controlled-wheel"), "installed_members": tl},
             "protected_ffi": {"version": "0.1.16",
                               "wheel": {"bytes": artifact.FFI_WHEEL_BYTES, "sha256": artifact.FFI_WHEEL_SHA256},
                               "installed_members": ffi},
             "compiler_environment": env,
             "evidence": {"producer": {"sha256": "5" * 64},
                          **{n: identity(n.encode()) for n in ("audit", "tl_first", "ffi_torch_first")}},
             "qualification": "artifact-identity-only"}
    wheel = tmp_path / "owned.whl"
    wheel.write_bytes(b"controlled-wheel")
    versions = {"tilelang": "0.1.10+fixture", "apache-tvm-ffi": "0.1.16",
                "torch": "2.16.dev-test", "triton": "3.2.0"}
    monkeypatch.setattr(artifact.importlib.metadata, "distribution",
                        lambda name: NS(version=versions[name], metadata={"Name": name},
                                        locate_file=lambda member: site / member))
    monkeypatch.setattr(artifact.importlib.util, "find_spec", lambda name: NS(origin=str(
        site / ("tilelang/3rdparty/tvm/python/tvm" if name == "tvm" else name) / "__init__.py")))
    for name in tuple(sys.modules):
        if name.partition(".")[0] in ("tilelang", "tvm_ffi", "tvm"):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(artifact, "_mapped_libraries", lambda: set())
    monkeypatch.delenv("TVM_IMPORT_PYTHON_PATH", raising=False)
    monkeypatch.delenv("TVM_LIBRARY_PATH", raising=False)
    receipt = artifact.publish(tmp_path / "attestation.json", value)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / receipt["cache_namespace"]))
    monkeypatch.delenv("TILELANG_TMP_DIR", raising=False)
    selected = artifact.OwnedArtifact(receipt["path"], receipt["bytes"], receipt["sha256"],
                                      str(wheel), receipt["required_hash_bytes"])
    return NS(site=site, value=value, env=env, versions=versions, selected=selected, receipt=receipt)


def test_complete_fixture_checks_actual_bytes_and_identity_cache_without_import(installed):
    record = installed.selected.inspect(installed.env, installed.versions)
    assert record["qualification"] == "artifact-identity-only"
    assert record["installed_member_count"] == 12
    assert record["hashed_bytes"] == installed.receipt["required_hash_bytes"]
    assert record["cache_namespace"] == installed.receipt["cache_namespace"]
    assert record["wheel"] == installed.value["tilelang"]["wheel"]


def test_compiler_report_binds_owned_identity_before_remaining_dependencies(installed, monkeypatch):
    monkeypatch.setattr(g3.Compiler, "_nvcc", lambda self: {
        **{k: installed.env[k] for k in ("cuda_home", "nvcc_sha256")}, "nvcc_version": "controlled"})
    monkeypatch.setattr(g3.Compiler, "_headers",
                        staticmethod(lambda: (installed.env["headers"], "/protected/cutlass", "/protected/cccl")))
    monkeypatch.setattr(g3.importlib.metadata, "version", installed.versions.__getitem__)
    monkeypatch.setattr(g3.importlib.metadata, "requires", lambda name: [])
    report = g3.Compiler("/protected/cuda", "a" * 64, installed.selected).inspect()
    assert report["owned_artifact"]["attestation_sha256"] == installed.selected.attestation_sha256
    assert report["packages"]["tilelang"] == "0.1.10+fixture"
    assert report["tilelang_base_requirements"] == []


@pytest.mark.parametrize("failure", ["extra-field", "path-escape", "second-ffi", "ffi-version", "image",
                                   "null-version", "known-unpatched-wheel"])
def test_closed_schema_refuses_unowned_or_ambiguous_inputs(installed, failure):
    value = copy.deepcopy(installed.value)
    if failure == "extra-field":
        value["fallback"] = True
    elif failure == "path-escape":
        value["tilelang"]["installed_members"]["../outside"] = identity(b"x")
    elif failure == "second-ffi":
        value["tilelang"]["installed_members"]["tilelang/lib/libtvm_ffi.so"] = identity(b"x")
    elif failure == "ffi-version":
        value["protected_ffi"]["version"] = "0.1.17"
    elif failure == "image":
        value["build"]["image_id"] = "pending"
    elif failure == "null-version":
        value["tilelang"]["version"] = None
    else:
        value["tilelang"]["wheel"]["sha256"] = artifact.UNPATCHED_TILELANG_WHEEL_SHA256
    with pytest.raises(ValueError):
        artifact.validate(value)


def test_hash_credit_precedes_wheel_and_member_reads(installed, monkeypatch):
    calls = []
    original = artifact._read
    def read(path, *args, **kwargs):
        calls.append(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(artifact, "_read", read)
    selected = artifact.OwnedArtifact(installed.selected.attestation, installed.selected.attestation_bytes,
                                      installed.selected.attestation_sha256, installed.selected.wheel,
                                      installed.selected.hash_byte_credit - 1)
    with pytest.raises(ValueError, match="byte credit"):
        selected.inspect(installed.env, installed.versions)
    assert calls == [installed.selected.attestation]


@pytest.mark.parametrize("failure", ["changed-member", "extra-member", "missing-member", "wrong-version",
                                   "foreign-import", "wrong-cache", "foreign-tmp", "changed-compiler"])
def test_actual_installation_and_cache_fail_closed(installed, monkeypatch, failure):
    expected = (ValueError, RuntimeError)
    if failure == "changed-member":
        (installed.site / "tilelang/compiler.py").write_bytes(b"# replaced compiler input!\n")
    elif failure == "extra-member":
        (installed.site / "tilelang/stale.py").write_bytes(b"# foreign")
    elif failure == "missing-member":
        (installed.site / "tilelang/compiler.py").unlink()
        expected = FileNotFoundError
    elif failure == "wrong-version":
        installed.versions["tilelang"] = "0.1.10+other"
    elif failure == "foreign-import":
        monkeypatch.setattr(artifact.importlib.util, "find_spec", lambda name: NS(origin="/foreign/__init__.py"))
    elif failure == "wrong-cache":
        monkeypatch.setenv("TILELANG_CACHE_DIR", "/cache/shared")
    elif failure == "foreign-tmp":
        monkeypatch.setenv("TILELANG_TMP_DIR", "/cache/foreign-tmp")
    else:
        installed.env = {**installed.env, "nvcc_sha256": "9" * 64}
    with pytest.raises(expected):
        installed.selected.inspect(installed.env, installed.versions)


@pytest.mark.parametrize("name", ["tilelang.foreign", "tvm_ffi.foreign", "tvm.foreign"])
def test_foreign_preloaded_compiler_descendants_refused(installed, monkeypatch, tmp_path, name):
    foreign = tmp_path / "foreign.py"
    foreign.write_bytes(b"# not from either verified package")
    monkeypatch.setitem(sys.modules, name, NS(__file__=str(foreign)))
    with pytest.raises(RuntimeError, match="unverified origin"):
        installed.selected.inspect(installed.env, installed.versions)


@pytest.mark.parametrize("name", ["TVM_IMPORT_PYTHON_PATH", "TVM_LIBRARY_PATH"])
def test_foreign_tvm_path_override_refused_before_import(installed, monkeypatch, name):
    monkeypatch.setenv(name, "/foreign/compiler")
    with pytest.raises(RuntimeError, match="path override"):
        installed.selected.inspect(installed.env, installed.versions)


def test_foreign_existing_native_mapping_refused_before_import(installed, monkeypatch):
    monkeypatch.setattr(artifact, "_mapped_libraries", lambda: {"/foreign/libtvm_runtime.so"})
    with pytest.raises(RuntimeError, match="verified TileLang"):
        installed.selected.inspect(installed.env, installed.versions)


def test_foreign_discoverable_tvm_refused_before_bootstrap(installed, monkeypatch):
    original = artifact.importlib.util.find_spec
    monkeypatch.setattr(artifact.importlib.util, "find_spec",
                        lambda name: NS(origin="/foreign/tvm/__init__.py") if name == "tvm" else original(name))
    with pytest.raises(RuntimeError, match="discoverable TVM"):
        installed.selected.inspect(installed.env, installed.versions)


def test_owned_tvm_paths_and_preloaded_descendants_remain_admissible(installed, monkeypatch):
    tvm_root = installed.site / "tilelang/3rdparty/tvm/python/tvm"
    monkeypatch.setenv("TVM_IMPORT_PYTHON_PATH", str(tvm_root.parent))
    monkeypatch.setenv("TVM_LIBRARY_PATH", str(installed.site / "tilelang/lib"))
    monkeypatch.setitem(sys.modules, "tvm.base", NS(__file__=str(tvm_root / "base.py")))
    monkeypatch.setitem(sys.modules, "tvm_ffi.valid", NS(__file__=str(installed.site / "tvm_ffi/__init__.py")))
    assert installed.selected.inspect(installed.env, installed.versions)["module_roots"]["tvm"] == str(tvm_root)


def test_metadata_read_rejects_writable_and_symlink_inputs(installed, tmp_path):
    selected = installed.selected
    Path(selected.attestation).chmod(0o644)
    with pytest.raises(ValueError, match="immutable"):
        selected.inspect(installed.env, installed.versions)
    Path(selected.attestation).chmod(0o444)
    link = tmp_path / "metadata-link"
    link.symlink_to(selected.attestation)
    with pytest.raises(ValueError, match="symlinks"):
        artifact.read_attestation(link, selected.attestation_bytes, selected.attestation_sha256)


def test_metadata_size_and_duplicate_keys_fail_before_admission(installed, tmp_path):
    with pytest.raises(ValueError, match="byte credit"):
        artifact.read_attestation("/absent", artifact.MAX_METADATA_BYTES + 1, "0" * 64)
    data = b'{"schema":1,"schema":2}'
    path = tmp_path / "duplicates.json"
    path.write_bytes(data)
    path.chmod(0o444)
    with pytest.raises(ValueError, match="duplicate"):
        artifact.read_attestation(path, len(data), hashlib.sha256(data).hexdigest())


def test_reader_retires_once_and_preserves_read_plus_close_failure(tmp_path, monkeypatch):
    path = tmp_path / "input"
    path.write_bytes(b"data")
    descriptor = os.open(path, os.O_RDONLY)
    primary, secondary = OSError("read failed"), OSError("close failed")
    calls = []
    class Reader:
        def fileno(self):
            return descriptor
        def read(self, size):
            raise primary
        def close(self):
            calls.append("close")
            os.close(descriptor)
            raise secondary
    monkeypatch.setattr(artifact.io, "FileIO", lambda *args, **kwargs: Reader())
    with pytest.raises(OSError) as caught:
        artifact._read(path, identity(b"data"))
    assert caught.value is primary and caught.value.__cause__ is secondary
    assert calls == ["close"]


def test_publisher_preserves_write_plus_close_failure(installed, tmp_path, monkeypatch):
    primary, secondary = OSError("write failed"), OSError("close failed")
    calls = []
    class Writer:
        def write(self, raw):
            raise primary
        def close(self):
            calls.append("close")
            raise secondary
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: Writer())
    target = tmp_path / "unpublished"
    with pytest.raises(OSError) as caught:
        artifact.publish(target, installed.value)
    assert caught.value is primary and caught.value.__cause__ is secondary
    assert calls == ["close"] and not target.exists()


def test_loaded_runtime_map_must_match_preimport_protected_identity(installed, monkeypatch):
    record = installed.selected.inspect(installed.env, installed.versions)
    mappings = {item["path"] for item in record["native_libraries"].values()}
    monkeypatch.setattr(artifact, "_mapped_libraries", lambda: mappings)
    assert {item["path"] for item in artifact.loaded_compiler(record)} == mappings
    mappings.add("/foreign/libtvm_ffi.so")
    with pytest.raises(RuntimeError, match="exactly"):
        artifact.loaded_compiler(record)


def test_missing_compiler_library_refused_before_compile(installed, monkeypatch):
    record = installed.selected.inspect(installed.env, installed.versions)
    mappings = {item["path"] for item in record["native_libraries"].values()}
    mappings.remove(record["native_libraries"]["libtvm_compiler.so"]["path"])
    monkeypatch.setattr(artifact, "_mapped_libraries", lambda: mappings)
    with pytest.raises(RuntimeError, match="exactly"):
        artifact.loaded_compiler(record)


def test_mapped_runtime_scan_preserves_primary_if_retirement_fails(monkeypatch):
    primary, secondary = OSError("maps read failed"), OSError("maps close failed")
    class Reader:
        def __iter__(self):
            raise primary
        def close(self):
            raise secondary
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: Reader())
    with pytest.raises(OSError) as caught:
        artifact._mapped_libraries()
    assert caught.value is primary and caught.value.__cause__ is secondary


@pytest.mark.parametrize("version", ["0.1.10", "0.1.10+unknown", "0.1.11", "999.0"])
def test_unattested_versions_never_reach_dependency_or_tilelang_import(monkeypatch, version):
    monkeypatch.setattr(g3.Compiler, "_nvcc", lambda self: {})
    packages = {"tilelang": version, "apache-tvm-ffi": "0.1.16", "torch": "fixture", "triton": "3.2.0"}
    monkeypatch.setattr(g3.importlib.metadata, "version", packages.__getitem__)
    monkeypatch.setattr(g3.importlib.metadata, "requires", lambda name: pytest.fail("dependency stage reached"))
    with pytest.raises(RuntimeError, match="explicit owned artifact"):
        g3.Compiler("/unused", "0" * 64).inspect()


def test_maintained_normalizer_is_shared_structural_only_and_nonoverwriting(installed, tmp_path, capsys):
    from tools import v100_gdn_chunk64_trial as trial
    value = copy.deepcopy(installed.value)
    value["evidence"]["producer"]["sha256"] = hashlib.sha256(Path(trial.__file__).read_bytes()).hexdigest()
    source = artifact.publish(tmp_path / "producer-input.json", value)
    output = tmp_path / "normalized.json"
    argv = ["--input", source["path"], "--input-bytes", str(source["bytes"]),
            "--input-sha256", source["sha256"], "--output", str(output)]
    result = trial.normalize_attestation(argv)
    assert result["qualification"] == "structural-normalization-only"
    assert result["required_hash_bytes"] == installed.receipt["required_hash_bytes"]
    assert json.loads(output.read_text()) == value
    assert not output.stat().st_mode & 0o222
    assert json.loads(capsys.readouterr().out)["sha256"] == result["sha256"]
    with pytest.raises(FileExistsError):
        trial.normalize_attestation(argv)

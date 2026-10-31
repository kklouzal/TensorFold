"""Selected include bytes/flags are build authority, never device qualification."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda import build
from tensorfold.cuda.sm70 import foundation_headers as headers


@pytest.fixture
def packet(tmp_path, monkeypatch):
    roots = {"cccl": str(tmp_path / "cccl"), "cutlass": str(tmp_path / "cutlass")}
    files = [Path(roots["cccl"]) / "cub/cub.cuh", Path(roots["cutlass"]) / "cutlass/cutlass.h"]
    for path in files:
        path.parent.mkdir(parents=True)
        path.write_text("// owned selected header " + path.name)
    manifest = tmp_path / "foundation.json"

    def publish():
        value = {"schema": "tensorfold.v100-source-foundation-install", "version": 2,
                 "runtime_policy": "torch2141-build-host-v1",
                 "include_roots": roots, "artifacts": [
                     {"path": str(path), "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]}
        manifest.write_text(json.dumps(value, sort_keys=True))

    publish()
    monkeypatch.setenv(headers.MANIFEST_ENV, str(manifest))
    monkeypatch.setattr(headers, "_binding", headers._UNSELECTED)
    return NS(manifest=manifest, roots=roots, files=files, publish=publish)


def test_actual_manifest_header_and_ordered_flag_identity_controls_build_cache(packet, monkeypatch):
    calls = []
    monkeypatch.setattr(build, "arch_flags", lambda *args: ["-gencode=arch=compute_70,code=sm_70"])
    monkeypatch.setattr(build, "_toolkit", lambda: [])
    monkeypatch.setattr(build, "_load", lambda module, name, sources, kwargs: calls.append((name, kwargs)) or "module")
    assert build.load("tf_owned", ["owned.cu"], need=(7, 0), extra_cuda_cflags=["-O3"]) == "module"
    first_name, options = calls[-1]
    assert options["extra_include_paths"] == [packet.roots["cccl"], packet.roots["cutlass"]]
    assert first_name.startswith("tf_owned_h") and first_name.endswith("_sm_70")
    build.load("tf_owned", ["owned.cu"], need=(7, 0), extra_cuda_cflags=["-O3"])
    assert calls[-1][0] == first_name
    build.load("tf_owned", ["owned.cu"], need=(7, 0), extra_cuda_cflags=["-O2"])
    assert calls[-1][0] != first_name


@pytest.mark.parametrize("change", ["bytes", "manifest", "extra", "missing", "symlink", "flags", "selection"])
def test_prepared_binding_refuses_header_or_compiler_authority_mutation(packet, monkeypatch, change):
    before = headers.selected_identity()
    assert before["headers_sha256"] and before["manifest_sha256"]
    if change == "bytes":
        packet.files[0].write_text("changed installed header")
    elif change == "manifest":
        packet.manifest.write_text(packet.manifest.read_text() + " ")
    elif change == "extra":
        (Path(packet.roots["cccl"]) / "unlisted.h").write_text("undeclared include")
    elif change == "missing":
        packet.files[0].unlink()
    elif change == "symlink":
        path = packet.files[0]
        path.unlink()
        path.symlink_to(packet.files[1])
    elif change == "flags":
        monkeypatch.setenv("NVCC_PREPEND_FLAGS", "--use_fast_math")
    else:
        monkeypatch.delenv(headers.MANIFEST_ENV)
        monkeypatch.setattr(headers, "DEFAULT_MANIFEST", str(packet.manifest.parent / "absent.json"))
    with pytest.raises((ValueError, FileNotFoundError)):
        headers.selected_identity()


def test_changed_valid_header_receipt_has_a_distinct_fresh_process_cache_identity(packet):
    options = {"extra_cuda_cflags": ["-O3"], "extra_include_paths": list(packet.roots.values())}
    old = headers.SelectedHeaders(packet.manifest).build_identity(options)
    packet.files[0].write_text("a different correct protected revision")
    packet.publish()
    fresh = headers.SelectedHeaders(packet.manifest).build_identity(options)
    assert fresh != old


def test_changed_header_during_build_never_publishes_the_loaded_module(packet, monkeypatch):
    monkeypatch.setattr(build, "arch_flags", lambda *args: ["-gencode=arch=compute_70,code=sm_70"])
    monkeypatch.setattr(build, "_toolkit", lambda: [])

    def compile_and_change(*args):
        packet.files[0].write_text("changed while the compiler held inputs")
        return "must not be published"

    monkeypatch.setattr(build, "_load", compile_and_change)
    with pytest.raises(ValueError, match="changed"):
        build.load("tf_owned", ["owned.cu"], need=(7, 0))


def test_header_mutation_refuses_final_preparation_adoption_before_state_binding(packet):
    from tensorfold.families.qwen4_exp.cuda.owner_preparation import OwnerPreparation
    from tests.test_owner_preparation import specification

    prepared = OwnerPreparation(specification(), role_rows={"decode": 4, "prefill": 4}, manifest_sha256="a" * 64)
    prepared.source_foundation = headers.selected_identity()
    prepared.prepared = True
    packet.files[0].write_text("new installation requires new preparation")
    with pytest.raises(ValueError, match="changed"):
        prepared.adopt(NS(meta={"owner_workspaces": prepared.workspaces}), NS())


@pytest.mark.parametrize("bad", ["missing-root", "nonpath-root", "uncovered", "digest", "bool-version"])
def test_incomplete_or_false_receipt_fails_before_native_build(packet, bad):
    value = json.loads(packet.manifest.read_text())
    if bad == "missing-root":
        del value["include_roots"]["cutlass"]
    elif bad == "nonpath-root":
        value["include_roots"]["cutlass"] = False
    elif bad == "uncovered":
        value["artifacts"].pop()
    elif bad == "digest":
        value["artifacts"][0]["sha256"] = "0" * 64
    else:
        value["version"] = True
    packet.manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        headers.selected_headers()


def test_other_hardware_keeps_existing_builder_contract(packet, monkeypatch):
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", "-I/legacy-override")
    monkeypatch.setattr(build, "arch_flags", lambda *args: ["-gencode=arch=compute_90,code=sm_90"])
    monkeypatch.setattr(build, "_toolkit", lambda: [])
    monkeypatch.setattr(headers, "selected_headers", lambda: pytest.fail("non-SM70 build consulted selected headers"))
    calls = []
    monkeypatch.setattr(build, "_load", lambda module, name, sources, kwargs: calls.append((name, kwargs)) or "module")
    build.load("tf_owned", ["owned.cu"], extra_cuda_cflags=["-O3"])
    assert calls == [("tf_owned_sm_90", {"extra_cuda_cflags": ["-O3", "-gencode=arch=compute_90,code=sm_90"]})]


def test_unselected_reference_image_does_not_acquire_include_policy(tmp_path, monkeypatch):
    monkeypatch.delenv(headers.MANIFEST_ENV, raising=False)
    monkeypatch.setattr(headers, "DEFAULT_MANIFEST", str(tmp_path / "absent"))
    monkeypatch.setattr(headers, "_binding", headers._UNSELECTED)
    monkeypatch.setenv("CPATH", "/legacy/include")
    assert headers.selected_identity() is None


def test_failed_header_read_preserves_primary_and_failed_close():
    primary, secondary = OSError("read failed"), OSError("close failed")
    calls = []

    def consume(stream):
        calls.append("read")
        raise primary

    def close():
        calls.append("close")
        raise secondary

    with pytest.raises(OSError) as caught:
        headers._read(NS(open=lambda mode: NS(close=close)), consume)
    assert caught.value is primary and caught.value.__cause__ is secondary
    assert calls == ["read", "close"]


@pytest.mark.parametrize("key,value", [("NVCC_PREPEND_FLAGS", "-I/override"), ("NVCC_APPEND_FLAGS", "-include /forced.h"),
                                       ("CPATH", "/override"), ("CPATH", ""), ("CPLUS_INCLUDE_PATH", "/override")])
def test_selected_binding_refuses_consumed_unbound_compiler_environment(packet, monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="unbound compiler input: " + key):
        headers.selected_headers()


@pytest.mark.parametrize("options", [{"extra_include_paths": ["/override"]},
                                     {"extra_cflags": ["-include", "/forced.h"]},
                                     {"extra_cuda_cflags": ["--pre-include=/forced.h"]},
                                     {"extra_cuda_cflags": ["-O3 -Xcompiler=-include,/forced.h"]},
                                     {"extra_cuda_cflags": ["--options-file=/options.txt"]},
                                     {"extra_cflags": ["@/options.txt"]}])
def test_selected_builder_refuses_header_or_response_file_injection_before_compile(packet, monkeypatch, options):
    monkeypatch.setattr(build, "arch_flags", lambda *args: ["-gencode=arch=compute_70,code=sm_70"])
    monkeypatch.setattr(build, "_toolkit", lambda: [])
    monkeypatch.setattr(build, "_load", lambda *args: pytest.fail("unbound include input reached compiler"))
    with pytest.raises(ValueError, match="unbound"):
        build.load("tf_owned", ["owned.cu"], need=(7, 0), **options)


def test_cpp_jit_ignored_flag_variables_are_not_blanket_banned(packet, monkeypatch):
    monkeypatch.setenv("CPPFLAGS", "-include /ignored.h")
    monkeypatch.setenv("CXXFLAGS", "-include /ignored.h")
    monkeypatch.setenv("C_INCLUDE_PATH", "/ignored-for-cxx")
    before = headers.selected_identity()
    monkeypatch.setenv("CPPFLAGS", "-include /another-ignored.h")
    monkeypatch.setenv("CXXFLAGS", "-include /another-ignored.h")
    monkeypatch.setenv("C_INCLUDE_PATH", "/another-ignored-for-cxx")
    assert headers.selected_identity() == before


@pytest.mark.parametrize("entry", ["provider", "gguf"])
def test_selected_environment_refuses_before_existing_nvcc_metadata_probes(packet, monkeypatch, entry):
    import torch
    from tensorfold.cuda.sm70 import provider, gguf_ops

    function = provider._native if entry == "provider" else gguf_ops.extension
    function.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_70"])
    monkeypatch.setattr(torch.version, "cuda", "12.9")
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", "--options-file=/unbound-options")
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: pytest.fail("compiler ran before flag validation"))
    try:
        with pytest.raises(ValueError, match="unbound compiler input: NVCC_PREPEND_FLAGS"):
            function()
    finally:
        function.cache_clear()


def test_actual_nvcc_prepend_shadow_is_refused_by_cached_and_prepared_authority(packet, monkeypatch, tmp_path):
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).is_file() or "release 12.9" not in subprocess.check_output([nvcc, "--version"], text=True):
        pytest.skip("requires the pinned CUDA12.9 compiler; no device execution")
    from tensorfold.families.qwen4_exp.cuda.owner_preparation import OwnerPreparation
    from tests.test_owner_preparation import specification

    packet.files[0].write_text("#define TF_SELECTED_CHOICE 1\n")
    packet.publish()
    identity = headers.selected_identity()
    prepared = OwnerPreparation(specification(), role_rows={"decode": 4, "prefill": 4}, manifest_sha256="a" * 64)
    prepared.source_foundation, prepared.prepared = identity, True
    override = tmp_path / "override/cub"
    override.mkdir(parents=True)
    (override / "cub.cuh").write_text("#define TF_SELECTED_CHOICE 2\n")
    source = tmp_path / "shadow.cu"
    source.write_text('#include <cub/cub.cuh>\nstatic_assert(TF_SELECTED_CHOICE == 1, "selected root shadowed");\n')
    injected = "-I" + str(override.parent)
    result = subprocess.run([nvcc, "-std=c++20", "-arch=sm_70", "-I" + packet.roots["cccl"],
                             "-c", str(source), "-o", str(tmp_path / "shadow.o")],
                            env={**os.environ, "NVCC_PREPEND_FLAGS": injected}, capture_output=True, text=True)
    assert result.returncode != 0 and "selected root shadowed" in result.stderr
    print("expected actual nvcc header precedence witness:\n" + result.stderr)
    metadata = subprocess.run([nvcc, "--version"], capture_output=True, text=True,
        env={**os.environ, "NVCC_PREPEND_FLAGS": "--options-file=" + str(tmp_path / "missing-options")})
    assert metadata.returncode != 0 and "Could not open options file" in metadata.stderr
    print("expected compiler metadata-query input witness:\n" + metadata.stderr)
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", injected)
    with pytest.raises(ValueError, match="unbound compiler input: NVCC_PREPEND_FLAGS"):
        headers.selected_identity()
    with pytest.raises(ValueError, match="unbound compiler input: NVCC_PREPEND_FLAGS"):
        prepared.adopt(NS(meta={"owner_workspaces": prepared.workspaces}), NS())


@pytest.mark.parametrize("version,policy", [(1, "torch2141-build-host-v1"), (2, "unknown"), (2, None)])
def test_selected_headers_refuse_legacy_or_unrecognized_runtime_policy(packet, version, policy):
    value = json.loads(packet.manifest.read_text())
    value.update(version=version, runtime_policy=policy)
    packet.manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="protected v2"):
        headers.SelectedHeaders(packet.manifest)

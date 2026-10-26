"""Installed source-foundation boundaries; these are CPU controls, not V100 qualification."""

import copy
import hashlib
import importlib.util
import builtins
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v100_source_install", ROOT / "deploy/v100/source_foundation.py")
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)


def runtime():
    return {"torch": "2.14.1+tfcu129sm70", "distribution": "2.14.1+tfcu129sm70", "cuda": "12.9",
            "arch_flags": "sm_70", "cublas": [12, 9, 2], "cublas_lt": 120902, "cudnn": 91002,
            "framework_nccl": [2, 32, 3], "framework_cudnn": 91002, "nccl_backend_available": True,
            "nccl": 23203, "mapped_libraries": ["/usr/local/cuda-12.9/lib64/libcublas.so.12.9.2.10",
            "/usr/lib/x86_64-linux-gnu/libcudnn.so.9.10.2",
            "/opt/tensorfold-native/nccl-2.32.3/lib/libnccl.so.2.32.3"]}


@pytest.mark.parametrize("field,value", [("torch", "2.14.1+cu126"), ("cuda", "12.6"),
                                          ("arch_flags", "sm_75 sm_80"), ("nccl", 22703),
                                          ("cudnn", 91100), ("cublas", [12, 6, 4])])
def test_old_or_different_native_providers_cannot_qualify_as_source_stack(field, value):
    identity = runtime()
    identity[field] = value
    with pytest.raises(ValueError):
        SOURCE.validate_runtime(identity)


@pytest.mark.parametrize("changed", ["nccl", "wheel-cublas", "missing-cudnn"])
def test_actual_mapped_provider_ownership_is_required(changed):
    identity = runtime()
    if changed == "nccl":
        identity["mapped_libraries"][-1] = "/usr/lib/x86_64-linux-gnu/libnccl.so.2.27.3"
    elif changed == "wheel-cublas":
        identity["mapped_libraries"][0] = "/opt/venv/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.12"
    else:
        identity["mapped_libraries"].pop(1)
    with pytest.raises(ValueError, match="mapped provider escaped"):
        SOURCE.validate_runtime(identity)


def test_source_cpu_consumer_extensions_remain_separate_from_bundled_accelerator_providers():
    identity = runtime()
    identity["mapped_libraries"].append("/opt/venv/lib/python3.12/site-packages/torchvision/_C.abi3.so")
    SOURCE.validate_runtime(identity)
    identity["mapped_libraries"].append("/opt/venv/lib/python3.12/site-packages/torchvision.libs/libcudart.fake.so")
    with pytest.raises(ValueError, match="binary-wheel accelerator"):
        SOURCE.validate_runtime(identity)


@pytest.mark.parametrize("change", [None, "old-producer", "sealed-library"])
def test_audit_policy_revision_preserves_and_checks_original_build_bytes(tmp_path, monkeypatch, change):
    prior = tmp_path / "audit-history"
    prior.mkdir()
    original = prior / "source_foundation.py"
    original.write_bytes(b"original build producer bytes")
    current = tmp_path / "source_foundation.py"
    current.write_bytes(b"new reviewed audit policy bytes")
    library = tmp_path / "libnative.so"
    library.write_bytes(b"unchanged compiled framework artifact")
    build = tmp_path / "source-build.json"
    build.write_bytes(SOURCE.canonical({"producer_inputs": {
        "deploy/v100/source_foundation.py": SOURCE.file_identity(original)}}))
    old_build_bytes = build.read_bytes()
    (prior / "audit-process.json").write_text('{"historical":"actual prior audit retained"}')
    pins = {"immutable": "source"}
    identity = {"torch_path": str(library), "mapped_libraries": [str(library)]}
    manifest = {"schema": "tensorfold.v100-source-foundation-install", "version": 2,
                "runtime_policy": SOURCE.RUNTIME_POLICY,
                "pins_sha256": hashlib.sha256(SOURCE.canonical(pins)).hexdigest(),
                "evidence_class": "source-built-software-identity", "release_qualified": False,
                "include_roots": dict(SOURCE.INCLUDE_ROOTS), "runtime": identity,
                "build_receipt": {"path": str(build)}, "pending_gates": ["compiler-6111901"],
                "artifacts": [{"path": str(current), **SOURCE.file_identity(original)},
                              {"path": str(library), **SOURCE.file_identity(library)},
                              {"path": str(build), **SOURCE.file_identity(build)}]}
    previous = prior / "v100-source-foundation.json"
    previous.write_text(json.dumps(manifest))
    old_manifest_bytes = previous.read_bytes()
    monkeypatch.setattr(SOURCE, "__file__", str(current))
    monkeypatch.setattr(SOURCE, "runtime_identity", lambda: identity)
    monkeypatch.setattr(SOURCE, "validate_runtime", lambda value: None)
    monkeypatch.setattr(SOURCE, "application_nccl_identity", lambda: {"path": str(library), "version": 23203})
    if change == "old-producer":
        original.write_bytes(b"substituted history")
    elif change == "sealed-library":
        library.write_bytes(b"substituted native provider")
    if change:
        with pytest.raises(ValueError, match="original framework-build producer|artifact changed"):
            SOURCE.revise_audit_policy(pins, prior)
    else:
        result = SOURCE.revise_audit_policy(pins, prior)
        revision = result["audit_policy_revision"]
        assert revision["original_build_producer"] == SOURCE.file_identity(original)
        assert revision["current_audit_producer"] == SOURCE.file_identity(current)
        assert result["release_qualified"] is False and result["pending_gates"] == ["compiler-6111901"]
        assert build.read_bytes() == old_build_bytes and previous.read_bytes() == old_manifest_bytes


def test_streamed_artifact_identity_rejects_nonregular_payload(tmp_path):
    payload = tmp_path / "artifact"
    payload.write_bytes(b"native source-built artifact")
    assert SOURCE.file_identity(payload) == {"bytes": 28, "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}
    with pytest.raises((ValueError, IsADirectoryError)):
        SOURCE.file_identity(tmp_path)


def test_streamed_artifact_read_stays_bounded_when_file_grows(tmp_path, monkeypatch):
    payload = tmp_path / "growing"
    payload.write_bytes(b"initial artifact")
    calls = []

    class Growing:
        def __enter__(self):
            self.stream = builtins.open(payload, "rb")
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def read(self, size):
            calls.append(size)
            if len(calls) > 4:
                raise AssertionError("artifact reader followed an indefinitely growing file")
            block = self.stream.read(size)
            with payload.open("ab") as writer:
                writer.write(b"x")
            return block

    monkeypatch.setattr(SOURCE, "open", lambda *args, **kwargs: Growing(), raising=False)
    with pytest.raises(ValueError, match="changed during hashing"):
        SOURCE.file_identity(payload)
    assert calls == [len(b"initial artifact"), 1]


@pytest.mark.parametrize("size,sha", [(True, "a" * 64), (-1, "a" * 64), (1, "short")])
def test_artifact_metadata_rejects_confusable_counts_or_partial_hashes(size, sha):
    with pytest.raises(ValueError, match="nonnegative byte count"):
        SOURCE.expected_identity({"bytes": size, "sha256": sha})


def test_audit_binds_actual_native_bytes_and_never_claims_release_qualification(tmp_path, monkeypatch):
    payload = (tmp_path / "libtorch.so").resolve()
    payload.write_bytes(b"native source-built artifact")
    pins = {"immutable": "selected source pins"}
    identity = {"torch_path": str(payload), "mapped_libraries": [str(payload)], "actual": "source-built"}
    monkeypatch.setattr(SOURCE, "runtime_identity", lambda: copy.deepcopy(identity))
    monkeypatch.setattr(SOURCE, "validate_runtime", lambda value: None)
    manifest = {"schema": "tensorfold.v100-source-foundation-install", "version": 2,
                "runtime_policy": SOURCE.RUNTIME_POLICY,
                "pins_sha256": hashlib.sha256(SOURCE.canonical(pins)).hexdigest(),
                "evidence_class": "source-built-software-identity", "release_qualified": False,
                "include_roots": dict(SOURCE.INCLUDE_ROOTS),
                "runtime": copy.deepcopy(identity), "build_receipt": {"path": str(payload)},
                "artifacts": [{"path": str(payload), **SOURCE.file_identity(payload)}],
                "pending_gates": ["compiler-6111901", "V100-device-graphs-collectives"]}
    result = SOURCE.audit_install(manifest, pins)
    assert result["software_integrity"] == "passed" and result["release_qualified"] is False
    assert "compiler-6111901" in result["pending_gates"]
    identity["actual"] = "different provider"
    with pytest.raises(ValueError, match="actual loaded"):
        SOURCE.audit_install(manifest, pins)
    identity["actual"] = "source-built"
    payload.write_bytes(b"native source-built corrupte")
    with pytest.raises(ValueError, match="artifact changed"):
        SOURCE.audit_install(manifest, pins)


def process_fixture(tmp_path, monkeypatch):
    library = (tmp_path / "libfoundation.so").resolve()
    library.write_bytes(b"sealed source library")
    identity = {"torch_path": str(library), "mapped_libraries": [str(library)], "actual": "source-built"}
    pins = {"immutable": "source"}
    manifest = {"schema": "tensorfold.v100-source-foundation-install", "version": 2,
                "runtime_policy": SOURCE.RUNTIME_POLICY,
                "pins_sha256": hashlib.sha256(SOURCE.canonical(pins)).hexdigest(),
                "evidence_class": "source-built-software-identity", "release_qualified": False,
                "include_roots": dict(SOURCE.INCLUDE_ROOTS), "runtime": copy.deepcopy(identity),
                "build_receipt": {"path": str(library)},
                "artifacts": [{"path": str(library), **SOURCE.file_identity(library)}],
                "pending_gates": ["compiler-6111901"]}
    monkeypatch.setattr(SOURCE, "runtime_identity", lambda: copy.deepcopy(identity))
    monkeypatch.setattr(SOURCE, "validate_runtime", lambda value: None)
    monkeypatch.setattr(SOURCE, "application_nccl_identity", lambda: {"path": str(library), "version": 23203})
    return pins, manifest, identity


def test_process_audit_binds_extra_code_without_weakening_strict_construction_audit(tmp_path, monkeypatch):
    pins, manifest, identity = process_fixture(tmp_path, monkeypatch)
    extra = (tmp_path / "server_kernel.so").resolve()
    extra.write_bytes(b"actual extra JIT module")
    identity["mapped_libraries"].append(str(extra))
    with pytest.raises(ValueError, match="actual loaded"):
        SOURCE.audit_install(manifest, pins)
    result = SOURCE.audit_process(manifest, pins)
    assert result["schema"] == "tensorfold.v100-source-foundation-process-audit"
    assert result["additional_process_libraries"] == [{"path": str(extra), **SOURCE.file_identity(extra)}]
    assert result["release_qualified"] is False and result["pending_gates"] == ["compiler-6111901"]
    extra.write_bytes(b"different extra JIT code")
    assert SOURCE.audit_process(manifest, pins)["runtime_sha256"] != result["runtime_sha256"]


def test_process_audit_refuses_known_library_family_relocation_even_with_equal_bytes(tmp_path, monkeypatch):
    pins, manifest, identity = process_fixture(tmp_path, monkeypatch)
    duplicate = tmp_path / "untrusted/libfoundation.so"
    duplicate.parent.mkdir()
    duplicate.write_bytes(Path(identity["mapped_libraries"][0]).read_bytes())
    identity["mapped_libraries"].append(str(duplicate.resolve()))
    with pytest.raises(ValueError, match="relocated or substituted"):
        SOURCE.audit_process(manifest, pins)


def test_process_audit_keeps_all_baseline_mappings_and_sealed_bytes_mandatory(tmp_path, monkeypatch):
    pins, manifest, identity = process_fixture(tmp_path, monkeypatch)
    identity["mapped_libraries"].clear()
    with pytest.raises(ValueError, match="lost a known"):
        SOURCE.audit_process(manifest, pins)
    identity["mapped_libraries"] = manifest["runtime"]["mapped_libraries"][:]
    Path(identity["torch_path"]).write_bytes(b"changed sealed source library")
    with pytest.raises(ValueError, match="artifact changed"):
        SOURCE.audit_process(manifest, pins)


def test_generated_image_recipes_use_only_hash_verified_offline_source_artifacts():
    pins = SOURCE.read_json(ROOT / "deploy/v100/source-pins.json")[0]
    generated = SOURCE.dockerfiles(pins)
    for name in ("Dockerfile.source-builder", "Dockerfile"):
        content = generated[name]
        assert content == (ROOT / "deploy/v100" / name).read_text()
        assert "apt-get update" not in content
        assert "sha256sum --strict --check" in content
        assert "--network=none" in content and "source_artifacts" in content
    assert "requirements-foundation-amd64.lock" not in generated["Dockerfile"]
    assert "capture-install" in generated["Dockerfile"] and "source_foundation.py audit" in generated["Dockerfile"]
    policy = generated["Dockerfile.audit-policy"]
    assert policy == (ROOT / "deploy/v100/Dockerfile.audit-policy").read_text()
    assert "ARG AUDIT_BASE_IMAGE" in policy and "FROM ${AUDIT_BASE_IMAGE}" in policy
    assert "audit-history/build-host-v1" in policy and "revise-runtime-policy" in policy
    assert "audit-policy-build-host-v1.json" in policy and "source_foundation.py audit-process" in policy
    assert "--network=none" in policy and "apt-get" not in policy
    assert "application_artifacts" in policy and "sha256sum --strict --check application-wheel.sha256" in policy
    assert "--no-deps /portable-app-wheels/tensorfold-*.whl" in policy and "/torch-*.whl" not in policy


def test_offline_locks_preserve_exact_resolved_artifact_hashes():
    for receipt_name, lock_name, bootstrap in (
            ("source-build-tools-amd64.json", "requirements-source-build-offline-amd64.lock", False),
            ("source-build-tools-amd64.json", "requirements-source-bootstrap-offline-amd64.lock", True),
            ("source-runtime-amd64.json", "requirements-source-runtime-offline-amd64.lock", False)):
        receipt = SOURCE.read_json(ROOT / "deploy/v100" / receipt_name)[0]
        lock = SOURCE.offline_lock(receipt, bootstrap=bootstrap)
        assert lock == (ROOT / "deploy/v100" / lock_name).read_text()
        assert "https:" not in lock and "--hash=sha256:" in lock


def test_runtime_closure_retains_import_only_triton_without_binary_cuda_providers():
    receipt = SOURCE.read_json(ROOT / "deploy/v100/source-runtime-amd64.json")[0]
    assert receipt["packages"]["triton"]["version"] == "3.8.0"
    assert not {"torch", "torchvision"} & receipt["packages"].keys()
    assert not any(name.startswith(("nvidia-", "cuda-")) for name in receipt["packages"])


@pytest.mark.skipif(shutil.which("dpkg-deb") is None, reason="the selected Ubuntu producer supplies dpkg-deb")
def test_apt_bundle_rejects_extra_packages_and_binds_real_deb_bytes(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for name in ("selected", "extra"):
        package = tmp_path / name
        (package / "DEBIAN").mkdir(parents=True)
        (package / "DEBIAN").chmod(0o755)
        (package / "DEBIAN/control").write_text(
            f"Package: {name}\nVersion: 1.0\nArchitecture: all\nMaintainer: Test <test@example.invalid>\n"
            "Description: owned fixture for the source bundle boundary\n")
        subprocess.run(["dpkg-deb", "--build", str(package), str(tmp_path / (name + ".deb"))], check=True,
                       stdout=subprocess.DEVNULL)
    expected = tmp_path / "packages.txt"
    expected.write_text("selected=1.0\n")
    shutil.copyfile(tmp_path / "selected.deb", bundle / "selected.deb")
    pins = {"base": "immutable-base", "native_packages": []}
    receipt = SOURCE.capture_apt(bundle, pins, expected)
    assert receipt["packages"] == [{"artifact": "selected.deb", "package": "selected", "version": "1.0",
                                    "architecture": "all", **SOURCE.file_identity(bundle / "selected.deb")}]
    shutil.copyfile(tmp_path / "extra.deb", bundle / "extra.deb")
    with pytest.raises(ValueError, match="exactly the installed"):
        SOURCE.capture_apt(bundle, pins, expected)


def test_completed_framework_receipt_refuses_cached_legacy_nccl_configuration(tmp_path):
    cache = tmp_path / "sources/pytorch/build/CMakeCache.txt"
    cache.parent.mkdir(parents=True)
    cache.write_text("CMAKE_CUDA_ARCHITECTURES:STRING=70\nNCCL_VERSION:STRING=2.27.3-1\n")
    with pytest.raises(ValueError, match="different CMake provider"):
        SOURCE.capture_build({}, tmp_path, ROOT)

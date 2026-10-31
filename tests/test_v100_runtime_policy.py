"""Pinned build/host separation and explicit immutable policy-rebind controls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("portable_source_policy", ROOT / "deploy/v100/source_foundation.py")
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)
REVISION = "04cf2557f315a350e7eea7ab064288def65db75a"
NVCC = "-gencode;arch=compute_70,code=sm_70"
HEADER = ('#define AT_CUDNN_ENABLED() 1\n#define NVCC_FLAGS_EXTRA "' + NVCC + '"\n').encode()


def configuration(cpu="AVX2", cuda=False):
    optional = ("  - CUDA Runtime 12.9\n  - NVCC architecture flags: " + NVCC + "\n  - CuDNN 91.0.2\n"
                if cuda else "")
    return ("PyTorch built with:\n  - GCC 13.3\n  - C++ Version: 202002\n"
            + "  - CPU capability usage: " + cpu + "\n" + optional
            + "  - Build settings: CXX_FLAGS=-O3 -pthread, CUDA_FLAGS=compute_70, COMMIT_SHA=" + REVISION + "\n")


@pytest.mark.parametrize("cpu", ["AVX2", "AVX512"])
@pytest.mark.parametrize("cuda", [False, True])
def test_build_identity_preserves_compiler_flags_across_host_observations(cpu, cuda):
    original, _ = SOURCE.torch_configuration_identity(configuration(), HEADER)
    actual, host = SOURCE.torch_configuration_identity(configuration(cpu, cuda), HEADER)
    assert actual == original
    assert host == {"torch_configuration": configuration(cpu, cuda), "cpu_dispatch": cpu,
                    "cuda_configuration_visible": cuda}


@pytest.mark.parametrize("old,new", [("GCC 13.3", "GCC 13.4"), ("-O3", "-Ofast"),
                                      ("CUDA_FLAGS=compute_70", "CUDA_FLAGS=compute_80"),
                                      (REVISION, "a" * 40)])
def test_stable_build_compiler_flags_and_source_differences_are_not_projected_away(old, new):
    expected, _ = SOURCE.torch_configuration_identity(configuration(), HEADER)
    actual, _ = SOURCE.torch_configuration_identity(configuration("AVX512", True).replace(old, new), HEADER)
    assert actual != expected


def test_conditional_nvcc_display_must_match_independently_sealed_header():
    with pytest.raises(ValueError, match="sealed NVCC"):
        SOURCE.torch_configuration_identity(configuration("AVX512", True).replace(NVCC, "compute_80"), HEADER)
    expected, _ = SOURCE.torch_configuration_identity(configuration(), HEADER)
    changed, _ = SOURCE.torch_configuration_identity(configuration(), HEADER.replace(b"compute_70", b"compute_80"))
    assert changed != expected


@pytest.mark.parametrize("change", ["missing-cpu", "duplicate-cpu", "missing-header", "duplicate-header"])
def test_unknown_or_ambiguous_configuration_is_refused(change):
    text, header = configuration(), HEADER
    if change == "missing-cpu":
        text = text.replace("  - CPU capability usage: AVX2\n", "")
    elif change == "duplicate-cpu":
        text += "  - CPU capability usage: AVX512\n"
    elif change == "missing-header":
        header = b"#define OTHER 1\n"
    else:
        header += HEADER
    with pytest.raises(ValueError):
        SOURCE.torch_configuration_identity(text, header)


@pytest.fixture
def rebind(tmp_path, monkeypatch):
    prior = tmp_path / "history"
    prior.mkdir()
    old_producer = prior / "source_foundation.py"
    old_producer.write_bytes(b"original framework-build producer")
    current = tmp_path / "source_foundation.py"
    current.write_bytes(b"new reviewed portable audit producer")
    library, header, build_path = tmp_path / "libtorch.so", tmp_path / "CUDAConfig.h", tmp_path / "source-build.json"
    library.write_bytes(b"unchanged original framework bytes")
    header.write_bytes(HEADER)
    build_path.write_bytes(SOURCE.canonical({"producer_inputs": {
        "deploy/v100/source_foundation.py": SOURCE.file_identity(old_producer)}}))
    pins = {"sources": {"pytorch": {"commit": REVISION}}}
    old_runtime = {"torch_git": REVISION, "torch_path": str(library), "immutable": "original",
                   "mapped_libraries": [str(library)], "host_driver_libraries": [],
                   "torch_config_sha256": hashlib.sha256(configuration().encode()).hexdigest()}
    previous = {"schema": "tensorfold.v100-source-foundation-install", "version": 1,
                "pins_sha256": hashlib.sha256(SOURCE.canonical(pins)).hexdigest(),
                "include_roots": dict(SOURCE.INCLUDE_ROOTS), "release_qualified": False,
                "evidence_class": "source-built-software-identity", "pending_gates": ["compiler-6111901"],
                "runtime": old_runtime, "build_receipt": {"path": str(build_path)},
                "artifacts": [{"path": str(current), **SOURCE.file_identity(old_producer)},
                              *({"path": str(path), **SOURCE.file_identity(path)}
                                for path in (library, header, build_path))]}
    previous_path = prior / "v100-source-foundation.json"
    previous_path.write_bytes(SOURCE.canonical(previous))
    policy = {"schema": "tensorfold.v100-runtime-audit-policy-input", "version": 1,
              "runtime_policy": SOURCE.RUNTIME_POLICY, "mapping_policy": SOURCE.MAPPING_POLICY, "torch_git": REVISION,
              "original_configuration": configuration(),
              "original_configuration_sha256": old_runtime["torch_config_sha256"],
              "original_producer": SOURCE.file_identity(old_producer),
              "original_source_build": SOURCE.file_identity(build_path),
              "cuda_header": {"path": str(header), **SOURCE.file_identity(header)},
              "original_manifests": [{"sha256": SOURCE.file_identity(previous_path)["sha256"]}]}
    policy_path = tmp_path / "runtime-policy.json"
    policy_path.write_bytes(SOURCE.canonical(policy))
    build_sha, host = SOURCE.torch_configuration_identity(configuration("AVX512", True), HEADER)
    actual = {key: value for key, value in old_runtime.items() if key != "torch_config_sha256"}
    actual.update(torch_build_configuration_sha256=build_sha, host_observations=host)
    monkeypatch.setattr(SOURCE, "__file__", str(current))
    monkeypatch.setattr(SOURCE, "runtime_identity", lambda: copy.deepcopy(actual))
    monkeypatch.setattr(SOURCE, "validate_runtime", lambda _: None)
    monkeypatch.setattr(SOURCE, "application_nccl_identity", lambda: {"path": str(library), "version": 23203})
    return pins, prior, policy_path, actual, library, header, build_path, old_producer, previous_path


def test_explicit_rebind_preserves_original_bytes_and_records_current_host(rebind):
    pins, prior, policy, actual, *_rest = rebind
    before = {path: path.read_bytes() for path in (prior / "source_foundation.py", prior / "v100-source-foundation.json")}
    result = SOURCE.revise_runtime_policy(pins, prior, policy)
    assert result["version"] == 2 and result["runtime_policy"] == SOURCE.RUNTIME_POLICY
    assert result["runtime"] == actual and result["runtime"]["host_observations"]["cpu_dispatch"] == "AVX512"
    assert all(path.read_bytes() == raw for path, raw in before.items())
    observed = SOURCE.audit_process(result, pins)
    assert observed["version"] == 1 and observed["release_qualified"] is False
    assert observed["host_observations"] == actual["host_observations"]
    original_runtime = copy.deepcopy(result["runtime"])
    actual["host_observations"] = SOURCE.torch_configuration_identity(configuration(), HEADER)[1]
    changed = SOURCE.audit_process(result, pins)
    assert changed["runtime_sha256"] != observed["runtime_sha256"]
    assert result["runtime"] == original_runtime


@pytest.mark.parametrize("fault", ["original-config", "policy-version", "original-producer", "build-receipt", "header", "library",
                                  "compiler", "cxx-flags", "cuda-flags", "source", "immutable", "manifest"])
def test_rebind_refuses_original_byte_or_stable_identity_changes(rebind, fault):
    pins, prior, policy_path, actual, library, header, build_path, producer, manifest = rebind
    if fault in ("original-config", "policy-version"):
        policy = json.loads(policy_path.read_text())
        if fault == "policy-version":
            policy["version"] = True
        else:
            policy["original_configuration"] += "changed"
        policy_path.write_bytes(SOURCE.canonical(policy))
    elif fault in ("original-producer", "build-receipt", "header", "library", "manifest"):
        path = {"original-producer": producer, "build-receipt": build_path, "header": header,
                "library": library, "manifest": manifest}[fault]
        path.write_bytes(path.read_bytes() + b"changed")
    elif fault in ("compiler", "cxx-flags", "cuda-flags"):
        old, new = {"compiler": ("GCC 13.3", "GCC 13.4"), "cxx-flags": ("-O3", "-Ofast"),
                    "cuda-flags": ("CUDA_FLAGS=compute_70", "CUDA_FLAGS=compute_80")}[fault]
        actual["torch_build_configuration_sha256"] = SOURCE.torch_configuration_identity(
            configuration("AVX512", True).replace(old, new), HEADER)[0]
    elif fault == "source":
        actual["torch_git"] = "a" * 40
    else:
        actual["immutable"] = "changed"
    with pytest.raises(ValueError):
        SOURCE.revise_runtime_policy(pins, prior, policy_path)


def test_runtime_audit_never_silently_accepts_old_manifest(rebind):
    pins, prior, *_ = rebind
    with pytest.raises(ValueError, match="unsupported"):
        SOURCE.audit_install(json.loads((prior / "v100-source-foundation.json").read_text()), pins)


def test_native_shared_anonymous_backing_is_data_not_a_software_artifact():
    import mmap
    mapping = mmap.mmap(-1, 2 << 20, flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE)
    try:
        maps = Path("/proc/self/maps").read_text()
        mapped, driver, data = SOURCE.mapped_runtime_artifacts(maps)
        assert data["mapping_count"] >= 1 and data["mapped_bytes"] >= 2 << 20
        assert data["schema"] == "tensorfold.host-shared-anonymous-mappings"
        assert not any("/dev/zero" in path for path in mapped | driver)
    finally:
        mapping.close()


@pytest.mark.parametrize("permissions,offset,device,inode,name", [
    ("rw-p", "00000000", "00:01", "123", "/dev/zero"),
    ("rwxs", "00000000", "00:01", "123", "/dev/zero"),
    ("r-xs", "00000000", "00:01", "123", "/dev/zero"),
    ("rw-s", "00000001", "00:01", "123", "/dev/zero"),
    ("rw-s", "00000000", "01:01", "123", "/dev/zero"),
    ("rw-s", "00000000", "00:01", "0", "/dev/zero"),
    ("r--p", "00000000", "00:01", "123", "/opt/source/retired-input.json"),
    ("rw-s", "00000000", "00:01", "123", "/opt/source/libsource.so"),
    ("r-xp", "00000000", "00:01", "123", "/usr/lib/libcuda.so.580.178.04"),
    ("r-xp", "00000000", "00:01", "123", "/tmp/retired-executable"),
])
def test_retired_source_provider_executable_and_unknown_mappings_remain_refused(
        permissions, offset, device, inode, name):
    row = f"1000-2000 {permissions} {offset} {device} {inode} {name} (deleted)\n"
    with pytest.raises(ValueError, match="retired mapped artifact"):
        SOURCE.mapped_runtime_artifacts(row)


def test_data_only_observations_do_not_stale_software_authority(rebind):
    pins, prior, policy, actual, *_rest = rebind
    result = SOURCE.revise_runtime_policy(pins, prior, policy)
    original = SOURCE.audit_process(result, pins)
    actual["host_data_observations"] = {"schema": "tensorfold.host-shared-anonymous-mappings", "version": 1,
                                        "backing": "/dev/zero", "mapping_count": 40, "mapped_bytes": 40 * (2 << 20)}
    observed = SOURCE.audit_process(result, pins)
    assert observed == original
    diagnostic = {}
    SOURCE.audit_install(result, pins, process_mode=True, host_data_observations=diagnostic)
    assert diagnostic == actual["host_data_observations"]
    assert "host_data_observations" not in result["runtime"]

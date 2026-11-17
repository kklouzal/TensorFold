"""Pinned build/host separation and runtime-audit policy controls."""
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
REVISION = "05adb434bd30d298eb408a72420b0496ee05f8e3"
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
def installed(tmp_path, monkeypatch):
    """A captured version-2 manifest whose runtime was built on AVX2 and now runs on an AVX512 CUDA host."""
    current = tmp_path / "source_foundation.py"
    current.write_bytes(b"current audit producer")
    library, header, build_path = tmp_path / "libtorch.so", tmp_path / "CUDAConfig.h", tmp_path / "source-build.json"
    library.write_bytes(b"installed framework bytes")
    header.write_bytes(HEADER)
    build_path.write_bytes(SOURCE.canonical({"producer_inputs": {
        "deploy/v100/source_foundation.py": SOURCE.file_identity(current)}}))
    pins = {"sources": {"pytorch": {"commit": REVISION}}}
    build_sha, host = SOURCE.torch_configuration_identity(configuration("AVX512", True), HEADER)
    actual = {"torch_git": REVISION, "torch_path": str(library), "immutable": "original",
              "mapped_libraries": [str(library)], "host_driver_libraries": [],
              "torch_build_configuration_sha256": build_sha, "host_observations": host}
    manifest = {"schema": "tensorfold.v100-source-foundation-install", "version": 2,
                "runtime_policy": SOURCE.RUNTIME_POLICY, "mapping_policy": SOURCE.MAPPING_POLICY,
                "pins_sha256": hashlib.sha256(SOURCE.canonical(pins)).hexdigest(),
                "include_roots": dict(SOURCE.INCLUDE_ROOTS), "release_qualified": False,
                "evidence_class": "source-built-software-identity", "pending_gates": ["compiler-6111901"],
                "runtime": copy.deepcopy(actual), "build_receipt": {"path": str(build_path), **SOURCE.file_identity(build_path)},
                "artifacts": [{"path": str(path), **SOURCE.file_identity(path)}
                              for path in sorted((current, library, header, build_path))]}
    monkeypatch.setattr(SOURCE, "__file__", str(current))
    monkeypatch.setattr(SOURCE, "verify_pristine_base", lambda expected=None: {})
    monkeypatch.setattr(SOURCE, "runtime_identity", lambda pins: copy.deepcopy(actual))
    monkeypatch.setattr(SOURCE, "validate_runtime", lambda _, pins: None)
    monkeypatch.setattr(SOURCE, "application_nccl_identity", lambda pins: {"path": str(library), "version": 23203})
    return pins, manifest, actual


def test_process_audit_records_current_host_observations(installed):
    pins, manifest, actual = installed
    observed = SOURCE.audit_process(manifest, pins)
    assert observed["version"] == 1 and observed["release_qualified"] is False
    assert observed["host_observations"] == actual["host_observations"]
    assert observed["host_observations"]["cpu_dispatch"] == "AVX512"
    original_runtime = copy.deepcopy(manifest["runtime"])
    actual["host_observations"] = SOURCE.torch_configuration_identity(configuration(), HEADER)[1]
    changed = SOURCE.audit_process(manifest, pins)
    assert changed["runtime_sha256"] != observed["runtime_sha256"]
    assert manifest["runtime"] == original_runtime


def test_runtime_audit_never_silently_accepts_old_manifest(installed):
    pins, manifest, _actual = installed
    with pytest.raises(ValueError, match="unsupported"):
        SOURCE.audit_install({**manifest, "version": 1}, pins)


def test_native_shared_anonymous_backing_is_data_not_a_software_artifact():
    import mmap
    mapping = mmap.mmap(-1, 2 << 20, flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE)
    try:
        maps = Path("/proc/self/maps").read_text()
        mapped, driver, data = SOURCE.mapped_runtime_artifacts(maps)
        assert data["shared_anonymous"]["mapping_count"] >= 1
        assert data["shared_anonymous"]["mapped_bytes"] >= 2 << 20
        assert data["schema"] == "tensorfold.host-shared-data-mappings" and data["version"] == 2
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


def test_data_only_observations_do_not_stale_software_authority(installed):
    pins, result, actual = installed
    original = SOURCE.audit_process(result, pins)
    actual["host_data_observations"] = {"schema": "tensorfold.host-shared-data-mappings", "version": 2,
        "shared_anonymous": {"backing": "/dev/zero", "mapping_count": 40, "mapped_bytes": 40 * (2 << 20)},
        "posix_semaphores": {"backing": "/dev/shm/sem.XXXXXX", "mapping_count": 3, "mapped_bytes": 3 * 4096}}
    observed = SOURCE.audit_process(result, pins)
    assert observed == original
    diagnostic = {}
    SOURCE.audit_install(result, pins, process_mode=True, host_data_observations=diagnostic)
    assert diagnostic == actual["host_data_observations"]
    assert "host_data_observations" not in result["runtime"]


@pytest.fixture
def live_spawn_queue():
    import multiprocessing

    queue = multiprocessing.get_context("spawn").Queue()
    try:
        yield queue
    finally:
        queue.close()
        queue.join_thread()


def test_live_spawn_queue_semaphores_are_data_while_the_queue_remains_owned(live_spawn_queue):
    import os

    maps = Path("/proc/self/maps").read_text()
    rows = [row for row in maps.splitlines() if "/dev/shm/sem." in row and row.endswith(" (deleted)")]
    assert len(rows) >= 3
    mapped, driver, data = SOURCE.mapped_runtime_artifacts(maps)
    semaphore = data["posix_semaphores"]
    assert semaphore["mapping_count"] == len(rows)
    assert semaphore["mapped_bytes"] == len(rows) * os.sysconf("SC_PAGE_SIZE")
    assert not any("/dev/shm/sem." in path for path in mapped | driver)
    # Closing a Queue endpoint does not retire its still-owned semaphore data.
    live_spawn_queue.close()
    live_spawn_queue.join_thread()
    assert SOURCE.mapped_runtime_artifacts(Path("/proc/self/maps").read_text())[2]["posix_semaphores"] == semaphore


def test_valid_semaphore_counter_equal_to_elf_magic_is_not_software():
    import multiprocessing

    value = int.from_bytes(b"\x7fELF", "little")
    semaphore = multiprocessing.get_context("spawn").Semaphore(value)
    assert semaphore.get_value() == value
    data = SOURCE.mapped_runtime_artifacts(Path("/proc/self/maps").read_text())[2]
    assert data["posix_semaphores"]["mapping_count"] >= 1
    assert semaphore.acquire(False)
    assert semaphore.get_value() == value - 1
    semaphore.release()
    assert semaphore.get_value() == value


@pytest.mark.parametrize("fault", ["private", "executable", "readonly", "offset", "device", "inode", "two-pages",
                                  "unaligned", "library-name", "provider-name", "unrelated-name", "long-name"])
def test_semaphore_role_requires_exact_mapping_shape(live_spawn_queue, fault):
    import os

    row = next(row for row in Path("/proc/self/maps").read_text().splitlines() if "/dev/shm/sem." in row)
    fields = row.split(maxsplit=5)
    start, end = (int(x, 16) for x in fields[0].split("-"))
    if fault in ("private", "executable", "readonly"):
        fields[1] = {"private": "rw-p", "executable": "rwxs", "readonly": "r--s"}[fault]
    elif fault == "offset":
        fields[2] = "00000001"
    elif fault == "device":
        fields[3] = "ff:ff"
    elif fault == "inode":
        fields[4] = "0"
    elif fault == "two-pages":
        fields[0] = f"{start:x}-{end + os.sysconf('SC_PAGE_SIZE'):x}"
    elif fault == "unaligned":
        fields[0] = f"{start + 1:x}-{end + 1:x}"
    else:
        fields[5] = {"library-name": "/dev/shm/sem.lib.so (deleted)",
                     "provider-name": "/dev/shm/libcuda.so (deleted)",
                     "unrelated-name": "/tmp/sem.Abc123 (deleted)",
                     "long-name": "/dev/shm/sem.Abc1234 (deleted)"}[fault]
    with pytest.raises(ValueError, match="retired mapped artifact|invalid semaphore-data"):
        SOURCE.mapped_runtime_artifacts(" ".join(fields))


@pytest.mark.parametrize("fault", ["filesystem", "mount-device", "mount-path"])
def test_semaphore_role_requires_actual_tmpfs_provenance(live_spawn_queue, monkeypatch, fault):
    maps = Path("/proc/self/maps").read_text()
    row = next(row for row in maps.splitlines() if "/dev/shm/sem." in row)
    actual = Path.read_text
    def read(path, *args, **kwargs):
        result = actual(path, *args, **kwargs)
        if str(path) == "/proc/self/mountinfo":
            rows = []
            for mount in result.splitlines():
                fields, filesystem = mount.split(" - ", 1)
                parts = fields.split()
                if parts[4] == "/dev/shm":
                    if fault == "filesystem":
                        filesystem = "ext4" + filesystem[len("tmpfs"):]
                    elif fault == "mount-device":
                        parts[2] = "255:255"
                    elif fault == "mount-path":
                        parts[4] = "/other-shm"
                rows.append(" ".join(parts) + " - " + filesystem)
            return "\n".join(rows)
        return result
    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(ValueError, match="invalid semaphore-data"):
        SOURCE.mapped_runtime_artifacts(row)


def test_current_mapping_policy_refuses_the_predecessor_at_runtime(installed):
    pins, result, _actual = installed
    result["mapping_policy"] = "shared-anonymous-data-v1"
    with pytest.raises(ValueError, match="unsupported installed"):
        SOURCE.audit_process(result, pins)


def test_captured_valid_queue_snapshot_remains_data_after_owner_release():
    import multiprocessing

    def capture():
        before = set(Path("/proc/self/maps").read_text().splitlines())
        queue = multiprocessing.get_context("spawn").Queue()
        try:
            snapshot = Path("/proc/self/maps").read_text()
            owned = {row for row in snapshot.splitlines() if row not in before and "/dev/shm/sem." in row}
            return snapshot, owned
        finally:
            queue.close()
            queue.join_thread()
    snapshot, owned = capture()  # Normal lexical retirement, no forced collection.
    assert len(owned) == 3
    assert not owned.intersection(Path("/proc/self/maps").read_text().splitlines())
    data = SOURCE.mapped_runtime_artifacts(snapshot)[2]
    assert data["posix_semaphores"]["mapping_count"] >= len(owned)

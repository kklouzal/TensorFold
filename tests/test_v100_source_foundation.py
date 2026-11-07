"""Source-build input boundaries; no GPU or source-built qualification implied."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v100_source_foundation", ROOT / "deploy/v100/source_foundation.py")
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)


def report():
    names = ("scikit-build-core", "cmake", "ninja", "numpy", "setuptools", "packaging", "pip", "pyyaml", "six", "typing-extensions")
    return {"environment": {"sys_platform": "linux", "platform_machine": "x86_64",
                            "implementation_name": "cpython", "python_version": "3.12"},
            "install": [{"metadata": {"name": name, "version": "1.0"}, "download_info": {
                "url": f"https://files.pythonhosted.org/packages/{name.replace('-', '_')}-1.0-py3-none-any.whl",
                "archive_info": {"hashes": {"sha256": "a" * 64}}}} for name in names]}


def test_source_build_tools_retain_exact_artifact_bytes_and_resolver_provenance():
    raw = report()
    lock, receipt = SOURCE.render_build_report(raw, "b" * 64)
    assert "numpy @ https://files.pythonhosted.org/packages/numpy-1.0-py3-none-any.whl --hash=sha256:" + "a" * 64 in lock
    assert receipt["resolver_report_sha256"] == "b" * 64
    assert receipt["lock_sha256"] == hashlib.sha256(lock.encode()).hexdigest()
    reordered = copy.deepcopy(raw)
    reordered["install"].reverse()
    assert SOURCE.render_build_report(reordered, "b" * 64) == (lock, receipt)


def test_precommit_framework_outputs_cannot_be_exported(tmp_path):
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "framework-bootstrap-active.json").write_text('{"phase":"precommit-build-inputs"}\n')
    with pytest.raises(ValueError, match="bootstrap cannot be exported"):
        SOURCE.capture_build({}, tmp_path, ROOT)


@pytest.mark.parametrize("jobs,final_outcome", [
    (jobs, outcome) for jobs in ("4", "24") for outcome in ("success", "pip-failure", "source-rejection")
] + [(jobs, "success") for jobs in (None, "", "1", "9", "10", "19", "20")])
def test_framework_driver_keeps_native_configuration_and_bootstrap_export_barrier(tmp_path, jobs, final_outcome):
    # Substitute only the external producer/pip boundaries to observe the real
    # shell driver's argument vectors and failures; no SDK build is claimed.
    state, prefix = tmp_path / "state with spaces", tmp_path / "native prefix"
    checkout = state / "sources/pytorch"
    checkout.mkdir(parents=True)
    subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
    (checkout / "README").write_text("independent shell-driver fixture\n")
    subprocess.run(["git", "-C", str(checkout), "add", "README"], check=True)
    subprocess.run(["git", "-C", str(checkout), "-c", "user.name=Fixture", "-c",
                    "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false", "commit", "-m", "Fixture"],
                   check=True, capture_output=True)
    path = prefix / "cccl-3.5.0/include/cub/version.cuh"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("fixture native prerequisite\n")
    for name in ("include/nccl.h", "lib/libnccl.so.2.32.3"):
        path = state / "build/nccl" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture native prerequisite\n")
    python = state / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(f'''#!{sys.executable}
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
if args[0] == "-":
    os.execv({sys.executable!r}, [{sys.executable!r}, *args])
with open(os.environ["TF_TEST_COMMAND_LOG"], "a") as log:
    log.write(json.dumps({{"args": args, "arch": os.environ.get("TORCH_CUDA_ARCH_LIST"),
                         "nccl": os.environ.get("NCCL_LIB_DIR"),
                         "parallel": {{name: os.environ.get(name) for name in
                             ("MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL", "OMP_NUM_THREADS")}}}}) + "\\n")
if args[0].endswith("source_foundation.py"):
    if args[1] == "verify-bootstrap-sources":
        (Path(os.environ["TF_SOURCE_STATE"]) / "receipts/bootstrap-sources.json").write_text(
            '{{"phase":"precommit-build-inputs"}}\\n')
    if args[1] == "capture-sources" and os.environ.get("TF_TEST_FINAL_OUTCOME") == "source-rejection":
        sys.exit(7)
elif args[:2] == ["-m", "pip"]:
    if "wheel" in args and os.environ.get("TF_TEST_FINAL_OUTCOME") == "pip-failure":
        sys.exit(8)
else:
    raise RuntimeError("unexpected shell boundary: " + repr(args))
''')
    python.chmod(0o755)
    log = tmp_path / "commands.jsonl"
    environment = {**os.environ, "TF_SOURCE_STATE": str(state), "TF_SOURCE_PREFIX": str(prefix),
                   "TF_PROJECT_ROOT": str(ROOT), "TF_TEST_COMMAND_LOG": str(log),
                   "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig")}
    if jobs is None:
        environment.pop("MAX_JOBS", None)
    else:
        environment["MAX_JOBS"] = jobs
    expected_parallel = {name: jobs or "4" for name in
                         ("MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL", "OMP_NUM_THREADS")}
    driver = ["bash", str(ROOT / "deploy/v100/source_build.sh")]
    bootstrap = subprocess.run([*driver, "framework-bootstrap"], env=environment, capture_output=True, text=True)
    assert bootstrap.returncode == 0, bootstrap.stdout + bootstrap.stderr
    marker = state / "receipts/framework-bootstrap-active.json"
    assert marker.is_file()
    assert (state / "receipts/framework-bootstrap-environment.json").is_file()
    assert not (state / "receipts/framework-build-environment.json").exists()
    environment["TF_TEST_FINAL_OUTCOME"] = final_outcome
    final = subprocess.run([*driver, "framework"], env=environment, capture_output=True, text=True)
    assert final.returncode == {"success": 0, "source-rejection": 7, "pip-failure": 8}[final_outcome], final.stderr
    assert marker.exists() == (final_outcome != "success")
    receipts = ["framework-bootstrap-environment.json"] + ([] if final_outcome == "source-rejection"
                                                          else ["framework-build-environment.json"])
    for receipt in receipts:
        actual = json.loads((state / "receipts" / receipt).read_text())
        assert {name: actual[name] for name in expected_parallel} == expected_parallel
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    pip = [row for row in commands if row["args"][:2] == ["-m", "pip"]]
    assert "install" in pip[0]["args"] and "-e" in pip[0]["args"]
    assert "--no-build-isolation" in pip[0]["args"] and "--no-deps" in pip[0]["args"]
    assert all(row["arch"] == "7.0" and row["nccl"] == str(prefix / "nccl-2.32.3/lib") for row in pip)
    assert all(row["parallel"] == expected_parallel for row in pip)
    expected = {
        "--config-settings=cmake.args=--fresh;-UNCCL_VERSION_DEFINED",
        "--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=70",
        f"--config-settings=cmake.define.CUB_INCLUDE_DIR={prefix}/cccl-3.5.0/include",
        f"--config-settings=cmake.define.NCCL_LIBRARIES={prefix}/nccl-2.32.3/lib/libnccl.so",
    }
    assert expected <= set(pip[0]["args"])
    if final_outcome == "source-rejection":
        assert len(pip) == 1
        assert commands[-1]["args"][1] == "capture-sources"
    else:
        assert "wheel" in pip[1]["args"]
        bootstrap_settings = [arg for arg in pip[0]["args"] if arg.startswith("--config-settings=")]
        assert bootstrap_settings == [arg for arg in pip[1]["args"] if arg.startswith("--config-settings=")]


@pytest.mark.parametrize("jobs", ["0", "-1", "25", "2147483648", "9" * 1000, "1.5", "04", " 4", "4;false"])
def test_framework_jobs_refuse_outside_supported_policy_before_side_effects(tmp_path, jobs):
    state, prefix = tmp_path / "state", tmp_path / "prefix"
    environment = {**os.environ, "MAX_JOBS": jobs, "TF_SOURCE_STATE": str(state),
                   "TF_SOURCE_PREFIX": str(prefix), "TF_PROJECT_ROOT": str(tmp_path / "missing-project")}
    result = subprocess.run(["bash", str(ROOT / "deploy/v100/source_build.sh"), "framework"],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 2
    assert "MAX_JOBS must be 1..24" in result.stderr
    assert not state.exists() and not prefix.exists()


@pytest.mark.parametrize("provider", ["Torch", "torchvision", "triton", "nvidia_nccl_cu12", "cuda-bindings"])
def test_binary_accelerator_providers_cannot_enter_source_build_tools(provider):
    data = report()
    data["install"][0]["metadata"]["name"] = provider
    with pytest.raises(ValueError, match="accelerator provider"):
        SOURCE.render_build_report(data, "b" * 64)


@pytest.mark.parametrize("field,value", [("sys_platform", "win32"), ("platform_machine", "aarch64"),
                                         ("implementation_name", "pypy"), ("python_version", "3.14")])
def test_source_build_report_must_match_declared_native_target(field, value):
    data = report()
    data["environment"][field] = value
    with pytest.raises(ValueError, match="native Linux AMD64 CPython"):
        SOURCE.render_build_report(data, "b" * 64)


@pytest.mark.parametrize("broken", ["duplicate", "missing", "foreign-url", "wrong-version", "wrong-name", "bad-sha"])
def test_malformed_or_incomplete_source_package_identity_is_rejected(broken):
    data = report()
    first = data["install"][0]
    if broken == "duplicate":
        data["install"].append(copy.deepcopy(first))
    elif broken == "missing":
        data["install"].pop()
    elif broken == "foreign-url":
        first["download_info"]["url"] = "https://unqualified.invalid/scikit_build_core-1.0-py3-none-any.whl"
    elif broken == "wrong-version":
        first["metadata"]["version"] = "2.0"
    elif broken == "wrong-name":
        first["download_info"]["url"] = "https://files.pythonhosted.org/packages/other-1.0-py3-none-any.whl"
    else:
        first["download_info"]["archive_info"]["hashes"]["sha256"] = "not-a-whole-file-hash"
    with pytest.raises(ValueError):
        SOURCE.render_build_report(data, "b" * 64)


def test_native_package_bytes_are_verified_before_installation(tmp_path):
    artifact = tmp_path / "library.deb"
    artifact.write_bytes(b"owned exact package payload")
    pins = {"native_packages": [{"artifact": artifact.name, "bytes": artifact.stat().st_size,
                                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}]}
    SOURCE.verify_native(pins, tmp_path)
    artifact.write_bytes(b"changed exact package payload")
    with pytest.raises(ValueError, match="identity mismatch"):
        SOURCE.verify_native(pins, tmp_path)
    pins["native_packages"][0]["artifact"] = "../library.deb"
    with pytest.raises(ValueError, match="declared local"):
        SOURCE.verify_native(pins, tmp_path)


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'])
def test_metadata_reader_rejects_ambiguous_or_nonfinite_data(tmp_path, raw):
    path = tmp_path / "report.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        SOURCE.read_json(path)


@pytest.mark.skipif(os.name != "posix", reason="the source-build producer targets native Linux")
def test_source_metadata_fifo_is_refused_before_blocking(tmp_path):
    fifo = tmp_path / "metadata"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        SOURCE.read_json(fifo)


def test_recorded_source_stack_is_distinct_from_the_historical_cu126_wheel_recipe():
    pins = json.loads((ROOT / "deploy/v100/source-pins.json").read_text())
    assert pins["targets"] == {"cuda": "12.9.2", "nvcc": "12.9.86", "cublas": "12.9.2.10",
        "pytorch": "2.14.1", "cccl": "3.5.0", "nccl": "2.32.3", "cutlass": "4.8.1",
        "cudnn": "9.10.2.21", "nsight_compute": "2025.2.1.3", "nsight_systems": "2025.3.2.474",
        "compute_sanitizer_package": "12.9.79"}
    assert "ubuntu24.04@sha256:" in pins["base"]
    assert all(len(source["commit"]) == 40 for source in pins["sources"].values())
    assert "candidate" in pins["status"] and "unrun" in pins["status"]

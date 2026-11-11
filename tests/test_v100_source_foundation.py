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
    (jobs, outcome) for jobs in ("4", "24") for outcome in ("success", "pip-failure", "source-rejection", "tool-rejection")
] + [(jobs, "success") for jobs in (None, "", "1", "9", "10", "19", "20", "64")])
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
    if args[1] == "source-configuration":
        print("tf_torch_wheel_version=2.14.1+tfcu129sm70")
        print("tf_nccl_version=2.32.3")
        sys.exit(0)
    if args[1] == "verify-bootstrap-sources":
        (Path(os.environ["TF_SOURCE_STATE"]) / "receipts/bootstrap-sources.json").write_text(
            '{{"phase":"precommit-build-inputs"}}\\n')
    if args[1] == "capture-sources" and os.environ.get("TF_TEST_FINAL_OUTCOME") == "source-rejection":
        sys.exit(7)
    if args[1] == "verify-ninja" and os.environ.get("TF_TEST_FINAL_OUTCOME") == "tool-rejection":
        sys.exit(9)
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
    assert final.returncode == {"success": 0, "source-rejection": 7, "pip-failure": 8, "tool-rejection": 9}[final_outcome], final.stderr
    assert marker.exists() == (final_outcome != "success")
    receipts = ["framework-bootstrap-environment.json"] + ([] if final_outcome in ("source-rejection", "tool-rejection")
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
    assert [row["args"][1] for row in commands[:3]] == ["source-configuration", "verify-pristine-base", "verify-ninja"]
    assert all(any(arg.startswith("--config-settings=cmake.define.CMAKE_MAKE_PROGRAM=") for arg in row["args"])
               for row in pip)
    expected = {
        "--config-settings=cmake.args=--fresh;-UNCCL_VERSION_DEFINED",
        "--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=70",
        f"--config-settings=cmake.define.CUB_INCLUDE_DIR={prefix}/cccl-3.5.0/include",
        f"--config-settings=cmake.define.NCCL_LIBRARIES={prefix}/nccl-2.32.3/lib/libnccl.so",
    }
    assert expected <= set(pip[0]["args"])
    if final_outcome in ("source-rejection", "tool-rejection"):
        assert len(pip) == 1
        assert commands[-1]["args"][1] == ("verify-ninja" if final_outcome == "tool-rejection" else "capture-sources")
    else:
        assert "wheel" in pip[1]["args"]
        bootstrap_settings = [arg for arg in pip[0]["args"] if arg.startswith("--config-settings=")]
        assert bootstrap_settings == [arg for arg in pip[1]["args"] if arg.startswith("--config-settings=")]


@pytest.mark.parametrize("jobs", ["0", "-1", "65", "2147483648", "9" * 1000, "1.5", "04", " 4", "4;false"])
def test_framework_jobs_refuse_outside_supported_policy_before_side_effects(tmp_path, jobs):
    state, prefix = tmp_path / "state", tmp_path / "prefix"
    environment = {**os.environ, "MAX_JOBS": jobs, "TF_SOURCE_STATE": str(state),
                   "TF_SOURCE_PREFIX": str(prefix), "TF_PROJECT_ROOT": str(tmp_path / "missing-project")}
    result = subprocess.run(["bash", str(ROOT / "deploy/v100/source_build.sh"), "framework"],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 2
    assert "MAX_JOBS must be 1..64" in result.stderr
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
        "pytorch": "2.16.0.dev20261111", "cccl": "3.5.0", "nccl": "2.32.6", "cutlass": "4.8.1",
        "cudnn": "9.10.2.21", "nsight_compute": "2025.2.1.3", "nsight_systems": "2025.3.2.474",
        "compute_sanitizer_package": "12.9.79"}
    assert "ubuntu24.04@sha256:" in pins["base"]
    assert all(len(source["commit"]) == 40 for source in pins["sources"].values())
    assert pins["sources"]["pytorch"]["commit"] == "09b29f3a97e07864066a2d574ffe03a852de9463"
    assert pins["sources"]["pytorch"]["effective_commit"] == "05adb434bd30d298eb408a72420b0496ee05f8e3"
    assert "candidate" in pins["status"] and "unrun" in pins["status"]


@pytest.fixture
def selected_ninja(tmp_path, monkeypatch):
    """Actual file/wheel ownership checks; execution is an explicit unit boundary."""
    import zipfile
    from types import SimpleNamespace
    tool = tmp_path / "bin/ninja"
    tool.parent.mkdir()
    payload = b"\x7fELF\x02\x01" + b"\0" * 12 + (62).to_bytes(2, "little") + b"unit-native-payload"
    tool.write_bytes(payload)
    tool.chmod(0o755)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel = wheels / "ninja-1.13.2-py3-none-manylinux2014_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("ninja-1.13.2.data/scripts/ninja", payload)
    selected = {"version": "1.13.2", "url": "https://files.pythonhosted.org/packages/" + wheel.name,
                "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}
    package = SimpleNamespace(version="1.13.2", files=[Path("bin/ninja")], locate_file=lambda _entry: tool)
    monkeypatch.setattr(SOURCE.platform, "machine", lambda: "x86_64")
    original_distribution = SOURCE.importlib.metadata.distribution
    monkeypatch.setattr(SOURCE.importlib.metadata, "distribution",
                        lambda name: package if name == "ninja" else original_distribution(name))
    monkeypatch.setattr(SOURCE.shutil, "which", lambda name: str(tool))
    calls = []
    def version(argv, **kwargs):
        calls.append(argv)
        assert argv == [str(tool), "--version"] and kwargs == {"text": True, "timeout": 10}
        return "1.13.2.git.kitware.jobserver-pipe-1\n"
    monkeypatch.setattr(SOURCE.subprocess, "check_output", version)
    return {"packages": {"ninja": selected}}, wheels, tool, wheel, package, calls


def test_selected_ninja_binds_actual_package_wheel_and_executable(selected_ninja):
    tools, wheels, tool, wheel, package, calls = selected_ninja
    actual = SOURCE.ninja_tool_identity(tools, wheels)
    assert actual["package_version"] == package.version == "1.13.2"
    assert actual["wheel"] == {"path": str(wheel), "bytes": wheel.stat().st_size,
                               "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}
    assert actual["executable"] == {"path": str(tool), **SOURCE.file_identity(tool)}
    assert actual["binary_version"] == "1.13.2.git.kitware.jobserver-pipe-1"
    assert len(calls) == 1


@pytest.mark.parametrize("change", ["old-lock", "old-package", "wheel-hash", "foreign-origin", "binary-bytes", "binary-version", "changed-during-query"])
def test_selected_ninja_refuses_unbound_or_unfixed_tool_before_build(selected_ninja, monkeypatch, change):
    tools, wheels, tool, wheel, package, calls = selected_ninja
    if change == "old-lock":
        tools["packages"]["ninja"]["version"] = "1.13.0"
    elif change == "old-package":
        package.version = "1.13.0"
    elif change == "wheel-hash":
        tools["packages"]["ninja"]["sha256"] = "0" * 64
    elif change == "foreign-origin":
        package.files = []
    elif change == "binary-bytes":
        tool.write_bytes(b"different ELF")
    else:
        def version(*args, **kwargs):
            if change == "changed-during-query":
                tool.write_bytes(tool.read_bytes() + b"changed")
            return "1.13.0" if change == "binary-version" else "1.13.2"
        monkeypatch.setattr(SOURCE.subprocess, "check_output", version)
    with pytest.raises(ValueError, match="Ninja"):
        SOURCE.ninja_tool_identity(tools, wheels)
    assert not calls



def test_native_report_regeneration_changes_only_ninja_and_retains_bootstrap(tmp_path):
    baseline = report()
    changed = copy.deepcopy(baseline)
    ninja = next(row for row in changed["install"] if row["metadata"]["name"] == "ninja")
    ninja["metadata"]["version"] = "1.13.2"
    ninja["download_info"]["url"] = "https://files.pythonhosted.org/packages/ninja-1.13.2-py3-none-manylinux2014_x86_64.whl"
    results = []
    for index, selected in enumerate((baseline, changed)):
        root = tmp_path / str(index)
        root.mkdir()
        source = root / "report.json"
        source.write_text(json.dumps(selected))
        args = [sys.executable, str(ROOT / "deploy/v100/source_foundation.py"), "render-build-tools",
                "--report", str(source), "--lock", str(root / "lock"), "--receipt", str(root / "receipt.json"),
                "--bootstrap-lock", str(root / "bootstrap"), "--constraints", str(root / "constraints")]
        subprocess.run(args, check=True, capture_output=True)
        actual = json.loads((root / "receipt.json").read_text())
        constraints = [line for line in (root / "constraints").read_text().splitlines() if not line.startswith("#")]
        assert constraints == [name + "==" + row["version"] for name, row in sorted(actual["packages"].items())]
        results.append((actual, (root / "bootstrap").read_bytes()))
    before, after = results
    assert before[1] == after[1]
    assert {name: row for name, row in before[0]["packages"].items() if name != "ninja"} == {
        name: row for name, row in after[0]["packages"].items() if name != "ninja"}
    assert after[0]["packages"]["ninja"]["version"] == "1.13.2"


def runtime_report():
    data = report()
    names = {row["metadata"]["name"] for row in data["install"]}
    for name in ("huggingface-hub", "tokenizers", "safetensors", "pytest", "ruff", "xgrammar", "triton"):
        if name not in names:
            version = "3.9.0" if name == "triton" else "1.0"
            data["install"].append({"metadata": {"name": name, "version": version}, "download_info": {
                "url": f"https://files.pythonhosted.org/packages/{name.replace('-', '_')}-{version}-py3-none-any.whl",
                "archive_info": {"hashes": {"sha256": "a" * 64}}}})
    for name, version in (("torch", "2.16.0.dev20261111+tfcu129sm70"), ("tensorfold", "0.6.6")):
        data["install"].append({"metadata": {"name": name, "version": version,
                                             "requires_dist": ["typing-extensions>=1"]},
                                "download_info": {"url": "file:///owned/" + name, "dir_info": {}},
                                "is_direct": True, "requested": True})
    pins = {"targets": {"pytorch": "2.16.0.dev20261111", "cuda": "12.9.2", "nccl": "2.32.6"},
            "sources": {"pytorch": {"commit": "a" * 40, "effective_commit": "b" * 40}}}
    return data, pins


def test_runtime_source_metadata_is_preserved_and_source_packages_are_installed_separately():
    data, pins = runtime_report()
    lock, receipt = SOURCE.render_build_report(data, "c" * 64, runtime=True, pins=pins)
    assert "torch @" not in lock and "tensorfold @" not in lock
    assert "triton @" in lock and receipt["packages"]["triton"]["version"] == "3.9.0"
    assert receipt["version"] == 2
    assert receipt["source_distributions"]["torch"] == {
        "version": "2.16.0.dev20261111+tfcu129sm70", "source_url": "file:///owned/torch",
        "requires_dist": ["typing-extensions>=1"], "requested_extras": [], "source_commit": "b" * 40}
    assert receipt["triton_execution"] == "import-only; no SM70 compiler/backend execution contract"


@pytest.mark.parametrize("change", ["missing-source", "wrong-source-version", "remote-source", "editable-source"])
def test_runtime_report_requires_actual_declared_owned_source_metadata(change):
    data, pins = runtime_report()
    source = data["install"][-2]
    if change == "missing-source":
        data["install"].pop()
    elif change == "wrong-source-version":
        source["metadata"]["version"] = "2.16.0+cu134"
    elif change == "remote-source":
        source["download_info"]["url"] = "https://example.invalid/torch"
    else:
        source["download_info"]["dir_info"] = {"editable": True}
    with pytest.raises(ValueError, match="source"):
        SOURCE.render_build_report(data, "c" * 64, runtime=True, pins=pins)

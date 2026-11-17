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
PINS = json.loads((ROOT / "deploy/v100/source-pins.json").read_text())


def report():
    names = ("scikit-build-core", "cmake", "ninja", "numpy", "setuptools", "packaging", "pip", "six", "typing-extensions",
             "cython", "setuptools-scm", "apache-tvm-ffi", "pybind11", "wheel")
    return {"environment": {"sys_platform": "linux", "platform_machine": "x86_64",
                            "implementation_name": "cpython", "python_version": "3.15"},
            "install": [{"metadata": {"name": name, "version": "1.0"}, "download_info": {
                "url": f"https://files.pythonhosted.org/packages/{name.replace('-', '_')}-1.0-py3-none-any.whl",
                "archive_info": {"hashes": {"sha256": "a" * 64}}}} for name in names]}


def test_source_build_tools_retain_exact_artifact_bytes_and_resolver_provenance():
    raw = report()
    lock, receipt = SOURCE.render_build_report(raw, "b" * 64, pins=PINS)
    assert "numpy @ https://files.pythonhosted.org/packages/numpy-1.0-py3-none-any.whl --hash=sha256:" + "a" * 64 in lock
    assert receipt["resolver_report_sha256"] == "b" * 64
    assert receipt["lock_sha256"] == hashlib.sha256(lock.encode()).hexdigest()
    reordered = copy.deepcopy(raw)
    reordered["install"].reverse()
    assert SOURCE.render_build_report(reordered, "b" * 64, pins=PINS) == (lock, receipt)


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
        print("tf_lint_python_version=3.15.2")
        print("tf_lint_python_build=20261110")
        print("tf_lint_python_sha256=" + "a" * 64)
        print("tf_lint_tools_sha256=" + "b" * 64)
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
        SOURCE.render_build_report(data, "b" * 64, pins=PINS)


@pytest.mark.parametrize("field,value", [("sys_platform", "win32"), ("platform_machine", "aarch64"),
                                         ("implementation_name", "pypy"), ("python_version", "3.14")])
def test_source_build_report_must_match_declared_native_target(field, value):
    data = report()
    data["environment"][field] = value
    with pytest.raises(ValueError, match="native Linux AMD64 CPython"):
        SOURCE.render_build_report(data, "b" * 64, pins=PINS)


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
        SOURCE.render_build_report(data, "b" * 64, pins=PINS)


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
                "--report", str(source), "--pins", str(ROOT / "deploy/v100/source-pins.json"), "--lock", str(root / "lock"), "--receipt", str(root / "receipt.json"),
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
    data["install"] = [row for row in data["install"]
                       if row["metadata"]["name"] not in ("cython", "setuptools-scm", "apache-tvm-ffi", "pybind11", "wheel")]
    for name, version in (("torch", "2.16.0.dev20261111+tfcu129sm70"), ("tensorfold", "0.6.6")):
        data["install"].append({"metadata": {"name": name, "version": version,
                                             "requires_dist": ["typing-extensions>=1"]},
                                "download_info": {"url": "file:///owned/" + name, "dir_info": {}},
                                "is_direct": True, "requested": True})
    for name, version in (("pyyaml", "6.0.3"), ("xgrammar", "0.2.9")):
        data["install"].append({"metadata": {"name": name, "version": version, "requires_dist": []},
                                "download_info": {"url": f"file:///owned/{name}-{version}-cp315-cp315-linux_x86_64.whl",
                                                  "archive_info": {"hashes": {"sha256": "d" * 64}}},
                                "is_direct": True, "requested": True})
    pins = {**copy.deepcopy(PINS), "targets": {"pytorch": "2.16.0.dev20261111", "cuda": "12.9.2", "nccl": "2.32.6"},
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
    assert receipt["triton_execution"] == "SM70 FP32 moe_partial ordered-FMA; protected CUDA12 tools; other kernels unqualified"
    # Owned cp315 builds are resolved from local wheels and verified against the build receipt.
    assert "pyyaml" not in receipt["packages"] and "xgrammar" not in receipt["packages"]
    assert receipt["source_distributions"]["pyyaml"] == {"version": "6.0.3", "owned_build": True,
                                                       "requires_dist": [], "requested_extras": []}


@pytest.mark.parametrize("change", ["official-wheel", "wrong-version", "not-direct"])
def test_runtime_owned_builds_must_come_from_their_declared_local_wheels(change):
    data, pins = runtime_report()
    owned = data["install"][-1]
    if change == "official-wheel":
        owned["download_info"]["url"] = "https://files.pythonhosted.org/packages/xgrammar-0.2.9-py3-none-any.whl"
    elif change == "wrong-version":
        owned["metadata"]["version"] = "0.2.8"
    else:
        owned["is_direct"] = False
    with pytest.raises(ValueError, match="owned runtime builds"):
        SOURCE.render_build_report(data, "c" * 64, runtime=True, pins=pins)


def test_build_tools_cannot_take_an_owned_package_from_an_official_wheel():
    data = report()
    data["install"].append({"metadata": {"name": "pyyaml", "version": "6.0.3"}, "download_info": {
        "url": "https://files.pythonhosted.org/packages/pyyaml-6.0.3-py3-none-any.whl",
        "archive_info": {"hashes": {"sha256": "a" * 64}}}})
    with pytest.raises(ValueError, match="owned"):
        SOURCE.render_build_report(data, "b" * 64, pins=PINS)


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


def lint_python_pins():
    return {"lint_python": {"version": "3.15.2", "build": "20261110", "bytes": 38360323,
        "sha256": "a" * 64,
        "url": "https://github.com/astral-sh/python-build-standalone/releases/download/20261110/"
               "cpython-3.15.2%2B20261110-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"}}


def test_lint_registry_selects_exact_archive_build_without_patching_uv():
    pins = lint_python_pins()
    registry = SOURCE.lint_python_registry(pins)
    assert registry == {"cpython-3.15.2-linux-x86_64-gnu": {
        "name": "cpython", "arch": {"family": "x86_64", "variant": None}, "os": "linux", "libc": "gnu",
        "major": 3, "minor": 15, "patch": 2, "prerelease": "", "url": pins["lint_python"]["url"],
        "sha256": "a" * 64, "variant": None, "build": "20261110"}}
    for field, value in (("url", "https://example.invalid/python.tar.gz"), ("bytes", 0),
                         ("sha256", "not-a-sha"), ("version", "3.15.2rc1")):
        altered = copy.deepcopy(pins)
        altered["lint_python"][field] = value
        with pytest.raises(ValueError):
            SOURCE.lint_python_registry(altered)


def test_lint_capture_requires_selected_managed_binary_version_and_registry(tmp_path, monkeypatch):
    pins = lint_python_pins()
    root = tmp_path / "managed"
    root.mkdir()
    binary = root / "python3.15"
    binary.write_bytes(b"independent managed interpreter fixture")
    registry = tmp_path / "registry.json"
    registry.write_bytes(SOURCE.canonical(SOURCE.lint_python_registry(pins)))
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return '{"implementation":"cpython","version":[3,15,2]}'
    monkeypatch.setattr(SOURCE.subprocess, "check_output", run)
    receipt = SOURCE.capture_lint_python(pins, binary, root, registry)
    assert receipt["interpreter"]["sha256"] == hashlib.sha256(binary.read_bytes()).hexdigest()
    assert commands[0][:3] == [str(binary.resolve()), "-I", "-c"]
    outsider = tmp_path / "outside-python"
    outsider.write_bytes(b"same version from an unselected installation")
    with pytest.raises(ValueError, match="escapes"):
        SOURCE.capture_lint_python(pins, outsider, root, registry)
    assert len(commands) == 1
    monkeypatch.setattr(SOURCE.subprocess, "check_output", lambda *args, **kwargs:
                        '{"implementation":"cpython","version":[3,15,1]}')
    with pytest.raises(ValueError, match="differs"):
        SOURCE.capture_lint_python(pins, binary, root, registry)
    registry.write_text('{}')
    with pytest.raises(ValueError, match="registry differs"):
        SOURCE.capture_lint_python(pins, binary, root, registry)


def test_framework_private_cutlass_version_comes_from_the_selected_header(tmp_path):
    header = tmp_path / "third_party/cutlass/include/cutlass/version.h"
    header.parent.mkdir(parents=True)
    header.write_text("#define CUTLASS_MAJOR 4\n#define CUTLASS_MINOR 8\n#define CUTLASS_PATCH 0\n")
    actual = SOURCE.framework_private_cutlass(tmp_path)
    assert actual == {"version": "4.8.0", "source": "pytorch/third_party/cutlass",
                      "version_header": SOURCE.file_identity(header)}
    header.write_text(header.read_text() + "#define CUTLASS_PATCH 1\n")
    with pytest.raises(ValueError, match="one exact component version"):
        SOURCE.framework_private_cutlass(tmp_path)


def test_isolated_lint_script_dependencies_keep_their_actual_python_lifetime():
    data = {"environment": {"sys_platform": "linux", "platform_machine": "x86_64",
                            "implementation_name": "cpython", "python_version": "3.15"},
            "install": [{"metadata": {"name": "cmakelint", "version": "1.4.1"}, "download_info": {
                "url": "https://files.pythonhosted.org/packages/cmakelint-1.4.1-py3-none-any.whl",
                "archive_info": {"hashes": {"sha256": "a" * 64}}}}]}
    data["install"].append({"metadata": {"name": "lintrunner", "version": "0.12.7"}, "download_info": {
        "url": "https://files.pythonhosted.org/packages/lintrunner-0.12.7-py3-none-any.whl",
        "archive_info": {"hashes": {"sha256": "c" * 64}}}})
    lock, receipt = SOURCE.render_build_report(data, "b" * 64, lint_script=True)
    assert "cmakelint @" in lock and receipt["packages"]["cmakelint"]["version"] == "1.4.1"
    assert receipt["schema"] == "tensorfold.v100-source-lint-script-tools"
    assert receipt["environment"]["python_version"] == "3.15"
    data["environment"]["python_version"] = "3.12"
    with pytest.raises(ValueError, match="native Linux AMD64 CPython3.15"):
        SOURCE.render_build_report(data, "b" * 64, lint_script=True)
    with pytest.raises(ValueError, match="separate interpreter"):
        SOURCE.render_build_report(data, "b" * 64, lint=True, lint_script=True)


@pytest.mark.parametrize("fault", ["", "producer", "columns"])
def test_export_captures_an_explicit_postbuild_package_observation_before_receipt(fault, tmp_path):
    state = tmp_path / "state"
    tools = state / "venv/bin"
    tools.mkdir(parents=True)
    marker = tmp_path / "capture-called"
    python = tools / "python"
    python.write_text(f"#!{sys.executable}\n" + """import os,sys
from pathlib import Path
args=sys.argv[1:]
if args[0] == '-':
    os.execv(sys.executable,[sys.executable,*args])
if args[1] == 'source-configuration':
    print('tf_torch_wheel_version=2.16.0.dev20261111+tfcu129sm70')
    print('tf_nccl_version=2.32.6')
    print('tf_lint_python_version=3.15.2')
    print('tf_lint_python_build=20261110')
    print('tf_lint_python_sha256='+'a'*64)
    print('tf_lint_tools_sha256='+'b'*64)
elif args[1] == 'capture-build':
    Path(os.environ['CAPTURE_MARKER']).write_text('validated export reached capture')
    sys.exit(23)
""")
    python.chmod(0o755)
    query = tools / "dpkg-query"
    query.write_text("#!/bin/sh\nif test \"$OBSERVATION_FAULT\" = columns; then printf 'pkg\\t1\\n'; "
                     "else printf 'pkg\\t1\\tamd64\\n'; fi\n"
                     "if test \"$OBSERVATION_FAULT\" = producer; then exit 42; fi\n")
    query.chmod(0o755)
    environment = {**os.environ, "TF_SOURCE_STATE": str(state), "TF_PROJECT_ROOT": str(ROOT),
                   "CAPTURE_MARKER": str(marker), "OBSERVATION_FAULT": fault, "MAX_JOBS": "4"}
    result = subprocess.run(["bash", str(ROOT / "deploy/v100/source_build.sh"), "export"],
                            env=environment, capture_output=True, text=True)
    observation = state / "receipts/builder-dpkg-observation.json"
    if fault:
        assert result.returncode == (42 if fault == "producer" else 1)
        assert not marker.exists() and not observation.exists()
    else:
        assert result.returncode == 23 and marker.is_file()
        actual = json.loads(observation.read_text())
        assert actual["stage"] == "export-time package database observation; no pre-build claim"
        assert actual["rows"] == 1
        payload = (state / "receipts/builder-dpkg.txt").read_bytes()
        assert actual["inventory"] == {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def runtime_archive(tmp_path, members):
    import io
    import tarfile
    pins = copy.deepcopy(PINS)
    archive = tmp_path / SOURCE.runtime_python(pins)["artifact"]
    with tarfile.open(archive, "w:gz") as bundle:
        for name, payload, mode in members:
            info = tarfile.TarInfo(name)
            if payload is None:
                info.type, info.mode = tarfile.DIRTYPE, 0o755
                bundle.addfile(info)
            elif isinstance(payload, tuple):
                info.type, info.linkname = tarfile.SYMTYPE, payload[0]
                bundle.addfile(info)
            else:
                info.size, info.mode = len(payload), mode
                bundle.addfile(info, io.BytesIO(payload))
    pins["runtime_python"].update(SOURCE.file_identity(archive))
    return pins, archive


def test_runtime_python_installs_unchanged_and_proves_an_existing_prefix(tmp_path, monkeypatch):
    pins, archive = runtime_archive(tmp_path, [
        ("python", None, 0), ("python/bin/python3.15", b"\x7fELF interpreter", 0o775),
        ("python/bin/python3", ("python3.15",), 0), ("python/lib/python3.15/os.py", b"import sys\n", 0o644)])
    target = tmp_path / "opt/python3.15"
    monkeypatch.setitem(pins["runtime_python"], "prefix", "/opt/python3.15")
    selected = SOURCE.runtime_python(pins)
    monkeypatch.setattr(SOURCE, "runtime_python", lambda _pins: {**selected, "prefix": str(target)})
    receipt = SOURCE.install_runtime_python(pins, archive, target)
    assert receipt["members"] == 4 and (target / "bin/python3").is_symlink()
    assert (target / "bin/python3.15").stat().st_mode & 0o777 == 0o755
    (target / "lib/python3.15/__pycache__").mkdir()
    (target / "lib/python3.15/__pycache__/os.cpython-315.pyc").write_bytes(b"cache")
    SOURCE.install_runtime_python(pins, archive, target)
    (target / "lib/python3.15/os.py").write_text("import changed\n")
    with pytest.raises(ValueError, match="differs from its archive"):
        SOURCE.install_runtime_python(pins, archive, target)
    (target / "lib/python3.15/os.py").write_text("import sys\n")
    (target / "lib/python3.15/extra.py").write_text("planted\n")
    with pytest.raises(ValueError, match="undeclared file"):
        SOURCE.install_runtime_python(pins, archive, target)


def test_runtime_python_pin_is_one_selected_standalone_315_artifact():
    selected = SOURCE.runtime_python(PINS)
    assert selected["executable"] == "/opt/python3.15/bin/python3.15"
    assert selected["artifact"] == "cpython-3.15.2+20261110-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
    for field, value in (("version", "3.14.10"), ("prefix", "/usr/local"), ("url", "https://example.invalid/x.tar.gz")):
        altered = copy.deepcopy(PINS)
        altered["runtime_python"][field] = value
        with pytest.raises(ValueError, match="runtime Python"):
            SOURCE.runtime_python(altered)


def test_owned_wheels_are_verified_against_the_completed_build_receipt(tmp_path):
    names = {"pyyaml": "pyyaml-6.0.3-cp315-cp315-linux_x86_64.whl",
             "xgrammar": "xgrammar-0.2.9-cp315-cp315-linux_x86_64.whl",
             "triton": "triton-3.2.0-cp315-cp315-linux_x86_64.whl"}
    for name, artifact in names.items():
        (tmp_path / artifact).write_bytes(name.encode())
    owned = SOURCE.capture_owned_wheels(PINS, tmp_path)
    SOURCE.verify_owned_wheels(PINS, {"owned_wheels": owned}, tmp_path)
    (tmp_path / names["xgrammar"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from its completed-build receipt"):
        SOURCE.verify_owned_wheels(PINS, {"owned_wheels": owned}, tmp_path)
    (tmp_path / names["triton"]).rename(tmp_path / "triton-3.2.0-cp312-cp312-linux_x86_64.whl")
    with pytest.raises(ValueError, match="CPython 3.15 tag"):
        SOURCE.capture_owned_wheels(PINS, tmp_path)


def test_owned_build_inputs_are_pinned_official_archives():
    builds = SOURCE.owned_python_builds(PINS)
    assert set(builds) == {"pyyaml", "xgrammar", "triton"}
    altered = copy.deepcopy(PINS)
    altered["owned_python_builds"]["pyyaml"]["url"] = "https://example.invalid/pyyaml-6.0.3.tar.gz"
    with pytest.raises(ValueError, match="official source archive"):
        SOURCE.owned_python_builds(altered)

"""Managed installation boundaries use real distributions/files and isolated pip."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import venv
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def guard(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "deploy/gb10"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("tested_foundation_guard", directory / "foundation_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def distribution(site, name, version):
    info = site / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    return info


@pytest.fixture
def sealed(guard, tmp_path, monkeypatch):
    foundation, application = tmp_path / "foundation", tmp_path / "application"
    fs, app = guard.site_directory(foundation), guard.site_directory(application)
    fs.mkdir(parents=True)
    app.mkdir(parents=True)
    distribution(fs, "torch", "1.0")
    module = fs / "torch.py"
    module.write_text("VERSION = '1.0'\n")
    nccl = tmp_path / "libnccl.so.2.32.3"
    nccl.write_bytes(b"native NCCL fixture")
    profile = tmp_path / "runtime-profile.json"
    profile.write_text("{}")
    manifest = {"roles": {"foundation": {"torch": {"version": "1.0"}},
                          "runtime": {"demo": {"version": "1.0", "url": "https://example.invalid/demo.whl", "sha256": "a" * 64}},
                          "verification": {}}}
    state = {"schema_version": 1, "foundation": str(foundation), "application": str(application),
             "manifest": manifest, "sdk_pattern": "^cuda-", "sdk_packages": {"cuda-sdk": "13.4.3"},
             "compiler_packages": {"gcc": "13"}, "native_files": [], "sdk_links": {},
             "nccl_library": str(nccl), "nccl_version": None,
             "runtime_profile_path": str(profile), "cpu_driver": str(nccl),
             "toolchain": {"cuda_home": "/selected", "cupti": {"path": str(nccl)}, "cupti_dir": str(tmp_path)},
             "files": {str(path): guard.identity(path, full=True) for path in guard.tree_files(foundation)}}
    monkeypatch.setattr(guard, "native_packages", lambda pattern: state["compiler_packages"]
                        if pattern == guard.COMPILER_PACKAGES else {"cuda-sdk": "13.4.3"})
    state["files"][str(profile)] = guard.identity(profile, full=True)
    state["state_digest"] = guard.state_digest(state)
    return state


def test_unchanged_sealed_foundation_passes_both_audits(guard, sealed):
    guard.verify(sealed)
    guard.verify(sealed, full=True)


def test_changed_file_is_refused_at_startup(guard, sealed):
    path = guard.site_directory(Path(sealed["foundation"])) / "torch.py"
    path.write_text("changed bytes\n")
    with pytest.raises(ValueError, match=r"file changed|distribution has no name"):
        guard.verify(sealed)


def test_full_audit_detects_same_size_mutation_with_restored_mtime(guard, sealed):
    path = guard.site_directory(Path(sealed["foundation"])) / "torch.py"
    before = path.stat()
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="file changed"):
        guard.verify(sealed, full=True)


@pytest.mark.parametrize("name,version,message", [("torch", "1.0", "shadows"),
                                                   ("nvidia_cublas_cu12", "1", "unapproved accelerator"),
                                                   ("demo", "2.0", "unapproved application")])
def test_app_distribution_cannot_replace_or_introduce_another_provider(guard, sealed, name, version, message):
    distribution(guard.site_directory(Path(sealed["application"])), name, version)
    with pytest.raises(ValueError, match=message):
        guard.verify(sealed)


def test_missing_protected_distribution_refuses(guard, sealed):
    info = guard.site_directory(Path(sealed["foundation"])) / "torch-1.0.dist-info/METADATA"
    info.write_text("Metadata-Version: 2.1\nName: other\nVersion: 1.0\n")
    with pytest.raises(ValueError, match="distribution differs"):
        guard.verify(sealed)


def test_native_or_compiler_package_change_refuses(guard, sealed, monkeypatch):
    monkeypatch.setattr(guard, "native_packages", lambda _: {"cuda-sdk": "13.4.1"})
    with pytest.raises(ValueError, match="native CUDA SDK"):
        guard.verify(sealed)
    monkeypatch.setattr(guard, "native_packages", lambda pattern: {"gcc": "14"}
                        if pattern == guard.COMPILER_PACKAGES else sealed["sdk_packages"])
    with pytest.raises(ValueError, match="compiler dependency"):
        guard.verify(sealed)


def test_unsealed_foundation_payload_is_refused_by_full_audit(guard, sealed):
    (guard.site_directory(Path(sealed["foundation"])) / "hidden.py").write_text("bad = True\n")
    with pytest.raises(ValueError, match="unsealed files"):
        guard.verify(sealed, full=True)


def test_lock_changed_to_conflicting_provider_is_refused_before_pip(guard, sealed, tmp_path, monkeypatch):
    lock = tmp_path / "runtime.lock"
    lock.write_text("nvidia-cublas-cu12 @ https://example.invalid/bad.whl --hash=sha256:" + "a" * 64 + "\n")
    monkeypatch.setattr(guard.subprocess, "run", lambda *_, **__: pytest.fail("must reject before installation"))
    with pytest.raises(ValueError, match="lock differs"):
        guard.install_role(sealed, role="runtime", lock=lock)


def test_failed_install_keeps_primary_and_failed_content_audit(guard, sealed, monkeypatch):
    primary = subprocess.CalledProcessError(42, ["pip"])
    path = guard.site_directory(Path(sealed["foundation"])) / "torch.py"

    def broken(*_, **__):
        path.write_text("mutated protected foundation\n")
        raise primary

    monkeypatch.setattr(guard.subprocess, "run", broken)
    with pytest.raises(BaseExceptionGroup) as caught:
        guard.run_install(sealed, ["pip"])
    assert caught.value.exceptions[0] is primary
    assert "file changed" in str(caught.value.exceptions[1])


def test_failed_install_preserves_unmodified_primary(guard, sealed, monkeypatch):
    primary = subprocess.CalledProcessError(42, ["pip"])

    def broken(*_, **__):
        raise primary

    monkeypatch.setattr(guard.subprocess, "run", broken)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        guard.run_install(sealed, ["pip"])
    assert caught.value is primary


def wheel(path, name, version, *, payload=None):
    info = f"{name}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{info}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        archive.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{info}/RECORD", "")
        archive.writestr(payload or f"{name}/__init__.py", f"__version__ = {version!r}\n")


@pytest.mark.parametrize("name,payload", [("torch", None), ("tensorfold", "nvidia/bad.py"),
                                          ("tensorfold", "../torch.py")])
def test_owned_application_wheel_refuses_protected_or_escaping_payload(guard, sealed, tmp_path, name, payload):
    path = tmp_path / f"{name}-1.0-py3-none-any.whl"
    wheel(path, name, "1.0", payload=payload)
    with pytest.raises(ValueError, match=r"owned TensorFold|protected or escaping"):
        guard.install_application(sealed, path)


def test_real_pip_conflict_cannot_replace_separate_foundation(guard, tmp_path):
    foundation, application = tmp_path / "foundation", tmp_path / "application"
    venv.EnvBuilder(with_pip=True).create(foundation)
    venv.EnvBuilder(with_pip=False).create(application)
    site = guard.site_directory(application)
    fs = guard.site_directory(foundation)
    distribution(fs, "torch", "1.0")
    (fs / "torch.py").write_text("__version__ = '1.0'\n")
    (site / "00_tensorfold_foundation.pth").write_text(f"import sys; sys.path.insert(0, {str(fs)!r})\n")
    constraint = tmp_path / "foundation.txt"
    constraint.write_text("torch===1.0\n")
    (application / "pip.conf").write_text(f"[global]\nconstraint = {constraint}\n")
    replacement = tmp_path / "torch-2.0-py3-none-any.whl"
    wheel(replacement, "torch", "2.0")
    environment = dict(os.environ, PIP_CONFIG_FILE=str(application / "pip.conf"), PIP_NO_INDEX="1",
                       PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PIP_CONSTRAINT", None)
    python = str(application / "bin/python")
    result = subprocess.run([python, "-m", "pip", "install", "--no-deps", str(replacement)],
                            env=environment, capture_output=True, text=True, check=False)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "conflict" in (result.stdout + result.stderr).lower()
    read = subprocess.run([python, "-c", "import torch; print(torch.__version__)"], env=environment,
                          check=True, capture_output=True, text=True)
    assert read.stdout.strip() == "1.0"
    assert not (site / "torch-2.0.dist-info").exists()
    assert (fs / "torch-1.0.dist-info").is_dir()


def test_sdk_selector_preserves_sdk_without_catching_os_libraries(guard):
    pattern = (Path(__file__).resolve().parents[1] / "deploy/gb10/foundation-sdk-pattern.txt").read_text().strip()
    raw = "cuda-nvcc-13-4\t13.4.92\nlibnccl2\t2.32.3\nlibcudnn9-cuda-13\t9.27\nlibcurl4t64\t1\nlibnvme1\t1\n"
    assert guard.sdk_versions(raw, pattern) == {"cuda-nvcc-13-4": "13.4.92", "libnccl2": "2.32.3",
                                               "libcudnn9-cuda-13": "9.27"}


def test_cache_namespace_includes_complete_sealed_state(guard, sealed, monkeypatch):
    import types
    seen = []
    monkeypatch.setitem(sys.modules, "foundation_tools", types.SimpleNamespace(
        toolchain_environment=lambda contract, *, cache_root, verify_content: seen.append((cache_root, verify_content)) or {}))
    sealed["cache_key"] = "a" * 64
    sealed["toolchain"]["cuda_home"] = "/usr/local/cuda"
    sealed["state_digest"] = guard.state_digest(sealed)
    assert guard.environment(sealed)["PYTHONNOUSERSITE"] == "1"
    assert seen == [("/cache/" + "a" * 64, False)]


def test_json_reader_rejects_oversized_document_before_decode(guard, tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"large": "x" * 1024}))
    with pytest.raises(ValueError, match="oversized"):
        guard.read_json(path, 32)


def test_seal_publishes_compact_early_environment_and_failure_is_fatal(guard, tmp_path, monkeypatch):
    import types
    foundation, application = tmp_path / "foundation", tmp_path / "application"
    venv.EnvBuilder(with_pip=False).create(foundation)
    venv.EnvBuilder(with_pip=False).create(application)
    distribution(guard.site_directory(foundation), "torch", "1.0")
    manifest = tmp_path / "foundation-manifest.json"
    manifest.write_text(json.dumps({"roles": {"foundation": {"torch": {"version": "1.0"}},
                                             "runtime": {}, "verification": {}}}))
    (tmp_path / "foundation-constraints.txt").write_text("torch===1.0\n")
    baseline, pattern = tmp_path / "base-sdk.tsv", tmp_path / "pattern"
    baseline.write_text("cuda-sdk\t13.4.3\n")
    (tmp_path / "base-sdk.sha256").write_text("")
    (tmp_path / "base-sdk-links.tsv").write_text("")
    pattern.write_text("^cuda-")
    monkeypatch.setattr(guard, "check_base_sdk", lambda *args: {"cuda-sdk": "13.4.3"})
    monkeypatch.setattr(guard, "native_packages", lambda pattern: {"gcc": "13"}
                        if pattern == guard.COMPILER_PACKAGES else {"cuda-sdk": "13.4.3"})
    cuda_home = tmp_path / "cuda"
    cpu_driver = cuda_home / "compat/libcuda.so.1"
    cpu_driver.parent.mkdir(parents=True)
    cpu_driver.write_bytes(b"controlled compatibility driver; no loading in this publication control")
    monkeypatch.setattr(guard, "native_files", lambda *args: {cpu_driver})
    monkeypatch.setitem(sys.modules, "foundation_runtime", types.SimpleNamespace(capture_runtime=lambda **kwargs: {"schema_version": 1}))
    monkeypatch.setitem(sys.modules, "foundation_tools", types.SimpleNamespace(
        toolchain_environment=lambda contract, *, cache_root, verify_content: {"CC": "/sealed/tensorfold-cc", "TRITON_CACHE_DIR": cache_root}))
    state_path = tmp_path / "state.json"
    tools = {name: {} for name in ("tools", "compilers", "wrappers", "helpers")}
    marker = tmp_path / "tool"
    marker.write_text("selected tool")
    monkeypatch.setattr(guard, "select_nccl", lambda *args: (str(marker), None))
    tools.update(headers=[], libdevice={"path": str(marker)}, cupti={"path": str(marker)},
                 cuda_home=str(cuda_home), cupti_dir=str(marker.parent))
    state = guard.seal(manifest_path=manifest, foundation=foundation, application=application,
                       sdk_baseline=baseline, pattern_path=pattern, state_path=state_path,
                       artifacts=tmp_path, toolchain=tools)
    result = subprocess.run([str(application / "bin/python"), "-c", "import os; print(os.environ['CC']); print(os.environ['TRITON_CACHE_DIR'])"],
                            capture_output=True, text=True, check=True, env=dict(os.environ, TF_NCCL_LIB="", TF_FOUNDATION_RUNTIME_PHASE=""))
    assert result.stdout.splitlines() == ["/sealed/tensorfold-cc", "/cache/" + state["cache_key"]]
    wrong_nccl = tmp_path / "wrong-libnccl.so"
    wrong_nccl.write_text("unselected provider")
    refusal = subprocess.run([str(application / "bin/python"), "-c", "print('must not run')"],
                             env=dict(os.environ, TF_NCCL_LIB=str(wrong_nccl), TF_FOUNDATION_RUNTIME_PHASE=""), capture_output=True, text=True, check=False)
    assert refusal.returncode == 78 and "TF_NCCL_LIB conflicts" in refusal.stderr
    prior_key = state["cache_key"]
    application_wheel = tmp_path / "tensorfold-1.0-py3-none-any.whl"
    wheel(application_wheel, "tensorfold", "1.0")
    monkeypatch.setattr(guard, "run_install", lambda *args: None)  # Pip boundary tested independently; observe actual publication.
    guard.install_application(state, application_wheel)
    assert state["cache_key"] != prior_key
    assert state["foundation_cache_key"] == prior_key
    assert state["application_wheel_sha256"] == guard.digest(application_wheel)
    result = subprocess.run([str(application / "bin/python"), "-c", "import os; print(os.environ['TRITON_CACHE_DIR'])"],
                            capture_output=True, text=True, check=True, env=dict(os.environ, TF_FOUNDATION_RUNTIME_PHASE=""))
    assert result.stdout.strip() == "/cache/" + state["cache_key"]
    (guard.site_directory(application) / "tensorfold-foundation-environment.json").unlink()
    result = subprocess.run([str(application / "bin/python"), "-c", "print('must not run')"], capture_output=True, text=True, check=False)
    assert result.returncode == 78
    assert "must not run" not in result.stdout
    assert "activation failed" in result.stderr


@pytest.mark.parametrize("fault", ["list", "versions"])
def test_sdk_capture_rejects_partial_failed_dpkg_producer(guard, tmp_path, fault):
    source = Path(__file__).resolve().parents[1] / "deploy/gb10"
    commands = tmp_path / "commands"
    commands.mkdir()
    provider = tmp_path / "provider.so"
    provider.write_text("original SDK payload")
    query = commands / "dpkg-query"
    query.write_text(f"#!{sys.executable}\nimport os,sys\nargs=sys.argv[1:]\n"
                     f"if '-L' in args:\n print({str(provider)!r}); sys.exit(42 if os.environ.get('FAULT')=='list' else 0)\n"
                     "print('cuda-sdk\\t13.4.3')\n"
                     "sys.exit(42 if len(args)>2 and os.environ.get('FAULT')=='versions' else 0)\n")
    query.chmod(0o755)
    mark = commands / "apt-mark"
    mark.write_text("#!/bin/sh\nexit 0\n")
    mark.chmod(0o755)
    env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"], FAULT=fault)
    destination = tmp_path / "captured"
    result = subprocess.run(["sh", str(source / "foundation_sdk.sh"), "capture", str(destination),
                             str(source / "foundation-sdk-pattern.txt")], env=env, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert not (destination / "base-sdk.sha256").exists()


def test_state_mutation_cannot_reuse_prior_cache_identity(guard, sealed):
    sealed["cache_key"] = "stale"
    with pytest.raises(ValueError, match="state identity changed"):
        guard.verify(sealed)


def test_install_and_audit_failures_survive_diagnostic_exhaustion(guard, sealed, monkeypatch):
    primary = subprocess.CalledProcessError(42, ["pip"])
    original = RuntimeError("native prior cause")
    primary.__cause__ = original
    path = guard.site_directory(Path(sealed["foundation"])) / "torch.py"

    def run(*_, **__):
        path.write_text("mutated protected foundation\n")
        raise primary

    def exhausted(*_, **__):
        raise MemoryError("group allocation refused")

    monkeypatch.setattr(guard.subprocess, "run", run)
    monkeypatch.setattr(guard, "BaseExceptionGroup", exhausted, raising=False)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        guard.run_install(sealed, ["pip"])
    assert caught.value is primary
    assert primary.__cause__ is original
    frame = caught.value.__traceback__
    statuses = []
    while frame:
        statuses.append(frame.tb_frame.f_locals.get("retained_audit"))
        frame = frame.tb_next
    assert any(isinstance(error, ValueError) and "file changed" in str(error) for error in statuses)


def test_base_directory_alias_is_checked_after_sealing(guard, sealed, tmp_path):
    first, second = tmp_path / "sdk-one", tmp_path / "sdk-two"
    first.mkdir()
    second.mkdir()
    alias = tmp_path / "cuda"
    alias.symlink_to(first, target_is_directory=True)
    sealed["sdk_links"] = {str(alias): str(first)}
    sealed["state_digest"] = guard.state_digest(sealed)
    guard.verify(sealed)
    alias.unlink()
    alias.symlink_to(second, target_is_directory=True)
    with pytest.raises(ValueError, match="base SDK alias changed"):
        guard.verify(sealed)


def test_managed_update_refuses_before_network_or_install(monkeypatch, capsys):
    from tensorfold import update
    monkeypatch.setenv("TF_FOUNDATION_STATE", "/sealed/state.json")
    monkeypatch.setattr(update, "latest_release", lambda **_: pytest.fail("must refuse before network"))
    assert update.update(force=True) == 1
    assert "rebuild the image" in capsys.readouterr().err


def test_driver_probe_uses_exact_typed_driver_api_and_reports_mapping(tmp_path, monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "deploy/gb10"
    spec = importlib.util.spec_from_file_location("controlled_cuda_driver_probe", directory / "cuda_driver_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    library = tmp_path / "libcuda.so.615.71.09"
    library.write_bytes(b"controlled driver payload")
    calls = []

    class Function:
        def __init__(self, name, value=None, status=0):
            self.name, self.value, self.status = name, value, status

        def __call__(self, argument):
            calls.append(self.name)
            if self.value is not None:
                argument._obj.value = self.value
            return self.status

    import types
    provider = types.SimpleNamespace(cuInit=Function("cuInit"),
        cuDriverGetVersion=Function("cuDriverGetVersion", 13040), cuDeviceGetCount=Function("cuDeviceGetCount", 2))
    monkeypatch.setattr(module.ctypes, "CDLL", lambda _: provider)
    monkeypatch.setattr(module.Path, "read_text", lambda _: "000-100 r-xp 0 0 0 " + str(library))
    assert module.probe("libcuda.so.1", 13040) == {"driver_api": 13040, "devices": 2, "library": str(library)}
    assert calls == ["cuInit", "cuDriverGetVersion", "cuDeviceGetCount"]
    assert provider.cuInit.argtypes == [module.ctypes.c_uint]
    provider.cuInit.status = 804
    with pytest.raises(RuntimeError, match="cuInit returned CUDA error 804"):
        module.probe("libcuda.so.1", 13040)


def test_managed_extension_build_uses_state_toolkit_and_refuses_pip_fallback(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tensorfold.cuda import build
    selected = tmp_path / "selected"
    (selected / "bin").mkdir(parents=True)
    (selected / "bin/nvcc").write_text("selected compiler")
    other = tmp_path / "other"
    (other / "bin").mkdir(parents=True)
    (other / "bin/nvcc").write_text("unselected compiler")
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"toolchain": {"cuda_home": str(selected)}}))
    monkeypatch.setenv("TF_FOUNDATION_STATE", str(state))
    monkeypatch.setenv("CUDA_HOME", str(selected))
    torch = SimpleNamespace(version=SimpleNamespace(cuda="13.4"))
    extension = SimpleNamespace(CUDA_HOME=str(selected))
    assert build.pip_toolkit(extension, torch) == []
    extension.CUDA_HOME = None
    with pytest.raises(RuntimeError, match="requires its selected native toolkit"):
        build.pip_toolkit(extension, torch)
    monkeypatch.setenv("CUDA_HOME", str(other))
    extension.CUDA_HOME = str(other)
    with pytest.raises(RuntimeError, match="requires its selected native toolkit"):
        build.pip_toolkit(extension, torch)


@pytest.mark.parametrize("changed", [False, True])
def test_checked_run_reprobes_final_driver_before_native_proof_and_exec(guard, sealed, monkeypatch, changed):
    env = {"TF_CUDA_DRIVER_PROOF": json.dumps({"library": "/host/libcuda.so.615.71.09", "mode": "native"})}
    sealed["toolchain"] = {"header_version": 13040}
    monkeypatch.setattr(guard, "read_json", lambda _: sealed)
    monkeypatch.setattr(guard, "environment", lambda _: env)
    monkeypatch.setattr(guard.sys, "argv", ["guard", "checked-run", "--", "python3", "serve.py"])
    calls = []

    def run(arguments, **options):
        calls.append(arguments)
        path = "/other/libcuda.so.615.71.09" if changed else "/host/libcuda.so.615.71.09"
        return subprocess.CompletedProcess(arguments, 0, json.dumps({"library": path, "driver_api": 13040, "devices": 1}), "")

    class Executed(BaseException):
        pass

    def execute(file, arguments, environment):
        calls.append((file, arguments, environment))
        raise Executed

    monkeypatch.setattr(guard.subprocess, "run", run)
    monkeypatch.setattr(guard.os, "execvpe", execute)
    if changed:
        with pytest.raises(ValueError, match="post-entrypoint driver provider differs"):
            guard.main()
        assert len(calls) == 1
    else:
        with pytest.raises(Executed):
            guard.main()
        assert len(calls) == 3
        assert calls[0][-4:] == ["--library", "libcuda.so.1", "--required-api", "13040"]
        assert calls[1][-1] == "native-proof"
        assert calls[2][:2] == ("python3", ["python3", "serve.py"])


@pytest.mark.parametrize("wheel_provider,nccl_mismatch", [(False, False), (True, False), (False, True)])
def test_loaded_cuda_provider_must_be_native_sdk_even_if_wheel_is_sealed(guard, sealed, tmp_path, monkeypatch, wheel_provider, nccl_mismatch):
    import types
    root = Path(sealed["foundation"])
    native = tmp_path / "native/libcublas.so.13"
    native.parent.mkdir()
    native.write_bytes(b"base SDK provider")
    wheel = guard.site_directory(root) / "nvidia/cublas/lib/libcublas.so.13"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"legitimate but unused wheel provider")
    sealed["native_files"] = [str(native)]
    sealed["files"][str(wheel)] = guard.identity(wheel, full=True)
    torch = types.ModuleType("torch")
    torch.__file__, torch.__version__ = str(root / "torch.py"), "1.0"
    torch.cuda = types.SimpleNamespace(is_initialized=lambda: False)
    triton = types.ModuleType("triton")
    triton.__file__, triton.__version__ = str(root / "triton.py"), "1.0"
    torchvision = types.ModuleType("torchvision")
    torchvision.__file__, torchvision.__version__ = str(root / "torchvision/__init__.py"), "0.30"
    loaded = [str(root / "torchvision" / name) for name in ("_C_stable.so", "image_stable.so")]
    operator = types.SimpleNamespace(default=types.SimpleNamespace(_schema="controlled image schema"))
    torch.ops = types.SimpleNamespace(loaded_libraries=loaded,
        image=types.SimpleNamespace(decode_jpegs_cuda=operator, encode_jpegs_cuda=operator))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "torchvision", torchvision)
    monkeypatch.setitem(sys.modules, "foundation_tools", types.SimpleNamespace(verify_triton_selection=lambda *args: {}))
    selected = wheel if wheel_provider else native
    original = guard.Path.read_text
    monkeypatch.setattr(guard.Path, "read_text", lambda path, *args, **kwargs:
                        f"0000-1000 r-xp 0 {os.major(selected.stat().st_dev):x}:{os.minor(selected.stat().st_dev):x} {selected.stat().st_ino} " + str(selected)
                        if path == Path("/proc/self/maps") else original(path, *args, **kwargs))
    row = {**guard.identity(native, full=True), "path": str(native), "soname": "libcublas.so.13",
           "needed": [], "provider_role": "native_foundation"}
    sealed["runtime"] = {"schema_version": 1, "index": {"libcublas.so.13": row}, "vision": {"providers": []}}
    monkeypatch.setitem(sys.modules, "tensorfold_foundation_bootstrap", types.SimpleNamespace(
        RUNTIME_OWNERS=("controlled same-process owner",), DRIVER_ROLE="controlled driver",
        ACTIVE_PROFILE_SHA256=sealed["files"][sealed["runtime_profile_path"]]["sha256"]))
    sealed["toolchain"] = {}
    sealed["nccl_version"] = 23203

    def get_version(pointer):
        pointer._obj.value = 23100 if nccl_mismatch else 23203
        return 0

    monkeypatch.setenv("TF_NCCL_LIB", sealed["nccl_library"])
    monkeypatch.setitem(sys.modules, "tensorfold.cuda.comm", types.SimpleNamespace(
        _library=lambda: types.SimpleNamespace(ncclGetVersion=get_version)))
    if nccl_mismatch:
        with pytest.raises(ValueError, match="NCCL native API version differs"):
            guard.native_proof(sealed)
    elif wheel_provider:
        with pytest.raises(RuntimeError, match="non-foundation runtime provider"):
            guard.native_proof(sealed)
    else:
        assert guard.native_proof(sealed)["loaded_sdk"] == [str(native)]


def test_nccl_native_or_protected_wheel_selection_is_explicit(guard, tmp_path):
    foundation = tmp_path / "foundation"
    native = tmp_path / "native/libnccl.so.2.32.3"
    native.parent.mkdir()
    native.write_bytes(b"native NCCL")
    manifest = {"roles": {"foundation": {"nvidia-nccl-cu13": {"version": "2.32.3"}}}}
    assert guard.select_nccl(manifest, foundation, {native}) == (str(native), 23203)
    package = guard.site_directory(foundation) / "nvidia/nccl/lib"
    package.mkdir(parents=True)
    wheel = package / "libnccl.so.2"
    wheel.write_bytes(b"protected wheel NCCL")
    manifest["nccl_provider"] = "nvidia-nccl-cu13"
    assert guard.select_nccl(manifest, foundation, {native}) == (str(wheel), 23203)
    manifest["roles"]["foundation"].clear()
    with pytest.raises(ValueError, match="outside protected foundation"):
        guard.select_nccl(manifest, foundation, {native})


def test_conflicting_nccl_override_is_not_silently_replaced(guard, sealed, monkeypatch):
    import types
    monkeypatch.setitem(sys.modules, "foundation_tools", types.SimpleNamespace(
        toolchain_environment=lambda *args, **kwargs: {}))
    sealed["cache_key"] = "a" * 64
    sealed["state_digest"] = guard.state_digest(sealed)
    monkeypatch.setenv("TF_NCCL_LIB", "/wrong/libnccl.so.2")
    with pytest.raises((ValueError, FileNotFoundError)):
        guard.environment(sealed)


def test_empty_nccl_override_selects_default_like_unmanaged_loader(guard, sealed, monkeypatch):
    import types
    monkeypatch.setitem(sys.modules, "foundation_tools", types.SimpleNamespace(
        toolchain_environment=lambda *args, **kwargs: {}))
    sealed["cache_key"] = "a" * 64
    sealed["state_digest"] = guard.state_digest(sealed)
    monkeypatch.setenv("TF_NCCL_LIB", "")
    assert guard.environment(sealed)["TF_NCCL_LIB"] == sealed["nccl_library"]


@pytest.mark.parametrize("missing", ["image_stable.so", "_C_stable.so", "schema"])
def test_vision_consumer_requires_actual_native_image_load_and_schemas(guard, tmp_path, missing):
    from types import SimpleNamespace
    directory = tmp_path / "torchvision"
    libraries = [str(directory / name) for name in ("_C_stable.so", "image_stable.so") if name != missing]
    operator = SimpleNamespace(default=SimpleNamespace(_schema="registered native CUDA image schema"))
    image = SimpleNamespace(decode_jpegs_cuda=operator)
    if missing != "schema":
        image.encode_jpegs_cuda = operator
    torch = SimpleNamespace(ops=SimpleNamespace(loaded_libraries=libraries, image=image))
    vision = SimpleNamespace(__file__=str(directory / "__init__.py"))
    with pytest.raises((ValueError, AttributeError)):
        guard.vision_proof(torch, vision)


@pytest.mark.parametrize("mismatch", [False, True])
def test_native_addon_path_is_validated_before_exact_install(guard, tmp_path, monkeypatch, mismatch):
    path = tmp_path / "approved.deb"
    path.write_bytes(b"exact approved Debian artifact")
    manifest = {"native_addons": [{"artifact": path.name, "sha256": guard.digest(path),
                                  "package": "libapproved", "version": "1.0"}]}
    baseline = tmp_path / "base-sdk.tsv"
    baseline.write_text("cuda-sdk\t13.4.3\n")
    monkeypatch.setattr(guard, "native_packages", lambda _: {"cuda-sdk": "13.4.3"})
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[0] == "dpkg-deb":
            version = "0.9" if mismatch else "1.0"
            return subprocess.CompletedProcess(command, 0, f"Package: libapproved\nVersion: {version}\n", "")
        if command[0] == "dpkg-query":
            return subprocess.CompletedProcess(command, 1, "", "not installed")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(guard.subprocess, "run", run)
    verified = []
    monkeypatch.setattr(guard, "check_base_sdk", lambda *args: verified.append(True))
    if mismatch:
        with pytest.raises(ValueError, match="package metadata differs"):
            guard.install_native_addons(manifest, artifacts=tmp_path, sdk_baseline=baseline, pattern="^cuda-")
        assert not any(command[0] == "dpkg" for command in commands)
    else:
        guard.install_native_addons(manifest, artifacts=tmp_path, sdk_baseline=baseline, pattern="^cuda-")
        assert [command for command in commands if command[0] == "dpkg"] == [["dpkg", "--install", str(path)]]
        assert verified == [True]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux shared-library maps and ELF integration")
@pytest.mark.parametrize("script,arguments,activated", [
    ("cuda_entrypoint.py", [], False), ("cuda_driver_probe.py", [], False),
    ("tensorfold-cc", [], False), ("tensorfold-cxx", [], False),
    ("foundation_guard.py", ["environment"], False),
    ("foundation_guard.py", ["--state", "unused", "native-proof"], True),
    ("consumer.py", [], True),
])
def test_bootstrap_preloads_only_final_consumer_interpreter(tmp_path, script, arguments, activated):
    import shutil
    # Driver acquisition stands in for a real admitted CUDA driver with the
    # already-loaded host libc. This control makes no CUDA call/device claim.
    maps = Path("/proc/self/maps").read_text().splitlines()
    driver = next(Path(line.split(maxsplit=5)[5]) for line in maps
                  if len(line.split(maxsplit=5)) == 6 and Path(line.split(maxsplit=5)[5]).name == "libc.so.6")
    application = tmp_path / "application"
    venv.EnvBuilder(with_pip=False).create(application)
    site = application / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    source = Path(__file__).resolve().parents[1] / "deploy/gb10/foundation_bootstrap.py"
    shutil.copyfile(source, site / "tensorfold_foundation_bootstrap.py")
    (site / "00_foundation.pth").write_text("import tensorfold_foundation_bootstrap\n")
    (site / "tensorfold_foundation_runtime.py").write_text(
        "import os,sys\ndef activate_runtime(contract):\n assert 'torch' not in sys.modules\n "
        "os.environ['CONTROL_CORE_READY']='1'\n return ('controlled core owner',)\n")
    profile = tmp_path / "runtime.json"
    profile.write_text('{"schema_version":1}')
    values = {"TF_FOUNDATION_RUNTIME_PROFILE": str(profile), "TF_FOUNDATION_CPU_DRIVER": str(driver),
              "TORCH_USE_RTLD_GLOBAL": "1"}
    (site / "tensorfold-foundation-environment.json").write_text(json.dumps(values))
    target = tmp_path / script
    target.write_text("import os,sys\nprint(os.environ.get('CONTROL_CORE_READY', 'no'))\n"
                      "import tensorfold_foundation_bootstrap as b\nprint(len(b.RUNTIME_OWNERS) if b.RUNTIME_OWNERS else 0)\n")
    proof = json.dumps({"library": str(driver), "mode": "native", "driver_api": 13040, "devices": 1})
    env = dict(os.environ, TF_FOUNDATION_RUNTIME_PHASE="cuda", TF_CUDA_DRIVER_PROOF=proof)
    env.pop("TORCH_USE_RTLD_GLOBAL", None)
    result = subprocess.run([str(application / "bin/python"), str(target), *arguments], env=env,
                            check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == (["1", "2"] if activated else ["no", "0"])

"""Managed installation boundaries use real distributions/files and isolated pip."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import venv
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def guard(monkeypatch):
    # Synthetic sealed states select their own provider. The actual managed
    # interpreter can legitimately arrive with its production NCCL published.
    monkeypatch.delenv("TF_NCCL_LIB", raising=False)
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


def test_owned_timestamp_preparation_preserves_payload_and_system_alias(guard, tmp_path):
    owned, system = tmp_path / "owned", tmp_path / "system"
    owned.write_bytes(b"owned copied payload")
    system.write_bytes(b"pristine system provider")
    for path in (owned, system):
        path.chmod(0o444)
        os.utime(path, ns=(1700000000123456789, 1700000000987654321))
    alias = tmp_path / "venv-python-alias"
    alias.symlink_to(system)
    before = guard.identity(owned, full=True)
    guard.normalize_image_mtime(owned)
    guard.normalize_image_mtime(alias)
    after = guard.identity(owned, full=True)
    assert after == {**before, "mtime_ns": 1700000000000000000}
    assert system.stat().st_mtime_ns == 1700000000987654321
    assert system.read_bytes() == b"pristine system provider"
    with pytest.raises(ValueError, match="not a regular file"):
        guard.normalize_image_mtime(tmp_path)


@pytest.mark.parametrize("normalize", [False, True])
def test_owned_timestamp_preparation_survives_whole_second_archive_export(guard, tmp_path, normalize):
    path = tmp_path / "protected-file"
    path.write_bytes(b"unchanged protected content")
    path.chmod(0o444)
    os.utime(path, ns=(1700000000123456789, 1700000000987654321))
    if normalize:
        guard.normalize_image_mtime(path)
    expected = guard.identity(path, full=True)
    image = tmp_path / "layer.tar"
    # The observed BuildKit export drops fractional timestamp precision. This
    # controlled integer-second tar boundary preserves bytes/mode/path exactly.
    with tarfile.open(image, "w", format=tarfile.USTAR_FORMAT) as archive:
        archive.add(path, arcname=path.name)
    path.unlink()
    with tarfile.open(image) as archive:
        member = archive.getmember(path.name)
        with archive.extractfile(member) as stream:
            path.write_bytes(stream.read())
        path.chmod(member.mode)
        os.utime(path, ns=(0, member.mtime * 10**9))
    actual = guard.identity(path, full=True)
    assert (actual == expected) is normalize
    assert actual["sha256"] == expected["sha256"]


@pytest.mark.parametrize("full", [False, True])
def test_post_seal_fractional_timestamp_mutation_still_refuses(guard, sealed, full):
    path = guard.site_directory(Path(sealed["foundation"])) / "torch.py"
    guard.normalize_image_mtime(path)
    sealed["files"][str(path)] = guard.identity(path, full=True)
    sealed["state_digest"] = guard.state_digest(sealed)
    before = path.stat()
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1))
    with pytest.raises(ValueError, match="protected foundation file changed"):
        guard.verify(sealed, full=full)


def test_owned_json_publication_has_image_stable_timestamp(guard, tmp_path):
    path = tmp_path / "owned-state.json"
    guard.write_json(path, {"sealed": True})
    assert path.stat().st_mtime_ns % 10**9 == 0
    assert json.loads(path.read_text()) == {"sealed": True}


@pytest.mark.parametrize("write_fails,close_fails,group_fails", [
    (True, True, False), (True, False, False), (False, True, False), (True, True, True)])
def test_json_publication_preserves_write_and_close_outcomes(guard, tmp_path, monkeypatch,
                                                            write_fails, close_fails, group_fails):
    path = tmp_path / "state.json"
    path.write_text("old published state")
    temporary = path.with_name(path.name + ".new")
    primary, close_error, allocation_error = OSError("write"), OSError("close"), MemoryError("group allocation")
    native_cause = LookupError("original native cause")
    primary.__cause__ = native_cause

    class ControlledFile:
        closes = 0

        def write(self, text):
            if write_fails:
                raise primary
            return len(text)

        def close(self):
            self.closes += 1
            if close_fails:
                raise close_error

        # Also represents the old context-manager protocol for this preserved
        # regression: its close error would otherwise replace the write root.
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    stream = ControlledFile()
    original_open = guard.Path.open
    monkeypatch.setattr(guard.Path, "open", lambda current, *a, **kw:
                        stream if current == temporary else original_open(current, *a, **kw))
    if group_fails:
        def failed_group(*args):
            raise allocation_error

        monkeypatch.setattr(guard, "BaseExceptionGroup", failed_group, raising=False)
    with pytest.raises(BaseException) as caught:
        guard.write_json(path, {"owned": "publication"})
    assert stream.closes == 1 and path.read_text() == "old published state"
    if write_fails and close_fails and not group_fails:
        assert isinstance(caught.value, BaseExceptionGroup)
        assert caught.value.exceptions == (primary, close_error)
    else:
        assert caught.value is (primary if write_fails else close_error)
    if write_fails:
        assert primary.__cause__ is native_cause
    if group_fails:
        frame = caught.value.__traceback__
        retained = []
        while frame:
            retained.append(frame.tb_frame.f_locals)
            frame = frame.tb_next
        assert any(scope.get("retained_close") is close_error
                   and scope.get("diagnostic_failure") is allocation_error for scope in retained)


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


def wheel(path, name, version, *, payload=None, body=None):
    import base64
    import csv
    import hashlib
    import io

    info = f"{name}-{version}.dist-info"
    files = {f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n".encode(),
             f"{info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
             payload or f"{name}/__init__.py": (f"__version__ = {version!r}\n" if body is None else body).encode()}
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for member, raw in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
        writer.writerow([member, "sha256=" + digest, len(raw)])
    writer.writerow([f"{info}/RECORD", "", ""])
    with zipfile.ZipFile(path, "w") as archive:
        for member, raw in files.items():
            archive.writestr(member, raw)
        archive.writestr(f"{info}/RECORD", record.getvalue())


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


def test_compiler_inventory_includes_versioned_runtime_packages_only(guard):
    expected = {"gcc-13:amd64": "13.3", "libstdc++6:amd64": "14.2", "libgcc-s1:amd64": "14.2",
                "libgomp1": "14.2", "libasan8": "14.2", "libtsan2": "14.2", "libubsan1": "14.2",
                "libquadmath0": "14.2", "libatomic1": "14.2", "libcc1-0": "14.2", "libitm1": "14.2"}
    raw = "".join(f"{name}\t{version}\n" for name, version in expected.items())
    raw += "libstdc++unrelated\t1\nlibcublas-13-4\t13.4\nnsight-compute-2026.3.1\t2026.3.1\n"
    assert guard.sdk_versions(raw, guard.COMPILER_PACKAGES) == expected


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
    addon = tmp_path / "approved-runtime.deb"
    addon.write_bytes(b"owned copied native artifact")
    manifest = tmp_path / "foundation-manifest.json"
    manifest.write_text(json.dumps({"roles": {"foundation": {"torch": {"version": "1.0"}},
                                             "runtime": {}, "verification": {}},
                                    "native_addons": [{"artifact": addon.name, "sha256": guard.digest(addon),
                                                       "package": "runtime-addon", "version": "1.0"}]}))
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
    native_timestamps = {path: path.stat().st_mtime_ns for path in (marker, cpu_driver)}
    monkeypatch.setattr(guard, "select_nccl", lambda *args: (str(marker), None))
    tools.update(headers=[], libdevice={"path": str(marker)}, cupti={"path": str(marker)},
                 cuda_home=str(cuda_home), cupti_dir=str(marker.parent))
    state = guard.seal(manifest_path=manifest, foundation=foundation, application=application,
                       sdk_baseline=baseline, pattern_path=pattern, state_path=state_path,
                       artifacts=tmp_path, toolchain=tools)
    for path in (addon, manifest, baseline, pattern, state_path, Path(state["runtime_profile_path"]),
                 guard.site_directory(application) / "tensorfold-foundation-environment.json"):
        assert path.stat().st_mtime_ns % 10**9 == 0
    assert {path: path.stat().st_mtime_ns for path in native_timestamps} == native_timestamps
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


@pytest.mark.parametrize("proof_status", [0, 17])
def test_checked_run_keeps_proof_stdout_private_and_failure_stderr_visible(
        guard, sealed, monkeypatch, capfd, proof_status):
    proof = {"library": "/host/libcuda.so.615.71.09", "mode": "native"}
    sealed["toolchain"] = {"header_version": 13040}
    monkeypatch.setattr(guard, "read_json", lambda _: sealed)
    monkeypatch.setattr(guard, "environment", lambda _: {"TF_CUDA_DRIVER_PROOF": json.dumps(proof)})
    monkeypatch.setattr(guard.sys, "argv", ["guard", "checked-run", "--", "application"])
    real_run = subprocess.run

    def run(arguments, **options):
        if arguments[-1] == "native-proof":
            code = ("import sys\nprint('private provider JSON')\n"
                    f"if {proof_status}: print('native proof failure', file=sys.stderr)\n"
                    f"raise SystemExit({proof_status})\n")
            # This child is only an I/O/status stand-in, using a synthetic
            # driver path. Managed .pth activation is tested independently.
            return real_run([sys.executable, "-S", "-c", code], **options)
        return subprocess.CompletedProcess(arguments, 0, json.dumps({**proof, "driver_api": 13040, "devices": 1}))

    class Executed(BaseException):
        pass

    def execute(*_):
        print("application output")
        raise Executed

    monkeypatch.setattr(guard.subprocess, "run", run)
    monkeypatch.setattr(guard.os, "execvpe", execute)
    if proof_status:
        with pytest.raises(subprocess.CalledProcessError) as caught:
            guard.main()
        assert caught.value.returncode == proof_status
    else:
        with pytest.raises(Executed):
            guard.main()
    captured = capfd.readouterr()
    assert captured.out == ("" if proof_status else "application output\n")
    assert captured.err == ("native proof failure\n" if proof_status else "")


def test_explicit_native_proof_keeps_its_json_output(guard, sealed, monkeypatch, capsys):
    report = {"selected_provider": "native foundation"}
    monkeypatch.setattr(guard, "read_json", lambda _: sealed)
    monkeypatch.setattr(guard, "verify", lambda _: None)
    monkeypatch.setattr(guard, "native_proof", lambda _: report)
    monkeypatch.setattr(guard.sys, "argv", ["guard", "native-proof"])
    guard.main()
    captured = capsys.readouterr()
    assert json.loads(captured.out) == report
    assert captured.err == ""


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


def test_same_version_owned_wheel_replaces_real_payload_before_recording_identity(guard, sealed, tmp_path, monkeypatch):
    foundation, application = Path(sealed["foundation"]), Path(sealed["application"])
    venv.EnvBuilder(with_pip=True).create(foundation)
    venv.EnvBuilder(with_pip=False).create(application)
    fs, site = guard.site_directory(foundation), guard.site_directory(application)
    (site / "00_fixture_foundation.pth").write_text(f"import sys; sys.path.insert(0, {str(fs)!r})\n")
    sealed["files"].update({str(path): guard.identity(path, full=True) for path in guard.tree_files(foundation)})
    sealed.update(cache_key="c" * 64, state_path=str(tmp_path / "state.json"))
    sealed["state_digest"] = guard.state_digest(sealed)
    # Provider publication has separate controls; preserve the actual audited
    # pip boundary here while keeping this scratch venv's toolchain synthetic.
    monkeypatch.setattr(guard, "publish_state", lambda state, path: state.update(state_digest=guard.state_digest(state)))
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    monkeypatch.setenv("PIP_CONFIG_FILE", os.devnull)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.delenv("PIP_CONSTRAINT", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    protected = (fs / "torch.py").read_bytes()
    old_identity = None
    for label in ("old", "new"):
        directory = tmp_path / label
        directory.mkdir()
        path = directory / "tensorfold-1.0-py3-none-any.whl"
        body = f"__version__ = '1.0'\nTOKEN = {label!r}\n"
        wheel(path, "tensorfold", "1.0", body=body)
        guard.install_application(sealed, path)
        assert (site / "tensorfold/__init__.py").read_text() == body
        result = subprocess.run([str(application / "bin/python"), "-c", "import tensorfold; print(tensorfold.TOKEN)"],
                                check=True, capture_output=True, text=True)
        assert result.stdout.strip() == label
        assert sealed["application_wheel_sha256"] == guard.digest(path)
        if old_identity is None:
            old_identity = sealed["application_wheel_sha256"]
        else:
            assert sealed["application_wheel_sha256"] != old_identity
        assert sealed["foundation_cache_key"] == "c" * 64 and (fs / "torch.py").read_bytes() == protected
        guard.verify(sealed, full=True)


def test_prepared_application_refuses_replacement_before_pip_or_state_mutation(guard, sealed, tmp_path, monkeypatch):
    import copy

    sealed["native_artifacts"] = {"manifest": "sealed immutable image bundle"}
    sealed["state_digest"] = guard.state_digest(sealed)
    before = copy.deepcopy(sealed)
    monkeypatch.setattr(guard, "run_install", lambda *args: pytest.fail("mutated a prepared application"))
    # Even wheel opening is deferred until this build-lifetime precondition.
    with pytest.raises(ValueError, match="rebuild the image"):
        guard.install_application(sealed, tmp_path / "not-opened.whl")
    assert sealed == before


@pytest.fixture
def native_contract(guard, sealed, tmp_path, monkeypatch):
    import hashlib

    from tensorfold.cuda import native_artifact as native

    sealed.update(foundation_cache_key="f" * 64, application_wheel_sha256="a" * 64,
                  state_path=str(tmp_path / "state.json"))
    sealed["state_digest"] = guard.state_digest(sealed)
    package = guard.site_directory(Path(sealed["application"])) / "tensorfold"
    package.mkdir()
    header = package / "owned.h"
    header.write_text("// transitive owned input\n")
    directory = tmp_path / "native"
    directory.mkdir()
    recipes = []
    for base in ("first", "second", "third", "fourth", "fifth"):
        source = package / (base + ".cpp")
        source.write_text('#include "owned.h"\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n'
                          '    m.def("execute", &execute);\n}\n')
        recipes.append({"name": base, "sources": [str(source)], "prepared": True,
                        "extra_cuda_cflags": ["-O3"], "verbose": False})
    monkeypatch.setattr(native, "recipes", lambda: tuple(recipes))
    monkeypatch.setattr(native, "abi", lambda: {"host_machine": "controlled ABI"})
    identity = {"foundation_cache_key": "f" * 64, "application_wheel_sha256": "a" * 64,
                "toolchain_sha256": hashlib.sha256(native.canonical(sealed["toolchain"])).hexdigest()}
    value = {"schema_version": 1, "kind": "tensorfold-native-artifacts", "identity": identity,
             "abi": native.abi(), "target_capability": [12, 1], "package_root": str(package), "entries": {}}
    for recipe in recipes:
        name, sources, options = native.compilation(recipe)
        binary = directory / (name + ".so")
        binary.write_bytes(b"controlled binary " + name.encode())
        request, owned = native.request(package, sources, options), native.inputs(package, sources)
        exports = native.binding_exports(sources)
        value["entries"][name] = {"module_name": name, "binary": binary.name,
            "binary_sha256": guard.digest(binary), "binary_bytes": binary.stat().st_size,
            "request": request, "inputs": owned, "exports": exports,
            "recipe_sha256": hashlib.sha256(native.canonical({"request": request, "inputs": owned,
                "exports": exports, "identity": identity, "abi": value["abi"]})).hexdigest()}
    path = directory / "manifest.json"
    guard.write_json(directory / "preparation.json", {"scope": "controlled fixture"})
    monkeypatch.setattr(guard, "publish_state", lambda state, path:
                        state.update(state_digest=guard.state_digest(state)))
    return native, sealed, value, path, header


@pytest.mark.parametrize("fault", [None, "extra-entry", "exports", "request", "missing-header", "header",
                                      "binary", "wheel", "abi", "extra-file", "extra-directory"])
def test_five_provider_seal_checks_final_canonical_contract(native_contract, guard, fault):
    native, state, value, path, header = native_contract
    entry = next(iter(value["entries"].values()))
    if fault == "extra-entry":
        value["entries"]["stale_extra_sm_121"] = dict(entry)
    elif fault == "exports":
        entry["exports"] = ["stale"]
    elif fault == "request":
        entry["request"]["options"]["extra_cuda_cflags"].append("--use_fast_math")
    elif fault == "missing-header":
        entry["inputs"].pop(header.name)
    elif fault == "header":
        header.write_text("// changed header\n")
    elif fault == "binary":
        (path.parent / entry["binary"]).write_bytes(b"changed binary")
    elif fault == "wheel":
        value["identity"]["application_wheel_sha256"] = "b" * 64
    elif fault == "abi":
        value["abi"] = {"host_machine": "foreign"}
    elif fault == "extra-file":
        (path.parent / "stale.so").write_bytes(b"unsealed extra library")
    elif fault == "extra-directory":
        nested = path.parent / "stale.build"
        nested.mkdir()
        (nested / "object.o").write_bytes(b"unpublished compiler intermediate")
    guard.write_json(path, value)
    if fault is not None:
        with pytest.raises(ValueError):
            guard.seal_native_artifacts(state, path)
        assert "native_artifacts" not in state
    else:
        guard.seal_native_artifacts(state, path)
        assert state["native_artifacts"]["sha256"] == guard.digest(path)
        assert json.loads(guard.native_activation(state)[native.ACTIVATION]) == state["native_artifacts"]
        assert str(header) in state["files"] and str(path.parent / "preparation.json") in state["files"]
        guard.verify(state, full=True)


def test_native_preparation_preserves_primary_and_failed_audit(guard, monkeypatch, tmp_path):
    import prepare_native

    primary, audit_error = RuntimeError("native compile failed"), ValueError("provider changed")
    checks = []

    def verify(state, *, full):
        checks.append(full)
        if len(checks) > 1:
            raise audit_error

    def failed(*args):
        raise primary

    monkeypatch.setattr(prepare_native, "guard", guard)
    monkeypatch.setattr(guard, "verify", verify)
    monkeypatch.setattr(prepare_native, "_prepare", failed)
    with pytest.raises(BaseExceptionGroup) as caught:
        prepare_native.prepare({}, tmp_path / "unused")
    assert caught.value.exceptions == (primary, audit_error) and checks == [True, True]


@pytest.mark.torch
@pytest.mark.skipif(any(Path('/dev').glob('nvidia[0-9]*')), reason="requires GPU-free preparation profile")
@pytest.mark.parametrize("fault", [None, "exports", "existing-destination", "already-sealed"])
def test_preparer_compiles_only_five_original_recipes_without_gpu_discovery(guard, monkeypatch, tmp_path, fault):
    from types import SimpleNamespace

    import prepare_native

    import tensorfold

    torch = pytest.importorskip("torch")
    from torch.utils import cpp_extension

    from tensorfold.cuda import native_artifact as native

    monkeypatch.setattr(prepare_native, "guard", guard)
    monkeypatch.setenv("TF_FOUNDATION_RUNTIME_PHASE", "cpu")
    monkeypatch.setattr(guard, "verify", lambda *args, **kwargs: None)
    monkeypatch.setattr(guard, "site_directory", lambda prefix: Path(tensorfold.__file__).resolve().parent.parent)
    sealed, compiled, normalized = [], [], []
    monkeypatch.setattr(guard, "seal_native_artifacts", lambda state, path: sealed.append(path))
    # This control reads original recipe inputs through the RO source mount.
    # Real metadata preparation/export is covered by the owned-install fixture below.
    monkeypatch.setattr(guard, "normalize_image_mtime", lambda path: normalized.append(path))

    def no_gpu(*args, **kwargs):
        pytest.fail("GPU-free preparation queried a runtime GPU")

    monkeypatch.setattr(torch.cuda, "get_device_capability", no_gpu)
    monkeypatch.setattr(torch.cuda, "device_count", no_gpu)

    def compile_recipe(**options):
        package = Path(tensorfold.__file__).resolve().parent
        required = {package / relative for relative in native.inputs(package, options["sources"])}
        assert required.issubset(normalized)
        compiled.append(options)
        binary = Path(options["build_directory"]) / (options["name"] + ".so")
        binary.write_bytes(b"controlled compiler output")
        exports = native.binding_exports(options["sources"])
        values = {name: lambda: None for name in exports}
        if fault == "exports":
            values.pop(exports[0])
        return SimpleNamespace(__file__=str(binary), **values)

    monkeypatch.setattr(cpp_extension, "load", compile_recipe)
    destination = tmp_path / "native"
    state = {"foundation_cache_key": "f" * 64, "application_wheel_sha256": "a" * 64,
             "application": "/controlled", "toolchain": {}}
    if fault == "already-sealed":
        state["native_artifacts"] = {"existing": "immutable application"}
        with pytest.raises(ValueError, match="already sealed"):
            prepare_native.prepare(state, destination)
        assert not compiled and not normalized and not destination.exists()
    elif fault == "existing-destination":
        destination.mkdir()
        with pytest.raises(FileExistsError):
            prepare_native.prepare(state, destination)
        assert not compiled and not sealed
    elif fault == "exports":
        with pytest.raises(ValueError, match="exports differ"):
            prepare_native.prepare(state, destination)
        assert not sealed and not (destination / "manifest.json").exists()
    else:
        result = prepare_native.prepare(state, destination)
        assert sealed == [destination / "manifest.json"] and len(compiled) == 5
        assert [row["name"] for row in compiled] == ["tensorfold_nvfp4_ck_v7_sm_121a", "tensorfold_qwen_b16_v6_sm_121",
                                                    "tensorfold_gdn_v3_sm_121", "tensorfold_prefill_attention_v1_sm_121",
                                                    "tensorfold_qmm_v6_sm_121"]
        assert [row["extra_cuda_cflags"][-1] for row in compiled] == [
            "-gencode=arch=compute_121a,code=sm_121a", *["-gencode=arch=compute_121,code=sm_121"] * 4]
        assert all("prepared" not in row for row in compiled)
        assert not any(destination.glob("*.build"))
        assert set(result["entries"]) == {row["name"] for row in compiled}


@pytest.mark.torch
@pytest.mark.skipif(any(Path('/dev').glob('nvidia[0-9]*')), reason="requires CPU load-only audit profile")
@pytest.mark.parametrize("fault", [None, "exports", "providers"])
def test_cpu_auditor_checks_exports_and_providers_without_resealing(native_contract, guard, monkeypatch, fault):
    from types import SimpleNamespace

    import foundation_runtime
    import prepare_native

    torch = pytest.importorskip("torch")
    native, state, value, path, _header = native_contract
    # capture_runtime and actual sealed images supply this profile. The
    # provider-mapping callback is controlled here; its input/flag forwarding is
    # the auditor's integration responsibility, not a substitute runtime proof.
    profile = {"schema_version": 1, "foundation": state["foundation"], "index": {},
               "torch": {"objects": [], "required": [], "order": []},
               "vision": {"providers": []}, "cupti_dir": None}
    state["runtime"] = profile
    state["state_digest"] = guard.state_digest(state)
    guard.write_json(path, value)
    guard.seal_native_artifacts(state, path)
    before = json.dumps(state, sort_keys=True)
    monkeypatch.setattr(prepare_native, "guard", guard)
    monkeypatch.setenv("TF_FOUNDATION_RUNTIME_PHASE", "cpu")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(guard, "native_proof", lambda state: {"providers": ["owned"]})
    observed = []

    def mapped_runtime(contract, *, allow_vision):
        assert contract is profile and allow_vision is True
        observed.append(contract)
        return ["foreign" if fault == "providers" else "owned"]

    monkeypatch.setattr(foundation_runtime, "mapped_runtime", mapped_runtime)
    monkeypatch.setattr(guard, "publish_state", lambda *args: pytest.fail("audit modified the sealed state"))
    monkeypatch.setattr(native, "_import", lambda *args:
                        SimpleNamespace(execute=0 if fault == "exports" else lambda: None))
    if fault:
        with pytest.raises(ValueError, match=r"exports differ|protected providers"):
            prepare_native.audit(state, path.parent)
    else:
        result = prepare_native.audit(state, path.parent)
        assert len(result["entries"]) == 5 and result["providers_unchanged"] is True
    assert observed == ([] if fault == "exports" else [profile])
    assert json.dumps(state, sort_keys=True) == before


@pytest.mark.torch
@pytest.mark.skipif(any(Path('/dev').glob('nvidia[0-9]*')), reason="requires GPU-free preparation profile")
def test_fresh_native_inputs_survive_whole_second_image_boundary(native_contract, guard, monkeypatch, tmp_path):
    from types import SimpleNamespace

    import prepare_native

    import tensorfold

    pytest.importorskip("torch")
    from torch.utils import cpp_extension

    native, state, value, _old_path, _header = native_contract
    package = Path(value["package_root"])
    initializer = package / "__init__.py"
    initializer.write_text("# installed-package identity fixture; never executed\n")
    monkeypatch.setattr(tensorfold, "__file__", str(initializer))
    monkeypatch.setattr(prepare_native, "guard", guard)
    monkeypatch.setenv("TF_FOUNDATION_RUNTIME_PHASE", "cpu")
    paths = {package / relative for entry in value["entries"].values() for relative in entry["inputs"]}
    before = {}
    for path in paths:
        path.chmod(0o444)
        os.utime(path, ns=(1700000000123456789, 1700000000987654321))
        before[path] = guard.identity(path, full=True)
    compiled = []

    def compile_recipe(**options):
        required = {package / relative for relative in native.inputs(package, options["sources"])}
        assert all(path.stat().st_mtime_ns % 10**9 == 0 for path in required)
        compiled.append(options["name"])
        binary = Path(options["build_directory"]) / (options["name"] + ".so")
        binary.write_bytes(b"controlled output; no compiler or GPU")
        return SimpleNamespace(__file__=str(binary), execute=lambda: None)

    monkeypatch.setattr(cpp_extension, "load", compile_recipe)
    prepare_native.prepare(state, tmp_path / "prepared")
    assert len(compiled) == 5
    for path in paths:
        assert guard.identity(path, full=True) == {
            **before[path], "mtime_ns": before[path]["mtime_ns"] // 10**9 * 10**9}
    # Exercise the observed image-export contract, not relaxed post-seal checks.
    archive_path = tmp_path / "image-inputs.tar"
    with tarfile.open(archive_path, "w", format=tarfile.USTAR_FORMAT) as archive:
        for path in sorted(paths):
            archive.add(path, arcname=str(path.relative_to(package)))
    for path in paths:
        path.unlink()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            with archive.extractfile(member) as stream:
                path = package / member.name
                path.write_bytes(stream.read())
            path.chmod(member.mode)
            os.utime(path, ns=(0, member.mtime * 10**9))
    guard.verify(state, full=True)
    changed = min(paths)
    stamp = changed.stat()
    os.utime(changed, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
    for full in (False, True):
        with pytest.raises(ValueError, match="protected foundation file changed"):
            guard.verify(state, full=full)
    with pytest.raises(ValueError, match="protected foundation file changed"):
        prepare_native.prepare(state, tmp_path / "must-not-repair")
    assert len(compiled) == 5 and changed.stat().st_mtime_ns == stamp.st_mtime_ns + 1

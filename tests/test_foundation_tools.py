"""Owned Triton tool/header selection and transparent compiler exec boundaries."""

import json
import os
import shutil
import signal
import subprocess
import sys
import venv
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.cuda import foundation_tools as tools

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name != "posix", reason="foundation fixtures require POSIX shell, signals and FIFOs")


def executable(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    path.chmod(0o755)
    return str(path)


def compiler_environment(**values):
    # Synthetic compiler contracts deliberately run outside the real image's
    # managed Python .pth activation. Never pass its publication into fixtures.
    inherited = {key: value for key, value in os.environ.items()
                 if not key.startswith(("TF_FOUNDATION_", "PYTHON"))
                 and key not in ("TF_NCCL_LIB", "TF_CUDA_DRIVER_PROOF", "COMPILER_STATUS", "COMPILER_SIGNAL")}
    return {**inherited, **values}


def wrapper_command(path, *arguments):
    # Keep the production wrapper bytes/argv0, but omit site.py for this child.
    # Actual managed entrypoint/provider behavior has separate integration gates.
    return [sys.executable, "-S", path, *arguments]


@pytest.fixture
def foundation(tmp_path):
    home = tmp_path / "CUDA foundation"
    include = home / "include"
    include.mkdir(parents=True)
    (include / "cuda.h").write_text("#define CUDA_VERSION 13040\n#define SELECTED_HEADER 43\n")
    (include / "secondary.h").write_text("#define SECONDARY_HEADER 1\n")
    for name in tools._TOOLS:
        executable(home / "bin" / name, "#!/bin/sh\nprintf 'tool release 13.4, V13.4.92\\n'\n")
    (home / "nvvm/libdevice").mkdir(parents=True)
    (home / "nvvm/libdevice/libdevice.10.bc").write_bytes(b"owned libdevice")
    (home / "extras/CUPTI/lib64").mkdir(parents=True)
    (home / "extras/CUPTI/lib64/libcupti.so").write_bytes(b"owned CUPTI")
    script = (ROOT / "deploy/gb10/foundation_compiler.py").read_text()
    wrappers = tmp_path / "compiler aliases"
    cc_alias = executable(wrappers / "tensorfold-cc", script)
    cxx_alias = executable(wrappers / "tensorfold-cxx", script)
    shutil.copyfile(tools.__file__, wrappers / "foundation_tools.py")
    compiler = executable(tmp_path / "compiler", "#!" + sys.executable + " -S\n"
                          "import json,os,sys\n"
                          "if sys.argv[1:] == ['--version']:\n"
                          " print('owned compiler 1');sys.exit(0)\n"
                          "print(json.dumps(sys.argv[1:]))\n"
                          "print('compiler stderr',file=sys.stderr)\n"
                          "if os.environ.get('COMPILER_SIGNAL'):\n"
                          " os.kill(os.getpid(),int(os.environ['COMPILER_SIGNAL']))\n"
                          "sys.exit(int(os.environ.get('COMPILER_STATUS','0')))\n")
    manifest = tmp_path / "sealed.json"
    kwargs = {"cuda_home": str(home), "cc": compiler, "cxx": compiler,
              "wrapper_cc": cc_alias, "wrapper_cxx": cxx_alias, "contract_path": str(manifest)}
    contract = tools.capture_toolchain(**kwargs)
    manifest.write_text(json.dumps({"toolchain": contract}))
    return SimpleNamespace(home=home, include=include, manifest=manifest, contract=contract, kwargs=kwargs)


def test_complete_environment_and_cache_identity(foundation, tmp_path):
    env = tools.toolchain_environment(foundation.contract, cache_root=str(tmp_path / "cache"))
    assert env["TRITON_PTXAS_PATH"] == env["TRITON_PTXAS_BLACKWELL_PATH"]
    assert env["TRITON_CUPTI_LIB_PATH"] == env["TRITON_CUPTI_LIB_BLACKWELL_PATH"]
    assert env["CC"] != env["CXX"]
    assert env["CUDACXX"] == str(foundation.home / "bin/nvcc")
    assert env["TRITON_CUDART_PATH"] == str(foundation.include)
    assert env["TRITON_LIBDEVICE_PATH"] == str(foundation.home / "nvvm/libdevice/libdevice.10.bc")
    assert all(foundation.contract["cache_key"] in env[key] for key in
               ("TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR"))
    (foundation.include / "secondary.h").write_text("#define SECONDARY_HEADER 2\n")
    changed = tools.capture_toolchain(**foundation.kwargs)
    assert changed["cache_key"] != foundation.contract["cache_key"]


def test_compiler_children_receive_only_their_synthetic_publication(foundation, monkeypatch):
    ambient = {"TF_FOUNDATION_STATE": "/actual/state.json", "TF_FOUNDATION_RUNTIME_PHASE": "cpu",
               tools.CONTRACT_ENV: "/actual/toolchain.json", "TF_NCCL_LIB": "/actual/libnccl.so",
               "TF_CUDA_DRIVER_PROOF": "/actual/driver-proof", "PYTHONPATH": "/actual/python",
               "PYTHONHOME": "/actual/home", "COMPILER_STATUS": "78", "COMPILER_SIGNAL": "15"}
    for key, value in ambient.items():
        monkeypatch.setenv(key, value)
    env = compiler_environment(**{tools.CONTRACT_ENV: str(foundation.manifest), "COMPILER_STATUS": "29"})
    assert env[tools.CONTRACT_ENV] == str(foundation.manifest)
    assert env["COMPILER_STATUS"] == "29"
    assert all(key not in env for key in ambient if key not in (tools.CONTRACT_ENV, "COMPILER_STATUS"))
    assert all(os.environ[key] == value for key, value in ambient.items())


def test_synthetic_wrapper_does_not_run_a_managed_site_publication(foundation, tmp_path):
    application = tmp_path / "managed-python"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(application)
    python = str(application / "bin/python")
    foreign = executable(tmp_path / "foreign-compiler", "#!/bin/sh\nprintf 'foreign compiler\\n'\n")
    foreign_manifest = tmp_path / "foreign.json"
    contract = tools.capture_toolchain(**{**foundation.kwargs, "cc": foreign, "cxx": foreign,
                                          "contract_path": str(foreign_manifest)})
    foreign_manifest.write_text(json.dumps({"toolchain": contract}))
    site = application / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    (site / "00_foreign_publication.pth").write_text(
        f"import os; os.environ[{tools.CONTRACT_ENV!r}] = {str(foreign_manifest)!r}\n")
    alias = foundation.kwargs["wrapper_cc"]
    env = compiler_environment(**{tools.CONTRACT_ENV: str(foundation.manifest), "COMPILER_STATUS": "29"})
    ordinary = subprocess.run([python, alias, "input.c"], env=env, capture_output=True, text=True, check=False)
    assert ordinary.returncode == 0 and ordinary.stdout == "foreign compiler\n"
    command = wrapper_command(alias, "input.c")
    command[0] = python
    isolated = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert isolated.returncode == 29
    assert json.loads(isolated.stdout) == ["-I" + str(foundation.include), "input.c"]
    assert isolated.stderr == "compiler stderr\n"


@pytest.mark.parametrize("changed", ["tool", "compiler", "wrapper", "helper", "header", "new_header", "libdevice", "cupti"])
def test_activation_refuses_changed_provider(foundation, tmp_path, changed):
    paths = {"tool": foundation.home / "bin/ptxas", "compiler": Path(foundation.kwargs["cc"]),
             "wrapper": Path(foundation.kwargs["wrapper_cc"]),
             "helper": Path(foundation.kwargs["wrapper_cc"]).parent / "foundation_tools.py",
             "header": foundation.include / "secondary.h", "new_header": foundation.include / "extra.h",
             "libdevice": foundation.home / "nvvm/libdevice/libdevice.10.bc",
             "cupti": foundation.home / "extras/CUPTI/lib64/libcupti.so"}
    with paths[changed].open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        tools.toolchain_environment(foundation.contract, cache_root=str(tmp_path))


def test_contract_mutation_and_relative_paths_refused(foundation):
    changed = deepcopy(foundation.contract)
    changed["include"] = "/other"
    with pytest.raises(ValueError, match="identity"):
        tools.compiler_command(changed, role="cc", arguments=[])
    with pytest.raises(ValueError, match="absolute"):
        tools.toolchain_environment(foundation.contract, cache_root="relative")
    with pytest.raises(ValueError, match="role"):
        tools.compiler_command(foundation.contract, role="unknown", arguments=[])


def test_managed_activation_reuses_owners_sdk_verification(foundation, tmp_path, monkeypatch):
    def unexpected_scan(*args, **kwargs):
        raise AssertionError("managed owner already verified the immutable SDK inventory")

    monkeypatch.setattr(tools, "_headers", unexpected_scan)
    monkeypatch.setattr(tools.hashlib, "file_digest", unexpected_scan)
    env = tools.toolchain_environment(foundation.contract, cache_root=str(tmp_path), verify_content=False)
    assert env["CC"] == foundation.kwargs["wrapper_cc"]
    Path(foundation.kwargs["cc"]).write_text("changed size")
    with pytest.raises(ValueError, match="changed"):
        tools.toolchain_environment(foundation.contract, cache_root=str(tmp_path), verify_content=False)


def test_content_mode_requires_boolean(foundation, tmp_path):
    with pytest.raises(ValueError, match="boolean"):
        tools.toolchain_environment(foundation.contract, cache_root=str(tmp_path), verify_content=0)


def test_capture_requires_valid_tools_and_owned_helpers(foundation):
    executable(foundation.home / "bin/ptxas", "#!/bin/sh\nprintf 'unrecognized tool\\n'\n")
    with pytest.raises(ValueError, match="recognized version"):
        tools.capture_toolchain(**foundation.kwargs)


def test_capture_refuses_header_directory_alias(foundation):
    (foundation.include / "alias").symlink_to(foundation.include, target_is_directory=True)
    with pytest.raises(ValueError, match="directory alias"):
        tools.capture_toolchain(**foundation.kwargs)


@pytest.mark.parametrize("kind", ["dangling", "fifo"])
def test_capture_refuses_unaccounted_header_entry(foundation, kind):
    path = foundation.include / "bad-entry"
    if kind == "dangling":
        path.symlink_to(foundation.include / "missing")
    else:
        os.mkfifo(path)
    with pytest.raises(ValueError, match="not a regular file or directory"):
        tools.capture_toolchain(**foundation.kwargs)


@pytest.mark.parametrize("provider", ["tool", "header", "libdevice"])
def test_capture_refuses_foreign_provider_alias(foundation, tmp_path, provider):
    foreign = tmp_path / "foreign"
    executable(foreign, "#!/bin/sh\nprintf 'release 13.4\\n'\n")
    path = {"tool": foundation.home / "bin/ptxas", "header": foundation.include / "cuda.h",
            "libdevice": foundation.home / "nvvm/libdevice/libdevice.10.bc"}[provider]
    path.unlink()
    path.symlink_to(foreign)
    with pytest.raises(ValueError, match="escapes"):
        tools.capture_toolchain(**foundation.kwargs)


@pytest.mark.parametrize("role", ["cc", "cxx"])
@pytest.mark.parametrize("status", [0, 29])
def test_wrapper_exec_preserves_arguments_streams_and_status(foundation, role, status):
    arguments = ["", "with spaces", 'quote"', "$literal", "é", "--version", "-ccbin=/literal"]
    env = compiler_environment(**{tools.CONTRACT_ENV: str(foundation.manifest), "COMPILER_STATUS": str(status)})
    result = subprocess.run(wrapper_command(foundation.kwargs["wrapper_" + role], *arguments),
                            env=env, capture_output=True, text=True, check=False)
    assert result.returncode == status
    assert json.loads(result.stdout) == ["-I" + str(foundation.include), *arguments]
    assert result.stderr == "compiler stderr\n"


def test_wrapper_requires_exact_manifest_and_refuses_changed_compiler(foundation):
    alias = foundation.kwargs["wrapper_cc"]
    env = compiler_environment()
    result = subprocess.run(wrapper_command(alias, "--version"), env=env, capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "sealed manifest" in result.stderr
    env[tools.CONTRACT_ENV] = str(foundation.manifest)
    Path(foundation.kwargs["cc"]).write_text("changed")
    result = subprocess.run(wrapper_command(alias, "--version"), env=env, capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "provider changed" in result.stderr


def test_wrapper_exec_preserves_compiler_termination_signal(foundation):
    env = compiler_environment(**{tools.CONTRACT_ENV: str(foundation.manifest), "COMPILER_SIGNAL": str(signal.SIGTERM)})
    result = subprocess.run(wrapper_command(foundation.kwargs["wrapper_cc"], "input.c"), env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == -signal.SIGTERM


@pytest.mark.parametrize("signal_name", ["SIGPIPE", "SIGXFSZ"])
def test_native_compiler_inherits_subprocess_restored_signals(foundation, tmp_path, signal_name):
    cc = shutil.which("cc")
    if cc is None or not hasattr(signal, signal_name):
        pytest.skip("native compiler and platform signal required")
    source = tmp_path / "native.c"
    source.write_text('#include <signal.h>\n#include <stdio.h>\n#include <string.h>\n'
                      'int main(int n,char **v){if(n==2&&!strcmp(v[1],"--version")){puts("native");return 0;}'
                      f'raise({signal_name});return 88;}}\n')
    native = tmp_path / "native-compiler"
    subprocess.run([cc, str(source), "-o", str(native)], check=True, capture_output=True)
    contract = tools.capture_toolchain(**{**foundation.kwargs, "cc": str(native), "cxx": str(native)})
    foundation.manifest.write_text(json.dumps({"toolchain": contract}))
    env = compiler_environment(**{tools.CONTRACT_ENV: str(foundation.manifest)})
    direct = subprocess.run([str(native), "input.c"], env=env, capture_output=True, check=False)
    wrapped = subprocess.run(wrapper_command(foundation.kwargs["wrapper_cc"], "input.c"), env=env,
                             capture_output=True, check=False)
    assert direct.returncode == wrapped.returncode == -getattr(signal, signal_name)


def mock_triton(monkeypatch, contract):
    env = tools.toolchain_environment(contract, cache_root="/cache")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    nvidia = SimpleNamespace(**{name: SimpleNamespace(path=env[key]) for name, key in tools._TRITON_TOOLS.items()},
                            libdevice_path=env["TRITON_LIBDEVICE_PATH"])
    knobs = SimpleNamespace(nvidia=nvidia, cache=SimpleNamespace(dir=env["TRITON_CACHE_DIR"]),
                            proton=SimpleNamespace(cupti_lib_dir=env["TRITON_CUPTI_LIB_PATH"],
                                                  cupti_lib_blackwell_dir=env["TRITON_CUPTI_LIB_BLACKWELL_PATH"]),
                            build=SimpleNamespace(cudart_path=env["TRITON_CUDART_PATH"],
                                                  cudacrt_path=env["TRITON_CUDACRT_PATH"], impl=None))
    resolver = SimpleNamespace(_find_compiler=lambda language: env["CC" if language == "c" else "CXX"])
    monkeypatch.setitem(sys.modules, "triton.runtime.build", resolver)
    return SimpleNamespace(knobs=knobs), resolver


def test_resolved_triton_providers_verified_without_gpu(foundation, monkeypatch):
    triton, _ = mock_triton(monkeypatch, foundation.contract)
    receipt = tools.verify_triton_selection(foundation.contract, triton)
    assert receipt["tools"]["ptxas"] == receipt["tools"]["ptxas_blackwell"]


@pytest.mark.parametrize("changed", ["ptxas", "ptxas_blackwell", "libdevice", "cupti", "compiler", "build_impl", "env", "cache"])
def test_resolved_triton_fallback_or_early_cache_refused(foundation, monkeypatch, changed):
    triton, resolver = mock_triton(monkeypatch, foundation.contract)
    if changed in ("ptxas", "ptxas_blackwell"):
        getattr(triton.knobs.nvidia, changed).path = foundation.kwargs["cc"]
    elif changed == "libdevice":
        triton.knobs.nvidia.libdevice_path = "/bundled/libdevice"
    elif changed == "cupti":
        triton.knobs.proton.cupti_lib_blackwell_dir = "/bundled/CUPTI"
    elif changed == "compiler":
        resolver._find_compiler = lambda language: "/previous/compiler"
    elif changed == "build_impl":
        triton.knobs.build.impl = object()
    elif changed == "cache":
        triton.knobs.cache.dir = "/old/cache"
    else:
        monkeypatch.setenv("TRITON_PTXAS_PATH", "/ignored/override")
    with pytest.raises(ValueError, match=r"Triton|foundation"):
        tools.verify_triton_selection(foundation.contract, triton)


@pytest.mark.parametrize("role,compiler,source", [("cc", "cc", "_Static_assert(SELECTED_HEADER == 43, \"header\");"),
                                                ("cxx", "c++", "static_assert(SELECTED_HEADER == 43, \"header\");")])
def test_real_compiler_header_precedence_and_query_modes(foundation, tmp_path, role, compiler, source):
    selected = shutil.which(compiler)
    if selected is None:
        pytest.skip("native C/C++ compiler not available")
    kwargs = {**foundation.kwargs, role: selected}
    contract = tools.capture_toolchain(**kwargs)
    foundation.manifest.write_text(json.dumps({"toolchain": contract}))
    env = compiler_environment(**{tools.CONTRACT_ENV: str(foundation.manifest)})
    alias = kwargs["wrapper_" + role]
    old = tmp_path / "bundled"
    old.mkdir()
    (old / "cuda.h").write_text("#define SELECTED_HEADER 31\n")
    src = tmp_path / ("probe.c" if role == "cc" else "probe.cpp")
    src.write_text("#include <cuda.h>\n" + source + "\n")
    result = subprocess.run(wrapper_command(alias, str(src), "-I" + str(old), "-fsyntax-only"), env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    for query in ("--version", "-dumpversion", "-dumpmachine"):
        plain = subprocess.run([selected, query], capture_output=True, check=False)
        wrapped = subprocess.run(wrapper_command(alias, query), env=env, capture_output=True, check=False)
        assert (wrapped.returncode, wrapped.stdout, wrapped.stderr) == (plain.returncode, plain.stdout, plain.stderr)

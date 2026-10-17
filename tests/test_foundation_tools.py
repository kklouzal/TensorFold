"""Owned Triton tool/header selection and transparent compiler exec boundaries."""

import json
import os
import shutil
import signal
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.cuda import foundation_tools as tools

ROOT = Path(__file__).resolve().parents[1]


def executable(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    path.chmod(0o755)
    return str(path)


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
    compiler = executable(tmp_path / "compiler", "#!" + sys.executable + "\n"
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


def test_capture_requires_valid_tools_and_owned_helpers(foundation):
    executable(foundation.home / "bin/ptxas", "#!/bin/sh\nprintf 'unrecognized tool\\n'\n")
    with pytest.raises(ValueError, match="recognized version"):
        tools.capture_toolchain(**foundation.kwargs)


def test_capture_refuses_header_directory_alias(foundation):
    (foundation.include / "alias").symlink_to(foundation.include, target_is_directory=True)
    with pytest.raises(ValueError, match="directory alias"):
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
    env = {**os.environ, tools.CONTRACT_ENV: str(foundation.manifest), "COMPILER_STATUS": str(status)}
    result = subprocess.run([foundation.kwargs["wrapper_" + role], *arguments],
                            env=env, capture_output=True, text=True, check=False)
    assert result.returncode == status
    assert json.loads(result.stdout) == ["-I" + str(foundation.include), *arguments]
    assert result.stderr == "compiler stderr\n"


def test_wrapper_requires_exact_manifest_and_refuses_changed_compiler(foundation):
    alias = foundation.kwargs["wrapper_cc"]
    env = {key: value for key, value in os.environ.items() if key != tools.CONTRACT_ENV}
    result = subprocess.run([alias, "--version"], env=env, capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "sealed manifest" in result.stderr
    env[tools.CONTRACT_ENV] = str(foundation.manifest)
    Path(foundation.kwargs["cc"]).write_text("changed")
    result = subprocess.run([alias, "--version"], env=env, capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "provider changed" in result.stderr


def test_wrapper_exec_preserves_compiler_termination_signal(foundation):
    env = {**os.environ, tools.CONTRACT_ENV: str(foundation.manifest), "COMPILER_SIGNAL": str(signal.SIGTERM)}
    result = subprocess.run([foundation.kwargs["wrapper_cc"], "input.c"], env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == -signal.SIGTERM


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
    env = {**os.environ, tools.CONTRACT_ENV: str(foundation.manifest)}
    alias = kwargs["wrapper_" + role]
    old = tmp_path / "bundled"
    old.mkdir()
    (old / "cuda.h").write_text("#define SELECTED_HEADER 31\n")
    src = tmp_path / ("probe.c" if role == "cc" else "probe.cpp")
    src.write_text("#include <cuda.h>\n" + source + "\n")
    result = subprocess.run([alias, str(src), "-I" + str(old), "-fsyntax-only"], env=env,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    for query in ("--version", "-dumpversion", "-dumpmachine"):
        plain = subprocess.run([selected, query], capture_output=True, check=False)
        wrapped = subprocess.run([alias, query], env=env, capture_output=True, check=False)
        assert (wrapped.returncode, wrapped.stdout, wrapped.stderr) == (plain.returncode, plain.stdout, plain.stderr)

"""Preparation-only selection of an immutable CUDA foundation for Triton.

The image owner captures this contract after installing the toolkit and owned
compiler aliases, stores it under ``toolchain`` in its sealed JSON manifest,
and publishes the returned environment before importing Torch or Triton. The
owner keeps these files immutable for the process lifetime. No SDK/component
version equality is inferred: exact files and their reported versions identify
the providers. Triton's supported environment knobs can silently fall back to
bundled tools, so the resolved descriptors must also be checked before use.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

CONTRACT_ENV = "TF_FOUNDATION_TOOLCHAIN_CONTRACT"
_SCHEMA = 1
_TOOLS = ("ptxas", "cuobjdump", "nvdisasm", "nvcc", "nvlink")
_TRITON_TOOLS = {
    "ptxas": "TRITON_PTXAS_PATH",
    "ptxas_blackwell": "TRITON_PTXAS_BLACKWELL_PATH",
    "cuobjdump": "TRITON_CUOBJDUMP_PATH",
    "nvdisasm": "TRITON_NVDISASM_PATH",
}


def _digest(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _file(path: str | Path, *, executable: bool = False, verify_content: bool = True) -> dict:
    selected = Path(path)
    if not selected.is_absolute():
        raise ValueError(f"foundation path must be absolute: {path}")
    real = selected.resolve(strict=True)
    if not real.is_file() or (executable and not os.access(real, os.X_OK)):
        raise ValueError(f"foundation provider is not a {'executable ' if executable else ''}file: {path}")
    record = {"path": str(selected), "realpath": str(real), "size": real.stat().st_size}
    if verify_content:
        with real.open("rb") as stream:
            record["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return record


def _check_file(record: dict, *, executable: bool = False, verify_content: bool = True) -> None:
    expected = record if verify_content else {key: value for key, value in record.items() if key != "sha256"}
    if _file(record["path"], executable=executable, verify_content=verify_content) != expected:
        raise ValueError(f"foundation provider changed: {record['path']}")


def _version(path: str) -> str:
    result = subprocess.run([path, "--version"], check=True, capture_output=True, text=True)
    return result.stdout + result.stderr


def _headers(include: Path) -> list[dict]:
    # SDK trees may contain file aliases; directory aliases would omit a subtree
    # under rglob, so refuse rather than deriving an incomplete cache identity.
    files = []
    for path in sorted(include.rglob("*")):
        if path.is_symlink() and path.is_dir():
            raise ValueError(f"foundation include directory alias is unsupported: {path}")
        if not path.is_dir() and not path.is_file():
            raise ValueError(f"foundation include entry is not a regular file or directory: {path}")
        if path.is_file():
            record = _file(path)
            if not Path(record["realpath"]).is_relative_to(include):
                raise ValueError(f"foundation header escapes selected include directory: {path}")
            files.append(record)
    if not files:
        raise ValueError("foundation include directory is empty")
    return files


def capture_toolchain(*, cuda_home: str, cc: str, cxx: str, wrapper_cc: str,
                      wrapper_cxx: str, contract_path: str, cupti_dir: str | None = None) -> dict:
    """Capture providers during image construction; return a JSON-safe contract.

    ``contract_path`` is the final sealed image manifest (not a temporary build
    path). The aliases and their sibling copy of this module must already exist.
    Both aliases execute the same script and retain their distinct basenames.
    """
    home = Path(cuda_home).resolve(strict=True)
    if not Path(cuda_home).is_absolute() or not Path(contract_path).is_absolute():
        raise ValueError("foundation and contract paths must be absolute")
    include = (home / "include").resolve(strict=True)
    if not include.is_relative_to(home):
        raise ValueError("CUDA include directory escapes selected toolkit")
    headers = _headers(include)
    cuda_header = (include / "cuda.h").read_text()
    match = re.search(r"^#define\s+CUDA_VERSION\s+(\d+)\s*$", cuda_header, re.MULTILINE)
    if match is None:
        raise ValueError("selected CUDA headers lack CUDA_VERSION")
    tools = {}
    versions = {}
    for name in _TOOLS:
        tools[name] = _file(home / "bin" / name, executable=True)
        if not Path(tools[name]["realpath"]).is_relative_to(home):
            raise ValueError(f"selected NVIDIA tool escapes CUDA toolkit: {name}")
        versions[name] = _version(tools[name]["path"])
        if not re.search(r"release \d+\.\d+", versions[name]):
            raise ValueError(f"selected NVIDIA tool has no recognized version: {name}")
    compilers = {"cc": _file(cc, executable=True), "cxx": _file(cxx, executable=True)}
    wrappers = {"cc": _file(wrapper_cc, executable=True), "cxx": _file(wrapper_cxx, executable=True)}
    for kind in compilers:
        if compilers[kind]["realpath"] == wrappers[kind]["realpath"]:
            raise ValueError("foundation compiler wrapper cannot execute itself")
        versions[kind] = _version(compilers[kind]["path"])
    if wrappers["cc"]["sha256"] != wrappers["cxx"]["sha256"]:
        raise ValueError("foundation compiler aliases must use the same owned script")
    if Path(wrapper_cc).name != "tensorfold-cc" or Path(wrapper_cxx).name != "tensorfold-cxx":
        raise ValueError("foundation compiler aliases must retain their cc/cxx role names")
    helpers = {}
    for kind, wrapper in wrappers.items():
        helpers[kind] = _file(Path(wrapper["path"]).parent / "foundation_tools.py")
    if any(value["sha256"] != _file(__file__)["sha256"] for value in helpers.values()):
        raise ValueError("foundation compiler aliases require the identical sibling helper")
    if cupti_dir is None:
        candidates = {path.resolve() for path in (home / "extras/CUPTI/lib64", home / "lib64")
                      if (path / "libcupti.so").is_file()}
        if len(candidates) != 1:
            raise ValueError("select one explicit foundation CUPTI directory")
        cupti = candidates.pop()
    else:
        cupti = Path(cupti_dir).resolve(strict=True)
    if not cupti.is_relative_to(home):
        raise ValueError("CUPTI provider must belong to selected CUDA toolkit")
    libdevice = _file(home / "nvvm/libdevice/libdevice.10.bc")
    cupti_record = _file(cupti / "libcupti.so")
    if any(not Path(record["realpath"]).is_relative_to(home) for record in (libdevice, cupti_record)):
        raise ValueError("selected CUDA library escapes toolkit")
    contract = {
        "schema": _SCHEMA, "cuda_home": str(home), "include": str(include),
        "header_version": int(match[1]), "headers": headers,
        "tools": tools, "versions": versions, "compilers": compilers,
        "wrappers": wrappers, "helpers": helpers,
        "libdevice": libdevice,
        "cupti": cupti_record, "cupti_dir": str(cupti),
        "contract_path": contract_path,
    }
    contract["cache_key"] = _digest(contract)
    return contract


def _validate_contract(contract: dict) -> None:
    if not isinstance(contract, dict) or type(contract.get("schema")) is not int or contract["schema"] != _SCHEMA:
        raise ValueError("unsupported foundation toolchain contract")
    body = {key: value for key, value in contract.items() if key != "cache_key"}
    if contract.get("cache_key") != _digest(body):
        raise ValueError("foundation toolchain contract identity changed")


def toolchain_environment(contract: dict, *, cache_root: str, verify_content: bool = True) -> dict[str, str]:
    """Validate immutable providers once, then return early-publication settings.

    ``cache_root`` must already identify the sealed runtime/architecture and
    transitive compiler environment (Python, Torch, Triton, standard headers,
    compiler subtools and libraries); this local key only adds the selected
    providers. No environment is mutated here. The owner publishes this mapping
    transactionally, before runtime descriptors/compiler caches are initialized.
    SDK-wide content hashing is preparation-only, never on a kernel launch.
    A managed owner may set ``verify_content=False`` only after its sealed
    inventory/package/file-identity verification and under the same immutable
    lifetime contract. That avoids duplicating its SDK scan; selected provider
    paths, sizes, executability and the sealed contract digest are still checked.
    """
    _validate_contract(contract)
    if type(verify_content) is not bool:
        raise ValueError("verify_content must be a boolean")
    if verify_content and _headers(Path(contract["include"])) != contract["headers"]:
        raise ValueError("foundation include tree changed")
    for group in ("tools", "compilers", "wrappers", "helpers"):
        for record in contract[group].values():
            _check_file(record, executable=group != "helpers", verify_content=verify_content)
    for name in ("libdevice", "cupti"):
        _check_file(contract[name], verify_content=verify_content)
    cache = Path(cache_root)
    if not cache.is_absolute():
        raise ValueError("foundation compiler cache root must be absolute")
    cache /= contract["cache_key"]
    env = {key: contract["tools"]["ptxas" if name == "ptxas_blackwell" else name]["path"]
           for name, key in _TRITON_TOOLS.items()}
    env.update({
        CONTRACT_ENV: contract["contract_path"],
        "CC": contract["wrappers"]["cc"]["path"], "CXX": contract["wrappers"]["cxx"]["path"],
        "CUDA_HOME": contract["cuda_home"], "CUDACXX": contract["tools"]["nvcc"]["path"],
        "TRITON_LIBDEVICE_PATH": contract["libdevice"]["path"],
        "TRITON_CUPTI_LIB_PATH": contract["cupti_dir"],
        "TRITON_CUPTI_LIB_BLACKWELL_PATH": contract["cupti_dir"],
        "TRITON_CUDART_PATH": contract["include"], "TRITON_CUDACRT_PATH": contract["include"],
        "TRITON_CACHE_DIR": str(cache / "triton"), "TORCH_EXTENSIONS_DIR": str(cache / "torch-extensions"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"), "CUDA_CACHE_PATH": str(cache / "cuda"),
        "TRITON_DUMP_DIR": str(cache / "triton-dump"), "TRITON_OVERRIDE_DIR": str(cache / "triton-override"),
    })
    return env


def compiler_command(contract: dict, *, role: str, arguments: list[str]) -> list[str]:
    """Prepare an exec argv without interpreting/rewriting compiler arguments.

    The image owner guarantees the admitted SDK tree remains immutable. Each
    compiler process rechecks its compiler, wrapper/helper and primary CUDA
    header identities; it does not rescan the complete SDK on every compile.
    A leading -I works for normal GCC/Clang compilation and their query modes,
    and precedes Triton's later bundled include arguments.
    """
    _validate_contract(contract)
    if role not in ("cc", "cxx"):
        raise ValueError("unknown foundation compiler role")
    for group in ("compilers", "wrappers", "helpers"):
        _check_file(contract[group][role], executable=group != "helpers")
    cuda_header = str(Path(contract["include"]) / "cuda.h")
    primary = [record for record in contract["headers"] if record["path"] == cuda_header]
    if len(primary) != 1:
        raise ValueError("foundation CUDA header identity is missing")
    _check_file(primary[0])
    compiler = contract["compilers"][role]["path"]
    return [compiler, "-I" + contract["include"], *arguments]


def verify_triton_selection(contract: dict, triton) -> dict:
    """Refuse ignored/invalid supported overrides without initializing CUDA."""
    _validate_contract(contract)
    selected = {}
    knobs = triton.knobs
    for name, env in _TRITON_TOOLS.items():
        expected = contract["tools"]["ptxas" if name == "ptxas_blackwell" else name]
        actual = getattr(knobs.nvidia, name).path
        if _file(actual, executable=True) != expected or os.environ.get(env) != expected["path"]:
            raise ValueError(f"Triton selected a different foundation tool: {name}")
        selected[name] = actual
    checks = (
        (knobs.nvidia.libdevice_path, contract["libdevice"]["path"], "libdevice"),
        (knobs.proton.cupti_lib_dir, contract["cupti_dir"], "CUPTI"),
        (knobs.proton.cupti_lib_blackwell_dir, contract["cupti_dir"], "Blackwell CUPTI"),
        (knobs.build.cudart_path, contract["include"], "CUDA headers"),
        (knobs.build.cudacrt_path, contract["include"], "CUDA CRT headers"),
    )
    for actual, expected, name in checks:
        if actual != expected:
            raise ValueError(f"Triton selected a different foundation provider: {name}")
    if knobs.build.impl is not None:
        raise ValueError("foundation compiler wrappers require the standard Triton build implementation")
    cache = os.environ.get("TRITON_CACHE_DIR")
    if cache is None or Path(cache).parts[-2:] != (contract["cache_key"], "triton") or knobs.cache.dir != cache:
        raise ValueError("Triton cache was initialized without the selected foundation identity")
    # Check the actual cached compiler resolver, not only its environment knob.
    from triton.runtime.build import _find_compiler

    for role, language in (("cc", "c"), ("cxx", "c++")):
        expected = contract["wrappers"][role]["path"]
        if _find_compiler(language) != expected or os.environ.get(role.upper()) != expected:
            raise ValueError(f"Triton compiler was initialized before foundation selection: {role}")
    return {"cache_key": contract["cache_key"], "tools": selected,
            "libdevice": contract["libdevice"]["path"], "include": contract["include"]}

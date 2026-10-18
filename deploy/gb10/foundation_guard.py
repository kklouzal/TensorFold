#!/usr/bin/env python3
"""Seal the owned CUDA foundation; install app roles without changing its providers.

Full content audits occur at construction/managed installation, not each launch.
Startup checks sealed metadata, paths, file sizes/modes/mtimes and actual loaded
providers. Separate prefixes prevent normal pip uninstall from owning foundation
files. This is accidental-mutation protection, not a sandbox against root.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path

from foundation_policy import canonical_name, is_accelerator_package

STATE_ENV = "TF_FOUNDATION_STATE"
CACHE_ROOT = "/cache"
MAX_JSON = 64 * 2**20
COMPILER_PACKAGES = r"^(gcc|g\+\+|cpp|binutils|libgcc|libstdc\+\+|libc6|libc-bin|libc-dev|linux-libc-dev|libgomp|libasan|libtsan|libubsan|libquadmath|libatomic|libcc1|libitm)(-|:|$)"


def read_json(path: Path, limit: int = MAX_JSON) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"oversized foundation document: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"foundation document must be an object: {path}")
    return value


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalize_image_mtime(path: Path) -> None:
    """Prepare owned regular files for whole-second image-layer timestamps.

    Run before sealing only. Do not follow venv aliases into system/SDK files;
    exact nanosecond comparisons remain active after the immutable seal.
    """
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        return
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"owned image payload is not a regular file: {path}")
    seconds_ns = info.st_mtime_ns // 10**9 * 10**9
    if info.st_mtime_ns != seconds_ns:
        os.utime(path, ns=(info.st_atime_ns, seconds_ns), follow_symlinks=False)


def identity(path: Path, *, full: bool) -> dict:
    real = path.resolve(strict=True)
    info = real.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"foundation provider is not a regular file: {path}")
    value = {"realpath": str(real), "size": info.st_size, "mtime_ns": info.st_mtime_ns,
             "mode": stat.S_IMODE(info.st_mode)}
    if full:
        value["sha256"] = digest(real)
    return value


def sdk_versions(raw: str, pattern: str) -> dict[str, str]:
    selected = {}
    for line in raw.splitlines():
        name, version = line.split("\t")
        if re.match(pattern, name):
            if name in selected:
                raise ValueError(f"duplicate SDK package: {name}")
            selected[name] = version
    if not selected:
        raise ValueError("the selected base has no protected CUDA SDK packages")
    return selected


def native_packages(pattern: str) -> dict[str, str]:
    result = subprocess.run(["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\n"],
                            check=True, capture_output=True, text=True)
    return sdk_versions(result.stdout, pattern)


def native_files(packages: dict) -> set[Path]:
    result = subprocess.run(["dpkg-query", "-L", *packages], check=True, capture_output=True, text=True)
    return {Path(line) for line in result.stdout.splitlines() if line.startswith("/")
            and Path(line).is_file()}


def tree_files(root: Path) -> set[Path]:
    return {path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts}


def site_directory(prefix: Path) -> Path:
    return prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def distributions(paths: list[str]) -> dict[str, list]:
    found: dict[str, list] = {}
    for dist in metadata.distributions(path=paths):
        raw_name = dist.metadata.get("Name")
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("installed foundation/application distribution has no name")
        name = canonical_name(raw_name)
        found.setdefault(name, []).append(dist)
    return found


def check_packages(manifest: dict, foundation: Path, application: Path) -> None:
    protected = manifest["roles"]["foundation"]
    own = distributions([str(site_directory(foundation))])
    active = distributions([str(site_directory(application))])
    for name, pin in protected.items():
        matches = own.get(name, [])
        if len(matches) != 1 or matches[0].version != pin["version"]:
            raise ValueError(f"protected foundation distribution differs: {name}")
        if name in active:
            raise ValueError(f"application prefix shadows protected distribution: {name}")
    for name in own.keys() | active.keys():
        if is_accelerator_package(name) and name not in protected:
            raise ValueError(f"unapproved accelerator distribution: {name}")
    allowed = manifest["roles"]["runtime"] | manifest["roles"]["verification"]
    for name, rows in active.items():
        if name == "tensorfold":
            continue
        if name not in allowed or len(rows) != 1 or rows[0].version != allowed[name]["version"]:
            raise ValueError(f"unapproved application distribution/version: {name}")
    # pip is the trusted venv bootstrap, not a resolver-selected CUDA provider.
    if set(own) - set(protected) - {"pip"}:
        raise ValueError(f"unsealed foundation distributions: {sorted(set(own) - set(protected) - {'pip'})}")



def write_json(path: Path, value: dict) -> None:
    # Build-owned destination; atomic publication does not leave partial state.
    temporary = path.with_name(path.name + ".new")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.chmod(0o644)
    normalize_image_mtime(temporary)
    temporary.replace(path)


def check_base_sdk(manifest: dict, baseline: Path, pattern: str) -> dict[str, str]:
    expected = sdk_versions(baseline.read_text(), pattern)
    current = native_packages(pattern)
    for addon in manifest.get("native_addons", []):
        matching = [name for name in current if name.split(":")[0] == addon["package"]]
        if re.match(pattern, addon["package"]):
            if len(matching) != 1 or current[matching[0]] != addon["version"]:
                raise ValueError("selected native addon version differs")
            if matching[0] in expected:
                raise ValueError("native addon may not replace a base SDK package")
            expected[matching[0]] = addon["version"]
    if current != expected:
        raise ValueError("base CUDA SDK packages changed during build preparation")
    subprocess.run(["sha256sum", "--check", "--status", str(baseline.with_name("base-sdk.sha256"))], check=True)
    for line in baseline.with_name("base-sdk-links.tsv").read_text().splitlines():
        name, target = line.split("\t")
        if not Path(name).is_symlink() or os.readlink(name) != target:  # noqa: PTH115 - preserve raw link text.
            raise ValueError(f"base CUDA SDK alias changed: {name}")
    return expected


def install_native_addons(manifest: dict, *, artifacts: Path, sdk_baseline: Path, pattern: str) -> None:
    # Exact local Debian artifacts; dpkg does not resolve/download another SDK.
    # A missing dependency or install failure invalidates this build layer.
    current = native_packages(pattern)
    if current != sdk_versions(sdk_baseline.read_text(), pattern):
        raise ValueError("native addons require the unchanged base SDK")
    packages = []
    for item in manifest.get("native_addons", []):
        path = artifacts / item["artifact"]
        if path.parent != artifacts or digest(path) != item["sha256"]:
            raise ValueError("native addon artifact identity differs")
        fields = subprocess.run(["dpkg-deb", "--field", str(path), "Package", "Version"],
                                check=True, capture_output=True, text=True).stdout
        values = dict(line.split(": ", 1) for line in fields.splitlines())
        if values != {"Package": item["package"], "Version": item["version"]}:
            raise ValueError("native addon package metadata differs")
        present = subprocess.run(["dpkg-query", "-W", "-f=${db:Status-Status}", item["package"]],
                                 check=False, capture_output=True, text=True)
        if present.returncode == 0 and present.stdout.strip() == "installed":
            raise ValueError("native addon cannot replace an installed base package")
        packages.append(str(path))
    if packages:
        subprocess.run(["dpkg", "--install", *packages], check=True)
        subprocess.run(["apt-mark", "hold", *(item["package"] for item in manifest["native_addons"])], check=True)
    check_base_sdk(manifest, sdk_baseline, pattern)


def select_nccl(manifest: dict, foundation: Path, native: set[Path]) -> tuple[str, int | None]:
    provider = manifest.get("nccl_provider", "native")
    if provider == "native":
        candidates = {path.resolve(strict=True) for path in native if re.fullmatch(r"libnccl\.so(?:\.\d+)*", path.name)}
    elif provider == "nvidia-nccl-cu13" and provider in manifest["roles"]["foundation"]:
        candidates = {(site_directory(foundation) / "nvidia/nccl/lib/libnccl.so.2").resolve(strict=True)}
        if not all(path.is_relative_to(foundation) for path in candidates):
            raise ValueError("selected NCCL wheel provider escapes foundation")
    else:
        raise ValueError("selected NCCL provider is outside protected foundation")
    if len(candidates) != 1:
        raise ValueError("select exactly one physical foundation NCCL provider")
    expected = manifest["roles"]["foundation"].get("nvidia-nccl-cu13")
    code = None
    if expected is not None:
        match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?", expected["version"])
        if match is None:
            raise ValueError("NCCL distribution version cannot be verified against native API")
        major, minor, patch = map(int, match.groups())
        code = major * 10000 + minor * 100 + patch
    return str(candidates.pop()), code


def seal(*, manifest_path: Path, foundation: Path, application: Path, sdk_baseline: Path,
         pattern_path: Path, state_path: Path, artifacts: Path, toolchain: dict) -> dict:
    manifest = read_json(manifest_path, 2 * 2**20)
    foundation, application = foundation.resolve(strict=True), application.resolve(strict=True)
    if foundation == application or foundation.is_relative_to(application) or application.is_relative_to(foundation):
        raise ValueError("foundation and application require separate, nonnested prefixes")
    pattern = pattern_path.read_text().strip()
    before = check_base_sdk(manifest, sdk_baseline, pattern)
    check_packages(manifest, foundation, application)
    site = site_directory(application)
    site.mkdir(parents=True, exist_ok=True)
    activation = site / "00_tensorfold_foundation.pth"
    bootstrap = site / "tensorfold_foundation_bootstrap.py"
    runtime_helper = site / "tensorfold_foundation_runtime.py"
    policy_helper = site / "foundation_policy.py"
    shutil.copyfile(Path(__file__).with_name("foundation_runtime.py"), runtime_helper)
    shutil.copyfile(Path(__file__).with_name("foundation_policy.py"), policy_helper)
    shutil.copyfile(Path(__file__).with_name("foundation_bootstrap.py"), bootstrap)
    hook = "try:\n import tensorfold_foundation_bootstrap\nexcept BaseException:\n import os; os._exit(78)"
    activation.write_text(f"import sys; sys.path.insert(0, {str(site_directory(foundation))!r}); exec({hook!r})\n")
    # --no-deps managed role installs own only app files. Constraints also guard
    # an ordinary resolver invocation; deliberate --isolated/--no-deps bypasses
    # remain detectable by origin/version/provider checks.
    constraint = manifest_path.with_name("foundation-constraints.txt")
    config = application / "pip.conf"
    config.write_text(f"[global]\nrequire-virtualenv = true\nconstraint = {constraint}\n")
    compiler_packages = native_packages(COMPILER_PACKAGES)
    compiler_files = native_files(compiler_packages)
    addons = {item["package"]: item["version"] for item in manifest.get("native_addons", [])}
    native = native_files(before | addons)
    nccl_library, nccl_version = select_nccl(manifest, foundation, native)
    addon_artifacts = {artifacts / item["artifact"]: item["sha256"] for item in manifest.get("native_addons", [])}
    owned_files = set(addon_artifacts) | tree_files(foundation) | {manifest_path, constraint, activation, config, bootstrap, runtime_helper, policy_helper, sdk_baseline, sdk_baseline.with_name("base-sdk.sha256"), sdk_baseline.with_name("base-sdk-links.tsv"), pattern_path}
    for key in ("wrappers", "helpers"):
        owned_files.update(Path(row["path"]) for row in toolchain[key].values())
    for path in sorted(owned_files):
        normalize_image_mtime(path)
    files = owned_files | native | compiler_files
    for key in ("tools", "compilers", "wrappers", "helpers"):
        files.update(Path(row["path"]) for row in toolchain[key].values())
    files.update(Path(row["path"]) for row in toolchain["headers"])
    files.update(Path(toolchain[key]["path"]) for key in ("libdevice", "cupti"))
    state = {"schema_version": 1, "manifest": manifest, "foundation": str(foundation),
             "application": str(application), "state_path": str(state_path), "sdk_pattern": pattern, "sdk_packages": before,
             "sdk_links": dict(line.split("\t") for line in sdk_baseline.with_name("base-sdk-links.tsv").read_text().splitlines()),
             "native_files": sorted(str(path.resolve(strict=True)) for path in native),
             "compiler_packages": compiler_packages, "toolchain": toolchain,
             "nccl_library": nccl_library, "nccl_version": nccl_version,
             "files": {str(path): identity(path, full=True) for path in sorted(files)}}
    if any(state["files"][str(path)]["sha256"] != expected for path, expected in addon_artifacts.items()):
        raise ValueError("native addon artifact changed before sealing")
    from foundation_runtime import capture_runtime
    state["runtime"] = capture_runtime(foundation=foundation, manifest=manifest, native_files=native | compiler_files,
        sealed_files=state["files"], nccl_library=nccl_library, cupti_library=toolchain["cupti"]["path"])
    profile_path = state_path.with_name("runtime-profile.json")
    state["runtime_profile_path"] = str(profile_path)
    write_json(profile_path, state["runtime"])
    state["files"][str(profile_path)] = identity(profile_path, full=True)
    cpu_driver = (Path(toolchain["cuda_home"]) / "compat/libcuda.so.1").resolve(strict=True)
    if str(cpu_driver) not in state["native_files"]:
        raise ValueError("CPU runtime validation requires the sealed compatibility driver for load-only admission")
    state["cpu_driver"] = str(cpu_driver)
    state["cache_key"] = hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    publish_state(state, state_path)
    return state


def publish_state(state: dict, path: Path) -> None:
    """Build-exclusive publication; a partial failure invalidates that image layer."""
    from foundation_tools import toolchain_environment
    values = toolchain_environment(state["toolchain"], cache_root="/cache/" + state["cache_key"], verify_content=False)
    values.update(TF_FOUNDATION_STATE=str(path), TF_NCCL_LIB=state["nccl_library"],
                  TF_FOUNDATION_RUNTIME_PROFILE=state["runtime_profile_path"], TF_FOUNDATION_CPU_DRIVER=state["cpu_driver"],
                  TORCH_USE_RTLD_GLOBAL="1", CUDA_PATH=state["toolchain"]["cuda_home"],
                  PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    environment_file = site_directory(Path(state["application"])) / "tensorfold-foundation-environment.json"
    write_json(environment_file, values)
    state["files"][str(environment_file)] = identity(environment_file, full=True)
    state["state_digest"] = state_digest(state)
    write_json(path, state)


def state_digest(state: dict) -> str:
    body = {key: value for key, value in state.items() if key != "state_digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify(state: dict, *, full: bool = False) -> None:
    if type(state.get("schema_version")) is not int or state["schema_version"] != 1:
        raise ValueError("unsupported foundation state")
    if state.get("state_digest") != state_digest(state):
        raise ValueError("foundation state identity changed")
    foundation, application = Path(state["foundation"]), Path(state["application"])
    check_packages(state["manifest"], foundation, application)
    if native_packages(state["sdk_pattern"]) != state["sdk_packages"]:
        raise ValueError("protected native CUDA SDK package versions changed")
    if native_packages(COMPILER_PACKAGES) != state["compiler_packages"]:
        raise ValueError("protected compiler dependency packages changed")
    for name, target in state["sdk_links"].items():
        if not Path(name).is_symlink() or os.readlink(name) != target:  # noqa: PTH115 - preserve raw link text.
            raise ValueError(f"protected base SDK alias changed: {name}")
    for name, expected in state["files"].items():
        actual = identity(Path(name), full=full)
        reference = expected if full else {key: value for key, value in expected.items() if key != "sha256"}
        if actual != reference:
            raise ValueError(f"protected foundation file changed: {name}")
    if full and not tree_files(foundation) <= {Path(path) for path in state["files"]}:
        raise ValueError("unsealed files were added to the foundation prefix")


def environment(state: dict) -> dict[str, str]:
    from foundation_tools import toolchain_environment
    verify(state)
    values = toolchain_environment(state["toolchain"], cache_root=str(Path(CACHE_ROOT) / state["cache_key"]), verify_content=False)
    override = os.environ.get("TF_NCCL_LIB")
    if override and Path(override).resolve(strict=True) != Path(state["nccl_library"]):
        raise ValueError("TF_NCCL_LIB conflicts with selected foundation provider")
    values.update(TF_NCCL_LIB=state["nccl_library"], TF_FOUNDATION_RUNTIME_PROFILE=state["runtime_profile_path"],
                  TF_FOUNDATION_CPU_DRIVER=state["cpu_driver"], TORCH_USE_RTLD_GLOBAL="1",
                  CUDA_PATH=state["toolchain"]["cuda_home"], PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    cupti = Path(state["toolchain"]["cupti"]["path"]).resolve(strict=True)
    requested_nvtx = os.environ.get("NVTX_INJECTION64_PATH")
    if requested_nvtx:
        if Path(requested_nvtx).resolve(strict=True) != cupti:
            raise ValueError("NVTX_INJECTION64_PATH conflicts with selected foundation CUPTI")
        values["NVTX_INJECTION64_PATH"] = str(cupti)
    libraries = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part]
    compat = str(Path(state["toolchain"]["cuda_home"]) / "compat")
    position = 1 if libraries[:1] == [compat] else 0
    libraries.insert(position, state["toolchain"]["cupti_dir"])
    values["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(libraries))
    values["PATH"] = state["toolchain"]["cuda_home"] + "/bin:" + os.environ.get("PATH", "")
    return values


def vision_proof(torch, torchvision) -> dict:
    """Exact official0.30 wheel load/schema contract; no GPU image execution."""
    directory = Path(torchvision.__file__).resolve().parent
    required = {directory / "_C_stable.so", directory / "image_stable.so"}
    loaded = {Path(path).resolve() for path in torch.ops.loaded_libraries}
    if not required <= loaded:
        raise ValueError("TorchVision native/image extensions did not actually load")
    schemas = {name: str(getattr(torch.ops.image, name).default._schema)
               for name in ("decode_jpegs_cuda", "encode_jpegs_cuda")}
    return {"loaded_native_paths": sorted(str(path) for path in required),
            "registered_cuda_image_schemas": schemas, "gpu_image_execution": "unrun"}


def native_proof(state: dict) -> dict:
    """Inspect the same-process prepared profile before and after actual consumers."""
    from foundation_runtime import mapped_runtime
    bootstrap = sys.modules.get("tensorfold_foundation_bootstrap")
    expected = state["files"][state["runtime_profile_path"]]["sha256"]
    if (bootstrap is None or not bootstrap.RUNTIME_OWNERS or expected != bootstrap.ACTIVE_PROFILE_SHA256):
        raise ValueError("native proof requires same-process foundation activation through guard run or the CUDA launcher")
    before = mapped_runtime(state["runtime"])
    import torch
    import triton
    from foundation_tools import verify_triton_selection
    root = Path(state["foundation"])
    for module in (torch, triton):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError(f"native module does not come from foundation: {module.__name__}")
    tools = verify_triton_selection(state["toolchain"], triton)
    if Path(os.environ.get("TF_NCCL_LIB", "")).resolve(strict=True) != Path(state["nccl_library"]):
        raise ValueError("TF_NCCL_LIB conflicts with selected foundation provider")
    import ctypes

    from tensorfold.cuda.comm import _library
    nccl = _library()
    nccl.ncclGetVersion.argtypes, nccl.ncclGetVersion.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_int
    version = ctypes.c_int()
    status = nccl.ncclGetVersion(ctypes.byref(version))
    if status != 0 or version.value <= 0 or (state["nccl_version"] is not None and version.value != state["nccl_version"]):
        raise ValueError(f"selected NCCL native API version differs: status={status}, version={version.value}")
    core = mapped_runtime(state["runtime"])
    import torchvision
    if not Path(torchvision.__file__).resolve().is_relative_to(root):
        raise ValueError("TorchVision does not come from protected foundation")
    vision = vision_proof(torch, torchvision)
    providers = mapped_runtime(state["runtime"], allow_vision=True)
    if torch.cuda.is_initialized():
        raise ValueError("foundation startup proof unexpectedly initialized CUDA")
    static = state["manifest"]["roles"]["foundation"].get("cuda-bindings")
    return {"torch": torch.__version__, "triton": triton.__version__, "providers_before_torch": before,
            "providers_after_torch": core, "providers": providers, "loaded_sdk": [row["path"] for row in providers],
            "tools": tools, "torchvision": torchvision.__version__, "vision": vision,
            "nccl_library": state["nccl_library"], "nccl_version": version.value, "cuda_initialized": False,
            "driver_role": bootstrap.DRIVER_ROLE,
            "static_runtime_exceptions": {} if static is None else {"cuda-bindings": {
                "version": static["version"], "url": static["url"], "sha256": static["sha256"],
                "authority": "official static CUDART has no supported replacement; user permits protected component runtime"}}}


def role_lines(manifest: dict, role: str) -> list[str]:
    return sorted(f"{name} @ {pin['url']} --hash=sha256:{pin['sha256']}"
                  for name, pin in manifest["roles"][role].items())


def run_install(state: dict, argv: list[str]) -> None:
    retained_primary = retained_audit = diagnostic_failure = combined = None
    prior_cause = prior_context = None
    prior_suppress = False
    try:
        subprocess.run(argv, check=True)
    except BaseException as primary:
        retained_primary = primary
        prior_cause, prior_context, prior_suppress = primary.__cause__, primary.__context__, primary.__suppress_context__
        try:
            verify(state, full=True)
        except BaseException as audit_error:
            retained_audit = audit_error
            try:
                combined = BaseExceptionGroup("installation failed and foundation audit failed", [retained_primary, retained_audit])
            except BaseException as error:
                diagnostic_failure = error
            if diagnostic_failure is not None:
                retained_primary.__cause__ = prior_cause
                retained_primary.__context__ = prior_context
                retained_primary.__suppress_context__ = prior_suppress
                raise retained_primary  # noqa: B904 - retain the primary native cause exactly.
            raise combined from None
        raise
    verify(state, full=True)


def install_role(state: dict, *, role: str, lock: Path) -> None:
    verify(state, full=True)
    lines = sorted(line.strip() for line in lock.read_text().splitlines() if line.strip() and not line.startswith("#"))
    if lines != role_lines(state["manifest"], role):
        raise ValueError("installation lock differs from the sealed, disjoint dependency role")
    run_install(state, [str(Path(state["application"]) / "bin/python"), "-m", "pip", "install",
                        "--require-hashes", "--no-deps", "-r", str(lock)])


def install_application(state: dict, wheel: Path) -> None:
    verify(state, full=True)
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        candidates = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(candidates) != 1:
            raise ValueError("application wheel requires exactly one METADATA")
        package = BytesParser().parsebytes(archive.read(candidates[0]))
        if canonical_name(package["Name"]) != "tensorfold":
            raise ValueError("only the owned TensorFold wheel is accepted as application")
        if any(name.startswith(("torch/", "triton/", "nvidia/")) or name.startswith("/")
               or ".." in Path(name).parts for name in names):
            raise ValueError("application wheel contains a protected or escaping payload")
    run_install(state, [str(Path(state["application"]) / "bin/python"), "-m", "pip", "install",
                        "--no-deps", str(wheel)])
    state["foundation_cache_key"] = state.get("foundation_cache_key", state["cache_key"])
    state["application_wheel_sha256"] = digest(wheel)
    state["cache_key"] = hashlib.sha256(json.dumps({"foundation": state["foundation_cache_key"],
        "application": state["application_wheel_sha256"]}, sort_keys=True).encode()).hexdigest()
    publish_state(state, Path(state["state_path"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("seal", "verify", "audit", "environment", "native-proof", "install", "application", "native-addons", "run", "checked-run"))
    parser.add_argument("--state", type=Path, default=Path(os.environ.get(STATE_ENV, "/opt/tensorfold-pins/foundation-state.json")))
    parser.add_argument("--manifest", type=Path, default=Path("/opt/tensorfold-pins/foundation-manifest.json"))
    parser.add_argument("--foundation", type=Path, default=Path("/opt/tensorfold-foundation"))
    parser.add_argument("--application", type=Path, default=Path("/opt/tensorfold-venv"))
    parser.add_argument("--sdk-baseline", type=Path, default=Path("/opt/tensorfold-pins/base-sdk.tsv"))
    parser.add_argument("--sdk-pattern", type=Path, default=Path("/opt/tensorfold-pins/foundation-sdk-pattern.txt"))
    parser.add_argument("--artifacts", type=Path, default=Path("/opt/tensorfold-foundation-artifacts"))
    parser.add_argument("--role", choices=("runtime", "verification"))
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--wheel", type=Path)
    raw = sys.argv[1:]
    split = raw.index("--") if "--" in raw else len(raw)
    args = parser.parse_args(raw[:split])
    command_argv = raw[split + 1:] if split < len(raw) else []
    if args.command == "native-addons":
        install_native_addons(read_json(args.manifest), artifacts=args.artifacts,
                              sdk_baseline=args.sdk_baseline, pattern=args.sdk_pattern.read_text().strip())
    elif args.command == "seal":
        from foundation_tools import capture_toolchain
        toolchain = capture_toolchain(cuda_home="/usr/local/cuda", cc="/usr/bin/gcc", cxx="/usr/bin/g++",
                    wrapper_cc="/opt/tensorfold-foundation-tools/tensorfold-cc",
                    wrapper_cxx="/opt/tensorfold-foundation-tools/tensorfold-cxx", contract_path=str(args.state))
        seal(manifest_path=args.manifest, foundation=args.foundation, application=args.application,
             sdk_baseline=args.sdk_baseline, pattern_path=args.sdk_pattern, state_path=args.state,
             artifacts=args.artifacts, toolchain=toolchain)
    else:
        state = read_json(args.state)
        if args.command in ("run", "checked-run"):
            argv = command_argv
            if not argv:
                parser.error("run requires an executable after --")
            values = dict(os.environ, **environment(state))
            values.pop("TF_FOUNDATION_RUNTIME_PHASE", None)
            if args.command == "checked-run":
                expected = json.loads(values["TF_CUDA_DRIVER_PROOF"])
                selected = subprocess.run([sys.executable, str(Path(__file__).with_name("cuda_driver_probe.py")),
                    "--library", "libcuda.so.1", "--required-api", str(state["toolchain"]["header_version"])],
                    env=values, check=True, capture_output=True, text=True, timeout=35)
                actual = json.loads(selected.stdout)
                if actual["library"] != expected["library"]:
                    raise ValueError("post-entrypoint driver provider differs from admitted CUDA proof")
                values["TF_CUDA_DRIVER_PROOF"] = json.dumps(dict(actual, mode=expected["mode"]))
                values["TF_FOUNDATION_RUNTIME_PHASE"] = "cuda"
                subprocess.run([sys.executable, str(Path(__file__).resolve()), "native-proof"],
                               env=values, check=True, stdout=subprocess.DEVNULL)
            else:
                if any(Path("/dev").glob("nvidia[0-9]*")):
                    raise RuntimeError("CPU guard run has NVIDIA devices; use the CUDA launcher for actual driver admission")
                values["TF_FOUNDATION_RUNTIME_PHASE"] = "cpu"
            os.execvpe(argv[0], argv, values)
        elif args.command in ("verify", "audit"):
            verify(state, full=args.command == "audit")
        elif args.command == "environment":
            print(json.dumps(environment(state)))
        elif args.command == "native-proof":
            verify(state)
            print(json.dumps(native_proof(state)))
        elif args.command == "install":
            if args.role is None or args.lock is None:
                parser.error("install requires --role and --lock")
            install_role(state, role=args.role, lock=args.lock)
        elif args.command == "application":
            if args.wheel is None:
                parser.error("application requires --wheel")
            install_application(state, args.wheel)


if __name__ == "__main__":
    main()

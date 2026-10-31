#!/usr/bin/env python3
"""Additive, independently sealed calibration role; never changes the SDK seal.

Preparation consumes a real native-platform pip report, not a hand-written lock.
Installation uses that report's complete dependency solution and hashed wheels.
The producer prefix is append-only on sys.path after managed SDK activation.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.machinery import PathFinder
import importlib.metadata as md
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
from urllib.parse import unquote, urlsplit
import zipfile

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

MODEL_OPT_COMMIT = "a851b028ecbd5f843eb1fc05b93cc2d0abed0de3"
MODEL_OPT_SOURCE_TYPE = "incremental-git-bundle"
MODEL_OPT_REPOSITORY = "https://github.com/kklouzal/TensorFold"
MODEL_OPT_UPSTREAM_REPOSITORY = "https://github.com/NVIDIA/Model-Optimizer"
MODEL_OPT_UPSTREAM_COMMIT = "e352caca3c36236fc15adea2627a9d42788380c4"
MODEL_OPT_BUNDLE_PATH = "deploy/gb10/calibration/modelopt-fork/modelopt-triton39.bundle"
MODEL_OPT_BUNDLE_SHA256 = "adfee7d91e371c90eebe8c4da33c4eb1572760322187d8f67662990c34a9221d"
BASE = "sha256:2a772b064f593ae9a4436562c4c26ceaa60cec55709992cbf966b74b16bc649c"
GUARD = "/opt/tensorfold-pins/foundation_guard.py"
PURE_BINDINGS = {"nvidia-modelopt", "nvidia-ml-py"}
BUILD_TOOLS = {"setuptools-scm", "vcs-versioning"}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    if path.stat().st_size > 64 * 2**20:
        raise ValueError("oversized calibration metadata")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("calibration metadata must be an object")
    return value


def write(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def unique_distributions(paths=None):
    """Deduplicate repeated search paths, never distinct metadata installations."""
    candidates = md.distributions() if paths is None else md.distributions(path=paths)
    seen = set()
    for dist in candidates:
        origins = [Path(dist.locate_file(member)).resolve(strict=True) for member in dist.files or []
                   if len(member.parts) == 2 and member.name in {"METADATA", "PKG-INFO"}
                   and member.parent.name.endswith((".dist-info", ".egg-info"))]
        if len(origins) != 1:
            raise ValueError("distribution must have one recorded metadata origin")
        if origins[0] not in seen:
            seen.add(origins[0])
            yield dist


def installed(paths=None) -> dict:
    result = {}
    for dist in unique_distributions(paths):
        name = canonicalize_name(dist.metadata["Name"])
        if name in result:
            raise ValueError(f"duplicate active distribution: {name}")
        result[name] = {"version": dist.version, "requires": dist.requires or []}
    return result


def baseline_files() -> dict:
    files = {}
    for dist in unique_distributions():
        for member in dist.files or []:
            path = Path(dist.locate_file(member))
            if path.is_file():
                files[str(path.resolve())] = sha(path)
    return files


def snapshot(destination: Path) -> None:
    """Capture active baseline and constrain EVERY installed version."""
    destination.mkdir(exist_ok=False)
    baseline = {"schema_version": 1, "base": BASE, "environment": default_environment(),
                "packages": installed(), "files": baseline_files()}
    if (baseline["environment"]["platform_machine"] != "aarch64"
            or baseline["packages"]["transformers"]["version"] != "5.17.0"):
        raise ValueError("this producer candidate requires the exact ARM64/HF5.17 baseline")
    write(destination / "baseline.json", baseline)
    (destination / "constraints.txt").write_text("".join(
        f"{name}=={row['version']}\n" for name, row in sorted(baseline["packages"].items())))


def closure(packages: dict, environment: dict) -> None:
    """Check default and transitively requested extras, including baseline edges."""
    pending = [(name, "") for name in packages]
    visited = set()
    while pending:
        name, extra = pending.pop()
        if (name, extra) in visited:
            continue
        visited.add((name, extra))
        row = packages[name]
        for raw in row["requires"]:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({**environment, "extra": extra}):
                continue
            dependency_name = canonicalize_name(requirement.name)
            dependency = packages.get(dependency_name)
            if (dependency is None or requirement.url or not requirement.specifier.contains(
                    Version(dependency["version"]), prereleases=True)):
                raise ValueError(f"unsatisfied dependency: {name}: {raw}")
            pending.extend((dependency_name, requested) for requested in requirement.extras)


def provider_name(name: str) -> bool:
    return (name in {"torch", "torchvision", "torchaudio", "triton", "cupti-python"}
            or name.startswith(("nvidia-", "cuda-", "triton-", "pytorch-triton")))


def wheel_record(path: Path) -> dict:
    from email.parser import BytesParser
    _, _, _, tags = parse_wheel_filename(path.name)
    if not tags.intersection(sys_tags()):
        raise ValueError("wheel does not support the actual native interpreter/platform")
    with zipfile.ZipFile(path) as wheel:
        members = wheel.namelist()
        metadata = [name for name in members if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("wheel must contain one METADATA")
        payload = BytesParser().parsebytes(wheel.read(metadata[0]))
        python_requirement = payload.get("Requires-Python")
        if python_requirement:
            from packaging.specifiers import SpecifierSet
            if not SpecifierSet(python_requirement).contains(default_environment()["python_full_version"]):
                raise ValueError("wheel does not support the actual Python version")
        for name in members:
            pieces = Path(name).parts
            if (name.startswith("/") or ".." in pieces or name.endswith(".pth")
                    or Path(name).name in {"sitecustomize.py", "usercustomize.py"}):
                raise ValueError(f"unsafe prefix activation/member: {name}")
            if re.match(r"lib(?:cuda|nvidia|cudart|cublas|cudnn|nvrtc|nvJitLink|nccl|cupti|stdc\+\+|gcc_s).*\.so", Path(name).name):
                raise ValueError(f"replacement native provider in wheel: {name}")
    name = canonicalize_name(payload["Name"])
    if provider_name(name) and name not in PURE_BINDINGS:
        raise ValueError(f"new accelerator/provider package is not permitted: {name}")
    return {"name": name, "version": payload["Version"],
            "requires": payload.get_all("Requires-Dist", []), "artifact": path.name,
            "sha256": sha(path), "tags": sorted(map(str, tags))}


def validate_role(kind: str, additions: dict) -> None:
    if kind == "producer":
        if "nvidia-modelopt" not in additions:
            raise ValueError("ModelOpt must be a new producer dependency")
    elif kind == "build-tools":
        if "setuptools-scm" not in additions or set(additions) - BUILD_TOOLS:
            raise ValueError("build-tools role permits only setuptools-scm and reviewed vcs-versioning closure")
        if Version(additions["setuptools-scm"]["version"]) < Version("8"):
            raise ValueError("ModelOpt requires setuptools-scm>=8")
    else:
        raise ValueError("unknown additive role kind")


def prepare(baseline_path: Path, report_path: Path, wheels: Path, provenance: Path,
            output: Path, kind: str = "producer") -> None:
    baseline, report, origin = read(baseline_path), read(report_path), read(provenance)
    if (baseline.get("base") != BASE or origin.get("source_commit") != MODEL_OPT_COMMIT
            or not re.fullmatch(r"[0-9a-f]{64}", origin.get("source_archive_sha256", ""))):
        raise ValueError("producer source/base provenance is not pinned")
    if report.get("environment") != baseline["environment"]:
        raise ValueError("resolver must run on the exact baseline platform/Python")
    if kind not in {"producer", "build-tools"}:
        raise ValueError("unknown additive role kind")
    additions = {}
    for item in report["install"]:
        meta = item["metadata"]
        name = canonicalize_name(meta["name"])
        if name in baseline["packages"] or name in additions:
            raise ValueError(f"resolver changes/reinstalls an existing package: {name}")
        download = item["download_info"]
        parsed = urlsplit(download["url"])
        if parsed.scheme not in {"https", "file"} or parsed.username or parsed.password:
            raise ValueError("wheel origin must be authenticated HTTPS or the pinned local ModelOpt wheel")
        filename = Path(unquote(parsed.path)).name
        if parsed.scheme == "file" and (name != "nvidia-modelopt" or kind != "producer"):
            raise ValueError("only the source-pinned ModelOpt wheel may be local")
        row = wheel_record(wheels / filename)
        expected = download["archive_info"]["hashes"]["sha256"]
        if row["name"] != name or row["version"] != meta["version"] or row["sha256"] != expected:
            raise ValueError("downloaded wheel differs from the full resolver report")
        if name == "nvidia-modelopt" and row["sha256"] != origin.get("wheel_sha256"):
            raise ValueError("ModelOpt built wheel differs from source provenance")
        additions[name] = row
    validate_role(kind, additions)
    packages = baseline["packages"] | additions
    closure(packages, baseline["environment"])
    output.mkdir(exist_ok=False)
    write(output / "manifest.json", {"schema_version": 1, "kind": kind, "baseline": baseline,
          "source": origin, "resolver_sha256": sha(report_path), "additions": additions})
    (output / "requirements.lock").write_text("".join(
        f"{name}=={row['version']} --hash=sha256:{row['sha256']}\n"
        for name, row in sorted(additions.items())))


def tree(prefix: Path) -> dict:
    files = {}
    for path in sorted(prefix.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"producer prefix must not contain aliases: {path}")
        if path.is_file():
            files[str(path.relative_to(prefix))] = sha(path)
    return files


def seal_inputs(prefix: Path, manifest_path: Path) -> dict:
    tools = Path(__file__).resolve().parent
    result = {"manifest_sha256": sha(manifest_path), "files": tree(prefix),
              "tools": tree(tools), "requirements_sha256": sha(manifest_path.with_name("requirements.lock"))}
    git_state = Path("/opt/tensorfold-calibration/git-state.json")
    if git_state.exists():
        result["git_state_sha256"] = sha(git_state)
    return result


def check_wheels(manifest: dict, wheels: Path) -> None:
    if installed() != manifest["baseline"]["packages"]:
        raise ValueError("build baseline differs from resolver baseline")
    closure(manifest["baseline"]["packages"] | manifest["additions"],
            manifest["baseline"]["environment"])
    validate_role(manifest["kind"], manifest["additions"])
    for name, row in manifest["additions"].items():
        artifact = row["artifact"]
        if Path(artifact).name != artifact or name in manifest["baseline"]["packages"]:
            raise ValueError("unsafe or overlapping producer wheel")
        if wheel_record(wheels / artifact) != row:
            raise ValueError(f"producer wheel/metadata changed: {name}")


def audit(prefix: Path, manifest: dict) -> None:
    subprocess.run([sys.executable, GUARD, "audit"], check=True)
    if installed() != manifest["baseline"]["packages"]:
        raise ValueError("baseline distributions or their metadata changed")
    if baseline_files() != manifest["baseline"]["files"]:
        raise ValueError("baseline distribution payloads changed")
    new = installed([str(prefix)])
    expected = {name: {"version": row["version"], "requires": row["requires"]}
                for name, row in manifest["additions"].items()}
    if new != expected:
        raise ValueError("producer prefix package metadata differs")
    closure(manifest["baseline"]["packages"] | new, manifest["baseline"]["environment"])
    baseline_members = set()
    for dist in unique_distributions():
        baseline_members.update(str(path) for path in (dist.files or []))
    native_helpers = None
    for path in prefix.rglob("*"):
        relative = str(path.relative_to(prefix))
        if path.is_file() and (relative in baseline_members or path.suffix == ".pth"
                or path.name in {"sitecustomize.py", "usercustomize.py"}):
            raise ValueError(f"producer shadows or activates baseline code: {relative}")
        if path.is_file():
            with path.open("rb") as stream:
                magic = stream.read(4)
            if magic == b"\x7fELF":
                if native_helpers is None:
                    from tensorfold_foundation_runtime import elf, provider_kind
                    native_helpers = elf, provider_kind
                elf, provider_kind = native_helpers
                soname = elf(path)["soname"]
                if provider_kind(path.name) or (soname and provider_kind(soname)):
                    raise ValueError(f"producer contains a replacement native provider: {relative}")


def provider_proof() -> None:
    state = read(Path(os.environ["TF_FOUNDATION_STATE"]))
    bootstrap = sys.modules.get("tensorfold_foundation_bootstrap")
    expected = state["files"][state["runtime_profile_path"]]["sha256"]
    if (bootstrap is None or not bootstrap.RUNTIME_OWNERS
            or bootstrap.ACTIVE_PROFILE_SHA256 != expected):
        raise RuntimeError("producer lacks actual same-process managed foundation activation")
    from tensorfold_foundation_runtime import mapped_runtime
    mapped_runtime(state["runtime"], allow_vision=True)


def build_import_contract(search_path: list[str]) -> dict:
    """Protect the public build APIs from this observed vendor-path ambiguity."""
    contract = {}
    for name in ("setuptools", "wheel", "packaging", "setuptools_scm", "vcs_versioning"):
        spec = PathFinder.find_spec(name, search_path)
        if spec is not None and spec.origin:
            contract[name] = {"origin": Path(spec.origin).resolve(strict=True),
                              "roots": tuple(Path(path).resolve(strict=True)
                                             for path in spec.submodule_search_locations or ())}
    return contract


def check_import_context(search_path: list[str], contract: dict) -> None:
    # Sealed Setuptools81 appends its own vendor fallback after all selected
    # baseline/producer paths. This does not grant public wheel/packaging aliases.
    vendor = str(contract["setuptools"]["origin"].parent / "_vendor") if "setuptools" in contract else None
    if sys.path != search_path and (vendor is None or sys.path != [*search_path, vendor]):
        raise ValueError("consumer changed the declared import search order")
    for name, module in tuple(sys.modules.items()):
        top = name.partition(".")[0]
        if top not in contract:
            continue
        filename = vars(module).get("__file__") if module is not None else None
        if not isinstance(filename, str):
            continue
        actual = Path(filename).resolve(strict=True)
        expected = contract[top]
        if not (actual == expected["origin"] or any(actual.is_relative_to(root) for root in expected["roots"])):
            raise ValueError(f"public build module resolved outside its selected origin: {name}: {actual}")


def run_script(script: str, script_args: list[str], prefix: Path, manifest: dict,
               seal: dict, manifest_path: Path) -> None:
    original_path, saved_path = sys.path, sys.path.copy()
    selected_path = [*saved_path, str(prefix)]
    import_contract = build_import_contract(selected_path)
    def finish():
        failures = []
        try:
            check_import_context(selected_path, import_contract)
        except BaseException as error:
            failures.append(error)
        try:
            original_path[:] = saved_path
            sys.path = original_path
        except BaseException as error:
            failures.append(error)
        try:
            audit(prefix, manifest)
        except BaseException as error:
            failures.append(error)
        try:
            if seal != seal_inputs(prefix, manifest_path):
                raise ValueError("producer changed its own sealed inputs")
        except BaseException as error:
            failures.append(error)
        try:
            provider_proof()
        except BaseException as error:
            failures.append(error)
        if failures:
            raise BaseExceptionGroup("producer final audits failed", failures)

    provider_proof()
    sys.path.append(str(prefix))
    try:
        subprocess.run([sys.executable, "-c",
            "import runpy,sys;sys.path.append(sys.argv.pop());"
            "sys.argv=['pip','check'];runpy.run_module('pip',run_name='__main__')",
            str(prefix)], check=True)
        sys.argv = [script, *script_args]
        runpy.run_path(script, run_name="__main__")
    except BaseException as primary:
        try:
            finish()
        except BaseException as cleanup:
            raise BaseExceptionGroup("producer and final audit both failed", [primary, cleanup]) from None
        raise
    else:
        finish()


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot_parser = commands.add_parser("snapshot")
    snapshot_parser.add_argument("--output", type=Path, required=True)
    prepare_parser = commands.add_parser("prepare")
    for flag in ("baseline", "report", "wheels", "provenance", "output"):
        prepare_parser.add_argument(f"--{flag}", type=Path, required=True)
    prepare_parser.add_argument("--kind", choices=("producer", "build-tools"), default="producer")
    wheel_parser = commands.add_parser("check-wheels")
    wheel_parser.add_argument("--manifest", type=Path, default=Path("/opt/tensorfold-calibration/manifest.json"))
    wheel_parser.add_argument("--wheels", type=Path, required=True)
    for command in ("seal", "run"):
        action = commands.add_parser(command)
        for flag, default in (("prefix", "prefix"), ("manifest", "manifest.json"), ("seal", "seal.json")):
            action.add_argument(f"--{flag}", type=Path, default=Path("/opt/tensorfold-calibration") / default)
        if command == "run":
            action.add_argument("script")
            action.add_argument("script_args", nargs=argparse.REMAINDER)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    if args.command == "snapshot":
        snapshot(args.output)
    elif args.command == "prepare":
        prepare(args.baseline, args.report, args.wheels, args.provenance, args.output, args.kind)
    elif args.command == "check-wheels":
        check_wheels(read(args.manifest), args.wheels)
    else:
        manifest = read(args.manifest)
        audit(args.prefix, manifest)
        if args.command == "seal":
            write(args.seal, seal_inputs(args.prefix, args.manifest))
        else:
            seal = read(args.seal)
            if seal != seal_inputs(args.prefix, args.manifest):
                raise ValueError("calibration prefix seal changed")
            if os.environ.get("TF_FOUNDATION_RUNTIME_PHASE") not in {"cpu", "cuda"}:
                raise RuntimeError("producer requires the original managed CPU/GPU guard launcher")
            # Check the effective union rather than installing/removing baseline packages.
            run_script(args.script, args.script_args, args.prefix, manifest, seal, args.manifest)


if __name__ == "__main__":
    main()

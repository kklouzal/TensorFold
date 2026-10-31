#!/usr/bin/env python3
"""Render target-validated hash locks from a bounded pip resolver report.

Linux CPython 3.12 ARM64 remains the default. AMD64 uses --architecture amd64.
A report produced with pip cross-platform wheel selection needs --cross-report;
its real resolver environment is retained while target tags, markers and extras
are independently checked. Dependency validation is not an AMD64 build claim.
"""

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

from foundation_policy import artifact_name, canonical_name, is_accelerator_package, validate_pins, wheel_url
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version
from platform_pins import PLATFORMS, platform_pins

HERE = Path(__file__).resolve().parent
TEST_ONLY = frozenset({"pytest", "pluggy", "iniconfig"})
BINARY_NIGHTLY_HOSTS = frozenset({"download.pytorch.org", "download-r2.pytorch.org"})
ALLOWED_HOSTS = BINARY_NIGHTLY_HOSTS | {"files.pythonhosted.org"}
NATIVE_ADDON_HOSTS = frozenset({"developer.download.nvidia.com"})


def _native_addons(pins: dict) -> list[dict]:
    """Validate local Debian payload identities; the guard checks base ownership."""
    entries = pins.get("native_addons", [])
    if not isinstance(entries, list) or len(entries) > 32:
        raise ValueError("native_addons must be a list of at most 32 records")
    packages, artifacts, result = set(), set(), []
    required = {"artifact", "sha256", "package", "version"}
    for entry in entries:
        if not isinstance(entry, dict) or not required <= entry.keys() or entry.keys() - required - {"url"}:
            raise ValueError("native addon requires artifact, sha256, package and version metadata")
        artifact = artifact_name(entry["artifact"], ".deb")
        digest, package, version = entry["sha256"], entry["package"], entry["version"]
        if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not isinstance(package, str) or not re.fullmatch(r"[a-z0-9][a-z0-9+.-]+", package)
                or not isinstance(version, str) or not 1 <= len(version) <= 256
                or not re.fullmatch(r"[0-9][A-Za-z0-9.+:~\-]*", version)):
            raise ValueError("invalid native addon package, version or SHA256")
        if artifact in artifacts or package in packages:
            raise ValueError("duplicate native addon package or artifact")
        if "url" in entry:
            url = entry["url"]
            if not isinstance(url, str):
                raise ValueError("native addon source must be an official HTTPS URL")
            parsed = urlsplit(url)
            if (parsed.scheme != "https" or parsed.hostname not in NATIVE_ADDON_HOSTS
                    or parsed.username or parsed.password or parsed.fragment
                    or any(character.isspace() for character in unquote(url))
                    or unquote(parsed.path.rsplit("/", 1)[-1]) != artifact):
                raise ValueError("native addon source must be an official HTTPS URL matching its artifact")
        packages.add(package)
        artifacts.add(artifact)
        result.append(dict(entry))
    return result


def target_tags(architecture: str):
    target = platform_pins(architecture)
    minimum = 5 if architecture == "amd64" else 17
    platforms = [f"manylinux_2_{minor}_{target.machine}" for minor in range(39, minimum - 1, -1)]
    platforms += [f"manylinux2014_{target.machine}", f"linux_{target.machine}"]
    if architecture == "amd64":
        platforms += ["manylinux2010_x86_64", "manylinux1_x86_64"]
    return set(cpython_tags((3, 12), abis=["cp312"], platforms=platforms)) | set(
        compatible_tags((3, 12), interpreter="cp312", platforms=platforms))


def _closure(selected: dict, environment: dict, *, cross: bool, roots: list[str] | None = None,
             required_roots: set[str] | None = None) -> set[str]:
    """Validate the reachable target graph, preserving each requested extra."""
    extras = {name: {""} for name in selected} if roots is None else {}

    def include(requirement: Requirement, parent: str) -> bool:
        dependency = canonical_name(requirement.name)
        if dependency not in selected:
            raise ValueError(f"report omitted active target dependency {parent} -> {requirement}")
        version = Version(selected[dependency]["version"])
        if not requirement.specifier.contains(version, prereleases=True):
            raise ValueError(f"target dependency conflict: {parent} requires {requirement}, got {version}")
        if requirement.url:
            raise ValueError(f"unreviewed direct-URL dependency in {parent}: {requirement}")
        before = len(extras.get(dependency, ()))
        extras.setdefault(dependency, {""}).update(requirement.extras)
        if len(extras[dependency]) > 256:
            raise ValueError("dependency extra count exceeds the audit limit")
        return len(extras[dependency]) != before

    def active(requirement: Requirement, name: str, requested: set[str]) -> bool:
        if cross and requirement.marker and any(value in str(requirement.marker)
                                                for value in ("platform_release", "platform_version")):
            raise ValueError(f"{name} requires a native report for kernel-specific dependency markers")
        return requirement.marker is None or any(
            requirement.marker.evaluate(dict(environment, extra=extra)) for extra in requested)

    if roots is not None:
        seeded = set()
        for text in roots:
            requirement = Requirement(text)
            if active(requirement, "foundation_roots", {""}):
                include(requirement, "foundation_roots")
                seeded.add(canonical_name(requirement.name))
        if required_roots is not None and not required_roots <= seeded:
            raise ValueError("foundation_roots omitted required audited wheels")
    changed = True
    while changed:
        changed = False
        for name in tuple(extras):
            metadata = selected[name]
            requirements = metadata.get("requires_dist", [])
            if requirements is None:
                requirements = []
            if not isinstance(requirements, list) or len(requirements) > 256:
                raise ValueError(f"invalid dependency metadata for {name}")
            for text in requirements:
                requirement = Requirement(text)
                if active(requirement, name, extras[name]):
                    changed |= include(requirement, name)
    return set(extras)


def render(report: dict, pins: dict, architecture: str = "arm64", *, cross_report: bool = False,
           report_sha256: str | None = None):
    target = platform_pins(architecture)
    validate_pins(pins, architecture)
    native_addons = _native_addons(pins)
    roots = pins["foundation_roots"]
    environment = report["environment"]
    if (environment["platform_system"] != "Linux" or environment["sys_platform"] != "linux"
            or environment["implementation_name"] != "cpython" or environment["python_version"] != pins["python"]
            or Version(environment["python_full_version"]).release[:2] != (3, 12)):
        raise ValueError("resolve the dependency report on Linux CPython 3.12")
    cross = environment["platform_machine"] != target.machine
    if cross and not cross_report:
        raise ValueError(f"resolve on {target.machine}, or explicitly declare --cross-report")
    if cross and environment["platform_machine"] not in {item.machine for item in PLATFORMS.values()}:
        raise ValueError("cross report came from an undeclared deployment architecture")
    active_environment = dict(default_environment(), **environment)
    active_environment["platform_machine"] = target.machine
    entries = report["install"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 256:
        raise ValueError("invalid dependency count")
    accepted = target_tags(architecture)
    selected, distributions, lines = {}, {}, {}
    for item in entries:
        metadata = item["metadata"]
        name = canonical_name(metadata["name"])
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or name in selected:
            raise ValueError("invalid or duplicate dependency name")
        version = metadata["version"]
        if metadata.get("requires_python") and not SpecifierSet(metadata["requires_python"]).contains(
                Version(environment["python_full_version"]), prereleases=True):
            raise ValueError(f"{name} does not support the report's Python version")
        download = item["download_info"]
        digest = download["archive_info"]["hashes"]["sha256"]
        url = download["url"]
        parsed = urlsplit(url)
        filename = unquote(parsed.path.rsplit("/", 1)[-1])
        wheel_name, wheel_version, _, tags = parse_wheel_filename(filename)
        if wheel_name != name or wheel_version != Version(version) or not tags & accepted:
            raise ValueError(f"wheel metadata/tags do not match linux/{architecture} CPython 3.12: {name}")
        allowed = BINARY_NIGHTLY_HOSTS if name in pins["wheels"] else ALLOWED_HOSTS
        nvidia_origin = (parsed.hostname == "pypi.nvidia.com" and parsed.port in (None, 443)
                         and (name == "cuda-toolkit" or name.startswith("nvidia-")))
        if (parsed.scheme != "https" or not (parsed.hostname in allowed or nvidia_origin)
                or parsed.username or parsed.password or parsed.fragment
                or any(character.isspace() for character in unquote(url))):
            raise ValueError(f"unapproved dependency origin for {name}")
        if name in pins["wheels"]:
            pin = pins["wheels"][name]
            if version != pin["version"] or digest != pin["sha256"]:
                raise ValueError(f"resolver changed the audited {name} wheel")
            url = wheel_url(name, pin)
            parsed = urlsplit(url)
            pinned_name, pinned_version, _, pinned_tags = parse_wheel_filename(unquote(parsed.path.rsplit("/", 1)[-1]))
            if pinned_name != name or pinned_version != Version(version) or not pinned_tags & accepted:
                raise ValueError(f"audited wheel metadata/tags do not match linux/{architecture} CPython 3.12: {name}")
        if any(character.isspace() for character in unquote(url)) or not parsed.path.endswith(".whl"):
            raise ValueError(f"dependency must be a wheel URL: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid SHA256 for {name}")
        selected[name] = metadata
        lines[name] = f"{name} @ {url} --hash=sha256:{digest}\n"
        distributions[name] = {"version": version, "url": url, "sha256": digest}
        distributions[name].update({key: metadata[key] for key in ("requires_dist", "requires_python") if key in metadata})
    if not set(pins["wheels"]) <= selected.keys() or not selected.keys() >= TEST_ONLY:
        raise ValueError("report omitted required nightly or verification packages")
    _closure(selected, active_environment, cross=cross)
    foundation = _closure(selected, active_environment, cross=cross, roots=roots, required_roots=set(pins["wheels"]))
    nccl_provider = pins.get("nccl_provider", "native")
    if nccl_provider == "nvidia-nccl-cu13" and nccl_provider not in foundation:
        raise ValueError("selected NCCL wheel provider must belong to the protected foundation closure")
    if foundation & TEST_ONLY:
        raise ValueError("verification-only packages cannot be foundation dependencies")
    runtime_closure = _closure(selected, active_environment, cross=cross, roots=sorted(selected.keys() - TEST_ONLY))
    if runtime_closure & TEST_ONLY:
        raise ValueError("verification-only packages cannot be runtime dependencies")
    foreign = sorted(name for name in selected.keys() - foundation if is_accelerator_package(name))
    if foreign:
        raise ValueError("accelerator packages outside protected foundation closure: " + ", ".join(foreign))
    roles = {"foundation": foundation, "runtime": selected.keys() - foundation - TEST_ONLY,
             "verification": TEST_ONLY}
    label = "ARM64" if architecture == "arm64" else "AMD64"
    mode = f"{label} cross-platform pip report (target marker closure checked)" if cross else f"{label} pip report"
    banner = f"# Generated by lock_dependencies.py from the {mode}; update pins and regenerate.\n"
    manifest = {"schema_version": 1,
                **{key: pins[key] for key in ("platform", "python", "base", "base_index", "cuda", "cache_key")},
                "foundation_roots": roots,
                "native_addons": native_addons,
                "nccl_provider": nccl_provider,
                "roles": {role: {name: distributions[name] for name in sorted(names)} for role, names in roles.items()},
                "resolver_environment": environment,
                "resolver_report_sha256": report_sha256 or hashlib.sha256(
                    json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "resolver_report_hash_format": "raw-file" if report_sha256 else "canonical-json",
                "cross_report": cross}
    if not re.fullmatch(r"[0-9a-f]{64}", manifest["resolver_report_sha256"]):
        raise ValueError("invalid resolver report SHA256")
    return {target.foundation: banner + "".join(lines[name] for name in sorted(foundation)),
            target.runtime: banner + "".join(lines[name] for name in sorted(roles["runtime"])),
            target.verification: banner + "".join(lines[name] for name in sorted(TEST_ONLY)),
            target.constraints: banner + "".join(
                f"{name}==={selected[name]['version']}\n" for name in sorted(foundation)),
            target.manifest: json.dumps(manifest, indent=2, sort_keys=True) + "\n"}


def write_locks(contents: dict[str, str], directory: Path) -> None:
    """Stage every validated lock before replacement; restore on an IO failure."""
    staged, original, committed = {}, {}, []
    try:
        for filename, content in contents.items():
            destination = directory / filename
            original[filename] = destination.read_bytes() if destination.exists() else None
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, prefix=".lock-",
                                             delete=False) as stream:
                staged[filename] = Path(stream.name)
                stream.write(content)
        for filename, path in staged.items():
            path.replace(directory / filename)
            committed.append(filename)
    except BaseException:
        for filename in reversed(committed):
            if original[filename] is None:
                (directory / filename).unlink()
            else:
                (directory / filename).write_bytes(original[filename])
        raise
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pins", type=Path, help="Selected pin metadata (default: the architecture's repository pins)")
    parser.add_argument("--output-directory", type=Path, default=HERE,
                        help="Generated lock/manifest directory (default: deploy/gb10)")
    parser.add_argument("--architecture", choices=tuple(PLATFORMS), default="arm64")
    parser.add_argument("--cross-report", action="store_true")
    args = parser.parse_args()
    with args.report.open("rb") as stream:
        report_bytes = stream.read(2 * 2**20 + 1)
    if len(report_bytes) > 2 * 2**20:
        raise ValueError("pip report exceeds the 2 MiB audit limit")
    report = json.loads(report_bytes)
    target = platform_pins(args.architecture)
    with (args.pins or HERE / target.pins).open("rb") as stream:
        pin_bytes = stream.read(2 * 2**20 + 1)
    if len(pin_bytes) > 2 * 2**20:
        raise ValueError("deployment pins exceed the 2 MiB audit limit")
    pins = json.loads(pin_bytes)
    contents = render(report, pins, args.architecture, cross_report=args.cross_report,
                      report_sha256=hashlib.sha256(report_bytes).hexdigest())
    args.output_directory.mkdir(parents=True, exist_ok=True)
    write_locks(contents, args.output_directory)
    counts = {role: len(packages) for role, packages in json.loads(contents[target.manifest])["roles"].items()}
    print(f"Locked {counts['foundation']} foundation, {counts['runtime']} runtime and "
          f"{counts['verification']} test packages for linux/{args.architecture}; dependency closure validated")


if __name__ == "__main__":
    main()

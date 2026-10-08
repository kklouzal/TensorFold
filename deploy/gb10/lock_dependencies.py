#!/usr/bin/env python3
"""Render target-validated hash locks from a bounded pip resolver report.

Linux CPython 3.12 ARM64 remains the default. AMD64 uses --architecture amd64.
A report produced with pip cross-platform wheel selection needs --cross-report;
its real resolver environment is retained while target tags, markers and extras
are independently checked. Dependency validation is not an AMD64 build claim.
"""

import argparse
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import unquote, urlsplit

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from platform_pins import PLATFORMS, platform_pins

HERE = Path(__file__).resolve().parent
TEST_ONLY = frozenset({"pytest", "pluggy", "iniconfig"})
ALLOWED_HOSTS = frozenset({"download.pytorch.org", "files.pythonhosted.org"})


def target_tags(architecture: str):
    target = platform_pins(architecture)
    minimum = 5 if architecture == "amd64" else 17
    platforms = [f"manylinux_2_{minor}_{target.machine}" for minor in range(39, minimum - 1, -1)]
    platforms += [f"manylinux2014_{target.machine}", f"linux_{target.machine}"]
    if architecture == "amd64":
        platforms += ["manylinux2010_x86_64", "manylinux1_x86_64"]
    return set(cpython_tags((3, 12), abis=["cp312"], platforms=platforms)) | set(
        compatible_tags((3, 12), interpreter="cp312", platforms=platforms))


def _closure(selected: dict, environment: dict, *, cross: bool) -> None:
    """Check every active target requirement, including transitive extras."""
    extras = {name: {""} for name in selected}
    changed = True
    while changed:
        changed = False
        for name, metadata in selected.items():
            requirements = metadata.get("requires_dist") or []
            if not isinstance(requirements, list) or len(requirements) > 256:
                raise ValueError(f"invalid dependency metadata for {name}")
            for text in requirements:
                requirement = Requirement(text)
                if cross and requirement.marker and any(value in str(requirement.marker)
                                                        for value in ("platform_release", "platform_version")):
                    raise ValueError(f"{name} requires a native report for kernel-specific dependency markers")
                active = requirement.marker is None or any(
                    requirement.marker.evaluate(dict(environment, extra=extra)) for extra in extras[name])
                if not active:
                    continue
                dependency = canonicalize_name(requirement.name)
                if dependency not in selected:
                    raise ValueError(f"report omitted active target dependency {name} -> {requirement}")
                version = Version(selected[dependency]["version"])
                if not requirement.specifier.contains(version, prereleases=True):
                    raise ValueError(f"target dependency conflict: {name} requires {requirement}, got {version}")
                if requirement.url:
                    raise ValueError(f"unreviewed direct-URL dependency in {name}: {requirement}")
                before = len(extras[dependency])
                extras[dependency].update(requirement.extras)
                if len(extras[dependency]) > 256:
                    raise ValueError("dependency extra count exceeds the audit limit")
                changed |= len(extras[dependency]) != before


def render(report: dict, pins: dict, architecture: str = "arm64", *, cross_report: bool = False):
    target = platform_pins(architecture)
    if pins["platform"] != f"linux/{architecture}" or pins["python"] != "3.12":
        raise ValueError("pin metadata does not describe the selected target")
    environment = report["environment"]
    if (environment["platform_system"] != "Linux" or environment["sys_platform"] != "linux"
            or environment["implementation_name"] != "cpython" or environment["python_version"] != pins["python"]):
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
    selected, runtime, test = {}, [], []
    for item in entries:
        metadata = item["metadata"]
        name = canonicalize_name(metadata["name"])
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
        if name in pins["wheels"]:
            pin = pins["wheels"][name]
            if version != pin["version"] or digest != pin["sha256"]:
                raise ValueError(f"resolver changed the audited {name} wheel")
            url = pin["url"]
            parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS or parsed.username or parsed.password:
            raise ValueError(f"unapproved dependency origin for {name}")
        if any(character.isspace() for character in unquote(url)) or not parsed.path.endswith(".whl"):
            raise ValueError(f"dependency must be a wheel URL: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid SHA256 for {name}")
        selected[name] = metadata
        line = f"{name} @ {url} --hash=sha256:{digest}\n"
        (test if name in TEST_ONLY else runtime).append((name, line))
    if not set(pins["wheels"]) <= selected.keys() or not TEST_ONLY <= selected.keys():
        raise ValueError("report omitted required nightly or verification packages")
    _closure(selected, active_environment, cross=cross)
    label = "ARM64" if architecture == "arm64" else "AMD64"
    mode = f"{label} cross-platform pip report (target marker closure checked)" if cross else f"{label} pip report"
    banner = f"# Generated by lock_dependencies.py from the {mode}; update pins and regenerate.\n"
    return {target.runtime: banner + "".join(line for _, line in sorted(runtime)),
            target.verification: banner + "".join(line for _, line in sorted(test))}


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
    parser.add_argument("--architecture", choices=tuple(PLATFORMS), default="arm64")
    parser.add_argument("--cross-report", action="store_true")
    args = parser.parse_args()
    if args.report.stat().st_size > 2 * 2**20:
        raise ValueError("pip report exceeds the 2 MiB audit limit")
    report = json.loads(args.report.read_text())
    target = platform_pins(args.architecture)
    pins = json.loads((HERE / target.pins).read_text())
    contents = render(report, pins, args.architecture, cross_report=args.cross_report)
    write_locks(contents, HERE)
    print(f"Locked {len(report['install']) - len(TEST_ONLY)} runtime and {len(TEST_ONLY)} test packages "
          f"for linux/{args.architecture}; dependency closure validated")


if __name__ == "__main__":
    main()

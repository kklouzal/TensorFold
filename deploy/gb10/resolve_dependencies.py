#!/usr/bin/env python3
"""Produce the native pip resolver report that lock_dependencies.py consumes.

Run on the target architecture (owned sdists need native metadata, so pip's
cross-platform selection cannot produce this report). The pinned interpreter
archive is verified against the selected pins and extracted into a temporary
directory; its bundled pip resolves one shared closure for the foundation,
application, verification and owned-build roles. The official Torch,
TorchVision and Triton wheels are requested by their pinned URLs and SHA256s (the
PyTorch index publishes no hashes, so a metadata-only dry run would record none);
the image's hash-checked install verifies the whole wheels. Only the selected
owned packages may resolve to sdists; pip builds their metadata in isolated
environments, which affects only this report, never the image. An optional
earlier manifest constrains every version it records (all roles), so a second
architecture or an interpreter change keeps the reviewed selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from foundation_policy import canonical_name, owned_builds, validate_pins
from packaging.requirements import Requirement
from platform_pins import PLATFORMS, platform_pins

HERE = Path(__file__).resolve().parent
# Linux application/verification roots: pyproject dependencies with the tui, test,
# vision and grammar extras, plus the tools the image's no-isolation wheel build uses.
APPLICATION_ROOTS = ("numpy", "huggingface_hub", "tokenizers", "safetensors", "Jinja2", "rich", "prompt_toolkit",
                     "pytest", "pillow", "transformers", "av", "xgrammar", "setuptools", "wheel", "packaging")


def constraints(manifest: Path) -> list[str]:
    with manifest.open("rb") as stream:
        raw = stream.read(2 * 2**20 + 1)
    if len(raw) > 2 * 2**20:
        raise ValueError("constraint manifest exceeds the 2 MiB audit limit")
    selected = {}
    for packages in json.loads(raw)["roles"].values():
        for name, record in packages.items():
            if selected.setdefault(name, record["version"]) != record["version"]:
                raise ValueError(f"constraint manifest records two versions of {name}")
    return [f"{name}=={version}\n" for name, version in sorted(selected.items())]


def resolve(architecture: str, archive: Path, output: Path, pins_path: Path, manifest: Path | None) -> None:
    target = platform_pins(architecture)
    if platform.system() != "Linux" or platform.machine() != target.machine:
        raise ValueError(f"resolve linux/{architecture} natively, not on {platform.system()}/{platform.machine()}")
    pins = json.loads(pins_path.read_bytes())
    validate_pins(pins, architecture, target.machine)
    interpreter = pins["interpreter"]
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != interpreter["sha256"]:
            raise ValueError("interpreter archive does not match the selected pins")
    # URL roots carry the official wheels' identities and their foundation_roots extras;
    # the generator checks the foundation roots themselves against the report.
    extras, other = {}, []
    for root in pins["foundation_roots"]:
        requirement = Requirement(root)
        if canonical_name(requirement.name) in pins["wheels"]:
            extras.setdefault(canonical_name(requirement.name), set()).update(requirement.extras)
        else:
            other.append(root)
    official = [f"{name}{'[' + ','.join(sorted(extras.get(name, ()))) + ']' if extras.get(name) else ''}"
                f" @ {pin['url']}#sha256={pin['sha256']}" for name, pin in sorted(pins["wheels"].items())]
    roots = [*official, *other, *APPLICATION_ROOTS,
             *sorted({text for record in owned_builds(pins).values() for text in record["build_requires"]})]
    owned = ",".join(sorted(owned_builds(pins))) or ":none:"
    with tempfile.TemporaryDirectory(prefix="tensorfold-resolve-") as temporary:
        work = Path(temporary)
        with tarfile.open(archive) as bundle:
            bundle.extractall(work / "interpreter", filter="data")
        python = work / "interpreter/python/bin" / f"python{pins['python']}"
        subprocess.run([str(python), "-m", "venv", str(work / "venv")], check=True)
        (work / "roots.txt").write_text("".join(f"{root}\n" for root in roots))
        (work / "constraints.txt").write_text("".join(constraints(manifest)) if manifest else "")
        environment = dict(os.environ, HOME=str(work / "home"), PIP_DISABLE_PIP_VERSION_CHECK="1",
                           PIP_CONFIG_FILE=os.devnull, CC="gcc", CXX="g++")
        command = [str(work / "venv/bin/python"), "-m", "pip", "install", "--dry-run", "--ignore-installed",
                   "--no-cache-dir", "--pre", "--only-binary=:all:", f"--no-binary={owned}",
                   "--index-url", "https://pypi.org/simple",
                   "-c", str(work / "constraints.txt"), "-r", str(work / "roots.txt"),
                   "--report", str(work / "report.json")]
        subprocess.run(command, env=environment, check=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(work / "report.json", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=tuple(PLATFORMS), required=True)
    parser.add_argument("--archive", type=Path, required=True, help="The pins' downloaded interpreter archive")
    parser.add_argument("--output", type=Path, required=True, help="Resolver report to write")
    parser.add_argument("--pins", type=Path, help="Selected pins (default: the architecture's repository pins)")
    parser.add_argument("--constraints-manifest", type=Path, help="Earlier foundation manifest whose versions to keep")
    args = parser.parse_args()
    resolve(args.architecture, args.archive, args.output, args.pins or HERE / platform_pins(args.architecture).pins,
            args.constraints_manifest)


if __name__ == "__main__":
    if sys.version_info < (3, 12):
        raise SystemExit("resolve_dependencies.py requires Python 3.12+ (tarfile data filter)")
    main()

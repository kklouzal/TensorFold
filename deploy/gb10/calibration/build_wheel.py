"""Root-operated public PEP517 backend consumer; no package installs.

Run only inside the reviewed build-tools role. The public backend runs in this
interpreter so it retains the append-only build-tools prefix. No global path
override, pip build isolation or changes to upstream source/metadata are used.
"""
import argparse
import importlib.metadata as md
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement

from role import MODEL_OPT_COMMIT, sha, wheel_record, write
from wheel_preflight import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_archive_sha256):
        raise ValueError("source archive SHA256 is required")
    source, output, executable, git_version = preflight(args.source, args.output)
    def git(*arguments):
        return subprocess.run([str(executable), "-C", str(source), *arguments], check=True,
                              capture_output=True, text=True).stdout.strip()
    if git("rev-parse", "HEAD") != MODEL_OPT_COMMIT or git("status", "--porcelain=v1"):
        raise ValueError("build requires the exact clean upstream Git checkout")
    if any(key.startswith("SETUPTOOLS_SCM_PRETEND_") for key in os.environ):
        raise ValueError("SCM version overrides are not permitted")
    spec = tomllib.loads((source / "pyproject.toml").read_text())["build-system"]
    if spec["build-backend"] != "setuptools.build_meta" or "backend-path" in spec:
        raise ValueError("upstream build backend changed")
    def validate_requirements(requirements):
        for raw in requirements:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate():
                continue
            if requirement.extras or requirement.url or not requirement.specifier.contains(md.version(requirement.name), prereleases=True):
                raise ValueError(f"build requirement is not already satisfied: {raw}")
    validate_requirements(spec["requires"])
    # Missing Git/permissions must fail before backend imports and their vendor
    # namespace side effects. The standalone preflight also precedes role audit.
    import setuptools.build_meta as backend
    output.mkdir(exist_ok=False)
    describe = git("describe", "--tags", "--always", "--dirty")
    previous = Path.cwd()
    try:
        os.chdir(source)
        dynamic = backend.get_requires_for_build_wheel()
        validate_requirements(dynamic)
        # Public PEP517 backend: upstream determines the wheel/SCM version.
        filename = backend.build_wheel(str(output))
    finally:
        os.chdir(previous)
    git("diff", "--exit-code", "HEAD", "--")
    if Path(filename).name != filename:
        raise ValueError("build backend returned an unsafe wheel path")
    record = wheel_record(output / filename)
    if record["name"] != "nvidia-modelopt":
        raise ValueError("build backend produced the wrong distribution")
    write(output / "source.json", {"source_commit": MODEL_OPT_COMMIT,
        "source_archive_sha256": args.source_archive_sha256,
        "wheel_sha256": sha(output / filename), "wheel_version": record["version"],
        "git_describe": describe, "build_backend": spec["build-backend"],
        "git_executable": str(executable), "git_version": git_version,
        "declared_build_requires": spec["requires"], "dynamic_build_requires": dynamic,
        "consumer_sha256": sha(Path(__file__).resolve()), "python": sys.version,
        "build_tools": {name: md.version(name) for name in ("setuptools", "setuptools-scm", "wheel")}})


if __name__ == "__main__":
    main()

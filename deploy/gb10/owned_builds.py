#!/usr/bin/env python3
"""Build the selected owned sdists into wheels inside the discarded build stage.

Each sdist is fetched by its exact URL and SHA256 and built without isolation
or an index, so only the generated build lock's wheels (installed beforehand)
can satisfy its build requirements; pip checks the declared ones. The pins'
per-package CFLAGS/CXXFLAGS are the only build-setting overrides. The image's
pinned interpreter names clang in its sysconfig; the base supplies GCC.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename


def build(manifest_path: Path, output: Path) -> dict[str, str]:
    with manifest_path.open("rb") as stream:
        raw = stream.read(2 * 2**20 + 1)
    if len(raw) > 2 * 2**20:
        raise ValueError("selected foundation manifest exceeds the 2 MiB audit limit")
    manifest = json.loads(raw)
    owned = manifest["owned_builds"]
    output.mkdir(parents=True)
    supported = set(sys_tags())
    built = {}
    for name, record in sorted(owned.items()):
        with tempfile.TemporaryDirectory(prefix=f"owned-{name}-") as temporary:
            directory = Path(temporary)
            requirement = directory / "requirement.txt"
            requirement.write_text(f"{name} @ {record['url']} --hash=sha256:{record['sha256']}\n")
            environment = dict(os.environ, CC="gcc", CXX="g++", **record["environment"])
            subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-index", "--no-cache-dir",
                            "--no-build-isolation", "--check-build-dependencies", "--require-hashes",
                            "--wheel-dir", str(directory / "wheels"), "-r", str(requirement)],
                           env=environment, check=True)
            wheels = list((directory / "wheels").iterdir())
            if len(wheels) != 1:
                raise ValueError(f"owned build {name} produced {len(wheels)} wheels")
            wheel_name, version, _, tags = parse_wheel_filename(wheels[0].name)
            if wheel_name != name or str(version) != record["version"] or not tags & supported:
                raise ValueError(f"owned build {name} produced an unexpected wheel: {wheels[0].name}")
            shutil.move(wheels[0], output / wheels[0].name)
            built[name] = wheels[0].name
    return built


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Selected foundation manifest")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the built wheels")
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.output), sort_keys=True))


if __name__ == "__main__":
    main()

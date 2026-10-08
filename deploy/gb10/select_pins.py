#!/usr/bin/env python3
"""Select target pins before installation and reject a mismatched base runtime."""

import argparse
import json
from pathlib import Path
import platform
import shutil
import sys

from platform_pins import PLATFORMS, platform_pins


def select(architecture: str, directory: Path) -> None:
    target = platform_pins(architecture)
    if platform.system() != "Linux" or platform.machine() != target.machine:
        raise ValueError(f"requested linux/{architecture} pins on {platform.system()}/{platform.machine()}")
    copies = ((target.pins, "nightly-pins.json"), (target.runtime, "requirements-runtime.lock"),
              (target.verification, "requirements-test.lock"))
    for source, _ in copies:
        path = directory / source
        if not path.is_file() or path.stat().st_size > 2 * 2**20:
            raise ValueError(f"missing or oversized deployment pins: {source}")
    pins = json.loads((directory / target.pins).read_text())
    if pins["platform"] != f"linux/{architecture}" or pins["python"] != f"{sys.version_info.major}.{sys.version_info.minor}":
        raise ValueError("selected pin metadata does not match the deployment runtime")
    for source, destination in copies:
        if source != destination:
            shutil.copyfile(directory / source, directory / destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=tuple(PLATFORMS), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    select(args.architecture, args.directory)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate NVIDIA forward compatibility freshly before the NGC entrypoint.

NGC caches its probe marker in the writable container layer but exports its
result only to the current process. A restart retains the marker and loses that
result. Use NVIDIA's shipped probe each startup; never infer success from a
marker or a symlink. Keep the NVIDIA entrypoint and its checks unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys


DRIVER_VERSION_FILE = Path("/proc/driver/nvidia/version")
COMPAT_LIBRARY = Path("/usr/local/cuda/compat/lib.real")
COMPAT_PROBE = "/usr/local/bin/cudaCheck"
NVIDIA_ENTRYPOINT = "/opt/nvidia/nvidia_entrypoint.sh"


def prepare_environment() -> dict[str, str]:
    """Return freshly validated launch variables; refuse failed FC probes."""

    environment = dict(os.environ)
    environment.pop("_CUDA_COMPAT_STATUS", None)
    driver = re.search(r"Kernel Module(?: for [a-z0-9_]+|\s)\s*([0-9]+(?:\.[0-9]+)+)",
                       DRIVER_VERSION_FILE.read_text(encoding="ascii"))
    image_driver = environment.get("CUDA_DRIVER_VERSION", "")
    if driver is None or re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", image_driver) is None:
        raise RuntimeError("cannot determine the host and container NVIDIA driver versions")
    if int(driver[1].split(".")[0]) >= int(image_driver.split(".")[0]):
        return environment
    if not COMPAT_LIBRARY.is_dir():
        raise RuntimeError(f"NVIDIA forward compatibility libraries are missing: {COMPAT_LIBRARY}")
    probe_environment = dict(environment, LD_LIBRARY_PATH=str(COMPAT_LIBRARY), LD_PRELOAD="")
    result = subprocess.run([COMPAT_PROBE], env=probe_environment, text=True,
                            capture_output=True, timeout=35, check=False)
    status = result.stdout.strip()
    if result.returncode != 0 or status != "CUDA Driver OK":
        detail = (status or result.stderr.strip())[:4096]
        raise RuntimeError(f"NVIDIA forward compatibility probe failed (exit {result.returncode}): {detail}")
    environment["_CUDA_COMPAT_STATUS"] = status
    libraries = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = str(COMPAT_LIBRARY) + (":" + libraries if libraries else "")
    return environment


def main() -> int:
    try:
        environment = prepare_environment()
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"TensorFold CUDA startup refused: {error}", file=sys.stderr)
        return 1
    os.execve(NVIDIA_ENTRYPOINT, [NVIDIA_ENTRYPOINT, *sys.argv[1:]], environment)


if __name__ == "__main__":
    raise SystemExit(main())

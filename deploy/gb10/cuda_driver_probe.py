#!/usr/bin/env python3
"""Fresh-process CUDA Driver API admission for the selected native/FC provider."""
from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path


def probe(library: str, required_api: int) -> dict:
    cuda = ctypes.CDLL(library)
    signatures = {"cuInit": ([ctypes.c_uint], ctypes.c_int),
                  "cuDriverGetVersion": ([ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
                  "cuDeviceGetCount": ([ctypes.POINTER(ctypes.c_int)], ctypes.c_int)}
    for name, (arguments, result) in signatures.items():
        function = getattr(cuda, name)
        function.argtypes, function.restype = arguments, result
    for name, argument in (("cuInit", 0),):
        code = getattr(cuda, name)(argument)
        if code != 0:
            raise RuntimeError(f"{name} returned CUDA error {code}")
    version, devices = ctypes.c_int(), ctypes.c_int()
    code = cuda.cuDriverGetVersion(ctypes.byref(version))
    if code != 0:
        raise RuntimeError(f"cuDriverGetVersion returned CUDA error {code}")
    if version.value < required_api:
        raise RuntimeError(f"driver API {version.value} is below selected SDK {required_api}")
    code = cuda.cuDeviceGetCount(ctypes.byref(devices))
    if code != 0 or devices.value < 1:
        raise RuntimeError(f"cuDeviceGetCount returned CUDA error {code}, devices={devices.value}")
    loaded = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and parts[5].startswith("/") and Path(parts[5]).name.startswith("libcuda.so"):
            loaded.add(str(Path(parts[5]).resolve(strict=True)))
    if len(loaded) != 1:
        raise RuntimeError("driver probe did not map exactly one libcuda provider")
    return {"driver_api": version.value, "devices": devices.value, "library": loaded.pop()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--required-api", required=True, type=int)
    args = parser.parse_args()
    try:
        result = probe(args.library, args.required_api)
    except (OSError, AttributeError, RuntimeError) as error:
        print(f"CUDA driver probe refused: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

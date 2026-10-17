"""Publish sealed tool/cache choices before any app-venv native import.

This tiny startup hook reads only the build-owned compact activation mapping.
Full origin/stat/provider checks belong to foundation_guard's launch boundary;
managed installs perform full hashes. Errors are fatal because site.py otherwise
logs .pth errors and continues with an incompletely configured interpreter.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def activate() -> None:
    path = Path(__file__).with_name("tensorfold-foundation-environment.json")
    with path.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("oversized foundation activation mapping")
    values = json.loads(raw)
    if (not isinstance(values, dict) or not values
            or any(not isinstance(key, str) or not isinstance(value, str)
                   or "\x00" in key or "\x00" in value for key, value in values.items())):
        raise ValueError("invalid foundation activation mapping")
    existing_nccl = os.environ.get("TF_NCCL_LIB")
    if (existing_nccl and "TF_NCCL_LIB" in values
            and Path(existing_nccl).resolve(strict=True) != Path(values["TF_NCCL_LIB"])):
        raise ValueError("TF_NCCL_LIB conflicts with selected foundation provider")
    os.environ.update(values)


try:
    activate()
except BaseException as error:
    try:
        print(f"TensorFold protected foundation activation failed: {error}; rebuild the image", file=sys.stderr)
    finally:
        os._exit(78)

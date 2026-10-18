"""Publish sealed tool/cache choices before any app-venv native import.

This tiny startup hook reads only the build-owned compact activation mapping.
Full origin/stat/provider checks belong to foundation_guard's launch boundary;
managed installs perform full hashes. Errors are fatal because site.py otherwise
logs .pth errors and continues with an incompletely configured interpreter.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def _preparatory_interpreter() -> bool:
    script = Path(sys.argv[0]).name if sys.argv else ""
    if script in {"cuda_entrypoint.py", "cuda_driver_probe.py", "tensorfold-cc", "tensorfold-cxx"}:
        return True
    if script != "foundation_guard.py":
        return False
    arguments = iter(sys.argv[1:])
    for argument in arguments:
        if argument.startswith("--"):
            if argument in {"--", "--help"}:
                return True
            if "=" not in argument:
                next(arguments, None)  # Every guard option takes one value.
        else:
            return argument != "native-proof"
    return True


def activate() -> tuple:
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
    requested_torch = os.environ.get("TORCH_USE_RTLD_GLOBAL")
    if requested_torch not in (None, "", "1"):
        raise ValueError("TORCH_USE_RTLD_GLOBAL conflicts with selected foundation")
    os.environ.update(values)
    phase = os.environ.get("TF_FOUNDATION_RUNTIME_PHASE")
    if not phase or _preparatory_interpreter():
        return None, None, None
    if phase not in {"cpu", "cuda"}:
        raise ValueError("invalid foundation runtime phase")
    profile = Path(values["TF_FOUNDATION_RUNTIME_PROFILE"])
    with profile.open("rb") as stream:
        raw_profile = stream.read(4 * 2**20 + 1)
    if len(raw_profile) > 4 * 2**20:
        raise ValueError("oversized foundation runtime profile")
    contract = json.loads(raw_profile)
    if phase == "cuda":
        proof = json.loads(os.environ["TF_CUDA_DRIVER_PROOF"])
        if (proof["mode"] not in {"native", "forward-compatible"} or type(proof["driver_api"]) is not int
                or proof["driver_api"] < 1000 or type(proof["devices"]) is not int or proof["devices"] < 1):
            raise ValueError("invalid admitted CUDA driver proof")
        driver = Path(proof["library"]).resolve(strict=True)
        driver_role = "fresh-driver-admitted"
    else:
        if any(Path("/dev").glob("nvidia[0-9]*")):
            raise ValueError("CPU runtime activation requires no NVIDIA devices")
        driver = Path(values["TF_FOUNDATION_CPU_DRIVER"]).resolve(strict=True)
        driver_role = "cpu-compatibility-load-only; no device eligibility claim"
    # This phase runs in the final consumer interpreter, after driver selection.
    # Both driver and core handles remain strongly owned until process exit.
    import ctypes

    from tensorfold_foundation_runtime import activate_runtime
    driver_owner = ctypes.CDLL(str(driver), mode=ctypes.RTLD_GLOBAL | os.RTLD_NOW)
    owners = (driver_owner, *activate_runtime(contract))
    return owners, hashlib.sha256(raw_profile).hexdigest(), driver_role


try:
    RUNTIME_OWNERS, ACTIVE_PROFILE_SHA256, DRIVER_ROLE = activate()
except BaseException as error:
    try:
        print(f"TensorFold protected foundation activation failed: {error}; rebuild the image", file=sys.stderr)
    finally:
        os._exit(78)

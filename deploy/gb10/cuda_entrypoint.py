#!/usr/bin/env python3
"""Admit actual native/forward-compatible CUDA, then keep NVIDIA's entrypoint.

The plain CUDA image ships compatibility libraries, not cudaCheck or a cached
probe protocol. Each launch tests the Driver API in fresh child processes. A
successful selected-library probe, not a marker or driver-name guess, admits it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

NVIDIA_ENTRYPOINT = "/opt/nvidia/nvidia_entrypoint.sh"


def _probe(environment: dict[str, str], library: str, required_api: int) -> dict:
    script = str(Path(__file__).with_name("cuda_driver_probe.py"))
    result = subprocess.run([sys.executable, script, "--library", library, "--required-api", str(required_api)],
                            env=environment, capture_output=True, text=True, timeout=35, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"CUDA driver probe failed (exit {result.returncode}): {result.stderr.strip()[:4096]}")
    value = json.loads(result.stdout)
    if (not isinstance(value, dict) or type(value.get("driver_api")) is not int
            or value["driver_api"] < required_api or type(value.get("devices")) is not int
            or value["devices"] < 1 or not isinstance(value.get("library"), str)
            or not Path(value["library"]).is_absolute() or not Path(value["library"]).name.startswith("libcuda.so")):
        raise ValueError("invalid CUDA driver probe result")
    return value


def prepare_environment() -> dict[str, str]:
    """No Torch/Triton import occurs before fresh FC and loader publication."""
    from foundation_guard import read_json, verify
    environment = dict(os.environ)
    environment.pop("_CUDA_COMPAT_STATUS", None)
    state_path = environment.get("TF_FOUNDATION_STATE")
    if not state_path:
        raise RuntimeError("CUDA launch requires its sealed foundation state")
    state = read_json(Path(state_path))
    verify(state)
    required = state["toolchain"]["header_version"]
    if type(required) is not int or required < 1000:
        raise ValueError("invalid selected CUDA Driver API requirement")
    compat = Path(state["toolchain"]["cuda_home"]) / "compat"
    native = dict(environment, LD_PRELOAD="")
    directories = [part for part in environment.get("LD_LIBRARY_PATH", "").split(":") if part]
    native["LD_LIBRARY_PATH"] = ":".join(part for part in directories if Path(part).resolve() != compat.resolve())
    native_failure = None
    try:
        proof = _probe(native, "libcuda.so.1", required)
        actual = Path(proof["library"]).resolve()
        if (actual.is_relative_to(Path(state["toolchain"]["cuda_home"]).resolve())
                or "compat" in actual.parts or "stubs" in actual.parts):
            raise RuntimeError("native search resolved an implicit compatibility/stub provider")
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        native_failure = error
    else:
        environment["LD_LIBRARY_PATH"] = native["LD_LIBRARY_PATH"]
        environment["TF_CUDA_DRIVER_PROOF"] = json.dumps(dict(proof, mode="native"))
        return environment
    library = compat / "libcuda.so.1"
    if not library.is_file() or str(library.resolve(strict=True)) not in state["native_files"]:
        raise RuntimeError(f"native CUDA failed and sealed compatibility provider is missing: {native_failure}") from native_failure
    candidate = dict(native)
    candidate["LD_LIBRARY_PATH"] = str(compat) + (":" + native["LD_LIBRARY_PATH"] if native["LD_LIBRARY_PATH"] else "")
    try:
        proof = _probe(candidate, str(library), required)
        if Path(proof["library"]).resolve(strict=True) != library.resolve(strict=True):
            raise ValueError("compatibility probe mapped a different libcuda provider")
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"native CUDA probe: {native_failure}; compatibility probe: {error}") from error
    environment["LD_LIBRARY_PATH"] = candidate["LD_LIBRARY_PATH"]
    environment["TF_CUDA_DRIVER_PROOF"] = json.dumps(dict(proof, mode="forward-compatible"))
    return environment


def main() -> int:  # noqa: RET503 - successful os.execve never returns.
    try:
        environment = prepare_environment()
        guard = str(Path(__file__).with_name("foundation_guard.py"))
        prepared = subprocess.run([sys.executable, guard, "environment"], env=environment,
                                  check=True, capture_output=True, text=True)
        values = json.loads(prepared.stdout)
        if not isinstance(values, dict) or any(not isinstance(key, str) or not isinstance(value, str)
                                               for key, value in values.items()):
            raise ValueError("invalid protected foundation environment")
        environment.update(values)
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"TensorFold CUDA startup refused: {error}", file=sys.stderr)
        return 1
    # NVIDIA's entrypoint may transform loader variables. Check actual providers
    # after it has executed, immediately before the final application exec.
    os.execve(NVIDIA_ENTRYPOINT, [NVIDIA_ENTRYPOINT, sys.executable, guard, "checked-run", "--", *sys.argv[1:]], environment)


if __name__ == "__main__":
    raise SystemExit(main())

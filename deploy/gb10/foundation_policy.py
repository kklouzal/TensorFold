"""Shared, data-only policy for immutable CUDA foundation artifacts."""
from __future__ import annotations

import re
from urllib.parse import urlsplit

OFFICIAL_WHEELS = {"torch", "torchvision", "triton"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def is_accelerator_package(name: str) -> bool:
    name = canonical_name(name)
    return (name in {"torch", "torchvision", "torchaudio", "triton", "cupti-python"}
            or name.startswith(("nvidia-", "cuda-", "pytorch-triton", "triton-")))


def artifact_name(value: object, suffix: str) -> str:
    if (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.+\-]+", value)
            or value.startswith(".") or not value.endswith(suffix)):
        raise ValueError(f"expected one artifact basename ending in {suffix}")
    return value


def wheel_url(name: str, pin: dict) -> str:
    """Accept genuine upstream nightly wheels with exact whole-artifact hashes."""
    if (not isinstance(pin, dict) or not isinstance(pin.get("version"), str)
            or not isinstance(pin.get("sha256"), str) or not SHA256.fullmatch(pin["sha256"])):
        raise ValueError(f"invalid version or wheel SHA256 for {name}")
    if set(pin) != {"version", "sha256", "url"}:
        raise ValueError("foundation wheels require exact official URL/version/hash fields")
    url = pin["url"]
    if not isinstance(url, str):
        raise ValueError("foundation wheel URL must be text")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in {"download.pytorch.org", "download-r2.pytorch.org"}
            or parsed.username or parsed.password or parsed.fragment or any(char.isspace() for char in url)
            or not parsed.path.endswith(".whl")):
        raise ValueError(f"{name} must use its genuine official nightly wheel")
    return url


def validate_pins(pins: dict, architecture: str) -> None:
    if (pins.get("schema_version") != 2 or pins.get("platform") != f"linux/{architecture}"
            or architecture not in {"amd64", "arm64"} or pins.get("python") != "3.12"):
        raise ValueError("foundation pins require schema2 and the selected Linux CPython3.12 target")
    for key in ("base", "base_index"):
        value = pins.get(key, "")
        if not re.fullmatch(r"nvcr\.io/nvidia/[A-Za-z0-9_./:+\-]+@sha256:[0-9a-f]{64}", value):
            raise ValueError(f"{key} must identify an immutable official NGC image")
    if pins.get("nccl_provider", "native") not in {"native", "nvidia-nccl-cu13"}:
        raise ValueError("nccl_provider must be native or nvidia-nccl-cu13")
    roots, wheels = pins.get("foundation_roots"), pins.get("wheels")
    if (not isinstance(roots, list) or not 1 <= len(roots) <= 32
            or any(not isinstance(root, str) or not root or len(root) > 1024 for root in roots)
            or not isinstance(wheels, dict) or set(wheels) != OFFICIAL_WHEELS):
        raise ValueError("foundation roots and exact torch/torchvision/triton artifacts are required")
    if not isinstance(pins.get("cuda"), str) or not re.fullmatch(r"[A-Za-z0-9_.+\-]{1,160}", pins.get("cache_key", "")):
        raise ValueError("foundation requires its CUDA release and a distinct bounded cache key")
    for name, pin in wheels.items():
        wheel_url(name, pin)

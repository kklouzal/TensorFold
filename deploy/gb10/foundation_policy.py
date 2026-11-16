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


PYTHON = "3.15"
PBS_RELEASES = "https://github.com/astral-sh/python-build-standalone/releases/download/"
OWNED_BUILD_ENVIRONMENT = frozenset({"CFLAGS", "CXXFLAGS"})


def interpreter_url(pins: dict, machine: str) -> str:
    """Exact python-build-standalone install_only_stripped archive for one target machine."""
    record = pins.get("interpreter")
    if not isinstance(record, dict) or set(record) != {"version", "artifact", "url", "sha256", "bytes"}:
        raise ValueError("interpreter requires exact version/artifact/url/sha256/bytes fields")
    version, artifact, url = record["version"], record["artifact"], record["url"]
    if (not isinstance(version, str) or not re.fullmatch(re.escape(PYTHON) + r"\.[0-9]{1,3}", version)
            or not isinstance(record["sha256"], str) or not SHA256.fullmatch(record["sha256"])
            or type(record["bytes"]) is not int or not 0 < record["bytes"] <= 2**30):
        raise ValueError(f"interpreter must be one CPython {PYTHON} patch release with exact size and SHA256")
    match = re.fullmatch(re.escape(f"cpython-{version}+") + r"([0-9]{8})"
                         + re.escape(f"-{machine}-unknown-linux-gnu-install_only_stripped.tar.gz"),
                         artifact_name(artifact, ".tar.gz"))
    if match is None or not isinstance(url, str) or url != (
            PBS_RELEASES + match[1] + "/" + artifact.replace("+", "%2B")):
        raise ValueError(f"interpreter must be the official {machine} python-build-standalone archive")
    return url


def owned_builds(pins: dict) -> dict:
    """Selected versions without a usable target wheel, built from exact PyPI sdists."""
    builds = pins.get("owned_builds", {})
    if not isinstance(builds, dict) or len(builds) > 16:
        raise ValueError("owned_builds must map at most 16 package names to sdist records")
    for name, record in builds.items():
        if (not isinstance(name, str) or canonical_name(name) != name or is_accelerator_package(name)
                or name in OFFICIAL_WHEELS):
            raise ValueError(f"owned build name must be a canonical application package: {name!r}")
        if not isinstance(record, dict) or set(record) != {"version", "url", "sha256", "build_requires", "environment"}:
            raise ValueError(f"owned build {name} requires version/url/sha256/build_requires/environment")
        version, url, requirements, environment = (record[key] for key in ("version", "url", "build_requires",
                                                                           "environment"))
        if (not isinstance(version, str) or not re.fullmatch(r"[0-9][A-Za-z0-9.!+_-]{0,63}", version)
                or not isinstance(record["sha256"], str) or not SHA256.fullmatch(record["sha256"])):
            raise ValueError(f"invalid owned build version or sdist SHA256: {name}")
        parsed = urlsplit(url) if isinstance(url, str) else None
        if (parsed is None or parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org"
                or parsed.port is not None or parsed.username or parsed.password or parsed.query or parsed.fragment
                or any(char.isspace() for char in url)
                or parsed.path.rsplit("/", 1)[-1] != f"{name.replace('-', '_')}-{version}.tar.gz"):
            raise ValueError(f"owned build {name} must use its exact PyPI sdist")
        if (not isinstance(requirements, list) or not 1 <= len(requirements) <= 32
                or any(not isinstance(item, str) or not item or len(item) > 256 for item in requirements)):
            raise ValueError(f"owned build {name} requires its bounded build requirements")
        if (not isinstance(environment, dict) or not set(environment) <= OWNED_BUILD_ENVIRONMENT
                or any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9=_.,+\- ]{1,256}", value)
                       for value in environment.values())):
            raise ValueError(f"owned build {name} environment allows only bounded CFLAGS/CXXFLAGS")
    return builds


def validate_pins(pins: dict, architecture: str, machine: str) -> None:
    if (pins.get("schema_version") != 3 or pins.get("platform") != f"linux/{architecture}"
            or architecture not in {"amd64", "arm64"} or pins.get("python") != PYTHON):
        raise ValueError(f"foundation pins require schema3 and the selected Linux CPython{PYTHON} target")
    interpreter_url(pins, machine)
    owned_builds(pins)
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

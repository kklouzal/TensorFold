"""Declared Linux deployment architectures and their generated pin artifacts."""

from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class PlatformPins:
    architecture: str
    machine: str
    pins: str
    runtime: str
    verification: str
    foundation: str
    constraints: str
    manifest: str


PLATFORMS = MappingProxyType({
    "arm64": PlatformPins("arm64", "aarch64", "nightly-pins.json", "requirements-runtime.lock",
                          "requirements-test.lock", "requirements-foundation.lock", "foundation-constraints.txt",
                          "foundation-manifest.json"),
    "amd64": PlatformPins("amd64", "x86_64", "nightly-pins-amd64.json", "requirements-runtime-amd64.lock",
                          "requirements-test-amd64.lock", "requirements-foundation-amd64.lock",
                          "foundation-constraints-amd64.txt", "foundation-manifest-amd64.json"),
})


def platform_pins(architecture: str) -> PlatformPins:
    try:
        return PLATFORMS[architecture]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported deployment architecture {architecture!r}; use arm64 or amd64") from exc

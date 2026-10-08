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


PLATFORMS = MappingProxyType({
    "arm64": PlatformPins("arm64", "aarch64", "nightly-pins.json", "requirements-runtime.lock",
                          "requirements-test.lock"),
    "amd64": PlatformPins("amd64", "x86_64", "nightly-pins-amd64.json", "requirements-runtime-amd64.lock",
                          "requirements-test-amd64.lock"),
})


def platform_pins(architecture: str) -> PlatformPins:
    try:
        return PLATFORMS[architecture]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported deployment architecture {architecture!r}; use arm64 or amd64") from exc

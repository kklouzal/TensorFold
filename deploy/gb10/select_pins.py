#!/usr/bin/env python3
"""Select target pins before installation and reject a mismatched base runtime."""

import argparse
import json
import platform
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

from foundation_policy import is_accelerator_package, validate_pins, wheel_url
from platform_pins import PLATFORMS, platform_pins


def _records(content: bytes) -> list[str]:
    """Generated locks contain one requirement per line and optional comments."""
    return [line for line in content.decode("utf-8").splitlines() if line.strip() and not line.startswith("#")]


def _validate_artifacts(pins: dict, manifest: dict, contents: dict[str, bytes], target) -> None:
    if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1):
        raise ValueError("selected foundation manifest requires schema1")
    for key in ("platform", "python", "base", "base_index", "cuda", "cache_key", "foundation_roots"):
        if key not in manifest or manifest[key] != pins[key]:
            raise ValueError(f"selected foundation manifest differs from pins: {key}")
    if "native_addons" not in manifest or manifest["native_addons"] != pins.get("native_addons", []):
        raise ValueError("selected foundation manifest differs from pins: native_addons")
    if "nccl_provider" not in manifest or manifest["nccl_provider"] != pins.get("nccl_provider", "native"):
        raise ValueError("selected foundation manifest differs from pins: nccl_provider")
    if not isinstance(manifest["native_addons"], list) or len(manifest["native_addons"]) > 32:
        raise ValueError("selected native addons exceed the generator schema")
    roles = manifest.get("roles")
    if not isinstance(roles, dict) or set(roles) != {"foundation", "runtime", "verification"}:
        raise ValueError("selected manifest must contain the three generated package roles")
    seen = set()
    for role, filename in (("foundation", target.foundation), ("runtime", target.runtime),
                           ("verification", target.verification)):
        packages = roles[role]
        if not isinstance(packages, dict) or len(packages) > 256:
            raise ValueError("selected package roles exceed the generator schema")
        expected = []
        for name, record in sorted(packages.items()):
            if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or name in seen:
                raise ValueError("selected package roles require unique canonical names")
            seen.add(name)
            if not isinstance(record, dict):
                raise ValueError(f"invalid selected distribution record: {name}")
            if "source" in record or "artifact" in record:
                raise ValueError(f"unsupported source/artifact wheel metadata: {name}")
            version, url, digest = (record.get(key) for key in ("version", "url", "sha256"))
            if (not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9.!+_-]{1,1024}", version)
                    or not isinstance(url, str) or any(character.isspace() for character in unquote(url))
                    or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                raise ValueError(f"invalid selected distribution identity: {name}")
            if name in pins["wheels"]:
                pin = pins["wheels"][name]
                audited = {"version": pin["version"], "url": wheel_url(name, pin), "sha256": pin["sha256"]}
                if role != "foundation" or any(record[key] != audited[key] for key in audited):
                    raise ValueError(f"selected foundation wheel differs from pins: {name}")
            else:
                parsed = urlsplit(url)
                if (parsed.scheme != "https"
                        or parsed.hostname not in {"download.pytorch.org", "download-r2.pytorch.org",
                                                   "files.pythonhosted.org"}
                        or parsed.username or parsed.password or not parsed.path.endswith(".whl")):
                    raise ValueError(f"invalid selected distribution origin: {name}")
            if role != "foundation" and is_accelerator_package(name):
                raise ValueError(f"accelerator package outside selected foundation: {name}")
            expected.append(f"{name} @ {url} --hash=sha256:{digest}")
        if _records(contents[filename]) != expected:
            raise ValueError(f"selected lock differs from manifest role: {role}")
    if len(seen) > 256 or not pins["wheels"].keys() <= roles["foundation"].keys():
        raise ValueError("selected manifest omits audited foundation wheels or exceeds the generator schema")
    if manifest["nccl_provider"] == "nvidia-nccl-cu13" and "nvidia-nccl-cu13" not in roles["foundation"]:
        raise ValueError("selected NCCL wheel provider must belong to the protected foundation closure")
    expected_constraints = [f"{name}==={record['version']}" for name, record in sorted(roles["foundation"].items())]
    if _records(contents[target.constraints]) != expected_constraints:
        raise ValueError("selected constraints differ from protected foundation versions")


def select(architecture: str, directory: Path, *, base_image: str | None = None) -> None:
    target = platform_pins(architecture)
    if platform.system() != "Linux" or platform.machine() != target.machine:
        raise ValueError(f"requested linux/{architecture} pins on {platform.system()}/{platform.machine()}")
    copies = ((target.pins, "nightly-pins.json"), (target.runtime, "requirements-runtime.lock"),
              (target.verification, "requirements-test.lock"), (target.foundation, "requirements-foundation.lock"),
              (target.constraints, "foundation-constraints.txt"), (target.manifest, "foundation-manifest.json"))
    contents = {}
    for source, _ in copies:
        path = directory / source
        if not path.is_file():
            raise ValueError(f"missing or oversized deployment pins: {source}")
        with path.open("rb") as stream:
            contents[source] = stream.read(2 * 2**20 + 1)
        if len(contents[source]) > 2 * 2**20:
            raise ValueError(f"missing or oversized deployment pins: {source}")
    pins = json.loads(contents[target.pins])
    if not isinstance(pins, dict):
        raise ValueError("selected deployment pins must be an object")
    validate_pins(pins, architecture)
    if pins["python"] != f"{sys.version_info.major}.{sys.version_info.minor}":
        raise ValueError("selected pin metadata does not match the deployment runtime")
    if base_image is not None and base_image not in (pins["base"], pins["base_index"]):
        raise ValueError("actual base image does not match the selected child or index digest")
    _validate_artifacts(pins, json.loads(contents[target.manifest]), contents, target)
    for source, destination in copies:
        if source != destination:
            (directory / destination).write_bytes(contents[source])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=tuple(PLATFORMS), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--base-image", help="Actual FROM reference; must match the selected immutable child or index")
    args = parser.parse_args()
    select(args.architecture, args.directory, base_image=args.base_image)


if __name__ == "__main__":
    main()

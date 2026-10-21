"""Prepare the three canonical GB10 native recipes without a visible GPU.

Run only under the managed CPU guard after installing the owned wheel. Derived
modules live in the immutable image, outside mounted compiler caches. The guard
seals their identities and owned transitive inputs; ordinary wheel builds do
not invoke this operation. No runtime capability is emulated or monkeypatched.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import time
from pathlib import Path

import foundation_guard as guard


def inputs(package: Path, sources: list[str]) -> dict[str, str]:
    """Close the owned literal quoted-include graph inside the installed wheel."""
    pending, result = [Path(path).resolve(strict=True) for path in sources], {}
    while pending:
        path = pending.pop()
        name = str(path.relative_to(package))
        if name in result:
            continue
        raw = path.read_bytes()
        result[name] = hashlib.sha256(raw).hexdigest()
        for included in re.findall(rb'^\s*#\s*include\s*"([^"\r\n]+)"', raw, re.MULTILINE):
            dependency = (path.parent / included.decode()).resolve(strict=True)
            dependency.relative_to(package)
            pending.append(dependency)
    return result


def prepare(state: dict, destination: Path) -> dict:
    guard.verify(state, full=True)
    return guard.audited_operation(state, lambda: _prepare(state, destination), "native preparation")


def _prepare(state: dict, destination: Path) -> dict:
    from torch.utils import cpp_extension

    import tensorfold
    from tensorfold.cuda import build, native_artifact
    from tensorfold.cuda.nvfp4 import checkpoint
    from tensorfold.families.gemma4.cuda import head
    from tensorfold.families.qwen3_5.cuda import b16

    if os.environ.get("TF_FOUNDATION_RUNTIME_PHASE") != "cpu" or any(Path("/dev").glob("nvidia[0-9]*")):
        raise RuntimeError("native preparation requires managed CPU load-only admission and no GPU devices")
    if "application_wheel_sha256" not in state or "foundation_cache_key" not in state:
        raise ValueError("native preparation requires the installed owned wheel")
    if destination.exists():
        raise FileExistsError("native preparation requires a new image-owned destination")
    destination.mkdir(parents=True)
    package = Path(tensorfold.__file__).resolve(strict=True).parent
    target = (12, 1)  # This tool prepares only the declared GB10 image region.
    recipes = [checkpoint._build_options(target), b16._build_options(), head._cap30_build_options()]
    toolchain = state["toolchain"]
    identity = {"foundation_cache_key": state["foundation_cache_key"],
                "application_wheel_sha256": state["application_wheel_sha256"],
                "toolchain_sha256": hashlib.sha256(native_artifact.canonical(toolchain)).hexdigest()}
    manifest = {"schema_version": 1, "kind": "tensorfold-native-artifacts", "identity": identity,
                "abi": native_artifact.abi(), "package_root": str(package), "target_capability": list(target),
                "toolchain": toolchain, "entries": {}}
    timings = {}
    for recipe in recipes:
        options = dict(recipe)
        base, sources = options.pop("name"), options.pop("sources")
        need = build.minimum(options.pop("need", build.MIN_CAPABILITY), options.get("arch_specific", False))
        specific = options.pop("arch_specific", False)
        if target < need or options.pop("prepared", False) is not True:
            raise ValueError("unsupported or noncanonical prepared native recipe")
        flags = build.target_flags(target, specific)
        name = base + "_" + flags[0].split(",code=", 1)[1]
        options["extra_cuda_cflags"] = [*options.get("extra_cuda_cflags", []), *flags]
        request = native_artifact.request(package, sources, options)
        owned = inputs(package, sources)
        work = destination / (name + ".build")
        work.mkdir()
        started = time.monotonic()
        module = cpp_extension.load(name=name, sources=sources, build_directory=str(work), **options)
        timings[name] = time.monotonic() - started
        binary = destination / (name + Path(module.__file__).suffix)
        shutil.copyfile(module.__file__, binary)
        binary.chmod(0o644)
        guard.normalize_image_mtime(binary)
        entry = {"module_name": name, "request": request, "inputs": owned,
                 "recipe_sha256": hashlib.sha256(native_artifact.canonical({"request": request, "inputs": owned,
                     "identity": identity, "abi": manifest["abi"]})).hexdigest(),
                 "binary": binary.name, "binary_sha256": guard.digest(binary), "binary_bytes": binary.stat().st_size}
        manifest["entries"][name] = entry
        shutil.rmtree(work)
    path = destination / "manifest.json"
    guard.write_json(path, manifest)
    if path.stat().st_size > native_artifact.MAX_MANIFEST:
        raise ValueError("native manifest exceeds its consumer bound")
    guard.write_json(destination / "preparation.json", {"scope": "GPU-free compile/import; no native device execution",
                                                       "seconds": timings})
    guard.seal_native_artifacts(state, path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=Path(os.environ.get(guard.STATE_ENV,
                                                               "/opt/tensorfold-pins/foundation-state.json")))
    parser.add_argument("--out", type=Path, default=Path("/opt/tensorfold-native"))
    args = parser.parse_args()
    prepare(guard.read_json(args.state), args.out.resolve())


if __name__ == "__main__":
    main()

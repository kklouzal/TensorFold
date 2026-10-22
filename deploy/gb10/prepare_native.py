"""Prepare or audit the five common GB10 native providers without a visible GPU.

Run only under the managed CPU guard after installing the owned wheel. Derived
modules live in the immutable image, outside mounted compiler caches. The guard
seals their identities and owned transitive inputs; ordinary wheel builds do
not invoke this operation. No runtime capability is emulated or monkeypatched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import foundation_guard as guard


def prepare(state: dict, destination: Path) -> dict:
    guard.verify(state, full=True)
    return guard.audited_operation(state, lambda: _prepare(state, destination), "native preparation")


def _prepare(state: dict, destination: Path) -> dict:
    from torch.utils import cpp_extension

    import tensorfold
    from tensorfold.cuda import native_artifact

    if os.environ.get("TF_FOUNDATION_RUNTIME_PHASE") != "cpu" or any(Path("/dev").glob("nvidia[0-9]*")):
        raise RuntimeError("native preparation requires managed CPU load-only admission and no GPU devices")
    if "application_wheel_sha256" not in state or "foundation_cache_key" not in state:
        raise ValueError("native preparation requires the installed owned wheel")
    if "native_artifacts" in state:
        raise ValueError("native preparation is already sealed; rebuild the image")
    if destination.exists():
        raise FileExistsError("native preparation requires a new image-owned destination")
    destination.mkdir(parents=True)
    package = Path(tensorfold.__file__).resolve(strict=True).parent
    if not package.is_relative_to(guard.site_directory(Path(state["application"])).resolve(strict=True)):
        raise ValueError("native preparation must use the installed owned package")
    toolchain = state["toolchain"]
    identity = {"foundation_cache_key": state["foundation_cache_key"],
                "application_wheel_sha256": state["application_wheel_sha256"],
                "toolchain_sha256": hashlib.sha256(native_artifact.canonical(toolchain)).hexdigest()}
    manifest = {"schema_version": 1, "kind": "tensorfold-native-artifacts", "identity": identity,
                "abi": native_artifact.abi(), "package_root": str(package),
                "target_capability": list(native_artifact.TARGET),
                "toolchain": toolchain, "entries": {}}
    timings = {}
    for recipe in native_artifact.recipes():
        name, sources, options = native_artifact.compilation(recipe)
        owned = native_artifact.inputs(package, sources)
        # pip writes fresh installed files with filesystem timestamps. Prepare
        # only these owned inputs before compile/first seal, because image
        # export preserves whole seconds. Existing seals were verified above.
        for relative in owned:
            guard.normalize_image_mtime(package / relative)
        request = native_artifact.request(package, sources, options)
        exports = native_artifact.binding_exports(sources)
        work = destination / (name + ".build")
        work.mkdir()
        guard.write_json(destination / "preparation.json", {
            "scope": "GPU-free compile/import calls; no native device execution",
            "status": "PREPARING", "active": name, "seconds": dict(timings)})
        print(f"Preparing native provider {name}", flush=True)
        started = time.monotonic()
        module = cpp_extension.load(name=name, sources=sources, build_directory=str(work), **options)
        native_artifact.check_exports(module, exports)
        timings[name] = time.monotonic() - started
        binary = destination / (name + Path(module.__file__).suffix)
        shutil.copyfile(module.__file__, binary)
        binary.chmod(0o644)
        guard.normalize_image_mtime(binary)
        entry = {"module_name": name, "request": request, "inputs": owned, "exports": exports,
                 "recipe_sha256": hashlib.sha256(native_artifact.canonical({"request": request, "inputs": owned,
                     "exports": exports, "identity": identity, "abi": manifest["abi"]})).hexdigest(),
                 "binary": binary.name, "binary_sha256": guard.digest(binary), "binary_bytes": binary.stat().st_size}
        manifest["entries"][name] = entry
        shutil.rmtree(work)
    path = destination / "manifest.json"
    guard.write_json(path, manifest)
    if path.stat().st_size > native_artifact.MAX_MANIFEST:
        raise ValueError("native manifest exceeds its consumer bound")
    guard.write_json(destination / "preparation.json", {
        "scope": "GPU-free compile/import calls; no native device execution",
        "status": "RECIPES_COMPLETE", "active": None, "seconds": timings})
    guard.seal_native_artifacts(state, path)
    return manifest


def audit(state: dict, destination: Path) -> dict:
    """CPU load-only check of final installed recipes, exports and providers."""
    guard.verify(state, full=True)
    return guard.audited_operation(state, lambda: _audit(state, destination), "native artifact audit")


def _audit(state: dict, destination: Path) -> dict:
    import torch
    from foundation_runtime import mapped_runtime

    from tensorfold.cuda import native_artifact

    if (os.environ.get("TF_FOUNDATION_RUNTIME_PHASE") != "cpu" or torch.cuda.is_initialized()
            or any(Path("/dev").glob("nvidia[0-9]*"))):
        raise RuntimeError("native artifact audit requires CPU load-only admission")
    contract = state.get("native_artifacts")
    path = destination / "manifest.json"
    if not isinstance(contract, dict) or path.resolve(strict=True) != Path(contract["manifest"]).resolve(strict=True):
        raise ValueError("native audit must use the sealed manifest")
    # Recheck the producer/schema/input contract before importing any library.
    manifest, _files = guard.validate_native_artifacts(state, path)
    if guard.digest(path) != contract["sha256"] or manifest["identity"] != contract["identity"]:
        raise ValueError("native audit activation differs from its sealed manifest")
    before = guard.native_proof(state)
    owners, records = [], {}
    for recipe in native_artifact.recipes():
        name, sources, _options = native_artifact.compilation(recipe)
        entry = manifest["entries"][name]
        binary = destination / entry["binary"]
        module = native_artifact._import(name, binary)
        owners.append(module)  # retain every loaded library through provider verification
        expected = native_artifact.binding_exports(sources)
        native_artifact.check_exports(module, expected)
        records[name] = {"binary": str(binary), "sha256": guard.digest(binary), "exports": expected}
    after = mapped_runtime(state["runtime"], allow_vision=True)
    if after != before["providers"] or torch.cuda.is_initialized():
        raise ValueError("native audit changed protected providers or initialized CUDA")
    return {"scope": "CPU load-only; no GPU/kernel/model/performance qualification", "entries": records,
            "manifest_sha256": guard.digest(path), "providers_unchanged": True, "cuda_initialized": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=Path(os.environ.get(guard.STATE_ENV,
                                                               "/opt/tensorfold-pins/foundation-state.json")))
    parser.add_argument("--out", type=Path, default=Path("/opt/tensorfold-native"))
    parser.add_argument("--audit", action="store_true", help="check the sealed bundle without compiling or using a GPU")
    args = parser.parse_args()
    state, destination = guard.read_json(args.state), args.out.resolve()
    if args.audit:
        print(json.dumps(audit(state, destination), sort_keys=True))
    else:
        prepare(state, destination)


if __name__ == "__main__":
    main()

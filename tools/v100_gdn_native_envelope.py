"""Resource/cache envelope for the explicitly selected standalone GDN trial.

The numerical procedure remains v100_gdn_chunk64_trial.py. This process owns
its CUDA context and cache until that synchronous procedure and retirement
callbacks finish. A failed diagnostic publication must preserve its primary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import stat
import sys
import time
import uuid
from pathlib import Path

from tensorfold.cleanup import error_name, raise_failures

COMPONENT_FILES = frozenset(("gdn_chunk64.py", "gdn_chunk64_artifact.py",
                             "gdn_chunk64_kernels.py", "gdn_chunk64_provenance.json"))


def snapshot(torch):
    free, total = torch.cuda.mem_get_info(0)
    return {
        "host_monotonic_seconds": time.monotonic(),
        "self_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "children_maxrss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "cgroup": {name: (Path("/sys/fs/cgroup") / name).read_text()
                   for name in ("memory.max", "memory.current", "memory.peak", "memory.events")},
        "torch_allocated_bytes": torch.cuda.memory_allocated(0),
        "torch_reserved_bytes": torch.cuda.memory_reserved(0),
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "device_free_bytes": free, "device_total_bytes": total,
        "scope": "Process-hierarchy and CUDA snapshots, not an allocation timeline; opaque allocations remain separate.",
    }


def retain_cache(source, target, *, entry_limit=8192, byte_limit=1 << 30):
    """Retain a retired compiler cache, without following SDK symlinks.

    Call only after synchronous compilation, GPU work and trace publication
    finish. No cache writer may outlive that handoff. Limits count logical
    bytes and directory entries, separately from tmpfs/cgroup capacity.
    """
    entries, logical_bytes = [], 0
    for path in source.rglob("*"):
        info = path.lstat()
        if len(entries) >= entry_limit:
            raise RuntimeError("native cache file-count grant exhausted")
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)):
            raise RuntimeError("unsupported native cache file type")
        if stat.S_ISREG(info.st_mode):
            logical_bytes += info.st_size
        if logical_bytes > byte_limit:
            raise RuntimeError("native cache logical-byte grant exhausted")
        entries.append(path)
    shutil.copytree(source, target, symlinks=True)
    identities = {}
    for path in entries:
        relative = path.relative_to(source)
        copied = target / relative
        if path.is_symlink():
            original, retained = os.readlink(path), os.readlink(copied)
            if original != retained:
                raise RuntimeError("native cache symlink changed during retention")
            identities[str(relative)] = {"symlink": retained}
        elif path.is_file():
            with path.open("rb") as stream:
                original = hashlib.file_digest(stream, "sha256").hexdigest()
            with copied.open("rb") as stream:
                retained = hashlib.file_digest(stream, "sha256").hexdigest()
            if original != retained:
                raise RuntimeError("native cache changed during retention")
            identities[str(relative)] = {"bytes": copied.stat().st_size, "sha256": retained}
    return {"files": identities, "logical_bytes": logical_bytes, "entry_count": len(entries),
            "scope": "Complete bounded cache copy before tmpfs retirement; no symlink targets followed"}


def complete_trial(action, report, *, after, retain, publish):
    """Attempt every retirement/report stage before transporting any failure."""
    primary, errors = None, []
    report["primary"], report["cleanup"] = None, []
    try:
        action()
    except BaseException as error:  # noqa: BLE001 - retire all stages before transporting the primary
        primary = error
        try:
            report["primary"] = {"type": error_name(error)}
        except BaseException as diagnostic:  # noqa: BLE001 - diagnostic failure cannot replace the primary
            errors.append(diagnostic)
    for stage, callback in (("after", after), ("retained_cache", retain)):
        try:
            report[stage] = callback()
        except BaseException as error:  # noqa: BLE001 - every retirement callback must be attempted
            errors.append(error)
            try:
                report["cleanup"].append({"stage": stage, "type": error_name(error)})
            except BaseException as diagnostic:  # noqa: BLE001 - retain formatting and resource failures
                errors.append(diagnostic)
    try:
        publish(report)
    except BaseException as error:  # noqa: BLE001 - preserve kernel failure through publication failure
        errors.append(error)
    raise_failures(primary, errors)


def selected_data(path, size, sha256):
    if (type(size) is not int or not 0 < size <= 1 << 20
            or re.fullmatch("[0-9a-f]{64}", sha256) is None):
        raise ValueError("native envelope requires bounded selected input identity")
    with path.open("rb") as stream:
        raw = stream.read(size + 1)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("native envelope input differs from its selected identity")
    # Root selected this immutable, generated DATA by exact bytes; it is not
    # an independently supplied model/request document.
    data = json.loads(raw)
    if data.get("schema") != "tensorfold.g3-native-data-v2":
        raise ValueError("unsupported native envelope DATA")
    if set(data["component_hashes"]) != COMPONENT_FILES:
        raise ValueError("native envelope needs all four installed component identities")
    return data


def invoke_procedure(raw, source, arguments):
    """Run the selected bytes and restore the process argument owner.

    The numerical procedure must return normally. A zero SystemExit (such as
    parser help) cannot make an unfinished trial or failed retirement exit0.
    """
    previous = sys.argv
    try:
        sys.argv = [str(source), *arguments]
        # Execute the already hashed bytes, avoiding a second file read/race.
        exec(compile(raw, str(source), "exec"),  # noqa: S102 - exact Root-selected trusted numerical procedure
             {"__name__": "__main__", "__file__": str(source)})
    except SystemExit as error:
        if error.code in (None, 0):
            raise RuntimeError("native numerical procedure exited before returning to its envelope") from error
        raise
    finally:
        sys.argv = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-bytes", type=int, required=True)
    parser.add_argument("--data-sha256", required=True)
    args = parser.parse_args(argv)
    data = selected_data(args.data, args.data_bytes, args.data_sha256)
    source = Path(data["tool_path"])
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != data["tool_sha256"]:
        raise RuntimeError("owned native numerical procedure changed")
    if sorted(os.sched_getaffinity(0)) != data["cpuset_cpus"]:
        raise RuntimeError("native CPU affinity differs from its selected plan")
    import tensorfold.cuda.sm70.gdn_chunk64 as component

    root = Path(component.__file__).parent
    for name, expected in data["component_hashes"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("installed GDN component differs: " + name)
    import torch

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (7, 0):
        raise RuntimeError("native trial requires exactly one actual SM70 GPU")
    torch.cuda.set_device(0)
    actual = "GPU-" + str(uuid.UUID(str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")))
    if actual != data["gpu_uuid"]:
        raise RuntimeError("native logical GPU differs from its selected physical UUID")
    torch.cuda.reset_peak_memory_stats(0)
    report = {"schema": "tensorfold.g3-native-envelope-v2", "data": data,
              "actual_gpu_uuid": actual, "before": snapshot(torch),
              "quality_acceptance": False, "performance_qualified": False}
    out = Path("/output")

    def action():
        invoke_procedure(raw, source, data["tool_argv"])

    def publish(value):
        with (out / "native-envelope.json").open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")

    complete_trial(action, report, after=lambda: snapshot(torch),
                   retain=lambda: retain_cache(Path("/cache"), out / "compile-cache"), publish=publish)


if __name__ == "__main__":
    main()

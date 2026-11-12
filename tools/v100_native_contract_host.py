"""Finite host owner for native-contract suites; JSON DATA, no generated Python.

Import/help are inert. Run from the canonical tools package or stage its unchanged
stdlib helper closure. The host never imports Torch or starts an inner model.

Safe external cancellation: retain run_token from controller-identity.json at
launch, then use --out SAME_DIRECTORY --cancel-token RETAINED_TOKEN. Never
precheck/kill a numeric controller PID. The request is generation-bound and
idempotent; host-result.json, not request publication, proves retirement.
Empty uuids selects a CPU-only runc worker; it never qualifies native CUDA.

Regular JSON remains capability-zero. An explicit verification_launcher contract
selects ONLY the owned file-capability launcher image/envelope, drops ALL Docker
capabilities and declares CHECKPOINT_RESTORE alone. The trusted inner must prove
UID/GID/caps, immutable Python/launcher and mapped providers BEFORE qualification;
the host validates that receipt without restoring loader environment or granting
capabilities itself. This opt-in is verification-only, never a serving default.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import stat
import subprocess
import time

if __package__:
    from . import v100_owned_model_contract as model_contract
    from .benchmark_output import write_json
    from .openai_protocol import object_json
    from .worker_lifetime import raise_failures
else:
    import v100_owned_model_contract as model_contract
    from benchmark_output import write_json
    from openai_protocol import object_json
    from worker_lifetime import raise_failures

SCHEMA = "tensorfold.native-contract-host-v1"
SHA = re.compile(r"[0-9a-f]{64}\Z")
UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
LIMIT = 8 << 20
VERIFICATION_MODE = "map-files-cap40-v1"
VERIFICATION_ENTRY = "/opt/qualification/bin/native-mapfile-cap-launcher"
VERIFICATION_SOURCE = "/opt/qualification/native_mapfile_cap_launcher.c"
VERIFICATION_PYTHON = "/opt/venv/bin/python3"
VERIFICATION_XATTR = "0100000200000000000000000001000000000000"
VERIFICATION_CAP = "0000010000000000"
_INJECTION_ENV = ("LD_PRELOAD", "LD_AUDIT", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _io():
    # Existing primary-safe local file/FD owners, loaded only after CLI parsing.
    if __package__:
        from . import gb10_io
    else:
        import gb10_io
    return gb10_io


def _regular_bytes(fd, maximum):
    info = os.fstat(fd)
    require(stat.S_ISREG(info.st_mode) and 0 <= info.st_size <= maximum, "bounded regular input required")
    chunks, left = [], info.st_size
    while left:
        chunk = os.read(fd, min(left, 65536))
        require(bool(chunk), "input truncated")
        chunks.append(chunk)
        left -= len(chunk)
    after = os.fstat(fd)
    require((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "input changed during read")
    return b"".join(chunks)


def read_bytes(path, maximum):
    with _io().owned_descriptor(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC) as fd:
        return _regular_bytes(fd, maximum)


def read_kernel(path, maximum):
    """Bound proc/cgroup pseudo-files whose stat size is intentionally zero."""
    with _io().owned_descriptor(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC) as fd:
        chunks, total = [], 0
        while True:
            chunk = os.read(fd, min(65536, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            require(total <= maximum, "kernel observation exceeds bound")
        return b"".join(chunks)


def file_identity(item):
    require(type(item) is dict and set(item) == {"path", "sha256", "bytes"}, "invalid input identity")
    require(type(item["bytes"]) is int and 0 <= item["bytes"] < 2**63, "declared raw file extent required")
    require(isinstance(item["sha256"], str) and SHA.fullmatch(item["sha256"]), "input SHA required")
    require(type(item["path"]) is str and Path(item["path"]).is_absolute(), "absolute input path required")
    return item


def bound_file(item):
    """Structured JSON only: retain its independent bounded materialization cap."""
    file_identity(item)
    require(0 < item["bytes"] <= LIMIT, "structured JSON exceeds read bound")
    raw = read_bytes(item["path"], item["bytes"])
    require(len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"], "input identity differs")
    return raw


def check_raw_file(item):
    """Stream one immutable declared extent; do not materialize raw asset bytes.

    Tokenizer/data/code extents are not structured-spec JSON budgets. The held
    descriptor AND path must retain the same regular-file identity throughout.
    """
    file_identity(item)
    helpers = _io()
    with helpers.owned_descriptor(item["path"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC) as fd:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size == item["bytes"], "raw input extent/type differs")
        checksum, remaining = hashlib.sha256(), item["bytes"]
        while remaining:
            chunk = os.read(fd, min(65536, remaining))
            require(bool(chunk), "raw input truncated")
            checksum.update(chunk)
            remaining -= len(chunk)
        require(helpers.stamp(os.fstat(fd)) == helpers.stamp(before)
                and helpers.stamp(os.stat(item["path"], follow_symlinks=False)) == helpers.stamp(before),
                "raw input changed during verification")
        require(checksum.hexdigest() == item["sha256"], "raw input SHA differs")
    return item


def _path(value):
    require(type(value) is str and Path(value).is_absolute() and "\x00" not in value, "absolute ordinary path required")
    require(".." not in Path(value).parts and ":" not in value and "," not in value, "ambiguous mount path")
    return value



def _verification_contract(value, image):
    require(type(value) is dict and set(value) == {"mode", "base_image_id", "launcher", "python", "providers"}
            and value["mode"] == VERIFICATION_MODE, "invalid explicit verification launcher mode")
    require(type(value["base_image_id"]) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", value["base_image_id"])
            and value["base_image_id"] != image, "distinct exact verification base image required")
    launcher = value["launcher"]
    require(type(launcher) is dict and set(launcher) == {"path", "bytes", "sha256", "source_bytes", "source_sha256", "file_capability_hex"}
            and launcher["path"] == VERIFICATION_ENTRY and launcher["file_capability_hex"] == VERIFICATION_XATTR,
            "owned file-capability launcher identity required")
    file_identity({key: launcher[key] for key in ("path", "bytes", "sha256")})
    file_identity({"path": VERIFICATION_SOURCE, "bytes": launcher["source_bytes"], "sha256": launcher["source_sha256"]})
    require(type(value["python"]) is dict and set(value["python"]) == {"path", "bytes", "sha256"}
            and value["python"]["path"] == VERIFICATION_PYTHON, "fixed protected Python identity required")
    file_identity(value["python"])
    require(launcher["bytes"] > 0 and launcher["source_bytes"] > 0 and value["python"]["bytes"] > 0,
            "nonempty verification executables/source required")
    providers = value["providers"]
    require(type(providers) is dict and providers, "nonempty explicit provider identities required")
    for path, identity in providers.items():
        require(type(identity) is dict and set(identity) == {"bytes", "sha256"}, "invalid provider identity")
        _path(path)
        require(".so" in Path(path).name, "mapped shared-library provider required")
        file_identity({"path": path, **identity})
        require(identity["bytes"] > 0, "nonempty mapped provider required")
    return value


def _verification_environment(env):
    require(all(key not in env for key in _INJECTION_ENV), "verification cannot inject Python or loader code")
    require(env.get("PYTHONOPTIMIZE") in (None, "", "0"), "verification must retain Python assertions")


def validate_spec(raw):
    required = {"schema", "name", "image_id", "uuids", "source", "inputs", "mounts", "environment",
                "entrypoint", "argv", "resources", "health_baseline", "timeouts", "disk_bytes", "diagnostics"}
    require(type(raw) is dict and set(raw) in (required, required | {"verification_launcher"},
            required | {"verification_launcher", "numa"}) and raw["schema"] == SCHEMA, "invalid native host schema/fields")
    require(type(raw["name"]) is str and re.fullmatch(r"[a-z][a-z0-9-]{0,62}", raw["name"]), "invalid attempt name")
    require(type(raw["image_id"]) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", raw["image_id"]), "immutable image required")
    ids = raw["uuids"]
    require(type(ids) is list and 0 <= len(ids) <= 8 and all(type(v) is str and UUID.fullmatch(v) for v in ids)
            and len(set(ids)) == len(ids), "distinct exact requested GPU UUIDs required")
    require(type(raw["inputs"]) is list and 1 <= len(raw["inputs"]) <= 128, "bounded immutable inputs required")
    require(type(raw["source"]) is dict and set(raw["source"]) == {"root", "manifest"}, "source inventory required")
    _path(raw["source"]["root"])
    file_identity(raw["source"]["manifest"])
    for item in raw["inputs"]:
        file_identity(item)
    require(type(raw["mounts"]) is list and 1 <= len(raw["mounts"]) <= 16, "bounded mounts required")
    targets = set()
    for mount in raw["mounts"]:
        require(type(mount) is dict and set(mount) == {"source", "target", "read_only"}, "invalid mount")
        _path(mount["source"])
        _path(mount["target"])
        require(type(mount["read_only"]) is bool and mount["target"] not in targets, "duplicate/invalid mount")
        targets.add(mount["target"])
    require({"/canonical", "/operator", "/evidence", "/cache"} <= targets, "explicit source/operator/evidence/cache mounts required")
    for left in targets:
        require(all(left == right or not Path(left).is_relative_to(Path(right)) for right in targets), "overlapping mount targets refused")
    mounts = {m["target"]: m for m in raw["mounts"]}
    require(mounts["/canonical"] == {"source": raw["source"]["root"], "target": "/canonical", "read_only": True}
            and mounts["/operator"]["read_only"] and not mounts["/evidence"]["read_only"]
            and not mounts["/cache"]["read_only"], "mount role isolation differs")
    env = raw["environment"]
    require(type(env) is dict and len(env) <= 128 and all(type(k) is str and re.fullmatch(r"[A-Z_][A-Z0-9_]*", k)
            and type(v) is str and len(v) <= 4096 and "\x00" not in v for k, v in env.items()), "invalid environment")
    require(not env.get("PYTEST_ADDOPTS"), "external pytest options cannot narrow the complete module contract")
    require(env.get("CUDA_VISIBLE_DEVICES") == ",".join(ids)
            and env.get("NVIDIA_VISIBLE_DEVICES") == (",".join(ids) if ids else "void"), "environment GPU list differs")
    if ids:
        require(env.get("NVIDIA_DRIVER_CAPABILITIES") == "compute,utility", "compute/utility capabilities required")
    _path(raw["entrypoint"])
    if "verification_launcher" in raw:
        _verification_contract(raw["verification_launcher"], raw["image_id"])
        require(raw["entrypoint"] == VERIFICATION_ENTRY, "verification must use the exact owned launcher entrypoint")
        _verification_environment(env)
        protected = ("/proc", "/sys", "/dev", "/opt", "/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/tmp")
        require(all(not (Path(target).is_relative_to(root) or Path(root).is_relative_to(target))
                    for target in targets for root in protected), "verification mount shadows protected process/image paths")
    require(type(raw["argv"]) is list and len(raw["argv"]) == 12
            and all(type(s) is str and len(s) <= 4096 and "\x00" not in s for s in raw["argv"]), "invalid inner argv")
    argv = raw["argv"]
    require(argv[:2] == ["-B", "/operator/v100_native_contract_driver.py"]
            and argv[2] == "--spec" and argv[4] == "--spec-sha256" and argv[6] == "--spec-bytes"
            and argv[8:] == ["--source", "/canonical", "--evidence", "/evidence"]
            and SHA.fullmatch(argv[5]) and re.fullmatch(r"[0-9]+", argv[7])
            and 0 < int(argv[7]) <= LIMIT and Path(argv[3]).parent == Path("/operator"),
            "canonical inner driver/data interface required; no embedded Python")
    inputs = {item["path"]: item for item in raw["inputs"]}
    operator = Path(mounts["/operator"]["source"])
    if 'numa' in raw:
        contract = model_contract.prepared_teacher_contract(numa=True)
        contract.validate_numa(raw['numa'], operator_root=str(operator))
        for key in contract.FILES:
            expected = raw['numa'][key]
            require(inputs.get(expected['path']) == expected, 'sealed NUMA payload lacks exact outer input authority')
        require(env.get('TENSORFOLD_NUMA_RECEIPT_DIR') == contract.RECEIPT_DIR
                and env.get('NCCL_IGNORE_CPU_AFFINITY') == '0', 'fixed NUMA receipt/NCCL environment required')
    require(str(operator / "v100_native_contract_driver.py") in inputs, "canonical driver input hash required")
    inner_identity = inputs.get(str(operator / Path(argv[3]).name))
    require(inner_identity is not None and inner_identity["sha256"] == argv[5]
            and inner_identity["bytes"] == int(argv[7]), "inner spec bytes/SHA not bound to mounted input")
    resources = raw["resources"]
    require(type(resources) is dict and set(resources) == {"nano_cpus", "memory_bytes", "memory_swap_bytes", "ipc", "tmpfs", "ulimits"}, "invalid resource contract")
    for key in ("nano_cpus", "memory_bytes", "memory_swap_bytes"):
        require(type(resources[key]) is int and 0 < resources[key] < 2**63, "positive resource bound required")
    require(resources["memory_swap_bytes"] == resources["memory_bytes"], "native contract requires zero swap")
    require(resources["ipc"] in ("host", "private"), "declared IPC envelope required")
    require(type(resources["tmpfs"]) is dict and set(resources["tmpfs"]) == {"/tmp"}
            and type(resources["tmpfs"]["/tmp"]) is str, "explicit bounded /tmp required")
    require("size=" in resources["tmpfs"]["/tmp"], "tmpfs size required")
    require(type(resources["ulimits"]) is list and resources["ulimits"], "ulimits required")
    names = set()
    for item in resources["ulimits"]:
        require(type(item) is dict and set(item) == {"Name", "Soft", "Hard"}
                and type(item["Name"]) is str and re.fullmatch(r"[a-z]+", item["Name"])
                and item["Name"] not in names and all(type(item[k]) is int and item[k] >= -1 for k in ("Soft", "Hard")), "invalid ulimit")
        names.add(item["Name"])
    timeouts = raw["timeouts"]
    require(type(timeouts) is dict and set(timeouts) == {"run", "poll", "monitor", "retire"}, "explicit finite timeouts required")
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in timeouts.values())
            and timeouts["poll"] <= timeouts["monitor"] <= timeouts["run"]
            and timeouts["run"] / timeouts["poll"] <= 65536, "invalid timeout ordering or poll capacity")
    diagnostics = raw["diagnostics"]
    require(type(diagnostics) is dict and set(diagnostics) == {"sample_limit", "process_limit", "read_limit_bytes"}
            and type(diagnostics["sample_limit"]) is int and 1 <= diagnostics["sample_limit"] <= 4096
            and type(diagnostics["process_limit"]) is int and 1 <= diagnostics["process_limit"] <= 256
            and type(diagnostics["read_limit_bytes"]) is int and 1024 <= diagnostics["read_limit_bytes"] <= 65536,
            "bounded diagnostic capacities required")
    require(timeouts["run"] / timeouts["monitor"] < diagnostics["sample_limit"], "monitor capacity cannot cover run deadline")
    require(type(raw["disk_bytes"]) is dict and set(raw["disk_bytes"]) == {"start", "running"}
            and all(type(v) is int and v >= 0 for v in raw["disk_bytes"].values()), "disk budgets required")
    health = raw["health_baseline"]
    require(type(health) is dict and set(ids) <= set(health) and len(health) <= 32, "complete health UUID inventory required")
    for uuid, values in health.items():
        require(UUID.fullmatch(uuid) and type(values) is list and len(values) == 6
                and all(type(x) is int and x >= 0 for x in values[:5]) and values[5] == "No"
                and values[1] == values[2] == values[4] == 0, "invalid ECC/retirement baseline")
    return raw


def source_inventory(spec):
    raw = object_json(bound_file(spec["source"]["manifest"]))
    require(raw.get("schema") == "tensorfold.reviewed-test-only-input-fixture-v1"
            and raw.get("snapshot_clean_commit") is False and raw.get("base_app_image") == spec.get("verification_launcher", {}).get("base_image_id", spec["image_id"])
            and raw.get("source_host") == spec["source"]["root"]
            and type(raw.get("base_commit")) is str and re.fullmatch(r"[0-9a-f]{40}", raw["base_commit"])
            and type(raw.get("archive_sha256")) is str and SHA.fullmatch(raw["archive_sha256"]),
            "explicit reviewed test-only fixture provenance required; not a clean commit")
    files, reviewed = raw.get("files"), raw.get("reviewed_test_hashes")
    require(type(files) is dict and 1 <= len(files) <= 100000 and type(reviewed) is dict and reviewed,
            "bounded source/reviewed-test inventory required")
    for name, row in files.items():
        require(type(name) is str and not Path(name).is_absolute() and ".." not in Path(name).parts
                and type(row) is dict and set(row) == {"type", "bytes", "sha256", "stored_mode"}
                and row["type"] == "file" and type(row["bytes"]) is int and row["bytes"] >= 0
                and type(row["sha256"]) is str and SHA.fullmatch(row["sha256"])
                and type(row["stored_mode"]) is int and 0 <= row["stored_mode"] <= 0o7777,
                "current reviewed fixture requires explicit regular-file extents/modes; symlinks unqualified")
    require(all(name in files and files[name]["sha256"] == value for name, value in reviewed.items()),
            "reviewed test hashes differ from inventory")
    return files, {key: value for key, value in raw.items() if key != "files"}


def validate_data(spec):
    """Bind the canonical inner mode/data BEFORE any output or Docker effect."""
    source_files, _ = source_inventory(spec)
    operator = next(m["source"] for m in spec["mounts"] if m["target"] == "/operator")
    inner_path = str(Path(operator) / Path(spec["argv"][3]).name)
    inner_input = next(item for item in spec['inputs'] if item['path'] == inner_path)
    inner = object_json(bound_file(inner_input))
    benchmark = inner.get('schema') == model_contract.BENCHMARK_SCHEMA
    numa = inner.get('schema') == model_contract.NUMA_PREPARED_TEACHER_SCHEMA
    prepared = inner.get('schema') in (model_contract.PREPARED_TEACHER_SCHEMA, model_contract.NUMA_PREPARED_TEACHER_SCHEMA)
    contract = model_contract.prepared_teacher_contract(numa=numa) if prepared else model_contract
    model = inner.get('schema') in (model_contract.SCHEMA, model_contract.BENCHMARK_SCHEMA,
                                   model_contract.PREPARED_TEACHER_SCHEMA, model_contract.NUMA_PREPARED_TEACHER_SCHEMA)
    require(('numa' in spec) == numa, 'NUMA host extension requires its exact versioned prepared operation')
    if model:
        if numa:
            contract.validate_host_operation(spec, inner, numa_bundle=_numa_authority(spec))
        else:
            (model_contract.validate_benchmark_host_operation if benchmark else contract.validate_host_operation)(spec, inner)
        for name, identity in inner['source_files'].items():
            require(name in source_files and all(source_files[name][key] == identity[key] for key in ('bytes', 'sha256')),
                    'model operator differs from complete reviewed source fixture')
        inputs = {Path(item['path']).name: item for item in spec['inputs'] if Path(item['path']).parent == Path(operator)}
        for name in ('v100_native_contract_driver.py', 'v100_owned_model_contract.py'):
            require(name in inputs and all(inputs[name][key] == inner['source_files']['tools/' + name][key]
                                          for key in ('bytes', 'sha256')), 'staged operator does not match canonical model source')
        for label in (('benchmark_plan', 'benchmark_client') if benchmark else ('teacher_plan', 'teacher_client')):
            item = inner['operation'][label]
            staged = inputs.get(Path(item['path']).name)
            require(staged is not None and all(staged[key] == item[key] for key in ('bytes', 'sha256')),
                    'model DATA not bound in readonly operator closure')
        if prepared:
            for label in ('case_authority', 'execution_manifest'):
                item = inner['operation'][label]
                staged = inputs.get(Path(item['path']).name)
                require(staged is not None and all(staged[key] == item[key] for key in ('bytes', 'sha256')),
                        'prepared teacher case/profile lacks its sealed operator input')
            require('v100_onecat_combinations.py' in inputs and all(
                inputs['v100_onecat_combinations.py'][key] == inner['source_files']['tools/v100_onecat_combinations.py'][key]
                for key in ('bytes', 'sha256')), 'prepared contract differs from canonical staged source')
        if numa:
            for key in contract.FILES:
                item = inner['operation']['numa'][key]
                staged = inputs.get(Path(item['path']).name)
                require(staged is not None and all(staged[k] == item[k] for k in ('bytes', 'sha256')),
                        'NUMA operation payload differs from staged Host input')
            for name in ('v100_prepared_teacher_numa.py', 'generate_numa_seccomp.py'):
                require(name in inputs and all(inputs[name][k] == inner['source_files']['tools/'+name][k]
                                               for k in ('bytes', 'sha256')), 'NUMA contract/generator source differs')
        result = {key: inner[key] for key in ('mode', 'source_commit', 'image_id', 'operation')}
    else:
        mode = "cuda" if spec["uuids"] else "cpu"
        require(inner.get("schema") == "tensorfold.v100-native-contract.v1" and inner.get("mode") == mode
                and inner.get("UUIDs") == spec["uuids"] and inner.get("SM") == ([7, 0] if spec["uuids"] else [])
                and inner.get("image_id") == spec["image_id"], "host/inner mode, devices or image differ")
        result = {key: inner[key] for key in ("mode", "source_commit", "image_id", "modules")}
    for item in spec['inputs']:
        if item is not inner_input:
            # Large target bytes are bound/hash-checked once by the inner owner.
            # The host input closure is operator/frontend authority DATA only.
            require(not model or not item['path'].endswith('.gguf'), 'model payloads must not be duplicate host hash inputs')
            check_raw_file(item)
    require(("verification_launcher" in inner) == ("verification_launcher" in spec)
            and inner.get("verification_launcher") == spec.get("verification_launcher"), "host/inner verification mode differs")
    if "verification_launcher" in spec:
        require(all(key in inner for key in ("app_root", "expected_app", "sdk_manifest", "fd", "torch")),
                "verification runtime identity fields required")
        result["runtime"] = {"app_root": inner["app_root"], "app_file_count": len(inner["expected_app"]),
            "app_inventory_sha256": hashlib.sha256(json.dumps(inner["expected_app"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "sdk_sha256": inner["sdk_manifest"]["sha256"], "fd_sha256": inner["fd"]["sha256"],
            "fd_source_sha256": inner["fd"]["source_sha256"], "torch_git": inner["torch"]["git_version"], "torch_cuda": inner["torch"]["cuda"]}
    return result


def _cpu_arg(nano):
    whole, fractional = divmod(nano, 10**9)
    return str(whole) if not fractional else str(whole) + "." + f"{fractional:09d}".rstrip("0")


def _numa_authority(spec):
    """Read only the explicit, independently captured verification-cap40 role."""
    source, _ = source_inventory(spec)
    return model_contract.prepared_teacher_contract(numa=True).frozen_host_inputs(spec, source)


def ownership_label(spec):
    return spec["name"] + ":" + hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def create_argv(spec, output):
    mounts = {m["target"]: m for m in spec["mounts"]}
    require(mounts["/evidence"]["source"] == str(output), "output/evidence mount differs")
    r = spec["resources"]
    argv = ["docker", "create", "--name", spec["name"], "--label", "org.tensorfold.native-contract=" + ownership_label(spec),
            "--runtime", "nvidia" if spec["uuids"] else "runc",
            *(["--gpus", '"device=' + ",".join(spec["uuids"]) + '"'] if spec["uuids"] else []),
            "--ipc", r["ipc"], "--network", "none", "--read-only", "--user", "1000:1000",
            "--cpus", _cpu_arg(r["nano_cpus"]), "--memory", str(r["memory_bytes"]), "--memory-swap", str(r["memory_swap_bytes"])]
    if "verification_launcher" in spec:
        argv.extend(("--cap-drop", "ALL", "--cap-add", "CHECKPOINT_RESTORE"))
    if 'numa' in spec:
        bundle = _numa_authority(spec)
        argv.extend(('--cpuset-cpus', bundle['cpuset_cpus'], '--cpuset-mems', bundle['cpuset_mems'],
                     '--security-opt', 'seccomp='+spec['numa']['seccomp']['path']))
    for key, value in sorted(r["tmpfs"].items()):
        argv.extend(("--tmpfs", key + ":" + value))
    for item in r["ulimits"]:
        argv.extend(("--ulimit", f'{item["Name"]}={item["Soft"]}:{item["Hard"]}'))
    for key, value in sorted(spec["environment"].items()):
        argv.extend(("-e", key + "=" + value))
    for m in spec["mounts"]:
        argv.extend(("--mount", f'type=bind,src={m["source"]},dst={m["target"]},readonly={str(m["read_only"]).lower()}'))
    return [*argv, "-w", "/canonical", "--entrypoint", spec["entrypoint"], spec["image_id"], *spec["argv"]]


def owns_container(document, spec, cid):
    return (document.get("Id") == cid and document.get("Name") == "/" + spec["name"]
            and document.get("Config", {}).get("Labels", {}).get("org.tensorfold.native-contract") == ownership_label(spec))


def admit(document, spec, cid):
    require(owns_container(document, spec, cid), "created container ownership differs")
    h, c = document["HostConfig"], document["Config"]
    r = spec["resources"]
    verification = "verification_launcher" in spec
    numa_bundle = _numa_authority(spec) if 'numa' in spec else None
    if numa_bundle is None:
        security_ok = h.get("SecurityOpt") in (None, [], *([["label=disable"]] if r["ipc"] == "host" else []))
    else:
        options = h.get('SecurityOpt')
        require(type(options) is list and len(options) in (1, 2)
                and all(type(item) is str for item in options) and len(set(options)) == len(options),
                'actual NUMA security options differ from fixed policy')
        selected = [item.removeprefix('seccomp=') for item in options if item.startswith('seccomp=')]
        other = [item for item in options if not item.startswith('seccomp=')]
        require(len(selected) == 1 and len(selected[0]) <= 2 << 20
                and other in ([], *([['label=disable']] if r['ipc'] == 'host' else [])),
                'only the sealed NUMA seccomp and existing host-IPC label option are permitted')
        # Docker CLI resolves seccomp=FILE to its JSON payload before the API
        # call. Compare the actual inspected policy, never a path-only claim.
        security_ok = object_json(selected[0]) == object_json(numa_bundle['raw']['seccomp'])
    require(document["Image"] == spec["image_id"] and h["Runtime"] == ("nvidia" if spec["uuids"] else "runc")
            and h["NetworkMode"] == "none" and h["ReadonlyRootfs"] is True and h["Privileged"] is False
            and not h.get("Devices")
            and (h.get("CapAdd") == ["CAP_CHECKPOINT_RESTORE"] and h.get("CapDrop") == ["ALL"]
                 if verification else not h.get("CapAdd") and not h.get("CapDrop"))
            and security_ok
            and not h.get("DeviceCgroupRules") and not h.get("PidMode") and not h.get("AutoRemove")
            and c["User"] == "1000:1000" and c["Entrypoint"] == [spec["entrypoint"]]
            and c["Cmd"] == spec["argv"] and c["WorkingDir"] == "/canonical", "image/command/isolation admission differs")
    if verification:
        require(not h.get("GroupAdd") and not h.get("Init")
                and set(document.get("NetworkSettings", {}).get("Networks", {})) == {"none"},
                "verification requires private init/group/network envelope")
    if numa_bundle is not None:
        require(h.get('CpusetCpus') == numa_bundle['cpuset_cpus']
                and h.get('CpusetMems') == numa_bundle['cpuset_mems'],
                'actual NUMA CPU/memory cgroup allowance differs from the sealed plan')
    requests = h.get("DeviceRequests")
    if spec["uuids"]:
        require(type(requests) is list and len(requests) == 1 and requests[0]["DeviceIDs"] == spec["uuids"]
                and requests[0].get("Driver") in ("", "nvidia") and type(requests[0].get("Count")) is int and requests[0]["Count"] == 0
                and requests[0].get("Options") in (None, {})
                and requests[0].get("Capabilities") == [["gpu"]], "exact requested GPU membership differs")
    else:
        require(requests in (None, []), "CPU route must request no GPU device")
    require(all(h[k] == r[v] for k, v in (("NanoCpus", "nano_cpus"), ("Memory", "memory_bytes"), ("MemorySwap", "memory_swap_bytes"), ("IpcMode", "ipc")))
            and h.get("Tmpfs") == r["tmpfs"], "resource admission differs")
    require(sorted(h.get("Ulimits", []), key=lambda x: x["Name"]) == sorted(r["ulimits"], key=lambda x: x["Name"]), "ulimit admission differs")
    require(len({x.split("=", 1)[0] for x in c["Env"]}) == len(c["Env"]), "duplicate inspect environment key")
    env = dict(x.split("=", 1) for x in c["Env"])
    require(not env.get("PYTEST_ADDOPTS"), "inherited pytest options cannot narrow the complete module contract")
    require(all(env.get(k) == v for k, v in spec["environment"].items()), "environment admission differs")
    if verification:
        _verification_environment(env)
    actual = []
    for m in document["Mounts"]:
        if m["Type"] == "tmpfs":
            require(m["Destination"] in r["tmpfs"], "unrequested tmpfs")
        else:
            require(m["Type"] == "bind" and type(m["RW"]) is bool, "unrequested/invalid mount type")
            actual.append({"source": m["Source"], "target": m["Destination"], "read_only": not m["RW"]})
    require(sorted(actual, key=lambda x: x["target"]) == sorted(spec["mounts"], key=lambda x: x["target"]), "mount identity differs")




def verification_operator(spec, spec_identity=None):
    """No unbound Python neighbour may run before the trusted PREQUAL gate."""
    root = Path(next(m["source"] for m in spec["mounts"] if m["target"] == "/operator"))
    allowed = {Path(item["path"]).name: item for item in
               [*spec["inputs"], spec["source"]["manifest"], *([spec_identity] if spec_identity else [])]
               if Path(item["path"]).parent == root}
    require(not root.is_symlink() and root.is_dir() and root.stat().st_uid == 1000
            and not stat.S_IMODE(root.stat().st_mode) & 0o222, "verification operator directory must be frozen/owned")
    actual = set()
    for path in root.iterdir():
        require(path.name in allowed and not path.is_symlink() and path.is_file()
                and path.stat().st_uid == 1000 and not stat.S_IMODE(path.stat().st_mode) & 0o222,
                "unmanaged or writable verification operator input")
        check_raw_file(allowed[path.name])
        actual.add(path.name)
    require(actual == set(allowed), "verification operator closure missing")
    return {name: allowed[name] for name in sorted(actual)}


def verification_image(candidate, base, spec):
    """Verify purpose-image lineage without changing/resealing its SDK."""
    require(candidate.get("Id") == spec["image_id"]
            and base.get("Id") == spec["verification_launcher"]["base_image_id"], "verification image IDs differ")
    layers, original = candidate.get("RootFS", {}).get("Layers"), base.get("RootFS", {}).get("Layers")
    require(type(layers) is list and type(original) is list and 0 < len(original) < len(layers) <= 4096
            and all(type(v) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", v) for v in layers)
            and layers[:len(original)] == original, "verification image is not the exact declared base derivative")
    return {"image_id": candidate["Id"], "base_image_id": base["Id"], "base_layers": original, "purpose_layers": layers}


def verification_receipt(raw, spec, expected):
    """Bind trusted inner PREQUAL evidence, never a substitute provider loader."""
    fields = {"schema", "mode", "image_id", "source_commit", "verification_launcher", "PID", "uids", "gids", "caps",
              "launcher", "python", "providers", "runtime", "loader_environment", "NoNewPrivs"}
    require(type(raw) is dict and set(raw) == fields and raw["schema"] == "tensorfold.verification-launcher-preflight-v1"
            and raw["mode"] == VERIFICATION_MODE and raw["verification_launcher"] == spec["verification_launcher"]
            and raw["image_id"] == spec["image_id"] and raw["source_commit"] == expected["source_commit"],
            "verification prequalification receipt identity/schema differs")
    require(type(raw["PID"]) is int and raw["PID"] == 1
            and all(type(raw[k]) is list and len(raw[k]) == 4 and all(type(v) is int and v == 1000 for v in raw[k])
                    for k in ("uids", "gids"))
            and raw["caps"] == {k: VERIFICATION_CAP for k in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")}
            and raw["NoNewPrivs"] == "0" and raw["loader_environment"] == {"LD_LIBRARY_PATH": None},
            "actual verification UID/GID/capabilities/loader environment differs")
    contract, launcher, python = spec["verification_launcher"], raw["launcher"], raw["python"]
    require(type(launcher) is dict and set(launcher) == {"path", "bytes", "sha256", "mode", "uid", "source", "file_capability_hex"}
            and {k: launcher[k] for k in ("path", "bytes", "sha256")} == {k: contract["launcher"][k] for k in ("path", "bytes", "sha256")}
            and type(launcher["mode"]) is int and launcher["mode"] == 0o555
            and type(launcher["uid"]) is int and launcher["uid"] == 0
            and launcher["file_capability_hex"] == VERIFICATION_XATTR
            and launcher["source"] == {"path": VERIFICATION_SOURCE, "bytes": contract["launcher"]["source_bytes"],
                "sha256": contract["launcher"]["source_sha256"], "mode": 0o444, "uid": 0},
            "actual owned launcher/source/xattr differs")
    require(type(python) is dict and set(python) == {"path", "resolved_path", "bytes", "sha256", "mode", "uid"}
            and {k: python[k] for k in ("path", "bytes", "sha256")} == contract["python"]
            and python["resolved_path"] == "/usr/bin/python3.12"
            and type(python["mode"]) is int and python["mode"] == 0o755
            and type(python["uid"]) is int and python["uid"] == 0, "actual protected Python differs")
    require(type(launcher["source"]) is dict and type(launcher["source"].get("uid")) is int
            and type(launcher["source"].get("mode")) is int, "launcher source identity types differ")
    require(raw["runtime"] == expected["runtime"], "prequalification installed App/SDK/FD/Torch differs")
    providers = raw["providers"]
    require(type(providers) is dict and providers.get("schema") == "tensorfold.native-elf-observation-v3"
            and providers.get("identity_method") == "kernel-vma-backed-fd-v1" and type(providers.get("PID")) is int
            and providers["PID"] == 1 and providers.get("missing_required_provider_paths") == []
            and providers.get("required_providers") == contract["providers"], "provider FD proof summary differs")
    rows = providers.get("mapped_elfs")
    require(type(rows) is list and rows and all(type(row) is dict for row in rows), "mapped provider FD rows required")
    for path, identity in contract["providers"].items():
        require(any(row.get("path") == path and type(row.get("bytes")) is int and row["bytes"] == identity["bytes"]
                    and row.get("sha256") == identity["sha256"] for row in rows), "required provider backing identity missing")
    return raw


def verification_status(text):
    """Host-side live status only after PREQUAL; EPERM is an owning failure."""
    wanted = {"Uid", "Gid", "CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb", "NoNewPrivs"}
    fields = {}
    for line in text.splitlines():
        name, separator, value = line.partition(":")
        if separator and name in wanted:
            require(name not in fields, "duplicate verification process status field")
            fields[name] = value.split()
    require(set(fields) == wanted
            and all(fields[k] == ["1000"]*4 for k in ("Uid", "Gid"))
            and all(fields[k] == [VERIFICATION_CAP] for k in wanted - {"Uid", "Gid", "NoNewPrivs"})
            and fields["NoNewPrivs"] == ["0"], "running verification identity/caps differs")
    return fields


def cgroup_entry(text):
    rows = [line[3:] for line in text.splitlines() if line.startswith("0::")]
    require(len(rows) == 1 and rows[0].startswith("/") and ".." not in Path(rows[0]).parts, "unambiguous cgroup-v2 membership required")
    return rows[0]


def process_group_allowed(cgroups, owned):
    group, parent = Path(cgroup_entry(cgroups)), Path(owned)
    require(group == parent or group.is_relative_to(parent), "foreign process cgroup")


def suite_result(raw, mode):
    require(type(raw) is dict and raw.get("schema") == "tensorfold.native-contract-result-v1" and raw.get("mode") == mode
            and {"pytest_exit", "JUnit", "post_errors", "primary_error", "diagnostic_errors", "context_fatal", "source_after_match", "first_failure_observed"} <= set(raw), "inner result schema differs")
    require(type(raw["post_errors"]) is list and type(raw["diagnostic_errors"]) is list, "inner error journals must be lists")
    require(type(raw.get("pytest_exit")) is int and raw["pytest_exit"] == 0 and not raw.get("post_errors")
            and raw.get("primary_error") is None and not raw.get("diagnostic_errors") and raw.get("context_fatal") is False
            and raw["first_failure_observed"] is False,
            "inner suite failed")
    junit = raw["JUnit"]
    require(type(junit) is dict, "JUnit counts must be an object")
    counts = {}
    for key in ("tests", "failures", "errors", "skipped"):
        value = junit[key]
        require((type(value) is int and value >= 0) or (type(value) is str and re.fullmatch(r"[0-9]+", value)), "invalid JUnit count")
        counts[key] = int(value)
    require(counts["tests"] > 0 and all(counts[key] == 0 for key in ("failures", "errors", "skipped")), "required suite failed/empty/skipped")
    require(raw.get("source_after_match") is True, "inner source identity changed/unverified")
    return counts


class ProcessDeparted(RuntimeError):
    """A pinned ownership observation raced an actual process departure."""


class System:
    """Linux/Docker boundary using existing bounded metadata-command owner."""
    def __init__(self, output, spec_identity=None):
        require(hasattr(os, "pidfd_open"), "pidfd-capable Linux host required; numeric-PID fallback forbidden")
        require(os.getuid() == 1000, "UID1000 controller/evidence ownership required before container effects")
        if __package__:
            from .gb10_controller import Host
        else:
            from gb10_controller import Host
        self.host, self.output = Host(output), output
        self.pidfd, self.group, self.init_pid = None, None, None
        self.spec_identity = spec_identity

    def command(self, argv, timeout=30):
        return self.host.command(argv, timeout).decode()

    def inspect(self, cid):
        rows = object_json(self.command(["docker", "inspect", "--format", "{{json .}}", cid]))
        return rows

    def snapshot(self, spec):
        manifest, fixture = source_inventory(spec)
        root = Path(spec["source"]["root"])
        actual = {}
        for path in root.rglob("*"):
            require(not path.is_symlink(), "source symlink refused")
            if path.is_file():
                name = str(path.relative_to(root))
                require(name in manifest, "unrecorded source file")
                row = manifest[name]
                require(type(row) is dict and row["type"] == "file", "source file identity schema differs")
                require(type(row["bytes"]) is int and row["bytes"] >= 0 and isinstance(row["sha256"], str)
                        and SHA.fullmatch(row["sha256"]), "invalid source file identity")
                require(stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) == row["stored_mode"]
                        and path.stat(follow_symlinks=False).st_size == row["bytes"]
                        and _io().sha256(path) == row["sha256"], "source file identity changed")
                actual[name] = row
        require(set(actual) == set(manifest), "source file missing")
        for item in spec["inputs"]:
            check_raw_file(item)
        if self.spec_identity is not None:
            bound_file(self.spec_identity)
        image = self.command(["docker", "image", "inspect", "--format", "{{.Id}}", spec["image_id"]]).strip()
        require(image == spec["image_id"], "image identity changed")
        result = {"source_manifest": spec["source"]["manifest"]["sha256"], "source_fixture": fixture, "source": actual,
                  "inputs": spec["inputs"], "image_id": image}
        if 'numa' in spec:
            bundle = _numa_authority(spec)
            provenance = bundle['manifest']['provenance']
            versions = object_json(self.command(['docker', 'version', '--format', '{{json .}}']))
            security = json.loads(self.command(['docker', 'info', '--format', '{{json .SecurityOptions}}']))
            require(versions == provenance['docker_versions'] and security == provenance['daemon_security'],
                    'captured CAP40 seccomp daemon/runtime authority changed; regenerate the exact role policy')
            result['numa_policy_authority'] = {'daemon_versions': versions, 'daemon_security': security,
                'capture_role': bundle['manifest']['capture_role'],
                'profile_sha256': spec['numa']['seccomp']['sha256'],
                'manifest_sha256': spec['numa']['seccomp_manifest']['sha256']}
        if "verification_launcher" in spec:
            result["verification_operator"] = verification_operator(spec, self.spec_identity)
            candidate = object_json(self.command(["docker", "image", "inspect", "--format", "{{json .}}", image]))
            base = object_json(self.command(["docker", "image", "inspect", "--format", "{{json .}}", spec["verification_launcher"]["base_image_id"]]))
            result["verification_image"] = verification_image(candidate, base, spec)
        return result

    def start(self, cid, log):
        return subprocess.Popen(["docker", "start", "--attach", cid], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def health(self, spec, owned=None):
        fields = "index,uuid,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total,retired_pages.single_bit_ecc.count,retired_pages.double_bit.count,retired_pages.pending"
        raw = self.command(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader"])
        facts = {}
        for row in csv.reader(raw.splitlines()):
            index, uuid, volatile, aggregate, single, double, pending = [v.strip() for v in row]
            require(uuid not in facts, "duplicate health UUID")
            facts[uuid] = [int(index), int(volatile), int(aggregate), int(single), int(double), pending]
        require(facts == spec["health_baseline"], "GPU health inventory changed")
        processes = self.command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader"])
        for row in csv.reader(processes.splitlines()):
            require(owned is not None, "unexpected pre/post GPU consumer")
            uuid, pid = row[0].strip(), int(row[1])
            require(uuid in spec["uuids"] and pid > 0, "unexpected GPU UUID/PID consumer")
            with _pidfd(pid) as fd:
                if select.select([fd], [], [], 0)[0]:
                    raise ProcessDeparted("GPU process departed before ownership proof")
                text = read_kernel(Path("/proc") / str(pid) / "cgroup", 65536).decode()
                process_group_allowed(text, owned)
                if select.select([fd], [], [], 0)[0]:
                    raise ProcessDeparted("GPU process departed during ownership proof")
        return {"facts": facts, "raw_gpu": raw, "raw_compute": processes}

    def bind(self, cid, document, spec):
        state = document["State"]
        require(state["Running"] and type(state["Pid"]) is int and state["Pid"] > 0, "running init PID required")
        pid = state["Pid"]
        self.pidfd = os.pidfd_open(pid)
        self.init_pid = pid
        if select.select([self.pidfd], [], [], 0)[0]:
            raise ProcessDeparted("container init already exited")
        group = cgroup_entry(read_kernel(Path("/proc") / str(pid) / "cgroup", 65536).decode())
        require(Path(group).name in (cid, "docker-" + cid + ".scope"), "container cgroup identity differs")
        self.group = group
        current = self.inspect(cid)
        if terminal(current):
            raise ProcessDeparted("container exited during startup binding")
        require(current["State"]["Pid"] == pid and not select.select([self.pidfd], [], [], 0)[0], "startup PID changed")
        return self.sample(spec)

    def verification_process(self, cid, spec):
        document = self.inspect(cid)
        require(owns_container(document, spec, cid), "verification process container identity differs")
        if terminal(document):
            raise ProcessDeparted("verification process already retired")
        require(self.pidfd is not None and not select.select([self.pidfd], [], [], 0)[0], "verification PID owner departed")
        pid = document["State"]["Pid"]
        require(pid == self.init_pid, "verification init differs from pinned PID owner")
        status = verification_status(read_kernel(Path("/proc") / str(pid) / "status", 65536).decode())
        process_group_allowed(read_kernel(Path("/proc") / str(pid) / "cgroup", 65536).decode(), self.group)
        current = self.inspect(cid)
        require(current["State"]["Pid"] == pid and not select.select([self.pidfd], [], [], 0)[0], "verification PID changed")
        return {"host_PID": pid, "status": status}

    def sample(self, spec):
        require(self.group is not None, "owned cgroup not bound")
        root = (Path("/sys/fs/cgroup") / self.group.lstrip("/")).resolve(strict=True)
        require(root.is_relative_to(Path("/sys/fs/cgroup")), "cgroup path escaped")
        values = {name: read_kernel(root / name, 65536).decode() for name in
                  ("memory.current", "memory.peak", "memory.events", "memory.max", "memory.swap.max", "cpu.max")}
        r = spec["resources"]
        require(values["memory.max"].strip() == str(r["memory_bytes"]) and values["memory.swap.max"].strip() == "0", "actual cgroup memory/swap differs")
        quota, period = values["cpu.max"].split()
        require(quota != "max" and int(quota) * 10**9 == r["nano_cpus"] * int(period), "actual CPU quota differs")
        events = dict(line.split() for line in values["memory.events"].splitlines())
        require(int(events.get("oom", 0)) == int(events.get("oom_kill", 0)) == 0, "owned cgroup OOM event")
        return {"cgroup": self.group, "values": values, "scope": "cumulative owned cgroup observations, not phase-exclusive allocation bounds"}

    def observe(self, cid, spec):
        """Bounded phase/compiler observations; never a phase-exclusive bound."""
        phase = None
        try:
            phase = object_json(read_bytes(self.output / "phase.json", 65536))
        except FileNotFoundError:
            pass
        rows = self.command(["docker", "top", cid, "-eo", "pid,ppid,args"]).splitlines()[1:]
        require(len(rows) <= spec["diagnostics"]["process_limit"], "process observation capacity exceeded")
        compilers = []
        for row in rows:
            fields = row.split(None, 2)
            require(len(fields) == 3 and fields[0].isdigit(), "invalid owned process observation")
            if not any(name in fields[2] for name in ("ninja", "nvcc", "cicc", "ptxas", "cc1plus", "collect2", "c++", "g++")):
                continue
            pid = int(fields[0])
            try:
                with _pidfd(pid) as fd:
                    root = Path("/proc") / str(pid)
                    process_group_allowed(read_kernel(root / "cgroup", 65536).decode(), self.group)
                    argv = read_kernel(root / "cmdline", spec["diagnostics"]["read_limit_bytes"]).replace(b"\0", b" ").decode()
                    status = read_kernel(root / "status", spec["diagnostics"]["read_limit_bytes"]).decode()
                    if select.select([fd], [], [], 0)[0]:
                        compilers.append({"pid": pid, "departed_during_observation": True})
                    else:
                        compilers.append({"pid": pid, "ppid": fields[1], "argv": argv,
                                          "memory": [s for s in status.splitlines() if s.startswith(("VmRSS:", "VmHWM:"))]})
            except (FileNotFoundError, ProcessLookupError):
                compilers.append({"pid": pid, "departed_before_observation": True})
        return {"phase": phase, "compiler_processes": compilers,
                "scope": "owned process samples/cumulative HWM; not exact phase peak or allocator bound"}

    def close(self):
        errors = []
        # Existing bounded command owner retains any child whose wait failed.
        # Keep those exact Popen owners; never signal a borrowed numeric PID.
        remaining = []
        for child in self.host.pending_clients:
            try:
                child.kill()
            except BaseException as error:
                errors.append(error)
            try:
                child.wait(timeout=5)
            except BaseException as error:
                remaining.append(child)
                errors.append(error)
        self.host.pending_clients = remaining
        if self.pidfd is not None:
            positive = False
            try:
                require(bool(select.select([self.pidfd], [], [], 0)[0]), "pinned container init retirement unverified")
                if self.group is not None:
                    path = Path("/sys/fs/cgroup") / self.group.lstrip("/") / "cgroup.procs"
                    try:
                        require(not read_kernel(path, 65536).strip(), "owned cgroup still contains processes")
                    except FileNotFoundError:
                        pass  # Removed group after pinned init exit has no remaining members.
                positive = True
            except BaseException as error:
                errors.append(error)
            if positive:
                fd, self.pidfd = self.pidfd, None
                try:
                    os.close(fd)  # Never retry ambiguous descriptor retirement.
                except BaseException as error:
                    errors.append(error)
        raise_failures(None, errors)


def _pidfd(pid):
    # Existing primary-safe FD context manager accepts an acquired descriptor only
    # through its path-open API, so this tiny owner handles pidfd acquisition.
    from contextlib import contextmanager

    @contextmanager
    def owned():
        fd, primary = os.pidfd_open(pid), None
        try:
            yield fd
        except BaseException as error:
            primary = error
        errors = []
        try:
            os.close(fd)
        except BaseException as error:
            errors.append(error)
        raise_failures(primary, errors)
    return owned()


def _read_at(directory, name, maximum=4096):
    with _io().owned_descriptor(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory) as fd:
        return object_json(_regular_bytes(fd, maximum))


def request_cancel(output, token):
    """Publish one generation-bound request; NEVER signal a numeric PID.

    Caller retains run_token from controller-identity.json at launch. Directory
    FD + token-named exclusive publication cannot cancel a reused output path.
    Return means requested, not acknowledgement or proven cohort retirement.
    """
    require(type(token) is str and SHA.fullmatch(token), "exact retained run token required")
    output = Path(output)
    require(output.is_absolute(), "absolute control output required")
    with _io().owned_descriptor(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW) as directory:
        require(os.fstat(directory).st_uid == os.getuid(), "control directory owner differs")
        identity = _read_at(directory, "controller-identity.json")
        require(identity.get("schema") == "tensorfold.native-contract-control-v1"
                and identity.get("run_token") == token, "stale/reused operation identity")
        name = "cancel-" + token + ".json"
        message = {"schema": "tensorfold.native-contract-cancel-v1", "run_token": token}
        try:
            existing = _read_at(directory, name)
        except FileNotFoundError:
            existing = None
        if existing is not None:
            require(existing == message, "cancellation channel contains different data")
            return {"requested": True, "already_requested": True, "acknowledged": False}
        import secrets
        temporary = ".cancel-" + secrets.token_hex(16)
        primary, errors, created, already = None, [], False, False
        try:
            with _io().owned_descriptor(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                        0o600, dir_fd=directory) as fd:
                created = True
                view = memoryview((json.dumps(message, sort_keys=True) + "\n").encode())
                while view:
                    count = os.write(fd, view)
                    require(count > 0, "control publication truncated")
                    view = view[count:]
                os.fsync(fd)
            try:
                os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
            except FileExistsError:
                require(_read_at(directory, name) == message, "cancellation channel contains different data")
                already = True
            os.fsync(directory)
        except BaseException as error:
            primary = error
        if created:
            try:
                os.unlink(temporary, dir_fd=directory)
            except BaseException as error:
                errors.append(error)
        raise_failures(primary, errors)
        return {"requested": True, "already_requested": already, "acknowledged": False}


def cancellation(directory, token):
    try:
        value = _read_at(directory, "cancel-" + token + ".json")
    except FileNotFoundError:
        return False
    require(value == {"schema": "tensorfold.native-contract-cancel-v1", "run_token": token},
            "invalid operation-owned cancellation request")
    return True


def terminal(document):
    state = document["State"]
    return state["Status"] in ("exited", "dead") and state["Running"] is False and state["Pid"] == 0


def _exception_type(error):
    # Native class-name descriptor; do not invoke exception/user repr or str.
    return type.__dict__["__name__"].__get__(type(error))[:64]


def run(spec, output, system=None, *, spec_identity=None):
    """Own one fresh CID and its attach client; retain every outcome before raise."""
    import shutil
    validate_spec(spec)
    expected_inner = validate_data(spec)
    output = Path(output)
    require(output.is_absolute() and not output.exists() and not output.is_symlink(), "fresh absolute output required")
    require(all(not parent.is_symlink() for parent in output.parents), "symlink output ancestry refused")
    argv = create_argv(spec, output)
    system = System(output, spec_identity) if system is None else system
    import secrets
    run_token = secrets.token_hex(32)
    output.mkdir(mode=0o700)
    primary, errors, cid, process, log, before, pre, final = None, [], None, None, None, None, None, None
    owned, cancelled, control_fd = False, [None], None
    previous = {}
    outcome = {"schema": "tensorfold.native-contract-host-result-v1", "name": spec["name"], "samples": [], "source_match": False}

    def stop(number, _frame):
        cancelled[0] = number

    try:
        control_fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        write_json(output / "controller-identity.json", {"schema": "tensorfold.native-contract-control-v1",
                   "run_token": run_token, "name": spec["name"], "host_spec_sha256": ownership_label(spec).split(":")[-1]})
        outcome["run_token"] = run_token
        before = system.snapshot(spec)
        pre = system.health(spec)
        require(shutil.disk_usage(output).free >= spec["disk_bytes"]["start"], "initial disk headroom")
        write_json(output / "preflight.json", {"inputs": before, "health": pre, "create_argv": argv})
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[number] = signal.signal(number, stop)
        require(not system.command(["docker", "container", "ls", "-a", "--filter", "name=^/" + spec["name"] + "$", "--format", "{{.ID}}"]).strip(), "attempt container name already exists")
        if cancelled[0] is not None or cancellation(control_fd, run_token):
            raise InterruptedError("native controller cancelled before create")
        outcome["create_attempted"] = True
        raw = system.command(argv, timeout=60).strip()
        require(bool(SHA.fullmatch(raw)), "Docker create did not return full CID")
        cid = raw
        document = system.inspect(cid)
        owned = owns_container(document, spec, cid)
        write_json(output / "created.json", document)
        admit(document, spec, cid)  # MUST precede any start.
        require(document["State"]["Status"] == "created" and document["State"]["Running"] is False
                and document["State"]["Pid"] == 0, "new container ran before admission")
        if cancelled[0] is not None or cancellation(control_fd, run_token):
            raise InterruptedError("native controller cancelled before start")
        log = (output / "console.log").open("xb")
        outcome["attach_publication_uncertain"] = True
        process = system.start(cid, log)
        outcome["attach_publication_uncertain"] = False
        deadline, next_sample = time.monotonic() + spec["timeouts"]["run"], 0
        bound = False
        while True:
            if cancelled[0] is not None or cancellation(control_fd, run_token):
                raise InterruptedError("native controller cancelled by owned request or signal " + str(cancelled[0]))
            require(time.monotonic() < deadline, "native container deadline")
            document = system.inspect(cid)
            if terminal(document):
                final = document
                break
            if document["State"]["Running"] and document["State"]["Pid"] > 0 and not bound:
                try:
                    outcome["cgroup_start"] = system.bind(cid, document, spec)
                    bound = True
                except (FileNotFoundError, ProcessLookupError, ProcessDeparted):
                    raced = system.inspect(cid)
                    if not terminal(raced):
                        raise
                    final = raced
                    break
            if bound and "verification_launcher" in spec and "verification_preflight" not in outcome:
                try:
                    receipt = object_json(read_bytes(output / "verification-launcher-preflight.json", LIMIT))
                except FileNotFoundError:
                    pass  # Trusted inner refuses CUDA/tests until it publishes this verified receipt.
                else:
                    outcome["verification_preflight"] = verification_receipt(receipt, spec, expected_inner)
                    try:
                        outcome["verification_process"] = system.verification_process(cid, spec)
                    except (FileNotFoundError, ProcessLookupError, ProcessDeparted):
                        require(terminal(system.inspect(cid)), "verification process disappeared while still running")
            if process.poll() is not None:
                # The attach client alone does not prove container retirement.
                require(terminal(system.inspect(cid)), "attach exited while container is not terminal")
                final = system.inspect(cid)
                break
            if bound and time.monotonic() >= next_sample:
                try:
                    require(len(outcome["samples"]) < spec["diagnostics"]["sample_limit"], "monitor capacity exhausted")
                    sample = {"health": system.health(spec, system.group), "cgroup": system.sample(spec),
                              "processes": system.observe(cid, spec)}
                    name = f"host-observation-{len(outcome['samples']):04d}.json"
                    write_json(output / name, sample)
                    outcome["samples"].append({"file": name})
                except (FileNotFoundError, ProcessLookupError, ProcessDeparted):
                    raced = system.inspect(cid)
                    if not terminal(raced):
                        raise
                    final = raced
                    break
                next_sample = time.monotonic() + spec["timeouts"]["monitor"]
            require(shutil.disk_usage(output).free >= spec["disk_bytes"]["running"], "running disk headroom")
            time.sleep(spec["timeouts"]["poll"])
        outcome["startup"] = {"cgroup_binding_observed": bound, "terminal_before_binding": not bound}
        code = process.wait(timeout=spec["timeouts"]["retire"])
        require(code == 0 and terminal(final) and final["State"]["ExitCode"] == 0 and not final["State"]["OOMKilled"], "native container failed")
        if "verification_launcher" in spec:
            receipt = object_json(read_bytes(output / "verification-launcher-preflight.json", LIMIT))
            verified = verification_receipt(receipt, spec, expected_inner)
            require("verification_preflight" not in outcome or outcome["verification_preflight"] == verified,
                    "verification prequalification receipt changed")
            outcome["verification_preflight"] = verified
        if expected_inner['mode'] in (model_contract.MODE, model_contract.BENCHMARK_MODE,
                                     model_contract.PREPARED_TEACHER_MODE, model_contract.NUMA_PREPARED_TEACHER_MODE):
            benchmark = expected_inner['mode'] == model_contract.BENCHMARK_MODE
            contract = (model_contract.prepared_teacher_contract(
                numa=expected_inner['mode'] == model_contract.NUMA_PREPARED_TEACHER_MODE)
                if expected_inner['mode'] in (model_contract.PREPARED_TEACHER_MODE, model_contract.NUMA_PREPARED_TEACHER_MODE)
                else model_contract)
            role = 'benchmark' if benchmark else 'teacher'
            outcome['inner'] = object_json(read_bytes(output / ('owned-' + role + '-operation-result.json'), LIMIT))
            (model_contract.validate_benchmark_host_result if benchmark else contract.validate_host_result)(
                outcome['inner'], expected_inner)
            binding_record = outcome['inner']['content_binding']
            binding = object_json(bound_file({'path': str(output / binding_record['file']),
                                             **{key: binding_record[key] for key in ('bytes', 'sha256')}}))
            assets = expected_inner['operation']
            require(binding.get('schema') == 'tensorfold.owned-model-content-binding-v1' and binding.get('completed') is True
                    and [row.get('identity') for row in binding.get('assets', [])] ==
                    (model_contract.benchmark_operation_assets if benchmark else contract.operation_assets)(assets),
                    'model content binding is incomplete or different from admitted assets')
        else:
            outcome["inner"] = object_json(read_bytes(output / "named-suite-result.json", LIMIT))
            suite_result(outcome["inner"], expected_inner["mode"])
            require(outcome["inner"]["source_commit"] == expected_inner["source_commit"]
                    and outcome["inner"]["image_id_declared"] == expected_inner["image_id"]
                    and outcome["inner"]["modules"] == expected_inner["modules"], "inner result identity differs")
        admit(final, spec, cid)
    except BaseException as error:
        primary = error
    if cid is None and outcome.get("create_attempted"):
        # A command/pipe retirement error can occur after Docker accepted create.
        # Recover only our exact immutable label/name; never adopt a foreign CID.
        try:
            recovered = system.inspect(spec["name"])
            if owns_container(recovered, spec, recovered.get("Id")):
                cid, owned = recovered["Id"], True
            else:
                errors.append(RuntimeError("ambiguous create did not establish owned container"))
        except BaseException as error:
            errors.append(error)
    if cid is not None and owned:
        for action in ("stop", "kill"):
            try:
                final = system.inspect(cid)
                if terminal(final) or (final["State"]["Status"] == "created" and not final["State"]["Running"] and not final["State"]["Pid"]):
                    break
                system.command(["docker", action, *(["-t", str(math.ceil(spec["timeouts"]["retire"]))] if action == "stop" else []), cid], timeout=spec["timeouts"]["retire"] + 5)
            except BaseException as error:
                errors.append(error)
        try:
            final = system.inspect(cid)
            require(not final["State"]["Running"] and final["State"]["Pid"] == 0, "container retirement unverified")
        except BaseException as error:
            errors.append(error)
    if process is not None:
        for action in (None, "terminate", "kill"):
            if action:
                try:
                    getattr(process, action)()
                except BaseException as error:
                    errors.append(error)
            try:
                process.wait(timeout=spec["timeouts"]["retire"])
                break
            except BaseException as error:
                errors.append(error)
    for number, handler in previous.items():
        try:
            signal.signal(number, handler)
        except BaseException as error:
            errors.append(error)
    for action in ((lambda: log.flush()), (lambda: os.fsync(log.fileno())), (lambda: log.close())) if log is not None else ():
        try:
            action()
        except BaseException as error:
            errors.append(error)
    for key, action in (("source_after", lambda: system.snapshot(spec)), ("post_health", lambda: system.health(spec)), ("system_close", system.close)):
        try:
            value = action()
            if key != "system_close":
                outcome[key] = value
        except BaseException as error:
            errors.append(error)
    if control_fd is not None:
        try:
            os.close(control_fd)
        except BaseException as error:
            errors.append(error)
    try:
        source_match = before is not None and before == outcome.get("source_after")
        health_match = pre is not None and pre == outcome.get("post_health")
        if not source_match or not health_match:
            errors.append(RuntimeError("source/health verification incomplete or changed"))
        outcome.update({"cid": cid, "container": final, "source_match": source_match, "health_match": health_match,
                        "primary_error": None if primary is None else repr(primary),
                        "cleanup_errors": [repr(e) for e in errors],
                        "attach_exit": None if process is None else process.returncode, "report_complete": True})
        write_json(output / "host-result.json", outcome)
    except BaseException as error:
        errors.append(error)
        # A cold diagnostic can itself fail, including user __repr__ and memory
        # exhaustion. Keep original objects for neutral propagation; fallback
        # contains only bounded native type names and already-validated identity.
        try:
            fallback = {"schema": "tensorfold.native-contract-host-incomplete-v1", "report_complete": False,
                        "cid": cid, "primary_type": None if primary is None else _exception_type(primary),
                        "cleanup_types": [_exception_type(value) for value in errors[:8]],
                        "cleanup_count": len(errors)}
            write_json(output / "host-result-incomplete.json", fallback)
        except BaseException as persistence:
            errors.append(persistence)
        try:
            import sys
            print("NATIVE_HOST_OUTCOME_UNPUBLISHED", "primary_type=", None if primary is None else _exception_type(primary),
                  "cleanup_types=", [_exception_type(value) for value in errors[:8]],
                  "cleanup_count=", len(errors), file=sys.stderr, flush=True)
        except BaseException as emission:
            errors.append(emission)
    raise_failures(primary, errors)
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--spec-sha256")
    parser.add_argument("--spec-bytes", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--cancel-token", help="retained run_token from controller-identity.json; safe request, never numeric PID kill")
    args = parser.parse_args(argv)
    if args.cancel_token is not None:
        if args.spec is not None or args.spec_sha256 is not None or args.spec_bytes is not None or args.validate_only:
            parser.error("cancel mode takes only --out and --cancel-token")
        print(json.dumps(request_cancel(args.out, args.cancel_token)))
        return 0
    if args.spec is None or args.spec_sha256 is None or args.spec_bytes is None:
        parser.error("run/validation requires --spec/--spec-sha256/--spec-bytes")
    spec = validate_spec(object_json(bound_file({"path": str(args.spec.absolute()), "sha256": args.spec_sha256, "bytes": args.spec_bytes})))
    create_argv(spec, args.out)
    if args.validate_only:
        validate_data(spec)
        return 0
    run(spec, args.out, spec_identity={"path": str(args.spec.absolute()), "sha256": args.spec_sha256, "bytes": args.spec_bytes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

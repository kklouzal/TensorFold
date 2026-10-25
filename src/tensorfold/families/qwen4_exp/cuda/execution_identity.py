"""Opt-in exact input binding for measured SM70 execution profiles.

The supported distributed boundary is one host/container with immutable shared
checkpoint mounts. Rank zero hashes bytes once; every rank verifies identical
opened file identities before and after that pass. Paths and cached disk hashes
alone never establish model identity. Normal startup performs none of this work.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import selectors
import subprocess
import time
from datetime import timedelta

from tensorfold.cuda.tensor_file import _metadata_object, _regular_stream, checkpoint_path
from tensorfold.engine.model_identity import canonical_runtime_json, capture_model_identity
from tensorfold.cleanup import finish, raise_failures, rollback

SCHEMA = "tensorfold.execution-profile-manifest"
VERSION = 1
_METADATA = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
             "special_tokens_map.json", "added_tokens.json", "chat_template.jinja")
_SOURCE_SUFFIXES = {".py", ".c", ".cpp", ".cu", ".cuh", ".h", ".so"}
_MAX_FILES = 65536
_MAX_DOCUMENT = 4 << 20
_CHANNEL = "tf_execution_identity_v1"
_RUNTIME_ENV = ("TENSORFOLD_SM70_PROFILE", "TENSORFOLD_PREFILL_ROWS", "TENSORFOLD_MULTI_GRAPH_MIB",
                "TENSORFOLD_MTP_COPY", "TF_COMM_BACKEND", "TF_NCCL_LIB", "NCCL_ALGO", "NCCL_PROTO",
                "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE",
                "NCCL_P2P_LEVEL", "NCCL_P2P_DIRECT_DISABLE", "NCCL_IB_DISABLE", "NCCL_NET",
                "NCCL_SOCKET_IFNAME", "NCCL_NTHREADS", "NCCL_BUFFSIZE", "NCCL_MIN_CTAS", "NCCL_MAX_CTAS",
                "CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_MODULE_LOADING", "CUDA_LAUNCH_BLOCKING",
                "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY")
_TUNING_PREFIXES = ("NCCL_", "CUDA_", "CUBLAS_", "CUDNN_", "TORCH_", "PYTORCH_", "OMP_", "MKL_", "GOMP_", "KMP_",
                    "TENSORFOLD_", "TF_")
_PROCESS_LABELS = frozenset({"TENSORFOLD_TP_FAILURE_FD", "TENSORFOLD_SOURCE_REVISION"})


def digest(value):
    return hashlib.sha256(canonical_runtime_json(value, max_bytes=16 << 20).encode()).hexdigest()


def tuning_environment_identity(environment):
    """Bind backend tuning inputs without publishing possibly sensitive values.

    Prefix coverage includes allocator/JIT/collective settings beyond the human
    readable list. Process descriptors and documentary revision labels do not
    select numerical or execution policy and are not persistent identities.
    """
    return digest({name: value for name, value in environment.items()
                   if name.startswith(_TUNING_PREFIXES) and name not in _PROCESS_LABELS})


def _stat(info):
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _opened_stats(files):
    result = {}
    for name, path in sorted(files.items()):
        with _regular_stream(path) as stream:
            result[name] = {"path": str(path), "stat": _stat(os.fstat(stream.fileno()))}
    return result


def _json(path, *, with_content=False):
    # Reuse the strict checkpoint JSON reader: UTF-8, finite scalars and no
    # duplicate keys. A profile document has its own smaller cumulative bound.
    with _regular_stream(path) as stream:
        before = os.fstat(stream.fileno())
        if not 0 < before.st_size <= _MAX_DOCUMENT:
            raise ValueError("execution profile document exceeds its 4 MiB bound")
        raw = stream.read(before.st_size + 1)
        if len(raw) != before.st_size or _stat(os.fstat(stream.fileno())) != _stat(before):
            raise ValueError("execution profile document changed during its read")
    body = _metadata_object(raw, path)
    return (body, raw) if with_content else body


def artifact_files(engine):
    """Derive the actual loader's complete selected input closure."""
    from ..gguf import GGUFCheckpoint, resolve_gguf

    entry = resolve_gguf(engine.model_dir)
    if entry is None:
        raise ValueError("execution profile manifests require an actual GGUF target")
    checkpoint = GGUFCheckpoint.open(entry)
    target_root = Path(entry).resolve().parent
    result = {"target/" + p.name: checkpoint_path(target_root, p.name) for p in checkpoint.paths}
    metadata_root = Path(engine.model_dir).resolve()
    if not metadata_root.is_dir():
        raise ValueError("execution profiles require the engine's prepared metadata directory")
    for name in _METADATA:
        if (metadata_root / name).exists():
            result["metadata/" + name] = checkpoint_path(metadata_root, name)
    if not {"metadata/config.json", "metadata/tokenizer.json", "metadata/tokenizer_config.json"} <= result.keys():
        raise ValueError("execution profiles require the actual target config and tokenizer assets")
    draft = (engine.mtp_model_dir if engine.method_geometry.method == "mtp" else
             engine.dspark_model_dir if engine.method_geometry.method == "dspark" else None)
    if draft is not None:
        root = Path(draft).resolve()
        # Both selected SM70 draft providers explicitly consume these two files.
        for name in ("config.json", "model.safetensors"):
            result["draft/" + name] = checkpoint_path(root, name)
    return result, checkpoint


def begin_execution_manifest(engine, path):
    """Capture opened input identities before the loader can observe weights."""
    manifest = _json(Path(path).expanduser().resolve(strict=True))
    _manifest_schema(manifest)
    files, _ = artifact_files(engine)
    stats = _opened_stats(files)
    _expected_files(manifest, files, stats)
    engine._execution_preload = {"manifest": digest(manifest), "artifacts": stats}
    return manifest


def _manifest_schema(manifest):
    if (manifest.get("schema") != SCHEMA or type(manifest.get("version")) is not int or manifest["version"] != VERSION
            or set(manifest).difference({"schema", "version", "artifacts", "evidence", "profiles", "preparation"})):
        raise ValueError("unsupported execution profile manifest")
    if "preparation" in manifest:
        from .owner_preparation import parse_preparation

        if manifest["preparation"] is None:
            raise ValueError("an explicit preparation section must be a bounded object")
        parse_preparation(manifest["preparation"])


def _expected_files(manifest, files, stats):
    expected = manifest.get("artifacts")
    if not isinstance(expected, dict) or set(expected) != set(files):
        raise ValueError("manifest artifacts must equal the actual complete target/draft/tokenizer inventory")
    for name, record in expected.items():
        if (not isinstance(record, dict) or set(record) != {"size", "sha256"}
                or type(record["size"]) is not int or record["size"] < 0
                or not isinstance(record["sha256"], str) or len(record["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in record["sha256"])
                or record["size"] != stats[name]["stat"][2]):
            raise ValueError("manifest artifact size or SHA256 is invalid for " + name)
    return expected


def _source_files():
    root = Path(__file__).resolve().parents[3]
    result = {}
    for path in root.rglob("*"):
        if path.suffix not in _SOURCE_SUFFIXES or not path.is_file():
            continue
        real = path.resolve(strict=True)
        if not real.is_relative_to(root):
            raise ValueError("installed TensorFold source resolves outside its package")
        result["source/" + str(path.relative_to(root))] = real
        if len(result) > _MAX_FILES:
            raise ValueError("installed source inventory exceeds the supported file bound")
    if not result:
        raise ValueError("no installed TensorFold source inputs were found")
    return result


def _command(argv):
    """Bound command output and deadline; this read-only query owns its child."""
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    data, deadline = bytearray(), time.monotonic() + 15.0

    def kill_if_live():
        if child.poll() is None:
            try:
                child.kill()
            except ProcessLookupError:
                if child.poll() is None:
                    raise

    def retire():
        finish((kill_if_live, lambda: child.wait(timeout=5.0), child.stdout.close))

    try:
        with selectors.DefaultSelector() as ready:
            ready.register(child.stdout, selectors.EVENT_READ)
            while True:
                left = deadline - time.monotonic()
                if left <= 0 or not ready.select(left):
                    raise TimeoutError("execution identity topology query timed out")
                chunk = os.read(child.stdout.fileno(), 65537 - len(data))
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > 65536:
                    raise ValueError("execution identity topology query exceeded its output bound")
        if child.wait(timeout=max(.001, deadline - time.monotonic())) != 0:
            raise ValueError("execution identity topology query failed: " + data.decode("utf-8", "replace")[:512])
        output = data.decode("utf-8", "strict")
    except BaseException as error:
        rollback(child, error, retire)
    retire()
    return output


def _small(path, limit=65536):
    with _regular_stream(path) as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("runtime identity input exceeds its size bound")
    return content


def _runtime(engine, *, previous_topology=None):
    import torch

    props = torch.cuda.get_device_properties(engine.w.device)
    uuid = str(props.uuid)
    if not uuid or len(uuid) > 256:
        raise ValueError("actual GPU UUID is required for execution profile binding")
    # Torch exposes uuid.UUID on supported releases; nvidia-smi requires the
    # NVIDIA GPU- namespace rather than the bare RFC4122 string representation.
    query_uuid = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
    distributions = sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())
    if len(distributions) > 4096:
        raise ValueError("runtime dependency inventory exceeds its bound")
    from tensorfold.cuda.sm70.attention_qualification import runtime_identity

    policy = runtime_identity.__wrapped__(engine.w.device)
    # TorchVersion is a str subclass. The strict identity serializer accepts
    # native JSON values; preserve the observed text at this runtime boundary.
    policy = {key: str(value) if isinstance(value, str) else value for key, value in policy.items()}
    numerics = {"execution_dtype": engine.w.cfg.execution_dtype,
                "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                # Registration observes actual mutable backend policy. Normal
                # dispatch retains its separately qualified startup cache.
                "runtime_policy": policy, "rope": engine.rope.metadata(),
                "eos": list(engine.w.cfg.eos), "method": engine.method_geometry.receipt(),
                "proposal_confidence": engine.confidence}
    runtime = {"python": platform.python_version(), "kernel": platform.release(),
               "torch": str(torch.__version__), "torch_git": torch.version.git_version,
               "cuda": torch.version.cuda, "dependencies": [list(d) for d in distributions],
               "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
               "graphs_enabled": engine.graphs_enabled,
               "environment": {k: os.environ.get(k) for k in _RUNTIME_ENV},
               "tuning_environment_sha256": tuning_environment_identity(os.environ)}
    from tensorfold.cuda.sm70.foundation_headers import selected_identity

    runtime["source_foundation_headers"] = selected_identity()
    # Only hashes of host-specific identifiers leave this process group.
    host = hashlib.sha256(_small("/proc/sys/kernel/random/boot_id") + b"\0" + _small("/etc/machine-id")).hexdigest()
    topology = {"rank": engine.rank, "host": host, "uuid": uuid, "name": props.name,
                "memory_bytes": props.total_memory, "sms": props.multi_processor_count,
                "capability": list(torch.cuda.get_device_capability(engine.w.device)),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "configured_gpu_operation": _command([
                    "nvidia-smi", "--id=" + query_uuid,
                    "--query-gpu=uuid,power.limit,clocks.applications.graphics,clocks.applications.memory,compute_mode,ecc.mode.current",
                    "--format=csv,noheader,nounits"]),
                # Physical link/mount identity is immutable during supported
                # normal reconfiguration. Failure/replacement is a separate
                # operational event. Mutable affinity and numeric facts above
                # are observed again for every registration.
                "gpu_topology": (previous_topology["gpu_topology"] if previous_topology else
                                 _command(["nvidia-smi", "topo", "-m"])),
                "gpu_inventory": (previous_topology["gpu_inventory"] if previous_topology else
                                  _command(["nvidia-smi", "--query-gpu=uuid,pci.bus_id,name,memory.total,driver_version",
                                            "--format=csv,noheader,nounits"])),
                "nvlink_status": (previous_topology["nvlink_status"] if previous_topology else
                                  _command(["nvidia-smi", "nvlink", "--status"])),
                "cpu_models": sorted({line.split(":", 1)[1].strip() for line in
                                      _small("/proc/cpuinfo", 4 << 20).decode().splitlines()
                                      if line.startswith("model name")}),
                "host_ram": next(line for line in _small("/proc/meminfo").decode().splitlines()
                                 if line.startswith("MemTotal:"))}
    libraries = {}
    for line in _small("/proc/self/maps", 4 << 20).decode().splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and parts[5].startswith("/") and ".so" in parts[5]:
            if parts[5].endswith(" (deleted)"):
                raise ValueError("execution profile binding cannot use a deleted loaded library")
            path = Path(parts[5]).resolve(strict=True)
            libraries["library/" + str(path)] = path
    if not libraries or len(libraries) > _MAX_FILES:
        raise ValueError("actual loaded native/runtime library inventory is unavailable or excessive")
    return host, numerics, runtime, topology, libraries


def _placement(engine):
    """Actual stationary CPU owner/offset tables; no address or slot generation."""
    migration = engine.multi.controller.record().get("migration")
    if migration is not None and migration["pending"]:
        raise ValueError("execution profiles cannot bind during a pending or mixed placement transition")
    hasher = hashlib.sha256()
    for state in engine.multi.states:
        if getattr(state, "_history_migration", None) is not None:
            raise ValueError("execution profiles cannot bind a partially migrated session")
        placement = getattr(state, "history_placement", None)
        caches = list(state.kc) + ([state.mtp_kc] if hasattr(state, "mtp_kc") else [])
        for layer, cache in enumerate(caches):
            record = {"slot": getattr(state, "history_slot", 0), "layer": layer,
                      "capacity": state.capacity, "rank": engine.rank,
                      "layout": "replicated" if placement is None else "unique"}
            if placement is not None:
                record.update(degree=placement.degree, page_tokens=placement.page_tokens,
                              owners=list(cache.owners), offsets=list(cache.offsets),
                              reserved_rows=cache.reserved_rows)
            hasher.update(canonical_runtime_json(record, max_bytes=4 << 20).encode() + b"\0")
    return hasher.hexdigest()


def _draft_vocabulary(engine):
    """Hash the decoder's already validated immutable CPU global-ID map.

    TokenMap owns this snapshot before serving. Registration neither rereads
    the selection file nor transfers IDs back from a CUDA tensor. TP rank order
    in the numerical identity binds every shard's complete selected sequence.
    """
    vocabulary = engine.w.cfg.vocab
    ids = engine.multi.draft_host
    if ids is None:
        return {"mode": "full-vocabulary", "vocabulary_size": vocabulary}
    canonical = ids.astype("<i8", copy=False)
    return {"mode": "selected-global-id-map", "vocabulary_size": vocabulary,
            "selected_count": int(ids.size), "encoding": "little-endian-int64",
            "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest()}


def _capture(files, before, expected, *, reuse=None):
    total = sum(value["stat"][2] for value in before.values())
    receipt = capture_model_identity(files, runtime_identity={"schema": SCHEMA, "version": VERSION},
        authorize=lambda name, path: path.resolve(strict=True),
        max_file_bytes=max(v["stat"][2] for v in before.values()), max_total_bytes=total,
        reuse=reuse, reuse_is_immutable=reuse is not None)
    contents = {f.logical_name: {"size": f.size, "sha256": f.sha256} for f in receipt.files}
    if any(contents[name] != record for name, record in expected.items()):
        raise ValueError("execution profile checkpoint content differs from its declared SHA256")
    if _opened_stats(files) != before:
        raise ValueError("execution profile inputs changed during the content pass")
    return receipt, contents


def _capture_reusing(files, before, expected, authorities):
    """Reuse issued immutable groups; hash only newly observed native inputs."""
    contents, retained = {}, []
    for authority in authorities:
        authority.verify_unchanged()
        keep = False
        for item in authority.files:
            if item.logical_name not in files:
                continue
            if (item.path != str(files[item.logical_name])
                    or list(item.stat_identity) != before[item.logical_name]["stat"]):
                raise ValueError("immutable execution input no longer matches its issued content authority")
            contents[item.logical_name] = {"size": item.size, "sha256": item.sha256}
            keep = True
        if keep:
            retained.append(authority)
    additional = {name: path for name, path in files.items() if name not in contents}
    if sum(len(a.files) for a in retained) + len(additional) > _MAX_FILES:
        raise ValueError("retained execution input authority exceeds its cumulative file bound")
    reused, read = len(contents), 0
    if additional:
        added, new_contents = _capture(additional, {n: before[n] for n in additional},
                                      {n: expected[n] for n in additional if n in expected})
        retained.append(added)
        contents.update(new_contents)
        read = added.bytes_read
    if any(contents.get(name) != record for name, record in expected.items()) or _opened_stats(files) != before:
        raise ValueError("execution content or opened input identity changed during rebind")
    return tuple(retained), contents, read, reused


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _nonnegative(value):
    return type(value) is int and value >= 0


def _positive_time(value):
    if type(value) not in (int, float) or value <= 0:
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _measurements(role, body, verified, *, ranks):
    """Validate substantive raw measurements within the trusted packet boundary.

    The administrator supplies the independently qualified immutable packet;
    this reader verifies its bytes, scope, recorded comparisons and arithmetic.
    It cannot establish that an administrator's external experiment occurred.
    """
    measured = body.get("measurements")
    if not isinstance(measured, dict):
        raise ValueError("raw execution proof needs role-specific measurements")
    if role == "correctness":
        cases = measured.get("comparisons")
        if not isinstance(cases, list) or not cases or len(cases) > 65536:
            raise ValueError("correctness proof needs bounded independent byte comparisons")
        for case in cases:
            if (not isinstance(case, dict) or not isinstance(case.get("contract"), str) or not case["contract"]
                    or case.get("reference_sha256") not in verified
                    or case.get("candidate_sha256") != case.get("reference_sha256")):
                raise ValueError("correctness proof lacks verified matching reference/candidate bytes")
    elif role == "resources":
        rows = measured.get("ranks")
        if not isinstance(rows, list) or len(rows) != ranks:
            raise ValueError("resource proof must cover every actual rank")
        for rank, row in enumerate(rows):
            if (not isinstance(row, dict) or row.get("rank") != rank
                    or any(not _nonnegative(row.get(k)) for k in
                           ("budget_bytes", "measured_peak_bytes", "protected_future_bytes",
                            "materialized_sessions", "context_tokens"))
                    or row["budget_bytes"] <= 0 or row["measured_peak_bytes"] <= 0
                    or row["measured_peak_bytes"] + row["protected_future_bytes"] > row["budget_bytes"]
                    or row["materialized_sessions"] != body["identity"]["resident_sessions"]
                    or row["context_tokens"] != body["identity"]["context_tokens"]):
                raise ValueError("resource proof does not preserve every full-context commitment")
    elif role == "performance":
        pairs = measured.get("pairs")
        analysis = measured.get("analysis")
        if not isinstance(pairs, list) or len(pairs) < 2 or len(pairs) > 65536 or not isinstance(analysis, dict):
            raise ValueError("performance proof needs repeated comparison pairs and declared uncertainty")
        objective = analysis.get("objective")
        if (objective not in ("committed_decode_tokens_per_second", "prompt_tokens_per_second")
                or not _positive_time(analysis.get("lower_gain"))
                or not _positive_time(analysis.get("upper_gain"))
                or analysis["upper_gain"] < analysis["lower_gain"]
                or not _positive_time(analysis.get("confidence")) or analysis["confidence"] >= 1
                or analysis.get("protocol_source_sha256") not in verified):
            raise ValueError("performance proof lacks a positive bounded same-region gain analysis")
        numerator = "committed_tokens" if objective.startswith("committed") else "prompt_tokens"
        rates = {"reference": [], "candidate": []}
        for pair in pairs:
            if not isinstance(pair, dict) or set(pair) != {"reference", "candidate"}:
                raise ValueError("performance comparison pairs are incomplete")
            for side, sample in pair.items():
                if (not isinstance(sample, dict) or not _positive_time(sample.get("elapsed_seconds"))
                        or any(not _nonnegative(sample.get(k)) for k in
                               ("prompt_tokens", "committed_tokens", "offered", "completed", "rejected", "timeouts"))
                        or sample["completed"] != sample["offered"] or sample["rejected"] or sample["timeouts"]
                        or sample[numerator] <= 0 or not _positive_time(sample.get("ttft_seconds"))
                        or not _positive_time(sample.get("tpot_seconds"))):
                    raise ValueError("performance samples lack useful work, latency or complete outcomes")
                try:
                    rate = sample[numerator] / sample["elapsed_seconds"]
                except OverflowError as error:
                    raise ValueError("performance rate exceeds its finite telemetry representation") from error
                if not math.isfinite(rate):
                    raise ValueError("performance rate must remain finite")
                rates[side].append(rate)
        if sum(rates["candidate"]) <= sum(rates["reference"]):
            raise ValueError("performance samples contradict their claimed useful-throughput gain")


def _evidence(manifest, root, *, identity=None, ranks=1):
    """Load content-addressed evidence from the administrator-owned packet."""
    result = {}
    references = manifest.get("evidence", [])
    if not isinstance(references, list) or len(references) > 768:
        raise ValueError("execution profile evidence reference bound exceeded")
    total = raw_total = raw_count = 0
    for reference in references:
        if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
            raise ValueError("evidence needs a relative path and exact SHA256")
        path = checkpoint_path(root, reference["path"])
        body, raw = _json(path, with_content=True)
        total += len(raw)
        if total > 16 << 20 or hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("execution profile evidence digest or cumulative size bound failed")
        if reference["sha256"] in result:
            raise ValueError("duplicate execution profile evidence reference")
        role = body.get("kind")
        proof = body.get("proof")
        if (role not in ("correctness", "resources", "performance") or not isinstance(proof, dict)
                or set(proof) != {"protocol", "record_sha256", "artifacts"} or not _sha(proof["record_sha256"])
                or proof["protocol"] != "tensorfold.execution-profile-" + role + "-v1"
                or not isinstance(proof["artifacts"], list) or not 1 <= len(proof["artifacts"]) <= 64
                or body.get("identity") != identity):
            raise ValueError("execution profile evidence lacks the bound role protocol and raw artifacts")
        raw_files, expected = {}, {}
        for index, artifact in enumerate(proof["artifacts"]):
            if (not isinstance(artifact, dict) or set(artifact) != {"path", "size", "sha256"}
                    or not _sha(artifact["sha256"]) or not _nonnegative(artifact["size"])
                    or not 0 < artifact["size"] <= 256 << 20):
                raise ValueError("raw execution proof artifact is invalid or excessive")
            raw_path = checkpoint_path(root, artifact["path"])
            name = str(index)
            raw_files[name] = raw_path
            expected[name] = {"size": artifact["size"], "sha256": artifact["sha256"]}
            raw_total += artifact["size"]
            raw_count += 1
        if raw_total > 1 << 30 or raw_count > 4096:
            raise ValueError("raw execution evidence exceeds cumulative byte/file bounds")
        stats = _opened_stats(raw_files)
        if any(stats[name]["stat"][2] != expected[name]["size"] for name in stats):
            raise ValueError("raw execution evidence size differs from its declaration")
        authority, contents = _capture(raw_files, stats, expected)
        verified = {record["sha256"] for record in contents.values()}
        record = next((raw_files[name] for name, value in contents.items()
                       if value["sha256"] == proof["record_sha256"]), None)
        if record is None:
            raise ValueError("execution proof record is not one of its verified raw artifacts")
        raw_body = _json(record)
        if (raw_body.get("schema") != "tensorfold.execution-profile-" + role
                or type(raw_body.get("version")) is not int or raw_body["version"] != 1
                or any(raw_body.get(k) != body.get(k) for k in ("identity", "workload", "execution"))):
            raise ValueError("execution evidence lacks verified exact-scope raw role measurements")
        _measurements(role, raw_body, verified, ranks=ranks)
        authority.verify_unchanged()
        result[reference["sha256"]] = body
    return result


_PHASES = ("preflight", "inventory", "content", "verification", "evidence", "commit")


class _ManifestRefusal(ValueError):
    """Every rank has consumed the same pre-commit refusal outcome."""


class _InputChanged(ValueError):
    """Observed execution inputs invalidate the previous measurement binding."""


def _fatal(engine, error):
    """An uncertain publication/commit cannot enter a different rank wait."""
    if getattr(engine, "_execution_manifest_uncertain", False):
        return
    engine._execution_manifest_uncertain = True
    engine._abort_comm = True
    poison = getattr(engine.multi, "poison_control", None)
    if callable(poison):
        poison(error)
    else:
        from tensorfold.cuda.cohort import notify_failure

        try:
            notify_failure()
        except BaseException as notification:
            raise_failures(error, [notification])


def _unretired(error):
    pending, seen = [error], set()
    while pending:
        item = pending.pop()
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        if len(seen) > 1024:
            return True
        namespace = BaseException.__dict__["__dict__"].__get__(item, type(item))
        if any(dict.get(namespace, name) is not None for name in (
                "_tensorfold_retained_owners", "_tensorfold_model_fd_scope", "_tensorfold_checkpoint_file_scope")):
            return True
        pending.extend((BaseException.__cause__.__get__(item, type(item)),
                        BaseException.__context__.__get__(item, type(item))))
        if BaseExceptionGroup in type.__dict__["__mro__"].__get__(type(item), type):
            pending.extend(BaseExceptionGroup.exceptions.__get__(item, type(item)))
    return False


def _phase(engine, channel, stage, operation, *, may_refuse=True):
    """Publish a rank-owned outcome before any dependent rank wait.

    Pure read/validation failures are agreed refusals. Transport, serialization,
    interrupted execution and commit failures may have partially published work:
    notify the parent supervisor and never enter a retirement rendezvous.
    """
    local_error = None
    try:
        value = operation()
        outcome = {"ok": True, "value": value}
    except (ValueError, OSError) as error:
        try:
            unsafe = not may_refuse or _unretired(error)
            if not unsafe:
                name = type.__dict__["__name__"].__get__(type(error), type)
                diagnostic = name[:80] + ": " + str(error)[:1024]
        except BaseException as inspection:
            try:
                _fatal(engine, error)
            except BaseException as notification:
                if notification is not error:
                    raise_failures(error, [inspection, notification])
            raise_failures(error, [inspection])
        if unsafe:
            _fatal(engine, error)
            raise
        local_error = error
        outcome = {"ok": False, "error": diagnostic,
                   "invalidates_binding": isinstance(error, _InputChanged)}
    except BaseException as error:
        _fatal(engine, error)
        raise
    if engine.comm is None:
        outcomes = [outcome]
    else:
        try:
            store = engine.comm.store
            store.set(f"{channel}/phase/{stage}/{engine.rank}", canonical_runtime_json(outcome, max_bytes=16 << 20))
            keys = [f"{channel}/phase/{stage}/{rank}" for rank in range(engine.tp)]
            store.wait(keys, timedelta(seconds=3600))
            outcomes = [json.loads(store.get(key)) for key in keys]
            if any(not isinstance(row, dict) or type(row.get("ok")) is not bool or
                   set(row) != ({"ok", "value"} if row["ok"] else {"ok", "error", "invalidates_binding"})
                   or not row["ok"] and type(row["invalidates_binding"]) is not bool for row in outcomes):
                raise RuntimeError("execution manifest phase outcome is malformed")
        except BaseException as error:
            _fatal(engine, error)
            raise
    failed = next(((rank, row["error"]) for rank, row in enumerate(outcomes) if not row["ok"]), None)
    if failed is not None:
        if any(row.get("invalidates_binding", False) for row in outcomes):
            engine.multi.controller.invalidate_binding("observed execution inputs changed during manifest preflight")
        label = "content verification" if stage == "content" else stage
        refusal = _ManifestRefusal(f"execution profile {label} refused on rank {failed[0]}: {failed[1]}")
        raise refusal from local_error
    return [row["value"] for row in outcomes]


def _rendezvous(engine, channel, stage):
    store = engine.comm.store
    store.set(f"{channel}/ready/{stage}/{engine.rank}", "1")
    store.wait([f"{channel}/ready/{stage}/{rank}" for rank in range(engine.tp)],
               timedelta(seconds=3600))  # Same startup/cohort wait contract as NCCL.ready.


def _retire_store(engine, channel, *, stages=None):
    """All consumers acknowledge before the coordinator removes owned keys."""
    store = engine.comm.store
    _rendezvous(engine, channel, "retirement")
    store.set(f"{channel}/retired/{engine.rank}", "1")
    if engine.rank == 0:
        store.wait([f"{channel}/retired/{rank}" for rank in range(engine.tp)], timedelta(seconds=60))
        keys = [f"{channel}/phase/{stage}/{rank}" for stage in (_PHASES if stages is None else stages)
                for rank in range(engine.tp)]
        keys += [f"{channel}/ready/retirement/{rank}" for rank in range(engine.tp)]
        keys += [f"{channel}/retired/{rank}" for rank in range(engine.tp)]
        finish((lambda key=key: store.delete_key(key) for key in keys))


def bind_execution_manifest(engine, manifest_path, *, rebind=False, manifest_root=None):
    """One bounded registration operation, with no persistent store inventory."""
    comm = engine.comm
    if comm is not None and not hasattr(comm, "store"):
        error = RuntimeError("execution profile binding requires the startup-owned same-host rank store")
        _fatal(engine, error)
        raise error
    if getattr(engine, "_execution_manifest_uncertain", False):
        raise RuntimeError("execution manifest cohort is poisoned; further binding is contained")
    poison = _CHANNEL + "/retirement-failure"
    if comm is not None:
        try:
            if comm.store.check([poison]):
                raise RuntimeError("execution manifest cohort retirement failed; further binding is contained")
        except BaseException as error:
            _fatal(engine, error)
            raise
    generation = getattr(engine, "_execution_manifest_generation", 0) + 1
    engine._execution_manifest_generation = generation
    channel = _CHANNEL + "/" + str(generation)

    def retire():
        if comm is None:
            return
        try:
            _retire_store(engine, channel)
        except BaseException as error:
            # One fixed diagnostic key bounds failed-cohort retention. Future
            # operations refuse before publishing a fresh generation.
            try:
                comm.store.set(poison, type(error).__name__ + ": " + str(error)[:1024])
            except BaseException as failure:
                _fatal(engine, error)
                raise_failures(error, [failure])
            _fatal(engine, error)
            raise

    # The reservation defers HTTP migration publication until all ranks have
    # committed or refused and consumed/retired their operation outcomes.
    try:
        with engine.multi.controller.profile_transaction():
            try:
                result = _bind_execution_manifest(engine, manifest_path, rebind=rebind, channel=channel,
                                                  manifest_root=manifest_root)
            except _ManifestRefusal as error:
                if getattr(engine, "_execution_manifest_uncertain", False):
                    raise
                rollback(engine, error, retire)
            retire()
            return result
    except _ManifestRefusal:
        raise
    except BaseException as error:
        _fatal(engine, error)
        raise


def _prepare_manifest(engine, manifest_path, *, rebind, manifest_root):
    path = (checkpoint_path(manifest_root, manifest_path) if manifest_root is not None else
            Path(manifest_path).expanduser().resolve(strict=True))
    manifest = _json(path)
    _manifest_schema(manifest)
    from .owner_preparation import parse_preparation

    actual_preparation = getattr(engine, "_owner_preparation", None)
    expected_preparation = None if actual_preparation is None else actual_preparation.specification
    if parse_preparation(manifest.get("preparation")) != expected_preparation:
        raise ValueError("native preparation is fixed at startup; profile rebind cannot replace its resource envelope")
    if (engine.execution_provider != "sm70" or engine.multi is None or engine.multi.controller is None
            or engine.indexer_history != "gpu"):
        raise ValueError("execution profiles require the resident SM70 GPU-history controller")
    artifacts, checkpoint = artifact_files(engine)
    artifact_stats = _opened_stats(artifacts)
    expected = _expected_files(manifest, artifacts, artifact_stats)
    preload = getattr(engine, "_execution_preload", None)
    if preload is None or artifact_stats != preload["artifacts"]:
        raise _InputChanged("checkpoint inputs changed across loading and execution profile binding")
    if not rebind and digest(manifest) != preload["manifest"]:
        raise ValueError("execution profile manifest changed across model loading")
    placement = _placement(engine)
    from tensorfold.token_metadata import tokenizer_override

    override = tokenizer_override()
    tokenization = {"override_sha256": None if override is None else override.sha256,
                    "override_sections": [] if override is None else list(override.differs),
                    "padding": False, "truncation": False,
                    "templates": None if engine.chat_template is None else [
                        engine.chat_template.provenance(tools=False), engine.chat_template.provenance(tools=True)]}
    migration = engine.multi.controller.record().get("migration")
    migration_epoch = None if migration is None else migration["request_epoch"]
    if rebind:
        previous = engine._execution_bind_local
        host, numerics, runtime, topology, libraries = _runtime(engine, previous_topology=previous["topology"])
        source = _source_files()
        files = {**artifacts, **source, **libraries}
        before = _opened_stats(files)
        if (any(before.get(name) != record for name, record in previous["files"].items()
                if not name.startswith("library/")) or expected != engine._execution_profile_artifacts):
            raise _InputChanged("immutable execution inputs changed before profile rebind")
    else:
        source = _source_files()
        host, numerics, runtime, topology, libraries = _runtime(engine)
        files = {**artifacts, **source, **libraries}
        before = _opened_stats(files)
    numerics = {**numerics, "draft_vocabulary": _draft_vocabulary(engine)}
    runtime = {**runtime, "decode_share": engine.multi.share}
    local = {"host": host, "manifest": digest(manifest), "artifacts": artifact_stats,
             "placement": placement,
             "tokenization": tokenization,
             "files": before, "numerics": numerics, "runtime": runtime, "topology": topology}
    local["mutable_facts_changed"] = bool(rebind and any(local[name] != previous[name] for name in
        ("host", "files", "numerics", "runtime", "topology", "placement", "tokenization")))
    return {"path": path, "manifest": manifest, "artifacts": artifacts, "checkpoint": checkpoint,
            "artifact_stats": artifact_stats, "expected": expected, "placement": placement,
            "migration_epoch": migration_epoch, "files": files, "before": before, "local": local}


def _profile_identity(engine, context, ranks, contents):
    checkpoint = context["checkpoint"]
    def group(prefix):
        return {k: v for k, v in contents.items() if k.startswith(prefix)}

    from .prompt_plan import CUTS, POINTS
    from tensorfold.cuda.markers import MIN_GAP

    retention = {"history_layout": engine.w.meta.get("history_layout", "replicated"),
                 "attention_degree": engine.w.meta.get("attention_degree", 1),
                 "indexer_history": engine.indexer_history,
                 "full_windows": engine.multi.full_windows, "keep": engine.multi.keep,
                 "recovery_points_enabled": engine.points is not None,
                 "recovery_points_per_piece": POINTS, "interior_points": CUTS, "minimum_gap": MIN_GAP,
                 "execution_envelope": engine.capacity_plan["execution_envelope"]}
    preparation = getattr(engine, "_owner_preparation", None)
    if preparation is not None:
        retention["owner_preparation"] = preparation.identity()
    identity = {
        "target": digest({"files": group("target/"), "config": contents["metadata/config.json"]}),
        "draft": digest({"files": group("draft/")}) if group("draft/") else "none",
        "tokenizer": digest({"files": group("metadata/"), "ranks": [row["tokenization"] for row in ranks]}),
        "codec": digest({"pair": engine.kv_pair.identity}),
        "numerics": digest({"ranks": [row["numerics"] for row in ranks]}),
        "source": digest({"files": group("source/")}),
        "kernel_build": digest({"libraries": group("library/")}),
        "runtime": digest({"ranks": [row["runtime"] for row in ranks]}),
        "topology": digest({"ranks": [row["topology"] for row in ranks]}),
        "placement": digest({"ranks": [row["placement"] for row in ranks]}),
        "retention": digest(retention),
        "target_weight_encoding": digest({"tensors": [[name, tensor.ggml_type, list(tensor.shape), tensor.nbytes]
                                                       for name, tensor in sorted(checkpoint.tensors.items())],
                                          "execution_dtype": engine.w.cfg.execution_dtype}),
        "context_tokens": engine.context_window, "resident_sessions": engine.streams,
        "history_layout": retention["history_layout"], "method": engine.method_geometry.method}
    return identity


def _bind_execution_manifest(engine, manifest_path, *, rebind, channel, manifest_root):
    """Matched read/validation phases followed by an all-rank commit vote."""
    context = {}

    def preflight():
        context.update(_prepare_manifest(engine, manifest_path, rebind=rebind, manifest_root=manifest_root))
        return context["local"]

    ranks = _phase(engine, channel, "preflight", preflight)
    local = context["local"]

    def inventory():
        if any(row["host"] != local["host"] or row["manifest"] != local["manifest"]
               or row["artifacts"] != context["artifact_stats"] for row in ranks):
            raise ValueError("execution profiles require one host and identical immutable shared checkpoint files on every rank")
        if (len(ranks) != engine.tp or any(row["topology"]["rank"] != rank for rank, row in enumerate(ranks))
                or len({row["topology"]["uuid"] for row in ranks}) != engine.tp):
            raise ValueError("execution profiles require the actual distinct GPU UUID to logical-rank mapping")
        if rebind and any(row["mutable_facts_changed"] for row in ranks):
            engine.multi.controller.invalidate_binding("observed execution inputs or placement changed")
        union = {}
        for row in ranks:
            for name, record in row["files"].items():
                if name in union and union[name] != record:
                    raise ValueError("rank source/runtime inputs do not share identical opened file identities")
                union[name] = record
        context["union"] = union

    _phase(engine, channel, "inventory", inventory)
    context.update(authority=None, authorities=())

    def content():
        if engine.rank != 0:
            return None
        union, expected = context["union"], context["expected"]
        capture_files = {name: Path(value["path"]) for name, value in union.items()}
        if rebind:
            prior = engine._execution_input_authorities
            if not prior or prior[0] is not engine._execution_input_authority:
                raise ValueError("current-process execution input authority was replaced")
            authorities, contents, read, reused = _capture_reusing(capture_files, union, expected, prior)
            authority = authorities[0]
        else:
            authority, contents = _capture(capture_files, union, expected)
            authorities, read, reused = (authority,), authority.bytes_read, 0
        context.update(authority=authority, authorities=authorities)
        return {"contents": contents, "bytes_read": read, "hashes_reused": reused}

    outcome = _phase(engine, channel, "content", content)[0]

    def verification():
        if _opened_stats(context["files"]) != context["before"]:
            raise ValueError("execution profile inputs changed before rank publication")

    _phase(engine, channel, "verification", verification)

    def evidence():
        from .adaptive_controller import AdaptiveController

        actual = _profile_identity(engine, context, ranks, outcome["contents"])
        manifest = context["manifest"]
        proofs = _evidence(manifest, context["path"].parent, identity=actual, ranks=engine.tp)
        table = manifest.get("profiles")
        if table is None:
            table = {"schema": "tensorfold.execution-profiles", "version": 1, "identity": actual, "profiles": []}
        maximum = engine.multi.controller.maximum
        staged = AdaptiveController(slots=engine.streams, row_budget=maximum["row_budget"],
                                    proposal_limit=maximum["proposal_limit"], prefill_tokens=maximum["prefill_tokens"])
        staged.bind_identity(actual, evidence=proofs)
        staged.install_profiles(table)
        context.update(identity=actual, evidence=proofs, table=table)
        return digest({"identity": actual, "table": table, "evidence": sorted(proofs)})

    prepared = _phase(engine, channel, "evidence", evidence)
    if any(value != prepared[0] for value in prepared[1:]):
        raise _ManifestRefusal("execution profile evidence refused: ranks prepared different exact profile replacements")

    def commit():
        # All fallible packet and table checks completed before this phase.
        # profile_transaction keeps administrative migration publication deferred
        # until every rank has committed/refused and consumed its outcome.
        engine.multi.controller.replace_verified_profiles(context["identity"], evidence=context["evidence"],
            table=context["table"], expected_migration_epoch=context["migration_epoch"])
        engine._execution_input_authority = context["authority"]
        engine._execution_input_authorities = context["authorities"]
        engine._execution_profile_identity = context["identity"]
        if not rebind:
            engine._execution_profile_root = context["path"].parent
        engine._execution_profile_artifacts = dict(context["expected"])
        engine._execution_bind_local = local
        return {"schema": SCHEMA, "version": VERSION, "identity": context["identity"],
                "manifest_sha256": local["manifest"], "verified_input_files": len(context["artifacts"]),
                "verified_input_bytes": sum(v["stat"][2] for v in context["artifact_stats"].values()),
                "content_passes": int(not rebind), "shared_immutable_same_host": True,
                "bytes_read_this_binding": outcome["bytes_read"], "hashes_reused_this_binding": outcome["hashes_reused"],
                "evidence_receipts": len(context["evidence"]),
                "profiles_installed": context["manifest"].get("profiles") is not None}

    return _phase(engine, channel, "commit", commit, may_refuse=False)[0]

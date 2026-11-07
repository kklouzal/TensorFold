"""Bound the physical PP1 rank assignment before initializing a GPU communicator.

These captured identities establish basic eligibility, not a qualified GPU mesh,
imported allocation, numerical profile or performance result.
"""

import hashlib
from types import SimpleNamespace
from uuid import UUID

from tensorfold.cleanup import rollback
from tensorfold.tensor_file import _regular_stream


def canonical_gpu_uuid(value):
    """Accept only stdlib UUIDs and bounded NVIDIA/RFC4122 record spellings."""
    if isinstance(value, UUID):
        return str(value)
    if type(value) is not str or len(value) > 64:
        raise ValueError("a bounded actual GPU UUID is required")
    if value.startswith("GPU-"):
        value = value[4:]
    try:
        return str(UUID(value))
    except (ValueError, AttributeError) as error:
        raise ValueError("an actual GPU UUID is required") from error


def host_identity():
    """Fingerprint the Linux host boot, independently of container machine-id.

    Supported Docker rank containers share the host kernel's boot ID. A runtime
    that virtualizes this input needs its own binding qualification; hostnames
    and container filesystem identities cannot substitute for it.
    """
    with _regular_stream("/proc/sys/kernel/random/boot_id") as source:
        raw = source.read(129)
    if len(raw) > 128:
        raise ValueError("host boot identity exceeds its bound")
    try:
        boot = UUID(raw.decode("ascii").strip())
    except (ValueError, UnicodeError) as error:
        raise ValueError("a valid Linux host boot identity is required") from error
    return hashlib.sha256(boot.bytes).hexdigest()


def capture_rank(rank, torch):
    # All supported rank processes expose their assigned device as device zero;
    # UUID, rather than that process-local ordinal, identifies the physical GPU.
    props = torch.cuda.get_device_properties(0)
    device_uuid = getattr(props, "uuid", None)
    # Only Torch's exact trusted binding may use its native string formatter.
    # Untrusted rank records still pass through the strict scalar validator.
    if type(device_uuid) is getattr(getattr(torch, "_C", None), "_CUuuid", None):
        device_uuid = str(device_uuid)
    return {"schema_version": 1, "rank": rank, "host": host_identity(),
            "uuid": canonical_gpu_uuid(device_uuid),
            "name": getattr(props, "name", None),
            "capability": [getattr(props, "major", None), getattr(props, "minor", None)],
            "memory_bytes": getattr(props, "total_memory", None)}


def validate_rank_mapping(records, world):
    """Validate ordered ownership; return canonical device IDs for exact comparison."""
    if type(world) is not int or world < 1 or not isinstance(records, (list, tuple)) or len(records) != world:
        raise ValueError("rank mapping must cover the complete cohort")
    hosts, devices = [], []
    for rank, record in enumerate(records):
        if (not isinstance(record, dict) or type(record.get("rank")) is not int or record["rank"] != rank
                or type(record.get("host")) is not str or not 1 <= len(record["host"]) <= 256):
            raise ValueError("rank mapping requires ordered ranks and bounded host identities")
        hosts.append(record["host"])
        devices.append(canonical_gpu_uuid(record.get("uuid")))
    if len(set(hosts)) != 1:
        raise ValueError("rank mapping requires one physical host")
    if len(set(devices)) != world:
        raise ValueError("rank mapping requires distinct physical GPU UUIDs")
    return tuple(devices)


def validate_pp1(records):
    devices = validate_rank_mapping(records, 8)
    for record in records:
        if (set(record) != {"schema_version", "rank", "host", "uuid", "name", "capability", "memory_bytes"}
                or type(record["schema_version"]) is not int or record["schema_version"] != 1
                or len(record["host"]) != 64 or any(c not in "0123456789abcdef" for c in record["host"])):
            raise ValueError("PP1 requires versioned stage facts and a canonical host fingerprint")
        name, capability, memory = record.get("name"), record.get("capability"), record.get("memory_bytes")
        if (type(name) is not str or not 1 <= len(name) <= 256
                or not {"V100", "SXM2", "32GB"}.issubset(name.upper().replace("-", " ").split())
                or not isinstance(capability, (list, tuple)) or len(capability) != 2
                or any(type(value) is not int for value in capability) or tuple(capability) != (7, 0)
                or type(memory) is not int or not 0 < memory < 2**63):
            raise ValueError("PP1 requires eight V100 SXM2 32GB SM70 devices with actual memory observations")
    return {"schema": "tensorfold.pp1-stage-identity-v1", "server_count": 1, "world_size": 8, "tp": 8,
            "host_sha256": records[0]["host"],
            "rank_gpu_sha256": [hashlib.sha256(value.encode("ascii")).hexdigest() for value in devices],
            "qualification": "captured rank identity only; GPU mesh/model/numerical/resource/performance gates pending"}


def prepare_pp1(engine, torch, *, required=True):
    """Use the existing CPU Store and agreed refusal/retirement before NCCL.

    Frames are capped below the existing startup value allowance. The two
    phases retire before other startup phases/communicator initialization; no
    model data, new listener, communicator or persistent key registry is owned.
    All SM70 TP8 ranks enter this envelope, including legacy admission. Mixed
    participation must refuse here rather than splitting CPU and NCCL waits.
    """
    from .execution_identity import _ManifestRefusal, _fatal, _phase, _retire_store, _unretired
    from .retention_resources import StoreFieldGuard

    cohort = SimpleNamespace(rank=engine.rank, tp=engine.tp, multi=None,
                             comm=SimpleNamespace(store=engine._startup_store),
                             store_fields=StoreFieldGuard(8192))
    channel, stages = "tensorfold/flashnext/pp1-stage-v1", ("capture", "validation")
    result = {}

    def capture():
        if type(required) is not bool:
            raise ValueError("PP1 admission participation must be an exact boolean")
        return {"rank": engine.rank, "required": required,
                "facts": capture_rank(engine.rank, torch) if required else None}

    def validate(packets):
        if (len(packets) != 8 or any(not isinstance(packet, dict)
                or set(packet) != {"rank", "required", "facts"}
                or type(packet["rank"]) is not int or packet["rank"] != rank
                or type(packet["required"]) is not bool or packet["required"] is not required
                for rank, packet in enumerate(packets))):
            raise ValueError("PP1 ranks selected different admission participation")
        if required:
            result["receipt"] = validate_pp1([packet["facts"] for packet in packets])
        else:
            if any(packet["facts"] is not None for packet in packets):
                raise ValueError("legacy admission cannot publish a PP1 stage identity")
            result["receipt"] = None
        return True

    try:
        try:
            packets = _phase(cohort, channel, stages[0], capture)
            _phase(cohort, channel, stages[1], lambda: validate(packets))
        except _ManifestRefusal as error:
            rollback(cohort, error, lambda: _retire_store(cohort, channel, stages=stages))
        _retire_store(cohort, channel, stages=stages)
        return result["receipt"]
    except BaseException as error:
        if (not isinstance(error, _ManifestRefusal) or getattr(cohort, "_execution_manifest_uncertain", False)
                or _unretired(error)):
            engine._abort_comm = True
            _fatal(cohort, error)
        raise

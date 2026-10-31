"""One-host NUMA admission for lossless raw-indexer RAM archives.

Resource counters are observations, not OS reservations. Full archive pages are
materialized and verified before startup is published. Existing global host and
cgroup admission remains separate; node budgets exclude dirty/unevictable data.
"""

import uuid

from tensorfold.cuda import capacity, numa_memory


POLICY = "lossless-fp16-ram-archive-v1"


def node_allowances(snapshots, reserve_bytes):
    """Clean file-LRU credit plus free pages, retaining the existing host reserve."""
    if type(reserve_bytes) is not int or reserve_bytes < 0 or not snapshots:
        raise ValueError("NUMA admission requires node snapshots and nonnegative host reserve bytes")
    nodes = sorted(snapshots)
    result = {}
    for offset, node in enumerate(nodes):
        snapshot = snapshots[node]
        counters = snapshot["counters"]
        names = ("MemTotal", "MemFree", "FilePages", "Active(file)", "Inactive(file)", "Dirty", "Writeback")
        if (type(node) is not int or node < 0 or snapshot.get("node") != node
                or any(type(counters.get(name)) is not int or counters[name] < 0 for name in names)):
            raise ValueError("NUMA memory snapshot has invalid byte counters or node identity")
        total = counters["MemTotal"]
        if not total or any(counters[name] > total for name in names[1:]):
            raise ValueError("NUMA byte counter exceeds physical node memory")
        clean = max(0, min(counters["FilePages"], counters["Active(file)"] + counters["Inactive(file)"])
                    - counters["Dirty"] - counters["Writeback"])
        # File LRU excludes anonymous/shmem and unevictable/mlocked pages.
        # Keep the configured reserve exactly once, distributed over used nodes.
        share = reserve_bytes // len(nodes) + int(offset < reserve_bytes % len(nodes))
        result[node] = max(0, min(total, counters["MemFree"] + clean) - share)
    return result


def _words(identity):
    value = uuid.UUID(str(identity).removeprefix("GPU-")).int
    return [(value >> shift) & 0xFFFFFFFF for shift in (96, 64, 32, 0)]


def _same_host_words():
    from pathlib import Path

    return _words(Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def prepare(plan, *, limit, maximum_owners, torch, device, rank, world, gather):
    """Agree on placement/grants before weights or any complete archive allocates.

    The caller already agreed on GPU-vs-RAM collective scheduling. Every local
    failure becomes a fixed-size status packet before any rank raises.
    """
    target = error = None
    host_bytes = maximum_owners * plan.host_bytes(limit)
    pinned_bytes = maximum_owners * plan.pinned_bytes
    packet = [0] * 15
    try:
        target = numa_memory.configure_gpu_numa(device, torch_module=torch)
        target.probe()
        if len(target.sockets) != 1 or target.page_size != plan.page_size:
            raise ValueError("RAM archives need an unambiguous CPU socket and matching OS page geometry")
        packet = [1, target.node, target.sockets[0], target.page_size, host_bytes, pinned_bytes,
                  *_words(target.gpu_uuid), *_same_host_words(), rank]
    except Exception as failure:
        error = failure
    rows = gather(packet) if world > 1 else [packet]
    failed = [index for index, row in enumerate(rows) if row[0] != 1]
    if failed:
        if error is not None:
            raise error
        raise RuntimeError(f"RAM indexer NUMA discovery/probe failed on ranks {failed}")
    if len(rows) != world or any(len(row) != len(packet) or row[-1] != index for index, row in enumerate(rows)):
        raise RuntimeError("invalid ordered NUMA placement packet")
    if any(row[10:14] != rows[0][10:14] for row in rows[1:]):
        raise ValueError("RAM indexer NUMA admission requires all tensor-parallel ranks on one Linux host")
    if len({tuple(row[6:10]) for row in rows}) != world:
        raise ValueError("RAM indexer ranks must own distinct physical GPU UUIDs")
    nodes = sorted({row[1] for row in rows})
    node_packet = [0] * (2 + len(nodes))
    node_error = None
    try:
        memory = capacity._meminfo()
        if memory is None:
            raise ValueError("RAM indexer admission needs observable host memory/reserve")
        reserve = capacity.host_stream_reserve_bytes(memory["MemTotal"])
        snapshots = {node: numa_memory.node_memory_snapshot(node) for node in nodes}
        allowances = node_allowances(snapshots, reserve)
        node_packet = [1, reserve, *(allowances[node] for node in nodes)]
    except Exception as failure:
        node_error = failure
    observations = gather(node_packet) if world > 1 else [node_packet]
    if any(row[0] != 1 for row in observations):
        if node_error is not None:
            raise node_error
        raise RuntimeError("a peer could not establish its NUMA memory allowance")
    allowances = {node: min(row[2 + index] for row in observations) for index, node in enumerate(nodes)}
    grants = [{"rank": index, "node": row[1], "bytes": row[4]} for index, row in enumerate(rows)]
    verified = numa_memory.verify_node_grants(grants, allowances)
    allocator = numa_memory.HostAllocator(target, byte_limit=host_bytes, pinned_limit=pinned_bytes)
    receipt = {"policy": POLICY, "placement": target.receipt, "maximum_owners": maximum_owners,
               "archive_limit": limit, "host_grant_bytes": host_bytes, "pinned_grant_bytes": pinned_bytes,
               "ranks": [{"rank": index, "node": row[1], "socket": row[2], "page_size": row[3],
                          "host_grant_bytes": row[4], "pinned_grant_bytes": row[5],
                          "gpu_uuid_words": row[6:10]} for index, row in enumerate(rows)],
               "node_grants": verified, "observed_node_allowances": allowances,
               "host_reserves_observed": [row[1] for row in observations], "snapshot_is_reservation": False}
    return allocator, receipt

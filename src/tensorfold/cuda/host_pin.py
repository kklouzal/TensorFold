"""Startup-only admission for optional read-only n-gram page locks.

GPU room never authorizes host allocation by itself. This conservative snapshot
keeps anonymous/kernel/pinned usage charged and credits only clean filesystem
LRU pages before charging the complete selected payload. It is not a reservation
against other processes: the existing lock syscall may still refuse, and its
table owner rolls back refused pins. Prefetch remains complete and evictable.
"""

from __future__ import annotations

import os
from mmap import PAGESIZE
from pathlib import Path

from .capacity import GIB, reserve_bytes


def _number(value: str, label: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"invalid {label}")
    return int(value)


def _rows(path: Path) -> dict[str, int]:
    result = {}
    for row in path.read_text().splitlines():
        fields = row.split()
        if len(fields) != 2 or fields[0] in result:
            raise ValueError(f"invalid memory counter at {path}")
        result[fields[0]] = _number(fields[1], str(path))
    return result


def _host(path: Path) -> dict | None:
    try:
        rows = path.read_text().splitlines()
    except FileNotFoundError:
        return None
    memory = {}
    for row in rows:
        fields = row.split()
        if not fields:
            continue
        if fields[0] in ("MemTotal:", "MemAvailable:"):
            if len(fields) != 3 or fields[2] != "kB" or fields[0] in memory:
                raise ValueError("invalid host memory counter")
            memory[fields[0]] = _number(fields[1], "host memory counter") * 1024
    if set(memory) != {"MemTotal:", "MemAvailable:"}:
        return None
    total, available = memory["MemTotal:"], memory["MemAvailable:"]
    if not total or available > total:
        raise ValueError("invalid available host memory")
    # Use the already established host-stream reserve policy, not a second GPU
    # reserve. MemAvailable estimates reclaim without swapping or counting
    # unevictable pages as available.
    reserve = (reserve_bytes(total, host=True)
               if os.environ.get("TENSORFOLD_MEMORY_RESERVE_GIB", "").strip() else 2 * GIB)
    return dict(total_bytes=total, available_bytes=available, reserve_bytes=reserve,
                room_bytes=max(0, available - reserve))


def _cgroups(membership: Path, root: Path, payload_bytes: int) -> list[dict]:
    try:
        paths = [row[3:] for row in membership.read_text().splitlines() if row.startswith("0::")]
    except FileNotFoundError:
        return []
    if not paths:
        return []
    if len(paths) != 1:
        raise ValueError("invalid cgroup-v2 process membership")
    relative = Path(paths[0])
    if not relative.is_absolute() or ".." in relative.parts:
        raise ValueError("invalid cgroup-v2 process membership")
    current = root.joinpath(*relative.parts[1:])
    leaf = current
    # A bounded direct-child check is enough for a childless current cgroup.
    # Never inspect/credit protected sibling descendants or scan the tree.
    entries = 0
    childless = True
    try:
        with os.scandir(current) as directory:
            for entry in directory:
                entries += 1
                if entries > 4096 or entry.is_dir(follow_symlinks=False):
                    childless = False
                    break
    except FileNotFoundError:
        return [dict(path=str(current), limit_bytes=None, room_bytes=0,
                     counter_capability=False, unprotected_childless_current_cgroup=False,
                     protection_observations=[], missing_current_cgroup=True)]
    unprotected = childless
    protected = []
    leaf_clean = None
    observed = []
    while True:
        exposed = False
        try:
            limit = (current / "memory.max").read_text().strip()
        except FileNotFoundError:
            limit = "max"                       # no exposed memory controller
        else:
            exposed = True
            for name in ("memory.min", "memory.low"):
                try:
                    protection = _number((current / name).read_text().strip(), "cgroup-v2 protection")
                except FileNotFoundError:
                    unprotected = False          # unknown independently versioned capability
                    protected.append(dict(path=str(current), counter=name, value=None))
                else:
                    if protection:
                        unprotected = False
                        protected.append(dict(path=str(current), counter=name, value=protection))
        if limit != "max" or (current == leaf and exposed):
            maximum = None if limit == "max" else _number(limit, "cgroup-v2 memory limit")
            before = _number((current / "memory.current").read_text().strip(), "cgroup-v2 usage")
            counters = _rows(current / "memory.stat")
            after = _number((current / "memory.current").read_text().strip(), "cgroup-v2 usage")
            required = ("anon", "kernel", "file", "active_file", "inactive_file", "file_dirty", "file_writeback")
            if not set(required) <= counters.keys():
                # An independently versioned valid schema may lack a counter.
                # It provides no proof for optional pins; do not invent a
                # kernel-version floor or fail unrelated model startup.
                observed.append(dict(path=str(current), limit_bytes=maximum,
                                     current_bytes_before=before, current_bytes_after=after,
                                     clean_file_credit_bytes=0, unreclaimable_bytes=max(before, after),
                                     room_bytes=0, counter_capability=False,
                                     unprotected_childless_current_cgroup=False,
                                     protection_observations=protected,
                                     missing_counters=sorted(set(required) - counters.keys()), counters=counters))
                if current == root:
                    break
                current = current.parent
                continue
            # LRU file lists exclude unevictable and swap-backed shmem pages.
            # Dirty/writeback may overlap: subtracting both is conservative.
            # Anonymous expert copies are never a reclaim credit. Stats are
            # sampled, not atomic; use both current readings and the distinct
            # anon+kernel floor rather than treating every file byte as free.
            usage = max(before, after)
            clean = max(0, min(counters["file"], counters["active_file"] + counters["inactive_file"])
                        - counters["file_dirty"] - counters["file_writeback"])
            if current == leaf:
                leaf_clean = clean
            # Never credit more than the selected payload conversion. This
            # does not spend unrelated/sibling file-LRU reclaim as if it were
            # guaranteed (a sibling can have memory.min protection).
            credit = min(usage, clean, leaf_clean or 0, payload_bytes)
            unreclaimable = max(usage - credit, counters["anon"] + counters["kernel"])
            observed.append(dict(path=str(current), limit_bytes=maximum,
                                 current_bytes_before=before, current_bytes_after=after,
                                 clean_file_candidate_bytes=clean, clean_file_credit_bytes=credit,
                                 unreclaimable_bytes=unreclaimable,
                                 room_bytes=None if maximum is None else max(0, maximum - unreclaimable),
                                 counter_capability=True, counters=counters))
        if current == root:
            break
        current = current.parent
    for group in observed:
        group["unprotected_childless_current_cgroup"] = unprotected
        group["protection_observations"] = protected
        if not unprotected:
            # Without this bounded reclaim proof, aggregate file
            # counters cannot authorize optional pinning. Refuse it;
            # do not double count or claim exact charge ownership.
            group["room_bytes"] = 0
    return observed


def admit(payload_bytes: int, *, device_room_bytes: int, future_bytes: int = 0) -> dict:
    """Require existing device admission plus actual host and cgroup room.

    ``payload_bytes`` counts all selected, initially unlocked table pages;
    ``future_bytes`` is unallocated shared-memory service storage (zero for a
    discrete GPU). Already allocated expert/scratch storage stays in observed
    usage and must not be added again. Host-stream reserve retains headroom for
    ordinary runtime/IO allocations. Unknown host memory refuses optional pins;
    an unavailable counter capability refuses optional pins. Malformed/exposed
    unreadable resource data fails startup rather than hiding
    broken admission. A False table.lock() remains the ordinary exhaustion path.
    """
    if any(type(value) is not int or value < 0 for value in (payload_bytes, future_bytes)):
        raise ValueError("pin payload and future storage require nonnegative integer bytes")
    if type(device_room_bytes) is not int:
        raise ValueError("device pin room requires integer bytes")
    if payload_bytes == 0:
        return dict(policy="observed-host-and-cgroup-ngram-pin-v1", payload_bytes=0,
                    future_bytes=future_bytes, needed_bytes=future_bytes, device_room_bytes=device_room_bytes,
                    host=None, cgroups=[], device_allows=False, host_allows=False,
                    cgroups_allow=False, admitted=False, snapshot_is_reservation=False,
                    exhaustion_policy="no selected pages; resource observation and pinning skipped")
    host = _host(Path("/proc/meminfo"))
    cgroups = _cgroups(Path("/proc/self/cgroup"), Path("/sys/fs/cgroup"), payload_bytes)
    needed = payload_bytes + future_bytes
    device_ok = device_room_bytes >= payload_bytes
    host_ok = host is not None and host["room_bytes"] >= needed
    for group in cgroups:
        # The same ordinary host headroom must fit each independent hard
        # limit; it is not another allocation or a duplicate GPU reserve.
        group["reserve_bytes"] = 0 if host is None else host["reserve_bytes"]
        group["pin_room_bytes"] = (None if group["room_bytes"] is None
                                   else max(0, group["room_bytes"] - group["reserve_bytes"]))
    cgroup_ok = all(group["counter_capability"] and group["unprotected_childless_current_cgroup"]
                    and (group["pin_room_bytes"] is None or group["pin_room_bytes"] >= needed)
                    for group in cgroups)
    return dict(policy="observed-host-and-cgroup-ngram-pin-v1", payload_bytes=payload_bytes,
                future_bytes=future_bytes, needed_bytes=needed, device_room_bytes=device_room_bytes,
                host=host, cgroups=cgroups, device_allows=device_ok, host_allows=host_ok,
                cgroups_allow=cgroup_ok, admitted=device_ok and host_ok and cgroup_ok,
                snapshot_is_reservation=False,
                exhaustion_policy="leave fully prefetched mappings evictable; table lock refusal rolls back owned pins")


def table_bytes(tables) -> int:
    """Conservative complete base-page footprint of the actual lock arrays.

    Constructors exclusively own these initially unpinned arrays. Shared tables
    are deduplicated by the caller; adjacent array edge pages may be counted
    twice, which can only refuse an optional pin. No storage address escapes.
    """
    page = PAGESIZE
    if type(page) is not int or page <= 0:
        raise ValueError("host pin admission requires a positive OS page size")
    total = 0
    for table in tables:
        # BF16/FP8 tables lock their values; affine and EXL3 tables lock their
        # words/scales/biases. These are the existing table.lock() contracts.
        arrays = table.values if hasattr(table, "values") else table.words + table.scales + table.biases
        for array in arrays:
            size, address = int(array.nbytes), int(array.ctypes.data)
            if size < 0 or address < 0:
                raise ValueError("invalid n-gram pin array extent")
            if size:
                total += ((address % page + size + page - 1) // page) * page
    return total

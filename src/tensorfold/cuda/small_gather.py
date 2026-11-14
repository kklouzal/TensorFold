"""Opt-in low-latency all-gather for small per-rank payloads (CUDA graph capturable).

``TF_SMALL_GATHER_BYTES`` (per-rank payload bytes, a multiple of 8; unset or 0
disables) selects it inside ``NCCL.all_gather`` for payloads up to that size;
larger payloads, zero-size calls, or a cohort whose topology/peer mapping is
unsupported keep NCCL. Each rank owns ``4 * world * capacity`` bytes of mailbox.

Ranks are separate processes with one visible GPU each. Every rank publishes
its PCI bus ID and NVML NVLink neighbours through the communicator's Store;
all ranks then compute the same routes: NVLink neighbours are written
directly, every other pair is relayed through common NVLink neighbours that
balance link load (``routes``). On the V100 hybrid cube mesh every
non-neighbour pair has two common neighbours. Measured on 8x V100-SXM2, direct
PCIe/SYS peer stores were slower at every size (4.1 vs 3.7 us at 8 B, 41 vs
4.6 us at 4 KiB), so a cohort without such a two-hop NVLink path keeps NCCL.
Peer memory is mapped with CUDA IPC and all ranks agree on success before any
rank selects the path.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
from functools import lru_cache
from pathlib import Path

ENV = "TF_SMALL_GATHER_BYTES"


def configured_capacity() -> int:
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return 0
    value = int(raw)
    if value < 0 or value % 8 or value > 1 << 30:
        raise ValueError(f"{ENV} must be 0 or a multiple of 8 bytes up to 1 GiB")
    return value


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build

    source = Path(__file__).with_name("small_gather.cu")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    return build.load("tf_small_gather_" + digest, [str(source)], need=(7, 0), extra_cuda_cflags=["-O3"],
                      extra_cflags=["-O3"])


def _bus(text: str) -> tuple[int, int, int, int]:
    """Normalize CUDA ("0000:1A:00.0") and NVML ("00000000:1A:00.0") spellings."""
    domain, bus, rest = text.strip().lower().split(":")
    device, function = rest.split(".")
    return int(domain, 16), int(bus, 16), int(device, 16), int(function, 16)


class _PciInfo(ctypes.Structure):
    _fields_ = [("busIdLegacy", ctypes.c_char * 16), ("domain", ctypes.c_uint), ("bus", ctypes.c_uint),
                ("device", ctypes.c_uint), ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint),
                ("busId", ctypes.c_char * 32)]


def nvlink_neighbours(bus_id: str) -> list[str]:
    """Remote PCI bus IDs of this GPU's active NVLinks (NVML sees GPUs outside CUDA_VISIBLE_DEVICES)."""
    nvml = ctypes.CDLL("libnvidia-ml.so.1")
    if nvml.nvmlInit_v2() != 0:
        raise RuntimeError("NVML initialization failed")
    try:
        handle = ctypes.c_void_p()
        if nvml.nvmlDeviceGetHandleByPciBusId_v2(bus_id.encode(), ctypes.byref(handle)) != 0:
            raise RuntimeError(f"NVML has no device at {bus_id}")
        remotes = []
        for link in range(18):  # NVML_NVLINK_MAX_LINKS
            active = ctypes.c_int()
            if nvml.nvmlDeviceGetNvLinkState(handle, link, ctypes.byref(active)) != 0 or active.value != 1:
                continue
            info = _PciInfo()
            if nvml.nvmlDeviceGetNvLinkRemotePciInfo_v2(handle, link, ctypes.byref(info)) != 0:
                raise RuntimeError("NVML NVLink remote query failed")
            remotes.append(info.busId.decode())
        return remotes
    finally:
        nvml.nvmlShutdown()


def routes(width: list[dict[int, int]], world: int, split: int):
    """Identical on every rank: (direct masks[source], relay masks[relay][residue][source]) or None.

    ``width[a][b]`` counts NVLinks a->b. A pair without a link is relayed by a
    common neighbour, which already holds the source's line; lines of each pair
    are dealt by ``index % split``. Relays are chosen by deterministic coordinate
    descent minimizing the busiest link's lines per NVLink (then the sum of squares),
    since a link-bound gather finishes with its busiest link.
    """
    direct = [0] * world
    forward = [[[0] * world for _ in range(split)] for _ in range(world)]
    load = {(a, b): split for a in range(world) for b in width[a]}  # own lines to each neighbour
    pairs = []
    for source in range(world):
        for destination in range(world):
            if source == destination:
                continue
            if destination in width[source]:
                direct[source] |= 1 << destination
                continue
            common = sorted(set(width[source]) & {m for m in range(world) if destination in width[m]})
            if not common:
                return None
            pairs.append((source, destination, common))
    choice = []
    for source, destination, common in pairs:
        picks = [common[k % len(common)] for k in range(split)]
        choice.append(picks)
        for relay in picks:
            load[relay, destination] += 1

    def cost():
        ratios = sorted((value / width[a][b] for (a, b), value in load.items()), reverse=True)
        return ratios[0], sum(r * r for r in ratios)

    best = cost()
    for _ in range(64):
        improved = False
        for (source, destination, common), picks in zip(pairs, choice):
            for k in range(split):
                for relay in common:
                    old = picks[k]
                    if relay == old:
                        continue
                    load[old, destination] -= 1
                    load[relay, destination] += 1
                    trial = cost()
                    if trial < best:
                        best, picks[k], improved = trial, relay, True
                    else:
                        load[relay, destination] -= 1
                        load[old, destination] += 1
        if not improved:
            break
    for (source, destination, _), picks in zip(pairs, choice):
        for k, relay in enumerate(picks):
            forward[relay][k][source] |= 1 << destination
    return direct, forward


def open_small_gather(store, key: str, rank: int, world: int, capacity: int):
    """Collective over the Store: every rank returns a connected native SmallGather, or every rank None.

    The owner calls ``gather(send_ptr, recv_ptr, bytes, stream)`` for 0 < bytes <=
    capacity, ``abort()`` before an aborting device fence and ``release()`` after
    its device fence.
    """
    prefix = f"{key}/small_gather/"
    store.set(prefix + f"capacity/{rank}", str(capacity))
    if any(store.get(prefix + f"capacity/{r}").decode() != str(capacity) for r in range(world)):
        raise RuntimeError(f"the ranks were started with different {ENV}")
    native = None
    try:
        native = extension().SmallGather(rank, world, capacity)
        split = extension().SmallGather.split
        bus = native.pci_bus_id()
        try:
            info = bus + "|" + ",".join(nvlink_neighbours(bus))
        except (OSError, RuntimeError, AttributeError) as error:
            info = "unavailable: " + str(error).replace("|", " ")
        store.set(prefix + f"topology/{rank}", info)
        store.set(prefix + f"handle/{rank}", native.handle())  # after the mailbox was zeroed
        rows = [store.get(prefix + f"topology/{r}").decode().split("|") for r in range(world)]
        plan = None
        if all(len(row) == 2 for row in rows):
            buses = {_bus(row[0]): r for r, row in enumerate(rows)}
            if len(buses) == world:
                width = [{} for _ in range(world)]
                for r, row in enumerate(rows):
                    for b in row[1].split(","):  # one entry per active NVLink
                        if b and _bus(b) in buses and buses[_bus(b)] != r:
                            peer = buses[_bus(b)]
                            width[r][peer] = width[r].get(peer, 0) + 1
                plan = routes(width, world, split)
        ok = plan is not None
        if ok:
            direct, forward = plan
            handles = [store.get(prefix + f"handle/{r}") for r in range(world)]
            try:
                incoming = sum(1 << r for r in range(world) if direct[r] >> rank & 1)
                native.connect(handles, direct[rank], incoming, [mask for row in forward[rank] for mask in row])
            except RuntimeError as error:
                print(f"[tensorfold] rank {rank} cannot map small all-gather peers: {error}", flush=True)
                ok = False
        store.set(prefix + f"ok/{rank}", "1" if ok else "0")
        if all(store.get(prefix + f"ok/{r}") == b"1" for r in range(world)):
            result, native = native, None
            return result
        return None
    finally:
        if native is not None:
            native.release()

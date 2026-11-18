"""Opt-in low-latency all-gather for small per-rank payloads (CUDA graph capturable).

``TF_SMALL_GATHER_BYTES`` (per-rank payload bytes, a multiple of 8 up to
``MAX_CAPACITY``; unset or 0 disables) selects it inside ``NCCL.all_gather`` for
payloads up to that size; larger payloads, zero-size calls, or a cohort whose
topology/peer mapping is unsupported keep NCCL. Each rank owns
``4 * world * capacity`` bytes of mailbox (``mailbox_bytes``), allocated with
cudaMalloc at communicator open outside the PyTorch allocator; admission
forecasts do not include it (as for NCCL's own buffers).

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
# Native kMaxCapacityBytes: at 256 KiB the measured in-graph latency equals
# NCCL's on 8x V100-SXM2 and NCCL is faster beyond (README).
MAX_CAPACITY = 256 * 1024


def configured_capacity() -> int:
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0 or value % 8 or value > MAX_CAPACITY:
        raise ValueError(f"{ENV} must be 0 or a multiple of 8 bytes up to {MAX_CAPACITY}, got {raw!r}")
    return value


BULK_ENV = "TF_SMALL_ALLREDUCE_BYTES"
# Native kMaxBulkBytes.
MAX_BULK = 8 << 20
# Unset: FP32 partials of 120 rows x 2560 per rank (SC117's 24-lane MTP4 verify), rounded down to a multiple of
# 16 x world.
DEFAULT_BULK = 120 * 2560 * 4


def configured_bulk(world: int) -> int:
    """Per-rank FP32 partial bytes the bulk row all-reduce admits (0: off). Only with the small all-gather.

    Its mailbox (``bulk_mailbox_bytes``) is allocated at communicator open, before Flash Next's capacity admission
    samples free device memory, so the admitted budget excludes it; geometry forecasts do not itemize it.
    """
    raw = os.environ.get(BULK_ENV, "").strip()
    if not raw:
        return DEFAULT_BULK // (16 * world) * 16 * world
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0 or value % (16 * world) or value > MAX_BULK:
        raise ValueError(f"{BULK_ENV} must be 0 or a multiple of {16 * world} bytes up to {MAX_BULK}, got {raw!r}")
    return value


def bulk_mailbox_bytes(world: int, bulk: int, regions: int) -> int:
    """Device bytes of one rank's bulk mailbox: two parities of phase-1, transit and phase-2 slots, plus flags."""
    slot = bulk // world
    return 2 * (world * (1 + regions) * slot + world * (slot // 2)) + 2 * world * (2 + regions) * 4 * 4


def mailbox_bytes(world: int, capacity: int) -> int:
    """Device bytes one rank allocates for one communicator (two parities x world slots of 16-byte lines)."""
    return 4 * world * capacity


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build

    source = Path(__file__).with_name("small_gather.cu")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    module = build.load("tf_small_gather_" + digest, [str(source)], need=(7, 0), extra_cuda_cflags=["-O3"],
                        extra_cflags=["-O3"])
    if module.SmallGather.max_capacity != MAX_CAPACITY or module.SmallGather.max_bulk != MAX_BULK:
        raise RuntimeError("small all-gather native capacity ceiling differs from its Python contract")
    return module


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


def reduce_routes(direct, forward, world: int, split: int, rank: int):
    """The all-reduce's scatter relays, from the gather routes (identical on every rank).

    Rank s's residue-k lines for a non-neighbour owner j travel through the relay that the gather
    assigns to s's residue-k lines for j. Returns (via[k * world + j] for this rank, ordinal[s * world + j],
    regions): ordinal numbers each source's non-neighbours, the transit region of that pair on its relays.
    """
    ordinal = [-1] * (world * world)
    regions = 0
    for source in range(world):
        far = [j for j in range(world) if j != source and not direct[source] >> j & 1]
        for index, j in enumerate(far):
            ordinal[source * world + j] = index
        regions = max(regions, len(far))
    via = [-1] * (split * world)
    for k in range(split):
        for j in range(world):
            if ordinal[rank * world + j] < 0:
                continue
            relays = [m for m in range(world) if forward[m][k][rank] >> j & 1]
            if len(relays) != 1:
                raise RuntimeError("small all-reduce relay routes are ambiguous")
            via[k * world + j] = relays[0]
    return via, ordinal, regions


def store_prefix(key: str, nonce: str) -> str:
    """Store namespace of one communicator open.

    ``nonce`` is shared by every rank of this open and differs between opens
    (the caller derives it from the NCCL unique ID), so a reopened communicator
    on the same Store and key never reads a previous open's handles or votes.
    """
    if not isinstance(nonce, str) or not nonce or "/" in nonce:
        raise ValueError("small all-gather requires a nonempty per-open nonce without '/'")
    return f"{key}/small_gather/{nonce}/"


def open_small_gather(store, key: str, rank: int, world: int, capacity: int, nonce: str, bulk: int = 0):
    """Collective over the Store: every rank returns a connected native SmallGather, or every rank None.

    The owner calls ``gather(send_ptr, recv_ptr, bytes, stream)`` for 0 < bytes <=
    capacity, ``abort()`` before an aborting device fence and ``release()`` after
    its device fence. A rank whose local setup raises first publishes failure
    markers for every key its peers still await (they then return None without
    waiting for the Store timeout) and re-raises.
    """
    prefix = store_prefix(key, nonce)
    store.set(prefix + f"capacity/{rank}", f"{capacity}/{bulk}")
    if any(store.get(prefix + f"capacity/{r}").decode() != f"{capacity}/{bulk}" for r in range(world)):
        raise RuntimeError(f"the ranks were started with different {ENV} or {BULK_ENV}")
    native = None
    pending = {"topology": "failed", "handle": b"", "bulk": b"", "ok": "0"}  # Keys peers await, with failure markers.

    def publish(name, value):
        store.set(prefix + f"{name}/{rank}", value)
        pending.pop(name, None)

    try:
        native = extension().SmallGather(rank, world, capacity)
        split = extension().SmallGather.split
        bus = native.pci_bus_id()
        try:
            info = bus + "|" + ",".join(nvlink_neighbours(bus))
        except (OSError, RuntimeError, AttributeError) as error:
            info = "unavailable: " + str(error).replace("|", " ")
        publish("topology", info)
        publish("handle", native.handle())  # after the mailbox was zeroed
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
                native.connect_reduce(*reduce_routes(direct, forward, world, split, rank))
            except RuntimeError as error:
                print(f"[tensorfold] rank {rank} cannot map small all-gather peers: {error}", flush=True)
                ok = False
        if ok and bulk:
            publish("bulk", native.open_bulk(bulk))
            bulk_handles = [store.get(prefix + f"bulk/{r}") for r in range(world)]
            try:
                native.connect_bulk(bulk_handles)
            except RuntimeError as error:
                print(f"[tensorfold] rank {rank} cannot map bulk all-reduce peers: {error}", flush=True)
                ok = False
        elif bulk:
            publish("bulk", b"")  # peers that planned routes still await this key
        publish("ok", "1" if ok else "0")
        if all(store.get(prefix + f"ok/{r}") == b"1" for r in range(world)):
            result, native = native, None
            return result
        return None
    except BaseException as error:
        # A failure marker parses as an unusable topology, an invalid handle or a
        # "0" vote, so every peer reaches its vote and returns None.
        for name, value in list(pending.items()):
            try:
                publish(name, value)
            except Exception as marker:  # noqa: BLE001 - the setup failure stays primary
                error.add_note(f"small all-gather failure marker {name!r} was not published: {marker}")
                break
        raise
    finally:
        if native is not None:
            native.release()

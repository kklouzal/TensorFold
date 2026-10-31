"""Explicit NUMA-local anonymous archives and registered small host stages.

The caller supplies admitted byte grants and drains all CPU/DMA consumers before
close. Exported Tensor views retain their native mapping even after owner close.
Only owned ranges are bound; thread memory policies and unrelated PLE mappings
are never changed. CPU copy workers must enter Placement.scope themselves.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import errno
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import sys
import threading
import uuid
import weakref

from tensorfold.cleanup import raise_failures

POLICY = "numa-local-private-anon-mbind-verified-v1"


def _ids(text):
    result = set()
    for item in text.strip().split(","):
        edges = item.split("-")
        if not 1 <= len(edges) <= 2 or any(not x.isascii() or not x.isdecimal() for x in edges):
            raise ValueError("malformed kernel CPU/node list")
        first, last = int(edges[0]), int(edges[-1])
        if first > last or last-first > 1048576:
            raise ValueError("invalid kernel CPU/node range")
        result.update(range(first, last+1))
    return result


def _gpu_uuid(value):
    return "GPU-"+str(uuid.UUID(str(value).removeprefix("GPU-")))


@dataclass(frozen=True)
class Placement:
    gpu_uuid: str
    pci_bus_id: str
    node: int
    cpus: tuple[int, ...]
    sockets: tuple[int, ...]
    page_size: int

    @property
    def receipt(self):
        return {"policy": POLICY, **asdict(self), "thread_memory_policy": "unchanged"}

    @contextmanager
    def scope(self):
        """Bind this exact OS thread, restoring its prior mask on every exit."""
        thread = threading.get_native_id()
        previous = os.sched_getaffinity(0)
        selected = set(self.cpus) & previous
        _check_allowed_node(self.node)
        if not selected:
            raise RuntimeError("GPU NUMA node has no CPUs in the caller's affinity allowance")
        os.sched_setaffinity(0, selected)
        primary = None
        try:
            yield self
            if os.sched_getaffinity(0) != selected:
                raise RuntimeError("NUMA CPU allowance changed during an owned operation")
            _check_allowed_node(self.node)
        except BaseException as error:
            primary = error
            raise
        finally:
            errors = []
            try:
                if threading.get_native_id() != thread:
                    raise RuntimeError("NUMA thread scope exited on another OS thread")
                os.sched_setaffinity(0, previous)
                if os.sched_getaffinity(0) != previous:
                    raise RuntimeError("NUMA scope could not restore its previous CPU allowance")
            except BaseException as error:
                errors.append(error)
            if errors:
                raise_failures(primary, errors)

    def probe(self):
        with self.scope():
            mapping = _native().Mapping(self.page_size, self.node)
            return _verify(mapping, self)


def _check_allowed_node(node):
    rows = Path("/proc/self/status").read_text().splitlines()
    allowed = next((line.split(":", 1)[1] for line in rows if line.startswith("Mems_allowed_list:")), None)
    if allowed is None or node not in _ids(allowed):
        raise RuntimeError("GPU NUMA node is no longer in the process memory allowance")


def discover_gpu_uuid(gpu_uuid, *, proc=Path("/proc"), sysfs=Path("/sys"), affinity=None):
    """Resolve actual CUDA identity through NVIDIA's kernel PCI inventory."""
    identity = _gpu_uuid(gpu_uuid)
    matches = []
    for file in (proc/"driver/nvidia/gpus").glob("*/information"):
        fields = dict(line.split(":", 1) for line in file.read_text().splitlines() if ":" in line)
        if _gpu_uuid(fields.get("GPU UUID", "").strip()) == identity:
            bus = fields["Bus Location"].strip().lower()
            if bus != file.parent.name.lower():
                raise ValueError("GPU UUID/PCI kernel inventory disagrees")
            matches.append(bus)
    if len(matches) != 1:
        raise RuntimeError("CUDA GPU UUID needs exactly one matching kernel PCI device")
    bus = matches[0]
    pci = sysfs/"bus/pci/devices"/bus
    node = int((pci/"numa_node").read_text())
    if node < 0 or node not in _ids((sysfs/"devices/system/node/online").read_text()):
        raise RuntimeError("GPU has no observable online host NUMA node")
    allowed = os.sched_getaffinity(0) if affinity is None else set(affinity)
    cpus = tuple(sorted(_ids((pci/"local_cpulist").read_text()) & allowed))
    status = (proc/"self/status").read_text().splitlines()
    mems = next((line.split(":", 1)[1] for line in status if line.startswith("Mems_allowed_list:")), None)
    if not cpus or mems is None or node not in _ids(mems):
        raise RuntimeError("GPU NUMA node is outside allowed CPU/memory nodes")
    sockets = tuple(sorted({int((sysfs/f"devices/system/cpu/cpu{cpu}/topology/physical_package_id").read_text())
                            for cpu in cpus}))
    return Placement(identity, bus, node, cpus, sockets, int(os.sysconf("SC_PAGE_SIZE")))


def configure_gpu_numa(device, *, torch_module=None):
    if sys.platform != "linux":
        raise RuntimeError("NUMA-local archive mode requires Linux NUMA placement APIs")
    if torch_module is None:
        import torch as torch_module
    return discover_gpu_uuid(torch_module.cuda.get_device_properties(device).uuid)


def ensure_copy_thread_local(placement):
    return placement.scope()


def backing_bytes(size, *, page_size=None):
    if type(size) is not int or size < 0:
        raise ValueError("host byte extent must be a nonnegative integer")
    page = int(os.sysconf("SC_PAGE_SIZE")) if page_size is None else page_size
    if type(page) is not int or page <= 0:
        raise ValueError("host page size must be a positive integer")
    return ((size+page-1)//page)*page+2*page if size else 0


@lru_cache(maxsize=1)
def _native():
    import torch
    from .build import load_cpu
    from .sm70.source_cache import stage_sources

    source = Path(__file__).with_suffix(".cpp")
    key = hashlib.sha256(source.read_bytes()+str(torch.__version__).encode()).hexdigest()[:20]
    staged = stage_sources([source], source_root=source.parent)
    return load_cpu("tf_host_numa_"+key, staged, with_cuda=False, extra_cflags=["-O2"], verbose=False)


def _proc_nodes(mapping, placement):
    start, stop = mapping.address, mapping.address+mapping.extent
    spans = []
    for line in Path("/proc/self/maps").read_text().splitlines():
        bounds = line.split(None, 1)[0].split("-")
        if int(bounds[0], 16) == start and int(bounds[1], 16) == stop:
            spans.append(line)
    if len(spans) != 1:
        raise RuntimeError("NUMA placement proof requires its dedicated guarded VMA")
    for line in Path("/proc/self/numa_maps").read_text().splitlines():
        fields = line.split()
        if int(fields[0], 16) == start:
            return {int(key[1:]): int(value) for token in fields[1:] if "=" in token
                    for key, value in [token.split("=", 1)] if key.startswith("N") and key[1:].isdecimal()}
    raise RuntimeError("owned NUMA mapping is absent from numa_maps")


def _verify(mapping, placement):
    method, unavailable = "move_pages-self-query", None
    try:
        nodes = mapping.nodes()
    except OSError as error:
        if error.errno not in (errno.EPERM, errno.EACCES, errno.ENOSYS):
            raise
        method, unavailable = "guarded-vma-numa_maps", error.errno
        nodes = _proc_nodes(mapping, placement)
    pages = mapping.extent//placement.page_size
    if nodes != {placement.node: pages}:
        raise RuntimeError(f"host pages are not entirely on GPU NUMA node {placement.node}: {nodes}")
    return {"method": method, "query_unavailable_errno": unavailable, "pages": pages,
            "node_pages": nodes, "extent_bytes": mapping.extent, "verified": True}


class OwnedHostBuffer:
    """Factory-owned range; close requires all DMA and CPU consumers quiescent."""

    def _initialize(self, allocator, shape, dtype, pinned, tag):
        import torch

        self.tensor = self._mapping = self._pin = None
        self._closed = False
        self._allocator = weakref.ref(allocator)
        self.receipt = {"placement": allocator.target.receipt, "shape": list(shape), "dtype": str(dtype),
                        "pinned": pinned, "tag": str(tag)[:160],
                        "mapping": "MAP_PRIVATE|MAP_ANONYMOUS", "bind_mode": 32770,
                        "bind_flags": 0, "write_first_touch": True}
        count = math.prod(shape)
        size = count*torch.empty((), dtype=dtype, device="meta").element_size()
        with allocator.target.scope():
            if size:
                self._mapping = _native().Mapping(size, allocator.target.node, allocator._budget)
                self.tensor = torch.frombuffer(self._mapping, dtype=dtype, count=count).view(shape)
                self.receipt["before_pin"] = _verify(self._mapping, allocator.target)
                if pinned:
                    from .expert_pin import _native as registration
                    native = registration()
                    pages = torch.frombuffer(self._mapping, dtype=torch.uint8)
                    self._pin = native.Registration(pages)
                    code = self._pin.pin()
                    if code:
                        raise RuntimeError(f"NUMA-local host registration failed: {native.error_text(code)}")
                    self.receipt["cuda_runtime"] = native.runtime_info()
                    self.receipt["after_pin"] = _verify(self._mapping, allocator.target)
                    if not self.tensor.is_pinned():
                        raise RuntimeError("registered NUMA stage is not visible as pinned to the loaded CUDA runtime")
            else:
                self.tensor = torch.empty(shape, dtype=dtype, device="cpu")
        self.receipt.update(payload_bytes=size, backing_bytes=backing_bytes(size, page_size=allocator.target.page_size))

    def close(self):
        if self._closed:
            return
        if self._pin is not None:
            code = self._pin.close()
            if code:
                from .expert_pin import PinCleanupError
                raise PinCleanupError(self, code)
        self._pin_charge = 0
        self._pin = self.tensor = self._mapping = None
        self._closed = True
        allocator = self._allocator() if hasattr(self, "_allocator") else None
        if allocator is not None:
            allocator._retired(self)


class HostAllocator:
    """One rank's explicit admitted budget; all partial owners stay journaled."""

    def __init__(self, target, *, byte_limit, pinned_limit):
        if type(target) is not Placement or any(type(n) is not int or n < 0 for n in (byte_limit, pinned_limit)):
            raise ValueError("host allocator needs an immutable placement and explicit nonnegative byte grants")
        self.target, self.byte_limit, self.pinned_limit = target, byte_limit, pinned_limit
        self._budget = _native().Budget(byte_limit)
        self.buffers = []
        self._lock = threading.RLock()
        self._failure = None
        self._closed = False

    def __call__(self, shape, dtype, *, pinned=False, tag=""):
        import torch

        if (not isinstance(shape, (tuple, list)) or any(type(n) is not int or n < 0 for n in shape)
                or type(pinned) is not bool):
            raise ValueError("host tensor needs nonnegative integer dimensions and an explicit pinned flag")
        size = math.prod(shape)*torch.empty((), dtype=dtype, device="meta").element_size()
        charge = backing_bytes(size, page_size=self.target.page_size)
        pin = max(0, charge-2*self.target.page_size) if pinned and size else 0
        with self._lock:
            if self._closed or self._failure is not None:
                raise RuntimeError("host allocation session is closed or failed")
            if charge > self.byte_limit-self.used or pin > self.pinned_limit-self.pinned:
                raise MemoryError("NUMA host allocation exceeds its admitted archive/staging grant")
            owner = object.__new__(OwnedHostBuffer)
            self.buffers.append(owner)  # journal before mmap, registration or publication can succeed
            owner.tensor = owner._mapping = owner._pin = None
            owner._closed = False
            owner._pin_charge = pin
            try:
                owner._initialize(self, shape, dtype, pinned, tag)
            except BaseException as error:
                self._failure = error
                raise
            return owner

    @property
    def used(self):
        return self._budget.held

    @property
    def pinned(self):
        return sum(owner._pin_charge for owner in self.buffers)

    def _retired(self, owner):
        with self._lock:
            self.buffers = [item for item in self.buffers if item is not owner]

    def allocate_archive(self, shape, dtype, **kwargs):
        return self(shape, dtype, pinned=False, **kwargs)

    def allocate_pinned(self, shape, dtype, **kwargs):
        return self(shape, dtype, pinned=True, **kwargs)

    @property
    def receipt(self):
        return {"policy": POLICY, "placement": self.target.receipt, "grant_bytes": self.byte_limit,
                "pinned_grant_bytes": self.pinned_limit, "charged_bytes": self.used,
                "pinned_extent_bytes": self.pinned, "allocations": [buffer.receipt for buffer in self.buffers
                                                                    if hasattr(buffer, "receipt")]}

    def close(self):
        with self._lock:
            self._closed = True
            errors = []
            for owner in reversed(tuple(self.buffers)):
                try:
                    owner.close()
                except BaseException as error:
                    errors.append(error)
            raise_failures(None, errors)


def node_memory_snapshot(node):
    """Raw kernel node counters; caller owns grants/reserve/reclaim policy."""
    if type(node) is not int or node < 0:
        raise ValueError("NUMA memory snapshot requires a nonnegative node")
    wanted = {"MemTotal", "MemFree", "FilePages", "Active(file)", "Inactive(file)",
              "Dirty", "Writeback", "Unevictable", "Mlocked", "Shmem"}
    counters = {}
    for line in Path(f"/sys/devices/system/node/node{node}/meminfo").read_text().splitlines():
        fields = line.split()
        if len(fields) == 5 and fields[:2] == ["Node", str(node)] and fields[2].endswith(":"):
            key = fields[2][:-1]
            if key in wanted:
                if not fields[3].isdecimal() or fields[4] != "kB" or key in counters:
                    raise ValueError("invalid NUMA memory counter")
                counters[key] = int(fields[3])*1024
    if set(counters) != wanted:
        raise RuntimeError("NUMA node memory counters cannot establish the required allowance")
    return {"node": node, "counters": counters, "snapshot_is_reservation": False}


def verify_node_grants(rows, allowances):
    """Validate a common rank packet against explicit parent-owned grants."""
    if not isinstance(allowances, dict) or any(type(k) is not int or k < 0 or type(v) is not int or v < 0
                                              for k, v in allowances.items()):
        raise ValueError("NUMA node allowances must contain nonnegative integer bytes")
    totals, ranks = {}, set()
    for row in rows:
        rank, node, size = (row[key] for key in ("rank", "node", "bytes"))
        if (any(type(n) is not int or n < 0 for n in (rank, node, size)) or rank in ranks
                or node not in allowances):
            raise ValueError("NUMA rank budget packet is malformed")
        ranks.add(rank)
        totals[node] = totals.get(node, 0)+size
    result = [{"node": node, "requested_bytes": size, "allowance_bytes": allowances[node]}
              for node, size in sorted(totals.items())]
    if any(row["requested_bytes"] > row["allowance_bytes"] for row in result):
        raise MemoryError("rank archives and stages exceed a GPU-local NUMA node allowance")
    return {"nodes": result, "admitted": True, "snapshot_is_reservation": False}

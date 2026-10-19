"""Growable fixed-address device buffers for attention caches, on CUDA virtual memory management.

An ``Arena`` reserves virtual addresses for the most rows its cache may hold once, then maps physical memory at its
end as the cache grows. Rows never move: a grow copies nothing, every view keeps its address (graphs and offset
tables stay valid), and the physical memory an arena holds is the rows it has mapped, rounded up to the driver's
allocation granularity (one map call, ~55 us on dev-01, per growth past a granule). The
memory goes back to the driver, not to the caching allocator, once the arena's last view is gone and the work queued
before that has finished: long fills cannot strand freed growth buffers in the allocator.

Contracts:
- Views are ``torch`` tensors over the arena's base (``view``); each holds the arena alive. Reads and writes stay
  below ``rows``, the mapped prefix. Mapping more never changes an existing view's bytes.
- An arena belongs to an ``owner`` token (a state and its clones). ``owned(t, owner)`` is the arena whose current
  view ``t`` is (object identity) when ``owner`` owns it: only then may it grow in place. A slice, an older view, or
  the same view under another owner (a copied cache list) is someone else's prefix and moves into an arena of its own.
- An arena's memory is used on the stream current when its last view goes (as the caching allocator assumes of a
  tensor). Its release is queued with an event on that stream (none during graph capture) and done by the next
  arena operation (``Arena()``, ``ensure``) or ``reap``, which wait for that work (a capture-time release syncs the
  device). Until then its bytes count in ``pending_bytes``: reusable, like the allocator's cached blocks.
- Driver failures raise at the operation that meets them; out of memory raises ``torch.OutOfMemoryError`` after the
  queued releases are done. A failed map undoes its partial steps; a failing undo is noted on the primary error.
- The process-wide ``_Driver`` per device (libcuda binding, granularity, queued releases, mapped total) is created on
  first use and lives until exit (the driver frees everything then); its queue and counters are lock-guarded, as a
  last view may go on any thread. The accounting (``mapped_bytes``, ``pending_bytes``) imports nothing: memory gates
  read it without torch.
"""

from __future__ import annotations

import ctypes
import threading
import weakref
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

_VMM_SUPPORTED = 102                      # CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
_PINNED, _DEVICE, _READWRITE = 1, 1, 3    # CU_MEM_ALLOCATION_TYPE_PINNED, CU_MEM_LOCATION_TYPE_DEVICE, PROT_READWRITE
_MINIMUM = 0                              # CU_MEM_ALLOC_GRANULARITY_MINIMUM
_OUT_OF_MEMORY = 2                        # CUDA_ERROR_OUT_OF_MEMORY


class _Location(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]


class _Prop(ctypes.Structure):            # CUmemAllocationProp (cuda.h, v1 layout)
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int), ("location", _Location),
                ("win32HandleMetaData", ctypes.c_void_p), ("allocFlags", _AllocFlags)]


class _Access(ctypes.Structure):          # CUmemAccessDesc
    _fields_ = [("location", _Location), ("flags", ctypes.c_int)]


class _Driver:
    """One device's libcuda virtual-memory entry points, its granularity, queued releases and mapped bytes."""

    def __init__(self, device: int):
        import torch

        torch.cuda.init()
        self.torch = torch
        lib = ctypes.CDLL("libcuda.so.1")
        u64, size, i32, flags = ctypes.c_uint64, ctypes.c_size_t, ctypes.c_int, ctypes.c_ulonglong
        sig = {"cuInit": [ctypes.c_uint], "cuDeviceGet": [ctypes.POINTER(i32), i32],
               "cuDeviceGetAttribute": [ctypes.POINTER(i32), i32, i32],
               "cuMemGetAllocationGranularity": [ctypes.POINTER(size), ctypes.POINTER(_Prop), i32],
               "cuMemAddressReserve": [ctypes.POINTER(u64), size, size, u64, flags],
               "cuMemAddressFree": [u64, size],
               "cuMemCreate": [ctypes.POINTER(u64), size, ctypes.POINTER(_Prop), flags],
               "cuMemRelease": [u64], "cuMemMap": [u64, size, size, u64, flags],
               "cuMemUnmap": [u64, size], "cuMemSetAccess": [u64, size, ctypes.POINTER(_Access), size],
               "cuGetErrorName": [i32, ctypes.POINTER(ctypes.c_char_p)]}
        self.fn = {}
        for name, args in sig.items():
            f = getattr(lib, name)
            f.argtypes, f.restype = args, i32
            self.fn[name] = f
        self.call("cuInit", 0)
        handle, supported, gran = i32(), i32(), size()
        self.call("cuDeviceGet", ctypes.byref(handle), device)
        self.call("cuDeviceGetAttribute", ctypes.byref(supported), _VMM_SUPPORTED, handle.value)
        if not supported.value:
            raise RuntimeError(f"CUDA device {device} lacks virtual memory management, which the attention caches use "
                               "(cuMemAddressReserve/cuMemCreate/cuMemMap)")
        self.prop = _Prop(type=_PINNED, requestedHandleTypes=0, location=_Location(_DEVICE, device))
        self.access = _Access(location=_Location(_DEVICE, device), flags=_READWRITE)
        self.call("cuMemGetAllocationGranularity", ctypes.byref(gran), ctypes.byref(self.prop), _MINIMUM)
        self.granularity = int(gran.value)
        self.device = device
        self.lock = threading.Lock()
        self.pending: list[tuple] = []        # (event or None, base, reserved, chunks) of arenas whose last view went
        self.mapped = 0                       # bytes mapped by live and queued arenas

    def call(self, name: str, *args) -> None:
        code = self.fn[name](*args)
        if code:
            text = ctypes.c_char_p()
            self.fn["cuGetErrorName"](code, ctypes.byref(text))
            label = text.value.decode() if text.value else str(code)
            if code == _OUT_OF_MEMORY:
                raise self.torch.OutOfMemoryError(f"{name}: {label} (attention cache arena)")
            raise RuntimeError(f"{name} failed: {label} (attention cache arena)")

    def undo(self, primary: BaseException, *steps: tuple) -> None:
        """Run cleanup calls after ``primary`` failed, noting (not raising) any that fail too."""

        for name, *args in steps:
            try:
                self.call(name, *args)
            except BaseException as cleanup:            # noqa: BLE001  (the primary failure is the one raised)
                primary.add_note(f"attention cache arena cleanup {name} also failed: {cleanup!r}")

    def retire(self, base: int, reserved: int, chunks: list) -> None:
        """An arena's last view went (its finalizer): queue the release behind the current stream's work."""

        cuda = self.torch.cuda
        event = None
        if not cuda.is_current_stream_capturing():
            event = cuda.Event()
            event.record(cuda.current_stream(self.device))
        with self.lock:
            self.pending.append((event, base, reserved, chunks))

    def pending_bytes(self) -> int:
        with self.lock:
            return sum(nbytes for item in self.pending for _, nbytes, _ in item[3])

    def reap(self, *, block: bool) -> None:
        """Release queued arenas whose work finished; ``block``: wait for it (a capture-time one syncs the device)."""

        cuda = self.torch.cuda
        with self.lock:
            keep = []
            for item in self.pending:
                event = item[0]
                ready = event.query() if event is not None else False
                if not ready:
                    if not block or (event is None and cuda.is_current_stream_capturing()):
                        keep.append(item)
                        continue
                    if event is None:
                        cuda.synchronize(self.device)
                    else:
                        event.synchronize()
                _, base, reserved, chunks = item
                for offset, nbytes, handle in reversed(chunks):
                    self.call("cuMemUnmap", base + offset, nbytes)
                    self.call("cuMemRelease", handle)
                    self.mapped -= nbytes
                chunks.clear()
                self.call("cuMemAddressFree", base, reserved)
            self.pending = keep


_DRIVERS: list[_Driver] = []
_ARENAS: "weakref.WeakValueDictionary[int, Arena]" = weakref.WeakValueDictionary()   # base address -> live arena


@lru_cache(maxsize=None)
def driver(device: int) -> _Driver:
    d = _Driver(device)
    _DRIVERS.append(d)
    return d


def mapped_bytes() -> int:
    """Device memory the arenas map, outside the caching allocator's statistics (queued releases included)."""

    return sum(d.mapped for d in _DRIVERS)


def pending_bytes() -> int:
    """The part of ``mapped_bytes`` whose arenas are gone: the next arena operation releases it before mapping."""

    return sum(d.pending_bytes() for d in _DRIVERS)


def reap() -> None:
    """Release every queued arena now, waiting for the work queued before each went."""

    for d in _DRIVERS:
        d.reap(block=True)


def owned(t: torch.Tensor, owner: object) -> "Arena | None":
    """The arena ``owner`` may grow in place through ``t``: its current view is ``t``. Else None (a plain tensor, a
    slice, a view an older grow made, or another owner's arena)."""

    arena = _ARENAS.get(t.data_ptr())
    if arena is None or arena.owner is not owner or arena._view is None or arena._view() is not t:
        return None
    return arena


class _Span:
    """The CUDA array interface of an arena's mapped prefix; a tensor made from it holds the arena alive."""

    def __init__(self, arena: "Arena", nbytes: int):
        self.arena = arena
        self.__cuda_array_interface__ = {"shape": (nbytes // 2,), "typestr": "<i2", "data": (arena.base, False),
                                         "version": 3}


class Arena:
    """A device buffer of up to ``capacity`` rows of ``row_shape`` ``dtype`` items at one fixed address."""

    def __init__(self, capacity: int, row_shape: tuple[int, ...], dtype: torch.dtype, device: torch.device,
                 owner: object = None):
        import torch

        device = torch.device(device)
        if device.type != "cuda" or type(capacity) is not int or capacity < 1:
            raise ValueError("an arena needs a CUDA device and a positive row capacity")
        index = device.index if device.index is not None else torch.cuda.current_device()
        self.drv = driver(index)
        if self.drv.pending:
            self.drv.reap(block=True)
        self.row_shape, self.dtype, self.device = tuple(row_shape), dtype, torch.device("cuda", index)
        self.owner = owner
        self.row_bytes = int(torch.Size(self.row_shape).numel()) * torch.empty((), dtype=dtype).element_size()
        if self.row_bytes % 2 or self.row_bytes == 0:
            raise ValueError("arena rows must be whole nonempty 16-bit words")
        g = self.drv.granularity
        self.capacity = capacity
        self.reserved = -(-capacity * self.row_bytes // g) * g
        base = ctypes.c_uint64()
        self.drv.call("cuMemAddressReserve", ctypes.byref(base), self.reserved, g, 0, 0)
        self.base = int(base.value)
        self.mapped = 0                                   # bytes mapped from base
        self.chunks: list[tuple[int, int, int]] = []      # (offset, bytes, handle), in map order
        self._view: weakref.ref | None = None             # the current view: its holders own in-place growth
        self._finalizer = weakref.finalize(self, self.drv.retire, self.base, self.reserved, self.chunks)
        self._finalizer.atexit = False
        _ARENAS[self.base] = self

    @property
    def rows(self) -> int:
        """Rows readable and writable through ``view``: the mapped prefix (at most ``capacity``)."""

        return min(self.capacity, self.mapped // self.row_bytes)

    def extra(self, rows: int) -> int:
        """The bytes ``ensure(rows)`` maps: none when they fit, else up to ``rows`` rounded to the granularity."""

        if rows <= self.rows:
            return 0
        if rows > self.capacity:
            raise ValueError(f"{rows} rows exceed the arena's {self.capacity}-row reservation")
        g = self.drv.granularity
        return -(-rows * self.row_bytes // g) * g - self.mapped

    def ensure(self, rows: int) -> None:
        """Map ``extra(rows)`` bytes at the end so ``rows`` rows fit; existing views keep their bytes."""

        nbytes = self.extra(rows)
        if not nbytes:
            return
        if self.drv.pending:
            self.drv.reap(block=True)
        handle = ctypes.c_uint64()
        try:
            self.drv.call("cuMemCreate", ctypes.byref(handle), nbytes, ctypes.byref(self.drv.prop), 0)
        except self.drv.torch.OutOfMemoryError:
            self.drv.reap(block=True)
            self.drv.call("cuMemCreate", ctypes.byref(handle), nbytes, ctypes.byref(self.drv.prop), 0)
        at = self.base + self.mapped
        try:
            self.drv.call("cuMemMap", at, nbytes, 0, handle.value, 0)
        except BaseException as error:
            self.drv.undo(error, ("cuMemRelease", handle.value))
            raise
        try:
            self.drv.call("cuMemSetAccess", at, nbytes, ctypes.byref(self.drv.access), 1)
        except BaseException as error:
            self.drv.undo(error, ("cuMemUnmap", at, nbytes), ("cuMemRelease", handle.value))
            raise
        self.chunks.append((self.mapped, nbytes, int(handle.value)))
        self.mapped += nbytes
        with self.drv.lock:
            self.drv.mapped += nbytes

    def view(self) -> torch.Tensor:
        """A tensor of the mapped rows, ``(rows, *row_shape)``, over the arena's base; it becomes the current view."""

        rows = self.rows
        flat = self.drv.torch.as_tensor(_Span(self, rows * self.row_bytes), device=self.device)
        out = flat.view(self.dtype).view(rows, *self.row_shape)
        self._view = weakref.ref(out)
        return out

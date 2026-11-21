"""DeepSeek-V4.1 Engram tables in host RAM, read by the GPU through mapped (zero-copy) host memory.

Every rank maps the IQ4_NL rows of the llama.cpp Engram file (one ``per_layer_token_embd`` tensor, the two per-layer
tables back to back) read-only straight from the file: the table is the file's own page-cache pages, one copy in host
RAM for all ranks, with no copy step (an earlier sealed-memfd copy cost a 103 GiB fill on rank 0 and a second
103 GiB of RAM beside the file's cached pages). Each rank faults its page tables in with parallel threads, then
registers the range with cudaHostRegisterReadOnly (without it the driver refuses a PROT_READ range: CUDA error 1,
m1c; a 110 GB shared read-write registration did not finish in 10 min, diag/hostreg2 2026-11-21). The registration
pins the pages (resident, never evicted or swapped while serving). The file is the trusted model input, as for the
weights; it must not be rewritten in place while a server runs (a rename or delete leaves the mapped pages intact).
Every rank registers its mapping with CUDA as mapped host memory, so
a step's gather reads only the rows it needs (48 x 144 B per token) straight from RAM inside the forward: no host
thread, no synchronization, and graph-capturable.

Binding rule (the user, 2026-11-21): the Engram table lives in system RAM only, never in VRAM, for any model: no
full or per-layer copy, no GPU row cache/LRU/hot-row residency, no prefetch into a persistent device table. The only
Engram data on a GPU is the caller's per-step working buffer (``lookup``'s ``out``: the current tokens' rows, decoded
and consumed in the same forward). ``HostEngram`` verifies at startup that its table address is host memory to the
CUDA driver (``device_bytes`` is then 0). This replaces 1Cat-vLLM's file reader (SharedEngramReader,
csrc/gguf_shared_engram.h) and the friend's RAM/holder patches 83/160 with one resident copy.

Decode arithmetic: llama.cpp/1Cat ``dequantize_block_iq4_nl``: value = half(float(d) * kvalues_iq4nl[q]); element
j < 16 of a 32-value block is the low nibble of byte j, element j + 16 its high nibble.
"""

from __future__ import annotations

import ctypes
import os
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from .kernels.triton_utils import jit, tl

ROW_BYTES = 144
ROW_VALUES = 256
HEADS = 24                    # hashed rows per token per table
KVALUES = (-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113)   # kvalues_iq4nl (the kernel's where-tree)
_CHUNK = 256 << 20
_REGISTER_PORTABLE, _REGISTER_MAPPED, _REGISTER_READ_ONLY = 1, 2, 8


class _Pointer:
    """A device-visible address for Triton (mapped host memory has no torch device tensor)."""

    def __init__(self, address: int):
        self.address, self.dtype = address, torch.uint8

    def data_ptr(self) -> int:
        return self.address


class HostEngram:
    """The two tables, shared by every rank on the host; ``rows``: rows of table 0 and table 1."""

    def __init__(self, path, offset: int, rows: tuple[int, int], rank: int, *, threads: int = 16):
        self.rows = tuple(int(r) for r in rows)
        self.size = sum(self.rows) * ROW_BYTES
        self.fd, self.address, self.registered = -1, 0, False
        self._base = self._length = 0                       # the page-aligned file mapping holding the table
        self.fill_seconds = self.register_seconds = 0.0   # startup phases
        started = time.perf_counter()
        try:
            self.fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            if os.fstat(self.fd).st_size < offset + self.size:
                raise ValueError(f"{path}: the Engram table [{offset}, +{self.size}) runs past the end of the file")
            page = os.sysconf("SC_PAGE_SIZE")
            first = offset // page * page
            self._length = offset - first + self.size
            self._base = _map(self.fd, self._length, _PROT_READ, first)
            self.address = self._base + offset - first
            # Every rank faults its page tables in parallel threads (rank 0 also reads any part of the file that is
            # not in the page cache); the registration's own single-threaded pin then finds them present.
            _populate(self._base, self._length, threads)
            self.fill_seconds = time.perf_counter() - started
            self._register(rank)
            self.pointer = _Pointer(self.address)     # UVA: the host address is valid on the device
        except BaseException:
            self.close()
            raise
        self.seconds = time.perf_counter() - started

    def _register(self, rank):
        t0 = time.perf_counter()
        try:
            self._register_now(rank)
        finally:
            self.register_seconds = time.perf_counter() - t0

    def _register_now(self, rank):
        # Every rank registers its own mapping at once. The earlier sealed-memfd table needed rank 0's registration
        # first (m1g: five of eight registrations made while rank 0's was in flight failed, "Cannot map memory ...
        # of 0x19bffe4 pages"); the file-backed table registered concurrently on all eight ranks without failure
        # (start/diag hostreg_file st-hreg2, 2 x 8 registrations: 15.8-17.9 s each vs 6.7 s + 15.8 s ordered).
        code = torch.cuda.cudart().cudaHostRegister(self._base, self._length,
                                                    _REGISTER_PORTABLE | _REGISTER_MAPPED | _REGISTER_READ_ONLY)
        if int(code) != 0:
            raise RuntimeError(f"rank {rank}: cudaHostRegister of the {self.size >> 30} GiB Engram table failed "
                               f"(CUDA error {int(code)}; the kernel log names the short page pin)")
        self.registered = True
        _require_host(self.address)

    device_bytes = 0          # Engram table bytes resident in VRAM: none, by construction (verified in __init__)

    def lookup(self, hashes: torch.Tensor, channel: int, out: torch.Tensor) -> torch.Tensor:
        """Decoded rows [T, 24 * 256] FP16 of table ``channel`` for ``hashes`` [T, 2, 24] (table-local ids;
        negative ids, i.e. padding tokens, decode as zeros)."""
        tokens = hashes.shape[0]
        if (hashes.ndim != 3 or hashes.shape[1:] != (2, HEADS) or hashes.dtype != torch.int64
                or not hashes.is_contiguous() or out.shape != (tokens, HEADS * ROW_VALUES)
                or out.dtype != torch.float16 or not out.is_contiguous() or channel not in (0, 1)):
            raise ValueError("Engram lookup needs contiguous int64 hashes [T, 2, 24] and FP16 rows [T, 6144]")
        if tokens:
            base = 0 if channel == 0 else self.rows[0]
            _gather_iq4nl[(tokens * HEADS,)](hashes, self.pointer, out, channel, base, self.rows[channel],
                                            HEADS=HEADS)
        return out

    def close(self):
        if self.registered:
            torch.cuda.cudart().cudaHostUnregister(self._base)
            self.registered = False
        if self._base:
            _unmap(self._base, self._length)
            self._base = self.address = 0
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.mmap.restype = ctypes.c_void_p
_LIBC.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
_LIBC.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_LIBC.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_PROT_READ, _MAP_SHARED, _MADV_POPULATE_READ = 1, 1, 22


_CU_POINTER_ATTRIBUTE_MEMORY_TYPE, _CU_MEMORYTYPE_HOST = 2, 1


def _require_host(address):
    """Fail unless the CUDA driver reports ``address`` as host memory (the no-VRAM Engram rule)."""
    cuda = ctypes.CDLL("libcuda.so.1")
    kind = ctypes.c_uint(0)
    code = cuda.cuPointerGetAttribute(ctypes.byref(kind), _CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                      ctypes.c_uint64(address))
    if code != 0 or kind.value != _CU_MEMORYTYPE_HOST:
        raise RuntimeError(f"the Engram table must be host memory (driver memory type {kind.value}, CUresult {code})")


def _map(fd, size, prot, offset=0):
    address = _LIBC.mmap(None, size, prot, _MAP_SHARED, fd, offset)
    if address in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_errno(), "mmap of the Engram table failed")
    return address


def _unmap(address, size):
    if _LIBC.munmap(ctypes.c_void_p(address), ctypes.c_size_t(size)) != 0:
        raise OSError(ctypes.get_errno(), "munmap of the Engram table failed")


def _populate(address, total, threads):
    """Fault in the read-only mapping's page tables (MADV_POPULATE_READ, Linux 5.14+), reading from storage any page
    not in the page cache, in ``_CHUNK`` pieces on ``threads`` threads."""
    def populate(first):
        if _LIBC.madvise(ctypes.c_void_p(address + first), ctypes.c_size_t(min(_CHUNK, total - first)),
                         _MADV_POPULATE_READ) != 0:
            raise OSError(ctypes.get_errno(), "populating the Engram table mapping failed")

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(populate, range(0, total, _CHUNK)))


@jit                    # the SM70 Triton 3.2 profile's launch options (ptx_version 83 for CUDA 12.9)
def _gather_iq4nl(hashes, table, out, CHANNEL, BASE, ROWS, HEADS: tl.constexpr):
    pid = tl.program_id(0)
    token = pid // HEADS
    head = pid % HEADS
    row = tl.load(hashes + token * (2 * HEADS) + CHANNEL * HEADS + head)
    ok = (row >= 0) & (row < ROWS)
    src = table + (BASE + tl.where(ok, row, 0)).to(tl.int64) * 144
    offs = tl.arange(0, 256)
    block = offs // 32
    within = offs % 32
    byte = tl.load(src + block * 18 + 2 + (within % 16), mask=ok, other=0).to(tl.int32)
    q = tl.where(within < 16, byte & 15, byte >> 4)
    lo = tl.load(src + block * 18, mask=ok, other=0).to(tl.uint16)
    hi = tl.load(src + block * 18 + 1, mask=ok, other=0).to(tl.uint16)
    d = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    kv = tl.where(q < 8,
                  tl.where(q < 4, tl.where(q < 2, tl.where(q == 0, -127.0, -104.0), tl.where(q == 2, -83.0, -65.0)),
                           tl.where(q < 6, tl.where(q == 4, -49.0, -35.0), tl.where(q == 6, -22.0, -10.0))),
                  tl.where(q < 12, tl.where(q < 10, tl.where(q == 8, 1.0, 13.0), tl.where(q == 10, 25.0, 38.0)),
                           tl.where(q < 14, tl.where(q == 12, 53.0, 69.0), tl.where(q == 14, 89.0, 113.0))))
    value = tl.where(ok, d * kv, 0.0).to(tl.float16)
    tl.store(out + token.to(tl.int64) * (HEADS * 256) + head * 256 + offs, value)

"""DeepSeek-V4.1 Engram tables in host RAM, read by the GPU through mapped (zero-copy) host memory.

Rank 0 copies the IQ4_NL rows of the llama.cpp Engram file (one ``per_layer_token_embd`` tensor, the two per-layer
tables back to back) into one sealed memfd; the other ranks map the same pages read-only
(``/proc/<pid>/fd/<fd>``, same host and user). The read-only mappings are registered with cudaHostRegisterReadOnly
(without it the driver refuses a PROT_READ range: CUDA error 1, m1c; a 110 GB shared read-write registration did not
finish in 10 min, diag/hostreg2 2026-11-21, while read-only took 149 s). The registration pins the pages
(resident, never swapped); rank 0 does not mlock them: an mlocked sealed range failed registration (CUDA error 1,
m1e) while the seven unlocked mappings of the same memfd registered in ~10 s each, concurrently. Every rank registers its mapping with CUDA as mapped host memory, so
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
_COPY = 256 << 20
_REGISTER_PORTABLE, _REGISTER_MAPPED, _REGISTER_READ_ONLY = 1, 2, 8
# Linux <linux/fcntl.h> sealing constants (the image's Python 3.15 fcntl module does not export them)
_F_ADD_SEALS = 1033
_F_SEAL_SEAL, _F_SEAL_SHRINK, _F_SEAL_GROW, _F_SEAL_WRITE = 1, 2, 4, 8


class _Pointer:
    """A device-visible address for Triton (mapped host memory has no torch device tensor)."""

    def __init__(self, address: int):
        self.address, self.dtype = address, torch.uint8

    def data_ptr(self) -> int:
        return self.address


class HostEngram:
    """The two tables, shared by every rank on the host; ``rows``: rows of table 0 and table 1."""

    def __init__(self, path, offset: int, rows: tuple[int, int], rank: int, store, *, threads: int = 16,
                 key: str = "ds41/engram"):
        self.rows = tuple(int(r) for r in rows)
        self.size = sum(self.rows) * ROW_BYTES
        self.fd, self.address, self.registered = -1, 0, False
        started = time.perf_counter()
        try:
            if rank == 0:
                self.fd = os.memfd_create("tensorfold-ds41-engram", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
                os.ftruncate(self.fd, self.size)
                writable = _map(self.fd, self.size, _PROT_READ | _PROT_WRITE)
                try:
                    _fill(writable, self.size, path, offset, threads)
                finally:
                    _unmap(writable, self.size)
                import fcntl
                fcntl.fcntl(self.fd, _F_ADD_SEALS, _F_SEAL_SEAL | _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_WRITE)
                self.address = _map(self.fd, self.size, _PROT_READ)
                self._register(rank)
                store.set(key, f"{os.getpid()} {self.fd} {self.size}")
            else:
                pid, fd, size = (int(v) for v in store.get(key).decode().split())
                if size != self.size:
                    raise RuntimeError(f"rank 0's Engram table has {size} bytes, expected {self.size}")
                self.fd = os.open(f"/proc/{pid}/fd/{fd}", os.O_RDONLY | os.O_CLOEXEC)
                self.address = _map(self.fd, self.size, _PROT_READ)
                self._register(rank)
            self.pointer = _Pointer(self.address)     # UVA: the host address is valid on the device
        except BaseException:
            self.close()
            raise
        self.seconds = time.perf_counter() - started

    def _register(self, rank):
        # Rank 0 registers before it publishes the table; the followers register after it, concurrently. m1g: five
        # of eight registrations made while rank 0's own was in flight failed (kernel "Cannot map memory ... of
        # 0x19bffe4 pages": the driver's long-term pin came back short); m1e's seven follower registrations after
        # rank 0 all passed.
        code = torch.cuda.cudart().cudaHostRegister(self.address, self.size,
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
            torch.cuda.cudart().cudaHostUnregister(self.address)
            self.registered = False
        if self.address:
            _unmap(self.address, self.size)
            self.address = 0
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.mmap.restype = ctypes.c_void_p
_LIBC.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
_LIBC.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_PROT_READ, _PROT_WRITE, _MAP_SHARED = 1, 2, 1


_CU_POINTER_ATTRIBUTE_MEMORY_TYPE, _CU_MEMORYTYPE_HOST = 2, 1


def _require_host(address):
    """Fail unless the CUDA driver reports ``address`` as host memory (the no-VRAM Engram rule)."""
    cuda = ctypes.CDLL("libcuda.so.1")
    kind = ctypes.c_uint(0)
    code = cuda.cuPointerGetAttribute(ctypes.byref(kind), _CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                      ctypes.c_uint64(address))
    if code != 0 or kind.value != _CU_MEMORYTYPE_HOST:
        raise RuntimeError(f"the Engram table must be host memory (driver memory type {kind.value}, CUresult {code})")


def _map(fd, size, prot):
    address = _LIBC.mmap(None, size, prot, _MAP_SHARED, fd, 0)
    if address in (None, ctypes.c_void_p(-1).value):
        raise OSError(ctypes.get_errno(), "mmap of the Engram table failed")
    return address


def _unmap(address, size):
    if _LIBC.munmap(ctypes.c_void_p(address), ctypes.c_size_t(size)) != 0:
        raise OSError(ctypes.get_errno(), "munmap of the Engram table failed")


def _fill(address, total, path, offset, threads):
    view = memoryview((ctypes.c_char * total).from_address(address)).cast("B")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        def copy(first):
            n = min(_COPY, total - first)
            done = 0
            while done < n:
                got = os.preadv(fd, [view[first + done:first + n]], offset + first + done)
                if got <= 0:
                    raise OSError(f"short read of the Engram table at byte {offset + first + done}")
                done += got

        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(copy, range(0, total, _COPY)))
    finally:
        os.close(fd)
        view.release()


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

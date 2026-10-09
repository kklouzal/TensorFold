"""NCCL all-gather on the current stream so CUDA graphs capture it; a rank-order sum after it keeps ranks bit-equal."""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import threading

import torch

from tensorfold.cleanup import rollback

_DTYPES = {torch.float32: 7, torch.bfloat16: 9, torch.int32: 2, torch.int64: 4}


class _UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


def _library() -> ctypes.CDLL:
    if os.name == "nt":
        raise RuntimeError("TensorFold does not run tensor-parallel (NCCL) on Windows: CUDA on Windows has no "
                           "libnccl to wrap; use one GPU per process there")
    candidates = [os.environ.get("TF_NCCL_LIB", "")]
    found = ctypes.util.find_library("nccl")
    if found:
        candidates.append(found)
    candidates += glob.glob("/usr/lib/*/libnccl.so.2") + glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/nccl/lib/libnccl.so.2")
    candidates += glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libnccl*.so*"))
    for path in candidates:
        if path:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
    raise RuntimeError("libnccl not found (set TF_NCCL_LIB)")


class NCCL:
    def __init__(self) -> None:
        """An empty valid owner; journal it before calling one-shot ``open``."""
        self._close_lock = threading.Lock()
        self.store = self.lib = self.device = None
        self.comm = ctypes.c_void_p()
        self.closed = self.retired = False
        self._native_release_started = self._native_retired = False
        self._release_error = None
        self._opened = False

    def open(self, rank: int, world: int, master: str, port: int) -> None:
        """Acquire into a published slot; failure closes or retains that slot.

        Native init writes directly into ``comm`` before Python regains control.
        The released GIL syscall cannot expose an unjournaled Python result.
        TCPStore's native provider owns its listener through reference lifetime.
        """
        acquiring = False
        try:
            with self._close_lock:
                if self.closed or self._opened:
                    raise RuntimeError("communicator open is one-shot")
                acquiring = True
                self._initialize(rank, world, master, port)
                self._opened = True
        except BaseException as error:
            if not acquiring:
                raise
            rollback(self, error, lambda: self.close(abort=True))

    def _initialize(self, rank: int, world: int, master: str, port: int) -> None:
        from datetime import timedelta

        if (type(world) is not int or not 1 <= world <= 2**31 - 1
                or type(rank) is not int or not 0 <= rank < world
                or type(port) is not int or not 0 <= port <= 65535 or not isinstance(master, str)):
            raise ValueError("NCCL requires integer world/rank/port ranges and a master hostname")
        from torch.distributed import TCPStore

        self.rank, self.world = rank, world
        self.lib = _library()
        lib = self.lib
        lib.ncclGetErrorString.restype = ctypes.c_char_p
        lib.ncclGetErrorString.argtypes = [ctypes.c_int]
        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(_UniqueId)]
        lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId, ctypes.c_int]
        lib.ncclAllGather.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
                                      ctypes.c_void_p]
        for function in (lib.ncclCommDestroy, lib.ncclCommAbort):
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_int
        self.store = TCPStore(master, port, world, rank == 0, timeout=timedelta(seconds=600))
        uid = _UniqueId()
        if rank == 0:
            self._check(self.lib.ncclGetUniqueId(ctypes.byref(uid)))
            self.store.set("tf_nccl_uid", bytes(uid.internal))
        else:
            raw = self.store.get("tf_nccl_uid")
            if not isinstance(raw, bytes) or len(raw) != ctypes.sizeof(_UniqueId):
                raise ValueError("NCCL rendezvous unique ID must contain exactly 128 bytes")
            ctypes.memmove(ctypes.addressof(uid), raw, 128)
        self.comm = ctypes.c_void_p()
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._check(self.lib.ncclCommInitRank(ctypes.byref(self.comm), world, uid, rank))
        if not self.comm.value:
            raise RuntimeError("NCCL initialization returned no communicator")

    def _check(self, code: int) -> None:
        if code != 0:
            raise RuntimeError(f"NCCL error {code}: {self.lib.ncclGetErrorString(code).decode()}")

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """Gather contiguous CUDA tensors on this communicator's device.

        Input production must precede the device's current stream. Both storage
        allocations remain borrowed until that stream completes; callers must
        not resize/mutate/free them meanwhile. Legal in-place input is exactly
        this rank's slice of recv. Other overlapping spans are invalid. NCCL2
        selects the communicator device internally, so another current device
        on the calling thread does not change which stream is submitted.
        """
        with self._close_lock:
            if self.closed or not self._opened:
                raise RuntimeError("all_gather: communicator admission is closed")
            self._all_gather(send, recv)

    def _all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """Submit while the caller holds this communicator's lifetime lease."""
        if not isinstance(send, torch.Tensor) or not isinstance(recv, torch.Tensor):
            raise ValueError("all_gather: input and output must be CUDA tensors")
        count, dtype = send.numel(), send.dtype
        if recv.numel() != count * self.world or dtype != recv.dtype:
            raise ValueError("all_gather: recv must hold world x send of the same dtype")
        if (dtype not in _DTYPES or send.device != self.device or recv.device != self.device
                or not send.is_contiguous() or not recv.is_contiguous()):
            raise ValueError("all_gather: supported dtype and contiguous storage on the communicator's CUDA device required")
        source, destination = send.data_ptr(), recv.data_ptr()
        size = count * send.element_size()
        if (source < destination + size * self.world and destination < source + size
                and source != destination + self.rank * size):
            raise ValueError("all_gather: overlapping input must be this rank's exact receive slice")
        stream = torch.cuda.current_stream(self.device).cuda_stream
        self._check(self.lib.ncclAllGather(source, destination, count, _DTYPES[dtype],
                                           self.comm, stream))

    def close(self, *, abort: bool = False) -> None:
        """Retire this owned communicator after its borrowers are quiescent.

        Callers stop graph replays and all borrowed store operations first.
        Ordinary close fences the device before destroy; failed startup/work
        aborts before fencing. Every active rank must participate in teardown.
        An entered destroy/abort pointer is never retried, even after an error
        or an interrupted return. Such an unresolved owner requires containment.
        TCPStore has native RAII lifetime; borrowed store references must retire
        before releasing this owner's final reference can close its listener.
        """
        if type(abort) is not bool:
            raise ValueError("typed communicator abort policy required")
        with self._close_lock:
            self.closed = True
            if self.retired:
                return
            if self._native_release_started and not self._native_retired:
                raise RuntimeError("native communicator retirement is unresolved; pointer cannot retry") from self._release_error
            fenced = False
            if self.comm.value and not self._native_retired:
                if not abort:
                    torch.cuda.synchronize(self.device)
                    fenced = True
                else:
                    # Rank zero's listener departure wakes its idle peer, which
                    # then enters the same failed-operation teardown policy.
                    self.store = None
                release = self.lib.ncclCommAbort if abort else self.lib.ncclCommDestroy
                self._native_release_started = True
                try:
                    self._check(release(self.comm))
                except BaseException as error:
                    self._release_error = error
                    raise
                self._native_retired = True
                self.comm = ctypes.c_void_p()
            if self.device is not None and not fenced:
                torch.cuda.synchronize(self.device)
            self.store = None
            self.retired = True

    def ready(self, label: str, *, every: float = 60.0, timeout: float = 3600.0) -> None:
        """Every rank finishes ``label`` before any goes on; a rank missing after ``timeout`` s is named."""
        with self._close_lock:
            if self.closed or not self._opened:
                raise RuntimeError("ready: communicator admission is closed")
            self._ready(label, every=every, timeout=timeout)

    def _ready(self, label: str, *, every: float, timeout: float) -> None:

        import time
        from datetime import timedelta
        from tensorfold.cuda.store_wait import timed_out

        self.store.set(f"tf_ready/{label}/{self.rank}", "1")
        others = [r for r in range(self.world) if r != self.rank]
        keys = [f"tf_ready/{label}/{r}" for r in others]
        interval = timedelta(seconds=every)
        started = time.monotonic()
        while True:
            try:
                self.store.wait(keys, interval)
                return
            except Exception as error:
                if not timed_out(error, keys, interval):
                    raise
            waited = time.monotonic() - started
            missing = ", ".join(str(r) for r in others)
            if waited >= timeout:
                raise RuntimeError(f"rank {self.rank} finished {label} but rank {missing} has not after "
                                   f"{waited / 60:.0f} min: check that rank's log (a CUDA extension build waiting on "
                                   "a lock names the lock there)")
            print(f"[tensorfold] rank {self.rank} finished {label}; waiting for rank {missing} ({waited:.0f} s)",
                  flush=True)

    def barrier(self) -> None:
        with self._close_lock:
            if self.closed or not self._opened:
                raise RuntimeError("barrier: communicator admission is closed")
            x = torch.zeros((1,), dtype=torch.float32, device="cuda")
            y = torch.zeros((self.world,), dtype=torch.float32, device="cuda")
            self._all_gather(x, y)
            torch.cuda.synchronize()

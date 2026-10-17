"""NCCL all-gather on the current stream so CUDA graphs capture it; a rank-order sum after it keeps ranks bit-equal."""

from __future__ import annotations

import ctypes
import ctypes.util
import glob
import os
import threading
from collections.abc import Callable
from typing import Protocol

import torch

from tensorfold.cleanup import rollback, raise_failures

_DTYPES = {torch.float32: 7, torch.bfloat16: 9, torch.int32: 2, torch.int64: 4,
           torch.uint8: 1, torch.int8: 0, torch.float16: 6}
BACKEND_ENV = "TF_COMM_BACKEND"


class Comm(Protocol):
    """What every engine assumes; ``all_gather_fast``, ``exchange``, ``check`` (and NCCL's ``store``) are optional."""

    rank: int
    world: int

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None: ...

    def barrier(self) -> None: ...

    def ready(self, label: str, *, every: float = 60.0, timeout: float = 3600.0) -> None: ...

    def close(self, *, abort: bool = False) -> None: ...


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

    def open(self, rank: int, world: int, master: str, port: int, *, timeout_s: float = 600,
             key: str = "tf_nccl_uid", store=None) -> None:
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
                self._initialize(rank, world, master, port, timeout_s=timeout_s, key=key, store=store)
                self._opened = True
        except BaseException as error:
            if not acquiring:
                raise
            rollback(self, error, lambda: self.close(abort=True))

    def _initialize(self, rank: int, world: int, master: str, port: int, *, timeout_s: float = 600,
                    key: str = "tf_nccl_uid", store=None) -> None:
        from datetime import timedelta
        from math import isfinite

        if (type(world) is not int or not 1 <= world <= 2**31 - 1
                or type(rank) is not int or not 0 <= rank < world
                or type(port) is not int or not 0 <= port <= 65535 or not isinstance(master, str)):
            raise ValueError("NCCL requires integer world/rank/port ranges and a master hostname")
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                or not isfinite(timeout_s) or timeout_s <= 0
                or not isinstance(key, str) or not key):
            raise ValueError("NCCL requires a positive finite timeout and nonempty rendezvous key")
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
        self.store = store if store is not None else TCPStore(master, port, world, rank == 0,
                                                             timeout=timedelta(seconds=timeout_s))
        uid = _UniqueId()
        if rank == 0:
            self._check(self.lib.ncclGetUniqueId(ctypes.byref(uid)))
            self.store.set(key, bytes(uid.internal))
        else:
            raw = self.store.get(key)
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

    def exchange(self, sends: list[torch.Tensor], recvs: list[torch.Tensor], peer: int) -> None:
        """Submit borrowed contiguous CUDA buffers; their storage survives current-stream completion.

        Admission and group submission share the communicator lifetime lease.
        Validate every buffer before entering the native group; closing may only
        retire it after caller graph/buffer borrowers have stopped.
        """
        with self._close_lock:
            if self.closed or not self._opened:
                raise RuntimeError("exchange: communicator admission is closed")
            if (len(sends) != len(recvs) or type(peer) is not int
                    or peer == self.rank or not 0 <= peer < self.world):
                raise ValueError("exchange: one receive per send, with another rank")
            for tensor in (*sends, *recvs):
                if (not isinstance(tensor, torch.Tensor) or tensor.dtype not in _DTYPES
                        or tensor.device != self.device or not tensor.is_contiguous()):
                    raise ValueError("exchange: supported contiguous tensors on the communicator's CUDA device required")
            reads = [(tensor.data_ptr(), tensor.numel() * tensor.element_size()) for tensor in sends]
            writes = [(tensor.data_ptr(), tensor.numel() * tensor.element_size()) for tensor in recvs]
            for index, (begin, size) in enumerate(writes):
                if any(size and extent and begin < other + extent and other < begin + size
                       for other, extent in [*reads, *writes[:index]]):
                    raise ValueError("exchange: receive storage must not overlap sends or other receives")
            lib = self.lib
            if not getattr(self, "_p2p", False):
                for name in ("ncclSend", "ncclRecv"):
                    getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                                   ctypes.c_void_p, ctypes.c_void_p]
                    getattr(lib, name).restype = ctypes.c_int
                lib.ncclGroupStart.argtypes = []
                lib.ncclGroupEnd.argtypes = []
                lib.ncclGroupStart.restype = lib.ncclGroupEnd.restype = ctypes.c_int
                self._p2p = True
            stream = torch.cuda.current_stream(self.device).cuda_stream
            self._check(lib.ncclGroupStart())
            primary = None
            try:
                for send, recv in zip(sends, recvs):
                    if send.numel():
                        self._check(lib.ncclSend(send.data_ptr(), send.numel(), _DTYPES[send.dtype], peer, self.comm, stream))
                    if recv.numel():
                        self._check(lib.ncclRecv(recv.data_ptr(), recv.numel(), _DTYPES[recv.dtype], peer, self.comm, stream))
            except BaseException as error:
                primary = error
            errors = []
            try:
                self._check(lib.ncclGroupEnd())
            except BaseException as error:
                errors.append(error)
            raise_failures(primary, errors)

    def broadcast(self, tensor: torch.Tensor, root: int = 0) -> None:
        """Borrow one contiguous tensor until current-stream completion; every rank submits the same shape/type."""
        with self._close_lock:
            self._buffer(tensor)
            if type(root) is not int or not 0 <= root < self.world:
                raise ValueError("broadcast requires an in-range root rank")
            function = self.lib.ncclBroadcast
            function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                 ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
            function.restype = ctypes.c_int
            self._check(function(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(), _DTYPES[tensor.dtype],
                                 root, self.comm, torch.cuda.current_stream(self.device).cuda_stream))

    def _buffer(self, tensor: torch.Tensor) -> None:
        if self.closed or not self._opened:
            raise RuntimeError("communicator admission is closed")
        if (not isinstance(tensor, torch.Tensor) or tensor.device != self.device
                or tensor.dtype not in _DTYPES or not tensor.is_contiguous()):
            raise ValueError("NCCL requires supported contiguous storage on the communicator device")

    def send(self, tensor: torch.Tensor, peer: int) -> None:
        self._transfer(tensor, peer, "ncclSend")

    def recv(self, tensor: torch.Tensor, peer: int) -> None:
        self._transfer(tensor, peer, "ncclRecv")

    def _transfer(self, tensor: torch.Tensor, peer: int, name: str) -> None:
        """Point-to-point borrowed storage, matched by the peer's identical type/count and submission order."""
        with self._close_lock:
            self._buffer(tensor)
            if type(peer) is not int or not 0 <= peer < self.world or peer == self.rank:
                raise ValueError("NCCL point-to-point peer must be another valid rank")
            function = getattr(self.lib, name)
            function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_void_p, ctypes.c_void_p]
            function.restype = ctypes.c_int
            self._check(function(tensor.data_ptr(), tensor.numel(), _DTYPES[tensor.dtype], peer, self.comm,
                                 torch.cuda.current_stream(self.device).cuda_stream))

    def check(self) -> None:
        """Observe deferred native errors without treating successful submission as completed communication."""
        with self._close_lock:
            if self.closed or not self._opened:
                raise RuntimeError("communicator admission is closed")
            error = ctypes.c_int()
            function = self.lib.ncclCommGetAsyncError
            function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
            function.restype = ctypes.c_int
            self._check(function(self.comm, ctypes.byref(error)))
            self._check(error.value)

    def barrier(self) -> None:
        with self._close_lock:
            if self.closed or not self._opened:
                raise RuntimeError("barrier: communicator admission is closed")
            x = torch.zeros((1,), dtype=torch.float32, device="cuda")
            y = torch.zeros((self.world,), dtype=torch.float32, device="cuda")
            self._all_gather(x, y)
            torch.cuda.synchronize()


def fast_gather(comm: Comm, send: torch.Tensor, recv: torch.Tensor) -> None:
    """A model exchange: the communicator's faster ``all_gather_fast`` if it has one, else ``all_gather``."""

    fast = getattr(comm, "all_gather_fast", None)
    if fast is None:
        comm.all_gather(send, recv)
    else:
        fast(send, recv)


def check(comm: Comm) -> None:
    """Raise a failure the communicator's transport recorded (call it after a synchronizing exchange)."""

    found = getattr(comm, "check", None)
    if found is not None:
        found()


def exchange(comm: Comm, sends: list[torch.Tensor], recvs: list[torch.Tensor], peer: int | None = None) -> None:
    """``comm.exchange`` when it has one, else (two ranks) each pair traded as bytes through the all-gather."""

    peer = 1 - comm.rank if peer is None else peer
    if len(sends) != len(recvs):
        raise ValueError("exchange: one receive per send")
    fn = getattr(comm, "exchange", None)
    if fn is not None:
        fn(sends, recvs, peer)
        return
    if comm.world != 2:
        raise ValueError("exchange: a communicator without exchange trades through its all-gather, two ranks only")
    if peer == comm.rank or not 0 <= peer < comm.world:
        raise ValueError("exchange: the peer must be another rank")
    for s, r in zip(sends, recvs):
        sb, rb = s.contiguous().view(-1).view(torch.uint8), r.view(-1).view(torch.uint8)
        n = max(sb.numel(), rb.numel())
        if n == 0:
            continue
        pad = torch.zeros((n,), dtype=torch.uint8, device=s.device)
        pad[:sb.numel()].copy_(sb)
        both = torch.empty((2 * n,), dtype=torch.uint8, device=s.device)
        comm.all_gather(pad, both)
        rb.copy_(both[peer * n:peer * n + rb.numel()])


class Transport:
    """A base for transports: whatever a subclass does not define (``store``, ``ready``, ...) is the wrapped NCCL's."""

    def __init__(self, base: Comm) -> None:
        self.base, self.rank, self.world = base, base.rank, base.world

    def __getattr__(self, name: str):
        if name == "base":                          # not set yet (a copy): no recursion
            raise AttributeError(name)
        return getattr(self.base, name)


# Process-owned transport registrations are written during single-threaded
# startup, before any communicator is opened, and stay stable while engines
# borrow their transports. A wrapper owns its extra resources and close/abort
# drains them before retiring base; bare Transport owns only base.
BACKENDS: dict[str, Callable[[Comm], Comm]] = {}


def register_backend(name: str, wrap: Callable[[Comm], Comm]) -> None:
    """Register before engine startup; wrap runs on all ranks and owns close/abort of its transport."""

    BACKENDS[name] = wrap


def open_comm(rank: int, world: int, master: str, port: int, *, backend: str | None = None, **nccl) -> Comm:
    """NCCL, wrapped by ``backend`` (default TF_COMM_BACKEND, else none); the ranks check they named the same one."""

    name = (backend if backend is not None else os.environ.get(BACKEND_ENV, "")).strip().lower() or "nccl"
    if name != "nccl" and name not in BACKENDS:
        raise ValueError(f"{BACKEND_ENV}={name!r}: expected nccl{''.join(', ' + b for b in sorted(BACKENDS))}")
    base = NCCL()
    base.open(rank, world, master, port, **nccl)     # open owns rollback of partial native acquisition
    try:
        store = getattr(base, "store", None)
        if store is not None and world > 1:          # every rank selects the same transport before its first exchange
            store.set(f"tf_comm_backend/{rank}", name)
            names = [store.get(f"tf_comm_backend/{r}").decode() for r in range(world)]
            if len(set(names)) > 1:
                raise RuntimeError(f"the ranks were started with different {BACKEND_ENV}: {', '.join(names)} (rank order)")
        return base if name == "nccl" else BACKENDS[name](base)
    except BaseException as error:
        rollback(base, error, lambda: base.close(abort=True))

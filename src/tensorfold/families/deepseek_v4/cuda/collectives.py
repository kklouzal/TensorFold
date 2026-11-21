"""Cross-rank sums and gathers of the DeepSeek-V4.1 SM70 lane over its one 8-rank communicator (TP8 attention/dense
and the ETP4 x EP2 experts).

``Collectives.sum(partial, torch.float16)`` (attention output, MoE output, decode indexer scores, embedding rows)
takes one of the Flash Next / Gemma 4 NVLink collectives of ``tensorfold.cuda.comm`` where its mailboxes admit the
shape (the communicator is opened with ``row_reduce``: ``engine.py``), selected per [rows, dims] by measured
in-graph latency (``_PATHS``; ledger section "DeepSeek V4.1 V100 lane", candidate K3):

- ``gather``: the small all-gather of the FP32 partials, then ``rank_sum`` (one Triton kernel);
- ``ll``: the low-latency small row all-reduce (``NCCL.all_reduce_rows_mode(bulk=False)``);
- ``bulk``: the bulk row all-reduce (``bulk=True``).

All three give bitwise the same FP16 values: every partial in rank order (rank 0 first, FP32 adds) rounded once to
FP16 (round to nearest even), on every rank. Shapes outside the table (prompt rows) and FP32 results keep the NCCL
FP32 all-reduce (NCCL's own sum order, the same bytes on every rank) rounded once. The selection depends only on
the shape, so every rank submits the same collective sequence (eager and inside the startup CUDA graphs).
"""

from __future__ import annotations

import torch

from .kernels.triton_utils import jit, tl, triton
from .weights import WORLD

_NCCL_SUM = 0
_NCCL_TYPES = {torch.float32: 7, torch.float16: 6}


@jit
def _rank_sum_kernel(parts, out, n, WORLD_: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    total = tl.load(parts + offsets, mask=mask)
    for r in tl.static_range(1, WORLD_):     # rank order, FP32 adds (no reassociation)
        total = total + tl.load(parts + r * n + offsets, mask=mask)
    tl.store(out + offsets, total.to(out.dtype.element_ty), mask=mask)


def rank_sum(parts: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """out[...] = round(parts[0] + parts[1] + ... + parts[W-1]) for FP32 parts [W, *out.shape] (contiguous)."""
    n = out.numel()
    block = 1024
    _rank_sum_kernel[(triton.cdiv(n, block),)](parts, out, n, WORLD_=parts.shape[0], BLOCK=block, num_warps=4)
    return out


def _table(rows: int, dims: int) -> str | None:
    """The measured path of an FP16-rounded [rows, dims] FP32 sum (None: NCCL)."""
    return _PATHS.get(dims, {}).get(rows)


# dims -> {rows: path}. Filled from the in-graph microbenchmark (8x V100-SXM2, 180 W).
_PATHS: dict[int, dict[int, str]] = {}


class Collectives:
    """FP32 sums and gathers over the 8-rank communicator (TP8 attention/dense and the ETP4 x EP2 experts)."""

    def __init__(self, world):
        self.world = world
        from tensorfold.cuda import comm
        self.small = getattr(world, "small_gather", None) is not None
        self.reduce = self.small and comm._SMALL_ALLREDUCE          # TF_SMALL_ALLREDUCE=0 keeps the gather / NCCL

    def _all_reduce(self, tensor):
        """In-place NCCL sum of an FP32 or FP16 tensor."""
        import ctypes
        comm = self.world
        fn = comm.lib.ncclAllReduce
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                       ctypes.c_void_p, ctypes.c_void_p]
        fn.restype = ctypes.c_int
        stream = torch.cuda.current_stream().cuda_stream
        comm._check(fn(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(), _NCCL_TYPES[tensor.dtype], _NCCL_SUM,
                       comm.comm, stream))

    def path(self, rows: int, dims: int) -> str:
        """The collective an FP16-rounded [rows, dims] sum takes: gather, ll, bulk or nccl. Every rank opened the same
        mailboxes (the communicator checks), so every rank selects the same path for the same shape."""
        path = _table(rows, dims) if self.small else None
        small = self.world.small_gather if path else None
        if path == "gather" and rows * dims * 4 <= self.world.small_gather_bytes:
            return path
        if path == "ll" and self.reduce and small.reduce_lines(rows, dims) > 0:
            return path
        if path == "bulk" and self.reduce and small.bulk_chunk_rows(rows, dims, False) > 0:
            return path
        return "nccl"

    def reduce_half(self, send: torch.Tensor, path: str) -> torch.Tensor:
        """Rank-ordered FP32 sum of FP32 [rows, dims] partials rounded once to FP16 through ``path`` (not nccl)."""
        rows, dims = send.shape
        recv = torch.empty(rows, dims, dtype=torch.float16, device=send.device)
        if path == "gather":
            parts = torch.empty((WORLD, rows, dims), dtype=torch.float32, device=send.device)
            self.world.all_gather(send, parts)
            return rank_sum(parts, recv)
        if not self.world.all_reduce_rows_mode(send, recv, path == "bulk"):
            raise RuntimeError(f"the {path} row all-reduce did not admit [{rows}, {dims}] (its table entry is invalid)")
        return recv

    def sum(self, partial: torch.Tensor, dtype=None) -> torch.Tensor:
        """8-rank sum of partials, accumulated in FP32 and rounded once to ``dtype`` (default: the partial's).
        FP16 results of 2-D partials in the measured table are rank-ordered (module docstring)."""
        dtype = dtype or partial.dtype
        if dtype == torch.float16 and partial.ndim == 2:
            path = self.path(*partial.shape)
            if path != "nccl":
                return self.reduce_half(partial.float().contiguous(), path)
        total = partial.float().contiguous()
        self._all_reduce(total)
        return total.to(dtype)

    def gather(self, part: torch.Tensor) -> torch.Tensor:
        """Concatenate the eight ranks' parts along the last dimension, rank order."""
        part = part.contiguous()
        recv = torch.empty((WORLD,) + tuple(part.shape), dtype=part.dtype, device=part.device)
        self.world.all_gather(part, recv)
        return torch.cat(list(recv.unbind(0)), dim=-1)

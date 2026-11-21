"""Cross-rank sums and gathers of the DeepSeek-V4.1 SM70 lane over its one 8-rank communicator (TP8 attention/dense
and the ETP4 x EP2 experts).

The communicator carries the Flash Next / Gemma 4 NVLink small all-gather mailbox (``engine.py`` sets
``TF_SMALL_GATHER_BYTES``): every all-gather up to its capacity per rank (decode logits rows, Engram projections,
control exchanges) takes it instead of NCCL (exact copies either way). ``Collectives.sum(partial, torch.float16)``
(attention output, MoE output, indexer scores) and ``exact_sum_half`` (embedding rows) select by the partial's
element count ``n`` per rank (measured in CUDA graphs, 8x V100-SXM2, 180 W; ledger candidate K3, us per call):

- ``n <= GATHER_MAX``: the small all-gather of the FP32 partials, then ``rank_sum`` (one Triton kernel)
  (1 x 5120: 11.2 vs LL 13.5 vs NCCL 28.8);
- ``n <= LL_MAX``: the low-latency small row all-reduce (``NCCL.all_reduce_rows_ll``) (2 / 4 / 8 / 11 x 5120:
  13.9 / 14.6 / 22.1 / 27.7 vs NCCL 34.4 / 32.5 / 24.1 / 30.4; 1 x 32768: 18.5 vs 24.4; 12 x 5120 29.6 vs 30.2 and
  from 13 x 5120 or 1 x 65536 NCCL is as fast or faster);
- larger (prompt rows, wide decode indexer scores): the NCCL FP32 all-reduce in place, rounded once.

The gather and LL paths give bitwise the same FP16 values: every partial in rank order (rank 0 first, FP32 adds)
rounded once to FP16 (round to nearest even), identical on every rank; the NCCL path's sum order is NCCL's own (the
same bytes on every rank). The rank-ordered bulk row all-reduce was slower than NCCL at every measured DS41 shape
(5120 x 1..96 rows, 32768 / 65536 x 1..32), so its mailbox is not opened. Selection depends only on the shape and
the communicator's configuration (identical on every rank, checked at open), so every rank submits the same
collective sequence, eager and inside the startup CUDA graphs.
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


GATHER_MAX = 5120          # elements per rank (one hidden row)
LL_MAX = 11 * 5120         # elements per rank


class Collectives:
    """FP32 sums and gathers over the 8-rank communicator (TP8 attention/dense and the ETP4 x EP2 experts)."""

    def __init__(self, world):
        self.world = world
        from tensorfold.cuda import comm
        self.small = getattr(world, "small_gather", None) is not None
        self.reduce = self.small and comm._SMALL_ALLREDUCE          # TF_SMALL_ALLREDUCE=0: no LL row all-reduce
        lib = getattr(world, "lib", None)
        if lib is not None:      # ncclAllReduce typed once (not per call: ~80 sums per prompt forward)
            import ctypes
            fn = lib.ncclAllReduce
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                           ctypes.c_void_p, ctypes.c_void_p]
            fn.restype = ctypes.c_int
            self._nccl_all_reduce = fn

    def _all_reduce(self, tensor):
        """In-place NCCL sum of an FP32 or FP16 tensor."""
        comm = self.world
        stream = torch.cuda.current_stream().cuda_stream
        comm._check(self._nccl_all_reduce(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(),
                                          _NCCL_TYPES[tensor.dtype], _NCCL_SUM, comm.comm, stream))

    def path(self, n: int, dims: int) -> str:
        """The collective an FP16-rounded sum of ``n`` FP32 elements per rank ([n / dims, dims]) takes:
        gather, ll or nccl (module docstring)."""
        if self.small and n <= GATHER_MAX and n * 4 <= self.world.small_gather_bytes:
            return "gather"
        if self.reduce and n <= LL_MAX and self.world.small_gather.reduce_lines(n // dims, dims) > 0:
            return "ll"
        return "nccl"

    def _reduce_half(self, send: torch.Tensor, path: str) -> torch.Tensor:
        rows, dims = send.shape
        recv = torch.empty(rows, dims, dtype=torch.float16, device=send.device)
        if path == "gather":
            parts = torch.empty((WORLD, rows, dims), dtype=torch.float32, device=send.device)
            self.world.all_gather(send, parts)
            return rank_sum(parts, recv)
        if not self.world.all_reduce_rows_ll(send, recv):
            raise RuntimeError(f"the small row all-reduce refused [{rows}, {dims}] after admitting it")
        return recv

    def sum(self, partial: torch.Tensor, dtype=None) -> torch.Tensor:
        """8-rank sum of [rows, dims] partials, accumulated in FP32 and rounded once to ``dtype`` (default: the
        partial's); FP16 results take the path ``path`` selects."""
        dtype = dtype or partial.dtype
        if dtype == torch.float16:
            path = self.path(partial.numel(), partial.shape[-1])
            if path != "nccl":
                return self._reduce_half(partial.float().contiguous().view(-1, partial.shape[-1]), path)
        total = partial.float().contiguous()
        self._all_reduce(total)
        return total.to(dtype)

    def exact_sum_half(self, rows: torch.Tensor) -> torch.Tensor:
        """Sum of FP16 [rows, dims] partials to which at most one rank contributes each nonzero element (exact in
        any order, so bitwise the same on every path): the rank-ordered paths where they are faster, else the NCCL
        FP16 all-reduce in place (half the FP32 bytes)."""
        if self.path(rows.numel(), rows.shape[-1]) != "nccl":
            return self.sum(rows, torch.float16)
        self._all_reduce(rows)
        return rows

    def gather(self, part: torch.Tensor) -> torch.Tensor:
        """Concatenate the eight ranks' parts along the last dimension, rank order."""
        part = part.contiguous()
        recv = torch.empty((WORLD,) + tuple(part.shape), dtype=part.dtype, device=part.device)
        self.world.all_gather(part, recv)
        return torch.cat(list(recv.unbind(0)), dim=-1)

"""Original mixed-GGUF Flash Next weights on the existing family interfaces.

Large expert banks and the vocabulary embedding retain their original bytes;
ordinary model projections are decoded once, in bounded chunks, to FP16.
GGUF value-head tiling and partial quantization-block shards are restored at
load time without requantization. No vLLM runtime or operator ABI is imported.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
import time

import numpy as np
import torch

from tensorfold.cuda.direct_read import Reader
from tensorfold.cuda.sm70 import gguf_ops
from tensorfold.cleanup import raise_failures

from ..gguf import FORMATS, GGUFCheckpoint, PackedTensor, decode_rows, metadata_config, resolve_gguf
from ..host_table import PREFETCH_READ, _MappedTable, _prefetch, _random_access
from ..ple_lifetime import PLETables, _retain_failure
from ..read_ahead import ReadAhead, row_ids
from ..table_file import TableFile
from .weight_types import AttnW, Config, GDNW, HC, LayerW, MoEW, MTPW, PLEW, Weights, draft_token_ids

STAGING_BYTES = 32 << 20
DECODE_BYTES = 32 << 20
MATMUL_ROWS = 32  # Fixed M preserves a row's reduction policy across prompt/decode batch sizes.
PLE_PREFETCH_WORKERS = 8


def _ranges(indices):
    """Keep arbitrary converter permutations as a small ordered list of row runs."""
    result = []
    for index in indices:
        if result and result[-1][0] + result[-1][1] == index:
            lo, count = result[-1]
            result[-1] = lo, count + 1
        else:
            result.append((int(index), 1))
    return tuple(result)


@dataclass(frozen=True)
class MatrixPlan:
    source: PackedTensor
    row_ranges: tuple[tuple[int, int], ...]
    columns: tuple[tuple[int, int], ...]
    packed: bool = False

    @property
    def n(self):
        return sum(count for _, count in self.row_ranges)

    @property
    def k(self):
        return sum(count for _, count in self.columns)

    @property
    def byte_segments(self):
        _, block, size = FORMATS[self.source.ggml_type]
        if len(self.columns) > 1 and any(lo % block or count % block for lo, count in self.columns):
            raise ValueError("permuted GGUF columns must contain complete source blocks")
        return tuple((lo // block * size, ((lo + count + block - 1) // block - lo // block) * size)
                     for lo, count in self.columns)

    @property
    def row_bytes(self):
        return sum(length for _, length in self.byte_segments)

    @property
    def column_offset(self):
        return self.columns[0][0] % FORMATS[self.source.ggml_type][1]

    @property
    def resident_bytes(self):
        return self.n * (self.row_bytes if self.packed else self.k * 2)


@dataclass(frozen=True)
class ArrayPlan:
    source: PackedTensor
    indices: tuple[int, ...] | None = None
    dtype: str = "float32"
    transform: str | None = None

    @property
    def shape(self):
        return self.source.shape if self.indices is None else (len(self.indices), *self.source.shape[1:])

    @property
    def resident_bytes(self):
        return math.prod(self.shape) * (4 if self.dtype == "float32" else 2)

    @property
    def transform_host_staging_bytes(self):
        if self.transform != "half_to_float":
            return 0
        full, selected = math.prod(self.source.shape), math.prod(self.shape)
        # The indexed CPU array and the original FP32 decode may overlap;
        # conversion also holds FP32 values, FP16 rounding and the FP32 result.
        return self.source.nbytes + max(4 * full + 4 * selected, 10 * selected)


@dataclass(frozen=True)
class ConcatPlan:
    parts: tuple


@dataclass(frozen=True)
class ExpertPlan:
    gate: MatrixPlan
    up: MatrixPlan
    down: MatrixPlan
    shared: tuple
    count: int
    width: int
    dims: int


@dataclass(frozen=True)
class TablePlan:
    source: PackedTensor


class GGUFDense:
    kernel = "gguf"
    implementation = "reference"
    wmma_policy = "sm70-m32-n64-k64-8cta-v1"

    def __init__(self, weight):
        if weight.dtype != torch.float16 or weight.ndim != 2 or not weight.is_contiguous():
            raise ValueError("GGUF dense weight must be contiguous FP16 [output,input]")
        self.weight = weight
        self.n, self.k = map(int, weight.shape)

    def matmul(self, x, *, out=None, f32=False):
        if self.implementation not in ("reference", "native32", "wmma"):
            raise ValueError("unknown GGUF dense execution implementation")
        if self.implementation == "native32" and x.shape[0] <= 32:
            return self.matmul_native32(x, out=out, f32=f32)
        if self.implementation == "wmma" and x.shape[0] <= 32 and self.k % 16 == 0 and self.n <= 16384:
            return self.matmul_wmma(x, out=out, f32=f32)
        return self.matmul_reference(x, out=out, f32=f32)

    def matmul_native32(self, x, *, out=None, f32=False):
        if out is None:
            out = torch.empty((x.shape[0], self.n), dtype=torch.float32 if f32 else torch.float16, device=x.device)
        return gguf_ops.dense32(x, self.weight, out)

    def matmul_wmma(self, x, *, out=None, f32=False):
        if out is None:
            out = torch.empty((x.shape[0], self.n), dtype=torch.float32 if f32 else torch.float16, device=x.device)
        return gguf_ops.dense_wmma(x, self.weight, out)

    def matmul_reference(self, x, *, out=None, f32=False):
        if x.ndim != 2 or x.shape[1] != self.k or x.dtype != torch.float16 or x.device != self.weight.device:
            raise ValueError("GGUF linear expects FP16 rows matching its input width and device")
        dtype = out.dtype if out is not None else torch.float32 if f32 else torch.float16
        if dtype not in (torch.float16, torch.float32):
            raise ValueError("GGUF linear output must be FP16 or FP32")
        if out is None:
            out = torch.empty((x.shape[0], self.n), dtype=dtype, device=x.device)
        if out.shape != (x.shape[0], self.n) or out.device != x.device:
            raise ValueError("GGUF linear output has the wrong shape or device")
        for start in range(0, x.shape[0], MATMUL_ROWS):
            count = min(MATMUL_ROWS, x.shape[0] - start)
            padded = torch.zeros((MATMUL_ROWS, self.k), dtype=torch.float16, device=x.device)
            padded[:count].copy_(x[start:start + count])
            result = (torch.mm(padded, self.weight.T, out_dtype=torch.float32) if dtype == torch.float32
                      else torch.mm(padded, self.weight.T))
            out[start:start + count].copy_(result[:count])
        return out

    def __call__(self, x, out=None):
        return self.matmul(x, out=out)

    def prefill(self, x, out=None):
        return self.matmul(x, out=out)

    def partials(self, x):
        return self.matmul(x, f32=True).unsqueeze(0)

    def device_tensors(self):
        return (self.weight,)

    def nbytes(self):
        return self.weight.numel() * self.weight.element_size()

    def can_matmul_prepared(self, dtype):
        return (self.implementation in ("reference", "native32") and self.weight.is_cuda
                and dtype in (torch.float16, torch.float32))

    def matmul_prepared(self, x, out):
        """Caller-owned complete M32 operands; native helper retains generic fallbacks."""
        if (not self.can_matmul_prepared(out.dtype) or x.dtype != torch.float16
                or x.device != self.weight.device or out.device != x.device
                or x.shape != (MATMUL_ROWS, self.k) or out.shape != (MATMUL_ROWS, self.n)
                or not x.is_contiguous() or not out.is_contiguous()):
            raise ValueError("prepared GGUF dense requires owned contiguous FP16 M32 input and matching output")
        if torch._C._overlaps(out, x) or torch._C._overlaps(out, self.weight):
            raise ValueError("prepared GGUF output must not overlap its inputs")
        return gguf_ops.dense32(x, self.weight, out)


class GGUFConcat:
    kernel = "gguf"
    implementation = "reference"

    def __init__(self, parts):
        self.parts = tuple(parts)
        self.n, self.k = sum(p.n for p in parts), parts[0].k
        if any(p.k != self.k for p in parts):
            raise ValueError("stacked GGUF projections must have one input width")
        self.fused = None
        offsets, rows = [], 0
        for part in self.parts:
            offsets.append(rows)
            rows += part.n
        # Actual Flash Next widths make every part's origin allocator-aligned.
        # Retain that alignment for the separate reference GEMMs as well as the
        # original allocation size. Unsupported generic stacks keep their owners.
        if (all(isinstance(part, GGUFDense) for part in self.parts)
                and all((at * self.k * 2) % 512 == 0 for at in offsets)):
            joined = torch.cat([part.weight for part in self.parts], dim=0)
            self.fused = GGUFDense(joined)
            for at, part in zip(offsets, self.parts):
                part.weight = joined[at:at + part.n]

    def matmul(self, x, *, out=None, f32=False):
        if self.implementation not in ("reference", "fused", "native32"):
            raise ValueError("unknown GGUF stacked projection implementation")
        if self.implementation == "fused":
            return self.matmul_fused(x, out=out, f32=f32)
        if self.implementation == "native32" and x.shape[0] <= 32:
            return self.matmul_native32(x, out=out, f32=f32)
        return self.matmul_reference(x, out=out, f32=f32)

    def matmul_native32(self, x, *, out=None, f32=False):
        if not all(isinstance(part, GGUFDense) for part in self.parts):
            raise ValueError("native32 stacked projections require FP16 dense parts")
        if out is None:
            out = torch.empty((x.shape[0], self.n), dtype=torch.float32 if f32 else torch.float16, device=x.device)
        return gguf_ops.concat32(x, (part.weight for part in self.parts), out)

    def matmul_fused(self, x, *, out=None, f32=False):
        if self.fused is None:
            raise ValueError("fused GGUF projections require aligned FP16 dense parts")
        return self.fused.matmul(x, out=out, f32=f32)

    def matmul_reference(self, x, *, out=None, f32=False):
        dtype = out.dtype if out is not None else torch.float32 if f32 else torch.float16
        if out is None:
            out = torch.empty((x.shape[0], self.n), dtype=dtype, device=x.device)
        column = 0
        for part in self.parts:
            part.matmul(x, out=out[:, column:column + part.n])
            column += part.n
        return out

    def __call__(self, x, out=None):
        return self.matmul(x, out=out)

    def prefill(self, x, out=None):
        return self.matmul(x, out=out)

    def partials(self, x):
        return self.matmul(x, f32=True).unsqueeze(0)

    def device_tensors(self):
        if self.fused is not None:
            return self.fused.device_tensors()
        return tuple(t for p in self.parts for t in p.device_tensors())

    def nbytes(self):
        if self.fused is not None:
            return self.fused.nbytes()
        return sum(p.nbytes() for p in self.parts)

    def can_matmul_prepared(self, dtype):
        return (self.implementation in ("reference", "native32") and bool(self.parts)
                and all(isinstance(part, GGUFDense) and part.can_matmul_prepared(dtype) for part in self.parts))

    def matmul_prepared(self, x, out):
        """Preserve original part GEMMs; only qualified native layouts write columns directly."""
        if (not self.can_matmul_prepared(out.dtype) or x.dtype != torch.float16
                or x.device != self.parts[0].weight.device or out.device != x.device
                or x.shape != (MATMUL_ROWS, self.k) or out.shape != (MATMUL_ROWS, self.n)
                or not x.is_contiguous() or not out.is_contiguous()):
            raise ValueError("prepared GGUF concat requires owned contiguous FP16 M32 input and matching output")
        if torch._C._overlaps(out, x) or any(torch._C._overlaps(out, part.weight) for part in self.parts):
            raise ValueError("prepared GGUF output must not overlap its inputs")
        return gguf_ops.concat32(x, (part.weight for part in self.parts), out)


@dataclass
class GGUFPacked:
    data: torch.Tensor
    ggml_type: int
    n: int
    k: int
    column_offset: int = 0

    def decode(self, ids=None, *, dtype=torch.float16):
        return gguf_ops.decode(self.data, self.ggml_type, self.k, column_offset=self.column_offset,
                               dtype=dtype, ids=ids)

    def device_tensors(self):
        return (self.data,)

    def nbytes(self):
        return self.data.numel()


class GGUFEmbedding:
    def __init__(self, packed):
        self.packed = packed
        self.n, self.k = packed.n, packed.k

    def lookup(self, ids, *, copies=1, out=None):
        if ids.dtype not in (torch.int32, torch.int64) or ids.device != self.packed.data.device:
            raise ValueError("GGUF embedding IDs must be integer CUDA tokens on the embedding device")
        values = self.packed.decode(ids.reshape(-1).to(torch.int64))
        if copies != 1:
            values = values[:, None, :].expand(-1, copies, -1).reshape(values.shape[0], copies * self.k)
        if out is not None:
            if out.shape != values.shape or out.dtype != torch.float16 or out.device != values.device:
                raise ValueError("GGUF embedding output does not match its FP16 token rows")
            out.copy_(values)
            return out
        return values

    def device_tensors(self):
        return self.packed.device_tensors()

    def nbytes(self):
        return self.packed.nbytes()


class GGUFExperts:
    """Original routed banks with bounded GPU dequantization of live expert slots."""
    kernel = "gguf"
    capturable = True
    implementation = "reference"  # Select once, before engine warmup and state/cache preparation.
    packed_rows = 32  # Six depth3 verification windows use24 rows; prompts keep explicit BMM dispatch.

    def __init__(self, gate, up, down, shared, *, count, width, dims):
        self.gate, self.up, self.down, self.shared = gate, up, down, tuple(shared)
        self.count, self.width, self.dims = count + 1, width, dims
        self.limit, self.swiglu = 0.0, True
        capacity = max(1, DECODE_BYTES // (6 * width * dims))
        self.pairs = min(16, 1 << (capacity.bit_length() - 1))
        self._prefill_plan = None

    def prepare_prefill(self, plan):
        """Bind the already-admitted prompt scratch policy before engine warmup."""
        from .gguf_prefill import PrefillPlan

        if not isinstance(plan, PrefillPlan):
            raise ValueError("GGUF prefill requires an immutable PrefillPlan")
        plan.validate(width=self.width, dims=self.dims)
        if self._prefill_plan is not None:
            raise ValueError("GGUF prefill workspace is already prepared")
        self._prefill_plan = plan

    def forward(self, x, picks, out=None):
        if self.implementation not in ("reference", "packed", "packed_splitk4", "packed_blocks", "packed_blocks2w", "packed_exact"):
            raise ValueError("unknown GGUF expert execution implementation")
        if self.implementation == "packed" and x.shape[0] <= self.packed_rows:
            return self.forward_packed(x, picks, out)
        if self.implementation == "packed_splitk4" and x.shape[0] <= self.packed_rows:
            return self.forward_splitk4(x, picks, out)
        if self.implementation == "packed_blocks" and x.shape[0] <= self.packed_rows:
            return self.forward_blocks(x, picks, out)
        if self.implementation == "packed_blocks2w" and x.shape[0] <= self.packed_rows:
            return self.forward_blocks2w(x, picks, out)
        if self.implementation == "packed_exact" and x.shape[0] <= self.packed_rows:
            return self.forward_exact(x, picks, out)
        return self.forward_reference(x, picks, out)

    def forward_packed(self, x, picks, out=None):
        if out is None:
            out = torch.empty((*picks.shape, self.dims), dtype=torch.float32, device=x.device)
        return gguf_ops.experts(x, picks, self.gate, self.up, self.down, self.shared, out)

    def forward_splitk4(self, x, picks, out=None):
        if out is None:
            out = torch.empty((*picks.shape, self.dims), dtype=torch.float32, device=x.device)
        return gguf_ops.experts(x, picks, self.gate, self.up, self.down, self.shared, out, split_k=4)

    def forward_blocks(self, x, picks, out=None):
        if out is None:
            out = torch.empty((*picks.shape, self.dims), dtype=torch.float32, device=x.device)
        return gguf_ops.experts(x, picks, self.gate, self.up, self.down, self.shared, out, blockwise=True)

    def forward_blocks2w(self, x, picks, out=None):
        if out is None:
            out = torch.empty((*picks.shape, self.dims), dtype=torch.float32, device=x.device)
        return gguf_ops.experts(x, picks, self.gate, self.up, self.down, self.shared, out,
                               blockwise=True, projection_warps=2)

    def forward_exact(self, x, picks, out=None):
        """Retain original BMM bytes while executing eligible target banks packed."""
        if self.width not in (80, 160) or self.dims != 2560 or x.shape[0] > self.packed_rows:
            return self.forward_reference(x, picks, out)
        if out is None:
            out = torch.empty((*picks.shape, self.dims), dtype=torch.float32, device=x.device)
        gguf_ops.experts_exact(x, picks, self.gate, self.up, self.down, self.shared, out)
        return self._write_shared(x, picks, out)

    def _write_shared(self, x, picks, out):
        if x.shape[0]:
            gate, up, down = self.shared
            shared = down.matmul((torch.nn.functional.silu(gate.matmul(x, f32=True))
                                  * up.matmul(x, f32=True)).half(), f32=True)
            out.copy_(torch.where((picks == self.count - 1)[..., None], shared[:, None], out))
        return out

    def forward_prefill(self, x, picks, out=None):
        """Keep prompt and cached-prefix tails on the same reference arithmetic."""
        if self._prefill_plan is not None:
            from .gguf_prefill import call_pairs

            pairs = call_pairs(self._prefill_plan, x.shape[0], picks.shape[1], width=self.width, dims=self.dims)
            if pairs == self.pairs:
                return self.forward_reference(x, picks, out)
            return self.forward_reference(x, picks, out, _pairs=pairs)
        return self.forward_reference(x, picks, out)

    def forward_reference(self, x, picks, out=None, *, _pairs=None):
        if (x.ndim != 2 or x.shape[1] != self.dims or x.dtype != torch.float16
                or picks.ndim != 2 or picks.shape[0] != x.shape[0]
                or picks.dtype not in (torch.int32, torch.int64) or picks.device != x.device):
            raise ValueError("GGUF expert input/slot rows have incompatible geometry or dtype")
        rows, slots = picks.shape
        pairs = self.pairs if _pairs is None else _pairs
        if out is None:
            out = torch.empty((rows, slots, self.dims), dtype=torch.float32, device=x.device)
        if (out.shape != (rows, slots, self.dims) or out.dtype not in (torch.float16, torch.float32)
                or out.device != x.device or not out.is_contiguous()):
            raise ValueError("GGUF expert output must be [rows,slots,hidden] FP16 or FP32")
        flattened = picks.reshape(-1)
        target = out.reshape(-1, self.dims)
        width_ids = torch.arange(self.width, device=x.device, dtype=torch.int64)
        hidden_ids = torch.arange(self.dims, device=x.device, dtype=torch.int64)
        for start in range(0, rows * slots, pairs):
            count = min(pairs, rows * slots - start)
            ids = torch.zeros(pairs, dtype=torch.int64, device=x.device)
            ids[:count].copy_(flattened[start:start + count])
            # The shared slot is evaluated separately. Its routed placeholder
            # only reads valid expert-zero bytes and is never published.
            routed = torch.where(ids == self.count - 1, 0, ids)
            row_ids_ = (torch.arange(pairs, device=x.device) + start).clamp_max(rows * slots - 1) // slots
            inputs = x.index_select(0, row_ids_)
            bank_rows = (routed[:, None] * self.width + width_ids).reshape(-1)
            gate = self.gate.decode(bank_rows).reshape(pairs, self.width, self.dims)
            up = self.up.decode(bank_rows).reshape(pairs, self.width, self.dims)
            gated = torch.bmm(inputs[:, None], gate.transpose(1, 2), out_dtype=torch.float32).squeeze(1)
            upward = torch.bmm(inputs[:, None], up.transpose(1, 2), out_dtype=torch.float32).squeeze(1)
            # Match the selected FP16 storage boundary after the SwiGLU product.
            activation = (torch.nn.functional.silu(gated) * upward).half()
            down_rows = (routed[:, None] * self.dims + hidden_ids).reshape(-1)
            down = self.down.decode(down_rows).reshape(pairs, self.dims, self.width)
            result = torch.bmm(activation[:, None], down.transpose(1, 2), out_dtype=torch.float32).squeeze(1)
            target[start:start + count].copy_(result[:count])
        return self._write_shared(x, picks, out)

    def device_tensors(self):
        return tuple(t for p in (self.gate, self.up, self.down, *self.shared) for t in p.device_tensors())

    def bytes_per_expert(self):
        return self.width * (self.gate.data.shape[1] + self.up.data.shape[1]) + self.dims * self.down.data.shape[1]


class GGUFPLETable(_MappedTable):
    """The fork's existing mapped/read-ahead lifetime over original IQ4_NL rows."""
    bits = 16  # Values handed to the family staging interface are decoded FP16.
    dtype = "float16"
    weight_scale = 1.0

    def __init__(self, source):
        with self._construction():
            if source.encoding != "IQ4_NL" or len(source.shape) != 2:
                raise ValueError("Flash Next GGUF PLE requires an IQ4_NL matrix")
            self.source = source
            file = TableFile(source.path, min_bytes=source.offset + source.nbytes)
            self._unpublished_file[0] = file
            self._files.append(file)
            self._unpublished_file[0] = None
            file.open()
            self._fds.append(file.fileno())
            self.values = self._arrays()
            self.values.append(np.memmap(file, mode="r", dtype=np.uint8, offset=source.offset,
                                         shape=(source.rows, source.row_bytes)))
            _random_access(self.values[0])
            self.rows, self.width = source.shape
            self.wrow, self.nbytes = source.row_bytes, source.nbytes

    def gather(self, ids):
        return self._life.call(self._gather_flat, row_ids(ids, self.rows))

    def _gather_flat(self, ids):
        packed = np.ascontiguousarray(self.values[0][ids])
        values = decode_rows(self.source, packed)
        result = values.astype(np.float16)
        if np.any(np.isfinite(values) & ~np.isfinite(result)):
            raise ValueError("IQ4_NL PLE row overflows FP16 execution")
        return result

    def prefetch(self, workers=PLE_PREFETCH_WORKERS):
        return self._life.call(_prefetch, self.values, workers)

    def lock(self):
        return self._life.call(self._lock_arrays, self.values)


class Planner:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.used = set()
        self.matrices = []
        self.arrays = []
        self.tables = []

    def matrix(self, name, *, rows=None, columns=None, packed=False):
        source = self.checkpoint.tensors[name]
        self.used.add(name)
        ranges = ((0, source.rows),) if rows is None else _ranges(rows)
        segments = ((0, source.shape[-1]),) if columns is None else tuple(columns)
        if any(lo < 0 or count <= 0 or lo + count > source.rows for lo, count in ranges):
            raise ValueError(f"{name}: invalid GGUF matrix row shard")
        if any(lo < 0 or count <= 0 or lo + count > source.shape[-1] for lo, count in segments):
            raise ValueError(f"{name}: invalid GGUF matrix column shard")
        result = MatrixPlan(source, ranges, segments, packed)
        result.byte_segments  # Establish block-preserving segment geometry before IO/allocation.
        self.matrices.append(result)
        return result

    def array(self, name, indices=None, *, dtype="float32", transform=None):
        self.used.add(name)
        result = ArrayPlan(self.checkpoint.tensors[name], None if indices is None else tuple(indices), dtype, transform)
        self.arrays.append(result)
        return result


def model_plan(path, tp=None, *, rope=None, eos_token_ids=None, mtp=False, draft_vocab=None, draft_ids=None,
               position_profile=None):
    """One immutable source of allocation and layout decisions for admission and load."""
    entry = resolve_gguf(path)
    if entry is None:
        raise ValueError("no GGUF checkpoint was selected")
    checkpoint = GGUFCheckpoint.open(entry)
    raw = metadata_config(entry)
    # Config.read uses the same metadata dispatch for raw GGUF and staged bundles.
    full = Config.read(path, rope=rope, eos_token_ids=eos_token_ids)
    if position_profile is not None:
        from ..position_profile import ResolvedPositionProfile

        if not isinstance(position_profile, ResolvedPositionProfile) or position_profile.target.rope != full.rope:
            raise ValueError("GGUF positional profile differs from the planned target transform")
    full = replace(full, quant="gguf", execution_dtype="float16",
                   ple_eos=int(raw["text_config"]["gguf_ple_eos_token_id"]))
    rank, world = (0, 1) if tp is None else tp
    if (type(world) is not int or world not in (1, 2, 4, 8) or type(rank) is not int or rank not in range(world)
            or any(v % world for v in (full.heads, full.nk, full.nv, full.moe_width, full.shared_width, full.vocab))):
        raise ValueError("GGUF Flash Next TP must legally divide heads, expert widths and vocabulary")
    selected = None
    if mtp:
        selected = draft_token_ids(draft_vocab, full.vocab) if draft_ids is None else draft_ids
        if selected is not None:
            if (not isinstance(selected, np.ndarray) or selected.dtype != np.int64 or selected.ndim != 1
                    or not selected.flags.c_contiguous or selected.size == 0
                    or selected[0] < 0 or selected[-1] >= full.vocab
                    or np.any(selected[1:] <= selected[:-1])):
                raise ValueError("GGUF draft IDs must be sorted unique global int64 token IDs within the vocabulary")
            if selected.size < world:
                raise ValueError("the draft vocabulary must contain at least one ID per tensor-parallel rank")
    if full.kv_heads >= world:
        if full.kv_heads % world:
            raise ValueError("GGUF key/value heads must divide TP or be replicated in equal rank groups")
        local_kv, first_kv = full.kv_heads // world, rank * (full.kv_heads // world)
    else:
        if world % full.kv_heads:
            raise ValueError("GGUF TP must replicate key/value heads in equal rank groups")
        local_kv, first_kv = 1, rank // (world // full.kv_heads)
    cfg = replace(full, heads=full.heads // world, kv_heads=local_kv, nk=full.nk // world, nv=full.nv // world,
                  moe_width=full.moe_width // world, shared_width=full.shared_width // world)
    planner = Planner(checkpoint)

    def hc(base, inject=True):
        down = planner.matrix(base + "down.weight")
        if inject:
            down = ConcatPlan((down, planner.matrix(base + "inject.weight")))
        up = planner.matrix(base + "up.weight")
        return HC(down, up, planner.array(base + "norm.weight"), inject, down, up)

    def moe(base):
        width = full.moe_width
        lo, hi = rank * cfg.moe_width, (rank + 1) * cfg.moe_width
        rows = [e * width + i for e in range(full.experts) for i in range(lo, hi)]
        gate = planner.matrix(base + "ffn_gate_exps.weight", rows=rows, packed=True)
        up = planner.matrix(base + "ffn_up_exps.weight", rows=rows, packed=True)
        down = planner.matrix(base + "ffn_down_exps.weight", columns=((lo, hi - lo),), packed=True)
        slo, shi = rank * cfg.shared_width, (rank + 1) * cfg.shared_width
        shared = (planner.matrix(base + "ffn_gate_shexp.weight", rows=range(slo, shi)),
                  planner.matrix(base + "ffn_up_shexp.weight", rows=range(slo, shi)),
                  planner.matrix(base + "ffn_down_shexp.weight", columns=((slo, shi - slo),)))
        router = ConcatPlan((planner.array(base + "ffn_gate_inp.weight", transform="half_to_float"),
                             planner.array(base + "ffn_gate_inp_shexp.weight", transform="half_to_float")))
        return MoEW(router, ExpertPlan(gate, up, down, shared, full.experts, cfg.moe_width, full.hidden))

    def gdn(base):
        # GGML's value dimension is tiled [repeat,key_head,DV], not the
        # family's [key_head,repeat,DV] order. Preserve source blocks while restoring it.
        repeat = full.nv // full.nk
        values = [h % repeat * full.nk + h // repeat for h in range(rank * cfg.nv, (rank + 1) * cfg.nv)]
        qrows = list(range(rank * cfg.nk * full.dk, (rank + 1) * cfg.nk * full.dk))
        vrows = [h * full.dv + i for h in values for i in range(full.dv)]
        channels = qrows + [full.nk * full.dk + i for i in qrows] + [2 * full.nk * full.dk + i for i in vrows]
        proj = ConcatPlan((planner.matrix(base + "attn_qkv.weight", rows=channels),
                           planner.matrix(base + "attn_gate.weight", rows=vrows),
                           planner.matrix(base + "ssm_beta.weight", rows=values),
                           planner.matrix(base + "ssm_alpha.weight", rows=values)))
        return GDNW(proj, planner.array(base + "ssm_conv1d.weight", channels, dtype="float16"),
                    None,
                    planner.array(base + "ssm_dt.bias", values),
                    planner.array(base + "ssm_norm.weight", dtype="float16"),
                    planner.matrix(base + "ssm_out.weight", columns=tuple((h * full.dv, full.dv) for h in values)),
                    a_neg=planner.array(base + "ssm_a", values, transform="negative_exp"))

    def attention(base):
        hd = full.head_dim
        qrows = range(rank * cfg.heads * 2 * hd, (rank + 1) * cfg.heads * 2 * hd)
        kvrows = range(first_kv * hd, (first_kv + local_kv) * hd)
        proj = ConcatPlan((planner.matrix(base + "attn_q.weight", rows=qrows),
                           planner.matrix(base + "attn_k.weight", rows=kvrows),
                           planner.matrix(base + "attn_v.weight", rows=kvrows),
                           planner.matrix(base + "indexer.q_proj.weight"),
                           planner.matrix(base + "indexer.k_proj.weight")))
        return AttnW(proj, planner.array(base + "attn_q_norm.weight"), planner.array(base + "attn_k_norm.weight"),
                     planner.array(base + "indexer.q_norm.weight"), planner.array(base + "indexer.k_norm.weight"),
                     planner.matrix(base + "attn_output.weight", columns=((rank * cfg.heads * hd, cfg.heads * hd),)))

    layers = []
    for i, kind in enumerate(full.layer_types):
        base = f"blk.{i}."
        linear = kind == "linear"
        layer = LayerW(i, linear, hc(base + "hc_attn_"), hc(base + "hc_ffn_"),
                       gdn(base) if linear else None, None if linear else attention(base), moe(base))
        if i in cfg.ple_layers:
            ngram = cfg.ngram(cfg.ple_layers.index(i))
            constants = raw["text_config"]["gguf_ple_constants"]
            ngram.check(constants["layer_multipliers"], constants["ngram_heads_offsets"],
                        constants["ngram_heads_vocab_sizes"])
            source = checkpoint.tensors["per_layer_token_embd.weight"]
            if source.shape != (ngram.rows, ngram.dims):
                raise ValueError("GGUF PLE table shape disagrees with its hash constants")
            planner.used.add(source.name)
            table = TablePlan(source)
            planner.tables.append(table)
            layer.ple = PLEW(table, planner.matrix(base + "ple_key.weight"),
                             planner.matrix(base + "ple_value.weight"), planner.array(base + "ple_norm_key.weight"),
                             planner.array(base + "ple_norm_query.weight"), planner.array(base + "ple_norm_conv.weight"),
                             planner.array(base + "ple_conv1d.weight", dtype="float16"), ngram)
        layers.append(layer)
    prototype = Weights(cfg, (planner.matrix("token_embd.weight", packed=True),), layers, hc("output_hc_", False),
                        planner.matrix("output.weight", rows=range(rank * (full.vocab // world),
                                                                  (rank + 1) * (full.vocab // world))), None,
                        around_one=True)
    if selected is not None:
        prototype.draft_ids = np.array_split(selected, world)[rank]
        prototype.draft_head = planner.matrix("output.weight", rows=prototype.draft_ids)
    unused = set(checkpoint.tensors) - planner.used
    if unused:
        raise ValueError("unmapped GGUF Flash Next tensors: " + ", ".join(sorted(unused)))
    prototype.meta.update(full=full, rank=rank, world=world, vocab_offset=rank * (full.vocab // world),
                          execution_provider="sm70", execution_dtype="float16", gguf_entrypoint=str(entry),
                          gdn_decay_representation="ggml-stored-negative-exp-fp32",
                          gdn_head_order="restored-key-major-before-tp",
                          draft_host_id_bytes=0 if selected is None else selected.nbytes + prototype.draft_ids.nbytes)
    if position_profile is not None:
        prototype.meta["position_profile"] = position_profile
    if checkpoint.metadata.get("tokenizer.ggml.tokens"):
        from ..gguf import token_id_identity

        prototype.meta["gguf_token_id_identity"] = token_id_identity(checkpoint.metadata)
    return prototype, planner


def weight_byte_plan(path, tp=None, *, mtp=False, draft_vocab=None, draft_ids=None, prefill_plan=None,
                     position_profile=None):
    # Separate stock-MTP admission is added by the common engine admission path.
    prototype, planner = model_plan(path, tp, mtp=mtp, draft_vocab=draft_vocab, draft_ids=draft_ids,
        **({"position_profile": position_profile, "rope": position_profile.target.rope}
           if position_profile is not None else {}))
    resident = sum(m.resident_bytes for m in planner.matrices) + sum(a.resident_bytes for a in planner.arrays)
    draft_id_bytes = 0 if prototype.draft_ids is None else prototype.draft_ids.nbytes
    resident += draft_id_bytes
    workspace = DECODE_BYTES + MATMUL_ROWS * max(prototype.cfg.streams * prototype.cfg.hidden,
                                               prototype.head.n) * 4
    if prefill_plan is not None:
        from .gguf_prefill import PrefillPlan

        if not isinstance(prefill_plan, PrefillPlan):
            raise ValueError("GGUF prefill requires an immutable PrefillPlan")
        prefill_plan.validate(width=prototype.cfg.moe_width, dims=prototype.cfg.hidden,
                              slots=prototype.cfg.top_k + 1)
        workspace += max(0, prefill_plan.workspace_bytes - DECODE_BYTES)
    workspace += prototype.cfg.rotary_dim // 2 * 4  # Shared Weights.nbytes accounts projections separately from RoPE.
    # Each background PLE warm owns one reusable buffer per worker, concurrent
    # with the loader's raw, selected and concatenated source spans. Conservatively
    # charge every mapped table even when prefetch is disabled by the caller.
    prefetch_host = len(planner.tables) * PLE_PREFETCH_WORKERS * PREFETCH_READ
    router_parts = [a for a in planner.arrays if a.transform == "half_to_float"]
    router_host = max((a.transform_host_staging_bytes for a in router_parts), default=0)
    return {"resident_bytes": resident, "staging_bytes": STAGING_BYTES,
            "workspace_bytes": workspace,
            "host_resident_bytes": sum(t.source.nbytes for t in planner.tables),
            "host_staging_bytes": max(3 * STAGING_BYTES, router_host) + prefetch_host
                                  + prototype.meta["draft_host_id_bytes"],
            "router_resident_bytes": sum(a.resident_bytes for a in router_parts),
            "router_host_staging_bytes": router_host,
            "draft_head_bytes": 0 if prototype.draft_head is None else prototype.draft_head.resident_bytes,
            "draft_id_bytes": draft_id_bytes, "draft_host_id_bytes": prototype.meta["draft_host_id_bytes"],
            "prefill_experts_workspace_bytes": 0 if prefill_plan is None else prefill_plan.workspace_bytes,
            "execution_dtype": "float16"}


def _packed(plan, reader, device):
    result = torch.empty((plan.n, plan.row_bytes), dtype=torch.uint8, device=device)
    destination = 0
    max_rows = max(1, STAGING_BYTES // plan.source.row_bytes)
    runs = []
    for first, count in plan.row_ranges:
        for start in range(first, first + count, max_rows):
            runs.append((start, min(max_rows, first + count - start)))
    at = 0
    while at < len(runs):
        group = [runs[at]]
        lo, hi = runs[at][0], runs[at][0] + runs[at][1]
        at += 1
        while at < len(runs):
            begin, count = runs[at]
            if max(hi, begin + count) - min(lo, begin) > max_rows:
                break
            group.append(runs[at])
            lo, hi = min(lo, begin), max(hi, begin + count)
            at += 1
        raw = reader.read(plan.source.path, plan.source.offset + lo * plan.source.row_bytes,
                          (hi - lo) * plan.source.row_bytes, "cpu").reshape(hi - lo, plan.source.row_bytes)
        indices = torch.tensor([index - lo for begin, count in group for index in range(begin, begin + count)],
                               dtype=torch.int64)
        parts = [raw[:, begin:begin + length].index_select(0, indices) for begin, length in plan.byte_segments]
        selected = parts[0].contiguous() if len(parts) == 1 else torch.cat(parts, dim=1)
        result[destination:destination + indices.numel()].copy_(selected.to(device))
        destination += indices.numel()
    return GGUFPacked(result, plan.source.ggml_type, plan.n, plan.k, plan.column_offset)


def _dense(plan, reader, device):
    result = torch.empty((plan.n, plan.k), dtype=torch.float16, device=device)
    max_rows = max(1, min(STAGING_BYTES // plan.source.row_bytes, DECODE_BYTES // (2 * plan.k)))
    destination = 0
    for first, count in plan.row_ranges:
        for start in range(first, first + count, max_rows):
            rows = min(max_rows, first + count - start)
            part = MatrixPlan(plan.source, ((start, rows),), plan.columns, True)
            packed = _packed(part, reader, device)
            result[destination:destination + rows].copy_(packed.decode())
            destination += rows
    return GGUFDense(result)


def _draft_dense(plan, reader, device):
    """Gather global output rows in bounded spans, including other target shards."""
    result = torch.empty((plan.n, plan.k), dtype=torch.float16, device=device)
    limit = max(1, min(STAGING_BYTES // plan.source.row_bytes, DECODE_BYTES // (2 * plan.k)))
    destination, batch, rows = 0, [], 0

    def publish():
        nonlocal destination
        part = replace(plan, row_ranges=tuple(batch), packed=True)
        packed = _packed(part, reader, device)
        if result.is_cuda:
            value = packed.decode()
        else:
            value = torch.from_numpy(decode_rows(plan.source, packed.data.numpy()).astype(np.float16))
        result[destination:destination + rows].copy_(value)
        destination += rows

    for first, count in plan.row_ranges:
        while count:
            take = min(count, limit - rows)
            batch.append((first, take))
            first, count, rows = first + take, count - take, rows + take
            if rows == limit:
                publish()
                batch, rows = [], 0
    if batch:
        publish()
    return GGUFDense(result)


def _array(plan, reader, device):
    source = plan.source
    if source.nbytes > STAGING_BYTES:
        raise ValueError("GGUF scalar/normalization tensor exceeds the bounded array reader")
    raw = reader.read(source.path, source.offset, source.nbytes, "cpu").reshape(source.rows, source.row_bytes)
    values = decode_rows(source, raw.numpy()).reshape(source.shape)
    if plan.indices is not None:
        values = values[np.asarray(plan.indices)]
    if plan.transform == "negative_exp":
        if not np.all(np.isfinite(values) & (values < 0)):
            raise ValueError("GGUF ssm_a must contain finite negative exp(A_log)")
    elif plan.transform == "half_to_float":
        # Router arithmetic consumes FP32 of the selected FP16 coefficients.
        # Retain that exact rounding while storing its already-cast Tensor.
        rounded = values.astype(np.float16)
        if np.any(np.isfinite(values) & ~np.isfinite(rounded)):
            raise ValueError("GGUF tensor overflows its selected execution dtype")
        values = rounded
    dtype = np.float32 if plan.dtype == "float32" else np.float16
    converted = values.astype(dtype)
    if np.any(np.isfinite(values) & ~np.isfinite(converted)):
        raise ValueError("GGUF tensor overflows its selected execution dtype")
    return torch.from_numpy(np.ascontiguousarray(converted)).to(device)


def materializer(reader, device, *, dense_loader=None, prepare_experts=None, load_table=None):
    """One planned owner maps to one runtime owner within a single load.

    The caller owns Reader/table retirement. Router ArrayPlan parts are the
    exception to caching: their joined Tensor owns the result, so retaining
    those temporary inputs would exceed the per-tensor staging contract.
    Dense import, expert preparation and tables are concrete caller policies;
    the default target invokes its existing loaders directly, without a
    per-matrix adapter. CPU MTP import controls keep their independent decoder.
    """
    dense_loader = _dense if dense_loader is None else dense_loader
    cache = {}

    def materialize(value):
        if isinstance(value, (MatrixPlan, ArrayPlan, ConcatPlan, ExpertPlan, TablePlan)):
            key = id(value)
            if key not in cache:
                cache[key] = materialize_owned(value)
            return cache[key]
        return materialize_owned(value)

    def materialize_owned(value):
        if isinstance(value, MatrixPlan):
            return _packed(value, reader, device) if value.packed else dense_loader(value, reader, device)
        if isinstance(value, ArrayPlan):
            return _array(value, reader, device)
        if isinstance(value, ConcatPlan):
            if all(isinstance(part, ArrayPlan) for part in value.parts):
                parts = tuple(_array(part, reader, device) for part in value.parts)
                return torch.cat([part if part.ndim == 2 else part[None] for part in parts])
            parts = tuple(materialize(part) for part in value.parts)
            if isinstance(parts[0], torch.Tensor):
                return torch.cat([part if part.ndim == 2 else part[None] for part in parts])
            return GGUFConcat(parts)
        if isinstance(value, ExpertPlan):
            bank = GGUFExperts(materialize(value.gate), materialize(value.up), materialize(value.down),
                               tuple(materialize(part) for part in value.shared), count=value.count,
                               width=value.width, dims=value.dims)
            if prepare_experts is not None:
                prepare_experts(bank)
            return bank
        if isinstance(value, TablePlan):
            if load_table is None:
                raise ValueError("GGUF graph contains a table without its owning loader")
            return load_table(value)
        if isinstance(value, (HC, GDNW, AttnW, MoEW, PLEW, LayerW, MTPW)):
            return type(value)(**{field.name: materialize(getattr(value, field.name)) for field in fields(value)})
        return value

    return materialize


def load_gguf(path, device="cuda", *, tp=None, mtp=False, rope=None, eos_token_ids=None,
              table_reads=None, ple_on_ssd=False, mtp_model_dir=None, draft_vocab=None, _draft_ids=None,
              _prefill_plan=None, position_profile=None, _position_prepared=None, **kwargs):
    """Load original checkpoint bytes into the maintained Flash Next weight graph."""
    if mtp and mtp_model_dir is None:
        raise ValueError("this target GGUF contains no MTP head; select --no-drafts until a matching draft is loaded")
    if kwargs.get("vram_experts") is not None or kwargs.get("expert_loader") is not None:
        raise ValueError("GGUF routed banks use their own admitted resident shards; external expert pools are unsupported")
    if position_profile is not None:
        from ..position_profile import PreparedPositionVectors

        if not isinstance(_position_prepared, PreparedPositionVectors):
            raise ValueError("GGUF positions require exact CPU preparation before model/device retention")
        _position_prepared.verify(position_profile, mtp_active=mtp)
    elif _position_prepared is not None:
        raise ValueError("prepared GGUF positional vectors lack their resolved profile")
    started = time.monotonic()
    prototype, planner = model_plan(path, tp, rope=rope, eos_token_ids=eos_token_ids,
                                    mtp=mtp, draft_vocab=draft_vocab, draft_ids=_draft_ids,
                                    position_profile=position_profile)
    # Even the direct legacy loader computes its exact small CPU vector before
    # Reader/device weight retention, rather than discovering range failure at
    # the end of a model import. Selected profiles adopt their prepared bytes.
    if _position_prepared is not None:
        target_inv = _position_prepared.target_cpu
    else:
        with torch.inference_mode(False), torch.device("cpu"):
            target_inv = prototype.cfg.rope.inverse_frequencies(torch)
            if not bool(torch.isfinite(target_inv).all()) or not bool((target_inv > 0).all()):
                raise ValueError("GGUF positions produce nonfinite or zero float32 inverse frequencies")
    if _prefill_plan is not None:
        from .gguf_prefill import PrefillPlan

        if not isinstance(_prefill_plan, PrefillPlan):
            raise ValueError("GGUF prefill requires an immutable PrefillPlan")
        _prefill_plan.validate(width=prototype.cfg.moe_width, dims=prototype.cfg.hidden,
                               slots=prototype.cfg.top_k + 1)
    reader, tables = Reader(), PLETables()
    # Same-host TP ranks reuse the kernel's shared read cache. Reader still owns
    # each bounded span; giant quantized banks are never materialized in full RAM.
    reader.direct = False

    def load_table(value):
        table = tables.acquire(ReadAhead(GGUFPLETable(value.source)))
        if table_reads is not None and not ple_on_ssd:
            from tensorfold.cuda.direct_read import in_background
            in_background(table.prefetch, tables.reads)
            table_reads.append(tables.reads[-1])
        return table

    prepare_experts = None
    if _prefill_plan is not None:
        def prepare_experts(bank):
            bank.prepare_prefill(_prefill_plan)

    materialize = materializer(reader, device, prepare_experts=prepare_experts, load_table=load_table)

    try:
        embedding = GGUFEmbedding(materialize(prototype.embed[0]))
        layers = [materialize(layer) for layer in prototype.layers]
        result = Weights(prototype.cfg, (embedding,), layers, materialize(prototype.mixer),
                         materialize(prototype.head), target_inv.to(device),
                         around_one=True)
        result.meta.update(prototype.meta, ple_tables=tables, rope=prototype.cfg.rope.metadata(),
                           load_seconds=time.monotonic() - started)
        if _prefill_plan is not None:
            result.meta["prefill_experts"] = _prefill_plan.receipt()
        if mtp:
            from .sm70_mtp_weights import attach
            attach(result, mtp_model_dir, tp=(result.meta["rank"], result.meta["world"]))
            if prototype.draft_head is not None:
                result.draft_head = _draft_dense(prototype.draft_head, reader, device)
                result.draft_ids = torch.from_numpy(prototype.draft_ids.copy()).to(device)
        reader.close()
        return result
    except BaseException as primary:
        _retain_failure(primary, "ple_tables", tables)
        failures = []
        for cleanup in (reader.close, lambda: tables.close(lambda: torch.cuda.synchronize(device))):
            try:
                cleanup()
            except BaseException as error:
                failures.append(error)
        raise_failures(primary, failures)

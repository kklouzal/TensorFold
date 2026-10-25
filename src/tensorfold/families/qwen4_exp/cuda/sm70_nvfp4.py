"""Retained NVFP4 software weights on the existing FP16 family linear face.

One serialized model producer owns the shared workspace. Weight tiles and the
fixed M32 operands have context-independent extents; no complete decoded weight
matrix is cached. This explicit numerical profile needs model qualification.
"""

from bisect import bisect_right

import torch
import numpy as np

from tensorfold.cuda.sm70 import nvfp4_ops
from tensorfold.cuda.sm70.nvfp4_layout import (EXPERT_DEVICE, EXPERT_FAMILIES, EXPERT_FIELDS, EXPERT_REFERENCE,
    INPUTS, LINEAR_FAMILIES, LINEAR_REFERENCE, LINEAR_WMMA, OUTPUTS, PROFILE, ROWS,
    expert_shape, expert_workspace_bytes)
from .sm70_mtp_weights import Dense as _DenseFace

NUMERICAL_PROFILE = PROFILE


def workspace_bytes(*, width=0, dims=0, kernel_family=LINEAR_REFERENCE, expert_family=EXPERT_REFERENCE,
                    selection_rows=0, slots=0, experts=0):
    if kernel_family not in LINEAR_FAMILIES:
        raise ValueError("unknown SM70 NVFP4 linear kernel family")
    base = (OUTPUTS * INPUTS * 2 + ROWS * INPUTS * 2 + 2 * ROWS * OUTPUTS * 4
            if kernel_family == LINEAR_REFERENCE else 0)
    if expert_family not in EXPERT_FAMILIES:
        raise ValueError("unknown SM70 NVFP4 expert kernel family")
    if not width or not dims:
        return base
    return base + (expert_workspace_bytes(selection_rows, slots, experts, width) if expert_family == EXPERT_DEVICE
                   else ROWS * (14 * width + 6 * dims + 8))


def host_workspace_bytes(rows, slots, experts, *, expert_family=EXPERT_REFERENCE):
    if expert_family not in EXPERT_FAMILIES:
        raise ValueError("unknown SM70 NVFP4 expert kernel family")
    return 0 if expert_family == EXPERT_DEVICE else rows * slots * 12 + (3 * experts + 1) * 8 + ROWS * 8


class Workspace:
    def __init__(self, device, *, width=0, dims=0, selection_rows=0, slots=0, experts=0,
                 kernel_family=LINEAR_REFERENCE, expert_family=EXPERT_REFERENCE):
        if kernel_family not in LINEAR_FAMILIES or expert_family not in EXPERT_FAMILIES:
            raise ValueError("unknown SM70 NVFP4 linear/expert kernel family")
        self.device, self.kernel_family = torch.device(device), kernel_family
        self.expert_family, self.width, self.dims = expert_family, width, dims
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.matrix_storage = ()
        if kernel_family == LINEAR_REFERENCE:
            self.decoded = torch.empty((OUTPUTS, INPUTS), dtype=torch.float16, device=device)
            self.input = torch.empty((ROWS, INPUTS), dtype=torch.float16, device=device)
            self.partial = torch.empty((ROWS, OUTPUTS), dtype=torch.float32, device=device)
            self.accum = torch.empty_like(self.partial)
            self.matrix_storage = (self.decoded, self.input, self.partial, self.accum)
        self.expert_tensors = ()
        if width and dims:
            pairs, tiles = expert_shape(selection_rows, slots, experts)
            self.selection_rows, self.slots, self.experts = selection_rows, slots, experts
            if expert_family == EXPERT_DEVICE:
                self.counts_device = torch.empty(experts, dtype=torch.int32, device=device)
                self.starts_device = torch.empty(experts + 1, dtype=torch.int32, device=device)
                self.members_device = torch.empty(pairs, dtype=torch.int32, device=device)
                self.tile_expert = torch.empty(tiles, dtype=torch.int32, device=device)
                self.tile_begin = torch.empty_like(self.tile_expert)
                self.work_count = torch.empty(1, dtype=torch.int32, device=device)
                self.activation_device = torch.empty((pairs, width), dtype=torch.float16, device=device)
                self.expert_tensors = (self.counts_device, self.starts_device, self.members_device, self.tile_expert,
                                       self.tile_begin, self.work_count, self.activation_device)
            else:
                self.gate = torch.empty((ROWS, width), dtype=torch.float32, device=device)
                self.up = torch.empty_like(self.gate)
                self.activation = torch.empty((ROWS, width), dtype=torch.float16, device=device)
                self.value = torch.empty_like(self.gate)
                self.expert_input = torch.empty((ROWS, dims), dtype=torch.float16, device=device)
                self.expert_output = torch.empty((ROWS, dims), dtype=torch.float32, device=device)
                self.ids = torch.empty(ROWS, dtype=torch.int64, device=device)
                self.host_selection = torch.empty((selection_rows, slots), dtype=torch.int32, device="cpu")
                self.host_selected = self.host_selection.numpy().reshape(-1)
                self.host_ids = torch.empty(ROWS, dtype=torch.int64, device="cpu")
                self.host_id_values = self.host_ids.numpy()
                self.members = np.empty(pairs, dtype=np.int64)
                self.counts = np.empty(experts, dtype=np.int64)
                self.starts = np.empty(experts + 1, dtype=np.int64)
                self.cursor = np.empty(experts, dtype=np.int64)
                self.expert_tensors = (self.gate, self.up, self.activation, self.value,
                                       self.expert_input, self.expert_output, self.ids)

    def matrix_tensors(self):
        return self.matrix_storage

    def device_tensors(self):
        return *self.matrix_tensors(), *self.expert_tensors


def warm(workspace):
    """Load the selected kernel and fixed-shape GEMM before capacity publication.

    The caller publishes/owns Workspace before its fallible allocation and
    measures this preparation's allocator/external residency separately.
    These temporary source tensors never become a model or history mirror.
    """
    device = workspace.device
    codes = torch.zeros((1, 8), dtype=torch.uint8, device=device)
    scales = torch.zeros((1, 1), dtype=torch.uint8, device=device)
    factor = torch.ones(1, dtype=torch.float32, device=device)
    projection = PackedLinear(codes, scales, factor, workspace)
    plain = FloatLinear(torch.zeros((16, 16), dtype=torch.float16, device=device))
    x = torch.zeros((1, 16), dtype=torch.float16, device=device)
    # FP16 and FP32 outputs have distinct native WMMA/conversion entrypoints.
    # Both must be resident before the owning engine measures preparation.
    for f32 in (False, True):
        projection.matmul(x, f32=f32)
        plain.matmul(x, f32=f32)
    if workspace.expert_family == EXPERT_DEVICE and workspace.expert_tensors:
        packed = PackedLinear(torch.zeros((16, 8), dtype=torch.uint8, device=device),
                              torch.zeros((16, 1), dtype=torch.uint8, device=device), factor, workspace)
        plain = FloatLinear(torch.zeros((16, 16), dtype=torch.float16, device=device))
        bank = ExpertBank(((packed, packed, packed), (plain, plain, plain)), workspace, width=16, dims=16)
        selected = torch.arange(workspace.slots, dtype=torch.int32, device=device).remainder_(2)[None]
        x = torch.zeros((1, 16), dtype=torch.float16, device=device)
        for dtype in (torch.float16, torch.float32):
            bank.forward(x, selected, out=torch.empty((1, workspace.slots, 16), dtype=dtype, device=device))


class FloatLinear(_DenseFace):
    """Excluded FP16 storage, fixed M32 FP32 GEMMs and final output narrowing.

    The existing native dense32 adapter already supports this exact policy;
    output=FP32 forces the compute/reduction type before an optional FP16 copy.
    Reusing it keeps the legacy GGUF and Blackwell numerical paths unchanged.
    """

    def matmul(self, x, out=None, f32=False):
        from tensorfold.cuda.sm70.gguf_ops import dense32

        dtype = out.dtype if out is not None else torch.float32 if f32 else torch.float16
        if out is None:
            out = torch.empty((x.shape[0], self.n), dtype=dtype, device=x.device)
        if (x.ndim != 2 or x.shape[1] != self.k or x.dtype != torch.float16 or x.device != self.weight.device
                or out.shape != (x.shape[0], self.n) or out.device != x.device
                or dtype not in (torch.float16, torch.float32)):
            raise ValueError("SM70 excluded projection differs from its FP16 operand/FP32 accumulation profile")
        if torch._C._overlaps(out, x) or torch._C._overlaps(out, self.weight):
            raise ValueError("SM70 excluded projection output overlaps source operands")
        for start in range(0, x.shape[0], ROWS):
            dense32(x[start:start + ROWS], self.weight, out[start:start + ROWS], force_f32=True)
        return out


class PackedLinear:
    """Original low-even E2M1 bytes, original E4M3 block bytes and one FP32 factor.

    Scale arithmetic is `(code * block) * global` in FP32, followed by FP16 RNE.
    Fixed M32/N64/K256 products accumulate in FP32; K tiles add in increasing
    order in FP32 and only the final caller output narrows. This is W4A16 and
    makes no native NVFP4/W4A4 or same-quality claim against a source recipe.
    """

    kernel = "sm70_dense"
    implementation = NUMERICAL_PROFILE

    def __init__(self, codes, scales, global_scale, workspace):
        if (codes.dtype != torch.uint8 or codes.ndim != 2 or not codes.is_contiguous()
                or scales.dtype != torch.uint8 or not scales.is_contiguous()
                or global_scale.dtype != torch.float32 or global_scale.numel() != 1
                or not global_scale.is_contiguous()):
            raise ValueError("SM70 NVFP4 requires admitted row-major U8 codes/E4M3 bytes and one FP32 factor")
        self.n, self.k = int(codes.shape[0]), int(codes.shape[1]) * 2
        if (self.n <= 0 or self.k <= 0 or self.k % 16 or tuple(scales.shape) != (self.n, self.k // 16)
                or scales.device != codes.device or global_scale.device != codes.device
                or workspace.device != codes.device):
            raise ValueError("SM70 NVFP4 source geometry/device differs from its admitted scale/workspace layout")
        if any(torch._C._overlaps(source, scratch) for source in (codes, scales, global_scale)
               for scratch in workspace.device_tensors()):
            raise ValueError("SM70 NVFP4 immutable source overlaps mutable workspace storage")
        self.codes, self.scales, self.global_scale, self.workspace = codes, scales, global_scale, workspace

    def device_tensors(self):
        return self.codes, self.scales, self.global_scale, *self.workspace.device_tensors()

    def nbytes(self):
        return self.codes.numel() + self.scales.numel() + 4

    def matmul(self, x, *, out=None, f32=False):
        if (x.ndim != 2 or x.shape[1] != self.k or x.dtype != torch.float16
                or x.device != self.codes.device):
            raise ValueError("SM70 NVFP4 matmul requires FP16 CUDA rows matching the admitted projection")
        if any(torch._C._overlaps(x, tensor) for tensor in self.workspace.matrix_tensors()):
            raise ValueError("SM70 NVFP4 input overlaps borrowed mutable tile storage")
        if x.device.type != "cuda":
            raise ValueError("SM70 NVFP4 matmul requires FP16 CUDA rows matching the admitted projection")
        dtype = out.dtype if out is not None else torch.float32 if f32 else torch.float16
        if dtype not in (torch.float16, torch.float32):
            raise ValueError("SM70 NVFP4 projection output must be FP16 or FP32")
        if out is None:
            out = torch.empty((x.shape[0], self.n), dtype=dtype, device=x.device)
        if out.device != x.device or tuple(out.shape) != (x.shape[0], self.n):
            raise ValueError("SM70 NVFP4 output shape/device differs from its projection")
        operands = (self.codes, self.scales, self.global_scale, *self.workspace.matrix_tensors())
        if torch._C._overlaps(out, x) or any(torch._C._overlaps(out, t) for t in operands):
            raise ValueError("SM70 NVFP4 output overlaps retained operands or borrowed tile storage")
        ws = self.workspace
        if ws.kernel_family == LINEAR_WMMA:
            return nvfp4_ops.matmul(x, self.codes, self.scales, self.global_scale, out)
        for first in range(0, x.shape[0], ROWS):
            rows = min(ROWS, x.shape[0] - first)
            for offset in range(0, self.n, OUTPUTS):
                columns = min(OUTPUTS, self.n - offset)
                ws.accum.zero_()
                for start in range(0, self.k, INPUTS):
                    width = min(INPUTS, self.k - start)
                    nvfp4_ops.decode_tile(self.codes, self.scales, self.global_scale, ws.decoded,
                                         row=offset, column=start, rows=columns, columns=width)
                    ws.input.zero_()
                    ws.input[:rows, :width].copy_(x[first:first + rows, start:start + width])
                    torch.mm(ws.input, ws.decoded.T, out_dtype=torch.float32, out=ws.partial)
                    ws.accum.add_(ws.partial)
                out[first:first + rows, offset:offset + columns].copy_(ws.accum[:rows, :columns])
        return out

    def __call__(self, x, out=None):
        return self.matmul(x, out=out)

    def prefill(self, x, out=None):
        return self.matmul(x, out=out)

    def partials(self, x):
        return self.matmul(x, f32=True).unsqueeze(0)


class FloatEmbedding:
    """An explicitly excluded FP16 table on native Torch gather, without Triton."""

    def __init__(self, weight):
        if weight.ndim != 2 or weight.dtype != torch.float16 or not weight.is_contiguous():
            raise ValueError("SM70 floating embedding requires its admitted contiguous FP16 table")
        self.weight = weight

    def lookup(self, ids, *, copies=1, out=None):
        if (ids.dtype not in (torch.int32, torch.int64) or ids.device != self.weight.device
                or type(copies) is not int or copies < 1):
            raise ValueError("SM70 embedding IDs/copies differ from admitted integer token rows")
        values = torch.index_select(self.weight, 0, ids.reshape(-1))
        shape = (ids.numel(), copies * self.weight.shape[1])
        if out is None:
            out = torch.empty(shape, dtype=torch.float16, device=values.device)
        if out.shape != shape or out.dtype != torch.float16 or out.device != values.device:
            raise ValueError("SM70 embedding output differs from its FP16 residual-stream rows")
        if out.numel() and torch._C._overlaps(out, self.weight):
            raise ValueError("SM70 embedding output overlaps immutable retained source storage")
        out.view(ids.numel(), copies, self.weight.shape[1]).copy_(values[:, None, :])
        return out

    def device_tensors(self):
        return (self.weight,)

    def nbytes(self):
        return self.weight.numel() * self.weight.element_size()


class ExpertBank:
    """Bounded library reference provider for the declared SM70 weight recipe.

    A CPU selection snapshot is an explicit synchronizing reference policy,
    not a graph-safe or performance-qualified kernel. Selected experts share
    one fixed32-row activation/tile workspace; no expert bank is expanded.
    """

    kernel = "sm70_dense"
    capturable = False

    def __init__(self, parts, workspace, *, width, dims):
        self.parts, self.workspace = tuple(parts), workspace
        self.count, self.width, self.dims = len(self.parts), width, dims
        if not self.parts or any(g.k != dims or g.n != u.n or u.k != dims
                                 or down.n != dims or down.k != g.n or g.n > width
                                 for g, u, down in self.parts):
            raise ValueError("SM70 NVFP4 routed/shared expert projection geometry disagrees")
        # Parts and their storage bindings stay immutable for this bank's
        # lifetime. Descriptors borrow raw pointers, so establish device and
        # source-storage authority before publishing any descriptor. Whole
        # storage intervals also cover shared global-factor bank views.
        intervals = []
        for group in self.parts:
            for part in group:
                if isinstance(part, PackedLinear):
                    sources = (part.codes, part.scales, part.global_scale)
                elif isinstance(part, FloatLinear):
                    sources = (part.weight,)
                else:
                    raise ValueError("unimplemented weight face in SM70 expert bank")
                if any(t.device != workspace.device for t in sources):
                    raise ValueError("SM70 expert source device differs from its immutable workspace")
                if isinstance(part, PackedLinear) and part.workspace is not workspace:
                    raise ValueError("SM70 expert parts require their single immutable shared workspace")
                for tensor in sources:
                    storage = tensor.untyped_storage()
                    if storage.nbytes():
                        intervals.append((storage.data_ptr(), storage.data_ptr() + storage.nbytes()))
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        self._source_starts = tuple(start for start, _ in merged)
        self._source_ends = tuple(end for _, end in merged)
        if any(self._overlaps_sources(tensor) for tensor in workspace.device_tensors()):
            raise ValueError("SM70 expert immutable source overlaps mutable workspace storage")
        self.descriptor = None
        if workspace.expert_family == EXPERT_DEVICE:
            if self.count > workspace.experts or width > workspace.width:
                raise ValueError("device expert bank exceeds its immutable prepared workspace")
            rows = []
            for group in self.parts:
                record = []
                for part in group:
                    if isinstance(part, PackedLinear):
                        data, scale, factor = part.codes, part.scales, part.global_scale
                        kind = 1
                    elif isinstance(part, FloatLinear):
                        data, scale, factor, kind = part.weight, None, None, 0
                    else:
                        raise ValueError("unimplemented weight face in device expert descriptor")
                    values = {"DATA": data.data_ptr(), "SCALES": 0 if scale is None else scale.data_ptr(),
                              "GLOBAL": 0 if factor is None else factor.data_ptr(), "ROWS": part.n,
                              "COLUMNS": part.k, "KIND": kind, "DATA_BYTES": data.numel() * data.element_size(),
                              "SCALE_ELEMENTS": 0 if scale is None else scale.numel()}
                    record.append([values[field] for field in EXPERT_FIELDS])
                rows.append(record)
            self.descriptor = torch.tensor(rows, dtype=torch.int64, device=workspace.device)

    def _overlaps_sources(self, tensor):
        if not tensor.numel():
            return False
        start = tensor.data_ptr()
        end = start + tensor.numel() * tensor.element_size()
        index = bisect_right(self._source_ends, start)
        return index < len(self._source_starts) and self._source_starts[index] < end

    def device_tensors(self):
        return (*(() if self.descriptor is None else (self.descriptor,)), *self.workspace.device_tensors(),
                *(tensor for group in self.parts for part in group for tensor in part.device_tensors()))

    def forward(self, x, picks, out=None):
        if (x.ndim != 2 or x.shape[1] != self.dims or x.dtype != torch.float16
                or picks.ndim != 2 or picks.shape[0] != x.shape[0]
                or picks.dtype != torch.int32 or picks.device != x.device or x.device != self.workspace.device
                or picks.shape[0] > self.workspace.selection_rows or picks.shape[1] != self.workspace.slots):
            raise ValueError("SM70 NVFP4 expert input/selection differs from its admitted family")
        if any(torch._C._overlaps(operand, scratch) for operand in (x, picks)
               for scratch in self.workspace.device_tensors()):
            raise ValueError("SM70 expert input/selection overlaps borrowed mutable workspace storage")
        shape = (*picks.shape, self.dims)
        if out is None:
            out = torch.empty(shape, dtype=torch.float32, device=x.device)
        if (tuple(out.shape) != shape or out.device != x.device or out.dtype not in (torch.float16, torch.float32)
                or not out.is_contiguous()):
            raise ValueError("SM70 NVFP4 expert output differs from admitted slots")
        if self._overlaps_sources(out):
            raise ValueError("SM70 expert output overlaps immutable retained source storage")
        borrowed = (x, picks, *self.workspace.device_tensors())
        if self.descriptor is not None:
            borrowed = (*borrowed, self.descriptor)
        if any(torch._C._overlaps(out, tensor) for tensor in borrowed):
            raise ValueError("SM70 expert output overlaps input or borrowed execution storage")
        if self.workspace.expert_family == EXPERT_DEVICE:
            return nvfp4_ops.experts(x, picks, self.descriptor, self.workspace, out, width=self.width)
        # Every selection is validated before any projection can borrow packed
        # storage. One finite snapshot replaces per-pair GPU scalar readbacks.
        ws = self.workspace
        ws.host_selection[:picks.shape[0]].copy_(picks)
        count = picks.numel()
        ws.counts.fill(0)
        for index in range(count):
            expert = int(ws.host_selected[index])
            if not 0 <= expert < self.count:
                raise ValueError("SM70 NVFP4 selected an expert outside the admitted bank")
            ws.counts[expert] += 1
        ws.starts[0] = 0
        np.cumsum(ws.counts, out=ws.starts[1:])
        ws.cursor[:] = ws.starts[:-1]
        for index in range(count):
            expert = int(ws.host_selected[index])
            ws.members[ws.cursor[expert]] = index
            ws.cursor[expert] += 1
        for expert in range(self.count):
            gate, up, down = self.parts[expert]
            width = gate.n
            begin, end = int(ws.starts[expert]), int(ws.starts[expert + 1])
            for start in range(begin, end, ROWS):
                chunk = ws.members[start:min(end, start + ROWS)]
                ws.host_id_values.fill(0)
                for index, pair in enumerate(chunk):
                    ws.host_id_values[index] = int(pair) // picks.shape[1]
                ws.ids.copy_(ws.host_ids)
                torch.index_select(x, 0, ws.ids, out=ws.expert_input)
                gate.matmul(ws.expert_input, out=ws.gate[:, :width])
                up.matmul(ws.expert_input, out=ws.up[:, :width])
                value = ws.value[:, :width]
                value.copy_(ws.gate[:, :width])
                torch.nn.functional.silu(value, inplace=True)
                value.mul_(ws.up[:, :width])
                ws.activation[:, :width].copy_(value)
                down.matmul(ws.activation[:, :width], out=ws.expert_output, f32=True)
                for index, pair in enumerate(chunk):
                    row, slot = divmod(int(pair), picks.shape[1])
                    out[row, slot].copy_(ws.expert_output[index])
        return out

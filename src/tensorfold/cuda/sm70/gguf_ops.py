"""Volta-safe operators for original GGML bytes, independent of vLLM registrations."""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess

import torch


@lru_cache(maxsize=1)
def extension():
    from torch.utils import cpp_extension
    from tensorfold.cuda import build

    if torch.cuda.get_device_capability() != (7, 0) or "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError("mixed GGUF FP16 execution requires a Volta-enabled PyTorch build and SM70 device")
    from .foundation_headers import selected_headers

    selected_headers()  # Validate injected argument channels before compiler metadata queries too.
    build._toolkit()
    if cpp_extension.CUDA_HOME is None:
        raise RuntimeError("mixed GGUF execution requires the native CUDA12 compiler")
    compiler = subprocess.check_output([str(Path(cpp_extension.CUDA_HOME) / "bin/nvcc"), "--version"])
    if b"release 12." not in compiler:
        raise RuntimeError("SM70 GGUF operators require CUDA12.x")
    here = Path(__file__).parent
    sources = [here / "gguf_ops.cpp", here / "gguf_ops.cu"]
    flags = ["-O3", "--fmad=false"]
    host_flags = ["-O3"]
    host_compiler = subprocess.check_output([os.environ.get("CXX", "c++"), "--version"])
    key = hashlib.sha256(compiler + b"\0" + host_compiler
                         + repr((flags, host_flags, torch.__version__, torch.version.cuda,
                                 torch._C._GLIBCXX_USE_CXX11_ABI)).encode())
    for path in sources:
        key.update(path.read_bytes())
    contracts = here.parent / "kernels/tensor_contracts.cuh"
    tables = here / "gguf_decode_tables.cuh"
    for header in (contracts, tables):
        key.update(header.read_bytes())
    from .source_cache import stage_sources

    staged = stage_sources(sources, headers=[contracts, tables])
    return build.load("tf_gguf_" + key.hexdigest()[:20], staged, need=(7, 0),
                      extra_cflags=host_flags, extra_cuda_cflags=flags)


def decode(packed, ggml_type, columns, *, column_offset=0, dtype=torch.float16, ids=None):
    """Decode only an explicit row/column view, preserving original quantization blocks.

    Optional row IDs are trusted products of host-validated token/row admission.
    The native kernel bounds-checks them to prevent out-of-range device reads.
    """
    if dtype not in (torch.float16, torch.float32):
        raise ValueError("GGUF decode output must be FP16 or FP32")
    return extension().decode(packed, int(ggml_type), int(column_offset), int(columns),
                              torch.empty(0, dtype=torch.int64, device=packed.device) if ids is None else ids,
                              ids is not None, dtype == torch.float32)


def experts(x, picks, gate, up, down, shared, out, *, split_k=1, blockwise=False, projection_warps=1):
    """Small-row packed expert provider; coefficients and SwiGLU round to FP16.

    FP32 dot reductions have a fixed warp order independent of the batch shape.
    This provider requires separate model-quality qualification against the
    reference cuBLAS route before its execution identity can be selected.
    """
    if split_k not in (1, 4):
        raise ValueError("packed expert K split must be1 or4")
    if blockwise and split_k != 1:
        raise ValueError("blockwise packed experts retain the unsplit v1 reduction")
    if projection_warps not in (1, 2) or projection_warps == 2 and not blockwise:
        raise ValueError("two projection warps require the unsplit blockwise expert provider")
    module = extension()
    method = (module.experts_blocks2w if projection_warps == 2 else module.experts_blocks if blockwise
              else module.experts if split_k == 1 else module.experts_splitk4)
    method(x, picks, gate.data, gate.ggml_type, gate.column_offset,
           up.data, up.ggml_type, up.column_offset,
           down.data, down.ggml_type, down.column_offset,
           *(part.weight for part in shared), out)
    return out


def dense32(x, weight, out, *, force_f32=False):
    """One native call around the unchanged fixed-M32 ATen matrix product."""
    extension().dense32(x, weight, out, force_f32)
    return out


def experts_exact(x, picks, gate, up, down, shared, out):
    """Original BMM arithmetic for the qualified D2560/K80 or K160 packed target route.

    The owning family evaluates its shared expert with the original Dense
    projections. Old packed operator entrypoints remain independent controls.
    """
    extension().experts_exact(x, picks, gate.data, gate.ggml_type, gate.column_offset,
        up.data, up.ggml_type, up.column_offset, down.data, down.ggml_type, down.column_offset,
        *(part.weight for part in shared), out)
    return out


def concat32(x, weights, out):
    """Share staging while retaining each original fixed-M32 ATen product."""
    extension().concat32(x, tuple(weights), out)
    return out


def dense_wmma(x, weight, out):
    """Optional fixed-partition Volta WMMA provider; separate math qualification required."""
    extension().dense_wmma(x, weight, out)
    return out


def owned_workspace_bytes(rows, slots, width=160):
    """Only reusable native route/activation scratch; shared Dense scratch is separate."""
    if (any(type(v) is not int or v <= 0 for v in (rows, slots))
            or type(width) is not int or width != 160 or rows * slots > 65535):
        raise ValueError("owned ETP4 scratch requires bounded positive rows/slots and width160")
    return rows * slots * (8 + 2 * width)


def grouped_workspace_words(pairs, resident_experts):
    """Fixed GPU-only route counters/maps/tiles; no expert-by-pair expansion."""
    if (type(pairs) is not int or not 0 <= pairs <= 65535
            or type(resident_experts) is not int or not 0 <= resident_experts <= 512):
        raise ValueError("grouped SC117 routes require bounded pairs and resident experts")
    tiles = min(pairs, (pairs + 15) // 16 + max(0, resident_experts - 1))
    return 2 * pairs + 2 * resident_experts + 3 * tiles + 2


@dataclass(frozen=True)
class _GroupedGateUp:
    """Explicit unselected SIMT experiment; never an automatic shape dispatch."""
    rows: int

    def __post_init__(self):
        if type(self.rows) is not int or self.rows not in (2, 4):
            raise ValueError("private grouped gate/up requires exactly 2 or 4 rows")


def experts_owned(x, picks, gate, up, down, *, first, experts, width, shared, local, activation, out,
                  grouping=None, _gate_up=None):
    """Explicit unselected ETP4 candidate, with caller-owned fixed native scratch.

    Foreign global slots become zero; invalid global IDs become NaN without any
    bank read. NaN is a fatal diagnostic result, not recoverable input repair.
    Shared results come from the original independent TP8 Dense path; publication
    preserves caller FP16 or FP32 slot storage with one explicit FP16 rounding.
    """
    if x.ndim != 2 or picks.ndim != 2 or picks.shape[0] != x.shape[0]:
        raise ValueError("owned inputs and picks require matching row matrices")
    owned_workspace_bytes(x.shape[0], picks.shape[1], width)
    if any(type(v) is not int for v in (first, experts, width)) or not 0 <= first < experts:
        raise ValueError("owned expert range requires native integer global placement")
    if grouping is not None and (experts != 512 or x.shape[1] != 2560):
        raise ValueError("grouped owned down160 requires SC117 hidden/global expert geometry")
    if _gate_up is not None and (type(_gate_up) is not _GroupedGateUp or grouping is None
                                or experts != 512 or x.shape[1] != 2560):
        raise ValueError("private grouped gate/up requires its typed selector and prepared SC117 grouping")
    native = extension()
    execute = native.experts_owned if grouping is None else native.experts_owned_grouped
    if _gate_up is not None:
        execute = native._experts_owned_grouped_gate_up
    extra = () if grouping is None else (grouping,)
    if _gate_up is not None:
        extra += (_gate_up.rows,)
    execute(x, picks, gate.data, gate.ggml_type, gate.column_offset,
        up.data, up.ggml_type, up.column_offset, down.data, down.ggml_type, down.column_offset,
        width, first, experts, shared, local, activation, out, *extra)
    return out


@dataclass(frozen=True)
class _OwnedDownFromActivation:
    """Cold-resolved down/publication callable; never computes routed GU.

    Activation is exclusively prepared GU scratch, with slot10's first Half
    reserved as immutable +0/poison row status. It is a read operand here.
    The caller computes the independent original shared output after GU init,
    retains every input/scratch/output through current-stream completion and
    graph replay, and serializes access. Failed work is not reusable before
    positive retirement; this helper neither retries nor rolls back GPU writes.
    """
    module: object
    device: object

    def run(self, x, picks, gate, up, down, *, first, shared, local, activation, out, grouping):
        operands = (x, picks, gate.data, up.data, down.data, shared, local, activation, out, grouping)
        if any(type(t) is not torch.Tensor for t in operands):
            raise TypeError("supplied GU requires plain Tensor operands")
        if x.device != self.device:
            raise ValueError("supplied GU device differs from cold preparation")
        if type(first) is not int or first not in (0, 256):
            raise ValueError("supplied GU requires E256 groupfirst0 or256")
        self.module.experts_owned_from_activation(x, picks,
            gate.data, gate.ggml_type, gate.column_offset,
            up.data, up.ggml_type, up.column_offset,
            down.data, down.ggml_type, down.column_offset,
            160, first, 512, shared, local, activation, out, grouping)
        # Three writable arenas; activation/weights are read-only in this call.
        torch.autograd.graph.increment_version((local, grouping, out))
        return out


def prepare_owned_from_activation(device):
    """Resolve all current grouped-down/publication functions before admission."""
    device = torch.device(device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("explicit indexed CUDA device required")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("supplied GU preparation requires an uncaptured stream")
        module = extension()
        module.prepare_owned_from_activation()
    return _OwnedDownFromActivation(module, device)


@dataclass(frozen=True)
class _OwnedGateUp:
    """Exact incumbent Grouped4 GU only, with caller-funded stable arenas."""
    module: object
    device: object

    def run(self, x, picks, gate, up, *, first, local, activation, grouping):
        operands = (x, picks, gate.data, up.data, local, activation, grouping)
        if any(type(t) is not torch.Tensor for t in operands):
            raise TypeError("owned GU requires plain Tensor operands")
        if x.device != self.device:
            raise ValueError("owned GU prepared device differs")
        if type(first) is not int or first not in (0, 256):
            raise ValueError("owned GU requires E256 first0|256")
        self.module.experts_owned_gate_up(x, picks, gate.data, gate.ggml_type, gate.column_offset,
                                         up.data, up.ggml_type, up.column_offset,
                                         first, local, activation, grouping)
        torch.autograd.graph.increment_version((local, activation, grouping))
        return activation


def prepare_owned_gate_up(device):
    """Resolve GU-only incumbent before admission/capture; no down work."""
    device = torch.device(device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("explicit indexed CUDA device required")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("owned GU preparation requires an uncaptured stream")
        module = extension()
        module.prepare_owned_gate_up()
    return _OwnedGateUp(module, device)

"""Volta-safe operators for original GGML bytes, independent of vLLM registrations."""

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
    projections; this routed core initializes shared slots to positive zero.
    Old packed operator entrypoints remain independent controls.
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

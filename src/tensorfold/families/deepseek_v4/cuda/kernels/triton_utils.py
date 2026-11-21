"""Triton entry points for the vendored DeepSeek-V4.1 kernels (replaces vllm.triton_utils).

Provenance: replaces 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f vllm/triton_utils/__init__.py
(blob not vendored; TensorFold-owned). Apache-2.0 SPDX of the users is kept in each module.

``INTERPRET`` mirrors Triton's own ``TRITON_INTERPRET`` switch, read once at import exactly as Triton 3.2's
``triton.jit`` does. Compiled (GPU) runs use the real libdevice. The Triton 3.2 interpreter has no libdevice, so
there ``tldevice.exp`` is ``tl.exp`` (NumPy FP32 exp): a CPU-test capability only, never a compiled path.
``on_triton_device`` is ``tensor.is_cuda`` in compiled runs; under the interpreter it also admits CPU tensors so
the wrappers that choose between a Triton kernel and a CPU torch path exercise the kernel on CPU.
``jit`` is ``triton.jit`` whose launches (``kernel[grid](...)`` and ``warmup``) also pass the process-lifetime
``tensorfold.cuda.sm70.triton_options.COMPILER_OPTIONS`` (the explicit SM70 Triton 3.2 profile's ``ptx_version=83``;
Triton 3.2 cannot derive a PTX version from CUDA 12.9's ptxas). Explicit launch options take precedence. With no
profile selected, or under the interpreter, it is exactly ``triton.jit``. Options do not change kernel arithmetic.
"""

import os

import triton
import triton.language as tl

from tensorfold.cuda.sm70.triton_options import COMPILER_OPTIONS

INTERPRET = os.environ.get("TRITON_INTERPRET", "0") == "1"

if INTERPRET:

    @triton.jit
    def _interpreter_exp(x):
        return tl.exp(x)

    class _InterpreterDevice:
        exp = _interpreter_exp

    tldevice = _InterpreterDevice()
else:
    import triton.language.extra.libdevice as tldevice


def jit(fn=None, **options):
    def decorate(function):
        kernel = triton.jit(function, **options)
        if COMPILER_OPTIONS and not INTERPRET:
            run = kernel.run

            def profiled_run(*args, **kwargs):
                return run(*args, **{**COMPILER_OPTIONS, **kwargs})

            kernel.run = profiled_run  # JITFunction.__getitem__ and warmup dispatch through self.run
        return kernel

    return decorate if fn is None else decorate(fn)


def on_triton_device(tensor):
    return tensor.is_cuda or INTERPRET


__all__ = ["INTERPRET", "jit", "on_triton_device", "tl", "tldevice", "triton"]

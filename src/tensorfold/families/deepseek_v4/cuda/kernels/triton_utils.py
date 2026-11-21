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
Triton 3.2 cannot derive a PTX version from CUDA 12.9's ptxas). Explicit launch options take precedence. Under
Triton 3.2.0 a launch of an already compiled specialization takes ``_cached_launch`` (Triton's own cache-hit steps
without the per-launch backend construction: the same compiled kernel and arguments). Under the interpreter it is
exactly ``triton.jit``. Options do not change kernel arithmetic.
``fma_dot`` is the vendored kernels' ``tl.dot`` as the friend's Triton 3.6 runs it on SM70: Triton >= 3.3 has no
MMAv1 (Volta) lowering and computes an FP16 dot as FP32 FMAs, K in order. Triton 3.2 would pick MMAv1, whose
results it cannot reduce on SM70 (tl.sum/tl.max of any MMAv1 dot result aborts: "Failed to compute parent layout
for slice layout" / ReduceOpConversion "Unexpected srcLayout", m1i/m1j and the CPU compile probe). FP16 operands
are exact in FP32, so the FP32 IEEE (FMA) dot is the same arithmetic.
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
        if not INTERPRET and (COMPILER_OPTIONS or triton.__version__ == _LAUNCH_VERSION):
            kernel.run = _cached_launch(kernel, kernel.run)  # __getitem__ and warmup dispatch through self.run
        return kernel

    return decorate if fn is None else decorate(fn)


_LAUNCH_VERSION = "3.2.0"     # the JITFunction.run whose cache-hit steps _cached_launch restates


def _cached_launch(kernel, run):
    """``kernel.run`` for launches of an already compiled specialization: Triton 3.2's ``JITFunction.run`` steps
    for a cache hit (the same binder, cache key, kernel cache, global-value check, grid canonicalization, stream and
    ``CompiledKernel.run`` call), without the per-launch target / backend construction (``make_backend``,
    ``get_current_target``: device capability and properties) that only a compile needs. A miss, a warmup, a pre-run
    hook or an unbuilt binder go through Triton's own ``run`` (which compiles and caches). The launched kernel and its
    arguments are identical. Pinned to Triton 3.2.0's ``JITFunction.run`` (ds41-short K36: about 0.5k Triton
    launches per DS41 prompt forward)."""
    from triton.runtime.driver import driver

    devices = []      # the active driver's (current device, current stream), resolved at the first launch: importing
    not_present = object()   # the kernels must not need a GPU driver (CPU tests and tools import them)
    cached = triton.__version__ == _LAUNCH_VERSION   # another Triton: its own run (plus the profile options)

    def launch(*args, grid, warmup, **kwargs):
        if COMPILER_OPTIONS:
            kwargs = {**COMPILER_OPTIONS, **kwargs}
        if not cached or warmup or kernel.binder is None or kernel.pre_run_hooks:
            return run(*args, grid=grid, warmup=warmup, **kwargs)
        kwargs["debug"] = kwargs.get("debug", False) or os.environ.get("TRITON_DEBUG", "0") == "1"
        if not devices:
            devices.extend((driver.active.get_current_device, driver.active.get_current_stream))
        current_device, current_stream = devices
        device = current_device()
        bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = kernel.binder(*args, **kwargs)
        compiled = kernel.cache[device].get(''.join(sig_and_spec) + str((constexpr_vals, excess_kwargs)))
        if compiled is None:
            return run(*args, grid=grid, warmup=False, **kwargs)
        for (name, _), (val, globals_dict) in kernel.used_global_vals.items():
            if (new := globals_dict.get(name, not_present)) != val:
                raise RuntimeError(
                    f"Global variable {name} has changed since we compiled this kernel, from {val} to {new}")
        if callable(grid):
            grid = grid(bound_args)
        size = len(grid)
        stream = current_stream(device)
        compiled.run(grid[0], grid[1] if size > 1 else 1, grid[2] if size > 2 else 1, stream, compiled.function,
                     compiled.packed_metadata, compiled.launch_metadata(grid, stream, *non_constexpr_vals),
                     kernel.CompiledKernel.launch_enter_hook, kernel.CompiledKernel.launch_exit_hook,
                     *non_constexpr_vals)
        return compiled

    return launch


@triton.jit
def fma_dot(a, b):
    return tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="ieee")


@triton.jit
def fma_dot_acc(a, b, acc):
    return tl.dot(a.to(tl.float32), b.to(tl.float32), acc, input_precision="ieee")


def on_triton_device(tensor):
    return tensor.is_cuda or INTERPRET


__all__ = ["INTERPRET", "fma_dot", "fma_dot_acc", "jit", "on_triton_device", "tl", "tldevice", "triton"]

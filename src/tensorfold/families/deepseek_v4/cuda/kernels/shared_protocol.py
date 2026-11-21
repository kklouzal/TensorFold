# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side shipping check ``_fail_if`` for the vendored V4.1 kernels.

Provenance: 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f
vllm/model_executor/layers/quantization/gguf_shared_protocol.py (blob 18ca5aec7011254f3e758fe522346c9d22dab7d2).
TensorFold changes:
- ``@triton.jit`` -> ``triton_utils.jit`` (launches add the SM70 Triton 3.2 profile options; arithmetic unchanged).
- Only ``_fail_if`` is vendored; the native coordinator helpers (_sys_load_acquire, _device_time_ns, plan/commit
  protocol) belong to vLLM's expert coordinator and are not used here.
- ``vllm.triton_utils`` -> ``triton_utils``.
- Triton-interpreter (CPU test) form: inline PTX is unsupported there, so the same predicate raises a Python
  ``RuntimeError`` (surfaced by Triton as ``InterpreterError``). Compiled runs keep the verbatim PTX trap.
"""

from .triton_utils import INTERPRET, jit, tl

if INTERPRET:

    @jit
    def _fail_if(failed):
        if failed != 0:
            raise RuntimeError("DeepSeek-V4.1 device check failed (_fail_if)")

else:

    @jit
    def _fail_if(failed):
        if failed != 0:
            tl.inline_asm_elementwise(
                "trap; mov.u32 $0, $1;",
                constraints="=r,r",
                args=[failed.to(tl.int32)],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )

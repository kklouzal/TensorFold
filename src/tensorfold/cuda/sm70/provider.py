"""Small FP16 reference provider for the Flash Next SM70 bring-up.

No existing BF16 provider or public serving admission is changed. Projections
and attention deliberately use FP32 arithmetic followed by one FP16 rounding;
this reference policy is independent of the modern providers' rounding order.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import subprocess

SM70 = (7, 0)


@lru_cache(maxsize=1)
def _native():
    import torch
    from torch.utils import cpp_extension
    from tensorfold.cuda import build

    if torch.cuda.get_device_capability() != SM70:
        raise RuntimeError("the SM70 provider requires a Volta compute-capability 7.0 device")
    if "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError("this PyTorch wheel has no sm_70 kernels; select a Volta-enabled foundation")
    if not torch.version.cuda or not torch.version.cuda.startswith("12."):
        raise RuntimeError("the SM70 provider requires a CUDA 12.x PyTorch foundation")
    build._toolkit()  # Retain the existing managed-foundation validation and toolkit selection.
    if cpp_extension.CUDA_HOME is None:
        raise RuntimeError("the SM70 provider requires a native CUDA 12.x compiler")
    compiler = subprocess.check_output([str(Path(cpp_extension.CUDA_HOME) / "bin/nvcc"), "--version"])
    if b"release 12." not in compiler:
        raise RuntimeError("SM70 native compilation requires CUDA 12.x")
    host_compiler = subprocess.check_output([os.environ.get("CXX", "c++"), "--version"])
    root = Path(__file__).parent
    sources = [root / "bindings.cpp", root / "primitives.cu"]
    header = root.parent / "kernels" / "tensor_contracts.cuh"
    flags = ["-O3", "--fmad=false"]
    host_flags = ["-O3"]
    identity = hashlib.sha256()
    for source in [*sources, header]:
        identity.update(source.name.encode() + b"\0" + source.read_bytes())
    identity.update(repr((flags, host_flags, torch.__version__, torch.version.cuda,
                          torch._C._GLIBCXX_USE_CXX11_ABI)).encode())
    identity.update(compiler + b"\0" + host_compiler)
    from .source_cache import stage_sources

    return build.load("tf_sm70_" + identity.hexdigest()[:20], stage_sources(sources, headers=[header]), need=SM70,
                      extra_cuda_cflags=flags, extra_cflags=host_flags)


def _half(*tensors):
    import torch

    if not tensors:
        return
    device = tensors[0].device
    for tensor in tensors:
        if tensor.device.type != "cuda" or tensor.device != device or tensor.dtype != torch.float16:
            raise ValueError("SM70 operands must be FP16 on the same CUDA device")


@dataclass(frozen=True)
class SM70Provider:
    """Component API, not a full-model engine; recurrence state is always FP32."""

    activation_dtype: str = "float16"
    accumulator_dtype: str = "float32"
    state_dtype: str = "float32"

    def __post_init__(self):
        if (self.activation_dtype, self.accumulator_dtype, self.state_dtype) != ("float16", "float32", "float32"):
            raise ValueError("SM70 precision policy is FP16 activations with FP32 accumulation/state")

    def prepare(self):
        """Compile and import before graph capture or model execution."""
        return _native()

    def rms_norm(self, x, weight, eps=1e-6):
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("RMSNorm epsilon must be finite and positive")
        return _native().rms_norm(x, weight, eps)

    def rope(self, x, inv_freq, positions):
        """Text rotary: pair the two halves of the rotary prefix; preserve its tail."""
        return _native().rope(x, inv_freq, positions)

    def linear(self, x, weight):
        _half(x, weight)
        self.prepare()
        if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
            raise ValueError("linear expects [rows, input] and [output, input]")
        return (x.float() @ weight.float().T).half()

    def attention(self, q, k, v, *, start=0, scale=None):
        """Bounded dense causal GQA reference; sparse QSA selection is a later adapter."""
        import torch

        _half(q, k, v)
        self.prepare()
        if (q.ndim != 3 or k.ndim != 3 or v.shape != k.shape or q.shape[2] != k.shape[2]
                or k.shape[1] <= 0 or q.shape[1] <= 0 or q.shape[2] <= 0 or q.shape[1] % k.shape[1]):
            raise ValueError("attention expects Q[R,H,D], K/V[T,HK,D] with HK dividing H")
        if type(start) is not int or start < 0 or start + q.shape[0] != k.shape[0]:
            raise ValueError("attention history must end at start + query rows")
        factor = q.shape[-1] ** -0.5 if scale is None else float(scale)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("attention scale must be finite and positive")
        heads = torch.arange(q.shape[1], device=q.device) // (q.shape[1] // k.shape[1])
        scores = torch.einsum("rhd,thd->hrt", q.float(), k[:, heads].float()) * factor
        mask = torch.arange(k.shape[0], device=q.device)[None, :] > (
            start + torch.arange(q.shape[0], device=q.device)[:, None])
        scores.masked_fill_(mask[None], -torch.inf)
        return torch.einsum("hrt,thd->rhd", scores.softmax(-1), v[:, heads].float()).half()

    def delta_chain(self, q, k, v, decay, beta, state):
        """Unnormalized delta-rule core; caller owns conv, Q/K norms and output gate.

        State [HV,DV,DK] is read-only; return disjoint final state. Supports the
        actual TP8 2/6-head geometry, without altering the existing BF16 GDN ABI.
        """
        import torch

        _half(q, k, v)
        self.prepare()
        if (q.ndim != 3 or q.shape != k.shape or v.ndim != 3 or v.shape[0] != q.shape[0]
                or q.shape[1] <= 0 or v.shape[1] <= 0 or q.shape[2] <= 0 or v.shape[2] <= 0
                or v.shape[1] % q.shape[1]):
            raise ValueError("delta rows require Q/K[R,HK,DK], V[R,HV,DV], HK dividing HV")
        if (state.shape != (v.shape[1], v.shape[2], q.shape[2])
                or decay.shape != v.shape[:2] or beta.shape != decay.shape):
            raise ValueError("delta state or gate geometry does not match its rows")
        if any(t.dtype != torch.float32 or t.device != q.device for t in (state, decay, beta)):
            raise ValueError("delta state, decay and beta must be FP32 on the rows' device")
        current = state.clone()
        heads = torch.arange(v.shape[1], device=q.device) // (v.shape[1] // q.shape[1])
        outputs = []
        for row in range(q.shape[0]):
            key, query = k[row, heads].float(), q[row, heads].float()
            current = current * decay[row, :, None, None]
            residual = (v[row].float() - torch.einsum("hvk,hk->hv", current, key)) * beta[row, :, None]
            current = current + residual[:, :, None] * key[:, None, :]
            outputs.append(torch.einsum("hvk,hk->hv", current, query).half())
        return (torch.stack(outputs) if outputs else v.new_empty(v.shape)), current

"""DeepSeek-V4.1 DSpark routed experts on SM70: MXFP4 banks, exact weight decode (``mxfp4_ds41.cu``).

Banks are one rank's original checkpoint bytes: gate/up uint8 [E, N, K/2] (two E2M1 codes per byte, low nibble =
even input) with E8M0 scales uint8 [E, N, K/32]; down uint8 [E, H, Kd/2] with scales [E, H, Kd/32]. Route ids are
int32 local expert indices, -1 for a route another rank owns (exactly zero contribution, no reads).

Modes (``q8``):
- False (default): the reference contract of 1Cat-vLLM's TurboMind MXFP4 path (``mxfp4_sm70_moe.py`` @4d57473d):
  FP16 activations x exact weights, FP32 accumulation. Expected to differ from TurboMind only by FP32 summation
  order, with one unverified caveat: the release's expert scales span 2^[-16, 0] (``dspark_weights.Audit``) while
  TurboMind decodes scales into an FP16 exponent field (2^[-14, 15] normal; ``e8m0_to_half2`` /
  ``AdjustUe8m0ScaleForHalf``); unless its global-factor rebasing covers them, its 91,072 blocks at 2^-15/2^-16
  (of 467M) would differ there. Here every block is exact.
- True: 1Cat dp4a style -- activations quantized per 32 to int8 with an FP32 scale; exact int32 dp4a; FP32 block
  accumulation. An explicit activation quantization (NOT the reference's FP16 activations).
Rounding boundaries: FP16 gate/up; vLLM ``silu_and_mul_with_clamp`` SwiGLU (limit 10); FP16 per-route down; FP32
route-weighted sum in route order rounded once to FP16 -- the stages of upstream ``execute_fp4``.
Split ETP can defer the two routed casts with ``experts_partials`` and ``finish_partials``. The collective
sums unweighted FP32 down planes per route and the separate shared-down plane before either is finalized.

All operands CUDA, contiguous, one device. ``out``/scratch tensors are caller-owned and fully written; passing them
keeps a call graph-capturable (no allocation). The caller retains operands until current-stream completion.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
Q8_BYTES = 36


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .source_cache import stage_sources

    if torch.cuda.get_device_capability() != (7, 0) or "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError("DeepSeek-V4.1 MXFP4 operators require a Volta-enabled PyTorch build and SM70 device")
    here = Path(__file__).parent
    sources = [here / "mxfp4_ds41.cpp", here / "mxfp4_ds41.cu"]
    flags = ["-O3"]
    identity = hashlib.sha256(repr(flags).encode())
    for path in sources:
        identity.update(path.name.encode() + b"\0" + path.read_bytes())
    staged = stage_sources(sources)
    return build.load("tf_mxfp4_ds41_" + identity.hexdigest()[:20], staged, need=(7, 0),
                      extra_cflags=["-O3"], extra_cuda_cflags=flags)


def prepare():
    """Build/load the extension outside capture (otherwise the first call does)."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare the DeepSeek-V4.1 MXFP4 extension before graph capture")
    return extension()


class Workspace:
    """Preallocated scratch for M tokens x top routes over one bank geometry (graph-capturable calls)."""

    def __init__(self, m, top, hidden, rows, device):
        routes = m * top
        self.m, self.top = m, top
        self.xq = torch.empty(m, hidden // 32, Q8_BYTES, dtype=torch.uint8, device=device)
        self.gate_up = torch.empty(routes, 2, rows, dtype=torch.float16, device=device)
        self.inter = torch.empty(routes, rows, dtype=torch.float16, device=device)
        self.iq = torch.empty(routes, rows // 32, Q8_BYTES, dtype=torch.uint8, device=device)
        self.out = torch.empty(m, hidden, dtype=torch.float16, device=device)


def experts(x, ids, probs, gate, gate_s, up, up_s, down, down_s, *, q8=False, ws=None):
    """Routed experts: FP16 x [M, K], int32 ids / FP32 probs [M, top] -> FP16 [M, H] (this rank's partial)."""
    m, top = ids.shape
    if (x.dtype != torch.float16 or x.ndim != 2 or x.shape[0] != m or ids.dtype != torch.int32
            or probs.dtype != torch.float32 or probs.shape != ids.shape):
        raise ValueError("MXFP4 experts take FP16 [M, K] activations with int32 ids and FP32 probabilities")
    if ws is None:
        ws = Workspace(m, top, x.shape[1], gate.shape[1], x.device)
    elif (ws.m, ws.top) != (m, top):
        raise ValueError("MXFP4 workspace geometry differs from the call")
    ext = extension()
    if q8:
        ext.quantize(ws.xq, x)
    ext.gate_up(ws.gate_up, x, ws.xq, ids, gate, gate_s, up, up_s, bool(q8))
    ext.swiglu(ws.inter, ws.iq, ws.gate_up, bool(q8))
    ext.down(ws.out, ws.inter, ws.iq, ids, probs, down, down_s, bool(q8))
    return ws.out


def experts_partials(x, ids, gate, gate_s, up, up_s, down, down_s, *, out):
    """Write one unweighted FP32 down plane per route, with +0 for unowned expert ids."""
    if (x.dtype != torch.float16 or x.ndim != 2 or ids.dtype != torch.int32
            or ids.ndim != 2 or ids.shape[0] != x.shape[0]
            or any(t.device != x.device for t in (ids, gate, gate_s, up, up_s, down, down_s, out))):
        raise ValueError("MXFP4 split experts require [M,K] FP16 rows and same-device [M,top] int32 ids")
    routes, channels = ids.numel(), gate.shape[1]
    gate_up = torch.empty((routes, 2, channels), dtype=torch.float16, device=x.device)
    inter = torch.empty((routes, channels), dtype=torch.float16, device=x.device)
    ext = extension()
    # Q8 buffers are unused by the FP16 gate/up and SwiGLU variants.
    ext.gate_up(gate_up, x, x, ids, gate, gate_s, up, up_s, False)
    ext.swiglu(inter, x, gate_up, False)
    ext.down_partials(out, inter, ids, down, down_s)


def finish_partials(planes, ids, probs, *, experts):
    """Complete the expert, routed and shared FP16 boundaries after the FP32 collective."""
    if planes.ndim != 3:
        raise ValueError("MXFP4 complete partials need [top+1,M,H] planes")
    out = torch.empty(planes.shape[1:], dtype=torch.float32, device=planes.device)
    extension().finish_partials(out, planes, ids, probs, experts)
    return out


# ------------------------------------------------------------------------------------- torch reference
def decode(codes, scales, dtype=torch.float32):
    """MXFP4 bytes [..., N, K/2] + E8M0 [..., N, K/32] -> exact values [..., N, K] (FP32 or FP64)."""
    table = torch.tensor(E2M1, dtype=dtype, device=codes.device)
    c = codes.long()
    values = torch.stack((table[c & 15], table[c >> 4]), dim=-1).flatten(-2)
    scale = torch.ldexp(torch.ones((), dtype=dtype, device=codes.device), scales.to(dtype) - 127)
    return values * scale.repeat_interleave(32, dim=-1)


def silu_and_mul_clamp(gate, up, limit=10.0):
    """vLLM silu_and_mul_with_clamp on FP16 operands (clamp, FP32 silu rounded to FP16, FP16 product)."""
    g = gate.float().clamp(max=limit).half().float()
    u = up.float().clamp(-limit, limit).half()
    return ((g / (1.0 + torch.exp(-g))).half().float() * u.float()).half()


def quantize_q8(x):
    """The Q8 mode's activation quantization: per 32, d = amax/127, q = rint(x/d) (dequantized FP32 values)."""
    v = x.float().view(*x.shape[:-1], -1, 32)
    d = v.abs().amax(-1, keepdim=True) / 127.0
    q = torch.where(d == 0, torch.zeros_like(v), torch.round(v / torch.where(d == 0, torch.ones_like(d), d)))
    return (q * d).view(x.shape)


def reference(x, ids, probs, gate, gate_s, up, up_s, down, down_s, *, q8=False, dtype=torch.float32):
    """Torch model of the kernel contract with the same rounding boundaries; ``dtype`` FP64 with q8=False gives
    the near-exact reference for error budgets (FP16 boundaries kept)."""
    m, top = ids.shape
    out = torch.zeros(m, down.shape[1], dtype=dtype, device=x.device)
    for t in range(m):
        xin = quantize_q8(x[t:t + 1]) if q8 else x[t:t + 1].float()
        for r in range(top):
            e = int(ids[t, r])
            if e < 0:
                continue
            g = (xin.to(dtype) @ decode(gate[e], gate_s[e], dtype).T).half()
            u = (xin.to(dtype) @ decode(up[e], up_s[e], dtype).T).half()
            h = silu_and_mul_clamp(g, u)
            hin = quantize_q8(h) if q8 else h.float()
            y = (hin.to(dtype) @ decode(down[e], down_s[e], dtype).T).half()
            out[t] += y[0].to(dtype) * probs[t, r].to(dtype)
    return out.half()

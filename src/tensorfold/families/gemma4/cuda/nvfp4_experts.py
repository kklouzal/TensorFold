"""Gemma CT-native W4A4 experts with HF BF16 GeGLU and ordered routing.

Original FP4 codes and E4M3 block-scale bytes are only permuted. The native
bank consumes original CT input multipliers and prepared FP32 reciprocals of
weight globals. Every row uses the same single K chain and BF16 down boundary.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tensorfold.cuda.nvfp4.experts_a4 import ExpertsA4, PlanBuffers, pack_projection_a4, uniform_input_bits
from tensorfold.cuda.nvfp4.host_layout import authority_fields_a4


def geglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """HF: BF16 tanh GELU storage, then a separate BF16 product with up."""

    return F.gelu(gate, approximate="tanh") * up


@triton.jit
def _combine(Y, IDS, WEIGHTS, OUT, N: tl.constexpr, R: tl.constexpr, RS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slots = tl.arange(0, RS)
    ids = tl.load(IDS + row * R + slots, mask=slots < R, other=2147483647)
    total = tl.full((BLOCK,), 0, tl.float32)
    for _ in tl.static_range(R):
        expert = tl.min(ids, 0)
        slot = tl.min(tl.where(ids == expert, slots, 2147483647), 0)
        ids = tl.where(slots == slot, 2147483647, ids)
        y = tl.load(Y + (row * R + slot) * N + n, mask=n < N, other=0).to(tl.float32)
        weight = tl.load(WEIGHTS + row * R + slot)
        weighted = (y * weight).to(tl.bfloat16).to(tl.float32)
        # HF visits ascending expert IDs and index_adds into a BF16 result.
        # Round every addition; an FP32 slot sum is a different contract.
        total = (total + weighted).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + row * N + n, total.to(tl.bfloat16), mask=n < N)


class Workspace:
    """One model's device routing plans and slot outputs, one buffer per row bucket.

    Every layer's bank shares it: layers run in order on one CUDA stream, so a
    bucket's buffer is never live twice. Buffers are never replaced, so CUDA
    graphs captured over a bucket keep valid addresses for the model's lifetime.
    """

    def __init__(self):
        self._buffers = {}

    def buffers(self, rows: int, slots: int, experts: int, dims: int, device) -> PlanBuffers:
        bucket = 1 << max(4, (rows - 1).bit_length())
        key = (bucket, slots, experts, dims, torch.device(device))
        buf = self._buffers.get(key)
        if buf is None:
            with torch.inference_mode(False):
                buf = self._buffers[key] = PlanBuffers(bucket, slots, experts, dims, device, prefill=True)
        return buf


class Experts:
    """Immutable native bank; slot outputs and plans live in the model's shared workspace."""

    def __init__(self, bank: ExpertsA4, workspace: Workspace | None = None):
        if (not isinstance(bank, ExpertsA4) or bank.direct_input
                or bank.swiglu is not geglu or not bank.capturable):
            raise ValueError("Gemma NVFP4 requires an original CT-calibrated resident native A4 bank")
        self.bank = bank
        self.workspace = Workspace() if workspace is None else workspace
        self.count, self.width, self.dims, self.device = bank.count, bank.width, bank.dims, bank.device

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.bank.device_tensors())

    def forward(self, x: torch.Tensor, picks: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """INT32 picks are constructed by the router; FP32 weights stay FP32.

        Target routing constructs only IDs in0..count-1 (the fused top-k over
        the router's ``count`` logits); the device plan groups them without a
        host read and traps on any other ID, and the native kernels still guard
        every indirect cell access.
        Fixed prefill=True selects SK1 and BF16 output for serial and verify.
        """

        if (not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16 or x.device != self.device or x.ndim != 2
                or x.shape[0] < 1 or x.shape[1] != self.dims):
            raise ValueError("Gemma NVFP4 experts require nonempty BF16 [rows,hidden] on the bank device")
        if (not isinstance(picks, torch.Tensor) or picks.dtype != torch.int32 or picks.device != self.device
                or picks.ndim != 2 or picks.shape[0] != x.shape[0] or not 1 <= picks.shape[1] <= self.count):
            raise ValueError("Gemma NVFP4 expert picks require INT32 [rows,slots] on the bank device")
        if (not isinstance(weights, torch.Tensor) or weights.dtype != torch.float32 or weights.device != self.device
                or tuple(weights.shape) != tuple(picks.shape)):
            raise ValueError("Gemma NVFP4 routing weights require matching FP32 [rows,slots] on the bank device")
        rows, slots = picks.shape
        weights = weights.contiguous()
        buf = self.workspace.buffers(rows, slots, self.count, self.dims, self.device)
        buf.pick[:rows].copy_(picks)
        buf.route(rows)
        self.bank.run_plan(x.contiguous(), buf, prefill=True)
        out = torch.empty((rows, self.dims), dtype=torch.bfloat16, device=self.device)
        _combine[(rows, triton.cdiv(self.dims, 256))](
            buf.y, buf.pick, weights, out, N=self.dims, R=slots, RS=triton.next_power_of_2(slots), BLOCK=256,
            num_warps=4, enable_fp_fusion=False)
        return out


def load_experts(read, name: str, *, count: int, width: int, dims: int, device, calibrated,
                 workspace: Workspace | None = None) -> Experts:
    """Load one resident bank, staging one original projection at a time on CUDA.

    The checkpoint owner validates all headers and CPU scalar calibrations
    before calling. ``calibrated[projection]`` is its (weight reciprocal,
    activation dequantization scale, original input multiplier) tuple. Reuse
    this prepared table without rereading or revalidating original scalars.
    Original scale bytes are reduced on CUDA and
    admitted before publishing the bank; permutation never interprets them as
    floating values. Preparation scratch is bounded by one projection's codes,
    block scales, native packed result and pack temporaries, plus3KiB globals
    for the target128-expert geometry. No CPU authority is retained.
    """

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Gemma native NVFP4 experts require CUDA")
    device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
    geometry = authority_fields_a4(count, width, dims)
    blobs = tuple(torch.empty(shape, dtype=torch.uint8, device=device) for shape, _ in geometry[:2])
    globals_ = tuple(torch.empty(shape, dtype=torch.float32, device="cpu") for shape, _ in geometry[2:])
    scale_top = torch.zeros((), dtype=torch.uint8, device=device)
    for expert in range(count):
        for index, (projection, n, k) in enumerate((("gate_proj", width, dims), ("up_proj", width, dims),
                                                   ("down_proj", dims, width))):
            base = f"{name}.{expert}.{projection}"
            weight, _, inverse = calibrated[base]
            if index < 2:
                globals_[0][expert, index], globals_[2][expert, index] = weight, inverse
            else:
                globals_[1][expert, 0], globals_[3][expert, 0] = weight, inverse
            codes = read(base + ".weight_packed", device=device)
            scales = read(base + ".weight_scale", device=device)
            if (codes.device != device or scales.device != device or codes.dtype != torch.uint8
                    or tuple(codes.shape) != (n, k // 2) or scales.dtype != torch.float8_e4m3fn
                    or tuple(scales.shape) != (n, k // 16)):
                raise ValueError(f"{base}: native NVFP4 codes/block scales differ from the declared geometry")
            scale_top = torch.maximum(scale_top, scales.view(torch.uint8).amax())
            packed = pack_projection_a4(codes, scales)
            (blobs[0][expert, index] if index < 2 else blobs[1][expert]).copy_(packed)
            del codes, scales, packed
    if int(scale_top) > 0x7e:
        raise ValueError(f"{name}: native A4 block scales must be finite unsigned E4M3 bytes in0x00..0x7e")
    # Compare the complete prepared FP32 gate/up multiplier bits on CPU before
    # transfer: equal bits let every row quantize once for all its experts.
    shared_input = uniform_input_bits(globals_[2])
    fields = (*blobs, *(value.to(device=device) for value in globals_))
    return Experts(ExpertsA4(fields, width, dims, swiglu=geglu, shared_input=shared_input), workspace)

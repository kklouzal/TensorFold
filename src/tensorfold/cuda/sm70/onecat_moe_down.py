"""Private 1Cat Q8_1/DP4A down candidates for original IQ4_NL.

This is an explicit additional activation quantization method, NOT the frozen
Half-MMA moe_down2 prototype or an equivalent current-TensorFold down claim.
Original leaves slots UNWRITTEN, including poison/foreign/capture paths; its
version does not advance. TF mode alone publishes per-route slots. Both
methods retain the identical internal Half per-route rounding: upstream-style route%4 warp
fold+FinalHalf, or routed-slot-order Float32 fold. Shared512 is independent and
never read/computed; foreign slots are zero. Finite-weight TF fold agrees with
the current slot fold supplied zero shared output. Nonfinite activations poison
before integer conversion, invalid IDs poison the whole row, never read banks.
Raw scales/nonfinite Float32 route weights retain arithmetic propagation.
Activation[:,10,0] is the producer's reserved +0/poison metadata cell (not live
shared data); any other bits poison even an all-foreign row. Status bits cover
IDs(1), unquantizable owned activation(2), producer poison(4), NOT a finite
output/quality certificate. All remaining shared activation/weight data ignored.

Caller exclusively owns Q8/status/route/output scratch and retains ALL operands
until positive current-stream completion and through graph capture/replay.
Prepared retains module/device only, not Tensor owners or cross-stream events.
The explicit private candidate can publish unweighted Half/F32 route slots,
including independent original shared results, for the existing transport/fold.
The historical folded APIs retain their separate rounding contracts. Selection
requires model quality/resource/performance qualification.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path

PIN = "1b57260a834d40faa32bfc6ac18f050414a8f042"
METHODS = ("onecat-q8-half-warp-fold", "tensorfold-q8-half-slot-f32-fold")


def layout(rows, first, *, ggml_type, column_offset, row_bytes, method):
    if type(rows) is not int or not 1 <= rows <= 32:
        raise ValueError("down component requires R1..32")
    if type(first) is not int or first not in (0, 256):
        raise ValueError("owned E256 first0|256 required")
    if type(ggml_type) is not int or ggml_type != 20:
        raise ValueError("IQ4_NL only; Q2_0 remains incumbent")
    if (
        type(column_offset) is not int
        or column_offset < 0
        or column_offset > 480
        or column_offset % 32
        or type(row_bytes) is not int
        or not 90 <= row_bytes <= 360
        or row_bytes % 18
        or (column_offset // 32 + 5) * 18 > row_bytes
    ):
        raise ValueError("original whole IQ4_NL blocks/ETP4 width160 required")
    if type(method) is not str or method not in METHODS:
        raise ValueError("explicit active-source fold comparison method required")
    return {
        "raw_shape": (256 * 2560, row_bytes),
        "activation_shape": (rows * 11, 160),
        "q8_shape": (rows * 11, 5, 36),
        "q8_bytes": rows * 11 * 5 * 36,
        "route_shape": (rows * 11, 2560),
        "route_bytes": rows * 11 * 2560 * 2,
        "route_written_bytes": 0 if method == METHODS[0] else rows * 11 * 2560 * 2,
        "status_shape": (rows,),
        "output_shape": (rows, 2560),
        "output_dtype": "float16" if method == METHODS[0] else "float32",
        "panels": tuple((i, min(8, rows - i)) for i in range(0, rows, 8)),
        "activation_error_contract": "explicit-Q8_1-approximation-not-qualified",
        "shared_expert": "excluded; independent live shared output must not alias scratch",
        "output_kind": "routed-rank-partial-shared-excluded; never multiply route0 weight again",
    }


def scratch_layout(rows):
    """Caller-funded per-role quantization scratch; output slots are preexisting.

    This is additional Q8_1 activation quantization, not an equivalence claim.
    No allocation or stream/graph lifecycle is hidden by the native API.
    """
    if type(rows) is not int or not 1 <= rows <= 32:
        raise ValueError("down scratch requires R1..32")
    return {"q8_shape": (rows * 11, 5, 36), "status_shape": (rows,),
            "q8_bytes": rows * 11 * 5 * 36, "status_bytes": rows * 4,
            "additional_device_bytes": rows * (11 * 5 * 36 + 4),
            "output_kind": "unweighted-slots-with-independent-shared",
            "activation_error_contract": "explicit-Q8_1-approximation-not-qualified"}


@dataclass(frozen=True)
class Prepared:
    module: object
    device: object

    def slots(self, activation, picks, raw, shared, q8, status, out, *, first, ggml_type, column_offset):
        """Publish unweighted slots, retaining the caller's FP16/FP32 boundary.

        Existing rank transport and ordered weighting/fold consume these slots.
        Caller retains all operands until current-stream/graph retirement.
        """
        import torch

        operands = (activation, picks, raw, shared, q8, status, out)
        if any(type(t) is not torch.Tensor for t in operands):
            raise TypeError("plain Tensor down operands required")
        if picks.ndim != 2 or raw.ndim != 2:
            raise ValueError("matrix picks/original packed rows required")
        layout(picks.shape[0], first, ggml_type=ggml_type,
               column_offset=column_offset, row_bytes=raw.shape[1], method=METHODS[1])
        if activation.device != self.device:
            raise ValueError("prepared device differs")
        self.module.slots(*operands, ggml_type, first, column_offset)
        torch.autograd.graph.increment_version((q8, status, out))
        return out

    def down(self, activation, picks, weights, raw, q8, slots, status, out, *, first, ggml_type, column_offset, method):
        import torch

        operands = (activation, picks, weights, raw, q8, slots, status, out)
        if any(type(t) is not torch.Tensor for t in operands):
            raise TypeError("plain Tensor down operands required")
        if picks.ndim != 2 or raw.ndim != 2:
            raise ValueError("matrix picks/original packed rows required")
        layout(
            picks.shape[0],
            first,
            ggml_type=ggml_type,
            column_offset=column_offset,
            row_bytes=raw.shape[1],
            method=method,
        )
        if activation.device != self.device:
            raise ValueError("prepared device differs")
        self.module.down(*operands, ggml_type, first, column_offset, method == METHODS[0])
        written = (q8, status, out) if method == METHODS[0] else (q8, slots, status, out)
        for tensor in written:
            if not tensor.is_inference():
                torch.autograd.graph.increment_version(tensor)
        return out


def prepare(device):
    import torch
    from tensorfold.cuda import build
    from .foundation_headers import selected_headers
    from .source_cache import stage_sources

    device = torch.device(device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("explicit indexed CUDA device required")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare requires uncaptured quiescent context")
        if torch.cuda.get_device_capability(device) != (7, 0):
            raise RuntimeError("SM70 down component required")
        selected_headers()
        root = Path(__file__).parent
        sources = [root / "onecat_moe_down.cpp", root / "onecat_moe_down.cu"]
        headers = [root.parent / "kernels/tensor_contracts.cuh", root / "gguf_decode_tables.cuh"]
        identity = hashlib.sha256()
        for path in [*sources, *headers]:
            identity.update(path.name.encode() + b"\0" + path.read_bytes())
        staged = stage_sources(sources, headers=headers)
        module = build.load(
            "tf_onecat_down_" + identity.hexdigest()[:20],
            staged,
            need=(7, 0),
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--fmad=false"],
        )
        module.prepare()
    return Prepared(module, device)

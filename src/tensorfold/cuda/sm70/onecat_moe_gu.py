"""Private 1Cat grouped GGUF GU candidates for owned E256 banks.

The explicit private owner prepares these operators before admission/capture.
Both epilogues share the same MMA and typed planes; their numerical and
performance qualification remains separate from implementation.
Original packed GGUF and original down/shared/fold paths remain independent.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path


PIN = "1b57260a834d40faa32bfc6ac18f050414a8f042"
METHODS = ("tensorfold-fp32-epilogue", "onecat-half-epilogue")
FORMATS = {"IQ4_XS": (3, 204800, 25600, 0),
           "IQ3_S": (5, 153600, 25600, 3072),
           "IQ3_XXS": (6, 153600, 12800, 3072),
           "IQ2_S": (7, 102400, 51200, 8192)}


def layout(rows, first, format_name):
    """Exact caller-owned byte extents, not allocator/resident-cost promises.

    Planes are the pinned lane-interleaved ABI, not original GGUF bytes.
    Repacking must be separately proved against original coefficient decode.
    Output is exclusively owned activation scratch, never the live shared
    expert output. Shared expert computation/publication happens separately
    after this component. Output retains 11 slots; shared/foreign are zero.
    Invalid IDs/duplicates/misplaced shared IDs poison the entire row with NaN.
    """
    if type(rows) is not int or not 1 <= rows <= 32:
        raise ValueError("component requires C1..32")
    if type(first) is not int or first not in (0, 256):
        raise ValueError("component requires one E256 group")
    if type(format_name) is not str or format_name not in FORMATS:
        raise ValueError("explicit supported GU plane format required")
    fmt, codes, scales, table = FORMATS[format_name]
    return {"format": fmt, "codes_shape": (256, codes),
            "scales_shape": (256, scales), "table_bytes": table,
            "activation_shape": (rows * 11, 160),
            "activation_bytes": rows * 11 * 160 * 2,
            "gate_up_plane_bytes": 2 * 256 * (codes + scales) + table,
            "panels": tuple((start, min(8, rows - start)) for start in range(0, rows, 8)),
            "device_scratch_bytes": 0}


@dataclass(frozen=True)
class Prepared:
    """Compiled module/device only, prepared before admission.

    Caller exclusively owns activation output and retains all operands stable
    until current-stream completion, and through graph capture/replay lifetime.
    This object does not retain Tensor owners or record cross-stream use;
    caller provides stream handoff/quiescence and positive GPU retirement.
    """
    module: object
    device: object

    def gate_up(self, x, picks, gate_codes, gate_scales, up_codes, up_scales,
                table, out, *, first, format_name, method):
        import torch

        if any(type(t) is not torch.Tensor for t in (x, picks, gate_codes, gate_scales, up_codes, up_scales, table, out)):
            raise TypeError("plain Tensor component operands required")
        facts = layout(x.shape[0], first, format_name)
        if type(method) is not str or method not in METHODS:
            raise ValueError("explicit comparison method required")
        if x.device != self.device:
            raise ValueError("prepared device differs")
        if out.is_inference() and not torch.is_inference_mode_enabled():
            raise RuntimeError("inference output mutation outside inference mode")
        self.module.gate_up(x, picks, gate_codes, gate_scales, up_codes, up_scales,
                            table, out, first, facts["format"], method == METHODS[1])
        torch.autograd.graph.increment_version(out)
        return out


def prepare(device):
    """Explicit native preparation; never invoked by the serving dispatcher."""
    import torch
    from tensorfold.cuda import build
    from .foundation_headers import selected_headers
    from .source_cache import stage_sources

    device = torch.device(device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("explicit indexed CUDA device required")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("preparation requires a quiescent uncaptured stream")
        if torch.cuda.get_device_capability(device) != (7, 0):
            raise RuntimeError("SM70 component only")
        selected_headers()
        here = Path(__file__).parent
        sources = [here / "onecat_moe_gu.cpp", here / "onecat_moe_gu.cu"]
        headers = [here / "onecat_dmv13_core.cuh", here.parent / "kernels/tensor_contracts.cuh"]
        identity = hashlib.sha256()
        for path in [*sources, *headers]:
            identity.update(path.read_bytes())
        staged = stage_sources(sources, headers=headers)
        module = build.load("tf_onecat_gu_" + identity.hexdigest()[:20], staged,
                            need=(7, 0), extra_cflags=["-O3"],
                            extra_cuda_cflags=["-O3", "--fmad=false"])
        module.prepare()
    return Prepared(module, device)


def _raw_iq2_region(rows, first):
    if type(rows) is not int or not 1 <= rows <= 32 or type(first) is not int or first not in (0, 256):
        raise ValueError('raw IQ2 GU requires R1..32/owned first0|256')


def raw_iq2_layout(rows, first):
    _raw_iq2_region(rows, first)
    return {'original_kind': 22, 'local_raw_shape': (256, 160, 820), 'table_bytes': 8192,
            'activation_shape': (rows*11, 160), 'activation_bytes': rows*11*160*2,
            'plane_bytes': 0, 'device_scratch_bytes': 0,
            'panels': tuple((start, min(8, rows-start)) for start in range(0, rows, 8))}


@dataclass(frozen=True)
class RawIQ2Prepared:
    """Unselected raw→Ld experiment, no intermediate planes or hidden owners.

    Caller retains raw/picks/activation/table and provides exclusive current-
    stream/capture lifetime. Math/epilogues stay the pinned GU implementations.
    """
    module: object
    device: object

    def gate_up_raw_iq2(self, x, picks, gate_raw, up_raw, table, out, *, first, method,
                        gate_column_offset=0, up_column_offset=0):
        import torch
        if any(type(t) is not torch.Tensor for t in (x, picks, gate_raw, up_raw, table, out)):
            raise TypeError('plain Tensor raw GU operands required')
        _raw_iq2_region(x.shape[0], first)
        if type(method) is not str or method not in METHODS:
            raise ValueError('explicit existing GU epilogue method required')
        for offset in (gate_column_offset, up_column_offset):
            if type(offset) is not int or offset < 0 or offset%256:
                raise ValueError('raw IQ2 complete256 input offset required')
        if x.device != self.device:
            raise ValueError('raw GU prepared device differs')
        if out.is_inference() and not torch.is_inference_mode_enabled():
            raise RuntimeError('inference output mutation outside inference mode')
        self.module.gate_up_raw_iq2(x, picks, gate_raw, up_raw, table, out, first,
                                   method == METHODS[1], gate_column_offset, up_column_offset)
        torch.autograd.graph.increment_version(out)
        return out

    def register_words(self, raw, out, *, expert, tile, group, column_offset=0):
        """Bounded component diagnostic: defined Ld.c/sc words only, not serving."""
        import torch
        if type(raw) is not torch.Tensor or type(out) is not torch.Tensor:
            raise TypeError('plain Tensor word probe operands required')
        if any(type(v) is not int for v in (expert, tile, group, column_offset)):
            raise ValueError('integer raw IQ2 diagnostic coordinates required')
        if out.device != self.device:
            raise ValueError('raw IQ2 diagnostic device differs')
        if out.is_inference() and not torch.is_inference_mode_enabled():
            raise RuntimeError('inference output mutation outside inference mode')
        self.module.raw_iq2_words(raw, out, expert, tile, group, column_offset)
        torch.autograd.graph.increment_version(out)
        return out


def prepare_raw_iq2(device):
    """Prepare the explicit original-byte IQ2 candidate before admission/capture."""
    import torch
    original = prepare(device)  # Preserve the packed control's preparation.
    with torch.cuda.device(original.device):
        original.module.prepare_raw_iq2()
    return RawIQ2Prepared(original.module, original.device)

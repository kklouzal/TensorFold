# SPDX-License-Identifier: Apache-2.0
"""Active owned GGUF→pinned GU planes, without a retained packed-bank copy.

One CTA per local expert scans the GU row predicate once. Invalid rows, foreign
IDs and shared slot512 do not activate an expert. Inactive output bytes retain
caller contents and MUST NOT be consumed. One immutable CTA owns every written
expert, even with repeated picks across rows. No counts/readback/allocator.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

from .onecat_moe_gu import FORMATS

_NAMES = {18: 'IQ3_XXS', 21: 'IQ3_S', 22: 'IQ2_S', 23: 'IQ4_XS'}


def _format_name(ggml_type):
    if type(ggml_type) is not int or ggml_type not in _NAMES:
        raise ValueError('original GGUF18/21/22/23 type required')
    return _NAMES[ggml_type]


def layout(ggml_type):
    fmt, codes, scales, table = FORMATS[_format_name(ggml_type)]
    return {'format': fmt, 'codes_shape': (256, codes), 'scales_shape': (256, scales),
            'pair_plane_bytes': 2*256*(codes+scales), 'workspace_bytes': 0,
            'local_experts': 256, 'rows_per_expert': 160, 'columns': 2560,
            'GU_table_bytes': table, 'activation': 'identical existing GU valid-row/top10 predicate'}


@dataclass(frozen=True)
class Prepared:
    """Caller owns stable raw/picks/output through current-stream completion.

    Capture/replay callers retain ALL tensors and this module through positive
    retirement. Source expert broadcast/positive gaps allowed; outputs contiguous
    disjoint views in one caller-funded slab. No allocation, Tensor ownership,
    stream handoff or graph lifecycle is hidden in this object.
    """
    module: object
    device: object

    def pack_active(self, gate_raw, up_raw, picks, gate_codes, gate_scales, up_codes, up_scales, *,
                    first, ggml_type, gate_column_offset=0, up_column_offset=0):
        import torch
        _format_name(ggml_type)
        if type(first) is not int or first not in (0, 256):
            raise ValueError('owned E256 first must be0 or256')
        for offset in (gate_column_offset, up_column_offset):
            if type(offset) is not int or offset < 0 or offset%256:
                raise ValueError('whole256-value source offset required')
        inputs = (gate_raw, up_raw, picks)
        outputs = (gate_codes, gate_scales, up_codes, up_scales)
        if any(type(t) is not torch.Tensor for t in (*inputs, *outputs)):
            raise TypeError('plain Tensor packing operands required')
        if any(t.device != self.device for t in (*inputs, *outputs)):
            raise ValueError('packing device differs from resident preparation')
        if not torch.is_inference_mode_enabled() and any(t.is_inference() for t in outputs):
            raise RuntimeError('inference output mutation outside inference mode')
        self.module.pack_active(gate_raw, up_raw, picks, *outputs, first, ggml_type,
                                gate_column_offset, up_column_offset)
        torch.autograd.graph.increment_version(outputs)
        return outputs


    def pack_union(self, gate_raw, up_raw, picks, gate_codes, gate_scales, up_codes, up_scales, *,
                   first, ggml_type, gate_column_offset=0, up_column_offset=0):
        """Pack the valid expert union for one bounded prefill/layer.

        All source/picks and frame views remain owned through every consuming
        GU panel. The caller serializes this layer before another frame writer.
        Unlike pack_active, the native envelope is at most 65,535 total pairs.
        """
        import torch
        _format_name(ggml_type)
        if type(first) is not int or first not in (0, 256):
            raise ValueError('owned E256 first must be0 or256')
        for offset in (gate_column_offset, up_column_offset):
            if type(offset) is not int or offset < 0 or offset%256:
                raise ValueError('whole256-value source offset required')
        inputs = (gate_raw, up_raw, picks)
        outputs = (gate_codes, gate_scales, up_codes, up_scales)
        if any(type(t) is not torch.Tensor for t in (*inputs, *outputs)):
            raise TypeError('plain Tensor packing operands required')
        if any(t.device != self.device for t in (*inputs, *outputs)):
            raise ValueError('packing device differs from resident preparation')
        if not torch.is_inference_mode_enabled() and any(t.is_inference() for t in outputs):
            raise RuntimeError('inference output mutation outside inference mode')
        self.module.pack_union(gate_raw, up_raw, picks, *outputs, first, ggml_type,
                               gate_column_offset, up_column_offset)
        torch.autograd.graph.increment_version(outputs)
        return outputs


def prepare(device):
    import torch
    from tensorfold.cuda import build
    from .foundation_headers import selected_headers
    from .source_cache import stage_sources

    device = torch.device(device)
    if device.type != 'cuda' or device.index is None:
        raise ValueError('explicit indexed CUDA device required')
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('packing preparation requires quiescent uncaptured stream')
        if torch.cuda.get_device_capability(device) != (7, 0):
            raise RuntimeError('SM70 packing only')
        selected_headers()
        here = Path(__file__).parent
        sources = [here/'onecat_gpu_packing.cpp', here/'onecat_gpu_packing.cu']
        headers = [here/'gguf_decode_tables.cuh']
        digest = hashlib.sha256()
        for source in (*sources, *headers):
            digest.update(source.name.encode()+b'\0'+source.read_bytes())
        staged = stage_sources(sources, headers=headers)
        module = build.load('tf_onecat_pack_'+digest.hexdigest()[:20], staged, need=(7, 0),
                            extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--fmad=false'])
        module.prepare()
    return Prepared(module, device)

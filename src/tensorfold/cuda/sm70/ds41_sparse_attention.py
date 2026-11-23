"""DeepSeek-V4.1 fused sparse attention for SM70 (the friend's patch 181, ds41mu_attn2.cu, at TP8: 8 heads/rank).

Replaces patch 87's per-64-token passes of ``mu_packed.gather_attention`` (a materialized [64, 640, 512] FP16
key buffer) + cuBLAS Q.K^T + ``_sink_softmax`` + cuBLAS P.K: one token per CTA streams its 128 window keys (this
forward's latents or the slot ring) and its 512 selected packed rows straight into m8n8k4 fragments; scores and
outputs stay in registers. Legacy compressed rows decode exactly as ``kv_pack.unpack_c_row``. RotorQuant
ISO64 H64 windows and compressed rows decode together in their stored basis. In that mode the caller supplies
rotated queries and receives rotated output.

Numerics: FP16 keys and the FP16 probability boundary are patch 87's; the score dot, the softmax sum and P.K
accumulate in FP32 in a different order than cuBLAS (the reference's P.K accumulation is cuBLAS's), so outputs
differ in the last FP16 bits (see tests/cuda/deepseek_v41/test_sm70_ds41_sparse_attention.py).

Operands are ``mu_packed.attend``'s: query [T, 8, 512] FP16, the layer's ring [S + 1, 128, D], this
forward's window latents [T, D] (FP16 D=512 or RotorQuant uint8 D=272), the Step's qsl / req_slot / positions /
valid / token_req, the pool's packed pages / block table / layout region and top-512 int32 positions (compressed
layers), the [8] FP32 sinks.
Graph-capturable (no host synchronization; allocates only ``out`` when not given).
"""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch

SCALE = 512 ** -0.5


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build

    if torch.cuda.get_device_capability() != (7, 0) or "sm_70" not in torch.cuda.get_arch_list():
        raise RuntimeError("DeepSeek-V4.1 sparse attention requires a Volta-enabled PyTorch build and SM70 device")
    from .source_cache import stage_sources

    here = Path(__file__).parent
    sources = [here / "ds41_sparse_attention.cpp", here / "ds41_sparse_attention.cu"]
    headers = [here / "rotorquant_q4.cuh", here / "rotorquant_q4_tables.cuh"]
    flags = ["-O3", "--fmad=false"]
    identity = hashlib.sha256(repr(flags).encode())
    for path in (*sources, *headers):
        identity.update(path.name.encode() + b"\0" + path.read_bytes())
    staged = stage_sources(sources, headers=headers)
    return build.load("tf_ds41_sparse_attention_" + identity.hexdigest()[:20], staged, need=(7, 0),
                      extra_cflags=["-O3"], extra_cuda_cflags=flags)


def prepare():
    """Build/load the extension before serving (otherwise the first call does)."""
    return extension()


def attend(query, positions, step, req_slot, window_ring, window, sink, *, cache=None, block_table=None,
           layout=None, source=None, topk=None, out=None):
    """Sparse attention output [T, 8, 512] FP16 (``mu_packed.attend``'s contract and operands)."""
    if out is None:
        out = torch.empty_like(query)
    compressed = cache is not None
    if compressed != (block_table is not None) or compressed != (topk is not None):
        raise ValueError("compressed sparse attention needs cache, block table and top-512 positions together")
    if compressed:
        ratio, _rows, c_offset, _i = layout.regions[source]
        block_size = layout.block_size
        rotor = window_ring.dtype == torch.uint8
        if layout.rotorquant != rotor:
            raise ValueError("compressed row format must match the window-ring mode")
        compressed_row_bytes = layout.compressed_row_bytes
    else:
        ratio, c_offset, block_size, compressed_row_bytes = 1, 0, 1, 0
    extension().attend(out, query, window_ring, window, step.qsl, req_slot, cache if compressed else None,
                       block_table if compressed else None, positions, step.valid, step.token_req,
                       topk if compressed else None, sink, SCALE, block_size, ratio, c_offset, 0, query.shape[0], compressed_row_bytes)
    return out

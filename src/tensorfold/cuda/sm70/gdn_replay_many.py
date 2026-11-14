"""One launch per State for the accepted-prefix GDN replay of every recurrent layer.

Bitwise the per-layer ``gdn_native.replay`` calls (same kernel body). A State's
saved-row pointer table is built once and reused while the State keeps the same
GDNScratch objects; ``replay_state`` returns False (caller keeps the per-layer
loop) for anything outside the validated geometry.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch

CUDA_FLAGS = ("-O3", "--fmad=false")


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .provider import _native
    from .source_cache import stage_sources

    prepared = _native()  # The selected SM70/toolkit/ABI admission and identity, as gdn_native.
    root = Path(__file__).parent
    sources = [root / "gdn_replay_many.cu"]
    headers = [root / "gdn_common.cuh"]
    digest = hashlib.sha256(prepared.__name__.encode() + repr(CUDA_FLAGS).encode())
    for source in [*sources, *headers]:
        digest.update(source.name.encode() + b"\0" + source.read_bytes())
    staged = stage_sources(sources, headers=headers, source_root=root.parent)
    return build.load("tf_sm70_gdn_replay_many_" + digest.hexdigest()[:20], staged, need=(7, 0),
                      extra_cuda_cflags=list(CUDA_FLAGS), extra_cflags=["-O3"])


def _table(state, rows):
    """The State's [layers, 4] k/v/g/b pointer table, validated once per scratch set."""
    scratch, rec = state.scratch, state.rec
    cached = getattr(state, "_gdn_replay_table", None)
    if (cached is not None and cached[0] is rec and len(cached[1]) == len(scratch)
            and all(a is b for a, b in zip(cached[1], scratch)) and rows <= cached[3]):
        return cached[2]
    layers, nv = rec.shape[1], rec.shape[2]
    nk = nv // 3
    capacity = None
    pointers = []
    for sc in scratch:
        k, v, g, b = sc.k, sc.v, sc.g, sc.b
        if (any(not x.is_cuda or x.device != rec.device or not x.is_contiguous() for x in (k, v, g, b))
                or k.dtype != torch.float32 or v.dtype != torch.float16 or g.dtype != torch.float32
                or b.dtype != torch.float32 or k.dim() != 3 or tuple(k.shape[1:]) != (nk, 128)
                or v.dim() != 3 or tuple(v.shape[1:]) != (nv, 128) or g.dim() != 2 or b.dim() != 2
                or g.shape[1] != nv or b.shape[1] != nv):
            return None
        held = min(k.shape[0], v.shape[0], g.shape[0], b.shape[0])
        capacity = held if capacity is None else min(capacity, held)
        pointers.append([k.data_ptr(), v.data_ptr(), g.data_ptr(), b.data_ptr()])
    if len(pointers) != layers or capacity is None or rows > capacity:
        return None
    table = torch.tensor(pointers, dtype=torch.int64).to(rec.device)
    state._gdn_replay_table = (rec, tuple(scratch), table, capacity)
    return table


def replay_state(state, rows):
    """Replay ``rows`` saved rows of every layer from rec[cur] into rec[1-cur]; False if not applicable."""
    rec, cur = state.rec, state.cur
    if (not isinstance(rec, torch.Tensor) or not rec.is_cuda or rec.dtype != torch.float32 or rec.dim() != 5
            or not rec.is_contiguous() or rec.shape[0] != 2 or not 1 <= rec.shape[1] <= 64
            or len(cur) != rec.shape[1] or any(parity not in (0, 1) for parity in cur) or not 1 <= rows <= 1024):
        return False
    table = _table(state, rows)
    if table is None:
        return False
    parity = 0
    for layer, bank in enumerate(cur):
        parity |= bank << layer
    extension().replay_many(rec, parity, table, rows)
    return True

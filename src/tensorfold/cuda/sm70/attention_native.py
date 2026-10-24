"""Fixed-position WMMA attention candidate for the existing FP16 cache ABI."""

from functools import lru_cache
import hashlib
from pathlib import Path


PARTIAL_THRESHOLDS = ((1, 1), (881, 10), (2731, 38), (2737, 39), (2809, 40), (2882, 41), (5096, 49), (5104, 50))
MAX_CONTEXT = 8192


def workspace_capacity(capacity):
    """Native storage covers qualified launches, independently of KV capacity."""
    if type(capacity) is not int or capacity < 0:
        raise ValueError("attention workspace capacity requires a nonnegative integer")
    return min(capacity, MAX_CONTEXT)

def partial_slots(capacity):
    maximum = 0
    for low, count in PARTIAL_THRESHOLDS:
        if capacity < low:
            break
        maximum = count
    return max((capacity + 127)//128, maximum)


def workspace_bytes(rows, heads, head_dim, capacity):
    """Exact retained score/probability/partial arrays; excludes existing AttnScratch fields."""
    if any(type(value) is not int or value < 0 for value in (rows, heads, head_dim, capacity)):
        raise ValueError("attention workspace geometry requires nonnegative integer dimensions")
    capacity = workspace_capacity(capacity)
    width = (capacity + 31) // 32 * 32
    tiles = partial_slots(capacity)
    return rows * heads * (6 * width + 4 * tiles * head_dim)


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .provider import _native
    from .source_cache import stage_sources

    prepared = _native()
    root = Path(__file__).parent
    sources = [root / "attention_bindings.cpp", root / "attention.cu"]
    identity = hashlib.sha256(prepared.__name__.encode())
    headers = [root.parent / "kernels/tensor_contracts.cuh", root / "pv_dispatch.cuh"]
    for source in [*sources, *headers]:
        identity.update(source.name.encode() + b"\0" + source.read_bytes())
    staged = stage_sources(sources, headers=headers, source_root=root.parent)
    return build.load("tf_sm70_attention_" + identity.hexdigest()[:20], staged, need=(7, 0),
                      extra_cuda_cflags=["-O3", "--fmad=false"], extra_cflags=["-O3"])


class Workspace:
    """Fixed qualified-context strides; larger KV owners retain their full extent."""

    def __init__(self, rows, heads, dim, capacity, device):
        import torch

        capacity = self.capacity = workspace_capacity(capacity)
        self.width = (capacity + 31) // 32 * 32
        self.scores = torch.empty((rows, heads, self.width), dtype=torch.float32, device=device)
        self.probabilities = torch.empty((rows, heads, self.width), dtype=torch.float16, device=device)
        self.partials = torch.empty((rows, heads, partial_slots(capacity), dim), dtype=torch.float32, device=device)


def attention(q, k, v, position, scratch, rows, scale, out, context, ks, vs, kbits, vbits):
    if (q.shape[1:] != (3, 256) or k.shape[1:] != (1, 256)
            or not 0 <= rows <= 24 or not 65 <= context <= 8192):
        raise ValueError("native attention requires qualified H3/KV1/HD256 rows<=24 and context65..8192")
    workspace = getattr(scratch, "native_workspace", None)
    if workspace is None:
        workspace = Workspace(scratch.out.shape[0], q.shape[1], q.shape[2],
                              getattr(scratch, "capacity", k.shape[0]), q.device)
        scratch.native_workspace = workspace
    if workspace.capacity < context:
        raise ValueError("native attention context exceeds its workspace capacity")
    extension().run(q, k, v, position, scratch.sparse, scratch.nk, scratch.ids,
                    workspace.scores, workspace.probabilities, workspace.partials, out,
                    ks if kbits else None, vs if vbits else None, rows, context,
                    kbits, vbits, scratch.qsa, scale)
    return out

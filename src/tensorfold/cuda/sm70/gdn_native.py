"""Native FP16 chain/replay for the existing Flash Next State/GDNScratch ABI."""

from functools import lru_cache
import hashlib
from pathlib import Path


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .provider import _native
    from .source_cache import stage_sources

    prepared = _native()  # Retain the selected SM70/toolkit/ABI admission and identity.
    root = Path(__file__).parent
    sources = [root / "gdn_bindings.cpp", root / "gdn.cu"]
    identity = hashlib.sha256(prepared.__name__.encode())
    headers = [root.parent / "kernels/tensor_contracts.cuh"]
    for source in [*sources, *headers]:
        identity.update(source.name.encode() + b"\0" + source.read_bytes())
    staged = stage_sources(sources, headers=headers)
    return build.load("tf_sm70_gdn_" + identity.hexdigest()[:20], staged, need=(7, 0),
                      extra_cuda_cflags=["-O3", "--fmad=false"], extra_cflags=["-O3"])


def chain(p, conv, weight, initial, decay_parameter, bias, norm, eps, rows, scratch, final, output, sums,
          *, negative_decay=False):
    return extension().chain(p, conv, weight, initial, decay_parameter, bias, norm, eps, rows, output, sums, final,
                             scratch.k, scratch.v, scratch.g, scratch.b, getattr(scratch, "keep_rows", None), negative_decay)


def replay(initial, scratch, rows, final):
    return extension().replay(initial, scratch.k, scratch.v, scratch.g, scratch.b, rows, final)


def batch_chain(p, weight, decay, bias, norm, eps, tables, layer, output, sums, *, negative_decay=False):
    """One unchanged chain body per independent(State,value-head) CTA."""
    from tensorfold.families.qwen4_exp.cuda.gdn_multi import NativeTables

    if type(tables) is not NativeTables:
        raise TypeError("native GDN batch requires the current owned frame")
    pointers, starts = tables.operands(layer, p, output, sums)
    return extension().batch_chain(p, weight, decay, bias, norm, eps, pointers, starts, output, sums, negative_decay)


def prepare_batch(device, value_heads):
    """Resolve the exact kernel functions before physical startup admission."""
    import torch
    with torch.cuda.device(device):
        extension().prepare_batch(value_heads)

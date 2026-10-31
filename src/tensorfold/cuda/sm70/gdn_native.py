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

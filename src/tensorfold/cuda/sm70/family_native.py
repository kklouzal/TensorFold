"""Capture-safe fused FP16 family operations behind the shared family ABI."""

from functools import lru_cache
import hashlib
from pathlib import Path


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .provider import _native
    from .source_cache import stage_sources

    prepared = _native()
    root = Path(__file__).parent
    sources = [root / "family_bindings.cpp", root / "family.cu"]
    identity = hashlib.sha256(prepared.__name__.encode())
    headers = [root / "torch_reduce.cuh", root.parent / "kernels/tensor_contracts.cuh"]
    for source in [*sources, *headers]:
        identity.update(source.name.encode() + b"\0" + source.read_bytes())
    staged = stage_sources(sources, headers=headers)
    return build.load("tf_sm70_family_" + identity.hexdigest()[:20], staged, need=(7, 0),
                      extra_cuda_cflags=["-O3", "--fmad=false"], extra_cflags=["-O3"])

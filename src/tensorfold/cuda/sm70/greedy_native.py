"""Private exact SM70 greedy packet epilogue, with no tensor allocation."""

from functools import lru_cache
import hashlib
from pathlib import Path


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from . import provider
    from .source_cache import stage_sources

    prepared = provider._native()
    root = Path(__file__).parent
    sources = [root / "greedy_bindings.cpp", root / "greedy.cu"]
    header = root.parent / "kernels/tensor_contracts.cuh"
    pinned = Path(provider.__file__).parent.parent / "kernels/tensor_contracts.cuh"
    if header.read_bytes() != pinned.read_bytes():
        raise ValueError("greedy epilogue must retain the pinned TensorFold tensor-contract header")
    flags = ["-O3", "--fmad=false"]
    identity = hashlib.sha256(prepared.__name__.encode())
    identity.update(repr(flags).encode())
    for source in [*sources, header]:
        identity.update(source.name.encode()+b"\0"+source.read_bytes())
    staged = stage_sources(sources, headers=[header], source_root=root.parent)
    return build.load("tf_sm70_greedy_"+identity.hexdigest()[:20], staged, need=(7, 0),
                      extra_cuda_cflags=flags, extra_cflags=["-O3"])

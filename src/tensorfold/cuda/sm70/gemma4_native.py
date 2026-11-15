"""Native Gemma 4 SM70 kernels (gemma4.cu), built once per source identity through tensorfold.cuda.build."""

from functools import lru_cache
import hashlib
from pathlib import Path

PROFILE = "gemma4-sm70-w4a16-v1"


@lru_cache(maxsize=1)
def extension():
    import torch
    from tensorfold.cuda import build
    from .source_cache import stage_sources

    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("Gemma 4 SM70 kernels require an actual SM70 device")
    root = Path(__file__).parent
    sources = [root / "gemma4_bindings.cpp", root / "gemma4.cu"]
    flags = ["-O3", "--use_fast_math=false"]
    identity = hashlib.sha256(repr(flags).encode())
    for source in sources:
        identity.update(source.name.encode() + b"\0" + source.read_bytes())
    module = build.load("tf_sm70_gemma4_" + identity.hexdigest()[:20], stage_sources(sources), need=(7, 0),
                        extra_cuda_cflags=["-O3"], extra_cflags=["-O3"])
    if module.profile != PROFILE:
        raise RuntimeError("compiled Gemma 4 SM70 numerical profile differs from the declared source")
    return module

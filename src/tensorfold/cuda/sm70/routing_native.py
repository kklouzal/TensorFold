"""Stable SM70 expert routing with the pinned Torch softmax and FP16 boundaries."""
from functools import lru_cache
import hashlib
from pathlib import Path

# Qualified explicitly by a prepared execution policy; ordinary routing remains selected.
_NATIVE_LOGITS = False


def supports_logits(x, weight, out):
    import torch

    if (not _NATIVE_LOGITS or not x.is_cuda or x.dtype != torch.float16 or x.ndim != 2
            or not 1 < x.shape[0] <= 32 or x.shape[1] != 2560 or x.stride(1) != 1 or x.stride(0) < 2560
            or weight.dtype != torch.float32 or weight.device != x.device or weight.shape != (513, 2560)
            or not weight.is_contiguous() or weight.data_ptr() % 512
            or out.dtype != torch.float32 or out.device != x.device or out.shape != (x.shape[0], 513)
            or not out.is_contiguous()):
        return False
    if torch.cuda.get_device_capability(x.device) != (7, 0):
        return False
    return not (torch._C._overlaps(out, x) or torch._C._overlaps(out, weight))


def logits(x, weight, out):
    """Write qualified FP32 router sums without staging or per-row launches."""
    extension().logits(x, weight, out)
    return out


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .provider import _native

    prepared = _native()
    root = Path(__file__).parent
    sources = [root / "routing_bindings.cpp", root / "routing.cu"]
    identity = hashlib.sha256(prepared.__name__.encode())
    for source in [*sources, root.parent / "kernels/tensor_contracts.cuh"]:
        identity.update(source.name.encode() + b"\0" + source.read_bytes())
    from .source_cache import stage_sources

    sources = stage_sources(sources, headers=[root.parent / "kernels/tensor_contracts.cuh"])
    return build.load("tf_sm70_routing_" + identity.hexdigest()[:20], [str(p) for p in sources], need=(7, 0),
                      extra_cuda_cflags=["-O3", "--fmad=false"], extra_cflags=["-O3"])


def indices(logits, count):
    import torch

    result = torch.empty((logits.shape[0], count), dtype=torch.int64, device=logits.device)
    extension().select(logits, result)
    return result


def route(logits, picks, weights):
    """Write selected int32 IDs and FP16-rounded FP32 expert/shared weights."""
    extension().route(logits, picks, weights)

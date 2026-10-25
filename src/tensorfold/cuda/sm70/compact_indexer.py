"""Generation-guarded bitwise publication for compact indexer state."""

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
    sources = [root / "compact_indexer_bindings.cpp", root / "compact_indexer.cu"]
    digest = hashlib.sha256(prepared.__name__.encode())
    for source in sources:
        digest.update(source.name.encode() + b"\0" + source.read_bytes())
    return build.load("tf_sm70_compact_indexer_" + digest.hexdigest()[:20], stage_sources(sources), need=(7, 0),
                      extra_cuda_cflags=["-O3", "--fmad=false"], extra_cflags=["-O3"])


def publish(source, destination, at, generations, expected, status, layer):
    """Caller owns contiguous buffers and validated row extent through retirement."""
    import torch

    if source.is_cuda:
        stream = torch.cuda.current_stream(source.device)
        for tensor in (source, destination, generations, status):
            tensor.record_stream(stream)
        return extension().publish(source, destination, at, generations, *expected, status, layer)
    tensors = (source, destination, generations, status)
    if (not all(value.device == source.device and value.is_contiguous() for value in tensors)
            or source.ndim < 1 or destination.ndim != source.ndim or source.dtype != destination.dtype
            or source.shape[1:] != destination.shape[1:] or type(at) is not int or at < 0
            or len(source) > len(destination) or at > len(destination) - len(source)
            or generations.dtype != torch.int64 or generations.numel() != 2
            or status.dtype != torch.int32 or status.numel() != 2):
        raise ValueError("compact publication requires matching contiguous owned rows and controls")
    for control in (generations, status):
        first, last = destination.data_ptr(), destination.data_ptr() + destination.nbytes
        begin, end = control.data_ptr(), control.data_ptr() + control.nbytes
        if first < end and begin < last:
            raise ValueError("compact publication destination aliases its generation/status control")
    if tuple(generations.tolist()) != expected:
        status[0].bitwise_or_(32)
        if int(status[1]) == -2:
            status[1] = layer
        return
    destination[at:at + len(source)].copy_(source)


def prepare(device):
    """Load and exercise the one byte-copy kernel before the capacity baseline."""
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        return
    extension()
    source = torch.zeros((1, 1), dtype=torch.float16, device=device)
    destination = torch.empty_like(source)
    generations = torch.zeros(2, dtype=torch.int64, device=device)
    status = torch.tensor([0, -2], dtype=torch.int32, device=device)
    publish(source, destination, 0, generations, (0, 0), status, -1)
    torch.cuda.synchronize(device)

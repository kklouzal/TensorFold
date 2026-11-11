"""Stable SM70 expert routing with the pinned Torch softmax and FP16 boundaries."""
from functools import lru_cache
import hashlib
import os
from pathlib import Path

# Qualified explicitly by a prepared execution policy; ordinary routing remains selected.
_NATIVE_LOGITS = False
# Process-fixed operator-only trial; it never enables top-k, padding or other family kernels.
_PRIVATE_SERIAL_LOGITS = os.environ.get("TENSORFOLD_PRIVATE_SM70_ROUTER_LOGITS", "inherit")
if _PRIVATE_SERIAL_LOGITS not in ("inherit", "native-serial-v1"):
    raise ValueError("TENSORFOLD_PRIVATE_SM70_ROUTER_LOGITS requires inherit or native-serial-v1")


def supports_logits(x, weight, out):
    import torch

    if (not (_NATIVE_LOGITS or _PRIVATE_SERIAL_LOGITS == "native-serial-v1") or not x.is_cuda or x.dtype != torch.float16 or x.ndim != 2
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


class PreparedMoESelection:
    """Resident routing callable, adopted only by actual owned static buffers."""
    def __init__(self, module, device):
        import torch
        self.module, self.device, self.route = module, torch.device(device), module.route

    def bind(self, buffers):
        import torch
        tensors = buffers.logits, buffers.pick, buffers.wts
        rows = buffers.rows
        if (type(rows) is not int or not 1 <= rows <= 256 or buffers.act.dtype != torch.float16
                or any(type(t) is not torch.Tensor or t.device != self.device or not t.is_contiguous()
                       or t.requires_grad or t.is_conj() or t.is_neg() for t in tensors)
                or tensors[0].dtype != torch.float32 or tensors[0].shape != (rows, 513)
                or tensors[1].dtype != torch.int32 or tensors[1].shape != (rows, 11)
                or tensors[2].dtype != torch.float32 or tensors[2].shape != (rows, 11)):
            raise ValueError('owned selection requires static Half/512/top10 buffers with1..256 rows')
        spans = [(t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in tensors]
        if any(a < d and c < b for index, (a, b) in enumerate(spans) for c, d in spans[index + 1:]):
            raise ValueError('owned selection buffers must be disjoint')
        return _OwnedMoESelection(self, tensors)


class _OwnedMoESelection:
    def __init__(self, prepared, tensors):
        self.prepared, self.owners = prepared, tensors
        self.rows = tensors[0].shape[0]
        self.pointers = tuple(t.data_ptr() for t in tensors)
        # The Torch reference writes each output in two separate view operations.
        # Inference tensors have no version counter; other owners increment twice.
        self.mutations = tuple(t for t in (tensors[1], tensors[2], tensors[1], tensors[2]) if not t.is_inference())
        self.inference_outputs = tensors[1].is_inference() or tensors[2].is_inference()

    def run(self, logits, buffers, top_k, experts):
        import torch
        source, picks, weights = self.owners
        if (self.inference_outputs and not torch.is_inference_mode_enabled()
                or type(top_k) is not int or top_k != 10 or type(experts) is not int or experts != 512
                or buffers.logits is not source or buffers.pick is not picks or buffers.wts is not weights
                or type(logits) is not torch.Tensor or logits.dtype != torch.float32 or logits.ndim != 2
                or not 1 <= logits.shape[0] <= self.rows or logits.shape[1] != 513
                or logits.device != self.prepared.device or not logits.is_contiguous()
                or logits.requires_grad or logits.is_conj() or logits.is_neg()
                or logits.data_ptr() != self.pointers[0]
                or source.data_ptr() != self.pointers[0] or source.shape != (self.rows, 513)
                or source.dtype != torch.float32 or source.device != self.prepared.device
                or not source.is_contiguous() or source.requires_grad or source.is_conj() or source.is_neg()
                or picks.data_ptr() != self.pointers[1] or weights.data_ptr() != self.pointers[2]
                or picks.requires_grad or weights.requires_grad
                or picks.dtype != torch.int32 or weights.dtype != torch.float32
                or picks.device != self.prepared.device or weights.device != self.prepared.device
                or picks.is_conj() or picks.is_neg() or weights.is_conj() or weights.is_neg()
                or picks.shape != (self.rows, 11) or weights.shape != picks.shape
                or not picks.is_contiguous() or not weights.is_contiguous()):
            return False
        self.prepared.route(logits, picks[:logits.shape[0]], weights[:logits.shape[0]])
        if self.mutations:
            torch.autograd.graph.increment_version(self.mutations)
        return True


def prepare_owned_selection(device):
    import torch
    device = torch.device(device)
    if device.type != 'cuda' or torch.cuda.get_device_capability(device) != (7, 0):
        raise ValueError('owned selection requires actual SM70 preparation')
    with torch.cuda.device(device):
        module = extension()
        module.prepare_route()
    return PreparedMoESelection(module, device)

"""Standalone explicit Q8 activation/raw-GGUF profile. NEVER enrolled by Engine.

Changes activation precision, current RN16 coefficient boundary and reduction
order. Status is a monotonic device latch; hot calls never inspect it on CPU.
Model use is prohibited until an admitted device-status veto before commit and
model/task-quality/resource gates exist. Shared/unsupported/MTP routes stay exact.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

import torch

from tensorfold.cleanup import raise_failures

CONTRACT = "standalone-q8_32-f32scale-rawgguf-dp4a-v1"
NONFINITE_INPUT, NONFINITE_WEIGHT, NONFINITE_HIDDEN, INVALID_PICK = 1, 2, 4, 8


@lru_cache(maxsize=1)
def extension():
    from tensorfold.cuda import build
    from .provider import _native
    from .source_cache import stage_sources

    prepared = _native()
    root = Path(__file__).parent
    sources = [root / "dp4a_experts.cpp", root / "dp4a_experts.cu"]
    headers = [root.parent / "kernels/tensor_contracts.cuh", root / "gguf_decode_tables.cuh"]
    identity = hashlib.sha256(prepared.__name__.encode() + CONTRACT.encode())
    for source in [*sources, *headers]:
        identity.update(source.name.encode() + b"\0" + source.read_bytes())
    return build.load("tf_dp4a_" + identity.hexdigest()[:20], stage_sources(sources, headers=headers), need=(7, 0),
                      extra_cuda_cflags=["-O3", "--fmad=false", "--ftz=false", "--prec-div=true"],
                      extra_cflags=["-O3"])


def _signature(tensor):
    return (tensor.data_ptr(), tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes(),
            tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device, tensor.requires_grad)


def eligible(bank, role):
    return (role == "target" and bank.count == 513 and bank.width == 80 and bank.dims == 2560
            and bank.gate.ggml_type in (18, 22) and bank.up.ggml_type in (18, 22)
            and bank.gate.column_offset == bank.up.column_offset == 0
            and bank.down.ggml_type == 20 and bank.down.column_offset in (0, 16)
            and bank.limit == 0.0 and bank.swiglu is True and len(bank.shared) == 3)


class Prototype:
    """Fixed binding/stream owner; data may change, storage must not.

    Never resets a failed status in a hot call. reset_after_fence is explicit
    cold recovery, not permission to commit a failed result. Graphs retain all
    original bindings and must be closed before this owner can release them.
    """
    def __init__(self, bank, x, picks, out, *, role="target"):
        if (x.dtype != torch.float16 or not x.is_cuda or x.ndim != 2 or x.shape[1] != 2560
                or not 1 <= x.shape[0] <= 32 or not x.is_contiguous()
                or picks.dtype != torch.int32 or picks.ndim != 2 or picks.shape[0] != x.shape[0]
                or not 1 <= picks.shape[1] <= 11 or not picks.is_contiguous() or picks.device != x.device
                or out.shape != (*picks.shape, 2560) or out.device != x.device or not out.is_contiguous()
                or out.dtype not in (torch.float16, torch.float32)):
            raise ValueError("standalone prototype needs fixed contiguous FP16/INT32 expert geometry")
        if any(torch._C._overlaps(out, value) for value in (x, picks, bank.gate.data, bank.up.data, bank.down.data, *(part.weight for part in bank.shared))):
            raise ValueError("prototype output must be disjoint from inputs/weights")
        if any(t.requires_grad for t in (x, out)):
            raise ValueError("standalone prototype is inference-only")
        self.bank, self.x, self.picks, self.out = bank, x, picks, out
        self.device = x.device
        self.stream = torch.cuda.current_stream(self.device)
        self.closed, self.failed, self.graph_owner = False, None, None
        self.exact_only = not eligible(bank, role)
        self.qx = self.qh = self.gate = self.up = self.hidden = self.status = None
        if not self.exact_only:
            rows, slots = picks.shape
            self.qx = torch.empty((rows, 80, 36), dtype=torch.uint8, device=self.device)
            self.qh = torch.empty((rows * slots, 3, 36), dtype=torch.uint8, device=self.device)
            self.gate = torch.empty((rows, slots, 80), dtype=torch.float32, device=self.device)
            self.up = torch.empty_like(self.gate)
            self.hidden = torch.empty((rows, slots, 80), dtype=torch.float16, device=self.device)
            self.status = torch.zeros((1,), dtype=torch.int32, device=self.device)
        self.scratch = () if self.exact_only else (self.qx, self.qh, self.gate, self.up, self.hidden, self.status)
        self.bindings = tuple((value, _signature(value)) for value in
            (x, picks, out, bank.gate.data, bank.up.data, bank.down.data, *(part.weight for part in bank.shared), *self.scratch))
        self.weights = (bank.gate, bank.up, bank.down, *bank.shared)
        self.weight_bindings = (bank.gate.data, bank.up.data, bank.down.data, *(part.weight for part in bank.shared))
        self.shared_policy = tuple((part.n, part.k, part.implementation) for part in bank.shared)
        self.layout = (bank.count, bank.width, bank.dims, bank.gate.ggml_type, bank.up.ggml_type,
                       bank.down.ggml_type, bank.gate.column_offset, bank.up.column_offset, bank.down.column_offset, bank.limit, bank.swiglu)

    def _validate(self):
        if self.closed or self.failed is not None:
            raise RuntimeError("prototype is closed or has uncertain native lifetime") from self.failed
        if torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError("prototype requires its original serialized CUDA stream")
        current = (self.x, self.picks, self.out, self.bank.gate.data, self.bank.up.data, self.bank.down.data,
                   *(part.weight for part in self.bank.shared), *((self.qx, self.qh, self.gate, self.up, self.hidden, self.status) if not self.exact_only else ()))
        if len(current) != len(self.bindings) or any(t is not old or _signature(t) != signature
                for t, (old, signature) in zip(current, self.bindings)):
            raise RuntimeError("prototype lost original tensor/storage bindings")
        now = (self.bank.count, self.bank.width, self.bank.dims, self.bank.gate.ggml_type, self.bank.up.ggml_type,
               self.bank.down.ggml_type, self.bank.gate.column_offset, self.bank.up.column_offset,
               self.bank.down.column_offset, self.bank.limit, self.bank.swiglu)
        parts = (self.bank.gate, self.bank.up, self.bank.down, *self.bank.shared)
        if len(parts) != len(self.weights) or any(value is not old for value, old in zip(parts, self.weights)):
            raise RuntimeError("prototype lost original packed/shared owners")
        if tuple((part.n, part.k, part.implementation) for part in self.bank.shared) != self.shared_policy:
            raise RuntimeError("prototype shared exact policy changed")
        if now != self.layout:
            raise RuntimeError("prototype packed layout changed")

    def _run(self):
        if self.exact_only:
            return self.bank.forward_exact(self.x, self.picks, self.out)
        module = extension()
        module.quantize(self.x, self.qx, 0, self.status)
        module.project(self.qx, self.picks, self.bank.gate.data, self.bank.gate.ggml_type, False, self.gate, self.status)
        module.project(self.qx, self.picks, self.bank.up.data, self.bank.up.ggml_type, False, self.up, self.status)
        module.activate(self.gate, self.up, self.hidden)
        module.quantize(self.hidden.view(-1, 80), self.qh, self.bank.down.column_offset, self.status)
        module.project(self.qh, self.picks, self.bank.down.data, 20, True, self.out, self.status)
        self.bank._write_shared(self.x, self.picks, self.out)  # Original exact shared arithmetic, no weight conversion.
        module.mask(self.out, self.status)  # Stream-ordered final device veto; no hot host status read.
        return self.out

    def run(self):
        self._validate()
        if torch.cuda.is_current_stream_capturing() and self.graph_owner is None:
            raise RuntimeError("prototype capture needs its explicit graph borrower")
        try:
            return self._run()
        except BaseException as error:
            self.failed = error
            raise

    def reset_after_fence(self):
        self._validate()
        if self.graph_owner is not None:
            raise RuntimeError("close graph borrower before cold status recovery")
        self.stream.synchronize()
        if self.status is not None:
            self.status.zero_()

    def capture(self):
        self._validate()
        if self.graph_owner is not None or self.stream.cuda_stream == 0:
            raise RuntimeError("one graph borrower requires a preselected nondefault construction stream")
        graph = Graph(self)
        self.graph_owner = graph  # Retain uncertain accepted capture before any native graph operation.
        graph.prepare()
        return graph

    def close(self):
        if self.closed:
            return
        if self.graph_owner is not None:
            raise RuntimeError("prototype graph borrower must close first")
        self.stream.synchronize()
        self.closed = True
        self.bank = self.x = self.picks = self.out = None
        self.bindings = self.weights = self.weight_bindings = self.scratch = ()
        self.qx = self.qh = self.gate = self.up = self.hidden = self.status = None


class Graph:
    def __init__(self, owner):
        self.owner, self.graph = owner, None
        self.closed = False
        self.capture_errors = [None]

    def prepare(self):
        self.owner.run()
        self.owner.stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        primary = None
        self.graph.capture_begin(capture_error_mode="thread_local")
        try:
            self.owner.run()
        except BaseException as error:
            primary = error
        try:
            self.graph.capture_end()
        except BaseException as error:
            self.capture_errors[0] = error
        if self.capture_errors[0] is not None:
            raise_failures(primary, self.capture_errors)
        if primary is not None:
            raise primary

    def replay(self):
        if self.closed:
            raise RuntimeError("prototype graph is closed")
        self.owner._validate()
        self.graph.replay()
        return self.owner.out

    def close(self):
        if self.closed:
            return
        self.owner.stream.synchronize()
        if self.graph is not None:
            self.graph.reset()
        self.graph = None
        self.owner.graph_owner = None
        self.owner = None
        self.closed = True

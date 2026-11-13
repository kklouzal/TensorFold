"""Unqualified owned Half frontend/epilogue for the explicit G3 component.

This is not a serving policy. It changes normalized q/k storage to unscaled
Half and supplies direct FP32 raw gates. The core owns its one query scale.
No conv history or recurrent State is mutated or published here. Full-prompt
frontend generation uses the original immutable three-row convolution history.

Preparation pays a fixed capacity and resolves kernels before binding. A bound
invocation borrows every input/output on one CPU thread and CUDA stream through
positive completion, including the independently bound core. No concurrent
workspace reuse, storage replacement, capture, or retry after launch failure is
permitted. The caller supplies finite projection/weights/raw core output and a
finite negative GGUF A_neg, or HF A_log whose negative exponential is finite.
The producer must prove the direct rate*softplus gate is finite/nonpositive;
value-domain proof belongs to the typed weight/producer boundary, not a GPU
readback at each launch. Native and model/quality/capacity/P3 gates are separate.
"""

import math
import threading
from dataclasses import dataclass

from .gdn_value8 import MAX_ROWS, _span

NUMERICAL_MODE = "g3-half-unscaled-qk-raw-gate-rounded-beta-v1"


def layout(capacity):
    if type(capacity) is not int or not 1 <= capacity <= MAX_ROWS:
        raise ValueError("G3 Half boundaries require capacity1..65535")
    return {"capacity": capacity, "frontend_bytes": capacity*2608,
            "raw_output_bytes": capacity*1536, "device_bytes": capacity*4144,
            "numerical_mode": NUMERICAL_MODE, "qualification": "private-unqualified"}


@dataclass(frozen=True)
class Workspace:
    capacity: int
    q: object
    k: object
    v: object
    raw_gate: object
    beta: object
    raw_output: object


def allocate_workspace(capacity, device, *, device_budget_bytes):
    facts = layout(capacity)
    if type(device_budget_bytes) is not int or device_budget_bytes < facts["device_bytes"]:
        raise MemoryError("explicit G3 Half frontend/raw-output credit is insufficient")
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("G3 Half workspace requires its actual CUDA device")
    return Workspace(capacity,
        torch.empty((1,capacity,2,128), dtype=torch.float16, device=device),
        torch.empty((1,capacity,2,128), dtype=torch.float16, device=device),
        torch.empty((1,capacity,6,128), dtype=torch.float16, device=device),
        torch.empty((1,capacity,6), dtype=torch.float32, device=device),
        torch.empty((1,capacity,6), dtype=torch.float32, device=device),
        torch.empty((1,capacity,6,128), dtype=torch.float16, device=device))


def _signature(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)


def _validate_operands(workspace, projection, conv, weight, parameter, bias, norm, output, sums, device):
    import torch

    if type(workspace) is not Workspace:
        raise TypeError("G3 Half boundaries require their paid workspace owner")
    layout(workspace.capacity)
    rows = projection.shape[0] if type(projection) is torch.Tensor and projection.dim() == 2 else 0
    if not 1 <= rows <= workspace.capacity:
        raise ValueError("G3 Half invocation exceeds funded capacity")
    specs = [(projection,(rows,2060),torch.float16), (conv,(3,1280),torch.float16),
             (weight,(1280,4),torch.float16), (parameter,(6,),torch.float32), (bias,(6,),torch.float32),
             (norm,(128,),torch.float16), (output,(rows,768),torch.float16), (sums,(rows,24),torch.float32),
             (workspace.q,(1,workspace.capacity,2,128),torch.float16),
             (workspace.k,(1,workspace.capacity,2,128),torch.float16),
             (workspace.v,(1,workspace.capacity,6,128),torch.float16),
             (workspace.raw_gate,(1,workspace.capacity,6),torch.float32),
             (workspace.beta,(1,workspace.capacity,6),torch.float32),
             (workspace.raw_output,(1,workspace.capacity,6,128),torch.float16)]
    for tensor,shape,dtype in specs:
        if (type(tensor) is not torch.Tensor or tuple(tensor.shape) != shape or tensor.dtype != dtype
                or tensor.device != device or not tensor.is_contiguous() or tensor.requires_grad
                or tensor.is_conj() or tensor.is_neg()):
            raise ValueError("G3 Half operands require exact contiguous plain same-device forward-only layout")
    reads = [projection,conv,weight,parameter,bias,norm]
    writes = [output,sums,workspace.q,workspace.k,workspace.v,workspace.raw_gate,workspace.beta,workspace.raw_output]
    for index,tensor in enumerate(writes):
        lo,hi = _span(tensor)
        if any(lo < _span(other)[1] and _span(other)[0] < hi for other in reads+writes[:index]):
            raise ValueError("G3 Half output/workspace storage must be disjoint from all inputs/writes")
    return rows, tuple(tensor for tensor,_,_ in specs)


@dataclass(frozen=True)
class Prepared:
    module: object
    device: object

    def bind(self, workspace, projection, conv, weight, bias, norm, eps, output, sums, *, a_log=None, a_neg=None):
        import torch

        if (a_log is None) == (a_neg is None):
            raise ValueError("G3 Half requires exactly one typed A_log or A_neg source")
        if type(eps) not in (int,float) or not math.isfinite(eps) or not 0 < eps <= 3.4028234663852886e38:
            raise ValueError("G3 Half requires finite positive FP32 epsilon")
        # Conversion itself must retain positivity in the native FP32 ABI.
        import struct
        if struct.unpack("f",struct.pack("f",eps))[0] <= 0:
            raise ValueError("G3 Half epsilon underflows its FP32 ABI")
        _context(self.device)
        parameter = a_log if a_neg is None else a_neg
        rows,tensors = _validate_operands(workspace,projection,conv,weight,parameter,bias,norm,output,sums,self.device)
        views = (workspace.q[:,:rows],workspace.k[:,:rows],workspace.v[:,:rows],
                 workspace.raw_gate[:,:rows],workspace.beta[:,:rows],workspace.raw_output[:,:rows])
        native_views = tuple(tensor[0] for tensor in
                             (workspace.q,workspace.k,workspace.v,workspace.raw_gate,workspace.beta,workspace.raw_output))
        return Bound(self, workspace, rows, tensors, tuple(map(_signature,tensors)), views, native_views, tuple(map(_signature,(*views,*native_views))), (output,sums),
                     float(eps), a_neg is not None, threading.current_thread(),
                     torch.cuda.current_stream(self.device).cuda_stream)


def _context(device):
    import torch

    if (device.type != "cuda" or torch.cuda.current_device() != device.index
            or torch.cuda.is_current_stream_capturing() or not torch.is_inference_mode_enabled()):
        raise RuntimeError("G3 Half boundaries require uncaptured inference on the prepared current CUDA device")


@dataclass
class Bound:
    prepared: Prepared
    workspace: Workspace
    rows: int
    tensors: tuple
    signatures: tuple
    core_operands: tuple
    native_views: tuple
    view_signatures: tuple
    output_pair: tuple
    eps: float
    negative_decay: bool
    thread: object
    stream: int
    error: BaseException | None = None

    def _ready(self):
        import torch

        if self.error is not None:
            raise RuntimeError("G3 Half launch failed; retain owners and contain State without retry") from self.error
        _context(self.prepared.device)
        if (threading.current_thread() is not self.thread
                or torch.cuda.current_stream(self.prepared.device).cuda_stream != self.stream):
            raise RuntimeError("G3 Half binding requires its original thread and CUDA stream")
        if (tuple(map(_signature,self.tensors)) != self.signatures
                or tuple(map(_signature,(*self.core_operands,*self.native_views))) != self.view_signatures):
            raise RuntimeError("G3 Half bound storage/layout changed; rebind after retirement")

    def front(self):
        """Enqueue once against original conv history; core operands retain owners.

        q/k are normalized and unscaled Half. raw_gate is direct FP32 rate times
        softplus, beta is Half(sigmoid) widened to FP32. No State publication.
        """
        import torch

        self._ready()
        p,conv,weight,parameter,bias,_,_,_,q,k,v,gate,beta,_ = self.tensors
        try:
            self.prepared.module.chunk64_front(p,conv,weight,parameter,bias,self.rows,
                                              *self.native_views[:5],self.negative_decay)
            torch.autograd.graph.increment_version((q,k,v,gate,beta))
        except BaseException as error:
            self.error = error
            raise
        return self.core_operands

    def back(self):
        """Consume the independently completed/enqueued core's separate raw Half.

        The caller establishes producer ordering on this same stream. Successful
        enqueue does not prove completion; keep this binding through retirement.
        """
        import torch

        self._ready()
        p,_,_,_,_,norm,out,sums,*_ = self.tensors
        try:
            self.prepared.module.chunk64_back(p,norm,self.eps,self.rows,self.native_views[5],out,sums)
            torch.autograd.graph.increment_version((out,sums))
        except BaseException as error:
            self.error = error
            raise
        return self.output_pair


def prepare(device):
    """Resolve only by explicit opt-in, outside requests and before admission."""
    from .gdn_value8 import prepare as prepare_value8

    native = prepare_value8(device)
    native.module.prepare_chunk64_boundaries()
    return Prepared(native.module,native.device)

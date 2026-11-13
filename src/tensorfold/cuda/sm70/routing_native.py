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


PREPARED_LOGITS_POLICY = 'owned-serial-router-v1'


def prepared_logits_identity():
    return {'policy': PREPARED_LOGITS_POLICY, 'supported_roles': ['decode', 'mtp'],
            'input': 'fp16[2..256,2560]', 'weight': 'aligned512-contiguous-fp32[513,2560]',
            'output': 'fp32[rows,513]', 'arithmetic': 'existing-serial-gemv-five-k512-tiles-16-lane-left-fold',
            'compiler': '-O3,--fmad=false;explicit-__fmaf_rn',
            'one_row': 'existing-serial-reference', 'prefill': 'existing-independent_rows-false',
            'selection_and_expert_weights': 'existing-selected-provider-unchanged',
            'required_contract': 'exact-pinned-serial-logit-and-routing-bits',
            'reference_bit_identity': False, 'qualification': 'private-unqualified'}


def _router_tensor(tensor, device, dtype, shape):
    import torch
    return (type(tensor) is torch.Tensor and tensor.device == device and tensor.dtype == dtype
            and tensor.shape == shape and tensor.is_contiguous() and not tensor.requires_grad
            and not tensor.is_conj() and not tensor.is_neg())


class PreparedSerialRouter:
    """One Weights owner's resident logit function; no process selector/workspace.

    Weights and Buffers retain their allocations through caller-owned stream
    completion. Close is called only after that owner's work has retired.
    """
    def __init__(self, module, device, role_rows):
        import torch
        roles = {role: rows for role, rows in role_rows.items() if role in ('decode', 'mtp')}
        if ('decode' not in roles or any(type(rows) is not int or not 1 <= rows <= 256 for rows in roles.values())):
            raise ValueError('prepared serial router requires bounded decode/MTP role envelopes')
        self.module, self.device, self.logits = module, torch.device(device), module.prepared_logits
        self.role_rows, self.weights, self.buffers = roles, None, {}
        self.weight_owner = None
        self.closed = False

    def record(self):
        root = Path(__file__).parent
        return {**prepared_logits_identity(), 'module': self.module.__name__,
                'module_path': self.module.__file__, 'role_rows': dict(self.role_rows),
                'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in ('routing_native.py', 'routing.cu', 'routing_bindings.cpp')},
                'persistent_tensor_bytes': 0, 'temporary_tensor_bytes': 0,
                'residency': 'serial-logit-function-prepared-before-state-admission',
                'binding': 'per-weights-and-role-buffers;no-process-policy-mutation',
                'invocation_evidence': False}

    def bind_weights(self, weights):
        import torch
        if self.closed or self.weight_owner is not None:
            raise RuntimeError('prepared serial router weights bind once before State admission')
        if (weights.device != self.device or weights.cfg.hidden != 2560
                or weights.cfg.experts != 512 or weights.cfg.top_k != 10
                or weights.cfg.execution_dtype != 'float16'):
            raise ValueError('prepared serial router requires the actual2560/512/top10 weights')
        roles = {'decode': [layer.moe.router for layer in weights.layers if layer.moe is not None]}
        if 'mtp' in self.role_rows:
            if weights.mtp is None or weights.mtp.layer.moe is None:
                raise ValueError('prepared serial router MTP role lacks its actual router weight')
            roles['mtp'] = [weights.mtp.layer.moe.router]
        owners = {}
        for role, tensors in roles.items():
            if not tensors or any(not _router_tensor(t, self.device, torch.float32, (513, 2560))
                                  or t.data_ptr() % 512 for t in tensors):
                raise ValueError('prepared serial router requires unchanged aligned FP32 router owners')
            owners[role] = {id(t): (t, t.data_ptr()) for t in tensors}
        self.weights = owners  # Publish only after all role weights validate.
        self.weight_owner = weights

    def bind_buffers(self, buffers):
        import torch
        role = buffers.owner_role
        rows = self.role_rows.get(role)
        if self.closed or self.weights is None or role in self.buffers:
            raise RuntimeError('prepared serial router buffers bind once after their weight owner')
        if (buffers.prefill or rows is None or buffers.rows != rows
                or not _router_tensor(buffers.mixed, self.device, torch.float16, (rows, 2560))
                or not _router_tensor(buffers.moe.logits, self.device, torch.float32, (rows, 513))
                or torch._C._overlaps(buffers.moe.logits, buffers.mixed)
                or any(torch._C._overlaps(buffers.moe.logits, t)
                       for t, _pointer in self.weights[role].values())):
            raise ValueError('prepared serial router requires its actual disjoint role buffers')
        bound = _BoundSerialRouter(self, role, buffers.mixed, buffers.moe.logits)
        self.buffers[role] = bound
        return bound

    def validate_adoption(self, weights, decoder):
        if (self.closed or self.weight_owner is not weights
                or getattr(weights, '_private_prepared_router', None) is not self or self.weights is None):
            raise ValueError('prepared serial router differs from its actual Weights owner')
        for role, buffers in (('decode', decoder.buf), ('mtp', decoder.mbuf)):
            if role not in self.role_rows:
                continue
            bound = self.buffers.get(role)
            if (bound is None or getattr(buffers, 'router_logits', None) is not bound
                    or buffers.mixed is not bound.source or buffers.moe.logits is not bound.output):
                raise ValueError('prepared serial router differs from its actual role Buffers owner')
        if getattr(decoder.pbuf, 'router_logits', None) is not None:
            raise ValueError('prepared serial router must leave prefill independent')

    def close(self):
        self.closed = True
        self.weights = None
        self.weight_owner = None
        self.buffers.clear()


class _BoundSerialRouter:
    def __init__(self, prepared, role, source, output):
        self.prepared, self.role, self.source, self.output = prepared, role, source, output
        self.rows = source.shape[0]
        self.pointers = source.data_ptr(), output.data_ptr()

    def run(self, x, weight, output):
        import torch
        owner = self.prepared
        if owner.closed:
            raise RuntimeError('prepared serial router owner is closed')
        rows = x.shape[0]
        if rows < 2:
            return False  # R1 retains the serial reference; empty work also stays incumbent.
        if (torch.is_grad_enabled() or any(t.requires_grad for t in (x, weight, output))
                or output.is_inference() and not torch.is_inference_mode_enabled()
                or any(t.is_conj() or t.is_neg() for t in (x, weight, output))):
            return False
        retained = owner.weights[self.role].get(id(weight))
        if (retained is None or retained[0] is not weight or retained[1] != weight.data_ptr()
                or rows > self.rows or x.data_ptr() != self.pointers[0] or output.data_ptr() != self.pointers[1]
                or self.source.data_ptr() != self.pointers[0] or self.output.data_ptr() != self.pointers[1]
                or self.source.shape != (self.rows, 2560) or self.output.shape != (self.rows, 513)):
            raise ValueError('prepared serial router operands differ from their cold ownership binding')
        owner.logits(x, weight, output)  # Retained function; no loader/device discovery here.
        if not output.is_inference():
            torch.autograd.graph.increment_version((output,) * rows)
        return True


def prepare_serial_router(device, role_rows):
    import torch
    device = torch.device(device)
    if device.type != 'cuda' or torch.cuda.get_device_capability(device) != (7, 0):
        raise ValueError('prepared serial router requires actual SM70 preparation')
    with torch.cuda.device(device):
        module = extension()
        if not all(callable(getattr(module, name, None)) for name in ('prepare_logits', 'prepared_logits')):
            raise ValueError('prepared serial router requires its built native entry and residency API')
        prepared = PreparedSerialRouter(module, device, role_rows)
        module.prepare_logits()
    return prepared

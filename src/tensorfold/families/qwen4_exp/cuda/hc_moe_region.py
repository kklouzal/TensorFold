"""Private TP8 MLP-HC plus pure-MoE capture component, without integration.

The caller issues the prior NCCL gather into canonical g_branch on the original
serialized stream. This region applies mode3 HC writeback/readout and the
original pure MoE body. The caller still owns State authority, MoE partials and
all later collectives. Preparation is startup-only, mutates scratch, and needs
fresh h/g_branch/inj_a inputs before replay. No serving catalogue is enrolled.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from tensorfold.cuda.sm70.router_blas import PROFILES
from .gguf_weights import GGUFConcat, GGUFDense
from .moe_region_catalogue import Bindings, _span
from .moe_region_graphs import MoERegion, region_tensors
from .weight_types import HC


HC_TENSORS = ('h', 'pss', 'normed', 'xs_normed', 'dn', 'dn_mix', 'act',
              'xs_act', 'inj_a', 'inj_m', 'up', 'xs_mixed', 'g_branch')


def validate_hc_region(weights, layer, buffers, rows):
    """Admit only the actual TP8 FP16 GGUF mode3 MLP-HC boundary."""
    from tensorfold.cuda.sm70 import family_ops, routing_native

    comm, cfg = weights.comm, weights.cfg
    if not family_ops._NATIVE_FAMILY or routing_native._NATIVE_LOGITS:
        raise ValueError('HC/MoE component requires the selected native family and canonical router selector')
    if (comm is None or type(comm.world) is not int or comm.world != 8
            or type(comm.rank) is not int or not 0 <= comm.rank < 8
            or weights.meta.get('world') != 8
            or weights.meta.get('sm70_profile') not in PROFILES
            or weights.meta.get('execution_provider') != 'sm70'):
        raise ValueError('HC/MoE component requires its selected SM70 TP8 model/communicator')
    if (type(rows) is not int or not 1 <= rows <= 32
            or any(type(getattr(cfg, key)) is not int or getattr(cfg, key) != value
                   for key, value in (('hidden', 2560), ('streams', 4), ('low', 320)))
            or cfg.execution_dtype != 'float16'
            or type(cfg.eps) not in (float, int) or not math.isfinite(cfg.eps) or cfg.eps <= 0):
        raise ValueError('HC/MoE component requires exact FP16 HC geometry and finite positive epsilon')
    if (buffers.prefill or buffers.hc_plans is not None or buffers.padded_linears is not None
            or buffers.split is not None or weights.x3 is not None):
        raise ValueError('HC/MoE component preserves only original GGUF decode readout')
    mtp = getattr(weights, 'mtp', None)
    if not any(value is layer for value in weights.layers) and (mtp is None or mtp.layer is not layer):
        raise ValueError('HC/MoE layer must be owned by the actual target or MTP model')
    hc = layer.mlp_hc
    if (type(hc) is not HC or hc.inject is not True or not isinstance(hc.down, GGUFConcat)
            or not isinstance(hc.up, GGUFDense) or len(hc.down.parts) != 2
            or hc.down.implementation != 'native32' or hc.up.implementation != 'native32'
            or (hc.down.n, hc.down.k, hc.up.n, hc.up.k) != (324, 10240, 10240, 320)
            or any(not isinstance(part, GGUFDense) or part.implementation != 'native32'
                   or (part.n, part.k) != shape
                   for part, shape in zip(hc.down.parts, ((320, 10240), (4, 10240)), strict=True))):
        raise ValueError('HC/MoE component requires the original separate down/inject/up projections')
    device, capacity = buffers.h.device, buffers.h.shape[0]
    shapes = {'h': (capacity, 10240), 'pss': (capacity, 10, 4),
              'normed': (capacity, 10240), 'xs_normed': (capacity, 320),
              'dn': (capacity, 324), 'dn_mix': (capacity, 320), 'act': (capacity, 320),
              'xs_act': (capacity, 10), 'inj_a': (capacity, 4), 'inj_m': (capacity, 4),
              'up': (capacity, 10240), 'mixed': (capacity, 2560),
              'xs_mixed': (capacity, 80), 'g_branch': (8 * capacity * 2560,)}
    f32 = {'pss', 'xs_normed', 'xs_act', 'xs_mixed', 'g_branch'}
    operands = [getattr(buffers, name) for name in HC_TENSORS] + list(region_tensors(buffers))
    for name, shape in shapes.items():
        tensor = getattr(buffers, name)
        if (type(tensor) is not torch.Tensor or not tensor.is_cuda or tensor.device != device
                or tensor.dtype != (torch.float32 if name in f32 else torch.float16)
                or tuple(tensor.shape) != shape or not tensor.is_contiguous() or tensor.requires_grad):
            raise ValueError('HC/MoE operand does not own its canonical CUDA shape/dtype/storage: ' + name)
    if (capacity < rows or comm.device != device or hc.scale.device != device
            or hc.scale.dtype != torch.float32 or hc.scale.shape != (10240,)):
        raise ValueError('HC/MoE component exceeds rows or has a different normalization weight')
    parameters = (hc.scale, *(part.weight for part in hc.down.parts), hc.up.weight)
    for dense in (*hc.down.parts, hc.up):
        if dense.weight.dtype != torch.float16 or tuple(dense.weight.shape) != (dense.n, dense.k):
            raise ValueError('HC/MoE original projections require their FP16 coefficient layout')
    for tensor in parameters:
        if tensor.device != device or tensor.requires_grad or not tensor.is_contiguous():
            raise ValueError('HC/MoE projections require fixed inference device storage')
    if any(tensor.requires_grad for tensor in operands):
        raise ValueError('HC/MoE component is an inference-only capture')
    for index, tensor in enumerate(operands):
        if any(torch._C._overlaps(tensor, other) for other in operands[index + 1:]):
            raise ValueError('HC/MoE canonical scratch/input operands must not overlap')
        if any(torch._C._overlaps(tensor, parameter) for parameter in parameters):
            raise ValueError('HC/MoE writable scratch must not alias model weights')


class HCModelBindings(Bindings):
    """MoE plus all target MLP-HC bindings, checked once by a forward lease."""
    def __init__(self, weights, buffers):
        super().__init__(weights, buffers)
        self.object(weights, 'comm')
        self.object(weights, 'x3')
        self.scalar(weights.comm, ('world', 'rank', 'device'))
        self.scalar(weights.cfg, ('hidden', 'streams', 'low', 'eps', 'execution_dtype'))
        self.scalar(buffers, ('prefill',))
        for name in ('hc_plans', 'padded_linears', 'split'):
            self.object(buffers, name)
        for name in HC_TENSORS:
            self.tensor(buffers, name)
        for layer in self.layers:
            self.object(layer, 'mlp_hc')
            hc = layer.mlp_hc
            self.scalar(hc, ('inject',))
            self.tensor(hc, 'scale')
            self.object(hc, 'down')
            self.object(hc, 'up')
            self.object(hc.down, 'parts')
            self.scalar(hc.down, ('n', 'k', 'kernel', 'implementation'))
            for dense in (*hc.down.parts, hc.up):
                self.tensor(dense, 'weight')
                self.scalar(dense, ('n', 'k', 'kernel', 'implementation'))
            self.scalar(layer.moe.experts, ('kernel',))
        self.math_policy = self.current_math_policy()
        self.tp_metadata = (weights.meta.get('world'), weights.meta.get('rank'))

    @staticmethod
    def current_math_policy():
        policy = torch.backends.cuda.matmul
        return (policy.allow_fp16_reduced_precision_reduction,
                policy.allow_fp16_accumulation, policy.allow_tf32)

    def validate(self):
        super().validate()
        if (self.weights.meta.get('world'), self.weights.meta.get('rank')) != self.tp_metadata:
            raise RuntimeError('HC/MoE captured TP metadata changed')
        if self.current_math_policy() != self.math_policy:
            raise RuntimeError('HC/MoE captured matmul numerical policy changed')


class _HCBindings(HCModelBindings):
    """The same seals scoped to one owned target or MTP layer."""
    def __init__(self, weights, layer, buffers):
        self.model = weights
        scope = SimpleNamespace(cfg=weights.cfg, layers=(layer,), meta=weights.meta,
                                comm=weights.comm, x3=weights.x3)
        super().__init__(scope, buffers)

    def current_policy(self):
        model = self.model
        mtp = getattr(model, 'mtp', None)
        return (super().current_policy(), id(model.cfg), id(model.meta), id(model.comm), id(model.x3),
                tuple(id(layer) for layer in model.layers), id(mtp), id(getattr(mtp, 'layer', None)))


class HCReadoutMoERegion(MoERegion):
    """One fixed layer/R, mode3 component; no State or collective is captured.

    Prepare is quiescent startup work, not a live-frame invocation: warm and
    capture change h/pss and scratch. Refresh inputs afterward. Metadata/storage
    guards do not replace the caller's producer-stream and model lifetime lease.
    """
    def __init__(self, weights, layer, buffers, rows, *, side=None, pool=None, capture_router=None):
        validate_hc_region(weights, layer, buffers, rows)
        super().__init__(weights, layer, buffers, rows, side=side, pool=pool, capture_router=capture_router)
        self.layer_index = layer.index
        self.capture_device = buffers.mixed.device
        self.bindings = _HCBindings(weights, layer, buffers)
        self.branch = buffers.g_branch[:8 * rows * 2560].view(8, rows, 2560)
        self.injection = buffers.inj_a[:rows]
        self.branch_signature, self.injection_signature = _span(self.branch), _span(self.injection)

    def prepare(self):
        if self.prepared or self.graph is not None or self.closed or self.retiring or self.failure is not None:
            raise RuntimeError('HC/MoE component cannot prepare twice or after retirement/failure')
        try:
            self.bindings.validate()  # No cold GPU work after constructor-time bindings changed.
            super().prepare()
        except BaseException as error:
            if self.failure is None:
                self.failure = error
            raise

    def _body(self):
        from .forward import hc_block

        cfg, buffers = self.weights.cfg, self.buffers
        hc_block(self.layer.mlp_hc, buffers, self.rows, cfg.eps, cfg.streams, cfg.low,
                 3, self.injection, buffers.inj_m, buffers.h, branch=self.branch)
        super()._body()  # Original moe_block(_local_only=True), before partial/NCCL.

    def replay(self, mode, branch, injection):
        if not self.prepared or self.closed or self.retiring or self.failure is not None:
            raise RuntimeError('HC/MoE component is not an admitted prepared owner') from self.failure
        try:
            if type(mode) is not int or mode != 3:
                raise ValueError('HC/MoE replay requires the actual TP8 mode3 pending result')
            if (type(branch) is not torch.Tensor or type(injection) is not torch.Tensor
                    or _span(branch) != self.branch_signature or _span(injection) != self.injection_signature):
                raise ValueError('HC/MoE replay requires canonical current g_branch and inj_a spans')
            if self.bindings is None:
                raise RuntimeError('HC/MoE component bindings have retired')
            self.bindings.validate()
            super().replay()
        except BaseException as error:
            if self.failure is None:
                self.failure = error
            raise

    def replay_prevalidated(self):
        """Trusted catalogue only, under its issued HC cut and forward lease.

        The caller has validated HCModelBindings for this exclusive frame and
        checked this exact layer/R/mode3/branch/injection cut before suppressing
        original HC. Its immediately following run consumes that cut exactly
        once; any unconsumed/failed cut poisons the frame, never eager-retries.
        This method supplies no HC capability proof itself. The original MoE
        lifecycle, external owner and serialized-stream checks still execute.
        """
        MoERegion.replay(self)

    def close(self):
        if self.closed:
            return
        self.retiring = True
        try:
            # A refused external rebinding must not redirect retirement to a
            # different device. The graph still owns its original operands.
            torch.cuda.synchronize(self.capture_device)
            self._release_fenced()
        except BaseException as error:
            if self.failure is None:
                self.failure = error
            raise

    def _release_fenced(self):
        super()._release_fenced()
        self.bindings = self.branch = self.injection = None
        self.weights = self.layer = self.buffers = None

    def record(self):
        return {'rows': self.rows, 'layer': self.layer_index, 'prepared': self.prepared,
                'closed': self.closed, 'retiring': self.retiring,
                'replays': None if self.counter_overflowed else self.hits, 'mode': 3, 'world': 8,
                'scope': 'private MLP-HC plus pure-MoE component; no State/collective/catalogue or full-model qualification'}

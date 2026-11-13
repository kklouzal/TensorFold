"""Explicit per-engine native-family trial; attention and expert policy stay separate.

Only private TP8/ETP4 FP16 NONE/MTP owners may adopt this capability. Calls borrow
ordinary Buffers/State storage on the decoder's current stream; there is no new
workspace, global selector, request-time capability discovery or tensor cache.
The existing checked native ABI owns geometry validation. Numerical qualification
is required: existing FP32 reductions/FP16 stores do not imply Torch bit identity.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

POLICY = 'owned-native-family-v1'
OPERATIONS = ('hc_writeback', 'hc_normed', 'hc_act', 'hc_reduce_act', 'hc_mix',
              'rmsnorm', 'attn_gate', 'ple_embed_bf16', 'ple_gate', 'ple_conv',
              'add_streams', 'moe_partial', 'shift_windows')


def numerical_identity():
    return {'policy': POLICY, 'operations': list(OPERATIONS),
            'scope': 'target-and-mtp;prefill-eager-and-exact-capture',
            'operands': 'fp16;model-scales-and-route-weights-fp32',
            'arithmetic': 'existing-sm70-family-fp32-reductions-fp16-stores',
            'compiler': '-O3,--fmad=false', 'reference_bit_identity': False,
            'attention_projection_and_cache': 'existing-owner-provider',
            'expert_kernels_and_linears': 'existing-selected-providers',
            'qualification': 'private-unqualified'}


def _published(*outputs, count=1):
    # Raw extension writes do not publish ATen mutation metadata. The Torch
    # reference has one copy_ per output; aliases share the same version counter.
    for output in outputs:
        if output is not None and not output.is_inference():
            torch.autograd.graph.increment_version([output] * count)


class PreparedFamily:
    def __init__(self, module, device):
        self.module, self.device = module, torch.device(device)

    def record(self):
        root = Path(__file__).parent
        return {**numerical_identity(), 'module': self.module.__name__,
                'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in ('owned_family.py', 'family.cu', 'family_bindings.cpp', 'torch_reduce.cuh')},
                'persistent_tensor_bytes': 0, 'temporary_tensor_bytes': 0,
                'residency': 'all-selected-entrypoints-warmed-before-state-admission',
                'binding': 'per-weights-and-buffers;no-process-policy-mutation'}

    def hc_writeback(self, h, hout, pss, streams, mode, branch=None, inject=None, y=None, wts=None):
        self.module.writeback(h, hout, pss, streams, mode, branch, inject, y, wts)
        _published(hout, pss)

    def hc_normed(self, h, pss, scale, normed, xs, streams, eps):
        self.module.normed(h, pss, scale, normed, xs, streams, eps)
        _published(normed, xs)

    def hc_act(self, dn, act, xs, inj, streams, low):
        self.module.act(dn, act, xs, inj, streams, low)
        _published(act, xs, inj)

    def hc_reduce_act(self, dn, act, xs, inj, streams, low):
        self.hc_act(dn, act, xs, inj, streams, low)

    def hc_mix(self, up, normed, mixed, xs, streams):
        self.module.mix(up, normed, mixed, xs, streams)
        _published(mixed, xs)

    def rmsnorm(self, x, weight, eps, group=None, out=None, xs=None):
        # All model callers supply outputs. Keep that resource contract explicit.
        if out is None or xs is None:
            raise ValueError('owned native RMS requires caller-owned outputs')
        self.module.rms(x, weight, out, xs, x.shape[-1] if group is None else group, eps)
        _published(out, xs)
        return out, xs

    def attn_gate(self, value, projection, out, xs, *, q_heads, head_dim):
        self.module.gate(value, projection, out, xs, q_heads, head_dim)
        _published(out, xs)

    def ple_embed_bf16(self, rows, values, heads, dh, out, xs, *, scale=1.0):
        # The historical API name is retained; this owned region consumes FP16.
        self.module.ple_embed(values, out, xs, rows, heads, dh, scale)
        _published(out, xs)

    def ple_gate(self, keys, vals, h, norm_key, norm_query, gated, pss, eps, streams,
                 valid_rows=None, independent_rows=False):
        self.module.ple_gate(keys, vals, h, norm_key, norm_query, gated, pss, streams,
                             eps, valid_rows, independent_rows)
        _published(gated, pss, count=h.shape[0] if independent_rows else 1)

    def ple_conv(self, gated, pss, norm, tail, weight, h, hout, nrow, eps, streams, dilation):
        self.module.ple_conv(gated, pss, norm, tail, weight, h, hout, nrow, streams, dilation, eps)
        _published(hout, nrow)

    def add_streams(self, embedding, hidden, out, streams):
        self.module.add(embedding, hidden, out, streams)
        _published(out)

    def moe_partial(self, values, weights, out, rows):
        self.module.partial(values, weights, out, rows)
        _published(out)

    def shift_windows(self, old, new, keep, channels):
        if old.shape[1] > 16:
            # Preserve the existing declared native capability boundary.
            from .family_ops import _reference
            return _reference['shift_windows'](old, new, keep, channels)
        self.module.shift(old, new, keep, channels)
        if old.numel():
            _published(old)


def prepare(device):
    """Load/warm every selected function inside the owner's residency window.

    All selected kernels are runtime-parameterized (no dtype/row template JIT).
    One initialized fixed-geometry frame therefore forces their code residency;
    this is preparation, not an operator/model correctness or performance gate.
    Warmup temporaries retire after positive stream completion, before admission.
    """
    device = torch.device(device)
    if device.type != 'cuda' or torch.cuda.get_device_capability(device) != (7, 0):
        raise ValueError('owned native family requires actual SM70 preparation')
    from .family_native import extension
    owner = PreparedFamily(extension(), device)
    with torch.cuda.device(device), torch.no_grad():
        def half(*shape):
            return torch.zeros(shape, device=device, dtype=torch.float16)
        def full(*shape):
            return torch.ones(shape, device=device, dtype=torch.float32)
        h, out, normed = half(1, 10240), half(1, 10240), half(1, 10240)
        pss, scale, xs = full(1, 10, 4), full(10240), full(1, 320)
        inject, branch, partial, slots, weights = half(1, 4), half(1, 2560), full(8, 1, 2560), half(1, 11, 2560), full(1, 11)
        for mode in range(5):
            owner.hc_writeback(h, out, pss, 4, mode,
                branch=branch if mode == 1 else partial if mode in (3, 4) else None,
                inject=inject if mode else None, y=slots if mode == 2 else None,
                wts=weights if mode == 2 else None)
        owner.hc_normed(out, pss, scale, normed, xs, 4, 1e-6)
        act, ax = half(1, 320), full(1, 10)
        owner.hc_act(full(1, 324), act, ax, inject, 4, 320)
        owner.hc_reduce_act(full(8, 1, 320), act, ax, None, 4, 320)
        mixed, mx = half(1, 2560), full(1, 80)
        owner.hc_mix(out, normed, mixed, mx, 4)
        owner.rmsnorm(h, scale, 1e-6, out=normed, xs=xs)
        owner.attn_gate(half(1, 3, 256), half(1, 1536), half(1, 768), full(1, 24), q_heads=3, head_dim=256)
        owner.ple_embed_bf16(1, half(4, 640), 4, 640, mixed, mx)
        for independent in (False, True):
            owner.ple_gate(h, mixed, h, scale, scale, out, full(1, 4), 1e-6, 4,
                valid_rows=torch.ones((1,), device=device, dtype=torch.int32), independent_rows=independent)
        owner.ple_conv(out, full(1, 4), scale, half(2, 10240), half(10240, 3), h, normed, out, 1e-6, 4, 1)
        owner.add_streams(mixed, half(4, 2560), h, 4)
        owner.moe_partial(slots, weights, full(1, 2560), 1)
        owner.shift_windows(half(1, 3, 2560), half(1, 1, 2560), 1, 2560)
        torch.cuda.synchronize(device)
    return owner

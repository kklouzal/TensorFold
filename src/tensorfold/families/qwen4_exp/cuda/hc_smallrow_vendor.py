"""Private HC-only logical-M vendor candidate over original FP16 operands.

No matrix storage is copied or merged. Higher rows, unsupported layouts,
autograd/alias cases and prefill use the original matrix owners. Different
vendor M dispatch is a numerical experiment, not an exact-M32 optimization.
"""
from dataclasses import replace
import sys

import torch

from tensorfold.cuda.sm70 import gguf_ops

from .gguf_weights import GGUFDense, GGUFConcat

HOST_ALLOWANCE = 256 << 10
POLICY = 'hc-only-vendor-logical-m1-3-original-f16-operands-v1'


class _SmallRow:
    hc_smallrow_vendor = True

    @property
    def n(self):
        return self.original.n

    @property
    def k(self):
        return self.original.k

    def _eligible(self, x, out, f32):
        if (self.original.implementation != 'native32' or type(x) is not torch.Tensor
                or x.layout != torch.strided or x.ndim != 2 or not 1 <= x.shape[0] <= 3
                or x.shape[1] != self.k or x.dtype != torch.float16 or not x.is_cuda
                or not x.is_contiguous() or x.is_neg() or x.is_conj() or x.requires_grad):
            return False
        dtype = out.dtype if type(out) is torch.Tensor else torch.float32 if f32 else torch.float16
        if out is not None and type(out) is not torch.Tensor or dtype != self.output_dtype:
            return False
        if out is not None and (out.layout != torch.strided or out.shape != (x.shape[0], self.n)
                or out.device != x.device or not out.is_contiguous() or out.requires_grad
                or out.is_neg() or out.is_conj()
                or out.is_inference() and not torch.is_inference_mode_enabled()
                or torch._C._overlaps(out, x)):
            return False
        for matrix, weight in zip(self.matrices, self.weights):
            if (type(weight) is not torch.Tensor or weight.dtype != torch.float16
                    or weight.shape != (matrix.n, matrix.k) or weight.device != x.device
                    or not weight.is_contiguous() or weight.requires_grad
                    or weight.is_neg() or weight.is_conj()
                    or out is not None and torch._C._overlaps(out, weight)):
                return False
        return True

    def matmul(self, x, *, out=None, f32=False):
        if not self._eligible(x, out, f32):
            return self.original.matmul(x, out=out, f32=f32)
        if out is None:
            out = torch.empty((x.shape[0], self.n), dtype=self.output_dtype, device=x.device)
        version = None if out.is_inference() else out._version
        gguf_ops.hc_smallrow(x, self.weights, out)
        # The raw copy is opaque to ATen's version machinery. Publish its
        # mutation for every normal Tensor, including saved non-grad inputs.
        if version is not None and out._version == version:
            torch.autograd.graph.increment_version(out)
        return out

    def prefill(self, x, out=None):
        return self.original.prefill(x, out=out)

    def __call__(self, x, out=None):
        # HC's non-qmm prefill calls the matrix owner directly. Decode uses
        # partials/matmul, so preserve this existing generic/prefill protocol.
        return self.original(x, out)


class Dense(_SmallRow, GGUFDense):
    def __init__(self, original, *, down):
        self.original = original
        self.output_dtype = torch.float32 if down else torch.float16

    @property
    def weight(self):
        return self.original.weight

    @property
    def weights(self):
        return (self.original.weight,)

    @property
    def matrices(self):
        return (self.original,)


class Concat(_SmallRow, GGUFConcat):
    output_dtype = torch.float32

    def __init__(self, original):
        self.original = original

    @property
    def parts(self):
        return self.original.parts

    @property
    def fused(self):
        return self.original.fused

    @property
    def weights(self):
        return tuple(part.weight for part in self.original.parts)

    @property
    def matrices(self):
        return self.original.parts


def bind_hc(hc):
    """Only an explicit HC object may acquire the private numeric adapters."""
    down, up = hc.down, hc.up
    if hc.prefill_down is not down or hc.prefill_up is not up:
        raise ValueError('HC small-row requires the original GGUF prefill aliases')
    if type(up) is not GGUFDense or (up.n, up.k) != (10240, 320):
        raise ValueError('HC small-row requires original up geometry')
    if hc.inject:
        if (type(down) is not GGUFConcat or len(down.parts) != 2
                or any(type(part) is not GGUFDense for part in down.parts)
                or [(part.n, part.k) for part in down.parts] != [(320, 10240), (4, 10240)]):
            raise ValueError('HC small-row requires separate original down/inject parts')
        bound_down = Concat(down)
    else:
        if type(down) is not GGUFDense or (down.n, down.k) != (320, 10240):
            raise ValueError('HC small-row requires original mixer down geometry')
        bound_down = Dense(down, down=True)
    return replace(hc, down=bound_down, up=Dense(up, down=False))


def enroll(weights):
    """Construction-only; called before any State, Buffer or graph exists."""
    from .prefix_policy import HC_SMALLROW_VENDOR_PROFILE

    if weights.meta.get('sm70_profile') != HC_SMALLROW_VENDOR_PROFILE:
        raise ValueError('HC small-row requires its explicit numerical profile')
    layers = list(weights.layers) + ([] if weights.mtp is None else [weights.mtp.layer])
    if len(weights.layers) != 48 or len(layers) > 49:
        raise ValueError('HC small-row requires the admitted target/stock-MTP layer geometry')
    # Resolve all geometry/aliases before mutating the model's HC references.
    prepared = [(layer, bind_hc(layer.attn_hc), bind_hc(layer.mlp_hc)) for layer in layers]
    mixer = bind_hc(weights.mixer)
    mtp_mixer = None if weights.mtp is None else bind_hc(weights.mtp.mixer)
    hcs = [hc for _, attention, mlp in prepared for hc in (attention, mlp)] + [mixer]
    if mtp_mixer is not None:
        hcs.append(mtp_mixer)
    controlled = (sys.getsizeof(prepared) + sum(sys.getsizeof(item) for item in prepared)
        + sys.getsizeof(hcs) + sum(sys.getsizeof(hc) + sys.getsizeof(vars(hc))
            + sys.getsizeof(hc.down) + sys.getsizeof(vars(hc.down))
            + sys.getsizeof(hc.up) + sys.getsizeof(vars(hc.up)) for hc in hcs))
    if controlled > HOST_ALLOWANCE:
        raise MemoryError('HC small-row metadata exceeds its prepaid operating allowance')
    for layer, attention, mlp in prepared:
        layer.attn_hc, layer.mlp_hc = attention, mlp
    weights.mixer = mixer
    if weights.mtp is not None:
        weights.mtp.mixer = mtp_mixer
    weights.meta['hc_smallrow_vendor'] = {'policy': POLICY, 'maximum_rows': 3,
        'hc_objects': 2 * len(layers) + 1 + (mtp_mixer is not None),
        'controlled_metadata_bytes_observed': controlled,
        'host_operating_bytes_per_rank': HOST_ALLOWANCE,
        'scope': 'HC down/inject/up only; prefill/higher/generic remain original; numerical unqualified'}

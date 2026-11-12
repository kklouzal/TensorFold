"""Unselected decoded-tile ATen/CUTLASS attention component for actual SM70.

This boundary consumes already selected, decoded FP16 BMHK tiles. It neither
reads Q8 history nor selects QSA keys, updates State, performs collectives or
claims the incumbent's arithmetic. Outputs are normalized Half attention and
Float32 natural-log LSE, not the native owner's unnormalized Float32 partials.

The caller exclusively owns inputs through positive current-stream completion
and graph retirement. Read-only input aliasing is permitted. Captures borrow
fixed input/mask storage and own library/output allocations in their graph pool;
no stream, synchronization, host mask readback or gradient graph is introduced.
"""

from dataclasses import dataclass
import math
import sys


POLICY = 'aten-cutlass-half-normalized-lse-v1'
_MAX_FLOAT32 = float.fromhex('0x1.fffffep+127')
SCHEMA = (
    'aten::_efficient_attention_forward(Tensor query, Tensor key, Tensor value, Tensor? bias, '
    'Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, SymInt? max_seqlen_q, SymInt? max_seqlen_k, '
    'float dropout_p, int custom_mask_type, bool compute_log_sumexp=False, *, float? scale=None, '
    'Tensor? seqlen_k=None, int? window_size=None) -> (Tensor output, Tensor logsumexp, '
    'Tensor philox_seed, Tensor philox_offset, SymInt max_seqlen_batch_q, SymInt max_seqlen_batch_k)'
)


@dataclass(frozen=True)
class TileSpec:
    batch: int
    query_rows: int
    key_rows: int
    query_heads: int
    kv_heads: int
    scale: float
    masked: bool = False
    head_dim: int = 256

    def __post_init__(self):
        positive = (self.batch, self.query_rows, self.query_heads, self.kv_heads, self.head_dim)
        if (any(type(value) is not int or not 0 < value <= (1 << 31) - 1 for value in positive)
                or type(self.key_rows) is not int or not 0 <= self.key_rows <= (1 << 31) - 1
                or self.head_dim != 256 or self.query_heads % self.kv_heads
                or type(self.masked) is not bool):
            raise ValueError('SM70 tile requires bounded BMHK head256 and divisible query/KV heads')
        if (type(self.scale) not in (float, int) or self.scale <= 0 or self.scale > _MAX_FLOAT32
                or not math.isfinite(self.scale)):
            raise ValueError('explicit positive finite Float32-range attention scale required')
        if max(self.batch * self.query_rows * self.query_heads * 256 * 4,
               self.batch * self.key_rows * self.kv_heads * 256 * 4) > sys.maxsize:
            raise OverflowError('decoded tile byte arithmetic exceeds the host address range')


def tensor_requirements(spec):
    """Known simultaneous tensor payloads, not an allocator/driver peak bound.

    The owned Torch CUDA source allocates Half output, LSE rounded to 32 query
    rows, two Int64 RNG placeholders even at dropout0, and at most a full
    Float32 output accumulator. Masked output publication can overlap these
    with one additional Half result and a compact Float32 LSE. Allocator/graph
    pool reservation and runtime-managed effects still require native evidence.
    """
    if type(spec) is not TileSpec:
        raise TypeError('actual immutable TileSpec required')
    b, m, n, h = spec.batch, spec.query_rows, spec.key_rows, spec.query_heads
    output = b * m * h * 256 * 2
    lse = b * h * ((m + 31) // 32 * 32) * 4
    masked = spec.masked and n > 0
    payloads = {
        'output_half_bytes': output,
        'native_lse_padded_bytes': lse if n else b * h * m * 4,
        'native_accumulator_bound_bytes': 2 * output if n else 0,
        'native_rng_placeholder_bytes': 16 if n else 0,
        'aligned_bias_bytes': b * m * ((n + 7) // 8 * 8) * 2 if masked else 0,
        'mask_inversion_bytes': b * m * n if masked else 0,
        'validity_bytes': b * m if masked or not n else 0,
        'published_masked_output_bytes': output if masked else 0,
        'published_masked_lse_bytes': b * h * m * 4 if masked else 0,
    }
    total = sum(payloads.values())
    if total > sys.maxsize:
        raise OverflowError('attention component payload exceeds the host address range')
    return {**payloads, 'known_peak_tensor_bytes': total,
            'scope': 'tensor payload bound; allocator rounding, graph pool and driver overhead unqualified',
            'qualification': 'unselected-component'}


def operator_identity():
    """Inspect the actual protected ATen entry point without activating CUDA."""
    import torch

    operator = torch.ops.aten._efficient_attention_forward.default
    if (str(operator._schema) != SCHEMA
            or not torch._C._dispatch_has_kernel_for_dispatch_key('aten::_efficient_attention_forward', 'CUDA')):
        raise ValueError('protected ATen efficient-attention schema/CUDA registration differs')
    return {'policy': POLICY, 'schema': SCHEMA, 'torch': torch.__version__,
            'torch_git': torch.version.git_version, 'torch_cuda': torch.version.cuda,
            'cuda_registered': True, 'scope': 'operator metadata only; no native execution proof'}


@dataclass(frozen=True)
class Partial:
    output: object
    logsumexp: object
    valid_rows: object
    policy: str = POLICY


def _publish_masked(output, logsumexp, valid_rows):
    import torch

    # Only an explicit empty selected-key set has the zero/-inf identity.
    # NaNs/nonfinites from every nonempty library result remain observable.
    return Partial(torch.where(valid_rows[:, :, None, None], output, 0),
                   torch.where(valid_rows[:, None, :], logsumexp, -math.inf), valid_rows)


def _plain_tile(tensor, shape, dtype, device):
    """Metadata-only boundary: ATen must not resolve an uncharged lazy view."""
    import torch

    return (type(tensor) is torch.Tensor and tuple(tensor.shape) == shape
            and tensor.device == device and tensor.dtype == dtype and not tensor.requires_grad
            and not tensor.is_neg() and not tensor.is_conj() and tensor.is_contiguous())


@dataclass(frozen=True)
class Prepared:
    spec: TileSpec
    device: object
    device_budget_bytes: int
    identity: dict

    def run(self, query, key, value, *, valid_keys=None):
        import torch

        s = self.spec
        operands = (query, key, value)
        shapes = ((s.batch, s.query_rows, s.query_heads, 256),
                  (s.batch, s.key_rows, s.kv_heads, 256), (s.batch, s.key_rows, s.kv_heads, 256))
        for tensor, shape in zip(operands, shapes):
            if not _plain_tile(tensor, shape, torch.float16, self.device):
                raise ValueError('decoded attention inputs require their exact resolved contiguous CUDA FP16 tile without gradients')
        if s.masked:
            if not _plain_tile(valid_keys, (s.batch, s.query_rows, s.key_rows), torch.bool, self.device):
                raise ValueError('selected-key validity requires the exact resolved contiguous CUDA bool tile')
        elif valid_keys is not None:
            raise ValueError('attention mask was not included in the component byte admission')
        if not s.key_rows:
            return Partial(torch.zeros(shapes[0], dtype=torch.float16, device=self.device),
                           torch.full((s.batch, s.query_heads, s.query_rows), -math.inf,
                                      dtype=torch.float32, device=self.device),
                           torch.zeros((s.batch, s.query_rows), dtype=torch.bool, device=self.device))
        bias, valid_rows = None, None
        if s.masked:
            # The actual native bias ABI needs aligned row strides. Expanded
            # heads share the same immutable QSA mask without a head copy.
            aligned = (s.key_rows + 7) // 8 * 8
            slab = torch.zeros((s.batch, s.query_rows, aligned), dtype=torch.float16, device=self.device)
            selected = slab[:, :, :s.key_rows]
            selected.masked_fill_(~valid_keys, -math.inf)
            bias = selected[:, None].expand(s.batch, s.query_heads, s.query_rows, s.key_rows)
            valid_rows = valid_keys.any(dim=-1)
        output, lse, _seed, _offset, _max_q, _max_k = torch.ops.aten._efficient_attention_forward.default(
            query, key, value, bias, None, None, None, None, 0.0, 0, True, scale=float(s.scale))
        lse = lse[:, :, :s.query_rows]
        return Partial(output, lse, None) if valid_rows is None else _publish_masked(output, lse, valid_rows)


def prepare(spec, *, device, device_budget_bytes):
    """Resolve this unselected component once, before any inference/capture.

    The explicit credit funds the known tensor extent. It does not grant this
    candidate model/State capacity, numerical acceptance or allocator overhead.
    A caller must retain q/k/v/masks and all returned borrowers through their
    actual current-stream work and captured graph retirement.
    """
    import torch

    needed = tensor_requirements(spec)['known_peak_tensor_bytes']
    if type(device_budget_bytes) is not int or not needed <= device_budget_bytes <= sys.maxsize:
        raise MemoryError('decoded attention component lacks its explicit tensor byte credit')
    device = torch.device(device)
    if device.type != 'cuda' or device.index is None:
        raise ValueError('decoded attention preparation requires one explicit CUDA device')
    identity = operator_identity()
    if torch.cuda.get_device_capability(device) != (7, 0):
        raise ValueError('decoded attention component requires the actual SM70 target')
    return Prepared(spec, device, device_budget_bytes, identity)

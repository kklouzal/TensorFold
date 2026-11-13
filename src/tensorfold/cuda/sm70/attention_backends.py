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
OWNER_FAMILY = 'aten-cutlass-q8-normalized-lse-owner-v1'
OWNER_MAX_KEYS = 2051
NATIVE_QUERY_LAYOUT = 'native-gqa-v1'
HEAD_QUERY_LAYOUT = 'fixed12-head-as-query-lse-v1'
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
    query_layout: str | None = None

    def __post_init__(self):
        positive = (self.batch, self.query_rows, self.query_heads, self.kv_heads, self.head_dim)
        if (any(type(value) is not int or not 0 < value <= (1 << 31) - 1 for value in positive)
                or type(self.key_rows) is not int or not 0 <= self.key_rows <= (1 << 31) - 1
                or self.head_dim != 256 or self.query_heads % self.kv_heads
                or type(self.masked) is not bool
                or self.query_layout is not None and type(self.query_layout) is not str
                or self.query_layout not in (None, NATIVE_QUERY_LAYOUT, HEAD_QUERY_LAYOUT)):
            raise ValueError('SM70 tile requires bounded BMHK head256 and divisible query/KV heads')
        if (type(self.scale) not in (float, int) or self.scale <= 0 or self.scale > _MAX_FLOAT32
                or not math.isfinite(self.scale)):
            raise ValueError('explicit positive finite Float32-range attention scale required')
        if max(self.batch * self.query_rows * self.query_heads * 256 * 4,
               self.batch * self.key_rows * self.kv_heads * 256 * 4) > sys.maxsize:
            raise OverflowError('decoded tile byte arithmetic exceeds the host address range')

    @property
    def heads_as_query_rows(self):
        # The explicit mask is one selected-visible-key vector per logical
        # query, shared by all12 heads. Other tile regimes keep native GQA.
        return (self.query_layout == HEAD_QUERY_LAYOUT and self.query_rows == 1
                and self.query_heads == 12 and self.kv_heads == 1 and self.masked)


def tensor_requirements(spec):
    """Known simultaneous tensor payloads, not an allocator/driver peak bound.

    The owned Torch CUDA source allocates Half output, LSE rounded to 32 query
    rows, two Int64 RNG placeholders even at dropout0, and at most a full
    Float32 output accumulator. Masked output publication can overlap these
    with one additional Half result and a compact Float32 LSE. Allocator/graph
    pool reservation and runtime-managed effects still require native evidence.
    The optional fixed12 head-row layout keeps this original conservative
    credit, including its larger native-GQA LSE padding; no fit credit is taken.
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
        head_rows = s.heads_as_query_rows
        bias, valid_rows = None, None
        if s.masked:
            # The actual native bias ABI needs aligned row strides. Expanded
            # heads share the same immutable QSA mask without a head copy.
            aligned = (s.key_rows + 7) // 8 * 8
            slab = torch.zeros((s.batch, s.query_rows, aligned), dtype=torch.float16, device=self.device)
            selected = slab[:, :, :s.key_rows]
            selected.masked_fill_(~valid_keys, -math.inf)
            bias = selected[:, None].expand(
                (s.batch, 1, s.query_heads, s.key_rows) if head_rows
                else (s.batch, s.query_heads, s.query_rows, s.key_rows))
            valid_rows = valid_keys.any(dim=-1)
        # Batch still identifies the original logical query/owner. Only its
        # 12 heads become compute rows; no State row or causal position changes.
        native_query = query.view(s.batch, 12, 1, 256) if head_rows else query
        output, lse, _seed, _offset, _max_q, _max_k = torch.ops.aten._efficient_attention_forward.default(
            native_query, key, value, bias, None, None, None, None, 0.0, 0, True, scale=float(s.scale))
        if head_rows:
            output = output.view(s.batch, 1, 12, 256)
            # Native LSE is[B,1,32]: only the first12 query rows are heads.
            # Padding is not a head and must never enter the rank-mass merge.
            lse = lse[:, 0, :12].unsqueeze(-1)
        else:
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


def owner_requirements(rows, panel_rows):
    """Additional persistent packing plus real per-panel operator temporaries."""
    if (type(rows) is not int or not 1 <= rows <= 256 or type(panel_rows) is not int
            or not 1 <= panel_rows <= 256):
        raise ValueError('owner tiles require the existing bounded256-row envelope and a positive panel')
    panel = min(rows, panel_rows)
    primitive = tensor_requirements(TileSpec(panel, 1, OWNER_MAX_KEYS, 12, 1, .0625, masked=True))
    payloads = {'query_pack_bytes': rows * 12 * 256 * 2,
                'local_lse_bytes': rows * 12 * 4,
                'key_value_panel_bytes': panel * OWNER_MAX_KEYS * 256 * 2 * 2,
                'valid_key_panel_bytes': panel * OWNER_MAX_KEYS}
    return {**payloads, 'persistent_bytes': sum(payloads.values()),
            'operator_temporary_bytes': primitive['known_peak_tensor_bytes'], 'panel_rows': panel,
            'max_keys': OWNER_MAX_KEYS, 'scope': 'tensor payload; allocator/driver/graph reserve separate'}


def _merge_normalized_owner(owner, views, normalized, local_lse):
    from . import paged_attention as native
    group = owner.rank // 4 * 4
    owner.gather_native(views['local_stats'], views['all_stats'])
    native.merge_statistics(views['all_stats'], views['row_table'], views['global_stats'],
        group_start=group, page_tokens=owner.native_page_tokens, layer=owner.native_layer)
    native.weight_owner_normalized(normalized, local_lse, views['global_stats'], views['row_table'],
        views['local_values'], group_start=group, page_tokens=owner.native_page_tokens, layer=owner.native_layer)
    owner.gather_native(views['local_values'], views['all_values'])
    # Retain original rank-order reduction, inverse Q8 basis and final Half
    # publication. Only the local normalized/LSE arithmetic is a new family.
    native.merge_values(views['all_values'], views['row_table'], views['out'], rank=owner.rank,
        page_tokens=owner.native_page_tokens, layer=owner.native_layer, value_bits=8, value_codec=0)
    return views['out']


class OwnerTiles:
    """Exclusive current-stream panels; the enclosing native owner fences release.

    Pack every query before borrowing the old query receive slab for local
    normalized results. Local LSE remains in a disjoint persistent slab while
    the ordinary statistics/value communication slabs advance their phases.
    ATen outputs are temporary allocations retained through their submitted
    copies; this path is not allocation-free or an incumbent numeric alias.
    """
    def __init__(self, rows, panel_rows, device, *, query_layout=None):
        import torch

        self.query = self.keys = self.values = self.valid = self.lse = self.panels = None
        self.closed = False
        self.rows, self.device = rows, torch.device(device)
        self.query_layout = query_layout
        self.requirements = owner_requirements(rows, panel_rows)
        self.panel_rows = self.requirements['panel_rows']
        template = TileSpec(self.panel_rows, 1, OWNER_MAX_KEYS, 12, 1, .0625,
                            masked=True, query_layout=query_layout)
        self.primitive = prepare(template, device=self.device,
                                 device_budget_bytes=self.requirements['operator_temporary_bytes'])
        # The owner family fixes key capacity2051. Per-count metadata is cold,
        # bounded and immutable; no serving-time descriptor construction or
        # context-dependent online-softmax tiling is introduced.
        self.panels = tuple([None] + [Prepared(
            TileSpec(count, 1, OWNER_MAX_KEYS, 12, 1, .0625, masked=True, query_layout=query_layout), self.device,
            self.primitive.device_budget_bytes, self.primitive.identity)
            for count in range(1, self.panel_rows + 1)])
        self.query = torch.empty((rows, 1, 12, 256), dtype=torch.float16, device=self.device)
        self.lse = torch.empty((rows, 12), dtype=torch.float32, device=self.device)
        elements = self.panel_rows * OWNER_MAX_KEYS * 256
        self.keys = torch.empty(elements, dtype=torch.float16, device=self.device)
        self.values = torch.empty_like(self.keys)
        self.valid = torch.empty(self.panel_rows * OWNER_MAX_KEYS, dtype=torch.bool, device=self.device)

    def run(self, owner, rows, views, key_capacity):
        from . import paged_attention as native

        if self.closed or not 1 <= rows <= self.rows or key_capacity != OWNER_MAX_KEYS:
            raise ValueError('ATen owner tile frame exceeds its prepared live row/key envelope')
        group = owner.rank // 4 * 4
        query = self.query[:rows]
        native.pack_owner_queries(views['q_all'], views['row_table'], query,
                                  group_start=group, page_tokens=owner.native_page_tokens, layer=owner.native_layer)
        # query_all has retired as an input after the pack, on this same stream.
        normalized = owner.queries.reshape(-1)[:rows * 12 * 256].view(rows, 12, 256)
        local_lse = self.lse[:rows]
        stats = views['local_stats']
        for first in range(0, rows, self.panel_rows):
            count = min(self.panel_rows, rows - first)
            key = self.keys[:count * key_capacity * 256].view(count, key_capacity, 1, 256)
            value = self.values[:count * key_capacity * 256].view_as(key)
            valid = self.valid[:count * key_capacity].view(count, 1, key_capacity)
            native.gather_owner_q8(views['selected_all'], views['row_table'], key, value, valid,
                first=first, group_start=group, page_tokens=owner.native_page_tokens, layer=owner.native_layer)
            primitive = self.panels[count]
            result = primitive.run(query[first:first + count], key, value, valid_keys=valid)
            normalized[first:first + count].copy_(result.output[:, 0])
            local_lse[first:first + count].copy_(result.logsumexp[:, :, 0])
            stats[first:first + count, :, 0].copy_(result.logsumexp[:, :, 0])
            # Keep independent validity: nonempty NaN/+inf LSE must fail the
            # existing global statistics guard, never become an empty rank.
            stats[first:first + count, :, 1].copy_(result.valid_rows.expand(count, 12))
            del result
        return _merge_normalized_owner(owner, views, normalized, local_lse)

    def record(self):
        return {'family': OWNER_FAMILY, 'requirements': dict(self.requirements),
                **({'query_layout': self.query_layout} if self.query_layout is not None else {}),
                'operator': dict(self.primitive.identity), 'closed': self.closed,
                'numeric_contract': 'Half normalized local values and natural-log LSE; ordered FP32 rank weights/merge',
                'key_capacity_policy': 'fixed2051 masked keys; no context-dependent shape or row dispatch',
                'prepared_panel_counts': list(range(1, self.panel_rows + 1)),
                'qualification': 'unselected; operator/component/model/PPL/resource/P3 gates remain separate'}

    def close(self):
        # Only after enclosing OwnerBuffers.retire_native positively fences all
        # current-stream raw-pointer, ATen and communication consumers.
        self.query = self.keys = self.values = self.valid = self.lse = self.panels = None
        self.closed = True


from .flash_v100_artifact import FAMILY as FLASH_OWNER_FAMILY


def flash_owner_requirements(rows, panel_rows):
    """Complete persistent role payload, including one fixed-P private native workspace."""
    if (type(rows) is not int or not 1 <= rows <= 256 or type(panel_rows) is not int
            or not 1 <= panel_rows <= 256):
        raise ValueError('Flash owner requires bounded rows/panel1..256')
    # Keep P independent of active tail count: one native owner per role.
    panel = min(rows, panel_rows)
    payloads = {'query_pack_bytes': rows * 12 * 256 * 2, 'local_lse_bytes': rows * 12 * 4,
        'panel_query_output_bytes': panel * 12 * 256 * 2 * 2,
        'panel_lse_bytes': panel * 12 * 4,
        'key_value_panel_bytes': panel * OWNER_MAX_KEYS * 256 * 2 * 2,
        'valid_key_panel_bytes': panel * OWNER_MAX_KEYS, 'nonempty_bytes': panel,
        'native_private_bytes': 2 * panel * OWNER_MAX_KEYS * 256 * 2 + 4 * (3 * panel + 2)}
    return {**payloads, 'persistent_bytes': sum(payloads.values()), 'operator_temporary_bytes': 0,
        'panel_rows': panel, 'max_keys': OWNER_MAX_KEYS,
        'scope': 'persistent tensor payload; cold extension/driver/allocator/host metadata measured separately'}


class FlashOwnerTiles:
    """One fixed-P compacted primitive; ghost compute rows never become State rows.

    Pack the full live query frame before receive-slab reuse, then copy active
    panels and zero only the query/mask ghost suffix. Gather touches only live
    descriptor prefixes. Every library launch uses P physical batches and12
    head-as-query rows; only active outputs/statistics enter ordered collectives.
    Source pack/gather retains the existing finite-producer status authority.
    """
    def __init__(self, rows, panel_rows, device, artifact, *, tensor_budget_bytes, native_owner):
        import threading
        import torch
        from . import flash_v100_forward as primitive

        self.closed = self.failed = self._retirement_failed = False
        self.primitive = self._loan = self.native_settings = None
        self.artifact, self.frames = artifact, {}
        self.common = None
        self.query = self.lse = self.panel_query = self.panel_out = self.panel_lse = None
        self.keys = self.values = self.valid = self.nonempty = None
        self.rows, self.device, self.thread = rows, torch.device(device), threading.current_thread()
        self.requirements = flash_owner_requirements(rows, panel_rows)
        needed = self.requirements['persistent_bytes']
        if type(tensor_budget_bytes) is not int or not needed <= tensor_budget_bytes <= sys.maxsize:
            raise MemoryError('Flash owner role lacks its complete persistent tensor credit')
        self.panel_rows = p = self.requirements['panel_rows']
        self._loan = artifact.loan() # Journal before any device allocation.
        self.query = torch.zeros((rows, 1, 12, 256), device=self.device, dtype=torch.float16)
        self.lse = torch.empty((rows, 12), device=self.device, dtype=torch.float32)
        self.panel_query = torch.zeros((p * 12, 1, 256), device=self.device, dtype=torch.float16)
        self.panel_out = torch.empty_like(self.panel_query)
        self.panel_lse = torch.empty((1, p * 12), device=self.device, dtype=torch.float32)
        self.keys = torch.zeros((p, OWNER_MAX_KEYS, 1, 256), device=self.device, dtype=torch.float16)
        self.values = torch.zeros_like(self.keys)
        self.valid = torch.zeros((p, 1, OWNER_MAX_KEYS), device=self.device, dtype=torch.bool)
        self.nonempty = torch.empty((p, 1), device=self.device, dtype=torch.bool)
        self.primitive = primitive.prepare_compacted(artifact.module, self.panel_query, self.keys, self.values,
            self.valid, self.panel_out, self.panel_lse, tensor_budget_bytes=self.requirements['native_private_bytes'])
        self.native_settings = self.primitive.settings()
        self.common = {'group_start': native_owner.rank // 4 * 4, 'page_tokens': native_owner.native_page_tokens,
                       'layer': native_owner.native_layer}
        for bucket in native_owner.native_buckets:
            views = native_owner.native_views(bucket)
            normalized = native_owner.queries.reshape(-1)[:bucket * 12 * 256].view(bucket, 12, 256)
            local_lse = self.lse[:bucket]
            parts = []
            for first in range(0, bucket, p):
                count = min(p, bucket - first)
                parts.append((first, count, self.query[first:first + count].view(count * 12, 1, 256),
                    self.panel_query[:count * 12], self.panel_query[count * 12:],
                    self.keys[:count], self.values[:count], self.valid[:count], self.valid[count:],
                    self.panel_out[:count * 12].view(count, 12, 256),
                    self.panel_lse[:, :count * 12].view(count, 12),
                    normalized[first:first + count], local_lse[first:first + count],
                    views['local_stats'][first:first + count, :, 0],
                    views['local_stats'][first:first + count, :, 1],
                    self.nonempty[:count].expand(count, 12)))
            self.frames[bucket] = (views, self.query[:bucket], normalized, local_lse, tuple(parts))

    def _check(self):
        import threading
        if self.closed or self.failed or self._retirement_failed:
            raise RuntimeError('Flash owner is retired, failed or retirement uncertain')
        if threading.current_thread() is not self.thread:
            raise RuntimeError('Flash owner belongs to another host Thread')
        self.artifact.module

    def run(self, owner, rows):
        import torch
        from . import paged_attention as native
        self._check()
        if type(rows) is not int or rows not in self.frames:
            raise ValueError('Flash owner row frame was not prepared before admission')
        views, query, normalized, local_lse, parts = self.frames[rows]
        common = self.common
        try:
            native.pack_owner_queries(views['q_all'], views['row_table'], query, **common)
            for (first, count, source_q, panel_q, ghost_q, key, value, valid, ghost_valid,
                    result, result_lse, target, target_lse, stats_lse, stats_nonempty, flags) in parts:
                panel_q.copy_(source_q)
                if count != self.panel_rows:
                    ghost_q.zero_(); ghost_valid.zero_()
                native.gather_owner_q8(views['selected_all'], views['row_table'], key, value, valid,
                                      first=first, **common)
                self.primitive.run(self.panel_query, self.keys, self.values, self.valid, self.panel_out, self.panel_lse)
                torch.any(self.valid, dim=2, out=self.nonempty)
                target.copy_(result); target_lse.copy_(result_lse)
                stats_lse.copy_(result_lse); stats_nonempty.copy_(flags)
            return _merge_normalized_owner(owner, views, normalized, local_lse)
        except BaseException:
            self.failed = True
            raise

    def handoff_to_current_thread(self):
        import threading
        if self.closed or self.failed or self._retirement_failed:
            raise RuntimeError('Flash owner is retired, failed or retirement uncertain')
        try:
            self.primitive.handoff_to_current_thread()
        except BaseException:
            self._retirement_failed = True
            raise
        self.thread = threading.current_thread()

    def record(self):
        metadata = sys.getsizeof(self.frames)
        for frame in self.frames.values():
            metadata += sys.getsizeof(frame) + sys.getsizeof(frame[-1]) + sys.getsizeof(frame[0])
            for part in frame[-1]: metadata += sys.getsizeof(part) + sum(sys.getsizeof(v) for v in part)
        return {'family': FLASH_OWNER_FAMILY, 'requirements': dict(self.requirements),
            'native': None if self.native_settings is None else dict(self.native_settings),
            'artifact': self.artifact.record(), 'prepared_rows': list(self.frames),
            'host_metadata_payload_bytes': metadata, 'closed': self.closed,
            'failed': self.failed, 'retirement_failed': self._retirement_failed,
            'numeric_contract': 'fixed-P/fixed12 head-as-row stablecompactHalfO/F32LSE; finite producer required; mass limitation retained',
            'qualification': 'unselected; actual owner/model/quality/resource/P3 gates separate'}

    def close(self):
        import torch
        if self.closed: return
        # A failed run may still retire positively; a failed retirement retains
        # every tensor/module/loan and forbids retries via the primitive owner.
        if self._retirement_failed: raise RuntimeError('Flash owner retirement uncertain; retry forbidden')
        try:
            if self.primitive is not None:
                self.primitive.close()
            elif any(isinstance(v, torch.Tensor) and v.is_cuda for v in vars(self).values()):
                torch.cuda.synchronize(self.device)
            if self._loan is not None:
                self.artifact.release(self._loan); self._loan = None
        except BaseException:
            self._retirement_failed = True
            raise
        self.primitive = None; self.frames.clear()
        self.query = self.lse = self.panel_query = self.panel_out = self.panel_lse = None
        self.keys = self.values = self.valid = self.nonempty = None
        self.common = self.thread = None
        self.closed = True

"""Optional owned M32 projection storage; logical TensorFold row views stay unchanged."""
from __future__ import annotations

from types import SimpleNamespace

import torch

# Private qualification policy; ordinary Buffers and generic matmul remain selected.
_ENABLED = False
ROWS = 32
OWNED_POLICY = "owned-prepared-m32-v1"
OWNED_MAX_ROWS = 120


def field_layout(c, rows, head_width, world, *, linear_layers=None):
    """(logical shape, element bytes) shared by allocation and admission."""
    wide = c.streams * c.hidden
    attention = c.heads * 2 * c.head_dim + 2 * c.kv_heads * c.head_dim + (c.index_heads + 1) * c.index_dim
    half = {"normed": wide, "act": c.low, "up": wide, "mixed": c.hidden,
            "pa": attention, "gated": c.heads * c.head_dim, "gout": c.nv * 128,
            "ple_emb": c.ple_dim, "ple_keys": wide, "ple_vals": c.hidden}
    half["branch"] = c.hidden
    result = {name: ((rows, width), 2) for name, width in half.items()}
    result["logits"] = ((0 if head_width == 0 else rows, head_width), 4)
    if world > 1:
        result["part_branch"] = ((rows, c.hidden), 4)
    linear = sum(kind == "linear" for kind in c.layer_types) if linear_layers is None else linear_layers
    projection = 2 * c.nk * 128 + 2 * c.nv * 128 + 2 * c.nv
    result["proj"] = ((linear, rows, projection), 2)
    return result


def workspace_bytes(c, rows, head_width, world, *, enabled=None, prefill=False, linear_layers=None, owned_panels=False):
    """Exact additional persistent bytes; original logical Buffers are already charged."""
    enabled = _ENABLED if enabled is None else enabled
    if not enabled or prefill or not 1 <= rows <= (OWNED_MAX_ROWS if owned_panels else ROWS):
        return 0
    physical_rows = ((rows + ROWS - 1) // ROWS) * ROWS if owned_panels else ROWS
    extra = 0
    for shape, size in field_layout(c, rows, head_width, world,
                                  **({"linear_layers": linear_layers} if linear_layers is not None else {})).values():
        if len(shape) == 2:
            extra += (physical_rows - rows) * shape[1] * size
        else:
            extra += shape[0] * (physical_rows - rows) * shape[2] * size
    maximum_input = max(c.streams * c.hidden, c.ple_dim, c.nv * 128, c.heads * c.head_dim, c.low)
    extra += ROWS * maximum_input * 2
    extra += physical_rows * (c.low + c.low + c.streams) * 4
    return extra


def workspace_bytes_from_text(text, rows, world, *, enabled=None, prefill=False, _pipeline_stage=None, owned_panels=False):
    """The same forecast before weights load, including replicated local KV heads."""
    enabled = _ENABLED if enabled is None else enabled
    if not enabled or prefill or not 1 <= rows <= (OWNED_MAX_ROWS if owned_panels else ROWS):
        return 0
    text = text.get("text_config") or text
    c = SimpleNamespace(
        streams=int(text.get("hc_count", 4)), hidden=int(text["hidden_size"]),
        low=int(text.get("hc_lowrank", 320)), heads=int(text["num_attention_heads"]) // world,
        kv_heads=max(1, int(text["num_key_value_heads"]) // world),
        head_dim=int(text.get("head_dim") or int(text["hidden_size"]) // int(text["num_attention_heads"])),
        index_heads=int(text.get("indexer_n_heads", 4)), index_dim=int(text.get("indexer_head_dim", 128)),
        nk=int(text["linear_num_key_heads"]) // world, nv=int(text["linear_num_value_heads"]) // world,
        ple_dim=int(text.get("ple_embed_dim") or text["hidden_size"]),
        layer_types=["linear" if kind == "linear_attention" else "attention" for kind in text["layer_types"]],
    )
    head_width = int(text["vocab_size"]) // world
    if _pipeline_stage is not None:
        from ..pipeline_stage import PipelineStage

        if not isinstance(_pipeline_stage, PipelineStage) or world != 8 or len(c.layer_types) != 48:
            raise ValueError("padded pipeline workspace requires the fixed global48 TP8 stage")
        c.layer_types = c.layer_types[_pipeline_stage.first:_pipeline_stage.stop]
        if not _pipeline_stage.owns_output:
            head_width = 0
    return workspace_bytes(c, rows, head_width, world, enabled=True, owned_panels=owned_panels)


class PaddedLinears:
    """All storage is owned before capture; only bounded reshape/slice views escape."""

    def __init__(self, w, rows, *, head_columns=None, linear_layers=None, owned_panels=False):
        self.rows, self.device = rows, w.device
        self.owned_panels = owned_panels
        self.physical_rows = ((rows + ROWS - 1) // ROWS) * ROWS if owned_panels else ROWS
        c = w.cfg
        self.layout = field_layout(c, rows, w.head.n if head_columns is None else head_columns,
                                   int(w.meta.get("world", 1)),
                                   **({"linear_layers": linear_layers} if linear_layers is not None else {}))
        self.owners = {}
        self.inputs = []
        self.outputs = []
        self.maximum_input = max(c.streams * c.hidden, c.ple_dim, c.nv * 128, c.heads * c.head_dim, c.low)
        self.input_storage = torch.zeros((ROWS * self.maximum_input,), dtype=torch.float16, device=w.device)
        self.partial = {width: torch.zeros((self.physical_rows, width), dtype=torch.float32, device=w.device)
                        for width in (c.low, c.low + c.streams)}
        self.outputs.extend(self.partial.values())

    def tensor(self, name, shape, dtype, *, zero=False):
        expected = self.layout.get(name)
        if expected is None:
            factory = torch.zeros if zero else torch.empty
            return factory(shape, dtype=dtype, device=self.device)
        if tuple(shape) != expected[0] or torch.empty((), dtype=dtype).element_size() != expected[1]:
            raise ValueError("padded projection allocation disagrees with its forecast")
        if len(shape) == 2:
            owner = torch.zeros((self.physical_rows, shape[1]), dtype=dtype, device=self.device)
            matrices = [owner]
            view = owner[:shape[0]]
        else:
            owner = torch.zeros((shape[0], self.physical_rows, shape[2]), dtype=dtype, device=self.device)
            matrices = list(owner.unbind(0))
            view = owner[:, :shape[1]]
        self.owners[name] = owner
        self.outputs.extend(matrices)
        if dtype == torch.float16:
            self.inputs.extend(matrices)
        return view

    @staticmethod
    def _matrix_view(value, owners):
        if value.ndim != 2 or not value.is_contiguous() or value.stride(0) != value.shape[1]:
            return None
        need = ROWS * value.shape[1]
        for owner in owners:
            if (owner.dtype == value.dtype and owner.device == value.device
                    and owner.data_ptr() == value.data_ptr() and owner.numel() >= need):
                return owner.reshape(-1)[:need].view(ROWS, value.shape[1])
        return None

    @staticmethod
    def _owned_panel_span(value, owners):
        """Only paid row-aligned panels; return their full initialized M32 extent.

        Contiguous flat-prefix views (the reduced draft head) are allowed as in
        the incumbent M32 helper. Offset views must begin at a complete M32 face
        for their actual width and fit entirely in one retained owner matrix.
        Pointer arithmetic is bounded against that immutable owner's allocation.
        """
        if (value.ndim != 2 or not value.is_contiguous() or value.shape[1] <= 0
                or value.stride(0) != value.shape[1]):
            return None
        width = value.shape[1]
        padded_rows = ((value.shape[0] + ROWS - 1) // ROWS) * ROWS
        element = value.element_size()
        for owner in owners:
            if owner.dtype != value.dtype or owner.device != value.device:
                continue
            offset = value.data_ptr() - owner.data_ptr()
            if offset < 0 or offset % (ROWS * width * element):
                continue
            first = offset // element
            if first + padded_rows * width <= owner.numel():
                return owner.reshape(-1)[first:first + padded_rows * width].view(padded_rows, width)
        return None

    def _matmul_panels(self, x, matrix, out):
        target = self._owned_panel_span(out, self.outputs)
        if target is None or torch._C._overlaps(x, out):
            return None
        source = self._owned_panel_span(x, self.inputs)
        if source is None and matrix.k > self.maximum_input:
            return None
        for start in range(0, x.shape[0], ROWS):
            count = min(ROWS, x.shape[0] - start)
            if source is None:
                panel = self.input_storage[:ROWS * matrix.k].view(ROWS, matrix.k)
                panel[:count].copy_(x[start:start + count])
            else:
                panel = source[start:start + ROWS]
            destination = target[start:start + ROWS]
            version = None if destination.is_inference() else destination._version
            matrix.matmul_prepared(panel, destination)
            if version is not None and destination._version == version:
                torch.autograd.graph.increment_version(destination)
        return out

    def matmul(self, x, matrix, out):
        eligible = getattr(matrix, "can_matmul_prepared", None)
        if (eligible is None or not eligible(out.dtype) or x.ndim != 2 or out.ndim != 2
                or not 1 <= x.shape[0] <= (self.rows if self.owned_panels else ROWS) or out.shape != (x.shape[0], matrix.n)
                or x.shape[1] != matrix.k or x.dtype != torch.float16):
            return None
        if self.owned_panels:
            return self._matmul_panels(x, matrix, out)
        target = self._matrix_view(out, self.outputs)
        if target is None:
            return None
        source = self._matrix_view(x, self.inputs)
        if source is None:
            if matrix.k > self.maximum_input:
                return None
            source = self.input_storage[:ROWS * matrix.k].view(ROWS, matrix.k)
            source[:x.shape[0]].copy_(x)
        matrix.matmul_prepared(source, target)
        return out

    def partials(self, matrix, x):
        owner = self.partial.get(matrix.n)
        if owner is None or not 1 <= x.shape[0] <= self.rows:
            return None
        out = owner[:x.shape[0]]
        result = self.matmul(x, matrix, out)
        return None if result is None else result.unsqueeze(0)


def create(w, rows, *, prefill, dtype, head_columns=None, linear_layers=None, owner_role=None):
    candidate = getattr(w, '_private_prepared_m32', None)
    if candidate is not None and not prefill:
        role = 'decode' if owner_role is None else owner_role
        if (dtype != torch.float16 or w.device != candidate.device
                or rows != candidate.role_rows.get(role) or not candidate.bound_weights):
            raise ValueError('owned prepared M32 Buffers differ from the pre-State contract')
        result = PaddedLinears(w, rows, head_columns=head_columns, linear_layers=linear_layers, owned_panels=True)
        candidate.adopt(role, result)
        return result
    if (not _ENABLED or prefill or dtype != torch.float16 or not 1 <= rows <= ROWS
            or w.meta.get("execution_provider") != "sm70" or w.device.type != "cuda"):
        return None
    return PaddedLinears(w, rows, **({"head_columns": head_columns} if head_columns is not None else {}),
                         **({"linear_layers": linear_layers} if linear_layers is not None else {}))


def owned_numerical_identity():
    return {'policy': OWNED_POLICY, 'roles': ['decode', 'mtp'], 'prefill': 'unchanged',
            'logical_rows_limit': OWNED_MAX_ROWS, 'panel_rows': ROWS,
            'physical_rows': 'ceil(logical_rows/32)*32',
            'matmul': 'existing-GGUFDense-dense32-or-GGUFConcat-concat32',
            'concat': 'original-part-GEMMs;original-order;original-full-N',
            'inputs': 'fp16', 'outputs': 'existing-fp16-or-fp32',
            'HC_partials': 'fp32', 'reference_bit_identity': False,
            'unsupported_outputs': 'existing-reference-provider',
            'qualification': 'private-unqualified'}


class PreparedM32:
    """Cold policy and weight residency; Buffers own the paid projection storage.

    This capability does not enable the process-global padded-linear selector.
    Each role adopts exactly one new PaddedLinears object before State creation.
    The ordinary geometry charges its extra padding/arenas; they are not also
    allocated in this preparation owner or credited as preallocated bytes.
    """
    def __init__(self, text, role_rows, device):
        self.device = torch.device(device)
        self.role_rows = {role: rows for role, rows in role_rows.items() if role in ('decode', 'mtp')}
        if (not self.role_rows or any(type(rows) is not int or not 1 <= rows <= OWNED_MAX_ROWS
                                     for rows in self.role_rows.values())):
            raise ValueError('owned prepared M32 requires bounded decode/MTP rows through120')
        self.forecast = {role: workspace_bytes_from_text(text, rows, 8, enabled=True, owned_panels=True)
                         for role, rows in self.role_rows.items()}
        self.buffers = {}
        self.bound_weights = self.binding_started = False
        self.binding_memory = None
        self.closed = False
        self.reconciled_bytes = {}

    def record(self):
        return {**owned_numerical_identity(), 'rows': dict(self.role_rows),
                'physical_rows_by_role': {role: ((rows + ROWS - 1) // ROWS) * ROWS
                                          for role, rows in self.role_rows.items()},
                'additional_buffer_tensor_bytes_by_role': dict(self.forecast),
                'future_buffer_tensor_bytes': sum(self.forecast.values()),
                'allocation': 'ordinary-Buffers-geometry;not-preallocated-owner-credit',
                'bound_weights': self.bound_weights,
                'adopted_roles': sorted(self.buffers), 'binding_memory': self.binding_memory,
                'reconciled_extra_bytes': dict(self.reconciled_bytes), 'closed': self.closed}

    def adopt(self, role, buffers):
        if (self.closed or role in self.buffers or buffers.rows != self.role_rows.get(role)
                or buffers.device != self.device or not buffers.owned_panels):
            raise ValueError('owned prepared M32 role differs from its unique paid Buffer contract')
        self.buffers[role] = buffers

    def verify_buffers(self, role, buffers):
        from math import prod
        if (self.closed or self.buffers.get(role) is not buffers
                or set(buffers.owners) != set(buffers.layout)):
            raise ValueError('owned prepared M32 lacks its complete paid field catalogue')
        logical = sum(prod(shape) * size for shape, size in buffers.layout.values())
        payload = sum(value.numel() * value.element_size() for value in
                      (*buffers.owners.values(), buffers.input_storage, *buffers.partial.values()))
        if payload - logical != self.forecast[role]:
            raise ValueError('owned prepared M32 allocation differs from its extra-byte forecast')
        self.reconciled_bytes[role] = payload - logical

    def close(self):
        # The parent retires graph consumers before this capability. Discard
        # only these extra owning references; ordinary Buffers keep their own
        # tensor views until the existing engine lifecycle retires them.
        self.buffers.clear()
        self.closed = True

    def bind_weights(self, weights):
        """Warm actual full-N fixed-M32 faces before State allocation.

        The existing helpers retain original per-part GEMMs. Temporary operands
        are shared serially; no persistent second model/weight or output copy is
        retained. OwnerPreparation reconciles opaque code/library commitment.
        """
        from .gguf_weights import GGUFDense, GGUFConcat
        if self.closed or self.binding_started or weights.device != self.device:
            raise RuntimeError('owned prepared M32 requires one matching pre-State weight binding')
        self.binding_started = True
        def hc(value):
            yield value.down, torch.float32
            yield value.up, torch.float16
        def layer(value):
            yield from hc(value.attn_hc)
            yield from hc(value.mlp_hc)
            body = value.gdn if value.linear else value.attn
            yield body.proj, torch.float16
            yield body.out if value.linear else body.o, torch.float32
            if value.ple is not None:
                yield value.ple.key, torch.float16
                yield value.ple.value, torch.float16
        calls = []
        if 'decode' in self.role_rows:
            for value in weights.layers:
                calls.extend(layer(value))
            calls.extend(hc(weights.mixer))
            calls.append((weights.head, torch.float32))
        if 'mtp' in self.role_rows:
            if weights.mtp is None:
                raise ValueError('owned prepared M32 MTP role lacks its actual method weights')
            calls.extend(layer(weights.mtp.layer))
            calls.extend(hc(weights.mtp.mixer))
            calls.append((weights.draft_head or weights.head, torch.float32))
        # Pointer/type contracts are immutable after load, and the prepared ABI
        # performs its established input/output validation before each launch.
        calls = list({(id(matrix), dtype): (matrix, dtype) for matrix, dtype in calls
                      if type(matrix) in (GGUFDense, GGUFConcat)}.values())
        if not calls or any(not matrix.can_matmul_prepared(dtype) for matrix, dtype in calls):
            raise ValueError('owned prepared M32 requires original reference/native32 dense faces')
        def memory():
            torch.cuda.synchronize(self.device)
            return {'allocated': int(torch.cuda.memory_allocated(self.device)),
                    'reserved': int(torch.cuda.memory_reserved(self.device)),
                    'free': int(torch.cuda.mem_get_info(self.device)[0])}
        before = memory()
        maximum_input, maximum_output = max(m.k for m, _ in calls), max(m.n for m, _ in calls)
        with torch.no_grad():
            input_storage = torch.zeros(ROWS * maximum_input, device=self.device, dtype=torch.float16)
            output_storage = torch.empty(ROWS * maximum_output, device=self.device, dtype=torch.float32)
            for matrix, dtype in calls:
                source = input_storage[:ROWS * matrix.k].view(ROWS, matrix.k)
                storage = output_storage if dtype == torch.float32 else output_storage.view(torch.float16)
                target = storage[:ROWS * matrix.n].view(ROWS, matrix.n)
                matrix.matmul_prepared(source, target)
            torch.cuda.synchronize(self.device)
        source = target = storage = input_storage = output_storage = None
        after = memory()
        active, reserved = after['allocated'] - before['allocated'], after['reserved'] - before['reserved']
        outside = before['free'] - after['free'] - reserved
        self.binding_memory = {'before': before, 'after': after,
            'committed_delta_bytes': active + outside if active >= 0 and outside >= 0 else None,
            'outside_active_delta_bytes': outside if active >= 0 and outside >= 0 else None,
            'cold_temporary_bytes': ROWS * (2 * maximum_input + 4 * maximum_output),
            'scope': 'actual fixed-M32 full-N original-part warmup before State allocation'}
        self.bound_weights = True
        return self.record()

"""Explicit, unselected prepared dense projections for owned TP8 trials.

The preparation owner pays all retained tensor storage before state admission.
Actual weights and Buffer call sites bind cold in adopt(), before model warmup
or capture. Runtime only indexes frozen contracts and launches prepared work.
MTP's separate buffers keep the existing provider; vendor GEMMs use invariant
M32 geometry for target serial/verification rows. Provider quality, capacity and
complete model/serving qualification remain mandatory.

Frames share their Buffers' exclusive, ordered CUDA lifetime: callers establish
stream dependencies when changing streams; host serialization alone is not GPU
ordering. Bound weight objects/storage metadata remain immutable until close.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import resource
from types import MappingProxyType
import weakref

import torch

from tensorfold.cuda.sm70 import dense_backends

POLICY = 'sm70-prepared-dense-projections-v1'


def _alignment(tensor):
    address, value = tensor.data_ptr(), 1
    while value < 256 and address % (2 * value) == 0:
        value *= 2
    return value


@dataclass(frozen=True)
class Selection:
    backend: str
    geometry: str
    roles: tuple[str, ...]
    panel_rows: int

    def __post_init__(self):
        if self.backend not in ('cublas', 'cublaslt', 'cutlass'):
            raise ValueError('dense projections require an explicit supported backend')
        if (self.geometry not in ('fixed-m32', 'direct')
                or self.backend != 'cutlass' and self.geometry != 'fixed-m32'):
            raise ValueError('vendor dense projections require fixed M32 serial/verify geometry')
        if (type(self.roles) is not tuple or not self.roles
                or any(type(role) is not str or role not in ('decode', 'prefill') for role in self.roles)
                or len(set(self.roles)) != len(self.roles)):
            raise ValueError('dense projection roles must explicitly select decode/prefill; MTP retains its provider')
        if (type(self.panel_rows) is not int or not 1 <= self.panel_rows <= (1 << 31) - 1
                or self.geometry == 'fixed-m32' and self.panel_rows != 32):
            raise ValueError('dense panel rows must be positive; fixed M32 requires32')

    @classmethod
    def parse(cls, value):
        if value is None:
            return None
        if (type(value) is not dict or set(value) != {'backend', 'geometry', 'roles', 'panel_rows'}
                or type(value['roles']) is not list):
            raise ValueError('dense projections require backend, geometry, explicit roles and panel_rows')
        return cls(value['backend'], value['geometry'], tuple(value['roles']), value['panel_rows'])

    def as_dict(self):
        return {'backend': self.backend, 'geometry': self.geometry,
                'roles': list(self.roles), 'panel_rows': self.panel_rows}

    @property
    def numerical_identity(self):
        return {'policy': POLICY, **self.as_dict(), 'MTP_dense': 'existing-reference-provider',
                'HC_decode_partials': 'float32', 'HC_prefill_output': 'float16', 'concat': 'existing-fused-weight-face'}


def forecast(selection, text, role_rows, *, world=8):
    """Exact tensor arenas; opaque CUDA handle/module residency is measured."""
    if type(selection) is not Selection or world != 8:
        raise ValueError('dense projections require their typed TP8 trial selection')
    text = text.get('text_config') or text
    def number(name, default=None):
        value = text.get(name, default)
        if type(value) is not int or value <= 0:
            raise ValueError('dense projection geometry requires positive ' + name)
        return value
    hidden, streams = number('hidden_size'), number('hc_count', 4)
    heads, kv_heads = number('num_attention_heads') // world, max(1, number('num_key_value_heads') // world)
    nk, nv = number('linear_num_key_heads') // world, number('linear_num_value_heads') // world
    if min(heads, nk, nv) <= 0:
        raise ValueError('dense projection geometry does not divide over TP8')
    dim = number('head_dim', hidden // number('num_attention_heads'))
    low = number('hc_lowrank', 320)
    index_dim, index_heads = number('indexer_head_dim', 128), number('indexer_n_heads', 4)
    wide, ple = streams * hidden, number('ple_embed_dim', hidden)
    projection = 2 * nk * 128 + 2 * nv * 128 + 2 * nv
    attention = heads * 2 * dim + 2 * kv_heads * dim + (index_heads + 1) * index_dim
    maximum_input = max(wide, ple, nv * 128, heads * dim, low)
    maximum_output = max(wide, hidden, projection, attention, low + streams, number('vocab_size') // world)
    layers = number('num_hidden_layers')
    ple_layers = text.get('ple_layer_ids') or []
    if type(ple_layers) is not list or len(ple_layers) > layers:
        raise ValueError('dense projection PLE geometry requires the bounded layer list')
    # Per target layer: two HC down/up pairs and one attention/GDN proj/out;
    # PLE adds key/value, and the terminal mixer/head adds three faces.
    face_bound = 6 * layers + 2 * len(ple_layers) + 3
    roles = {}
    for role in selection.roles:
        rows = role_rows.get(role)
        if type(rows) is not int or rows <= 0:
            raise ValueError('dense projections name an absent/unbounded role')
        fixed = selection.geometry == 'fixed-m32'
        partial_rows = max(rows, 32) if fixed else rows
        roles[role] = {'input_elements': 32 * maximum_input if fixed else 0,
            'output_float_elements': 32 * maximum_output if fixed else 0,
            'partial_shape': (partial_rows, low + streams) if role == 'decode' else (0, low + streams)}
    tensor_bytes = sum(2 * role['input_elements'] + 4 * role['output_float_elements']
        + 4 * role['partial_shape'][0] * role['partial_shape'][1] for role in roles.values())
    # One scratch LD plus at most three direct pointer-alignment classes:
    # a32-row Half panel advances a multiple of64 bytes (Float:128).
    contracts_per_face = 4 if selection.geometry == 'fixed-m32' else selection.panel_rows
    max_plans = face_bound * contracts_per_face * len(roles)
    if max_plans > (1 << 63) - 1 or tensor_bytes > (1 << 63) - 1:
        raise ValueError('dense projection resource arithmetic exceeds the supported range')
    return {'roles': roles, 'tensor_bytes': tensor_bytes, 'maximum_input': maximum_input,
            'maximum_output': maximum_output, 'face_bound': face_bound, 'max_plans': max_plans,
            'cold_temporary_bytes': (selection.panel_rows * (2 * maximum_input + 4 * maximum_output)
                                     if selection.geometry == 'direct' else 0),
            'future_temporary_bytes': 0,
            'opaque_residency': 'measure cold handle/module allocation and reconcile after actual weight binding'}


class Owner:
    @classmethod
    def prepare(cls, selection, text, role_rows, device, *, world=8, register):
        owner = cls()
        owner.selection = selection
        owner.resources = forecast(selection, text, role_rows, world=world)
        owner.device = torch.device(device)
        owner.role_rows = dict(role_rows)
        owner._context = None
        owner._arenas, owner._frames, owner._cache = {}, {}, {}
        owner._attachments = []
        owner.closed = owner.adopted = False
        owner.adoption_started = False
        owner.binding_started = owner.bound_weights = False
        owner.binding_memory = None
        owner.native_module = None
        owner._bound = {}
        owner.adoption_memory = None
        register(owner)  # Journal before native acquisition or allocation.
        owner._context = dense_backends.Context(owner.device, backend=selection.backend,
                                                max_plans=owner.resources['max_plans'])
        owner.native_module = type(owner._context._native).__module__
        for role, resource in owner.resources['roles'].items():
            owner._arenas[role] = arena = {}
            arena['input'] = torch.zeros(resource['input_elements'], dtype=torch.float16, device=device)
            arena['output'] = torch.empty(resource['output_float_elements'], dtype=torch.float32, device=device)
            arena['partial'] = torch.empty(resource['partial_shape'], dtype=torch.float32, device=device)
        actual = sum(t.numel() * t.element_size() for arena in owner._arenas.values() for t in arena.values())
        if actual != owner.resources['tensor_bytes']:
            raise MemoryError('dense projection tensors disagree with their pre-admission forecast')
        return owner

    def _plan(self, x, weight, out):
        key = (tuple(x.shape), tuple(weight.shape), tuple(out.shape), tuple(x.stride()),
               tuple(weight.stride()), tuple(out.stride()), out.dtype,
               *((_alignment(x), _alignment(weight), _alignment(out)) if self.selection.backend != 'cutlass' else ()))
        plan = self._cache.get(key)
        if plan is None:
            if self.bound_weights:
                raise ValueError('dense adoption reached an unprebound shape/stride/dtype/alignment contract')
            plan = dense_backends.prepare(self._context, x, weight, out)
            self._cache[key] = plan
        return plan

    def bind_weights(self, weights):
        """Cold library/module preparation after weight load, BEFORE State allocation.

        Bound weights and their storage metadata are immutable until close.
        Actual Buffer adoption subsequently performs only bounded host views and
        sealed-plan lookups; no new device allocation or operator warmup occurs.
        """
        from .gguf_weights import GGUFDense, GGUFConcat

        if self.closed or self.binding_started or weights.device != self.device:
            raise RuntimeError('dense projections require one matching pre-State weight binding')
        self._context._check()
        self.binding_started = True
        before = self._memory()
        temporary_input = temporary_output = None
        if self.selection.geometry == 'direct':
            temporary_input = torch.zeros(self.selection.panel_rows * self.resources['maximum_input'],
                                           dtype=torch.float16, device=self.device)
            temporary_output = torch.empty(self.selection.panel_rows * self.resources['maximum_output'],
                                            dtype=torch.float32, device=self.device)

        def calls(role):
            def hc(value):
                yield value.down, torch.float32 if role == 'decode' else torch.float16
                yield value.prefill_up if role == 'prefill' and value.prefill_up is not None else value.up, torch.float16
            for layer in weights.layers:
                yield from hc(layer.attn_hc)
                yield from hc(layer.mlp_hc)
                body = layer.gdn if layer.linear else layer.attn
                yield body.proj, torch.float16
                yield body.out if layer.linear else body.o, torch.float32
                if layer.ple is not None:
                    yield layer.ple.key, torch.float16
                    yield layer.ple.value, torch.float16
            yield from hc(weights.mixer)
            yield weights.head, torch.float32

        for role in self.selection.roles:
            self._bound[role] = bound = {}
            input_arena = self._arenas[role]['input'] if temporary_input is None else temporary_input
            output_arena = self._arenas[role]['output'] if temporary_output is None else temporary_output
            for matrix, dtype in calls(role):
                if type(matrix) not in (GGUFDense, GGUFConcat):
                    continue
                face = matrix if type(matrix) is GGUFDense else matrix.fused
                if face is None:
                    continue
                if matrix in bound:
                    continue
                bound[matrix] = face.weight
                row_contracts = (32,) if self.selection.geometry == 'fixed-m32' else range(1, self.selection.panel_rows + 1)
                for rows in row_contracts:
                    if rows * matrix.k > input_arena.numel() or rows * matrix.n > output_arena.numel():
                        raise ValueError('dense actual weight exceeds its pre-State cold scratch forecast')
                    source = input_arena[:rows * matrix.k].view(rows, matrix.k)
                    storage = output_arena if dtype == torch.float32 else output_arena.view(torch.float16)
                    target = storage[:rows * matrix.n].view(rows, matrix.n)
                    self._plan(source, face.weight, target)
        # Drop every temporary view before sampling settled residency. Ordinary
        # reusable allocator spare is recorded separately, never called private.
        source = target = storage = input_arena = output_arena = None
        temporary_input = temporary_output = None
        self._cache = MappingProxyType(self._cache)
        self._bound = MappingProxyType({role: MappingProxyType(faces) for role, faces in self._bound.items()})
        self.bound_weights = True
        after = self._memory()
        active = after['torch_allocated_bytes'] - before['torch_allocated_bytes']
        reserved = after['torch_reserved_bytes'] - before['torch_reserved_bytes']
        physical = before['physical_free_bytes'] - after['physical_free_bytes']
        outside = physical - reserved
        self.binding_memory = {'before': before, 'after': after,
            'committed_delta_bytes': active + outside if active >= 0 and outside >= 0 else None,
            'outside_active_delta_bytes': outside if active >= 0 and outside >= 0 else None,
            'cold_temporary_bytes': self.resources['cold_temporary_bytes'],
            'scope': 'isolated after-weight/pre-State library warmup; unknown/noisy deltas cannot qualify admission'}
        return self.record()

    def adopt(self, weights, decoder):
        if self.closed or not self.bound_weights or self.adoption_started or weights.device != self.device:
            raise RuntimeError('dense projections require one matching cold adoption')
        self._context._check()
        if int(weights.meta.get('world', 0)) != 8 or weights.meta.get('execution_provider') != 'sm70':
            raise ValueError('dense projections require actual owned SM70 TP8 weights')
        before = self._memory()
        self.adoption_started = True
        buffers = {'decode': decoder.buf, 'prefill': decoder.pbuf}
        for role in self.selection.roles:
            frame = Frame(self, weights, buffers[role], role)
            self._frames[role] = frame
            frame.prepare()
        # Publish only after the complete catalogue has prepared successfully.
        self.adoption_memory = {'before': before, 'after': self._memory(),
            'scope': 'host-only sealed-plan/Buffer view adoption; no operator warmup or device allocation'}
        after = self.adoption_memory['after']
        if (after['torch_allocated_bytes'] != before['torch_allocated_bytes']
                or after['torch_reserved_bytes'] != before['torch_reserved_bytes']):
            raise RuntimeError('dense Buffer adoption unexpectedly allocated device memory')
        for role, frame in self._frames.items():
            self._attachments.append((weakref.ref(buffers[role]), frame))
            buffers[role].dense_linears = frame
        self.adopted = True
        return self.record()

    def _memory(self):
        torch.cuda.synchronize(self.device)
        return {'physical_free_bytes': int(torch.cuda.mem_get_info(self.device)[0]),
                'torch_allocated_bytes': int(torch.cuda.memory_allocated(self.device)),
                'torch_reserved_bytes': int(torch.cuda.memory_reserved(self.device)),
                'host_peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024}

    def handoff_to_current_thread(self):
        if self.closed or not self.adopted:
            raise RuntimeError('dense projections require adopted live resources before handoff')
        self._context.handoff_to_current_thread()

    def record(self):
        return {'selection': self.selection.as_dict(), 'tensor_bytes': self.resources['tensor_bytes'],
            'numerical_identity': self.selection.numerical_identity,
            'max_plans': self.resources['max_plans'], 'prepared_plans': len(self._cache),
            'roles': {role: frame.record() for role, frame in self._frames.items()},
            'adopted': self.adopted, 'closed': self.closed, 'adoption_memory': self.adoption_memory,
            'bound_weights': self.bound_weights, 'binding_memory': self.binding_memory,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'native_module': self.native_module,
            'host_metadata': {'prepared_plan_bound': self.resources['max_plans'],
                'prepared_plans': len(self._cache), 'bound_weight_faces': sum(len(faces) for faces in self._bound.values()),
                'frame_call_sites': sum(len(frame.faces) for frame in self._frames.values()),
                'direct_target_views': sum(len(value[-1]) for frame in self._frames.values()
                                           for value in frame.faces.values() if value[0] == 'fixed'),
                'HC_row_views': sum(len(views) - 1 for frame in self._frames.values()
                                    for views in frame.partial_outputs.values())},
            'method_dense': 'existing-reference-provider', 'quality': 'unqualified-private-trial'}

    def close(self):
        if self.closed:
            return
        if self._context is not None:
            self._context.close()  # Graph borrowers retire first; positive device fence before release.
        for reference, frame in self._attachments:
            buffers = reference()
            if buffers is not None and getattr(buffers, 'dense_linears', None) is frame:
                buffers.dense_linears = None
            frame.close()
        for frame in self._frames.values():
            frame.close()
        self._attachments.clear()
        self._frames.clear()
        self._cache = {}
        self._bound = {}
        self._arenas.clear()
        self.closed = True


class Frame:
    def __init__(self, owner, weights, buffers, role):
        from .gguf_weights import GGUFDense, GGUFConcat

        self._context = owner._context
        self._prepare_plan = owner._plan
        self._types = (GGUFDense, GGUFConcat)
        self._buffers, self._weights = buffers, weights  # Cold construction only; dropped by prepare().
        self.role, self.rows = role, buffers.rows
        self.geometry, self.panel_rows = owner.selection.geometry, owner.selection.panel_rows
        self.backend = owner.selection.backend
        self.bound_faces = owner._bound[role]
        self.arena = owner._arenas[role]
        self.faces, self.excluded = {}, set()
        self.partial_outputs = {}
        self.domains = {}
        self.closed = False
        self.calls = []

    def prepare(self):
        from .gguf_weights import GGUFDense, GGUFConcat

        b, w = self._buffers, self._weights
        if b.execution_dtype != torch.float16 or b.owner_role != self.role:
            raise ValueError('dense projections require the exact Half execution role')
        if self.role == 'decode' and self.rows > self.arena['partial'].shape[0]:
            raise ValueError('dense projection decode frame exceeds its paid partial arena')
        self.partial_outputs = {}

        def add(matrix, x, out, *, partial=False):
            if type(matrix) not in self._types:
                return
            face = matrix if type(matrix) is GGUFDense else matrix.fused
            if face is None:
                self.excluded.add(matrix)
                return
            if type(face) is not GGUFDense or face.weight.dtype != torch.float16:
                raise ValueError('dense projections require contiguous native Half weight faces')
            if matrix in self.faces:
                return
            if self.bound_faces.get(matrix) is not face.weight:
                raise ValueError('dense adoption reached a matrix absent from pre-State weight binding')
            if partial:
                if matrix.n > self.arena['partial'].shape[1]:
                    raise ValueError('HC partial width exceeds its paid dense arena')
                # The existing fused HC consumer reads tightly packed [SK,R,N].
                # Different HC widths reuse one flat arena, never strided columns.
                out = self.arena['partial'].reshape(-1)[:self.arena['partial'].shape[0] * matrix.n].view(-1, matrix.n)
                self.partial_outputs[matrix] = tuple([None] + [out[:r].unsqueeze(0) for r in range(1, self.rows + 1)])
            if x.shape[1] != matrix.k or out.shape[1] != matrix.n:
                raise ValueError('dense projection call site disagrees with actual matrix geometry')
            limit = min(self.panel_rows, self.rows, out.shape[0])
            if not limit:
                return
            if self.geometry == 'fixed-m32':
                if 32 * matrix.k > self.arena['input'].numel() or 32 * matrix.n > self.arena['output'].numel():
                    raise ValueError('dense matrix exceeds its pre-admission pad envelope')
                source = self.arena['input'][:32 * matrix.k].view(32, matrix.k)
                output_arena = self.arena['output'] if out.dtype == torch.float32 else self.arena['output'].view(torch.float16)
                padded_out = output_arena[:32 * matrix.n].view(32, matrix.n)
                scratch_plan = self._prepare_plan(source, face.weight, padded_out)
                direct_targets = {}
                # These inactive rows are owned execution scratch. Every exposed
                # direct target lies wholly inside one declared output matrix.
                for start in range(0, out.shape[0] - 31, 32):
                    target = out[start:start + 32]
                    if tuple(target.stride()) != (matrix.n, 1):
                        continue
                    if self.backend != 'cutlass' and _alignment(target) != _alignment(padded_out):
                        continue
                    direct_targets[target.data_ptr()] = (target, self._prepare_plan(source, face.weight, target))
                value = ('fixed', face.weight, source, padded_out, scratch_plan, _alignment(source),
                         MappingProxyType(direct_targets))
            else:
                plans = tuple([None] + [self._prepare_plan(x[:r], face.weight, out[:r]) for r in range(1, limit + 1)])
                value = ('direct', face.weight, plans)
            self.faces[matrix] = value
            self.domains[matrix] = (out.dtype, tuple(x.stride()), tuple(out.stride()))
            self.calls.append({'n': matrix.n, 'k': matrix.k, 'dtype': str(out.dtype),
                'output_stride': list(out.stride()), 'concat_fused': type(matrix) is GGUFConcat,
                'maximum_panel_rows': 32 if self.geometry == 'fixed-m32' else limit})

        def hc(value):
            if self.role == 'decode':
                add(value.down, b.normed, b.dn, partial=True)
            else:
                add(value.down, b.normed, b.dn if value.down.n == b.dn.shape[1] else b.dn_mix)
            add(value.prefill_up if b.prefill and value.prefill_up is not None else value.up, b.act, b.up)

        linear_index = 0
        for layer in w.layers:
            hc(layer.attn_hc)
            hc(layer.mlp_hc)
            if layer.linear:
                add(layer.gdn.proj, b.mixed, b.proj[0 if b.prefill else linear_index])
                add(layer.gdn.out, b.gout, b.part_branch if b.world > 1 else b.branch)
                linear_index += 1
            else:
                add(layer.attn.proj, b.mixed, b.pa)
                add(layer.attn.o, b.gated, b.part_branch if b.world > 1 else b.branch)
            if layer.ple is not None:
                add(layer.ple.key, b.ple_emb, b.ple_keys)
                add(layer.ple.value, b.ple_emb, b.ple_vals)
        hc(w.mixer)
        add(w.head, b.mixed, b.logits)
        if self.role == 'prefill' and getattr(w, 'mtp', None) is not None:
            # MTP prompt absorption borrows pbuf. Only actual distinct method
            # faces decline: unknown target matrices still fail closed, and a
            # shared target alias remains in its target contract. The supported
            # shared-GGUF importer aliases embedding/head, not these MTP faces.
            method = w.mtp
            method_faces = [method.fc_e, method.fc_h, method.mixer.down, method.mixer.up]
            for hc_value in (method.layer.attn_hc, method.layer.mlp_hc):
                method_faces.extend((hc_value.down, hc_value.up))
            body = method.layer.gdn if method.layer.linear else method.layer.attn
            method_faces.extend((body.proj, body.out if method.layer.linear else body.o))
            if method.layer.ple is not None:
                method_faces.extend((method.layer.ple.key, method.layer.ple.value))
            self.excluded.update(matrix for matrix in method_faces
                                 if type(matrix) in self._types and matrix not in self.faces)
        self.faces = MappingProxyType(self.faces)
        self.domains = MappingProxyType(self.domains)
        self.excluded = frozenset(self.excluded)
        self.partial_outputs = MappingProxyType(self.partial_outputs)
        self.calls = tuple(self.calls)
        self._buffers = self._weights = self._prepare_plan = None
        self.bound_faces = None

    def matmul(self, x, matrix, out):
        if self.closed:
            raise RuntimeError('dense projection frame is closed')
        if type(matrix) not in self._types or matrix in self.excluded:
            return None
        self._context._check()  # Before any pad/copy effect, including wrong-thread calls.
        value = self.faces.get(matrix)
        if value is None:
            raise ValueError('dense projection reached a matrix absent from its cold call catalogue')
        expected_dtype, input_stride, output_stride = self.domains[matrix]
        if (not dense_backends._plain(x) or not dense_backends._plain(out)
                or x.ndim != 2 or out.ndim != 2 or x.dtype != torch.float16
                or out.dtype != expected_dtype or x.device != value[1].device or out.device != x.device
                or x.shape[1] != matrix.k or tuple(x.stride()) != input_stride or tuple(out.stride()) != output_stride
                or out.is_inference() and not torch.is_inference_mode_enabled()):
            raise ValueError('dense projection input/output domain differs from its cold call contract')
        rows = x.shape[0]
        if not 0 < rows <= self.rows or out.shape != (rows, matrix.n):
            raise ValueError('dense projection rows exceed the adopted execution frame')
        if torch._C._overlaps(out, x) or torch._C._overlaps(out, value[1]):
            raise ValueError('dense projection output overlaps its inputs')
        for start in range(0, rows, self.panel_rows):
            count = min(self.panel_rows, rows - start)
            source, target = x[start:start + count], out[start:start + count]
            if value[0] == 'fixed':
                _, weight, padded_x, padded_out, plan, input_alignment, direct_targets = value
                # A complete prefill M32 face can borrow the live input under
                # the already sealed plan's exact stride/dtype/alignment class.
                # Tails and other domains retain initialized paid staging.
                direct_source = (self.role == 'prefill' and count == 32
                    and tuple(source.stride()) == tuple(padded_x.stride())
                    and _alignment(source) >= input_alignment)
                if not direct_source:
                    padded_x[:count].copy_(source)
                operand = source if direct_source else padded_x
                direct = direct_targets.get(target.data_ptr())
                if direct is not None:
                    direct[1].run(operand, weight, direct[0])
                else:
                    plan.run(operand, weight, padded_out)
                    target.copy_(padded_out[:count])
            else:
                _, weight, plans = value
                if count >= len(plans):
                    raise ValueError('dense panel exceeds its prepared call-site row contracts')
                plans[count].run(source, weight, target)
        return out

    def partials(self, matrix, x):
        if type(matrix) not in self._types or matrix in self.excluded:
            return None
        outputs = self.partial_outputs.get(matrix)
        rows = x.shape[0]
        if outputs is None or not 0 < rows < len(outputs):
            raise ValueError('dense HC partials lack their paid FP32 row contract')
        result = outputs[rows]
        self.matmul(x, matrix, result[0])
        return result

    def record(self):
        return {'role': self.role, 'rows': self.rows, 'geometry': self.geometry,
            'panel_rows': self.panel_rows, 'call_sites': list(self.calls), 'reference_faces': len(self.excluded),
            'input_pad': 'complete aligned prefill M32 borrows source;other panels copy active prefix into initialized staging',
            'output_pad': 'direct full-M32 owned targets where admitted, otherwise explicit active-row copy',
            'HC_decode_partials': 'Float32', 'MTP': 'reference-provider-unmodified'}

    def close(self):
        self.faces = self.domains = self.partial_outputs = MappingProxyType({})
        self.excluded = frozenset()
        self.arena = {}
        self._context = self._buffers = self._weights = self._prepare_plan = None
        self.bound_faces = None
        self.closed = True

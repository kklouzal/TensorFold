"""Unselected HC-only BitBLAS reference with fixed physical M32 arithmetic.

Original Half weights are borrowed unchanged. Inject4 has a separately funded,
immutable zero-padded N8 copy; model weight values/storage cannot change after
binding. Down320/inject4 retain independent Float32 partials for decode; prefill
and up publish Half with an explicit Float32-to-Half copy. MTP keeps its provider.

One Buffers role exclusively owns its arenas. Callers establish CUDA stream
ordering across uses; graphs and every operand must retire before owner.close().
The three ctypes/driver code mappings are process-resident opaque footprint.
Close fences and releases tensor/catalogue borrows; it does not claim to unload
CUDA code. Runtime indexes a sealed row/face catalogue, creates no Tensor views, allocates
no device storage and performs no compilation, file access or host readback.
"""
import resource
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import torch

from tensorfold.cuda.sm70 import bitblas_library as library

POLICY = 'sm70-bitblas-hc-fixed-m32-v1'


@dataclass(frozen=True)
class Selection:
    manifest: str
    manifest_sha256: str
    roles: tuple[str, ...] = ('decode', 'prefill')

    def __post_init__(self):
        if (type(self.manifest) is not str or not Path(self.manifest).is_absolute()
                or type(self.manifest_sha256) is not str or len(self.manifest_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.manifest_sha256)):
            raise ValueError('HC selection requires an absolute sealed manifest and lowercase SHA256')
        if (type(self.roles) is not tuple or not self.roles or len(set(self.roles)) != len(self.roles)
                or any(v not in ('decode', 'prefill') for v in self.roles)):
            raise ValueError('HC selection requires explicit target decode/prefill roles; MTP is unchanged')

    @property
    def backend(self):
        return 'bitblas-hc'

    @property
    def numerical_identity(self):
        return {'policy':POLICY,'geometry':'fixed-m32','roles':list(self.roles),
                'HC_decode_partials':'float32','HC_prefill_and_up':'explicit-float32-to-half-copy',
                'MTP':'existing-reference-provider'}

    @classmethod
    def parse(cls,value):
        if value is None:
            return None
        if (type(value) is not dict or set(value) != {'backend','manifest','manifest_sha256','roles'}
                or value['backend'] != 'bitblas-hc' or type(value['roles']) is not list):
            raise ValueError('HC selection requires its explicit backend/manifest/SHA/target-role schema')
        return cls(value['manifest'],value['manifest_sha256'],tuple(value['roles']))

    def as_dict(self):
        return {'backend': 'bitblas-hc', 'manifest': self.manifest,
                'manifest_sha256': self.manifest_sha256, 'roles': list(self.roles)}


def forecast(selection, text, role_rows, *, world=8):
    if type(selection) is not Selection:
        raise TypeError('typed HC selection required')
    text = text.get('text_config') or text
    if (world != 8 or text.get('hidden_size') != 2560 or text.get('hc_count', 4) != 4
            or text.get('hc_lowrank', 320) != 320 or text.get('num_hidden_layers') != 48):
        raise ValueError('sealed HC exports require actual SC117 TP8 D2560/S4/L320/48-layer geometry')
    roles = {}
    for role in selection.roles:
        rows = role_rows.get(role)
        if type(rows) is not int or not 0 < rows <= (1 << 31)-1:
            raise ValueError('HC role rows require an admitted positive bounded integer')
        q, tail = divmod(rows, 32)
        refs = 32*q*(q+1)//2 + tail*(q+1)
        roles[role] = {'rows': rows, 'input_half_elements': 32*10240,
                       'output_float_elements': 32*10240,
                       'partial_float_elements': max(rows, 32)*324 if role == 'decode' else 0,
                       'program_tuple_bound': 3*(rows+1), 'program_step_reference_bound': 3*refs,
                       'unique_step_bound': 3*rows, 'tensor_view_bound': 24*rows+32}
    padded = 96*8*10240*2  # Original4 rows remain retained; charge full derived8.
    tensors = padded + sum(2*v['input_half_elements']+4*(v['output_float_elements']+v['partial_float_elements'])
                           for v in roles.values())
    if tensors >= (1 << 63) or any(v['program_step_reference_bound'] >= (1 << 63) for v in roles.values()):
        raise ValueError('HC resource arithmetic exceeds its supported range')
    return {'roles': roles, 'padded_inject_weights': 96, 'padded_weight_bytes': padded,
            'tensor_bytes': tensors, 'bound_HC_faces': 194, 'original_HC_weight_faces':290, 'padded_weight_views':96, 'native_libraries': 3,
            'host_catalogue': 'finite row-program/view/reference bounds; Python/native metadata residency measured cold',
            'opaque_residency': 'loaded ELF/CUDA runtime and host Tensor metadata measured before service admission'}


def _plain(value):
    return (type(value) is torch.Tensor and not value.requires_grad
            and not value.is_neg() and not value.is_conj())


def _matrix(matrix, n, k, device):
    from .gguf_weights import GGUFDense

    if (type(matrix) is not GGUFDense or (matrix.n, matrix.k) != (n, k)
            or not _plain(matrix.weight) or matrix.weight.device != device
            or matrix.weight.dtype != torch.float16 or not matrix.weight.is_contiguous()
            or tuple(matrix.weight.shape) != (n, k) or matrix.weight.data_ptr() % 16):
        raise ValueError('HC binding requires original contiguous aligned Half Dense weights')
    return matrix.weight


@dataclass(frozen=True)
class _Binding:
    kind: str
    n: int
    weights: tuple


@dataclass(frozen=True)
class _Step:
    source: object
    input_prefix: object
    input32: object
    output32: object
    active_output: object
    target: object
    inject_output32: object = None
    inject_active_output: object = None
    inject_target: object = None

    def run(self, binding, kernels):
        self.input_prefix.copy_(self.source)
        kernel = kernels['hc-up' if binding.kind == 'up' else 'hc-down']
        library.call_library(kernel, self.input32, binding.weights[0], self.output32, torch)
        self.target.copy_(self.active_output)
        if self.inject_target is not None:
            library.call_library(kernels['hc-inject'], self.input32, binding.weights[1], self.inject_output32, torch)
            self.inject_target.copy_(self.inject_active_output)


class Owner:
    @classmethod
    def prepare(cls, selection, text, role_rows, device, *, world=8, register):
        owner = cls()
        owner.selection = selection
        owner.resources = forecast(selection, text, role_rows, world=world)
        owner.device = torch.device(device)
        if owner.device.type != 'cuda' or owner.device.index is None:
            raise ValueError('HC preparation requires an explicit CUDA device')
        owner._thread = threading.current_thread()
        owner.closed = owner.poisoned = owner.failed_retirement = owner.bound_weights = owner.adopted = False
        owner.binding_started = owner.adoption_started = False
        owner._arenas, owner._kernels, owner._weights, owner._bindings, owner._frames = {}, {}, [], {}, {}
        owner._attachments = []
        owner._padded = None
        owner._device_resources = False
        owner._model_identity = None
        owner._hcs = ()
        owner.preparation_memory = owner.binding_memory = owner.adoption_memory = None
        register(owner)  # Journal all partial ownership before library/device effects.
        manifest = library.hc_manifest(selection.manifest, selection.manifest_sha256)
        owner.manifest = manifest
        with torch.cuda.device(owner.device):
            if torch.cuda.is_current_stream_capturing() or torch.cuda.get_device_capability() != (7, 0):
                raise RuntimeError('HC preparation requires an uncaptured genuine SM70 context')
            before = owner._memory()
            owner._device_resources = True
            for role in library.HC_KERNELS:
                item = manifest['kernels'][role]
                owner._kernels[role] = library.load_verified_library(Path(selection.manifest).parent/role,
                    item['record'], item['dimensions'], item['alignment_bytes'])
            owner._padded = torch.zeros((96,8,10240),dtype=torch.float16,device=device)
            owner._weights = tuple(owner._padded[index] for index in range(96))
            for role, resource in owner.resources['roles'].items():
                owner._arenas[role] = arena = {}
                arena['input'] = torch.zeros(resource['input_half_elements'], dtype=torch.float16, device=device)
                arena['output'] = torch.empty(resource['output_float_elements'], dtype=torch.float32, device=device)
                arena['partial'] = torch.empty(resource['partial_float_elements'], dtype=torch.float32, device=device)
        owner.preparation_memory = {'before':before,'after':owner._memory(),
                                    'scope':'all funded tensor arenas/padded weights plus three library/module initializations'}
        return owner

    def _memory(self):
        torch.cuda.synchronize(self.device)
        free,total = torch.cuda.mem_get_info(self.device)
        return {'physical_free_bytes': int(free), 'physical_total_bytes': int(total),
                'torch_allocated_bytes': int(torch.cuda.memory_allocated(self.device)),
                'torch_reserved_bytes': int(torch.cuda.memory_reserved(self.device)),
                'host_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}

    def _check(self):
        if self.closed or self.poisoned or self.failed_retirement:
            raise RuntimeError('HC owner is closed or retirement uncertain')
        if threading.current_thread() is not self._thread:
            raise RuntimeError('HC owner belongs to its current host thread')

    def bind_weights(self, weights):
        from .gguf_weights import GGUFConcat

        self._check()
        if (self.binding_started or weights.device != self.device
                or weights.meta.get('world') != 8 or weights.meta.get('execution_provider') != 'sm70'
                or len(weights.layers) != 48):
            raise ValueError('HC binding requires one matching actual TP8 target')
        if self._padded.is_inference() and not torch.is_inference_mode_enabled():
            raise RuntimeError('HC inference weights require inference-mode binding')
        self.binding_started = True
        self._model_identity = weights
        before = self._memory()
        hcs = tuple(hc for layer in weights.layers for hc in (layer.attn_hc, layer.mlp_hc)) + (weights.mixer,)
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('HC weights bind before capture and service admission')
            for index, hc in enumerate(hcs):
                injecting = index < 96
                if bool(hc.inject) != injecting:
                    raise ValueError('HC target injection inventory differs')
                if injecting:
                    if type(hc.down) is not GGUFConcat or len(hc.down.parts) != 2 or (hc.down.n, hc.down.k) != (324, 10240):
                        raise ValueError('HC down requires original320+4 Concat parts')
                    down = _matrix(hc.down.parts[0], 320, 10240, self.device)
                    original = _matrix(hc.down.parts[1], 4, 10240, self.device)
                    padded = self._weights[index]
                    padded[:4].copy_(original)
                    self._bindings[hc.down] = _Binding('down', 324, (down, padded))
                else:
                    down = _matrix(hc.down, 320, 10240, self.device)
                    self._bindings[hc.down] = _Binding('down', 320, (down,))
                up = _matrix(hc.up, 10240, 320, self.device)
                self._bindings[hc.up] = _Binding('up', 10240, (up,))
            self._kernels = MappingProxyType(self._kernels)
            self._bindings = MappingProxyType(self._bindings)
            actual = sum(t.numel()*t.element_size() for arena in self._arenas.values() for t in arena.values())
            actual += sum(t.numel()*t.element_size() for t in self._weights)
            if (actual != self.resources['tensor_bytes'] or len(self._bindings) != self.resources['bound_HC_faces']
                    or len(self._weights) != self.resources['padded_inject_weights']):
                raise MemoryError('HC actual weight/scratch inventory differs from its funded forecast')
            torch.cuda.synchronize(self.device)
        # Retain original wrappers/weight owners as well as each borrowed Tensor.
        self._hcs = hcs
        self.bound_weights = True
        self.binding_memory = {'before': before, 'after': self._memory(),
                               'scope': 'cold original-weight binding plus immutable inject copies'}
        return self.record()

    def adopt(self, weights, decoder):
        self._check()
        if not self.bound_weights or self.adoption_started or weights is not self._model_identity:
            raise RuntimeError('HC adoption requires one matching prepared weight owner')
        self.adoption_started = True
        before = self._memory()
        for role in self.selection.roles:
            buffers = decoder.buf if role == 'decode' else decoder.pbuf
            if getattr(buffers, 'dense_linears', None) is not None:
                raise ValueError('HC owner cannot overwrite an existing prepared dense owner')
            frame = Frame(self, buffers, role)
            self._frames[role] = frame
        after = self._memory()
        if any(before[k] != after[k] for k in ('torch_allocated_bytes','torch_reserved_bytes')):
            raise RuntimeError('HC Buffer catalogue adoption unexpectedly allocated device storage')
        # Publish after every role and resource check prepared successfully.
        for role, frame in self._frames.items():
            buffers = decoder.buf if role == 'decode' else decoder.pbuf
            self._attachments.append((weakref.ref(buffers), frame))
            buffers.dense_linears = frame
        self.adopted = True
        self.adoption_memory = {'before': before, 'after': after, 'scope': 'sealed host view/row-program preparation'}
        return self.record()

    def handoff_to_current_thread(self):
        if self.closed or self.poisoned or self.failed_retirement:
            raise RuntimeError('HC owner is closed or retirement uncertain')
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('HC handoff requires prior worker/graph retirement')
            try:
                torch.cuda.synchronize(self.device)
            except BaseException:
                self.failed_retirement = True
                raise
        self._thread = threading.current_thread()

    def close(self):
        if self.closed:
            return
        if self.failed_retirement:
            raise RuntimeError('HC retirement is uncertain; retry is forbidden')
        if not self._device_resources:
            self.closed = True
            return
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('HC close requires graph retirement before resource release')
            try:
                torch.cuda.synchronize(self.device)
            except BaseException:
                self.failed_retirement = True
                raise
        for ref, frame in self._attachments:
            buffers = ref()
            if buffers is not None and getattr(buffers, 'dense_linears', None) is frame:
                buffers.dense_linears = None
        for frame in self._frames.values():
            frame.close()
        self._frames, self._arenas, self._bindings, self._kernels = {}, {}, {}, {}
        self._weights, self._attachments, self._hcs = (), [], ()
        self._padded = None
        self._model_identity = None
        self.closed = True

    def record(self):
        return {'policy': POLICY, 'selection': self.selection.as_dict(), 'resources': self.resources,
                'bound_weights': self.bound_weights, 'adopted': self.adopted,
                'native_libraries': {role: {'dimensions': list(value.dimensions), 'native': value.native_provider,
                    'runtime': value.runtime_provider} for role,value in self._kernels.items()},
                'preparation_memory': self.preparation_memory, 'binding_memory': self.binding_memory, 'adoption_memory': self.adoption_memory,
                'frames': {role: frame.record() for role, frame in self._frames.items()},
                'numerical_policy': 'original separate HC parts; fixed physical M32 Half inputs/F32 accumulation; explicit Half up/prefill cast',
                'scope': 'target HC only; all other Dense and MTP retain their existing provider',
                'quality_acceptance': False, 'performance_acceptance': False}


class Frame:
    def __init__(self, owner, buffers, role):
        if (buffers.owner_role != role or buffers.execution_dtype != torch.float16
                or buffers.rows != owner.resources['roles'][role]['rows']):
            raise ValueError('HC frame requires its exact admitted Half Buffers role')
        self._owner, self.role, self.rows = owner, role, buffers.rows
        self.closed = False
        self._bindings = owner._bindings
        self._programs, self._domains, self._partials = {}, {}, {}
        self._arena = owner._arenas[role]
        self._read_inputs = (buffers.normed,buffers.act)
        self._views = 0
        for kind, n in (('down', 324), ('down', 320), ('up', 10240)):
            source = buffers.act if kind == 'up' else buffers.normed
            if kind == 'up':
                target = buffers.up
            elif role == 'decode':
                target = self._arena['partial'][:max(self.rows, 32)*n].view(-1, n)
            else:
                target = buffers.dn if n == 324 else buffers.dn_mix
            self._prepare(kind, n, source, target)
        self._programs = MappingProxyType(self._programs)
        self._domains = MappingProxyType(self._domains)
        self._partials = MappingProxyType(self._partials)
        if self._views > owner.resources['roles'][role]['tensor_view_bound']:
            raise MemoryError('HC row catalogue exceeded its funded finite view bound')

    def _prepare(self, kind, n, source, target):
        k = 320 if kind == 'up' else 10240
        dtype = torch.float32 if kind == 'down' and self.role == 'decode' else torch.float16
        if (not _plain(source) or not _plain(target) or source.dtype != torch.float16 or target.dtype != dtype
                or source.device != self._owner.device or target.device != source.device
                or tuple(source.shape) != (self.rows, k) or target.shape[0] < self.rows or target.shape[1] != n
                or source.stride(1) != 1 or target.stride(1) != 1
                or source.stride(0) < k or target.stride(0) < n):
            raise ValueError('HC frame Buffer input/output geometry or storage differs')
        if (any(torch._C._overlaps(read,target) for read in self._read_inputs)
                or any(torch._C._overlaps(source,t) for t in self._arena.values())
                or any(torch._C._overlaps(target,self._arena[k]) for k in ('input','output'))
                or any(torch._C._overlaps(target,w) for binding in self._bindings.values() for w in binding.weights)):
            raise ValueError('HC frame Buffer storage aliases its inputs, weights or execution scratch')
        a32 = self._arena['input'][:32*k].view(32, k)
        c32 = self._arena['output'][:32*(10240 if kind == 'up' else 320)].view(32, -1)
        inject32 = self._arena['output'][:32*8].view(32, 8) if n == 324 else None
        programs, full, views = [None], [], 0
        for rows in range(1, self.rows+1):
            start = (rows-1)//32*32
            count = rows-start
            step = _Step(source[start:rows], a32[:count], a32, c32, c32[:count], target[start:rows, :c32.shape[1]],
                         inject32, None if inject32 is None else inject32[:count, :4],
                         None if inject32 is None else target[start:rows, 320:324])
            views += 4 + (2 if inject32 is not None else 0)
            programs.append(tuple(full)+(step,))
            if count == 32:
                full.append(step)
        key = (kind, n)
        self._programs[key] = tuple(programs)
        self._domains[key] = (source.data_ptr(), source.stride(), target.data_ptr(), target.stride(), dtype)
        if kind == 'down' and self.role == 'decode':
            values = [None]
            for r in range(1,self.rows+1):
                active = target[:r]
                values.append((active.unsqueeze(0),active))
            self._partials[n] = tuple(values)
            views += 2*self.rows
        self._views += views+8

    def matmul(self, x, matrix, out):
        self._owner._check()
        if self.closed:
            raise RuntimeError('HC frame is closed')
        binding = self._bindings.get(matrix)
        if binding is None:
            return None
        key = (binding.kind, binding.n)
        source_ptr, source_stride, out_ptr, out_stride, dtype = self._domains[key]
        if (not _plain(x) or not _plain(out) or x.device != self._owner.device or out.device != x.device
                or x.dtype != torch.float16 or out.dtype != dtype or x.ndim != 2 or out.ndim != 2
                or not 0 < x.shape[0] <= self.rows or out.shape != (x.shape[0], binding.n)
                or x.data_ptr() != source_ptr or out.data_ptr() != out_ptr
                or x.stride() != source_stride or out.stride() != out_stride
                or (out.is_inference() or self._arena['input'].is_inference()) and not torch.is_inference_mode_enabled()):
            raise ValueError('HC call differs from its sealed Buffer face/row domain')
        try:
            for step in self._programs[key][x.shape[0]]:
                step.run(binding, self._owner._kernels)
        except BaseException:
            self._owner.poisoned = True
            raise
        return out

    def partials(self, matrix, x):
        self._owner._check()
        binding = self._bindings.get(matrix)
        if binding is None:
            return None
        if self.role != 'decode' or binding.kind != 'down' or not 0 < x.shape[0] <= self.rows:
            raise ValueError('HC partials require their decode down row domain')
        result, target = self._partials[binding.n][x.shape[0]]
        self.matmul(x, matrix, target)
        return result

    def record(self):
        resource = self._owner.resources['roles'][self.role]
        return {'role': self.role, 'rows': self.rows, 'row_program_families': 3,
                'prepared_tensor_views': self._views, 'program_step_reference_bound': resource['program_step_reference_bound'],
                'runtime': 'precreated Tensor views/current-stream copies and kernels; inactive M32 inputs initialized, inactive outputs not published'}

    def close(self):
        self._programs, self._domains, self._partials, self._bindings = {}, {}, {}, {}
        self._arena = {}
        self._read_inputs = ()
        self.closed = True

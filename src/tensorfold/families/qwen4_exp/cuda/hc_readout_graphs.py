"""Unselected startup-owned live-prefix M32 graphs of the incumbent GGUF HC readout.

The caller supplies an already joint-admitted additional graph budget, proves
exclusive/quiescent pre-listener preparation, and retains the ordinary serialized
Buffers frame through output consumption. The predicate below is a guard, not
proof of those obligations. This module never enrolls itself or overrides the
caller's --no-graphs policy. It changes no readout operators or GEMM shapes.
"""
from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace
import sys
import threading
import time

from tensorfold.cleanup import raise_failures

POLICY = 'sm70-gguf-hc-readout-r1-direct-m32-templates-v4'
PHYSICAL_ROWS = 32
MAX_LOGICAL_ROWS = 24


class HCGraphBudgetExceeded(MemoryError):
    """Only an observed caller-budget refusal, never a CUDA/library failure."""


@dataclass
class _Template:
    hc: object
    h: object
    inject: object
    graph: object = None
    pool: object = None
    reset_attempted: bool = False
    reset_error: object = None
    capture_error: object = None
    bindings: tuple = ()
    mutations: tuple = ()
    writes: tuple = ()
    inference_outputs: bool = False
    source: object = None
    sources: tuple = ()
    banks: tuple = ()
    construction: object = None
    progress: object = None
    direct: bool = False
    bank_owner: object = None


@dataclass
class _Entry:
    hc: object
    h: object
    inject: object
    bindings: tuple
    sources: tuple
    template: _Template
    direct: _Template
    replays: int = 0



class _HC32Linears:
    """HC-only input staging; preserve the original separate M32 GEMMs.

    Called only during warmup/capture. Padding must be zero immediately before
    EACH GEMM, including after nonfinite padded normalization/activation values.
    Ordinary Buffers and the optional global PaddedLinears policy are untouched.
    """
    def __init__(self, owner):
        self.owner = owner

    def _input(self, matrix, value):
        owner = self.owner
        stage = owner._staging[:PHYSICAL_ROWS * matrix.k].view(PHYSICAL_ROWS, matrix.k)
        stage.copy_(owner.torch.where(owner._active, value, 0))
        return stage

    def matmul(self, x, matrix, out):
        return matrix.matmul_reference(self._input(matrix, x), out=out, f32=out.dtype == self.owner.torch.float32)

    def partials(self, matrix, x):
        return matrix.matmul_reference(self._input(matrix, x), f32=True).unsqueeze(0)


class HCReadoutGraphs:
    """A fixed 97-key target-only catalogue; every ordinary miss returns False.

    Construct/journal before prepare(). Logical keys share only representation-
    and output-role-compatible templates, each with a separate private pool.
    R1 replays the original program on original weights. R2..24 use fresh
    bitwise weight copies before template replay on the prepared stream. Input
    banks may be shared across role graphs; graph-private pools never are.
    """
    def __init__(self, w, b, original_readout, *, budget_bytes, torch_module=None):
        if type(budget_bytes) is not int or not 0 < budget_bytes <= sys.maxsize:
            raise ValueError('HC graphs require an explicit positive admitted byte budget')
        if not callable(original_readout):
            raise TypeError('the incumbent readout callback is required')
        if torch_module is None:
            import torch as torch_module
        from .gguf_weights import GGUFDense, GGUFConcat

        t, c = torch_module, w.cfg
        if (getattr(b, 'prefill', True) or w.mtp is not None or w.x3 is not None
                or len(w.layers) != 48 or (c.hidden, c.streams, c.low) != (2560, 4, 320)
                or w.comm is None or w.comm.world != 8 or not 0 <= w.comm.rank < 8
                or w.meta.get('execution_provider') != 'sm70' or b.h.device.type != 'cuda'
                or b.h.dtype != t.float16 or b.rows < 1
                or getattr(b, 'padded_linears', None) is not None):
            raise ValueError('HC readout graphs require the target-only SC117 TP8 GGUF decode region')
        self.torch, self.device = t, b.h.device
        self._dense_type, self._concat_type = GGUFDense, GGUFConcat
        self._w, self._b, self._original = w, b, original_readout
        self.budget, self.eps, self.streams, self.low = budget_bytes, c.eps, c.streams, c.low
        if type(self.eps) not in (int, float) or not 0 < self.eps < float('inf'):
            raise ValueError('HC normalization needs its finite positive original epsilon')
        self._lock = threading.Lock()
        self.prepared = self.closed = self.started = False
        self.failure = self.retirement_failure = None
        self.side = self.prepared_stream = None
        self.before = self.after = None
        self.observed_peak_bytes = self.settled_bytes = self.capture_seconds = 0
        # All slots exist before the first CUDA allocation/capture/effect.
        self._errors, self._assembly = [], [None]
        self._entries = [None] * 97
        self._templates = []
        template_map, bank_groups = {}, {}
        self._progress = []
        self._last_stage, self._last_template, self._last_memory = 'not-started', None, None
        self.bank_tensor_bytes = 0
        self._bank_stages = self._total_replays = self._r1_replays = 0
        self.logical_rows = min(b.rows, MAX_LOGICAL_ROWS)
        self._scratch = self._staging = self._row_indices = self._live_rows = self._active = None
        self._physical_inputs = self._workspace_bindings = ()
        self.workspace_tensor_bytes = 0
        self._queued_rows = None
        self._index = {}
        self._leases = []
        self._tensor_names = ('h', 'streams', 'pss', 'normed', 'xs_normed', 'dn', 'dn_mix',
                        'act', 'xs_act', 'inj_a', 'inj_m', 'up', 'mixed', 'xs_mixed')
        self._inputs = tuple(getattr(b, name) for name in self._tensor_names)
        for tensor in self._inputs:
            self._lease(tensor)
        if len({tensor.untyped_storage().data_ptr() for tensor in self._inputs}) != len(self._inputs):
            raise ValueError('HC graph Buffer fields require independent original mutation counters/storage')
        hcs = [(getattr(layer, role), b.h, b.inj_a if role == 'attn_hc' else b.inj_m)
               for layer in w.layers for role in ('attn_hc', 'mlp_hc')]
        hcs.append((w.mixer, b.streams, None))
        for index, (hc, h, inject) in enumerate(hcs):
            matrices = self._matrices(hc)
            if id(hc) in self._index:
                raise ValueError('HC graph keys must have independent identity')
            actual_inject = inject if hc.inject else None
            bindings = self._bindings(hc, matrices)
            sources = tuple(matrix.weight for matrix in matrices) + (hc.scale,)
            role = ('streams' if h is b.streams else 'h',
                    None if actual_inject is None else 'inj_a' if actual_inject is b.inj_a else 'inj_m')
            geometry = (type(hc.down), tuple((matrix.k, matrix.n, matrix.weight.dtype) for matrix in matrices),
                        hc.scale.dtype, tuple(hc.scale.shape))
            key = (role, geometry)
            template = template_map.get(key)
            if template is None:
                # All journals exist before prepare() can allocate device storage.
                progress = [0] * 6  # banks, warm, begin, end, replay, fence
                template = _Template(None, h, actual_inject, source=hc, sources=sources,
                                     construction=[], progress=progress, bank_owner=bank_groups.get(geometry))
                if geometry not in bank_groups:
                    bank_groups[geometry] = template
                template_map[key] = template
                self._templates.append(template)
            direct = _Template(hc, h, None if actual_inject is None else actual_inject[:1],
                               bindings=bindings, progress=[0]*6, direct=True)
            entry = self._entries[index] = _Entry(hc, h, actual_inject, bindings, sources, template, direct)
            self._index[id(hc)] = entry
            for tensor in (hc.scale, *[matrix.weight for matrix in matrices]):
                self._lease(tensor)
        self._owners = (w, b, *[entry.hc for entry in self._entries])
        self.template_count = len(self._templates)
        self.bank_group_count = len(bank_groups)
        self._programs = [*self._templates, *[entry.direct for entry in self._entries]]
        self._progress = [program.progress for program in self._programs]
        self._phase_memory = [[None] * 3 for _ in self._programs]
        self._fence_error, self._restore_error = len(self._programs), len(self._programs)+1
        self._errors = [None] * (len(self._programs)+2)

    def _matrices(self, hc):
        down, up = hc.down, hc.up
        if type(down) is self._concat_type and hc.inject:
            parts = down.parts
            expected = ((10240, 320), (10240, 4))
        else:
            parts = (down,)
            expected = ((10240, 324 if hc.inject else 320),)
        if (type(up) is not self._dense_type or up.implementation != 'reference'
                or (up.k, up.n) != (320, 10240) or down.implementation != 'reference'
                or len(parts) != len(expected)
                or any(type(part) is not self._dense_type or part.implementation != 'reference'
                       or (part.k, part.n) != shape for part, shape in zip(parts, expected))):
            raise ValueError('HC graphs require original Dense/split-Concat reference geometry')
        return (*parts, up)

    @staticmethod
    def _bindings(hc, matrices):
        return (hc.down, hc.up, hc.scale, *matrices, *[matrix.weight for matrix in matrices])

    def _lease(self, tensor):
        if (tensor.device != self.device or tensor.requires_grad or tensor.is_neg() or tensor.is_conj()
                or not tensor.is_contiguous()):
            raise ValueError('HC graph tensors require stable contiguous non-gradient storage')
        storage = tensor.untyped_storage()
        self._leases.append((tensor, storage, self._signature(tensor, storage)))

    @staticmethod
    def _signature(tensor, storage):
        return (tensor.data_ptr(), storage.data_ptr(), storage.nbytes(), tuple(tensor.shape),
                tuple(tensor.stride()), tensor.storage_offset(), tensor.dtype, tensor.device)

    @property
    def enabled(self):
        if self.failure is not None or self.retirement_failure is not None:
            raise RuntimeError('HC graph owner is poisoned') from (self.failure if self.failure is not None else self.retirement_failure)
        return self.prepared and not self.closed

    @property
    def budget_declined(self):
        """Cold classification only after successful retirement, without secondary errors."""
        return (self.closed and isinstance(self.failure, HCGraphBudgetExceeded)
                and self.retirement_failure is None and all(error is None for error in self._errors))

    def _memory(self):
        cuda = self.torch.cuda
        return {'free': int(cuda.mem_get_info(self.device)[0]),
                'allocated': int(cuda.memory_allocated(self.device)),
                'reserved': int(cuda.memory_reserved(self.device))}

    def _observe(self):
        current = self._memory()
        self._last_memory = current  # Numeric failure evidence survives retirement.
        physical = self.before['free'] - current['free']
        active = current['allocated'] - self.before['allocated']
        if physical < 0 or active < 0:
            raise RuntimeError('HC preparation residency is not attributable under concurrent memory changes')
        self.observed_peak_bytes = max(self.observed_peak_bytes, physical, active,
                                       current['reserved'] - self.before['reserved'])
        if self.observed_peak_bytes > self.budget:
            raise HCGraphBudgetExceeded('HC readout graphs exceed their caller-admitted preparation budget')
        return current

    def _prepare_frame(self):
        # Journal the frame before allocating any device storage. These tensors
        # belong to the same measured/joint-admitted lifetime as the template pools.
        self._scratch = SimpleNamespace(rows=PHYSICAL_ROWS, prefill=False, hc_plans=None,
                                        padded_linears=_HC32Linears(self))
        with self.torch.inference_mode(False):
            for name, original in zip(self._tensor_names, self._inputs):
                if original.ndim < 2 or original.shape[0] != self._b.rows:
                    raise ValueError('HC frame requires the actual complete row-major Buffer fields')
                value = self.torch.zeros((PHYSICAL_ROWS, *original.shape[1:]),
                                         dtype=original.dtype, device=self.device)
                setattr(self._scratch, name, value)
                self._lease(value)
            self._staging = self.torch.zeros((PHYSICAL_ROWS * 10240,), dtype=self.torch.float16, device=self.device)
            self._live_rows = self.torch.zeros((), dtype=self.torch.int32, device=self.device)
            self._row_indices = self.torch.arange(PHYSICAL_ROWS, device=self.device).view(PHYSICAL_ROWS, 1)
        for value in (self._staging, self._live_rows, self._row_indices):
            self._lease(value)
        self._physical_inputs = tuple(getattr(self._scratch, name) for name in self._tensor_names)
        self._workspace_bindings = tuple((name, getattr(self, name)) for name in
                                         ('_scratch', '_staging', '_live_rows', '_row_indices'))
        self.workspace_tensor_bytes = sum(value.numel() * value.element_size() for value in
                                         (*self._physical_inputs, self._staging, self._live_rows, self._row_indices))
        self._live_rows.fill_(self.logical_rows)

    def _prepare_banks(self, template):
        # The private HC has the exact original wrapper type/scalars. Original
        # key weights and their metadata are never modified or rebound.
        template.hc = copy(template.source)
        if template.bank_owner is not None:
            # These are shared INPUT banks, not shared graph-private pools.
            # The same prepared stream serializes the prior reader and the next
            # key's overwrite across both output-role-specific template graphs.
            leader = template.bank_owner
            template.hc.down, template.hc.up, template.hc.scale = leader.hc.down, leader.hc.up, leader.hc.scale
            template.hc.prefill_down, template.hc.prefill_up = leader.hc.down, leader.hc.up
            template.banks = leader.banks
            template.bindings = self._bindings(template.hc, self._matrices(template.hc))
            template.progress[0] = 1
            return
        def bank(source):
            value = self.torch.zeros(tuple(source.shape), dtype=source.dtype, device=self.device)
            template.construction.append(value)
            return value
        with self.torch.inference_mode(False):
            source = template.source
            if type(source.down) is self._concat_type:
                template.hc.down = self._concat_type([self._dense_type(bank(part.weight)) for part in source.down.parts])
            else:
                template.hc.down = self._dense_type(bank(source.down.weight))
            template.hc.down.implementation = source.down.implementation
            if type(source.down) is self._concat_type:
                for target, original in zip(template.hc.down.parts, source.down.parts, strict=True):
                    target.implementation = original.implementation
            template.hc.up = self._dense_type(bank(source.up.weight))
            template.hc.up.implementation = source.up.implementation
            template.hc.scale = bank(source.scale)
        template.hc.prefill_down, template.hc.prefill_up = template.hc.down, template.hc.up
        matrices = self._matrices(template.hc)
        template.banks = tuple(matrix.weight for matrix in matrices) + (template.hc.scale,)
        for value in template.banks:
            self._lease(value)
        # Concat may join its temporary construction tensors into one bank.
        # Only final live storages count as retained banks; all construction
        # overlap is still measured by the surrounding preparation phase.
        storages = {value.untyped_storage().data_ptr(): value.untyped_storage() for value in template.banks}
        self.bank_tensor_bytes += sum(storage.nbytes() for storage in storages.values())
        template.construction.clear()
        template.bindings = self._bindings(template.hc, matrices)
        template.progress[0] = 1

    def _stage_key(self, template, sources):
        # Every call refreshes every operand, even for repeated logical keys.
        # Same-stream ordering retires the preceding reader before overwrite.
        # Any partial enqueue failure poisons the whole owner in its caller.
        if self._bank_stages >= sys.maxsize:
            raise OverflowError('HC weight staging counter exhausted')
        next_stage = self._bank_stages+1
        for target, source in zip(template.banks, sources, strict=True):
            target.copy_(source)
        self._bank_stages = next_stage

    def _run_original(self, entry, *, learn=False):
        # All ordinary fields were initialized at preparation and remain defined.
        # Fixed-size masks never read beyond those owned fields. No State or TP
        # descriptor is captured; only the validated host row count varies.
        try:
            with self.torch.inference_mode(False):
                self._active = self._row_indices < self._live_rows
                name = 'streams' if entry.h is self._b.streams else 'h'
                physical_h = getattr(self._scratch, name)
                for source, target in ((entry.h, physical_h), (self._b.pss, self._scratch.pss)):
                    shape = (self.logical_rows, *([1] * (source.ndim - 1)))
                    mask = self._active[:self.logical_rows].view(shape)
                    target[:self.logical_rows].copy_(self.torch.where(mask, source[:self.logical_rows], 0))
                # Inputs' remaining physical rows stay zero: the original callback
                # never mutates h/streams/PSS. Every other read is overwrite-first.
                versions = tuple(value._version for value in self._physical_inputs)
                inject = None if entry.inject is None else (
                    self._scratch.inj_a if entry.inject is self._b.inj_a else self._scratch.inj_m)
                self._original(entry.hc, self._scratch, physical_h, PHYSICAL_ROWS, self.eps,
                               self.streams, self.low, inject, normed=False)
                writes = tuple((name, value._version - before) for name, value, before in
                               zip(self._tensor_names, self._physical_inputs, versions)
                               if value._version != before)
                if any(name in ('h', 'streams', 'pss') or count < 1 for name, count in writes):
                    raise RuntimeError('HC original callback changed a readonly input or invalid mutation count')
                if learn:
                    entry.writes = writes
                    entry.inference_outputs = any(getattr(self._b, name).is_inference() for name, _ in writes)
                elif writes != entry.writes:
                    raise RuntimeError('HC original callback mutation contract changed during capture')
            # Masked commits preserve every inactive suffix bit. CPU mutation
            # multiplicities are learned from the actual original callback, not
            # reimplemented HC arithmetic. Replay publishes the captured deltas.
            for name, count in entry.writes:
                target = getattr(self._b, name)[:self.logical_rows]
                source = getattr(self._scratch, name)[:self.logical_rows]
                mask = self._active[:self.logical_rows].view((self.logical_rows, *([1] * (target.ndim - 1))))
                for _ in range(count):
                    target.copy_(self.torch.where(mask, source, target))
        finally:
            self._active = None

    def _run_program(self, entry, *, learn=False):
        if entry.direct:
            # The R1 specialization is the original program on original weights
            # and ordinary Buffers. It never touches M32 scratch/banks/row scalar.
            return self._original(entry.hc, self._b, entry.h, 1, self.eps,
                                  self.streams, self.low, entry.inject, normed=False)
        return self._run_original(entry, learn=learn)

    def _capture(self, entry):
        versions = tuple(None if tensor.is_inference() else tensor._version for tensor in self._inputs)
        entry.pool = self.torch.cuda.graph_pool_handle()
        entry.graph = self.torch.cuda.CUDAGraph()  # Journal before capture_begin.
        entry.graph.capture_begin(pool=entry.pool, capture_error_mode='thread_local')
        entry.progress[2] = 1
        primary = None
        try:
            self._run_program(entry)
        except BaseException as error:
            primary = error
        try:
            entry.graph.capture_end()  # Once, even when the original readout failed.
            entry.progress[3] = 1
        except BaseException as error:
            entry.capture_error = error
        if entry.capture_error is not None:
            self._assembly[0] = entry.capture_error
            raise_failures(primary, self._assembly)
        if primary is not None:
            raise_failures(primary, ())
        # Replay bypasses ATen's CPU mutation bookkeeping. Preserve the fixed
        # original program's increments without duplicating its arithmetic.
        entry.mutations = tuple(tensor for tensor, before in zip(self._inputs, versions)
                                if before is not None for _ in range(tensor._version - before))
        entry.graph.replay()  # Initialize every captured output before consumers.
        entry.progress[4] = 1

    def prepare(self, *, quiescent):
        """Caller proves joint funding/exclusivity; capture only once at startup.

        The supplied guard must observe the actual pre-listener lifetime owner,
        not an HTTP idle snapshot. No State or global graph policy is changed.
        """
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('HC graph owner already has an active operation')
        try:
            if (self.started or self.closed or self.failure is not None
                    or not callable(quiescent) or quiescent() is not True):
                raise RuntimeError('HC graphs need one-time exclusive pre-listener preparation')
            if self._b.hc_plans is not None and self._b.hc_plans is not self:
                raise RuntimeError('HC graph preparation cannot capture another active plan')
            self.validate_adoption()
        except BaseException:
            self._lock.release()
            raise
        previous = primary = None
        try:
            self.started = True
            cuda = self.torch.cuda
            previous = self.prepared_stream = cuda.current_stream(self.device)
            cuda.synchronize(self.device)
            cuda.empty_cache()
            self.before = self._memory()
            self.side = cuda.Stream(device=self.device)
            self.side.wait_stream(previous)
            cuda.set_stream(self.side)
            start = time.monotonic()
            # Existing inputs have defined zero semantics for every warm/capture.
            for tensor in self._inputs:
                tensor.zero_()
            self._last_stage = 'frame'
            self._prepare_frame()
            self._observe()
            for index, template in enumerate(self._templates):
                self._last_template, self._last_stage = index, 'banks'
                self._prepare_banks(template)
                cuda.synchronize(self.device)
                self._phase_memory[index][0] = self._observe()
                self._last_stage = 'weight-copy'
                self._stage_key(template, template.sources)
                self._last_stage = 'warm'
                self._run_program(template, learn=True)
                cuda.synchronize(self.device)
                template.progress[1] = 1
                self._phase_memory[index][1] = self._observe()
                cuda.empty_cache()
                self._last_stage = 'capture'
                self._capture(template)
                cuda.synchronize(self.device)
                template.progress[5] = 1
                self._last_stage = 'post-capture'
                self._phase_memory[index][2] = self._observe()
            for offset, logical in enumerate(self._entries):
                index, direct = self.template_count+offset, logical.direct
                self._last_template, self._last_stage = index, 'r1-warm'
                direct.inference_outputs = logical.template.inference_outputs
                self._run_program(direct)
                cuda.synchronize(self.device)
                direct.progress[1] = 1
                self._phase_memory[index][1] = self._observe()
                cuda.empty_cache()
                self._last_stage = 'r1-capture'
                self._capture(direct)
                cuda.synchronize(self.device)
                direct.progress[5] = 1
                self._last_stage = 'r1-post-capture'
                self._phase_memory[index][2] = self._observe()
            previous.wait_stream(self.side)
            cuda.synchronize(self.device)
            cuda.empty_cache()
            self._last_stage = 'settled'
            self.after = self._observe()
            self.settled_bytes = max(self.before['free'] - self.after['free'],
                                     self.after['allocated'] - self.before['allocated'])
            self.capture_seconds = time.monotonic() - start
            self.validate_adoption()
        except BaseException as error:
            primary = self.failure = error
        if previous is not None:
            try:
                self.torch.cuda.set_stream(previous)
            except BaseException as error:
                self._errors[self._restore_error] = error
                if primary is None:
                    primary = self.failure = error
        try:
            if primary is not None:
                self._finish_retirement(primary)
            self.prepared = True
            # Capture only reads this private scalar. Its initial fill and all
            # preparation work have positively retired before publication.
            self._queued_rows = self.logical_rows
            try:
                result = self.record()
            except BaseException as error:
                self.failure = error
                self._finish_retirement(error)
        finally:
            self._lock.release()
        return result

    def validate_adoption(self):
        """Cold boundary only; caller cannot resize/rebind tensors afterwards."""
        if self.closed:
            raise RuntimeError('HC graph owner is closed')
        roster = [getattr(layer, role) for layer in self._w.layers for role in ('attn_hc', 'mlp_hc')]
        roster.append(self._w.mixer)
        if len(roster) != 97 or any(hc is not entry.hc for hc, entry in zip(roster, self._entries)):
            raise RuntimeError('HC graph catalogue was rebound')
        if any(getattr(self._b, name) is not tensor for name, tensor in zip(self._tensor_names, self._inputs)):
            raise RuntimeError('HC graph Buffers tensor was rebound')
        for tensor, storage, signature in self._leases:
            if self._signature(tensor, tensor.untyped_storage()) != signature or storage.data_ptr() != signature[1]:
                raise RuntimeError('HC graph tensor/storage changed after enrollment')
        if self._scratch is not None:
            if (any(getattr(self, name) is not value for name, value in self._workspace_bindings)
                    or any(getattr(self._scratch, name) is not value
                           for name, value in zip(self._tensor_names, self._physical_inputs))):
                raise RuntimeError('HC physical frame was rebound')
        for entry in self._entries:
            current = self._bindings(entry.hc, self._matrices(entry.hc))
            if len(current) != len(entry.bindings) or any(a is not b for a, b in zip(current, entry.bindings)):
                raise RuntimeError('HC graph weight or scale was rebound')

        for template in self._templates:
            if template.hc is not None and template.banks:
                current = self._bindings(template.hc, self._matrices(template.hc))
                if len(current) != len(template.bindings) or any(a is not b for a, b in zip(current, template.bindings)):
                    raise RuntimeError('HC private weight bank was rebound')

    def __call__(self, hc, h, R, eps, streams, low, inject, normed=False):
        if self.closed or self.failure is not None or self.retirement_failure is not None:
            raise RuntimeError('HC graph owner is unavailable') from (self.failure if self.failure is not None else self.retirement_failure)
        if not self.prepared:
            return False
        if type(R) is not int or not 1 <= R <= self.logical_rows or normed is not False or eps != self.eps or streams != self.streams or low != self.low:
            return False
        entry = self._index.get(id(hc))
        if entry is None or entry.hc is not hc or h is not entry.h:
            return False
        if ((inject is None) != (entry.inject is None)
                or inject is not None and (inject.data_ptr() != entry.inject.data_ptr()
                    or tuple(inject.shape) != (R, self.streams) or inject.dtype != entry.inject.dtype
                    or inject.device != entry.inject.device or not inject.is_contiguous()
                    or inject.requires_grad or inject.is_neg() or inject.is_conj())):
            return False
        program = entry.direct if R == 1 else entry.template
        if program.inference_outputs and not self.torch.is_inference_mode_enabled():
            return False  # Retain the original ATen inference-mutation refusal.
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('HC graph replay overlaps another operation')
        try:
            if self.closed or self.failure is not None or self.retirement_failure is not None:
                raise RuntimeError('HC graph owner retired before replay admission') from (self.failure if self.failure is not None else self.retirement_failure)
            if self.torch.cuda.current_stream(self.device) != self.prepared_stream:
                raise RuntimeError('HC graph requires its prepared serialized current stream')
            if entry.replays >= sys.maxsize or self._total_replays >= sys.maxsize:
                raise OverflowError('HC graph replay counter exhausted')
            next_entry = entry.replays+1
            next_total = self._total_replays+1
            next_r1 = self._r1_replays+(R == 1)
            if R != 1:
                # Direct R1 never uses or invalidates this M32-only value cache.
                if self._queued_rows != R:
                    self._live_rows.fill_(R)
                    self._queued_rows = R
                self._stage_key(entry.template, entry.sources)
            program.graph.replay()
            if program.mutations:
                self.torch.autograd.graph.increment_version(program.mutations)
            entry.replays = next_entry
            self._total_replays = next_total
            self._r1_replays = next_r1
        except BaseException as error:
            self._queued_rows = None
            self.failure = error
            raise
        finally:
            self._lock.release()
        return True

    def _transport(self, primary):
        first = primary
        for error in self._errors:
            if first is None and error is not None:
                first = error
        if first is None:
            return
        try:
            errors = [error for error in self._errors if error is not None]
        except BaseException as error:
            self._assembly[0] = error
            raise_failures(first, self._assembly)
        raise_failures(first, errors)

    def _retire(self, primary=None):
        self._queued_rows = None
        if self.retirement_failure is not None:
            self._assembly[0] = self.retirement_failure
            raise_failures(primary, self._assembly)
        if self.started:
            try:
                self.torch.cuda.synchronize(self.device)
            except BaseException as error:
                self.retirement_failure = self._errors[self._fence_error] = error
                self._transport(primary)
        for index, entry in enumerate(self._programs):
            if entry.graph is None:
                continue
            entry.reset_attempted = True  # A failed native reset is never retried.
            try:
                entry.graph.reset()
            except BaseException as error:
                entry.reset_error = self._errors[index] = error
                if self.retirement_failure is None:
                    self.retirement_failure = error
            else:
                entry.graph = entry.pool = None
        if self.retirement_failure is None:
            self._entries.clear()
            self._index.clear()
            self._templates.clear()
            self._programs.clear()
            self._leases.clear()
            self._inputs = self._owners = self._physical_inputs = self._workspace_bindings = ()
            self._scratch = self._staging = self._row_indices = self._live_rows = self._active = None
            self._w = self._b = self._original = self.side = self.prepared_stream = None
            self.closed = True
        self._transport(primary)

    def close(self):
        if self.closed:
            return
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('HC graph retirement overlaps an active operation')
        try:
            self._finish_retirement()
        finally:
            self._lock.release()

    def _finish_retirement(self, primary=None):
        self._queued_rows = None
        # Even cold traversal/status transport can exhaust. An interrupted
        # retirement retains this owner and forbids native retries; the caller
        # must retain it through terminal containment, rather than drop it.
        try:
            self._retire(primary)
        except BaseException as cleanup:
            if cleanup is primary:
                if not self.closed and self.retirement_failure is None:
                    self.retirement_failure = cleanup
                raise
            if self.retirement_failure is None:
                self.retirement_failure = cleanup
            self._assembly[0] = cleanup
            raise_failures(primary, self._assembly)

    def record(self):
        return {'policy': POLICY, 'prepared': self.prepared, 'closed': self.closed,
                'rows': list(range(1, self.logical_rows + 1)), 'physical_rows': PHYSICAL_ROWS, 'catalogue_keys': 97,
                'workspace_tensor_bytes': self.workspace_tensor_bytes,
                'captured_keys': sum(e.direct.graph is not None and e.template.graph is not None for e in self._entries),
                'covered_keys': sum(e.direct.graph is not None and e.template.graph is not None for e in self._entries),
                'r1_graph_count': 97, 'captured_r1_graphs': sum(e.direct.graph is not None for e in self._entries),
                'captured_graphs': sum(program.graph is not None for program in self._programs),
                'bank_group_count': self.bank_group_count,
                'template_rows': list(range(2, self.logical_rows + 1)),
                'template_count': self.template_count, 'captured_templates': sum(t.graph is not None for t in self._templates),
                'bank_tensor_bytes': self.bank_tensor_bytes, 'bank_stages': self._bank_stages,
                'preparation_progress': [dict(path='m32' if index < self.template_count else 'r1',
                    **dict(zip(('banks', 'warm', 'begin', 'end', 'replay', 'fence'), row)))
                    for index, row in enumerate(self._progress)],
                'last_preparation_stage': self._last_stage, 'last_preparation_program': self._last_template,
                'last_memory': self._last_memory,
                'program_memory': [dict(zip(('banks', 'warm', 'capture'), row)) for row in self._phase_memory],
                'replays': self._total_replays, 'r1_replays': self._r1_replays,
                'template_replays': self._total_replays-self._r1_replays,
                'graph_budget_bytes': self.budget,
                'observed_peak_delta_bytes': self.observed_peak_bytes, 'settled_charge_bytes': self.settled_bytes,
                'before': self.before, 'after': self.after, 'capture_seconds': self.capture_seconds,
                'private_pool_spare_bytes': None, 'pool_policy': 'separate-per-graph; shared-input-banks-only; no private spare credit',
                'qualification': 'source candidate; actual target bits/resources/model/P3 unqualified'}

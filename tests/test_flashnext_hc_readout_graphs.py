"""CPU ownership/dispatch tests; doubles do not establish CUDA bit identity."""
import gc
from contextlib import nullcontext
import math
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch
import weakref

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = 'tensorfold.families.qwen4_exp.cuda.hc_readout_graphs'
WEIGHTS = 'tensorfold.families.qwen4_exp.cuda.gguf_weights'


class Storage:
    def __init__(self, ptr):
        self.ptr = ptr
    def data_ptr(self):
        return self.ptr
    def nbytes(self):
        return 1 << 20


class Tensor:
    next_ptr = 512
    def __init__(self, shape=(4, 10240), storage=None, dtype='half'):
        self.shape, self.device, self.dtype = shape, DEVICE, dtype
        self.requires_grad, self.zeros = False, 0
        self.fills = []
        self.storage = storage or Storage(Tensor.next_ptr)
        if not hasattr(self.storage, 'version'):
            self.storage.version = 0
        Tensor.next_ptr += 512
    @property
    def _version(self):
        return self.storage.version
    @_version.setter
    def _version(self, value):
        self.storage.version = value
    @property
    def ndim(self):
        return len(self.shape)
    def numel(self):
        return math.prod(self.shape)
    def element_size(self):
        return 2 if self.dtype == 'half' else 4
    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.shape[0])
            shape = (len(range(start, stop, step)), *self.shape[1:])
        else:
            shape = self.shape[1:]
        return Tensor(shape, self.storage, self.dtype)
    def view(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        return Tensor(shape, self.storage, self.dtype)
    def __lt__(self, other):
        return Tensor(self.shape, dtype='bool')
    def data_ptr(self):
        return self.storage.ptr
    def untyped_storage(self):
        return self.storage
    def stride(self):
        value, result = 1, []
        for dimension in reversed(self.shape):
            result.insert(0, value)
            value *= dimension
        return tuple(result)
    def storage_offset(self):
        return 0
    def is_neg(self):
        return False
    def is_inference(self):
        return False
    def is_conj(self):
        return False
    def is_contiguous(self):
        return True
    def zero_(self):
        self.zeros += 1
        self._version += 1
        return self
    def fill_(self, value):
        self.fills.append(value)
        self.filled = value
        self._version += 1
        return self
    def copy_(self, source):
        self.last_source = source
        self._version += 1
        return self


def fake_torch(cuda):
    def increment(tensors):
        for tensor in tensors:
            tensor._version += 1
    return NS(autograd=NS(graph=NS(increment_version=increment)), cuda=cuda, float16='half', float32='float', int32='int',
              inference_mode=lambda enabled: nullcontext(),
              is_inference_mode_enabled=lambda: False,
              zeros=lambda shape, dtype, device: Tensor(shape, dtype=dtype),
              arange=lambda rows, device: Tensor((rows,), dtype='long'),
              where=lambda mask, value, other: Tensor(value.shape, dtype=value.dtype))


DEVICE = NS(type='cuda', index=0)


class Dense:
    def __init__(self, k, n=None):
        if isinstance(k, Tensor):
            self.weight = k
            self.n, self.k = k.shape
        else:
            self.k, self.n, self.weight = k, n, Tensor((n, k))
        self.implementation = 'reference'


class Concat:
    implementation = 'reference'
    def __init__(self, parts):
        self.parts = tuple(parts)
        self.k, self.n = parts[0].k, sum(part.n for part in parts)


class Stream:
    def __init__(self):
        self.waits = []
    def wait_stream(self, other):
        self.waits.append(other)


class CUDA:
    def __init__(self):
        self.stream = Stream()
        self.events, self.graphs, self.failures = [], [], {}
        self.resident = 0
    def event(self, name):
        self.events.append(name)
        fault = self.failures.get(name)
        if fault is not None:
            raise fault
    def current_stream(self, device):
        self.event('current_stream')
        return self.stream
    def set_stream(self, stream):
        self.event('set_stream')
        self.stream = stream
    def Stream(self, device):
        self.event('stream_create')
        return Stream()
    def synchronize(self, device):
        self.event('fence')
    def empty_cache(self):
        self.event('empty_cache')
    def mem_get_info(self, device):
        self.event('memory')
        return 10**9-self.resident, 10**9
    def memory_allocated(self, device):
        return self.resident
    def memory_reserved(self, device):
        return self.resident
    def graph_pool_handle(self):
        self.event('pool')
        return object()
    def CUDAGraph(self):
        cuda = self
        class Graph:
            resets = 0
            replays = 0
            def capture_begin(self, **kwargs):
                self.pool = kwargs['pool']
                cuda.event('begin')
                cuda.resident += 100
            def capture_end(self):
                cuda.event('end')
            def replay(self):
                cuda.event('replay')
                self.replays += 1
            def reset(self):
                self.resets += 1
                cuda.event('reset')
                cuda.resident -= 100
        graph = Graph()
        self.graphs.append(graph)
        return graph


def setup(*, budget=100000, callback=None, rows=4, split=False):
    fake_weights = ModuleType(WEIGHTS)
    fake_weights.GGUFDense = Dense
    fake_weights.GGUFConcat = Concat
    spec = importlib.util.spec_from_file_location(MODULE, ROOT/'src/tensorfold/families/qwen4_exp/cuda/hc_readout_graphs.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {MODULE: module, WEIGHTS: fake_weights}):
        spec.loader.exec_module(module)
        def hc(inject=True):
            return NS(inject=inject, down=Concat([Dense(10240, 320), Dense(10240, 4)]) if split and inject else Dense(10240, 324 if inject else 320),
                      up=Dense(320, 10240), scale=Tensor())
        w = NS(cfg=NS(hidden=2560, streams=4, low=320, eps=1e-6), mtp=None, x3=None,
               layers=[NS(attn_hc=hc(), mlp_hc=hc()) for _ in range(48)], mixer=hc(False),
               comm=NS(world=8, rank=0), meta={'execution_provider': 'sm70'})
        b = NS(prefill=False, rows=rows, hc_plans=None)
        for name in ('h', 'streams', 'pss', 'normed', 'xs_normed', 'dn', 'dn_mix',
                     'act', 'xs_act', 'inj_a', 'inj_m', 'up', 'mixed', 'xs_mixed'):
            setattr(b, name, Tensor((rows, 4) if name.startswith('inj_') else (rows, 10240)))
        calls, cuda = [], CUDA()
        def original(*args, **kwargs):
            calls.append((args, kwargs))
            if callback is not None:
                callback(cuda, calls)
        owner = module.HCReadoutGraphs(w, b, original, budget_bytes=budget,
                                      torch_module=fake_torch(cuda))
    return owner, w, b, cuda, calls


def invoke(owner, w, b, **changes):
    values = dict(hc=w.layers[0].attn_hc, h=b.h, R=1, eps=w.cfg.eps,
                  streams=4, low=320, inject=b.inj_a[:1], normed=False)
    values.update(changes)
    return owner(**values)


def test_catalogue_prejournals_without_cuda_and_captures_incumbent_callback():
    owner, w, b, cuda, calls = setup()
    assert not cuda.events and len(owner._entries) == 97 and not owner.enabled
    b.hc_plans = owner
    record = owner.prepare(quiescent=lambda: True)
    assert record['captured_keys'] == 97 and record['settled_charge_bytes'] == 10000
    assert record['captured_templates'] == 3 and record['captured_r1_graphs'] == 97
    assert len({id(graph.pool) for graph in cuda.graphs}) == record['captured_graphs'] == 100
    assert len(calls) == 200 and all(t.zeros == 1 for t in owner._inputs)
    assert all(args[3:7] == (32 if index < 6 else 1, w.cfg.eps, 4, 320)
               and kwargs == {'normed': False} for index, (args, kwargs) in enumerate(calls))
    assert calls[-1][0][2] is b.streams and calls[-1][0][7] is None
    assert record['rows'] == [1, 2, 3, 4] and record['physical_rows'] == 32
    assert invoke(owner, w, b) is True and owner._entries[0].direct.graph.replays == 2
    owner.close()
    assert owner.closed and all(graph.resets == 1 for graph in cuda.graphs)
    owner.close()


@pytest.mark.parametrize('changes', [dict(R=5), dict(R=True), dict(normed=True), dict(eps=.1),
                                    dict(h=Tensor()), dict(inject=Tensor((1, 4))), dict(hc=object())])
def test_ordinary_miss_does_not_capture_or_replay(changes):
    owner, w, b, cuda, _ = setup()
    assert not invoke(owner, w, b)
    owner.prepare(quiescent=lambda: True)
    before = list(cuda.events)
    assert invoke(owner, w, b, **changes) is False
    assert cuda.events == before
    owner.close()


def test_failed_quiescence_and_repeated_prepare_do_not_destroy_existing_owner():
    owner, w, b, cuda, _ = setup()
    with pytest.raises(RuntimeError, match='pre-listener'):
        owner.prepare(quiescent=lambda: False)
    assert not cuda.events and not owner.started
    owner.prepare(quiescent=lambda: True)
    before = list(cuda.events)
    with pytest.raises(RuntimeError, match='pre-listener'):
        owner.prepare(quiescent=lambda: True)
    assert owner.enabled and cuda.events == before and invoke(owner, w, b)
    owner.close()


@pytest.mark.parametrize('target', ['buffer', 'weight', 'implementation', 'storage'])
def test_adoption_rejects_rebound_cold_contract(target):
    owner, w, b, cuda, _ = setup()
    if target == 'buffer':
        b.h = Tensor()
    elif target == 'weight':
        w.layers[0].attn_hc.down.weight = Tensor()
    elif target == 'implementation':
        w.layers[0].attn_hc.up.implementation = 'native32'
    else:
        b.h.storage.ptr += 512
    with pytest.raises((RuntimeError, ValueError)):
        owner.prepare(quiescent=lambda: True)
    assert not cuda.events and not owner.started
    owner.close()


def test_budget_refusal_fences_and_resets_before_releasing_owners():
    owner, _, _, cuda, _ = setup(budget=1)
    with pytest.raises(MemoryError, match='admitted'):
        owner.prepare(quiescent=lambda: True)
    assert owner.closed and owner.observed_peak_bytes == 100
    assert cuda.graphs[0].resets == 1 and cuda.events[-2:] == ['fence', 'reset']


def test_body_and_capture_end_failures_preserve_primary_and_retire():
    primary, secondary = OSError('original readout'), RuntimeError('capture end')
    def callback(cuda, calls):
        if len(calls) == 2:
            cuda.failures['end'] = secondary
            raise primary
    owner, _, _, cuda, _ = setup(callback=callback)
    with pytest.raises(OSError) as raised:
        owner.prepare(quiescent=lambda: True)
    assert raised.value is primary and raised.value.__cause__ is secondary
    assert owner.closed and cuda.graphs[0].resets == 1


def test_failed_fence_retains_graphs_and_never_retries_uncertain_retirement():
    owner, _, _, cuda, _ = setup()
    owner.prepare(quiescent=lambda: True)
    error = OSError('uncertain device')
    cuda.failures['fence'] = error
    with pytest.raises(OSError) as raised:
        owner.close()
    assert raised.value is error and owner._owners and not owner.closed
    assert all(graph.resets == 0 for graph in cuda.graphs)
    before = list(cuda.events)
    with pytest.raises(OSError):
        owner.close()
    assert cuda.events == before


def test_reset_failure_attempts_independent_graphs_once_and_retains_owners():
    owner, _, _, cuda, _ = setup()
    owner.prepare(quiescent=lambda: True)
    cuda.failures['reset'] = RuntimeError('native reset')
    with pytest.raises(RuntimeError):
        owner.close()
    assert all(graph.resets == 1 for graph in cuda.graphs) and owner._owners
    before = list(cuda.events)
    with pytest.raises(RuntimeError):
        owner.close()
    assert cuda.events == before


def test_replay_failure_poisoning_and_wrong_stream_refusal():
    owner, w, b, cuda, _ = setup()
    owner.prepare(quiescent=lambda: True)
    cuda.stream = Stream()
    with pytest.raises(RuntimeError, match='serialized'):
        invoke(owner, w, b)
    with pytest.raises(RuntimeError, match='poisoned'):
        _ = owner.enabled
    owner.close()


def test_closed_owner_releases_all_retained_buffers_without_gc_cycle():
    owner, w, b, _, calls = setup()
    owner.prepare(quiescent=lambda: True)
    wr, br = weakref.ref(w) if hasattr(w, '__weakref__') else None, None
    # SimpleNamespace itself is not weak-referenceable; a tensor is sufficient
    # to detect a retained Buffers/closure/lease path after successful retirement.
    br = weakref.ref(b.h)
    calls.clear()
    gc.disable()
    try:
        owner.close()
        del b, w
        assert br() is None and wr is None
    finally:
        gc.enable()


def test_record_failure_retires_successfully_captured_catalogue():
    owner, _, _, cuda, _ = setup()
    primary = MemoryError('receipt allocation')
    with patch.object(owner, 'record', side_effect=primary):
        with pytest.raises(MemoryError) as raised:
            owner.prepare(quiescent=lambda: True)
    assert raised.value is primary and owner.closed
    assert all(graph.resets == 1 for graph in cuda.graphs)


def test_catalogue_rebind_and_competing_preparation_are_rejected_before_effects():
    owner, w, _, cuda, _ = setup()
    w.layers[0].attn_hc = w.layers[1].attn_hc
    with pytest.raises(RuntimeError, match='catalogue'):
        owner.prepare(quiescent=lambda: True)
    assert not cuda.events
    owner.close()
    owner, _, b, cuda, _ = setup()
    b.hc_plans = object()
    with pytest.raises(RuntimeError, match='another active plan'):
        owner.prepare(quiescent=lambda: True)
    assert not cuda.events
    owner.close()


def test_stream_restore_failure_preserves_body_primary_and_existing_cause():
    primary, previous, restore = OSError('readout'), ValueError('original cause'), OSError('stream restore')
    primary.__cause__ = previous
    def callback(cuda, calls):
        if len(calls) == 2:
            cuda.failures['set_stream'] = restore
            raise primary
    owner, _, _, cuda, _ = setup(callback=callback)
    with pytest.raises(OSError) as raised:
        owner.prepare(quiescent=lambda: True)
    assert raised.value is primary and owner.closed
    assert isinstance(primary.__cause__, BaseExceptionGroup)
    assert set(primary.__cause__.exceptions) == {previous, restore}
    assert cuda.graphs[0].resets == 1


def test_replay_and_close_refuse_cpu_overlap_without_device_effects():
    owner, w, b, cuda, _ = setup()
    owner.prepare(quiescent=lambda: True)
    owner._lock.acquire()
    before = list(cuda.events)
    try:
        with pytest.raises(RuntimeError, match='overlaps'):
            invoke(owner, w, b)
        with pytest.raises(RuntimeError, match='overlaps'):
            owner.close()
        assert cuda.events == before
    finally:
        owner._lock.release()
    assert invoke(owner, w, b)
    owner.close()


def test_counter_exhaustion_and_replay_error_stop_before_another_submission():
    owner, w, b, cuda, _ = setup()
    owner.prepare(quiescent=lambda: True)
    owner._entries[0].replays = sys.maxsize
    before = cuda.graphs[0].replays
    with pytest.raises(OverflowError):
        invoke(owner, w, b)
    assert cuda.graphs[0].replays == before
    owner.close()
    owner, w, b, cuda, _ = setup()
    owner.prepare(quiescent=lambda: True)
    primary = RuntimeError('replay failed')
    cuda.failures['replay'] = primary
    with pytest.raises(RuntimeError) as raised:
        invoke(owner, w, b)
    assert raised.value is primary and owner.failure is primary
    owner.close()


@pytest.mark.parametrize('site', ['_retire', '_transport'])
def test_cleanup_preparation_or_transport_exhaustion_preserves_readout_primary(site):
    primary, exhaustion = OSError('original readout'), MemoryError('cold cleanup exhausted')
    def callback(cuda, calls):
        if len(calls) == 2:
            raise primary
    owner, _, _, cuda, _ = setup(callback=callback)
    with patch.object(owner, site, side_effect=exhaustion):
        with pytest.raises(OSError) as raised:
            owner.prepare(quiescent=lambda: True)
    assert raised.value is primary and primary.__cause__ is exhaustion
    assert owner.retirement_failure is exhaustion
    if site == '_retire':
        assert owner._owners and not owner.closed and cuda.graphs[0].resets == 0
    else:
        assert owner.closed and cuda.graphs[0].resets == 1
    before = list(cuda.events)
    if not owner.closed:
        with pytest.raises(MemoryError) as raised:
            owner.close()
        assert raised.value is exhaustion
    assert cuda.events == before


def test_close_preparation_exhaustion_retains_owned_native_resources():
    owner, _, _, cuda, _ = setup()
    owner.prepare(quiescent=lambda: True)
    exhaustion = MemoryError('close traversal')
    with patch.object(owner, '_retire', side_effect=exhaustion):
        with pytest.raises(MemoryError) as raised:
            owner.close()
    assert raised.value is exhaustion and owner._owners and not owner.closed
    assert all(graph.resets == 0 for graph in cuda.graphs)


def test_actual_split_concat_reference_is_retained_and_part_rebinding_refused():
    owner, w, _, cuda, _ = setup()
    # Recreate enrollment with the real loader's split320+4 representation.
    owner.close()
    w.layers[0].attn_hc.down = Concat([Dense(10240, 320), Dense(10240, 4)])
    fake = ModuleType(WEIGHTS)
    fake.GGUFDense, fake.GGUFConcat = Dense, Concat
    _, _, b, _, calls = setup()
    with patch.dict(sys.modules, {WEIGHTS: fake}):
        fresh = type(owner)(w, b, lambda *a, **kw: calls.append(a), budget_bytes=100000,
                            torch_module=fake_torch(cuda))
    fresh.prepare(quiescent=lambda: True)
    assert fresh._entries[0].hc.down is w.layers[0].attn_hc.down
    w.layers[0].attn_hc.down.parts[1].weight = Tensor((4, 10240))
    with pytest.raises(RuntimeError, match='rebound'):
        fresh.validate_adoption()
    fresh.close()


def test_replay_publishes_exact_original_mutation_multiplicities():
    def callback(cuda, calls):
        b = calls[-1][0][1]
        b.normed._version += 1
        b.xs_normed._version += 2
    owner, w, b, _, _ = setup(callback=callback)
    published = []
    def increment(tensors):
        published.append(tensors)
        for tensor in tensors:
            tensor._version += 1
    owner.torch.autograd = NS(graph=NS(increment_version=increment))
    owner.prepare(quiescent=lambda: True)
    versions = b.normed._version, b.xs_normed._version, b.h._version
    assert invoke(owner, w, b)
    assert (b.normed._version, b.xs_normed._version, b.h._version) == (versions[0]+1, versions[1]+2, versions[2])
    assert published == [(b.normed, b.xs_normed, b.xs_normed)]
    owner.close()


def test_unqualified_shared_mutation_storage_and_prepared_slabs_refuse_enrollment():
    owner, w, b, cuda, _ = setup()
    owner.close()
    fake = ModuleType(WEIGHTS)
    fake.GGUFDense, fake.GGUFConcat = Dense, Concat
    b.normed.storage = b.h.storage
    with patch.dict(sys.modules, {WEIGHTS: fake}):
        with pytest.raises(ValueError, match='independent'):
            type(owner)(w, b, lambda *a, **kw: None, budget_bytes=100000,
                        torch_module=fake_torch(cuda))
        b.normed = Tensor()
        b.padded_linears = object()
        with pytest.raises(ValueError, match='region'):
            type(owner)(w, b, lambda *a, **kw: None, budget_bytes=100000,
                        torch_module=fake_torch(cuda))
    assert not cuda.events


def test_exhausted_retirement_reusing_primary_identity_still_forbids_retry():
    primary = MemoryError('exhaustion')
    def callback(cuda, calls):
        if len(calls) == 2:
            raise primary
    owner, _, _, cuda, _ = setup(callback=callback)
    with patch.object(owner, '_retire', side_effect=primary):
        with pytest.raises(MemoryError) as raised:
            owner.prepare(quiescent=lambda: True)
    assert raised.value is primary and owner.retirement_failure is primary and owner._owners
    before = list(cuda.events)
    with pytest.raises(MemoryError):
        owner.close()
    assert cuda.events == before


def test_only_clean_typed_budget_refusal_is_eligible_for_joint_decline():
    owner, _, _, _, _ = setup(budget=1)
    with pytest.raises(MemoryError):
        owner.prepare(quiescent=lambda: True)
    assert owner.budget_declined
    def callback(cuda, calls):
        if len(calls) == 2:
            cuda.failures['set_stream'] = OSError('restore failed')
    owner, _, _, _, _ = setup(budget=1, callback=callback)
    with pytest.raises(MemoryError):
        owner.prepare(quiescent=lambda: True)
    assert owner.closed and not owner.budget_declined


def test_live_row_selector_reuses_exact97_graphs_without_recapture():
    owner, w, b, cuda, calls = setup()
    owner.prepare(quiescent=lambda: True)
    count = len(calls)
    for rows in (4, 1, 3, 2, 4):
        assert invoke(owner, w, b, R=rows, inject=b.inj_a[:rows])
        if rows != 1:
            assert owner._live_rows.filled == rows
    assert len(cuda.graphs) == 100 and len(calls) == count
    assert owner.record()['replays'] == 5
    owner.close()
    assert owner._scratch is None and not owner._physical_inputs


def test_physical_frame_rebind_is_a_cold_boundary_error():
    owner, _, _, _, _ = setup()
    owner.prepare(quiescent=lambda: True)
    owner._scratch.normed = Tensor((32, 10240))
    with pytest.raises(RuntimeError, match='physical frame'):
        owner.validate_adoption()
    owner.close()


def test_partial_frame_allocation_failure_is_journalled_and_retired():
    owner, _, _, cuda, _ = setup()
    factory = owner.torch.zeros
    created = []
    primary = MemoryError('physical frame allocation failed')
    def allocate(*args, **kwargs):
        if len(created) == 3:
            raise primary
        result = factory(*args, **kwargs)
        created.append(weakref.ref(result))
        return result
    owner.torch.zeros = allocate
    with pytest.raises(MemoryError) as caught:
        owner.prepare(quiescent=lambda: True)
    assert caught.value is primary and owner.closed and owner._scratch is None
    assert not cuda.graphs and 'fence' in cuda.events
    # The authentic exception traceback can retain the last allocation local.
    # Owner journals must be released after the fence; traceback retention is
    # bounded failure evidence, not a still-active CUDA consumer.
    assert not owner._leases and not owner._physical_inputs and not owner._workspace_bindings


def test_callback_readonly_input_mutation_is_refused_before_commit():
    def callback(cuda, calls):
        calls[-1][0][1].pss._version += 1
    owner, _, _, _, _ = setup(callback=callback)
    with pytest.raises(RuntimeError, match='readonly input'):
        owner.prepare(quiescent=lambda: True)
    assert owner.closed


def test_changed_callback_output_mutations_poison_capture():
    def callback(cuda, calls):
        calls[-1][0][1].normed._version += 1 if len(calls) == 1 else 2
    owner, _, _, cuda, _ = setup(callback=callback)
    with pytest.raises(RuntimeError, match='mutation contract'):
        owner.prepare(quiescent=lambda: True)
    assert owner.closed and cuda.graphs[0].resets == 1


def test_inference_output_requires_the_original_mutation_context():
    def callback(cuda, calls):
        calls[-1][0][1].normed._version += 1
    owner, w, b, cuda, _ = setup(callback=callback)
    b.normed.is_inference = lambda: True
    owner.prepare(quiescent=lambda: True)
    assert owner._entries[0].direct.inference_outputs and owner._entries[0].template.inference_outputs
    before = cuda.graphs[0].replays
    assert not invoke(owner, w, b)
    assert cuda.graphs[0].replays == before
    owner.torch.is_inference_mode_enabled = lambda: True
    assert invoke(owner, w, b)
    owner.close()


def test_each_hc_gemm_masks_padding_and_calls_reference_not_native():
    owner, _, _, _, _ = setup()
    owner.prepare(quiescent=lambda: True)
    masked, calls = [], []
    owner._active = object()
    original_where = owner.torch.where
    def where(mask, value, other):
        masked.append((mask, value.shape, other))
        return original_where(mask, value, other)
    owner.torch.where = where
    class Matrix:
        k, n = 10240, 320
        def matmul_reference(self, source, *, out=None, f32=False):
            calls.append((source.shape, out, f32))
            return out
    matrix = Matrix()
    adapter = owner._scratch.padded_linears
    out = Tensor((32, 320), dtype='float')
    assert adapter.matmul(Tensor((32, 10240)), matrix, out) is out
    matrix.k, matrix.n = 320, 10240
    out = Tensor((32, 10240))
    assert adapter.matmul(Tensor((32, 320)), matrix, out) is out
    assert masked == [(owner._active, (32, 10240), 0), (owner._active, (32, 320), 0)]
    assert [row[0] for row in calls] == [(32, 10240), (32, 320)]
    assert [row[2] for row in calls] == [True, False]
    owner._active = None
    owner.close()


def test_cached_rows_remove_redundant_submissions_across_keys_and_transitions():
    owner, w, b, _, _ = setup(rows=24)
    owner.prepare(quiescent=lambda: True)
    scalar = owner._live_rows
    assert scalar.fills == [24] and owner._queued_rows == 24
    for hc in [getattr(layer, role) for layer in w.layers for role in ('attn_hc', 'mlp_hc')]:
        # Use each enrolled key's exact original injection owner.
        entry = owner._index[id(hc)]
        inject = entry.inject[:24]
        assert owner(hc, b.h, 24, w.cfg.eps, 4, 320, inject)
    assert owner(w.mixer, b.streams, 24, w.cfg.eps, 4, 320, None)
    assert scalar.fills == [24]  # No scalar submission for the entire97-key round.
    for rows in (24, 24, 1, 1, 23, 23, 24, 1):
        assert invoke(owner, w, b, R=rows, inject=b.inj_a[:rows])
        assert owner(w.layers[0].mlp_hc, b.h, rows, w.cfg.eps, 4, 320, b.inj_m[:rows])
    assert scalar.fills == [24, 23, 24]
    owner.close()
    assert owner._queued_rows is None


def test_row_fill_failure_invalidates_and_poison_prevents_any_replay():
    owner, w, b, cuda, _ = setup()
    owner.prepare(quiescent=lambda: True)
    primary = RuntimeError('row enqueue failure')
    with patch.object(owner._live_rows, 'fill_', side_effect=primary):
        before = cuda.graphs[0].replays
        with pytest.raises(RuntimeError) as caught:
            invoke(owner, w, b, R=2, inject=b.inj_a[:2])
        assert caught.value is primary
        assert owner._queued_rows is None and cuda.graphs[0].replays == before
        with pytest.raises(RuntimeError, match='unavailable'):
            invoke(owner, w, b)
    owner.close()


def test_same_rows_still_enforce_stream_and_failed_retirement_blocks_reuse():
    owner, w, b, cuda, _ = setup()
    owner.prepare(quiescent=lambda: True)
    cuda.stream = Stream()
    with pytest.raises(RuntimeError, match='serialized'):
        invoke(owner, w, b, R=4, inject=b.inj_a[:4])
    assert owner._queued_rows is None
    owner.close()
    owner, w, b, cuda, _ = setup()
    owner.prepare(quiescent=lambda: True)
    failure = RuntimeError('uncertain retirement')
    cuda.failures['fence'] = failure
    with pytest.raises(RuntimeError):
        owner.close()
    assert owner._queued_rows is None
    with pytest.raises(RuntimeError, match='poisoned'):
        _ = owner.enabled
    before = list(cuda.events)
    with pytest.raises(RuntimeError, match='unavailable'):
        invoke(owner, w, b)
    assert cuda.events == before


def test_poison_transport_never_evaluates_exception_truthiness():
    class Failure(RuntimeError):
        def __bool__(self):
            raise AssertionError('exception truthiness is not a failure contract')
    owner, w, b, cuda, _ = setup()
    owner.prepare(quiescent=lambda: True)
    primary = Failure('native replay failure')
    cuda.failures['replay'] = primary
    with pytest.raises(Failure):
        invoke(owner, w, b)
    with pytest.raises(RuntimeError, match='poisoned') as caught:
        _ = owner.enabled
    assert caught.value.__cause__ is primary
    with pytest.raises(RuntimeError, match='unavailable') as caught:
        invoke(owner, w, b)
    assert caught.value.__cause__ is primary
    cuda.failures.clear()
    owner.close()


def test_few_templates_preserve_roles_and_cover_all97_keys():
    def callback(cuda, calls):
        hc, frame, _, _, _, _, _, inject = calls[-1][0]
        if hc.inject:
            inject._version += 1
    owner, w, b, cuda, _ = setup(rows=24, split=True, callback=callback)
    owner.prepare(quiescent=lambda: True)
    record = owner.record()
    assert record['template_count'] == record['captured_templates'] == 3
    assert record['captured_r1_graphs'] == 97 and len(cuda.graphs) == 100
    assert record['bank_group_count'] == 2
    assert owner._templates[0].banks is owner._templates[1].banks
    assert record['captured_keys'] == record['covered_keys'] == 97
    assert owner._entries[0].template is owner._entries[2].template
    assert owner._entries[0].template is not owner._entries[1].template
    assert owner._entries[0].template.writes == (('inj_a', 1),)
    assert owner._entries[1].template.writes == (('inj_m', 1),)
    assert owner._entries[96].template.writes == ()
    versions = b.inj_a._version, b.inj_m._version
    assert invoke(owner, w, b)
    assert (b.inj_a._version, b.inj_m._version) == (versions[0]+1, versions[1])
    assert record['bank_tensor_bytes'] > 0
    owner.close()
    closed = owner.record()
    assert closed['captured_templates'] == 0 and closed['covered_keys'] == 0
    assert len(closed['preparation_progress']) == len(closed['program_memory']) == 100
    assert all(row['warm'] is not None and row['capture'] is not None
               for row in closed['program_memory'])
    assert all(row['banks'] is not None for row in closed['program_memory'][:3])
    assert all(row['begin'] == row['end'] == row['replay'] == row['fence'] == 1
               for row in closed['preparation_progress'])


def test_fresh_banks_follow_A_B_A_without_rebinding_or_mutating_sources():
    owner, w, b, _, _ = setup(split=True)
    owner.prepare(quiescent=lambda: True)
    a, other = owner._entries[0], owner._entries[2]
    template = a.template
    assert template is other.template and template.hc is not a.hc
    source_versions = {id(source): source._version for entry in (a, other) for source in entry.sources}
    before = owner.record()['bank_stages']
    for entry in (a, other, a):
        assert owner(entry.hc, b.h, 2, w.cfg.eps, 4, 320, b.inj_a[:2])
        assert all(target is not source and target.last_source is source
                   for target, source in zip(template.banks, entry.sources, strict=True))
    assert owner.record()['bank_stages'] == before+3
    assert all(source._version == source_versions[id(source)]
               for entry in (a, other) for source in entry.sources)
    owner.close()


@pytest.mark.parametrize('position', range(4))
def test_partial_weight_copy_failure_never_replays_or_reuses_template(position):
    owner, w, b, cuda, _ = setup(split=True)
    owner.prepare(quiescent=lambda: True)
    entry = owner._entries[2]
    primary = OSError('bitwise weight copy failed')
    before = entry.template.graph.replays
    with patch.object(entry.template.banks[position], 'copy_', side_effect=primary):
        with pytest.raises(OSError) as caught:
            owner(entry.hc, b.h, 2, w.cfg.eps, 4, 320, b.inj_a[:2])
        assert caught.value is primary and entry.template.graph.replays == before
        assert owner.failure is primary and owner._queued_rows is None
        events = list(cuda.events)
        with pytest.raises(RuntimeError, match='unavailable'):
            invoke(owner, w, b)
        assert cuda.events == events
    owner.close()


def test_template_private_bank_rebind_refused_and_decline_progress_survives_close():
    owner, _, _, _, _ = setup(split=True)
    owner.prepare(quiescent=lambda: True)
    owner._entries[0].template.hc.up.weight = Tensor((10240, 320))
    with pytest.raises(RuntimeError, match='private weight bank'):
        owner.validate_adoption()
    owner.close()
    owner, _, _, _, _ = setup(budget=1, split=True)
    with pytest.raises(MemoryError):
        owner.prepare(quiescent=lambda: True)
    record = owner.record()
    assert owner.closed and record['captured_templates'] == 0
    assert record['last_preparation_stage'] == 'post-capture'
    assert record['last_preparation_program'] == 0
    assert record['preparation_progress'][0]['begin'] == record['preparation_progress'][0]['fence'] == 1
    assert record['last_memory']['allocated'] == 100


def test_direct_R1_round_never_stages_banks_or_touches_private_scalar():
    owner, _, _, _, _ = setup(rows=24, split=True)
    owner.prepare(quiescent=lambda: True)
    before = owner.record()['bank_stages']
    scalar = list(owner._live_rows.fills)
    with patch.object(owner, '_stage_key', side_effect=AssertionError('R1 must use original weights')):
        for entry in owner._entries:
            inject = entry.inject[:1] if entry.inject is not None else None
            assert owner(entry.hc, entry.h, 1, owner.eps, 4, 320, inject)
            assert entry.direct.hc is entry.hc
    record = owner.record()
    assert record['bank_stages'] == before and owner._live_rows.fills == scalar
    assert record['r1_replays'] == record['replays'] == 97 and record['template_replays'] == 0
    owner.close()


def test_R1_multirow_R1_preserves_independent_paths_and_shared_input_bank_order():
    owner, w, b, _, _ = setup(rows=24, split=True)
    owner.prepare(quiescent=lambda: True)
    a, m = owner._entries[0], owner._entries[1]
    assert a.template.banks is m.template.banks
    assert a.template.graph is not m.template.graph
    stages = owner.record()['bank_stages']
    for entry, rows in ((a, 1), (a, 24), (m, 2), (a, 1), (m, 24), (a, 1)):
        assert owner(entry.hc, entry.h, rows, owner.eps, 4, 320, entry.inject[:rows])
    assert owner.record()['bank_stages'] == stages+3
    assert owner.record()['r1_replays'] == owner.record()['template_replays'] == 3
    assert owner._live_rows.fills == [24, 2, 24]
    owner.close()


def test_partial_direct_capture_failure_retires_every_journalled_program():
    primary = OSError('directR1 capture failed')
    def callback(cuda, calls):
        # Three templates each warm/capture, then direct0 warm/capture.
        if len(calls) == 8:
            raise primary
    owner, _, _, cuda, _ = setup(callback=callback)
    with pytest.raises(OSError) as caught:
        owner.prepare(quiescent=lambda: True)
    assert caught.value is primary and owner.closed
    assert len(cuda.graphs) == 4 and all(graph.resets == 1 for graph in cuda.graphs)
    assert len(owner._errors) == 102
    record = owner.record()
    assert record['captured_graphs'] == 0
    assert record['preparation_progress'][3]['path'] == 'r1'
    assert record['preparation_progress'][3]['begin'] == 1

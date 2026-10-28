"""Metadata and accepted-capture ownership only; actual numerical proof is TP8."""
from contextlib import nullcontext
from itertools import count
from math import prod
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70 import family_ops, routing_native, router_blas
from tensorfold.families.qwen4_exp.cuda import forward, hc_moe_region as module, moe_region_graphs
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFConcat, GGUFDense
from tensorfold.families.qwen4_exp.cuda.weight_types import HC


class Storage:
    ids = count(1)
    def __init__(self, size):
        self.pointer, self.size = next(self.ids) * 1048576, size
    def data_ptr(self):
        return self.pointer
    def nbytes(self):
        return self.size


class Tensor:
    is_cuda = True
    requires_grad = False
    device = torch.device('cuda', 0)
    def __init__(self, shape, dtype=torch.float16, *, storage=None, offset=0):
        self.shape, self.dtype = tuple(shape), dtype
        self.offset = offset
        self.storage = storage or Storage(prod(shape) * (2 if dtype == torch.float16 else 4))
    def stride(self):
        return tuple(prod(self.shape[i + 1:]) for i in range(len(self.shape)))
    def is_contiguous(self):
        return True
    def data_ptr(self):
        return self.storage.data_ptr() + self.offset * (2 if self.dtype == torch.float16 else 4)
    def untyped_storage(self):
        return self.storage
    def storage_offset(self):
        return self.offset
    def __getitem__(self, index):
        start, stop, step = index.indices(self.shape[0])
        assert step == 1
        return Tensor((stop - start, *self.shape[1:]), self.dtype, storage=self.storage,
                      offset=self.offset + start * prod(self.shape[1:]))
    def view(self, *shape):
        assert prod(shape) == prod(self.shape)
        return Tensor(shape, self.dtype, storage=self.storage, offset=self.offset)


def dense(n, k):
    value = object.__new__(GGUFDense)
    value.n, value.k, value.weight = n, k, Tensor((n, k))
    return value


@pytest.fixture
def fixture(monkeypatch):
    monkeypatch.setattr(GGUFDense, 'implementation', 'native32')
    monkeypatch.setattr(GGUFConcat, 'implementation', 'native32')
    monkeypatch.setattr(family_ops, '_NATIVE_FAMILY', True)
    monkeypatch.setattr(routing_native, '_NATIVE_LOGITS', False)
    cfg = NS(hidden=2560, streams=4, low=320, eps=1e-6, execution_dtype='float16', experts=512, top_k=10)
    down = object.__new__(GGUFConcat)
    down.parts, down.n, down.k = (dense(320, 10240), dense(4, 10240)), 324, 10240
    hc = HC(down, dense(10240, 320), Tensor((10240,), torch.float32), True)
    def packed():
        return NS(data=Tensor((1, 1)), ggml_type=12, n=1, k=1, column_offset=0)
    bank = NS(kernel='gguf', implementation='packed_exact', packed_rows=32, count=513,
              dims=2560, width=80, limit=0.0, swiglu=True, gate=packed(), up=packed(), down=packed(),
              shared=(dense(80, 2560), dense(80, 2560), dense(2560, 80)))
    layer = NS(index=0, mlp_hc=hc, moe=NS(experts=bank, router=Tensor((513, 2560), torch.float32)))
    weights = NS(cfg=cfg, layers=[layer], comm=NS(world=8, rank=0, device=Tensor.device), x3=None,
                 meta={'world': 8, 'sm70_profile': router_blas.PROFILE, 'execution_provider': 'sm70'})
    capacity = 34
    buffers = NS(prefill=False, rows=capacity, hc_plans=None, padded_linears=None, split=None)
    shapes = {'h': (capacity, 10240), 'pss': (capacity, 10, 4), 'normed': (capacity, 10240),
              'xs_normed': (capacity, 320), 'dn': (capacity, 324), 'dn_mix': (capacity, 320),
              'act': (capacity, 320), 'xs_act': (capacity, 10), 'inj_a': (capacity, 4),
              'inj_m': (capacity, 4), 'up': (capacity, 10240), 'mixed': (capacity, 2560),
              'xs_mixed': (capacity, 80), 'g_branch': (8 * capacity * 2560,)}
    for name, shape in shapes.items():
        dtype = torch.float32 if name in ('pss', 'xs_normed', 'xs_act', 'xs_mixed', 'g_branch') else torch.float16
        setattr(buffers, name, Tensor(shape, dtype))
    buffers.moe = NS(logits=Tensor((capacity, 513), torch.float32), pick=Tensor((capacity, 11), torch.int32),
                     wts=Tensor((capacity, 11), torch.float32), y=Tensor((capacity, 11, 2560)),
                     act=Tensor((capacity, 1)))
    buffers.router_blas = object()
    log = []
    graph = NS(capture_begin=lambda **kw: log.append('begin'), capture_end=lambda: log.append('end'),
               replay=lambda: log.append('replay'), reset=lambda: log.append('reset'))
    current = NS(cuda_stream=1, wait_stream=lambda side: log.append('join-side'))
    side = NS(wait_stream=lambda stream: log.append('join-input'), synchronize=lambda: log.append('warm-done'))
    cuda = NS(is_current_stream_capturing=lambda: False, current_stream=lambda device: current,
              Stream=lambda **kw: side, stream=lambda stream: nullcontext(), CUDAGraph=lambda: graph,
              synchronize=lambda device: log.append('fence'))
    math_policy = NS(allow_fp16_reduced_precision_reduction=True, allow_fp16_accumulation=False, allow_tf32=False)
    fake_torch = NS(Tensor=Tensor, float16=torch.float16, float32=torch.float32, cuda=cuda,
                    backends=NS(cuda=NS(matmul=math_policy)),
                    _C=NS(_overlaps=lambda a, b: a.storage is b.storage))
    monkeypatch.setattr(module, 'torch', fake_torch)
    monkeypatch.setattr(moe_region_graphs, 'torch', fake_torch)
    class CapturePlans:
        def __init__(self, values, rows):
            self.owners, self.rows = tuple(values), rows
    monkeypatch.setattr(router_blas, 'RolePlans', CapturePlans)
    def hc_call(hc, buf, rows, eps, streams, low, mode, injection, inj_out, h, *, branch):
        assert mode == 3 and branch.shape == (8, rows, 2560)
        assert injection.shape == (rows, 4) and inj_out is buffers.inj_m and h is buffers.h
        log.append('HC')
    def moe_call(layer, w, buf, rows, *, _local_only):
        assert _local_only is True and buf.mixed is buffers.mixed
        log.append('MoE')
    monkeypatch.setattr(forward, 'hc_block', hc_call)
    monkeypatch.setattr(forward, 'moe_block', moe_call)
    return NS(weights=weights, layer=layer, buffers=buffers, log=log, graph=graph, cuda=cuda)


def component(fixture):
    return module.HCReadoutMoERegion(fixture.weights, fixture.layer, fixture.buffers, 30)


def test_original_hc_then_original_pure_moe_and_explicit_pending_cut(fixture):
    region = component(fixture)
    region.prepare()
    assert [v for v in fixture.log if v in ('HC', 'MoE')] == ['HC', 'MoE', 'HC', 'MoE']
    assert region.capture_router is not fixture.buffers.router_blas
    region.replay(3, fixture.buffers.g_branch[:8 * 30 * 2560].view(8, 30, 2560), fixture.buffers.inj_a[:30])
    assert fixture.log[-1] == 'replay' and region.hits == 1
    region.close()
    assert region.closed and region.bindings is None and region.branch is None and region.injection is None
    assert region.weights is None and region.layer is None and region.buffers is None
    assert region.record()['closed'] is True
    assert fixture.log[-2:] == ['fence', 'reset']


@pytest.mark.parametrize('changed', ('mode', 'branch', 'injection'))
def test_wrong_pending_domain_refuses_before_graph_issue(fixture, changed):
    region = component(fixture)
    region.prepare()
    mode, branch, injection = 3, region.branch, region.injection
    if changed == 'mode':
        mode = 4
    elif changed == 'branch':
        branch = Tensor(branch.shape, torch.float32)
    else:
        injection = fixture.buffers.inj_a[1:31]
    with pytest.raises(ValueError):
        region.replay(mode, branch, injection)
    assert 'replay' not in fixture.log and region.failure is not None
    region.close()


@pytest.mark.parametrize('changed', ('h', 'g_branch', 'down', 'epsilon', 'communicator', 'math'))
def test_captured_hc_binding_change_is_not_silent_stale_replay(fixture, changed):
    region = component(fixture)
    region.prepare()
    if changed in ('h', 'g_branch'):
        old = getattr(fixture.buffers, changed)
        setattr(fixture.buffers, changed, Tensor(old.shape, old.dtype))
    elif changed == 'down':
        fixture.layer.mlp_hc.down.parts[0].weight = Tensor((320, 10240))
    elif changed == 'epsilon':
        fixture.weights.cfg.eps = 1e-5
    elif changed == 'communicator':
        fixture.weights.comm = NS(world=8, rank=0, device=Tensor.device)
    else:
        module.torch.backends.cuda.matmul.allow_tf32 = True
    with pytest.raises(RuntimeError):
        region.replay(3, region.branch, region.injection)
    assert 'replay' not in fixture.log
    region.close()


@pytest.mark.parametrize('changed', ('world', 'prefill', 'hc_plans', 'padded', 'inject', 'alias', 'dtype'))
def test_unsupported_or_aliasing_cut_rejected_before_any_cuda_submission(fixture, changed):
    if changed == 'world':
        fixture.weights.comm.world = 1
    elif changed == 'prefill':
        fixture.buffers.prefill = True
    elif changed == 'hc_plans':
        fixture.buffers.hc_plans = object()
    elif changed == 'padded':
        fixture.buffers.padded_linears = object()
    elif changed == 'inject':
        fixture.layer.mlp_hc.inject = False
    elif changed == 'alias':
        fixture.buffers.normed = fixture.buffers.h
    else:
        fixture.layer.mlp_hc.up.weight = Tensor((10240, 320), torch.float32)
    with pytest.raises(ValueError):
        component(fixture)
    assert fixture.log == []


def test_accepted_capture_primary_and_end_failure_retain_original(fixture, monkeypatch):
    region = component(fixture)
    primary, secondary = ValueError('captured-HC'), OSError('capture-end')
    calls = 0
    original = forward.hc_block
    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        original(*args, **kwargs)
        if calls == 2:
            raise primary
    monkeypatch.setattr(forward, 'hc_block', fail)
    fixture.graph.capture_end = lambda: (_ for _ in ()).throw(secondary)
    with pytest.raises(ValueError) as caught:
        region.prepare()
    assert caught.value is primary and caught.value.__cause__ is secondary and region.failure is primary
    with pytest.raises(RuntimeError):
        region.replay(3, region.branch, region.injection)
    assert region.failure is primary and 'replay' not in fixture.log
    region.close()


def test_failed_fence_retains_hc_storage_and_blocks_replay(fixture):
    region = component(fixture)
    region.prepare()
    fixture.cuda.synchronize = lambda device: (_ for _ in ()).throw(OSError('fence'))
    with pytest.raises(OSError, match='fence'):
        region.close()
    assert region.bindings is not None and region.branch is not None and region.graph is fixture.graph
    with pytest.raises(RuntimeError):
        region.replay(3, region.branch, region.injection)
    assert 'replay' not in fixture.log


def test_changed_binding_before_prepare_has_no_gpu_submission(fixture):
    region = component(fixture)
    fixture.buffers.h = Tensor(fixture.buffers.h.shape)
    with pytest.raises(RuntimeError, match='Tensor'):
        region.prepare()
    assert fixture.log == [] and region.graph is None and region.failure is not None
    region.close()


def test_retirement_uses_original_device_after_refused_external_rebinding(fixture):
    region = component(fixture)
    region.prepare()
    original = region.capture_device
    fixture.buffers.mixed.device = torch.device('cuda', 1)
    with pytest.raises(RuntimeError):
        region.replay(3, region.branch, region.injection)
    fenced = []
    fixture.cuda.synchronize = lambda device: fenced.append(device)
    region.close()
    assert fenced == [original] and region.closed and region.weights is None


def test_owned_mtp_layer_is_bound_even_outside_target_layer_list(fixture):
    fixture.weights.layers = []
    fixture.weights.mtp = NS(layer=fixture.layer)
    region = component(fixture)
    region.prepare()
    fixture.layer.moe.router = Tensor((513, 2560), torch.float32)
    with pytest.raises(RuntimeError):
        region.replay(3, region.branch, region.injection)
    assert 'replay' not in fixture.log
    region.close()


def test_unowned_layer_is_rejected_before_capture(fixture):
    fixture.weights.layers = []
    fixture.weights.mtp = None
    with pytest.raises(ValueError, match='owned'):
        component(fixture)
    assert fixture.log == []


def test_parent_successful_child_retirement_preserves_named_failure_roots(monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parent / 'cuda' / 'test_sm70_hc_moe_region_tp8.py'
    spec = importlib.util.spec_from_file_location('hc_parent_outcome_control', path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    primary, prior = ValueError('worker failed'), OSError('historical source failure')
    primary.__cause__ = prior
    retired = []
    class Process:
        def __init__(self):
            self.alive, self.exitcode = True, None
        def is_alive(self):
            return self.alive
        def kill(self):
            self.alive = False
        def join(self, timeout):
            self.exitcode = 17
            retired.append(self)
    class Context:
        processes = [Process() for _ in range(8)]
        def join(self, timeout):
            raise primary
    monkeypatch.setattr(oracle.torch.cuda, 'device_count', lambda: 8)
    monkeypatch.setattr(oracle.torch.cuda, 'get_device_capability', lambda index: (7, 0))
    monkeypatch.setattr(oracle.torch.multiprocessing, 'spawn', lambda *args, **kwargs: Context())
    with pytest.raises(ValueError) as caught:
        oracle.test_actual_tp8_mode3_hc_and_pure_moe_graph_preserve_all_outputs_and_tails()
    assert caught.value is primary and primary.__cause__ is prior and len(retired) == 8
    assert not getattr(primary, '__notes__', ())

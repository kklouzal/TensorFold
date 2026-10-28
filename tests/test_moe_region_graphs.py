"""Private capture ownership/failure controls; numerical oracle is actual CUDA."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.cuda import moe_region_graphs as module


class Tensor:
    dtype = 'half'
    device = 'cuda:0'
    is_cuda = True
    shape = (32, 2560)
    def is_contiguous(self):
        return True


@pytest.fixture
def setup(monkeypatch):
    log = []
    graph = SimpleNamespace(capture_begin=lambda **kw: log.append('begin'),
                            capture_end=lambda: log.append('end'),
                            replay=lambda: log.append('replay'), reset=lambda: log.append('reset'))
    original = SimpleNamespace(cuda_stream=1, wait_stream=lambda side: log.append('join-side'))
    side = SimpleNamespace(wait_stream=lambda current: log.append('join-input'),
                           synchronize=lambda: log.append('warm-done'))
    cuda = SimpleNamespace(is_current_stream_capturing=lambda: False,
                           current_stream=lambda device: original, Stream=lambda **kw: side,
                           stream=lambda stream: nullcontext(), CUDAGraph=lambda: graph,
                           synchronize=lambda device: log.append('fence'))
    monkeypatch.setattr(module,'torch',SimpleNamespace(float16='half',cuda=cuda))
    expert = SimpleNamespace(kernel='gguf',implementation='packed_exact',packed_rows=32,
                             count=513,dims=2560,width=80)
    weights = SimpleNamespace(cfg=SimpleNamespace(experts=512,top_k=10))
    layer = SimpleNamespace(index=0,moe=SimpleNamespace(experts=expert))
    buffers = SimpleNamespace(prefill=False,mixed=Tensor(),moe=SimpleNamespace(
        logits=Tensor(),pick=Tensor(),wts=Tensor(),y=Tensor()))
    region = module.MoERegion(weights,layer,buffers,32)
    region._body = lambda: log.append('body')
    return region,graph,cuda,original,log


def test_prepare_replay_and_positive_retirement(setup):
    region,_,_,_,log=setup
    region.prepare()
    assert log==['join-input','body','warm-done','begin','body','end','join-side']
    region.replay()
    assert region.hits==1
    region.close()
    assert log[-3:]==['replay','fence','reset']
    assert region.closed and region.graph is None
    region.close()


def test_body_primary_and_end_secondary_are_both_retained(setup):
    region,graph,_,_,log=setup
    primary,secondary=ValueError('body'),OSError('end')
    calls=0
    def body():
        nonlocal calls
        calls+=1
        if calls==2:
            raise primary
    def end():
        log.append('end')
        raise secondary
    region._body=body
    graph.capture_end=end
    with pytest.raises(ValueError) as caught:
        region.prepare()
    assert caught.value is primary and caught.value.__cause__ is secondary
    assert log.count('end')==1 and region.graph is graph
    assert region.failure is caught.value and not region.prepared


def test_end_only_failure_and_begin_rejection(setup):
    region,graph,_,_,log=setup
    error=OSError('end')
    graph.capture_end=lambda: (_ for _ in ()).throw(error)
    with pytest.raises(OSError) as caught:
        region.prepare()
    assert caught.value is error and region.failure is error
    # A second component with a rejected begin must never attempt capture_end.
    second=module.MoERegion(region.weights,region.layer,region.buffers,32)
    second._body=lambda: None
    graph.capture_begin=lambda **kw: (_ for _ in ()).throw(ValueError('begin'))
    graph.capture_end=lambda: log.append('unexpected-end')
    with pytest.raises(ValueError,match='begin'):
        second.prepare()
    assert 'unexpected-end' not in log


def test_stream_or_external_binding_change_refuses_before_replay(setup):
    region,_,_,original,log=setup
    region.prepare()
    original.cuda_stream=2
    with pytest.raises(RuntimeError,match='serialized stream'):
        region.replay()
    assert 'replay' not in log and region.failure is not None


def test_external_tensor_rebinding_is_not_silent_stale_input(setup):
    region,_,_,_,log=setup
    region.prepare()
    region.buffers.mixed=Tensor()
    with pytest.raises(RuntimeError,match='buffer owners'):
        region.replay()
    assert 'replay' not in log


def test_uncertain_fence_keeps_graph_owner(setup):
    region,graph,cuda,_,_=setup
    region.prepare()
    cuda.synchronize=lambda device: (_ for _ in ()).throw(OSError('fence'))
    with pytest.raises(OSError,match='fence'):
        region.close()
    assert region.graph is graph and not region.closed and region.tensors
    with pytest.raises(RuntimeError,match='not an admitted'):
        region.replay()


def test_exhausted_counter_refuses_before_gpu_submission(setup):
    region,_,_,_,log=setup
    region.prepare()
    region.hits=(1<<63)-1
    with pytest.raises(OverflowError):
        region.replay()
    assert 'replay' not in log


def test_stream_restore_failure_preserves_owned_operation_primary(setup):
    region,_,cuda,_,log=setup
    primary,secondary=ValueError('warm-body'),OSError('stream-restore')
    class RestoreFailure:
        def __enter__(self):
            return None
        def __exit__(self,*args):
            raise secondary
    cuda.stream=lambda side: RestoreFailure()
    region._body=lambda: (_ for _ in ()).throw(primary)
    with pytest.raises(ValueError) as caught:
        region.prepare()
    assert caught.value is primary and caught.value.__cause__ is secondary
    assert region.failure is primary and region.graph is None and 'begin' not in log


def test_capture_router_is_dedicated_and_retired_after_graph(setup,monkeypatch):
    from tensorfold.cuda.sm70 import router_blas
    region,_,_,_,_=setup
    main=object()
    weight=object()
    region.buffers.router_blas=main
    region.layer.moe.router=weight
    class PreparedCapture:
        def __init__(self,weights,rows):
            self.owners=tuple(weights)
            self.rows=rows
    monkeypatch.setattr(router_blas,'RolePlans',PreparedCapture)
    seen=[]
    region._body=lambda: seen.append(region.local_buffers.router_blas)
    region.prepare()
    assert region.capture_router is not main
    assert region.capture_router.owners==(weight,) and region.capture_router.rows==32
    assert seen==[region.capture_router,region.capture_router]
    assert region.local_buffers.mixed is region.buffers.mixed
    assert region.local_buffers.moe is region.buffers.moe
    region.close()
    assert region.capture_router is None and region.local_buffers is None


def test_capture_router_constructor_failure_retains_partial_owner(setup,monkeypatch):
    from tensorfold.cuda.sm70 import router_blas
    region,_,_,_,log=setup
    region.buffers.router_blas=object()
    region.layer.moe.router=object()
    primary=MemoryError('capture-plan')
    class PartialCapture:
        def __init__(self,weights,rows):
            self.owners=tuple(weights)
            raise primary
    monkeypatch.setattr(router_blas,'RolePlans',PartialCapture)
    with pytest.raises(MemoryError) as caught:
        region.prepare()
    assert caught.value is primary and region.failure is primary
    assert region.capture_router is not None and 'begin' not in log
    region.close()
    assert region.capture_router is None


def test_body_only_failure_does_not_transport_a_none_secondary(setup):
    region,_,_,_,log=setup
    primary=ValueError('capture-body-only')
    calls=0
    def body():
        nonlocal calls
        calls+=1
        if calls==2:
            raise primary
    region._body=body
    with pytest.raises(ValueError) as caught:
        region.prepare()
    assert caught.value is primary and caught.value.__cause__ is None
    assert region.failure is primary and log.count('end')==1


def test_body_end_and_stream_restore_failures_keep_all_named_outcomes(setup):
    region,graph,cuda,_,_=setup
    primary,end_error,restore_error=ValueError('capture'),OSError('end'),RuntimeError('restore')
    enters=0
    class Context:
        def __enter__(self):
            nonlocal enters
            enters+=1
        def __exit__(self,*args):
            if enters==2:
                raise restore_error
    cuda.stream=lambda side: Context()
    calls=0
    def body():
        nonlocal calls
        calls+=1
        if calls==2:
            raise primary
    region._body=body
    graph.capture_end=lambda: (_ for _ in ()).throw(end_error)
    with pytest.raises(ValueError) as caught:
        region.prepare()
    def leaves(error):
        return sum((leaves(item) for item in error.exceptions),[]) if isinstance(error,BaseExceptionGroup) else [error]
    assert caught.value is primary
    assert end_error in leaves(primary.__cause__) and restore_error in leaves(primary.__cause__)


def test_failed_graph_reset_cannot_reopen_replay(setup):
    region,graph,_,_,log=setup
    region.prepare()
    error=OSError('graph-reset')
    graph.reset=lambda: (_ for _ in ()).throw(error)
    with pytest.raises(OSError) as caught:
        region.close()
    assert caught.value is error and region.failure is error
    assert region.retiring and not region.closed and region.graph is graph
    with pytest.raises(RuntimeError,match='not an admitted'):
        region.replay()
    assert 'replay' not in log

"""Capture lifecycle controls; packet numerical oracle requires actual CUDA."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.cuda import candidate_region_graphs as module


class Storage:
    def data_ptr(self):
        return 1

    def nbytes(self):
        return 32 * 65 * 4


class Tensor:
    shape = (32, 65)
    ndim = 2
    dtype = 'float32'
    device = 'cuda:0'
    is_cuda = True
    requires_grad = False

    def __init__(self):
        self.storage = Storage()

    def untyped_storage(self):
        return self.storage

    def storage_offset(self):
        return 0

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return 1

    def stride(self):
        return (65, 1)

    def numel(self):
        return 32 * 65


@pytest.fixture
def setup(monkeypatch):
    log = []
    graph = SimpleNamespace(capture_begin=lambda **kw: log.append('begin'),
                            capture_end=lambda: log.append('end'),
                            replay=lambda: log.append('replay'), reset=lambda: log.append('reset'))
    current = SimpleNamespace(cuda_stream=1, wait_stream=lambda side: log.append('join-side'))
    side = SimpleNamespace(wait_stream=lambda current: log.append('join-input'),
                           synchronize=lambda: log.append('warm-done'))
    cuda = SimpleNamespace(is_current_stream_capturing=lambda: False,
                           current_stream=lambda device: current, Stream=lambda **kw: side,
                           stream=lambda stream: nullcontext(), CUDAGraph=lambda: graph,
                           synchronize=lambda device: log.append('fence'))
    monkeypatch.setattr(module, 'torch', SimpleNamespace(float32='float32', cuda=cuda))
    logits = Tensor()
    weights = SimpleNamespace(meta={'execution_provider': 'sm70'})
    buffers = SimpleNamespace(prefill=False, cand=Tensor(), logits=logits)
    owner = module.CandidateRegion(weights, buffers, logits, 32)
    owner._body = lambda: log.append('body')
    return owner, graph, cuda, current, log


def test_prepare_replay_and_retirement(setup):
    owner, _, _, _, log = setup
    owner.prepare()
    assert log == ['join-input', 'body', 'warm-done', 'begin', 'body', 'end', 'join-side']
    owner.replay()
    owner.close()
    assert log[-3:] == ['replay', 'fence', 'reset']
    assert owner.hits == 1 and owner.closed and owner.graph is None
    owner.close()


def test_capture_body_primary_and_end_secondary(setup):
    owner, graph, _, _, log = setup
    primary, secondary = ValueError('body'), OSError('end')
    calls = 0

    def body():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise primary

    def end():
        log.append('end')
        raise secondary

    owner._body, graph.capture_end = body, end
    with pytest.raises(ValueError) as caught:
        owner.prepare()
    assert caught.value is primary and caught.value.__cause__ is secondary
    assert log.count('end') == 1 and owner.graph is graph and owner.failure is primary


def test_rejected_begin_does_not_end(setup):
    owner, graph, _, _, log = setup
    graph.capture_begin = lambda **kw: (_ for _ in ()).throw(OSError('begin'))
    with pytest.raises(OSError, match='begin'):
        owner.prepare()
    assert 'end' not in log and owner.graph is graph


@pytest.mark.parametrize('changed', ['stream', 'packet', 'input', 'nested', 'layout'])
def test_changed_replay_contract_refuses_before_launch(setup, changed):
    owner, _, cuda, current, log = setup
    owner.prepare()
    if changed == 'stream':
        current.cuda_stream = 2
    elif changed == 'packet':
        owner.buffers.cand = Tensor()
    elif changed == 'input':
        owner.buffers.logits = Tensor()
    elif changed == 'nested':
        cuda.is_current_stream_capturing = lambda: True
    else:
        owner.logits.shape = (32, 64)
    with pytest.raises(RuntimeError):
        owner.replay()
    assert 'replay' not in log and owner.failure is not None


def test_failed_fence_retains_graph_and_closes_admission(setup):
    owner, graph, cuda, _, _ = setup
    owner.prepare()
    cuda.synchronize = lambda device: (_ for _ in ()).throw(OSError('fence'))
    with pytest.raises(OSError, match='fence'):
        owner.close()
    assert owner.graph is graph and not owner.closed and owner.retiring
    with pytest.raises(RuntimeError):
        owner.replay()


def test_failed_reset_retains_external_storage_until_retry(setup):
    owner, graph, _, _, _ = setup
    owner.prepare()
    original = owner.input_owner
    graph.reset = lambda: (_ for _ in ()).throw(OSError('reset'))
    with pytest.raises(OSError, match='reset'):
        owner.close()
    assert owner.input_owner is original and owner.graph is graph and not owner.closed
    graph.reset = lambda: None
    owner.close()
    assert owner.closed and owner.input_owner is None and owner.logits is None


def test_rebound_tensor_keeps_prior_storage_until_positive_retirement(setup):
    import weakref

    owner, graph, _, _, _ = setup
    original = weakref.ref(owner.logits.storage)
    owner.prepare()
    # Cold/quiescent rebind containment: no replay has been accepted. This is
    # not a promise that concurrent Storage.resize_ can be made safe.
    owner.logits.storage = Storage()
    assert original() is not None
    graph.reset = lambda: (_ for _ in ()).throw(OSError('reset'))
    with pytest.raises(OSError, match='reset'):
        owner.close()
    assert original() is not None
    graph.reset = lambda: None
    owner.close()
    assert original() is None


def test_side_stream_restore_preserves_warm_primary(setup):
    owner, _, cuda, _, log = setup
    primary, secondary = ValueError('warm'), OSError('restore')

    class RestoreFailure:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            raise secondary

    cuda.stream = lambda stream: RestoreFailure()
    owner._body = lambda: (_ for _ in ()).throw(primary)
    with pytest.raises(ValueError) as caught:
        owner.prepare()
    assert caught.value is primary and caught.value.__cause__ is secondary
    assert owner.failure is primary and owner.graph is None and 'begin' not in log

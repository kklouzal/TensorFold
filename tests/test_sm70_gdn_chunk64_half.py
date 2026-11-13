"""CPU boundary/oracle controls only: no CUDA compilation or native execution."""
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.sm70 import gdn_chunk64_half as half
from tools.v100_gdn_chunk64_trial import half_epilogue_reference, half_frontend_reference


def operands(rows=1, capacity=None):
    capacity = rows if capacity is None else capacity
    workspace = half.Workspace(capacity,
        torch.empty(1,capacity,2,128,dtype=torch.float16),
        torch.empty(1,capacity,2,128,dtype=torch.float16),
        torch.empty(1,capacity,6,128,dtype=torch.float16),
        torch.empty(1,capacity,6),torch.empty(1,capacity,6),
        torch.empty(1,capacity,6,128,dtype=torch.float16))
    tensors = [torch.zeros(rows,2060,dtype=torch.float16),torch.zeros(3,1280,dtype=torch.float16),
               torch.zeros(1280,4,dtype=torch.float16),torch.zeros(6),torch.zeros(6),
               torch.ones(128,dtype=torch.float16),torch.empty(rows,768,dtype=torch.float16),torch.empty(rows,24)]
    return workspace,tensors


@pytest.mark.parametrize('rows',[1,63,64,65,65535])
def test_explicit_numerical_mode_and_paid_separate_raw_output(rows):
    facts = half.layout(rows)
    assert facts['frontend_bytes'] == 2608*rows
    assert facts['raw_output_bytes'] == 1536*rows
    assert facts['device_bytes'] == 4144*rows
    assert facts['numerical_mode'] == 'g3-half-unscaled-qk-raw-gate-rounded-beta-v1'
    assert facts['qualification'] == 'private-unqualified'


@pytest.mark.parametrize('rows',[0,-1,True,1.0,65536])
def test_capacity_is_bounded_before_dependency_or_allocation(rows):
    with pytest.raises(ValueError):
        half.allocate_workspace(rows,'cuda:0',device_budget_bytes=10**9)


def test_insufficient_credit_never_allocates(monkeypatch):
    monkeypatch.setattr(torch,'empty',lambda *a,**k: pytest.fail('unfunded allocation'))
    for credit in (4143,True,None):
        with pytest.raises(MemoryError):
            half.allocate_workspace(1,'cuda:0',device_budget_bytes=credit)
    with pytest.raises(ValueError,match='CUDA'):
        half.allocate_workspace(1,'cpu',device_budget_bytes=4144)


def test_full_capacity_payload_and_active_rows_are_validated_on_real_cpu_storage():
    workspace,tensors = operands(63,65)
    rows,owners = half._validate_operands(workspace,*tensors,torch.device('cpu'))
    assert rows == 63 and len(owners) == 14
    assert sum(t.numel()*t.element_size() for t in vars(workspace).values() if isinstance(t,torch.Tensor)) == 65*4144
    tensors[0] = torch.zeros(66,2060,dtype=torch.float16)
    with pytest.raises(ValueError,match='capacity'):
        half._validate_operands(workspace,*tensors,torch.device('cpu'))


@pytest.mark.parametrize('fault',['dtype','shape','grad','lazy-negative','alias-input','alias-raw'])
def test_exact_operand_and_disjoint_write_contract(fault):
    workspace,tensors = operands()
    if fault == 'dtype':
        tensors[2] = tensors[2].float()
    elif fault == 'shape':
        tensors[1] = tensors[1][:2]
    elif fault == 'grad':
        tensors[3].requires_grad_()
    elif fault == 'lazy-negative':
        tensors[3] = torch._neg_view(tensors[3])
    elif fault == 'alias-input':
        tensors[6] = tensors[0][0,:768].view(1,768)
    else:
        tensors[6] = workspace.raw_output.view(1,768)
    with pytest.raises(ValueError):
        half._validate_operands(workspace,*tensors,torch.device('cpu'))


def test_decay_source_and_epsilon_rejected_before_cuda_context():
    prepared = half.Prepared(None,torch.device('cpu'))
    workspace,tensors = operands()
    p,conv,w,a,b,n,out,sums = tensors
    for sources in ({},{'a_log':a,'a_neg':a}):
        with pytest.raises(ValueError,match='exactly one'):
            prepared.bind(workspace,p,conv,w,b,n,1e-6,out,sums,**sources)
    for eps in (0,-1,True,float('nan'),float('inf'),1e-50,1e39):
        with pytest.raises(ValueError,match='epsilon'):
            prepared.bind(workspace,p,conv,w,b,n,eps,out,sums,a_log=a)
    with pytest.raises(RuntimeError,match='CUDA'):
        prepared.bind(workspace,p,conv,w,b,n,1e-6,out,sums,a_log=a)


def bound_cpu_control(monkeypatch,module):
    # Simulate only the launch interface; tensors/alias/layout checks are real.
    # This deliberately makes no GPU, ordering or native arithmetic assertion.
    monkeypatch.setattr(half,'_context',lambda device: None)
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device: SimpleNamespace(cuda_stream=7))
    workspace,tensors = operands(1,2)
    p,conv,w,a,b,n,out,sums = tensors
    prepared = half.Prepared(module,torch.device('cpu'))
    return prepared.bind(workspace,p,conv,w,b,n,1e-6,out,sums,a_neg=a),tensors


def test_binding_retains_views_and_launch_failure_poisons_both_boundaries(monkeypatch):
    primary = RuntimeError('authentic launch failure')
    def fail(*args):
        raise primary
    bound,_ = bound_cpu_control(monkeypatch,SimpleNamespace(chunk64_front=fail))
    core = bound.core_operands
    assert [tuple(t.shape) for t in core] == [(1,1,2,128),(1,1,2,128),(1,1,6,128),(1,1,6),(1,1,6),(1,1,6,128)]
    assert all(t.is_contiguous() for t in core)
    with pytest.raises(RuntimeError) as caught:
        bound.front()
    assert caught.value is primary and bound.error is primary
    with pytest.raises(RuntimeError,match='without retry') as caught:
        bound.back()
    assert caught.value.__cause__ is primary


def test_bound_storage_and_stream_changes_fail_before_dispatch(monkeypatch):
    module = SimpleNamespace(chunk64_front=lambda *args: pytest.fail('stale dispatch'))
    bound,tensors = bound_cpu_control(monkeypatch,module)
    tensors[0].resize_(2,1030)
    with pytest.raises(RuntimeError,match='storage/layout'):
        bound.front()
    bound,_ = bound_cpu_control(monkeypatch,module)
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device: SimpleNamespace(cuda_stream=8))
    with pytest.raises(RuntimeError,match='original thread'):
        bound.front()


@pytest.mark.parametrize('rows',[1,63,64,65])
def test_frontend_oracle_full_prompt_history_unscaled_half_and_direct_gate(rows):
    _,tensors = operands(rows)
    p,history,w,parameter,bias,*_ = tensors
    w[:,3] = 1
    p[:,0] = 2
    p[:,256] = 2
    p[:,512:1280] = 1
    p[:,2048:2054] = 0.3
    p[:,2054:2060] = 21
    parameter.fill_(-8)
    q,k,v,raw,beta = half_frontend_reference(p,history,w,parameter,bias,negative_decay=True)
    assert q.dtype == k.dtype == v.dtype == torch.float16
    assert raw.dtype == beta.dtype == torch.float32
    assert torch.equal(q[...,0],torch.tensor([1.,0.],dtype=torch.float16).expand(1,rows,2))
    assert torch.equal(q,k)
    assert torch.equal(raw,torch.full((1,rows,6),-168.))
    assert torch.equal(torch.exp(raw),torch.zeros_like(raw))  # log(savedexp) would lose the finite producer.
    expected_beta = (1/(1+torch.exp(-p[:,2048:2054].float()))).half().float()[None]
    assert torch.equal(beta,expected_beta)
    assert not torch.equal(beta,(1/(1+torch.exp(-p[:,2048:2054].float())))[None])
    assert torch.equal(v,((torch.tensor(1.)/(1+torch.exp(torch.tensor(-1.)))).half()).expand_as(v))


def test_frontend_oracle_declared_a_log_and_a_neg_agree_for_exact_rate_and_history():
    _,tensors = operands(2)
    p,history,w,parameter,bias,*_ = tensors
    history[2,0] = 2
    w[:,2] = 1
    hf = half_frontend_reference(p,history,w,parameter,bias,negative_decay=False)
    gguf = half_frontend_reference(p,history,w,-torch.ones_like(parameter),bias,negative_decay=True)
    assert all(torch.equal(a,b) for a,b in zip(hf,gguf))
    assert hf[0][0,0,0,0] == 1 and hf[0][0,1,0,0] == 0


def test_epilogue_oracle_half_casts_sigmoid_output_and_group32_sums():
    _,tensors = operands(2)
    p,_,_,_,_,norm,*_ = tensors
    raw = torch.ones(1,2,6,128,dtype=torch.float16)
    output,sums = half_epilogue_reference(raw,p,norm,1e-6)
    assert output.dtype == torch.float16 and sums.dtype == torch.float32
    assert torch.equal(output,torch.full_like(output,0.5))
    assert torch.equal(sums,torch.full_like(sums,16))
    raw.zero_()
    output,sums = half_epilogue_reference(raw,p,norm,1e-6)
    assert torch.count_nonzero(output) == torch.count_nonzero(sums) == 0


@pytest.mark.parametrize('view,fault',[(0,'resize'),(5,'set'),(6,'resize'),(11,'set')])
def test_active_and_native_view_metadata_changes_refuse_dispatch(monkeypatch,view,fault):
    module = SimpleNamespace(chunk64_front=lambda *args: pytest.fail('changed view reached native dispatch'),
                             chunk64_back=lambda *args: pytest.fail('changed view reached native dispatch'))
    bound,_ = bound_cpu_control(monkeypatch,module)
    tensor = (*bound.core_operands,*bound.native_views)[view]
    original_full = tuple(map(half._signature,bound.tensors))
    if fault == 'resize':
        tensor.resize_(tensor.numel())
    else:
        tensor.set_(torch.empty_like(tensor))
    assert tuple(map(half._signature,bound.tensors)) == original_full
    for submit in (bound.front,bound.back):
        with pytest.raises(RuntimeError,match='storage/layout'):
            submit()

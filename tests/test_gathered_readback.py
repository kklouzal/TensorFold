"""CPU host/ownership controls; no CUDA DMA, collective or performance claim."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen4_exp.cuda import decode
from tensorfold.families.qwen4_exp.cuda.gathered_readback import POLICY, GatheredReadback, layout
from tensorfold.families.qwen4_exp.cuda.owner_preparation import OwnerPreparation, parse_preparation
from tensorfold.families.qwen4_exp.cuda.state import CAND


class HostPacket(torch.Tensor):
    """CPU storage with only the wrapper's CUDA metadata gate substituted."""
    @property
    def is_cuda(self):
        return True


class Buffer:
    pass


def fixture(rows=3, maximum=7, role='decode'):
    w=NS(meta={'world':8,'execution_provider':'sm70'},comm=NS(world=8))
    b=Buffer()
    b.rows,b.owner_role,b.world,b.execution_dtype=maximum,role,8,torch.float16
    b.cand_all=torch.empty(8*maximum*65,dtype=torch.float32).as_subclass(HostPacket)
    g=b.cand_all[:8*rows*65].view(8,rows,65)
    scores=torch.arange(CAND,dtype=torch.float32).neg().view(1,1,CAND).expand(8,rows,CAND)
    g[:,:,:CAND].copy_(scores)
    ids=(torch.arange(8*CAND,dtype=torch.int32)+torch.iinfo(torch.int32).max-8*CAND).view(8,1,CAND)
    g[:,:,CAND:2*CAND].view(torch.int32).copy_(ids.expand(8,rows,CAND))
    g[:,:,2*CAND].fill_(1.0)
    owner=GatheredReadback(8,maximum,role,'cpu')
    owner.bind(w,b)
    plan=NS(widths=(CAND,)*8,graph_ids_safe=True,check_model=lambda *args:None)
    return w,b,owner,plan


@pytest.mark.parametrize('rows',[1,3,7])
@pytest.mark.parametrize('policy',[None,Sampling(73,top_k=1),Sampling(73,top_k=20,top_p=0.7,min_p=0.05)])
def test_original_transport_and_selection_match_with_int64_ids_and_reuse(rows,policy):
    w,b,owner,plan=fixture(rows)
    g=b.cand_all[:8*rows*65].view(8,rows,65).cpu().numpy()
    old_values=g[:,:,:CAND].astype(np.float32)
    old_ids=np.ascontiguousarray(g[:,:,CAND:2*CAND]).view(np.int32).astype(np.int64)
    with owner.borrow(w,b.cand_all,rows) as (values,ids,norms):
        assert np.array_equal(values.view(np.int32),old_values.view(np.int32))
        assert np.array_equal(ids,old_ids) and ids.dtype==np.int64
        assert np.array_equal(norms,g[:,:,2*CAND])
    positions=list(range(17,17+rows))
    before=decode.choose_gathered(w,b.cand_all,rows,positions,policy,with_prob=True,plan=plan)
    candidate=decode.choose_gathered(w,b.cand_all,rows,positions,policy,with_prob=True,plan=plan,readback=owner)
    assert candidate==before
    saved=(list(candidate[0]),list(candidate[1]))
    b.cand_all[:8*rows*65].view(8,rows,65)[:,:,0].fill_(-5)
    decode.choose_gathered(w,b.cand_all,rows,positions,policy,with_prob=True,plan=plan,readback=owner)
    assert candidate==saved  # No returned list/number borrows reusable storage.


@pytest.mark.parametrize('field,value',[(0,float('nan')),(0,float('inf')),(64,float('nan')),
                                       (64,float('inf')),(64,-float('inf'))])
def test_same_domain_failures_as_original(field,value):
    w,b,owner,plan=fixture(1)
    g=b.cand_all[:8*65].view(8,1,65)
    g[:,:,field].fill_(value)
    messages=[]
    for selected in (None,owner):
        with pytest.raises(ValueError) as error:
            decode.choose_gathered(w,b.cand_all,1,[1],None,plan=plan,readback=selected)
        messages.append(str(error.value))
    assert messages[0]==messages[1]
    assert not owner.lock.locked()


def test_padding_masks_and_id_validation_preserved():
    w,b,owner,plan=fixture(1)
    plan.widths=(2,)*8
    g=b.cand_all[:8*65].view(8,1,65)
    g[:,:,2:CAND].fill_(-float('inf'))
    g[:,:,CAND+2:2*CAND].view(torch.int32).fill_(-1)
    original=decode.choose_gathered(w,b.cand_all,1,[1],None,plan=plan)
    assert decode.choose_gathered(w,b.cand_all,1,[1],None,plan=plan,readback=owner)==original
    g[0,0,CAND+2].view(torch.int32).fill_(1)
    with pytest.raises(ValueError,match='padding'):
        decode.choose_gathered(w,b.cand_all,1,[1],None,plan=plan,readback=owner)


def test_fallback_releases_active_borrow_before_nested_sampler(monkeypatch):
    w,b,owner,_plan=fixture(1)
    b.cand_all[:8*65].view(8,1,65)[:,:,:CAND].fill_(0)
    logits=torch.zeros(1,256).as_subclass(HostPacket)
    calls=[]

    def fallback(*args,**kwargs):
        with owner.borrow(w,b.cand_all,1):
            calls.append(True)
        return [123]

    monkeypatch.setattr(decode,'tp_sample_rows',fallback)
    assert decode.choose_gathered(w,b.cand_all,1,[1],None,logits=logits,readback=owner)==[123]
    assert calls==[True]


def test_nested_close_and_exception_release():
    w,b,owner,_=fixture(1)
    with pytest.raises(LookupError), owner.borrow(w,b.cand_all,1):
        with pytest.raises(RuntimeError,match='active borrower'), owner.borrow(w,b.cand_all,1):
            pytest.fail('nested borrow accepted')
        with pytest.raises(RuntimeError,match='while borrowed'):
            owner.close()
        raise LookupError('consumer failed')
    with owner.borrow(w,b.cand_all,1):
        pass
    owner.close()
    with pytest.raises(RuntimeError,match='original live'), owner.borrow(w,b.cand_all,1):
        pytest.fail('closed owner accepted')


@pytest.mark.parametrize('mutation',['packet','packet-resize','host','ids','role','weights','comm','world','dtype'])
def test_stale_binding_or_storage_refused(mutation):
    w,b,owner,_=fixture(1)
    weights=w
    if mutation=='packet':
        b.cand_all=b.cand_all.clone()
    elif mutation=='packet-resize':
        b.cand_all.resize_(1)
    elif mutation=='host':
        owner.host=owner.host.clone()
    elif mutation=='ids':
        owner.ids=np.empty_like(owner.ids)
    elif mutation=='role':
        b.owner_role='mtp'
    elif mutation=='weights':
        weights=NS(**vars(w))
    elif mutation=='comm':
        w.comm=NS(world=8)
    elif mutation=='world':
        w.meta['world']=16
    else:
        b.execution_dtype=torch.bfloat16
    with pytest.raises((ValueError,RuntimeError)), owner.borrow(weights,b.cand_all,1):
        pytest.fail('stale owner accepted')


def test_copy_failure_poison_prevents_reuse(monkeypatch):
    w,b,owner,_=fixture(1)

    def fail(*args,**kwargs):
        raise RuntimeError('copy failure')

    monkeypatch.setattr(torch.Tensor,'copy_',fail)
    with pytest.raises(RuntimeError,match='copy failure'), owner.borrow(w,b.cand_all,1):
        pytest.fail('failed copy exposed packet')
    assert owner.poisoned and not owner.lock.locked()
    with pytest.raises(RuntimeError,match='original live'), owner.borrow(w,b.cand_all,1):
        pytest.fail('poisoned owner accepted')


def test_default_off_and_aggregate_host_credit_before_allocation():
    base={'kernel_family':'sm70-paged-owner-rn-ex2-p2051-v1','graph_family':'uncaptured',
          'graph_budget_bytes':0,'rn_heads_per_block':12,'trial':True}
    assert 'gathered_readback' not in parse_preparation(base)
    selected={**base,'gathered_readback':{'policy':POLICY,'host_budget_bytes':4128*5}}
    assert parse_preparation(selected)['gathered_readback']==selected['gathered_readback']
    OwnerPreparation(selected,role_rows={'decode':3,'mtp':2,'prefill':16},manifest_sha256='a'*64)
    with pytest.raises(MemoryError,match='aggregate host credit'):
        OwnerPreparation(selected,role_rows={'decode':4,'mtp':2,'prefill':16},manifest_sha256='a'*64)
    with pytest.raises(ValueError,match='explicit trial'):
        parse_preparation({**selected,'trial':False})
    assert layout(8,256)=={'pinned_host_bytes':532480,'pageable_host_bytes':524288,'host_payload_bytes':1056768}


def test_forecast_is_retained_and_host_allocation_deferred_until_complete_admission(monkeypatch):
    base={'kernel_family':'sm70-paged-owner-rn-ex2-p2051-v1','graph_family':'uncaptured',
          'graph_budget_bytes':0,'rn_heads_per_block':12,'trial':True,
          'gathered_readback':{'policy':POLICY,'host_budget_bytes':4128*5}}
    owner=OwnerPreparation(base,role_rows={'decode':3,'mtp':2,'prefill':16},manifest_sha256='a'*64)
    assert owner.gathered_readbacks=={}
    assert owner.gathered_host_payload_bytes()==4128*5
    assert owner.gathered_readback_forecast=={'decode':layout(8,3),'mtp':layout(8,2)}
    with pytest.raises(RuntimeError,match='admitted prepared owner'):
        owner.allocate_gathered_readbacks(NS(tp=8),'cpu')
    owner.prepared=True
    monkeypatch.setattr(owner,'_phase',lambda engine,name,operation,**kwargs:[operation()]*8)
    owner.allocate_gathered_readbacks(NS(tp=8),'cpu')
    assert set(owner.gathered_readbacks)=={'decode','mtp'}
    assert sum(o.cost['host_payload_bytes'] for o in owner.gathered_readbacks.values())==4128*5
    with pytest.raises(RuntimeError,match='one admitted prepared owner'):
        owner.allocate_gathered_readbacks(NS(tp=8),'cpu')
    for readback in owner.gathered_readbacks.values():
        readback.close()

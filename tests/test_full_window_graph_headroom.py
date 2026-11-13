"""Worst-rank physical headroom after complete resident capacity and service promises."""
import copy
from types import SimpleNamespace as NS
import pytest
from tensorfold.families.qwen4_exp.cuda import service_memory as memory
from tests.test_ram_graph_credit import decoder, future  # noqa: F401 - actual decoder fixture dependency

GIB=1<<30


def receipt(bits=8):
    slots=24 if bits==8 else 32
    ranks=[]
    for rank in range(8):
        free=(30 if rank!=7 else 29)*GIB
        ranks.append({'rank':rank,'allocation_baseline':{'physical_free_bytes':free,
            'effective_device_floor_bytes':4*GIB,'usable_serving_budget_bytes':free-4*GIB,
            'remaining_absolute_grant_bytes':None,'preallocated_project_bytes':GIB}})
    return {'native_window':262144,'context_window':262144,'cache_slots':262149,
        'session_capacity':{'context_tokens':262144,'full_context_commitments':slots,'resident_capacity':slots,'ranks':ranks},
        'budget_bytes':25*GIB,'weight_bytes_estimate':12*GIB,'startup_peak_bytes_estimate':13*GIB,
        'serving_peak_bytes_estimate':24*GIB,'total_bytes_estimate':24*GIB,
        'cache_workspace_bytes_estimate':12*GIB,'mapped_table_bytes':0}


@pytest.mark.parametrize('bits,slots',[(8,24),(4,32)])
def test_required_capacity_region_binds_full_context_and_physical_tail(bits,slots):
    r=receipt(bits)
    kwargs=dict(key_bits=bits,value_bits=bits,slots=slots,full_windows=slots,physical_capacity=262149)
    assert memory.full_window_headroom_required(r,**kwargs)
    assert memory.full_window_graph_budget(r)==29*GIB
    assert not memory.full_window_headroom_required(r,**{**kwargs,'slots':8})
    assert not memory.full_window_headroom_required(r,**{**kwargs,'physical_capacity':262144})
    r['context_window']=260060
    assert not memory.full_window_headroom_required(r,**kwargs)


def test_remaining_absolute_grant_and_preallocated_owners_are_not_credited_twice():
    r=receipt();base=r['session_capacity']['ranks'][7]['allocation_baseline']
    base['remaining_absolute_grant_bytes']=26*GIB;base['usable_serving_budget_bytes']=25*GIB
    assert memory.full_window_graph_budget(r)==26*GIB # NOT +preallocated, +floor or +allocatorreserved.
    base['usable_serving_budget_bytes']+=1
    with pytest.raises(ValueError):memory.full_window_graph_budget(r)


@pytest.mark.parametrize('failure',['missing','rank','negative','bool','inconsistent','lower'])
def test_unproven_rank_baselines_never_create_graph_credit(failure):
    r=receipt()
    if failure=='missing':r['session_capacity']['ranks'][3].pop('allocation_baseline')
    elif failure=='rank':r['session_capacity']['ranks'][2]['rank']=1
    elif failure=='negative':r['session_capacity']['ranks'][0]['allocation_baseline']['remaining_absolute_grant_bytes']=-1
    elif failure=='bool':r['session_capacity']['ranks'][0]['allocation_baseline']['physical_free_bytes']=True
    elif failure=='inconsistent':r['session_capacity']['ranks'][0]['allocation_baseline']['usable_serving_budget_bytes']+=1
    else:r['budget_bytes']=31*GIB
    with pytest.raises(ValueError):memory.full_window_graph_budget(r)


def test_new_receipt_preserves_original_header_and_reserves_observed_room_once():
    r=receipt();old=copy.deepcopy(r);proof={'policy':memory.FULL_GRAPH_HEADROOM_POLICY,
        'agreed_graph_bytes':5*GIB,'admitted_project_budget_bytes':29*GIB,'future_service':future()}
    memory.refine_materialized_graph_plan(r,proof)
    assert r['budget_bytes']==29*GIB and r['serving_peak_bytes_estimate']==24*GIB
    assert r['observed_graph_admission']['original_header']['budget_bytes']==old['budget_bytes']
    memory.reserve_graph_plan(r,5*GIB)
    assert r['total_bytes_estimate']==29*GIB and r['context_window']==262144 and r['cache_slots']==262149
    assert r['session_capacity']['full_context_commitments']==24
    with pytest.raises(RuntimeError):memory.refine_materialized_graph_plan(r,proof)


def test_gpu_full_window_observation_uses_all_rank_minimum_and_keeps_future(decoder,monkeypatch):
    multi,dec,calls,captures=decoder
    dec.w.meta['indexer_history']='gpu';dec.graph_project_budget=10000
    for state in dec.free:state.indexer_history_owner=None
    # fixture graph capture publishes the actual resolution, leaving GPU work mocked only at its boundary.
    monkeypatch.setattr(multi.multi_graphs,'live_bytes',lambda **kwargs:future()['future_bytes']+500)
    dec.capture_graphs()
    assert captures==[(200,True)] # peer's500→200minimum; no header_room31 cap.
    proof=dec.graph_record['observed_graph_admission']
    assert proof['policy']==memory.FULL_GRAPH_HEADROOM_POLICY and proof['live_reserve_bytes']==0
    assert proof['future_service']['workspace_bytes']==2*GIB
    assert proof['future_service']['snapshot_bytes']==80 and proof['future_service']['copy_overlap_bytes']==40


def test_gpu_history_does_not_make_partial_windows_full(decoder,monkeypatch):
    multi,dec,calls,captures=decoder;dec.w.meta['indexer_history']='gpu';dec.graph_project_budget=10000
    dec.free[0].capacity-=1
    monkeypatch.setattr(multi.multi_graphs,'live_bytes',lambda **kwargs:future()['future_bytes']+500)
    with pytest.raises(ValueError):dec.capture_graphs()
    assert not captures


def test_prepared_live_excludes_opaque_driver_and_private_spare_once(monkeypatch):
    import torch
    from tensorfold.cuda import capacity,kv_arena
    from tensorfold.families.qwen4_exp.cuda.owner_preparation import OwnerPreparation
    owner=object.__new__(OwnerPreparation);owner.private_spare_bytes=10;owner.outside_active_bytes=20
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda:100)
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda:160)
    calls=[]
    def available(runtime,**kwargs):calls.append(kwargs);return 200
    monkeypatch.setattr(capacity,'available_bytes',available)
    monkeypatch.setattr(capacity,'cuda_limit_bytes',lambda:300)
    monkeypatch.setattr(kv_arena,'mapped_bytes',lambda:30)
    # physical200+ordinaryspare35=235; cap300-active100-mapped30-preexisting20-newdriver40-private15=95.
    assert owner.live_bytes(reserve=0,additional_private_spare_bytes=15,additional_outside_active_bytes=40)==95
    assert calls==[{'reserve':0,'full_window_headroom':True}]


def test_scoped_cold_baseline_cannot_reject_required_capacity_for_unused_floor(monkeypatch):
    from tensorfold.cuda import capacity
    runtime=NS(cuda=NS(mem_get_info=lambda:(30*GIB,32*GIB),memory_allocated=lambda:GIB,
        memory_reserved=lambda:2*GIB,get_device_properties=lambda device:NS(is_integrated=False)))
    monkeypatch.setenv('TENSORFOLD_CUDA_MEMORY_RESERVE_GIB','4')
    monkeypatch.setenv(capacity.LIMIT_ENV,'29')
    original=capacity.device_allocation_baseline(runtime,preallocated_device_bytes=GIB)
    scoped=capacity.device_allocation_baseline(runtime,reserve=0,preallocated_device_bytes=GIB,
                                               full_window_headroom=True)
    assert original['usable_serving_budget_bytes']==26*GIB
    assert scoped['usable_serving_budget_bytes']==28*GIB # absolute29 minus preallocated1, notphysical30+alreadyallocated.
    assert scoped['effective_device_floor_bytes']==0
    assert capacity.available_bytes(runtime,reserve=0,full_window_headroom=True)==29*GIB
    assert capacity.available_bytes(runtime)==26*GIB
    with pytest.raises(ValueError):capacity.device_allocation_baseline(runtime,full_window_headroom=True)


@pytest.mark.parametrize('enabled,capture,geometry',[(1,True,object()),(True,False,object()),(True,True,None)])
def test_unbound_cold_floor_policy_refuses_before_header_or_device_access(enabled,capture,geometry):
    from tensorfold.cuda import capacity
    with pytest.raises(ValueError,match='measured resident-session'):
        capacity.admit('/absent',None,None,NS(),None,None,capture_device_baseline=capture,
            memory_reserve=0,full_window_headroom=enabled,session_geometry=geometry)

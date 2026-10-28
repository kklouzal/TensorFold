"""Catalogue resource/data/lifetime contracts
actual CUDA oracles are separate."""
import ast
from pathlib import Path
from types import SimpleNamespace as N

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import moe_region_catalogue as module


def model():
    def packed():
        return N(data=torch.ones((2,3),dtype=torch.uint8),ggml_type=12,n=2,k=3,column_offset=0)
    def dense():
        return N(weight=torch.ones((2,3),dtype=torch.float16),n=2,k=3,implementation='native32')
    bank=N(count=513,width=80,dims=2560,implementation='packed_exact',packed_rows=32,limit=0.0,swiglu=True,
           gate=packed(),up=packed(),down=packed(),shared=tuple(dense() for _ in range(3)))
    layers=[N(index=i,moe=N(router=torch.ones((2,3)),experts=bank)) for i in range(48)]
    w=N(layers=layers,cfg=N(experts=512,top_k=10),device=torch.device('cpu'),meta={
        'sm70_profile':module.PROFILE,'execution_provider':'sm70','sm70_moe_region_graphs_trial':True})
    b=N(mixed=torch.ones((32,3),dtype=torch.float16),prefill=False,rows=32,router_blas=object(),
        input_preparation_context=None,moe=N(logits=torch.zeros((32,4)),pick=torch.zeros((32,2),dtype=torch.int32),
        wts=torch.zeros((32,2)),y=torch.zeros((32,2,3),dtype=torch.float16),act=torch.empty((32,2,3))))
    return w,b


@pytest.fixture
def owner(monkeypatch):
    w,b=model()
    log=[]
    dec=N(w=w,buf=b,_memory_error=None)
    parent=N(dec=dec,admitted=True,budget=1<<30,allocator_backend='native')
    cat=module.Catalogue(parent)
    cat.bindings=module.Bindings(w,b)
    cat.ready=True
    cat.stream_id=7
    monkeypatch.setattr(module.torch.cuda,'current_stream',lambda device:N(cuda_stream=7))
    monkeypatch.setattr(module.torch.cuda,'synchronize',lambda device:log.append('global-fence'))
    cat.capture_router=object()
    for rows in module.ROWS:
        for layer in w.layers:
            cat.regions[rows,layer.index]=N(graph=object(),replay=lambda r=rows,i=layer.index:log.append((r,i)),
                _release_fenced=lambda r=rows,i=layer.index:log.append(('reset',r,i)))
    cat.pools={row:(0,row) for row in module.ROWS}
    return cat,w,b,log


def test_declared_funding_and_explicit_profile_gate(monkeypatch):
    plan=module.funding()
    assert plan['maximum_graphs']==144 and plan['rows']==(30,31,32)
    assert plan['future_host_bytes_per_rank']==1<<30
    assert plan['capture_role_host_bound_bytes']<plan['future_host_bytes_per_rank']
    monkeypatch.setenv(module.ENV,'1')
    assert module.requested(module.PROFILE)
    with pytest.raises(ValueError,match='batched-router'):
        module.requested('reference')
    monkeypatch.setenv(module.ENV,'2')
    with pytest.raises(ValueError,match='0 or1'):
        module.requested(module.PROFILE)


def test_storage_rebind_detected_and_old_storage_retained():
    w,b=model()
    bindings=module.Bindings(w,b)
    original=b.mixed.untyped_storage()
    old_pointer=original.data_ptr()
    b.mixed.set_(torch.empty_like(b.mixed))  # Tensor identity stays fixed.
    assert bindings.tensors[0][2] is b.mixed
    assert bindings.tensors[0][3].data_ptr()==old_pointer
    with pytest.raises(RuntimeError,match='storage/span/layout'):
        bindings.validate()


def test_inplace_input_values_are_dynamic_but_layout_and_weight_binding_are_not():
    w,b=model()
    bindings=module.Bindings(w,b)
    b.mixed.add_(1)
    bindings.validate()
    w.layers[0].moe.router=torch.ones_like(w.layers[0].moe.router)
    with pytest.raises(RuntimeError,match='storage/span/layout'):
        bindings.validate()


def test_scalar_and_effective_profile_invalidation():
    w,b=model()
    bindings=module.Bindings(w,b)
    w.layers[0].moe.experts.width=160
    with pytest.raises(RuntimeError,match='scalar/layout'):
        bindings.validate()
    w,b=model()
    bindings=module.Bindings(w,b)
    w.meta['sm70_profile']='other'
    with pytest.raises(RuntimeError,match='effective profile'):
        bindings.validate()


def test_exact_row_cycle_and_other_rows_eager(owner):
    cat,w,b,log=owner
    assert cat.begin(w,b,29) is False
    assert not cat.active and not log
    for rows in (32,30,31,32):
        assert cat.begin(w,b,rows)
        for layer in w.layers:
            assert cat.run(layer,b,rows)
        cat.finish()
    assert cat.frames==4 and cat.hits==192
    assert cat.row_frames=={30:1,31:1,32:2}
    assert log[:48]==[(32,i) for i in range(48)]
    assert log[48]==(30,0)


def test_partial_or_reordered_cycle_poison_and_no_fallback(owner):
    cat,w,b,log=owner
    cat.begin(w,b,32)
    with pytest.raises(RuntimeError,match='order') as caught:
        cat.run(w.layers[1],b,32)
    with pytest.raises(RuntimeError) as finished:
        cat.finish(caught.value)
    assert finished.value is caught.value and cat.failure is caught.value
    assert not cat.active and cat.retiring and cat.regions
    assert cat.dec._memory_error is caught.value
    with pytest.raises(RuntimeError,match='not reusable'):
        cat.begin(w,b,29)  # Poison is not an optional eager capability decline.
    assert not log


def test_missing_final_layer_is_not_a_completed_frame(owner):
    cat,w,b,_=owner
    cat.begin(w,b,30)
    cat.run(w.layers[0],b,30)
    with pytest.raises(RuntimeError,match='full ordered'):
        cat.finish()
    assert cat.frames==0 and not cat.active and cat.retiring
    assert cat.frame_lock.acquire(blocking=False)
    cat.frame_lock.release()


def test_binding_failure_notifies_after_lease_release_preserving_primary(owner):
    cat,w,b,log=owner
    secondary=OSError('parent-notification')
    def fail(error):
        assert cat.frame_lock.acquire(blocking=False)
        cat.frame_lock.release()
        raise secondary
    b.input_preparation_context=N(fail=fail)
    b.mixed.set_(torch.empty_like(b.mixed))
    with pytest.raises(RuntimeError) as caught:
        cat.begin(w,b,32)
    assert caught.value is cat.failure and caught.value.__cause__ is secondary
    assert not cat.active and not log


def test_close_fences_before_graph_reset_and_shared_release(owner):
    cat,_,_,log=owner
    role=cat.capture_router
    first=cat.regions[30,0]
    def reset():
        assert log==['global-fence'] and cat.capture_router is role
        log.append('first-reset')
    first._release_fenced=reset
    cat.close()
    assert cat.closed and cat.capture_router is None and not cat.regions
    assert log[0]=='global-fence'


def test_close_failure_retains_borrowed_storage_and_refuses_reuse(owner):
    cat,w,b,_=owner
    error=OSError('graph-reset')
    cat.regions[30,0]._release_fenced=lambda: (_ for _ in ()).throw(error)
    role=cat.capture_router
    with pytest.raises(OSError) as caught:
        cat.close()
    assert caught.value is error and cat.capture_router is role and cat.bindings is not None
    assert not cat.closed and cat.retiring
    with pytest.raises(RuntimeError):
        cat.begin(w,b,32)


def test_close_refuses_active_borrower_before_fence(owner):
    cat,w,b,log=owner
    cat.begin(w,b,32)
    with pytest.raises(RuntimeError,match='borrower'):
        cat.close()
    assert not log and cat.capture_router is not None
    with pytest.raises(ValueError):
        cat.finish(ValueError('cancelled-partial'))


def test_host_operating_gate_has_no_vendor_bound_claim(owner,monkeypatch):
    cat,_,_,_=owner
    cat.host_before={'VmRSS':1000,'VmHWM':10000}
    monkeypatch.setattr(module,'_host_memory',lambda:{'VmRSS':1001+module.HOST_ALLOWANCE,'VmHWM':10000})
    with pytest.raises(MemoryError,match='host operating'):
        cat._host_gate()
    assert cat.host_observed_peak_growth==0
    assert cat.host_observed_retained_growth==module.HOST_ALLOWANCE+1


def test_actual_engine_funding_promotes_to_final_admission_before_floor():
    path=Path(__file__).resolve().parents[1]/'src/tensorfold/families/qwen4_exp/cuda/engine.py'
    tree=ast.parse(path.read_text())
    constructor=next(n for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='FlashNextEngine'
                     for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_initialize')
    block=next(n for n in ast.walk(constructor) if isinstance(n,ast.If) and
               isinstance(n.test,ast.Call) and isinstance(n.test.func,ast.Name) and n.test.func.id=='getattr' and
               len(n.test.args)>1 and isinstance(n.test.args[1],ast.Constant) and n.test.args[1].value=='_moe_region_graphs_requested')
    promote=next(n for n in ast.walk(constructor) if isinstance(n,ast.For) and
                 isinstance(n.iter,ast.Tuple) and any(isinstance(v,ast.Constant) and v.value=='host_workspace_bytes' for v in n.iter.elts))
    admit=next(n for n in ast.walk(constructor) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='admit')
    assert block.lineno<promote.lineno<admit.lineno
    options={'host_workspace_bytes':137,'device_workspace_bytes':19}
    host={}
    scope={'self':N(_moe_region_graphs_requested=True),'options':options,'tp':8,'host_capacity':host,
           '__package__':'tensorfold.families.qwen4_exp.cuda'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[block,promote],type_ignores=[])),'engine-funding','exec'),scope)
    assert host['host_workspace_bytes']==137+8*module.HOST_ALLOWANCE
    assert options['device_workspace_bytes']==19  # GPU role is paid in settled graphs, never twice.
    assert any(k.arg is None and isinstance(k.value,ast.Name) and k.value.id=='host_capacity' for k in admit.keywords)
    floor=next(n for n in ast.walk(constructor) if isinstance(n,ast.If) and 'streams < 24' in ast.unparse(n.test))
    from tensorfold.cuda.sm70 import execution_policy
    names={name:getattr(execution_policy,name) for name in dir(execution_policy) if name.endswith('_TRIAL')}
    names.update(self=N(sm70_profile=module.PROFILE),streams=23)
    code=compile(ast.fix_missing_locations(ast.Module(body=[floor],type_ignores=[])),'engine-floor','exec')
    with pytest.raises(ValueError,match='at least24'):
        exec(code,names)
    names['streams']=24
    exec(code,names)


def test_all_four_private_pools_are_excluded_from_available_cached_bytes(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import multi_graphs
    owner=N(failure=None,failure_frames=None,allocator_backend='native',pool=(0,1),
            dec=N(w=N(device=N(index=0))),moe_regions=N(pools={30:(0,2),31:(0,3),32:(0,4)}),
            private_spare_bytes=None)
    def snapshot(*,mempool_id,include_traces):
        return [{'device':0,'segment_pool_id':mempool_id,'total_size':100+10*mempool_id[1],
                 'allocated_size':100,'active_size':100}]
    monkeypatch.setattr(multi_graphs.torch.cuda,'memory_snapshot',snapshot)
    monkeypatch.setattr(multi_graphs.torch.cuda,'memory_reserved',lambda:1000)
    monkeypatch.setattr(multi_graphs.torch.cuda,'memory_allocated',lambda:500)
    monkeypatch.setattr(multi_graphs,'free_bytes',lambda:5000)
    monkeypatch.setattr(multi_graphs,'live_bytes',lambda:5500)
    multi_graphs.RoundGraphs._account_pool(owner)
    assert owner.private_spare_bytes==100
    assert [r['family'] for r in owner.private_pool_records]==['n1','moe-target-r30','moe-target-r31','moe-target-r32']


def test_supported_shared_tuple_and_member_storage_changes_refuse():
    w,b=model()
    bindings=module.Bindings(w,b)
    bank=w.layers[0].moe.experts
    bank.shared=(bank.shared[1],bank.shared[0],bank.shared[2])
    with pytest.raises(RuntimeError,match='original model/buffer'):
        bindings.validate()
    w,b=model()
    bindings=module.Bindings(w,b)
    w.layers[0].moe.experts.shared[0].weight.set_(torch.empty((2,3),dtype=torch.float16))
    with pytest.raises(RuntimeError,match='storage/span/layout'):
        bindings.validate()


def _leaves(error):
    if isinstance(error,BaseExceptionGroup):
        return sum((_leaves(part) for part in error.exceptions),[])
    return [error]


def test_actual_round_finally_attempts_all_frontiers_and_preserves_compute_primary():
    from tensorfold.families.qwen4_exp.cuda.capture_lifetime import finish_forward_frontiers
    path=Path(__file__).resolve().parents[1]/'src/tensorfold/families/qwen4_exp/cuda/multi.py'
    tree=ast.parse(path.read_text())
    round_function=next(n for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='MultiDecoder'
                        for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_round')
    source_try=next(n for n in ast.walk(round_function) if isinstance(n,ast.Try) and
                    any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                        and node.func.id=='finish_forward_frontiers' for node in ast.walk(ast.Module(body=n.finalbody,type_ignores=[]))))
    primary=ValueError('original-compute')
    observer_error,gdn_error,moe_error=OSError('observer'),RuntimeError('gdn-seal'),OSError('moe-retire')
    log=[]
    def frontier(label,error):
        def run(original):
            assert original is primary
            log.append(label)
            raise error
        return run
    buf=N(gdn_tables=object(),attn_step=object(),gdn_state_batch=None)
    scope={'self':N(buf=buf,controller=None),'gdn_primary':None,'moe_active':True,
           'frontier_journal':([None]*3,[None],[None]*2,[None]*3),
           'observer_finish':frontier('observer',observer_error),
           'gdn_finish':frontier('gdn',gdn_error),'moe_finish':frontier('moe',moe_error),
           'primary_to_raise':primary,'finish_forward_frontiers':finish_forward_frontiers}
    actual=ast.Try(body=[ast.Raise(exc=ast.Name(id='primary_to_raise',ctx=ast.Load()),cause=None)],
                   handlers=source_try.handlers,orelse=[],finalbody=source_try.finalbody)
    code=compile(ast.fix_missing_locations(ast.Module(body=[actual],type_ignores=[])),'actual-round-finally','exec')
    with pytest.raises(ValueError) as caught:
        exec(code,scope)
    assert caught.value is primary
    assert log==['observer','gdn','moe']
    assert {id(value) for value in _leaves(primary.__cause__)}=={id(observer_error),id(gdn_error),id(moe_error)}
    assert buf.gdn_tables is None and buf.attn_step is None


def test_actual_round_finally_closes_real_moe_cycle_after_gdn_seal_failure(owner):
    from tensorfold.families.qwen4_exp.cuda.capture_lifetime import finish_forward_frontiers
    cat,w,b,_=owner
    cat.begin(w,b,32)
    primary=ValueError('compute')
    seal_error=OSError('seal')
    def seal(original):
        assert original is primary
        raise seal_error
    journal=([None]*3,[None],[None]*2,[None]*3)
    with pytest.raises(ValueError) as caught:
        finish_forward_frontiers(primary,seal,cat.finish,None,journal)
    assert caught.value is primary and primary.__cause__ is seal_error
    assert not cat.active and cat.retiring and cat.capture_router is not None and cat.bindings is not None
    assert cat.frame_lock.acquire(blocking=False)
    cat.frame_lock.release()


def test_capture_charge_does_not_subtract_private_from_physical_authority_twice():
    from tensorfold.families.qwen4_exp.cuda.multi_graphs import RoundGraphs
    before={'physical_free_bytes':1200,'general_available_bytes':1250,'unavailable_owned_bytes':300}
    after={'physical_free_bytes':1100,'general_available_bytes':1150,'unavailable_owned_bytes':400,
           'private_reserved_bytes':400,'private_active_bytes':0}
    parent=N(bytes=0,room_bytes=0,budget=1000,future_bytes=1000,private_spare_bytes=400,
             spare=lambda:100,capture_observation=lambda:after)
    row=RoundGraphs.charge_capture(parent,before,['test'])
    assert row['service_spare_after_private_exclusion']==150
    assert row['charged_bytes']==100 and parent.bytes==100
    # The former spare()-private_spare arithmetic would have falsely refused -300.


def test_capture_charge_debits_new_unavailable_owned_bytes_even_without_driver_drop():
    from tensorfold.families.qwen4_exp.cuda.multi_graphs import RoundGraphs
    before={'physical_free_bytes':1200,'general_available_bytes':1100,'unavailable_owned_bytes':100}
    after={'physical_free_bytes':1200,'general_available_bytes':1100,'unavailable_owned_bytes':180,
           'private_reserved_bytes':180,'private_active_bytes':0}
    parent=N(bytes=0,room_bytes=0,budget=1000,future_bytes=1000,private_spare_bytes=20,
             capture_observation=lambda:after)
    row=RoundGraphs.charge_capture(parent,before,['cached-to-owned'])
    assert row['charged_bytes']==80 and parent.bytes==80

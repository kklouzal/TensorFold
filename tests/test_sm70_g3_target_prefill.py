"""CPU accepted-frame/status/funding controls; no native/model acceptance."""
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import gdn_chunk64_prefill as adapter
from tensorfold.families.qwen4_exp.cuda import gdn_prefill_policy as policy
from tensorfold.families.qwen4_exp.cuda.forward import Cut
from tensorfold.families.qwen4_exp.cuda.owner_preparation import parse_preparation
from tensorfold.families.qwen4_exp.cuda.state import State
from tests.test_sm70_gdn_chunk64_prefill import cpu_prepared, cpu_state


def request():
    return {'policy':policy.POLICY,'tensor_budget_bytes':4936*256+21,'journal_host_budget_bytes':1<<26,
        'cold_host_budget_bytes':1<<28,'warmup_device_budget_bytes':5752*256+804656,
        'cuda_home':'/protected/cuda','nvcc_sha256':'a'*64,
        'artifact':{'attestation':'/operator/owned.json','attestation_bytes':1,'attestation_sha256':'b'*64,
                    'wheel':'/operator/owned.whl','hash_byte_credit':1,'native_base_wheel':None}}


def geometry():
    return {'num_hidden_layers':48,'layer_types':['linear_attention']*36+['full_attention']*12,
        'linear_num_key_heads':16,'linear_num_value_heads':48,'linear_key_head_dim':128,
        'linear_value_head_dim':128,'linear_conv_kernel_dim':4}


def prepared_group(monkeypatch, *, host_credit=1<<26):
    prepared,tensors,calls = cpu_prepared(monkeypatch,6)
    prepared.configure_groups(1,6,host_budget_bytes=host_credit)
    a,b = cpu_state(),cpu_state()
    b.conv.mul_(-1)
    for state in (a,b):
        state.kv_status = torch.tensor([0,-2],dtype=torch.int32)
    segs = ((a,0,2),(b,2,6))
    cuts = (Cut(1,at=0),Cut(2,at=2))
    return prepared,tensors,calls,segs,cuts


def submit_all(prepared,group,tensors,segs):
    p,_,weight,parameter,bias,norm,out,sums = tensors
    for state,start,end in segs:
        group.submit(state,0,p[start:end],weight,bias,norm,1e-6,out[start:end],sums[start:end],
                     a_log=parameter,at=start)


def test_packed_group_reuses_one_slab_without_layer_checks_or_waits(monkeypatch):
    prepared,tensors,calls,segs,cuts = prepared_group(monkeypatch)
    checks = []
    original = State.kv_check
    def checked(state):
        checks.append(state)
        return original(state)
    monkeypatch.setattr(State,'kv_check',checked)
    with torch.inference_mode():
        group = prepared.begin_group(segs,cuts,cut_budget_bytes=adapter.cut_bytes(1,2))
        assert len(checks) == 2
        for state,_,_ in segs:
            state.kv_begin()
        group.target_started()
        submit_all(prepared,group,tensors,segs)
        assert len(checks) == 2  # Original final check remains pending, no per-layer readback.
        assert all(state._kv_pending for state,_,_ in segs)
        assert prepared.event.recorded == prepared.event.waited == 0
        group.seal()
        assert prepared.event.recorded == 1 and prepared.event.waited == 0
        with pytest.raises(RuntimeError,match='whole-frame'):
            group.wait_native_reuse()
        assert prepared.active is group and all(state._kv_pending for state,_,_ in segs)
        for state,_,_ in segs:
            state.kv_check()  # The actual existing authoritative status latch.
        group.mark_validated()
        snapshots = [(cut.rec,cut.conv,cut.rec.clone(),cut.conv.clone()) for cut in cuts]
        group.wait_native_reuse()
    assert prepared.event.waited == 1 and group.retired and prepared.active is None
    assert all(command is None for command in prepared.group_journal)
    assert all(not state.history_native_readers for state,_,_ in segs)
    assert all(torch.equal(rec,rr) and torch.equal(conv,cc) for rec,conv,rr,cc in snapshots)
    assert sum(kind=='front' for kind,*_ in calls if type(kind) is str) == 2


def test_codec_failure_is_not_cleared_or_committed_and_poison_is_group_wide(monkeypatch):
    prepared,tensors,_,segs,cuts = prepared_group(monkeypatch)
    with torch.inference_mode():
        group = prepared.begin_group(segs,cuts,cut_budget_bytes=adapter.cut_bytes(1,2))
        group.target_started()
        for state,_,_ in segs:state.kv_begin()
        submit_all(prepared,group,tensors,segs)
        group.seal()
        failure = RuntimeError('authoritative codec/cohort failure')
        group.fail(failure)
        with pytest.raises(RuntimeError) as caught:group.wait_native_reuse()
    assert caught.value is failure and prepared.active is group
    assert all(state._history_error is state._kv_error is failure and state._kv_pending for state,_,_ in segs)
    assert all(state.pos == 0 for state,_,_ in segs)
    assert any(command is not None for command in prepared.group_journal)


def test_group_bounds_and_cross_state_cut_alias_refuse_before_submission(monkeypatch):
    prepared,_,calls,segs,cuts = prepared_group(monkeypatch)
    cuts[0].rec = segs[1][0].rec[0]
    cuts[0].conv = segs[1][0].conv
    with torch.inference_mode(),pytest.raises(ValueError,match='disjoint'):
        prepared.begin_group(segs,cuts,cut_budget_bytes=adapter.cut_bytes(1,2))
    assert prepared.active is None and calls == []
    assert all(not state.history_native_readers for state,_,_ in segs)


def test_unvalidated_frame_cannot_retire_or_publish_complete(monkeypatch):
    prepared,_,_,segs,cuts = prepared_group(monkeypatch)
    with torch.inference_mode():
        group = prepared.begin_group(segs,cuts,cut_budget_bytes=adapter.cut_bytes(1,2))
        group.target_started()
        with pytest.raises(RuntimeError,match='every linear'):
            group.seal()
        with pytest.raises(RuntimeError,match='whole-frame'):
            group.mark_validated()
        with pytest.raises(RuntimeError,match='before target'):
            group.wait_native_reuse()


@pytest.mark.parametrize('key',['tensor_budget_bytes','journal_host_budget_bytes','cold_host_budget_bytes','warmup_device_budget_bytes'])
def test_closed_request_refuses_boolean_or_missing_credit(key):
    value = request()
    value[key] = True
    with pytest.raises(ValueError):policy.Request.parse(value)
    del value[key]
    with pytest.raises(ValueError):policy.Request.parse(value)


def test_actual_pbuf_capacity_is_not_attention_role_minimum_or_layer_multiplied():
    selected = policy.Request.parse(request())
    facts = policy.forecast(geometry(),256,256,selected)
    assert facts['tensor_payload_bytes'] == 1263637 and facts['legacy_replay_bytes_omitted'] == 667648
    assert facts['journal_commands'] == 36*256 and facts['maximum_spans'] == 4*36*256
    assert facts['cut_snapshot_payload_credit'] == 0  # Actual admitted State count binds this later.
    value = request();value['tensor_budget_bytes'] -= 1
    with pytest.raises(MemoryError):policy.forecast(geometry(),256,256,policy.Request.parse(value))
    with pytest.raises(ValueError):policy.forecast(geometry(),256,257,selected)


def test_new_mode_is_numeric_and_absent_baseline_spec_is_unchanged():
    from tensorfold.families.qwen4_exp.cuda.owner_preparation import RN_KERNEL_FAMILY
    base = {'kernel_family':RN_KERNEL_FAMILY,'rn_heads_per_block':12,'graph_family':'uncaptured',
            'graph_budget_bytes':0,'trial':True}
    original = parse_preparation(base)
    assert 'g3_prefill' not in original
    parsed = parse_preparation({**base,'g3_prefill':request()})
    assert parsed['g3_prefill'] == request()
    assert {key:value for key,value in parsed.items() if key != 'g3_prefill'} == original
    with pytest.raises(ValueError):parse_preparation({**base,'trial':False,'g3_prefill':request()})
    for invalid in (None,False,True,{}):
        with pytest.raises(ValueError):parse_preparation({**base,'g3_prefill':invalid})


@pytest.mark.parametrize('codec_fails',[False,True])
def test_actual_target_compute_owns_phase_and_preserves_final_codec_check(monkeypatch,codec_fails):
    from tensorfold.families.qwen4_exp.cuda import compact_indexer, forward

    prepared,tensors,_,segs,cuts = prepared_group(monkeypatch)
    buffers = SimpleNamespace(prefill=True,split=None,ids=torch.zeros(6,dtype=torch.long),
        h=torch.empty(6,4*2560),g3_prefill_owner=SimpleNamespace(
            begin=lambda w,b,segs,cuts:prepared.begin_group(segs,cuts,cut_budget_bytes=adapter.cut_bytes(1,2))))
    weights = SimpleNamespace(cfg=SimpleNamespace(hidden=2560,streams=4),layers=[object()],mixer=None,
                              meta={},comm=None)
    monkeypatch.setattr(compact_indexer,'validate_target_indexer_frame',lambda *args:None)
    monkeypatch.setattr(forward,'_embed',lambda w,ids,copies,dest:dest.zero_())
    monkeypatch.setattr(forward.image_rows,'embed',lambda *args:None)
    def layer(*args,**kwargs):
        assert buffers.g3_target_frame is prepared.active
        submit_all(prepared,buffers.g3_target_frame,tensors,segs)
    monkeypatch.setattr(forward,'layer_forward',layer)
    monkeypatch.setattr(forward,'finish',lambda *args,**kwargs:torch.empty(0))
    original = forward._kv_check
    observations = []
    primary = RuntimeError('authoritative whole-frame codec failure')
    def final_check(*args,**kwargs):
        group = buffers.g3_target_frame
        observations.append(group)
        assert group.sealed and not group.validated and prepared.event.waited == 0
        assert all(state._kv_pending for state,_,_ in segs)
        if codec_fails:raise primary
        return original(*args,**kwargs)
    monkeypatch.setattr(forward,'_kv_check',final_check)
    with torch.inference_mode():
        if codec_fails:
            with pytest.raises(RuntimeError) as caught:forward.compute(weights,segs,buffers,cuts=cuts)
            assert caught.value is primary
        else:
            forward.compute(weights,segs,buffers,cuts=cuts)
    assert len(observations) == 1 and buffers.g3_target_frame is None
    if codec_fails:
        assert prepared.active is observations[0] and prepared.event.waited == 0
        assert all(state._history_error is primary and state._kv_pending for state,_,_ in segs)
    else:
        assert observations[0].validated and observations[0].retired and prepared.event.waited == 1
        assert prepared.active is None and all(not state._kv_pending for state,_,_ in segs)


def test_requested_capability_fails_before_native_initialization_when_host_authority_absent(monkeypatch):
    from tensorfold.cuda import capacity
    from tensorfold.cuda.sm70 import gdn_chunk64

    owner = policy.Owner(policy.Request.parse(request()),policy.forecast(geometry(),256,256,policy.Request.parse(request())))
    def missing_authority(*args,**kwargs):raise MemoryError('host physical authority unavailable')
    monkeypatch.setattr(capacity,'admit_host_commitment',missing_authority)
    monkeypatch.setattr(gdn_chunk64.Compiler,'inspect',lambda self:pytest.fail('native initialization before authority'))
    with pytest.raises(MemoryError,match='host'):
        owner.admit_host()
    assert owner.prepared is None


def test_reader_registration_failure_rolls_back_prior_readers_and_preserves_primary(monkeypatch):
    prepared,_,_,segs,cuts = prepared_group(monkeypatch)
    primary = MemoryError('reader registration exhausted')
    class FailedReaders:
        def add(self,reader):raise primary
        def discard(self,reader):pass
    segs[1][0].history_native_readers = FailedReaders()
    with torch.inference_mode(),pytest.raises(MemoryError) as caught:
        prepared.begin_group(segs,cuts,cut_budget_bytes=adapter.cut_bytes(1,2))
    assert caught.value is primary and prepared.active is None
    assert not segs[0][0].history_native_readers


def test_requested_missing_gdn_package_is_a_clear_cold_failure_not_fallback(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    from tensorfold.cuda import capacity
    from tensorfold.cuda.sm70 import gdn_chunk64

    selected = policy.Request.parse(request())
    owner = policy.Owner(selected,policy.forecast(geometry(),256,256,selected))
    monkeypatch.setattr(capacity,'admit_host_commitment',lambda *args,**kwargs:{'scope':'CPU-procedure-only'})
    owner.admit_host()
    primary = PackageNotFoundError('tilelang')
    def missing(self):raise primary
    monkeypatch.setattr(gdn_chunk64.Compiler,'inspect',missing)
    with pytest.raises(RuntimeError,match='protected owned GDN image') as caught:
        owner.prepare(torch.device('cpu'))
    assert caught.value.__cause__ is primary and owner.prepared is None


def test_latched_original_codec_failure_is_preserved_before_new_gdn_submission(monkeypatch):
    prepared,tensors,calls,segs,cuts = prepared_group(monkeypatch)
    with torch.inference_mode():
        group = prepared.begin_group(segs,cuts,cut_budget_bytes=adapter.cut_bytes(1,2))
        group.target_started()
        primary = RuntimeError('original codec failure remains owning')
        segs[1][0]._kv_error = primary
        with pytest.raises(RuntimeError) as caught:submit_all(prepared,group,tensors,segs)
    assert caught.value is primary and calls == [] and prepared.active is group
    assert all(state._history_error is state._kv_error is primary for state,_,_ in segs)


def test_explicit_worker_handoff_reuses_compiled_set_and_refuses_live_group(monkeypatch):
    import threading

    prepared,_,_,segs,cuts = prepared_group(monkeypatch)
    owner = policy.Owner(policy.Request.parse(request()),{})
    owner.prepared = prepared
    compiled = prepared.core.cumsum,prepared.core.kkt,prepared.core.core
    prepared.core.owner = threading.Thread()
    owner.handoff_to_current_thread()
    assert prepared.core.owner is threading.current_thread() and prepared.event.waited == 1
    assert (prepared.core.cumsum,prepared.core.kkt,prepared.core.core) == compiled
    with torch.inference_mode():
        group = prepared.begin_group(segs,cuts,cut_budget_bytes=adapter.cut_bytes(1,2))
        with pytest.raises(RuntimeError,match='live borrower'):owner.handoff_to_current_thread()
        assert prepared.event.waited == 1
        group.wait_native_reuse()  # Unsubmitted frame: no device consumer exists.


@pytest.mark.parametrize('version',['legacy','rn12','aten'])
def test_existing_closed_teacher_contracts_cannot_admit_g3_as_transport(version):
    from tensorfold.families.qwen4_exp.cuda.owner_preparation import RN_KERNEL_FAMILY
    from tools import v100_target_window_teacher as window

    value = {'kernel_family':RN_KERNEL_FAMILY,'rn_heads_per_block':12,'graph_family':'uncaptured',
        'graph_budget_bytes':0,'trial':True,'collective_family':'tp8-allgather-v1','row_buckets':{},
        'native_gdn_capture':True,'native_hc_writeback_capture':True,
        'exact_graph_policy':{'policy':'paid-subset-v1','keys':[['verify',8,8],['mtp',8,8]],'evidence_sha256':'d'*64},
        'g3_prefill':request()}
    contract = {'legacy':None,'rn12':window.PREPARATION_CONTRACT,'aten':window.ATEN_PREPARATION_CONTRACT}[version]
    with pytest.raises(ValueError):
        window.validate_preparation(value,{key:False for key in window.numerical.AXES},contract)


@pytest.mark.parametrize('changed',['thread','stream'])
def test_group_actor_proof_precedes_state_checks_or_target_writes(monkeypatch,changed):
    import threading

    prepared,_,calls,segs,cuts = prepared_group(monkeypatch)
    if changed == 'thread':prepared.core.owner = threading.Thread()
    else:monkeypatch.setattr(torch.cuda,'current_stream',lambda device:SimpleNamespace(cuda_stream=18))
    with torch.inference_mode(),pytest.raises(RuntimeError,match='actor thread/device/stream'):
        prepared.begin_group(segs,cuts,cut_budget_bytes=adapter.cut_bytes(1,2))
    assert calls == [] and prepared.active is None
    assert all(not state.history_native_readers and not state._kv_pending for state,_,_ in segs)

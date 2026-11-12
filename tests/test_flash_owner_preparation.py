"""Typed Flash pre-State resource/lifetime contract; no GPU or model emulation."""
from types import SimpleNamespace as NS
import pytest

from tensorfold.cuda.sm70 import flash_v100_artifact as artifact
from tensorfold.cuda.sm70.attention_backends import flash_owner_requirements
from tensorfold.families.qwen4_exp.cuda import owner_preparation as prep, owner_execution as execution


def specification(**changes):
    value = dict(kernel_family=execution.FLASH_KERNEL_FAMILY, graph_family=execution.UNCAPTURED,
        graph_budget_bytes=0, trial=True, flash_tiles=dict(panel_rows=3, tensor_budget_bytes=1 << 30,
            native_manifest='/operator/flash-v100/manifest.json', native_manifest_sha256='a'*64))
    value.update(changes)
    return value


@pytest.fixture
def validated(monkeypatch):
    result = NS(manifest_sha256='a'*64, profile_sha256='b'*64,
                record=lambda: {'manifest_sha256':'a'*64, 'profile_sha256':'b'*64})
    monkeypatch.setattr(artifact, 'validate_manifest', lambda path, sha: result)
    return result


@pytest.mark.parametrize('change', [{'trial':False}, {'graph_family':execution.GRAPH_FAMILY,'graph_budget_bytes':1},
    {'flash_tiles':None}, {'flash_tiles':{}},
    {'flash_tiles':dict(panel_rows=True,tensor_budget_bytes=1024,native_manifest='/x',native_manifest_sha256='a'*64)},
    {'flash_tiles':dict(panel_rows=3,tensor_budget_bytes=1024,native_manifest='relative',native_manifest_sha256='a'*64)},
    {'flash_tiles':dict(panel_rows=3,tensor_budget_bytes=1024,native_manifest='/x',native_manifest_sha256='A'*64)},
    {'reference_arithmetic':{'policy':prep.ARITHMETIC_POLICY,'graph_budget_bytes':1}},
    {'aten_tiles':{'panel_rows':1,'tensor_budget_bytes':1024}}])
def test_flash_schema_refuses_implicit_unbounded_or_other_numeric_modes(change):
    with pytest.raises(ValueError): prep.parse_preparation(specification(**change))


def test_new_family_preserves_existing_controls_and_pays_every_actual_role_before_validation(monkeypatch, validated):
    roles = {'decode':4,'mtp':4,'prefill':8}
    contract = specification(native_gdn_capture=True, native_hc_writeback_capture=True, gdn_value8_capture=True)
    parsed = prep.parse_preparation(contract)
    assert parsed['native_gdn_capture'] and parsed['native_hc_writeback_capture'] and parsed['gdn_value8_capture']
    owner = prep.OwnerPreparation(contract,role_rows=roles,manifest_sha256='c'*64)
    expected = sum(flash_owner_requirements(rows,3)['persistent_bytes'] for rows in roles.values())
    assert sum(v['persistent_bytes'] for v in owner.flash_requirements.values()) == expected
    assert owner.additional_temporary_bytes == 0
    assert owner.numerical_selection['attention']['native_profile_sha256'] == 'b'*64
    monkeypatch.setattr(artifact,'validate_manifest',lambda *a:pytest.fail('unfunded request touched artifact'))
    under = specification(flash_tiles={**contract['flash_tiles'],'tensor_budget_bytes':expected-1})
    with pytest.raises(MemoryError,match='aggregate'): prep.OwnerPreparation(under,role_rows=roles,manifest_sha256='c'*64)
    with pytest.raises(ValueError): prep.parse_preparation({**specification(),'kernel_family':execution.KERNEL_FAMILY})


def test_cold_load_and_all_role_resources_are_charged_before_state_and_retire_in_order(monkeypatch,validated):
    import torch
    from tensorfold.cuda import capacity
    from tensorfold.cuda.sm70 import foundation_headers
    from tensorfold.families.qwen4_exp.cuda import history_owner, execution_identity
    events=[];memory={'free':1<<30,'active':0,'reserved':0};created=[]
    artifact_owner=NS(record=lambda:{'profile_sha256':'b'*64})
    def load(value):
        assert value is validated and not created
        events.append('load');memory['free']-=64 # Independent driver/code residency, not Torch payload.
        return artifact_owner
    monkeypatch.setattr(artifact,'load_native',load)
    monkeypatch.setattr(foundation_headers,'selected_identity',lambda:{'source':'test'})
    monkeypatch.setattr(execution_identity,'_retire_store',lambda *a,**k:None)
    monkeypatch.delenv(capacity.LIMIT_ENV,raising=False)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda device:None)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(memory['free'],1<<30))
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda device:memory['active'])
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda device:memory['reserved'])
    class Role:
        def __init__(self,rows): self.rows=rows;self.closed=False;self.bytes=0
        def prepare_native(self,*,pair,page_tokens,buckets): self.native_buckets=execution.row_buckets(self.rows,buckets)
        def configure_native_family(self,family,heads,**kwargs):
            assert family==execution.FLASH_KERNEL_FAMILY and heads is None
            assert kwargs['flash_artifact'] is artifact_owner and kwargs['flash_panel_rows']==3
            self.bytes=kwargs['flash_tensor_budget_bytes']
            memory['active']+=self.bytes;memory['reserved']+=self.bytes;memory['free']-=self.bytes
        def configure_native_transport(self,*a): pass
        def allocation_record(self):return {'device_tensor_bytes':self.bytes}
        def warm_reference_transport(self):pass
        def clear_native_inputs(self,*a):pass
        def warm_native_transport(self,*a):pass
        def wait_native_inputs(self):pass
        def native_run(self,rows):events.append(('warm',self.rows,rows))
        def handoff_native_thread(self):events.append(('handoff',self.rows))
        def close(self):self.closed=True;events.append(('close',self.rows))
    def factory(text,*,rows,register,**kwargs):
        assert events[0]=='load'
        owner=Role(rows);register(owner);created.append(owner);return owner
    monkeypatch.setattr(history_owner.OwnerBuffers,'from_text',factory)
    def close_artifact():
        assert all(owner.closed for owner in created);events.append('artifact-close')
    artifact_owner.close=close_artifact
    owner=prep.OwnerPreparation(specification(),role_rows={'decode':4,'mtp':4,'prefill':8},manifest_sha256='c'*64)
    def phase(engine,name,operation,**kwargs):
        value=operation();return [value]*8
    owner._phase=phase
    engine=NS(comm=NS(),rank=0,tp=8,multi=None,_calls={},position_profile=object(),method_geometry=NS(method='mtp'))
    pair=NS(identity='q8',key=NS(bits=8,codec=0),value=NS(bits=8,codec=0))
    text=dict(num_attention_heads=24,num_key_value_heads=2,head_dim=256,indexer_budget=2048,indexer_compress_ratio=4)
    owner.prepare(engine,text,pair,page_tokens=128,device='cpu')
    assert owner.prepared and owner.requires_worker_handoff and len(created)==3
    assert owner.committed_bytes()==memory['active']+64
    assert owner.outside_active_bytes==64
    owner.handoff_to_current_thread()
    assert len([e for e in events if isinstance(e,tuple) and e[0]=='handoff'])==3
    owner.close();assert events[-1]=='artifact-close' and owner.flash_artifact is None


def test_role_retirement_failure_keeps_native_artifact_alive(validated):
    owner=prep.OwnerPreparation(specification(),role_rows={'decode':1,'prefill':1},manifest_sha256='c'*64)
    def fail():raise RuntimeError('uncertain role fence')
    owner.workspaces={'decode':NS(close=fail)}
    owner.flash_artifact=NS(close=lambda:pytest.fail('closed artifact before role retirement'))
    with pytest.raises(RuntimeError,match='uncertain'):owner.close()
    assert not owner.closed and owner.flash_artifact is not None

"""Archive credit geometry/real shared planner controls; no GPU/model loading."""
import builtins
import sys
from pathlib import Path
from types import ModuleType

import pytest

from gemma_cuda_target_metadata_test import inventory, tiny_config
from test_gemma_cuda_dflash_metadata import config as draft_config
from tensorfold.cuda import capacity as common
from tensorfold.families.gemma4.cuda import capacity, state
from tensorfold.families.gemma4.dflash_metadata import DraftConfig
from tensorfold.families.gemma4.sparkulator_metadata import SparkulatorConfig
from tensorfold.engine.prefix_blocks import PrefixBlocks


def valid_credit(extra=1):
    # Derive the existing empty-store metadata boundary, never duplicate its
    # private arithmetic or invent a different admission minimum.
    store = PrefixBlocks(100_000)
    minimum = store.stats['charged_bytes']
    store.close()
    return minimum+extra


@pytest.fixture
def scenario(monkeypatch, tmp_path):
    raw = tiny_config()
    runtime = ModuleType('torch')
    metadata = ModuleType('tensorfold.checkpoint_metadata')
    draft_dir = tmp_path/'draft'
    selected = {'spark': False, 'budget': None, 'host_budget': None}
    metadata.read_config = lambda path: ({'model_type': 'gemma4_assistant'} if selected['spark'] and Path(path)==draft_dir else raw)
    monkeypatch.setitem(sys.modules,'torch',runtime)
    monkeypatch.setitem(sys.modules,'tensorfold.checkpoint_metadata',metadata)
    monkeypatch.setattr(common,'headers',lambda path: {name:{'dtype':v[3],'shape':v[4]} for name,v in inventory(raw)[1].items()})
    monkeypatch.setattr(common,'unified',lambda torch: False)
    monkeypatch.setattr(common,'admit_host_commitment',lambda n: None)
    calls = []

    def boundary(directory, context, explicit, torch, geometry, rule, *, weight_estimator,
                 device_workspace_bytes, host_workspace_bytes):
        assert torch is runtime
        weights, host_staging = weight_estimator(directory)
        calls.append(dict(context=context,explicit=explicit,geometry=geometry,weights=weights,
                          device_workspace=device_workspace_bytes,host_workspace=host_workspace_bytes,
                          host_staging=host_staging))
        native = raw['text_config']['max_position_embeddings']
        if selected['host_budget'] is not None and max(host_staging,host_workspace_bytes)>selected['host_budget']:
            raise ValueError('selected host/cgroup budget cannot fit unchanged workspaces')
        if selected['budget'] is None:
            window = native if context is None else context
        else:
            # Existing shared admission algorithms, not a copied fitting rule.
            total = common.Geometry(lambda n:geometry.needed(n)+device_workspace_bytes,0,
                                    maximum_slots=geometry.maximum_slots)
            plan = common.make_plan(native,context,context is not None if explicit is None else explicit,
                                    selected['budget'],weights,total)
            window = common.choose(plan)
        return {'context_window':window,'geometry':geometry,'weights':weights}

    monkeypatch.setattr(common,'admit',boundary)
    return raw,selected,calls,tmp_path,draft_dir


@pytest.mark.parametrize('value',[-1,True,False,0.0,1.0,float('nan'),float('inf'),'1',sys.maxsize+1])
def test_invalid_credit_is_refused_before_torch_or_header_effects(monkeypatch,value):
    original = builtins.__import__
    def guarded(name,*args,**kwargs):
        if name == 'torch':
            raise AssertionError('invalid credit imported runtime')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(ValueError,match='prefix_cache_bytes'):
        capacity.admit(Path('/never-open'),None,None,prefix_cache_bytes=value)


@pytest.mark.parametrize('context',[64,128,256])
@pytest.mark.parametrize('streams,keep',[(1,1),(1,3),(2,3)])
def test_default_exact_legacy_archive_algebra_and_explicit_zero(scenario,context,streams,keep):
    raw,_,calls,path,_ = scenario
    cfg = inventory(raw)[0]
    default = capacity.admit(path,context,True,streams=streams,keep=keep)
    zero = capacity.admit(path,context,True,streams=streams,keep=keep,prefix_cache_bytes=0)
    for at in (64,128,256):
        legacy = keep*state.capacity_bytes(cfg,at,state.widest_forward(16))
        assert default['geometry'].needed(at)-zero['geometry'].needed(at)==legacy
    assert default['prefix_cache_bytes']==default['prefix_cache_default_bytes']==keep*state.capacity_bytes(cfg,context,1024)
    assert zero['prefix_cache_bytes']==0 and zero['prefix_cache_default_bytes']==default['prefix_cache_default_bytes']
    assert default['weights']==zero['weights']
    assert calls[-1]['device_workspace']==calls[-2]['device_workspace']
    assert calls[-1]['host_workspace']==calls[-2]['host_workspace']


def test_explicit_small_and_above_legacy_grants_count_once_without_context_clamp(scenario):
    raw,_,calls,path,_ = scenario
    default = capacity.admit(path,None,False,keep=3)
    credit = default['prefix_cache_default_bytes']*7+17
    zero = capacity.admit(path,None,False,keep=3,prefix_cache_bytes=0)
    large = capacity.admit(path,None,False,keep=3,prefix_cache_bytes=credit)
    small = capacity.admit(path,128,False,streams=2,keep=3,prefix_cache_bytes=valid_credit(7))
    small_zero = capacity.admit(path,128,False,streams=2,keep=3,prefix_cache_bytes=0)
    assert calls[0]['context'] is None and calls[0]['explicit'] is False
    assert calls[1]['context']==raw['text_config']['max_position_embeddings'] and calls[1]['explicit'] is True
    assert large['prefix_cache_bytes']==credit>large['prefix_cache_default_bytes']
    for at in (64,128,256):
        assert large['geometry'].needed(at)-zero['geometry'].needed(at)==credit
        assert small['geometry'].needed(at)-small_zero['geometry'].needed(at)==valid_credit(7)
    assert large['context_window']==256 and small['context_window']==128


def test_dflash_archive_replaces_all_target_and_companion_snapshot_credit(scenario,monkeypatch):
    raw,_,_,path,directory = scenario
    cfg = inventory(raw)[0]
    draft = DraftConfig.from_dict({**draft_config(),'num_target_layers':cfg.layers,'vocab_size':cfg.vocab,
                                  'dflash_config':{'target_layer_ids':[0,1],'mask_token_id':4}})
    from tensorfold.families.gemma4 import dflash_metadata
    headers = {n:{'dtype':'BF16','shape':s} for n,s in draft.shapes().items()}
    monkeypatch.setattr(dflash_metadata,'checked_headers',lambda path:(draft,headers))
    plain = capacity.admit(path,128,True,drafter_dir=directory,keep=3)
    zero = capacity.admit(path,128,True,drafter_dir=directory,keep=3,prefix_cache_bytes=0)
    for at in (64,128,256):
        companion = sum(state.companion_rings(draft.layer_types,draft.sliding_window,16,1024,at))*draft.kv_heads*draft.head_dim*4
        expected = 3*(state.capacity_bytes(cfg,at,1024)+companion)
        assert plain['geometry'].needed(at)-zero['geometry'].needed(at)==expected
    assert plain['prefix_cache_bytes']==plain['prefix_cache_default_bytes']
    assert zero['weights']==plain['weights']


def test_spark_live_is_independent_and_snapshot_archive_counts_once(scenario,monkeypatch):
    raw,selected,_,path,directory = scenario
    text = raw['text_config']
    text.update(hidden_size=2816,num_hidden_layers=30,vocab_size=262144,num_attention_heads=16,
                num_key_value_heads=8,num_global_key_value_heads=2,head_dim=256,global_head_dim=512,
                intermediate_size=2112,moe_intermediate_size=704,num_experts=128,top_k_experts=8,
                sliding_window=1024,max_position_embeddings=262144,
                layer_types=['sliding_attention']*5+['full_attention']+['sliding_attention']*5+['full_attention']
                            +['sliding_attention']*5+['full_attention']+['sliding_attention']*5+['full_attention']
                            +['sliding_attention']*5+['full_attention'])
    selected['spark'] = True
    spark = SparkulatorConfig()
    from tensorfold.families.gemma4 import sparkulator_metadata
    monkeypatch.setattr(sparkulator_metadata,'checked_headers',lambda path:(spark,{}))
    cfg = inventory(raw)[0]
    base = capacity.admit(path,32768,True,drafter_dir=directory,streams=2,keep=3)
    zero = capacity.admit(path,32768,True,drafter_dir=directory,streams=2,keep=3,prefix_cache_bytes=0)
    custom = capacity.admit(path,32768,True,drafter_dir=directory,streams=2,keep=3,prefix_cache_bytes=valid_credit(1234))
    for at in (8192,32768,262144):
        expected = 3*(state.capacity_bytes(cfg,at,1024)+spark.snapshot_bytes(at))
        assert base['geometry'].needed(at)-zero['geometry'].needed(at)==expected
        assert custom['geometry'].needed(at)-zero['geometry'].needed(at)==valid_credit(1234)
    assert base['prefix_cache_default_bytes']==3*(state.capacity_bytes(cfg,32768,1024)+spark.snapshot_bytes(32768))
    assert custom['prefix_cache_bytes']==valid_credit(1234) and custom['context_window']==32768
    assert base['weights']==zero['weights']==custom['weights']


@pytest.mark.parametrize("requested_context",[None,0])
def test_real_common_scalar_planner_rejects_unfit_credit_instead_of_shortening(scenario,requested_context):
    _,selected,calls,path,_ = scenario
    probe = capacity.admit(path,None,False,prefix_cache_bytes=valid_credit(1000))
    call = calls[-1]
    selected['budget'] = max(call['weights'].resident+call['weights'].staging,
                            call['weights'].resident+probe['geometry'].needed(128)+call['device_workspace'])
    with pytest.raises(ValueError,match='requested context 256'):
        capacity.admit(path,requested_context,False,prefix_cache_bytes=valid_credit(1000))
    assert calls[-1]['context']==256 and calls[-1]['explicit'] is True
    fitted = capacity.admit(path,128,False,prefix_cache_bytes=valid_credit(1000))
    assert fitted['context_window']==128
    auto = capacity.admit(path,None,False)
    assert 0<auto['context_window']<256
    assert calls[-1]['context'] is None and calls[-1]['explicit'] is False


def test_unchanged_host_budget_failure_propagates_before_plan_publication(scenario):
    _,selected,_,path,_ = scenario
    selected['host_budget'] = 0
    with pytest.raises(ValueError,match='host/cgroup budget'):
        capacity.admit(path,None,None,prefix_cache_bytes=0)


@pytest.mark.parametrize('which',['one','below','threshold'])
def test_unconstructible_positive_credit_is_refused_before_torch(monkeypatch,which):
    minimum = valid_credit()-1
    credit = {'one':1,'below':minimum-1,'threshold':minimum}[which]
    original = builtins.__import__
    def guarded(name,*args,**kwargs):
        if name == 'torch':
            raise AssertionError('unconstructible credit imported runtime')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(ValueError,match='empty index metadata'):
        capacity.admit(Path('/never-open'),None,None,prefix_cache_bytes=credit)


def test_smallest_constructible_credit_reaches_real_header_geometry(scenario):
    _,_,calls,path,_ = scenario
    credit = valid_credit()
    result = capacity.admit(path,128,True,prefix_cache_bytes=credit)
    assert result['prefix_cache_bytes']==credit and result['context_window']==128
    assert calls[-1]['context']==128 and calls[-1]['explicit'] is True


def test_none_and_disabled_zero_do_not_call_positive_store_validator(scenario,monkeypatch):
    _,_,_,path,_ = scenario
    import tensorfold.engine.prefix_blocks as blocks
    def refused(_):
        raise AssertionError('disabled/default changed to explicit positive validation')
    monkeypatch.setattr(blocks,'validate_prefix_budget',refused)
    assert capacity.admit(path,128,True,prefix_cache_bytes=0)['prefix_cache_bytes']==0
    assert capacity.admit(path,128,True)['prefix_cache_bytes']>0

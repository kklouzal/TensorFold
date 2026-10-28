import pytest
from tools.moe_screen_coverage import validate_case, validate_prepared_profile

KEYS={(rows,layer) for rows in (30,31,32) for layer in range(48)}


def delta():
    return {'router_blas.target.calls':0,'router_blas.mtp.calls':17,'gdn.batch_submitted_layers':36,
            'owner.prelude_submitted_batches':12,'owner.prelude_submitted_mtp_batches':4,
            'moe_regions.frames':3,'moe_regions.hits':144,
            'moe_regions.rows_30_frames':1,'moe_regions.rows_31_frames':1,'moe_regions.rows_32_frames':1}


def test_zero_eager_target_with_real_qualified_graph_cycles_is_valid():
    source=delta()
    result=validate_case(24,source,prepared_keys=KEYS)
    assert result['eager_target_wrapper_calls']==0 and result['captured_moe_replays']==144
    assert source['router_blas.target.calls']==0  # Never fabricate a wrapper increment.


@pytest.mark.parametrize('field',('router_blas.mtp.calls','gdn.batch_submitted_layers',
                                  'owner.prelude_submitted_mtp_batches'))
def test_graph_target_does_not_waive_independent_runtime_evidence(field):
    source=delta()
    source[field]=0
    with pytest.raises(ValueError):
        validate_case(24,source,prepared_keys=KEYS)


def test_partial_cycles_and_wrong_keyspace_are_not_valid_graph_coverage():
    source=delta()
    source['moe_regions.hits']=143
    with pytest.raises(ValueError,match='complete qualified'):
        validate_case(24,source,prepared_keys=KEYS)
    with pytest.raises(ValueError,match='144'):
        validate_case(24,delta(),prepared_keys=KEYS-{(32,47)})


def test_unsupported_rows_keep_independent_eager_authority_and_n1_baseline():
    source=delta()
    source.update({'router_blas.target.calls':48,'moe_regions.frames':0,'moe_regions.hits':0,
        'moe_regions.rows_30_frames':0,'moe_regions.rows_31_frames':0,'moe_regions.rows_32_frames':0})
    assert validate_case(8,source,prepared_keys=KEYS)['eager_target_wrapper_calls']==48
    n1={name:0 for name in ('moe_regions.frames','moe_regions.hits','moe_regions.rows_30_frames',
                              'moe_regions.rows_31_frames','moe_regions.rows_32_frames')}
    assert validate_case(1,n1,prepared_keys=KEYS)['captured_moe_replays']==0


def test_captured_fusion_uses_its_sealed_backend_not_python_wrapper_increments():
    profile='source-router-gdn-shared-fusion-trial'
    ready={'sm70_profile':profile,'native_flags':{'gguf_experts_implementation':'packed_blocks2w'},
           'shared_fusion':{'enabled':True,'banks':49,'public_count':513,'slots':11,
                           'target_output':'float16','mtp_output':'float32'},
           'capacity_plan':{'multi_graphs':{'moe_regions':{'numeric_profile':profile,
               'graph_contract':'pure-moe-target-exactR-ordered-pools-v1'}}}}
    assert validate_prepared_profile(ready)=='packed_blocks2w'
    assert validate_case(24,delta(),prepared_keys=KEYS)['eager_target_wrapper_calls']==0
    ready['native_flags']['gguf_experts_implementation']='packed_exact'
    with pytest.raises(ValueError,match='Fusion backend'):
        validate_prepared_profile(ready)


def test_candidate_eager_wrappers_not_fabricated_and_n1_bypass_allowed():
    from tools.moe_screen_coverage import validate_candidate_case
    source={f'candidate_regions.{role}.rows_{r}':0 for role in ('target','mtp') for r in range(1,33)}
    assert validate_candidate_case(1,source)['replays']=={'target':0,'mtp':0}
    with pytest.raises(ValueError,match='replay'):
        validate_candidate_case(24,source)
    source['candidate_regions.target.rows_32']=3
    source['candidate_regions.mtp.rows_14']=7
    assert validate_candidate_case(24,source)['replays']=={'target':3,'mtp':7}
    del source['candidate_regions.target.rows_1']
    with pytest.raises(ValueError,match='unavailable'):
        validate_candidate_case(24,source)


@pytest.mark.parametrize('profile', ['source-router-gdn-shared-fusion-trial', 'source-batch-router-blas-trial'])
def test_hc_boundary_requires_its_own_seal_not_old_moe_credit(profile):
    ready={'sm70_profile':profile,'native_flags':{'gguf_experts_implementation':'packed_blocks2w'},
           'shared_fusion':{'enabled':True,'banks':49,'public_count':513,'slots':11,
                           'target_output':'float16','mtp_output':'float32'},
           'capacity_plan':{'multi_graphs':{'moe_regions':{'numeric_profile':profile,
               'graph_contract':'pure-moe-target-exactR-ordered-pools-v1'}}}}
    with pytest.raises(ValueError,match='HC/MoE'):
        validate_prepared_profile(ready,hc=True)
    graph=ready['capacity_plan']['multi_graphs']['moe_regions']
    graph['graph_contract']='mlp-hc-mode3-moe-target-exactR-ordered-pools-v1'
    with pytest.raises(ValueError,match='HC/MoE'):
        validate_prepared_profile(ready,hc=True)
    graph['funding']={'boundary':'MLP-HC-mode3+pure-MoE'}
    assert validate_prepared_profile(ready,hc=True)==('packed_blocks2w' if profile.endswith('shared-fusion-trial') else 'packed_exact')


def test_hot_row_policy_is_hash_bound_preserves_n1_and_drives_full_cycle_counter_validation(tmp_path):
    import hashlib
    import json
    from tools.moe_screen_coverage import held_hot_rows
    from tensorfold.families.qwen4_exp.cuda.moe_region_catalogue import HOT_ROWS
    path = tmp_path/'source-hold.json'
    policy = {'schema':'tensorfold.hc-moe-hot16-HOLD-v1','rows':list(HOT_ROWS)}
    path.write_text(json.dumps(policy))
    rows = held_hot_rows(path,hashlib.sha256(path.read_bytes()).hexdigest())
    assert rows==HOT_ROWS and min(rows)>3 and max(rows)==32
    with pytest.raises(ValueError,match='hash'):
        held_hot_rows(path,'0'*64)
    source=delta()
    for r in rows:
        source.setdefault(f'moe_regions.rows_{r}_frames',0)
    source['moe_regions.rows_5_frames']=2
    source['moe_regions.frames']=5
    source['moe_regions.hits']=240
    keys={(r,i) for r in rows for i in range(48)}
    result=validate_case(8,source,prepared_keys=keys,rows=rows)
    assert result['completed_captured_forwards']==5 and result['captured_moe_replays']==240
    del source['moe_regions.rows_5_frames']
    with pytest.raises(ValueError,match='unavailable'):
        validate_case(8,source,prepared_keys=keys,rows=rows)
    policy['rows'][0]=3
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError,match='N1'):
        held_hot_rows(path,hashlib.sha256(path.read_bytes()).hexdigest())


def test_hot_shared_profile_requires_exact_owned_rowset_and_pool_policy():
    from tensorfold.families.qwen4_exp.cuda.moe_region_catalogue import HOT_ROWS
    profile='source-router-gdn-shared-fusion-trial'
    graph={'numeric_profile':profile,'rows':list(HOT_ROWS),'graphs':768,
           'graph_contract':'mlp-hc-mode3-moe-hot16-independent-pool-v1',
           'funding':{'rows':list(HOT_ROWS),'maximum_graphs':768,'boundary':'MLP-HC-mode3+pure-MoE',
                      'pool_policy':'one-independent-no-private-output-pool'}}
    ready={'sm70_profile':profile,'native_flags':{'gguf_experts_implementation':'packed_blocks2w'},
           'shared_fusion':{'enabled':True,'banks':49,'public_count':513,'slots':11,
                           'target_output':'float16','mtp_output':'float32'},
           'capacity_plan':{'multi_graphs':{'moe_regions':graph}}}
    assert validate_prepared_profile(ready,hc=True,shared=True,rows=HOT_ROWS)=='packed_blocks2w'
    graph['funding']['pool_policy']='ordered-pool-per-exact-R'
    with pytest.raises(ValueError,match='policy'):
        validate_prepared_profile(ready,hc=True,shared=True,rows=HOT_ROWS)

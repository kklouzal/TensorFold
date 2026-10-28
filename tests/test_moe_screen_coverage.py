import pytest
from tools.moe_screen_coverage import validate_case

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

"""Every C8 service-reader predicate runs against retained actual Source40 HTTP fields."""
import copy
import json
from pathlib import Path

import pytest

from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
from tools import v100_target_window_teacher as window


def actual():
    data=json.loads((Path(__file__).parent/'fixtures/c8-source40-actual-http-stats.json').read_text())
    data['instance']=data['stats']['server']['instance_id']
    profile=data['profile'];data['prepared']={'manifest_canonical_sha256':digest(profile),
        'manifest_artifact_files':len(profile['artifacts']),
        'manifest_artifact_bytes':sum(row['size']for row in profile['artifacts'].values())}
    return data


def validate(data):
    return window.validate_service(data['stats'],data['client'],data['instance'],
                                   prepared_manifest=data['prepared'])


def test_full_actual_http_and_fresh_registration_contract_with_uncaptured_owner():
    data=actual();families=data['stats']['runtime']['execution']['execution_families']
    assert families['effective']['graph_family']=='uncaptured' and families['graph_rows']==0
    # These are separate actually captured MultiDecoder step graphs, not the
    # absent OwnerPreparation bucket graph family reported by graph_rows.
    plan=data['stats']['capacity_plan']
    assert plan['execution_envelope']['target_rows']==120
    assert len(plan['multi_graphs']['exact_graph_catalogue']['captured_keys'])==36
    assert plan['multi_graphs']['captured']is True
    assert validate(data)['resident_capacity']==24


@pytest.mark.parametrize('fault',['owner_rows120','owner_rowsbool','unprepared','wrong_owner','trial',
    'step_rows','missing_key','capture_failed','instance','model','context','slots',
    'manifest','files','bytes','passes','passesbool','reuse','reusebool','shared'])
def test_actual_http_contract_keeps_graph_capacity_identity_and_registration_guards(fault):
    data=copy.deepcopy(actual());stats=data['stats'];families=stats['runtime']['execution']['execution_families']
    binding=stats['runtime']['capacity_plan']['execution_profile_binding'];plan=stats['capacity_plan']
    if fault=='owner_rows120':families['graph_rows']=120
    elif fault=='owner_rowsbool':families['graph_rows']=False
    elif fault=='unprepared':families['native_prepared']=False
    elif fault=='wrong_owner':families['effective']['graph_family']='bucketed'
    elif fault=='trial':families['trial']['kernel_family']='other'
    elif fault=='step_rows':plan['execution_envelope']['target_rows']=119
    elif fault=='missing_key':plan['multi_graphs']['exact_graph_catalogue']['captured_keys'].pop()
    elif fault=='capture_failed':plan['multi_graphs']['captured']=False
    elif fault=='instance':stats['server']['instance_id']='different-instance'
    elif fault=='model':stats['model']['id']='other'
    elif fault=='context':plan['session_capacity']['context_tokens']-=1
    elif fault=='slots':plan['session_capacity']['resident_capacity']=8
    elif fault=='manifest':binding['manifest_sha256']='0'*64
    elif fault=='files':binding['verified_input_files']-=1
    elif fault=='bytes':binding['verified_input_bytes']-=1
    elif fault=='passes':binding['content_passes']=0
    elif fault=='passesbool':binding['content_passes']=True
    elif fault=='reuse':binding['hashes_reused_this_binding']=1
    elif fault=='reusebool':binding['hashes_reused_this_binding']=False
    else:binding['shared_immutable_same_host']=False
    with pytest.raises(ValueError):validate(data)

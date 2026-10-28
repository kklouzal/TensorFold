"""Tail runtime and inherited row authority are separate, hash-bound protocols."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest
from tools.descriptor_row_policy import validate


@pytest.fixture
def package(tmp_path):
    source=Path(os.environ['TENSORFOLD_TAIL_POLICY_PACKAGE'])
    for name in ('candidate.json','source-hold.json','row-policy.json','wheel-provenance.json',
                 'row-policy-wheel-provenance.json','descriptor-parent-source-hold.json',
                 'descriptor-parent-wheel-provenance.json','tail-structural-proof.json'):
        shutil.copy(source/name,tmp_path/name)
    return tmp_path


def test_exact_tail_only_runtime_delta_keeps_original_row_authority(package):
    result=validate(package)
    assert result['current_HOLD_sha256']=='dc400f0548828db6410592240c8c9f7dde8c30efead0bd91fe39dd0942ba5653'
    assert result['row_policy_sha256']=='8021154e213e1383b2a26290d14cb4d5318dfb0c80eee7cb003e3f9fc5e2979d'
    assert len(result['rows'])==16 and min(result['rows'])>3


@pytest.mark.parametrize('path',['/moe_region_catalogue.py','/engine.py','/multi_graphs.py','/history_owner.py'])
def test_tail_cannot_hide_any_other_runtime_change(package,path):
    target=package/'wheel-provenance.json'
    value=json.loads(target.read_text())
    next(row for row in value['source_files'] if row['path'].endswith(path))['sha256']='0'*64
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='compact-indexer-only'):
        validate(package)


def test_tail_method_only_proof_is_required(package):
    path=package/'tail-structural-proof.json'
    value=json.loads(path.read_text())
    value['graph_funding_rows_methods_unchanged']=False
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='structural proof'):
        validate(package)


def test_wrong_descriptor_parent_is_rejected_before_row_use(package):
    path=package/'descriptor-parent-source-hold.json'
    value=json.loads(path.read_text())
    value['commit']='0'*40
    path.write_text(json.dumps(value))
    candidate=package/'candidate.json'
    c=json.loads(candidate.read_text())
    c['descriptor_parent_hold_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    candidate.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='exact descriptor parent'):
        validate(package)

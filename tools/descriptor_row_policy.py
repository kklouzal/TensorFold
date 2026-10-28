"""Bind descriptor APP authority separately from its inherited HC row policy."""
import hashlib
import json
from pathlib import Path


def validate(package):
    package = Path(package)
    def load(name):
        return json.loads((package / name).read_text())

    def sha(name):
        return hashlib.sha256((package / name).read_bytes()).hexdigest()

    candidate = load('candidate.json')
    if candidate['row_policy_file'] != 'row-policy.json':
        raise ValueError('row policy must be the explicit package document')
    if (sha('source-hold.json') != candidate['source_hold_sha256']
            or sha('row-policy.json') != candidate['row_policy_sha256']
            or sha('row-policy-wheel-provenance.json') != candidate['row_policy_provenance_sha256']):
        raise ValueError('current APP or inherited row authority hash differs')
    current, parent = load('source-hold.json'), load('row-policy.json')
    if (current['schema'] != 'tensorfold.descriptor-host-words-HOLD-v1'
            or parent['schema'] != 'tensorfold.hc-moe-hot16-HOLD-v1'
            or current['inherited_HOLD']['sha256'] != candidate['row_policy_sha256']
            or current['APP'] != candidate['wheel_sha256']
            or any(current[key] != parent[key] for key in ('rows', 'funding', 'pool_contract'))):
        raise ValueError('descriptor successor does not inherit this exact row policy')
    current_provenance = load('wheel-provenance.json')
    parent_provenance = load('row-policy-wheel-provenance.json')
    if (current_provenance['wheel']['sha256'] != candidate['wheel_sha256']
            or parent_provenance['wheel']['sha256'] != parent['APP']
            or current_provenance['compiled_extensions'] != parent_provenance['compiled_extensions']):
        raise ValueError('application or native extension provenance differs')
    old = {row['path']: row for row in parent_provenance['source_files']}
    new = {row['path']: row for row in current_provenance['source_files']}
    if (old.keys() != new.keys() or len(new) != 727
            or [name for name in old if old[name] != new[name]]
            != ['tensorfold/families/qwen4_exp/cuda/history_owner.py']):
        raise ValueError('inherited graph policy requires exact history-owner-only APP delta')
    rows = current['rows']
    if (type(rows) is not list or len(rows) != 16 or any(type(row) is not int or not 4 <= row <= 32 for row in rows)
            or rows != sorted(set(rows))):
        raise ValueError('inherited hot16 rows must preserve N1 and logical32')
    return {'current_HOLD_sha256': candidate['source_hold_sha256'],
            'row_policy_file': candidate['row_policy_file'],
            'row_policy_sha256': candidate['row_policy_sha256'], 'rows': rows,
            'scope': 'row authority only; current installed APP remains independently verified'}

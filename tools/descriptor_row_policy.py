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
    tail = current.get('schema') == 'tensorfold.compact-tail-slices-HOLD-v1'
    descriptor = current
    if tail:
        if (sha('descriptor-parent-source-hold.json') != candidate['descriptor_parent_hold_sha256']
                or sha('descriptor-parent-wheel-provenance.json') != candidate['descriptor_parent_provenance_sha256']):
            raise ValueError('descriptor parent authority differs')
        descriptor = load('descriptor-parent-source-hold.json')
        if current['base'] != descriptor['commit']:
            raise ValueError('Tail runtime does not descend from this exact descriptor parent')
        proof = load('tail-structural-proof.json')
        if (sha('tail-structural-proof.json') != current['evidence']['structural-proof.json']['sha256']
                or not proof.get('all_other_module_ASTexact') or not proof.get('indexedfallback_ASTexact')
                or not proof.get('graph_funding_rows_methods_unchanged')):
            raise ValueError('Tail method-only structural proof is unavailable')
    if (descriptor['schema'] != 'tensorfold.descriptor-host-words-HOLD-v1'
            or parent['schema'] != 'tensorfold.hc-moe-hot16-HOLD-v1'
            or descriptor['inherited_HOLD']['sha256'] != candidate['row_policy_sha256']
            or current['APP'] != candidate['wheel_sha256']
            or any(descriptor[key] != parent[key] for key in ('rows', 'funding', 'pool_contract'))):
        raise ValueError('runtime successor does not inherit this exact row policy')
    current_provenance = load('wheel-provenance.json')
    parent_provenance = load('row-policy-wheel-provenance.json')
    if (current_provenance['wheel']['sha256'] != candidate['wheel_sha256']
            or parent_provenance['wheel']['sha256'] != parent['APP']
            or current_provenance['compiled_extensions'] != parent_provenance['compiled_extensions']):
        raise ValueError('application or native extension provenance differs')
    old = {row['path']: row for row in parent_provenance['source_files']}
    new = {row['path']: row for row in current_provenance['source_files']}
    if old.keys() != new.keys() or len(new) != 727:
        raise ValueError('inherited graph policy requires the same727-file package')
    descriptor_rows = new
    if tail:
        descriptor_provenance = load('descriptor-parent-wheel-provenance.json')
        descriptor_rows = {row['path']: row for row in descriptor_provenance['source_files']}
        if (descriptor_provenance['wheel']['sha256'] != descriptor['APP']
                or descriptor_provenance['compiled_extensions'] != current_provenance['compiled_extensions']
                or descriptor_rows.keys() != new.keys()
                or [name for name in descriptor_rows if descriptor_rows[name] != new[name]]
                != ['tensorfold/families/qwen4_exp/cuda/compact_indexer.py']):
            raise ValueError('Tail runtime requires exact compact-indexer-only APP delta')
    if [name for name in old if old[name] != descriptor_rows[name]] != ['tensorfold/families/qwen4_exp/cuda/history_owner.py']:
        raise ValueError('inherited graph policy requires exact history-owner-only descriptor delta')
    rows = parent['rows']
    if (type(rows) is not list or len(rows) != 16 or any(type(row) is not int or not 4 <= row <= 32 for row in rows)
            or rows != sorted(set(rows))):
        raise ValueError('inherited hot16 rows must preserve N1 and logical32')
    return {'current_HOLD_sha256': candidate['source_hold_sha256'],
            'row_policy_file': candidate['row_policy_file'],
            'row_policy_sha256': candidate['row_policy_sha256'], 'rows': rows,
            'scope': 'row authority only; current installed APP remains independently verified'}

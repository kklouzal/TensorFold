"""Typed numerical trials on the existing full-vocabulary NONE/NUMA teacher.

This version changes only three explicit preparation axes. It does not qualify
MTP, graphs, C8 batching, or free generation. Historical expert21 DATA stays strict.
"""
from __future__ import annotations

from pathlib import Path

if __package__:
    from . import v100_prepared_teacher_numa as numa
else:
    import v100_prepared_teacher_numa as numa

base, legacy = numa.base, numa.legacy
require = legacy.require
SCHEMA = 'tensorfold.v100-numerical-teacher-operation-v3'
MODE = 'owned-numerical-teacher-model-v3'
HOST_RESULT_SCHEMA = 'tensorfold.owned-numerical-teacher-operation-result-v3'
RESULT_SCHEMA = 'tensorfold.numerical-teacher-model-result-v3'
CLIENT_SCHEMA = 'tensorfold.numerical-teacher-client-v3'
CLIENT_REPORT_SCHEMA = 'tensorfold.numerical-teacher-client-report-v3'
SCOPE, SERVICE = numa.SCOPE, numa.SERVICE
FILES, RECEIPT_DIR, PLACEMENT_MEMBER = numa.FILES, numa.RECEIPT_DIR, numa.PLACEMENT_MEMBER
REQUIRED_PREFIXES, MODEL_OBSERVER_LIMITS = numa.REQUIRED_PREFIXES, numa.MODEL_OBSERVER_LIMITS
identity, absolute, selectors = legacy.identity, legacy.absolute, base.selectors
validate_teacher_completion = legacy.validate_teacher_completion
validate_numa, load_inputs, frozen_host_inputs = numa.validate_numa, numa.load_inputs, numa.frozen_host_inputs
publish_inputs, collect, evidence = numa.publish_inputs, numa.collect, numa.evidence
validate_reference = numa.validate_reference
AXES = {'native_family': 'owned-native-family-v1', 'prepared_linears': 'owned-prepared-m32-v1',
        'owned_q8_window_prepare': True}


def validate_trial(value):
    require(type(value) is dict and set(value) == set(AXES)
            and all(type(v) is bool for v in value.values()), 'three explicit boolean numerical axes required')
    return value


def preparation_projection(value, trial):
    validate_trial(trial)
    require(type(value) is dict, 'typed numerical preparation required')
    expected = {k: v for k, v in AXES.items() if trial[k]}
    require({k: v for k, v in value.items() if k in AXES} == expected,
            'numerical axes differ from their exact preparation selection')
    projected = {k: v for k, v in value.items() if k not in AXES}
    base.validate_preparation(projected)
    return projected


def operation_projection(value):
    require(type(value) is dict and 'numerical_trial' in value, 'versioned numerical trial required')
    return {**{k: v for k, v in value.items() if k != 'numerical_trial'},
            'prepared_execution': preparation_projection(value['prepared_execution'], value['numerical_trial'])}


def client_projection(value):
    return {**{k: v for k, v in value.items() if k != 'numerical_trial'}, 'schema': numa.CLIENT_SCHEMA,
            'prepared_execution': preparation_projection(value['prepared_execution'], value['numerical_trial'])}


def validate_operation(value):
    numa.validate_operation(operation_projection(value))
    return value


def validate_teacher_client(value):
    require(type(value) is dict and value.get('schema') == CLIENT_SCHEMA and 'numerical_trial' in value,
            'closed versioned numerical client required')
    numa.validate_teacher_client(client_projection(value))
    return value


def validate_sources(value):
    numa.validate_sources(value)
    require('tools/v100_numerical_teacher.py' in value, 'numerical contract must join frozen source closure')
    return value


def validate_teacher_operation_plan(operation, plan, client, **kwargs):
    validate_operation(operation)
    validate_teacher_client(client)
    require(client['numerical_trial'] == operation['numerical_trial'], 'client numerical trial differs')
    profile = kwargs['profile']
    require(type(profile) is dict and profile.get('preparation') == operation['prepared_execution'],
            'actual numerical manifest differs from operation')
    if plan.get('mode') == 'capture':
        require(not any(operation['numerical_trial'].values()), 'only numerical control captures the teacher prefix')
    return numa.validate_teacher_operation_plan(operation_projection(operation), plan, client_projection(client),
        **{**kwargs, 'profile': {**profile, 'preparation': operation_projection(operation)['prepared_execution']}})


def operation_assets(value):
    validate_operation(value)
    return [*legacy.operation_assets(value), value['case_authority'], value['execution_manifest'],
            *[value['numa'][key] for key in FILES]]


def qualifier_argv(operation, client, uuids, output):
    validate_operation(operation)
    validate_teacher_client(client)
    return numa.qualifier_argv(operation_projection(operation), client_projection(client), uuids, output)


def validate_host_operation(host, inner, *, numa_bundle=None):
    validate_operation(inner['operation'])
    validate_sources(inner['source_files'])
    numa.validate_host_operation(host, {**inner, 'schema': numa.SCHEMA, 'mode': numa.MODE,
        'operation': operation_projection(inner['operation'])}, numa_bundle=numa_bundle)
    return inner


def model_projection(value):
    require(type(value) is dict and value.get('schema') == RESULT_SCHEMA, 'versioned numerical model result required')
    return {**value, 'schema': numa.RESULT_SCHEMA}


def validate_model_result(value):
    numa.validate_model_result(model_projection(value))
    return value


def reference_result_files(model):
    return numa.reference_result_files(model_projection(model))


def validate_result_files(root, model, operation):
    validate_operation(operation)
    return numa.validate_result_files(root, model_projection(model), operation)


def validate_host_result(value, expected):
    result = legacy._validate_host_result(value, expected, schema=HOST_RESULT_SCHEMA, mode=MODE,
                                          result_validator=validate_model_result)
    require(result['teacher_plan_sha256'] == expected['operation']['teacher_plan']['sha256']
            and result['teacher_client_sha256'] == expected['operation']['teacher_client']['sha256'],
            'numerical result DATA binding differs')
    return result


def validate_completed_teacher_host(value, *, authority_sha256, authority_bytes, image_id=None):
    require(type(value) is dict and type(value.get('inner')) is dict, 'closed numerical Host record required')
    inner = value['inner']
    validate_operation(inner['operation'])
    validate_host_result(inner, {'operation': inner['operation'], 'source_commit': inner['source_commit'],
                                'image_id': inner['image_id_declared']})
    projected = {**inner, 'schema': numa.HOST_RESULT_SCHEMA, 'mode': numa.MODE,
                 'operation': operation_projection(inner['operation']),
                 'model_result': model_projection(inner['model_result'])}
    numa.validate_completed_teacher_host({**value, 'inner': projected}, authority_sha256=authority_sha256,
                                        authority_bytes=authority_bytes, image_id=image_id)
    return inner


def validate_startup(row, selected):
    trial = validate_trial(selected['numerical_trial'])
    require(row.get('teacher_profile', {}).get('teacher_semantics', {}).get('prepared_execution') ==
            selected['prepared_execution'], 'all-rank actual numerical preparation differs')
    profile = row['teacher_profile']
    registration = profile.get('teacher_execution_registration', {})
    validate_numerics(registration.get('numerics'), trial)
    _normalized_retention(row, selected)
    projected = {**row, 'teacher_profile': {**profile, 'teacher_semantics': {
        **profile['teacher_semantics'], 'prepared_execution': preparation_projection(selected['prepared_execution'], trial)}}}
    base.validate_startup(projected, operation_projection(selected))
    for axis, receipt in (('native_family', 'native_family'), ('prepared_linears', 'prepared_linears')):
        require((receipt in row) == trial[axis], 'numerical capability receipt missing or unrequested')
        if trial[axis]:
            require(row[receipt].get('requested') == AXES[axis], 'actual numerical capability policy differs')
    q8 = row.get('owned_q8_preparation', {})
    require(bool(q8.get('window_roles')) == trial['owned_q8_window_prepare'], 'Q8 window role receipt differs')
    if trial['owned_q8_window_prepare']:
        require(set(q8['window_roles']) == {'decode'}, 'NONE teacher must retain only target Q8 window role')
    return row


def expected_numerics(trial):
    validate_trial(trial)
    result = {}
    if trial['native_family']:
        from tensorfold.cuda.sm70.owned_family import numerical_identity
        result['native_family'] = numerical_identity()
    if trial['prepared_linears']:
        from tensorfold.families.qwen4_exp.cuda.padded_linears import owned_numerical_identity
        result['prepared_linears'] = owned_numerical_identity()
    if trial['owned_q8_window_prepare']:
        result['owned_q8_window_preparation'] = {
            'policy': 'owned-q8-window-preparation-v1', 'roles': ['decode'],
            'codec': 'original-floor-q8', 'native_rows_per_dispatch': 1,
            'native_dispatches_per_window': 'rows', 'maximum_window_rows': 256,
            'compact_indexer': 'reference-window-transaction', 'qualified': False}
    return result or None


def validate_numerics(value, trial):
    expected = expected_numerics(trial)
    require(type(value) is dict and ('owner_preparation_numerics' in value) == (expected is not None)
            and value.get('owner_preparation_numerics') == expected,
            'actual numerical registration differs from runtime-owned absent/present identity')
    return value


def validate_engine(engine, operation):
    """Use genuine retained-capability validators before installing teacher hooks."""
    if __package__:
        from . import v100_owned_qualification as owner
    else:
        import v100_owned_qualification as owner
    trial = validate_trial(operation['numerical_trial'])
    family, linears, q8 = owner._native_family_receipt(engine), owner._prepared_m32_receipt(engine), owner._owned_q8_preparation_receipt(engine)
    require((family is not None) == trial['native_family'] and (linears is not None) == trial['prepared_linears']
            and bool((q8 or {}).get('window_roles')) == trial['owned_q8_window_prepare'],
            'actual retained native/prepared/Q8 capabilities differ from numerical trial')


def validate_replay_pair(control, candidate):
    validate_operation(control)
    validate_operation(candidate)
    require(not any(control['numerical_trial'].values()) and control['selector'] == selectors()[0],
            'numerical replay requires original all-axis-off expert incumbent control')
    for key in ('model_assets', 'model_entrypoint', 'frontend_assets', 'frontend_root', 'case_authority', 'teacher_scope', 'numa'):
        require(control[key] == candidate[key], 'numerical replay changed unrelated input: ' + key)
    require(operation_projection(control)['prepared_execution'] == operation_projection(candidate)['prepared_execution'],
            'numerical replay changed unrelated preparation')


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('spec', 'out', 'operation-gate', 'review'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--spec-sha256', required=True)
    parser.add_argument('--spec-bytes', type=int, required=True)
    args = parser.parse_args(argv)
    base.complete_cli(args.spec, {'sha256': args.spec_sha256, 'bytes': args.spec_bytes}, args.out.absolute(),
                      args.operation_gate, args.review, contract=__import__(__name__, fromlist=['']))
    return 0


def _manifest(side, control):
    quality = base._quality()
    requested = side['inner']['operation']['execution_manifest']
    inputs = side['host']['source_after']['inputs']
    matches = [r for r in inputs if Path(r['path']).name == Path(requested['path']).name
               and all(r[k] == requested[k] for k in ('bytes', 'sha256'))]
    require(len(matches) == 1, 'actual numerical manifest lacks its unique frozen Host input')
    raw = quality._owned_teacher_read(matches[0]['path'], requested['bytes'], control)
    require(len(raw) == requested['bytes'] and quality.hashlib.sha256(raw).hexdigest() == requested['sha256'],
            'numerical manifest raw bytes differ')
    profile = quality._owned_teacher_object(raw)
    require(profile.get('preparation') == side['inner']['operation']['prepared_execution'],
            'closed numerical manifest actual preparation differs')
    return {**profile, 'preparation': preparation_projection(profile['preparation'],
            side['inner']['operation']['numerical_trial'])}


def _normalized_retention(row, operation):
    from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
    raw = row['teacher_profile'].get('numerical_trial_retention')
    registered = row['teacher_profile']['teacher_execution_registration']['binding']['identity']
    require(type(raw) is dict and registered['retention'] == digest(raw),
            'numerical retention raw input does not bind its issued fingerprint')
    owner = raw.get('owner_preparation')
    require(type(owner) is dict and owner.get('specification') == {
        k: v for k, v in operation['prepared_execution'].items() if k != 'trial'},
        'actual retained preparation identity differs from explicit numerical selection')
    normalized = {**owner, 'specification': {k: v for k, v in owner['specification'].items() if k not in AXES}}
    # These are actual paid capability/resource records. They are retained in
    # output, not asserted equal across policies with different workspace/code.
    for key in ('native_family', 'prepared_linears', 'allocations'):
        normalized.pop(key, None)
    return {**raw, 'owner_preparation': normalized}


def compare(reference, candidate, *, control):
    """Consume both complete owners and reuse the full-vocabulary F64 reader."""
    quality = base._quality()
    import sys
    contract = sys.modules[__name__]
    left = quality._prepared_expert_side(reference, 'capture', control, contract)
    right = quality._prepared_expert_side(candidate, 'replay', control, contract)
    old, new = left['inner']['operation'], right['inner']['operation']
    validate_replay_pair(old, new)
    require(left['inner']['image_id_declared'] == right['inner']['image_id_declared'], 'same immutable image required')
    for key in ('source_manifest', 'source_fixture', 'source', 'image_id'):
        require(left['host']['source_after'].get(key) == right['host']['source_after'].get(key)
                and left['host']['source_after'].get(key) is not None, 'canonical source/image changed')
    require(_manifest(left, control) == _manifest(right, control), 'numerical trial changed unrelated manifest inputs')
    a, b = left['authority'], right['authority']
    expected = {'teacher/teacher-authority.json': {k: left['authority_ref'][k] for k in ('bytes', 'sha256')}}
    for group in ('startup_receipts', 'rank_receipts'):
        expected.update({'teacher/' + r['file']: {k: r[k] for k in ('bytes', 'sha256')} for r in a[group]})
    expected.update({'teacher/' + r['vector']['file']: {k: r['vector'][k] for k in ('bytes', 'sha256')} for r in a['frames']})
    expected.update(reference_result_files(left['model']))
    require(new['reference']['files'] == expected and b['reference_sha256'] == left['authority_ref']['sha256']
            and all(new['reference']['host_result'][k] == reference['host_result'][k] for k in ('bytes', 'sha256'))
            and all(new['reference']['completion'][k] == reference['completion'][k] for k in ('bytes', 'sha256')),
            'numerical trial changed its exact positively closed reference/CLI/NUMA/vector closure')
    for key in ('input_ids', 'vocab', 'widths', 'teacher_ids', 'runtime_sources'):
        require(a[key] == b[key], 'numerical trial changed frozen tokens or runtime sources')
    for key in ('case_id', 'model_sha256', 'frontend_sha256'):
        require(a['declared_identity'][key] == b['declared_identity'][key], 'numerical case/model/frontend differs')
    for side, operation in ((left, old), (right, new)):
        require(side['authority']['declared_identity']['profile_sha256'] == operation['execution_manifest']['sha256'],
                'numerical plan must name its own actual profile; no identity aliases')
    def semantic(side, operation):
        selected = operation['selector']
        actual = quality._candidate_semantics(side, None if selected['gu'] == 'incumbent' else selected['gu'],
            selected['down'], selected['iq2_source'], selected['prefill_method'], teacher_scope=SCOPE)
        require(actual['prepared_execution'] == operation['prepared_execution'], 'actual numerical State preparation differs')
        return {**actual, 'prepared_execution': preparation_projection(actual['prepared_execution'], operation['numerical_trial'])}
    require(semantic(left, old) == semantic(right, new), 'numerical trial changed unrelated State/RN/row semantics')
    for rank in range(8):
        require(_normalized_retention(left['startups'][rank], old) == _normalized_retention(right['startups'][rank], new),
                'numerical trial changed unrelated retention policy')
        lr, rr = left['registrations'][rank], right['registrations'][rank]
        for registration, operation, startup in ((lr, old, left['startups'][rank]), (rr, new, right['startups'][rank])):
            nums = registration['numerics']
            validate_numerics(nums, operation['numerical_trial'])
            require(nums.get('owned_expert_native') == startup['plan_identity'],
                    'actual numerical owner/Plan identity differs')
        def numbers(registration, operation):
            result = dict(registration['numerics'])
            result.pop('owner_preparation_numerics', None)
            s = operation['selector']
            result['owned_expert_native'] = quality._candidate_plan_identity(result['owned_expert_native'],
                None if s['gu'] == 'incumbent' else s['gu'], s['down'], s['iq2_source'], s['prefill_method'], teacher_scope=SCOPE)
            return result
        require(numbers(lr, old) == numbers(rr, new), 'numerical trial changed unrelated arithmetic identity')
        for key, value in lr['binding']['identity'].items():
            if key not in ('numerics', 'kernel_build', 'retention'):
                require(rr['binding']['identity'].get(key) == value, 'numerical trial changed unrelated execution identity: ' + key)
        require({k: v for k, v in lr['contents'].items() if k.startswith('source/')} ==
                {k: v for k, v in rr['contents'].items() if k.startswith('source/')}, 'installed source content changed')
        for side, operation in ((left, old), (right, new)):
            startup, mapped = side['startups'][rank], side['native'][rank].values()
            if operation['numerical_trial']['native_family']:
                module = startup['native_family']['actual'].get('module')
                require(type(module) is str and module.startswith('tf_sm70_family_')
                        and any(Path(item['path']).name == module + '.so' for item in mapped),
                        'retained native family has no matching executable VMA provider')
            if operation['numerical_trial']['owned_q8_window_prepare']:
                provider = startup['owned_q8_preparation'].get('native_module', {})
                require(any(item['path'] == provider.get('path') for item in mapped),
                        'retained Q8 window has no matching executable VMA provider')
        old_native, new_native = left['native'][rank], right['native'][rank]
        require(all(name in new_native and new_native[name]['sha256'] == item['sha256'] for name, item in old_native.items()),
                'numerical trial changed a baseline mapped native provider')
        allowed = ('tf_sm70_family_', 'tf_onecat_gu_', 'tf_onecat_pack_', 'tf_onecat_down_')
        require(all(name.startswith(allowed) for name in new_native.keys() - old_native.keys()),
                'numerical trial introduced an unrelated native provider')
    rows, metrics = quality._candidate_teacher_metrics(left, right, control,
        execution_families=(old['prepared_execution'], new['prepared_execution']))
    return {'schema': 'tensorfold.numerical-teacher-comparison-v3', 'complete': True,
            'quality_acceptance': False, 'P3_valid': False, 'numerical_trial': new['numerical_trial'],
            'selector': new['selector'], 'reference': reference, 'candidate': candidate,
            'frames': rows, 'quality_metrics': metrics,
            'actual_profiles': [a['actual_owned_profile'], b['actual_owned_profile']],
            'coverage': {'context': 262144, 'resident_states': 24, 'active_states': 1, 'method': 'none',
                         'graphs': False, 'frames': len(rows), 'vocab': 248320,
                         'uncovered': ['C8 target windows', 'MTP4 proposal/verification', 'graph replay', 'free generation']}}


if __name__ == '__main__':
    raise SystemExit(main())

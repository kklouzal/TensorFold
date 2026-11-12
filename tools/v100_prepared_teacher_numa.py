"""Versioned prepared-teacher NUMA/CAP40 DATA and receipt authority.

The v1 teacher contracts remain historical oracles. This required execution
region binds a separately captured verification-cap40 seccomp baseline, one
sealed source-owned rank plan, and all24 real worker placement receipts.
"""
from __future__ import annotations

import hashlib
import os
import sys
import types
from pathlib import Path

if __package__:
    from . import v100_onecat_combinations as base
else:
    import v100_onecat_combinations as base

legacy = base.legacy
SCHEMA = 'tensorfold.v100-prepared-expert-teacher-numa-operation-v2'
MODE = 'owned-prepared-expert-teacher-numa-model-v2'
HOST_RESULT_SCHEMA = 'tensorfold.owned-prepared-expert-teacher-numa-operation-result-v2'
RESULT_SCHEMA = 'tensorfold.prepared-expert-teacher-numa-model-result-v2'
CLIENT_SCHEMA = 'tensorfold.prepared-expert-teacher-numa-client-v2'
CLIENT_REPORT_SCHEMA = 'tensorfold.prepared-expert-teacher-numa-client-report-v2'
SCOPE = base.SCOPE
NUMA_SCHEMA = 'tensorfold.prepared-teacher-numa-input-v1'
CAPTURE_ROLE = 'verification-cap40-v1'
POLICY_PURPOSE = 'tensorfold-numa-verification-cap40-worker'
RECEIPT_DIR = '/evidence/numa'
PLACEMENT_RELATIVE = 'src/tensorfold/cuda/rank_placement.py'
PLACEMENT_MEMBER = 'cuda/rank_placement.py'
FILES = {'plan': ('numa-plan.json', 1 << 20),
         'baseline_seccomp': ('numa-baseline-seccomp.json', 2 << 20),
         'seccomp': ('numa-seccomp.json', 2 << 20),
         'seccomp_manifest': ('numa-seccomp-manifest.json', 2 << 20)}
STAGES = ('bootstrap', 'post-cuda', 'ready')
EXTRA_REFERENCE_FILES = {'teacher/' + name for name, _ in FILES.values()} | {
    'teacher/numa-readiness.json'} | {f'numa/rank-{rank}-{stage}.json' for rank in range(8) for stage in STAGES}
identity = legacy.identity
absolute = legacy.absolute
require = legacy.require
selectors = base.selectors
validate_teacher_completion = legacy.validate_teacher_completion
REQUIRED_PREFIXES = legacy.REQUIRED_PREFIXES
MODEL_OBSERVER_LIMITS = legacy.MODEL_OBSERVER_LIMITS
SERVICE = base.SERVICE


def validate_numa(value, *, operator_root='/operator'):
    require(type(value) is dict and set(value) == {'schema', 'placement_source', *FILES}
            and value['schema'] == NUMA_SCHEMA, 'closed verification-cap40 NUMA input required')
    source = value['placement_source']
    require(type(source) is dict and set(source) == {'bytes', 'sha256'}
            and type(source['bytes']) is int and 0 < source['bytes'] <= 8 << 20
            and legacy.sha(source['sha256']), 'exact installed placement source identity required')
    for key, (name, limit) in FILES.items():
        identity(value[key])
        require(value[key]['path'] == str(Path(operator_root)/name) and value[key]['bytes'] <= limit,
                'NUMA input must retain its exact flat sealed operator path/bound')
    return value


def _reference_projection(value):
    return {**value, 'files': {name: item for name, item in value['files'].items() if name not in EXTRA_REFERENCE_FILES}}


def validate_reference(value):
    require(type(value) is dict and set(value) == {'root', 'authority', 'files', 'host_result', 'completion'}
            and type(value['files']) is dict and EXTRA_REFERENCE_FILES <= set(value['files']),
            'prepared NUMA replay requires its full placement/policy/24-receipt closure')
    projected = _reference_projection(value)
    legacy.validate_reference(projected)
    for name in EXTRA_REFERENCE_FILES:
        item = value['files'][name]
        require(type(item) is dict and set(item) == {'bytes', 'sha256'}, 'closed NUMA reference identity required')
        identity({'path': '/reference/' + name, **item})
        require(item['bytes'] <= (8 << 20 if name.endswith('numa-readiness.json') else 2 << 20),
                'NUMA reference exceeds its structured artifact bound')
    require(sum(item['bytes'] for item in value['files'].values()) <= 256 << 20,
            'complete prepared NUMA reference exceeds existing cumulative256MiB envelope')
    return value


def _operation_projection(value):
    result = {key: item for key, item in value.items() if key != 'numa'}
    if 'reference' in result:
        result['reference'] = _reference_projection(result['reference'])
    return result


def validate_operation(value):
    require(type(value) is dict and 'numa' in value, 'new prepared execution requires sealed NUMA/CAP40 authority')
    validate_numa(value['numa'])
    if 'reference' in value:
        validate_reference(value['reference'])
    base.validate_operation(_operation_projection(value))
    return value


def validate_sources(value):
    base.validate_sources(value)
    require({'tools/v100_prepared_teacher_numa.py', 'tools/generate_numa_seccomp.py'} <= set(value),
            'NUMA contract and authoritative seccomp generator must join the source closure')
    return value


def validate_teacher_client(value):
    require(type(value) is dict and value.get('schema') == CLIENT_SCHEMA and 'numa' in value,
            'versioned prepared NUMA teacher client required')
    validate_numa(value['numa'])
    base.validate_teacher_client({**{key: item for key, item in value.items() if key != 'numa'},
                                  'schema': base.CLIENT_SCHEMA})
    return value


def _read(path, expected, maximum):
    raw = base._quality()._owned_teacher_read(path, maximum)
    require(len(raw) == expected['bytes'] and hashlib.sha256(raw).hexdigest() == expected['sha256'],
            'sealed NUMA artifact bytes/SHA differ')
    return raw


def load_inputs(value, *, placement=None, generator=None):
    """Use the source-owned plan/policy validators; never execute input code."""
    if placement is None:
        from tensorfold.cuda import rank_placement as placement
    if generator is None:
        if __package__:
            from . import generate_numa_seccomp as generator
        else:
            import generate_numa_seccomp as generator
    _read(Path(placement.__file__), value['placement_source'], 8 << 20)
    raw = {key: _read(item['path'], item, FILES[key][1]) for key, item in value.items() if key in FILES}
    plan = placement.load_plan(value['plan']['path'], value['plan']['sha256'])
    require(len(plan['ranks']) == 8, 'prepared teacher requires exact eight-rank NUMA placement')
    manifest = base._quality()._owned_teacher_object(raw['seccomp_manifest'])
    baseline = base._quality()._owned_teacher_object(raw['baseline_seccomp'])
    require(manifest.get('schema_version') == 2 and manifest.get('purpose') == POLICY_PURPOSE
            and manifest.get('capture_role') == CAPTURE_ROLE, 'default-role seccomp is not CAP40 verification authority')
    rendered = generator.render(baseline, manifest['provenance'], purpose='worker', capture_role=CAPTURE_ROLE)
    require(all(rendered[source] == raw[target] for source, target in
                (('baseline-seccomp.json', 'baseline_seccomp'), ('seccomp.json', 'seccomp'), ('manifest.json', 'seccomp_manifest'))),
            'NUMA CAP40 seccomp differs from its actual captured baseline/frozen generator')
    return {'plan': plan, 'raw': raw, 'manifest': manifest}


def publish_inputs(operation, bundle, teacher_output):
    """Copy the four actual held payloads unchanged; never regenerate a policy."""
    root = Path(teacher_output)
    for key, (name, maximum) in FILES.items():
        data = bundle['raw'][key]
        target = root/name
        with target.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        target.chmod(0o444)
        _read(target, operation['numa'][key], maximum)


def _ref(root, path, expected=None, *, maximum=1 << 20):
    path = Path(path)
    require(not path.is_symlink() and path.resolve(strict=True).is_relative_to(Path(root).resolve(strict=True)),
            'NUMA evidence must remain an ordinary file inside its exact result root')
    data = base._quality()._owned_teacher_read(path, maximum)
    identity = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    if expected is not None:
        require(identity == {k: expected[k] for k in ('bytes', 'sha256')}, 'NUMA evidence file changed')
    return {'file': str(path.relative_to(root)), **identity}, data


def collect(operation, teacher_output, *, native_records=None):
    """Validate all actual placement stages before POST and again after close."""
    from tensorfold.cuda import rank_placement as placement
    root = Path(teacher_output).parent
    inputs = {}
    copied = {**operation['numa']}
    for key, (name, maximum) in FILES.items():
        path = Path(teacher_output)/name
        ref, _ = _ref(root, path, operation['numa'][key], maximum=maximum)
        inputs[key] = ref
        copied[key] = {**operation['numa'][key], 'path': str(path)}
    bundle = load_inputs(copied)
    rows, pids = [], set()
    for rank in range(8):
        _, data = _ref(root, Path(teacher_output)/f'rank-{rank}.json', maximum=8 << 20)
        startup = base._quality()._owned_teacher_object(data)
        expected_pid = startup.get('teacher_worker_PID')
        require(type(expected_pid) is int and expected_pid > 1 and startup.get('rank') == rank,
                'NUMA readiness lacks its real teacher startup worker')
        stages = {}
        for stage in STAGES:
            ref, raw = _ref(root, root/'numa'/f'rank-{rank}-{stage}.json')
            placement.validate_receipt(base._quality()._owned_teacher_object(raw), bundle['plan'], rank, stage,
                plan_sha256=operation['numa']['plan']['sha256'],
                source_sha256=operation['numa']['placement_source']['sha256'], pid=expected_pid)
            stages[stage] = ref
        if native_records is not None:
            ref = native_records[rank]
            _, raw = _ref(root, root/ref['file'], ref, maximum=4 << 20)
            native = base._quality()._owned_teacher_object(raw)
            require(native.get('rank') == rank and native.get('PID') == expected_pid,
                    'NUMA PID differs across bootstrap/teacher/actual closed VMA observer')
        pids.add(expected_pid)
        rows.append({'rank': rank, 'pid': expected_pid, 'stages': stages})
    require(len(pids) == 8, 'NUMA readiness requires eight distinct actual workers')
    return {'schema': 'tensorfold.prepared-teacher-numa-readiness-v1', 'inputs': inputs, 'ranks': rows,
            'plan_sha256': operation['numa']['plan']['sha256'], 'placement_source': operation['numa']['placement_source']}


def evidence(operation, teacher_output, before, native_records):
    after = collect(operation, teacher_output, native_records=native_records)
    require(after == before, 'NUMA payloads/24 worker receipts changed between generation and retirement')
    root = Path(teacher_output).parent
    ref, raw = _ref(root, Path(teacher_output)/'numa-readiness.json', maximum=8 << 20)
    require(base._quality()._owned_teacher_object(raw) == before, 'published before-generation NUMA gate changed')
    result = {**after, 'schema': 'tensorfold.prepared-teacher-numa-evidence-v1', 'readiness': ref}
    validate_evidence_refs(result)
    return result


def frozen_host_inputs(host, source_files):
    """Host-only authenticated stdlib validators; no runtime-package overlay."""
    value = host['numa']
    operator = next(m['source'] for m in host['mounts'] if m['target'] == '/operator')
    validate_numa(value, operator_root=operator)
    from_file = Path(host['source']['root'])/PLACEMENT_RELATIVE
    expected = source_files.get(PLACEMENT_RELATIVE)
    require(type(expected) is dict and all(expected.get(k) == value['placement_source'][k] for k in ('bytes', 'sha256')),
            'frozen placement validator differs from installed expected source')
    code = _read(from_file, expected, 8 << 20)
    placement = types.ModuleType('_owned_cap40_numa_placement')
    placement.__file__ = str(from_file)
    exec(compile(code, str(from_file), 'exec'), placement.__dict__)  # noqa: S102 - frozen source-authenticated stdlib only
    if __package__:
        from . import generate_numa_seccomp as generator
    else:
        import generate_numa_seccomp as generator
    expected_generator = source_files.get('tools/generate_numa_seccomp.py')
    require(type(expected_generator) is dict, 'frozen seccomp generator identity missing')
    _read(Path(generator.__file__), expected_generator, 8 << 20)
    bundle = load_inputs(value, placement=placement, generator=generator)
    require(bundle['manifest']['provenance'].get('role_authority') ==
            {'image_id': host['image_id'], 'verification_launcher': host['verification_launcher']},
            'CAP40 policy was not freshly derived for this exact verification image/launcher contract')
    require(bundle['plan']['ordered_gpu_uuids'] == host['uuids']
            and set(bundle['plan']['cpuset_cpus']) <= set(os.sched_getaffinity(0)),
            'NUMA authority GPU order/available CPU allowance differs')
    bundle['cpuset_cpus'] = ','.join(map(str, bundle['plan']['cpuset_cpus']))
    bundle['cpuset_mems'] = ','.join(map(str, bundle['plan']['cpuset_mems']))
    return bundle


def validate_teacher_operation_plan(operation, plan, client, **kwargs):
    validate_operation(operation)
    validate_teacher_client(client)
    require(operation['numa'] == client['numa'], 'client NUMA authority differs from its closed operation')
    bundle = load_inputs(operation['numa'])
    projection = {**{key: item for key, item in client.items() if key != 'numa'}, 'schema': base.CLIENT_SCHEMA}
    result = base.validate_teacher_operation_plan(_operation_projection(operation), plan, projection, **kwargs)
    require(bundle['plan']['ordered_gpu_uuids'] and len(bundle['plan']['ranks']) == 8,
            'NUMA teacher plan rank authority missing')
    return result


def validate_startup(row, selected):
    return base.validate_startup(row, selected)


def operation_assets(value):
    validate_operation(value)
    # Avoid the v1 fixed reference projection here: every new receipt/policy
    # artifact is a real held input, not an inferred proof substitution.
    return [*legacy.operation_assets(value), value['case_authority'], value['execution_manifest'],
            *[value['numa'][key] for key in FILES]]


def qualifier_argv(operation, client, uuids, output):
    validate_operation(operation)
    validate_teacher_client(client)
    projected = {**{key: item for key, item in client.items() if key != 'numa'}, 'schema': base.CLIENT_SCHEMA}
    args = base.qualifier_argv(_operation_projection(operation), projected, uuids, output)
    split = args.index('--')
    return [*args[:split], '--numa-plan', operation['numa']['plan']['path'], '--numa-plan-sha256',
            operation['numa']['plan']['sha256'], *args[split:]]


def validate_host_operation(host, inner, *, numa_bundle=None):
    validate_operation(inner['operation'])
    validate_sources(inner['source_files'])
    require(numa_bundle is not None and 'verification_launcher' in host and 'numa' in host,
            'prepared NUMA requires its independent trusted CAP40 host authority')
    projected = {**inner, 'schema': base.SCHEMA, 'mode': base.MODE,
                 'operation': _operation_projection(inner['operation'])}
    base.validate_host_operation(host, projected)
    require(host['environment'].get('TENSORFOLD_NUMA_RECEIPT_DIR') == RECEIPT_DIR
            and host['environment'].get('NCCL_IGNORE_CPU_AFFINITY') == '0',
            'NUMA receipt/NCCL affinity environment must be fixed')
    source = inner['operation']['numa']['placement_source']
    require(all(inner['expected_app'].get(PLACEMENT_MEMBER, {}).get(k) == source[k] for k in ('bytes', 'sha256')),
            'installed App lacks the exact expected placement implementation')
    operator = next(m['source'] for m in host['mounts'] if m['target'] == '/operator')
    validate_numa(host['numa'], operator_root=operator)
    require(host['numa']['placement_source'] == source and all(
        {k: host['numa'][key][k] for k in ('bytes', 'sha256')} ==
        {k: inner['operation']['numa'][key][k] for k in ('bytes', 'sha256')} for key in FILES),
        'Host/Driver NUMA payload identities differ')
    return inner


def validate_model_result(value):
    require(type(value) is dict and value.get('schema') == RESULT_SCHEMA and 'numa' in value,
            'versioned model result requires its actual NUMA readiness/24-receipt proof')
    base.validate_model_result({**{k: item for k, item in value.items() if k != 'numa'}, 'schema': base.RESULT_SCHEMA})
    validate_evidence_refs(value['numa'])
    return value


def validate_evidence_refs(value):
    require(type(value) is dict and set(value) == {'schema', 'inputs', 'readiness', 'ranks', 'plan_sha256', 'placement_source'}
            and value['schema'] == 'tensorfold.prepared-teacher-numa-evidence-v1'
            and type(value['inputs']) is dict and set(value['inputs']) == set(FILES)
            and type(value['ranks']) is list and len(value['ranks']) == 8
            and legacy.sha(value['plan_sha256']), 'closed actual NUMA evidence references required')
    source = value['placement_source']
    require(type(source) is dict and set(source) == {'bytes', 'sha256'} and type(source['bytes']) is int
            and 0 < source['bytes'] <= 8 << 20 and legacy.sha(source['sha256']),
            'NUMA evidence placement source identity differs')
    for key, (name, limit) in FILES.items():
        identity(value['inputs'][key], key='file')
        require(value['inputs'][key]['file'] == 'teacher/' + name and value['inputs'][key]['bytes'] <= limit,
                'NUMA evidence input extent/path differs')
    identity(value['readiness'], key='file')
    require(value['readiness']['file'] == 'teacher/numa-readiness.json' and value['readiness']['bytes'] <= 8 << 20,
            'NUMA readiness file differs')
    pids = set()
    for rank, row in enumerate(value['ranks']):
        require(type(row) is dict and set(row) == {'rank', 'pid', 'stages'} and type(row['rank']) is int
                and row['rank'] == rank and type(row['pid']) is int and row['pid'] > 1
                and type(row['stages']) is dict and set(row['stages']) == set(STAGES), 'NUMA evidence rank/PID/stages differ')
        pids.add(row['pid'])
        for stage, ref in row['stages'].items():
            identity(ref, key='file')
            require(ref['file'] == f'numa/rank-{rank}-{stage}.json' and ref['bytes'] <= 1 << 20,
                    'NUMA worker evidence receipt path/extent differs')
    require(len(pids) == 8, 'NUMA evidence needs eight distinct real worker PIDs')
    return value


def reference_result_files(model):
    validate_model_result(model)
    value = model['numa']
    refs = [*value['inputs'].values(), value['readiness']]
    for row in value['ranks']:
        refs.extend(row['stages'].values())
    return {ref['file']: {key: ref[key] for key in ('bytes', 'sha256')} for ref in refs}


def validate_result_files(root, model, operation):
    """Consume the real closed model's placement artifacts, not summary labels."""
    validate_model_result(model)
    root = Path(root)
    evidence_refs = model['numa']
    observed = collect(operation, root/'teacher', native_records=model['child_native_records'])
    require({k: item for k, item in evidence_refs.items() if k not in ('readiness', 'schema')} ==
            {k: item for k, item in observed.items() if k != 'schema'},
            'closed NUMA evidence differs from actual plan/24 receipts/startup/VMA workers')
    for name, expected in reference_result_files(model).items():
        _ref(root, root/name, expected, maximum=8 << 20)
    _, data = _ref(root, root/evidence_refs['readiness']['file'], evidence_refs['readiness'], maximum=8 << 20)
    require(base._quality()._owned_teacher_object(data) == observed, 'before-generation NUMA readiness proof changed')
    return observed


def validate_host_result(value, expected):
    result = legacy._validate_host_result(value, expected, schema=HOST_RESULT_SCHEMA, mode=MODE,
                                          result_validator=validate_model_result)
    require(result['teacher_plan_sha256'] == expected['operation']['teacher_plan']['sha256']
            and result['teacher_client_sha256'] == expected['operation']['teacher_client']['sha256'],
            'NUMA prepared result DATA binding differs')
    return result


def validate_completed_teacher_host(value, *, authority_sha256, authority_bytes, image_id=None):
    require(type(value) is dict and type(value.get('inner')) is dict, 'closed NUMA Host record required')
    inner = value['inner']
    validate_operation(inner['operation'])
    validate_host_result(inner, {'operation': inner['operation'], 'source_commit': inner['source_commit'],
                                'image_id': inner['image_id_declared']})
    projected_result = {**{k: item for k, item in inner['model_result'].items() if k != 'numa'}, 'schema': base.RESULT_SCHEMA}
    projected = {**inner, 'schema': base.HOST_RESULT_SCHEMA, 'mode': base.MODE,
                 'operation': _operation_projection(inner['operation']), 'model_result': projected_result}
    base.validate_completed_teacher_host({**value, 'inner': projected}, authority_sha256=authority_sha256,
        authority_bytes=authority_bytes, image_id=image_id)
    return inner


if __name__ == '__main__':
    raise SystemExit(base.main(['run-numa', *sys.argv[1:]]))

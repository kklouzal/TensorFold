"""Versioned CAP40/NUMA DATA/receipt wiring; no fixture is native evidence."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import test_numa_seccomp as policy_tests
from tests import test_rank_placement as placement_tests
from tests.test_v100_prepared_teacher_contract import data
from tools import v100_prepared_teacher_numa as numa

pytest_plugins = ['tests.test_numa_seccomp', 'tests.test_rank_placement']


def identify(path):
    raw = path.read_bytes()
    return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def payload(tmp_path, hardware, cap40_source):
    placement = placement_tests.placement
    _, proc, sysfs = hardware
    devices = [f'GPU-00000000-0000-0000-0000-{rank+1:012x}' for rank in range(8)]
    for rank, gpu in enumerate(devices):
        bus, node = f'0000:{rank+1:02x}:00.0', rank//4
        info = proc/'driver/nvidia/gpus'/bus
        info.mkdir(parents=True, exist_ok=True)
        (info/'information').write_text(f'GPU UUID: {gpu}\nBus Location: {bus}\n')
        pci = sysfs/'bus/pci/devices'/bus
        pci.mkdir(parents=True, exist_ok=True)
        (pci/'numa_node').write_text(str(node))
        (pci/'local_cpulist').write_text('0-3,8-11' if node == 0 else '4-7,12-15')
    plan = placement.build_plan(devices, proc=proc, sysfs=sysfs)
    source = identify(Path(placement.__file__))
    source = {key: source[key] for key in ('bytes', 'sha256')}
    directory = tmp_path/'payload'
    directory.mkdir()
    (directory/'numa-plan.json').write_text(json.dumps(plan))
    producer = policy_tests.profile
    baseline, provenance = producer.capture('fixed', bundle_config=cap40_source.source.config,
        capture_role=producer.CAP40_ROLE, verification_contract=cap40_source.authority)
    rendered = producer.render(baseline, provenance, purpose='worker', capture_role=producer.CAP40_ROLE)
    for canonical, flat in [('baseline-seccomp.json', 'numa-baseline-seccomp.json'),
                            ('seccomp.json', 'numa-seccomp.json'), ('manifest.json', 'numa-seccomp-manifest.json')]:
        (directory/flat).write_bytes(rendered[canonical])
    value = {'schema': numa.NUMA_SCHEMA, 'placement_source': source,
             **{key: identify(directory/name) for key, (name, _) in numa.FILES.items()}}
    return value, plan


def canonical(value):
    return {**value, **{key: {**value[key], 'path': '/operator/'+name}
                       for key, (name, _) in numa.FILES.items()}}


def actual_receipts(root, plan, authority, *, write_startup=True):
    root.mkdir(exist_ok=True)
    teacher = root/'teacher'
    teacher.mkdir(exist_ok=True)
    (root/'numa').mkdir(exist_ok=True)
    for rank, target in enumerate(plan['ranks']):
        if write_startup:
            (teacher/f'rank-{rank}.json').write_text(json.dumps({'rank': rank, 'teacher_worker_PID': 100+rank}))
        for stage in numa.STAGES:
            record = placement_tests.receipt(plan, stage)
            record.update(rank=rank, pid=100+rank, **{key: target[key] for key in ('gpu_uuid', 'pci_bus_id', 'node', 'cpus')})
            record['plan_sha256'] = authority['plan']['sha256']
            record['bootstrap_source'].update(bytes=authority['placement_source']['bytes'],
                                             sha256=authority['placement_source']['sha256'])
            record['memory_policy']['nodes'] = [target['node']]
            record['anonymous']['page_nodes'] = [target['node']]*4
            if stage != 'bootstrap':
                record['actual_cuda_gpu_uuid'] = target['gpu_uuid']
                record['cuda_pinned']['page_nodes'] = [target['node']]*5
                record['threads'] = [{'tid': 100+rank, 'name': 'fixture-worker', 'cpus': target['cpus']}]
            (root/'numa'/f'rank-{rank}-{stage}.json').write_text(json.dumps(record))
    return teacher


def test_cap40_payload_is_reconstructed_by_actual_authoritative_generator(payload):
    value, plan = payload
    numa.validate_numa(value, operator_root=str(Path(value['plan']['path']).parent))
    bundle = numa.load_inputs(value)
    assert bundle['plan'] == plan and bundle['manifest']['capture_role'] == numa.CAPTURE_ROLE
    changed = copy.deepcopy(value)
    changed['placement_source']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='bytes/SHA'):
        numa.load_inputs(changed)


def test_default_role_or_modified_policy_cannot_substitute_for_cap40(payload):
    value, _ = payload
    path = Path(value['seccomp_manifest']['path'])
    manifest = json.loads(path.read_text())
    manifest['capture_role'] = 'default'
    path.write_text(json.dumps(manifest))
    changed = {**value, 'seccomp_manifest': identify(path)}
    with pytest.raises(ValueError, match='CAP40'):
        numa.load_inputs(changed)


def test_24_receipts_bind_original_payload_and_pid_before_generation_and_close(tmp_path, payload):
    value, plan = payload
    operation = {**data()[0], 'numa': canonical(value)}
    teacher = actual_receipts(tmp_path/'evidence', plan, value)
    bundle = numa.load_inputs(value)
    numa.publish_inputs(operation, bundle, teacher)
    before = numa.collect(operation, teacher)
    assert len(before['ranks']) == 8 and all(set(row['stages']) == set(numa.STAGES) for row in before['ranks'])
    (teacher/'numa-readiness.json').write_text(json.dumps(before))
    native = []
    (teacher.parent/'child-native').mkdir()
    for rank in range(8):
        path = teacher.parent/'child-native'/f'rank-{rank}.json'
        path.write_text(json.dumps({'rank': rank, 'PID': 100+rank}))
        ref = identify(path)
        native.append({'file': str(path.relative_to(teacher.parent)), **{key: ref[key] for key in ('bytes', 'sha256')}})
    checked = numa.evidence(operation, teacher, before, native)
    assert checked['plan_sha256'] == value['plan']['sha256']
    target = teacher.parent/'numa/rank-4-ready.json'
    record = json.loads(target.read_text())
    record['pid'] += 1
    target.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='identity'):
        numa.evidence(operation, teacher, before, native)


def test_versioned_operation_client_and_argv_require_all_numa_authority(payload):
    value, plan = payload
    operation, _, _, _, client = data()
    operation['numa'] = canonical(value)
    client.update(schema=numa.CLIENT_SCHEMA, numa=canonical(value))
    assert numa.validate_operation(operation) and numa.validate_teacher_client(client)
    args = numa.qualifier_argv(operation, client, plan['ordered_gpu_uuids'], '/evidence/teacher')
    split = args.index('--')
    assert args.index('--numa-plan') < split
    assert args[args.index('--numa-plan-sha256')+1] == value['plan']['sha256']
    with pytest.raises(ValueError):
        numa.base.validate_operation(operation)


def test_readiness_failure_prevents_first_post(monkeypatch, tmp_path):
    from tools import v100_owned_qualification as owned
    selected = {'data': {'schema': numa.CLIENT_SCHEMA}, 'prepared_operation': {}, 'sha256': 'a'*64}
    client = owned._TeacherClient({'data': {}}, selected, tmp_path)
    calls = []
    transport = SimpleNamespace(call=lambda *args, **kwargs: calls.append(args))
    control = SimpleNamespace(check=lambda: None)
    monkeypatch.setattr(owned, '_owned_client_open', lambda *args: (transport, control, {}, 'instance'))
    monkeypatch.setattr(numa, 'collect', lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('bad placement')))
    with pytest.raises(ValueError, match='bad placement'):
        client.run()
    assert client.post_attempts == 0 and calls == []


def test_host_exact_cap40_image_policy_cpuset_and_inspected_seccomp(payload, cap40_source):
    from tests.test_v100_native_contract_host import CID, Contracts, document, file_record
    from tools import v100_native_contract_host as host
    value, plan = payload
    owner = Contracts()
    owner.setUp()
    try:
        spec, inner = owner.model_spec()
        spec['image_id'] = cap40_source.authority['image_id']
        spec['verification_launcher'] = copy.deepcopy(cap40_source.authority['verification_launcher'])
        spec['uuids'] = plan['ordered_gpu_uuids']
        spec['health_baseline'] = {gpu: [rank, 0, 0, 0, 0, 'No'] for rank, gpu in enumerate(spec['uuids'])}
        spec['environment'].update(CUDA_VISIBLE_DEVICES=','.join(spec['uuids']),
            NVIDIA_VISIBLE_DEVICES=','.join(spec['uuids']), TENSORFOLD_PREFILL_ROWS='256',
            TENSORFOLD_SM70_PROFILE='reference', TENSORFOLD_PRIVATE_SM70_GROUPED_DOWN='grouped-v1',
            TENSORFOLD_NUMA_RECEIPT_DIR=numa.RECEIPT_DIR, NCCL_IGNORE_CPU_AFFINITY='0')
        spec['numa'] = {**value}
        for key, (name, _) in numa.FILES.items():
            target = owner.operator/name
            target.write_bytes(Path(value[key]['path']).read_bytes())
            record = file_record(target)
            spec['inputs'].append(record)
            spec['numa'][key] = record
        operation = inner['operation']
        operation.update({key: item for key, item in data()[0].items()
            if key in ('case_authority', 'execution_manifest', 'prepared_execution', 'teacher_scope', 'selector')})
        operation['numa'] = canonical(spec['numa'])
        for key in ('case_authority', 'execution_manifest'):
            target = owner.operator/Path(operation[key]['path']).name
            target.write_text('{}\n')
            record = file_record(target)
            spec['inputs'].append(record)
            operation[key] = {**record, 'path': '/operator/'+target.name}
        manifest_path = Path(spec['source']['manifest']['path'])
        manifest = json.loads(manifest_path.read_text())
        manifest['base_app_image'] = spec['verification_launcher']['base_image_id']
        relative = numa.PLACEMENT_RELATIVE
        source = owner.source/relative
        source.parent.mkdir(parents=True)
        source.write_bytes(Path(placement_tests.placement.__file__).read_bytes())
        src = file_record(source)
        manifest['files'][relative] = {'type': 'file', 'stored_mode': source.stat().st_mode & 0o7777,
                                      **{key: src[key] for key in ('bytes', 'sha256')}}
        for name in ('v100_onecat_combinations.py', 'v100_native_contract_host.py',
                     'v100_prepared_teacher_numa.py', 'generate_numa_seccomp.py'):
            source, staged = owner.source/'tools'/name, owner.operator/name
            source.write_bytes((Path(host.__file__).parent/name).read_bytes())
            staged.write_bytes(source.read_bytes())
            record = file_record(source)
            inner['source_files']['tools/'+name] = {key: record[key] for key in ('bytes', 'sha256')}
            manifest['files']['tools/'+name] = {'type': 'file', 'stored_mode': source.stat().st_mode & 0o7777,
                                               **inner['source_files']['tools/'+name]}
            if not any(item['path'] == str(staged) for item in spec['inputs']):
                spec['inputs'].append(file_record(staged))
        inner.update(schema=numa.SCHEMA, mode=numa.MODE, image_id=spec['image_id'], UUIDs=spec['uuids'],
                     verification_launcher=spec['verification_launcher'])
        inner['expected_app'][numa.PLACEMENT_MEMBER] = {**value['placement_source'], 'mode': 0o644}
        manifest_path.write_text(json.dumps(manifest))
        spec['source']['manifest'] = file_record(manifest_path)
        (owner.operator/'spec.json').write_text(json.dumps(inner))
        spec['inputs'][1] = file_record(owner.operator/'spec.json')
        spec['argv'][5], spec['argv'][7] = spec['inputs'][1]['sha256'], str(spec['inputs'][1]['bytes'])
        host.validate_spec(spec)
        assert host.validate_data(spec)['mode'] == numa.MODE
        command = host.create_argv(spec, owner.output)
        assert command[command.index('--cap-add')+1] == 'CHECKPOINT_RESTORE'
        assert command[command.index('--cpuset-cpus')+1] == ','.join(map(str, plan['cpuset_cpus']))
        observed = document(spec)
        observed['HostConfig'].update(CpusetCpus=','.join(map(str, plan['cpuset_cpus'])), CpusetMems='0,1',
                                     SecurityOpt=['seccomp='+Path(value['seccomp']['path']).read_text()])
        host.admit(observed, spec, CID)
        changed = copy.deepcopy(observed)
        changed['HostConfig']['SecurityOpt'] = ['seccomp=unconfined']
        with pytest.raises(ValueError):
            host.admit(changed, spec, CID)
        changed = copy.deepcopy(observed)
        changed['HostConfig']['CpusetMems'] = '0'
        with pytest.raises(ValueError, match='allowance'):
            host.admit(changed, spec, CID)
        wrong_image = copy.deepcopy(spec)
        wrong_image['image_id'] = 'sha256:'+'f'*64
        with pytest.raises(ValueError, match='exact verification image'):
            host.create_argv(wrong_image, owner.output)
    finally:
        owner.doCleanups()


def numa_side(root, mode, selector, payload, reference=None):
    from tests.test_v100_owned_teacher_comparison import _write
    from tests.test_v100_prepared_teacher_comparison import side
    value, plan = payload
    selected, authority, profile = side(root, mode, selector, reference)
    host = json.loads((root/'host-result.json').read_text())
    inner, model = host['inner'], host['inner']['model_result']
    operation = inner['operation']
    operation['numa'] = canonical(value)
    if reference is not None:
        captured = json.loads((Path(reference[0]['root'])/'host-result.json').read_text())
        operation['reference']['files'].update(numa.reference_result_files(captured['inner']['model_result']))
    teacher = actual_receipts(root, plan, value, write_startup=False)
    numa.publish_inputs(operation, numa.load_inputs(value), teacher)
    before = numa.collect(operation, teacher)
    (teacher/'numa-readiness.json').write_text(json.dumps(before))
    model.update(schema=numa.RESULT_SCHEMA,
                 numa=numa.evidence(operation, teacher, before, model['child_native_records']))
    client_path = root/model['client_report']['file']
    client = json.loads(client_path.read_text())
    client['schema'] = numa.CLIENT_REPORT_SCHEMA
    model['client_report'] = _write(root, model['client_report']['file'], client)
    inner.update(schema=numa.HOST_RESULT_SCHEMA, mode=numa.MODE)
    inner['content_binding'] = _write(root, 'model-content-binding.json', {
        'completed': True, 'assets': [{'identity': item} for item in numa.operation_assets(operation)]})
    _write(root, 'owned-teacher-operation-result.json', inner)
    host_ref = _write(root, 'host-result.json', host)
    completion = json.loads((root/'completion.json').read_text())
    completion['host_result_sha256'] = host_ref['sha256']
    completion_ref = _write(root, 'completion.json', completion)
    selected.update(host_result=host_ref, completion={'path': str(root/'completion.json'),
        **{key: completion_ref[key] for key in ('bytes', 'sha256')}})
    return selected, authority, profile


def test_versioned_reader_binds_complete24_receipt_reference_without_weakening_numeric_checks(tmp_path, payload):
    from tools import v100_rotorquant_quality as quality
    first, authority, _ = numa_side(tmp_path/'capture', 'capture', numa.selectors()[0], payload)
    selector = numa.selectors()[-1]
    second, _, _ = numa_side(tmp_path/'replay', 'replay', selector, payload, (first, authority))
    report = quality.compare_owned_candidate_teacher(first, second, method=selector['gu'], down_method=selector['down'],
        iq2_source=selector['iq2_source'], prefill_method=selector['prefill_method'], teacher_scope=numa.SCOPE,
        read_timeout_s=30, numa=True)
    assert report['schema'] == 'tensorfold.prepared-expert-teacher-numa-comparison-v2'
    assert report['quality_metrics']['ppl_ratio'] == 1 and not report['quality_acceptance']
    assert all(len(proof['ranks']) == 8 for proof in report['numa_evidence'])
    receipt = tmp_path/'replay/numa/rank-6-ready.json'
    receipt.write_bytes(receipt.read_bytes()+b' ')
    with pytest.raises(ValueError):
        quality.compare_owned_candidate_teacher(first, second, method=selector['gu'], down_method=selector['down'],
            iq2_source=selector['iq2_source'], prefill_method=selector['prefill_method'], teacher_scope=numa.SCOPE,
            read_timeout_s=30, numa=True)

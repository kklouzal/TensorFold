"""Pure closed DATA/controller controls; fixtures make no native/model claim."""
import copy
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from tools import v100_native_contract_driver as driver
from tools import v100_onecat_combinations as prepared
from tools import v100_owned_model_contract as legacy


def data():
    from tests.test_v100_native_contract_driver import DriverTests

    operation = DriverTests().model_operation()
    spec = {**prepared.FAMILY, 'graph_budget_bytes': 0, 'trial': True,
            'collective_family': 'tp8-allgather-v1', 'rn_heads_per_block': 12, 'row_buckets': {}}
    def file(path):
        return {'path': '/operator/' + path, 'bytes': 1, 'sha256': 'c' * 64}
    operation.update(case_authority=file('case.json'), execution_manifest=file('profile.json'),
                     prepared_execution=spec, teacher_scope=prepared.SCOPE, selector=prepared.selectors()[0])
    case = {'schema': 'tensorfold.prepared-expert-teacher-case-v1', 'vocab': 248320, 'context': 262144,
            'input_ids': [1, 2, 3], 'max_frames': 64,
            'identity': {'case_id': 'explicit_fixture', 'model_sha256': 'a'*64, 'frontend_sha256': 'b'*64}}
    plan = {'schema': 'tensorfold.owned-teacher-plan-v1', 'mode': 'capture', 'vocab': case['vocab'],
            'context': case['context'], 'input_ids': case['input_ids'], 'max_frames': 64, 'reference': None,
            'read_timeout_s': 30, 'identity': {**case['identity'], 'run_id': 'fixture', 'profile_sha256': 'c'*64}}
    client = DriverTests().teacher_client()
    client.update(schema=prepared.CLIENT_SCHEMA, teacher_scope=prepared.SCOPE, expected_service=dict(prepared.SERVICE),
                  prepared_execution=copy.deepcopy(spec), selector=copy.deepcopy(operation['selector']),
                  execution_manifest_sha256='c'*64)
    return operation, case, {'preparation': copy.deepcopy(spec)}, plan, client


def result(operation):
    def ref(name, rank=None):
        value = {'file': name, 'bytes': 1, 'sha256': 'a'*64}
        return value if rank is None else {**value, 'rank': rank}
    model = {'schema': legacy.RESULT_SCHEMA, 'teacher_plan_sha256': operation['teacher_plan']['sha256'],
             'teacher_client_sha256': operation['teacher_client']['sha256'], 'launch_exit': 0,
             'client_report': ref('teacher/teacher-client-report.json'),
             'teacher_authority': ref('teacher/teacher-authority.json'),
             'rank_receipts': [ref(f'teacher/teacher-rank{r}/summary.json', r) for r in range(8)],
             'startup_receipts': [ref(f'teacher/rank-{r}.json', r) for r in range(8)],
             'child_native_records': [ref(f'child-native/teacher-closed-rank-{r}-pid-{r+2}.json', r) for r in range(8)]}
    inner = {'schema': prepared.HOST_RESULT_SCHEMA, 'mode': prepared.MODE, 'source_commit': '1'*40,
             'image_id_declared': 'sha256:'+'2'*64, 'operation': operation, 'source_after_match': True,
             'content_binding_complete': True, 'content_binding': ref('model-content-binding.json'),
             'parent_cuda_uninitialized': True, 'model_result': model, 'primary_error': None, 'post_errors': [],
             'scope': 'CPU schema fixture only'}
    return {'schema': 'tensorfold.native-contract-host-result-v1', 'report_complete': True,
            'source_match': True, 'health_match': True, 'primary_error': None, 'cleanup_errors': [], 'attach_exit': 0,
            'container': {'Image': inner['image_id_declared'], 'State': {'Status': 'exited', 'Running': False,
                'Pid': 0, 'ExitCode': 0, 'OOMKilled': False}}, 'inner': inner}


def test_exact21_selectors_and_full_preparation_round_trip_preserve_v1():
    operation, case, profile, plan, client = data()
    assert len(prepared.selectors()) == 21
    assert len({json.dumps(row, sort_keys=True) for row in prepared.selectors()}) == 21
    assert prepared.validate_teacher_operation_plan(operation, plan, client, case=case, profile=profile) == {
        'mode': 'capture', 'reference_sha256': None, 'teacher_ids': None}
    assert json.loads(json.dumps(operation)) == operation
    with pytest.raises(ValueError):
        legacy.validate_operation(operation)
    with pytest.raises(ValueError):
        legacy.validate_teacher_client(client)
    from tests.test_v100_native_contract_driver import DriverTests
    assert legacy.validate_operation(DriverTests().model_operation())
    assert legacy.validate_teacher_client(DriverTests().teacher_client())


@pytest.mark.parametrize('selector', prepared.selectors())
def test_closed_argv_has_only_the_declared_expert_axis_and_none_geometry(selector):
    operation, _, _, _, client = data()
    operation['selector'] = client['selector'] = selector
    args = prepared.qualifier_argv(operation, client,
        ['GPU-00000000-0000-0000-0000-'+format(r, '012x') for r in range(8)], '/evidence/teacher')
    for key, value in (('--draft-method', 'none'), ('--parallel', '24'), ('--context', '262144')):
        assert args[args.index(key)+1] == value
    assert '--no-graphs' in args and '--execution-rows' not in args and '--drafter' not in args
    assert args[args.index('--execution-profile-manifest')+1] == '/operator/profile.json'
    if selector['gu'] == 'incumbent':
        assert args[args.index('--grouped-gate-up-rows')+1] == '4' and '--onecat-gu-method' not in args
    else:
        assert args[args.index('--onecat-gu-method')+1] == selector['gu']
        assert args[args.index('--onecat-down-method')+1] == selector['down']


@pytest.mark.parametrize('field,value', [('rn_heads_per_block', 4), ('trial', False), ('graph_budget_bytes', 1),
    ('native_gdn_capture', True), ('dense_projections', {}), ('kernel_family', 'foreign')])
def test_undeclared_numerical_or_graph_axes_refuse(field, value):
    operation, *_ = data()
    operation['prepared_execution'][field] = value
    with pytest.raises(ValueError):
        prepared.validate_operation(operation)


def test_completed_host_uses_actual_declared_image_projection_and_preserves_errors():
    operation, *_ = data()
    host = result(operation)
    assert prepared.validate_completed_teacher_host(host, authority_sha256='a'*64, authority_bytes=1) == host['inner']
    for field, value in [('attach_exit', 1), ('source_match', False), ('health_match', False)]:
        changed = copy.deepcopy(host)
        changed[field] = value
        with pytest.raises(ValueError):
            prepared.validate_completed_teacher_host(changed, authority_sha256='a'*64, authority_bytes=1)
    with pytest.raises(ValueError):
        legacy.validate_completed_teacher_host(host, authority_sha256='a'*64, authority_bytes=1)


def test_driver_new_schema_keeps_original_capability_and_source_envelope():
    from tests.test_v100_native_contract_driver import DriverTests
    owner = DriverTests()
    owner.setUp()
    try:
        spec = owner.model_spec()
        spec.update(schema=prepared.SCHEMA, mode=prepared.MODE, operation=data()[0])
        for name in ('v100_onecat_combinations.py', 'v100_native_contract_host.py'):
            path = Path(driver.__file__).parent/name
            spec['source_files']['tools/'+name] = {k: driver.file_identity(path)[k] for k in ('bytes', 'sha256')}
        assert driver.validate_spec(spec) is spec
        for key, value in [('verification_launcher', None), ('required_native_elf_prefixes', [])]:
            with pytest.raises((ValueError, TypeError)):
                driver.validate_spec({**spec, key: value})
    finally:
        owner.doCleanups()


def test_driver_prepared_dispatch_reuses_model_owner_and_passes_only_typed_operation(monkeypatch):
    from tests.test_v100_native_contract_driver import DriverTests
    owner = DriverTests()
    owner.setUp()
    try:
        owner.model_run_controls()
        operation, case, profile, plan, client = data()
        owner.spec.update(schema=prepared.SCHEMA, mode=prepared.MODE, operation=operation)
        owner.spec['UUIDs'] = ['GPU-00000000-0000-0000-0000-'+format(r, '012x') for r in range(8)]
        for name in ('v100_onecat_combinations.py', 'v100_native_contract_host.py'):
            source = owner.source/'tools'/name
            source.write_bytes((Path(driver.__file__).parent/name).read_bytes())
            owner.spec['source_files']['tools/'+name] = {k: driver.file_identity(source)[k] for k in ('bytes', 'sha256')}
        payload = {'teacher-plan.json': plan, 'teacher-client.json': client, 'case.json': case, 'profile.json': profile}
        called = []
        with monkeypatch.context() as scoped:
            scoped.setattr(driver, '_bound_model_json', lambda path, **kw: payload[Path(path).name])
            scoped.setattr(driver, '_call_owned_model', lambda args, **kw: called.append((args, kw)) or 0)
            owner.execute()
        assert len(called) == 1 and called[0][1] == {'prepared_teacher_operation': operation}
        assert '--no-graphs' in called[0][0] and '--execution-rows' not in called[0][0]
        recorded = json.loads((owner.evidence/'owned-teacher-operation-result.json').read_text())
        assert recorded['schema'] == prepared.HOST_RESULT_SCHEMA and recorded['mode'] == prepared.MODE
        assert recorded['parent_cuda_uninitialized'] is True and recorded['primary_error'] is None
    finally:
        owner.doCleanups()


def test_host_prepared_dispatch_binds_flat_case_profile_and_new_contract_source():
    from tests.test_v100_native_contract_host import Contracts, file_record
    from tools import v100_native_contract_host as host
    owner = Contracts()
    owner.setUp()
    try:
        spec, inner = owner.model_spec()
        additions = data()[0]
        operation = inner['operation']
        operation.update({key: value for key, value in additions.items() if key in
                          ('case_authority', 'execution_manifest', 'prepared_execution', 'teacher_scope', 'selector')})
        for key, name in [('case_authority', 'case.json'), ('execution_manifest', 'profile.json')]:
            path = owner.operator/name
            path.write_text('{}\n')
            record = file_record(path)
            spec['inputs'].append(record)
            operation[key] = {**record, 'path': '/operator/'+name}
        manifest_path = Path(spec['source']['manifest']['path'])
        manifest = json.loads(manifest_path.read_text())
        for name in ('v100_onecat_combinations.py', 'v100_native_contract_host.py'):
            source, staged = owner.source/'tools'/name, owner.operator/name
            source.write_text('# pure declared source fixture; never executed\n')
            staged.write_bytes(source.read_bytes())
            record = file_record(source)
            inner['source_files']['tools/'+name] = {k: record[k] for k in ('bytes', 'sha256')}
            manifest['files']['tools/'+name] = {'type': 'file', 'stored_mode': source.stat().st_mode & 0o7777,
                                               **inner['source_files']['tools/'+name]}
            spec['inputs'].append(file_record(staged))
        manifest_path.write_text(json.dumps(manifest))
        spec['source']['manifest'] = file_record(manifest_path)
        inner.update(schema=prepared.SCHEMA, mode=prepared.MODE)
        (owner.operator/'spec.json').write_text(json.dumps(inner))
        spec['inputs'][1] = file_record(owner.operator/'spec.json')
        spec['argv'][5], spec['argv'][7] = spec['inputs'][1]['sha256'], str(spec['inputs'][1]['bytes'])
        spec['environment'].update(TENSORFOLD_PREFILL_ROWS='256', TENSORFOLD_SM70_PROFILE='reference',
                                   TENSORFOLD_PRIVATE_SM70_GROUPED_DOWN='grouped-v1')
        assert host.validate_data(spec)['mode'] == prepared.MODE
        wrong = copy.deepcopy(spec)
        wrong['environment']['TENSORFOLD_PRIVATE_SM70_ROUTER_LOGITS'] = 'native-serial-v1'
        with pytest.raises(ValueError, match='unrelated'):
            host.validate_data(wrong)
    finally:
        owner.doCleanups()


def test_owned_external_host_interruption_drains_child_and_restores_handlers(tmp_path):
    ready, done = tmp_path/'ready', tmp_path/'done'
    code = ('import signal,time,pathlib,sys; '
            f'done=pathlib.Path({str(done)!r}); '
            'signal.signal(signal.SIGINT,lambda *args:(done.write_text("retired"),sys.exit(0))); '
            f'pathlib.Path({str(ready)!r}).write_text("ready"); '
            'time.sleep(3)')
    old = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    def interrupt():
        deadline = time.monotonic()+2
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        if ready.exists():
            os.kill(os.getpid(), signal.SIGINT)  # This test's own current process, not a borrowed controller PID.
    sender = threading.Thread(target=interrupt)
    sender.start()
    try:
        with pytest.raises(RuntimeError, match='controller interrupted'):
            prepared._run_host_cli([sys.executable, '-c', code])
        assert done.read_text() == 'retired'
        assert {number: signal.getsignal(number) for number in old} == old
    finally:
        sender.join(timeout=3)

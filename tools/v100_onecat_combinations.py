"""Closed P256/E24/NONE expert-teacher DATA and an external Host CLI owner.

No dataset, tolerance, inference algorithm or arbitrary command is selected here.
The ordinary native-contract Host/driver owns all model/CUDA effects. This
controller observes its actual CLI exit separately before publishing completion.
"""
from __future__ import annotations

import argparse
import hashlib
import signal
import subprocess
import sys
import threading
from pathlib import Path

if __package__:
    from . import v100_owned_model_contract as legacy
else:
    import v100_owned_model_contract as legacy

SCHEMA = 'tensorfold.v100-prepared-expert-teacher-operation-v1'
MODE = 'owned-prepared-expert-teacher-model-v1'
HOST_RESULT_SCHEMA = 'tensorfold.owned-prepared-expert-teacher-operation-result-v1'
CLIENT_SCHEMA = 'tensorfold.prepared-expert-teacher-client-v1'
SCOPE = 'rn12-p256-none-v1'
FAMILY = {'kernel_family': 'sm70-paged-owner-rn-ex2-p2051-v1', 'graph_family': 'uncaptured'}
SERVICE = {**legacy.SERVICE, **FAMILY}
REQUIRED_PREFIXES = legacy.REQUIRED_PREFIXES
MODEL_OBSERVER_LIMITS = legacy.MODEL_OBSERVER_LIMITS
RESULT_SCHEMA = legacy.RESULT_SCHEMA
identity = legacy.identity
absolute = legacy.absolute
validate_model_result = legacy.validate_model_result
validate_teacher_completion = legacy.validate_teacher_completion
require = legacy.require


def selectors():
    result = [{'gu': 'incumbent', 'down': 'tensorfold', 'iq2_source': 'active-frame',
               'prefill_method': 'tensorfold'}]
    for gu in ('tensorfold', 'tensorfold-fp32-epilogue', 'onecat-half-epilogue'):
        for down in ('tensorfold', 'onecat-q8-slots'):
            for iq2 in (('active-frame',) if gu == 'tensorfold' else ('active-frame', 'raw')):
                for prefill in ('tensorfold', 'candidate'):
                    result.append({'gu': gu, 'down': down, 'iq2_source': iq2, 'prefill_method': prefill})
    return result


def validate_preparation(value):
    required = {*FAMILY, 'graph_budget_bytes', 'trial', 'collective_family', 'rn_heads_per_block', 'row_buckets'}
    require(type(value) is dict and set(value) == required,
            'expert teacher requires its finite RN12 preparation; other numerical axes are separate')
    require(all(type(value[k]) is str and value[k] == v for k, v in FAMILY.items())
            and type(value['graph_budget_bytes']) is int and value['graph_budget_bytes'] == 0
            and value['trial'] is True and value['collective_family'] == 'tp8-allgather-v1'
            and type(value['rn_heads_per_block']) is int and value['rn_heads_per_block'] == 12
            and value['row_buckets'] == {} and type(value['row_buckets']) is dict,
            'expert teacher preparation differs from RN12 uncaptured scope')
    return value


def validate_operation(value):
    required = {'teacher_plan', 'teacher_client', 'model_entrypoint', 'model_assets', 'frontend_root',
                'frontend_assets', 'execution_manifest', 'case_authority', 'prepared_execution', 'teacher_scope', 'selector'}
    require(type(value) is dict and set(value) in (required, required | {'reference'}),
            'closed prepared expert-teacher operation required')
    for key in ('teacher_plan', 'teacher_client', 'case_authority', 'execution_manifest'):
        identity(value[key])
        require(Path(value[key]['path']).parent == Path('/operator'), 'prepared teacher inputs must be flat operator DATA')
        require(value[key]['bytes'] <= (16 << 20 if key == 'execution_manifest' else 8 << 20),
                'prepared teacher input exceeds existing metadata envelope')
    require(len({value[k]['path'] for k in ('teacher_plan', 'teacher_client', 'case_authority', 'execution_manifest')}) == 4,
            'prepared teacher inputs must be distinct')
    legacy._validate_model_assets({**value, 'execution_manifest': None})
    validate_preparation(value['prepared_execution'])
    require(value['teacher_scope'] == SCOPE and type(value['selector']) is dict
            and value['selector'] in selectors(), 'explicit finite expert selector and NONE scope required')
    if 'reference' in value:
        legacy.validate_reference(value['reference'])
    return value


def validate_sources(value):
    legacy.validate_sources(value)
    require({'tools/v100_onecat_combinations.py', 'tools/v100_native_contract_host.py'} <= set(value),
            'prepared teacher contract and external Host CLI must join the source closure')
    return value


def validate_case(value):
    require(type(value) is dict and set(value) == {'schema', 'vocab', 'context', 'input_ids', 'max_frames', 'identity'}
            and value['schema'] == 'tensorfold.prepared-expert-teacher-case-v1', 'sealed explicit teacher case DATA required')
    require(type(value['vocab']) is int and value['vocab'] == 248320
            and type(value['context']) is int and value['context'] == 262144
            and type(value['input_ids']) is list and 0 < len(value['input_ids']) < value['context']
            and all(type(t) is int and 0 <= t < value['vocab'] for t in value['input_ids'])
            and type(value['max_frames']) is int and 1 <= value['max_frames'] <= 64
            and len(value['input_ids']) + value['max_frames'] <= value['context'], 'bounded SC117 teacher case geometry required')
    declared = value['identity']
    require(type(declared) is dict and set(declared) == {'case_id', 'model_sha256', 'frontend_sha256'}
            and type(declared['case_id']) is str and legacy.re.fullmatch(r'[A-Za-z0-9_-]{1,64}', declared['case_id'])
            and all(legacy.sha(declared[k]) for k in ('model_sha256', 'frontend_sha256')),
            'explicit case/model/frontend declarations required; they do not replace loaded content proof')
    return value


def validate_teacher_client(value):
    extra = {'teacher_scope', 'prepared_execution', 'selector', 'execution_manifest_sha256'}
    require(type(value) is dict and value.get('schema') == CLIENT_SCHEMA and extra <= set(value),
            'closed prepared teacher client required')
    validate_preparation(value['prepared_execution'])
    require(value['teacher_scope'] == SCOPE and type(value['selector']) is dict and value['selector'] in selectors()
            and legacy.sha(value['execution_manifest_sha256'])
            and type(value.get('expected_service')) is dict and value['expected_service'] == SERVICE
            and all(type(value['expected_service'][k]) is type(v) for k, v in SERVICE.items()),
            'prepared teacher client family/scope/selector differs')
    projected = {key: item for key, item in value.items() if key not in extra}
    projected.update(schema='tensorfold.owned-teacher-client-v1', expected_service=dict(legacy.SERVICE))
    legacy.validate_teacher_client(projected)  # Reuse only the unchanged HTTP/deadline/response DATA contract.
    return value


def validate_startup(row, selected):
    profile = row.get('teacher_profile', {})
    semantics = profile.get('teacher_semantics', {})
    require(type(row.get('rank')) is int and 0 <= row['rank'] < 8 and row.get('world') == 8
            and all(type(profile.get(k)) is int for k in ('rank', 'world', 'target_banks', 'mtp_banks'))
            and profile.get('rank') == row['rank'] and profile.get('world') == 8
            and profile.get('capability') == [7, 0] and profile.get('mtp_banks') == 0
            and profile.get('target_banks') == 48 and profile.get('plan_identity') == row.get('plan_identity')
            and semantics.get('prepared_execution') == selected['prepared_execution']
            and all(type(semantics.get(k)) is int for k in ('prefill_rows', 'target_rows', 'context'))
            and semantics.get('prefill_rows') == 256 and semantics.get('target_rows') == 24
            and semantics.get('context') == 262144,
            'prepared teacher startup differs from frozen full numerical/row authority')
    selector = selected['selector']
    _quality()._candidate_plan_identity(row.get('plan_identity'),
        None if selector['gu'] == 'incumbent' else selector['gu'], selector['down'], selector['iq2_source'],
        selector['prefill_method'], teacher_scope=SCOPE)
    return row


def validate_teacher_operation_plan(operation, plan, client, *, prepared=None, reference=None, case=None, profile=None):
    validate_operation(operation)
    validate_case(case)
    validate_teacher_client(client)
    require(type(profile) is dict and profile.get('preparation') == operation['prepared_execution'],
            'sealed execution manifest differs from the frozen full RN preparation')
    require(client['teacher_plan_sha256'] == operation['teacher_plan']['sha256']
            and client['prepared_execution'] == operation['prepared_execution']
            and client['selector'] == operation['selector'] and client['teacher_scope'] == operation['teacher_scope'],
            'prepared client/operation/plan binding differs')
    require(client['execution_manifest_sha256'] == operation['execution_manifest']['sha256'],
            'prepared teacher client sealed profile SHA differs')
    require(type(plan) is dict and plan.get('schema') == 'tensorfold.owned-teacher-plan-v1'
            and all(plan.get(k) == case[k] for k in ('vocab', 'context', 'input_ids'))
            and type(plan.get('identity')) is dict
            and all(plan['identity'].get(k) == v for k, v in case['identity'].items())
            and plan['identity'].get('profile_sha256') == operation['execution_manifest']['sha256'],
            'prepared teacher plan differs from explicit case/profile DATA')
    if plan.get('mode') == 'capture':
        require('reference' not in operation and plan.get('reference') is None
                and operation['selector'] == selectors()[0] and plan.get('max_frames') == case['max_frames'],
                'only original incumbent may capture its finite conditional greedy prefix')
        return {'mode': 'capture', 'reference_sha256': None, 'teacher_ids': None}
    require(plan.get('mode') == 'replay' and 'reference' in operation and type(prepared) is dict
            and prepared.get('data') == plan and prepared.get('input_sha256') == operation['teacher_plan']['sha256']
            and prepared.get('reference_sha256') == operation['reference']['authority']['sha256']
            and plan.get('reference') == {k: operation['reference']['authority'][k] for k in ('path', 'sha256')},
            'prepared replay requires its canonical positively closed capture')
    ids = prepared.get('teacher_ids')
    require(type(ids) is list and 0 < len(ids) <= case['max_frames'] and plan.get('max_frames') == len(ids)
            and type(reference) is dict and reference.get('teacher_ids') == ids,
            'prepared replay frozen sequence differs')
    expected = {'teacher/teacher-authority.json': {k: operation['reference']['authority'][k] for k in ('bytes', 'sha256')}}
    for rows in (reference['rank_receipts'], reference['startup_receipts']):
        for row in rows:
            expected['teacher/' + row['file']] = {k: row[k] for k in ('bytes', 'sha256')}
    for frame in reference['frames']:
        row = frame['vector']
        expected['teacher/' + row['file']] = {k: row[k] for k in ('bytes', 'sha256')}
    require(expected == operation['reference']['files'], 'prepared replay actual closure differs')
    return {'mode': 'replay', 'reference_sha256': operation['reference']['authority']['sha256'], 'teacher_ids': tuple(ids)}


def operation_assets(value):
    validate_operation(value)
    return [*legacy.operation_assets(value), value['case_authority'], value['execution_manifest']]


def qualifier_argv(operation, client, uuids, output):
    validate_operation(operation)
    validate_teacher_client(client)
    args = legacy._qualifier_argv(operation, client, uuids, output,
        ['--teacher-plan', operation['teacher_plan']['path'], '--teacher-client', operation['teacher_client']['path']])
    split = args.index('--')
    selected = operation['selector']
    flags = (['--grouped-gate-up-rows', '4'] if selected['gu'] == 'incumbent' else
             ['--onecat-gu-method', selected['gu'], '--onecat-down-method', selected['down'],
              '--onecat-iq2-source', selected['iq2_source'], '--onecat-prefill-method', selected['prefill_method'],
              '--onecat-device-budget-bytes', '117976064', '--onecat-global-host-bytes', '67108864'])
    return [*args[:split], *flags, *args[split:], '--execution-profile-manifest', operation['execution_manifest']['path']]


def validate_host_operation(host, inner):
    validate_operation(inner['operation'])
    validate_sources(inner['source_files'])
    legacy._validate_model_host(host, inner, schema=SCHEMA, mode=MODE)
    require(host['environment'].get('TENSORFOLD_PREFILL_ROWS') == '256'
            and host['environment'].get('TENSORFOLD_SM70_PROFILE', 'reference') == 'reference'
            and host['environment'].get('TENSORFOLD_PRIVATE_SM70_GROUPED_DOWN') == 'grouped-v1',
            'prepared teacher fixed NONE numeric environment differs')
    for key in ('TENSORFOLD_PRIVATE_SM70_MOE_PARTIAL', 'TENSORFOLD_PRIVATE_SM70_ROUTER_LOGITS',
                'TENSORFOLD_PRIVATE_SM70_EMPTY_PREFIX'):
        require(host['environment'].get(key) in (None, 'inherit'), 'unrelated teacher numerical selector forbidden')
    if 'reference' in inner['operation']:
        mounts = [m for m in host['mounts'] if m['target'] == '/reference']
        require(len(mounts) == 1 and mounts[0]['read_only'] is True, 'prepared replay reference needs its own RO mount')
        source = legacy.PurePosixPath(mounts[0]['source'])
        for mount in host['mounts']:
            if not mount['read_only']:
                writable = legacy.PurePosixPath(mount['source'])
                require(not source.is_relative_to(writable) and not writable.is_relative_to(source),
                        'prepared reference cannot have a writable source alias')
        for item in operation_assets(inner['operation']):
            if item['path'].startswith('/reference/'):
                owners = [m for m in host['mounts'] if legacy.PurePosixPath(item['path']).is_relative_to(legacy.PurePosixPath(m['target']))]
                require(owners == mounts, 'prepared replay file must retain its exact RO bind')
    return inner


def validate_host_result(value, expected):
    result = legacy._validate_host_result(value, expected, schema=HOST_RESULT_SCHEMA, mode=MODE,
                                          result_validator=validate_model_result)
    require(result['teacher_plan_sha256'] == expected['operation']['teacher_plan']['sha256']
            and result['teacher_client_sha256'] == expected['operation']['teacher_client']['sha256'],
            'prepared result DATA binding differs')
    return result


def validate_completed_teacher_host(value, *, authority_sha256, authority_bytes, image_id=None):
    require(type(value) is dict and value.get('schema') == 'tensorfold.native-contract-host-result-v1'
            and value.get('report_complete') is True and value.get('source_match') is True
            and value.get('health_match') is True and value.get('primary_error') is None
            and value.get('cleanup_errors') == [] and type(value.get('attach_exit')) is int and value['attach_exit'] == 0,
            'prepared Host operation lacks positive completion')
    state = value.get('container', {}).get('State', {})
    require(state.get('Status') == 'exited' and state.get('Running') is False and type(state.get('Pid')) is int
            and state['Pid'] == 0 and type(state.get('ExitCode')) is int and state['ExitCode'] == 0
            and state.get('OOMKilled') is False, 'prepared container lacks positive retirement')
    inner = value.get('inner')
    require(type(inner) is dict and value['container'].get('Image') == inner.get('image_id_declared')
            and (image_id is None or image_id == inner['image_id_declared']), 'prepared immutable image differs')
    validate_operation(inner['operation'])
    result = validate_host_result(inner, {'operation': inner['operation'], 'source_commit': inner['source_commit'],
                                         'image_id': inner['image_id_declared']})
    require(result['teacher_authority'] == {'file': 'teacher/teacher-authority.json',
            'bytes': authority_bytes, 'sha256': authority_sha256}, 'prepared authority differs from the owning result')
    return inner


def _quality():
    if __package__:
        from . import v100_rotorquant_quality
    else:
        import v100_rotorquant_quality
    return v100_rotorquant_quality


def _read(path, expected=None):
    raw = _quality()._owned_teacher_read(path, 8 << 20)
    actual = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    if expected is not None:
        require(actual == expected, 'prepared controller DATA SHA/extent differs')
    return _quality()._owned_teacher_object(raw), actual


def _run_host_cli(command):
    """Own the fixed Host child until its callback completes, including cancellation."""
    if __package__:
        from .worker_lifetime import Task, drain, raise_failures
    else:
        from worker_lifetime import Task, drain, raise_failures
    cancelled, lock = threading.Event(), threading.Lock()
    process, status, secondary = [None], [None], [None] * 5

    def invoke():
        child, primary, errors = None, None, [None] * 2
        try:
            child = subprocess.Popen(command)
            with lock:
                process[0] = child
                if cancelled.is_set():
                    child.send_signal(signal.SIGINT)
            status[0] = child.wait()
        except BaseException as error:  # noqa: BLE001 - own acquisition/wait failures through child retirement
            primary = error
        finally:
            if child is not None and child.poll() is None:
                try:
                    child.send_signal(signal.SIGINT)
                except BaseException as error:  # noqa: BLE001 - preserve cancellation failure alongside primary
                    errors[0] = error
                try:
                    child.wait()  # The existing Host owns bounded CID cancellation/retirement.
                except BaseException as error:  # noqa: BLE001 - an uncertain child wait cannot be hidden
                    errors[1] = error
        raise_failures(primary, errors)

    task = Task(invoke, name='prepared-teacher-host-cli')
    old_handlers, primary = {}, None

    def interrupted(number, frame):
        raise RuntimeError('prepared teacher controller interrupted by signal ' + str(number))

    try:
        for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            old_handlers[number] = signal.signal(number, interrupted)
        task.start()
        while not task.done.wait(.1):
            pass
    except BaseException as error:  # noqa: BLE001 - cancellation also covers KeyboardInterrupt and exhaustion
        primary = error
        cancelled.set()
        try:
            with lock:
                if process[0] is not None:
                    process[0].send_signal(signal.SIGINT)
        except BaseException as error:  # noqa: BLE001 - retain failed notification while still draining
            secondary[0] = error
    try:
        drain([task], primary=primary)
    except BaseException as error:  # noqa: BLE001 - drain preserves original BaseException identity
        primary = error
    finally:
        for index, (number, handler) in enumerate(old_handlers.items(), 1):
            try:
                signal.signal(number, handler)
            except BaseException as error:  # noqa: BLE001 - attempt every signal restoration before propagation
                secondary[index] = error
    raise_failures(primary, secondary)
    require(type(status[0]) is int, 'actual Host CLI callback did not publish its exit')
    return status[0]


def complete_cli(host_spec, expected, output, operation_gate, review, *, contract=None):
    """Observe the fixed canonical Host subprocess; never infer its CLI exit from attach."""
    if __package__:
        from . import v100_native_contract_host as host
    else:
        import v100_native_contract_host as host
    contract = legacy.prepared_teacher_contract() if contract is None else contract
    data, actual = _read(host_spec, expected)
    host.validate_spec(data)
    selected = host.validate_data(data)
    require(selected['mode'] == contract.MODE, 'expert controller only runs its exact closed prepared-teacher mode')
    sources, _ = host.source_inventory(data)
    for name, path in (('v100_native_contract_host.py', host.__file__),
                       ('v100_owned_model_contract.py', legacy.__file__),
                       ('v100_onecat_combinations.py', __file__),
                       *([('v100_prepared_teacher_numa.py', contract.__file__)] if contract.MODE == legacy.NUMA_PREPARED_TEACHER_MODE else [])):
        requested = Path(path)
        require('tools/' + name in sources and not requested.is_symlink()
                and requested.resolve() == (Path(data['source']['root']) / 'tools' / name).resolve()
                and not requested.stat().st_mode & 0o222,
                'external prepared controller must execute its frozen reviewed source closure')
        expected_source = sources['tools/' + name]
        raw = _quality()._owned_teacher_read(requested, expected_source['bytes'])
        require(all(expected_source[key] == value for key, value in
                        (('bytes', len(raw)), ('sha256', hashlib.sha256(raw).hexdigest()))),
                'executing closed Host/controller contract differs from reviewed frozen source')
    gate, gate_id = _read(operation_gate)
    reviewed, review_id = _read(review)
    for record, schema in ((gate, 'tensorfold.prepared-teacher-operation-gate-v1'),
                           (reviewed, 'tensorfold.prepared-teacher-review-v1')):
        require(type(record) is dict and set(record) == {'schema', 'host_input_sha256', 'approved'}
                and record['schema'] == schema and record['host_input_sha256'] == actual['sha256']
                and record['approved'] is True, 'explicit gate/review DATA must bind this exact Host input')
    require(not output.exists(), 'prepared teacher output must be fresh')
    code = _run_host_cli([sys.executable, str(Path(host.__file__).resolve()), '--spec', str(host_spec.absolute()),
                          '--spec-sha256', actual['sha256'], '--spec-bytes', str(actual['bytes']), '--out', str(output)])
    terminal = {'schema': 'tensorfold.prepared-teacher-external-cli-v1', 'host_cli_exit': code,
                'host_input': actual, 'scope': 'Actual fixed Host subprocess exit; not a Docker attach inference'}
    primary = RuntimeError('prepared Host CLI did not complete positively: ' + str(code)) if code else None
    try:
        if not output.exists():
            output.mkdir(mode=0o700)  # A failed pre-effect Host may not have created its journal.
        _quality()._owned_teacher_json(output / 'prepared-teacher-cli-terminal.json', terminal, 8 << 20)
    except BaseException as error:  # noqa: BLE001 - preserve failed Host exit when terminal publication also fails
        if __package__:
            from .worker_lifetime import raise_failures
        else:
            from worker_lifetime import raise_failures
        raise_failures(primary, [error])
    if primary is not None:
        raise primary
    outcome, host_id = _read(output / 'host-result.json')
    authority, authority_id = _read(output / 'teacher/teacher-authority.json')
    contract.validate_completed_teacher_host(outcome, authority_sha256=authority_id['sha256'], authority_bytes=authority_id['bytes'])
    completion = {'schema': 'tensorfold.owned-teacher-completion-v1', 'host_result_sha256': host_id['sha256'],
                  'authority_sha256': authority_id['sha256'], 'host_cli_exit': code,
                  'operation_gate_sha256': gate_id['sha256'], 'review_sha256': review_id['sha256'],
                  'terminal_observation_sha256': hashlib.sha256((output / 'prepared-teacher-cli-terminal.json').read_bytes()).hexdigest()}
    validate_teacher_completion(completion, host_result_sha256=host_id['sha256'], authority_sha256=authority_id['sha256'])
    _quality()._owned_teacher_json(output / 'prepared-teacher-completion.json', completion, 8 << 20)
    return authority


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    listed = commands.add_parser('selectors', help='publish the finite21 selectors; no model operation')
    listed.add_argument('--out', type=Path, required=True)
    for name in ('run', 'run-numa'):
        run = commands.add_parser(name, help='one separately leased, versioned closed prepared-teacher Host operation')
        run.add_argument('--spec', type=Path, required=True)
        run.add_argument('--spec-sha256', required=True)
        run.add_argument('--spec-bytes', type=int, required=True)
        run.add_argument('--out', type=Path, required=True)
        run.add_argument('--operation-gate', type=Path, required=True)
        run.add_argument('--review', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'selectors':
        _quality()._owned_teacher_json(args.out.absolute(), {
            'schema': 'tensorfold.prepared-expert-teacher-selector-matrix-v1', 'teacher_scope': SCOPE,
            'selectors': selectors(), 'quality_acceptance': False, 'P3_valid': False}, 8 << 20)
        return 0
    require(legacy.sha(args.spec_sha256) and type(args.spec_bytes) is int and 0 < args.spec_bytes <= 8 << 20,
            'bounded explicit Host input identity required')
    complete_cli(args.spec, {'bytes': args.spec_bytes, 'sha256': args.spec_sha256}, args.out.absolute(),
                 args.operation_gate, args.review,
                 contract=legacy.prepared_teacher_contract(numa=True) if args.command == 'run-numa' else None)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

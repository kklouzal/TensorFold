"""Standalone, explicitly authorized host owner for the native ragged gate.

The runtime remains the frozen installed image and its original owned qualifier.
Import/help/validation never create containers. Run requires a separately sealed
Root GO for this exact DATA, controller closure, image, source and current boot.
No inference request, benchmark, teacher, profiler or application patch is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

if __package__:
    from . import v100_adaptive_qualification as diagnostics
    from . import v100_onecat_campaign as campaign
    from . import v100_short_container as lifetime
else:
    import v100_adaptive_qualification as diagnostics
    import v100_onecat_campaign as campaign
    import v100_short_container as lifetime

SCHEMA = 'tensorfold.native-ragged-gate-operation-v1'
SHA = re.compile('[0-9a-f]{64}')
HEALTH = {
    'gpu': ['nvidia-smi', '--query-gpu=index,uuid,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total,retired_pages.single_bit_ecc.count,retired_pages.double_bit.count,retired_pages.pending', '--format=csv,noheader'],
    'links': ['nvidia-smi', 'nvlink', '-s'],
    'link_errors': ['nvidia-smi', 'nvlink', '-e'],
    'xids': ['sudo', '-n', 'journalctl', '-k', '-b', '--no-pager', '-o', 'cat', '-g', 'NVRM.*Xid'],
}
HELPERS = ('v100_ragged_replay_gate.py', 'v100_onecat_campaign.py', 'v100_short_container.py',
           'v100_adaptive_qualification.py', 'worker_lifetime.py', 'benchmark_output.py',
           'bench_concurrent.py', 'openai_protocol.py')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path, maximum):
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink() and path.is_file(), 'absolute regular DATA file required')
    with path.open('rb') as source:
        raw = source.read(maximum + 1)
    require(len(raw) <= maximum, 'DATA exceeds its bounded byte envelope')
    return raw


def object_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate DATA field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _value: require(False, 'nonfinite DATA'))


def fields(value, names):
    require(type(value) is dict and set(value) == set(names), 'closed DATA fields differ')


def absolute(value):
    require(type(value) is str and Path(value).is_absolute() and '..' not in Path(value).parts,
            'absolute DATA path required')
    return Path(value)


def asset(value, maximum):
    fields(value, ('path', 'bytes', 'sha256'))
    require(type(value['bytes']) is int and 0 < value['bytes'] <= maximum
            and type(value['sha256']) is str and SHA.fullmatch(value['sha256']), 'bounded asset identity required')
    raw = read(absolute(value['path']), maximum)
    require(len(raw) == value['bytes'] and hashlib.sha256(raw).hexdigest() == value['sha256'], 'asset identity differs')
    require(not Path(value['path']).stat().st_mode & 0o222, 'sealed asset must be read-only')
    return raw


def helper_paths():
    return {'v100_ragged_replay_gate.py': Path(__file__),
            'v100_onecat_campaign.py': Path(campaign.__file__),
            'v100_short_container.py': Path(lifetime.__file__),
            'v100_adaptive_qualification.py': Path(diagnostics.__file__),
            'worker_lifetime.py': Path(sys.modules[campaign.raise_failures.__module__].__file__),
            'benchmark_output.py': Path(sys.modules[diagnostics.write_json.__module__].__file__),
            'bench_concurrent.py': Path(diagnostics.bench.__file__),
            'openai_protocol.py': Path(diagnostics.protocol.__file__)}


def validate_operation(value):
    fields(value, ('schema', 'source', 'image', 'installed_check', 'profile', 'oracle', 'demand',
                   'model', 'numa', 'resources', 'health_baseline', 'controller_closure', 'output'))
    require(value['schema'] == SCHEMA, 'ragged operation schema differs')
    fields(value['source'], ('root', 'source_uid'))
    absolute(value['source']['root'])
    absolute(value['output'])
    require(type(value['source']['source_uid']) is str and SHA.fullmatch(value['source']['source_uid']), 'source UID required')
    require(type(value['image']) is str and re.fullmatch('sha256:[0-9a-f]{64}', value['image']), 'immutable image ID required')
    fields(value['installed_check'], ('root', 'inputs', 'terminal'))
    installed_root = absolute(value['installed_check']['root'])
    for name, filename in (('inputs', 'check-inputs.json'), ('terminal', 'container-result.json')):
        asset(value['installed_check'][name], 8 << 20)
        require(Path(value['installed_check'][name]['path']) == installed_root / filename, 'installed receipt path differs')
    model = value['model']
    fields(model, ('gpus', 'models_root', 'target', 'mtp', 'cache', 'max_waiting_input_bytes'))
    require(type(model['gpus']) is list and len(model['gpus']) == len(set(model['gpus'])) == 8
            and all(type(item) is str and re.fullmatch('GPU-[0-9a-f-]{36}', item) for item in model['gpus']), 'eight ordered GPU UUIDs required')
    for name in ('models_root', 'cache'):
        absolute(model[name])
    for name in ('target', 'mtp'):
        item = model[name]
        require(type(item) is str and not Path(item).is_absolute() and '..' not in Path(item).parts and item.endswith('.gguf'), 'relative GGUF model path required')
    require(type(model['max_waiting_input_bytes']) is int and 0 < model['max_waiting_input_bytes'] < 2**63, 'finite original input credit required')
    fields(value['numa'], ('plan', 'plan_sha256', 'seccomp_directory', 'seccomp_manifest_sha256'))
    for name in ('plan', 'seccomp_directory'):
        absolute(value['numa'][name])
    for name in ('plan_sha256', 'seccomp_manifest_sha256'):
        require(type(value['numa'][name]) is str and SHA.fullmatch(value['numa'][name]), 'NUMA authority digest required')
    resources = value['resources']
    fields(resources, ('cpus', 'memory', 'startup_timeout_s', 'operation_timeout_s'))
    require(type(resources['cpus']) is int and 0 < resources['cpus'] <= 64, 'bounded host CPU grant required')
    require(type(resources['memory']) is str and re.fullmatch('[1-9][0-9]*[mg]', resources['memory']), 'explicit host memory/no-swap grant required')
    require(all(type(resources[name]) is int and 0 < resources[name] < 2**31 for name in ('startup_timeout_s', 'operation_timeout_s'))
            and resources['startup_timeout_s'] <= resources['operation_timeout_s'], 'finite ordered timeouts required')
    fields(value['health_baseline'], HEALTH)
    require(all(type(item) is str and SHA.fullmatch(item) for item in value['health_baseline'].values()), 'sealed health digest required')
    fields(value['controller_closure'], HELPERS)
    loaded = helper_paths()
    for name, row in value['controller_closure'].items():
        asset(row, 2 << 20)
        require(Path(row['path']).resolve() == Path(__file__).with_name(name).resolve() == loaded[name].resolve(),
                'actual imported host controller helper origin differs')
    oracle = object_json(asset(value['oracle'], 4096))
    expected_oracle = {'schema': 'tensorfold.native-ragged-replay-input-v1', 'key': [28, 8],
        'layouts': [[5, 5, 5, 5, 5, 1, 1, 1], [1, 5, 2, 4, 5, 3, 4, 4]],
        'slots': 10, 'maximum_context_rows': 10, 'comparison': 'bit-exact'}
    require(json.dumps(oracle, sort_keys=True) == json.dumps(expected_oracle, sort_keys=True), 'closed native oracle DATA differs')
    demand = object_json(asset(value['demand'], 1 << 20))
    require(demand.get('schema') == 'tensorfold.source27.actual-graph-demand-result-v1'
            and demand.get('all8_event_and_histogram_census_identical') is True, 'actual Source27 demand evidence required')
    profile = object_json(asset(value['profile'], 16 << 20))
    require(profile.get('schema') == 'tensorfold.execution-profile-manifest' and profile.get('version') == 1
            and type(profile.get('artifacts')) is dict and profile['artifacts'], 'original execution profile required')
    policy = profile['preparation']['exact_graph_policy']
    fields(policy, ('policy', 'keys', 'evidence_sha256'))
    keys = {(role, n * rows, n) for n in (1, 2, 4, 8, 16, 24)
            for rows in (range(1, 6) if n == 1 else (1, 5)) for role in ('verify', 'mtp')}
    keys.update({('verify', 28, 8), ('mtp', 28, 8)})
    require(policy['policy'] == 'observed-demand-v1' and policy['evidence_sha256'] == value['demand']['sha256']
            and type(policy['keys']) is list and len(policy['keys']) == 32
            and all(type(k) is list and len(k) == 3 and type(k[0]) is str
                    and all(type(n) is int for n in k[1:]) for k in policy['keys'])
            and {tuple(k) for k in policy['keys']} == keys, 'all original30 plus only two ragged keys required')
    expected_preparation = {'kernel_family': 'sm70-paged-owner-rn-ex2-p2051-v1', 'graph_family': 'uncaptured',
        'graph_budget_bytes': 0, 'trial': True, 'collective_family': 'tp8-allgather-v1', 'rn_heads_per_block': 12,
        'native_gdn_capture': True, 'native_hc_writeback_capture': True, 'native_family': 'owned-native-family-v1',
        'owned_q8_window_prepare': True, 'prepared_linears': 'owned-prepared-m32-v1', 'owned_q8_window_dispatch': 'grid-v1',
        'exact_graph_policy': policy}
    require(json.dumps(profile['preparation'], sort_keys=True) == json.dumps(expected_preparation, sort_keys=True),
            'original AUTO/native/grid preparation required')
    return value


def authorization(value, *, operation_sha, spec, boot_id):
    fields(value, ('schema', 'approved', 'operation_sha256', 'controller_sha256', 'source_uid', 'image', 'boot_id'))
    require(value['schema'] == 'tensorfold.native-ragged-gate-root-go-v1' and value['approved'] is True
            and value['operation_sha256'] == operation_sha and value['source_uid'] == spec['source']['source_uid']
            and value['image'] == spec['image'] and value['boot_id'] == boot_id
            and value['controller_sha256'] == spec['controller_closure']['v100_ragged_replay_gate.py']['sha256'],
            'exact current-boot Root GO required')


def qualifier_argv(spec):
    model = spec['model']
    return ['/opt/venv/bin/python', '-B', '/tools/v100_owned_qualification.py', '--gpus', ','.join(model['gpus']),
        '--out', '/evidence/qualification', '--execution-rows', '120', '--numa-plan', '/operator/numa/plan.json',
        '--numa-plan-sha256', spec['numa']['plan_sha256'], '--grouped-gate-up-rows', '4',
        '--ragged-replay-input', '/operator/ragged-input.json', '--', '/models/' + model['target'],
        '--backend', 'cuda', '--context', '262144', '--parallel', '24', '--kv-full-windows', '24',
        '--kv-dtype', 'int8', '--indexer-history', 'gpu', '--expert-tensor-parallel-size', '4',
        '--draft-method', 'mtp', '--drafter', '/models/' + model['mtp'], '--mtp-drafts', '4', '--mtp-confidence', '0.7',
        '--no-prefill-fp8', '--no-thinking', '--no-update-check', '--host', '127.0.0.1', '--port', '8000',
        '--name', 'providerone-sc117-ragged-component', '--max-pending-requests', '24', '--max-http-connections', '25',
        '--max-waiting-input-bytes', str(model['max_waiting_input_bytes']), '--execution-profile-manifest', '/operator/profile.json']


def container_argv(spec, source, bundle, operator, output):
    gpus = ','.join(spec['model']['gpus'])
    args = ['docker', 'create', '--name', 'tf-ragged-' + output.name, '--runtime', 'nvidia', '--gpus', '"device=' + gpus + '"',
        '--network', 'none', '--read-only', '--user', '1000:1000', '--cpus', str(spec['resources']['cpus']),
        '--memory', spec['resources']['memory'], '--memory-swap', spec['resources']['memory'],
        '--cpuset-cpus', bundle['cpuset_cpus'], '--cpuset-mems', bundle['cpuset_mems'],
        '--security-opt', 'seccomp=' + str(operator / 'numa/seccomp.json'), '--ipc', 'host',
        '--ulimit', 'memlock=-1:-1', '--ulimit', 'nofile=1048576:1048576', '--ulimit', 'core=0:0',
        '--tmpfs', '/tmp:rw,exec,nosuid,nodev,size=8g,mode=1777']
    env = {'NVIDIA_VISIBLE_DEVICES': gpus, 'CUDA_VISIBLE_DEVICES': gpus, 'NVIDIA_DRIVER_CAPABILITIES': 'compute,utility',
        'TENSORFOLD_NUMA_RECEIPT_DIR': '/evidence/numa', 'NCCL_IGNORE_CPU_AFFINITY': '0',
        'TENSORFOLD_NO_UPDATE_CHECK': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
        'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'TF_GGUF_ASSETS_DIR': '/cache/gguf_assets', 'TENSORFOLD_SM70_PROFILE': 'reference',
        'TENSORFOLD_PRIVATE_SM70_GROUPED_DOWN': 'grouped-v1', 'TENSORFOLD_PREFILL_ROWS': '256'}
    for name, value in env.items():
        args += ['-e', name + '=' + value]
    for host, target, mode in ((spec['model']['models_root'], '/models', 'ro'), (str(source / 'source/tools'), '/tools', 'ro'),
            (str(operator), '/operator', 'ro'), (str(output), '/evidence', 'rw'), (spec['model']['cache'], '/cache', 'rw')):
        args += ['-v', host + ':' + target + ':' + mode]
    qualifier = qualifier_argv(spec)
    return args + ['-w', '/tools', '--entrypoint', qualifier[0], spec['image'], *qualifier[1:]]


def command(argv):
    result = diagnostics.capture(argv, timeout=15, limit=1 << 20)
    require(result['status'] == 'ok' and result['exit_code'] == 0, 'host command failed: ' + repr(result))
    return result['output_utf8']


def health(spec):
    require(not command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']).strip(), 'GPU consumer remains')
    result = {}
    for name, argv in HEALTH.items():
        row = diagnostics.capture(argv, timeout=15, limit=1 << 20)
        no_matches = name == 'xids' and row['exit_code'] == 1 and not row['output_utf8']
        require((row['status'] == 'ok' and row['exit_code'] == 0) or no_matches, 'health command failed: ' + repr(row))
        require(row['output_sha256'] == spec['health_baseline'][name], 'GPU/link/Xid baseline changed: ' + name)
        result[name] = row
    return result


def collect(output, spec, frozen):
    """Consume bounded actual all-rank receipts; no output/model oracle substitute."""
    rows = []
    expected_sources = {name: frozen['files'][path]['sha256'] for name, path in (
        ('engine', 'src/tensorfold/families/qwen4_exp/cuda/engine.py'),
        ('graph_controller', 'src/tensorfold/families/qwen4_exp/cuda/multi_graphs.py'),
        ('graph_policy', 'src/tensorfold/families/qwen4_exp/cuda/exact_graphs.py'),
        ('observer_tool', 'tools/v100_owned_qualification.py'))}
    expected_keys = object_json(asset(spec['profile'], 16 << 20))['preparation']['exact_graph_policy']['keys']
    expected_layouts = object_json(asset(spec['oracle'], 4096))['layouts']
    for rank in range(8):
        path = output / 'qualification' / f'ragged-replay-rank-{rank}.json'
        row = object_json(read(path.absolute(), 8 << 20))
        require(row['schema'] == 'tensorfold.native-ragged-replay-oracle-v1' and row['rank'] == rank
                and row['input_sha256'] == spec['oracle']['sha256'] and len(row['cases']) == 2
                and all(case['exact'] is True for case in row['cases'])
                and [case['lengths'] for case in row['cases']] == expected_layouts
                and all(case['rows'] == 28 and case['participants'] == 8 and len(set(case['slots'])) == 8 for case in row['cases'])
                and row['runtime_sources'] == expected_sources, 'native oracle rank receipt differs')
        for phase in ('before_probe', 'after_probe'):
            accounting = row['accounting'][phase]
            require(accounting['rank'] == rank and accounting['schema'] == 'tensorfold.native-ragged-accounting-v1'
                    and accounting['catalogue']['evidence_sha256'] == spec['demand']['sha256']
                    and sorted(accounting['catalogue']['captured_keys']) == sorted(expected_keys)
                    and all(sorted(accounting['catalogue'][name]) == sorted(expected_keys) for name in ('requested_keys', 'selected_keys', 'available_keys'))
                    and not accounting['catalogue']['declined_keys'] and len(accounting['actual_states']) == 24
                    and accounting['capture_settlement']['room_bytes'] <= accounting['capture_settlement']['budget_bytes']
                    and accounting['capacity']['graph_admission'] == 'AUTO'
                    and accounting['capacity']['context_window'] == 262144 and accounting['capacity']['reserved'] == 24
                    and accounting['capacity']['workspace_bytes'] == 2 * 2**30
                    and accounting['quiescent_sample']['physical_free_bytes'] >= accounting['capacity']['effective_device_floor_bytes']
                    and all(st['logical_context'] == 262144 and st['capacity'] >= 262144 and st['pos'] == st['mtp_len'] == 0 for st in accounting['actual_states']),
                    'complete all-rank settled accounting required')
        require(row['accounting']['before_probe']['capture_settlement'] == row['accounting']['after_probe']['capture_settlement'],
                'graph settlement changed during probe')
        startup_path = output / 'qualification' / f'rank-{rank}.json'
        startup = object_json(read(startup_path.absolute(), 8 << 20))
        require(startup['rank'] == rank and startup['world'] == 8 and startup['state_count'] == 24
                and startup['capability'] == [7, 0] and startup['tool_sha256'] == expected_sources['observer_tool']
                and startup['runtime_cli_sha256'] == frozen['files']['src/tensorfold/cli.py']['sha256']
                and startup['history_host_obligation'] == {'global_bytes': row['resources']['global_host_obligation_bytes'],
                    'input_sha256': spec['oracle']['sha256']}, 'startup/source/admission rank binding differs')
        rows.append({'rank': rank, 'oracle': {'file': str(path.relative_to(output)), **campaign.identity(path)},
                     'startup': {'file': str(startup_path.relative_to(output)), **campaign.identity(startup_path)}})
    return rows


def unchanged_receipts(output, spec, frozen, expected):
    require(collect(output, spec, frozen) == expected, 'native oracle/startup receipts changed after publication')


_CAPS = """import json,pathlib,sys
pids=json.loads(sys.argv[1]);rows=[]
for pid in pids:
 raw=pathlib.Path('/proc',str(pid),'status').read_text();values={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in raw.splitlines() if ':' in line}
 if not (values['CapEff']=='0000000000000000' and values['Uid'].split()==['1000']*4 and values['Gid'].split()==['1000']*4): raise RuntimeError('ordinary rank UID/GID/CapEff differs')
 rows.append({'pid':pid,'CapEff':values['CapEff'],'Uid':values['Uid'],'Gid':values['Gid']})
print(json.dumps({'actual_rank_processes':rows}))
"""


def run(spec, output, *, root_go, operation_sha, owner_factory=lifetime.OwnedContainer):
    authorization(root_go, operation_sha=operation_sha, spec=spec,
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    require(output.is_absolute() and output == absolute(spec['output']), 'output must match sealed one-attempt operation')
    source = absolute(spec['source']['root'])
    frozen = campaign.frozen(source)
    require(frozen['source_uid'] == spec['source']['source_uid'], 'frozen source differs')
    binding = campaign.immutable_image(spec['image'])
    require(binding['id'] == spec['image'], 'actual image differs')
    for name in HELPERS:
        if name != 'v100_ragged_replay_gate.py':
            expected_helper = frozen['files']['tools/' + name]
            require(all(spec['controller_closure'][name][key] == expected_helper[key] for key in ('bytes', 'sha256')),
                    'host helper differs from original frozen runtime source')
    installed = campaign.installed_check_receipt(absolute(spec['installed_check']['root']), frozen['source_uid'], spec['image'])
    settings = {'gpus': spec['model']['gpus'], 'numa': spec['numa']}
    bundle = campaign._numa_inputs(source, frozen, settings)
    manifest = object_json(bundle['files']['manifest.json'])
    require(manifest['provenance']['container']['image'] == spec['image'], 'ordinary seccomp image authority differs')
    output.mkdir(parents=True, exist_ok=False)
    operator = output / 'operator'
    operator.mkdir()
    for name, key, limit in (('profile.json', 'profile', 16 << 20), ('ragged-input.json', 'oracle', 4096)):
        (operator / name).write_bytes(asset(spec[key], limit))
        (operator / name).chmod(0o444)
    campaign._stage_numa(bundle, operator, output)
    operator.chmod(0o555)
    owner, primary, errors, receipts, ready_numa = owner_factory(), None, [], None, None
    started = time.monotonic()
    deadline = started + spec['resources']['startup_timeout_s']
    campaign.publish(output / 'inputs.json', {'operation': spec, 'operation_sha256': operation_sha, 'root_go': root_go, 'installed': installed, 'image': binding,
        'controller': campaign.identity(Path(__file__)), 'scope': 'standalone native component; no quality/P3 qualification'})
    before = health(spec)
    try:
        args = container_argv(spec, source, bundle, operator, output)
        owner.create(args, lambda argv, result: campaign.publish(output / 'create.json', {'argv': argv, 'cid': result.stdout.decode().strip()}))
        command(['docker', 'start', owner.cid])
        while len(list((output / 'qualification').glob('rank-*.json'))) != 8:
            state = object_json(command(['docker', 'inspect', '--format', '{{json .State}}', owner.cid]))
            require(state['Running'], 'model exited before all startup receipts')
            if time.monotonic() >= deadline:
                raise TimeoutError('original all-rank startup deadline expired')
            time.sleep(min(1, max(0, deadline - time.monotonic())))
        receipts = collect(output, spec, frozen)
        ready = output / 'readiness'
        ready.mkdir()
        remaining = min(deadline - time.monotonic(), started + spec['resources']['operation_timeout_s'] - time.monotonic())
        require(remaining > 0, 'operation deadline expired before readiness')
        campaign.logged(['docker', 'exec', owner.cid, '/opt/venv/bin/python', '-B', '-c', campaign._WAIT_READY,
                         str(remaining), 'q8'], ready, remaining + 5)
        settled = object_json(read((output / 'readiness.json').absolute(), 16 << 20))['stats']
        require(all(settled['load'][name] == 0 for name in ('requests_in_flight', 'requests_running', 'requests_waiting'))
                and all(n == 0 for n in settled['load']['lanes'].values())
                and settled['totals']['requests'] == settled['totals']['generated_tokens'] == 0,
                'settled READY must have no inference request or active State owner')
        ready_numa = campaign.numa_receipts(source, frozen, output, bundle)
        pids = [row['pid'] for row in ready_numa['ranks']]
        caps = output / 'capabilities'
        caps.mkdir()
        campaign.logged(['docker', 'exec', owner.cid, '/opt/venv/bin/python', '-B', '-c', _CAPS, json.dumps(pids)], caps, 30)
    except BaseException as error:  # noqa: BLE001 - Retain cancellation until mandatory owned-CID retirement.
        primary = error
    finally:
        owner.retire()
    errors.extend(error for error in owner.errors if error is not None)
    for label, action in (
        ('server.log', lambda: campaign.logged(['docker', 'logs', owner.cid], output / 'logs', 60) if owner.cid else None),
        ('source', lambda: campaign.verify_source(source / 'source', frozen['files'])),
        ('inputs', lambda: validate_operation(spec)),
        ('health', lambda: health(spec)),
        ('receipts', lambda: unchanged_receipts(output, spec, frozen, receipts) if receipts is not None else None),
        ('numa', lambda: require(campaign.numa_receipts(source, frozen, output, bundle) == ready_numa, 'NUMA lifetime receipts changed') if ready_numa else None),
    ):
        try:
            if label == 'server.log':
                (output / 'logs').mkdir()
            result = action()
            if label == 'health':
                campaign.publish(output / 'health.json', {'before': before, 'after': result})
        except BaseException as error:  # noqa: BLE001 - Every post-retirement failure belongs to the nonzero outcome.
            errors.append(error)
    removed = False
    if owner.cid and owner.state and owner.state.get('Status') == 'exited' and owner.state.get('Pid') == 0:
        try:
            require(command(['docker', 'rm', owner.cid]).strip() == owner.cid, 'owned removal receipt differs')
            removed = True
        except BaseException as error:  # noqa: BLE001 - Removal/cancellation failure must remain in the owning outcome.
            errors.append(error)
    failed = owner.exit_code(primary) or bool(errors)
    try:
        campaign.publish(output / 'result.json', {'schema': 'tensorfold.native-ragged-gate-result-v1', 'cid': owner.cid,
            'state': owner.state, 'server_log': 'logs/console.log', 'exit_code': int(failed), 'primary': None if primary is None else repr(primary),
            'cleanup': [repr(error) for error in errors], 'oracle_receipts': receipts,
            'elapsed_s': time.monotonic() - started, 'removed': removed, 'P3_valid': False, 'quality_qualified': False})
    except BaseException as error:  # noqa: BLE001 - Preserve publication failure alongside the original and cleanup errors.
        campaign.raise_failures(primary, [*errors, error])
    return int(failed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operation', type=Path, required=True)
    parser.add_argument('--operation-sha256', required=True)
    parser.add_argument('--root-go', type=Path, required=True)
    parser.add_argument('--root-go-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    raw = read(args.operation, 1 << 20)
    require(hashlib.sha256(raw).hexdigest() == args.operation_sha256, 'operation digest differs')
    spec = validate_operation(object_json(raw))
    go = read(args.root_go, 65536)
    require(hashlib.sha256(go).hexdigest() == args.root_go_sha256, 'Root GO digest differs')
    go_value = object_json(go)
    return run(spec, args.out, root_go=go_value, operation_sha=args.operation_sha256)


if __name__ == '__main__':
    raise SystemExit(main())

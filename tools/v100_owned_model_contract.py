"""Closed DATA contract for one cold owned-teacher model operation.

Shared by the host, inert driver and canonical qualifier. This module performs
no IO/import of the runtime and executes no operation. It is not an argv runner.
"""
from __future__ import annotations

from pathlib import PurePosixPath
import re
import math
import urllib.parse

SCHEMA = 'tensorfold.v100-owned-teacher-operation-v1'
MODE = 'owned-teacher-model-v1'
RESULT_SCHEMA = 'tensorfold.owned-teacher-model-result-v1'
HOST_RESULT_SCHEMA = 'tensorfold.owned-teacher-operation-result-v1'
CHILD_ENV = 'TENSORFOLD_NATIVE_CONTRACT_CHILD_EVIDENCE'
DRIVER_ENV = 'TF_NATIVE_CONTRACT_DRIVER'
DRIVER_SHA_ENV = 'TF_NATIVE_CONTRACT_DRIVER_SHA256'
PREFIX_ENV = 'TF_NATIVE_CONTRACT_REQUIRED_PREFIXES'
REQUIRED_PREFIXES = ['tf_gguf_']
FRONTEND_NAMES = {'config.json', 'generation_config.json', 'ready.json', 'tokenizer.json', 'tokenizer_config.json'}
REQUIRED_TOOLS = {'tools/v100_owned_qualification.py', 'tools/launch_cuda_tp.py',
                  'tools/v100_native_contract_driver.py', 'tools/v100_owned_model_contract.py',
                  'tools/v100_rotorquant_quality.py', 'tools/v100_rotorquant_live.py'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value, length=64):
    return type(value) is str and re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is not None


def absolute(value):
    return (type(value) is str and value.startswith('/') and '\0' not in value
            and '..' not in PurePosixPath(value).parts and str(PurePosixPath(value)) == value)


def relative(value):
    return (type(value) is str and bool(value) and '\0' not in value and not value.startswith('/')
            and '..' not in PurePosixPath(value).parts and value != '.' and str(PurePosixPath(value)) == value)


def identity(value, *, key='path', rank=False):
    fields = {key, 'bytes', 'sha256'} | ({'rank'} if rank else set())
    require(type(value) is dict and set(value) == fields, 'closed file identity required')
    require((absolute(value[key]) if key == 'path' else relative(value[key]))
            and type(value['bytes']) is int and 0 < value['bytes'] < 2**63 and sha(value['sha256']),
            'ordinary file path, positive extent and SHA256 required')
    if rank:
        require(type(value['rank']) is int and 0 <= value['rank'] < 8, 'fixed teacher rank required')
    return value


def validate_operation(value):
    fields = {'teacher_plan', 'teacher_client', 'model_entrypoint', 'model_assets',
              'frontend_root', 'frontend_assets', 'execution_manifest'}
    require(type(value) is dict and set(value) == fields, 'closed owned teacher operation required')
    for name in ('teacher_plan', 'teacher_client'):
        identity(value[name])
        require(PurePosixPath(value[name]['path']).parent == PurePosixPath('/operator'), 'teacher DATA must be flat operator input')
    require(value['execution_manifest'] is None, 'first reference operation has no execution-profile manifest')
    model = value['model_assets']
    require(type(model) is list and len(model) == 2, 'exact two target shards required; no drafter/media')
    for item in model:
        identity(item)
        require(item['path'].startswith('/models/hf/hub/'), 'target must retain canonical absolute model namespace')
    require(max(item['bytes'] for item in model) <= 64 << 30 and sum(item['bytes'] for item in model) <= 96 << 30,
            'first teacher target content pass exceeds declared region')
    paths = [item['path'] for item in model]
    first = PurePosixPath(paths[0])
    require(paths[0] == value['model_entrypoint'] and paths[1] != paths[0]
            and first.parent == PurePosixPath(paths[1]).parent
            and first.name.endswith('-00001-of-00002.gguf')
            and PurePosixPath(paths[1]).name == first.name.replace('-00001-of-00002.gguf', '-00002-of-00002.gguf'),
            'original ordered target shard names required')
    require(absolute(value['frontend_root']) and PurePosixPath(value['frontend_root']).parent == PurePosixPath('/frontend')
            and sha(PurePosixPath(value['frontend_root']).name), 'prepared frontend key must be under separate RO /frontend')
    assets = value['frontend_assets']
    require(type(assets) is list and len(assets) == len(FRONTEND_NAMES), 'complete prepared frontend closure required')
    for item in assets:
        identity(item)
        require(PurePosixPath(item['path']).parent == PurePosixPath(value['frontend_root']), 'frontend asset outside prepared directory')
    require(sum(item['bytes'] for item in assets) <= 64 << 20, 'prepared frontend closure exceeds first operation limit')
    require({PurePosixPath(item['path']).name for item in assets} == FRONTEND_NAMES, 'frontend filenames differ')
    return value


def validate_sources(value):
    require(type(value) is dict and REQUIRED_TOOLS <= value.keys() and len(value) <= 128, 'bounded canonical operator closure required')
    for name, item in value.items():
        require(relative(name) and name.startswith('tools/') and name.endswith('.py')
                and type(item) is dict and set(item) == {'bytes', 'sha256'}
                and type(item['bytes']) is int and item['bytes'] > 0 and sha(item['sha256']), 'canonical tool source identity required')
    return value


def qualifier_argv(operation, client, uuids, output):
    validate_operation(operation)
    validate_teacher_client(client)
    require(type(uuids) is list and len(uuids) == 8 and len(set(uuids)) == 8
            and all(type(value) is str and re.fullmatch(r'GPU-[0-9a-f-]{36}', value) for value in uuids), 'exact eight GPU UUIDs required')
    require(absolute(output), 'absolute fresh teacher output required')
    return ['--gpus', ','.join(uuids), '--master-port', '29551', '--out', output,
            '--teacher-plan', operation['teacher_plan']['path'], '--teacher-client', operation['teacher_client']['path'],
            '--', operation['model_entrypoint'], '--backend', 'cuda', '--context', '262144', '--parallel', '24',
            '--kv-full-windows', '24', '--kv-dtype', 'int8', '--indexer-history', 'gpu',
            '--expert-tensor-parallel-size', '4', '--draft-method', 'none', '--no-graphs', '--no-prefill-fp8',
            '--no-thinking', '--host', '127.0.0.1', '--port', '8000', '--name', client['model'],
            '--max-pending-requests', '24', '--max-http-connections', '25', '--max-waiting-input-bytes', '226493088',
            '--output-queue-tokens', '32', '--max-tokens', '64', '--no-update-check']


def validate_model_result(value):
    fields = {'schema', 'teacher_plan_sha256', 'teacher_client_sha256', 'launch_exit', 'client_report',
              'teacher_authority', 'rank_receipts', 'startup_receipts', 'child_native_records'}
    require(type(value) is dict and set(value) == fields and value['schema'] == RESULT_SCHEMA,
            'closed typed teacher result required')
    require(sha(value['teacher_plan_sha256']) and sha(value['teacher_client_sha256'])
            and type(value['launch_exit']) is int and value['launch_exit'] == 0, 'teacher inputs and positive launcher completion required')
    for key in ('client_report', 'teacher_authority'):
        identity(value[key], key='file')
        require(value[key]['file'] == 'teacher/' + ('teacher-client-report.json' if key == 'client_report' else 'teacher-authority.json'),
                'fixed teacher result artifact path required')
    for key in ('rank_receipts', 'startup_receipts', 'child_native_records'):
        rows = value[key]
        require(type(rows) is list and len(rows) == 8, 'complete eight-rank teacher result required')
        for rank, row in enumerate(rows):
            identity(row, key='file', rank=True)
            require(row['rank'] == rank, 'teacher result rank order differs')
            pattern = (f'teacher/teacher-rank{rank}/summary.json' if key == 'rank_receipts' else
                       f'teacher/rank-{rank}.json' if key == 'startup_receipts' else None)
            require(row['file'] == pattern if pattern is not None else re.fullmatch(
                rf'child-native/teacher-closed-rank-{rank}-pid-[1-9][0-9]*\.json', row['file']) is not None,
                'teacher result artifact role/rank path differs')
        require(len({row['file'] for row in rows}) == 8, 'teacher result files must be distinct')
    return value


SERVICE = {'context_tokens': 262144, 'resident_capacity': 24, 'configured_ceiling': 24,
           'kv_dtype': 'int8', 'kernel_family': 'torch-reference', 'graph_family': 'uncaptured',
           'tp': 8, 'pp': 1, 'etp': 4, 'ep': 2}
MODEL_OBSERVER_LIMITS = {'map_lines': 65536, 'line_characters': 16384, 'elfs': 64,
                         'segments': 256, 'mounts': 4096, 'path_characters': 4096,
                         'elf_bytes': 256 << 20, 'total_elf_bytes': 1 << 30, 'modules': 16384}


def validate_teacher_client(value):
    keys = {'schema', 'teacher_plan_sha256', 'base', 'model', 'expected_service',
            'readiness_timeout_s', 'whole_timeout_s', 'socket_timeout_s', 'response_bytes'}
    require(type(value) is dict and set(value) == keys
            and value['schema'] == 'tensorfold.owned-teacher-client-v1', 'closed teacher client DATA required')
    require(sha(value['teacher_plan_sha256']) and type(value['model']) is str
            and 0 < len(value['model']) <= 256 and '\0' not in value['model'], 'teacher plan and served model identity required')
    require(type(value['base']) is str, 'literal loopback HTTP base required')
    parsed = urllib.parse.urlsplit(value['base'])
    require(parsed.scheme == 'http' and parsed.hostname == '127.0.0.1' and parsed.username is None
            and parsed.password is None and parsed.path in ('', '/') and not parsed.query and not parsed.fragment
            and parsed.netloc == '127.0.0.1:' + str(parsed.port) and parsed.port == 8000,
            'fixed private loopback teacher service port required')
    service = value['expected_service']
    require(type(service) is dict and set(service) == set(SERVICE)
            and all(type(service[k]) is type(v) and service[k] == v for k, v in SERVICE.items()),
            'first teacher service profile differs')
    for key in ('readiness_timeout_s', 'whole_timeout_s', 'socket_timeout_s'):
        require(type(value[key]) in (int, float) and math.isfinite(value[key]) and value[key] > 0,
                'finite positive teacher client timeouts required')
    require(value['readiness_timeout_s'] <= 3600 and value['socket_timeout_s'] <= 300
            and value['whole_timeout_s'] <= 7500 and value['readiness_timeout_s'] <= value['whole_timeout_s']
            and value['socket_timeout_s'] <= value['whole_timeout_s'], 'teacher client deadline exceeds operation envelope')
    require(type(value['response_bytes']) is int and value['response_bytes'] == 1048576, 'fixed bounded client response required')
    return value


def child_observer_resources():
    # CPython3.12 observation-only reservation: two <=16MiB maps texts with
    # <=65536 bounded split lines, <=64 ELF /256 VMA records, <=4096 mounts,
    # 1MiB hash chunk and bounded 4MiB serialized record. The 512MiB allowance
    # includes simultaneous parsed/text/JSON forms; runtime allocator RSS is a
    # separate measured property, not established by this source projection.
    return {'schema': 'tensorfold.teacher-child-observer-resources-v1',
            'world': 8, 'per_rank_host_bytes': 512 << 20, 'global_host_bytes': 8 * (512 << 20),
            'record_bytes': 4 << 20, 'limits': dict(MODEL_OBSERVER_LIMITS),
            'projection': {'worker_controlled_peak_bytes': 400 << 20, 'parent_overlap_bytes': 256 << 20,
                           'simultaneous_controlled_bytes': (8 * 400 + 256) << 20,
                           'scope': 'Conservative bounded CPython3.12 object/serialization projection; not physical RSS.'}}


def validate_host_operation(host, inner):
    """Bind only the declared first-model envelope; old suite policy is separate."""
    validate_operation(inner['operation'])
    validate_sources(inner['source_files'])
    require(inner['mode'] == MODE and inner['schema'] == SCHEMA and len(host['uuids']) == 8
            and inner['UUIDs'] == host['uuids'] and inner['SM'] == [7, 0]
            and inner['image_id'] == host['image_id'] and 'verification_launcher' in host,
            'model host/image/eight-device/capability contract differs')
    r, t = host['resources'], host['timeouts']
    require(r['nano_cpus'] == 64 * 10**9 and r['memory_bytes'] == 340 << 30
            and r['memory_swap_bytes'] == r['memory_bytes'] and r['ipc'] == 'host'
            and t['run'] == 7800 and t['retire'] == 90, 'fixed first model resource/deadline envelope differs')
    mounts = host['mounts']
    frontend = [m for m in mounts if m['target'] == '/frontend']
    require(len(frontend) == 1 and frontend[0]['read_only'] is True
            and host['environment'].get('TF_GGUF_ASSETS_DIR') == '/frontend', 'separate readonly prepared frontend required')
    assets = [*inner['operation']['model_assets'], *inner['operation']['frontend_assets']]
    for item in assets:
        owners = [m for m in mounts if PurePosixPath(item['path']).is_relative_to(PurePosixPath(m['target']))]
        require(len(owners) == 1 and owners[0]['read_only'] is True, 'model/frontend asset lacks unique readonly bind owner')
    for key in ('TENSORFOLD_PRIVATE_SM70_MOE_PARTIAL', 'TENSORFOLD_PRIVATE_SM70_ROUTER_LOGITS',
                'TENSORFOLD_PRIVATE_SM70_EMPTY_PREFIX', 'TENSORFOLD_PRIVATE_SM70_GROUPED_DOWN'):
        require(host['environment'].get(key) in (None, 'inherit'), 'first teacher operation must inherit default selectors')
    require(host['environment'].get('TENSORFOLD_SM70_PROFILE', 'reference') == 'reference', 'first teacher profile must remain reference')
    return inner


def validate_host_result(value, expected):
    fields = {'schema', 'mode', 'source_commit', 'image_id_declared', 'operation', 'source_after_match',
              'content_binding_complete', 'content_binding', 'parent_cuda_uninitialized', 'model_result', 'primary_error', 'post_errors', 'scope'}
    require(type(value) is dict and set(value) == fields and value['schema'] == HOST_RESULT_SCHEMA
            and value['mode'] == MODE and value['operation'] == expected['operation']
            and value['source_commit'] == expected['source_commit'] and value['image_id_declared'] == expected['image_id']
            and value['source_after_match'] is True and value['content_binding_complete'] is True
            and value['parent_cuda_uninitialized'] is True and value['primary_error'] is None
            and value['post_errors'] == [], 'owned teacher operation did not complete its owning contract')
    identity(value['content_binding'], key='file')
    require(value['content_binding']['file'] == 'model-content-binding.json', 'fixed owned content binding record required')
    result = validate_model_result(value['model_result'])
    require(result['teacher_plan_sha256'] == expected['operation']['teacher_plan']['sha256']
            and result['teacher_client_sha256'] == expected['operation']['teacher_client']['sha256'],
            'owned teacher final result has different input identity')
    return value

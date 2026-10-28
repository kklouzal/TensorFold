"""GPU-local CPU scheduling for one owned TP8 child, before its backend imports.

This does not set memory policy, migrate pages or modify the supervisor. The
same exec entrypoint accepts a finite probe or an ordinary external TP8 CLI.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

# The held package initializer/error transport are pure stdlib, with no Torch.
# The exact staged import preflight verifies this boundary before any launch.
from tensorfold.cleanup import raise_failures

SCHEMA = 'tensorfold.gpu-local-worker-affinity-v1'
MAX_DOCUMENT = 1 << 20
MAX_THREADS = 4096  # The held qualification container's existing total PID limit.
MAX_RECEIPT = 256 << 10  # Existing per-follower diagnostic receipt bound.


def cpu_ids(text):
    result = set()
    for item in text.strip().split(','):
        if not re.fullmatch(r'[0-9]+(?:-[0-9]+)?', item):
            raise ValueError('invalid CPU list')
        pair = item.split('-')
        first, last = int(pair[0]), int(pair[-1])
        if not 0 <= first <= last < MAX_THREADS:
            raise ValueError('CPU list exceeds the held profile')
        result.update(range(first, last + 1))
    if not result:
        raise ValueError('empty CPU list')
    return result


def cpu_list(cpus):
    values = sorted(cpus)
    if not values:
        return ''
    runs, first, last = [], values[0], values[0]
    for value in values[1:]:
        if value != last + 1:
            runs.append(str(first) if first == last else f'{first}-{last}')
            first = value
        last = value
    runs.append(str(first) if first == last else f'{first}-{last}')
    return ','.join(runs)


def pci_id(value):
    match = re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', value)
    if match is None or int(match[1], 16) > 0xffff:
        raise ValueError('invalid PCI identity')
    return f'{int(match[1], 16):04x}:{match[2].lower()}:{match[3].lower()}.{match[4]}'


def file_operation(path, mode, operation):
    errors = [None]  # Reserve the cleanup slot before acquiring the file.
    primary = cause = context = None
    suppressed, result = False, None
    target = Path(path).open(mode)
    try:
        result = operation(target)
    except BaseException as error:
        primary = error
        cause = BaseException.__cause__.__get__(primary)
        context = BaseException.__context__.__get__(primary)
        suppressed = BaseException.__suppress_context__.__get__(primary)
    try:
        target.close()
    except BaseException as error:
        errors[0] = error
    if primary is not None:
        BaseException.__cause__.__set__(primary, cause)
        BaseException.__context__.__set__(primary, context)
        BaseException.__suppress_context__.__set__(primary, suppressed)
    if errors[0] is not None:
        raise_failures(primary, errors)
    if primary is not None:
        raise_failures(primary, ())
    return result


def policy(path, expected_sha):
    raw = file_operation(path, 'rb', lambda source: source.read(MAX_DOCUMENT + 1))
    if len(raw) > MAX_DOCUMENT or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('worker topology policy size or hash differs')
    value = json.loads(raw)
    ranks = value.get('ranks')
    if value.get('schema') != SCHEMA or type(ranks) is not list or len(ranks) != 8:
        raise ValueError('worker topology requires its explicit eight-rank profile')
    uuids, buses = set(), set()
    for rank, row in enumerate(ranks):
        if (row.get('rank') != rank or type(row.get('rank')) is not int
                or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', row.get('gpu_uuid', ''))
                or type(row.get('numa_node')) is not int or row['numa_node'] < 0):
            raise ValueError('ambiguous rank GPU or NUMA identity')
        bus = pci_id(row['pci_bus_id'])
        cpus = cpu_ids(row['local_cpulist'])
        if row['gpu_uuid'] in uuids or bus in buses or cpu_list(cpus) != row['local_cpulist']:
            raise ValueError('duplicate GPU or noncanonical local CPU policy')
        uuids.add(row['gpu_uuid'])
        buses.add(bus)
    return value


def gpu_devices():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True, timeout=30)
    if len(result.stdout) > 65536:
        raise ValueError('GPU identity response exceeds its bounded profile')
    devices = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) != 3 or not fields[0].isdecimal():
            raise ValueError('invalid GPU identity response')
        devices.append({'index': fields[0], 'gpu_uuid': fields[1], 'pci_bus_id': pci_id(fields[2])})
    return devices


def local_topology(bus):
    node = int((Path('/sys/bus/pci/devices') / bus / 'numa_node').read_text())
    if node < 0:
        raise ValueError('GPU PCI device has no authoritative local NUMA node')
    cpus = cpu_ids((Path('/sys/devices/system/node') / f'node{node}' / 'cpulist').read_text())
    online = cpu_ids(Path('/sys/devices/system/cpu/online').read_text())
    if not cpus <= online:
        raise ValueError('GPU-local CPU topology includes offline CPUs')
    return {'numa_node': node, 'local_cpulist': cpu_list(cpus), 'online_cpulist': cpu_list(online)}


def process_identity():
    raw = Path('/proc/self/stat').read_text()
    return {'pid': os.getpid(), 'parent_pid': os.getppid(), 'session': os.getsid(0),
            'start_ticks': int(raw.rsplit(') ', 1)[1].split()[19])}


def thread_ids():
    tids = sorted(int(p.name) for p in Path('/proc/self/task').iterdir() if p.name.isdecimal())
    if len(tids) > MAX_THREADS:
        raise ValueError('thread census exceeds held total PID capacity')
    return tids


def thread_snapshot(allowed=None):
    tids = thread_ids()
    rows, started = [], time.monotonic_ns()
    for tid in tids:
        try:
            cpus = os.sched_getaffinity(tid)
            name = (Path('/proc/self/task') / str(tid) / 'comm').read_text().rstrip('\n')
        except (FileNotFoundError, ProcessLookupError):
            rows.append({'tid': tid, 'retired_before_read': True})
            continue
        if allowed is not None and not cpus <= allowed:
            raise RuntimeError(f'worker helper {tid} escaped its GPU-local allowed CPUs')
        rows.append({'tid': tid, 'name': name, 'cpus': cpu_list(cpus)})
    return {'started_monotonic_ns': started, 'ended_monotonic_ns': time.monotonic_ns(), 'threads': rows,
            'scope': 'bounded boundary census; vanished threads recorded; not continuous all-time affinity proof'}


def write_receipt(path, value):
    raw = (json.dumps(value, indent=1, allow_nan=False) + '\n').encode()
    if len(raw) > MAX_RECEIPT:
        raise ValueError('worker affinity evidence exceeds existing follower receipt bound')
    def write(target):
        if target.write(raw) != len(raw):
            raise OSError('short worker affinity receipt write')
    file_operation(path, 'xb', write)


def execute(args):
    if 'torch' in sys.modules:
        raise RuntimeError('GPU-local bootstrap must precede Torch/backend imports')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        raise ValueError('worker exec requires an explicit command')
    document = policy(args.policy, args.policy_sha)
    row = document['ranks'][args.rank]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    devices = gpu_devices()
    if len(devices) != 8 or len({d['index'] for d in devices}) != 8:
        raise ValueError('the held GPU profile requires eight distinct visible devices')
    if {(d['gpu_uuid'], d['pci_bus_id']) for d in devices} != {
            (r['gpu_uuid'], pci_id(r['pci_bus_id'])) for r in document['ranks']}:
        raise ValueError('GPU UUID/PCI topology changed from its held policy')
    selected_gpu = [d for d in devices if visible in (d['index'], d['gpu_uuid'])]
    if len(selected_gpu) != 1 or selected_gpu[0]['gpu_uuid'] != row['gpu_uuid']:
        raise ValueError('worker visible GPU is ambiguous or differs from its rank')
    bus = pci_id(row['pci_bus_id'])
    topology = local_topology(bus)
    if topology['numa_node'] != row['numa_node'] or topology['local_cpulist'] != row['local_cpulist']:
        raise ValueError('GPU-local NUMA topology changed from its held policy')
    before = os.sched_getaffinity(0)
    selected = before & cpu_ids(topology['local_cpulist'])
    if not selected:
        raise ValueError('GPU locality and inherited CPU allowance have empty intersection')
    if len(thread_ids()) != 1:
        raise RuntimeError('bind the fresh worker before helper threads exist')
    identity = process_identity()
    if args.mode == 'bind':
        os.sched_setaffinity(0, selected)
    expected = selected if args.mode == 'bind' else before
    if os.sched_getaffinity(0) != expected or local_topology(bus) != topology:
        raise RuntimeError('worker affinity or topology changed during initialization')
    if gpu_devices() != devices:
        raise RuntimeError('GPU UUID/PCI enumeration changed during initialization')
    borrowed = os.environ.get('TENSORFOLD_TP_FAILURE_FD')
    if borrowed is not None:
        if not borrowed.isdecimal() or not stat.S_ISFIFO(os.fstat(int(borrowed)).st_mode) or not os.get_inheritable(int(borrowed)):
            raise ValueError('supervised failure pipe must survive the worker exec')
    receipt = {'schema': SCHEMA, 'rank': args.rank, 'mode': args.mode, 'policy_sha256': args.policy_sha,
               'identity': identity, 'gpu': selected_gpu[0], 'topology': topology,
               'inherited_cpus': cpu_list(before), 'selected_cpus': cpu_list(selected), 'actual_cpus': cpu_list(expected),
               'input_cuda_visible_devices': visible, 'exec_cuda_visible_devices': row['gpu_uuid'],
               'torch_loaded_before_bind': False, 'failure_fd': None if borrowed is None else int(borrowed),
               'bootstrap_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'snapshot': thread_snapshot(selected if args.mode == 'bind' else None),
               'memory_policy': 'unchanged; no page placement or migration claimed'}
    write_receipt(args.receipt, receipt)
    # CUDA's numeric ordinal order need not equal nvidia-smi/NVML order.
    # Select the resolved held UUID explicitly before any CUDA import.
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=row['gpu_uuid'])
    os.execvpe(command[0], command, environment)


def sample(receipt_path, *, rank, policy_sha, mode):
    raw = file_operation(receipt_path, 'rb', lambda source: source.read(MAX_RECEIPT + 1))
    if len(raw) > MAX_RECEIPT:
        raise ValueError('worker affinity bootstrap receipt exceeds bound')
    receipt = json.loads(raw)
    identity = process_identity()
    if (receipt['schema'] != SCHEMA or receipt['rank'] != rank or receipt['policy_sha256'] != policy_sha
            or mode not in ('bind', 'observe-reference') or receipt['mode'] != mode
            or os.environ.get('CUDA_VISIBLE_DEVICES') != receipt['exec_cuda_visible_devices']
            or receipt['exec_cuda_visible_devices'] != receipt['gpu']['gpu_uuid']
            or any(receipt['identity'][key] != identity[key] for key in ('pid', 'session', 'start_ticks'))
            or receipt['bootstrap_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            or local_topology(receipt['gpu']['pci_bus_id']) != receipt['topology']):
        raise RuntimeError('worker affinity receipt/process/topology identity changed')
    if os.sched_getaffinity(0) != cpu_ids(receipt['actual_cpus']):
        raise RuntimeError('worker main-thread affinity changed after exec')
    allowed = cpu_ids(receipt['selected_cpus']) if receipt['mode'] == 'bind' else None
    return {'identity': identity, 'bootstrap': str(receipt_path), 'policy_sha256': policy_sha,
            'mode': receipt['mode'], 'gpu': receipt['gpu'], 'topology': receipt['topology'],
            'selected_cpus': receipt['selected_cpus'], 'snapshot': thread_snapshot(allowed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--policy-sha', required=True)
    parser.add_argument('--rank', type=int, choices=range(8), required=True)
    parser.add_argument('--mode', choices=('bind', 'observe-reference'), default='bind')
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    failures = [None]
    try:
        execute(args)
    except BaseException as primary:
        cause = BaseException.__cause__.__get__(primary)
        context = BaseException.__context__.__get__(primary)
        suppressed = BaseException.__suppress_context__.__get__(primary)
        if 'TENSORFOLD_TP_FAILURE_FD' in os.environ:
            try:
                # Only the failure path imports this existing pure-stdlib protocol.
                from tensorfold.cuda.cohort import notify_failure
                notify_failure()
            except BaseException as secondary:
                failures[0] = secondary
        BaseException.__cause__.__set__(primary, cause)
        BaseException.__context__.__set__(primary, context)
        BaseException.__suppress_context__.__set__(primary, suppressed)
        if failures[0] is not None:
            raise_failures(primary, failures)
        raise


if __name__ == '__main__':
    main()

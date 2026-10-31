"""Private, finite stop/profile/retire/restore transaction; execute only through gpu-queue.

The existing guard owns its producer lease and GPU container. This controller
preserves the inherited queue lock in the guard, never kills a quarantined
guardian, and never restores production from an unproved retirement outcome.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

# Canonical tools and hash-bound packet consumers provide this same shared IO
# module and exact project-owned cleanup source. No installed model is imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gb10_io import opened_file, owned_descriptor, raise_failures

EVIDENCE = Path('/home/kklouzal/tf-campaign/experiments/performance-resume-20261021')
TOOLS = Path('/home/kklouzal/tf-campaign/bin')
LOCK = Path('/home/kklouzal/tf-campaign/gpu.lock')
HOLD = Path('/home/kklouzal/tf-campaign/gpu-retirement-hold.json')
LABEL = 'org.tensorfold.installed-native'
SOURCE = '259b4238ae16d8019c106603f104b18cbffb385a'
IMAGE = 'sha256:244312373625cacc8bd82af1372b058eec8af5d19ee217a6bc628b62b027896d'
HOT = '/assets/guard-state/hot'
LIMIT = 1 << 20
INTERRUPTION_AUTHORIZATION = 'you may interrupt'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def raise_transaction_failures(message, *values):
    failures = []
    for value in values:
        if value is not None and not any(value is prior for prior in failures):
            failures.append(value)
    raise BaseExceptionGroup(message, failures) from None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path, maximum=LIMIT, expected_sha256=None):
    with Path(path).open('rb') as stream:
        raw = stream.read(maximum + 1)
    require(len(raw) <= maximum, f'JSON exceeds bound: {path}')
    if expected_sha256 is not None:
        require(hashlib.sha256(raw).hexdigest() == expected_sha256, f'JSON snapshot digest differs: {path}')
    return json.loads(raw)


def absent(path):
    try:
        Path(path).lstat()
    except FileNotFoundError:
        return True
    return False  # Other inspection errors propagate; absence is never guessed.


def publish(path, value):
    """Retain the original durable JSON publication with primary-safe owners."""
    temporary = path.with_suffix(path.suffix + '.new')
    with opened_file(temporary, 'x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o600)
    os.replace(temporary, path)
    with owned_descriptor(path.parent, os.O_RDONLY | os.O_DIRECTORY) as fd:
        os.fsync(fd)


def canonical_configuration(container):
    require(type(container['Config']) is dict and type(container['HostConfig']) is dict,
            'container configuration sections must be objects')
    mounts = container['Mounts']
    require(type(mounts) is list and all(type(row) is dict and type(row.get('Destination')) is str for row in mounts),
            'mount identity requires complete entries with string destinations')
    require(len({row['Destination'] for row in mounts}) == len(mounts), 'duplicate mount destination')
    host_config = container['HostConfig'].copy()
    ulimits = host_config.get('Ulimits')
    if ulimits is not None:
        require(type(ulimits) is list and all(type(row) is dict and type(row.get('Name')) is str
                and row['Name'] for row in ulimits), 'ulimit identity requires complete entries with nonempty string names')
        require(len({row['Name'] for row in ulimits}) == len(ulimits), 'duplicate ulimit name')
        host_config['Ulimits'] = sorted(ulimits, key=lambda row: row['Name'])
    # Moby inspect appends daemon-default ulimits from a map; Mounts also
    # originates from a map. Only these two keyed lists are order-independent.
    # Preserve every entry field, absent/null distinctions and other array order.
    return {'Config': container['Config'], 'HostConfig': host_config,
            'Mounts': sorted(mounts, key=lambda row: row['Destination'])}


def fingerprint(container):
    return digest(canonical_configuration(container))


def inspection_metadata(container):
    """Retain comparison identities, never literal environment/label values."""
    config = canonical_configuration(container)
    record = {'schema_version': 1, 'cid': container['Id'], 'image': container['Image'],
              'canonical_fingerprint': digest(config),
              'Config': {name: digest(value) for name, value in config['Config'].items()},
              'HostConfig': {name: digest(value) for name, value in config['HostConfig'].items()},
              'Mounts': {row['Destination']: digest(row) for row in config['Mounts']}}
    require(len(json.dumps(record).encode()) <= LIMIT, 'inspection metadata exceeds1MiB')
    return record


def queue_descriptors():
    owner = os.environ.get('TFC_LOCK_HELD', '')
    require(owner.isascii() and owner.isdecimal(), 'run the controller through gpu-queue')
    pid, ancestors = os.getpid(), set()
    while pid > 1 and pid not in ancestors:
        ancestors.add(pid)
        pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
    require(int(owner) in ancestors, 'FIFO owner is not this process or its ancestor')
    require(not os.environ.get('TFC_OWNED_CONTAINER_LABEL') and
            not os.environ.get('TFC_OWNED_CONTAINER_LEASE_FD'), 'do not nest this transaction in another owned guard')
    identity = LOCK.stat()
    descriptors = []
    for entry in Path('/proc/self/fd').iterdir():
        try:
            info = os.fstat(int(entry.name))
        except OSError:
            continue
        if (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino):
            fcntl.flock(int(entry.name), fcntl.LOCK_EX | fcntl.LOCK_NB)
            descriptors.append(int(entry.name))
    require(descriptors, 'the inherited GPU admission lock descriptor is missing')
    # A separate open description must be excluded by our retained lock.
    with LOCK.open('a') as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError('exclusive GPU admission is not established')
    return tuple(descriptors)


class Host:
    def __init__(self, directory, descriptors=()):
        self.directory, self.calls = directory, []
        self.uncertain = False
        self.lifecycle_uncertain = False
        self.descriptors, self.pending_clients = descriptors, []

    def command(self, argv, timeout=30):
        """Bound the original client; preserve command and independent IO roots."""
        record = {'argv': argv, 'timeout_s': timeout, 'complete': False}
        self.calls.append(record)
        lifecycle = argv[:2] in (['docker', 'stop'], ['docker', 'start'])
        prior_lifecycle_uncertainty = self.lifecycle_uncertain
        if lifecycle:
            self.lifecycle_uncertain = True
        process = selector = None
        buffers = {'stdout': bytearray(), 'stderr': bytearray()}
        deadline = time.monotonic() + timeout
        primary, result = None, None
        prior, errors = [None, None, None], [None] * 13

        def failed(index, error):
            nonlocal primary
            errors[index] = error
            if primary is None:
                primary = error
                prior[0] = BaseException.__cause__.__get__(error)
                prior[1] = BaseException.__context__.__get__(error)
                prior[2] = BaseException.__suppress_context__.__get__(error)

        try:
            process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=self.descriptors)
            record['client_pid'] = process.pid
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ, 'stdout')
            selector.register(process.stderr, selectors.EVENT_READ, 'stderr')
            while selector.get_map():
                remaining = deadline - time.monotonic()
                require(remaining > 0, 'bounded Docker client timed out')
                for key, _ in selector.select(min(remaining, 0.5)):
                    chunk = os.read(key.fileobj.fileno(), min(65536, LIMIT + 1 - len(buffers[key.data])))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    buffers[key.data].extend(chunk)
                    require(len(buffers[key.data]) <= LIMIT, 'bounded Docker client output exceeded1MiB')
            code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            # Close the selector before accepting the command as before. A
            # failed close is not retried; its raw owner stays traceback-owned.
            owner, selector = selector, None
            owner.close()
            record['exit'] = code
            require(code == 0, f'Docker client failed with exit{code}; private payload not emitted')
            record['complete'] = True
            result = bytes(buffers['stdout'])
        except BaseException as error:  # noqa: BLE001 - Retain the command root through every independent cleanup.
            failed(0, error)
            try:
                record['complete'] = False
                self.uncertain = True
                if lifecycle:
                    self.lifecycle_uncertain = True
                if process is None:
                    record['client_publication_unproven'] = True
            except BaseException as status_error:  # noqa: BLE001 - Retain status failure without losing the command root.
                failed(1, status_error)
            if selector is not None:
                owner, selector = selector, None
                try:
                    owner.close()
                except BaseException as close_error:  # noqa: BLE001 - Retire the client even if selector close fails.
                    failed(2, close_error)
            if process is not None:
                try:
                    process.kill()
                except BaseException as kill_error:  # noqa: BLE001 - A failed kill cannot prevent the bounded wait.
                    failed(3, kill_error)
                try:
                    process.wait(timeout=5)
                except BaseException as wait_error:  # noqa: BLE001 - Any failed wait leaves retirement unproved.
                    # Timeout or another failed wait does not establish client
                    # retirement. Keep the same pending-client recovery owner.
                    failed(4, wait_error)
                    try:
                        record['client_retirement_unproven'] = True
                        self.pending_clients.append(process)
                    except BaseException as retention_error:  # noqa: BLE001 - Keep retention and wait failures together.
                        failed(5, retention_error)
        finally:
            try:
                record['stdout_sha256'] = hashlib.sha256(buffers['stdout']).hexdigest()
                record['stdout_bytes'] = len(buffers['stdout'])
            except BaseException as metadata_error:  # noqa: BLE001 - Fingerprint failure cannot skip either pipe close.
                failed(6, metadata_error)
            try:
                record['stderr_sha256'] = hashlib.sha256(buffers['stderr']).hexdigest()
                record['stderr_bytes'] = len(buffers['stderr'])
            except BaseException as metadata_error:  # noqa: BLE001 - Fingerprint failure cannot skip either pipe close.
                failed(7, metadata_error)
            if process is not None:
                try:
                    process.stdout.close()
                except BaseException as close_error:  # noqa: BLE001 - Close stderr and publish after a stdout-close failure.
                    failed(8, close_error)
                try:
                    process.stderr.close()
                except BaseException as close_error:  # noqa: BLE001 - Retain stderr retirement before publication.
                    failed(9, close_error)
            if primary is not None:
                try:
                    record['complete'] = False
                    self.uncertain = True
                    if lifecycle:
                        self.lifecycle_uncertain = True
                except BaseException as status_error:  # noqa: BLE001 - Retain incomplete-capture status failure.
                    failed(10, status_error)
            try:
                publish(self.directory / 'commands.json', self.calls)
            except BaseException as publication_error:  # noqa: BLE001 - Keep publication beside command and close roots.
                failed(11, publication_error)
            try:
                if primary is None and lifecycle and record['complete']:
                    self.lifecycle_uncertain = prior_lifecycle_uncertainty
                elif primary is not None:
                    record['complete'] = False
                    self.uncertain = True
                    if lifecycle:
                        self.lifecycle_uncertain = True
            except BaseException as status_error:  # noqa: BLE001 - A status error cannot replace an earlier failure.
                failed(12, status_error)
        if primary is not None:
            BaseException.__cause__.__set__(primary, prior[0])
            BaseException.__context__.__set__(primary, prior[1])
            BaseException.__suppress_context__.__set__(primary, prior[2])
            try:
                failures = [error for error in errors if error is not None]
            except BaseException as diagnostic:
                # Raw statuses and prior native links remain frame-owned if
                # preparing the adopted transport itself exhausts.
                raise primary from diagnostic
            raise_failures(primary, failures)
        return result

    def clients_retired(self):
        self.pending_clients[:] = [process for process in self.pending_clients if process.poll() is None]
        return not self.pending_clients

    def inspect(self, cid):
        records = json.loads(self.command(['docker', 'container', 'inspect', cid]))
        require(type(records) is list and len(records) == 1 and records[0]['Id'] == cid, 'exact CID inspection failed')
        return records[0]

    def health(self):
        with urllib.request.urlopen('http://127.0.0.1:8888/health', timeout=5) as response:
            raw = response.read(65537)
        require(len(raw) <= 65536, 'health response exceeds64KiB')
        return json.loads(raw)

    def affinity(self, pid):
        status = Path(f'/proc/{pid}/status').read_text()
        return next(line.split(':', 1)[1].strip() for line in status.splitlines()
                    if line.startswith('Cpus_allowed_list:'))

    def process_identity(self, pid):
        try:
            with Path(f'/proc/{pid}/stat').open('rb') as stream:
                raw = stream.read(8193)
        except FileNotFoundError:
            return None
        require(len(raw) <= 8192, 'process identity exceeds bound')
        head, tail = raw.rsplit(b')', 1)
        require(head.split(b'(', 1)[0].strip() == str(pid).encode(), 'process identity PID differs')
        fields = tail.split()
        require(len(fields) >= 20 and fields[19].isdigit(), 'process start identity is unavailable')
        return {'start_ticks': int(fields[19]), 'state': fields[0].decode('ascii')}

    def start_guard(self, argv, descriptors):
        # Guard Sink/quarantine owns monitoring/output bounds. Do not add a
        # host RLIMIT that could terminate this safety owner independently.
        with (self.directory / 'guard-launch.stdout').open('xb') as stdout, \
             (self.directory / 'guard-launch.stderr').open('xb') as stderr:
            return subprocess.Popen(argv,
                                    stdout=stdout, stderr=stderr, pass_fds=descriptors, start_new_session=True)


def idle(health):
    return (health.get('ok') is True and health.get('busy') is False and
            type(health.get('requests_running')) is int and health['requests_running'] == 0)


def interruption_health(health):
    require(type(health) is dict and health.get('ok') is True and type(health.get('busy')) is bool
            and type(health.get('requests_running')) is int and health['requests_running'] >= 0,
            'authorized interruption still requires typed healthy production status')
    return {key: health[key] for key in ('ok', 'busy', 'requests_running')}


def process_ids(raw, *, docker_top=False):
    lines = raw.decode('ascii').splitlines()
    if docker_top:
        require(lines and lines.pop(0).strip() == 'PID', 'exact Docker PID table header required')
    require(len(lines) <= 4096 and all(re.fullmatch(r'[1-9][0-9]{0,9}', line.strip()) for line in lines),
            'bounded positive process PID list required')
    result = [int(line.strip()) for line in lines]
    require(len(result) == len(set(result)), 'duplicate process PID observation')
    return result


def driver_ready(raw):
    fields = [line.split(':', 1) for line in raw.decode('utf-8').splitlines() if ':' in line]
    require([value.strip() for key, value in fields if key.strip() == 'Attached GPUs'] == ['1']
            and [value.strip() for key, value in fields if key.strip() == 'GPU Recovery Action'] == ['None'],
            'exact single-GPU Recovery Action=None required before the guarded workload')
    return {'attached_gpus': 1, 'gpu_recovery_action': 'None', 'query_sha256': hashlib.sha256(raw).hexdigest()}


def read_workload(packet, manifest):
    workload = read_json(packet / 'workload.json', expected_sha256=manifest['files']['workload.json'])
    require(type(workload) is dict and set(workload) == {'image', 'source_revision'} and
            type(workload['image']) is str and re.fullmatch(r'sha256:[0-9a-f]{64}', workload['image']) and
            type(workload['source_revision']) is str and re.fullmatch(r'[0-9a-f]{40}', workload['source_revision']),
            'workload requires exactly an immutable image and full source revision')
    command = read_json(packet / 'profile-command.json', expected_sha256=manifest['files']['profile-command.json'])
    require(type(command) is list and len(command) >= 3 and
            all(type(arg) is str and '\0' not in arg for arg in command) and
            command[:3] == ['/opt/harness/cuda_entrypoint.py', 'python3', '/assets/bootstrap.py'],
            'finite workload must use the installed CUDA entrypoint, Python and sealed bootstrap')
    return {**workload, 'command': command}


def native_options_preflight(packet, manifest, output):
    """Run the actual pinned producer parser before any production lifecycle action.

    The audited four-module import closure only defines Python objects. options()
    reads bounded JSON, hashes that same snapshot and stats declared mount paths;
    it performs no Docker, GPU, process, environment or filesystem mutation.
    Temporary module bindings bypass stale bytecode/cached foreign modules and
    are restored before returning from this single-threaded controller stage.
    """
    names = ('owned_container', 'owned_command', 'owned_environment', 'installed_native')
    snapshots = {}
    receipt = {'schema_version': 1, 'status': 'RUNNING', 'controller_stop_attempted': False,
               'expected_options_sha256': manifest['files']['profile-options.json'], 'helper_sha256': {}}
    try:
        for name in names:
            path = TOOLS / (name + '.py')
            with path.open('rb') as stream:
                raw = stream.read(LIMIT + 1)
            digest = hashlib.sha256(raw).hexdigest()
            require(len(raw) <= LIMIT and digest == manifest['helpers'][path.name],
                    'native preflight helper snapshot differs: ' + path.name)
            snapshots[name] = (path, raw)
            receipt['helper_sha256'][path.name] = digest
        missing = object()
        previous = {name: sys.modules.get(name, missing) for name in names}
        modules = {}
        try:
            for name in names:
                path, raw = snapshots[name]
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                modules[name] = sys.modules[name] = module
                exec(compile(raw, str(path), 'exec'), module.__dict__)  # noqa: S102 - exact pinned maintained source snapshot
            _config, digest = modules['installed_native'].options(packet / 'profile-options.json')
            require(digest == manifest['files']['profile-options.json'],
                    'actual native options snapshot differs from the reviewed packet')
            receipt.update(status='PASS', options_sha256=digest)
        finally:
            for name, original in previous.items():
                if original is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = original
    except BaseException as primary:
        try:
            receipt.update(status='FAIL', error=''.join(traceback.format_exception(primary)))
            publish(output, receipt)
        except BaseException as publication:  # noqa: BLE001 - retain parser and publication failures together
            raise BaseExceptionGroup('Native preflight and receipt publication failed', [primary, publication]) from None
        raise
    publish(output, receipt)  # A failed PASS publication also prevents any production stop.
    return receipt


def validate_assets(root, expected_hash):
    manifest = root / 'assets-manifest.json'
    data = read_json(manifest, expected_sha256=expected_hash)
    require(data.get('schema_version') == 1 and type(data.get('files')) is dict and 1 <= len(data['files']) <= 2048,
            'invalid bounded asset manifest')
    require('bootstrap.py' in data['files'], 'required sealed bootstrap asset missing')
    for name, sha in data['files'].items():
        path = root / name
        require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts
                and not name.startswith('guard-state/') and not path.is_symlink()
                and path.is_file() and path.resolve().is_relative_to(root.resolve())
                and type(sha) is str and re.fullmatch('[0-9a-f]{64}', sha), 'invalid asset identity')
        require(file_hash(path) == sha, f'profile asset changed: {name}')
    actual = set()
    for directory, folders, files in os.walk(root):
        parent = Path(directory)
        if parent == root:
            folders[:] = [name for name in folders if name != 'guard-state']
        require(all(not (parent / name).is_symlink() for name in folders), 'asset directory symlink is unsupported')
        for name in files:
            relative = str((parent / name).relative_to(root))
            if relative != 'assets-manifest.json':
                actual.add(relative)
                require(len(actual) <= 2048, 'asset file count exceeds bound')
    require(actual == set(data['files']), 'asset manifest must bind the complete immutable snapshot')
    template = 'gemma4-chat-template-google-4d7ae498.jinja'
    if template in data['files']:
        require(data['files'][template] == 'ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4',
                'template differs from production')
    return data


class Transaction:
    def __init__(self, packet, root, token, host, descriptors, workload, interruption_authorization=None):
        self.packet, self.root, self.token, self.host, self.descriptors = packet, root, token, host, descriptors
        self.workload = workload
        require(interruption_authorization in (None, INTERRUPTION_AUTHORIZATION), 'explicit user interruption authorization required')
        self.interruption_authorization = interruption_authorization
        self.production_gpu_retirement_verified = interruption_authorization is None
        self.baseline = read_json(packet / 'production-baseline.json')
        self.state = {'status': 'RUNNING', 'phase': 'preflight', 'owner_label': f'{LABEL}={token}',
                      'source_revision': SOURCE, 'image': IMAGE, 'restore_completed': False,
                      'workload_image': workload['image'], 'workload_source_revision': workload['source_revision']}
        self.guard = None
        self.guard_attempted = self.stopped = False
        self.stop_attempted = self.stop_verified = self.restore_attempted = False
        self.interrupted = None
        if interruption_authorization is not None:
            self.state.update(interruption_authorization={'source': 'user', 'text': interruption_authorization,
                              'scope': 'bounded performance window 2026-10-21'},
                              production_gpu_retirement_verified=False)

    def prepare_interruption(self):
        observations = []
        self.state['authorized_health_observations'] = observations
        for index in range(2):
            if index:
                time.sleep(1)
            health = self.host.health()
            observations.append({**interruption_health(health), 'health_sha256': digest(health),
                                 'observed_unix_ns': time.time_ns()})
        raw = self.host.command(['sudo', '-n', 'journalctl', '-k', '-n', '1', '--no-pager', '-o', 'json'], timeout=20)
        anchor = json.loads(raw)
        cursor = anchor.get('__CURSOR')
        require(type(cursor) is str and 1 <= len(cursor) <= 4096 and '\0' not in cursor,
                'exact kernel journal cursor required before interruption')
        roles = {}
        for role in ('gemma', 'chatterbox', 'parakeet'):
            pids = process_ids(self.host.command(['docker', 'top', self.baseline[role]['cid'], '-eo', 'pid'], timeout=20), docker_top=True)
            require(pids, 'running permanent container has no process membership')
            for pid in pids:
                require(pid not in roles, 'permanent container process memberships overlap')
                identity = self.host.process_identity(pid)
                require(identity is not None and identity['state'] not in ('Z', 'X'),
                        'live permanent container process start identity required')
                roles[pid] = {'role': role, **identity}
        compute = process_ids(self.host.command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], timeout=20))
        contexts = {}
        for pid in compute:
            require(pid in roles, 'unknown compute process before interruption')
            identity = self.host.process_identity(pid)
            require(identity is not None and identity['state'] not in ('Z', 'X')
                    and identity['start_ticks'] == roles[pid]['start_ticks'],
                    'CUDA PID start identity changed since container membership')
            role = roles[pid]['role']
            members = process_ids(self.host.command(['docker', 'top', self.baseline[role]['cid'], '-eo', 'pid'], timeout=20), docker_top=True)
            after_membership = self.host.process_identity(pid)
            require(pid in members and after_membership is not None
                    and after_membership['start_ticks'] == identity['start_ticks']
                    and after_membership['state'] not in ('Z', 'X'),
                    'CUDA process start identity is not bound across fresh exact-container membership')
            contexts[str(pid)] = {'role': role, **after_membership}
        snapshot = {'schema_version': 1, 'authorization': self.state['interruption_authorization'],
                    'health_observations': observations, 'compute_contexts': contexts, 'kernel_cursor': cursor,
                    'kernel_anchor_sha256': hashlib.sha256(raw).hexdigest(),
                    'driver': driver_ready(self.host.command(['nvidia-smi', '-q'], timeout=20))}
        publish(self.root / 'production-interruption-before.json', snapshot)
        self.interruption_snapshot = snapshot
        self.state['production_interruption_before_sha256'] = file_hash(self.root / 'production-interruption-before.json')

    def verify_interruption_retirement(self):
        actual = self.verify('gemma', False)
        require(type(actual['State'].get('Pid')) is int and actual['State']['Pid'] == 0,
                'stopped exact production container must have no running init PID')
        contexts = self.interruption_snapshot['compute_contexts']
        allowed = {pid: value for pid, value in contexts.items() if value['role'] != 'gemma'}
        production = {pid: value for pid, value in contexts.items() if value['role'] == 'gemma'}
        samples = []
        for index in range(2):
            if index:
                time.sleep(0.5)
            observed = process_ids(self.host.command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], timeout=20))
            require({str(pid) for pid in observed} == set(allowed),
                    'production/unknown/new compute context remains or a preserved peer context disappeared')
            identities = {}
            for pid in observed:
                key = str(pid)
                identity = self.host.process_identity(pid)
                require(identity is not None and identity['start_ticks'] == allowed[key]['start_ticks']
                        and identity['state'] not in ('Z', 'X'), 'peer CUDA process start identity changed')
                identities[key] = identity
            for key, before in production.items():
                identity = self.host.process_identity(int(key))
                require(identity is None or identity['start_ticks'] != before['start_ticks'] or identity['state'] in ('Z', 'X'),
                        'stopped production CUDA worker is still live')
            samples.append({'observed_unix_ns': time.time_ns(), 'peer_compute_contexts': identities})
        driver = driver_ready(self.host.command(['nvidia-smi', '-q'], timeout=20))
        cursor = self.interruption_snapshot['kernel_cursor']
        raw = self.host.command(['sudo', '-n', 'journalctl', '-k', '--cursor=' + cursor, '--no-pager', '-o', 'json'], timeout=20)
        records = [json.loads(line) for line in raw.splitlines()]
        require(1 <= len(records) <= 4096 and records[0].get('__CURSOR') == cursor,
                'kernel journal anchor is unavailable; new-fault interval is unproved')
        for record in records[1:]:
            message = record.get('MESSAGE')
            require(type(message) is str and not re.search(r'NV_ERR_NO_MEMORY|oom-killer|Out of memory|Xid|NVRM', message),
                    'new kernel GPU/memory fault or uninspectable kernel message after interruption admission')
        self.verify_peers()
        require(absent(HOLD), 'durable GPU hold appeared during production retirement')
        proof = {'schema_version': 1, 'production_cuda_workers': production, 'peer_context_samples': samples,
                 'driver': driver, 'kernel_records_after_anchor': len(records) - 1,
                 'kernel_interval_sha256': hashlib.sha256(raw).hexdigest(),
                 'before_sha256': self.state['production_interruption_before_sha256'],
                 'production_cuda_workers_retired': True, 'unknown_or_new_compute_processes': False}
        publish(self.root / 'production-gpu-retirement.json', proof)
        self.state.update(production_gpu_retirement_verified=True,
                          production_gpu_retirement_sha256=file_hash(self.root / 'production-gpu-retirement.json'))
        self.production_gpu_retirement_verified = True

    def checkpoint(self, phase):
        self.state['phase'] = phase
        publish(self.root / 'transaction.json', self.state)

    def inspect(self, name):
        require(name in ('gemma', 'chatterbox', 'parakeet'), 'inspection role is not a permanent container')
        actual = self.host.inspect(self.baseline[name]['cid'])
        record = inspection_metadata(actual)
        # At most three fixed role files, atomically replaced before identity or
        # state checks. Publication failures retain the existing owning policy.
        publish(self.root / f'inspection-{name}-latest.json', record)
        return actual, record['canonical_fingerprint']

    def verify(self, name, running):
        expected = self.baseline[name]
        actual, identity = self.inspect(name)
        require(actual['Image'] == expected['image'] and identity == expected['fingerprint'],
                f'{name} immutable container configuration changed')
        require(actual['State']['Running'] is running and not actual['State'].get('Restarting') and
                not actual['State'].get('OOMKilled'), f'{name} runtime state differs')
        if running:
            require(actual['State'].get('Health', {}).get('Status') == 'healthy', f'{name} is not healthy')
            if name == 'gemma':
                require(self.host.affinity(actual['State']['Pid']) == '5-9,15-19',
                        'production effective CPU partition changed')
        return actual

    def verify_peers(self):
        for name in ('chatterbox', 'parakeet'):
            self.verify(name, True)

    def interrupt(self, signum, _frame):
        if self.interrupted is None:
            self.interrupted = signum
            if self.guard is not None:
                self.guard.send_signal(signal.SIGTERM)

    def guard_argv(self):
        native = [sys.executable, str(TOOLS / 'installed_native.py'), '--image', self.workload['image'],
                  '--source-revision', self.workload['source_revision'], '--owner-token', self.token,
                  '--out', str(self.root / 'native/worker'), '--docker-options-json', str(self.packet / 'profile-options.json'),
                  '--timeout-s', '600', '--', *self.workload['command']]
        return [sys.executable, str(TOOLS / 'gpu_guard.py'), '--dir', str(self.root / 'guard'), '--name', 'gemma-profile',
                '--hot-flag', str(self.root / 'assets/guard-state/hot'), '--owned-container-label', f'{LABEL}={self.token}',
                '--timeout', '900', '--', *native]

    def retirement(self):
        require(self.guard is not None and self.guard.poll() is not None, 'guard producer is not retired')
        guard = read_json(self.root / 'guard/guard.json')
        native = read_json(self.root / 'native/worker/report.json')
        owned = guard.get('owned_retirement', {})
        require(guard.get('verdict') == 'ok' and guard.get('started') is True and
                owned.get('label') == f'{LABEL}={self.token}' and
                type(owned.get('version')) is int and owned['version'] == 1 and
                owned.get('retirement_proven') is True and owned.get('producer_retired') is True and
                owned.get('action_complete') is True and not guard.get('guard_error') and
                native.get('retirement_proven') is True and native.get('owner_label') == f'{LABEL}={self.token}' and
                native.get('image') == self.workload['image'] and
                native.get('source_revision') == self.workload['source_revision'],
                'guard/native clean exact-owned retirement is unproven')
        require(absent(HOLD), 'durable GPU hold remains; production stays stopped')
        remaining = self.host.command(['docker', 'ps', '-aq', '--no-trunc', '--filter', f'label={LABEL}={self.token}'])
        require(not remaining.strip(), 'the exact owned profile container remains')
        self.state['profile_status'] = native.get('status')
        self.state['guard_exit'] = self.guard.returncode
        self.state['guard_receipt_sha256'] = file_hash(self.root / 'guard/guard.json')
        self.state['native_receipt_sha256'] = file_hash(self.root / 'native/worker/report.json')
        self.state['retirement_verified'] = True
        return native.get('status') == 'PASS' and self.guard.returncode == 0

    def restore(self):
        require(not self.host.uncertain and absent(HOLD), 'uncertain control action or durable hold forbids restore')
        self.verify_peers()
        self.verify('gemma', False)
        self.checkpoint('restore-start')
        self.restore_attempted = True
        self.host.command(['docker', 'start', self.baseline['gemma']['cid']], timeout=60)
        deadline = time.monotonic() + 900  # Existing production health start-period, not a weakened GPU guard.
        while time.monotonic() < deadline:
            actual, identity = self.inspect('gemma')
            require(actual['Image'] == IMAGE and identity == self.baseline['gemma']['fingerprint'],
                    'restored production identity changed')
            require(actual['State']['Running'] is True and not actual['State'].get('OOMKilled'), 'production stopped during restore')
            if actual['State'].get('Health', {}).get('Status') == 'healthy':
                if idle(self.host.health()):
                    self.verify('gemma', True)
                    self.verify_peers()
                    self.state['restore_completed'] = True
                    self.checkpoint('restored-healthy')
                    return
            time.sleep(2)
        raise RuntimeError('same-CID production restore did not become healthy within900s')

    def quarantine_control(self):
        """Retain the existing queue admission; root supplies actual recovery observations.

        This is not an approval gate or another scheduler. No Docker action is
        retried here, and no recurring telemetry is appended. TERM/INT do not
        release an unresolved control actor. The root recovery observation must
        bind this exact call history and healthy restored permanent/voice CIDs.
        """
        self.state.update(admission_retained=True, manual_recovery_required=True,
                          control_calls_sha256=digest(self.host.calls))
        try:
            self.checkpoint('production-control-quarantine')
        except BaseException as error:
            self.state['quarantine_receipt_error'] = ''.join(traceback.format_exception(error))
            try:
                print(json.dumps({'transaction': self.state}), file=sys.stderr)
            except BaseException as diagnostic:
                self.state['quarantine_stderr_error'] = ''.join(traceback.format_exception(diagnostic))
        path = self.root / 'production-control-recovery.json'
        diagnosed = False
        while True:
            try:
                info = path.lstat()
                require(not path.is_symlink() and info.st_uid in {0, os.geteuid()} and not info.st_mode & 0o077,
                        'recovery observation must be private and owner-controlled')
                data = read_json(path, maximum=65536)
                require(type(data.get('schema_version')) is int and data['schema_version'] == 1 and
                        data.get('owner_token') == self.token and
                        data.get('control_calls_sha256') == self.state['control_calls_sha256'] and
                        data.get('control_actors_retired') is True and data.get('docker_lifecycle_reconciled') is True,
                        'root recovery observation does not prove this exact control operation')
                require(self.host.clients_retired(), 'known Docker client remains live')
                for name, expected in self.baseline.items():
                    observed = data.get('containers', {}).get(name, {})
                    require(all(observed.get(key) == expected[key] for key in ('cid', 'image', 'fingerprint')) and
                            observed.get('running') is True and observed.get('healthy') is True,
                            'root recovery has not restored the exact healthy permanent/voice container')
                require(data.get('gemma_effective_cpus') == '5-9,15-19' and idle(data.get('gemma_health', {}))
                        and absent(HOLD), 'root recovery is not idle, affinity-preserving and free of durable guard hold')
                self.state.update(control_recovery_canonical_sha256=digest(data), recovered_by_root=True,
                                  restore_completed=True, admission_retained=False, manual_recovery_required=False)
                return
            except FileNotFoundError:
                pass
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                if not diagnosed:
                    self.state['control_recovery_error'] = str(error)
                    try:
                        self.checkpoint('production-control-quarantine')
                    except OSError as receipt_error:
                        self.state['control_recovery_receipt_error'] = str(receipt_error)
                    diagnosed = True
            time.sleep(1)

    def run(self):
        profile_pass = False
        primary = None
        restore_failure = None
        primary_diagnostic = restore_diagnostic = None
        try:
            require(absent(HOLD), 'durable GPU hold exists')
            self.verify_peers()
            self.verify('gemma', True)
            if self.interruption_authorization is None:
                require(idle(self.host.health()), 'production has an active request; refusing stop')
                time.sleep(1)
                require(idle(self.host.health()), 'production did not remain idle across the1s observation window')
                self.state['idle_observation_window_s'] = 1
            else:
                self.prepare_interruption()
            require(self.interrupted is None, 'transaction interrupted before stop')
            self.checkpoint('stop-production')
            self.stop_attempted = True
            self.host.command(['docker', 'stop', '--time', '30', self.baseline['gemma']['cid']], timeout=60)
            self.stopped = True
            self.verify('gemma', False)
            self.stop_verified = True
            self.checkpoint('production-stopped')
            if self.interruption_authorization is not None:
                self.verify_interruption_retirement()
            require(self.interrupted is None, 'transaction interrupted before profile')
            self.checkpoint('guard-launching')
            self.guard_attempted = True  # A failed Popen publication is ambiguous: never infer no actor.
            self.guard = self.host.start_guard(self.guard_argv(), self.descriptors)
            if self.interrupted is not None:
                self.guard.send_signal(signal.SIGTERM)  # The signal may have arrived before handle publication.
            self.state['guard_pid'] = self.guard.pid
            self.checkpoint('profile-running')
            deadline = time.monotonic() + 1320  # 300s admission cooling +900s guard + bounded terminal headroom.
            while self.guard.poll() is None:
                require(not (self.root / 'guard/guard-quarantine.json').exists(), 'guard quarantined; retain monitor/lock and recover explicitly')
                require(time.monotonic() < deadline, 'guardian still live; retain monitor/lock and investigate explicitly')
                time.sleep(0.5)
            profile_pass = self.retirement()
        except BaseException as error:
            primary = error
            try:
                self.state['error'] = ''.join(traceback.format_exception(error))
            except BaseException as diagnostic:  # noqa: BLE001 - diagnosis cannot bypass stop/retirement containment
                primary_diagnostic = diagnostic
        try:
            # No unconditional finally restore. Unknown/live guardian, trips, holds and uncertain Docker actions remain stopped.
            if self.control_unproved():
                self.quarantine_control()
            else:
                safe = self.stopped and not self.host.uncertain and (not self.guard_attempted or self.state.get('retirement_verified') is True)
                if safe:
                    self.restore()
                elif self.stopped:
                    self.state['manual_recovery_required'] = True
        except BaseException as error:
            restore_failure = error
            primary = primary or error
            try:
                self.state['restore_error'] = ''.join(traceback.format_exception(error))
            except BaseException as diagnostic:  # noqa: BLE001 - diagnosis cannot bypass ambiguous-restore quarantine
                restore_diagnostic = diagnostic
            if self.control_unproved():
                self.quarantine_control()
        if self.interrupted is not None:
            self.state['interrupted_signal'] = self.interrupted
        self.state['status'] = 'PASS' if primary is None and self.interrupted is None and profile_pass and self.state['restore_completed'] else 'FAIL'
        try:
            self.checkpoint('complete' if self.state['restore_completed'] else 'held-or-manual-recovery')
        except BaseException as error:
            try:
                print(json.dumps({'transaction': self.state, 'receipt_error': ''.join(traceback.format_exception(error))}),
                      file=sys.stderr)
            except BaseException as diagnostic:
                raise_transaction_failures('Transaction and receipt/diagnostic publication failed',
                                           primary, restore_failure, primary_diagnostic, restore_diagnostic, error, diagnostic)
            if primary_diagnostic is not None or restore_diagnostic is not None:
                raise_transaction_failures('Transaction and failure diagnosis/publication failed',
                                           primary, restore_failure, primary_diagnostic, restore_diagnostic, error)
            return 1
        if primary_diagnostic is not None or restore_diagnostic is not None:
            raise_transaction_failures('Transaction and failure diagnosis failed',
                                       primary, restore_failure, primary_diagnostic, restore_diagnostic)
        return 0 if self.state['status'] == 'PASS' else 1

    def control_unproved(self):
        if self.state.get('recovered_by_root') is True and self.state['restore_completed']:
            return False
        return (self.host.lifecycle_uncertain or (self.stop_attempted and not self.stop_verified)
                or (self.stop_attempted and not self.production_gpu_retirement_verified)
                or (self.restore_attempted and not self.state['restore_completed']))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--packet-dir', type=Path, required=True)
    parser.add_argument('--packet-manifest-sha256', required=True)
    parser.add_argument('--assets-manifest-sha256', required=True)
    parser.add_argument('--owner-token', required=True)
    parser.add_argument('--allow-active-production-request', action='store_true')
    parser.add_argument('--interruption-authorization')
    args = parser.parse_args()
    require((not args.allow_active_production_request and args.interruption_authorization is None)
            or (args.allow_active_production_request and args.interruption_authorization == INTERRUPTION_AUTHORIZATION),
            'active request admission requires the sealed explicit user authorization')
    require(re.fullmatch('[0-9a-f]{32}', args.owner_token), 'fresh32hex token required')
    for value in (args.packet_manifest_sha256, args.assets_manifest_sha256):
        require(re.fullmatch('[0-9a-f]{64}', value), 'reviewed manifest SHA256 required')
    packet = args.packet_dir.resolve(strict=True)
    require(Path(__file__).resolve() == packet / 'profile_transaction.py', 'execute the reviewed packet controller itself')
    manifest_path = packet / 'packet-manifest.json'
    manifest = read_json(manifest_path, expected_sha256=args.packet_manifest_sha256)
    require(args.assets_manifest_sha256 == manifest['assets_manifest_sha256'], 'profile assets differ from the reviewed packet')
    for name, expected in manifest['files'].items():
        require(name in {'profile_transaction.py', 'production-baseline.json', 'profile-options.json', 'profile-command.json', 'workload.json'}
                and file_hash(packet / name) == expected, 'reviewed packet member differs')
    require(set(manifest['files']) == {'profile_transaction.py', 'production-baseline.json', 'profile-options.json', 'profile-command.json', 'workload.json'}, 'packet members incomplete')
    workload = read_workload(packet, manifest)
    require(set(manifest['helpers']) == {'gpu-queue', 'gpu_guard.py', 'installed_native.py',
            'owned_environment.py', 'owned_command.py', 'owned_container.py'}, 'maintained helper bindings incomplete')
    for name, expected in manifest['helpers'].items():
        require(file_hash(TOOLS / name) == expected, 'maintained host helper changed: ' + name)
    validate_assets(EVIDENCE / 'assets', args.assets_manifest_sha256)
    state_directory = EVIDENCE / 'assets/guard-state'
    require(absent(state_directory) or (not state_directory.is_symlink() and state_directory.is_dir()
            and not any(state_directory.iterdir())), 'guard-state must be a fresh regular empty directory')
    require(absent(EVIDENCE / 'transaction.json') and absent(EVIDENCE / 'guard') and absent(EVIDENCE / 'native'),
            'use a fresh transaction evidence root; prior attempt must remain immutable')
    descriptors = queue_descriptors()
    os.umask(0o077)
    for relative in ('cache', 'native/evidence', 'assets/guard-state', 'transaction-control'):
        (EVIDENCE / relative).mkdir(parents=True, mode=0o700, exist_ok=relative == 'assets/guard-state')
    native_options_preflight(packet, manifest, EVIDENCE / 'native-options-preflight.json')
    host = Host(EVIDENCE / 'transaction-control', descriptors)
    transaction = Transaction(packet, EVIDENCE, args.owner_token, host, descriptors, workload,
                              args.interruption_authorization)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, transaction.interrupt)
    return transaction.run()


if __name__ == '__main__':
    raise SystemExit(main())

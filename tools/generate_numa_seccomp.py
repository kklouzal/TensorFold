#!/usr/bin/env python3
"""Derive a narrowly extended NUMA profile from a running Docker default.

Run on the Linux Docker host, with read access to its OCI runtime bundle.
Only linux.seccomp and selected non-secret provenance are retained; the full
OCI configuration and container environment are never emitted. A generated
directory is bound to the captured daemon/runtime versions and capabilities.
Regenerate and requalify it after changing any of those inputs.
"""

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

if __package__:
    from .worker_lifetime import raise_failures
else:
    from worker_lifetime import raise_failures


LIMIT = 2 * 2**20
INSPECT = ('{"id":{{json .Id}},"running":{{json .State.Running}},'
           '"pid":{{json .State.Pid}},'
           '"image":{{json .Image}},"privileged":{{json .HostConfig.Privileged}},'
           '"cap_add":{{json .HostConfig.CapAdd}},"cap_drop":{{json .HostConfig.CapDrop}},'
           '"devices":{{json .HostConfig.Devices}},"device_requests":{{json .HostConfig.DeviceRequests}},'
           '"security_opt":{{json .HostConfig.SecurityOpt}},"apparmor":{{json .AppArmorProfile}}}')
RUNTIME_ROOTS = (
    Path('/run/containerd/io.containerd.runtime.v2.task/moby'),
    Path('/run/docker/containerd/daemon/io.containerd.runtime.v2.task/moby'),
    Path('/run/docker/runtime-runc/moby'),
)
CAP40_ROLE = 'verification-cap40-v1'
CAP40_CAPTURE_ARGS = ('-I', '-S', '-c', 'import signal\nwhile True: signal.pause()\n')
CAP40_INSPECT = INSPECT[:-1] + (
    ',"user":{{json .Config.User}},"entrypoint":{{json .Config.Entrypoint}},"command":{{json .Config.Cmd}},'
    '"readonly_root":{{json .HostConfig.ReadonlyRootfs}},"pid_mode":{{json .HostConfig.PidMode}},'
    '"network_mode":{{json .HostConfig.NetworkMode}},"groups":{{json .HostConfig.GroupAdd}},'
    '"mounts":{{json .Mounts}},"tmpfs":{{json (index .HostConfig "Tmpfs")}},'
    '"userns_mode":{{json .HostConfig.UsernsMode}}}')
IMAGE_INSPECT = '{"Id":{{json .Id}},"RootFS":{{json .RootFS}}}'


def _host_authority():
    if __package__:
        from . import v100_native_contract_host
    else:
        import v100_native_contract_host
    return v100_native_contract_host


def _cap40_authority(value, image):
    if (type(value) is not dict or set(value) != {'image_id', 'verification_launcher'} or value['image_id'] != image
            or type(image) is not str or re.fullmatch(r'sha256:[0-9a-f]{64}', image) is None):
        raise ValueError('CAP40 capture needs its exact trusted image/verification-launcher authority')
    _host_authority()._verification_contract(value['verification_launcher'], image)


def _cap40_isolation(info):
    host = _host_authority()
    if (info.get('cap_add') != ['CAP_CHECKPOINT_RESTORE'] or info.get('cap_drop') != ['ALL']
            or info.get('user') != '1000:1000' or info.get('readonly_root') is not True
            or info.get('pid_mode') not in ('', 'private') or info.get('userns_mode') not in ('', None)
            or info.get('network_mode') != 'none' or info.get('groups') or info.get('mounts') or info.get('tmpfs')
            or info.get('entrypoint') != [host.VERIFICATION_ENTRY] or info.get('command') != list(CAP40_CAPTURE_ARGS)):
        raise ValueError('CAP40 capture requires the fixed CPU-only UID1000 readonly launcher envelope')


def _cap40_oci(process, root):
    host = _host_authority()
    caps = process.get('capabilities', {})
    allowed = {'bounding', 'effective', 'permitted', 'inheritable', 'ambient'}
    if (type(caps) is not dict or set(caps) - allowed
            or any(caps.get(name) != ['CAP_CHECKPOINT_RESTORE'] for name in ('bounding', 'effective', 'permitted'))
            or caps.get('inheritable', []) or caps.get('ambient', [])
            or process.get('user', {}).get('uid') != 1000 or process.get('user', {}).get('gid') != 1000
            or process.get('user', {}).get('additionalGids', []) not in ([], [1000])
            or process.get('noNewPrivileges', False) is not False or root.get('readonly') is not True
            or process.get('args') != [host.VERIFICATION_ENTRY, *CAP40_CAPTURE_ARGS]):
        raise ValueError('resolved OCI process differs from the fixed CAP40 launcher role')


def _proc_bytes(pid, name):
    def opener(path, flags): return os.open(path, flags | os.O_NOFOLLOW | os.O_NONBLOCK)
    with open(Path('/proc', str(pid), name), 'rb', opener=opener) as stream:
        data = stream.read(65537)
    if len(data) > 65536: raise ValueError('CAP40 process metadata exceeds its bounded envelope')
    return data


def _process_facts(pid):
    wanted = {'Uid', 'Gid', 'Groups', 'CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb', 'NoNewPrivs', 'Seccomp', 'Seccomp_filters', 'NSpid'}
    status = {}
    for line in _proc_bytes(pid, 'status').decode('ascii').splitlines():
        key, _, value = line.partition(':')
        if key in wanted: status[key] = value.strip()
    for key in ('Uid', 'Gid', 'Groups', 'NSpid'): status[key] = [int(v) for v in status[key].split()]
    for key in ('NoNewPrivs', 'Seccomp', 'Seccomp_filters'): status[key] = int(status[key])
    stat_tail = _proc_bytes(pid, 'stat').rsplit(b')', 1)[-1].split()
    return {'status': status, 'start_time_ticks': int(stat_tail[19]),
            'argv': [v.decode('utf8') for v in _proc_bytes(pid, 'cmdline').split(b'\0') if v]}


def _image_file(path, expected, *, proc_exe=False):
    def opener(filename, flags):
        return os.open(filename, flags | os.O_NONBLOCK | (0 if proc_exe else os.O_NOFOLLOW))
    with open(path, 'rb', opener=opener) as stream:
        before = os.fstat(stream.fileno())
        readonly = bool(os.fstatvfs(stream.fileno()).f_flag & os.ST_RDONLY)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected['bytes']:
            raise ValueError('protected CAP40 image file type/extent differs')
        hashed, remaining = hashlib.sha256(), before.st_size
        while remaining:
            data = stream.read(min(remaining, 1 << 20))
            if not data: raise ValueError('protected CAP40 image file truncated')
            remaining -= len(data); hashed.update(data)
        if stream.read(1): raise ValueError('protected CAP40 image file grew')
        after = os.fstat(stream.fileno())
        if any(getattr(before, k) != getattr(after, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
            raise ValueError('protected CAP40 image file changed')
    result = {'bytes': before.st_size, 'sha256': hashed.hexdigest(), 'uid': before.st_uid,
              'mode': stat.S_IMODE(before.st_mode), 'readonly': readonly}
    if result['sha256'] != expected['sha256']:
        raise ValueError('protected CAP40 image file hash differs')
    return result


def _cap40_files(pid, authority):
    host = _host_authority(); contract = authority['verification_launcher']
    root = Path('/proc', str(pid), 'root')
    launcher = _image_file(root / host.VERIFICATION_ENTRY.lstrip('/'), contract['launcher'])
    launcher['file_capability_hex'] = os.getxattr(root / host.VERIFICATION_ENTRY.lstrip('/'), 'security.capability').hex()
    source = _image_file(root / host.VERIFICATION_SOURCE.lstrip('/'),
        {'bytes': contract['launcher']['source_bytes'], 'sha256': contract['launcher']['source_sha256']})
    python = _image_file(Path('/proc', str(pid), 'exe'), contract['python'], proc_exe=True)
    return {'launcher': launcher, 'source': source, 'python': python}


def validate_capture_role(provenance, *, capture_role='default'):
    """Pure fixed-role authority used by capture and later exact regeneration."""
    if capture_role == 'default':
        if provenance.get('capture_role', 'default') != 'default':
            raise ValueError('CAP40 provenance cannot become a default-capability authority')
        return
    if capture_role != CAP40_ROLE or provenance.get('capture_role') != CAP40_ROLE:
        raise ValueError('unknown or mismatched fixed capture role')
    host = _host_authority(); info = provenance['container']; authority = provenance['role_authority']
    if (type(info.get('id')) is not str or re.fullmatch('[0-9a-f]{64}', info['id']) is None
            or type(info.get('pid')) is not int or info['pid'] <= 0
            or 'name=seccomp,profile=builtin' not in provenance['daemon_security']):
        raise ValueError('CAP40 capture lacks its actual CID/PID/builtin-seccomp identity')
    bundle = Path(provenance['bundle_path'])
    if bundle.name != 'config.json' or bundle.parent.name != info['id']:
        raise ValueError('CAP40 runtime bundle differs from its inspected CID')
    _validate_cgroups(info['id'], provenance['identity']['oci_cgroups_path'], provenance['identity']['process_cgroups'])
    _cap40_authority(authority, info['image']); _cap40_isolation(info)
    if (info.get('running') is not True or info.get('privileged') is not False or info.get('security_opt')
            or info.get('devices') or info.get('device_requests')):
        raise ValueError('CAP40 role cannot add privilege/security bypass/devices')
    process = provenance['resolved_process']
    _cap40_oci(process, provenance['resolved_root'])
    if process['capabilities'] != provenance['resolved_capabilities']:
        raise ValueError('CAP40 OCI capability provenance differs')
    facts = provenance['actual_process']; status = facts['status']
    if (set(status) != {'Uid', 'Gid', 'Groups', 'CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb', 'NoNewPrivs', 'Seccomp', 'Seccomp_filters', 'NSpid'}
            or status['Uid'] != [1000]*4 or status['Gid'] != [1000]*4
            or status['Groups'] not in ([], [1000])
            or any(status[name] != host.VERIFICATION_CAP for name in ('CapInh','CapPrm','CapEff','CapBnd','CapAmb'))
            or status['NoNewPrivs'] != 0 or status['Seccomp'] != 2 or type(status['Seccomp_filters']) is not int
            or status['Seccomp_filters'] < 1 or status['NSpid'][-1:] != [1]
            or status['NSpid'][:1] != [info['pid']]
            or type(facts['start_time_ticks']) is not int or facts['start_time_ticks'] <= 0
            or facts['argv'] != [host.VERIFICATION_PYTHON, *CAP40_CAPTURE_ARGS]):
        raise ValueError('actual post-launcher UID/GID/caps/process differs from fixed CAP40 role')
    lineage = provenance['image_lineage']
    host.verification_image({'Id':lineage['image_id'],'RootFS':{'Layers':lineage['purpose_layers']}},
        {'Id':lineage['base_image_id'],'RootFS':{'Layers':lineage['base_layers']}}, authority)
    files = provenance['protected_files']; contract = authority['verification_launcher']
    for name, wanted in (('launcher', contract['launcher']), ('source', {
            'bytes':contract['launcher']['source_bytes'],'sha256':contract['launcher']['source_sha256']}), ('python', contract['python'])):
        actual = files[name]
        if (any(actual[key] != wanted[key] for key in ('bytes','sha256')) or actual['uid'] != 0
                or actual['readonly'] is not True):
            raise ValueError('actual protected CAP40 image authority differs')
    if (files['launcher']['mode'] != 0o555 or files['source']['mode'] != 0o444
            or files['launcher']['file_capability_hex'] != host.VERIFICATION_XATTR
            or files['python']['mode'] & 0o6022 or not files['python']['mode'] & 0o100):
        raise ValueError('actual CAP40 launcher/Python protection differs')


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path):
    def opener(filename, flags):
        return os.open(filename, flags | os.O_NOFOLLOW | os.O_NONBLOCK)

    with open(path, 'rb', opener=opener) as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('JSON input must be a regular non-symlink file')
        data = stream.read(LIMIT + 1)
        after = os.fstat(stream.fileno())
    if len(data) > LIMIT:
        raise ValueError('JSON input exceeds the 2 MiB audit limit')
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, field) != getattr(after, field) for field in fields) or len(data) != before.st_size:
        raise ValueError('JSON input changed during capture')
    return data


def _read_json(path):
    return json.loads(_read_bytes(path))


def _docker(executable, *arguments):
    # No shell or environment inspection. The fixed CAP40 format admits only
    # its known command/envelope before retaining the selected provenance.
    result = subprocess.run([executable, *arguments], capture_output=True, timeout=30, check=False)
    if result.returncode:
        raise RuntimeError(f'Docker metadata command failed: {arguments[0]} (exit {result.returncode})')
    if len(result.stdout) > LIMIT:
        raise ValueError('Docker metadata exceeds the 2 MiB audit limit')
    return json.loads(result.stdout)


def _process_cgroups(pid):
    return Path('/proc', str(pid), 'cgroup').read_text()


def _validate_cgroups(cid, cgroup_path, cgroups):
    # systemd's OCI notation ends in :docker:<ID>; cgroupfs ends in /<ID>.
    oci_leaf = cgroup_path.rsplit('/', 1)[-1].rsplit(':', 1)[-1]
    expected_leaves = {cid, 'docker-' + cid + '.scope'}
    actual_leaves = {line.split(':', 2)[-1].rsplit('/', 1)[-1] for line in cgroups.splitlines()}
    if oci_leaf not in expected_leaves or not actual_leaves.intersection(expected_leaves):
        raise ValueError('OCI bundle and running PID cgroups do not identify the inspected container')


def capture(container, *, docker='docker', bundle_config=None, capture_role='default', verification_contract=None):
    """Capture the resolved filter, refusing a source with elevated isolation."""
    if capture_role not in ('default', CAP40_ROLE) or capture_role == 'default' and verification_contract is not None:
        raise ValueError('unknown capture role or unused verification authority')
    inspect = CAP40_INSPECT if capture_role == CAP40_ROLE else INSPECT
    info = _docker(docker, 'inspect', '--format', inspect, container)
    if not isinstance(info, dict) or not re.fullmatch(r'[0-9a-f]{64}', str(info.get('id', ''))):
        raise ValueError('Docker did not resolve one full container identity')
    if (info.get('running') is not True or info.get('privileged') is not False or info.get('security_opt')
            or capture_role == 'default' and (info.get('cap_add') or info.get('cap_drop'))):
        raise ValueError('source must be running with default capabilities and security options, without privilege')
    if capture_role == CAP40_ROLE:
        _cap40_authority(verification_contract, info.get('image'))
        _cap40_isolation(info)
    if info.get('devices') or info.get('device_requests'):
        raise ValueError('source must be CPU-only, without requested devices')
    security = _docker(docker, 'info', '--format', '{{json .SecurityOptions}}')
    if not isinstance(security, list) or 'name=seccomp,profile=builtin' not in security:
        raise ValueError('Docker daemon must report the builtin seccomp profile')
    versions = _docker(docker, 'version', '--format', '{{json .}}')
    candidates = ([Path(bundle_config)] if bundle_config else
                  [root / info['id'] / 'config.json' for root in RUNTIME_ROOTS])
    paths = [path for path in candidates if path.is_file()]
    if len(paths) != 1:
        raise ValueError('expected exactly one readable runtime bundle; specify --bundle-config if needed')
    if paths[0].resolve().parent.name != info['id'] or paths[0].name != 'config.json':
        raise ValueError('runtime bundle must be <full inspected container ID>/config.json')
    config = _read_json(paths[0])
    pid = info.get('pid')
    if type(pid) is not int or pid <= 0:
        raise ValueError('Docker did not report a running host PID')
    cgroups = _process_cgroups(pid)
    cgroup_path = config.get('linux', {}).get('cgroupsPath', '')
    _validate_cgroups(info['id'], cgroup_path, cgroups)
    seccomp = config.get('linux', {}).get('seccomp')
    # Capabilities are part of the provenance because Docker resolves its
    # builtin capability-conditioned rules before writing this OCI profile.
    capabilities = config.get('process', {}).get('capabilities')
    if not isinstance(capabilities, dict):
        raise ValueError('runtime bundle does not expose resolved capabilities')
    provenance = {'container': info, 'docker_versions': versions, 'daemon_security': security,
                  'resolved_capabilities': capabilities, 'bundle_path': str(paths[0]),
                  'identity': {'oci_cgroups_path': cgroup_path, 'process_cgroups': cgroups}}
    if capture_role == CAP40_ROLE:
        process = config.get('process', {})
        _cap40_oci(process, config.get('root', {}))
        observed = _process_facts(pid)
        files = _cap40_files(pid, verification_contract)
        candidate = _docker(docker, 'image', 'inspect', '--format', IMAGE_INSPECT, info['image'])
        base = _docker(docker, 'image', 'inspect', '--format', IMAGE_INSPECT,
                       verification_contract['verification_launcher']['base_image_id'])
        lineage = _host_authority().verification_image(candidate, base, verification_contract)
        if _process_facts(pid) != observed or _process_cgroups(pid) != cgroups:
            raise ValueError('CAP40 process identity changed during capture')
        provenance.update(capture_role=CAP40_ROLE, role_authority=verification_contract,
            resolved_process={name: process[name] for name in ('user','args','capabilities')}
                | {'noNewPrivileges':process.get('noNewPrivileges',False)},
            resolved_root={'readonly':config['root']['readonly']}, actual_process=observed,
            protected_files=files, image_lineage=lineage)
        validate_capture_role(provenance, capture_role=CAP40_ROLE)
    if _docker(docker, 'inspect', '--format', inspect, info['id']) != info:
        raise ValueError('container identity or isolation changed during capture')
    return seccomp, provenance


def numa_rules(*, worker=False):
    """Allow physical-node BIND and self page queries, with no migration flags.

    MPOL_F_STATIC_NODES (1 << 15) preserves physical node identities when
    allowed node sets change. Actual placement still needs verification;
    the deployment must retain access to the selected node.
    """
    def equal(index, value):
        return {'index': index, 'value': value, 'op': 'SCMP_CMP_EQ'}

    rules = [
        {'names': ['mbind'], 'action': 'SCMP_ACT_ALLOW', 'args': [equal(2, 32770), equal(5, 0)]},
        {'names': ['get_mempolicy'], 'action': 'SCMP_ACT_ALLOW'},
        {'names': ['move_pages'], 'action': 'SCMP_ACT_ALLOW',
         'args': [equal(0, 0), equal(3, 0), equal(5, 0)]},
    ]

    if worker:
        rules.append({'names':['set_mempolicy'],'action':'SCMP_ACT_ALLOW','args':[equal(0,32770)]})
    return rules


def render(baseline, provenance, *, purpose='archive', capture_role='default'):
    if (not isinstance(baseline, dict) or baseline.get('defaultAction') != 'SCMP_ACT_ERRNO'
            or not isinstance(baseline.get('syscalls'), list) or not baseline['syscalls']
            or not isinstance(baseline.get('architectures'), list) or not baseline['architectures']):
        raise ValueError('expected a resolved default-deny OCI seccomp profile')
    if set(baseline) - {'defaultAction', 'defaultErrnoRet', 'architectures', 'flags', 'syscalls'}:
        raise ValueError('unsupported resolved seccomp fields; review before extending this profile')
    if purpose not in ('archive','worker'):
        raise ValueError('NUMA profile purpose must be archive or worker')
    validate_capture_role(provenance, capture_role=capture_role)
    if capture_role == CAP40_ROLE and purpose != 'worker':
        raise ValueError('fixed CAP40 capture supports only the narrow worker policy')
    added = numa_rules(worker=purpose=='worker')
    names = {name for rule in added for name in rule['names']}
    for rule in baseline['syscalls']:
        if not isinstance(rule, dict) or not isinstance(rule.get('names'), list):
            raise ValueError('invalid resolved syscall rule')
        if names.intersection(rule['names']):
            raise ValueError('baseline already mentions a required NUMA syscall; review overlapping policy')
    profile = copy.deepcopy(baseline)
    profile['syscalls'].extend(added)
    baseline_bytes, profile_bytes = encoded(baseline), encoded(profile)
    manifest = {'schema_version': 1, 'purpose': 'tensorfold-numa-local-rank-worker' if purpose=='worker' else 'tensorfold-numa-local-archive',
                'baseline_sha256': digest(baseline_bytes), 'profile_sha256': digest(profile_bytes),
                'generator_sha256': digest(Path(__file__).read_bytes()),
                'added_rules': added, 'provenance': provenance,
                'references': ['https://docs.docker.com/engine/security/seccomp/',
                               'https://github.com/opencontainers/runtime-spec/blob/v1.2.1/config-linux.md#seccomp'],
                'qualification': 'generated; actual placement and target runtime checks still required'}
    if capture_role == CAP40_ROLE:
        manifest.update(schema_version=2, purpose='tensorfold-numa-verification-cap40-worker', capture_role=CAP40_ROLE)
    return {'baseline-seccomp.json': baseline_bytes, 'seccomp.json': profile_bytes,
            'manifest.json': encoded(manifest)}


def _write_file(path, data):
    stream = path.open('xb')
    primary, failures = None, []
    try:
        os.fchmod(stream.fileno(), 0o644)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    except BaseException as error:
        primary = error
    try:
        stream.close()
    except BaseException as error:
        failures.append(error)
    raise_failures(primary, failures)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    primary, failures = None, []
    try:
        os.fsync(descriptor)
    except BaseException as error:
        primary = error
    try:
        os.close(descriptor)
    except BaseException as error:
        failures.append(error)
    raise_failures(primary, failures)


def publish(output, files):
    """Publish all three files together; an identical retry is read-only.

    The operator owns the parent directory and serializes generation there.
    A successful rename publishes the complete directory. A later parent fsync
    failure reports committed content with unconfirmed crash durability.
    """
    output = Path(output)
    if output.is_symlink():
        raise ValueError('output directory must not be a symlink')
    if output.exists():
        if (not output.is_dir() or {path.name for path in output.iterdir()} != set(files)
                or any((output / name).is_symlink() or not (output / name).is_file()
                       or _read_bytes(output / name) != data for name, data in files.items())):
            raise ValueError('output exists with different contents; use a new directory')
        return
    if not output.parent.is_dir():
        raise ValueError('output parent directory must already exist')
    temporary = Path(tempfile.mkdtemp(prefix='.' + output.name + '.', dir=output.parent))
    committed = False
    primary, failures = None, []
    try:
        for name, data in files.items():
            _write_file(temporary / name, data)
        temporary.chmod(0o755)
        _sync_directory(temporary)
        os.rename(temporary, output)
        committed = True
        _sync_directory(output.parent)
    except BaseException as error:
        primary = error
        if committed:
            try:
                BaseException.add_note(error, 'profile directory was published; parent directory durability was not confirmed')
            except BaseException as annotation:
                failures.append(annotation)
    finally:
        if not committed:
            try:
                shutil.rmtree(temporary)
            except BaseException as cleanup:
                failures.append(cleanup)
    raise_failures(primary, failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', required=True,
                        help='running CPU-only container using this daemon\'s default isolation')
    parser.add_argument('--output-directory', required=True, type=Path,
                        help='new output directory, or identical previous generation')
    parser.add_argument('--bundle-config', type=Path,
                        help='alternate runtime-root/<full container ID>/config.json')
    parser.add_argument('--purpose', choices=('archive','worker'), default='archive',
                        help='worker adds only physical-node static thread BIND to the archive/self-query policy')
    parser.add_argument('--docker', default='docker', help='Docker executable on this host')
    parser.add_argument('--capture-role', choices=('default', CAP40_ROLE), default='default',
                        help='fixed verification role independently captures its CAP40-conditioned default filter')
    parser.add_argument('--verification-contract', type=Path,
                        help='CAP40-only closed image_id/verification_launcher JSON authority')
    args = parser.parse_args()
    authority = None if args.verification_contract is None else _read_json(args.verification_contract)
    baseline, provenance = capture(args.container, docker=args.docker, bundle_config=args.bundle_config,
                                   capture_role=args.capture_role, verification_contract=authority)
    files = render(baseline, provenance, purpose=args.purpose, capture_role=args.capture_role)
    publish(args.output_directory, files)
    print(json.dumps({'profile': str(args.output_directory / 'seccomp.json'),
                      'manifest': str(args.output_directory / 'manifest.json'),
                      'profile_sha256': digest(files['seccomp.json'])}, sort_keys=True))


if __name__ == '__main__':
    main()

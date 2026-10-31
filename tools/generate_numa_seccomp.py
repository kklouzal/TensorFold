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
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile

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
    # No shell and no inspect of environment, mounts or command arguments.
    result = subprocess.run([executable, *arguments], capture_output=True, timeout=30, check=False)
    if result.returncode:
        raise RuntimeError(f'Docker metadata command failed: {arguments[0]} (exit {result.returncode})')
    if len(result.stdout) > LIMIT:
        raise ValueError('Docker metadata exceeds the 2 MiB audit limit')
    return json.loads(result.stdout)


def _process_cgroups(pid):
    return Path('/proc', str(pid), 'cgroup').read_text()


def capture(container, *, docker='docker', bundle_config=None):
    """Capture the resolved filter, refusing a source with elevated isolation."""
    info = _docker(docker, 'inspect', '--format', INSPECT, container)
    if not isinstance(info, dict) or not re.fullmatch(r'[0-9a-f]{64}', str(info.get('id', ''))):
        raise ValueError('Docker did not resolve one full container identity')
    if (info.get('running') is not True or info.get('privileged') is not False
            or info.get('cap_add') or info.get('cap_drop') or info.get('security_opt')):
        raise ValueError('source must be running with default capabilities and security options, without privilege')
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
    # systemd's OCI notation ends in :docker:<ID>; cgroupfs ends in /<ID>.
    # Both must identify this container and its actual running host process.
    oci_leaf = cgroup_path.rsplit('/', 1)[-1].rsplit(':', 1)[-1]
    expected_leaves = {info['id'], 'docker-' + info['id'] + '.scope'}
    actual_leaves = {line.split(':', 2)[-1].rsplit('/', 1)[-1] for line in cgroups.splitlines()}
    if oci_leaf not in expected_leaves or not actual_leaves.intersection(expected_leaves):
        raise ValueError('OCI bundle and running PID cgroups do not identify the inspected container')
    seccomp = config.get('linux', {}).get('seccomp')
    # Capabilities are part of the provenance because Docker resolves its
    # builtin capability-conditioned rules before writing this OCI profile.
    capabilities = config.get('process', {}).get('capabilities')
    if not isinstance(capabilities, dict):
        raise ValueError('runtime bundle does not expose resolved capabilities')
    provenance = {'container': info, 'docker_versions': versions, 'daemon_security': security,
                  'resolved_capabilities': capabilities, 'bundle_path': str(paths[0]),
                  'identity': {'oci_cgroups_path': cgroup_path, 'process_cgroups': cgroups}}
    if _docker(docker, 'inspect', '--format', INSPECT, info['id']) != info:
        raise ValueError('container identity or isolation changed during capture')
    return seccomp, provenance


def numa_rules():
    """Allow physical-node BIND and self page queries, with no migration flags.

    MPOL_F_STATIC_NODES (1 << 15) preserves physical node identities when
    allowed node sets change. Actual placement still needs verification;
    the deployment must retain access to the selected node.
    """
    def equal(index, value):
        return {'index': index, 'value': value, 'op': 'SCMP_CMP_EQ'}

    return [
        {'names': ['mbind'], 'action': 'SCMP_ACT_ALLOW', 'args': [equal(2, 32770), equal(5, 0)]},
        {'names': ['get_mempolicy'], 'action': 'SCMP_ACT_ALLOW'},
        {'names': ['move_pages'], 'action': 'SCMP_ACT_ALLOW',
         'args': [equal(0, 0), equal(3, 0), equal(5, 0)]},
    ]


def render(baseline, provenance):
    if (not isinstance(baseline, dict) or baseline.get('defaultAction') != 'SCMP_ACT_ERRNO'
            or not isinstance(baseline.get('syscalls'), list) or not baseline['syscalls']
            or not isinstance(baseline.get('architectures'), list) or not baseline['architectures']):
        raise ValueError('expected a resolved default-deny OCI seccomp profile')
    if set(baseline) - {'defaultAction', 'defaultErrnoRet', 'architectures', 'flags', 'syscalls'}:
        raise ValueError('unsupported resolved seccomp fields; review before extending this profile')
    added = numa_rules()
    names = {name for rule in added for name in rule['names']}
    for rule in baseline['syscalls']:
        if not isinstance(rule, dict) or not isinstance(rule.get('names'), list):
            raise ValueError('invalid resolved syscall rule')
        if names.intersection(rule['names']):
            raise ValueError('baseline already mentions a required NUMA syscall; review overlapping policy')
    profile = copy.deepcopy(baseline)
    profile['syscalls'].extend(added)
    baseline_bytes, profile_bytes = encoded(baseline), encoded(profile)
    manifest = {'schema_version': 1, 'purpose': 'tensorfold-numa-local-archive',
                'baseline_sha256': digest(baseline_bytes), 'profile_sha256': digest(profile_bytes),
                'generator_sha256': digest(Path(__file__).read_bytes()),
                'added_rules': added, 'provenance': provenance,
                'references': ['https://docs.docker.com/engine/security/seccomp/',
                               'https://github.com/opencontainers/runtime-spec/blob/v1.2.1/config-linux.md#seccomp'],
                'qualification': 'generated; actual placement and target runtime checks still required'}
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
    parser.add_argument('--docker', default='docker', help='Docker executable on this host')
    args = parser.parse_args()
    baseline, provenance = capture(args.container, docker=args.docker, bundle_config=args.bundle_config)
    files = render(baseline, provenance)
    publish(args.output_directory, files)
    print(json.dumps({'profile': str(args.output_directory / 'seccomp.json'),
                      'manifest': str(args.output_directory / 'manifest.json'),
                      'profile_sha256': digest(files['seccomp.json'])}, sort_keys=True))


if __name__ == '__main__':
    main()

"""Owned profile transformation, boundary and atomic publication contracts."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('tools.generate_numa_seccomp', ROOT / 'tools/generate_numa_seccomp.py')
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


@pytest.fixture
def baseline():
    return {'defaultAction': 'SCMP_ACT_ERRNO', 'defaultErrnoRet': 1,
            'architectures': ['SCMP_ARCH_X86_64', 'SCMP_ARCH_X86'],
            'flags': ['SECCOMP_FILTER_FLAG_LOG'],
            'syscalls': [{'names': ['read', 'write'], 'action': 'SCMP_ACT_ALLOW'},
                         {'names': ['clone3'], 'action': 'SCMP_ACT_ERRNO', 'errnoRet': 38},
                         {'names': ['socket'], 'action': 'SCMP_ACT_ALLOW',
                          'args': [{'index': 0, 'value': 38, 'op': 'SCMP_CMP_LT'}]}]}


def test_only_three_narrow_rules_are_added_with_existing_rules_untouched(baseline):
    previous = copy.deepcopy(baseline)
    files = profile.render(baseline, {'versions': 'recorded'})
    actual = json.loads(files['seccomp.json'])
    assert baseline == previous
    added = actual['syscalls'][len(baseline['syscalls']):]
    actual['syscalls'] = actual['syscalls'][:len(baseline['syscalls'])]
    assert actual == baseline
    assert added == [
        {'action': 'SCMP_ACT_ALLOW', 'names': ['mbind'],
         'args': [{'index': 2, 'op': 'SCMP_CMP_EQ', 'value': 32770},
                  {'index': 5, 'op': 'SCMP_CMP_EQ', 'value': 0}]},
        {'action': 'SCMP_ACT_ALLOW', 'names': ['get_mempolicy']},
        {'action': 'SCMP_ACT_ALLOW', 'names': ['move_pages'],
         'args': [{'index': 0, 'op': 'SCMP_CMP_EQ', 'value': 0},
                  {'index': 3, 'op': 'SCMP_CMP_EQ', 'value': 0},
                  {'index': 5, 'op': 'SCMP_CMP_EQ', 'value': 0}]},
    ]
    assert 'set_mempolicy' not in {name for rule in added for name in rule['names']}
    manifest = json.loads(files['manifest.json'])
    assert manifest['profile_sha256'] == profile.digest(files['seccomp.json'])
    assert manifest['baseline_sha256'] == profile.digest(files['baseline-seccomp.json'])
    assert files == profile.render(baseline, {'versions': 'recorded'})


@pytest.mark.parametrize('invalid', ['allow', 'empty', 'unknown', 'overlap', 'bad-rule', 'no-arch'])
def test_unreviewed_or_overlapping_profiles_are_rejected(baseline, invalid):
    if invalid == 'allow':
        baseline['defaultAction'] = 'SCMP_ACT_ALLOW'
    elif invalid == 'empty':
        baseline['syscalls'] = []
    elif invalid == 'unknown':
        baseline['listenerPath'] = '/unreviewed/listener'
    elif invalid == 'overlap':
        baseline['syscalls'].append({'names': ['mbind'], 'action': 'SCMP_ACT_ERRNO'})
    elif invalid == 'bad-rule':
        baseline['syscalls'].append({'action': 'SCMP_ACT_ALLOW'})
    else:
        baseline['architectures'] = []
    with pytest.raises(ValueError):
        profile.render(baseline, {})


def test_directory_publication_is_complete_idempotent_and_never_rewrites(baseline, tmp_path):
    files = profile.render(baseline, {})
    output = tmp_path / 'numa'
    profile.publish(output, files)
    before = {p.name: (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes()) for p in output.iterdir()}
    profile.publish(output, files)
    assert {p.name: (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes()) for p in output.iterdir()} == before
    assert output.stat().st_mode & 0o777 == 0o755
    assert all(p.stat().st_mode & 0o777 == 0o644 for p in output.iterdir())
    assert len(list(tmp_path.iterdir())) == 1
    (output / 'seccomp.json').write_text('different')
    with pytest.raises(ValueError, match='different contents'):
        profile.publish(output, files)
    assert (output / 'seccomp.json').read_text() == 'different'


def test_symlinks_and_nonordinary_outputs_are_not_followed(baseline, tmp_path):
    files = profile.render(baseline, {})
    real = tmp_path / 'real'
    real.mkdir()
    link = tmp_path / 'link'
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        profile.publish(link, files)
    output = tmp_path / 'numa'
    profile.publish(output, files)
    target = output / 'seccomp.json'
    data = target.read_bytes()
    target.unlink()
    (real / 'seccomp.json').write_bytes(data)
    target.symlink_to(real / 'seccomp.json')
    with pytest.raises(ValueError, match='different contents'):
        profile.publish(output, files)


def test_existing_output_retry_has_bounded_reads(baseline, tmp_path):
    files = profile.render(baseline, {})
    output = tmp_path / 'numa'
    profile.publish(output, files)
    (output / 'seccomp.json').write_bytes(b' ' * (profile.LIMIT + 1))
    with pytest.raises(ValueError, match='2 MiB'):
        profile.publish(output, files)


def test_before_publish_failure_leaves_no_partial_directory(baseline, tmp_path, monkeypatch):
    def fail(*args):
        raise OSError('owned sync failure')
    monkeypatch.setattr(profile.os, 'fsync', fail)
    with pytest.raises(OSError, match='owned sync failure'):
        profile.publish(tmp_path / 'numa', profile.render(baseline, {}))
    assert list(tmp_path.iterdir()) == []


def test_publish_cleanup_and_annotation_failure_preserve_primary_and_prior_cause(baseline, tmp_path, monkeypatch):
    prior = LookupError('prior cause')
    primary = OSError('write failed')
    primary.__cause__ = prior
    primary.__notes__ = 123
    cleanup = OSError('temporary removal failed')

    def fail_sync(*args):
        raise primary

    def fail_cleanup(*args):
        raise cleanup

    monkeypatch.setattr(profile.os, 'fsync', fail_sync)
    monkeypatch.setattr(profile.shutil, 'rmtree', fail_cleanup)
    with pytest.raises(OSError) as error:
        profile.publish(tmp_path / 'numa', profile.render(baseline, {}))
    assert error.value is primary
    cause = error.value.__cause__
    assert isinstance(cause, BaseExceptionGroup)
    assert prior in cause.exceptions and cleanup in cause.exceptions
    assert any(isinstance(item, TypeError) for item in cause.exceptions)


def test_after_publish_failure_reports_committed_directory(baseline, tmp_path, monkeypatch):
    real = profile.os.fsync
    count = 0

    def sync(descriptor):
        nonlocal count
        count += 1
        if count == 5:
            raise OSError('parent sync failed')
        return real(descriptor)

    monkeypatch.setattr(profile.os, 'fsync', sync)
    files = profile.render(baseline, {})
    output = tmp_path / 'numa'
    with pytest.raises(OSError, match='parent sync failed') as error:
        profile.publish(output, files)
    assert any('was published' in note for note in error.value.__notes__)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == files


@pytest.fixture
def capture_source(tmp_path, monkeypatch, baseline):
    info = {'id': 'a' * 64, 'pid': 1234, 'running': True, 'privileged': False,
            'cap_add': None, 'cap_drop': None, 'security_opt': None}
    secret = 'this environment must never enter an output'
    bundle = tmp_path / info['id']
    bundle.mkdir()
    config = bundle / 'config.json'
    config.write_text(json.dumps({'linux': {'seccomp': baseline,
                                          'cgroupsPath': 'system.slice:docker:' + info['id']},
                                 'process': {'env': [secret], 'capabilities': {'effective': ['CAP_CHOWN']}}}))

    def docker(executable, action, *args):
        if action == 'inspect':
            return info
        if action == 'info':
            return ['name=seccomp,profile=builtin']
        assert action == 'version'
        return {'Server': {'Version': 'recorded-version'}}

    monkeypatch.setattr(profile, '_docker', docker)
    monkeypatch.setattr(profile, '_process_cgroups', lambda pid: '0::/system.slice/docker-' + 'a' * 64 + '.scope\n')
    return SimpleNamespace(info=info, config=config, secret=secret)


def test_capture_retains_only_seccomp_and_selected_provenance(capture_source, baseline):
    actual, provenance = profile.capture('cpu-probe', bundle_config=capture_source.config)
    assert actual == baseline
    assert provenance['docker_versions']['Server']['Version'] == 'recorded-version'
    assert provenance['resolved_capabilities'] == {'effective': ['CAP_CHOWN']}
    assert capture_source.secret.encode() not in b''.join(profile.render(actual, provenance).values())


def test_capture_refuses_an_explicit_bundle_for_another_container(capture_source):
    capture_source.info['id'] = 'b' * 64
    with pytest.raises(ValueError, match='inspected container ID'):
        profile.capture('different-probe', bundle_config=capture_source.config)


def test_capture_refuses_a_mislabelled_bundle_with_another_oci_identity(capture_source):
    config = json.loads(capture_source.config.read_text())
    config['linux']['cgroupsPath'] = 'system.slice:docker:' + 'b' * 64
    capture_source.config.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='cgroups do not identify'):
        profile.capture('cpu-probe', bundle_config=capture_source.config)


def test_capture_refuses_a_pid_from_another_container(capture_source, monkeypatch):
    monkeypatch.setattr(profile, '_process_cgroups', lambda pid: '0::/system.slice/docker-' + 'b' * 64 + '.scope\n')
    with pytest.raises(ValueError, match='cgroups do not identify'):
        profile.capture('cpu-probe', bundle_config=capture_source.config)


def test_capture_refuses_restart_during_collection(capture_source, monkeypatch):
    original = profile._docker
    count = 0

    def docker(executable, action, *args):
        nonlocal count
        value = original(executable, action, *args)
        if action == 'inspect':
            count += 1
            if count == 2:
                value = value | {'pid': value['pid'] + 1}
        return value

    monkeypatch.setattr(profile, '_docker', docker)
    with pytest.raises(ValueError, match='changed during capture'):
        profile.capture('cpu-probe', bundle_config=capture_source.config)


@pytest.mark.parametrize('field,value', [('running', False), ('privileged', True),
                                        ('cap_add', ['SYS_NICE']), ('cap_drop', ['CHOWN']),
                                        ('security_opt', ['seccomp=unconfined'])])
def test_capture_refuses_changed_isolation(capture_source, field, value):
    capture_source.info[field] = value
    with pytest.raises(ValueError, match='default capabilities'):
        profile.capture('probe', bundle_config=capture_source.config)


@pytest.mark.parametrize('field', ['devices', 'device_requests'])
def test_capture_refuses_gpu_or_other_requested_devices(capture_source, field):
    capture_source.info[field] = [{'device': 'example'}]
    with pytest.raises(ValueError, match='CPU-only'):
        profile.capture('probe', bundle_config=capture_source.config)


def test_config_limit_is_checked_before_parsing(tmp_path):
    path = tmp_path / 'huge.json'
    path.write_bytes(b' ' * (profile.LIMIT + 1))
    with pytest.raises(ValueError, match='2 MiB'):
        profile._read_json(path)


def test_config_input_refuses_symlinks_and_nonregular_files(tmp_path):
    regular = tmp_path / 'regular.json'
    regular.write_text('{}')
    link = tmp_path / 'link'
    link.symlink_to(regular)
    with pytest.raises(OSError):
        profile._read_json(link)
    fifo = tmp_path / 'fifo'
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match='regular non-symlink'):
        profile._read_json(fifo)


def test_config_input_detects_mutation_during_read(tmp_path, monkeypatch):
    path = tmp_path / 'changing.json'
    path.write_text('{}')
    original = profile.os.fstat
    count = 0

    def fstat(descriptor):
        nonlocal count
        result = original(descriptor)
        count += 1
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_mode')
        value = SimpleNamespace(**{field: getattr(result, field) for field in fields})
        if count == 2:
            value.st_mtime_ns += 1
        return value

    monkeypatch.setattr(profile.os, 'fstat', fstat)
    with pytest.raises(ValueError, match='changed during capture'):
        profile._read_json(path)


def test_optional_compose_merge_only_adds_a_required_profile(tmp_path):
    docker = shutil.which('docker')
    if docker is None:
        pytest.skip('requires host Docker Compose')
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('TF_', 'TENSORFOLD_', 'MODEL_ROOT', 'GGUF_FILE'))}
    environment.update(MODEL_ROOT='/models', GGUF_FILE='/models/model.gguf')
    empty_env = tmp_path / 'empty.env'
    empty_env.write_text('')
    command = [docker, 'compose', '--env-file', str(empty_env), '-f',
               str(ROOT / 'deploy/v100/compose.yaml')]

    def run(arguments, extra=None):
        return subprocess.run(arguments, capture_output=True, text=True, timeout=30,
                              env=environment | (extra or {}), check=False)

    base = run(command + ['config', '--format', 'json'])
    assert base.returncode == 0, base.stderr
    command += ['-f', str(ROOT / 'deploy/v100/compose.numa.yaml'), 'config', '--format', 'json']
    missing = run(command)
    assert missing.returncode != 0 and 'TF_NUMA_SECCOMP' in missing.stderr
    merged = run(command, {'TF_NUMA_SECCOMP': '/profiles/seccomp.json'})
    assert merged.returncode == 0, merged.stderr
    expected, actual = json.loads(base.stdout), json.loads(merged.stdout)
    assert actual['services']['tensorfold'].pop('security_opt') == ['seccomp=/profiles/seccomp.json']
    assert actual == expected

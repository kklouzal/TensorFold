"""Owned profile transformation, boundary and atomic publication contracts."""

import copy
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
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


def test_v100_compose_requires_worker_plan_and_override_preserves_it(tmp_path):
    docker = shutil.which('docker')
    if docker is None:
        pytest.skip('requires host Docker Compose')
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('TF_', 'TENSORFOLD_', 'MODEL_ROOT', 'GGUF_FILE'))}
    environment.update(MODEL_ROOT='/models', GGUF_FILE='/models/model.gguf')
    required={'TF_GPU_UUIDS':','.join('GPU-00000000-0000-0000-0000-'+str(i+1).zfill(12)for i in range(8)),
              'TF_NUMA_PLAN':'/profiles/plan.json','TF_NUMA_PLAN_SHA256':'a'*64,
              'TF_NUMA_CPUSET':'0-63','TF_NUMA_SECCOMP':'/profiles/seccomp.json'}
    empty_env = tmp_path / 'empty.env';empty_env.write_text('')
    command = [docker, 'compose', '--env-file', str(empty_env), '-f',str(ROOT / 'deploy/v100/compose.yaml')]
    def run(arguments, extra):
        return subprocess.run(arguments,capture_output=True,text=True,timeout=30,env=environment|extra,check=False)
    for name in required:
        missing=run(command+['config','--format','json'],{k:v for k,v in required.items()if k!=name})
        assert missing.returncode!=0 and name in missing.stderr
    base=run(command+['config','--format','json'],required)
    assert base.returncode==0,base.stderr
    merged=run(command+['-f',str(ROOT/'deploy/v100/compose.numa.yaml'),'config','--format','json'],required)
    assert merged.returncode==0,merged.stderr
    expected,actual=json.loads(base.stdout),json.loads(merged.stdout)
    service=actual['services']['tensorfold']
    assert service['security_opt']==['seccomp=/profiles/seccomp.json']
    assert service['cpuset']=='0-63' and service['environment']['NCCL_IGNORE_CPU_AFFINITY']=='0'
    assert '--numa-plan' in service['command'] and '--numa-plan-sha256' in service['command']
    assert any(v['target']=='/run/tensorfold-v100/numa/plan.json' and v['read_only']for v in service['volumes'])
    assert actual==expected


def test_worker_adds_only_exact_static_thread_binding_over_archive(baseline):
    archive=profile.render(baseline,{'versions':'recorded'})
    worker=profile.render(baseline,{'versions':'recorded'},purpose='worker')
    a=json.loads(archive['seccomp.json']);b=json.loads(worker['seccomp.json'])
    assert b['syscalls'][:-1]==a['syscalls']
    assert b['syscalls'][-1]=={'names':['set_mempolicy'],'action':'SCMP_ACT_ALLOW',
                             'args':[{'index':0,'value':32770,'op':'SCMP_CMP_EQ'}]}
    assert json.loads(worker['manifest.json'])['purpose']=='tensorfold-numa-local-rank-worker'
    assert b['architectures']==a['architectures'] and b['defaultAction']==a['defaultAction']


@pytest.fixture
def cap40_source(capture_source,monkeypatch):
    host=profile._host_authority();info=capture_source.info
    authority={'image_id':'sha256:'+'c'*64,'verification_launcher':{
        'mode':host.VERIFICATION_MODE,'base_image_id':'sha256:'+'d'*64,
        'launcher':{'path':host.VERIFICATION_ENTRY,'bytes':16976,'sha256':'1'*64,
            'source_bytes':8200,'source_sha256':'2'*64,'file_capability_hex':host.VERIFICATION_XATTR},
        'python':{'path':host.VERIFICATION_PYTHON,'bytes':8020928,'sha256':'3'*64},
        'providers':{'/opt/venv/lib/python3.12/site-packages/tensorfold/_fd_owner.abi3.so':{'bytes':1,'sha256':'4'*64}}}}
    info.update(image=authority['image_id'],cap_add=['CAP_CHECKPOINT_RESTORE'],cap_drop=['ALL'],user='1000:1000',
        readonly_root=True,pid_mode='',userns_mode='',network_mode='none',groups=None,mounts=[],tmpfs=None,
        entrypoint=[host.VERIFICATION_ENTRY],command=list(profile.CAP40_CAPTURE_ARGS))
    config=json.loads(capture_source.config.read_text())
    config['root']={'readonly':True}
    config['process'].update(user={'uid':1000,'gid':1000,'additionalGids':[1000]},
        args=[host.VERIFICATION_ENTRY,*profile.CAP40_CAPTURE_ARGS],
        capabilities={k:['CAP_CHECKPOINT_RESTORE'] for k in ('bounding','effective','permitted')})
    capture_source.config.write_text(json.dumps(config))
    facts={'status':{'Uid':[1000]*4,'Gid':[1000]*4,'Groups':[1000],
        **{k:host.VERIFICATION_CAP for k in ('CapInh','CapPrm','CapEff','CapBnd','CapAmb')},
        'NoNewPrivs':0,'Seccomp':2,'Seccomp_filters':1,'NSpid':[info['pid'],1]},
        'start_time_ticks':12345,'argv':[host.VERIFICATION_PYTHON,*profile.CAP40_CAPTURE_ARGS]}
    files={'launcher':{'bytes':16976,'sha256':'1'*64,'uid':0,'mode':0o555,'readonly':True,'file_capability_hex':host.VERIFICATION_XATTR},
        'source':{'bytes':8200,'sha256':'2'*64,'uid':0,'mode':0o444,'readonly':True},
        'python':{'bytes':8020928,'sha256':'3'*64,'uid':0,'mode':0o755,'readonly':True}}
    original=profile._docker
    def docker(executable,action,*args):
        if action=='image':
            image=args[-1];layers=['sha256:'+'e'*64]
            return {'Id':image,'RootFS':{'Layers':layers+(['sha256:'+'f'*64] if image==authority['image_id'] else [])}}
        return original(executable,action,*args)
    monkeypatch.setattr(profile,'_docker',docker)
    monkeypatch.setattr(profile,'_process_facts',lambda pid:copy.deepcopy(facts))
    monkeypatch.setattr(profile,'_cap40_files',lambda *a:copy.deepcopy(files))
    return SimpleNamespace(source=capture_source,authority=authority,facts=facts,files=files)


def test_cap40_role_independently_captures_and_derives_only_worker_rules(cap40_source,baseline):
    c=cap40_source
    actual,provenance=profile.capture('fixed',bundle_config=c.source.config,capture_role=profile.CAP40_ROLE,
                                      verification_contract=c.authority)
    files=profile.render(actual,provenance,purpose='worker',capture_role=profile.CAP40_ROLE)
    record=json.loads(files['manifest.json']);derived=json.loads(files['seccomp.json'])
    assert record['schema_version']==2 and record['capture_role']==profile.CAP40_ROLE
    assert record['purpose']=='tensorfold-numa-verification-cap40-worker'
    assert derived['syscalls']==baseline['syscalls']+profile.numa_rules(worker=True)
    assert provenance['actual_process']['status']['CapEff']=='0000010000000000'
    assert c.source.secret.encode() not in b''.join(files.values())
    assert files==profile.render(actual,provenance,purpose='worker',capture_role=profile.CAP40_ROLE)
    with pytest.raises(ValueError):profile.render(actual,provenance,purpose='worker')
    with pytest.raises(ValueError):profile.render(actual,provenance,capture_role=profile.CAP40_ROLE)
    with pytest.raises(ValueError):profile.capture('fixed',bundle_config=c.source.config)


@pytest.mark.parametrize('field,value', [('cap_add',['CAP_SYS_ADMIN']),('cap_drop',None),('user','0:0'),
    ('readonly_root',False),('pid_mode','host'),('userns_mode','host'),('groups',['0']),('mounts',[{}]),
    ('tmpfs',{'/tmp':''}),('network_mode','bridge'),('command',['-c','arbitrary']),('entrypoint',['/bin/sh']),
    ('security_opt',['seccomp=unconfined']),('privileged',True)])
def test_cap40_capture_refuses_every_other_envelope(cap40_source,field,value):
    c=cap40_source;c.source.info[field]=value
    with pytest.raises(ValueError):profile.capture('fixed',bundle_config=c.source.config,
        capture_role=profile.CAP40_ROLE,verification_contract=c.authority)


@pytest.mark.parametrize('field,value', [('Uid',[0]*4),('Groups',[1000,0]),('CapAmb','0000000000000000'),
    ('CapPrm','0000010000200000'),('NoNewPrivs',1),('Seccomp',0),('Seccomp_filters',0),('NSpid',[1234,2])])
def test_cap40_capture_requires_actual_post_launcher_facts(cap40_source,field,value):
    c=cap40_source;c.facts['status'][field]=value
    with pytest.raises(ValueError):profile.capture('fixed',bundle_config=c.source.config,
        capture_role=profile.CAP40_ROLE,verification_contract=c.authority)


@pytest.mark.parametrize('field', ['caps','uid','args','groups','readonly'])
def test_cap40_oci_resolved_role_cannot_be_relabelled(cap40_source,field):
    c=cap40_source;config=json.loads(c.source.config.read_text())
    if field=='caps':config['process']['capabilities']['effective'].append('CAP_SYS_NICE')
    elif field=='uid':config['process']['user']['uid']=0
    elif field=='args':config['process']['args']=['/bin/sh']
    elif field=='groups':config['process']['user']['additionalGids'].append(0)
    else:config['root']['readonly']=False
    c.source.config.write_text(json.dumps(config))
    with pytest.raises(ValueError):profile.capture('fixed',bundle_config=c.source.config,
        capture_role=profile.CAP40_ROLE,verification_contract=c.authority)


def test_cap40_immutable_image_authority_and_pid_generation_are_required(cap40_source,monkeypatch):
    c=cap40_source;c.files['launcher']['file_capability_hex']='00'
    with pytest.raises(ValueError):profile.capture('fixed',bundle_config=c.source.config,
        capture_role=profile.CAP40_ROLE,verification_contract=c.authority)
    c.files['launcher']['file_capability_hex']=profile._host_authority().VERIFICATION_XATTR
    count=0
    def changed(pid):
        nonlocal count
        count+=1;facts=copy.deepcopy(c.facts)
        if count==2:facts['start_time_ticks']+=1
        return facts
    monkeypatch.setattr(profile,'_process_facts',changed)
    with pytest.raises(ValueError,match='changed during capture'):
        profile.capture('fixed',bundle_config=c.source.config,capture_role=profile.CAP40_ROLE,verification_contract=c.authority)

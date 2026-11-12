"""Container NUMA authority/staging boundaries; native placement has separate gates."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tools import v100_onecat_campaign as campaign


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


@pytest.fixture
def inputs(monkeypatch, tmp_path):
    cpu = min(os.sched_getaffinity(0))
    devices = [f'GPU-00000000-0000-0000-0000-{rank+1:012x}' for rank in range(8)]
    plan = {'ordered_gpu_uuids': devices, 'ranks': [{'rank': n} for n in range(8)],
            'cpuset_cpus': [cpu], 'cpuset_mems': [0, 1]}
    root = tmp_path / 'authority'; root.mkdir()
    plan_path = root / 'plan.json'; plan_path.write_bytes(encode(plan))
    policy = root / 'seccomp'; policy.mkdir()
    data = {'baseline-seccomp.json': encode({'baseline': True}),
            'seccomp.json': encode({'defaultAction': 'SCMP_ACT_ERRNO'}),
            'manifest.json': encode({'purpose': 'tensorfold-numa-local-rank-worker', 'provenance': {'captured': True}})}
    for name, value in data.items():
        (policy/name).write_bytes(value)
    calls = []
    def load(path, sha):
        assert path == plan_path and sha == hashlib.sha256(plan_path.read_bytes()).hexdigest()
        calls.append('plan-validated')
        return plan
    def render(baseline, provenance, *, purpose):
        assert baseline == {'baseline': True} and provenance == {'captured': True} and purpose == 'worker'
        calls.append('policy-reconstructed')
        return data
    monkeypatch.setattr(campaign, '_numa_modules', lambda *a: (NS(load_plan=load), NS(render=render)))
    settings = {'gpus': devices, 'numa': {'plan': str(plan_path),
        'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(), 'seccomp_directory': str(policy),
        'seccomp_manifest_sha256': hashlib.sha256(data['manifest.json']).hexdigest()}}
    return tmp_path, settings, data, calls


def test_required_numaplan_and_reconstructed_policy_stage_readonly_with_receipts(inputs):
    root, settings, data, calls = inputs
    bundle = campaign._numa_inputs(root, {'source_uid': 'fixture'}, settings)
    operator = root/'operator'; operator.mkdir()
    output = root/'output'; output.mkdir()
    staged = campaign._stage_numa(bundle, operator, output)
    assert calls == ['plan-validated', 'policy-reconstructed']
    assert staged['operator_plan'] == '/operator/numa/plan.json'
    assert staged['plan']['ordered_gpu_uuids'] == settings['gpus']
    assert staged['receipt_directory'] == '/evidence/numa'
    assert (operator/'numa').stat().st_mode & 0o777 == 0o555
    for name in ['plan.json', *data]:
        assert (operator/'numa'/name).stat().st_mode & 0o777 == 0o444
        assert campaign.identity(operator/'numa'/name) == staged['files'][name]
    assert (output/'numa').is_dir()


@pytest.mark.parametrize('mutation', ['missing', 'wrong-order', 'policy-bytes', 'extra-file', 'archive-purpose'])
def test_numa_authority_mismatch_refuses_before_operator_creation(inputs, mutation):
    root, settings, data, _ = inputs
    if mutation == 'missing':
        del settings['numa']
    elif mutation == 'wrong-order':
        settings['gpus'] = list(reversed(settings['gpus']))
    elif mutation == 'policy-bytes':
        (Path(settings['numa']['seccomp_directory'])/'seccomp.json').write_bytes(b'{}')
    elif mutation == 'extra-file':
        (Path(settings['numa']['seccomp_directory'])/'unexpected').write_bytes(b'bad')
    else:
        data['manifest.json'] = encode({'purpose': 'tensorfold-numa-local-archive', 'provenance': {'captured': True}})
        (Path(settings['numa']['seccomp_directory'])/'manifest.json').write_bytes(data['manifest.json'])
        settings['numa']['seccomp_manifest_sha256'] = hashlib.sha256(data['manifest.json']).hexdigest()
    with pytest.raises(ValueError):
        campaign._numa_inputs(root, {'source_uid': 'fixture'}, settings)
    assert not (root/'operator').exists()


def test_bounded_numa_file_rejects_links_and_excess_before_read(tmp_path):
    target = tmp_path/'data'; target.write_bytes(b'12345')
    link = tmp_path/'link'; link.symlink_to(target)
    with pytest.raises(ValueError):
        campaign._bounded_numa_file(link)
    with pytest.raises(ValueError):
        campaign._bounded_numa_file(target, maximum=4)

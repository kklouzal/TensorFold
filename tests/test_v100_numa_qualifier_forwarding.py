"""NUMA policy reaches the common pre-exec launcher without public serve flags."""
import pytest

from tools import v100_owned_qualification as owned
from tools import launch_cuda_tp as launcher


def test_private_qualifier_forwards_sealed_numa_plan_before_serve_separator(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(launcher, 'main', lambda argv, **kwargs: calls.append((argv, kwargs)) or 0)
    plan = tmp_path / 'numa-plan.json'
    result = owned.main(['--gpus', ','.join(str(n) for n in range(8)), '--out', str(tmp_path/'out'),
                         '--numa-plan', str(plan), '--numa-plan-sha256', 'a'*64,
                         '--', 'model', '--no-drafts'])
    assert result == 0 and len(calls) == 1
    command = calls[0][0]
    assert command[command.index('--numa-plan')+1] == str(plan.absolute())
    assert command[command.index('--numa-plan-sha256')+1] == 'a'*64
    assert command.index('--numa-plan') < command.index('--')
    assert command[command.index('--')+1:] == ['model', '--no-drafts']


@pytest.mark.parametrize('arguments', [['--numa-plan', '/plan'], ['--numa-plan-sha256', 'a'*64]])
def test_incomplete_numa_authority_refuses_before_output_or_launch(monkeypatch, tmp_path, arguments):
    monkeypatch.setattr(launcher, 'main', lambda *a, **k: pytest.fail('incomplete NUMA authority launched'))
    output = tmp_path / 'out'
    with pytest.raises(SystemExit):
        owned.main(['--gpus', ','.join(str(n) for n in range(8)), '--out', str(output), *arguments,
                    '--', 'model', '--no-drafts'])
    assert not output.exists()

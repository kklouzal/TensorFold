"""Capacity procedure contracts; synthetic served identities, no live HTTP."""
import hashlib
import json
from types import SimpleNamespace

import pytest
from tools import gemma_shared_prefix_capacity_qualify as q


class Out:
    def __init__(self):
        self.rows = {}
    def save(self, name, value):
        self.rows[name] = value


def reply(temperature=0, ignore=False):
    tokens = list(range(100, 116)) if ignore else [100, 1]
    return {'prompt': q.PROMPT, 'output': len(tokens), 'cached': q.PROMPT,
        'tensorfold': {'token_ids': tokens, 'drafts': True},
        'message': {'content': 'READY' if temperature == 0 else 'Useful answer'},
        'finish': 'length' if ignore else 'stop',
        'usage': {'prompt_tokens': q.PROMPT, 'completion_tokens': len(tokens), 'total_tokens': q.PROMPT + len(tokens)}}


def test_exact_native_window_and_tail_survive_served_crop():
    assert (q.PROMPT, q.OUTPUT, q.CONTEXT) == (262128, 16, 262144)
    full = list(range(q.PROMPT + 11))
    class Client:
        out = Out()
        def tokens(self, name, body):
            return full if name == 'native-source' else full[:3] + full[-5:]
    client = Client()
    actual = q.boundary_ids(client, {'source_body': {'messages': [{'content': 'public'}]}})
    assert len(actual) == 262128 and actual[:3] == full[:3] and actual[-4096:] == full[-4096:]
    assert client.out.rows['native-served-input.json']['verified_frame_suffix'] == 5


@pytest.mark.parametrize('temperature,ignore', [(0, False), (0, True), (1, False), (1, True)])
def test_ordinary_gold_and_fixed_work_with_exact_own_replay(temperature, ignore):
    value = reply(temperature, ignore)
    actual = q.validate_native(value, temperature=temperature, ignore_eos=ignore)
    assert q.validate_native(value, temperature=temperature, ignore_eos=ignore, previous=actual) == actual


@pytest.mark.parametrize('mutation', ['gold', 'work', 'total', 'cached', 'boolean_id', 'draft'])
def test_original_independent_oracles_cannot_be_waived(mutation):
    value = reply(0, mutation == 'work')
    if mutation == 'gold':
        value['message']['content'] = 'NOT READY'
    elif mutation == 'work':
        value['tensorfold']['token_ids'].pop()
    elif mutation == 'total':
        value['usage']['total_tokens'] = 131072
    elif mutation == 'cached':
        value['cached'] -= 1
    elif mutation == 'boolean_id':
        value['tensorfold']['token_ids'][0] = True
    else:
        value['tensorfold']['drafts'] = False
    with pytest.raises(ValueError):
        q.validate_native(value, temperature=0, ignore_eos=mutation == 'work', previous=[100, 1])


def test_closed_input_hash_and_content_are_checked_before_effects(tmp_path):
    spec = q.default_inputs()
    path = tmp_path / 'input.json'
    raw = json.dumps(spec).encode()
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert q.load_inputs(path, digest) == spec
    with pytest.raises(ValueError, match='identity'):
        q.load_inputs(path, '0' * 64)
    spec['source_body']['messages'][0]['content'] = 'bad'
    raw = json.dumps(spec).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match='content/gold'):
        q.load_inputs(path, hashlib.sha256(raw).hexdigest())


def test_native_schedule_keeps_both_gold_modes_and_stops_on_first_failure(monkeypatch):
    calls, out, rows = [], Out(), []
    monkeypatch.setattr(q, 'boundary_ids', lambda *args: [7] * q.PROMPT)
    monkeypatch.setattr(q, 'settle', lambda *args: {'idle': True})
    primary = RuntimeError('native model failed')
    class Client:
        def stats(self, name):
            return {}
        def generate(self, name, path, body):
            calls.append(body)
            if len(calls) == 3:
                raise primary
            return reply(body['temperature'], body['ignore_eos'])
    client = Client()
    client.out = out
    with pytest.raises(RuntimeError) as caught:
        q.run(client, {'model': 'm'}, rows=rows)
    assert caught.value is primary and len(calls) == 3 and len(rows) == 2
    assert all(body['seed'] == 7 and body['max_tokens'] == 16 for body in calls)
    assert calls[0]['ignore_eos'] is False and calls[2]['ignore_eos'] is True


def test_settlement_requires_actual_typed_idle_and_uses_local_clock_binding(monkeypatch):
    ticks, calls = [0.], []
    monkeypatch.setattr(q, 'time', SimpleNamespace(monotonic=lambda: ticks[0], sleep=lambda n: ticks.__setitem__(0, ticks[0] + n)))
    class Health:
        def __init__(self, *args, **kwargs):
            pass
        def json(self, name, path):
            calls.append(name)
            return {'ok': True, 'busy': len(calls) == 1, 'requests_running': int(len(calls) == 1)}
    monkeypatch.setattr(q.core, 'Client', Health)
    client = SimpleNamespace(port=18888, out=Out(), arm='baseline', stats=lambda name: {})
    assert q.settle(client, 'case', timeout=3)['observed_idle_after_s'] == 1
    assert len(calls) == 2


def test_failure_publication_preserves_authentic_primary(monkeypatch, tmp_path):
    primary, secondary = RuntimeError('model failed'), OSError('summary failed')
    monkeypatch.setattr(q, 'load_inputs', lambda *args: {})
    monkeypatch.setattr(q, 'run', lambda *args, **kwargs: (_ for _ in ()).throw(primary))
    monkeypatch.setattr(q.core, 'Evidence', lambda path: SimpleNamespace(save=lambda *args: (_ for _ in ()).throw(secondary)))
    with pytest.raises(RuntimeError) as caught:
        q.main(['--inputs', 'unused', '--inputs-sha256', '0' * 64, '--base-url', 'http://127.0.0.1:18888',
                '--arm', 'candidate', '--out', str(tmp_path / 'out')])
    assert caught.value is primary

"""Prospective timing/input contracts using only synthetic HTTP entities."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tools import gemma_shared_prefix_timing_qualify as q


def wire(tokens=128, done=True):
    def event(value):
        return ('data: ' + json.dumps(value) + '\n\n').encode()
    raw = event({'choices': [{'index': 0, 'delta': {'content': 'public response'}, 'finish_reason': None}]})
    raw += event({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'length'}]})
    raw += event({'choices': [], 'usage': {'prompt_tokens': 10, 'completion_tokens': tokens,
        'total_tokens': 10 + tokens, 'prompt_tokens_details': {'cached_tokens': 4}},
        'tensorfold': {'token_ids': [7] * tokens, 'drafts': True, 'prefill_s': 1., 'decode_s': 2., 'queue_s': 0.}})
    return raw + (b'data: [DONE]\n\n' if done else b'')


def test_mirrored_schedule_is_closed_and_preserves_dependency_order():
    rows = q.schedule()
    assert len(rows) == 36
    assert [row['region'] for row in rows[::6]] == ['short', 'code64k', 'prose64k', 'prose64k', 'code64k', 'short']
    for start in range(0, 36, 3):
        assert [row['state'] for row in rows[start:start + 3]] == ['cold', 'warm', 'canonical-growth']
    assert q.generation('m', 1) == {'model': 'm', 'max_tokens': 128, 'temperature': 1, 'seed': 7,
        'ignore_eos': True, 'return_token_ids': True, 'stream': True, 'stream_options': {'include_usage': True},
        'top_k': 64, 'top_p': .95, 'min_p': 0.}
    assert 'top_k' not in q.generation('m', 0)


def test_served_nonce_composition_preserves_fixed_raw_extent_and_final_task():
    original = list(range(65536))
    value = {'name': 'code64k', 'raw_ids': original, 'frame_prefix_tokens': 8}
    result = q.cold_raw(value, [901, 902])
    assert len(result) == 65536 and result[:8] == original[:8] and result[8:10] == [901, 902]
    assert result[-2048:] == original[-2048:] and original == list(range(65536))
    value['name'] = 'short'
    value['raw_ids'] = list(range(30))
    assert len(q.cold_raw(value, [901, 902])) == 32


def test_tokenizer_only_rebinding_keeps_raw_and_new_chat_separate():
    full, calls = list(range(65600)), []
    cropped = full[:63488] + full[-2048:]
    class Client:
        def tokens(self, name, body):
            calls.append((name, body))
            if name.endswith('source-open'):
                return full
            if name.endswith('empty-open'):
                return full[:3] + full[-4:]
            if name.endswith('rebound-open'):
                return cropped + [999]
            assert body['add_generation_prompt'] is False
            return cropped[:-2]
        def json(self, name, path, body):
            assert path == '/v1/detokenize' and body['tokens'] == cropped[3:-4]
            return {'prompt': 'public reconstructed content'}
    value = q.prepare_region(Client(), 'code64k', 'public original', cropped,
                             source_content_sha256=hashlib.sha256(b'public original').hexdigest())
    assert value['raw_matches_legacy'] and not value['chat_equals_raw']
    assert len(value['raw_ids']) == 65536 and len(value['chat_open_ids']) == 65537
    assert value['chat_body']['messages'][0]['content'] == 'public reconstructed content'
    assert len(calls) == 4


def test_bounded_strict_sse_keeps_fullwork_phases_and_publication_gap(monkeypatch):
    monkeypatch.setattr(q, 'time', SimpleNamespace(monotonic=lambda: 104.))
    class Client:
        def call(self, *args):
            return wire(), 'tag', {'arrival_monotonic': 100., 'full_wall_s': 3., 'terminal_s': 2.9}
    result, record = q.exchange(Client(), 'case', '/v1/completions', q.generation('m', 1), 10)
    assert result['output'] == 128 and result['cached'] == 4
    assert record['evidence_publication_gap_s'] == 1
    assert record['uncached_original_input_tokens'] == 6
    assert record['phase_decode_tps'] == 127 / 2
    assert record['whole_request_output_tps'] == 128 / 3


@pytest.mark.parametrize('tokens,done', [(127, True), (128, False)])
def test_incomplete_output_or_missing_DONE_cannot_be_a_success_sample(tokens, done):
    client = SimpleNamespace(call=lambda *args: (wire(tokens, done), 'tag',
        {'arrival_monotonic': 100., 'full_wall_s': 3., 'terminal_s': 2.9}))
    with pytest.raises(ValueError):
        q.exchange(client, 'case', '/v1/completions', q.generation('m', 1), 10)


def test_reference_identity_and_extent_checked_before_parsing(tmp_path):
    path = tmp_path / 'ref.json'
    raw = b'{"public": true}'
    path.write_bytes(raw)
    ref = {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    assert q.read_ref(ref) == {'public': True}
    ref['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='changed'):
        q.read_ref(ref)


def test_descriptive_distributions_keep_changed_inputs_visible():
    def row(value, sha):
        return {'region': 'short', 'temperature': 0, 'state': 'canonical-growth', 'input_kind': 'Chat',
            'timing': {'full_wall_s': value}, 'full_cell_s': value + 1,
            'next_probe': {'timing': {'first_visible_s': value / 2}}, 'HTTP_done_to_probe_arrival_s': .1,
            'request_sha256': sha, 'output_ids_sha256': sha}
    summary = q.distributions([row(1., 'a'), row(2., 'b')])[0]
    assert summary['same_input_all_repeats'] is False and summary['output_ids_equal'] is False
    assert summary['measures']['HTTP_complete_wall_s'] == {'raw': [1., 2.], 'observed': 2, 'missing': 0, 'minimum': 1., 'median': 1.5, 'maximum': 2.}


def test_growth_uses_actual_wire_assistant_then_served_closed_open_ids(monkeypatch):
    calls, saved = [], {}
    monkeypatch.setattr(q, 'schedule', lambda: [{'region': 'short', 'temperature': 0, 'state': 'canonical-growth', 'traversal': 0}])
    monkeypatch.setattr(q.native, 'settle', lambda *args: {'idle': True})
    def exchange(client, name, path, body, expected):
        calls.append((name, body, expected))
        result = {'message': {'role': 'assistant', 'content': 'ACTUAL', 'reasoning_content': '', 'tool_calls': []}}
        return result, {'timing': {'arrival_monotonic': 1., 'full_wall_s': 1.},
                        'request_sha256': 's', 'output_ids_sha256': 'o'}
    monkeypatch.setattr(q, 'exchange', exchange)
    class Client:
        out = SimpleNamespace(save=lambda name, value: saved.__setitem__(name, value))
        def stats(self, name):
            return {}
        def tokens(self, name, body):
            assert body['messages'][1]['content'] == 'ACTUAL'
            return [1, 2] if body.get('add_generation_prompt') is False else [1, 2, 3]
    region = {'name': 'short', 'raw_ids': [1, 2], 'frame_prefix_tokens': 1,
              'chat_open_ids': [1, 2], 'chat_equals_raw': True,
              'chat_body': {'model': 'm', 'messages': [{'role': 'user', 'content': 'public'}],
                            'chat_template_kwargs': {'enable_thinking': True}}}
    result = q.run(Client(), {'model': 'm', 'cold_nonce_ids': {'t0-temp0': [7]}}, {'short': region})
    assert calls[1][1]['messages'][1]['content'] == 'ACTUAL' and calls[1][2] == 3
    assert len(calls) == 3 and calls[-1][1]['max_tokens'] == 1
    assert result[0]['next_probe_scope'].startswith('observed gap')


def frozen_inputs(tmp_path, mutation=None):
    refs = {}
    for name in q.REGIONS:
        raw = list(range(30 if name == 'short' else 65536))
        region = {'name': name, 'raw_ids': raw, 'legacy_raw_ids': raw.copy(), 'raw_matches_legacy': True,
            'chat_body': {'model': 'm', 'messages': [{'role': 'user', 'content': 'public'}],
                          'chat_template_kwargs': {'enable_thinking': True}},
            'chat_open_ids': raw + [3], 'chat_closed_ids': raw[:-2], 'chat_equals_raw': False, 'frame_prefix_tokens': 3}
        if name == 'code64k' and mutation == 'raw_binding':
            region['legacy_raw_ids'][-1] += 1
            region['raw_matches_legacy'] = False
        if name == 'code64k' and mutation == 'boolean_prefix':
            region['frame_prefix_tokens'] = True
        path = tmp_path / (name + '.json')
        data = json.dumps(region).encode()
        path.write_bytes(data)
        refs[name] = {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    spec = {'schema': q.SCHEMA, 'model': 'm', 'authority': {'model_revision': q.native.MODEL_REVISION,
            'tokenizer_sha256': q.native.TOKENIZER_SHA, 'template_sha256': q.native.TEMPLATE_SHA},
        'legacy_input_sha256': q.LEGACY_SHA, 'generator_sha256': q.GENERATOR_SHA, 'regions': refs,
        'cold_nonce_ids': {key: [7, 8] for key in q.NONCES}}
    if mutation == 'nonce_missing':
        spec['cold_nonce_ids'].pop(q.NONCES[0])
    path = tmp_path / 'manifest.json'
    data = json.dumps(spec).encode()
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def test_full_closed_manifest_keeps_new_chat_region_distinct(tmp_path):
    spec, regions = q.load_inputs(*frozen_inputs(tmp_path))
    assert spec['schema'] == q.SCHEMA and len(regions['code64k']['raw_ids']) == 65536
    assert len(regions['code64k']['chat_open_ids']) == 65537 and regions['code64k']['chat_equals_raw'] is False


@pytest.mark.parametrize('mutation', ['raw_binding', 'boolean_prefix', 'nonce_missing'])
def test_bad_or_unbound_inputs_fail_before_model_work(tmp_path, mutation):
    with pytest.raises(ValueError):
        q.load_inputs(*frozen_inputs(tmp_path, mutation))



def test_source_content_hash_refuses_before_any_http_inspection():
    client = SimpleNamespace(tokens=lambda *args: pytest.fail('unbound content reached HTTP'))
    with pytest.raises(ValueError, match='identity differs before tokenization'):
        q.prepare_region(client, 'code64k', 'wrong public source', [1], source_content_sha256='0' * 64)



def test_actual_B_result_schema_projects_phases_without_invented_defaults():
    path = Path(__file__).parent / 'fixtures' / 'gemma_shared_prefix_actual_B_raw_result_v2.json'
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '1228128219ad5e4022d9b0b5a7f6e8ef020614300aa091c7151ded824d5107a8'
    actual = json.loads(raw)
    stats = actual['server']
    assert stats['drafts'] is True and len(stats['token_ids']) == actual['output_tokens']
    result = q.phase_projection(stats, actual['prompt_tokens'], actual['output_tokens'], actual['cached_tokens'])
    assert result['effective_prefill_tps'] == 23996 / 6.6511
    assert result['phase_decode_tps'] == 5 / .4511
    assert result['optional_server_phases']['cache_publication_s'] == .6062054699286819
    assert result['optional_server_phases']['frontend_settlement_s'] == .6075970779638737
    assert not result['unavailable_server_phases']
    assert actual['timing']['terminal_s'] is actual['timing']['first_visible_s'] is None  # real NONstream record
    legacy = dict(stats)
    legacy.pop('cache_publication')
    legacy.pop('frontend_settlement_s')
    projected = q.phase_projection(legacy, 24000, 6, 4)
    assert projected['optional_server_phases']['cache_publication_s'] is None
    assert projected['optional_server_phases']['frontend_settlement_s'] is None

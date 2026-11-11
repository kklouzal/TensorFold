"""Prospective B/A2 fixed-work SSE timings; no service lifecycle or promotion.

Client.call owns bounded HTTP transport. openai_protocol validates the preserved
SSE entity. Its evidence publication precedes our next probe: that exact gap is
measured, never called immediate/zero-gap queued overlap. Historical A1 is not
rerun or relabeled by this tool. Importing performs no I/O.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import math
import re
import statistics
import time
from pathlib import Path

if __package__:
    from . import gemma_shared_prefix_qualify as core
    from . import gemma_shared_prefix_capacity_qualify as native
    from . import openai_protocol as protocol
else:
    import gemma_shared_prefix_qualify as core
    import gemma_shared_prefix_capacity_qualify as native
    import openai_protocol as protocol

SCHEMA = 'tensorfold.gemma-regional-timing-input-v1'
LEGACY_SHA = 'fb9aafe715f75ad7b43046ddc59567d025df881c0bcd4119ceba63851e6f2d63'
GENERATOR_SHA = 'e5d2e735baa6abdabe66318655671ab5a5cb1f28ec9fbf9e7ed3f1f55e520f24'
HISTORICAL_RUNNER_SHA = '10f314f926fb2a181e63a609849941631bac6dd809c1a8fbc4ff0084f05ad9c2'
REGIONS = ('short', 'code64k', 'prose64k')
NONCES = tuple(f't{traversal}-temp{temperature}' for traversal in (0, 1) for temperature in (0, 1))
TOKENS = 128


def ids(value, maximum=262144):
    core.require(type(value) is list and 0 < len(value) <= maximum
                 and all(type(token) is int and 0 <= token < 262144 for token in value), 'bounded native token IDs required')
    return value


def read_ref(ref):
    core.require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'}, 'closed frozen input reference required')
    size = core.integer(ref['bytes'], 'reference bytes')
    core.require(0 < size <= core.CAP and type(ref['sha256']) is str
                 and re.fullmatch('[0-9a-f]{64}', ref['sha256']), 'bounded reference identity required')
    raw = core.read_bound(ref['path'], size)
    core.require(len(raw) == size and hashlib.sha256(raw).hexdigest() == ref['sha256'], 'referenced input changed')
    return protocol.object_json(raw)


def load_inputs(path, sha256):
    raw = core.read_bound(path, core.CAP)
    core.require(hashlib.sha256(raw).hexdigest() == sha256, 'timing manifest identity differs')
    spec = protocol.object_json(raw)
    core.require(set(spec) == {'schema', 'model', 'authority', 'legacy_input_sha256', 'generator_sha256',
                              'regions', 'cold_nonce_ids'} and spec['schema'] == SCHEMA,
                 'closed prospective timing input schema required')
    core.require(spec['authority'] == {'model_revision': native.MODEL_REVISION,
        'tokenizer_sha256': native.TOKENIZER_SHA, 'template_sha256': native.TEMPLATE_SHA}
        and spec['legacy_input_sha256'] == LEGACY_SHA and spec['generator_sha256'] == GENERATOR_SHA,
        'historical source/asset binding differs')
    core.require(type(spec['model']) is str and type(spec['regions']) is dict and set(spec['regions']) == set(REGIONS)
                 and type(spec['cold_nonce_ids']) is dict and set(spec['cold_nonce_ids']) == set(NONCES), 'three regions/four prospective cold tapes required')
    for value in spec['cold_nonce_ids'].values():
        ids(value, 128)
    regions = {name: read_ref(spec['regions'][name]) for name in REGIONS}
    for name, region in regions.items():
        core.require(set(region) == {'name', 'raw_ids', 'legacy_raw_ids', 'raw_matches_legacy', 'chat_body',
                                     'chat_open_ids', 'chat_closed_ids', 'chat_equals_raw', 'frame_prefix_tokens'}
                     and region['name'] == name, 'closed region binding differs')
        for key in ('raw_ids', 'legacy_raw_ids', 'chat_open_ids', 'chat_closed_ids'):
            ids(region[key])
        core.require(type(region['raw_matches_legacy']) is bool and region['raw_matches_legacy'] ==
                     (region['raw_ids'] == region['legacy_raw_ids']), 'raw legacy equivalence misreported')
        core.require(type(region['chat_equals_raw']) is bool and region['chat_equals_raw'] ==
                     (region['chat_open_ids'] == region['raw_ids']), 'Chat/raw equivalence misreported')
        core.require(type(region['frame_prefix_tokens']) is int
                     and 0 < region['frame_prefix_tokens'] < len(region['raw_ids']), 'served frame prefix required')
        body = region['chat_body']
        core.require(type(body) is dict and set(body) == {'model', 'messages', 'chat_template_kwargs'}
                     and body['model'] == spec['model'] and type(body['messages']) is list and len(body['messages']) == 1
                     and type(body['messages'][0]) is dict and set(body['messages'][0]) == {'role', 'content'}
                     and body['messages'][0]['role'] == 'user' and type(body['messages'][0]['content']) is str
                     and body['chat_template_kwargs'] == {'enable_thinking': True}, 'frozen actual Chat body required')
        if name != 'short':
            core.require(region['raw_matches_legacy'], 'historical RAW corpus no longer matches served source; replan before generation')
            core.require(len(region['raw_ids']) == len(region['legacy_raw_ids']) == 65536,
                         '64K RAW region extent differs')
    return spec, regions


def prepare_region(client, name, content, legacy_ids=None, *, source_content_sha256=None):
    """Tokenizer-only preparation; never generates or accesses local tokenizers."""
    core.require(name in REGIONS and type(content) is str and content, 'public source required')
    if name != 'short':
        core.require(type(source_content_sha256) is str
                     and hashlib.sha256(content.encode()).hexdigest() == source_content_sha256,
                     'historical public source content identity differs before tokenization')
    body = {'model': 'gemma4-26b-a4b', 'messages': [{'role': 'user', 'content': content}],
            'chat_template_kwargs': {'enable_thinking': True}}
    full = client.tokens(name + '-source-open', body)
    empty = copy.deepcopy(body)
    empty['messages'][0]['content'] = ''
    frame = client.tokens(name + '-empty-open', empty)
    prefix, suffix = core.prefix_length(full, frame), 0
    while suffix < min(len(full), len(frame)) - prefix and full[-suffix - 1] == frame[-suffix - 1]:
        suffix += 1
    core.require(0 < prefix < len(full) and suffix > 0, 'actual Chat frame cannot be separated')
    raw = full if name == 'short' else full[:63488] + full[-2048:]
    core.require(name == 'short' or len(full) >= 65536 and len(raw) == 65536, 'source cannot supply64K native crop')
    if name != 'short':
        decoded = client.json(name + '-crop-content', '/v1/detokenize', {'tokens': raw[prefix:len(raw) - suffix]})
        core.require(type(decoded.get('prompt')) is str, 'native detokenized public content missing')
        body['messages'][0]['content'] = decoded['prompt']
    opened = client.tokens(name + '-rebound-open', body)
    closed = client.tokens(name + '-rebound-closed', {**body, 'add_generation_prompt': False})
    legacy = raw if legacy_ids is None else ids(legacy_ids)
    return {'name': name, 'raw_ids': raw, 'legacy_raw_ids': legacy, 'raw_matches_legacy': raw == legacy,
            'chat_body': body, 'chat_open_ids': opened, 'chat_closed_ids': closed,
            'chat_equals_raw': opened == raw, 'frame_prefix_tokens': prefix}


def prepare_nonces(client):
    return {key: client.tokens('nonce-' + key, {'model': 'gemma4-26b-a4b',
        'prompt': f' Qualification nonce shared-prefix-{key}. ', 'add_special_tokens': False}) for key in NONCES}


def cold_raw(region, nonce):
    prefix, raw = region['frame_prefix_tokens'], region['raw_ids']
    if region['name'] == 'short':
        return raw[:prefix] + nonce + raw[prefix:]
    core.require(prefix + len(nonce) < len(raw) - 2048, 'cold nonce would change final task/framing')
    result = raw[:prefix] + nonce + raw[prefix + len(nonce):]
    core.require(len(result) == 65536 and result[-2048:] == raw[-2048:], 'fixed64K cold tape differs')
    return result


def schedule():
    """36 primary cells;12 declared Chat seeds +36 one-token next probes."""
    rows = []
    for traversal in (0, 1):
        for region in (REGIONS if traversal == 0 else tuple(reversed(REGIONS))):
            for temperature in ((0, 1) if traversal == 0 else (1, 0)):
                for state in ('cold', 'warm', 'canonical-growth'):
                    rows.append({'traversal': traversal, 'region': region, 'temperature': temperature, 'state': state})
    return rows


def generation(model, temperature, *, tokens=TOKENS):
    body = {'model': model, 'max_tokens': tokens, 'temperature': temperature, 'seed': 7,
            'ignore_eos': True, 'return_token_ids': True, 'stream': True, 'stream_options': {'include_usage': True}}
    if temperature:
        body.update(top_k=64, top_p=.95, min_p=0.)
    return body



def phase_projection(stats, prompt, output, cached):
    """Actual emitted field names; absent optional phases remain unavailable."""
    for key in ('prefill_s', 'decode_s', 'queue_s'):
        value = stats.get(key)
        core.require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'invalid server phase ' + key)
    optional = {key: stats.get(key) for key in ('prepare_s', 'frontend_settlement_s', 'request_wall_s')}
    publication = stats.get('cache_publication')
    core.require(publication is None or type(publication) is dict, 'invalid publication phase container')
    optional['cache_publication_s'] = None if publication is None else publication.get('publication_s')
    for key, value in optional.items():
        core.require(value is None or type(value) in (int, float) and math.isfinite(value) and value >= 0,
                     'invalid optional server phase ' + key)
    return {'effective_prefill_tps': (prompt - cached) / stats['prefill_s'] if stats['prefill_s'] > 0 else None,
        'phase_decode_tps': (output - 1) / stats['decode_s'] if stats['decode_s'] > 0 else None,
        'optional_server_phases': optional, 'unavailable_server_phases': [key for key, value in optional.items() if value is None],
        'server_phase_precision': 'as emitted, possibly rounded; no subresolution claims'}


def exchange(client, name, path, body, expected_prompt):
    raw, tag, marks = client.call(name, path, body)
    returned = time.monotonic()
    # Reuse the maintained strict bounded reader, not a second SSE transport.
    for chunk in protocol.sse_objects(io.BytesIO(raw), limits=protocol.Limits(core.CAP, 65536)):
        protocol.chunk_fields(chunk)
    result = core.normalize(raw, True)
    output = protocol.completion_usage(result['usage'], body['max_tokens'], prompt=True)
    core.require(output == body['max_tokens'] and result['usage']['prompt_tokens'] == expected_prompt
                 and result['usage']['total_tokens'] == expected_prompt + output and result['finish'] == 'length',
                 'fixed accepted work/native input extent differs')
    actual = ids(result['tensorfold'].get('token_ids'), body['max_tokens'])
    core.require(len(actual) == output and result['tensorfold'].get('drafts') is True,
                 'actual accepted IDs/draft path missing')
    cached = core.integer(result['usage'].get('prompt_tokens_details', {}).get('cached_tokens'), 'cached usage')
    core.require(cached <= expected_prompt and marks['terminal_s'] is not None, 'cache or terminal accounting differs')
    phases = phase_projection(result['tensorfold'], expected_prompt, output, cached)
    result.update(prompt=expected_prompt, output=output, cached=cached)
    record = {'tag': tag, 'request_sha256': native.digest(body), 'timing': marks,
        'client_call_return_monotonic': returned,
        'evidence_publication_gap_s': returned - marks['arrival_monotonic'] - marks['full_wall_s'],
        'prompt_tokens': expected_prompt, 'accepted_output_tokens': output, 'cached_tokens': cached,
        'uncached_original_input_tokens': expected_prompt - cached, 'output_ids_sha256': native.digest(actual),
        'server_phases': result['tensorfold'], 'whole_request_output_tps': output / marks['full_wall_s'],
        **phases, 'DONE_to_HTTP_close_s': marks['full_wall_s'] - marks['terminal_s'],
        'scope': 'client HTTP/text-arrival phases; SSE pieces maybatchtokens; server phases retain actual scope'}
    return result, record


def run(client, spec, regions, *, rows=None):
    client.stats('timing-initial')
    rows = [] if rows is None else rows
    for number, cell in enumerate(schedule()):
        region = regions[cell['region']]
        name = f'{number:02d}-{cell["region"]}-t{cell["temperature"]}-{cell["state"]}'
        controls = generation(spec['model'], cell['temperature'])
        nonce = spec['cold_nonce_ids'][f't{cell["traversal"]}-temp{cell["temperature"]}']
        raw = cold_raw(region, nonce)
        seed_record = None
        if cell['state'] == 'canonical-growth':
            seed_body = {**region['chat_body'], **controls}
            seed, seed_record = exchange(client, name + '-seed', '/v1/chat/completions', seed_body,
                                         len(region['chat_open_ids']))
            messages = copy.deepcopy(region['chat_body']['messages']) + [core.assistant(seed)]
            completed = {**region['chat_body'], 'messages': messages}
            closed = client.tokens(name + '-actual-completed', {**completed, 'add_generation_prompt': False})
            messages += [{'role': 'user', 'content': 'Continue with one additional concrete example.'}]
            body = {**region['chat_body'], **controls, 'messages': messages}
            prompt = client.tokens(name + '-actual-next', body)
            client.out.save(name + '-actual-growth-input.json', {'body': body, 'open_ids': prompt,
                'closed_ids': closed, 'compatible_prefix_tokens': core.prefix_length(closed, prompt)})
            path, count = '/v1/chat/completions', len(prompt)
        else:
            body, path, count = {**controls, 'prompt': raw}, '/v1/completions', len(raw)
        record = exchange(client, name, path, body, count)[1]
        # Current Client publishes evidence before returning. The probe exposes
        # only the observed arrival/queue delay AFTER that measured gap.
        probe_body = {**generation(spec['model'], 0, tokens=1), 'prompt': regions['short']['raw_ids']}
        probe = exchange(client, name + '-next-probe', '/v1/completions', probe_body,
                         len(regions['short']['raw_ids']))[1]
        record.update(cell)
        record['input_kind'] = 'actual-re-rendered-Chat' if cell['state'] == 'canonical-growth' else 'frozen-native-RAW'
        record['chat_equals_raw'] = region['chat_equals_raw']
        record['chat_seed'] = seed_record
        record['next_probe'] = probe
        record['HTTP_done_to_probe_arrival_s'] = (probe['timing']['arrival_monotonic']
            - record['timing']['arrival_monotonic'] - record['timing']['full_wall_s'])
        record['next_probe_scope'] = 'observed gap includes evidence publication/validation; NOT immediate queued overlap'
        record['settlement_after_primary_and_probe'] = native.settle(client, name)
        record['full_cell_s'] = time.monotonic() - (seed_record or record)['timing']['arrival_monotonic']
        record['full_cell_scope'] = 'through observed idle after primary+probe; includes declared Chat seed/tokenization when present'
        client.out.save(name + '-validated.json', record)
        rows.append(record)
    return rows



def distributions(rows):
    groups = {}
    for row in rows:
        key = (row['region'], row['temperature'], row['state'], row['input_kind'])
        groups.setdefault(key, []).append(row)
    result = []
    for key, values in groups.items():
        measures = {}
        for name, getter in (('HTTP_complete_wall_s', lambda r: r['timing']['full_wall_s']),
                             ('full_cell_s', lambda r: r['full_cell_s']),
                             ('next_probe_TTFT_s', lambda r: r['next_probe']['timing']['first_visible_s']),
                             ('client_gap_s', lambda r: r['HTTP_done_to_probe_arrival_s'])):
            samples = [getter(row) for row in values]
            finite = [value for value in samples if value is not None]
            measures[name] = {'raw': samples, 'observed': len(finite), 'missing': len(samples) - len(finite), 'minimum': min(finite) if finite else None,
                'median': statistics.median(finite) if finite else None, 'maximum': max(finite) if finite else None}
        result.append({'region': key[0], 'temperature': key[1], 'state': key[2], 'input_kind': key[3],
            'same_input_all_repeats': len({row['request_sha256'] for row in values}) == 1,
            'output_ids_equal': len({row['output_ids_sha256'] for row in values}) == 1,
            'measures': measures, 'scope': 'descriptive only; non-significance is not equivalence'})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--inputs-sha256', required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--arm', required=True, choices=('baseline', 'candidate'))
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    spec, regions = load_inputs(args.inputs, args.inputs_sha256)
    out = core.Evidence(args.out)
    result, primary, errors = {'schema': 'tensorfold.gemma-regional-timing-result-v1', 'passed': False,
        'input_sha256': args.inputs_sha256, 'historical_runner_sha256': HISTORICAL_RUNNER_SHA,
        'arm': args.arm, 'endpoint': args.base_url, 'cells': [],
        'tool_sha256': hashlib.sha256(core.read_bound(Path(__file__), core.CAP)).hexdigest(),
        'traversals': 2, 'primary_cells': 36, 'additional_chat_seed_requests': 12, 'one_token_probes': 36,
        'comparison': 'B->A2 same-input cells only; A1 unchangedcore evidence carried externally',
        'P3_acceptance': False, 'scope': 'raw per-region observations; no thresholds, equivalence or promotion'}, None, [None]
    try:
        run(core.Client(args.base_url, out, args.arm), spec, regions, rows=result['cells'])
        result['distributions'] = distributions(result['cells'])
        result['passed'] = True
    except BaseException as error:
        primary = error
    try:
        out.save('result.json', result)
    except BaseException as error:
        errors[0] = error
    core.finish_errors(primary, errors)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

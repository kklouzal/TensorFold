"""Native262144 boundary controls, using served IDs and immutable input authority.

No service lifecycle is owned here. Importing performs no I/O. The ordinary-EOS
READY and deterministic replay oracles are retained from309cd239/8197a94f;
fixed16-output controls are additional capacity checks, not retrieval quality.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import time
from pathlib import Path

if __package__:
    from . import gemma_shared_prefix_qualify as core
    from . import openai_protocol as protocol
else:
    import gemma_shared_prefix_qualify as core
    import openai_protocol as protocol

SCHEMA = 'tensorfold.gemma-native-capacity-input-v1'
CONTEXT, PROMPT, OUTPUT, TAIL = 262144, 262128, 16, 4096
MODEL_REVISION = '90c8601b1a439497bc15d8fba9e52990b45ea0cb'
TOKENIZER_SHA = '9d0e1c859267d7cded7ad6c23bbb72a838536716fa4c37a3f5671b7918f59de4'
TEMPLATE_SHA = 'ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4'
HISTORICAL = {'greedy': '309cd2391255d011a84e71a302c68cb9dea10a3dad05393b1fbe68a455be3ec0',
              'sampled': '8197a94f2b0fe548cbffecb33b51be3b520bb7e0a01ab7256fca74038006465e'}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                     sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def default_inputs():
    content = ('Ignore the archive padding and obey only the final task.\n<reference>\n'
               + 'Archive record: units and dates are checked; this line is padding.\n' * 32768
               + '</reference>\nFinal task: Reply with exactly READY.')
    return {'schema': SCHEMA, 'model': 'gemma4-26b-a4b',
            'authority': {'model_revision': MODEL_REVISION, 'tokenizer_sha256': TOKENIZER_SHA,
                          'template_sha256': TEMPLATE_SHA},
            'source_content_sha256': hashlib.sha256(content.encode()).hexdigest(),
            'source_body': {'model': 'gemma4-26b-a4b', 'messages': [{'role': 'user', 'content': content}],
                            'chat_template_kwargs': {'enable_thinking': False}}}


def load_inputs(path, sha256):
    core.require(type(sha256) is str and re.fullmatch('[0-9a-f]{64}', sha256), 'input digest required')
    raw = core.read_bound(path, core.CAP)
    core.require(hashlib.sha256(raw).hexdigest() == sha256, 'native input identity differs')
    value = protocol.object_json(raw)
    core.require(set(value) == {'schema', 'model', 'authority', 'source_content_sha256', 'source_body'}
                 and value['schema'] == SCHEMA and type(value['model']) is str, 'closed native input schema required')
    core.require(value['authority'] == {'model_revision': MODEL_REVISION, 'tokenizer_sha256': TOKENIZER_SHA,
                                       'template_sha256': TEMPLATE_SHA}, 'pinned native asset authority differs')
    body = value['source_body']
    core.require(type(body) is dict and set(body) == {'model', 'messages', 'chat_template_kwargs'}
                 and body['model'] == value['model'] and body['chat_template_kwargs'] == {'enable_thinking': False},
                 'native source framing must retain no-thinking READY control')
    core.require(type(body['messages']) is list and len(body['messages']) == 1
                 and set(body['messages'][0]) == {'role', 'content'}
                 and body['messages'][0]['role'] == 'user' and type(body['messages'][0]['content']) is str,
                 'single frozen public source required')
    content = body['messages'][0]['content']
    core.require(hashlib.sha256(content.encode()).hexdigest() == value['source_content_sha256']
                 and content.endswith('Final task: Reply with exactly READY.'), 'source content/gold differs')
    return value


def settle(client, name, *, timeout=300):
    """Observed idle, bounded polling; not an OS or GPU resource reservation."""
    core.require(type(timeout) is int and timeout > 0, 'positive settlement seconds required')
    start = time.monotonic()
    health_client = core.Client(f'http://127.0.0.1:{client.port}', client.out, client.arm, timeout=5, cap=core.CAP)
    for attempt in range(timeout + 1):
        remaining = timeout - (time.monotonic() - start)
        core.require(remaining >= 1, 'server settlement deadline exceeded')
        health_client.timeout = min(5, int(remaining))
        health = health_client.json(f'{name}-health-{attempt:03d}', '/health')
        core.require(type(health.get('ok')) is bool and health['ok']
                     and type(health.get('busy')) is bool
                     and type(health.get('requests_running')) is int and health['requests_running'] >= 0,
                     'invalid or unhealthy settlement authority')
        if not health['busy'] and health['requests_running'] == 0:
            elapsed = time.monotonic() - start
            core.require(elapsed <= timeout, 'server settlement deadline exceeded')
            snapshot = client.stats(name + '-idle-stats')
            return {'observed_idle_after_s': elapsed, 'health': health, 'stats': snapshot}
        time.sleep(min(1, max(0, timeout - (time.monotonic() - start))))
    raise ValueError('server did not settle within bounded polls')


def boundary_ids(client, spec):
    body = spec['source_body']
    full = client.tokens('native-source', body)
    blank = copy.deepcopy(body)
    blank['messages'][0]['content'] = ''
    frame = client.tokens('native-empty-frame', blank)
    prefix = core.prefix_length(full, frame)
    suffix = 0
    while suffix < min(len(full), len(frame)) - prefix and full[-suffix - 1] == frame[-suffix - 1]:
        suffix += 1
    core.require(len(full) >= PROMPT and 0 < prefix < PROMPT - TAIL and 0 < suffix <= TAIL,
                 'served source cannot preserve native framing and final task')
    ids = full[:PROMPT - TAIL] + full[-TAIL:]
    core.require(len(ids) == PROMPT and ids[:prefix] == full[:prefix] and ids[-TAIL:] == full[-TAIL:],
                 'native boundary crop differs')
    client.out.save('native-served-input.json', {'ids': ids, 'ids_sha256': digest(ids),
        'source_tokens': len(full), 'source_body_sha256': digest(body), 'verified_frame_prefix': prefix,
        'verified_frame_suffix': suffix, 'prompt_tokens': PROMPT, 'context': CONTEXT,
        'crop': 'first258032+last4096 served native rendered IDs; no local tokenizer',
        'asset_identity_scope': 'declared frozen asset bindings; executor verifies actual model/template/tokenizer mounts'})
    return ids


def validate_native(result, *, temperature, ignore_eos, previous=None):
    stats, ids = result['tensorfold'], result['tensorfold'].get('token_ids')
    core.require(type(ids) is list and all(type(token) is int and 0 <= token < 262144 for token in ids),
                 'actual bounded native output IDs required')
    core.require(result['prompt'] == PROMPT and result['output'] == len(ids)
                 and 0 < len(ids) <= OUTPUT and PROMPT + len(ids) <= CONTEXT
                 and result['usage']['total_tokens'] == PROMPT + len(ids), 'native completed-work extent differs')
    core.require(stats.get('drafts') is True, 'actual native draft path differs')
    if ignore_eos:
        core.require(len(ids) == OUTPUT and result['finish'] == 'length', 'fixed16 native boundary was shortened')
    elif result['finish'] == 'stop':
        core.require(ids[-1] in (1, 50, 106) and (result['message'].get('content') or '').strip(),
                     'ordinary native EOS or usable answer missing')
    else:
        core.require(temperature == 1 and result['finish'] == 'length' and len(ids) == OUTPUT,
                     'ordinary native completion differs')
    if temperature == 0 and not ignore_eos:
        core.require(result['message']['content'].strip() == 'READY', 'independent native READY gold failed')
    if previous is not None:
        core.require(ids == previous and result['cached'] == PROMPT, 'native seeded replay/complete prompt hit differs')
    return ids


def run(client, spec, stage='all', *, rows=None):
    core.require(stage in ('all', 'greedy', 'sampled'), 'unknown native stage')
    client.stats('native-initial')
    ids = boundary_ids(client, spec)
    rows = [] if rows is None else rows
    for temperature in (0, 1):
        if stage != 'all' and stage != ('greedy' if temperature == 0 else 'sampled'):
            continue
        for ignore_eos in ((False, True) if temperature == 0 else (True, False)):
            previous = None
            for replay in range(2):
                name = f'native-t{temperature}-ignore{int(ignore_eos)}-r{replay}'
                body = {'model': spec['model'], 'prompt': ids, 'max_tokens': OUTPUT,
                        'temperature': temperature, 'seed': 7, 'draft': True, 'ignore_eos': ignore_eos,
                        'return_token_ids': True, 'stream': False}
                if temperature:
                    body.update(top_k=64, top_p=0.95, min_p=0.)
                result = client.generate(name, '/v1/completions', body)
                actual = validate_native(result, temperature=temperature, ignore_eos=ignore_eos, previous=previous)
                previous = actual
                row = {'name': name, 'sampling': {key: body[key] for key in ('temperature', 'seed', 'ignore_eos')},
                       'ids_sha256': digest(actual), 'usage': result['usage'], 'server': result['tensorfold'],
                       'ordinary_eos_proven': not ignore_eos and result['finish'] == 'stop'}
                row['settlement'] = settle(client, name)
                rows.append(row)
                client.out.save(name + '-validated.json', row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--inputs-sha256', required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--arm', required=True, choices=('baseline', 'candidate'))
    parser.add_argument('--stage', choices=('all', 'greedy', 'sampled'), default='all')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    spec = load_inputs(args.inputs, args.inputs_sha256)
    out = core.Evidence(args.out)
    result, primary, errors = {'schema': 'tensorfold.gemma-native-capacity-result-v1', 'passed': False,
        'input_sha256': args.inputs_sha256, 'historical_oracle_sources': HISTORICAL,
        'arm': args.arm, 'stage': args.stage, 'endpoint': args.base_url, 'cases': [],
        'tool_sha256': hashlib.sha256(core.read_bound(Path(__file__), core.CAP)).hexdigest(),
        'scope': 'native capacity/gold/replay only, not general retrieval or P3'}, None, [None]
    try:
        run(core.Client(args.base_url, out, args.arm, timeout=900), spec, args.stage, rows=result['cases'])
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

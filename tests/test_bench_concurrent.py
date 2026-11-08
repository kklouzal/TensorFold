"""HTTP/SSE and exactness evidence boundaries; no server or GPU is needed."""
import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def bench():
    path = Path(__file__).resolve().parents[1] / "tools/bench_concurrent.py"
    spec = importlib.util.spec_from_file_location("bench_concurrent_boundary", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


@pytest.mark.parametrize("left,right,want", [
    ({}, {}, None), ({"token_sha": None}, {"token_sha": None}, None),
    ({"token_sha": ""}, {"token_sha": ""}, None),
    ({"token_sha": 123}, {"token_sha": 123}, None),
    ({"token_sha": "abc"}, {"token_sha": "abc"}, True),
    ({"token_sha": "abc"}, {"token_sha": "def"}, False),
    ({"token_sha": "abc", "error": "failed"}, {"token_sha": "abc"}, None),
])
def test_missing_or_failed_hash_evidence_never_proves_equality(bench, left, right, want):
    assert bench.compare_hash(left, right) is want


class Response(io.BytesIO):
    status = 200

    def getheader(self, name, default=None):
        return "text/event-stream"


def sse(*events, done=True):
    result = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    return result + (b"data: [DONE]\n\n" if done else b"")


@pytest.mark.parametrize("payload,reason", [
    (sse({"error": {"message": "GPU rejected invalid state"}}), "SSE error"),
    (sse({"error": None}), "SSE error"),
    (sse(["invalid event"]), "invalid object"),
    (sse({"usage": {"completion_tokens": 4}}, done=False), "before [DONE]"),
    (sse({"usage": {"completion_tokens": False}}), "bounded positive"),
    (sse({"usage": {"completion_tokens": 0}}), "bounded positive"),
    (sse({"usage": {"completion_tokens": 17}}), "bounded positive"),
    (sse({}), "bounded positive"),
    (b"data: " + b"x" * 65536 + b"\n", "bounded size"),
], ids=["engine-error", "null-error", "non-object-event", "missing-done", "boolean-token-count",
        "zero-token-count", "excess-token-count", "missing-token-count", "oversized-event"])
def test_sse_error_incomplete_or_invalid_usage_is_a_failed_stream(bench, monkeypatch, payload, reason):
    monkeypatch.setattr(bench.urllib.request, "urlopen", lambda *a, **kw: Response(payload))
    row = bench.stream("http://unused", "fixture", bench.PROMPTS[0], 16, 0, 123)
    assert row["tokens"] == 0 and reason in row["error"]
    assert row["token_sha"] is None and row["decode_tps"] is None


def test_valid_sse_keeps_optional_missing_hash_explicit(bench, monkeypatch):
    payload = sse(*({"choices": [{"text": "x"}]} for _ in range(4)),
                  {"usage": {"completion_tokens": 4}})
    monkeypatch.setattr(bench.urllib.request, "urlopen", lambda *a, **kw: Response(payload))
    row = bench.stream("http://unused", "fixture", bench.PROMPTS[0], 16, 0, 123)
    assert "error" not in row and row["tokens"] == 4 and len(row["pieces"]) == 4
    assert row["token_sha"] is None


def test_usage_and_hash_without_text_pieces_cannot_claim_measured_rates(bench, monkeypatch):
    payload = sse({"usage": {"completion_tokens": 4}, "tensorfold": {"token_sha": "abc"}})
    monkeypatch.setattr(bench.urllib.request, "urlopen", lambda *a, **kw: Response(payload))
    row = bench.stream("http://unused", "fixture", bench.PROMPTS[0], 16, 0, 123)
    assert "error" not in row and row["tokens"] == 4 and row["token_sha"] == "abc"
    assert row["unmeasured"] is True and row["decode_tps"] is None


@pytest.mark.parametrize("kind,strict,want", [
    ("valid", True, 0), ("missing", False, 0), ("missing", True, 1),
    ("unequal", False, 1), ("unequal", True, 1),
    ("error", False, 1), ("unmeasured", False, 0), ("unmeasured", True, 1),
])
def test_cli_retains_result_and_exits_by_verified_cells(bench, monkeypatch, tmp_path, kind, strict, want):
    def row(item, seed, tokens, draft=True):
        digest = None if kind == "missing" else f"{item['name']}-{seed}-{draft if kind == 'unequal' else True}"
        result = {"prompt": item["name"], "seed": seed, "tokens": tokens,
                  "first": 1.0, "last": 2.0, "sent": 0.0, "pieces": [(1.0, 1), (2.0, 1)],
                  "ttft_s": 1.0, "decode_tps": tokens - 1, "token_sha": digest}
        if kind == "error":
            result.update(error="SSE error: GPU failed", decode_tps=None)
        elif kind == "unmeasured":
            result.update(unmeasured=True, decode_tps=None)
        return result

    def together(base, model, specs, tokens, temp, *rest, **kwargs):
        assert kwargs['limits'].response_bytes == 128 << 20
        assert kwargs['limits'].line_bytes == 2048 << 10
        return [row(item, seed, tokens) for item, seed in specs]
    def stream(base, model, item, tokens, temp, seed, draft=True, **kwargs):
        assert kwargs['limits'].response_bytes == 128 << 20
        assert kwargs['limits'].line_bytes == 2048 << 10
        return row(item, seed, tokens, draft)
    monkeypatch.setattr(bench, 'together', together)
    monkeypatch.setattr(bench, 'stream', stream)
    output = tmp_path / "retained.json"
    argv = ["bench_concurrent.py", "http://unused", "fixture", "--levels", "1,4", "--reps", "1",
            "--temperatures", "0", "--alone", "--serial", "--output", str(output),
            "--response-mib", "128", "--sse-line-kib", "2048"]
    if strict:
        argv.append("--strict")
    monkeypatch.setattr(sys, "argv", argv)
    assert bench.main() == want
    report = json.loads(output.read_text())
    assert report["passed"] is (want == 0) and report["strict"] is strict
    assert len(report["cells"]) == 4
    if kind == "missing":
        assert all(cell["alone"]["equal"] == 0 and cell["alone"]["unverified"] > 0
                   and cell["serial"]["equal"] == 0 and cell["serial"]["unverified"] > 0
                   for cell in report["cells"])
    if kind == "unequal":
        assert all(cell["serial"]["unequal"] > 0 for cell in report["cells"])


def test_strict_no_seed_is_refused_before_requests(bench, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench", "http://unused", "fixture", "--strict", "--no-seed"])
    monkeypatch.setattr(bench, "together", lambda *a, **kw: pytest.fail("invalid gate made a request"))
    with pytest.raises(SystemExit) as caught:
        bench.main()
    assert caught.value.code == 2


def matrix_fixture():
    import hashlib
    plan = {'schema': 'tensorfold.phase-matrix-plan-v1', 'run_identity': 'a' * 32, 'model': 'fixture',
            'profile': 'q8', 'admin_cap': 24, 'levels': [1, 2, 32], 'context': 262144, 'tokens': 3,
            'reps': 1, 'inputs': [6], 'tokenizer_sha256': 'b' * 64, 'filler_ids': [7],
            'cohorts': [], 'excluded_levels': [32]}
    for cid, level in enumerate((1, 2)):
        cohort = {'id': cid, 'level': level, 'input_tokens': 6, 'tokens': 3, 'repetition': 0,
                  'warmup': cid == 0, 'requests': []}
        for index in range(level):
            ids = [10 + cid * 2 + index] + [7] * 5
            cohort['requests'].append({'prefix': ids[:1], 'input_tokens': 6, 'index': index,
                'prompt_sha256': hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()})
        plan['cohorts'].append(cohort)
    return plan


def test_matrix_defaults_include_two_and_explicit_profile_caps(bench):
    assert bench.STANDARD_LEVELS == (1, 2, 4, 8, 16, 24, 32)
    assert bench.PROFILE_CAPS == {'q8': 24, 'q4': 32}


def test_matrix_data_roundtrip_rejects_changed_ids_and_boolean_counts(bench, tmp_path):
    plan = matrix_fixture()
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    assert bench.load_matrix(path) == plan
    plan['cohorts'][0]['requests'][0]['prefix'] = [False]
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        bench.load_matrix(path)
    plan = matrix_fixture()
    plan['cohorts'][0]['tokens'] = True
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        bench.load_matrix(path)


def test_natural_matrix_preparation_exact_lengths_unique_requests_and_admin_exclusion(bench, tmp_path):
    from test_segment_encoder import bpe
    tokenizer = bpe()
    tokenizer.save(str(tmp_path / 'tokenizer.json'))
    plan = bench.prepare_matrix(tmp_path, 'fixture', 'q8', list(bench.STANDARD_LEVELS), [256], 4, 1)
    assert plan['excluded_levels'] == [32]
    assert [c['level'] for c in plan['cohorts']] == [1, 1, 2, 4, 8, 16, 24]
    hashes = set()
    for cohort in plan['cohorts']:
        for request in cohort['requests']:
            ids = bench.matrix_ids(plan, request)
            assert len(ids) == 256 and 'Run ' in tokenizer.decode(ids[:len(request['prefix'])])
            assert request['prompt_sha256'] not in hashes
            hashes.add(request['prompt_sha256'])


@pytest.mark.parametrize('fault', [None, 'overlap', 'zero-time', 'restart', 'cached', 'stale-observer', 'hash', 'observer-instance', 'materialization'])
def test_matrix_runner_keeps_one_service_denominators_raw_errors_and_actual_overlap(bench, monkeypatch, tmp_path, fault):
    from types import SimpleNamespace as NS
    from tools import v100_rotorquant_live as live
    monkeypatch.setitem(sys.modules, 'v100_rotorquant_live', live)
    from tools import v100_adaptive_qualification as qualification
    monkeypatch.setitem(sys.modules, 'v100_adaptive_qualification', qualification)
    plan = matrix_fixture()
    calls, stats_calls = [], [0]
    class Client:
        def __init__(self, *args, **kwargs):
            pass
        def call(self, label, body=None, **kwargs):
            calls.append((label, body, kwargs))
            if body is None and kwargs.get('path') == '/stats':
                stats_calls[0] += 1
                return {'server': {'instance_id': 'other' if fault == 'restart' and stats_calls[0] > 1 else 'one'},
                        'object': 'tensorfold.stats',
                        'model': {'id': 'fixture', 'kv_dtype': 'int8', 'context_length': 262144},
                        'capacity_plan': {'session_capacity': {'resident_capacity': 24, 'configured_ceiling': 24,
                            'context_tokens': 262144, 'full_context_commitments': 24},
                            'kv_full_windows': {'materialized': fault != 'materialization', 'reserved': 24, 'stream_slots': 24,
                                'ranks': [{'rank': rank, 'materialized': True, 'startup_capacities': [262149] * 24,
                                           'allocated_cache_bytes': 100, 'torch_allocated_bytes': 200, 'torch_reserved_bytes': 300}
                                          for rank in range(8)]}},
                        'load': {'requests_in_flight': 0, 'requests_running': 0, 'requests_waiting': 0}}
            if body is None:
                return {'execution': {'work_observations': {'benchmark': {
                    'run_identity': 'stale' if fault == 'stale-observer' else plan['run_identity'], 'server_instance_id': 'other' if fault == 'observer-instance' else 'one',
                    'cohorts': {str(c['id']): {'events_ready': True, 'completed': c['level'],
                        'active_owner_highwater': 1 if fault == 'overlap' else c['level'],
                        'prompt_hashes': sorted(r['prompt_sha256'] for r in c['requests']),
                        'prefill_ms': 0 if fault == 'zero-time' else 20., 'decode_ms': 40.}
                        for c in plan['cohorts']}}}}}
            kwargs['observer']({}, ['natural text'])
            return {'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'natural text'}, 'finish_reason': 'length'}],
                    'usage': {'prompt_tokens': 6, 'completion_tokens': 3, 'total_tokens': 9,
                              'prompt_tokens_details': {'cached_tokens': 0}},
                    'tensorfold': {'token_ids': [1, 2, 3],
                                   'token_sha': 'wrong' if fault == 'hash' else __import__('hashlib').sha256(b'1,2,3').hexdigest()[:12], 'cached': 1 if fault == 'cached' else 0,
                                   'prefill_s': .2, 'decode_s': .4}}
    monkeypatch.setattr(live, 'Client', Client)
    args = NS(base='http://unused', model='fixture', output=tmp_path / 'report.json', timeout=.2,
              response_mib=16, sse_line_kib=64)
    if fault in ('restart', 'stale-observer', 'observer-instance', 'materialization'):
        with pytest.raises(ValueError):
            bench.run_matrix(args, plan)
        report = json.loads(args.output.read_text())
        assert report['error']['type'] == 'ValueError' and report['passed'] is False
        assert 'unrun_cohorts' in report
        return
    result = bench.run_matrix(args, plan)
    assert result == (0 if fault is None else 2)
    report = json.loads(args.output.read_text())
    assert report['excluded_by_admin_cap'] == [32]
    assert sum(body is not None for label, body, kw in calls) == (1 if fault in ('cached', 'hash') else 3)
    assert all(body['ignore_eos'] and body['return_token_ids'] for label, body, kw in calls if body is not None)
    if fault is None:
        assert report['cohorts'][1]['rates']['prefill_envelope_tps'] == 600
        assert report['cohorts'][1]['rates']['decode_envelope_tps'] == 100
        assert report['cohorts'][1]['requests'][0]['prefill_reported_host_tps'] == 30
        assert report['cohorts'][1]['requests'][0]['decode_reported_host_tps'] == 5


def test_matrix_geometry_is_rejected_before_any_token_list_allocation(bench, monkeypatch, tmp_path):
    plan = matrix_fixture()
    plan['cohorts'][0]['input_tokens'] = 1 << 30
    plan['cohorts'][0]['requests'][0]['input_tokens'] = 1 << 30
    path = tmp_path / 'malformed.json'
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(bench, 'matrix_ids', lambda *a: pytest.fail('allocated before context validation'))
    with pytest.raises(ValueError):
        bench.load_matrix(path)


def repro_fixture():
    import hashlib
    configuration = {'id': 'fixture', 'kv_dtype': 'rotorquant-iso4-sm70-v1',
                     'context_length': 262144, 'mtp_drafts': 4}
    cases = []
    for index in range(8):
        body = {'model': 'fixture', 'prompt': [index] * 64, 'max_tokens': 128, 'temperature': 0,
                'draft': True, 'ignore_eos': True, 'stream': True,
                'stream_options': {'include_usage': True}, 'return_token_ids': True}
        ids = [index + 1] * 128
        reply = {'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'text'}, 'finish_reason': 'length'}],
                 'usage': {'prompt_tokens': 64, 'completion_tokens': 128, 'total_tokens': 192,
                           'prompt_tokens_details': {'cached_tokens': 0}},
                 'tensorfold': {'token_ids': ids, 'token_sha': hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()[:12],
                                'cached': 0, 'drafts': True, 'drafted': 100, 'accepted': 80}}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        cases.append({'index': index, 'body': body, 'body_sha256': digest, 'reference': reply})
    return {'schema': 'tensorfold.raw-completion-repro-input-v1', 'model_configuration': configuration, 'cases': cases}


@pytest.mark.parametrize('fault', [None, 'hash', 'geometry', 'mode', 'configuration'])
def test_repro_data_validation_preserves_original_bodies(bench, tmp_path, fault):
    data = repro_fixture()
    if fault == 'hash':
        data['cases'][0]['body']['prompt'][0] = 999
    elif fault == 'geometry':
        data['cases'][0]['body']['max_tokens'] = 127
    elif fault == 'mode':
        data['cases'][0]['body']['stream'] = False
    elif fault == 'configuration':
        data['model_configuration']['mtp_drafts'] = 6
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(data))
    if fault:
        with pytest.raises(ValueError):
            bench.load_repro(path, 'fixture')
    else:
        assert bench.load_repro(path, 'fixture') == {**data, '_source_sha256': __import__('hashlib').sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize('fault', [None, 'different', 'cached', 'bad-sha', 'wrong-dispatch', 'serial-proposals', 'transport', 'restart', 'final-stats', 'repeat-idle'])
def test_repro_runs_twenty_or_conditional_twenty_eight_and_retains_failures(bench, monkeypatch, tmp_path, fault):
    import copy
    import hashlib
    from types import SimpleNamespace as NS
    from tools import v100_rotorquant_live as live
    from tools import v100_adaptive_qualification as qualification
    monkeypatch.setitem(sys.modules, 'v100_rotorquant_live', live)
    monkeypatch.setitem(sys.modules, 'v100_adaptive_qualification', qualification)
    data, calls = repro_fixture(), []
    final_reads = [0]
    transport = OSError('retained original transport failure')
    transport.add_note('authentic primary note')
    class Client:
        def __init__(self, *args, **kwargs):
            pass
        def call(self, label, body=None, **kwargs):
            calls.append((label, copy.deepcopy(body), kwargs.get('path')))
            if body is None:
                if fault == 'repeat-idle' and label == 'control-3':
                    raise OSError('conditional repeat idle failed')
                if sum(body is not None for _, body, _ in calls) == 20:
                    final_reads[0] += 1
                    if fault == 'final-stats' and final_reads[0] == 2:
                        raise OSError('final stats failed')
                return {'server': {'instance_id': 'changed' if fault == 'restart' and len(calls) > 2 else 'one'},
                        'object': 'tensorfold.stats', 'model': data['model_configuration'],
                        'capacity_plan': {'expert_parallel': {'tp': 8, 'pp': 1, 'etp': 4, 'ep': 2,
                            'implementation': 'owned-etp4-native-slots-v1'}, 'session_capacity': {'resident_capacity': 32, 'configured_ceiling': 32,
                            'context_tokens': 262144, 'full_context_commitments': 32},
                            'kv_full_windows': {'materialized': True, 'reserved': 32, 'stream_slots': 32,
                                'ranks': [{'rank': rank, 'materialized': True, 'startup_capacities': [262149] * 32,
                                           'allocated_cache_bytes': 100, 'torch_allocated_bytes': 200, 'torch_reserved_bytes': 300}
                                          for rank in range(8)]}},
                        'load': {'requests_in_flight': 0, 'requests_running': 0, 'requests_waiting': 0}}
            index = body['prompt'][0]
            if fault == 'transport' and index == 2:
                raise transport
            reply = copy.deepcopy(data['cases'][index]['reference'])
            runtime = reply['tensorfold']
            runtime.update(drafts=body['draft'], drafted=100 if body['draft'] else 0, accepted=80 if body['draft'] else 0)
            if fault in ('different', 'repeat-idle') and index == 2 and body['draft']:
                runtime['token_ids'][6] = 555
                runtime['token_sha'] = hashlib.sha256(','.join(map(str, runtime['token_ids'])).encode()).hexdigest()[:12]
            elif fault == 'cached':
                runtime['cached'] = 1
            elif fault == 'bad-sha':
                runtime['token_sha'] = 'bad'
            elif fault == 'wrong-dispatch':
                runtime['drafts'] = not body['draft']
            elif fault == 'serial-proposals' and not body['draft']:
                runtime['drafted'] = 1
            kwargs['observer']({}, ['text'])
            return reply
    monkeypatch.setattr(live, 'Client', Client)
    source = tmp_path / 'input.json'
    source.write_text(json.dumps(data))
    data = bench.load_repro(source, 'fixture')
    source.unlink()  # Run must use exactly the already loaded bytes/hash, not reread DATA.
    args = NS(base='http://unused', model='fixture', output=tmp_path / 'out.json', repro_input=source,
              timeout=1, response_mib=16, sse_line_kib=64)
    if fault not in (None, 'different'):
        with pytest.raises((ValueError, OSError)) as caught:
            bench.run_repro(args, data)
        if fault == 'transport':
            assert caught.value is transport
            assert 'authentic primary note' in caught.value.__notes__
    else:
        assert bench.run_repro(args, data) == 0
    report = json.loads(args.output.read_text())
    assert 'final_stats' in report or 'final_stats_error' in report
    requests = [(label, body, path) for label, body, path in calls if body is not None]
    for label, body, path in requests:
        original = data['cases'][body['prompt'][0]]['body']
        assert body == (original if body['draft'] else {**original, 'draft': False})
        assert path == '/v1/completions'
    if fault in (None, 'different'):
        assert report['completed'] and not report['unrun_stages']
        assert len(requests) == (28 if fault == 'different' else 20)
        assert report['all_exact'] is (fault is None)
        assert len([row for row in report['stages'] if row['name'] == 'baseline-c8-mtp-repeat']) == (fault == 'different')
        assert any(not item['passed'] for item in report['differences']) is (fault == 'different')
    else:
        assert not report['completed'] and report['error']
        assert bool(report['unrun_stages']) is (fault != 'final-stats')
        if fault == 'repeat-idle':
            assert report['conditional_repeat_required'] is True
            assert 'baseline-c8-mtp-repeat' in report['unrun_stages']
        if fault not in ('restart', 'final-stats', 'repeat-idle'):
            assert any(error for row in report['stages'] for error in row['errors'])


def test_repro_loader_reads_once_bounded_and_rejects_oversize_before_parse(bench, monkeypatch):
    reads = []
    class Source:
        def open(self, mode):
            assert mode == 'rb'
            class Stream(io.BytesIO):
                def read(self, size):
                    reads.append(size)
                    return super().read(size)
            return Stream(b' ' * ((16 << 20) + 2))
    monkeypatch.setattr(bench.protocol, 'object_json', lambda raw: pytest.fail('oversize reached JSON parser'))
    with pytest.raises(ValueError, match='exceeds16MiB'):
        bench.load_repro(Source(), 'fixture')
    assert reads == [(16 << 20) + 1]


@pytest.mark.parametrize('raw', [b'{"schema":1,"schema":2}', b'{"schema":NaN}', b'{"schema":1e999}'])
def test_repro_loader_reuses_strict_object_json(bench, tmp_path, raw):
    path = tmp_path / 'invalid.json'
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        bench.load_repro(path, 'fixture')

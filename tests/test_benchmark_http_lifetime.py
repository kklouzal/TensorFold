"""Bounded real loopback HTTP exercises benchmark request owners; no model/SDK."""
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import threading
import sys
from types import SimpleNamespace
import unittest
import urllib.error

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location('loopback_owned_' + name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


benchmark = load('bench_concurrent')
calibration = load('fit_draft_calibration')
single = load('bench_openai')
prefill = load('prefill_cold')


@contextlib.contextmanager
def server(fail_seed=None, mode=None, simultaneous=0):
    lock, records, finished = threading.Lock(), [], []
    rendezvous = threading.Barrier(simultaneous) if simultaneous else None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            size = int(self.headers['Content-Length'])
            if not 0 < size <= 65536:
                raise ValueError('fixture request bound')
            body = json.loads(self.rfile.read(size))
            with lock:
                records.append((self.path, body))
            try:
                if rendezvous:
                    rendezvous.wait(timeout=3)
                if body.get('seed', 20) == fail_seed:
                    self.send_error(500, 'fixture failure')
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream' if body.get('stream') else 'application/json')
                self.end_headers()
                if not body.get('stream'):
                    if mode == 'json_error':
                        self.wfile.write(b'{"error": {"message": "fixture HTTP200 error"}}')
                        return
                    self.wfile.write(b'{"choices": [{"text": "fixture"}], "usage": {"completion_tokens": 4}}')
                    return
                for i in range(4):
                    choice = {'text': 'word' + str(i)} if self.path == '/v1/completions' else \
                        {'delta': {'content': 'word' + str(i)}}
                    self.wfile.write(('data: ' + json.dumps({'choices': [choice]}) + '\n\n').encode())
                    self.wfile.flush()
                usage = {'usage': {'completion_tokens': body['max_tokens'], 'prompt_tokens': 17}, 'tensorfold': {'token_sha': 'seed-' + str(body.get('seed', 20))}}
                self.wfile.write(('data: ' + json.dumps(usage) + '\n\n').encode())
                if mode != 'missing_done':
                    self.wfile.write(b'data: [DONE]\n\n')
            finally:
                with lock:
                    finished.append(body.get('seed', 20))

    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    owner = threading.Thread(target=httpd.serve_forever, kwargs={'poll_interval': .01}, name='benchmark-loopback-owner')
    owner.start()
    try:
        yield 'http://127.0.0.1:' + str(httpd.server_port), records, finished
    finally:
        httpd.shutdown()
        httpd.server_close()
        owner.join(timeout=3)
        if owner.is_alive():
            raise AssertionError('fixture server owner did not join')


class HttpLifetime(unittest.TestCase):
    def specs(self):
        return [(benchmark.PROMPTS[i % 2], 20 + i) for i in range(4)]

    def test_real_four_streams_have_complete_usage_hashes_seed_order_and_bodies(self):
        with server(simultaneous=4) as (base, records, finished):
            results = benchmark.together(base, 'fixture', self.specs(), 4, 1.)
            self.assertEqual([r['seed'] for r in results], [20, 21, 22, 23])
            self.assertEqual([r['token_sha'] for r in results], ['seed-' + str(i) for i in range(20, 24)])
            self.assertTrue(all(r['tokens'] == 4 and len(r['pieces']) == 4 and not r.get('error') for r in results))
            self.assertEqual(benchmark.aggregates(results)['failed'], 0)
        self.assertCountEqual(finished, [20, 21, 22, 23])
        for path, body in records:
            self.assertEqual(body['model'], 'fixture')
            self.assertEqual(body['stream_options'], {'include_usage': True})
            self.assertEqual((body['max_tokens'], body['temperature'], body['top_k'], body['top_p']), (4, 1., 20, .95))
            item = benchmark.PROMPTS[(body['seed'] - 20) % 2]
            self.assertEqual(path, '/v1/chat/completions' if item['kind'] == 'chat' else '/v1/completions')
            self.assertEqual(body.get('prompt', body.get('messages', [{}])[0].get('content')), item['prompt'])

    def test_http_failure_or_unterminated_sse_is_observed_without_losing_other_streams(self):
        for options in ({'fail_seed': 21}, {'mode': 'missing_done'}):
            with self.subTest(options=options), server(simultaneous=4, **options) as (base, records, finished):
                results = benchmark.together(base, 'fixture', self.specs(), 4, 0.)
                self.assertEqual([r['seed'] for r in results], [20, 21, 22, 23])
                self.assertEqual(len(records), 4)
                self.assertEqual(benchmark.aggregates(results)['failed'], 1 if 'fail_seed' in options else 4)
            self.assertCountEqual(finished, [20, 21, 22, 23])

    def test_real_single_stream_and_prefill_require_done_and_typed_usage(self):
        for method in ('single', 'prefill'):
            for mode in (None, 'missing_done'):
                with self.subTest(method=method, mode=mode), server(mode=mode) as (base, records, finished):
                    def invoke():
                        if method == 'single':
                            return single.stream(base, 'fixture', single.PROMPTS[0], 4, 0., 20)
                        return prefill.one(base, 'fixture', [{'role': 'user', 'content': 'fixture'}])
                    if mode:
                        with self.assertRaisesRegex(ValueError, r'before \[DONE\]'):
                            invoke()
                    else:
                        result = invoke()
                        self.assertGreaterEqual(result['ttft_s'], 0)
                        self.assertEqual(result['tokens'] if method == 'single' else result['prompt_tokens'],
                                         4 if method == 'single' else 17)
                self.assertEqual(len(records), 1)
                self.assertEqual(finished, [20])

    def test_real_calibration_sends_full_original_corpus_or_drains_failed_batch(self):
        with server() as (base, records, finished), contextlib.redirect_stdout(io.StringIO()) as output:
            calibration.collect(SimpleNamespace(base=base, model='fixture', tokens=4, streams=3, seed=7))
        self.assertIn('sent 32 of 32', output.getvalue())
        self.assertEqual(len(records), 32)
        self.assertEqual(len(finished), 32)
        actual = [(p, b.get('prompt', b.get('messages', [{}])[0].get('content')), b['temperature'], b['seed'])
                  for p, b in records]
        expected = [('/v1/chat/completions' if kind == 'chat' else '/v1/completions', text, temperature, 7 + i)
                    for temperature in (1., 0.) for i, (kind, text) in enumerate(calibration.CORPUS)]
        self.assertCountEqual(actual, expected)
        for options, error in [({'fail_seed': 7}, urllib.error.HTTPError), ({'mode': 'json_error'}, ValueError)]:
            with self.subTest(options=options), server(**options) as (base, records, finished), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(error):
                    calibration.collect(SimpleNamespace(base=base, model='fixture', tokens=4, streams=3, seed=7))
            self.assertEqual(len(records), 3)
            self.assertEqual(len(finished), 3)
            self.assertEqual(output.getvalue(), '')


if __name__ == '__main__':
    unittest.main()

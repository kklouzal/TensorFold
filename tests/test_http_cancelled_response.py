"""A canceled generation without a complete HTTP reply closes its transport owner."""
import http.client
import json
import unittest

from tensorfold.server.cancellation import RequestCancelled
from tests.test_http_service_limits import _App, _running, _sources


class CancelledResponse(unittest.TestCase):
    def test_cancel_before_output_closes_even_when_the_peer_is_still_connected(self):
        class CancelledApp(_App):
            def chat(self, *args, **kwargs):
                raise RequestCancelled('controlled request cancellation')

            def run(self, *args, **kwargs):
                raise RequestCancelled('controlled request cancellation')

        _, generic, cuda, _ = _sources()
        for factory in (generic, cuda):
            for stream in (False, True):
                with self.subTest(backend=factory.__module__, stream=stream), \
                     _running(factory(CancelledApp(limit=1))) as (server, failures):
                    client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
                    try:
                        body = {'messages': [{'role': 'user', 'content': 'controlled request'}], 'stream': stream}
                        client.request('POST', '/v1/chat/completions', json.dumps(body),
                                       {'Content-Type': 'application/json'})
                        with self.assertRaises(http.client.RemoteDisconnected):
                            client.getresponse()
                        self.assertEqual(failures, [])
                    finally:
                        client.close()

    def test_cuda_model_metadata_tolerates_query_and_trailing_slash(self):
        for path in ('/v1/models?fixture=1', '/models/?fixture=1'):
            with self.subTest(path=path), _running(_sources()[2](_App(limit=None))) as (server, _):
                client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=2)
                try:
                    client.request('GET', path)
                    response = client.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertEqual(json.loads(response.read())['data'][0]['id'], 'source-fixture')
                finally:
                    client.close()

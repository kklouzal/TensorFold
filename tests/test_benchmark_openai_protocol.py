"""SSE framing/resource/schema differential controls; no SDK/model execution."""
import importlib.util
import io
import json
from pathlib import Path
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('owned_openai_protocol', ROOT / 'tools/openai_protocol.py')
protocol = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = protocol
spec.loader.exec_module(protocol)


class Response(io.BytesIO):
    status = 200
    content_type = 'text/event-stream'
    def getheader(self, name, default=None):
        return self.content_type


class Protocol(unittest.TestCase):
    def test_every_byte_read_boundary_cr_lf_crlf_multiline_bom_unicode(self):
        for ending in (b'\n', b'\r', b'\r\n'):
            body = ending.join([b'\xef\xbb\xbf:comment', b'event: message', b'data: {"choices":',
                b'data: [{"delta":{"content":"caf\xc3\xa9"}}]}', b'', b'data: [DONE]', b'']) + ending
            expected = [{'choices': [{'delta': {'content': 'caf\u00e9'}}]}]
            for boundary in range(1, len(body)):
                class Split(Response):
                    first = True
                    def read1(self, amount):
                        self.assert_bound(amount)
                        if self.first:
                            self.first = False
                            amount = min(amount, boundary)
                        return super().read1(amount)
                    def assert_bound(self, amount):
                        if amount > 4096:
                            raise AssertionError('unbounded read')
                with self.subTest(ending=ending, boundary=boundary):
                    self.assertEqual(list(protocol.sse_objects(Split(body))), expected)

    def test_unterminated_done_data_is_not_a_dispatched_event(self):
        for data in (b'data: [DONE]', b'data: [DONE]\n', b'data: {}\n', b'data: {}\rdata: [DONE]\n'):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, r'before \[DONE\]'):
                list(protocol.sse_objects(Response(data)))

    def test_line_and_total_bounds_precede_unbounded_materialization(self):
        for data, limits in [
            (b'data: ' + b'x' * 100 + b'\n\n', protocol.Limits(line_bytes=32)),
            (b':heartbeat\n\n' * 10, protocol.Limits(response_bytes=32)),
        ]:
            with self.subTest(limits=limits), self.assertRaisesRegex(ValueError, 'bounded size'):
                list(protocol.sse_objects(Response(data), limits=limits))

    def test_larger_valid_line_is_available_via_explicit_operation_owned_budget(self):
        chunk = {'choices': [{'text': 'x' * 100}]}
        body = b'data: ' + json.dumps(chunk).encode() + b'\n\ndata: [DONE]\n\n'
        with self.assertRaisesRegex(ValueError, 'bounded size'):
            list(protocol.sse_objects(Response(body), limits=protocol.Limits(line_bytes=32)))
        self.assertEqual(list(protocol.sse_objects(Response(body), limits=protocol.Limits(line_bytes=256))), [chunk])
        for value in (0, -1, True, 2.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol.Limits(response_bytes=value)

    def test_nested_schema_invalids_error_objects_duplicate_keys_and_nonfinite(self):
        for value in [{'usage': []}, {'tensorfold': []}, {'tensorfold': {'token_sha': 7}}, {'choices': None},
                      {'choices': [None]}, {'choices': [{'delta': []}]}, {'choices': [{'text': 1}]},
                      {'choices': [{'delta': {'content': False}}]}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol.chunk_fields(value)
        for data in (b'[]', b'{"error":null}', b'{"usage":{},"usage":{}}', b'{"score":NaN}', b'{"score":Infinity}', b'{"score":1e400}'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                protocol.object_json(data)

    def test_typed_bounded_completion_and_positive_prompt_counts(self):
        for value in (None, False, 0, -1, 3, '2', 2.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol.completion_usage({'completion_tokens': value}, 2)
        for value in (None, False, 0, -1, '2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                protocol.completion_usage({'completion_tokens': 2, 'prompt_tokens': value}, 2, prompt=True)
        self.assertEqual(protocol.completion_usage({'completion_tokens': 2, 'prompt_tokens': 9}, 2, prompt=True), 2)

    def test_response_status_and_media_type_are_mandatory(self):
        response = Response(b'')
        protocol.response_type(response, 'text/event-stream')
        for status, media in [(201, 'text/event-stream'), (200, 'application/json'), (500, 'text/event-stream')]:
            response.status, response.content_type = status, media
            with self.assertRaises(ValueError):
                protocol.response_type(response, 'text/event-stream')

    def test_bounded_json_response_contains_real_usage_and_refuses_http200_error(self):
        response = Response(json.dumps({'choices': [{'text': 'fixture'}], 'usage': {'completion_tokens': 2}}).encode())
        response.content_type = 'application/json; charset=utf-8'
        self.assertEqual(protocol.json_response(response, 2)['usage']['completion_tokens'], 2)
        for raw in (b'{"error":{}}', b'{"choices":[]}', b'{}' * 100):
            response = Response(raw)
            response.content_type = 'application/json'
            with self.subTest(raw=raw[:32]), self.assertRaises(ValueError):
                protocol.json_response(response, 2, limits=protocol.Limits(response_bytes=64))


if __name__ == '__main__':
    unittest.main()

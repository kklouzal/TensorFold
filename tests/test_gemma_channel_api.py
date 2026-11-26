"""Replay malformed channel output through the real CUDA presentation path.

The scripted engine deliberately does not enforce generation constraints: these
checks exercise presentation of received tokens, not grammar sampling quality.
"""
import ast
import json
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import pytest

from tests.test_gemma_api_contract import GEMMA_CALLS, STOP, app, grammars as grammars, render
from tests.test_strict_tools import tool


def replay(tmp_path, text, *, opened=False, constrained=None):
    channel = opened if isinstance(opened, str) else 'thought'
    template = GEMMA_CALLS + ('<|channel>' + channel + '\n' if opened else '')
    instance = app(tmp_path, text, constrained, template)

    def generate(prompt, max_tokens, sampling, on_tokens, constraint=None, **kwargs):
        for token in instance.engine.script[:max_tokens]:
            if on_tokens([token]) or token == STOP:
                break
        return {}

    instance.engine.generate = generate
    return instance


CASES = [
    ('<|channel>thought\nx<channel|><|channel>thought <|channel>synthesis <channel|>Answer.', 'Answer.'),
    ('<|channel>synthesis\nnotes<channel|>Answer.', 'Answer.'),
    ('Visible. <|channel>analysis notes<channel|>Answer.', 'Visible. Answer.'),
    ('<channel|>Answer.', 'Answer.'),
    ('Visible. <|channel>synthesis\nnotes', 'Visible. '),
]


@pytest.mark.parametrize('raw,expected', CASES)
@pytest.mark.parametrize('opened', [False, True])
@pytest.mark.parametrize('stream', [False, True])
def test_cuda_routes_all_channels_without_content_leak(tmp_path, raw, expected, opened, stream):
    instance = replay(tmp_path, raw, opened=opened)
    deltas = []
    body = {'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': stream}
    result = instance.run(body, True, lambda value: deltas.append(value) or True)
    # A prompt-opened channel owns any leading body until its first close.
    if opened and raw.startswith('Visible.'):
        expected = 'Answer.' if '<channel|>' in raw else ''
    assert result['content'] == expected
    assert '<|channel>' not in result['reasoning'] and '<channel|>' not in result['reasoning']
    emitted = ''.join((value or {}).get('content', '') for value in deltas)
    # App.run supplies cumulative presentation deltas even for a non-streaming
    # handler; the HTTP layer chooses whether to serialize them as SSE.
    assert emitted == expected
    if 'notes' in raw:
        assert 'notes' in result['reasoning']


@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('opened', [False, True])
def test_cuda_tool_turn_keeps_literal_markers_in_arguments(tmp_path, grammars, strict, stream, opened):
    literal = '<|channel>thought <|channel>synthesis <channel|>'
    raw = ('<|channel>thought\nwhy<channel|><|channel>synthesis\nmore<channel|>'
           + render({'text': literal}))
    instance = replay(tmp_path, raw, opened=opened, constrained=grammars)
    specification = {'type': 'object', 'properties': {'text': {'type': 'string'}},
                     'required': ['text'], 'additionalProperties': False}
    deltas = []
    body = {'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': stream,
            'tools': [tool('f', specification, strict=strict)]}
    result = instance.run(body, True, lambda value: deltas.append(value) or True)
    assert result['content'] == ''
    assert result['finish'] == 'tool_calls'
    assert json.loads(result['calls'][0]['function']['arguments']) == {'text': literal}
    assert 'why' in result['reasoning'] and 'more' in result['reasoning']
    assert not any((value or {}).get('content') for value in deltas)


@pytest.mark.parametrize('raw,expected', CASES[:2])
@pytest.mark.parametrize('opened', [False, True])
@pytest.mark.parametrize('stream', [False, True])
def test_http_message_and_sse_serialize_only_the_routed_answer(tmp_path, raw, expected, opened, stream):
    from tests.test_cuda_admission import http_server, post
    from tests.test_strict_tools import _reply

    instance = replay(tmp_path, raw, opened=opened)
    with http_server(instance) as port:
        status, response = post(port, {'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': stream}, True)
    assert status == 200, response
    content, calls, finish = _reply(stream, response)
    assert (content, calls, finish) == (expected, [], 'stop')
    assert '<|channel>' not in response and '<channel|>' not in response


@pytest.mark.parametrize('opened', [False, True])
@pytest.mark.parametrize('stream', [False, True])
def test_unclosed_channel_cannot_publish_a_call_when_no_tools_are_declared(tmp_path, opened, stream):
    call = render({'text': 'literal <|channel>thought <channel|>'})
    instance = replay(tmp_path, '<|channel>analysis\nWhy ' + call, opened=opened)
    deltas = []
    result = instance.run({'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': stream},
                          True, lambda value: deltas.append(value) or True)
    assert result['content'] == '' and result['calls'] is None
    assert call in result['reasoning']
    assert not any((value or {}).get('content') for value in deltas)


def test_strict_final_parser_uses_routed_answer_even_without_sampler_grammar(tmp_path, grammars):
    specification = {'type': 'object', 'properties': {'text': {'type': 'string'}},
                     'required': ['text'], 'additionalProperties': False}
    raw = '<|channel>synthesis\nprivate notes<channel|>' + render({'text': 'value'})
    instance = replay(tmp_path, raw, constrained=grammars)
    body = {'messages': [{'role': 'user', 'content': 'Hi'}],
            'tools': [tool('f', specification, strict=True)]}
    prepared = instance._prepare(body, True)
    # Exercise the final parser's supported no-grammar branch independently of
    # sampling. Strict result parsing must consume the same routed text.
    prepared.grammar = None
    result = instance.run(body, True, lambda value: True, prepared=prepared)
    assert result['content'] == ''
    assert result['reasoning'] == 'private notes'
    assert json.loads(result['calls'][0]['function']['arguments']) == {'text': 'value'}


@pytest.mark.parametrize("with_tools", [False, True])
def test_native_cuda_tokenizes_gemma_chat_with_exact_controls(tmp_path, with_tools, monkeypatch):
    from tensorfold.engine import exact_sampling
    from tests.test_cuda_admission import http_server
    from tests.test_token_routes import call

    plain = GEMMA_CALLS + "{% if add_generation_prompt and enable_thinking %}<|channel>thought\n{% endif %}"
    instance = app(tmp_path, "unused", template={"default": plain, "tool_use": plain + " "})
    instance.aliases = ("gemma-alias",)
    instance.native_context_window = instance.context_window = 4
    instance.sampling.update(top_k=0, top_p=1.0)
    body = {"model": "gemma-alias", "messages": [{"role": "user", "content": "Hello."}],
            "chat_template_kwargs": {"enable_thinking": True}, "temperature": 1.0}
    if with_tools:
        schema = {"type": "object", "properties": {"value": {"type": "string"}},
                  "required": ["value"], "additionalProperties": False}
        body["tools"] = [tool("f", schema, strict=True)]
    prepared = instance._prepare(body, True, for_generation=False)
    expected = prepared.prompt
    prepared.close()

    def count_seed(tokens):
        assert not tokens  # count validates sampling without keying it to the whole prompt
        return 0

    monkeypatch.setattr(exact_sampling, "seed_for", count_seed)
    with http_server(instance) as port:
        status, count = call(port, "/v1/tokenize", {**body, "return_tokens": False})
        assert status == 200, count
        assert count["count"] == len(expected) > count["context_length"] == count["max_model_len"] == 4
        assert count["model"] == "gemma-alias"
        assert count["harness_tokenizer_version"] == 1
        assert count["chat_template_sha256"] == instance.template.selected(tools=with_tools).sha256
        assert count["capabilities"]["thinking_control"]["keyword"] == "enable_thinking"
        assert "tokens" not in count and "prompt" not in count
        status, inspected = call(port, "/v1/tokenize", {**body, "return_prompt": True})
        assert status == 200, inspected
        assert inspected["tokens"] == expected
        assert inspected["prompt"] == instance.tok.decode(expected, skip_special_tokens=False)
        assert "<|channel>thought\n" in inspected["prompt"]
        for field in ("return_tokens", "return_prompt"):
            status, refused = call(port, "/v1/tokenize", {**body, field: 1})
            assert status == 400 and f"{field} must be a boolean" in refused["error"]["message"]
        assert call(port, "/v1/tokenize", {**body, "model": "other"})[0] == 404
    assert not instance.engine.calls


@pytest.mark.parametrize('headers', [(), ('<|tool_call>call:other{',), ('<|tool_call>call:echo{',)])
@pytest.mark.parametrize('quoted', [False, True])
def test_native_boundary_shields_only_quoted_tool_values_independent_of_names(headers, quoted):
    from tensorfold.engine.grammar import AnswerBoundary
    prefix = '<|tool_call>call:echo{x:' + ('<|"|>literal ' if quoted else '0,')
    pieces = {1: prefix, 101: '<channel|>'}
    boundary = AnswerBoundary(None, think_end=101, think_starts=(100,), thought_open=True,
                              decode=lambda ids, **kwargs: ''.join(pieces[token] for token in ids),
                              answer_headers=headers)
    boundary.advance([1, 101])
    assert boundary.active is (not quoted)
    assert (boundary.answer_offset is None) is quoted


@pytest.mark.parametrize('channel', ['analysis', 'synthesis', 'unknown.name'])
@pytest.mark.parametrize('stream', [False, True])
def test_implicit_unknown_prompt_header_owns_the_generated_body(tmp_path, channel, stream):
    instance = replay(tmp_path, 'notes<channel|>Answer.', opened=channel)
    deltas = []
    result = instance.run({'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': stream},
                          True, lambda value: deltas.append(value) or True)
    assert result['content'] == 'Answer.' and result['reasoning'] == 'notes'
    assert ''.join((value or {}).get('content', '') for value in deltas) == 'Answer.'


@pytest.mark.parametrize('raw', [
    '<|channel>thought\nA<|chan<channel|>visible<|channel>analysis\nnel>Z',
    '<|tool_<|channel>thought\nR<channel|>call>hello',
])
def test_partial_fragments_at_native_boundary_never_retract_streamed_fields(tmp_path, raw):
    from tensorfold.server.gemma_values import split_gemma_thinking
    expected_reason, expected_answer = split_gemma_thinking(raw, finished=True, allow_tool_calls=False)
    instance = replay(tmp_path, raw)
    deltas = []
    result = instance.run({'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': True},
                          True, lambda value: deltas.append(value) or True)
    deltas.append(result['final'])
    assert (result['reasoning'], result['content']) == (expected_reason, expected_answer)
    assert ''.join((value or {}).get('reasoning_content', '') for value in deltas) == expected_reason
    assert ''.join((value or {}).get('content', '') for value in deltas) == expected_answer
    assert '<|channel>' not in result['reasoning'] + result['content']


@pytest.mark.parametrize('opened', [False, True])
def test_proved_native_call_intent_still_streams_arguments_without_thought_close(tmp_path, grammars, opened):
    specification = {'type': 'object', 'properties': {'text': {'type': 'string'}},
                     'required': ['text'], 'additionalProperties': False}
    raw = ('' if opened else '<|channel>thought\n') + 'Why. ' + render({'text': 'abcdef'})
    instance = replay(tmp_path, raw, opened=opened, constrained=grammars)
    instance.default_thinking = True  # configure the matcher's reasoning/intent phase too
    deltas = []
    result = instance.run({'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': True,
                           'tools': [tool('f', specification, strict=True)]},
                          True, lambda value: deltas.append(value) or True)
    fragments = [call.get('function', {}).get('arguments', '') for value in deltas
                 for call in (value or {}).get('tool_calls', ())]
    assert len([part for part in fragments if part]) > 1
    assert json.loads(''.join(fragments)) == {'text': 'abcdef'}
    assert result['reasoning'] == 'Why. ' and result['content'] == ''


def test_non_cuda_prompt_phase_uses_only_the_actual_generation_turn():
    # Importing ChatApp requires MLX. Execute this exact host-only method on
    # Linux as a protocol control; full MLX serving remains a platform check.
    from tensorfold.server.text import CHANNEL_MARKERS
    source = Path(__file__).parents[1] / 'src/tensorfold/server/app.py'
    tree = ast.parse(source.read_bytes())
    chat = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ChatApp')
    method = next(node for node in chat.body if isinstance(node, ast.FunctionDef)
                  and node.name == '_gemma_prompt_thought_open')
    namespace = {'CHANNEL_MARKERS': CHANNEL_MARKERS}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(source), 'exec'), namespace)
    pieces = {1: '<|turn>', 2: '<|channel>', 3: 'thought', 4: '\n', 5: '<channel|>',
              6: 'model\n', 7: '<turn|>', 8: 'user\n', 9: 'analysis'}
    decoded = []

    def decode(ids):
        decoded.append(list(ids))
        return ''.join(pieces[token] for token in ids)

    ids = {value: key for key, value in pieces.items()}
    subject = SimpleNamespace(think_markers=CHANNEL_MARKERS,
                              _token_id=lambda mark: ids.get(mark, -1),
                              tokenizer_lock=Lock(), tokenizer=SimpleNamespace(decode=decode))
    opened = namespace['_gemma_prompt_thought_open']
    assert opened(subject, [1, 8, 7, 1, 6, 2, 3, 4])
    assert decoded == [[2, 3, 4]]  # never decode/copy the full conversation
    assert not opened(subject, [1, 8, 2, 3, 4, 7, 1, 6])
    assert not opened(subject, [1, 6, 2, 3, 4, 5])
    assert opened(subject, [1, 6, 2, 9, 4])  # Unknown native channels are non-final too.


def _raw(instance, prompt):
    deltas = []
    result = instance.run({'prompt': prompt, 'stream': True}, False, lambda value: deltas.append(value) or True)
    contents = [(value or {}).get('content', '') for value in deltas]
    return result, [text for text in contents if text]


def test_raw_completion_without_open_channel_streams_each_round(tmp_path):
    # No template frames a raw prompt: unless the prompt itself opened a channel, visible text is not held
    # for an answer boundary that can only resolve at the end of the reply.
    instance = replay(tmp_path, 'Visible answer.')
    result, contents = _raw(instance, 'Hi')
    assert result['content'] == 'Visible answer.'
    assert ''.join(contents) == 'Visible answer.'
    assert len(contents) == len('Visible answer.')       # one delta per generated token, not one at the end


def test_raw_completion_ending_inside_an_open_thought_keeps_the_holdback(tmp_path):
    instance = replay(tmp_path, 'notes<channel|>Answer.')
    result, contents = _raw(instance, 'Hi<|channel>thought\n')
    assert result['content'] == 'Answer.'
    assert 'notes' not in ''.join(contents)


def test_raw_prompt_channel_detection_uses_real_tokens_and_a_short_header():
    from tensorfold.server.gemma_values import prompt_ids_open_thought
    from tests.test_gemma_api_contract import GemmaTokenizer

    tok = GemmaTokenizer()
    assert prompt_ids_open_thought(tok, tok.encode('Hi<|channel>thought\n').ids)
    assert prompt_ids_open_thought(tok, tok.encode('Hi<|channel>').ids)
    assert not prompt_ids_open_thought(tok, tok.encode('Hi').ids)
    assert not prompt_ids_open_thought(tok, tok.encode('Hi<|channel>thought\nnotes<channel|>').ids)
    assert not prompt_ids_open_thought(tok, tok.encode('Hi<|channel>thought\nlong body text').ids)


@pytest.mark.parametrize('raw,expected', [
    ('Visible. <|channel>thought\nnotes<channel|>Answer.', 'Visible. Answer.'),
    ('<|channel>thought\nnotes<channel|>Answer.', 'Answer.'),
    ('Visible. <|channel>synthesis\nnotes', 'Visible. '),
])
def test_raw_completion_reply_opening_a_channel_routes_it_as_before(tmp_path, raw, expected):
    # Text before the reply's own native opener has already streamed; from the opener on the answer boundary
    # hides and separates the channel exactly as when it was created at the start.
    instance = replay(tmp_path, raw)
    result, contents = _raw(instance, 'Hi')
    assert result['content'] == expected
    assert ''.join(contents) == expected
    assert 'notes' not in ''.join(contents)
    if raw.startswith('Visible.'):
        assert contents[0] == 'V'                       # the plain prefix streamed before the opener arrived


def test_raw_completion_prefix_before_a_reply_channel_keeps_a_possible_stop_start(tmp_path):
    # The text before the reply's own opener is final once the opener arrives: an end that could only begin a stop
    # string (here "EN" of "END") is content, not a held-back partial match.
    instance = replay(tmp_path, 'Visible EN<|channel>thought\nnotes<channel|>Answer.')
    deltas = []
    result = instance.run({'prompt': 'Hi', 'stream': True, 'stop': ['END']}, False,
                          lambda value: deltas.append(value) or True)
    contents = ''.join((value or {}).get('content', '') for value in deltas)
    assert result['content'] == 'Visible ENAnswer.'
    assert contents == 'Visible ENAnswer.'

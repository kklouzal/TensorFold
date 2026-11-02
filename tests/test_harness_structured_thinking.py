"""Thinking/schema/tool state integration through real grammar, fake CPU model."""
import json

import pytest

from tests import test_strict_tools as strict_tools
from tests.test_strict_tools import ScriptEngine, _app, tool

grammars = strict_tools.grammars


@pytest.mark.parametrize('stream', [False, True])
def test_response_format_data_is_not_rewritten_or_executed_as_advertised_tools(tmp_path, grammars, stream):
    value = '<tool_call>literal</tool_call> <|channel>thought <channel|> </think>'
    payload = json.dumps({'x': value}, separators=(',', ':'))
    app = _app(tmp_path, ScriptEngine('why\n</think>' + payload), grammars)
    app.default_thinking = True
    emitted = []
    body = {'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': stream,
            'tools': [tool('f', strict=False)],
            'response_format': {'type': 'json_schema', 'json_schema': {'schema': {
                'type': 'object', 'properties': {'x': {'type': 'string'}}, 'required': ['x'],
                'additionalProperties': False}}}}
    result = app.run(body, True, lambda delta: emitted.append(delta) or True)
    assert result['content'] == payload and json.loads(result['content']) == {'x': value}
    assert result['calls'] is None and result['reasoning'] == 'why\n'
    emitted.append(result['final'])
    assert ''.join((delta or {}).get('content', '') for delta in emitted) == payload
    assert not any((delta or {}).get('tool_calls') for delta in emitted)


@pytest.mark.parametrize('stream', [False, True])
def test_strict_tool_inside_open_unframed_thinking_activates_before_arguments(tmp_path, grammars, stream):
    call = '<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>'
    app = _app(tmp_path, ScriptEngine('why\n' + call), grammars)
    app.default_thinking = True
    body = {'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': stream, 'tool_choice': 'required',
            'tools': [tool('f', {'type': 'object', 'properties': {'x': {'type': 'integer'}},
                                 'required': ['x'], 'additionalProperties': False})]}
    result = app.run(body, True, lambda delta: True)
    assert result['finish'] == 'tool_calls' and result['reasoning'] == 'why\n'
    assert result['calls'][0]['function']['name'] == 'f'
    assert json.loads(result['calls'][0]['function']['arguments']) == {'x': 1}
    assert result['content'] == ''


def test_native_eos_in_thinking_stays_a_reasoning_only_stop_under_schema(tmp_path, grammars):
    app = _app(tmp_path, ScriptEngine('why'), grammars)
    app.default_thinking = True
    body = {'messages': [{'role': 'user', 'content': 'Hi'}], 'response_format': {'type': 'json_object'}}
    result = app.run(body, True, lambda delta: True)
    assert result['finish'] == 'stop' and result['reasoning'] == 'why'
    assert not result['content'] and result['calls'] is None


@pytest.mark.parametrize('terminal', [False, True])
def test_lenient_tool_stream_finalfeed_only_on_actual_normal_terminal(tmp_path, grammars, monkeypatch, terminal):
    from tensorfold.cuda import server
    calls = []
    class Stream:
        index = -1
        streamed = False
        complete = True
        def feed(self, text, *, finished=False):
            calls.append(finished)
            return []
    monkeypatch.setattr(server, 'ToolCallStreamer', lambda tools: Stream())
    app = _app(tmp_path, ScriptEngine('plain'), grammars)
    body = {'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': True,
            'max_tokens': 20 if terminal else 1, 'tools': [tool('f', strict=False)]}
    result = app.run(body, True, lambda delta: True)
    assert result['finish'] == ('stop' if terminal else 'length')
    assert calls.count(True) == int(terminal)
    if terminal:
        assert calls[-1] is True


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('ending', ['eos', 'cap', 'userstop'])
def test_lenient_ambiguous_quote_uses_same_terminal_authority_for_stream_and_final_parse(
        tmp_path, grammars, stream, ending):
    envelope = '<tool_call><function=echo><parameter=text>"raw unmatched</parameter></function></tool_call>'
    engine = ScriptEngine(envelope + (' HALT' if ending == 'userstop' else ''))
    app = _app(tmp_path, engine, grammars)
    emitted = []
    body = {'messages': [{'role': 'user', 'content': 'Hi'}], 'stream': stream,
            'max_tokens': len(engine.script)-1 if ending == 'cap' else len(engine.script)+1,
            'tools': [tool('echo', {'type': 'object', 'properties': {'text': {'type': 'string'}}}, strict=False)]}
    if ending == 'userstop':
        body['stop'] = ' HALT'
    result = app.run(body, True, lambda delta: emitted.append(delta) or True)
    if ending == 'eos':
        assert result['finish'] == 'tool_calls'
        assert json.loads(result['calls'][0]['function']['arguments']) == {'text': '"raw unmatched'}
    else:
        assert result['finish'] == ('length' if ending == 'cap' else 'stop')
        assert not result['calls']
        partial = ''.join(call.get('function', {}).get('arguments', '')
                          for delta in emitted for call in ((delta or {}).get('tool_calls') or ()))
        with pytest.raises(json.JSONDecodeError):
            json.loads(partial)  # partial prefixes may remain; no premature completed raw fallback


@pytest.mark.parametrize('ending', ['answer', 'thought-eos', 'header-eos', 'backtick', 'direct', 'omitted'])
def test_native_gemma_framing_only_callbacks_never_leak_or_slice_reasoning(tmp_path, ending):
    """Native header/recorded body IDs; sparse decoder isolates presentation, not model matching."""
    from tensorfold.cuda import server
    from tensorfold.engine.grammar import AnswerBoundary, Spec

    pieces = {100: '<|channel>', 45518: 'thought', 107: '\n', 818: 'The', 2430: ' user',
              563: ' is', 10980: ' asking', 101: '<channel|>', 200: '{"ok":true}', 106: '<eos>',
              201: 'The character ` starts code.', 202: 'The answer.'}
    class Tok:
        control_token_ids = unused_token_ids = frozenset()
        def decode(self, ids, **kwargs):
            return ''.join(pieces[int(t)] for t in ids)
        def token_to_id(self, text):
            return {'<channel|>': 101, '<|channel>': 100}.get(text)
        def id_to_token(self, token):
            return pieces[token]
    class Constraint(AnswerBoundary):
        def fresh_boundary(self):
            return AnswerBoundary(self.vocab, think_end=self.think_end, decode=self.decode,
                                  think_starts=self.think_starts, thought_open=self.initial_thought_open)
    class Grammars:
        def constraint(self, compiled, *, think_end=None, decode=None, **kwargs):
            return Constraint(262144, think_end=think_end, decode=decode,
                              think_starts=kwargs.get('think_starts', ()), thought_open=kwargs.get('thought_open', False))
    chunks = [] if ending in ('direct', 'omitted') else [[100], [45518]]
    reason = ''
    if ending not in ('header-eos', 'direct'):
        chunks.append([107, 201] if ending == 'backtick' else [107, 818, 2430, 563, 10980, 107])
        reason = 'The character ` starts code.' if ending == 'backtick' else 'The user is asking\n'
        if ending == 'omitted':
            reason = '\n' + reason  # no channel header owns this actual leading newline
    if ending in ('answer', 'backtick', 'direct', 'omitted'):
        chunks.append([200] if ending == 'direct' else [101, 202 if ending == 'backtick' else 200])
    chunks.append([106])
    ids = [token for chunk in chunks for token in chunk]
    class Engine:
        eos = (106,)
        def __init__(self, streaming):
            self.streaming = streaming
        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True, constraint=None):
            for chunk in chunks if self.streaming else [ids]:
                constraint.advance(chunk)
                if on_tokens(chunk):
                    break
            return {'cached': 0}
    observed = []
    results = []
    for streaming in (False, True):
        app = _app(tmp_path, Engine(streaming), Grammars())
        app.tok = Tok()
        prepared = server.PreparedRequest([3], len(ids)+1, [], True, None,
                                         grammar=(Spec('regex', 'presentation-only'), object()))
        emitted = []
        result = app.run({'stream': streaming, 'return_token_ids': True}, True,
                         lambda delta: emitted.append(delta) or True, prepared=prepared)
        results.append(result)
        if streaming:
            emitted.append(result['final'])
            observed = emitted
    assert results[0]['stats']['token_ids'] == results[1]['stats']['token_ids'] == ids
    assert results[0]['reasoning'] == results[1]['reasoning'] == reason
    expected_answer = 'The answer.' if ending == 'backtick' else '{"ok":true}' if ending in ('answer', 'direct', 'omitted') else ''
    assert results[0]['content'] == results[1]['content'] == expected_answer
    assert ''.join((delta or {}).get('reasoning_content', '') for delta in observed) == reason
    assert ''.join((delta or {}).get('content', '') for delta in observed) == expected_answer
    assert not any('<|channel>' in (delta or {}).get('reasoning_content', '') for delta in observed)
    assert results[1]['finish'] == 'stop'  # native EOS is not promoted or force-closed
    expected_reason_tokens = 0 if ending == 'direct' else ids.index(101)+1 if 101 in ids else len(ids)
    assert results[0]['reasoning_tokens'] == results[1]['reasoning_tokens'] == expected_reason_tokens

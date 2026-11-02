"""Whitespace is not companion prose; no-call outcomes and literal data stay exact."""
import pytest

from tensorfold.server.tool_policy import ToolCallPolicy, visible_tool_prose
from tensorfold.server.tools import parse_tool_calls_from_content

TOOLS = [{'type': 'function', 'function': {'name': 'echo', 'parameters': {'type': 'object'}}}]
CALL = '<|tool_call>call:echo{}<tool_call|>'


@pytest.mark.parametrize('white', [' ', ' \n', '\t\r\n', '\u2003'])
def test_cumulative_tool_visibility_holds_only_ambiguous_whitespace(white):
    assert visible_tool_prose(white) == visible_tool_prose(white, calls=True) == ''
    assert visible_tool_prose(white, calls=False) == white
    prose = white + 'Meaningful ```text\n data \n``` ' + white
    assert visible_tool_prose(prose) == visible_tool_prose(prose, calls=True) == prose


@pytest.mark.parametrize('parallel', [False, True])
def test_whitespace_only_companion_is_never_emitted_alongside_actual_call(parallel):
    policy = ToolCallPolicy({'parallel_tool_calls': parallel})
    assert not policy.delta({'content': ' \n'})
    reply = policy.finish({'content': ' \n' + CALL, 'finish_reason': 'stop'}, TOOLS, parse_tool_calls_from_content)
    assert reply['tool_calls'] and reply['content'] == ''
    assert policy.flush() == ''


@pytest.mark.parametrize('parallel', [False, True])
def test_no_call_whitespace_is_delayed_then_restored_without_fake_call(parallel):
    policy = ToolCallPolicy({'parallel_tool_calls': parallel})
    assert policy.delta(' \n') == ''
    reply = policy.finish({'content': ' \n', 'finish_reason': 'stop'}, TOOLS, parse_tool_calls_from_content)
    assert reply['content'] == ' \n' and not reply.get('tool_calls') and reply['finish_reason'] == 'stop'
    assert policy.flush() == ' \n'


def test_meaningful_prose_releases_original_whitespace_without_rewriting_its_data():
    policy = ToolCallPolicy({})
    assert policy.delta(' \n') == ''
    assert policy.delta('Text `literal <|tool_call>`') == ' \nText `literal <|tool_call>`'
    assert policy.delta(' \n') == ' \n'


@pytest.mark.parametrize('parallel', [False, True])
def test_response_format_data_bypasses_all_tool_presentation(parallel):
    policy = ToolCallPolicy({'parallel_tool_calls': parallel, 'response_format': {'type': 'json_object'}})
    data = {'content': ' \n{"literal":"<|tool_call>call:echo{}<tool_call|>"}'}
    assert policy.delta(data) is data
    def forbidden(*args, **kwargs):
        raise AssertionError('response_format data must not be parsed as calls')
    reply = {'content': data['content'], 'finish_reason': 'stop'}
    assert policy.finish(reply, TOOLS, forbidden) is reply
    assert policy.flush() == ''


def test_partial_call_preserves_actual_cap_and_does_not_discard_no_call_whitespace():
    policy = ToolCallPolicy({})
    assert policy.delta(' \n') == ''
    partial = '<|tool_call>call:echo{'
    reply = policy.finish({'content': partial, 'finish_reason': 'length',
                           'tool_parse_finished': False, 'tool_calls_complete': False},
                          TOOLS, parse_tool_calls_from_content)
    assert reply['finish_reason'] == 'length' and not reply.get('tool_calls')
    assert policy.flush() == ' \n'


@pytest.mark.parametrize('parallel', [False, True])
def test_tool_argument_whitespace_is_data_not_companion_content(parallel):
    import json

    policy = ToolCallPolicy({'parallel_tool_calls': parallel})
    call = '<|tool_call>call:echo{text:<|"|> \n<|"|>}<tool_call|>'
    reply = policy.finish({'content': ' \n' + call, 'finish_reason': 'stop'}, TOOLS, parse_tool_calls_from_content)
    assert reply['content'] == ''
    assert json.loads(reply['tool_calls'][0]['function']['arguments']) == {'text': ' \n'}
    if parallel:
        delta = {'tool_calls': [{'index': 0, 'function': {'arguments': '{"text":" \\n"}'}}]}
        assert policy.delta(delta) is delta

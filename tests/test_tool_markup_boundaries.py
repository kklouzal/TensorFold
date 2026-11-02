"""Actual call framing versus quoted examples/opaque argument data; all delta cuts."""
import json

import pytest

from tensorfold.engine.tool_draft import ToolCallStreamer
from tensorfold.server.tools import normalize_qwen_tool_markup, parse_tool_calls_from_content
from tensorfold.server.text import hide_tool_calls
from tensorfold.server.tool_policy import ToolCallPolicy
from tensorfold.cuda.reply_text import parse_tool_calls as parse_cuda

TOOLS = [{'type': 'function', 'function': {'name': 'echo', 'parameters': {
    'type': 'object', 'properties': {'text': {'type': 'string'}}}}}]
NATIVE = '<|tool_call>call:echo{text:<|"|>hello<|"|>}<tool_call|>'
XML = '<tool_call><function=echo><parameter=text>hello</parameter></function></tool_call>'


def calls(parser, text, **kwargs):
    content, result = parser(text, TOOLS, **kwargs)
    return content, [(x['function']['name'], json.loads(x['function']['arguments'])) for x in result or []]


def test_repeated_native_envelopes_are_one_frame_but_distinct_calls_remain_distinct():
    doubled = '<|tool_call>' + NATIVE + '<tool_call|>'
    assert calls(parse_tool_calls_from_content, doubled) == ('', [('echo', {'text': 'hello'})])
    assert calls(parse_tool_calls_from_content, NATIVE + NATIVE) == ('', [('echo', {'text': 'hello'})] * 2)
    assert calls(parse_tool_calls_from_content, NATIVE + NATIVE, max_calls=1) == ('', [('echo', {'text': 'hello'})])


@pytest.mark.parametrize('literal', [NATIVE, XML, '{"name":"echo","arguments":{"text":"hello"}}'])
@pytest.mark.parametrize('wrapper', ['`{}`', '```json\n{}\n```', '~~~text\n{}\n~~~'])
def test_quoted_tool_examples_never_become_calls(literal, wrapper):
    raw = wrapper.format(literal)
    assert normalize_qwen_tool_markup(raw) == raw
    assert calls(parse_tool_calls_from_content, raw) == (raw, [])
    assert calls(parse_cuda, raw) == (raw, [])
    streamer = ToolCallStreamer(TOOLS)
    policy = ToolCallPolicy({'parallel_tool_calls': False})
    deltas, visible = [], []
    for end in range(1, len(raw) + 1):
        deltas += streamer.feed(raw[:end])
        visible.append(policy.delta(raw[end - 1:end]))
        assert hide_tool_calls(raw[:end], finished=False) == raw[:end]
    visible.append(policy.flush())
    assert not deltas and ''.join(visible) == raw


def test_native_string_values_keep_nested_tool_and_channel_literals():
    value = '<tool_call><tool_call>literal</tool_call> <|channel>not a channel<channel|>'
    raw = '<|tool_call>call:echo{text:<|"|>' + value + '<|"|>}<tool_call|>'
    assert normalize_qwen_tool_markup(raw) == raw
    assert calls(parse_tool_calls_from_content, raw) == ('', [('echo', {'text': value})])


@pytest.mark.parametrize('parser', [parse_tool_calls_from_content, parse_cuda])
def test_xml_parameter_tool_markers_are_data_and_do_not_truncate_outer_frame(parser):
    value = 'İ echo `unterminated </tool_call><tool_call> payload'
    raw = '<tool_call><function=echo><parameter=text>' + value + '</parameter></function></tool_call>'
    assert calls(parser, raw) == ('', [('echo', {'text': value})])
    streamer = ToolCallStreamer(TOOLS)
    deltas = []
    for end in range(len(raw) + 1):
        deltas += streamer.feed(raw[:end])
    arguments = ''.join(d['tool_calls'][0]['function'].get('arguments', '') for d in deltas)
    assert json.loads(arguments) == {'text': value}


def test_json_string_with_tool_markup_is_not_an_outer_call_or_repaired_value():
    raw = json.dumps({'data': XML + '<tool_call><tool_call>'})
    assert normalize_qwen_tool_markup(raw) == raw
    assert calls(parse_tool_calls_from_content, raw) == (raw, [])
    assert calls(parse_cuda, raw) == (raw, [])


def test_multiple_xml_functions_are_split_without_repairing_argument_bytes():
    first = '<function=echo><parameter=text>literal </function><function=echo> in value</parameter></function>'
    second = '<function=echo><parameter=text>second</parameter></function>'
    raw = '<tool_call>' + first + second + '</tool_call>'
    expected = [('echo', {'text': 'literal </function><function=echo> in value'}), ('echo', {'text': 'second'})]
    assert calls(parse_tool_calls_from_content, raw) == ('', expected)
    assert calls(parse_cuda, raw) == ('', expected)


@pytest.mark.parametrize('opener,closer', [
    ('<ns:tool_call>', '</ns:tool_call>'),
    ('<TOOL_CALL>', '</TOOL_CALL>'),
    ('<ToOl_CaLl>', '</tOoL_cAlL>'),
])
def test_partial_xml_openers_preserve_single_call_text_filter_monotonicity(opener, closer):
    raw = 'Before ' + opener + '<function=echo><parameter=text>hello</parameter></function>' + closer + ' After'
    for cut in range(len(raw) + 1):
        policy = ToolCallPolicy({'parallel_tool_calls': False})
        shown = policy.delta(raw[:cut]) + policy.delta(raw[cut:]) + policy.flush()
        assert shown == 'Before  After'
    policy = ToolCallPolicy({'parallel_tool_calls': False})
    assert ''.join(policy.delta(char) for char in raw) + policy.flush() == 'Before  After'


@pytest.mark.parametrize('raw', ['ordinary <name:toolbox> text', 'ordinary <name> text',
                                  '`<ns:tool_call>`', '"<TOOL_CALL>"',
                                  '```text\n<ns:tool_call>\n```'])
def test_partial_xml_noncall_or_quoted_data_remains_exact(raw):
    policy = ToolCallPolicy({'parallel_tool_calls': False})
    assert ''.join(policy.delta(char) for char in raw) + policy.flush() == raw


@pytest.mark.parametrize('value', ['{"a":"x</parameter>y","b":[1,2]}', "{'a':'x</parameter>y','b':[1,2]}"])
@pytest.mark.parametrize('kind', ['object', 'string'])
@pytest.mark.parametrize('step', [1, 7, 10000])
def test_json_parameter_delimiter_literal_does_not_swallow_complete_envelope(kind, step, value):
    from tensorfold.markup_lex import markup_regions
    raw = '<tool_call><function=echo><parameter=text>' + value + '</parameter></function></tool_call>'
    tools = [{'type': 'function', 'function': {'name': 'echo', 'parameters': {
        'type': 'object', 'properties': {'text': {'type': kind}}}}}]
    expected = {'text': {'a': 'x</parameter>y', 'b': [1, 2]} if kind == 'object' else value}
    regions = markup_regions(raw)
    assert len(regions) == 1 and regions[0].kind == 'call' and regions[0].complete
    assert regions[0].end == len(raw)
    for parser in (parse_tool_calls_from_content, parse_cuda):
        for maximum in (None, 1):
            _, result = parser(raw, tools, max_calls=maximum)
            assert json.loads(result[0]['function']['arguments']) == expected
    streamer = ToolCallStreamer(tools)
    deltas = []
    for end in range(1, len(raw) + 1, step):
        deltas += streamer.feed(raw[:end])
    deltas += streamer.feed(raw, finished=True)
    arguments = ''.join(delta['tool_calls'][0]['function'].get('arguments', '') for delta in deltas)
    assert json.loads(arguments) == expected
    policy = ToolCallPolicy({'parallel_tool_calls': False})
    assert ''.join(policy.delta(char) for char in raw) + policy.flush() == ''


@pytest.mark.parametrize('value', ['{"a":"cut', '"raw unmatched', "'raw unmatched"])
@pytest.mark.parametrize('kind', ['object', 'string'])
def test_malformed_quoted_value_preserves_exact_raw_data_only_at_normal_finality(value, kind):
    from tensorfold.markup_lex import markup_regions
    tools = [{'type': 'function', 'function': {'name': 'echo', 'parameters': {
        'type': 'object', 'properties': {'text': {'type': kind}}}}}]
    raw = '<tool_call><function=echo><parameter=text>' + value + '</parameter></function></tool_call>'
    assert not markup_regions(raw)[0].complete
    assert markup_regions(raw, finished=True)[0].complete
    for parser in (parse_tool_calls_from_content, parse_cuda):
        for maximum in (None, 1):
            _, result = parser(raw, tools, max_calls=maximum)
            assert json.loads(result[0]['function']['arguments']) == {'text': value}
    streamer, deltas = ToolCallStreamer(tools), []
    for end in range(len(raw) + 1):
        deltas += streamer.feed(raw[:end])
    partial = ''.join(delta['tool_calls'][0]['function'].get('arguments', '') for delta in deltas)
    assert value not in partial
    deltas += streamer.feed(raw, finished=True)
    assert json.loads(''.join(delta['tool_calls'][0]['function'].get('arguments', '') for delta in deltas)) == {'text': value}


def test_closing_looking_bytes_inside_future_valid_json_never_finalize_a_prefix():
    value = '{"a":"inside </parameter></function></tool_call> still data"}'
    raw = '<tool_call><function=echo><parameter=text>' + value + '</parameter></function></tool_call>'
    tools = [{'type': 'function', 'function': {'name': 'echo', 'parameters': {
        'type': 'object', 'properties': {'text': {'type': 'object'}}}}}]
    streamer, deltas = ToolCallStreamer(tools), []
    for end in range(len(raw) + 1):
        deltas += streamer.feed(raw[:end])
    deltas += streamer.feed(raw, finished=True)
    assert json.loads(''.join(delta['tool_calls'][0]['function'].get('arguments', '') for delta in deltas)) == {'text': json.loads(value)}


@pytest.mark.parametrize('step', [1, 7, 10000])
def test_malformed_first_quote_cannot_borrow_from_a_separate_second_call(step):
    raw_value = '{"a":"cut'
    first = '<tool_call><function=echo><parameter=text>' + raw_value + '</parameter></function></tool_call>'
    second = '<tool_call><function=echo><parameter=text>"hello"</parameter></function></tool_call>'
    raw = first + second
    tools = [{'type': 'function', 'function': {'name': 'echo', 'parameters': {
        'type': 'object', 'properties': {'text': {'type': 'object'}}}}}]
    expected = [{'text': raw_value}, {'text': '"hello"'}]
    for parser in (parse_tool_calls_from_content, parse_cuda):
        _, result = parser(raw, tools)
        assert [json.loads(call['function']['arguments']) for call in result] == expected
    streamer, deltas = ToolCallStreamer(tools), []
    for end in range(1, len(raw) + 1, step):
        deltas += streamer.feed(raw[:end])
    deltas += streamer.feed(raw, finished=True)
    arguments = {}
    for delta in deltas:
        for call in delta['tool_calls']:
            arguments[call['index']] = arguments.get(call['index'], '') + call['function'].get('arguments', '')
    assert [json.loads(arguments[index]) for index in sorted(arguments)] == expected


@pytest.mark.parametrize('value,expected', [
    ('{"a":"</parameter></function></tool_call><tool_call>data"',
     {'a': '</parameter></function></tool_call><tool_call>data'}),
    ("{'a': '''line one\n</parameter></function></tool_call><tool_call>\nline two'''}",
     {'a': 'line one\n</parameter></function></tool_call><tool_call>\nline two'}),
])
def test_valid_repaired_json_and_python_triple_string_keep_whole_envelope_bytes(value, expected):
    tools = [{'type': 'function', 'function': {'name': 'echo', 'parameters': {
        'type': 'object', 'properties': {'text': {'type': 'object'}}}}}]
    raw = '<tool_call><function=echo><parameter=text>' + value + '</parameter></function></tool_call>'
    for parser in (parse_tool_calls_from_content, parse_cuda):
        _, result = parser(raw, tools)
        assert json.loads(result[0]['function']['arguments']) == {'text': expected}
    streamer, deltas = ToolCallStreamer(tools), []
    for end in range(len(raw) + 1):
        deltas += streamer.feed(raw[:end])
    deltas += streamer.feed(raw, finished=True)
    assert json.loads(''.join(delta['tool_calls'][0]['function'].get('arguments', '') for delta in deltas)) == {'text': expected}


@pytest.mark.parametrize('ended', [False, True])
@pytest.mark.parametrize('prior', [False, True])
@pytest.mark.parametrize('reason', ['length', 'stop'])
def test_tool_parse_finality_preserves_cap_userstop_and_prior_calls(ended, prior, reason):
    from tensorfold.server.tools import tool_calls_complete
    value = '"raw unmatched'
    raw = '<tool_call><function=echo><parameter=text>' + value + '</parameter></function></tool_call>'
    text = (XML if prior else '') + raw
    for parser in (parse_tool_calls_from_content, parse_cuda):
        content, result = parser(text, TOOLS, finished=ended)
        expected = ([{'text': 'hello'}] if prior else []) + ([{'text': value}] if ended else [])
        assert [json.loads(call['function']['arguments']) for call in result or []] == expected
        assert tool_calls_complete(text, TOOLS, finished=ended) is ended
        if not ended:
            assert raw in content
    streamer = ToolCallStreamer(TOOLS)
    deltas = []
    for cut in range(len(text) + 1):
        deltas += streamer.feed(text[:cut])
    assert not streamer.complete
    if ended:
        deltas += streamer.feed(text, finished=True)
    assert streamer.complete is ended
    # This is the Mac public-handler policy's internal engine outcome handoff,
    # not an extra wire field or a synthetic tool completion at a reply cap.
    result = ToolCallPolicy({}).finish({'content': text, 'finish_reason': reason,
                'tool_parse_finished': ended, 'tool_calls_complete': streamer.complete},
                TOOLS, parse_tool_calls_from_content)
    assert result['finish_reason'] == ('tool_calls' if ended else reason)
    expected = ([{'text': 'hello'}] if prior else []) + ([{'text': value}] if ended else [])
    assert [json.loads(call['function']['arguments']) for call in result.get('tool_calls', [])] == expected


def test_unambiguous_complete_and_quoted_example_do_not_require_eos_for_call_availability():
    from tensorfold.server.tools import tool_calls_complete
    for text in (XML, 'Before ' + XML, '`<tool_call><function=echo><parameter=text>"cut`'):
        assert tool_calls_complete(text, TOOLS, finished=False)
    _, result = parse_tool_calls_from_content(XML, TOOLS, finished=False)
    assert json.loads(result[0]['function']['arguments']) == {'text': 'hello'}
    policy = ToolCallPolicy({})
    result = policy.finish({'content': XML, 'finish_reason': 'length', 'tool_parse_finished': False,
                           'tool_calls_complete': True}, TOOLS, parse_tool_calls_from_content)
    assert result['finish_reason'] == 'tool_calls'


def test_streamer_completed_count_tracks_multiple_functions_under_one_real_envelope():
    raw = '<tool_call><function=echo><parameter=text>one</parameter></function>'
    raw += '<function=echo><parameter=text>two</parameter></function></tool_call>'
    streamer = ToolCallStreamer(TOOLS)
    streamer.feed(raw)
    assert streamer.index == 1 and streamer.complete

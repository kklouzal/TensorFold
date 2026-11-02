"""Independent T4 fixture matrix: classification, literal data and every delta boundary."""
import pytest

from tensorfold.markup_lex import CALL_MARKS, markup_regions, structural_tool_opener
from tensorfold.server.gemma_values import (GEMMA_OPEN, GEMMA_CLOSE, THOUGHT_OPEN, THOUGHT_CLOSE,
                                           hide_gemma_calls, split_gemma_thinking)


@pytest.mark.parametrize('name', ['thought', 'analysis', '', 'wrong.name'])
@pytest.mark.parametrize('closed', [False, True])
@pytest.mark.parametrize('prefix', ['', 'Visible. '])
def test_channel_cardinality_matrix(name, closed, prefix):
    raw = prefix + '<|channel>' + name + '\nBody'
    if closed:
        raw += '<channel|>Answer'
    expected = (('Body', prefix + 'Answer') if closed else
                ('Body', prefix) if name == 'thought' else ('', prefix + 'Body'))
    assert split_gemma_thinking(raw, finished=True, expect_thinking=True) == expected
    previous = ('', '')
    # Every character cut includes partial marker/header/codepoint boundaries.
    for end in range(len(raw) + 1):
        current = split_gemma_thinking(raw[:end], finished=False, expect_thinking=True)
        assert all(now.startswith(before) for before, now in zip(previous, current))
        previous = current
    assert all(final.startswith(before) for before, final in zip(previous, expected))


@pytest.mark.parametrize('marker', ['<|channel>', '<channel|>', '<|turn>', '<turn|>', '<|think|>'])
@pytest.mark.parametrize('quote', ['`{}`', '``{}``', '```text\n{}\n```', '~~~text\n{}\n~~~', '{{"value":"{}"}}'])
def test_quoted_code_and_json_string_markers_are_exact_data(marker, quote):
    raw = quote.format(marker)
    assert split_gemma_thinking(raw, finished=True) == ('', raw)
    known = THOUGHT_OPEN + '\n' + raw
    assert split_gemma_thinking(known, finished=True) == (raw, '')


@pytest.mark.parametrize('raw,expected', [
    (THOUGHT_OPEN + '\nFirst\n<|channel>other\nSecond' + THOUGHT_CLOSE + '\nAnswer', ('First\nSecond', 'Answer')),
    ('Unframed reasoning\n</think>\nAnswer', ('Unframed reasoning\n', 'Answer')),
    (THOUGHT_OPEN + '\nKnown open reasoning', ('Known open reasoning', '')),
    ('Plain answer', ('', 'Plain answer')),
    ('<|think|>Reason<channel|>Answer', ('Reason', 'Answer')),
    ('<|turn>Answer<turn|>', ('', 'Answer')),
])
def test_repeated_unframed_and_native_eos_cases(raw, expected):
    assert split_gemma_thinking(raw, finished=True, expect_thinking=True) == expected
    previous = ('', '')
    for end in range(len(raw) + 1):
        current = split_gemma_thinking(raw[:end], finished=False, expect_thinking=True)
        assert all(now.startswith(before) for before, now in zip(previous, current))
        previous = current
    assert all(final.startswith(before) for before, final in zip(previous, expected))


def test_proved_reasoning_phase_is_cleaned_without_invented_prompt_opener():
    raw = 'Unframed prefix\n<|channel>wrong\nMore\n`<turn|>`'
    reason, answer = split_gemma_thinking(raw, finished=True, reasoning_only=True)
    assert reason == 'Unframed prefix\nMore\n`<turn|>`' and answer == ''


def test_native_tool_arguments_are_opaque_and_open_thought_call_is_routed_only_when_complete():
    call = GEMMA_OPEN + 'call:show{text:<|"|>literal <|channel>wrong\nvalue<channel|><|"|>}' + GEMMA_CLOSE
    raw = THOUGHT_OPEN + '\nReason. ' + call
    assert split_gemma_thinking(raw, finished=True) == ('Reason. ', call)
    assert split_gemma_thinking(raw, finished=False) == ('Reason. ', '')
    mention = raw + THOUGHT_CLOSE + 'Answer'
    assert split_gemma_thinking(mention, finished=True) == ('Reason. ' + call, 'Answer')
    quoted = '```text\n' + call + '\n```'
    assert hide_gemma_calls(quoted, finished=True) == quoted


def test_removing_reasoning_never_synthesizes_a_tool_opener():
    raw = '<|tool_' + THOUGHT_OPEN + 'Hidden' + THOUGHT_CLOSE + 'call>call:show{}' + GEMMA_CLOSE
    _, answer = split_gemma_thinking(raw, finished=True)
    assert not any(region.kind == 'call' for region in markup_regions(answer))


@pytest.mark.parametrize('opener,closer', CALL_MARKS)
@pytest.mark.parametrize('prefix', ['', 'Thought. ', '```text\n', '`', '{"value":"', '<|"|>'])
def test_shared_grammar_opener_predicate_never_activates_quoted_data(opener, closer, prefix):
    assert structural_tool_opener(prefix + opener, opener) is (prefix in ('', 'Thought. '))


def test_xml_parameter_quotes_and_unicode_do_not_hide_following_real_envelope():
    first = '<tool_call><function=run><parameter=cmd>İ echo `unterminated</parameter></function></tool_call>'
    second = '<tool_call>'
    assert structural_tool_opener(first + second, second)


def test_literal_stream_decoder_preserves_control_data_and_utf8_boundaries():
    from tensorfold.cuda.reply_text import StreamDecoder
    class Tokens:
        control_token_ids = frozenset({202})
        unused_token_ids = frozenset({203})
        pieces = {1: b'{"value":"', 2: b'"}', 10: b'\xf0', 11: b'\x9f', 12: b'\xa6', 13: b'\x89',
                  202: b'<|turn>', 203: b'UNUSED', 0: b''}
        def decode(self, ids, **kwargs):
            return b''.join(self.pieces[i] for i in ids).decode('utf-8', errors='replace')
    decoder = StreamDecoder(Tokens(), (0,), literal=True)
    assert decoder.add([1, 10]) == ''
    assert decoder.add([11, 12, 13]) == '{"value":"🦉'
    assert decoder.add([202, 203, 2, 0]) == '{"value":"🦉<|turn>UNUSED"}'
    assert decoder.final() == '{"value":"🦉<|turn>UNUSED"}'
    ordinary = StreamDecoder(Tokens(), (0,))
    assert ordinary.add([1, 202, 203, 2]) == '{"value":""}'


@pytest.mark.parametrize('name', ['other', 'thought', ''])
def test_repeated_channel_header_never_publishes_undecided_name(name):
    prefix = THOUGHT_OPEN + '\nFirst\n'
    raw = prefix + '<|channel>' + name + '\nSecond' + THOUGHT_CLOSE + 'Answer'
    previous = ('', '')
    for end in range(len(raw) + 1):
        current = split_gemma_thinking(raw[:end], finished=False, expect_thinking=True)
        assert all(now.startswith(before) for before, now in zip(previous, current))
        previous = current
    assert split_gemma_thinking(prefix + '<|channel>o', finished=False) == ('First\n', '')
    assert split_gemma_thinking(raw, finished=True) == ('First\nSecond', 'Answer')


def test_stopped_initial_known_thought_name_prefix_stays_framing():
    for length in range(len('thought') + 1):
        assert split_gemma_thinking(' \n<|channel>' + 'thought'[:length], finished=True) == ('', ' \n')
    raw = '<|channel>thought\nNative EOS reason'
    assert split_gemma_thinking(raw, finished=True) == ('Native EOS reason', '')
    for wrapper in ('`{}`', '"{}"', '```text\n{}\n```'):
        literal = wrapper.format('<|channel>tho')
        assert split_gemma_thinking(literal, finished=True) == ('', literal)


@pytest.mark.parametrize('opener,header', [
    ('<tool_call>', '<tool_call>\n<function=echo>\n'),
    ('<|tool_call>', '<|tool_call>call:echo{'),
])
def test_strict_header_intent_does_not_turn_bare_openers_into_calls(opener, header):
    from tensorfold.markup_lex import pending_call_header, structural_call_header, structural_marker
    headers = (header,)
    bare = 'Reason ' + opener + ' 9 days<channel|>'
    assert structural_call_header(bare, opener, headers) is None
    assert structural_marker(bare, '<channel|>', call_headers=headers)
    assert not structural_marker('Reason ' + header + '<channel|>', '<channel|>', call_headers=headers)
    assert structural_call_header('Reason ' + header, opener, headers) == (len('Reason '), len('Reason ' + header))
    for end in range(len(opener), len(header)):
        prefix = 'Reason ' + header[:end]
        assert structural_call_header(prefix, opener, headers) is None
        assert pending_call_header(prefix, opener, headers) == len('Reason ')
    assert pending_call_header('Reason ' + header, opener, headers) is None
    for wrapper in ('`{}`', '"{}"', '```text\n{}\n```'):
        assert structural_call_header(wrapper.format(header), opener, headers) is None
        assert pending_call_header(wrapper.format(header[:len(opener)]), opener, headers) is None
    assert structural_call_header('Reason ' + opener + 'call:unoffered{', opener, headers) is None

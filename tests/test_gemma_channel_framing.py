"""Native channel framing is structural; only genuine tool strings are data."""
import logging

import pytest

from tensorfold.server.gemma_values import split_gemma_thinking
from tensorfold.server.text import CHANNEL_MARKERS, THINK_MARKERS, split_thinking


def monotonic(raw, expected, **options):
    assert split_gemma_thinking(raw, finished=True, **options) == expected
    previous = ('', '')
    for at in range(len(raw)+1):
        current = split_gemma_thinking(raw[:at], finished=False, **options)
        assert all(now.startswith(old) for old, now in zip(previous, current)), (at, previous, current)
        previous = current
    assert all(final.startswith(old) for old, final in zip(previous, expected))


@pytest.mark.parametrize('name', ['thought', 'analysis', 'final', 'answer', 'unknown.name', '🦉', ''])
@pytest.mark.parametrize('closed', [False, True])
def test_every_channel_name_is_nonfinal_until_native_close(name, closed):
    raw = 'Visible. <|channel>  '+name+'\r\nReason'
    if closed:
        raw += '<channel|>Answer'
    monotonic(raw, ('Reason', 'Visible. '+('Answer' if closed else '')), expect_thinking=True)


@pytest.mark.parametrize('raw,expected', [
    ('A<|channel>thought\nOne<|channel>analysis\nTwo<channel|>B<channel|>C', ('OneTwo', 'ABC')),
    ('A<|channel><|channel>\nTwo<channel|>B', ('Two', 'AB')),
    ('A<|channel> \tthought \r\n  Reason<channel|>B', ('Reason', 'AB')),
    ('A<|channel>\nReason<channel|>B', ('Reason', 'AB')),
    ('Before `<|channel>analysis\nhidden<channel|>` after', ('hidden', 'Before `` after')),
    ('{"x":"<|channel>thought\nsecret<channel|>public"}', ('secret', '{"x":"public"}')),
    ('A<channel|>B<channel|>C', ('', 'ABC')),
    ('A<|channel>', ('', 'A')),
    ('A<|channel>unknown', ('', 'A')),
])
def test_repeated_nested_empty_whitespace_quote_and_stray_frames(raw, expected):
    monotonic(raw, expected)


@pytest.mark.parametrize('expect', [False, True])
def test_opened_channel_requires_proved_phase_and_preserves_field_order(expect):
    raw = 'First<|channel>analysis\nSecond<channel|>A<|channel>thought\nThird<channel|>B'
    monotonic(raw, ('FirstSecondThird', 'AB'), opened=True, expect_thinking=expect)
    assert split_thinking(raw, finished=True, markers=CHANNEL_MARKERS,
                          opened=True, expect_thinking=expect) == ('FirstSecondThird', 'AB')
    monotonic('Plain<channel|>answer', ('', 'Plainanswer'), expect_thinking=expect)
    assert split_thinking('Reason</think>answer', finished=True, markers=THINK_MARKERS, opened=True) == ('Reason', 'answer')


@pytest.mark.parametrize('prefix', ['', '<|channel>thought\nReason. '])
def test_native_tool_string_channel_bytes_are_literal_at_every_boundary(prefix):
    call = '<|tool_call>call:echo{text:<|"|>literal <|channel>alien\nvalue<channel|><|"|>}<tool_call|>'
    raw = prefix + call
    expected = ('Reason. ', call) if prefix else ('', call)
    monotonic(raw, expected)


def test_headers_and_markers_inside_ordinary_quoted_regions_are_held_during_stream():
    for wrapper in ('`{}`', '```text\n{}\n```', '"{}"', '{"x":"{}"}'):
        raw = wrapper.replace('{}', '<|channel>unknown\nReason<channel|>Answer', 1)
        final = split_gemma_thinking(raw, finished=True)
        assert '<|channel>' not in ''.join(final) and '<channel|>' not in ''.join(final)
        monotonic(raw, final)


def test_unknown_logs_finished_only_once_per_name_with_bounded_records(caplog):
    raw = ''.join('<|channel>'+name+'\nR<channel|>' for name in ['alien', 'alien', *[f'x{i}' for i in range(20)]])
    with caplog.at_level(logging.WARNING, logger='tensorfold.server.gemma_values'):
        for at in range(len(raw)+1):
            split_gemma_thinking(raw[:at], finished=False)
        assert not caplog.records
        assert split_gemma_thinking(raw, finished=True) == ('R'*22, '')
    assert len(caplog.records) == 9
    assert sum('alien' in item.getMessage() for item in caplog.records) == 1
    assert 'suppressed' in caplog.records[-1].getMessage()


@pytest.mark.parametrize('opened', [False, True])
def test_utf8_each_byte_and_every_two_chunk_cut_preserve_monotonic_fields(opened):
    import codecs
    reason = '思考🦉 café'
    call = '<|tool_call>call:echo{text:<|"|>値🙂 <|channel>literal<channel|><|"|>}<tool_call|>'
    prefix = '' if opened else '前🙂 <|channel>thought\n'
    raw = prefix + reason + '<channel|>答えé ' + call
    expected = (reason, ('' if opened else '前🙂 ') + '答えé ' + call)
    encoded = raw.encode('utf-8')

    def consume(chunks):
        decoder = codecs.getincrementaldecoder('utf-8')('strict')
        text, previous = '', ('', '')
        for chunk in chunks:
            text += decoder.decode(chunk, final=False)
            current = split_gemma_thinking(text, finished=False, opened=opened)
            assert all(now.startswith(old) for old, now in zip(previous, current))
            previous = current
        text += decoder.decode(b'', final=True)
        assert text == raw
        final = split_gemma_thinking(text, finished=True, opened=opened)
        assert final == expected
        assert all(now.startswith(old) for old, now in zip(previous, final))

    consume([bytes([byte]) for byte in encoded])
    for cut in range(len(encoded)+1):
        consume((encoded[:cut], encoded[cut:]))


@pytest.mark.parametrize('opened', [False, True])
@pytest.mark.parametrize('allow', [False, True])
def test_tool_recovery_requires_offered_tool_context_in_nonfinal_channel(opened, allow):
    call = '<|tool_call>call:echo{text:<|"|>literal <|channel>unknown\n<channel|> 🦉<|"|>}<tool_call|>'
    raw = ('' if opened else '<|channel>analysis\n') + 'Reason. ' + call
    expected = ('Reason. ', call) if allow else ('Reason. '+call, '')
    monotonic(raw, expected, opened=opened, allow_tool_calls=allow)
    assert split_thinking(raw, finished=True, markers=CHANNEL_MARKERS, opened=opened,
                          allow_tool_calls=allow) == expected
    # A real channel close makes this a reasoning mention in either context.
    assert split_gemma_thinking(raw+'<channel|>Answer', finished=True, opened=opened,
                               allow_tool_calls=allow) == ('Reason. '+call, 'Answer')


def test_channel_tool_context_option_does_not_change_other_marker_parser():
    raw = 'Reason</think>Answer'
    assert split_thinking(raw, finished=True, markers=THINK_MARKERS, allow_tool_calls=False) == ('Reason', 'Answer')


@pytest.mark.parametrize('header', ['<|channel>thought', '<|channel>analysis\n', '<|channel>final\r\n',
                                    '<|channel>\n', '<|channel>', '<|channel>  \t🦉.分析 \t\r\n', '<|channel>thought\u00a0\n'])
def test_trusted_prompt_terminal_any_header_establishes_open_nonfinal_phase(header):
    from tensorfold.server.gemma_values import prompt_opens_thought
    assert prompt_opens_thought('<|turn>model\n'+header)


@pytest.mark.parametrize('suffix', ['<|channel>analysis\nBODY', '<|channel>thought<channel|>',
                                    '<|channel>analysis\n<turn|>', '<|channel>analysis\n<|turn>model\n',
                                    'No native header', '<|channel>analysis body'])
def test_prompt_helper_never_infers_from_body_or_closed_or_previous_turn(suffix):
    from tensorfold.server.gemma_values import prompt_opens_thought
    assert not prompt_opens_thought(suffix)


@pytest.mark.parametrize('prefix,opened,expect,reason', [
    ('<|channel>thought\nReason. ', False, False, 'Reason. '),
    ('Reason. ', True, True, 'Reason. '),
    ('Reason `unfinished ', True, False, 'Reason `unfinished '),
    ('"Unfinished quote ', True, False, '"Unfinished quote '),
    ('<|channel>analysis\nReason. ```text\n', False, False, 'Reason. ```text\n'),
    ('Unframed 🦉 reason. ', False, True, 'Unframed 🦉 reason. '),
    ('<|channel>thought\nReason<|channel>undecided', False, False, 'Reason'),
])
def test_trusted_answer_boundary_streams_call_without_thought_close_or_prefix_reparse(prefix, opened, expect, reason):
    call = '<|tool_call>call:echo{text:<|"|>値🦉 <|channel>literal<channel|><|"|>}<tool_call|>'
    raw, offset = prefix+call, len(prefix)
    previous = ('', '')
    for end in range(len(raw)+1):
        hint = offset if end >= offset else None
        current = split_gemma_thinking(raw[:end], finished=False, opened=opened,
            expect_thinking=expect, answer_start=hint)
        assert all(now.startswith(before) for before, now in zip(previous, current)), (end, previous, current)
        previous = current
    expected = (reason, call)
    final = split_gemma_thinking(raw, finished=True, opened=opened, expect_thinking=expect, answer_start=offset)
    assert final == expected
    assert all(now.startswith(before) for before, now in zip(previous, final))
    assert split_thinking(raw, finished=True, markers=CHANNEL_MARKERS, opened=opened,
                         expect_thinking=expect, answer_start=offset) == expected


@pytest.mark.parametrize('offset', [-1, 4, True, 1.5, '1'])
def test_answer_boundary_rejects_invalid_character_offsets(offset):
    with pytest.raises(ValueError, match='character offset'):
        split_gemma_thinking('abc', finished=False, answer_start=offset)


def test_answer_boundary_requires_actual_tool_context_and_handles_empty_tail():
    with pytest.raises(ValueError, match='tool intent'):
        split_gemma_thinking('Reason', finished=False, allow_tool_calls=False, answer_start=6)
    assert split_gemma_thinking('Reason', finished=False, expect_thinking=True, answer_start=6) == ('Reason', '')
    assert split_gemma_thinking('Reason', finished=True, expect_thinking=True, answer_start=6) == ('Reason', '')


@pytest.mark.parametrize('opened', [False, True])
@pytest.mark.parametrize('closed', [False, True])
def test_foreign_Qwen_close_never_ends_active_native_channel(opened, closed):
    body = 'private A\n</think>\nprivate B'
    raw = ('' if opened else '<|channel>analysis\n') + body
    if closed:
        raw += '<channel|>Answer'
    monotonic(raw, (body, 'Answer' if closed else ''), opened=opened, allow_tool_calls=False)


def test_foreign_close_legacy_behavior_outside_native_is_unchanged():
    monotonic('Unframed reasoning\n</think>\nAnswer', ('Unframed reasoning\n', 'Answer'), expect_thinking=True)
    assert split_thinking('Reason</think>Answer', finished=True, markers=THINK_MARKERS) == ('Reason', 'Answer')


def test_trusted_tool_boundary_can_end_native_phase_even_with_foreign_close_prose():
    prefix = '<|channel>thought\nprivate A\n</think>\nprivate B '
    call = '<|tool_call>call:echo{text:<|"|>ok<|"|>}<tool_call|>'
    raw = prefix+call
    expected = ('private A\n</think>\nprivate B ', call)
    assert split_gemma_thinking(raw, finished=True, answer_start=len(prefix)) == expected
    previous = ('', '')
    for end in range(len(raw)+1):
        result = split_gemma_thinking(raw[:end], finished=False,
            answer_start=len(prefix) if end >= len(prefix) else None)
        assert all(now.startswith(old) for old, now in zip(previous, result))
        previous = result
    assert all(now.startswith(old) for old, now in zip(previous, expected))

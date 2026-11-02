"""Explicit budget and required-call transitions, without model/runtime work."""
from tensorfold.engine.call_gate import CallGate, ThinkBudget

EOS, END, OPEN, WHY, FENCE, NL, DATA = range(7)
TEXT = {EOS: '', END: '</think>', OPEN: '<tool_call>', WHY: 'why', FENCE: '```', NL: '\n', DATA: 'payload'}


def decode(ids, **kwargs):
    return ''.join(TEXT[int(t)] for t in ids)


def budget(limit):
    from tensorfold.engine.grammar import AnswerBoundary
    gate = ThinkBudget(limit, [NL, END], END, stop_ids=(EOS,), decode=decode)
    gate.intent = AnswerBoundary(7, think_end=END, answer_start=OPEN, answer_opener='<tool_call>',
                                 answer_headers=('<tool_call>payload',), decode=decode,
                                 prefix_valid=lambda _: True)  # phase-only oracle; native matcher covered separately
    return gate


def test_explicit_budget_does_not_replace_native_eos_at_its_limit():
    gate = budget(2)
    assert gate.cut([WHY, EOS]) is None
    gate.observe(WHY)
    gate.observe(EOS)
    assert not gate.open and gate.cut([DATA]*4) is None


def test_actual_call_disarms_budget_before_payload_while_quoted_marker_does_not():
    gate = budget(4)
    assert gate.cut([WHY, NL, OPEN, DATA]) is None
    for token in [WHY, NL, OPEN, DATA]:
        gate.observe(token)
    assert not gate.open and gate.cut([DATA]*8) is None
    quoted = budget(4)
    assert quoted.cut([FENCE, NL, OPEN, DATA]) == (3, [NL, END])
    for token in [FENCE, NL, OPEN]:
        quoted.observe(token)
    quoted.intent.advance([FENCE, NL, OPEN])
    assert quoted.open and quoted.cut([DATA]) == (0, [NL, END])


def test_required_call_omitted_thought_opener_is_not_forced_before_its_real_boundary():
    gate = CallGate.after_prompt([], OPEN, lambda _: False, think_end=END, thinking=True,
                                decode=decode)
    assert gate.cut([WHY, NL]) is None
    for token in [WHY, NL]:
        gate.observe(token)
    assert gate.cut([OPEN, DATA]) is None
    gate.observe(OPEN)
    gate.observe(DATA)
    assert not gate.watching and not gate.done  # nonstrict calls remain free reasoning
    gate.observe(END)
    assert gate.watching and gate.cut([DATA]) == (0, [OPEN])


def test_required_call_quoted_opener_never_marks_a_call_started():
    gate = CallGate.after_prompt([], OPEN, lambda _: False, think_end=END, thinking=True,
                                decode=decode)
    assert gate.cut([FENCE, NL, OPEN, DATA]) is None
    for token in [FENCE, NL, OPEN, DATA]:
        gate.observe(token)
    assert not gate.done and not gate.watching


def test_quoted_close_does_not_end_budget_but_trusted_fix_notifies_exact_published_index():
    from tensorfold.engine.call_gate import generate_gated
    gate = budget(4)
    events, published, prompts = [], [], []
    def generate(prompt, count, callback):
        prompts.append(list(prompt))
        if len(prompts) == 1:
            callback([FENCE, NL, END, DATA])  # quoted END inside unclosed code; fourth token hits budget
        else:
            callback([DATA])
        return {}
    generate_gated(generate, [DATA], 6, [gate], lambda tokens: published.extend(tokens) or False,
                   on_think_close=events.append)
    assert events == [4]
    assert published[:5] == [FENCE, NL, END, NL, END]
    assert not gate.open and prompts[1] == [DATA, FENCE, NL, END, NL, END]


def test_reply_cap_before_forced_close_does_not_invent_an_activation_event():
    from tensorfold.engine.call_gate import generate_gated
    events, published = [], []
    gate = budget(1)
    def generate(prompt, count, callback):
        callback([WHY])
        return {}
    generate_gated(generate, [], 1, [gate], lambda tokens: published.extend(tokens) or False,
                   on_think_close=events.append)
    assert published == [NL] and events == []
